#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import dotenv

dotenv.load_dotenv("/home/robot/lerobot/.env")

import dataclasses
import logging
import time
from contextlib import nullcontext
from pprint import pformat
from typing import Any

import torch
from accelerate import Accelerator
from termcolor import colored
from torch.optim import Optimizer
from tqdm import tqdm

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.sampler import EpisodeAwareSampler  # 轨迹感知采样器
from lerobot.datasets.utils import cycle  # 循环迭代器
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import close_envs
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.rl.wandb_utils import WandBLogger
from lerobot.scripts.lerobot_eval import eval_policy_all
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker  # 指标跟踪工具
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.utils.utils import (
    format_big_number,
    has_method,
    init_logging,
    inside_slurm,
)


def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    accelerator: Accelerator,
    lr_scheduler=None,
    lock=None,
    rabc_weights_provider=None,
) -> tuple[MetricsTracker, dict]:
    """
    Performs a single training step to update the policy's weights. -- 执行单个训练步骤来更新策略的权重。

    This function executes the forward and backward passes, clips gradients, and steps the optimizer and
    learning rate scheduler. Accelerator handles mixed-precision training automatically.
    此函数执行前向和后向传递, 裁剪梯度, 并执行优化器和学习率调度程序。Accelerator 自动处理混合精度训练。
    该函数负责前向传播、损失计算、反向传播和参数更新。

    Args:
        train_metrics: A MetricsTracker instance to record training statistics. 即：一个 MetricsTracker 实例，用于记录训练统计信息。
        policy: The policy model to be trained. 即：要训练的策略模型。
        batch: A batch of training data. 即：一批训练数据。
        optimizer: The optimizer used to update the policy's parameters. 即：用于更新策略参数的优化器。
        grad_clip_norm: The maximum norm for gradient clipping. 即：梯度裁剪的最大范数。
        accelerator: The Accelerator instance for distributed training and mixed precision. 即：用于分布式训练和混合精度的 Accelerator 实例。
        lr_scheduler: An optional learning rate scheduler. 即：可选的学习率调度器。
        lock: An optional lock for thread-safe optimizer updates. 即：用于线程安全优化器更新的可选锁。
        rabc_weights_provider: Optional RABCWeights instance for sample weighting. 即：用于样本加权的可选 RABCWeights 实例。

    Returns:
        A tuple containing:
        - The updated MetricsTracker with new statistics for this step. 即：更新后的 MetricsTracker, 带有此步骤的新统计信息。
        - A dictionary of outputs from the policy's forward pass, for logging purposes. 即：来自策略前向传递的输出字典，用于日志记录。
    """
    start_time = time.perf_counter()
    policy.train()

    rabc_batch_weights = None  # 初始化 RA-BC 权重
    rabc_batch_stats = None  # 初始化 RA-BC 统计

    # 如果启用了 RA-BC（稳健动作行为克隆），计算样本权重
    if rabc_weights_provider is not None:
        rabc_batch_weights, rabc_batch_stats = (
            rabc_weights_provider.compute_batch_weights(batch)
        )

    # 自动混合精度上下文（由 accelerator 处理）
    with accelerator.autocast():
        # case1 如果使用 RA-BC：则使用每个样本的损失以实现适当的加权
        if rabc_batch_weights is not None:
            # 获取每个样本的损失
            per_sample_loss, output_dict = policy.forward(batch, reduction="none")

            # * 应用 RA-BC 权重: L_RA-BC = Σ(w_i * l_i) / (Σw_i + ε)
            # rabc_batch_weights 已经归一化为总和等于批次大小
            epsilon = 1e-6
            # 使用加权平均计算最终损失
            loss = (per_sample_loss * rabc_batch_weights).sum() / (
                rabc_batch_weights.sum() + epsilon
            )
            # 记录原始平均权重（归一化之前）- 这是有意义的指标
            output_dict["rabc_mean_weight"] = rabc_batch_stats["raw_mean_weight"]
            output_dict["rabc_num_zero_weight"] = rabc_batch_stats["num_zero_weight"]
            output_dict["rabc_num_full_weight"] = rabc_batch_stats["num_full_weight"]
        else:
            # case2 如果是普通训练：直接计算平均损失
            loss, output_dict = policy.forward(batch)

        # TODO(rcadene): policy.unnormalize_outputs(out_dict)

    # 使用加速器的反向传播方法
    accelerator.backward(loss)

    # 如果指定了梯度裁剪范数，则执行梯度裁剪：防止梯度爆炸
    if grad_clip_norm > 0:
        # 对策略参数应用梯度裁剪
        grad_norm = accelerator.clip_grad_norm_(policy.parameters(), grad_clip_norm)
    else:
        # 当裁剪范数为 0 或负数时，不进行有效裁剪（无穷大阈值）
        grad_norm = torch.nn.utils.clip_grad_norm_(
            policy.parameters(), float("inf"), error_if_nonfinite=False
        )

    # 优化器更新参数
    with lock if lock is not None else nullcontext():
        optimizer.step()

    # 清零优化器梯度
    optimizer.zero_grad()

    # 在每个批次而不是每个 epoch 时通过 pytorch 调度器进行学习率更新
    if lr_scheduler is not None:
        lr_scheduler.step()

    # 如果模型有特殊的 update 方法（如 EMA 更新），则执行
    if has_method(accelerator.unwrap_model(policy, keep_fp32_wrapper=True), "update"):
        # 解包加速器包装的模型并调用其更新方法
        accelerator.unwrap_model(policy, keep_fp32_wrapper=True).update()

    # 将数据填入指标跟踪器
    train_metrics.loss = loss.item()  # 记录当前批次的损失值
    train_metrics.grad_norm = grad_norm.item()  # 记录梯度范数值
    train_metrics.lr = optimizer.param_groups[0]["lr"]  # 记录当前学习率
    train_metrics.update_s = time.perf_counter() - start_time  # 记录本次更新的耗时

    return train_metrics, output_dict  # 返回更新后的指标和输出字典


@parser.wrap()
def train(cfg: TrainPipelineConfig, accelerator: Accelerator | None = None):
    """
    Main function to train a policy. 即：训练策略的主要函数。

    This function orchestrates the entire training pipeline, including:
    - Setting up logging, seeding, and device configuration.
    - Creating the dataset, evaluation environment (if applicable), policy, and optimizer.
    - Handling resumption from a checkpoint.
    - Running the main training loop, which involves fetching data batches and calling `update_policy`.
    - Periodically logging metrics, saving model checkpoints, and evaluating the policy.
    - Pushing the final trained model to the Hugging Face Hub if configured.

    Args:
        cfg: A `TrainPipelineConfig` object containing all training configurations.
        accelerator: Optional Accelerator instance. If None, one will be created automatically.
    """
    # step0. 验证参数的合法性
    cfg.validate()

    # step1. 如果没有提供，则自动创建加速器
    # ? accelerator 是什么？好像之前学过了吧？
    # Create Accelerator if not provided
    # It will automatically detect if running in distributed mode or single-process mode
    # We set step_scheduler_with_optimizer=False to prevent accelerate from adjusting the lr_scheduler steps based on the num_processes
    # We set find_unused_parameters=True to handle models with conditional computation
    if accelerator is None:
        from accelerate.utils import DistributedDataParallelKwargs

        # 创建 DDP 参数，启用未使用参数的查找功能，处理条件计算的模型
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

        # 如果策略设备设置为 CPU，则强制使用 CPU 设备
        # Accelerate auto-detects the device based on the available hardware and ignores the policy.device setting.
        # Force the device to be CPU when policy.device is set to CPU.
        force_cpu = cfg.policy.device == "cpu"
        # 创建 Accelerator 实例
        accelerator = Accelerator(
            step_scheduler_with_optimizer=False,
            kwargs_handlers=[ddp_kwargs],
            cpu=force_cpu,
        )

    init_logging(accelerator=accelerator)

    # Determine if this is the main process (for logging and checkpointing)
    # When using accelerate, only the main process should log to avoid duplicate outputs
    is_main_process = accelerator.is_main_process

    # Only log on main process
    if is_main_process:
        logging.info(pformat(cfg.to_dict()))  # 打印 cfg

    # step2. 初始化 WandB 日志（仅在主进程上）
    # Initialize wandb only on main process
    if cfg.wandb.enable and cfg.wandb.project and is_main_process:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        if is_main_process:
            logging.info(
                colored("Logs will be saved locally.", "yellow", attrs=["bold"])
            )
            # INFO 2026-04-07 01:35:53 ot_train.py:232 Logs will be saved locally.

    if cfg.seed is not None:
        set_seed(cfg.seed, accelerator=accelerator)

    # 设置 CUDNN 性能优化
    device = accelerator.device
    if cfg.cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # step3. 创建数据集 (确保多进程下数据下载同步)
    # Dataset loading synchronization: main process downloads first to avoid race conditions
    if is_main_process:
        logging.info("Creating dataset")
        # INFO 2026-04-07 01:35:53 ot_train.py:251 Creating dataset
        # ? 为什么调用两次 make_dataset(cfg) ？答：这是分布式训练的安全加载流程！
        # 第1次调用：仅在主进程（rank=0）中执行，负责【下载】数据集
        dataset = make_dataset(cfg)

    # 分布式训练中的「栅栏同步」! 等待所有进程都到达这里才会继续执行
    # 在什么地方调用？
    #   1. 数据集加载完成后：确保所有进程都加载好数据集再继续
    #   2. 创建策略完成后：确保所有进程都初始化好模型
    #   3. 保存检查点前：确保所有进程训练到同一阶段
    #   4. 评估完成后：防止评估还在进行就开始下一轮训练
    # 没有这个调用会发生什么？
    #   -- 有的进程训练快有的慢，导致模型参数不同步
    #   -- 出现「部分进程已经开始下一轮，部分还在上一轮」的死锁
    accelerator.wait_for_everyone()

    # Now all other processes can safely load the dataset
    if not is_main_process:
        # 第2次调用：其他进程（rank>0）开始【加载】，此时数据集已经存在
        dataset = make_dataset(cfg)

    # step4. 创建评估环境（仅在仿真环境且主进程时开启）
    # 当不传入 --env 参数时，cfg.env 默认为 None,不会调用 make_env()，不会创建任何仿真环境
    # 因为真机评估是用独立的 lerobot_eval.py 脚本，而不会在训练脚本里执行
    # Create environment used for evaluating checkpoints during training on simulation data.
    # On real-world data, no need to create an environment as evaluations are done outside train.py,
    # using the eval.py instead, with gym_dora environment and dora-rs.
    eval_env = None
    # 启用定期评估功能（默认 0 表示不评估）
    # 配置了仿真环境，才可以在训练中评估
    # 只在主进程执行，避免多进程重复评估
    if cfg.eval_freq > 0 and cfg.env is not None and is_main_process:
        logging.info("Creating env")
        eval_env = make_env(
            cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs
        )

    # step5. 创建策略 (Policy)
    if is_main_process:
        logging.info("Creating policy")
        # INFO 2026-04-07 01:35:56 ot_train.py:290 Creating policy

    # * 创建策略模型
    # ps 注意需要用到数据集元数据或者仿真环境配置，来让策略知道输入输出数据的结构
    policy = make_policy(
        cfg=cfg.policy,  # 策略配置对象
        ds_meta=dataset.meta,  # 传入数据集元数据
        rename_map=cfg.rename_map,  # 创建策略时使用的重命名映射
    )

    # 如果启用了 PEFT（高效微调，如 LoRA），进行封装
    if cfg.peft is not None:
        logging.info("Using PEFT! Wrapping model.")
        # Convert CLI peft config to dict for overrides
        peft_cli_overrides = dataclasses.asdict(cfg.peft)
        policy = policy.wrap_with_peft(peft_cli_overrides=peft_cli_overrides)

    # Wait for all processes to finish policy creation before continuing
    accelerator.wait_for_everyone()

    # step6. 配置预处理器和后处理器（用于数据归一化）
    # Create processors - only provide dataset_stats if not resuming from saved processors
    processor_kwargs = {}  # 预处理处理器
    postprocessor_kwargs = {}  # 后处理处理器
    if (
        cfg.policy.pretrained_path and not cfg.resume
    ) or not cfg.policy.pretrained_path:
        # Only provide dataset_stats when not resuming from saved processor state
        processor_kwargs["dataset_stats"] = dataset.meta.stats

    # SARM 模型特殊处理，可忽略
    # For SARM, always provide dataset_meta for progress normalization
    if cfg.policy.type == "sarm":
        processor_kwargs["dataset_meta"] = dataset.meta

    # 配置预处理器的覆盖参数（设备类型、归一化映射等）
    if cfg.policy.pretrained_path is not None:
        processor_kwargs["preprocessor_overrides"] = {
            "device_processor": {"device": device.type},
            "normalizer_processor": {
                "stats": dataset.meta.stats,
                "features": {
                    **policy.config.input_features,
                    **policy.config.output_features,
                },
                "norm_map": policy.config.normalization_mapping,
            },
        }
        processor_kwargs["preprocessor_overrides"]["rename_observations_processor"] = {
            "rename_map": cfg.rename_map
        }
        postprocessor_kwargs["postprocessor_overrides"] = {
            "unnormalizer_processor": {
                "stats": dataset.meta.stats,
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
            },
        }

    # * 创建前、后两大处理器
    # ps 工作流程: 数据集原始数据 → preprocessor → 模型输入 → 模型 → 模型输出 → postprocessor → 机器人动作
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        **processor_kwargs,
        **postprocessor_kwargs,
    )

    # step7. 创建优化器和调度器
    if is_main_process:
        logging.info("Creating optimizer and scheduler")
        # INFO 2026-04-07 01:37:13 ot_train.py:360 Creating optimizer and scheduler
    # 创建优化器和学习率调度器
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)

    # 如果启用了 RA-BC，加载预先计算好的进度数据（默认不启用）
    # Load precomputed SARM progress for RA-BC if enabled
    # Generate progress using: src/lerobot/policies/sarm/compute_rabc_weights.py
    rabc_weights = None
    if cfg.use_rabc:
        from lerobot.utils.rabc import RABCWeights

        # Get chunk_size from policy config
        chunk_size = getattr(policy.config, "chunk_size", None)
        if chunk_size is None:
            raise ValueError("Chunk size is not found in policy config")

        head_mode = getattr(cfg, "rabc_head_mode", "sparse")
        logging.info(f"Loading SARM progress for RA-BC from {cfg.rabc_progress_path}")
        logging.info(
            f"Using chunk_size={chunk_size} from policy config, head_mode={head_mode}"
        )
        rabc_weights = RABCWeights(
            progress_path=cfg.rabc_progress_path,
            chunk_size=chunk_size,
            head_mode=head_mode,
            kappa=getattr(cfg, "rabc_kappa", 0.01),
            epsilon=getattr(cfg, "rabc_epsilon", 1e-6),
            device=device,
        )

    step = 0  # 记录策略更新的步数 (forward + backward + optim)

    # step8. 处理断点续训
    if cfg.resume:
        step, optimizer, lr_scheduler = load_training_state(
            cfg.checkpoint_path, optimizer, lr_scheduler
        )

    # 打印模型参数量信息
    num_learnable_params = sum(
        p.numel() for p in policy.parameters() if p.requires_grad
    )
    num_total_params = sum(p.numel() for p in policy.parameters())

    # 打印其它日志
    if is_main_process:
        logging.info(
            colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}"
        )
        # INFO 2026-04-07 01:37:13 ot_train.py:406 Output dir: outputs/train/smolval_so101_test

        if cfg.env is not None:
            logging.info(f"{cfg.env.task=}")
            logging.info("Creating environment processors")
            env_preprocessor, env_postprocessor = make_env_pre_post_processors(
                env_cfg=cfg.env, policy_cfg=cfg.policy
            )
        logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
        # INFO 2026-04-07 01:37:13 ot_train.py:413 cfg.steps=100000 (100K)
        logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
        # INFO 2026-04-07 01:37:13 ot_train.py:414 dataset.num_frames=19631 (20K)
        logging.info(f"{dataset.num_episodes=}")
        # INFO 2026-04-07 01:37:13 ot_train.py:415 dataset.num_episodes=50

        # ?什么叫有效的 batch_size？答: 这是分布式训练中的核心概念！
        num_processes = accelerator.num_processes  # 进程数 / 可用 GPU 数
        effective_bs = cfg.batch_size * num_processes  # 有效 bs = 进程数 * 显卡数
        logging.info(
            f"Effective batch size: {cfg.batch_size} x {num_processes} = {effective_bs}"
        )
        # INFO 2026-04-07 01:37:13 ot_train.py:420 Effective batch size: 8 x 1 = 8

        logging.info(
            f"{num_learnable_params=} ({format_big_number(num_learnable_params)})"
        )
        # INFO 2026-04-07 01:37:13 ot_train.py:422 num_learnable_params=99880992 (100M)
        logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")
        # INFO 2026-04-07 01:37:13 ot_train.py:423 num_total_params=450046176 (450M)

    # step9. 创建 Dataloader
    # - 当提供了自定义 sampler 时，shuffle 参数必须设为 False
    # - EpisodeAwareSampler 是机器人训练的特殊要求：不能把不同轨迹的帧混合在一起采样
    if hasattr(cfg.policy, "drop_n_last_frames"):
        # case1 使用特殊采样器，不在这里 shuffle
        shuffle = False
        sampler = EpisodeAwareSampler(
            dataset.meta.episodes["dataset_from_index"],
            dataset.meta.episodes["dataset_to_index"],
            episode_indices_to_use=dataset.episodes,
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
        )
    else:
        # case2 使用默认采样器
        # 简单随机打乱, 全局随机打乱所有帧，不区分 episode, ❌ 和 sampler 互斥
        shuffle = True
        sampler = None  # 自定义采样逻辑, EpisodeAwareSampler 会在 episode 内部打乱，不会跨episode采样

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,  # 8
        shuffle=shuffle and not cfg.dataset.streaming,
        sampler=sampler,
        pin_memory=device.type == "cuda",
        drop_last=False,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )
    # print(f"Dataloader Length = {len(dataloader)}")  # Length = 2454，即有 2454 个 batch
    logging.info(f"Dataloader Length = {len(dataloader)}")

    # step10. 使用加速器封装所有对象
    accelerator.wait_for_everyone()
    policy, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        policy, optimizer, dataloader, lr_scheduler
    )
    # 将 dataloader 变为无限循环迭代器
    dl_iter = cycle(dataloader)

    policy.train()

    # 初始化训练指标计量器
    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }

    # Keep global batch size for logging; MetricsTracker handles world size internally.
    effective_batch_size = cfg.batch_size * accelerator.num_processes
    train_tracker = MetricsTracker(
        cfg.batch_size,
        dataset.num_frames,
        dataset.num_episodes,
        train_metrics,
        initial_step=step,
        accelerator=accelerator,
    )

    # 设置进度条
    if is_main_process:
        progbar = tqdm(
            total=cfg.steps - step,
            desc="Training",
            unit="step",
            disable=inside_slurm(),
            position=0,
            leave=True,
        )
        logging.info(
            f"Start offline training on a fixed dataset, with effective batch size: {effective_batch_size}"
        )
        # INFO 2026-04-07 01:37:13 ot_train.py:494 Start offline training on a fixed dataset, with effective batch size: 8

    # step11. ================== 主训练循环 ==================
    logging.info(f"数据集总帧数: {dataset.num_frames}")  # 19631
    logging.info(f"每个 batch 大小: {cfg.batch_size}")  # 8
    logging.info(f"总 batch 数: {len(dataloader)}")
    logging.info(f"GPU 数量: {accelerator.num_processes}")  # 1
    logging.info(f"有效 batch_size: {effective_batch_size}")  # 8

    # 总共循环 100_000 个 steps
    for _ in range(step, cfg.steps):
        # 记录数据加载开始时间
        start_time = time.perf_counter()

        # 从 DataLoader 中取出一个 batch（如果到尽头会自动循环）
        batch = next(dl_iter)
        # print(f"Batch type: ", type(batch))  # Batch type:  <class 'dict'>
        # print(f"Batch keys: ", batch.keys())
        # dict_keys(
        #     [
        #         "observation.images.top",
        #         "observation.images.wrist",
        #         "action",
        #         "observation.state",
        #         "timestamp",
        #         "frame_index",
        #         "episode_index",
        #         "index",
        #         "task_index",
        #         "action_is_pad",
        #         "observation.state_is_pad",
        #         "observation.images.top_is_pad",
        #         "observation.images.wrist_is_pad",
        #         "task",
        #     ]
        # )
        # print(f"Batch length = {len(batch)}")  # Batch length = 14

        # 用预处理器把原始数据转成模型可接受的格式
        batch = preprocessor(batch)

        # 记录数据加载花费的时间
        train_tracker.dataloading_s = time.perf_counter() - start_time

        # * 调用 update_policy 进行一轮完整的训练步骤，包括：前向——>反向——>优化
        train_tracker, output_dict = update_policy(
            train_tracker,
            policy,
            batch,
            optimizer,
            cfg.optimizer.grad_clip_norm,
            accelerator=accelerator,
            lr_scheduler=lr_scheduler,
            rabc_weights_provider=rabc_weights,
        )

        # Note: eval and checkpoint happens *after* the `step`th training update has completed, so we
        # 训练完成，step 数 +1
        step += 1

        # 主进程更新进度条
        if is_main_process:
            progbar.update(1)

        # 指标追踪器步进，计算平均指标
        train_tracker.step()

        # 检查当前 step 是否到了日志、保存或评估的时间点
        # 每隔 200 次输出一次日志
        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0 and is_main_process
        # 20_000 次保存一次模型，并且最后一次必然保存，不管有没有到 20000：
        is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps
        # 20_000 次评估一次模型
        is_eval_step = cfg.eval_freq > 0 and step % cfg.eval_freq == 0

        # 日志步骤：打印指标并上传到 wandb
        if is_log_step:
            logging.info(train_tracker)
            # INFO 2026-04-07 01:38:48 ot_train.py:567 step:200 smpl:2K ep:4 epch:0.08 loss:1.728 grdn:17.987 lr:1.0e-05 updt_s:0.463 data_s:0.009
            if wandb_logger:
                wandb_log_dict = train_tracker.to_dict()
                if output_dict:
                    wandb_log_dict.update(output_dict)
                # Log RA-BC statistics if enabled
                if rabc_weights is not None:
                    rabc_stats = rabc_weights.get_stats()
                    wandb_log_dict.update(
                        {
                            "rabc_delta_mean": rabc_stats["delta_mean"],
                            "rabc_delta_std": rabc_stats["delta_std"],
                            "rabc_num_frames": rabc_stats["num_frames"],
                        }
                    )
                wandb_logger.log_dict(wandb_log_dict, step)

            # 重置平均值，为下一段日志做准备
            train_tracker.reset_averages()

        # 保存步骤：保存模型检查点
        if cfg.save_checkpoint and is_saving_step:
            if is_main_process:
                logging.info(f"Checkpoint policy after step {step}")
                # INFO 2026-04-07 04:12:54 ot_train.py:590 Checkpoint policy after step 20000
                checkpoint_dir = get_step_checkpoint_dir(
                    cfg.output_dir, cfg.steps, step
                )
                save_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    step=step,
                    cfg=cfg,
                    policy=accelerator.unwrap_model(policy),
                    optimizer=optimizer,
                    scheduler=lr_scheduler,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                )

                # 更新“最新”软链接
                update_last_checkpoint(checkpoint_dir)

                if wandb_logger:
                    wandb_logger.log_policy(checkpoint_dir)

            accelerator.wait_for_everyone()

        # * 模型评估 (仅主进程且指定 --env 的情况下才进行评估)，对于真机不执行评估，因为有单独的脚本
        if cfg.env and is_eval_step:
            if is_main_process:
                step_id = get_step_identifier(step, cfg.steps)
                logging.info(f"Eval policy at step {step}")
                # ! 由于未配置仿真环境，所以此处不进行评估
                with torch.no_grad(), accelerator.autocast():
                    eval_info = eval_policy_all(
                        envs=eval_env,  # dict[suite][task_id] -> vec_env
                        policy=accelerator.unwrap_model(policy),
                        env_preprocessor=env_preprocessor,
                        env_postprocessor=env_postprocessor,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        n_episodes=cfg.eval.n_episodes,
                        videos_dir=cfg.output_dir / "eval" / f"videos_step_{step_id}",
                        max_episodes_rendered=4,
                        start_seed=cfg.seed,
                        max_parallel_tasks=cfg.env.max_parallel_tasks,
                    )

                # 处理并记录评估指标（如成功率、总奖励）
                # overall metrics (suite-agnostic)
                aggregated = eval_info["overall"]

                # optional: per-suite logging
                for suite, suite_info in eval_info.items():
                    logging.info("Suite %s aggregated: %s", suite, suite_info)

                # meters/tracker
                eval_metrics = {
                    "avg_sum_reward": AverageMeter("∑rwrd", ":.3f"),
                    "pc_success": AverageMeter("success", ":.1f"),
                    "eval_s": AverageMeter("eval_s", ":.3f"),
                }
                eval_tracker = MetricsTracker(
                    cfg.batch_size,
                    dataset.num_frames,
                    dataset.num_episodes,
                    eval_metrics,
                    initial_step=step,
                    accelerator=accelerator,
                )
                eval_tracker.eval_s = aggregated.pop("eval_s")
                eval_tracker.avg_sum_reward = aggregated.pop("avg_sum_reward")
                eval_tracker.pc_success = aggregated.pop("pc_success")
                if wandb_logger:
                    wandb_log_dict = {**eval_tracker.to_dict(), **eval_info}
                    wandb_logger.log_dict(wandb_log_dict, step, mode="eval")
                    wandb_logger.log_video(
                        eval_info["overall"]["video_paths"][0], step, mode="eval"
                    )

            accelerator.wait_for_everyone()

    # step12. 训练结束后的收尾
    if is_main_process:
        progbar.close()

    if eval_env:
        close_envs(eval_env)

    # 如果配置了上传，将最终模型推送到 Hugging Face Hub
    if is_main_process:
        logging.info("# End of training")
        # INFO 2026-04-07 03:27:25 ot_train.py:566 End of training

        if cfg.policy.push_to_hub:
            unwrapped_policy = accelerator.unwrap_model(policy)
            if cfg.policy.use_peft:
                unwrapped_policy.push_model_to_hub(cfg, peft_model=unwrapped_policy)
            else:
                unwrapped_policy.push_model_to_hub(cfg)
            preprocessor.push_to_hub(cfg.policy.repo_id)
            postprocessor.push_to_hub(cfg.policy.repo_id)

    # Properly clean up the distributed process group
    accelerator.wait_for_everyone()

    # 结束训练会话
    accelerator.end_training()


def main():
    register_third_party_plugins()
    train()


if __name__ == "__main__":
    main()
