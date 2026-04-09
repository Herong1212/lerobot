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

from __future__ import annotations

import importlib
import logging
from typing import Any, TypedDict, Unpack

import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.feature_utils import dataset_to_policy_features
from lerobot.envs.configs import EnvConfig
from lerobot.envs.utils import env_to_policy_features
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.groot.configuration_groot import GrootConfig
from lerobot.policies.multi_task_dit.configuration_multi_task_dit import (
    MultiTaskDiTConfig,
)
from lerobot.policies.pi0.configuration_pi0 import PI0Config
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.sac.configuration_sac import SACConfig
from lerobot.policies.sac.reward_model.configuration_classifier import (
    RewardClassifierConfig,
)
from lerobot.policies.sarm.configuration_sarm import SARMConfig
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.tdmpc.configuration_tdmpc import TDMPCConfig
from lerobot.policies.utils import validate_visual_features_consistency
from lerobot.policies.vqbet.configuration_vqbet import VQBeTConfig
from lerobot.policies.wall_x.configuration_wall_x import WallXConfig
from lerobot.policies.xvla.configuration_xvla import XVLAConfig
from lerobot.processor import PolicyProcessorPipeline
from lerobot.processor.converters import (
    batch_to_transition,
    policy_action_to_transition,
    transition_to_batch,
    transition_to_policy_action,
)
from lerobot.types import PolicyAction
from lerobot.utils.constants import (
    ACTION,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)


def get_policy_class(name: str) -> type[PreTrainedPolicy]:
    """
    Retrieves a policy class by its registered name.

    This function uses dynamic imports to avoid loading all policy classes into memory
    at once, improving startup time and reducing dependencies.

    Args:
        name: The name of the policy. Supported names are "tdmpc", "diffusion", "act",
            "multi_task_dit", "vqbet", "pi0", "pi05", "sac", "reward_classifier", "smolvla", "wall_x".
    Returns:
        The policy class corresponding to the given name.

    Raises:
        NotImplementedError: If the policy name is not recognized.
    """
    if name == "tdmpc":
        from lerobot.policies.tdmpc.modeling_tdmpc import TDMPCPolicy

        return TDMPCPolicy
    elif name == "diffusion":
        from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

        return DiffusionPolicy
    elif name == "act":
        from lerobot.policies.act.modeling_act import ACTPolicy

        return ACTPolicy
    elif name == "multi_task_dit":
        from lerobot.policies.multi_task_dit.modeling_multi_task_dit import (
            MultiTaskDiTPolicy,
        )

        return MultiTaskDiTPolicy
    elif name == "vqbet":
        from lerobot.policies.vqbet.modeling_vqbet import VQBeTPolicy

        return VQBeTPolicy
    elif name == "pi0":
        from lerobot.policies.pi0.modeling_pi0 import PI0Policy

        return PI0Policy
    elif name == "pi0_fast":
        from lerobot.policies.pi0_fast.modeling_pi0_fast import PI0FastPolicy

        return PI0FastPolicy
    elif name == "pi05":
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy

        return PI05Policy
    elif name == "sac":
        from lerobot.policies.sac.modeling_sac import SACPolicy

        return SACPolicy
    elif name == "reward_classifier":
        from lerobot.policies.sac.reward_model.modeling_classifier import Classifier

        return Classifier
    elif name == "smolvla":
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        return SmolVLAPolicy
    elif name == "sarm":
        from lerobot.policies.sarm.modeling_sarm import SARMRewardModel

        return SARMRewardModel
    elif name == "groot":
        from lerobot.policies.groot.modeling_groot import GrootPolicy

        return GrootPolicy
    elif name == "xvla":
        from lerobot.policies.xvla.modeling_xvla import XVLAPolicy

        return XVLAPolicy
    elif name == "wall_x":
        from lerobot.policies.wall_x.modeling_wall_x import WallXPolicy

        return WallXPolicy
    else:
        try:
            return _get_policy_cls_from_policy_name(name=name)
        except Exception as e:
            raise ValueError(f"Policy type '{name}' is not available.") from e


def make_policy_config(policy_type: str, **kwargs) -> PreTrainedConfig:
    """
    Instantiates a policy configuration object based on the policy type.

    This factory function simplifies the creation of policy configuration objects by
    mapping a string identifier to the corresponding config class.

    Args:
        policy_type: The type of the policy. Supported types include "tdmpc",
                     "multi_task_dit", "diffusion", "act", "vqbet", "pi0", "pi05", "sac",
                     "smolvla", "reward_classifier", "wall_x".
        **kwargs: Keyword arguments to be passed to the configuration class constructor.

    Returns:
        An instance of a `PreTrainedConfig` subclass.

    Raises:
        ValueError: If the `policy_type` is not recognized.
    """
    if policy_type == "tdmpc":
        return TDMPCConfig(**kwargs)
    elif policy_type == "diffusion":
        return DiffusionConfig(**kwargs)
    elif policy_type == "act":
        return ACTConfig(**kwargs)
    elif policy_type == "multi_task_dit":
        return MultiTaskDiTConfig(**kwargs)
    elif policy_type == "vqbet":
        return VQBeTConfig(**kwargs)
    elif policy_type == "pi0":
        return PI0Config(**kwargs)
    elif policy_type == "pi05":
        return PI05Config(**kwargs)
    elif policy_type == "sac":
        return SACConfig(**kwargs)
    elif policy_type == "smolvla":
        return SmolVLAConfig(**kwargs)
    elif policy_type == "reward_classifier":
        return RewardClassifierConfig(**kwargs)
    elif policy_type == "groot":
        return GrootConfig(**kwargs)
    elif policy_type == "xvla":
        return XVLAConfig(**kwargs)
    elif policy_type == "wall_x":
        return WallXConfig(**kwargs)
    else:
        try:
            config_cls = PreTrainedConfig.get_choice_class(policy_type)
            return config_cls(**kwargs)
        except Exception as e:
            raise ValueError(f"Policy type '{policy_type}' is not available.") from e


class ProcessorConfigKwargs(TypedDict, total=False):
    """
    A TypedDict defining the keyword arguments for processor configuration.

    This provides type hints for the optional arguments passed to `make_pre_post_processors`,
    improving code clarity and enabling static analysis.

    Attributes:
        preprocessor_config_filename: The filename for the preprocessor configuration.
        postprocessor_config_filename: The filename for the postprocessor configuration.
        preprocessor_overrides: A dictionary of overrides for the preprocessor configuration.
        postprocessor_overrides: A dictionary of overrides for the postprocessor configuration.
        dataset_stats: Dataset statistics for normalization.
    """

    preprocessor_config_filename: str | None
    postprocessor_config_filename: str | None
    preprocessor_overrides: dict[str, Any] | None
    postprocessor_overrides: dict[str, Any] | None
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None


def make_pre_post_processors(
    policy_cfg: PreTrainedConfig,
    pretrained_path: str | None = None,
    **kwargs: Unpack[ProcessorConfigKwargs],
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    为给定策略创建或加载预处理和后处理管道。

    This function acts as a factory. It can either load existing processor pipelines
    from a pretrained path or create new ones from scratch based on the policy
    configuration. Each policy type has a dedicated factory function for its
    processors (e.g., `make_tdmpc_pre_post_processors`).

    Args:
        policy_cfg: 需要创建处理器的策略配置，告诉函数要为哪种类型的机器人策略创建处理器（如 ACT、TDMPC 等）
        pretrained_path: 可选参数，指定从哪里加载预训练的处理器管道。
            如果提供此路径，则从此路径加载管道。
        **kwargs: 处理器配置的关键字参数，定义在 [ProcessorConfigKwargs](file:///home/robot/lerobot/src/lerobot/policies/factory.py#L197-L216) 中。

    Returns:
        A tuple containing the input (pre-processor) and output (post-processor) pipelines.

    返回： 一个包含输入（预处理器 pre-processor) 和输出（后处理器 post-processor) 管道的元组：
                第一个元素: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]] - 预处理器管道
                    作用：处理输入数据（如摄像头图像、传感器读数等），将其转换为策略模型可以理解的格式
                    例子：将原始图像像素值(0-255)归一化到(-1,1)范围，或者将关节角度标准化
                第二个元素: PolicyProcessorPipeline[PolicyAction, PolicyAction] - 后处理器管道
                    作用：处理策略模型输出的动作，将其转换为机器人实际可以执行的格式
                    例子：将模型输出的标准化动作值反归一化为实际关节角度或末端执行器位置

    Raises:
        NotImplementedError: If a processor factory is not implemented for the given
            policy configuration type.
    """
    # 如果提供了预训练路径，则从该路径加载处理器管道
    if pretrained_path:
        # TODO(Steven): Temporary patch, implement correctly the processors for Gr00t
        if isinstance(policy_cfg, GrootConfig):
            # GROOT handles normalization in groot_pack_inputs_v3 step
            # Need to override both stats AND normalize_min_max since saved config might be empty
            preprocessor_overrides = {}
            postprocessor_overrides = {}
            preprocessor_overrides["groot_pack_inputs_v3"] = {
                "stats": kwargs.get("dataset_stats"),
                "normalize_min_max": True,
            }

            # Also ensure postprocessing slices to env action dim and unnormalizes with dataset stats
            env_action_dim = policy_cfg.output_features[ACTION].shape[0]
            postprocessor_overrides["groot_action_unpack_unnormalize_v1"] = {
                "stats": kwargs.get("dataset_stats"),
                "normalize_min_max": True,
                "env_action_dim": env_action_dim,
            }
            kwargs["preprocessor_overrides"] = preprocessor_overrides
            kwargs["postprocessor_overrides"] = postprocessor_overrides

        # 从预训练路径加载预处理器和后处理器
        return (
            PolicyProcessorPipeline.from_pretrained(
                pretrained_model_name_or_path=pretrained_path,
                config_filename=kwargs.get(
                    "preprocessor_config_filename",
                    f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
                ),
                overrides=kwargs.get("preprocessor_overrides", {}),
                to_transition=batch_to_transition,
                to_output=transition_to_batch,
            ),
            PolicyProcessorPipeline.from_pretrained(
                pretrained_model_name_or_path=pretrained_path,
                config_filename=kwargs.get(
                    "postprocessor_config_filename",
                    f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json",
                ),
                overrides=kwargs.get("postprocessor_overrides", {}),
                to_transition=policy_action_to_transition,
                to_output=transition_to_policy_action,
            ),
        )

    # 根据策略类型（policy type）创建新的处理器
    if isinstance(policy_cfg, TDMPCConfig):
        from lerobot.policies.tdmpc.processor_tdmpc import (
            make_tdmpc_pre_post_processors,
        )

        processors = make_tdmpc_pre_post_processors(
            config=policy_cfg,
            dataset_stats=kwargs.get("dataset_stats"),
        )

    elif isinstance(policy_cfg, DiffusionConfig):
        from lerobot.policies.diffusion.processor_diffusion import (
            make_diffusion_pre_post_processors,
        )

        processors = make_diffusion_pre_post_processors(
            config=policy_cfg,
            dataset_stats=kwargs.get("dataset_stats"),
        )

    elif isinstance(policy_cfg, ACTConfig):
        from lerobot.policies.act.processor_act import make_act_pre_post_processors

        processors = make_act_pre_post_processors(
            config=policy_cfg,
            dataset_stats=kwargs.get("dataset_stats"),
        )

    elif isinstance(policy_cfg, MultiTaskDiTConfig):
        from lerobot.policies.multi_task_dit.processor_multi_task_dit import (
            make_multi_task_dit_pre_post_processors,
        )

        processors = make_multi_task_dit_pre_post_processors(
            config=policy_cfg,
            dataset_stats=kwargs.get("dataset_stats"),
        )

    elif isinstance(policy_cfg, VQBeTConfig):
        from lerobot.policies.vqbet.processor_vqbet import (
            make_vqbet_pre_post_processors,
        )

        processors = make_vqbet_pre_post_processors(
            config=policy_cfg,
            dataset_stats=kwargs.get("dataset_stats"),
        )

    elif isinstance(policy_cfg, PI0Config):
        from lerobot.policies.pi0.processor_pi0 import make_pi0_pre_post_processors

        processors = make_pi0_pre_post_processors(
            config=policy_cfg,
            dataset_stats=kwargs.get("dataset_stats"),
        )

    elif isinstance(policy_cfg, PI05Config):
        from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors

        processors = make_pi05_pre_post_processors(
            config=policy_cfg,
            dataset_stats=kwargs.get("dataset_stats"),
        )

    elif isinstance(policy_cfg, SACConfig):
        from lerobot.policies.sac.processor_sac import make_sac_pre_post_processors

        processors = make_sac_pre_post_processors(
            config=policy_cfg,
            dataset_stats=kwargs.get("dataset_stats"),
        )

    elif isinstance(policy_cfg, RewardClassifierConfig):
        from lerobot.policies.sac.reward_model.processor_classifier import (
            make_classifier_processor,
        )

        processors = make_classifier_processor(
            config=policy_cfg,
            dataset_stats=kwargs.get("dataset_stats"),
        )

    # 如果是 SmolVLA 策略配置
    elif isinstance(policy_cfg, SmolVLAConfig):
        from lerobot.policies.smolvla.processor_smolvla import (
            make_smolvla_pre_post_processors,
        )

        processors = make_smolvla_pre_post_processors(
            config=policy_cfg,  # 传入策略配置
            dataset_stats=kwargs.get("dataset_stats"),  # 数据集统计信息
        )

    elif isinstance(policy_cfg, SARMConfig):
        from lerobot.policies.sarm.processor_sarm import make_sarm_pre_post_processors

        processors = make_sarm_pre_post_processors(
            config=policy_cfg,
            dataset_stats=kwargs.get("dataset_stats"),
            dataset_meta=kwargs.get("dataset_meta"),
        )
    elif isinstance(policy_cfg, GrootConfig):
        from lerobot.policies.groot.processor_groot import (
            make_groot_pre_post_processors,
        )

        processors = make_groot_pre_post_processors(
            config=policy_cfg,
            dataset_stats=kwargs.get("dataset_stats"),
        )

    elif isinstance(policy_cfg, XVLAConfig):
        from lerobot.policies.xvla.processor_xvla import (
            make_xvla_pre_post_processors,
        )

        processors = make_xvla_pre_post_processors(
            config=policy_cfg,
            dataset_stats=kwargs.get("dataset_stats"),
        )

    elif isinstance(policy_cfg, WallXConfig):
        from lerobot.policies.wall_x.processor_wall_x import (
            make_wall_x_pre_post_processors,
        )

        processors = make_wall_x_pre_post_processors(
            config=policy_cfg,
            dataset_stats=kwargs.get("dataset_stats"),
        )

    else:
        try:
            processors = _make_processors_from_policy_config(
                config=policy_cfg,
                dataset_stats=kwargs.get("dataset_stats"),
            )
        except Exception as e:
            raise ValueError(
                f"Processor for policy type '{policy_cfg.type}' is not implemented."
            ) from e

    return processors


def make_policy(
    cfg: PreTrainedConfig,
    ds_meta: LeRobotDatasetMetadata | None = None,
    env_cfg: EnvConfig | None = None,
    rename_map: dict[str, str] | None = None,
) -> PreTrainedPolicy:
    """
    创建一个策略模型。

    This factory function handles the logic of creating a policy, which requires
    determining the input and output feature shapes. These shapes can be derived
    either from a `LeRobotDatasetMetadata` object or an `EnvConfig` object. The function
    can either initialize a new policy from scratch or load a pretrained one.

    Args:
        cfg: The configuration for the policy to be created. If `cfg.pretrained_path` is
             set, the policy will be loaded with weights from that path.
        ds_meta: Dataset metadata used to infer feature shapes and types. Also provides
                 statistics for normalization layers.
        env_cfg: Environment configuration used to infer feature shapes and types.
                 One of `ds_meta` or `env_cfg` must be provided.
        rename_map: Optional mapping of dataset or environment feature keys to match
                 expected policy feature names (e.g., `"left"` → `"camera1"`).

    Returns:
        An instantiated and device-placed policy model. 即：一个已实例化并放置在指定设备上的策略模型

    Raises:
        ValueError: If both or neither of `ds_meta` and `env_cfg` are provided. 即：如果不允许同时提供或都没有提供 `ds_meta` 和 `env_cfg`。
        NotImplementedError: If attempting to use an unsupported policy-backend
                             combination (e.g., VQBeT with 'mps'). 如果尝试使用不支持的策略-后端组合 (例如, VQBeT 与 'mps'）。
    """
    # 检查是否提供了数据集元数据或环境配置（只能二选一）
    # 为什么需要这些信息？因为策略需要知道输入的数据是什么样子的（比如图像尺寸、动作维度等）
    if bool(ds_meta) == bool(env_cfg):
        raise ValueError(
            "Either one of a dataset metadata or a sim env must be provided."
        )

    # 特定于 VQBeT 策略的兼容性检查（MPS后端不支持）
    # 注意：目前如果你尝试在MPS后端运行VQBeT，会出现错误
    # 原因：MPS后端尚未实现 'aten::unique_dim' 操作
    # NOTE: Currently, if you try to run vqbet with mps backend, you'll get this error.
    # TODO(aliberts, rcadene): Implement a check_backend_compatibility in policies?
    # NotImplementedError: The operator 'aten::unique_dim' is not currently implemented for the MPS device. If
    # you want this op to be added in priority during the prototype phase of this feature, please comment on
    # https://github.com/pytorch/pytorch/issues/77764. As a temporary fix, you can set the environment
    # variable `PYTORCH_ENABLE_MPS_FALLBACK=1` to use the CPU as a fallback for this op. WARNING: this will be
    # slower than running natively on MPS.
    if cfg.type == "vqbet" and cfg.device == "mps":
        raise NotImplementedError(
            "Current implementation of VQBeT does not support `mps` backend. "
            "Please use `cpu` or `cuda` backend."
        )

    # 获取策略类（根据策略类型获取相应的策略类）
    policy_cls = get_policy_class(cfg.type)

    kwargs = {}
    # 根据是否有数据集元数据决定使用哪种方式获取特征
    if ds_meta is not None:
        # 从数据集元数据获取特征信息
        # 例如：如果数据集中有左摄像头图像(640x480)和右摄像头图像(640x480)，以及7维关节角度作为动作
        features = dataset_to_policy_features(ds_meta.features)
    else:
        # 从环境配置获取特征信息
        if not cfg.pretrained_path:
            logging.warning(
                "You are instantiating a policy from scratch and its features are parsed from an environment "
                "rather than a dataset. Normalization modules inside the policy will have infinite values "
                "by default without stats from a dataset."
            )
        if env_cfg is None:
            raise ValueError("env_cfg cannot be None when ds_meta is not provided")
        features = env_to_policy_features(env_cfg)

    # 设置输出特征（通常是动作）
    cfg.output_features = {
        key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION
    }
    # 如果没有设置输入特征，则自动设置（除动作外的所有特征）
    if not cfg.input_features:
        cfg.input_features = {
            key: ft for key, ft in features.items() if key not in cfg.output_features
        }
    kwargs["config"] = cfg

    # 如果可用，传递数据集统计信息给策略（某些策略如SARM需要）
    # Pass dataset_stats to the policy if available (needed for some policies like SARM)
    if ds_meta is not None and hasattr(ds_meta, "stats"):
        # 这些统计数据包括均值、标准差等，用于归一化输入数据
        kwargs["dataset_stats"] = ds_meta.stats

    if ds_meta is not None:
        kwargs["dataset_meta"] = ds_meta

    # 检查PEFT配置是否正确
    if not cfg.pretrained_path and cfg.use_peft:
        raise ValueError(
            "Instantiating a policy with `use_peft=True` without a checkpoint is not supported since that requires "
            "the PEFT config parameters to be set. For training with PEFT, see `lerobot_train.py` on how to do that."
        )

    # 根据不同情况加载策略：
    if cfg.pretrained_path and not cfg.use_peft:
        # 加载预训练策略并根据需要覆盖配置
        # Load a pretrained policy and override the config if needed (for example, if there are inference-time
        # hyperparameters that we want to vary).
        kwargs["pretrained_name_or_path"] = cfg.pretrained_path
        # 例如：加载已经训练好的 ACT 模型，可以直接用于推理
        policy = policy_cls.from_pretrained(**kwargs)
    elif cfg.pretrained_path and cfg.use_peft:
        # 加载预训练的 PEFT 模型（LoRA 适配器）
        # Load a pretrained PEFT model on top of the policy. The pretrained path points to the folder/repo
        # of the adapter and the adapter's config contains the path to the base policy. So we need the
        # adapter config first, then load the correct policy and then apply PEFT.
        from peft import PeftConfig, PeftModel

        logging.info("Loading policy's PEFT adapter.")

        peft_pretrained_path = cfg.pretrained_path
        peft_config = PeftConfig.from_pretrained(peft_pretrained_path)

        kwargs["pretrained_name_or_path"] = peft_config.base_model_name_or_path
        if not kwargs["pretrained_name_or_path"]:
            # This means that there's a bug or we trained a policy from scratch using PEFT.
            # It is more likely that this is a bug so we'll raise an error.
            raise ValueError(
                "No pretrained model name found in adapter config. Can't instantiate the pre-trained policy on which "
                "the adapter was trained."
            )

        # 例如：加载基础 ACT 模型，然后应用微调过的适配器
        policy = policy_cls.from_pretrained(**kwargs)
        policy = PeftModel.from_pretrained(
            policy, peft_pretrained_path, config=peft_config
        )

    else:
        # 创建一个全新的策略
        # 例如：从头开始创建一个新的策略模型，通常用于训练阶段
        policy = policy_cls(**kwargs)

    # 将策略移动到指定设备上
    policy.to(cfg.device)
    assert isinstance(policy, torch.nn.Module)

    # 可选：编译策略以提高性能
    # policy = torch.compile(policy, mode="reduce-overhead")

    # 如果没有重命名映射，则验证视觉特征一致性
    if not rename_map:
        validate_visual_features_consistency(cfg, features)
        # TODO: (jadechoghari) - add a check_state(cfg, features) and check_action(cfg, features)

    return policy


def _get_policy_cls_from_policy_name(name: str) -> type[PreTrainedConfig]:
    """Get policy class from its registered name using dynamic imports.

    This is used as a helper function to import policies from 3rd party lerobot plugins.

    Args:
        name: The name of the policy.
    Returns:
        The policy class corresponding to the given name.
    """
    if name not in PreTrainedConfig.get_known_choices():
        raise ValueError(
            f"Unknown policy name '{name}'. Available policies: {PreTrainedConfig.get_known_choices()}"
        )

    config_cls = PreTrainedConfig.get_choice_class(name)
    config_cls_name = config_cls.__name__

    model_name = config_cls_name.removesuffix(
        "Config"
    )  # e.g., DiffusionConfig -> Diffusion
    if model_name == config_cls_name:
        raise ValueError(
            f"The config class name '{config_cls_name}' does not follow the expected naming convention."
            f"Make sure it ends with 'Config'!"
        )
    cls_name = model_name + "Policy"  # e.g., DiffusionConfig -> DiffusionPolicy
    module_path = config_cls.__module__.replace(
        "configuration_", "modeling_"
    )  # e.g., configuration_diffusion -> modeling_diffusion

    module = importlib.import_module(module_path)
    policy_cls = getattr(module, cls_name)
    return policy_cls


def _make_processors_from_policy_config(
    config: PreTrainedConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[Any, Any]:
    """Create pre- and post-processors from a policy configuration using dynamic imports.

    This is used as a helper function to import processor factories from 3rd party lerobot plugins.

    Args:
        config: The policy configuration object.
        dataset_stats: Dataset statistics for normalization.
    Returns:
        A tuple containing the input (pre-processor) and output (post-processor) pipelines.
    """

    policy_type = config.type
    function_name = f"make_{policy_type}_pre_post_processors"
    module_path = config.__class__.__module__.replace(
        "configuration_", "processor_"
    )  # e.g., configuration_diffusion -> processor_diffusion
    logging.debug(
        f"Instantiating pre/post processors using function '{function_name}' from module '{module_path}'"
    )
    module = importlib.import_module(module_path)
    function = getattr(module, function_name)
    return function(config, dataset_stats=dataset_stats)
