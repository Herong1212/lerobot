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
""" 
Visualize data of **all** frames of any episode of a dataset of type LeRobotDataset. 
即：用于可视化 LeRobot 数据集中的任意 episode 的所有帧数据。
主要功能：
    - 加载本地或 HuggingFace Hub 上的 LeRobotDataset
    - 可视化指定 episode 的图像、动作、状态、奖励等数据
    - 支持三种模式：本地直接查看、远程服务(启动 gRPC/HTTP 服务器)、保存为文件


Note: The last frame of the episode doesn't always correspond to a final state.
That's because our datasets are composed of transition from state to state up to
the antepenultimate state associated to the ultimate action to arrive in the final state.
However, there might not be a transition from a final state to another state.

Note: This script aims to visualize the data used to train the neural networks.
~What you see is what you get~. When visualizing image modality, it is often expected to observe
lossy compression artifacts since these images have been decoded from compressed mp4 videos to
save disk space. The compression factor applied has been tuned to not affect success rate.

Examples:

- Visualize data stored on a local machine:
```
local$ lerobot-dataset-viz \
    --repo-id lerobot/pusht \
    --episode-index 0
```

- Visualize data stored on a distant machine with a local viewer:
```
distant$ lerobot-dataset-viz \
    --repo-id lerobot/pusht \
    --episode-index 0 \
    --save 1 \
    --output-dir path/to/directory

local$ scp distant:path/to/directory/lerobot_pusht_episode_0.rrd .
local$ rerun lerobot_pusht_episode_0.rrd
```

- Visualize data stored on a distant machine through streaming:
```
distant$ lerobot-dataset-viz \
    --repo-id lerobot/pusht \
    --episode-index 0 \
    --mode distant \
    --grpc-port 9876

local$ rerun rerun+http://IP:GRPC_PORT/proxy
```

"""

import pprint
import dotenv

dotenv.load_dotenv("/home/robot/lerobot/.env")

import argparse
import gc
import logging
import time
from pathlib import Path

import numpy as np
import rerun as rr
import torch
import torch.utils.data
import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import ACTION, DONE, OBS_STATE, REWARD
from lerobot.utils.utils import init_logging


def to_hwc_uint8_numpy(chw_float32_torch: torch.Tensor) -> np.ndarray:
    """
    将 PyTorch 的 [通道, 高, 宽] 格式的浮点张量转换为 OpenCV/Rerun 习惯的 [高, 宽, 通道] 格式字节数组。

    参数:
        chw_float32_torch (torch.Tensor) -> 形状为 (C, H, W)，值范围 0.0-1.0 的浮点张量。

    返回:
        hwc_uint8_numpy (np.ndarray) -> 形状为 (H, W, C)，值范围 0-255 的 uint8 数组。

    异常:
        AssertionError: 当输入张量的数据类型不是float32、维度不是3维或通道数不符合要求时抛出
    """
    # 验证输入张量的数据类型为float32
    assert chw_float32_torch.dtype == torch.float32
    # 验证输入张量为3维张量
    assert chw_float32_torch.ndim == 3
    # 获取张量形状并验证为通道优先格式
    c, h, w = chw_float32_torch.shape
    # 确保第一个维度是通道数（通道数应该小于高度和宽度）
    assert (
        c < h and c < w
    ), f"expect channel first images, but instead {chw_float32_torch.shape}"

    # 执行转换：缩放数值范围到[0,255] -> 转换数据类型为uint8 -> 重新排列维度顺序为HWC -> 转换为numpy数组
    # .type(torch.uint8)：将数据类型从 float32 转换为 uint8
    # .permute(1, 2, 0)：重新排列张量维度顺序，从 CHW 转换为 HWC
    hwc_uint8_numpy = (
        (chw_float32_torch * 255).type(torch.uint8).permute(1, 2, 0).numpy()
    )
    return hwc_uint8_numpy


def visualize_dataset(
    dataset: LeRobotDataset,
    episode_index: int,
    batch_size: int = 32,
    num_workers: int = 0,
    mode: str = "local",
    web_port: int = 9090,
    grpc_port: int = 9876,
    save: bool = False,
    output_dir: Path | None = None,
    display_compressed_images: bool = False,
    **kwargs,
) -> Path | None:

    # step1 参数校验（save 模式必须指定 output_dir） │
    if save:
        assert (
            output_dir is not None
        ), "Set an output directory where to write .rrd files with `--output-dir path/to/directory`."

    repo_id = dataset.repo_id

    # step2 创建 DataLoader 加载数据
    logging.info("Loading dataloader")
    # * DataLoader 是 PyTorch 内置的类，用于批量加载数据的工具，可以自动将数据集分割成多个 batch，方便批量处理
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
    )  # *返回的 dataloader 本质上是一个 iterator 对象，可以用 for 循环遍历，如：for batch in dataloader:
    print(len(dataloader))  # 1500 / 32 = 46.875 ≈ 47

    # step3 初始化 Rerun 服务
    logging.info("Starting Rerun")

    if mode not in ["local", "distant"]:
        raise ValueError(mode)

    # 判断是否需要在本地直接弹出可视化窗口（非保存模式且为 local 时弹出）
    spawn_local_viewer = mode == "local" and not save
    # 初始化 Rerun 会话，设置唯一的 App ID
    rr.init(f"{repo_id}/episode_{episode_index}", spawn=spawn_local_viewer)

    # 强制垃圾回收（解决多进程内存泄漏问题）
    # Manually call python garbage collector after `rr.init` to avoid hanging in a blocking flush
    # when iterating on a dataloader with `num_workers` > 0
    # TODO(rcadene): remove `gc.collect` when rerun version 0.16 is out, which includes a fix
    gc.collect()

    # step4 如果是 distant 模式，启动服务器
    # ? gRPC 与 Web 必须是同时启动的吗？还是只采用一种即可
    # 答：由于这两种属于两种截然不同的连接协议，所以选择其中一种来查看即可！
    # 如果在本地安装了 Rerun 客户端（推荐）：只需关注 gRPC 端口。性能最强。由于 5060 显卡支持硬件加速，本地客户端渲染最流畅，几乎无延迟。
    # 如果只想通过浏览器查看（免安装）：则必须通过 Web 端口。最方便。无需安装任何软件，但浏览器解析 Wasm 性能有限，看高分辨率视频会卡顿。
    if mode == "distant":
        server_uri = rr.serve_grpc(grpc_port=grpc_port)  # 开启底层数据传输服务
        logging.info(
            f"Connect to a Rerun Server: rerun rerun+http://IP:{grpc_port}/proxy"
        )

        # 同时也开启 Web 端的查看器服务
        rr.serve_web_viewer(
            open_browser=False,
            web_port=web_port,
            connect_to=server_uri,
        )

    logging.info("Logging to Rerun")

    # step5 遍历 DataLoader，记录每一帧数据
    # first_index: 第一帧的索引（用于归一化 frame_index），用于对齐时间轴，让当前 episode 从第 0 帧开始显示
    first_index = None

    # 遍历所有 batch。整个 dataloader 的帧索引为 15000~16499，每个 batch 包含 batch_size=32 个数据帧，其中某索引可能 15000~15031
    for batch in tqdm.tqdm(dataloader, total=len(dataloader)):
        print(f"batch type: ", type(batch))  # <class 'dict'>
        print(f"dataloader length: ", len(dataloader))  # 47

        # 记录本回合的第一帧索引
        if first_index is None:
            # * a 本身是 1 个 1 维 Tensor，里面包含 32 个标量数值
            a = batch["index"]
            print(a)  # 拿到 index 字段的 Tensor，Tensor([15000, 15001, ..., 15031])
            print(f"a.type: {type(a)}")  # a.type: <class 'torch.Tensor'>
            print(f"a.ndim = {a.ndim}")  # a.ndim = 1
            print(f"a.shape = {a.shape}")  # a.shape = torch.Size([32])

            a0 = batch["index"][0]
            print(a0)  # 拿到这个 Tensor 的第 0 个元素，tensor(15000)
            print(f"a0.type: {type(a0)}")  # a0.type: <class 'torch.Tensor'>
            print(f"a0.ndim = {a0.ndim}")  # a0.ndim = 0
            print(f"a0.shape = {a0.shape}")  # a0.shape = torch.Size([])，这是标量！

            a0_0 = batch["index"][0].item()
            print(a0_0)  # 将 scalar Tensor 转为 Python 纯数值，15000
            print(f"a0_0.type: {type(a0_0)}")  # a0_0.type: <class 'int'>
            first_index = batch["index"][0].item()

        # 遍历 batch 中 32 帧中的的每一帧
        for i in range(len(batch["index"])):
            # 设置 Rerun 里的当前帧编号（用于拖动进度条），减去 first_index 来做偏移
            # 效果：
            #     全局索引	    减去15000后	    显示的帧编号
            #      15000	        0	         第0帧
            #      15001	        1	         第1帧
            #      15002	        2	         第2帧
            #      ...	           ...	          ...
            #      15031	        31	         第31帧
            rr.set_time("frame_index", sequence=batch["index"][i].item() - first_index)
            # 设置 Rerun 里的精确时间戳（单位为: 秒）
            rr.set_time("timestamp", timestamp=batch["timestamp"][i].item())

            # A. 遍历该数据集包含的所有相机，并将图像记录到 Rerun
            for key in dataset.meta.camera_keys:
                img = to_hwc_uint8_numpy(batch[key][i])
                img_entity = (rr.Image(img).compress() if display_compressed_images else rr.Image(img))
                # 记录相机图像
                rr.log(key, entity=img_entity)

            # B. 处理动作（控制指令），每一维记录为一个标量曲线
            if ACTION in batch:
                for dim_idx, val in enumerate(batch[ACTION][i]):
                    # 记录动作曲线
                    rr.log(f"{ACTION}/{dim_idx}", rr.Scalars(val.item()))

            # C. 如果数据集中包含观测状态（如关节角），也记录为标量曲线
            if OBS_STATE in batch:
                for dim_idx, val in enumerate(batch[OBS_STATE][i]):
                    # 记录状态曲线
                    rr.log(f"state/{dim_idx}", rr.Scalars(val.item()))

            # D. 记录其他可选的训练指标（如完成标志、奖励、成功标志）
            if DONE in batch:
                rr.log(DONE, rr.Scalars(batch[DONE][i].item()))

            if REWARD in batch:
                rr.log(REWARD, rr.Scalars(batch[REWARD][i].item()))

            if "next.success" in batch:
                rr.log("next.success", rr.Scalars(batch["next.success"][i].item()))

    # step6 结果输出与服务维持
    # 如果是本地保存模式，将录制好的内容存为 .rrd 文件
    if mode == "local" and save:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        repo_id_str = repo_id.replace("/", "_")
        rrd_path = output_dir / f"{repo_id_str}_episode_{episode_index}.rrd"
        rr.save(rrd_path)
        return rrd_path

    # 如果是远程模式，保持进程运行直到手动中断 (Ctrl+C)
    elif mode == "distant":
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Ctrl-C received. Exiting.")


def main():
    """
    主函数，用于解析命令行参数并启动数据集可视化流程。

    该函数通过 argparse 解析用户输入的参数，加载指定 Hugging Face 仓库中的 LeRobotDataset 数据集，
    并调用 visualize_dataset 函数进行可视化或保存为 .rrd 文件。

    参数说明：
        --repo-id (str, 必须): Hugging Face 数据集仓库 ID (如 lerobot/pusht)。
        --episode-index (int, 必须): 要可视化的 episode 索引。
        --root (Path, 可选): 本地数据集存储根目录。默认为 None, 表示从 Hugging Face 缓存或在线下载。
        --output-dir (Path, 可选): 当启用保存模式时，指定输出 .rrd 文件的目录路径。
        --batch-size (int, 可选): DataLoader 加载数据的批次大小，默认为 32。
        --num-workers (int, 可选): DataLoader 使用的工作进程数，默认为 4。
        --mode (str, 可选): 可视化模式，支持 "local" 或 "distant"，默认为 "local"。
        --web-port (int, 可选): 在 distant 模式下使用的 Web 端口，默认为 9090。
        --ws-port (int, 可选): 已废弃，由 --grpc-port 参数替代。
        --grpc-port (int, 可选): 在 distant 模式下使用的 gRPC 服务的端口号，默认为 9876。
        --save (int, 可选): 是否将可视化结果保存为 .rrd 文件 (1 表示保存），默认为 0。
        --tolerance-s (float, 可选): 时间戳校验容忍度（秒），默认为 1e-4。

    注意事项：
        - 若使用 Windows 系统，建议设置 --num-workers=0 避免多进程问题。
        - 启用 --save=1 将不会弹出本地 viewer。
    """
    parser = argparse.ArgumentParser()

    # 添加命令行参数：DataLoader 每次加载的批量大小（越大加载越快，但需匹配内存）
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size loaded by DataLoader.",
    )
    # 添加命令行参数：DataLoader 的工作线程数量（0 表示主线程加载，>0 多进程加速，Windows 建议设 0）。
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,  # Windows、单机笔记本建议设为 0，Linux、多核服务器可以设为 >0
        help="Number of processes of Dataloader for loading the data.",
    )
    # 添加命令行参数：Hugging Face 数据集仓库 ID（如 lerobot/pusht）
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Name of hugging face repository containing a LeRobotDataset dataset (e.g. `lerobot/pusht`).",
    )
    # 添加命令行参数：要加载的 episode 索引（减少内存占用）。
    parser.add_argument(
        "--episode-index",
        type=int,
        required=True,
        help="Episode to visualize.",
    )
    # 添加命令行参数：本地数据集根目录（若为 None，从 Hugging Face 缓存或仓库下载）。
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Root directory for the dataset stored locally (e.g. `--root data`). By default, the dataset will be loaded from hugging face cache folder, or downloaded from the hub if available.",
    )
    # 添加命令行参数：保存 .rrd 文件的目标目录（save=True 时必传）。
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory path to write a .rrd file when `--save 1` is set.",
    )
    # 添加命令行参数：可视化模式（local=本地弹出 viewer、distant=远程 websocket 服务）
    parser.add_argument(
        "--mode",
        type=str,
        default="local",
        help=(
            "Mode of viewing between 'local' or 'distant'. "
            "'local' requires data to be on a local machine. It spawns a viewer to visualize the data locally. "
            "'distant' creates a server on the distant machine where the data is stored. "
            "Visualize the data by connecting to the server with `rerun rerun+http://IP:GRPC_PORT/proxy` on the local machine."
        ),
    )
    # ! 添加命令行参数：远程模式下的 HTTP Web 服务器端口（仅 mode="distant" 生效）。通信协议为 gRPC（二进制）
    parser.add_argument(
        "--web-port",
        type=int,
        default=9090,
        help="Web port for rerun.io when `--mode distant` is set.",
    )
    # ! 已弃用，同 grpc-port
    parser.add_argument(
        "--ws-port",
        type=int,
        help="deprecated, please use --grpc-port instead.",
    )
    # ! 添加命令行参数：远程模式下的 gRPC 服务器端口（仅 mode="distant" 生效）。通信协议为 HTTP/WebSocket
    parser.add_argument(
        "--grpc-port",
        type=int,
        default=9876,
        help="gRPC port for rerun.io when `--mode distant` is set.",
    )

    # 添加命令行参数：是否保存为 .rrd 文件（保存后不弹出 viewer）。
    parser.add_argument(
        "--save",
        type=int,
        default=0,
        help=(
            "Save a .rrd file in the directory provided by `--output-dir`. "
            "It also deactivates the spawning of a viewer. "
            "Visualize the data by running `rerun path/to/file.rrd` on your local machine."
        ),
    )

    # 添加命令行参数：时间戳容差（检查数据集中的时间戳是否符合指定的 FPS，确保数据帧率合规，默认 1e-4 秒）。
    parser.add_argument(
        "--tolerance-s",
        type=float,
        default=1e-4,
        help=(
            "Tolerance in seconds used to ensure data timestamps respect the dataset fps value"
            "This is argument passed to the constructor of LeRobotDataset and maps to its tolerance_s constructor argument"
            "If not given, defaults to 1e-4."
        ),
    )

    # 添加命令行参数：是否显示压缩图像（而不是未压缩图像），若不指定，则为高清原图，否则进行有损压缩
    parser.add_argument(
        "--display-compressed-images",
        action="store_true",
        help="If set, display compressed images in Rerun instead of uncompressed ones.",
    )

    # 解析命令行参数
    args = parser.parse_args()
    kwargs = vars(args)  # 👉 把 Namespace 对象 转成 dict 对象

    # ? 这里为什么要 pop 出来？因为下面的 dataset 实例化时要传入 repo_id、root、tolerance_s 三个变量，
    # 而 visualize_dataset(dataset, **vars(args)) 中的 **vars(args) 也要传入这三个变量，所以只能在一个里面传入，同时传入会冲突！
    repo_id = kwargs.pop("repo_id")
    root = kwargs.pop("root")
    tolerance_s = kwargs.pop("tolerance_s")

    if kwargs["ws_port"] is not None:
        logging.warning(
            "--ws-port is deprecated and will be removed in future versions. Please use --grpc-port instead."
        )
        logging.warning("Setting grpc_port to ws_port value.")
        kwargs["grpc_port"] = kwargs.pop("ws_port")

    init_logging()

    logging.info("Loading dataset")

    # 加载指定 episode 的 LeRobotDataset 数据集
    # 实例化数据集：注意 episodes=[args.episode_index] 参数，这意味着不会加载完整数据集，只加载选择的那个 episode。
    # ? 如果不选择默认加载哪个？答：好像是必传参数
    dataset = LeRobotDataset(
        repo_id,
        episodes=[args.episode_index],  # 只加载指定 episode
        root=root,
        tolerance_s=tolerance_s,
    )

    print(f"# Using Args: ")
    pprint.pprint(args)

    # 启动可视化流程
    visualize_dataset(dataset, **vars(args))


if __name__ == "__main__":
    main()
