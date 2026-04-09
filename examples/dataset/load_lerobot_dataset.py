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
This script demonstrates the use of `LeRobotDataset` class for handling and processing robotic datasets from Hugging Face.
It illustrates how to load datasets, manipulate them, and apply transformations suitable for machine learning tasks in PyTorch.

Features included in this script:
- Viewing a dataset's metadata and exploring its properties.
- Loading an existing dataset from the hub or a subset of it.
- Accessing frames by episode number.
- Using advanced dataset features like timestamp-based frame selection.
- Demonstrating compatibility with PyTorch DataLoader for batch processing.

The script ends with examples of how to batch process data using PyTorch's DataLoader.
"""

import dotenv

dotenv.load_dotenv("/home/robot/lerobot/.env")

from pprint import pprint

import torch
from huggingface_hub import HfApi

import huggingface_hub

print(huggingface_hub.__version__)  # 1.8.0

import lerobot
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# * 此脚本演示了如何处理机器人数据，特别是如何处理带有时间序列（History/Future）特征的数据。


def main():
    # step1 查看可用数据集清单
    # We ported a number of existing datasets ourselves, use this to see the list:
    print("List of available datasets:")
    pprint(lerobot.available_datasets)

    # step2 从 Hugging Face Hub 搜索社区数据集
    # You can also browse through the datasets created/ported by the community on the hub using the hub api:
    hub_api = (
        HfApi()
    )  # 初始化 Hugging Face Hub 的 API 客户端，用于调用 Hub 的数据集查询接口
    repo_ids = [
        info.id
        for info in hub_api.list_datasets(
            task_categories="robotics",
            filter=["LeRobot"],
            limit=10,  # 强制限制返回数量，减少通信时间
        )
    ]
    pprint(repo_ids)

    # Or simply explore them in your web browser directly at:
    # https://huggingface.co/datasets?other=LeRobot
    # 功能：提供 Hub 数据集的网页链接，用户可通过浏览器可视化浏览数据集详情（如描述、示例、使用说明）

    # step3 加载数据集的元数据 (Metadata)
    print("# Loading the dataset meta...")
    # Let's take this one for this example
    repo_id = "lerobot/aloha_mobile_cabinet"
    # 实例化一个数据集元数据对象，仅下载数据集的元数据文件（如 dataset_info.json 配置文件），不下载图像、动作等大文件，速度极快
    # 返回 ds_meta 对象，包含数据集结构信息，详细内容见 aloha_mobile_cabinet_ds_meta.json
    ds_meta = LeRobotDatasetMetadata(repo_id)

    # By instantiating just this class, you can quickly access useful information about the content and the
    # structure of the dataset without downloading the actual data yet (only metadata files — which are
    # lightweight).
    print(f"Total number of episodes: {ds_meta.total_episodes}")  # 85
    print(
        f"Average number of frames per episode: {ds_meta.total_frames / ds_meta.total_episodes:.3f}"
    )  # 1500.000
    print(f"Frames per second used during data collection: {ds_meta.fps}")  # 50
    print(f"Robot type: {ds_meta.robot_type}")  # aloha
    print(
        f"keys to access images from cameras: {ds_meta.camera_keys=}\n"
    )  # List[str, str, str], ds_meta.camera_keys = ["observation.images.cam_high", "observation.images.cam_left_wrist", "observation.images.cam_right_wrist"]

    print("Tasks:")
    print(ds_meta.tasks)

    # * 比较重要！数据集的所有模态字段（如 action 动作、observation.state 关节状态、摄像头图像）及格式（如形状、数据类型）
    print("Features:")
    pprint(ds_meta.features)

    # You can also get a short summary by simply printing the object:
    print("# ds_meta:")
    print(ds_meta)
    # LeRobotDatasetMetadata({
    #     Repository ID: 'lerobot/aloha_mobile_cabinet',
    #     Total episodes: '85',
    #     Total frames: '127500',
    #     Features: '['observation.images.cam_high', 'observation.images.cam_left_wrist', 'observation.images.cam_right_wrist', 'observation.state', 'observation.effort', 'action', 'episode_index', 'frame_index', 'timestamp', 'next.done', 'index', 'task_index']',
    # })',

    print("# Loading the dataset...")
    # step4 加载实际数据集 (Dataset)
    # case1 加载特定 episode（只下载第 0, 10, 11, 23 回合的数据）：
    # dataset = LeRobotDataset(repo_id, episodes=[0, 10, 11, 23], download_videos=True)

    # print(f"Selected episodes: {dataset.episodes}")  # [0, 10, 11, 23]
    # print(f"Number of episodes selected: {dataset.num_episodes}")  # 4，选中的回合数
    # print(f"Number of frames selected: {dataset.num_frames}")  # 6000，总帧数

    # case2 加载完整数据集（生产环境常用）：
    dataset = LeRobotDataset(repo_id)
    print(f"Number of episodes selected: {dataset.num_episodes}")  # 85，选中的回合数
    print(f"Number of frames selected: {dataset.num_frames}")  # 127500，总帧数

    # The previous metadata class is contained in the 'meta' attribute of the dataset:
    print("# Dataset metadata:")
    print(dataset.meta)
    # LeRobotDatasetMetadata({
    #     Repository ID: 'lerobot/aloha_mobile_cabinet',
    #     Total episodes: '85',
    #     Total frames: '127500',
    #     Features: '['observation.images.cam_high', 'observation.images.cam_left_wrist', 'observation.images.cam_right_wrist', 'observation.state', 'observation.effort', 'action', 'episode_index', 'frame_index', 'timestamp', 'next.done', 'index', 'task_index']',
    # }),

    # You can inspect the dataset using its repr:
    print("# Dataset:")
    print(dataset)
    # LeRobotDataset({
    #     Repository ID: 'lerobot/aloha_mobile_cabinet',
    #     Number of selected episodes: '85',
    #     Number of selected samples: '127500',
    #     Features: '['observation.images.cam_high', 'observation.images.cam_left_wrist', 'observation.images.cam_right_wrist', 'observation.state', 'observation.effort', 'action', 'episode_index', 'frame_index', 'timestamp', 'next.done', 'index', 'task_index']',
    # })

    # step5 访问具体帧数据
    # LeRobotDataset 是按“帧”索引的。我们可以找到第一回合的起止索引：
    # LeRobot datasets also subclasses PyTorch datasets so you can do everything you know and love from working with the latter, like iterating through the dataset.
    # The __getitem__ iterates over the frames of the dataset. Since our datasets are also structured by episodes, you can access the frame indices of any episode using dataset.meta.episodes. Here, we access
    # frame indices associated to the first episode:
    episode_index = 0
    # dataset.meta.episodes["dataset_from_index"]：这部分返回的是一个长列表（或数组），它的长度等于 total_episodes（在你的例子里是 85）
    print(type(dataset.meta.episodes))  # <class 'datasets.arrow_dataset.Dataset'>
    print(dataset.meta.episodes.shape)  # (85, 21)
    print(
        dataset.meta.episodes["dataset_from_index"]
    )  # Column([0, 1500, 3000, 4500, 6000, ...])
    # ...[episode_index]：是对上面那个列表进行取值，即如果想看第 0 个回合，就取索引 0
    # 元数据中存储每个 episode 的帧索引范围（起始帧 ~ 结束帧的下一个索引）
    from_idx = dataset.meta.episodes["dataset_from_index"][episode_index]  # 0
    to_idx = dataset.meta.episodes["dataset_to_index"][episode_index]  # 1500

    # 获取第一回合的所有图像帧
    # Then we grab all the image frames from the first camera:
    camera_key = dataset.meta.camera_keys[0]  # 'observation.images.cam_high'
    # * dataset[idx] 返回的是一个 Dict[str, torch.Tensor] 类型的单个样本，此时是 LeRobotDataset 类处理后的标准输出结果，用于专门喂给神经网络
    print(type(dataset[5]))
    print(dataset[5].keys())
    # dict_keys(
    #     [
    #         "observation.images.cam_high",
    #         "observation.images.cam_left_wrist",
    #         "observation.images.cam_right_wrist",
    #         "observation.state",
    #         "observation.effort",
    #         "action",
    #         "episode_index",
    #         "frame_index",
    #         "timestamp",
    #         "next.done",
    #         "index",
    #         "task_index",
    #         "task",
    #     ]
    # )

    # 列表推导式，返回一个 1500 帧的 list，每个元素都是一个 tensor，shape = torch.Size([3, 480, 640])
    frames = [dataset[idx][camera_key] for idx in range(from_idx, to_idx)]

    # The objects returned by the dataset are all torch.Tensors
    # * PyTorch 默认是 Channel-first (通道在前)
    print(type(frames[0]))  # <class 'torch.Tensor'>
    print(frames[0].shape)  # (C, H, W) = torch.Size([3, 480, 640])

    # Since we're using pytorch, the shape is in pytorch, channel-first convention (c, h, w).
    # We can compare this shape with the information available for that feature
    pprint(dataset.features[camera_key])
    # {
    #     "dtype": "video",
    #     "names": ["height", "width", "channel"],
    #     "shape": (480, 640, 3),
    #     "video_info": {
    #         "has_audio": False,
    #         "video.codec": "av1",
    #         "video.fps": 50.0,
    #         "video.is_depth_map": False,
    #         "video.pix_fmt": "yuv420p",
    #     },
    # }

    # In particular:
    print(dataset.features[camera_key]["shape"])  # (480, 640, 3) = (h, w, c)
    # The shape is in (h, w, c) which is a more universal format.

    # step6 高级功能：加载时间序列数据 (Temporal Data)
    # * 具身智能核心：模型通常需要看“过去几帧”来决定“未来一串动作”
    # delta_timestamps 定义了相对于“当前帧”的时间偏移（单位：秒）
    # 机器人学习中，单帧图像往往无法表达物体的运动趋势（Velocity）。通过设置 [-0.1, 0]，模型可以同时获得前一时刻和当前时刻，从而感知“动量”。
    # For many machine learning applications we need to load the history of past observations or trajectories of
    # future actions. Our datasets can load previous and future frames for each key/modality, using timestamps
    # differences with the current loaded frame. For instance:
    # 定义每个模态需要加载的 “时序偏移”（单位：秒），用于获取当前帧的历史或未来数据，适配机器人时序建模需求（如论文 3.1.5 节动作块预测）
    delta_timestamps = {
        # 加载 4 张图像：取当前时刻、前 0.2s、前 0.5s、前 1s 的 4 张图
        camera_key: [-1, -0.5, -0.20, 0],
        # 加载 6 个状态向量：取当前帧、前0.1s、前 0.2s、前 0.5s、前 1s、前 1.5s 的 6 个状态向量
        "observation.state": [-1.5, -1, -0.5, -0.20, -0.10, 0],
        # 加载 34 个动作向量：取从现在开始往后 64 帧（当前帧、1 帧、2 帧...63 帧）的预选动作序列 (用于 Action Chunking)
        "action": [t / dataset.fps for t in range(64)],
    }
    # Note that in any case, these delta_timestamps values need to be multiples of (1/fps) so that added to any
    # timestamp, you still get a valid timestamp. 注意：偏移量必须是 1/fps 的整数倍（如 30 FPS 下，最小偏移是 1/30 ≈ 0.033 秒），确保时序对齐

    # 重新实例化数据集对象，此时会添加 delta_timestamps 所指定的数据
    dataset = LeRobotDataset(repo_id, delta_timestamps=delta_timestamps)
    # 由于设置了 delta_timestamps（假设取了 4 个时间点），它返回的是 (4, 3, 480, 640)。这是一个包含 4 帧连续图像的小片段。

    print(dataset)
    ds_len = len(dataset)
    print(ds_len)  # 127500, dataset 果然存放的是每一帧！
    print(type(dataset))  # <class 'lerobot.datasets.lerobot_dataset.LeRobotDataset'>
    pprint(dataset.features)
    # 打印对象当前所有属性的名值对（注意：输出可能非常多，建议配合断点查看）
    pprint(vars(dataset))
    # 列出 dataset 对象所有的属性和方法（包括带下划线的隐藏成员）
    pprint(dir(dataset))

    # * dataset[5] 返回的是一个包含这一帧（以及相关联的时间窗口）所有信息的字典
    # 原因: 深度学习训练（尤其是随机 Shuffle 时）需要的是“以帧为中心的采样”。模型在学习时，通常是看当前的图像 + 之前的几帧，来预测下一步的动作。
    # ! 这里的 keys()会多出来 3 个，是由于 delta_timestamps 为对齐时间戳而补的 xxx_is_pad，其中 True 表示这一帧是复制出来的填充数据，而 False 表示是原始真实的传感器数据
    sample = dataset[5]
    print(sample)  # 触发👉(return) LeRobotDataset.__getitem__()
    print(type(dataset[5]))  # <class 'dict'>
    pprint(dataset[5].keys())

    # 图像形状：(4, C, H, W) -> 4 代表 4 帧图像堆叠
    print(f"\n{dataset[0][camera_key].shape=}")
    # dataset[0][camera_key].shape=torch.Size([4, 3, 480, 640])

    # 状态形状：(6, C) -> 6 帧状态，C 是状态维度
    print(f"{dataset[0]['observation.state'].shape=}")
    # dataset[0]['observation.state'].shape=torch.Size([6, 14])

    # 动作形状：(64, C) -> 64 步未来动作，C 是动作维度 (如 6 自由度+夹爪)
    print(f"{dataset[0]['action'].shape=}\n")
    # dataset[0]['action'].shape=torch.Size([64, 14])

    # step7 使用 DataLoader 进行批量处理
    # 从 dataset 里随机抓了 32 个这样的“小片段”
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=0,  # 4 改为 0，不启动子进程
        batch_size=32,
        shuffle=True,
    )
    # dataloader: <torch.utils.data.dataloader.DataLoader object at 0x73520081f6b0>
    for batch in dataloader:
        dl_len = len(dataloader)
        print(dl_len)  # 3985，表示 127500 个数据被分为了 3985 个批次

        # batch[key] 形状：(Batch_Size, Temporal_Steps, Dimensions...)
        print(f"{batch[camera_key].shape=}")  # (32, 4, c, h, w)
        # batch[camera_key].shape=torch.Size([32, 4, 3, 480, 640])

        print(f"{batch['observation.state'].shape=}")  # (32, 6, c)
        # batch['observation.state'].shape=torch.Size([32, 6, 14])

        print(f"{batch['action'].shape=}")  # (32, 64, c)
        # batch["action"].shape = torch.Size([32, 64, 14])

        break


if __name__ == "__main__":
    main()
