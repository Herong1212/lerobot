#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
Example script demonstrating dataset tools utilities. --- 展示如何使用 LeRobot 提供的 API 对数据集进行检查、合并或修改。

This script shows how to:
1. Delete episodes from a dataset --- 从数据集中删除某些回合。
2. Split a dataset into train/val sets --- 划分数据集为训练集和验证集。
3. Add/remove features --- 新增或删除特征。
4. Merge datasets --- 合并多个数据集。

常见操作：
    验证完整性：检查视频帧和标签是否对齐。
    数据集转换：比如把其他格式的机器人数据转换成 LeRobot 格式。

Usage:
    python examples/dataset/use_dataset_tools.py
"""
import dotenv

dotenv.load_dotenv("/home/robot/lerobot/.env")

import numpy as np

from lerobot.datasets.dataset_tools import (
    add_features,
    delete_episodes,
    merge_datasets,
    modify_features,
    remove_feature,
    split_dataset,
)
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def main():
    repo_id = "lerobot/pusht"
    dataset = LeRobotDataset(repo_id)

    print(
        f"Original dataset: {dataset.meta.total_episodes} episodes, {dataset.meta.total_frames} frames"
    )
    # Original dataset: 206 episodes, 25650 frames

    print(f"Features: {list(dataset.meta.features.keys())}")
    # Features: ['observation.image', 'observation.state', 'action', 'episode_index', 'frame_index', 'timestamp', 'next.reward', 'next.done', 'next.success', 'index', 'task_index']

    print("\n # 1. Deleting episodes 0 and 2...")

    # delete_episodes 会生成一个新的数据集副本，存放在指定的 repo_id 路径下
    filtered_dataset = delete_episodes(
        dataset, episode_indices=[0, 2], repo_id="lerobot/pusht_filtered"
    )
    print(f"Filtered dataset: {filtered_dataset.meta.total_episodes} episodes")

    print("\n # 2. Splitting dataset into train/val...")
    
    # split_dataset 会返回一个字典，Key 是你定义的名称（train/val）
    # 它不会真正复制物理文件，而是通过索引映射来实现逻辑上的划分
    splits = split_dataset(
        dataset,
        splits={"train": 0.8, "val": 0.2},
    )
    print(
        f"Train split: {splits['train'].meta.total_episodes} episodes"
    )  # Train split: 164 episodes
    print(
        f"Val split: {splits['val'].meta.total_episodes} episodes"
    )  # Val split: 42 episodes

    print("\n # 3. Adding features...")

    # 准备数据：随机生成一组和总帧数一样长的奖励值 [Total_Frames, 1]
    reward_values = np.random.randn(dataset.meta.total_frames).astype(np.float32)

    # 准备逻辑：定义一个函数来计算“成功”. 比如：如果当前帧索引接近回合末尾，则判定为成功
    def compute_success(row_dict, episode_index, frame_index):
        episode_length = 10
        return float(frame_index >= episode_length - 10)

    # * add_features() 可以在 Parquet 表格中增加新列
    dataset_with_features = add_features(
        dataset,
        features={
            "reward": (
                reward_values,
                {"dtype": "float32", "shape": (1,), "names": None},
            ),
            "success": (
                compute_success,
                {"dtype": "float32", "shape": (1,), "names": None},
            ),
        },
        repo_id="lerobot/pusht_with_features",
    )

    print(f"New features: {list(dataset_with_features.meta.features.keys())}")

    print("\n # 4. Removing the success feature...")

    # 某些特征（比如中间计算的临时变量）不需要了，可以剔除以节省空间
    dataset_cleaned = remove_feature(
        dataset_with_features, feature_names="success", repo_id="lerobot/pusht_cleaned"
    )
    print(f"Features after removal: {list(dataset_cleaned.meta.features.keys())}")

    print("\n # 5. Using modify_features to add and remove features simultaneously...")

    # 这是最推荐的高效方法，一步到位
    dataset_modified = modify_features(
        dataset_with_features,
        add_features={
            "discount": (
                np.ones(dataset.meta.total_frames, dtype=np.float32) * 0.99,
                {"dtype": "float32", "shape": (1,), "names": None},
            ),
        },
        remove_features="reward",
        repo_id="lerobot/pusht_modified",
    )
    print(f"Modified features: {list(dataset_modified.meta.features.keys())}")

    print("\n # 6. Merging train and val splits back together...")

    # 模拟从两个不同来源的数据集拼成一个大数据集
    merged = merge_datasets(
        [splits["train"], splits["val"]], output_repo_id="lerobot/pusht_merged"
    )
    print(f"Merged dataset: {merged.meta.total_episodes} episodes")

    print("\n # 7. Complex workflow example...")

    # 比如某些模型不需要全景相机，只需要手腕相机，就可以在这里过滤
    if len(dataset.meta.camera_keys) > 1:
        camera_to_remove = dataset.meta.camera_keys[0]
        print(f"Removing camera: {camera_to_remove}")
        dataset_no_cam = remove_feature(
            dataset, feature_names=camera_to_remove, repo_id="pusht_no_first_camera"
        )
        print(f"Remaining cameras: {dataset_no_cam.meta.camera_keys}")

    print("\n # Done! Check ~/.cache/huggingface/lerobot/ for the created datasets.")


if __name__ == "__main__":
    main()
