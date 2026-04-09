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
This example demonstrates how to use image transforms with LeRobot datasets for data augmentation during training.

Image transforms are applied to camera frames to improve model robustness and generalization. They are applied
at training time only, not during dataset recording, allowing you to experiment with different augmentations
without re-recording data.

作用：展示如何对数据集中的图像进行动态变换。

核心功能：
    Data Augmentation: 包括随机裁剪、旋转、亮度调节、噪声注入等;
    标准化：将图片从 480p 缩放到模型需要的尺寸 (如 224 x 224);
"""

import dotenv

dotenv.load_dotenv(".env")

import torch
from torchvision.transforms import v2
from torchvision.transforms.functional import to_pil_image

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.transforms import (
    ImageTransformConfig,  # 单个变换的配置类
    ImageTransforms,  # 变换的执行类
    ImageTransformsConfig,  # 整体变换组合的配置类
)


def save_image(tensor, filename):
    """
    Helper function to save a tensor as an image file.
    """
    # 数据的形状预期是 [C, H, W]，即 [通道, 高, 宽]
    if tensor.dim() == 3:
        if tensor.max() > 1.0:
            tensor = tensor / 255.0
        tensor = torch.clamp(tensor, 0.0, 1.0)  # 裁剪到 0-1 之间，防止溢出
        pil_image = to_pil_image(tensor)  # 转换为 PIL 图片格式并保存
        pil_image.save(filename)
        print(f"Saved: {filename}")
    else:
        print(f"Skipped {filename}: unexpected tensor shape {tensor.shape}")


def example_1_default_transforms():
    """
    Example 1: Use default transform configuration and save original vs transformed images
    """

    print("\n # Example 1: Default Transform Configuration with Image Saving")

    repo_id = "pepijn223/record_main_0"  # Example dataset

    try:
        # step1 Load dataset without transforms (original)
        dataset_original = LeRobotDataset(repo_id=repo_id)

        # step2 Load dataset with transforms enabled
        transforms_config = ImageTransformsConfig(
            enable=True,  # 必须手动设为 True，默认是不开启的
            max_num_transforms=2,  # 每次读取一帧时，随机从池子里选 2 个变换应用，如"brightness"、"hue"等
            random_order=False,  # 是否随机化变换的顺序，如先模糊再缩小，和先缩小再模糊，结果像素分布是不一样的
        )

        # step3 Load dataset with transforms (transformed)
        dataset_with_transforms = LeRobotDataset(
            repo_id=repo_id, image_transforms=ImageTransforms(transforms_config)
        )

        # 保存原始图片和转换后的图片，来进行对比
        if len(dataset_original) > 0:
            frame_idx = 0  # 选择第一帧（索引为0）作为样本进行比较

            # 获取原始数据集中指定帧的数据
            original_sample = dataset_original[frame_idx]
            # 获取原始数据集中指定帧的数据
            transformed_sample = dataset_with_transforms[frame_idx]

            print(f"Saving comparison images (frame {frame_idx}):")

            # 遍历数据集元数据中的所有相机键（每个键代表一个摄像头的图像）
            for cam_key in dataset_original.meta.camera_keys:
                # 检查原始样本和变换后样本中都存在该相机键对应的图像
                if cam_key in original_sample and cam_key in transformed_sample:
                    # 处理相机键名称，替换其中的特殊字符（"."和"/"）为下划线，这样可以避免文件命名中的问题
                    cam_name = cam_key.replace(".", "_").replace("/", "_")

                    # 保存原始图像，文件名为"相机名_original.png"
                    save_image(original_sample[cam_key], f"{cam_name}_original.png")
                    # 保存变换后的图像，文件名为"相机名_transformed.png"
                    save_image(
                        transformed_sample[cam_key], f"{cam_name}_transformed.png"
                    )

    except Exception as e:
        print(f"Could not load dataset '{repo_id}': {e}")


def example_2_custom_transforms():
    """
    Example 2: Create custom transform configuration and save examples
    """

    print("\n # Example 2: Custom Transform Configuration")

    repo_id = "pepijn223/record_main_0"  # Example dataset

    try:
        # Create custom transform configuration with strong effects
        custom_transforms_config = ImageTransformsConfig(
            enable=True,
            max_num_transforms=2,  # 每帧最多应用 2 个变换
            random_order=True,  # Apply transforms in random order
            tfs={
                "brightness": ImageTransformConfig(
                    weight=1.0,  # * 权重越大，被选中的概率越高
                    type="ColorJitter",  # 对应 torchvision 的 ColorJitter
                    kwargs={"brightness": (0.5, 1.5)},  # 亮度变化范围，0.5 变为亮度一半
                ),
                "contrast": ImageTransformConfig(
                    weight=1.0,  # Higher weight = more likely to be selected
                    type="ColorJitter",
                    kwargs={"contrast": (0.6, 1.4)},  # Strong contrast
                ),
                "sharpness": ImageTransformConfig(
                    weight=0.5,  # Lower weight = less likely to be selected
                    type="SharpnessJitter",
                    kwargs={"sharpness": (0.2, 2.0)},  # Strong sharpness variation
                ),
            },
        )

        dataset_with_custom_transforms = LeRobotDataset(
            repo_id=repo_id, image_transforms=ImageTransforms(custom_transforms_config)
        )

        # Save examples with strong transforms
        if len(dataset_with_custom_transforms) > 0:
            sample = dataset_with_custom_transforms[0]  # 采样第一帧
            print("Saving custom transform examples:")

            for cam_key in dataset_with_custom_transforms.meta.camera_keys:
                if cam_key in sample:
                    cam_name = cam_key.replace(".", "_").replace("/", "_")
                    save_image(sample[cam_key], f"{cam_name}_custom_transforms.png")

    except Exception as e:
        print(f"Could not load dataset '{repo_id}': {e}")


def example_3_torchvision_transforms():
    """
    Example 3: Use pure torchvision transforms and save examples
    """

    print("\n # Example 3: Pure Torchvision Transforms")

    repo_id = "pepijn223/record_main_0"  # Example dataset

    try:
        # 使用 v2.Compose 创建一个标准的 torchvision 变换 Pipeline
        # Compose 的作用是将多个变换操作按顺序“串联”起来
        torchvision_transforms = v2.Compose(
            [
                # 1. ColorJitter: 颜色抖动
                # 随机改变图像的亮度(0.3)、对比度(0.3)、饱和度(0.3)和色调(0.1)，模拟不同光照环境（如室内灯光强弱、色温变化）
                v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                # 2. GaussianBlur: 高斯模糊
                # 使用大小为 3x3 的卷积核，模糊程度(sigma)在 0.1 到 2.0 之间随机，模拟相机对焦不准、镜头有污渍或机器人高速运动产生的运动模糊
                v2.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
                # 3. RandomRotation: 随机旋转
                # 在正负 10 度之间随机旋转图像，模拟相机安装时的微小角度偏差或机器人底盘的不平稳
                v2.RandomRotation(degrees=10),  # Small rotation
            ]
        )

        dataset_with_torchvision = LeRobotDataset(
            repo_id=repo_id, image_transforms=torchvision_transforms
        )

        # Save examples with torchvision transforms
        if len(dataset_with_torchvision) > 0:
            # ! 因为这里没有传 delta_timestamps，所以 LeRobot 默认只取当前这一帧。就不存在 时间戳对齐的问题，所以也就没有 xxx_is_pad 的键值对
            sample = dataset_with_torchvision[0]
            print("Saving torchvision transform examples:")

            for cam_key in dataset_with_torchvision.meta.camera_keys:
                if cam_key in sample:  # ['observation.images.front']
                    cam_name = cam_key.replace(".", "_").replace("/", "_")
                    save_image(sample[cam_key], f"{cam_name}_torchvision.png")

    except Exception as e:
        print(f"Could not load dataset '{repo_id}': {e}")


def main():
    """
    Run all examples
    """
    print("LeRobot Dataset Image Transforms Examples")

    # example_1_default_transforms()
    # example_2_custom_transforms()
    example_3_torchvision_transforms()


if __name__ == "__main__":
    main()
