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
Helper to find the camera devices available in your system.

Example:

```shell
lerobot-find-cameras
```
"""

# NOTE(Steven): RealSense can also be identified/opened as OpenCV cameras. If you know the camera is a RealSense, use the `lerobot-find-cameras realsense` flag to avoid confusion.
# NOTE(Steven): macOS cameras sometimes report different FPS at init time, not an issue here as we don't specify FPS when opening the cameras, but the information displayed might not be truthful.

import argparse
import concurrent.futures
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from lerobot.cameras.configs import ColorMode
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.realsense.camera_realsense import RealSenseCamera
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig

import pprint

pp = pprint.PrettyPrinter(indent=2)

logger = logging.getLogger(__name__)


def find_all_opencv_cameras() -> list[dict[str, Any]]:
    """
    Finds all available OpenCV cameras plugged into the system.

    Returns:
        A list of all available OpenCV cameras with their metadata.
    """
    all_opencv_cameras_info: list[dict[str, Any]] = []
    logger.info("Searching for OpenCV cameras...")
    try:
        opencv_cameras = OpenCVCamera.find_cameras()
        for cam_info in opencv_cameras:
            all_opencv_cameras_info.append(cam_info)
        logger.info(f"Found {len(opencv_cameras)} OpenCV cameras.")
    except Exception as e:
        logger.error(f"Error finding OpenCV cameras: {e}")

    return all_opencv_cameras_info


def find_all_realsense_cameras() -> list[dict[str, Any]]:
    """
    Finds all available RealSense cameras plugged into the system.

    Returns:
        A list of all available RealSense cameras with their metadata.
    """
    all_realsense_cameras_info: list[dict[str, Any]] = []
    logger.info("Searching for RealSense cameras...")
    try:
        realsense_cameras = RealSenseCamera.find_cameras()
        for cam_info in realsense_cameras:
            all_realsense_cameras_info.append(cam_info)
        logger.info(f"Found {len(realsense_cameras)} RealSense cameras.")
    except ImportError:
        logger.warning(
            "Skipping RealSense camera search: pyrealsense2 library not found or not importable."
        )
    except Exception as e:
        logger.error(f"Error finding RealSense cameras: {e}")

    return all_realsense_cameras_info


def find_and_print_cameras(
    camera_type_filter: str | None = None,
) -> list[dict[str, Any]]:
    """
    Finds available cameras based on an optional filter and prints their information. 根据可选的过滤器查找可用相机，并打印它们的详细信息。

    Args:
        camera_type_filter: Optional string to filter cameras ("realsense" or "opencv"). 可选的字符串，用于过滤相机类型（"realsense" 或 "opencv"）。
                            If None, lists all cameras. 如果为 None, 则列出探测到的所有相机。

    Returns:
        A list of all available cameras matching the filter, with their metadata. 一个包含所有符合过滤条件的相机元数据（Metadata）的字典列表。
    """
    # 初始化一个空列表，用于存放所有查找到的相机信息
    all_cameras_info: list[dict[str, Any]] = []

    # case1 如果提供了过滤器字符串，统一转换为小写，防止因大小写输入错误导致匹配失败
    if camera_type_filter:
        camera_type_filter = camera_type_filter.lower()

    # case2 如果没有过滤器，或者过滤器指定为 "opencv"，则调用 OpenCV 探测函数并将结果添加到总列表
    if camera_type_filter is None or camera_type_filter == "opencv":
        all_cameras_info.extend(find_all_opencv_cameras())
    # case3 如果没有过滤器，或者过滤器指定为 "realsense"，则调用 RealSense 探测函数并将结果添加到总列表
    if camera_type_filter is None or camera_type_filter == "realsense":
        all_cameras_info.extend(find_all_realsense_cameras())

    # 检查最终列表是否为空（即系统中没发现任何相机）
    if not all_cameras_info:
        if camera_type_filter:
            # 如果指定了类型但没搜到，打印特定类型的警告
            logger.warning(f"No {camera_type_filter} cameras were detected.")
        else:
            # 如果全面搜索也没搜到，打印通用警告
            logger.warning("No cameras (OpenCV or RealSense) were detected.")
    # 如果搜到了相机，开始在终端进行漂亮的格式化打印
    else:
        print("\n--- Detected Cameras ---")
        # 遍历相机列表，i 是索引（Camera #0, #1...），cam_info 是该相机的属性字典
        for i, cam_info in enumerate(all_cameras_info):
            print(f"Camera #{i}:")

            # 遍历单个相机信息字典中的所有键值对
            for key, value in cam_info.items():
                # 特殊处理：如果键是 "default_stream_profile"（默认流配置），它通常是一个嵌套字典
                if key == "default_stream_profile" and isinstance(value, dict):
                    # 将下划线替换为空格并首字母大写，美化输出名（例如 default_stream_profile -> Default stream profile）
                    print(f"  {key.replace('_', ' ').capitalize()}:")
                    # 递归打印嵌套字典中的子项（如 Width, Height, Fps）
                    for sub_key, sub_value in value.items():
                        print(f"    {sub_key.capitalize()}: {sub_value}")
                else:
                    # 对于普通属性（如 ID, Type），直接美化键名并打印其值
                    print(f"  {key.replace('_', ' ').capitalize()}: {value}")
            print("-" * 20)
    return all_cameras_info


def save_image(
    img_array: np.ndarray,
    camera_identifier: str | int,
    images_dir: Path,
    camera_type: str,
):
    """
    Saves a single image to disk using Pillow. Handles color conversion if necessary.
    """
    try:
        img = Image.fromarray(img_array, mode="RGB")

        safe_identifier = str(camera_identifier).replace("/", "_").replace("\\", "_")
        filename_prefix = f"{camera_type.lower()}_{safe_identifier}"
        filename = f"{filename_prefix}.png"

        path = images_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        img.save(str(path))
        logger.info(f"Saved image: {path}")
    except Exception as e:
        logger.error(
            f"Failed to save image for camera {camera_identifier} (type {camera_type}): {e}"
        )


def create_camera_instance(cam_meta: dict[str, Any]) -> dict[str, Any] | None:
    """Create and connect to a camera instance based on metadata."""
    cam_type = cam_meta.get("type")
    cam_id = cam_meta.get("id")
    instance = None

    logger.info(f"Preparing {cam_type} ID {cam_id} with default profile")

    try:
        if cam_type == "OpenCV":
            cv_config = OpenCVCameraConfig(
                index_or_path=cam_id,
                color_mode=ColorMode.RGB,
            )
            instance = OpenCVCamera(cv_config)
        elif cam_type == "RealSense":
            rs_config = RealSenseCameraConfig(
                serial_number_or_name=cam_id,
                color_mode=ColorMode.RGB,
            )
            instance = RealSenseCamera(rs_config)
        else:
            logger.warning(
                f"Unknown camera type: {cam_type} for ID {cam_id}. Skipping."
            )
            return None

        if instance:
            logger.info(f"Connecting to {cam_type} camera: {cam_id}...")
            # ! 这里报错了！ERROR:__main__:Failed to connect or configure RealSense camera 241122300596: Failed to open RealSenseCamera(241122300596).Run `lerobot-find-cameras realsense` to find available cameras.
            instance.connect(warmup=True)
            return {"instance": instance, "meta": cam_meta}
    except Exception as e:
        logger.error(f"Failed to connect or configure {cam_type} camera {cam_id}: {e}")
        if instance and instance.is_connected:
            instance.disconnect()
        return None


def process_camera_image(
    cam_dict: dict[str, Any], output_dir: Path, current_time: float
) -> concurrent.futures.Future | None:
    """Capture and process an image from a single camera."""
    cam = cam_dict["instance"]
    meta = cam_dict["meta"]
    cam_type_str = str(meta.get("type", "unknown"))
    cam_id_str = str(meta.get("id", "unknown"))

    try:
        image_data = cam.read()

        return save_image(
            image_data,
            cam_id_str,
            output_dir,
            cam_type_str,
        )
    except TimeoutError:
        logger.warning(
            f"Timeout reading from {cam_type_str} camera {cam_id_str} at time {current_time:.2f}s."
        )
    except Exception as e:
        logger.error(f"Error reading from {cam_type_str} camera {cam_id_str}: {e}")
    return None


def cleanup_cameras(cameras_to_use: list[dict[str, Any]]):
    """Disconnect all cameras."""
    logger.info(f"Disconnecting {len(cameras_to_use)} cameras...")
    for cam_dict in cameras_to_use:
        try:
            if cam_dict["instance"] and cam_dict["instance"].is_connected:
                cam_dict["instance"].disconnect()
        except Exception as e:
            logger.error(
                f"Error disconnecting camera {cam_dict['meta'].get('id')}: {e}"
            )


def save_images_from_all_cameras(
    output_dir: Path,
    record_time_s: float = 2.0,
    camera_type: str | None = None,
):
    """
    连接到检测到的相机（可选择性地按类型过滤）并从每台相机保存图像。
    使用默认流配置文件设置宽度、高度和 FPS。

    Args:
        output_dir: 保存图片的目录
        record_time_s: 记录图片的持续时间（秒）
        camera_type: 可选字符串，用于筛选相机（ "realsense" 或 "opencv" )
                            如果为 None, 则使用所有检测到的相机。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving images to {output_dir}")

    # all_camera_metadata：是一个包含多个相机元素的 list，每个元素是一个 dict，包含相机的元数据，如 name='OpenCV Camera @ /dev/video0'、type='OpenCV'、id='/dev/video0' 等信息
    all_camera_metadata = find_and_print_cameras(camera_type_filter=camera_type)

    if not all_camera_metadata:
        logger.warning("No cameras detected matching the criteria. Cannot save images.")
        return

    # 初始化要使用的相机列表
    cameras_to_use = []
    for cam_meta in all_camera_metadata:
        # 为每个相机创建实例
        camera_instance = create_camera_instance(cam_meta)

        # 如果成功创建了相机实例，将其添加到列表中
        if camera_instance:
            cameras_to_use.append(camera_instance)

    if not cameras_to_use:
        logger.warning("No cameras could be connected. Aborting image save.")
        return

    # 记录开始捕获图像的日志
    logger.info(
        f"Starting image capture for {record_time_s} seconds from {len(cameras_to_use)} cameras."
    )

    # 记录开始时间，用于计算总捕获时间
    start_time = time.perf_counter()

    print(f"# Using Cameras Numbers: {len(cameras_to_use)}")

    # * 创建一个线程池执行器，最大工作线程数为相机数量的两倍
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=len(cameras_to_use) * 2
    ) as executor:
        try:
            # 主循环：持续捕获图像直到达到指定时间
            while time.perf_counter() - start_time < record_time_s:
                # 创建一个未来对象列表，用于跟踪异步任务
                futures = []

                # 获取当前捕获时间
                current_capture_time = time.perf_counter()

                # 对于每个可用相机，处理其图像
                for cam_dict in cameras_to_use:
                    # 处理单个相机的图像捕获
                    future = process_camera_image(
                        cam_dict, output_dir, current_capture_time
                    )
                    # 如果有返回未来对象，将其添加到列表中
                    if future:
                        futures.append(future)

                # 如果有任何待完成的任务，等待它们完成
                if futures:
                    concurrent.futures.wait(futures)

        except KeyboardInterrupt:
            logger.info("Capture interrupted by user.")

        # 无论正常结束还是异常中断都会执行清理操作
        finally:
            print("\nFinalizing image saving...")

            # 关闭线程池，wait=True表示等待所有任务完成后再关闭
            executor.shutdown(wait=True)
            # 断开所有相机连接
            cleanup_cameras(cameras_to_use)
            print(f"Image capture finished. Images saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Unified camera utility script for listing cameras and capturing images."
    )

    parser.add_argument(
        "camera_type",
        type=str,
        nargs="?",
        default=None,
        choices=["realsense", "opencv"],
        help="Specify camera type to capture from (e.g., 'realsense', 'opencv'). Captures from all if omitted.",
    )

    # 图片保存目录
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="outputs/captured_images",
        help="Directory to save images. Default: outputs/captured_images",
    )

    # 6s 之后会拍照，保存到上面的 output-dir
    parser.add_argument(
        "--record-time-s",
        type=float,
        default=6.0,
        help="Time duration to attempt capturing frames. Default: 6 seconds.",
    )
    args = parser.parse_args()
    pp.pprint(args)
    # Namespace(camera_type=None, output_dir=PosixPath('outputs/captured_images'), record_time_s=6.0)

    save_images_from_all_cameras(**vars(args))


if __name__ == "__main__":
    main()
