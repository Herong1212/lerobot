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

from dataclasses import dataclass
from pathlib import Path

from ..configs import CameraConfig, ColorMode, Cv2Backends, Cv2Rotation

__all__ = ["OpenCVCameraConfig", "ColorMode", "Cv2Rotation", "Cv2Backends"]


# 1、这是一个类装饰器，由 draccus 库提供的功能
#    作用：将 OpenCVCameraConfig 注册为 CameraConfig 的子类，并赋予标识名 "opencv"
#    当配置文件中使用 type: "opencv" 时，系统会自动创建 OpenCVCameraConfig 实例
#    这是一种设计模式，允许框架根据字符串配置动态创建相应的配置类实例
#
#    例如：
#    camera_config:
#        _target_: opencv
#        index_or_path: 0
#        fps: 30
#
#    draccus 会根据 _target_: opencv 找到并实例化 OpenCVCameraConfig
# 2、dataclass 装饰器会自动根据类属性生成 __init__() 方法
#    生成后的 __init__ 签名大致如下：
#    def __init__(self, fps, width, height, index_or_path,
#                color_mode=ColorMode.RGB, rotation=Cv2Rotation.NO_ROTATION,
#                warmup_s=1, fourcc=None, backend=Cv2Backends.ANY)
#    注意：fps、width、height 是从父类 CameraConfig 继承的
@CameraConfig.register_subclass("opencv")
@dataclass
class OpenCVCameraConfig(CameraConfig):
    """
    Configuration class for OpenCV-based camera devices or video files.

    This class provides configuration options for cameras accessed through OpenCV,
    supporting both physical camera devices and video files. It includes settings
    for resolution, frame rate, color mode, and image rotation.

    Example configurations:
    ```python
    # Basic configurations
    OpenCVCameraConfig(0, 30, 1280, 720)   # 1280x720 @ 30FPS
    OpenCVCameraConfig(/dev/video4, 60, 640, 480)   # 640x480 @ 60FPS

    # Advanced configurations with FOURCC format
    OpenCVCameraConfig(128422271347, 30, 640, 480, rotation=Cv2Rotation.ROTATE_90, fourcc="MJPG")     # With 90° rotation and MJPG format
    OpenCVCameraConfig(0, 30, 1280, 720, fourcc="YUYV")     # With YUYV format
    ```

    Attributes:
        index_or_path: Either an integer representing the camera device index, or a Path object pointing to a video file.
                        一个代表相机设备的 int 索引或一个视频文件的 path 对象。

        fps: Requested frames per second for the color stream.
        width: Requested frame width in pixels for the color stream.
        height: Requested frame height in pixels for the color stream.
        color_mode: Color mode for image output (RGB or BGR). Defaults to RGB.
        rotation: Image rotation setting (0°, 90°, 180°, or 270°). Defaults to no rotation.
        warmup_s: Time reading frames before returning from connect (in seconds)

        fourcc: FOURCC code for video format (e.g., "MJPG", "YUYV", "I420"). Defaults to None (auto-detect).
                    视频格式的 FOURCC 编码（如 "MJPG", "YUYV", "I420"）。
                    ! FOURCC 必须是 4 个字符的字符串！否则会报错！

        backend: OpenCV backend identifier (https://docs.opencv.org/3.4/d4/d15/group__videoio__flags__base.html). Defaults to ANY.
                    OpenCV 后端标识符。

    Note:
        - Only 3-channel color output (RGB/BGR) is currently supported.
        - FOURCC codes must be 4-character strings (e.g., "MJPG", "YUYV"). Some common FOUCC codes: https://learn.microsoft.com/en-us/windows/win32/medfound/video-fourccs#fourcc-constants
        - Setting FOURCC can help achieve higher frame rates on some cameras.
    """

    # index_or_path: 相机设备索引 或 视频文件路径，可以是 int 或 Path
    index_or_path: int | Path
    color_mode: ColorMode = ColorMode.RGB
    rotation: Cv2Rotation = Cv2Rotation.NO_ROTATION
    warmup_s: int = 1  # 预热时间(s)，连接相机后，会先读取若干秒的帧以稳定相机设置
    fourcc: str | None = None  # 视频格式的 FOURCC 编码，默认值为 None（自动检测）
    # OpenCV 视频捕获后端，默认值为 Cv2Backends.ANY（自动选择）
    backend: Cv2Backends = Cv2Backends.ANY

    # ==========================================================================
    # 初始化后处理方法
    # ==========================================================================
    def __post_init__(self) -> None:
        """
        dataclass 的初始化后处理方法。

        在由 dataclass 自动生成的 __init__() 方法执行完毕后，
        会自动调用 __post_init__() 方法。

        这个方法的作用是：
        1. 确保所有枚举类型的属性都转换为正确的枚举实例
        2. 验证 fourcc 参数的格式是否正确

        输入参数：
            无 (self 是类实例本身）

        返回值：
            None (不返回任何值)
        """
        # 将 color_mode 转换为 ColorMode 枚举实例
        self.color_mode = ColorMode(self.color_mode)
        # 将 rotation 转换为 Cv2Rotation 枚举实例
        self.rotation = Cv2Rotation(self.rotation)
        # 将 backend 转换为 Cv2Backends 枚举实例
        self.backend = Cv2Backends(self.backend)

        # 检查条件（如果满足以下任一条件，则抛出错误）：
        # 1. fourcc 不是 None，且
        # 2. fourcc 不是字符串类型，或 fourcc 长度不等于 4
        if self.fourcc is not None and (
            not isinstance(self.fourcc, str) or len(self.fourcc) != 4
        ):
            raise ValueError(
                f"`fourcc` must be a 4-character string (e.g., 'MJPG', 'YUYV'), but '{self.fourcc}' is provided."
            )
