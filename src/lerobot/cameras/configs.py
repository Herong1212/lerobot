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

import abc
from dataclasses import dataclass
from enum import Enum

# ####################################################################
# ########################### 核心配置类 ##############################
# ####################################################################

# 以下各类定义了相机的属性参数（FPS、宽、高、颜色模式等）

import draccus  # type: ignore  # TODO: add type stubs for draccus


# 定义颜色模式枚举：限制只能使用 RGB 或 BGR
class ColorMode(str, Enum):
    RGB = "rgb"
    BGR = "bgr"

    @classmethod
    def _missing_(cls, value: object) -> None:
        # 如果输入的颜色模式不在定义范围内，抛出详细的错误提示
        raise ValueError(
            f"`color_mode` is expected to be in {list(cls)}, but {value} is provided."
        )


# 定义 OpenCV 旋转角度枚举：处理相机倒装或侧装的情况
class Cv2Rotation(int, Enum):
    NO_ROTATION = 0
    ROTATE_90 = 90
    ROTATE_180 = 180
    ROTATE_270 = -90

    @classmethod
    def _missing_(cls, value: object) -> None:
        # 如果旋转角度不合法，抛出错误
        raise ValueError(
            f"`rotation` is expected to be in {list(cls)}, but {value} is provided."
        )


# 定义 OpenCV 后端 API 枚举：指定系统通过什么协议访问相机硬件
# 参考自 OpenCV 官方的视频 I/O 标志：Subset from https://docs.opencv.org/3.4/d4/d15/group__videoio__flags__base.html
class Cv2Backends(int, Enum):
    ANY = 0  # 自动选择最合适的后端
    V4L2 = 200  # Video4Linux2（Linux 系统下最常用的驱动）
    DSHOW = 700  # DirectShow（Windows 常用）
    PVAPI = 800  # Pleora GigE Vision 驱动
    ANDROID = 1000  # Android 平台后端
    AVFOUNDATION = 1200  # Apple / macOS 后端
    MSMF = 1400  # Microsoft Media Foundation（Windows 新标准）

    @classmethod
    def _missing_(cls, value: object) -> None:
        # 如果指定的后端 API 不受支持，抛出错误
        raise ValueError(
            f"`backend` is expected to be in {list(cls)}, but {value} is provided."
        )


# 相机配置的基类（抽象类）
@dataclass(kw_only=True)  # kw_only=True 强制要求在实例化时必须使用关键字传参
class CameraConfig(draccus.ChoiceRegistry, abc.ABC):  # type: ignore  # TODO: add type stubs for draccus
    # draccus.ChoiceRegistry 允许 LeRobot 根据配置中的 type 字段自动选择子类

    fps: int | None = None  # 帧率：每秒抓取的图像数量，默认为 None（使用硬件默认值）
    width: int | None = None  # 宽度：图像像素宽度
    height: int | None = None  # 高度：图像像素高度

    @property
    def type(self) -> str:
        # 动态获取当前子类的注册名称（例如 "opencv" 或 "intelrealsense"）
        return str(self.get_choice_name(self.__class__))
