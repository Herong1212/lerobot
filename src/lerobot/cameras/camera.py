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
import warnings
from typing import Any

from numpy.typing import NDArray  # type: ignore  # TODO: add type stubs for numpy.typing

# NDArray[Any] 表示一个 NumPy 多维数组，元素类型为 Any
# 在相机场景中，帧数据通常是一个 3 维 NumPy 数组 (高度, 宽度, 通道数)

from .configs import CameraConfig


# * 所有相机的核心基类，定义了“相机”这个物体的所有标准行为（如：连接、断开、读图像）
class Camera(abc.ABC):
    """
    相机实现的基类。

    为不同后端的相机操作定义标准接口。
    所有子类必须实现所有抽象方法。

    管理基本相机属性 (FPS、分辨率) 和核心操作：
    - 连接/断开连接
    - 帧捕获（同步/异步/最新）

    NOTE 这是一个抽象基类 (ABC)，不能直接实例化，必须由具体相机类型继承实现。

    Attributes:
        fps (int | None): Configured frames per second
        width (int | None): Frame width in pixels
        height (int | None): Frame height in pixels
    """

    def __init__(self, config: CameraConfig):
        """Initialize the camera with the given configuration.

        Args:
            config: Camera configuration containing FPS and resolution.
        """
        self.fps: int | None = config.fps
        self.width: int | None = config.width
        self.height: int | None = config.height

    # ==========================================================================
    # 上下文管理器方法（支持 with 语句）
    # ==========================================================================
    def __enter__(self):
        """
        Context manager entry.
        Automatically connects to the camera.

        上下文管理器入口方法。

        当使用 `with` 语句时自动调用。
        自动连接到相机，使相机准备好捕获帧。

        输入参数：
            无

        返回值：
            self: 返回相机实例本身，以便在 with 语句中使用

        示例：
            with OpenCVCamera(config) as cam:
                # 这里的 cam 就是 __enter__ 返回的 self
                # 进入 with 块时，connect() 已经被调用
                frame = cam.read()
            # 离开 with 块时，__exit__ 会被调用，自动断开连接
        """
        self.connect()  # 调用抽象方法 connect() 连接到相机
        return self  # 返回自身，赋值给 as 后面的变量

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """
        Context manager exit.
        Automatically disconnects, ensuring resources are released even on error.

        上下文管理器出口方法。

        当离开 `with` 语句块时自动调用。
        自动断开相机连接，确保即使发生错误也能释放资源。

        输入参数：
            exc_type: 异常类型（如果 with 块中发生异常）
                     如果没有异常，值为 None
            exc_value: 异常实例（如果 with 块中发生异常）
                      如果没有异常，值为 None
            traceback: 回溯对象，用于调试
                      如果没有异常，值为 None

        返回值：
            None

        示例：
            with OpenCVCamera(config) as cam:
                frame = cam.read()
                # 如果这里发生异常，__exit__ 仍会被调用，确保资源释放
            # 无论是否正常退出，__exit__ 都会调用 disconnect()
        """
        self.disconnect()  # 调用抽象方法 disconnect() 断开连接并释放资源

    # ==========================================================================
    # 析构函数（垃圾回收时的安全措施）
    # ==========================================================================
    def __del__(self) -> None:
        """
        Destructor safety net.
        Attempts to disconnect if the object is garbage collected without cleanup.

        析构函数安全网。

        当对象被垃圾回收且没有正确清理时，尝试断开连接。
        这是一个安全措施，防止相机资源泄漏。

        注意：不应该依赖 __del__ 进行资源管理，
        应该始终使用 with 语句或手动调用 disconnect()。

        输入参数：
            无

        返回值：
            None

        示例：
            cam = OpenCVCamera(config)
            cam.connect()
            del cam  # 对象被删除时，__del__ 会尝试断开连接
        """
        try:
            # 尝试检查并断开连接
            # 这里使用 try-except 是因为在对象销毁过程中检查属性可能失败
            if self.is_connected:  # 访问抽象属性 is_connected
                self.disconnect()  # 如果已连接，调用 disconnect() 断开
        except Exception:  # nosec B110
            # 捕获所有异常并忽略
            # 这是有意为之的，因为析构函数不应该抛出异常
            # nosec B110 是安全检查注释，表示这里忽略空 except 块是安全的
            pass

    # ==========================================================================
    # 抽象属性和方法（必须由子类实现）
    # ==========================================================================

    @property
    @abc.abstractmethod
    def is_connected(self) -> bool:
        """
        检查相机当前是否已连接。

        这是一个抽象属性，子类必须提供实现。

        Returns:
            bool: 相机已连接并准备好捕获帧, 相机未连接或无法使用

        示例：
        cam = OpenCVCamera(config)
        print(cam.is_connected)  # False(尚未连接）
        cam.connect()
        print(cam.is_connected)  # True(已连接）
        cam.disconnect()
        print(cam.is_connected)  # False(已断开）
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def find_cameras() -> list[dict[str, Any]]:
        """
        检测系统中连接的可用相机。

        这是一个抽象静态方法，子类必须提供实现。
        静态方法意味着可以在不创建相机实例的情况下调用。

        输入参数：
            无

        返回值：
            list[dict[str, Any]]: 包含检测到的相机信息的字典列表，
                                    每个字典包含一个相机的详细信息，如：
                                    - 设备索引
                                    - 设备名称
                                    - 支持的分辨率
                                    - 序列号等

        示例：
            # 查找所有可用的 OpenCV 相机
            cameras = OpenCVCamera.find_cameras()
            # 返回示例：
            # [
            #     {"index": 0, "name": "USB Camera", "width": 640, "height": 480},
            #     {"index": 1, "name": "Integrated Webcam", "width": 1280, "height": 720}
            # ]

            # 查找所有可用的 RealSense 相机
            realsense_cameras = RealSenseCamera.find_cameras()
            # 返回示例：
            # [
            #     {"serial_number": "0123456789", "name": "Intel RealSense D455"}
            # ]
        """
        pass

    @abc.abstractmethod
    def connect(self, warmup: bool = True) -> None:
        """
        建立与相机的连接。

        这是一个抽象方法，子类必须提供实现。

        输入参数：
            warmup: bool （默认 True)
                   - True: 连接后捕获预热帧。对于需要时间调整
                          捕获设置（如曝光、白平衡）的相机很有用。
                   - False: 跳过预热帧，立即返回。

        返回值：
            None

        示例：
            cam = OpenCVCamera(config)
            cam.connect()  # 使用默认预热（warmup=True）
            cam.connect(warmup=False)  # 跳过预热

            # 使用上下文管理器（推荐做法）
            with OpenCVCamera(config) as cam:
                # connect() 在这里自动调用
                frame = cam.read()
            # disconnect() 在这里自动调用
        """
        pass

    @abc.abstractmethod
    def read(self) -> NDArray[Any]:
        """
        从相机同步捕获并返回单个完整的帧。

        相机状态时间线图解
        假设 FPS=30（每帧间隔 33ms）：

        时间轴：  0ms    33ms   66ms   99ms  132ms  165ms  198ms  231ms  264ms  297ms
        相机：   [帧0] → [帧1] → [帧2] → [帧3] → [帧4] → [帧5] → [帧6] → [帧7] → [帧8] → ...
                  ↑       ↑       ↑       ↑       ↑       ↑       ↑       ↑       ↑
            当调用connect()之后，相机就开始在后台持续捕获（像视频一样），帧存入缓冲区

        ### connect() 和 read() 的详细过程

        # ============= 第 0 步：调用 connect() =============
        cam.connect()
        # 此时相机开始工作：
        # - 相机驱动在后台持续捕获帧
        # - 每 33ms 产生新的一帧 (FPS=30)
        # - 新帧被存入内部缓冲区
        # - 相机状态：正在运行，持续捕获中


        # ============= 第 1 步：调用 img = cam.read() =============
        # 假设在 t=50ms 时调用 read()
        img = cam.read()
        # 会发生什么？
        # 1. read() 会等待当前正在捕获的帧 (也就是[帧 1]) 完成
        # 2. 假设当前正在捕获帧 1 (t=33ms 开始的这一帧）
        # 3. read() 阻塞等待这一帧完成（约 33ms 后拿到） —— 这里会阻塞！！！
        # 4. 返回这一帧的图像数据
        # 5. 调用耗时：约 10-33ms (取决于调用时机）

        # 捕获时间大概多久？
        # - 通常是 1/FPS 的时间, FPS=30 时约 33ms
        # - 如果刚好在帧完成的瞬间调用，可能几乎不阻塞


        # ============= 第 2 步：不调用 disconnect()，相机什么状态？ =============
        # 相机仍然在后台持续捕获！
        # read() 只是从缓冲区取了一帧，不影响相机运行
        # 相机状态：仍在运行，持续产生新帧
        # 缓冲区中的帧：
        # - 旧帧可能被新帧覆盖
        # - 具体行为取决于实现


        # ============= 第 3 步：执行 model(img) 推理 (100ms)============
        action = model(img)  # 推理耗时 100ms

        # 在这 100ms 期间：
        # 相机仍在后台持续捕获！
        # 产生了约 100ms / 33ms ≈ 3 帧
        # 帧序列：[帧A] → [帧B] → [帧C]（都在缓冲区中）

        # 相机状态：仍在运行，持续产生新帧
        # 缓冲区：存储了最新的几帧（旧的被覆盖）


        # ============= 第 4 步：下一次 read() =============
        img2 = cam.read()
        # 返回当前最新的帧（可能是帧 C)

        # 相机状态：仍在运行
        # 需要手动调用 disconnect() 才会停止

        # ============= 相机状态时间线如上👆 =============

        read() 时序图：
        相机捕获：  [帧0] → [帧1] → [帧2] → [帧3] → [帧4] → [帧5] → [帧6] → ...
        时间：       0ms    33ms   66ms   99ms  132ms  165ms  198ms
        read():       |____等待____|          |____等待____|
        推理：               |________100ms________|          |________100ms___
        实际使用的帧：       帧1                       帧3
        被丢掉的帧：                 帧2                      帧4, 帧5


        这是一个阻塞调用，会等待硬件和其 SDK 完成当前帧的捕获。
        这意味着在帧准备好之前，程序会暂停在这里。

        这是一个抽象方法，子类必须提供实现。

        输入参数：
            无

        返回值：
            np.ndarray: 捕获的帧，作为 NumPy 多维数组
                        通常是一个 3D 数组，形状为 (高度, 宽度, 通道数)
                        - 对于彩色图像：形状为 (H, W, 3), 3 个通道是 RGB 或 BGR
                        - 对于灰度图像：形状为 (H, W, 1) 或 (H, W)

        示例：
            cam = OpenCVCamera(config)
            cam.connect()
            frame = cam.read()  # 阻塞直到捕获一帧
            print(frame.shape)  # 输出：(480, 640, 3) 表示 640x480 彩色图像
            cam.disconnect()
        """
        pass

    @abc.abstractmethod
    def async_read(self, timeout_ms: float = ...) -> NDArray[Any]:
        """
        返回最新的未消费帧。

        时序图：
        相机捕获：  [帧0] → [帧1] → [帧2] → [帧3] → [帧4] → [帧5] → [帧6] → ...
        时间：       0ms    33ms   66ms   99ms  132ms  165ms  198ms
        async_read: |新帧?|               |新帧?|
        推理：               |________100ms________|          |________100ms___
        实际使用的帧：       帧0                       帧3
        被丢掉的帧：                                         帧1, 帧2(未被消费）


        此方法检索后台线程捕获的最新帧。
        如果缓冲区中已经有新帧（自上次调用以来捕获的），它会立即返回。

        仅在以下情况下阻塞最多 `timeout_ms`:
        - 缓冲区为空
        - 最新帧已被之前的 async_read() 调用消费

        本质上，此方法返回最新的未消费帧，如有必要则等待
        新帧在指定超时内到达。

        这是一个抽象方法，子类必须提供实现。

        使用场景：
            - 适用于控制循环，确保每个处理的帧都是最新的
            - 有效地将循环同步到相机的 FPS
            - 超时原因通常包括：
              1. 相机 FPS 过低
              2. 处理负载过重
              3. 相机已断开连接

        输入参数：
            timeout_ms: float (默认 ...，通常为 200ms 即 0.2 秒）
                       等待新帧的最大时间（毫秒）

        返回值：
            np.ndarray: 捕获的帧，作为 NumPy 多维数组

        异常：
            TimeoutError: 如果在 timeout_ms 内没有新帧到达

        示例：
            cam = OpenCVCamera(config)
            cam.connect()
            try:
                # 等待最多 200ms 获取新帧
                frame = cam.async_read(timeout_ms=200)
            except TimeoutError:
                print("超时：未收到新帧")
            cam.disconnect()

            # 在实时控制循环中使用
            while running:
                try:
                    frame = cam.async_read(timeout_ms=100)
                    # 处理帧（确保帧是最新的）
                    process_frame(frame)
                except TimeoutError:
                    # 处理超时情况
                    handle_timeout()
        """
        pass

    # ==========================================================================
    # 具体方法（有默认实现，子类可以覆盖）
    # ==========================================================================

    def read_latest(self, max_age_ms: int = 500) -> NDArray[Any]:
        """
        立即返回最近捕获的帧（窥视）。

        相机捕获：  [帧0] → [帧1] → [帧2] → [帧3] → [帧4] → [帧5] → [帧6] → ...
        时间：       0ms    33ms   66ms   99ms  132ms  165ms  198ms
        read_latest: |瞬间!|               |瞬间!|
        推理：               |________100ms________|          |________100ms___
        实际使用的帧：       可能帧0                   可能帧3
        陈旧的危险：         如果相机卡了，可能用的还是很久前的帧

        此方法是非阻塞的，返回内存缓冲区中的任何内容。
        帧可能是陈旧的，意味着它可能是很久以前捕获的
        （例如相机挂起的场景）。

        与 async_read() 的区别：
            - async_read(): 等待新帧，确保帧是最新的
            - read_latest(): 立即返回，帧可能是旧的

        使用场景：
            适用于需要零延迟或解耦频率的场景，也就是实时画面显示以及
            需要保证有帧的场景，如：
            - UI 可视化
            - 日志记录
            - 监控屏幕


        输入参数：
            max_age_ms: int(默认 500 毫秒）
                       帧的最大允许年龄（毫秒）
                       如果帧比这更旧，将抛出 TimeoutError

        返回值：
            NDArray[Any]: 帧图像(NumPy 数组）

        异常：
            TimeoutError: 如果最新帧比 max_age_ms 更旧
            NotConnectedError: 如果相机未连接
            RuntimeError: 如果相机已连接但尚未捕获任何帧

        示例：
            cam = OpenCVCamera(config)
            cam.connect()
            # 立即获取当前可用的任何帧（不等待）
            frame = cam.read_latest(max_age_ms=500)
            cam.disconnect()

            # 在 UI 中使用（始终显示最新的帧，即使是旧的）
            while ui_running:
                try:
                    frame = cam.read_latest()
                    display_image(frame)  # 立即显示，不阻塞 UI
                except TimeoutError:
                    display_error("帧太旧")

        注意：
            此方法当前发出 FutureWarning, 因为默认实现
            只是调用 async_read()。在未来的版本中，
            子类应该覆盖 read_latest() 方法。
        """
        # 发出警告，通知开发者此方法未实现
        # 建议使用具体的子类实现
        warnings.warn(
            f"{self.__class__.__name__}.read_latest() is not implemented. "
            "Please override read_latest(); it will be required in future releases.",
            # f 字符串动态包含当前类名
            # FutureWarning 表示这是一个关于未来变化的警告
            FutureWarning,  # Python 内置的警告类别
            stacklevel=2,  # 警告显示在调用者位置，而不是此函数内部
        )
        return self.async_read()  # 默认实现：调用 async_read()

    @abc.abstractmethod
    def disconnect(self) -> None:
        """
        断开与相机的连接并释放资源。

        这是一个抽象方法，子类必须提供实现。
            应该释放所有相机占用的资源，如：
            - 关闭相机设备句柄
            - 停止捕获线程
            - 释放内存缓冲区

        输入参数：
            无

        返回值：
            None

        示例：
            cam = OpenCVCamera(config)
            cam.connect()
            frame = cam.read()
            cam.disconnect()  # 释放相机资源

            # 推荐使用 with 语句，自动调用 disconnect()
            with OpenCVCamera(config) as cam:
                frame = cam.read()
            # 离开 with 块时自动调用 disconnect()
        """
        pass
