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

import logging
import os
import platform
import select
import subprocess
import sys
import time
from copy import copy, deepcopy
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from accelerate import Accelerator


# 检测当前进程是否运行在 SLURM 集群调度系统下
def inside_slurm():
    """
    检查 Python 进程是否通过 SLURM 启动

    返回:
        bool: 如果进程是通过 SLURM 启动的（具有 SLURM_JOB_ID 环境变量）则返回 True，
              否则返回 False
    """
    # 检查是否存在 SLURM_JOB_ID 环境变量以确定是否在 SLURM 环境中运行
    # TODO(rcadene): return False for interactive mode `--pty bash`
    return "SLURM_JOB_ID" in os.environ


# NOTE ✨ 整个项目的日志初始化系统
# ✅ 这是最重要的函数：支持多 GPU 训练时仅主进程输出控制台日志，避免刷屏；同时支持文件日志、进程PID显示、日志等级区分；还自动屏蔽第三方库的冗余日志
def init_logging(
    log_file: Path | None = None,
    display_pid: bool = False,
    console_level: str = "INFO",
    file_level: str = "DEBUG",
    accelerator: Accelerator | None = None,
):
    """
    初始化 LeRobot 的日志配置。

    在多 GPU 训练中，只有主进程记录到控制台以避免重复输出。
    非主进程的控制台日志被抑制但仍可记录到文件。

    Args:
        log_file: 可选的日志文件写入路径
        display_pid: 在日志消息中包含进程 ID (对调试多进程有用）
        console_level: 控制台输出的日志级别
        file_level: 文件输出的日志级别
        accelerator: 可选的 Accelerator 实例（用于多 GPU 检测）
    """

    # 自定义日志格式化函数
    def custom_format(record: logging.LogRecord) -> str:
        dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fnameline = f"{record.pathname}:{record.lineno}"
        pid_str = f"[PID: {os.getpid()}] " if display_pid else ""
        return f"{record.levelname} {pid_str}{dt} {fnameline[-15:]:>15} {record.getMessage()}"

    # 创建并设置自定义格式化器
    formatter = logging.Formatter()
    formatter.format = custom_format

    # 获取根日志记录器并重置其级别
    logger = logging.getLogger()
    logger.setLevel(logging.NOTSET)

    # 清除任何现有的处理器
    logger.handlers.clear()

    # 确定在分布式训练中是否为主进程
    is_main_process = accelerator.is_main_process if accelerator is not None else True

    # 控制台日志记录（仅主进程）
    if is_main_process:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(console_level.upper())
        logger.addHandler(console_handler)
    else:
        # 抑制非主进程的控制台输出
        logger.addHandler(logging.NullHandler())
        logger.setLevel(logging.ERROR)

    # 添加文件日志记录（如果指定）
    if log_file is not None:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(file_level.upper())
        logger.addHandler(file_handler)

    # 抑制 httpx 库的日志记录以减少冗余输出
    logging.getLogger("httpx").setLevel(logging.WARNING)


# 大数字格式化，自动加 K/M/B/T 后缀
def format_big_number(num, precision=0):
    """
    将大数字格式化为带有适当后缀的字符串表示。用于显示模型参数量、数据集大小等统计数字

    参数:
        num (float/int): 需要格式化的数字
        precision (int): 小数点后保留的位数，默认为 0

    返回值:
        str: 格式化后的数字字符串，带有相应的数量级后缀 (K, M, B, T, Q 等)
             如果数字过大超出预定义后缀范围，则返回处理后的数值

    示例:
        format_big_number(500)      -> "500"     # 小于 1000, 无后缀
        format_big_number(1500)     -> "1.5K"    # 千级别 (1.5 x 10^3)
        format_big_number(2000000)  -> "2.0M"    # 百万级别 (2.0 x 10^6)
        format_big_number(3000000000) -> "3.0B"  # 十亿级别 (3.0 x 10^9)
        format_big_number(8000000000000) -> "8.0T" # 万亿级别 (8.0 x 10^12)
        format_big_number(5000000000000000) -> "5.0Q" # 千万亿级别 (5.0 x 10^15)
    """
    # 定义数量级后缀数组，分别代表千、百万、十亿、万亿、千万亿
    suffixes = ["", "K", "M", "B", "T", "Q"]
    divisor = 1000.0

    for suffix in suffixes:
        if abs(num) < divisor:
            return f"{num:.{precision}f}{suffix}"
        num /= divisor

    return num


# 跨平台文本转语音播报。机器人操作时的语音反馈，训练完成提醒
def say(text: str, blocking: bool = False):
    """
    跨平台文本转语音播报功能

    参数:
        text (str): 需要播报的文本内容
        blocking (bool, optional): 是否阻塞执行，默认为 False。
            如果为 True，程序会等待语音播报完成后再继续执行；如果为 False，语音播报在后台执行，程序立即返回

    返回值:
        None

    异常:
        RuntimeError: 当操作系统不支持文本转语音功能时抛出

    示例:
        >>> say("Hello, world!")  # 在后台播放语音，程序立即返回
        >>> say("Processing complete.", blocking=True)  # 等待语音播放完成后继续执行
    """
    system = platform.system()

    if system == "Darwin":
        cmd = ["say", text]

    elif system == "Linux":
        cmd = ["spd-say", text]
        if blocking:
            cmd.append("--wait")

    elif system == "Windows":
        cmd = [
            "PowerShell",
            "-Command",
            "Add-Type -AssemblyName System.Speech; "
            f"(New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{text}')",
        ]

    else:
        raise RuntimeError("Unsupported operating system for text-to-speech.")

    if blocking:
        subprocess.run(cmd, check=True)
    else:
        subprocess.Popen(
            cmd, creationflags=subprocess.CREATE_NO_WINDOW if system == "Windows" else 0
        )


# 同时记录日志 + 语音播报
def log_say(text: str, play_sounds: bool = True, blocking: bool = False):
    """
    同时记录日志并可选择性地播放语音

    Args:
        text (str): 要记录和播放的文本内容
        play_sounds (bool, optional): 是否播放声音，默认为 True
        blocking (bool, optional): 播放时是否阻塞执行，默认为 False

    Returns:
        None

    Example:
        >>> log_say("Training completed successfully!")  # 记录日志并播放语音
        >>> log_say("Warning: Low battery level", play_sounds=False)  # 仅记录日志，不播放语音
        >>> log_say("System ready", blocking=True)  # 记录日志并阻塞播放语音
    """
    logging.info(text)

    if play_sounds:
        say(text, blocking)


def get_channel_first_image_shape(image_shape: tuple) -> tuple:
    """
    将图像形状转换为通道优先格式（channel-first format）。

    参数:
        image_shape (tuple): 输入的图像形状元组，通常是 (height, width, channel) 或 (channel, height, width) 格式

    返回:
        tuple: 转换后的通道优先格式的图像形状元组 (channel, height, width)
    示例：
        假设输入图像形状是 (height=480, width=640, channel=3) 的格式
        input_shape = (480, 640, 3)
        result = get_channel_first_image_shape(input_shape)
        print(result)  # 输出: (3, 480, 640) - 转换为 (channel, height, width) 格式
    """
    shape = copy(image_shape)
    if shape[2] < shape[0] and shape[2] < shape[1]:  # (h, w, c) -> (c, h, w)
        shape = (shape[2], shape[0], shape[1])
    elif not (shape[0] < shape[1] and shape[0] < shape[2]):
        raise ValueError(image_shape)

    return shape


# 安全检测对象是否包含某个可调用方法
def has_method(cls: object, method_name: str) -> bool:
    return hasattr(cls, method_name) and callable(getattr(cls, method_name))


# 验证字符串是否是合法的 Numpy 数据类型
def is_valid_numpy_dtype_string(dtype_str: str) -> bool:
    """
    Return True if a given string can be converted to a numpy dtype.
    """
    try:
        # Attempt to convert the string to a numpy dtype
        np.dtype(dtype_str)
        return True
    except TypeError:
        # If a TypeError is raised, the string is not a valid dtype
        return False


# 非阻塞检测是否按下了回车键。用于控制流程交互，录制/遥操作时的暂停继续
def enter_pressed() -> bool:
    if platform.system() == "Windows":
        import msvcrt

        if msvcrt.kbhit():
            key = msvcrt.getch()
            return key in (b"\r", b"\n")  # enter key
        return False
    else:
        return (
            select.select([sys.stdin], [], [], 0)[0]
            and sys.stdin.readline().strip() == ""
        )


# 终端光标控制，用于实时进度条刷新
def move_cursor_up(lines):
    """Move the cursor up by a specified number of lines."""
    print(f"\033[{lines}A", end="")


# 格式化时间戳为 天/时/分/秒
def get_elapsed_time_in_days_hours_minutes_seconds(elapsed_time_s: float):
    days = int(elapsed_time_s // (24 * 3600))
    elapsed_time_s %= 24 * 3600
    hours = int(elapsed_time_s // 3600)
    elapsed_time_s %= 3600
    minutes = int(elapsed_time_s // 60)
    seconds = elapsed_time_s % 60
    return days, hours, minutes, seconds


# 临时屏蔽 HuggingFace datasets 库的进度条
class SuppressProgressBars:
    """
    Context manager to suppress progress bars.

    Example
    --------
    ```python
    with SuppressProgressBars():
        # Code that would normally show progress bars
    ```
    """

    def __enter__(self):
        from datasets.utils.logging import disable_progress_bar

        disable_progress_bar()

    def __exit__(self, exc_type, exc_val, exc_tb):
        from datasets.utils.logging import enable_progress_bar

        enable_progress_bar()


# ✨ 这是代码里最常用的辅助类：高精度性能计时器
class TimerManager:
    """
    Lightweight utility to measure elapsed time.

    Examples
    --------
    ```python
    # Example 1: Using context manager
    timer = TimerManager("Policy", log=False)
    for _ in range(3):
        with timer:
            time.sleep(0.01)
    print(timer.last, timer.fps_avg, timer.percentile(90))  # Prints: 0.01 100.0 0.01
    ```

    ```python
    # Example 2: Using start/stop methods
    timer = TimerManager("Policy", log=False)
    timer.start()
    time.sleep(0.01)
    timer.stop()
    print(timer.last, timer.fps_avg, timer.percentile(90))  # Prints: 0.01 100.0 0.01
    ```
    """

    def __init__(
        self,
        label: str = "Elapsed-time",
        log: bool = True,
        logger: logging.Logger | None = None,
    ):
        self.label = label
        self.log = log
        self.logger = logger
        self._start: float | None = None
        self._history: list[float] = []

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def start(self):
        self._start = time.perf_counter()
        return self

    def stop(self) -> float:
        if self._start is None:
            raise RuntimeError("Timer was never started.")
        elapsed = time.perf_counter() - self._start
        self._history.append(elapsed)
        self._start = None
        if self.log:
            if self.logger is not None:
                self.logger.info(f"{self.label}: {elapsed:.6f} s")
            else:
                logging.info(f"{self.label}: {elapsed:.6f} s")
        return elapsed

    def reset(self):
        self._history.clear()

    @property
    def last(self) -> float:
        return self._history[-1] if self._history else 0.0

    @property
    def avg(self) -> float:
        return mean(self._history) if self._history else 0.0

    @property
    def total(self) -> float:
        return sum(self._history)

    @property
    def count(self) -> int:
        return len(self._history)

    @property
    def history(self) -> list[float]:
        return deepcopy(self._history)

    @property
    def fps_last(self) -> float:
        return 0.0 if self.last == 0 else 1.0 / self.last

    @property
    def fps_avg(self) -> float:
        return 0.0 if self.avg == 0 else 1.0 / self.avg

    def percentile(self, p: float) -> float:
        """
        Return the p-th percentile of recorded times.
        """
        if not self._history:
            return 0.0
        return float(np.percentile(self._history, p))

    def fps_percentile(self, p: float) -> float:
        """
        FPS corresponding to the p-th percentile time.
        """
        val = self.percentile(p)
        return 0.0 if val == 0 else 1.0 / val
