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
import importlib
import importlib.metadata
import logging
from typing import Any

from draccus.choice_types import ChoiceRegistry


def is_package_available(
    pkg_name: str, import_name: str | None = None, return_version: bool = False
) -> tuple[bool, str] | bool:
    """
    Check if the package spec exists and grab its version to avoid importing a local directory.

    Args:
        pkg_name: The name of the package as installed via pip (e.g. "python-can").
        import_name: The actual name used to import the package (e.g. "can").
                     Defaults to pkg_name if not provided.
        return_version: Whether to return the version string.
    """
    if import_name is None:
        import_name = pkg_name

    # Check if the module spec exists using the import name
    package_exists = importlib.util.find_spec(import_name) is not None
    package_version = "N/A"
    if package_exists:
        try:
            # Primary method to get the package version
            package_version = importlib.metadata.version(pkg_name)

        except importlib.metadata.PackageNotFoundError:
            # Fallback method: Only for "torch" and versions containing "dev"
            if pkg_name == "torch":
                try:
                    package = importlib.import_module(import_name)
                    temp_version = getattr(package, "__version__", "N/A")
                    # Check if the version contains "dev"
                    if "dev" in temp_version:
                        package_version = temp_version
                        package_exists = True
                    else:
                        package_exists = False
                except ImportError:
                    # If the package can't be imported, it's not available
                    package_exists = False
            else:
                # For packages other than "torch", don't attempt the fallback and set as not available
                package_exists = False
        logging.debug(f"Detected {pkg_name} version: {package_version}")
    if return_version:
        return package_exists, package_version
    else:
        return package_exists


_transformers_available = is_package_available("transformers")
_peft_available = is_package_available("peft")
_scipy_available = is_package_available("scipy")
_reachy2_sdk_available = is_package_available("reachy2_sdk")
_can_available = is_package_available("python-can", "can")
_unitree_sdk_available = is_package_available("unitree-sdk2py", "unitree_sdk2py")
_pygame_available = is_package_available("pygame")


def make_device_from_device_class(config: ChoiceRegistry) -> Any:
    """
    Dynamically instantiates an object from its `ChoiceRegistry` configuration.

    This factory uses the module path and class name from the `config` object's
    type to locate and instantiate the corresponding device class (not the config).
    It derives the device class name by removing a trailing 'Config' from the config
    class name and tries a few candidate modules where the device implementation is
    commonly located.
    """
    if not isinstance(config, ChoiceRegistry):
        raise ValueError(f"Config should be an instance of `ChoiceRegistry`, got {type(config)}")

    config_cls = config.__class__
    module_path = config_cls.__module__  # typical: lerobot_teleop_mydevice.config_mydevice
    config_name = config_cls.__name__  # typical: MyDeviceConfig

    # Derive device class name (strip "Config")
    if not config_name.endswith("Config"):
        raise ValueError(f"Config class name '{config_name}' does not end with 'Config'")

    device_class_name = config_name[:-6]  # typical: MyDeviceConfig -> MyDevice

    # Build candidate modules to search for the device class
    parts = module_path.split(".")
    parent_module = ".".join(parts[:-1]) if len(parts) > 1 else module_path
    candidates = [
        parent_module,  # typical: lerobot_teleop_mydevice
        parent_module + "." + device_class_name.lower(),  # typical: lerobot_teleop_mydevice.mydevice
    ]

    # handle modules named like "config_xxx" -> try replacing that piece with "xxx"
    last = parts[-1] if parts else ""
    if last.startswith("config_"):
        candidates.append(".".join(parts[:-1] + [last.replace("config_", "")]))

    # de-duplicate while preserving order
    seen: set[str] = set()
    candidates = [c for c in candidates if not (c in seen or seen.add(c))]

    tried: list[str] = []
    for candidate in candidates:
        tried.append(candidate)
        try:
            module = importlib.import_module(candidate)
        except ImportError:
            continue

        if hasattr(module, device_class_name):
            cls = getattr(module, device_class_name)
            if callable(cls):
                try:
                    return cls(config)
                except TypeError as e:
                    raise TypeError(
                        f"Failed to instantiate '{device_class_name}' from module '{candidate}': {e}"
                    ) from e

    raise ImportError(
        f"Could not locate device class '{device_class_name}' for config '{config_name}'. "
        f"Tried modules: {tried}. Ensure your device class name is the config class name without "
        f"'Config' and that it's importable from one of those modules."
    )


def register_third_party_plugins() -> None:
    """
    发现并导入第三方 LeRobot 插件，以便它们可以注册自己。

    This function uses `importlib.metadata` to find packages installed in the environment
    (including editable installs) starting with 'lerobot_robot_', 'lerobot_camera_',
    'lerobot_teleoperator_', or 'lerobot_policy_' and imports them.

    此函数使用 `importlib.metadata` 查找环境中安装（包括以 -e . 安装方式安装的）的以 'lerobot_robot_'、'lerobot_camera_'、
    'lerobot_teleoperator_' 或 'lerobot_policy_' 开头的包并导入它们。
    """
    # 定义插件包名称的前缀元组，用于识别LeRobot插件
    prefixes = ("lerobot_robot_", "lerobot_camera_", "lerobot_teleoperator_", "lerobot_policy_")
    # 创建一个列表，用于存储成功导入的插件名称
    imported: list[str] = []
    # 创建一个列表，用于存储导入失败的插件名称
    failed: list[str] = []

    def attempt_import(module_name: str):
        # 尝试导入指定模块的内部函数
        try:
            # 使用importlib实际导入模块
            importlib.import_module(module_name)
            # 将成功导入的模块添加到imported列表
            imported.append(module_name)
            # 记录成功导入的插件日志
            logging.info("Imported third-party plugin: %s", module_name)
        except Exception:
            # 如果导入过程中出现异常，则记录错误日志
            logging.exception("Could not import third-party plugin: %s", module_name)
            # 将导入失败的模块添加到failed列表
            failed.append(module_name)

    # 遍历所有已安装的包分布
    for dist in importlib.metadata.distributions():
        # 获取包的名称
        dist_name = dist.metadata.get("Name")

        # 检查包名称是否存在
        if not dist_name:
            continue
        # 检查包名称是否以定义的前缀之一开始
        if dist_name.startswith(prefixes):
            attempt_import(dist_name)

    # 记录调试信息，显示导入插件的摘要统计
    logging.debug("Third-party plugin import summary: imported=%s failed=%s", imported, failed)
