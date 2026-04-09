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
import inspect
import pkgutil
import sys
from argparse import ArgumentError
from collections.abc import Callable, Iterable, Sequence
from functools import wraps
from pathlib import Path
from pkgutil import ModuleInfo
from types import ModuleType
from typing import Any, TypeVar, cast

import draccus

from lerobot.utils.utils import has_method

F = TypeVar("F", bound=Callable[..., object])

PATH_KEY = "path"
PLUGIN_DISCOVERY_SUFFIX = "discover_packages_path"


def get_cli_overrides(
    field_name: str, args: Sequence[str] | None = None
) -> list[str] | None:
    """Parses arguments from cli at a given nested attribute level.

    For example, supposing the main script was called with:
    python myscript.py --arg1=1 --arg2.subarg1=abc --arg2.subarg2=some/path

    If called during execution of myscript.py, get_cli_overrides("arg2") will return:
    ["--subarg1=abc" "--subarg2=some/path"]
    """
    if args is None:
        args = sys.argv[1:]
    attr_level_args = []
    detect_string = f"--{field_name}."
    exclude_strings = (
        f"--{field_name}.{draccus.CHOICE_TYPE_KEY}=",
        f"--{field_name}.{PATH_KEY}=",
    )
    for arg in args:
        if arg.startswith(detect_string) and not arg.startswith(exclude_strings):
            denested_arg = f"--{arg.removeprefix(detect_string)}"
            attr_level_args.append(denested_arg)

    return attr_level_args


def parse_arg(arg_name: str, args: Sequence[str] | None = None) -> str | None:
    if args is None:
        args = sys.argv[1:]
    prefix = f"--{arg_name}="
    for arg in args:
        if arg.startswith(prefix):
            return arg[len(prefix) :]
    return None


def parse_plugin_args(plugin_arg_suffix: str, args: Sequence[str]) -> dict[str, str]:
    """Parse plugin-related arguments from command-line arguments.

    This function extracts arguments from command-line arguments that match a specified suffix pattern.
    It processes arguments in the format '--key=value' and returns them as a dictionary.

    Args:
        plugin_arg_suffix (str): The suffix to identify plugin-related arguments.
        cli_args (Sequence[str]): A sequence of command-line arguments to parse.

    Returns:
        dict: A dictionary containing the parsed plugin arguments where:
            - Keys are the argument names (with '--' prefix removed if present)
            - Values are the corresponding argument values

    Example:
        >>> args = ["--env.discover_packages_path=my_package", "--other_arg=value"]
        >>> parse_plugin_args("discover_packages_path", args)
        {'env.discover_packages_path': 'my_package'}
    """
    plugin_args = {}
    for arg in args:
        if "=" in arg and plugin_arg_suffix in arg:
            key, value = arg.split("=", 1)
            # Remove leading '--' if present
            if key.startswith("--"):
                key = key[2:]
            plugin_args[key] = value
    return plugin_args


class PluginLoadError(Exception):
    """Raised when a plugin fails to load."""


def load_plugin(plugin_path: str) -> None:
    """Load and initialize a plugin from a given Python package path.

    This function attempts to load a plugin by importing its package and any submodules.
    Plugin registration is expected to happen during package initialization, i.e. when
    the package is imported the gym environment should be registered and the config classes
    registered with their parents using the `register_subclass` decorator.

    Args:
        plugin_path (str): The Python package path to the plugin (e.g. "mypackage.plugins.myplugin")

    Raises:
        PluginLoadError: If the plugin cannot be loaded due to import errors or if the package path is invalid.

    Examples:
        >>> load_plugin("external_plugin.core")  # Loads plugin from external package

    Notes:
        - The plugin package should handle its own registration during import
        - All submodules in the plugin package will be imported
        - Implementation follows the plugin discovery pattern from Python packaging guidelines

    See Also:
        https://packaging.python.org/en/latest/guides/creating-and-discovering-plugins/
    """
    try:
        package_module = importlib.import_module(plugin_path, __package__)
    except (ImportError, ModuleNotFoundError) as e:
        raise PluginLoadError(
            f"Failed to load plugin '{plugin_path}'. Verify the path and installation: {str(e)}"
        ) from e

    def iter_namespace(ns_pkg: ModuleType) -> Iterable[ModuleInfo]:
        return pkgutil.iter_modules(ns_pkg.__path__, ns_pkg.__name__ + ".")

    try:
        for _finder, pkg_name, _ispkg in iter_namespace(package_module):
            importlib.import_module(pkg_name)
    except ImportError as e:
        raise PluginLoadError(
            f"Failed to load plugin '{plugin_path}'. Verify the path and installation: {str(e)}"
        ) from e


def get_path_arg(field_name: str, args: Sequence[str] | None = None) -> str | None:
    return parse_arg(f"{field_name}.{PATH_KEY}", args)


def get_type_arg(field_name: str, args: Sequence[str] | None = None) -> str | None:
    return parse_arg(f"{field_name}.{draccus.CHOICE_TYPE_KEY}", args)


def filter_arg(field_to_filter: str, args: Sequence[str] | None = None) -> list[str]:
    if args is None:
        return []
    return [arg for arg in args if not arg.startswith(f"--{field_to_filter}=")]


def filter_path_args(
    fields_to_filter: str | list[str], args: Sequence[str] | None = None
) -> list[str]:
    """
    Filters command-line arguments related to fields with specific path arguments.

    Args:
        fields_to_filter (str | list[str]): A single str or a list of str whose arguments need to be filtered.
        args (Sequence[str] | None): The sequence of command-line arguments to be filtered.
            Defaults to None.

    Returns:
        list[str]: A filtered list of arguments, with arguments related to the specified
        fields removed.

    Raises:
        ArgumentError: If both a path argument (e.g., `--field_name.path`) and a type
            argument (e.g., `--field_name.type`) are specified for the same field.
    """
    if isinstance(fields_to_filter, str):
        fields_to_filter = [fields_to_filter]

    filtered_args = [] if args is None else list(args)

    for field in fields_to_filter:
        if get_path_arg(field, args):
            if get_type_arg(field, args):
                raise ArgumentError(
                    argument=None,
                    message=f"Cannot specify both --{field}.{PATH_KEY} and --{field}.{draccus.CHOICE_TYPE_KEY}",
                )
            filtered_args = [
                arg for arg in filtered_args if not arg.startswith(f"--{field}.")
            ]

    return filtered_args


def wrap(config_path: Path | None = None) -> Callable[[F], F]:
    """
    HACK: Similar to draccus.wrap but does three additional things:
        - Will remove '.path' arguments from CLI in order to process them later on.
        - If a 'config_path' is passed and the main config class has a 'from_pretrained' method, will
          initialize it from there to allow to fetch configs from the hub directly
        - Will load plugins specified in the CLI arguments. These plugins will typically register
            their own subclasses of config classes, so that draccus can find the right class to instantiate
            from the CLI '.type' arguments
    """

    # NOTE 命令行参数在这里进行序列化（解析）
    def wrapper_outer(fn: F) -> F:

        # 当运行下面命令时：
        # python src/lerobot/scripts/lerobot_record.py --robot.type=so100_follower --robot.port=/dev/ttyUSB0

        # sys.argv 的值是：
        #   sys.argv = [
        #   'src/lerobot/scripts/lerobot_record.py',    # sys.argv[0] - 脚本名
        #   '--robot.type=so100_follower',              # sys.argv[1]
        #   '--robot.port=/dev/ttyUSB0'                 # sys.argv[2]
        #   ]

        # sys.argv[1:] 就是去掉脚本名后的参数列表
        # cli_args = sys.argv[1:]
        # ['--robot.type=so100_follower', '--robot.port=/dev/ttyUSB0', ......]

        @wraps(fn)
        def wrapper_inner(*args: Any, **kwargs: Any) -> Any:
            # 获取被装饰函数的第一个参数
            # 对于 record(cfg: RecordConfig) 来说，
            #   1、fn 就是 record 函数；
            #   2、argspec.args[0] = 'cfg'  (第一个参数名，也就是 cfg: RecordConfig)；
            argspec = inspect.getfullargspec(fn)
            # argspec.annotations = {'cfg': RecordConfig, 'return': LeRobotDataset}
            # 所以 argtype = RecordConfig
            argtype = argspec.annotations[argspec.args[0]]

            # 检查是否已经传入了配置对象
            if len(args) > 0 and type(args[0]) is argtype:
                # 如果已经传入，直接使用
                cfg = args[0]  # * 使用传入的对象
                args = args[1:]
            else:
                # 否则从命令行解析
                cli_args = sys.argv[1:]  # * 获取所有命令行参数
                # 以 python src/lerobot/scripts/lerobot_record.py
                # --robot.type=so100_follower --robot.port=/dev/tty.usbmodem58760431541
                # --robot.cameras="{laptop: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}"
                # --robot.id=black --dataset.repo_id=ghr/demo_dataset --dataset.num_episodes=2
                # --dataset.single_task="This is a demo dataset" --dataset.streaming_encoding=true
                # --dataset.encoder_threads=2 --display_data=true --teleop.type=so100_leader
                # --teleop.port=/dev/tty.usbmodem58760431551 --teleop.id=blue 为例还解析命令行参数解析过程...
                # cli_args = [
                #     '--robot.type=so100_follower',
                #     '--robot.port=/dev/tty.usbmodem58760431541',
                #     '--robot.cameras={laptop: {type: opencv, ...}}',
                #     '--robot.id=black',
                #     '--dataset.repo_id=ghr/demo_dataset',
                #     '--dataset.num_episodes=2',
                #     '--dataset.single_task=This is a demo dataset',
                #     '--dataset.streaming_encoding=true',
                #     '--dataset.encoder_threads=2',
                #     '--display_data=true',
                #     '--teleop.type=so100_leader',
                #     '--teleop.port=/dev/tty.usbmodem58760431551',
                #     '--teleop.id=blue'
                # ]
                # 处理插件参数（如果有的话）
                plugin_args = parse_plugin_args(PLUGIN_DISCOVERY_SUFFIX, cli_args)
                # PLUGIN_DISCOVERY_SUFFIX = "discover_packages_path"
                # 如果没有 --*.discover_packages_path 参数，plugin_args = {}

                # 遍历插件并加载（这里没有插件，跳过）
                for plugin_cli_arg, plugin_path in plugin_args.items():
                    try:
                        load_plugin(plugin_path)
                    except PluginLoadError as e:
                        # add the relevant CLI arg to the error message
                        raise PluginLoadError(
                            f"{e}\nFailed plugin CLI Arg: {plugin_cli_arg}"
                        ) from e
                    cli_args = filter_arg(plugin_cli_arg, cli_args)

                # 处理 config_path 参数（如果有的话）
                config_path_cli = parse_arg("config_path", cli_args)
                # 如果没有 --config_path=xxx，则 config_path_cli = None

                # case1 分支 1: 检查 RecordConfig 是否有 __get_path_fields__ 方法
                if has_method(argtype, "__get_path_fields__"):
                    path_fields = argtype.__get_path_fields__()
                    # path_fields = ["policy"]  （来自 RecordConfig 的定义）
                    cli_args = filter_path_args(path_fields, cli_args)
                    # 移除所有 --policy.xxx 参数，因为它们会被单独处理
                    # 这里没有 --policy.xxx，所以 cli_args 不变
                # case2 分支 2: 检查是否有 from_pretrained 方法
                if has_method(argtype, "from_pretrained") and config_path_cli:
                    cli_args = filter_arg("config_path", cli_args)
                    # RecordConfig 没有 from_pretrained 方法，跳过
                    cfg = argtype.from_pretrained(config_path_cli, cli_args=cli_args)
                # case3 分支 3: 检查完毕, 开始最终的序列化
                else:
                    # 调用 draccus.parse() 解析配置, 解析过程👇
                    #   1. 识别 RecordConfig 是一个 dataclass
                    #   2. 遍历其字段：robot, dataset, teleop, policy, ...
                    #   3. 对每个字段递归解析：
                    #       3.1 --robot.type=so100_follower
                    #           → 查找 RobotConfig 的子类中名为 "so100_follower" 的类
                    #           → 找到 SOFollowerRobotConfig
                    #           → 继续解析 SOFollowerRobotConfig 的字段
                    #               --robot.port=/dev/tty.usbmodem58760431541
                    #               → 设置 port = "/dev/tty.usbmodem58760431541"
                    #
                    #               --robot.id=black
                    #               → 设置 id = "black"

                    #               --robot.cameras={laptop: {type: opencv, ...}} "类型为 dict[str, CameraConfig]"
                    #               → 解析 cameras 字典
                    #               → 对于 key = "laptop" 相机：
                    #                   type: opencv → 使用 OpenCVCameraConfig
                    #                   index_or_path: 0
                    #                   width: 640
                    #                   height: 480
                    #                   fps: 30
                    #       3.2 同样解析 dataset、teleop 等字段...
                    #   4. 最终生成完整的 RecordConfig 对象

                    # * 此时的 cfg 是一个完整的 RecordConfig 对象了!!!
                    cfg = draccus.parse(
                        config_class=argtype, config_path=config_path, args=cli_args
                    )  # argtype = RecordConfig, config_path = None, cli_args = [...一大堆]

            # 将解析好的 cfg 传入 record() 函数
            response = fn(cfg, *args, **kwargs)  # 等价于：record(cfg)

            return response

        return cast(F, wrapper_inner)

    return cast(Callable[[F], F], wrapper_outer)
