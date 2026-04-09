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
lerobot_record.py` 是 LeRobot 框架中的数据集录制脚本，支持：
    1、遥操作录制: 人工操作机械臂, 记录动作和图像;
    2、策略录制: 用训练好的 AI 模型控制机器人，自动录制数据;

Example:

```shell
lerobot-record \
    --robot.type=so100_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.cameras="{laptop: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
    --robot.id=black \
    --dataset.repo_id=<my_username>/<my_dataset_name> \
    --dataset.num_episodes=2 \
    --dataset.single_task="Grab the cube" \
    --dataset.streaming_encoding=true \
    --dataset.encoder_threads=2 \
    --display_data=true
    # <- Optional: specify video codec (auto, h264, hevc, libsvtav1). Default is libsvtav1. \
    # --dataset.vcodec=h264 \
    # <- Teleop optional if you want to teleoperate to record or in between episodes with a policy \
    # --teleop.type=so100_leader \
    # --teleop.port=/dev/tty.usbmodem58760431551 \
    # --teleop.id=blue \
    # <- Policy optional if you want to record with a policy \
    # --policy.path=${HF_USER}/my_policy \
```

Example recording with bimanual so100:
```shell
lerobot-record \
  --robot.type=bi_so_follower \
  --robot.left_arm_config.port=/dev/tty.usbmodem5A460822851 \
  --robot.right_arm_config.port=/dev/tty.usbmodem5A460814411 \
  --robot.id=bimanual_follower \
  --robot.left_arm_config.cameras='{
    wrist: {"type": "opencv", "index_or_path": 1, "width": 640, "height": 480, "fps": 30},
    top: {"type": "opencv", "index_or_path": 3, "width": 640, "height": 480, "fps": 30},
  }' --robot.right_arm_config.cameras='{
    wrist: {"type": "opencv", "index_or_path": 2, "width": 640, "height": 480, "fps": 30},
    front: {"type": "opencv", "index_or_path": 4, "width": 640, "height": 480, "fps": 30},
  }' \
  --teleop.type=bi_so_leader \
  --teleop.left_arm_config.port=/dev/tty.usbmodem5A460852721 \
  --teleop.right_arm_config.port=/dev/tty.usbmodem5A460819811 \
  --teleop.id=bimanual_leader \
  --display_data=true \
  --dataset.repo_id=${HF_USER}/bimanual-so-handover-cube \
  --dataset.num_episodes=25 \
  --dataset.single_task="Grab and handover the red cube to the other arm" \
  --dataset.streaming_encoding=true \
  # --dataset.vcodec=auto \
  --dataset.encoder_threads=2
```
"""

import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from pprint import pformat
from typing import Any

from lerobot.cameras import (  # noqa: F401
    CameraConfig,  # noqa: F401
)
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.reachy2_camera.configuration_reachy2_camera import (
    Reachy2CameraConfig,
)  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import (
    RealSenseCameraConfig,
)  # noqa: F401
from lerobot.cameras.zmq.configuration_zmq import ZMQCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.feature_utils import build_dataset_frame, combine_feature_dicts
from lerobot.datasets.image_writer import safe_stop_image_writer
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import (
    aggregate_pipeline_dataset_features,
    create_initial_features,
)
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import make_robot_action
from lerobot.processor import (
    PolicyAction,
    PolicyProcessorPipeline,
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
    make_default_processors,
)
from lerobot.processor.rename_processor import rename_stats
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_openarm_follower,
    bi_so_follower,
    earthrover_mini_plus,
    hope_jr,
    koch_follower,
    make_robot_from_config,
    omx_follower,
    openarm_follower,
    reachy2,
    so_follower,
    unitree_g1 as unitree_g1_robot,
)
from lerobot.teleoperators import (  # noqa: F401
    Teleoperator,
    TeleoperatorConfig,
    bi_openarm_leader,
    bi_so_leader,
    homunculus,
    koch_leader,
    make_teleoperator_from_config,
    omx_leader,
    openarm_leader,
    openarm_mini,
    reachy2_teleoperator,
    so_leader,
    unitree_g1,
)
from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.control_utils import (
    init_keyboard_listener,
    is_headless,
    predict_action,
    sanity_check_dataset_name,
    sanity_check_dataset_robot_compatibility,
)
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import (
    init_logging,
    log_say,
)
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

import pprint

pp = pprint.PrettyPrinter(indent=2)


@dataclass
class DatasetRecordConfig:
    # Dataset identifier. By convention it should match '{hf_username}/{dataset_name}' (e.g. `lerobot/test`).
    repo_id: str  # 数据集标识，格式通常为 '用户名/数据集名'
    # A short but accurate description of the task performed during the recording (e.g. "Pick the Lego block and drop it in the box on the right.")
    single_task: str  # 任务描述，如 "Grab the cube"
    # Root directory where the dataset will be stored (e.g. 'dataset/path'). If None, defaults to $HF_LEROBOT_HOME/repo_id.
    root: str | Path | None = None  # 本地存储根目录，默认 ~/.cache/huggingface/lerobot

    # Limit the frames per second.
    fps: int = 30
    # Number of seconds for data recording for each episode.
    episode_time_s: int | float = 60  # 每个回合（Episode）的最长录制时间（秒）
    # Number of seconds for resetting the environment after each episode.
    reset_time_s: int | float = 60  # 回合结束后，留给人工复位环境的时间
    # Number of episodes to record.
    num_episodes: int = 50

    # Upload dataset to Hugging Face hub.
    push_to_hub: bool = True  # 录制完成后是否自动上传到 Hugging Face Hub
    # Upload on private repository on the Hugging Face hub.
    private: bool = False  # 是否设为私有数据集
    # Add tags to your dataset on the hub.
    tags: list[str] | None = None  # 在 hub 中显示的数据集标签

    # Number of subprocesses handling the saving of frames as PNG. Set to 0 to use threads only;
    # set to ≥1 to use subprocesses, each using threads to write images. The best number of processes
    # and threads depends on your system. We recommend 4 threads per camera with 0 processes.
    # If fps is unstable, adjust the thread count. If still unstable, try using 1 or more subprocesses.
    num_image_writer_processes: int = 0  # 图像写入进程数。0 表示只使用线程
    # Number of threads writing the frames as png images on disk, per camera.
    # Too many threads might cause unstable teleoperation fps due to main thread being blocked.
    # Not enough threads might cause low camera fps.
    num_image_writer_threads_per_camera: int = 4  # 每个相机对应的图像写入 4 个线程数

    # Encode frames in the dataset into video
    video: bool = True  # 是否将图像帧编码为视频文件以节省空间
    # Number of episodes to record before batch encoding videos
    # Set to 1 for immediate encoding (default behavior), or higher for batched encoding
    video_encoding_batch_size: int = 1  # 多少个回合批量编码一次视频
    # Video codec for encoding videos. Options: 'h264', 'hevc', 'libsvtav1', 'auto',
    # or hardware-specific: 'h264_videotoolbox', 'h264_nvenc', 'h264_vaapi', 'h264_qsv'.
    # Use 'auto' to auto-detect the best available hardware encoder.
    vcodec: str = "libsvtav1"  # 视频编码器，libsvtav1 压缩率高，h264 兼容性好
    # Enable streaming video encoding: encode frames in real-time during capture instead
    # of writing PNG images first. Makes save_episode() near-instant. More info in the documentation: https://huggingface.co/docs/lerobot/streaming_video_encoding
    streaming_encoding: bool = False  # 是否开启流式编码（边录边编），开启后保存速度极快
    # Maximum number of frames to buffer per camera when using streaming encoding.
    # ~1s buffer at 30fps. Provides backpressure if the encoder can't keep up.
    encoder_queue_maxsize: int = 30  # 编码队列最大帧数缓存
    # Number of threads per encoder instance. None = auto (codec default).
    # Lower values reduce CPU usage, maps to 'lp' (via svtav1-params) for libsvtav1 and 'threads' for h264/hevc..
    encoder_threads: int | None = None  # 编码使用的线程数

    # Rename map for the observation to override the image and state keys
    rename_map: dict[str, str] = field(default_factory=dict)  # 重命名映射

    def __post_init__(self):
        if self.single_task is None:
            raise ValueError("You need to provide a task as argument in `single_task`.")


@dataclass
class RecordConfig:
    robot: RobotConfig
    dataset: DatasetRecordConfig
    # Whether to control the robot with a teleoperator
    teleop: TeleoperatorConfig | None = None  # 遥操作配置（可选）
    # Whether to control the robot with a policy
    policy: PreTrainedConfig | None = None  # 策略配置（可选）

    # Display all cameras on screen
    display_data: bool = False  # 是否显示数据
    # Display data on a remote Rerun server
    display_ip: str | None = None  # Rerun 服务器 IP
    # Port of the remote Rerun server
    display_port: int | None = None  # Rerun 服务器端口
    # Whether to  display compressed images in Rerun
    display_compressed_images: bool = False  # 是否显示压缩图像

    # Use vocal synthesis to read events.
    play_sounds: bool = True  # 是否播放语音提示
    # Resume recording on an existing dataset.
    resume: bool = False  # 是否恢复录制

    def __post_init__(self):
        """
        在对象初始化后执行的后处理方法。
        解析命令行参数以获取预训练模型路径，并根据需要加载预训练配置。
        同时验证是否设置了策略或遥操作控制器。
        """
        # HACK: We parse again the cli args here to get the pretrained path if there was one.
        # 从命令行参数中获取策略模型的路径
        policy_path = parser.get_path_arg("policy")

        # 如果存在策略路径，则继续处理
        if policy_path:
            # 获取策略相关的命令行覆盖参数
            cli_overrides = parser.get_cli_overrides("policy")

            # 从预训练模型路径创建配置对象，并应用命令行覆盖参数
            self.policy = PreTrainedConfig.from_pretrained(
                policy_path, cli_overrides=cli_overrides
            )
            # 设置预训练路径属性
            self.policy.pretrained_path = policy_path

        # 检查是否同时没有设置遥操作器和策略，如果是则抛出错误
        if self.teleop is None and self.policy is None:
            raise ValueError(
                "Choose a policy, a teleoperator or both to control the robot"
            )

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """
        获取路径字段列表，用于配置解析器从策略中加载配置

        该方法支持通过命令行参数 `--policy.path=local/dir` 的方式指定策略路径，
        使解析器能够正确识别并加载相应的配置文件。

        Args:
            cls: 类对象引用

        Returns:
            list[str]: 包含路径字段名称的列表，此处返回包含 "policy" 字符串的列表
        """
        return ["policy"]


""" --------------- record_loop() data flow --------------------------
       [ Robot ]
           V
     [ robot.get_observation() ] ---> raw_obs
           V
     [ robot_observation_processor ] ---> processed_obs
           V
     .-----( ACTION LOGIC )------------------.
     V                                       V
     [ From Teleoperator ]                   [ From Policy ]
     |                                       |
     |  [teleop.get_action] -> raw_action    |   [predict_action]
     |          |                            |          |
     |          V                            |          V
     | [teleop_action_processor]             |          |
     |          |                            |          |
     '---> processed_teleop_action           '---> processed_policy_action
     |                                       |
     '-------------------------.-------------'
                               V
                  [ robot_action_processor ] --> robot_action_to_send
                               V
                    [ robot.send_action() ] -- (Robot Executes)
                               V
                    ( Save to Dataset )
                               V
                  ( Rerun Log / Loop Wait )
"""


@safe_stop_image_writer
def record_loop(
    robot: Robot,  # 机器人实例
    events: dict,  # 事件字典（键盘控制）
    fps: int,  # 目标帧率
    # * 1、遥操作动作处理器
    teleop_action_processor: RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ],  # runs after teleop
    # * 2、机器人动作处理器
    robot_action_processor: RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ],  # runs before robot
    # * 3、机器人观测处理器
    robot_observation_processor: RobotProcessorPipeline[
        RobotObservation, RobotObservation
    ],  # runs after robot
    dataset: LeRobotDataset | None = None,  # 数据集对象（如果是只测试不录制则为None）
    teleop: Teleoperator | list[Teleoperator] | None = None,  # 遥操作设备
    policy: PreTrainedPolicy | None = None,  # 策略模型
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]] | None = None,
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction] | None = None,
    control_time_s: int | None = None,  # 控制时长
    single_task: str | None = None,
    display_data: bool = False,
    display_compressed_images: bool = False,  # 是否压缩
):
    """
    3种处理器：
    ### 三种处理器的角色和位置

    ┌─────────────────────────────────────────────────────────────────┐
    │                      完整的数据流                                │
    │                                                                 │
    │   [Robot]                                                       │
    │     │                                                           │
    │     │ raw_obs (原始观测：图像、关节位置)                          │
    │     ▼                                                           │
    │   ┌───────────────────────────────┐                             │
    │   │ ① robot_observation_processor │  <-- 处理观测               │
    │   │    (观测处理器)                │                             │
    │   └───────────────┬───────────────┘                             │
    │                   │                                             │
    │                   │ processed_obs (处理后的观测)                 │
    │                   ▼                                             │
    │   ┌───────────────┴───────────────┐                             │
    │   │        ACTION LOGIC           │                             │
    │   │  ┌─────────────────────────┐  │                             │
    │   │  │ Teleoperator            │  │                             │
    │   │  │   │                     │  │                             │
    │   │  │   │ raw_teleop_action   │  │                             │
    │   │  │   ▼                     │  │                             │
    │   │  │ ┌─────────────────────┐ │  │                             │
    │   │  │ │② teleop_action_     │ │  │                             │
    │   │  │ │  processor          │ │  │                             │
    │   │  │ └─────────┬───────────┘ │  │                             │
    │   │  └───────────┼─────────────┘  │                             │
    │   │              │                │                             │
    │   │              ▼                │                             │
    │   │   ┌─────────────────────┐     │                             │
    │   │   │③ robot_action_      │     │                             │
    │   │   │  processor          │     │                             │
    │   │   └─────────┬───────────┘     │                             │
    │   └─────────────┼─────────────────┘                             │
    │                 │                                               │
    │                 │ robot_action_to_send                          │
    │                 ▼                                               │
    │            [Robot] ← 执行动作                                    │
    └─────────────────────────────────────────────────────────────────┘


    录制循环的工作流程（时间线）：
        开始循环 (fps=30, 每帧 ~33ms)
        │
        ├── t=0ms:     start_loop_t = 当前时间
        │
        ├── t=0ms:     obs = robot.get_observation()
        │              │
        │              └── 相机捕获图像 (约33ms)
        │                  关节电机读取位置
        │
        ├── t=5ms:     obs_processed = processor(obs)
        │              │
        │              └── 处理观测数据（重命名、转换等）
        │
        ├── t=6ms:     --- 动作来源选择 ---
        │              │
        │              ├── 如果有 policy:
        │              │   action = predict_action(obs, policy)  # AI 推理 (约50-100ms)
        │              │
        │              └── 如果有 teleop:
        │                  action = teleop.get_action()          # 获取手柄位置
        │
        ├── t=106ms:   robot_action = processor(action)
        │              │
        │              └── 处理动作（归一化、限制范围等）
        │
        ├── t=107ms:   robot.send_action(robot_action)
        │              │
        │              └── 发送关节目标位置
        │                  电机执行动作
        │
        ├── t=108ms:   dataset.add_frame({obs, action, task})
        │              │
        │              └── 将观测+动作+任务保存到数据集
        │
        ├── t=110ms:   dt_s = 已用时间 (约 10ms)
        │              sleep_time = 1/fps - dt_s
        │                         = 33ms - 10ms = 23ms
        │              precise_sleep(sleep_time)  # 等待到33ms
        │
        └── t=33ms:    下一帧...
    """
    if dataset is not None and dataset.fps != fps:
        raise ValueError(
            f"The dataset fps should be equal to requested fps ({dataset.fps} != {fps})."
        )

    teleop_arm = teleop_keyboard = None
    if isinstance(teleop, list):
        teleop_keyboard = next(
            (t for t in teleop if isinstance(t, KeyboardTeleop)), None
        )
        teleop_arm = next(
            (
                t
                for t in teleop
                if isinstance(
                    t,
                    (
                        so_leader.SO100Leader
                        | so_leader.SO101Leader
                        | koch_leader.KochLeader
                        | omx_leader.OmxLeader
                    ),
                )
            ),
            None,
        )

        if not (
            teleop_arm
            and teleop_keyboard
            and len(teleop) == 2
            and robot.name == "lekiwi_client"
        ):
            raise ValueError(
                "For multi-teleop, the list must contain exactly one KeyboardTeleop and one arm teleoperator. Currently only supported for LeKiwi robot."
            )

    # Reset policy and processor if they are provided
    if policy is not None and preprocessor is not None and postprocessor is not None:
        policy.reset()
        preprocessor.reset()
        postprocessor.reset()

    # step1. 初始化时间戳和循环变量
    no_action_count = 0
    timestamp = 0
    start_episode_t = time.perf_counter()

    # NOTE 在设定的回合时间内录制某 episode
    while timestamp < control_time_s:
        start_loop_t = time.perf_counter()

        # 如果按下特定键，提前结束本回合
        if events["exit_early"]:
            events["exit_early"] = False
            break

        # step2. 读取机器人当前原始观测（包括图像 + 关节角度）
        obs = robot.get_observation()

        # Applies a pipeline to the raw robot observation, default is IdentityProcessor
        # 处理原始观测数据，此时为处理后的观测数据
        obs_processed = robot_observation_processor(obs)

        # step3. 构造数据集格式的帧
        if policy is not None or dataset is not None:
            observation_frame = build_dataset_frame(
                dataset.features, obs_processed, prefix=OBS_STR
            )

        # step4. 获取动作（来自 policy 还是来自 teleop？）
        # case1 如果有 AI 策略，由 AI 决策
        if (
            policy is not None
            and preprocessor is not None
            and postprocessor is not None
        ):
            action_values = predict_action(
                observation=observation_frame,
                policy=policy,
                device=get_safe_torch_device(policy.config.device),
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                use_amp=policy.config.use_amp,
                task=single_task,
                robot_type=robot.robot_type,
            )

            act_processed_policy: RobotAction = make_robot_action(
                action_values, dataset.features
            )

        # case2 如果是人类遥操作，读取 Leader 臂动作
        elif policy is None and isinstance(teleop, Teleoperator):
            if robot.name == "unitree_g1":
                teleop.send_feedback(obs)
            act = teleop.get_action()

            # Applies a pipeline to the raw teleop action, default is IdentityProcessor
            act_processed_teleop = teleop_action_processor((act, obs))

        elif policy is None and isinstance(teleop, list):
            arm_action = teleop_arm.get_action()
            arm_action = {f"arm_{k}": v for k, v in arm_action.items()}
            keyboard_action = teleop_keyboard.get_action()
            base_action = robot._from_keyboard_to_base_action(keyboard_action)
            act = {**arm_action, **base_action} if len(base_action) > 0 else arm_action
            act_processed_teleop = teleop_action_processor((act, obs))
        else:
            no_action_count += 1
            if no_action_count == 1 or no_action_count % 10 == 0:
                logging.warning(
                    "No policy or teleoperator provided, skipping action generation. "
                    "This is likely to happen when resetting the environment without a teleop device. "
                    "The robot won't be at its rest position at the start of the next episode."
                )
            continue

        # step5. 执行动作
        # Applies a pipeline to the action, default is IdentityProcessor
        if policy is not None and act_processed_policy is not None:
            action_values = act_processed_policy
            robot_action_to_send = robot_action_processor((act_processed_policy, obs))
        else:
            action_values = act_processed_teleop
            robot_action_to_send = robot_action_processor((act_processed_teleop, obs))

        # Send action to robot
        # Action can eventually be clipped using `max_relative_target`,
        # so action actually sent is saved in the dataset. action = postprocessor.process(action)
        # TODO(steven, pepijn, adil): we should use a pipeline step to clip the action, so the sent action is the action that we input to the robot.
        _sent_action = robot.send_action(robot_action_to_send)

        # step6. 存入数据集
        # Write to dataset
        if dataset is not None:
            action_frame = build_dataset_frame(
                dataset.features, action_values, prefix=ACTION
            )
            frame = {**observation_frame, **action_frame, "task": single_task}
            dataset.add_frame(frame)  # 将本帧数据加入缓存

        if display_data:
            log_rerun_data(
                observation=obs_processed,
                action=action_values,
                compress_images=display_compressed_images,
            )

        # step7. 控制频率：睡眠多余的时间，确保稳定的 FPS
        dt_s = time.perf_counter() - start_loop_t

        sleep_time_s: float = 1 / fps - dt_s
        if sleep_time_s < 0:
            logging.warning(
                f"Record loop is running slower ({1 / dt_s:.1f} Hz) than the target FPS ({fps} Hz). Dataset frames might be dropped and robot control might be unstable. Common causes are: 1) Camera FPS not keeping up 2) Policy inference taking too long 3) CPU starvation"
            )

        precise_sleep(max(sleep_time_s, 0.0))

        timestamp = time.perf_counter() - start_episode_t


@parser.wrap()
def record(cfg: RecordConfig) -> LeRobotDataset:
    """
    记录机器人数据集，通过连接机器人和可选的远程操作设备来收集数据

    Args:
        cfg (RecordConfig): 记录配置对象，包含数据集、机器人、远程操作等配置信息

    Returns:
        LeRobotDataset: 包含记录数据的 LeRobot 数据集对象
    """
    # 这里开始接上面 @parser.wrap() 装饰器返回的值, 进而 record() 函数开始执行，使用解析好的配置...

    init_logging()
    # logging.info(pformat(asdict(cfg)))  # 等价于下面几步:
    cfg_dict = asdict(cfg)
    # asdict(cfg) 将 dataclass 转为普通字典
    # {
    #     'robot': {'type': 'so100_follower', 'port': '/dev/ttyUSB0', ...},
    #     'dataset': {'repo_id': 'ghr/demo_dataset', 'num_episodes': 2, ...},
    #     'teleop': {'type': 'so100_leader', ...},
    #     'policy': None,
    #     ...
    # }
    # pformat() 将字典格式化为漂亮的字符串（适合打印/日志）
    formatted_str = pformat(cfg_dict)
    # "{\n    'robot': {\n        'type': 'so100_follower',\n        ...\n    },\n    ..."
    logging.info(formatted_str)

    print(f"# cfg type: ", type(cfg))  # <class '__main__.RecordConfig'>
    print("# cfg: ")
    pp.pprint(cfg)
    # RecordConfig(robot=SOFollowerRobotConfig(port='/dev/tty.usbmodem58760431541',
    #                                      disable_torque_on_disconnect=True,
    #                                      max_relative_target=None,
    #                                      cameras={ 'laptop': OpenCVCameraConfig(fps=30,
    #                                                                             width=640,
    #                                                                             height=480,
    #                                                                             index_or_path=0,
    #                                                                             color_mode=<ColorMode.RGB: 'rgb'>,
    #                                                                             rotation=<Cv2Rotation.NO_ROTATION: 0
    #                                                                             warmup_s=1,
    #                                                                             fourcc=None,
    #                                                                             backend=<Cv2Backends.ANY: 0>)},
    #                                      use_degrees=True,
    #                                      id='black',
    #                                      calibration_dir=None),
    #          dataset=DatasetRecordConfig(repo_id='ghr/demo_dataset',
    #                                      single_task='This is a demo dataset',
    #                                      root=None,
    #                                      fps=30,
    #                                      episode_time_s=60,
    #                                      reset_time_s=60,
    #                                      num_episodes=2,
    #                                      push_to_hub=True,
    #                                      private=False,
    #                                      tags=None,
    #                                      num_image_writer_processes=0,
    #                                      num_image_writer_threads_per_camera=4,
    #                                      video=True,
    #                                      video_encoding_batch_size=1,
    #                                      vcodec='libsvtav1',
    #                                      streaming_encoding=True,
    #                                      encoder_queue_maxsize=30,
    #                                      encoder_threads=2,
    #                                      rename_map={}),
    #          teleop=SOLeaderTeleopConfig(port='/dev/tty.usbmodem58760431551',
    #                                      use_degrees=True,
    #                                      id='blue',
    #                                      calibration_dir=None),
    #          policy=None,
    #          display_data=True,
    #          display_ip=None,
    #          display_port=None,
    #          display_compressed_images=False,
    #          play_sounds=True,
    #          resume=False)

    if cfg.display_data:
        init_rerun(session_name="recording", ip=cfg.display_ip, port=cfg.display_port)
    display_compressed_images = (
        True
        if (
            cfg.display_data
            and cfg.display_ip is not None
            and cfg.display_port is not None
        )
        else cfg.display_compressed_images
    )

    robot = make_robot_from_config(cfg.robot)
    teleop = (
        make_teleoperator_from_config(cfg.teleop) if cfg.teleop is not None else None
    )

    teleop_action_processor, robot_action_processor, robot_observation_processor = (
        make_default_processors()
    )

    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(
                action=robot.action_features
            ),  # TODO(steven, pepijn): in future this should be come from teleop or policy
            use_videos=cfg.dataset.video,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(
                observation=robot.observation_features
            ),
            use_videos=cfg.dataset.video,
        ),
    )

    dataset = None
    listener = None

    try:
        # case1 开启断点续训，则加载现有保存的片段继续录制
        if cfg.resume:
            num_cameras = len(robot.cameras) if hasattr(robot, "cameras") else 0
            dataset = LeRobotDataset.resume(
                cfg.dataset.repo_id,
                root=cfg.dataset.root,
                batch_encoding_size=cfg.dataset.video_encoding_batch_size,
                vcodec=cfg.dataset.vcodec,
                streaming_encoding=cfg.dataset.streaming_encoding,
                encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
                encoder_threads=cfg.dataset.encoder_threads,
                image_writer_processes=(
                    cfg.dataset.num_image_writer_processes if num_cameras > 0 else 0
                ),
                image_writer_threads=(
                    cfg.dataset.num_image_writer_threads_per_camera * num_cameras
                    if num_cameras > 0
                    else 0
                ),
            )
            sanity_check_dataset_robot_compatibility(
                dataset, robot, cfg.dataset.fps, dataset_features
            )

        # case2 创建空数据集从头录制
        else:
            # Create empty dataset or load existing saved episodes
            sanity_check_dataset_name(cfg.dataset.repo_id, cfg.policy)
            dataset = LeRobotDataset.create(
                cfg.dataset.repo_id,
                cfg.dataset.fps,
                root=cfg.dataset.root,
                robot_type=robot.name,
                features=dataset_features,
                use_videos=cfg.dataset.video,
                image_writer_processes=cfg.dataset.num_image_writer_processes,
                image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera
                * len(robot.cameras),
                batch_encoding_size=cfg.dataset.video_encoding_batch_size,
                vcodec=cfg.dataset.vcodec,
                streaming_encoding=cfg.dataset.streaming_encoding,
                encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
                encoder_threads=cfg.dataset.encoder_threads,
            )

        # 加载预训练策略
        policy = (
            None
            if cfg.policy is None
            else make_policy(cfg.policy, ds_meta=dataset.meta)
        )
        preprocessor = None
        postprocessor = None
        if cfg.policy is not None:
            preprocessor, postprocessor = make_pre_post_processors(
                policy_cfg=cfg.policy,
                pretrained_path=cfg.policy.pretrained_path,
                dataset_stats=rename_stats(dataset.meta.stats, cfg.dataset.rename_map),
                preprocessor_overrides={
                    "device_processor": {"device": cfg.policy.device},
                    "rename_observations_processor": {
                        "rename_map": cfg.dataset.rename_map
                    },
                },
            )

        # * 连接机器人和远程操作设备
        robot.connect()
        if teleop is not None:
            teleop.connect()

        # 初始化键盘监听器
        listener, events = init_keyboard_listener()

        # 提示用户关于流编码的性能优化建议
        if not cfg.dataset.streaming_encoding:
            logging.info(
                "Streaming encoding is disabled. If you have capable hardware, consider enabling it for way faster episode saving. --dataset.streaming_encoding=true --dataset.encoder_threads=2 # --dataset.vcodec=auto. More info in the documentation: https://huggingface.co/docs/lerobot/streaming_video_encoding"
            )

        # NOTE 开始录制主循环
        with VideoEncodingManager(dataset):
            recorded_episodes = 0
            while (
                recorded_episodes < cfg.dataset.num_episodes
                and not events["stop_recording"]
            ):
                # 此处有语音播报...
                log_say(f"Recording episode {dataset.num_episodes}", cfg.play_sounds)

                # * 开始录制当前 episode...
                record_loop(
                    robot=robot,
                    events=events,
                    fps=cfg.dataset.fps,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    teleop=teleop,
                    policy=policy,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    dataset=dataset,
                    control_time_s=cfg.dataset.episode_time_s,
                    single_task=cfg.dataset.single_task,
                    display_data=cfg.display_data,
                    display_compressed_images=display_compressed_images,
                )

                # Execute a few seconds without recording to give time to manually reset the environment
                # Skip reset for the last episode to be recorded
                if not events["stop_recording"] and (
                    (recorded_episodes < cfg.dataset.num_episodes - 1)
                    or events["rerecord_episode"]
                ):
                    # 重置环境（不录制）
                    log_say("Reset the environment", cfg.play_sounds)

                    record_loop(
                        robot=robot,
                        events=events,
                        fps=cfg.dataset.fps,
                        teleop_action_processor=teleop_action_processor,
                        robot_action_processor=robot_action_processor,
                        robot_observation_processor=robot_observation_processor,
                        teleop=teleop,
                        control_time_s=cfg.dataset.reset_time_s,
                        single_task=cfg.dataset.single_task,
                        display_data=cfg.display_data,
                    )

                if events["rerecord_episode"]:
                    log_say("Re-record episode", cfg.play_sounds)
                    events["rerecord_episode"] = False
                    events["exit_early"] = False
                    dataset.clear_episode_buffer()
                    continue

                # 保存录制的这个 episode
                dataset.save_episode()
                recorded_episodes += 1

    finally:
        # 清理资源并保存数据集
        log_say("Stop recording", cfg.play_sounds, blocking=True)

        if dataset:
            dataset.finalize()

        if robot.is_connected:
            robot.disconnect()
        if teleop and teleop.is_connected:
            teleop.disconnect()

        if not is_headless() and listener:
            listener.stop()

        if cfg.dataset.push_to_hub:
            dataset.push_to_hub(tags=cfg.dataset.tags, private=cfg.dataset.private)

        log_say("Exiting", cfg.play_sounds)
    return dataset


def main():
    register_third_party_plugins()
    record()


if __name__ == "__main__":
    main()
