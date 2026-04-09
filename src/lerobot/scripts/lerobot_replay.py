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
Replays the actions of an episode from a dataset on a robot.

Examples:

```shell
lerobot-replay \
    --robot.type=so100_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.id=black \
    --dataset.repo_id=<USER>/record-test \
    --dataset.episode=0
```

Example replay with bimanual so100:
```shell
lerobot-replay \
  --robot.type=bi_so_follower \
  --robot.left_arm_port=/dev/tty.usbmodem5A460851411 \
  --robot.right_arm_port=/dev/tty.usbmodem5A460812391 \
  --robot.id=bimanual_follower \
  --dataset.repo_id=${HF_USER}/bimanual-so100-handover-cube \
  --dataset.episode=0
```

"""

import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat
import pprint

pp = pprint.PrettyPrinter(indent=2)

from lerobot.configs import parser
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.processor import (
    make_default_robot_action_processor,
)
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
    unitree_g1,
)
from lerobot.utils.constants import ACTION
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import (
    init_logging,
    log_say,
)


@dataclass
class DatasetReplayConfig:
    # Dataset identifier. By convention it should match '{hf_username}/{dataset_name}' (e.g. `lerobot/test`).
    repo_id: str
    # Episode to replay.
    episode: int
    # Root directory where the dataset will be stored (e.g. 'dataset/path'). If None, defaults to $HF_LEROBOT_HOME/repo_id.
    root: str | Path | None = None
    # Limit the frames per second. By default, uses the policy fps.
    fps: int = 30


@dataclass
class ReplayConfig:
    robot: RobotConfig
    dataset: DatasetReplayConfig
    # Use vocal synthesis to read events.
    play_sounds: bool = True


@parser.wrap()
def replay(cfg: ReplayConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))
    cfg_dict = asdict(cfg)
    formatted_str = pformat(cfg_dict)
    logging.info(formatted_str)

    print(f"# cfg type: ", type(cfg))  # <class '__main__.RecordConfig'>
    print("# cfg: ")
    pp.pprint(cfg)
    # cfg = ReplayConfig(
    #     robot=SOFollowerRobotConfig(type="so101_follower", port="/dev/ttyUSB0", id="black"),
    #     dataset=DatasetReplayConfig(repo_id="lerobot/pusht", episode=0),
    #     play_sounds=True,
    # )

    # * 创建动作处理器
    # 确保数据集中的抽象动作值能正确映射到具体机器人的电机指令上
    robot_action_processor = make_default_robot_action_processor()

    robot = make_robot_from_config(cfg.robot)  # 实例化一个 SO101Follower 对象
    dataset = LeRobotDataset(
        cfg.dataset.repo_id, root=cfg.dataset.root, episodes=[cfg.dataset.episode]
    )

    # 从数据集中筛选出所有 "action" 列
    actions = dataset.select_columns(ACTION)

    robot.connect()

    try:
        log_say("Replaying episode", cfg.play_sounds, blocking=True)

        # 核心循环：遍历本回合所有的帧（dataset.num_frames）
        for idx in range(dataset.num_frames):
            # 记录本帧开始的时间戳，用于后续控制 FPS
            start_episode_t = time.perf_counter()

            # step1. 从 actions 数组中取出第 idx 帧的动作数据（通常是 Tensor 或数组）
            action_array = actions[idx][ACTION]

            # step2. 将数组转换为字典格式 {电机名: 目标值}
            # 输入: [0.1, 0.5, ...]; 输出: {"joint_1": 0.1, "joint_2": 0.5, ...}
            action = {}
            for i, name in enumerate(dataset.features[ACTION]["names"]):
                action[name] = action_array[i]

            # step3. 获取机器人当前观测（关节实际位置、速度等）
            # 作用: 有些动作处理器需要根据当前位置来计算相对移动量
            robot_obs = robot.get_observation()

            # step4. 处理动作：将数据集中的通用动作转换为机器人能理解的具体数值
            # 输入: (原始目标动作字典, 原始机器人当前状态); 输出: 处理后的动作（可直接传递给机器人来执行）
            processed_action = robot_action_processor((action, robot_obs))

            # step5. 发送指令给硬件：电机开始旋转
            # 数据变化: 机器人产生物理位移
            _ = robot.send_action(processed_action)

            # step6. 频率控制（保持同步）
            # 计算这一帧已经消耗了多久 (dt_s)
            dt_s = time.perf_counter() - start_episode_t

            # 如果跑快了，就精确睡眠一段时间，确保按照 dataset.fps (如30帧) 运行
            precise_sleep(max(1 / dataset.fps - dt_s, 0.0))
    finally:
        robot.disconnect()


def main():
    register_third_party_plugins()
    replay()


if __name__ == "__main__":
    main()
