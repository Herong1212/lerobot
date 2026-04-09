#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

from lerobot.types import RobotAction, RobotObservation

from .converters import (
    observation_to_transition,
    robot_action_observation_to_transition,
    transition_to_observation,
    transition_to_robot_action,
)
from .pipeline import IdentityProcessorStep, RobotProcessorPipeline


def make_default_teleop_action_processor() -> (
    RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction]
):
    teleop_action_processor = RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ](
        steps=[IdentityProcessorStep()],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )
    return teleop_action_processor


# NOTE 这是一个工厂函数，返回的是一个`RobotProcessorPipeline`对象
def make_default_robot_action_processor() -> (
    RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction]
):
    """
    作用：
        - 将数据集中的动作转换为机器人可执行的动作；
        - 处理动作和观测的预处理；
        - 确保动作在机器人安全范围内；
    处理流程：
        数据集动作 → 动作处理器 → 机器人可执行动作 → 发送给机器人

    输入：(action, robot_obs) 元组
            action: RobotAction = dict[str, Any], 数据集中的动作，如 {"motor_0": 0.1, "motor_1": 0.2}
            robot_obs: RobotObservation = dict[str, Any], 机器人的当前观测，如关节位置、速度等

    输出: processed_action
            RobotAction = dict[str, Any], 机器人可执行的动作，如 {"joint_0": 0.1, "joint_1": 0.2}

    """
    robot_action_processor = RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ](
        # 处理器步骤列表，恒等处理器（原样传递），也可以添加自定义处理器（如限幅、归一化等）
        steps=[IdentityProcessorStep()],
        # 转换函数1：将 `(action, robot_obs)` 转换为 `Transition`，`Transition` 是 LeRobot 的统一数据格式
        to_transition=robot_action_observation_to_transition,
        # 转换函数2：将 `Transition` 转换为 `RobotAction`，最终输出机器人可执行的动作
        to_output=transition_to_robot_action,
    )
    return robot_action_processor


def make_default_robot_observation_processor() -> (
    RobotProcessorPipeline[RobotObservation, RobotObservation]
):
    robot_observation_processor = RobotProcessorPipeline[
        RobotObservation, RobotObservation
    ](
        steps=[IdentityProcessorStep()],
        to_transition=observation_to_transition,
        to_output=transition_to_observation,
    )
    return robot_observation_processor


def make_default_processors():
    teleop_action_processor = make_default_teleop_action_processor()
    robot_action_processor = make_default_robot_action_processor()
    robot_observation_processor = make_default_robot_observation_processor()
    return (
        teleop_action_processor,
        robot_action_processor,
        robot_observation_processor,
    )
