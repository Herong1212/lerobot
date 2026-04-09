import dotenv

dotenv.load_dotenv("/home/robot/lerobot/.env")

import torch

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.feature_utils import hw_to_dataset_features
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.policies.utils import build_inference_frame, make_robot_action
from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig

# 定义运行参数：跑 5 个回合，每个回合最多走 20 步
MAX_EPISODES = 5
MAX_STEPS_PER_EPISODE = 20


def main():
    device = torch.device("cuda")  # or "cuda" or "cpu"
    model_id = "lerobot/pi0_base"

    # step1 加载预训练模型
    model = PI0Policy.from_pretrained(model_id)

    # step2. 创建预处理器和后处理器
    # 关键！它会自动根据模型卡片里的信息，处理图像缩放、均值归一化等
    preprocess, postprocess = make_pre_post_processors(
        model.config,
        model_id,
        # This overrides allows to run on MPS, otherwise defaults to CUDA (if available)
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    # step3. 硬件连接配置
    # find ports using lerobot-find-port
    follower_port = ...  # something like "/dev/tty.usbmodem58760431631"

    # the robot ids are used the load the right calibration files
    follower_id = ...  # something like "follower_so100"

    # step4. 相机配置：必须与模型训练时的分辨率、名称完全对齐！
    # 如果模型训练时用了三路相机，你部署时也必须提供三路，否则维度会报错
    # Robot and environment configuration
    # Camera keys must match the name and resolutions of the ones used for training!
    # You can check the camera keys expected by a model in the info.json card on the model card on the Hub
    camera_config = {
        "base_0_rgb": OpenCVCameraConfig(
            index_or_path=0, width=640, height=480, fps=30
        ),
        "left_wrist_0_rgb": OpenCVCameraConfig(
            index_or_path=1, width=640, height=480, fps=30
        ),
        "right_wrist_0_rgb": OpenCVCameraConfig(
            index_or_path=2, width=640, height=480, fps=30
        ),
    }

    # 初始化机器人硬件并连接
    robot_cfg = SO100FollowerConfig(
        port=follower_port, id=follower_id, cameras=camera_config
    )
    robot = SO100Follower(robot_cfg)
    robot.connect()

    # 设定任务指令和机器人类型（用于多任务/多机型模型）
    task = ""  # something like "pick the red block"
    robot_type = ""  # something like "so100_follower" for multi-embodiment datasets

    # step5. 特征映射：将硬件产生的原始键名转换为数据集标准的键名
    # 比如硬件叫 'qpos'，数据集可能叫 'observation.state'
    # This is used to match the raw observation keys to the keys expected by the policy
    action_features = hw_to_dataset_features(robot.action_features, "action")
    obs_features = hw_to_dataset_features(robot.observation_features, "observation")
    dataset_features = {**action_features, **obs_features}

    # step6. --- 开始主循环 ---
    for _ in range(MAX_EPISODES):
        for _ in range(MAX_STEPS_PER_EPISODE):
            # A. 感知：从机器人传感器获取当前画面和关节角度
            obs = robot.get_observation()

            # B. 组装：将原始观测包装成模型推理所需的“帧”格式
            obs_frame = build_inference_frame(
                observation=obs,
                ds_features=dataset_features,
                device=device,
                task=task,
                robot_type=robot_type,
            )

            # C. 预处理：图像归一化、维度重排 (H,W,C -> C,H,W) 等
            obs = preprocess(obs_frame)

            # D. 推理：模型根据图像和任务，选择最优动作
            # ! 此时输出的 action 通常是归一化后的数值（如 -1 到 1）
            action = model.select_action(obs)

            # E. 后处理：将模型输出还原为真实的物理数值（如角度、弧度）
            action = postprocess(action)

            # F. 适配：根据硬件定义的格式，包装成可发送的字典
            action = make_robot_action(action, dataset_features)

            # G. 执行：将指令通过串口发送给机械臂舵机
            robot.send_action(action)

        print("Episode finished! Starting new episode...")


if __name__ == "__main__":
    main()
