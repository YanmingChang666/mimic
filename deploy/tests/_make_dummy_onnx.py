"""_make_dummy_onnx.py — 生成一个「结构等价」的假 SMP 策略 onnx，用于离线验证部署栈。

真实策略由 scripts/export_onnx.py 从 mjlab 导出（本容器没装 mjlab，跑不了）。这里用 torch 造一个
输入 obs[1,101]、输出 actions[1,29] 的 MLP，并按 mjlab 约定挂上 metadata（joint_names 用一个
非平凡的关节序，以真正检验电机序↔策略序重排），供 test_steering_deploy.py 端到端跑通。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import torch
import torch.nn as nn

# 策略(seq)关节序：故意不同于 Unitree 电机序(MOTOR_ORDER)，以检验重排正确性。
SEQ_JOINT_NAMES = [
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint", "left_hip_roll_joint",
    "right_hip_roll_joint", "waist_roll_joint", "left_hip_yaw_joint", "right_hip_yaw_joint",
    "waist_pitch_joint", "left_knee_joint", "right_knee_joint", "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint", "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint", "left_ankle_roll_joint",
    "right_ankle_roll_joint", "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint", "left_wrist_roll_joint", "right_wrist_roll_joint",
    "left_wrist_pitch_joint", "right_wrist_pitch_joint", "left_wrist_yaw_joint", "right_wrist_yaw_joint",
]


class _MLP(nn.Module):
    def __init__(self, num_obs=101, num_act=29):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_obs, 512), nn.ELU(),
            nn.Linear(512, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, num_act),
        )

    def forward(self, obs):
        return self.net(obs)


def make_dummy_onnx(path: str, num_obs: int = 101) -> str:
    torch.manual_seed(0)
    model = _MLP(num_obs=num_obs).eval()
    dummy = torch.zeros(1, num_obs)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, dummy, path, input_names=["obs"], output_names=["actions"],
                      opset_version=17, dynamic_axes=None)

    n = len(SEQ_JOINT_NAMES)
    stiffness = (np.arange(n) + 10.0)             # seq 序，逐关节不同
    damping = (np.arange(n) * 0.1 + 1.0)
    default = (np.arange(n) * 0.01)
    action_scale = (0.1 + np.arange(n) * 0.01)

    m = onnx.load(path)
    def add(k, v):
        e = m.metadata_props.add(); e.key = k; e.value = v
    add("joint_names", ",".join(SEQ_JOINT_NAMES))
    add("joint_stiffness", ",".join(f"{x:.4f}" for x in stiffness))
    add("joint_damping", ",".join(f"{x:.4f}" for x in damping))
    add("default_joint_pos", ",".join(f"{x:.4f}" for x in default))
    add("action_scale", ",".join(f"{x:.4f}" for x in action_scale))
    add("observation_names", "base_lin_vel,base_ang_vel,projected_gravity,joint_pos,joint_vel,actions,command")
    onnx.save(m, path)
    return path


if __name__ == "__main__":
    p = make_dummy_onnx(str(Path(__file__).parent / "_dummy_smp_policy.onnx"))
    print("wrote", p)
