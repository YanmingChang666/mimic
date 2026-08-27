"""ctrlcomp.py — FSM 与外层(仿真/真机)之间的两个共享黑板（SMP Forward/Steering 版）。

  StateAndCmd ：外层每帧写入(机器人状态 + 遥控器指令) → state 读取。
  PolicyOutput：state 每帧写入(目标关节角 + PD 增益) → 外层读取并下发电机。

关节量按 **Unitree 电机序**(legs 12, waist 3, arms 14) 排列，G1 共 29 维。

SMP 观测(101 维)所需状态字段（与 g1_smp_env_cfg 的 actor 观测组对齐）：
  base_lin_vel(3) base_ang_vel(3) projected_gravity(3) joint_pos(29) joint_vel(29) actions(29)
  + command(5, steering) —— command 由摇杆经 vel_cmd 生成，见 SteeringMode。
"""

from __future__ import annotations

import numpy as np

from common.fsm_utils import FSMCommand


class StateAndCmd:
  def __init__(self, num_joints: int = 29):
    self.num_joints = num_joints
    # ---- 机器人本体感受状态（电机序 / 机体系）----
    self.q = np.zeros(num_joints, dtype=np.float32)          # 关节位置 (rad)
    self.dq = np.zeros(num_joints, dtype=np.float32)         # 关节速度 (rad/s)
    self.base_lin_vel = np.zeros(3, dtype=np.float32)        # 机体系线速度 (m/s)；真机需状态估计
    self.ang_vel = np.zeros(3, dtype=np.float32)             # 机体系角速度 (rad/s)
    self.projected_gravity = np.array([0.0, 0.0, -1.0], dtype=np.float32)  # 重力投影(姿态)

    # ---- 遥控器指令 ----
    self.skill_cmd = FSMCommand.INVALID   # 技能切换（消费后置回 INVALID）
    # 摇杆原始指令 [fwd, left, turn] ∈ [-1,1]：左摇杆前后/左右 + 右摇杆左右。
    # SteeringMode 据此生成 5 维 steering 指令。
    self.vel_cmd = np.zeros(3, dtype=np.float32)


class PolicyOutput:
  def __init__(self, num_joints: int = 29):
    self.actions = np.zeros(num_joints, dtype=np.float32)   # 目标关节角 (rad)，电机序
    self.kps = np.zeros(num_joints, dtype=np.float32)
    self.kds = np.zeros(num_joints, dtype=np.float32)
