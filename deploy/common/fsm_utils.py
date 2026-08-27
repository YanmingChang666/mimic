"""fsm_utils.py — FSM 枚举 + SMP 部署共用的数学工具。

SMP 的下游任务(Forward/Steering)是**速度/朝向指令驱动**的行走策略，遥控器摇杆直接映射为
steering 指令，因此这里的状态只需 3 个：阻尼保护 / 就绪位 / 行走(SteeringMode)。

  FSMStateName：状态身份枚举。
  FSMCommand  ：遥控器组合键映射出的指令枚举。
"""

from __future__ import annotations

from enum import Enum, unique

import numpy as np


@unique
class FSMStateName(Enum):
  INVALID = -1
  PASSIVE = 1      # 阻尼保护（开机默认）
  FIXEDPOSE = 2    # 位控保持默认站姿
  STEERING = 3     # SMP Forward/Steering 策略：速度+朝向指令行走


@unique
class FSMCommand(Enum):
  INVALID = -1
  PASSIVE = 1      # L1(仿真)/F1(真机)：阻尼保护
  POS_RESET = 2    # Start：进入就绪位
  STEER = 3        # R1+A：进入行走(SteeringMode)


def get_gravity_orientation(quaternion) -> np.ndarray:
  """机体姿态四元数(w,x,y,z) → 重力方向在机体系下的单位投影（projected_gravity 观测）。"""
  qw, qx, qy, qz = quaternion[0], quaternion[1], quaternion[2], quaternion[3]
  g = np.zeros(3)
  g[0] = 2 * (-qz * qx + qw * qy)
  g[1] = -2 * (qz * qy + qw * qx)
  g[2] = 1 - 2 * (qw * qw + qz * qz)
  return g


def quat_rotate_inverse(q, v) -> np.ndarray:
  """用四元数 q(w,x,y,z) 的逆旋转向量 v(x,y,z)：把世界系向量转到机体系。

  用于 sim2sim 里把 MuJoCo 世界系的 base 线/角速度转成策略训练时的机体系观测。
  """
  q = np.asarray(q, dtype=np.float64)
  v = np.asarray(v, dtype=np.float64)
  q_w = q[0]
  q_vec = q[1:]
  a = v * (2.0 * q_w**2 - 1.0)
  b = np.cross(q_vec, v) * q_w * 2.0
  c = q_vec * (np.dot(q_vec, v)) * 2.0
  return a - b + c
