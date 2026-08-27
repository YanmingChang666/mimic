"""onnx_policy.py — 从导出的 policy.onnx 元数据加载 BeyondMimic 策略参数（单一来源）。

BeyondMimic 的 play.py 把「权威」控制参数烘焙进 policy.onnx 的 metadata：关节刚度/阻尼、
默认关节角、逐关节 action_scale、关节顺序、观测维度。本模块把这些读出来，并同时给出
两套关节序，供部署闭环复用：

  seq 序   ：ONNX ``joint_names`` 顺序（网络内部序）。metadata 里的增益/默认角/scale 均此序。
  motor 序 ：Unitree G1 SDK 电机下标序（legs 12, waist 3, arms 14）。仿真 d.qpos[7:] /
             真机 motor_cmd[i] 都用此序。

sim 与 real 都用它 → 消除真机端手抄 yaml 增益导致的 sim2real 漂移（对应旧
gen_deploy_config.py 的动机，这里直接在运行时读取，无需中间 yaml）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import onnx

# Unitree G1 29-DOF 电机序（== 旧 deploy 脚本里的 joint_xml / gen_deploy_config.py 的 MOTOR_ORDER）。
MOTOR_ORDER = [
  "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
  "left_ankle_pitch_joint", "left_ankle_roll_joint",
  "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
  "right_ankle_pitch_joint", "right_ankle_roll_joint",
  "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
  "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
  "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
  "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
  "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]


@dataclass
class PolicyBundle:
  """policy.onnx 解析结果 + 两套关节序的重排索引。"""

  onnx_path: str
  joint_names: list           # seq 序
  num_obs: int
  num_actions: int
  anchor_body_name: str
  # seq 序参数（与 metadata 原样一致）
  stiffness_seq: np.ndarray
  damping_seq: np.ndarray
  default_seq: np.ndarray
  action_scale_seq: np.ndarray
  # motor 序参数（下发用）
  stiffness_motor: np.ndarray
  damping_motor: np.ndarray
  default_motor: np.ndarray
  # 重排索引：seq_arr[idx_seq_for_motor] -> motor_arr;  motor_arr[idx_motor_for_seq] -> seq_arr
  idx_seq_for_motor: np.ndarray
  idx_motor_for_seq: np.ndarray

  def seq_to_motor(self, seq_arr: np.ndarray) -> np.ndarray:
    """把 seq 序的逐关节数组重排为 motor 序。"""
    return np.asarray(seq_arr)[self.idx_seq_for_motor]

  def motor_to_seq(self, motor_arr: np.ndarray) -> np.ndarray:
    """把 motor 序的逐关节数组重排为 seq 序。"""
    return np.asarray(motor_arr)[self.idx_motor_for_seq]


def _floats(meta: dict, key: str) -> np.ndarray:
  return np.array([float(x) for x in meta[key].split(",")], dtype=np.float32)


def load_policy_bundle(onnx_path: str) -> PolicyBundle:
  """读取 policy.onnx 的 metadata + 图 IO，返回 PolicyBundle（不创建推理会话）。"""
  model = onnx.load(onnx_path)
  meta = {p.key: p.value for p in model.metadata_props}
  required = ["joint_names", "joint_stiffness", "joint_damping", "default_joint_pos", "action_scale"]
  missing = [k for k in required if k not in meta]
  if missing:
    raise SystemExit(
      f"[onnx_policy] {onnx_path} 缺少 metadata {missing}；"
      "请用本仓库 play.py(attach_onnx_metadata) 导出，把增益烘焙进 onnx。"
    )

  joint_names = [s.strip() for s in meta["joint_names"].split(",")]  # seq 序
  name_to_seq = {n: i for i, n in enumerate(joint_names)}
  name_to_motor = {n: i for i, n in enumerate(MOTOR_ORDER)}
  missing_motor = [j for j in MOTOR_ORDER if j not in name_to_seq]
  if missing_motor:
    raise SystemExit(f"[onnx_policy] policy joint_names 缺少电机关节: {missing_motor}")

  idx_seq_for_motor = np.array([name_to_seq[j] for j in MOTOR_ORDER], dtype=np.int64)
  idx_motor_for_seq = np.array([name_to_motor[j] for j in joint_names], dtype=np.int64)

  stiffness_seq = _floats(meta, "joint_stiffness")
  damping_seq = _floats(meta, "joint_damping")
  default_seq = _floats(meta, "default_joint_pos")
  action_scale_seq = _floats(meta, "action_scale")

  # num_obs 取自图里名为 "obs" 的输入的最后一维；num_actions = 关节数
  num_obs = None
  for inp in model.graph.input:
    if inp.name == "obs":
      num_obs = int(inp.type.tensor_type.shape.dim[-1].dim_value)
  if num_obs is None:
    raise SystemExit(f"[onnx_policy] {onnx_path} 图中未找到名为 'obs' 的输入。")

  return PolicyBundle(
    onnx_path=onnx_path,
    joint_names=joint_names,
    num_obs=num_obs,
    num_actions=len(joint_names),
    anchor_body_name=meta.get("anchor_body_name", "torso_link"),
    stiffness_seq=stiffness_seq,
    damping_seq=damping_seq,
    default_seq=default_seq,
    action_scale_seq=action_scale_seq,
    stiffness_motor=stiffness_seq[idx_seq_for_motor],
    damping_motor=damping_seq[idx_seq_for_motor],
    default_motor=default_seq[idx_seq_for_motor],
    idx_seq_for_motor=idx_seq_for_motor,
    idx_motor_for_seq=idx_motor_for_seq,
  )
