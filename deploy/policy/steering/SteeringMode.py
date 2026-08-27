"""SteeringMode.py — SMP Forward/Steering 策略（★核心状态）。

跑导出的 SMP 策略 onnx（actor + 经验观测归一化已折进图内），把遥控器摇杆映射为 5 维
steering 指令，组装 101 维观测推理，输出目标关节角。

观测布局(101，与 g1_smp_env_cfg 的 actor 观测组顺序一致；无手动 scale，归一化在 onnx 内)：
  [base_lin_vel(3)] [base_ang_vel(3)] [projected_gravity(3)]
  [joint_pos-default(29)] [joint_vel(29)] [last_action(29)]
  [command(5) = tar_dir_x, tar_dir_y, tar_speed, face_dir_x, face_dir_y]   (机器人朝向系)

关节序：state_cmd.q/dq 与 policy_output.actions 均电机序；网络内部按 onnx joint_names(seq 序)，
用 bundle 互转。command 在朝向系表达(与训练一致，偏航不变)，直接由摇杆生成。

command_mode:
  "forward"  —— 固定 +x 方向与朝向，仅左摇杆 Y 控速(0~max_speed)。对应 Smp-Forward-G1。
  "steering" —— 左摇杆定行进方向+速度，右摇杆 X 定朝向。对应 Smp-Steering-G1。
"""

import numpy as np
import onnxruntime

from FSM.FSMState import FSMState
from common.ctrlcomp import StateAndCmd, PolicyOutput
from common.fsm_utils import FSMStateName, FSMCommand
from common.onnx_policy import load_policy_bundle

FACE_MAX_ANGLE = np.pi   # 右摇杆满偏对应的朝向角(rad)


class SteeringMode(FSMState):
    def __init__(self, state_cmd: StateAndCmd, policy_output: PolicyOutput,
                 policy_path: str, command_mode: str = "steering", max_speed: float = 2.0):
        super().__init__()
        self.state_cmd = state_cmd
        self.policy_output = policy_output
        self.name = FSMStateName.STEERING
        self.name_str = "steering_mode"
        self.command_mode = command_mode
        self.max_speed = float(max_speed)

        self.bundle = load_policy_bundle(policy_path)
        self.session = onnxruntime.InferenceSession(policy_path)
        self.num_obs = self.bundle.num_obs          # 期望 101 (96 + command 5)
        self.num_actions = self.bundle.num_actions  # 29
        self.command_dim = self.num_obs - 96
        if self.command_dim not in (5,):
            print(f"[SteeringMode][warn] command_dim={self.command_dim} (期望 5)；"
                  f"num_obs={self.num_obs}，请确认导出的是 Forward/Steering 策略。")

        self.action_buffer = np.zeros(self.num_actions, dtype=np.float32)  # seq 序
        self.last_obs = np.zeros(self.num_obs, dtype=np.float32)
        self.command = np.zeros(self.command_dim, dtype=np.float32)

        # onnx 预热
        warm = np.zeros((1, self.num_obs), dtype=np.float32)
        try:
            self._in = self.session.get_inputs()[0].name
            self._out = self.session.get_outputs()[0].name
            for _ in range(5):
                self.session.run([self._out], {self._in: warm})
        except Exception as e:   # noqa: BLE001
            print(f"[SteeringMode][warn] onnx 预热失败: {e}")
        print(f"SteeringMode: num_obs={self.num_obs} command_mode={self.command_mode} "
              f"max_speed={self.max_speed}")

    def enter(self):
        self.action_buffer = np.zeros(self.num_actions, dtype=np.float32)

    def _build_command(self):
        """摇杆 → 5 维 steering 指令(朝向系): [dirx, diry, speed, facex, facey]。"""
        cmd = np.zeros(self.command_dim, dtype=np.float32)
        if self.command_dim != 5:
            return cmd
        fwd, left, turn = (float(self.state_cmd.vel_cmd[0]),
                           float(self.state_cmd.vel_cmd[1]),
                           float(self.state_cmd.vel_cmd[2]))
        if self.command_mode == "forward":
            speed = np.clip(fwd, 0.0, 1.0) * self.max_speed
            cmd[:] = [1.0, 0.0, speed, 1.0, 0.0]
            return cmd
        # steering：左摇杆定方向+速度，右摇杆定朝向
        v = np.array([fwd, left], dtype=np.float32)   # 机体系 [+x 前, +y 左]
        mag = float(np.linalg.norm(v))
        if mag > 1e-3:
            tar_dir = v / mag
            speed = min(mag, 1.0) * self.max_speed
        else:
            tar_dir = np.array([1.0, 0.0], dtype=np.float32)
            speed = 0.0
        theta = -turn * FACE_MAX_ANGLE
        face_dir = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)
        cmd[:] = [tar_dir[0], tar_dir[1], speed, face_dir[0], face_dir[1]]
        return cmd

    def _build_obs(self):
        obs = np.zeros(self.num_obs, dtype=np.float32)
        q_seq = self.bundle.motor_to_seq(self.state_cmd.q)
        dq_seq = self.bundle.motor_to_seq(self.state_cmd.dq)
        self.command = self._build_command()
        o = 0
        obs[o:o + 3] = np.asarray(self.state_cmd.base_lin_vel).reshape(-1)[:3]; o += 3
        obs[o:o + 3] = np.asarray(self.state_cmd.ang_vel).reshape(-1)[:3];      o += 3
        obs[o:o + 3] = np.asarray(self.state_cmd.projected_gravity).reshape(-1)[:3]; o += 3
        obs[o:o + 29] = q_seq - self.bundle.default_seq;  o += 29
        obs[o:o + 29] = dq_seq;                           o += 29
        obs[o:o + 29] = self.action_buffer;               o += 29
        obs[o:o + self.command_dim] = self.command
        return obs

    def run(self):
        obs = self._build_obs()
        self.last_obs = obs.copy()
        action = self.session.run([self._out], {self._in: obs[None].astype(np.float32)})[0]
        action = np.asarray(action).reshape(-1).astype(np.float32)
        self.action_buffer = action.copy()

        target_seq = self.bundle.default_seq + action * self.bundle.action_scale_seq
        self.policy_output.actions = self.bundle.seq_to_motor(target_seq).astype(np.float32)
        self.policy_output.kps = self.bundle.stiffness_motor.astype(np.float32)
        self.policy_output.kds = self.bundle.damping_motor.astype(np.float32)

    def exit(self):
        self.action_buffer = np.zeros(self.num_actions, dtype=np.float32)

    def checkChange(self):
        if self.state_cmd.skill_cmd == FSMCommand.POS_RESET:
            self.state_cmd.skill_cmd = FSMCommand.INVALID
            return FSMStateName.FIXEDPOSE
        elif self.state_cmd.skill_cmd == FSMCommand.PASSIVE:
            self.state_cmd.skill_cmd = FSMCommand.INVALID
            return FSMStateName.PASSIVE
        else:
            self.state_cmd.skill_cmd = FSMCommand.INVALID
            return FSMStateName.STEERING
