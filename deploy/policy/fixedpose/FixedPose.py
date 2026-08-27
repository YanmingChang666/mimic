"""FixedPose.py — 就绪/默认位（位控平滑保持）。

从当前关节角，用训练时的 PD 增益，在 2s 内线性插值到「就绪位」，然后保持。
就绪位默认取 ONNX 的 default_joint_pos（电机序）；若构造时传入 start_pose（如 motion 第 0 帧，
电机序），则保持到该位姿，消除进入动作跟踪瞬间的位姿跳变。

所有量均为 **电机序**，与 StateAndCmd.q / PolicyOutput.actions 一致（G1 = 29）。
"""

import numpy as np

from FSM.FSMState import FSMState
from common.ctrlcomp import StateAndCmd, PolicyOutput
from common.fsm_utils import FSMStateName, FSMCommand


class FixedPose(FSMState):
    def __init__(self, state_cmd: StateAndCmd, policy_output: PolicyOutput, bundle,
                 start_pose=None, control_dt: float = 0.02):
        super().__init__()
        self.state_cmd = state_cmd
        self.policy_output = policy_output
        self.name = FSMStateName.FIXEDPOSE
        self.name_str = "fixed_pose"
        self.control_dt = control_dt
        self.n = state_cmd.num_joints

        self.kps = np.asarray(bundle.stiffness_motor, dtype=np.float32)
        self.kds = np.asarray(bundle.damping_motor, dtype=np.float32)
        # 目标就绪位：默认关节角，或外部指定（motion 第 0 帧）
        self.target_pose = (np.asarray(start_pose, dtype=np.float32)
                            if start_pose is not None
                            else np.asarray(bundle.default_motor, dtype=np.float32))

    def enter(self):
        print("Moving to ready pose.")
        self.num_step = max(int(2.0 / self.control_dt), 1)
        self.cur_step = 0
        self.init_dof_pos = np.asarray(self.state_cmd.q, dtype=np.float32).copy()

    def run(self):
        self.cur_step += 1
        alpha = min(self.cur_step / self.num_step, 1.0)
        self.policy_output.actions = (self.init_dof_pos * (1 - alpha)
                                      + self.target_pose * alpha).astype(np.float32)
        self.policy_output.kps = self.kps.copy()
        self.policy_output.kds = self.kds.copy()

    def exit(self):
        self.policy_output.actions = self.target_pose.copy()
        self.policy_output.kps = self.kps.copy()
        self.policy_output.kds = self.kds.copy()

    def checkChange(self):
        if self.state_cmd.skill_cmd == FSMCommand.STEER:
            self.state_cmd.skill_cmd = FSMCommand.INVALID
            return FSMStateName.STEERING
        elif self.state_cmd.skill_cmd == FSMCommand.PASSIVE:
            self.state_cmd.skill_cmd = FSMCommand.INVALID
            return FSMStateName.PASSIVE
        else:
            self.state_cmd.skill_cmd = FSMCommand.INVALID
            return FSMStateName.FIXEDPOSE
