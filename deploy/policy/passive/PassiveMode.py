"""PassiveMode.py — 阻尼保护模式（开机默认，最安全）。

电机只提供阻尼、不主动发力：kp=0, kd=常数, 目标角=0。任何时候按 L1(仿真)/F1(真机) 都能进来。
从这里按 Start 进入就绪位(FixedPose)。
"""

import numpy as np

from FSM.FSMState import FSMState
from common.ctrlcomp import StateAndCmd, PolicyOutput
from common.fsm_utils import FSMStateName, FSMCommand


class PassiveMode(FSMState):
    def __init__(self, state_cmd: StateAndCmd, policy_output: PolicyOutput, bundle, kd: float = 8.0):
        super().__init__()
        self.state_cmd = state_cmd
        self.policy_output = policy_output
        self.name = FSMStateName.PASSIVE
        self.name_str = "passive_mode"
        self.n = state_cmd.num_joints
        self.kd = float(kd)   # 与 command_helper.create_damping_cmd 的阻尼一致

    def enter(self):
        self.policy_output.kps = np.zeros(self.n, dtype=np.float32)
        self.policy_output.kds = np.full(self.n, self.kd, dtype=np.float32)

    def run(self):
        self.policy_output.actions = np.zeros(self.n, dtype=np.float32)
        self.policy_output.kps = np.zeros(self.n, dtype=np.float32)
        self.policy_output.kds = np.full(self.n, self.kd, dtype=np.float32)

    def exit(self):
        self.policy_output.kps = np.zeros(self.n, dtype=np.float32)
        self.policy_output.kds = np.full(self.n, self.kd, dtype=np.float32)

    def checkChange(self):
        if self.state_cmd.skill_cmd == FSMCommand.POS_RESET:
            self.state_cmd.skill_cmd = FSMCommand.INVALID
            return FSMStateName.FIXEDPOSE
        else:
            self.state_cmd.skill_cmd = FSMCommand.INVALID
            return FSMStateName.PASSIVE
