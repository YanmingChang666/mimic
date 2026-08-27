"""FSM.py — 有限状态机调度核心（SMP Forward/Steering 3 状态版）。

状态：PassiveMode(阻尼) / FixedPose(就绪位) / SteeringMode(SMP 行走策略)。
开机默认 PassiveMode。切换按 旧.exit() → 选新 → 新.enter()。数据流经共享黑板
StateAndCmd / PolicyOutput（见 common/ctrlcomp.py）。
"""

from enum import Enum, unique

from common.ctrlcomp import StateAndCmd, PolicyOutput
from common.fsm_utils import FSMStateName
from FSM.FSMState import FSMState
from policy.passive.PassiveMode import PassiveMode
from policy.fixedpose.FixedPose import FixedPose
from policy.steering.SteeringMode import SteeringMode


@unique
class FSMMode(Enum):
    CHANGE = 1
    NORMAL = 2


class FSM:
    def __init__(self, state_cmd: StateAndCmd, policy_output: PolicyOutput,
                 policy_path: str, command_mode: str = "steering", max_speed: float = 2.0):
        self.state_cmd = state_cmd
        self.policy_output = policy_output
        self.sim_counter = 0
        self.FSMmode = FSMMode.NORMAL

        # SteeringMode 内加载 onnx（含增益/默认角/关节序）；另两个复用其 bundle
        self.steering_policy = SteeringMode(state_cmd, policy_output, policy_path,
                                            command_mode=command_mode, max_speed=max_speed)
        bundle = self.steering_policy.bundle
        self.passive_mode = PassiveMode(state_cmd, policy_output, bundle)
        self.fixed_pose = FixedPose(state_cmd, policy_output, bundle)
        print("initialized all policies!!!")

        self.cur_policy: FSMState = self.passive_mode
        print("current policy is", self.cur_policy.name_str)

    def run(self):
        if self.FSMmode == FSMMode.NORMAL:
            self.cur_policy.run()
            next_name = self.cur_policy.checkChange()
            if next_name != self.cur_policy.name:
                self.FSMmode = FSMMode.CHANGE
                self.cur_policy.exit()
                self._select(next_name)
                print("Switched to", self.cur_policy.name_str)
        elif self.FSMmode == FSMMode.CHANGE:
            self.cur_policy.enter()
            self.sim_counter = 0
            self.FSMmode = FSMMode.NORMAL
            self.cur_policy.run()

    def _select(self, name: FSMStateName):
        if name == FSMStateName.PASSIVE:
            self.cur_policy = self.passive_mode
        elif name == FSMStateName.FIXEDPOSE:
            self.cur_policy = self.fixed_pose
        elif name == FSMStateName.STEERING:
            self.cur_policy = self.steering_policy
