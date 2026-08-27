"""FSMState.py — 有限状态机所有「状态/技能」的抽象基类。

BeyondMimic 部署里每个控制模式（阻尼保护 PassiveMode / 就绪位 FixedPose / 动作跟踪 MimicMode）
都是 FSMState 的子类。FSM 在任意时刻只持有一个当前状态，按固定节拍调用它的 run()；当
checkChange() 返回一个不同的状态名时，FSM 执行 旧.exit() → 新.enter() 完成切换。
"""

from common.fsm_utils import FSMStateName


class FSMState:
    def __init__(self):
        self.name = FSMStateName.INVALID   # 本状态枚举名（FSM 用它判断是否需要切换）
        self.name_str = "invalid"          # 人类可读名（打印日志用）
        self.control_dt = 0.02             # 期望控制周期（秒），50Hz=0.02s

    def enter(self):
        """进入本状态时调用一次：初始化（清缓冲、重排增益、复位相位等）。"""
        raise NotImplementedError("enter() must be implemented")

    def run(self):
        """每个控制周期一次：读 state_cmd → 推理/计算 → 写 policy_output。"""
        raise NotImplementedError("run() must be implemented")

    def exit(self):
        """离开本状态时调用一次：清理。"""
        raise NotImplementedError("exit() must be implemented")

    def checkChange(self):
        """每个控制周期一次：据遥控器指令决定下一个状态名（返回自身 name = 不切换）。"""
        raise NotImplementedError("checkChange() must be implemented")
