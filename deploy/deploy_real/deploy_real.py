"""deploy_real.py — SMP Forward/Steering sim2real（Unitree DDS + 宇树无线遥控器 + FSM）。

与 deploy_mujoco.py 对称，区别只在状态来源/指令去向：
  - 状态：DDS 订阅 LowState（关节/IMU/内置遥控器）。
  - 指令：FSM 输出的目标角/kp/kd 填 LowCmd，加 CRC，DDS 发布。
  - 遥控器：机器人内置无线遥控器（RemoteController）。

增益/默认角/action_scale/关节序来自 policy.onnx 元数据；硬件参数(net/topics/imu)来自 yaml。

⚠️ base_lin_vel（机体系线速度）：SMP 观测需要它，但真机 IMU 测不到，需状态估计。
   本脚本默认 estimate_base_lin_vel() 返回零占位（接口预留）。要真机稳定，二选一：
     (a) 用你的里程计/状态估计填 estimate_base_lin_vel()；
     (b) 训练时用去掉 base_lin_vel 的观测配置(Wo-State-Estimation)，重新导出 onnx。

遥控器映射（与仿真一致）：
  start → 就绪位   R1+A → 行走   F1 → 阻尼   select → 退出
  左摇杆 → 行进方向+速度   右摇杆 X → 朝向

用法：python deploy_real/deploy_real.py enp4s0 configs/g1_smp.yaml \
        --policy policy.onnx --command-mode steering --max-speed 2.0
"""

import sys
import time
import argparse
from pathlib import Path

ROOT = Path(__file__).parent.parent.absolute()
sys.path.append(str(ROOT))

import numpy as np
import yaml

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG
from unitree_sdk2py.utils.crc import CRC

from common.command_helper import create_damping_cmd, init_cmd_hg, MotorMode
from common.fsm_utils import FSMCommand, get_gravity_orientation
from common.remote_controller import RemoteController, KeyMap
from common.ctrlcomp import StateAndCmd, PolicyOutput
from FSM.FSM import FSM

NUM_JOINTS = 29


class RealConfig:
    def __init__(self, path: str):
        with open(path, "r") as f:
            c = yaml.safe_load(f)
        self.control_dt = c.get("control_dt", 0.02)
        self.msg_type = c.get("msg_type", "hg")
        self.imu_type = c.get("imu_type", "pelvis")   # SMP base 观测在 pelvis(imu) 系
        self.lowcmd_topic = c.get("lowcmd_topic", "rt/lowcmd")
        self.lowstate_topic = c.get("lowstate_topic", "rt/lowstate")


class Controller:
    def __init__(self, config: RealConfig, policy_path: str, command_mode: str, max_speed: float):
        self.config = config
        self.remote_controller = RemoteController()
        self.state_cmd = StateAndCmd(NUM_JOINTS)
        self.policy_output = PolicyOutput(NUM_JOINTS)
        self.fsm = FSM(self.state_cmd, self.policy_output, policy_path,
                       command_mode=command_mode, max_speed=max_speed)

        if config.msg_type != "hg":
            raise ValueError("本部署仅支持 G1(hg) 报文类型。")
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.low_state = unitree_hg_msg_dds__LowState_()
        self.mode_pr_ = MotorMode.PR
        self.mode_machine_ = 0
        self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdHG)
        self.lowcmd_publisher_.Init()
        self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateHG)
        self.lowstate_subscriber.Init(self.LowStateHgHandler, 10)
        self.wait_for_low_state()
        init_cmd_hg(self.low_cmd, self.mode_machine_, self.mode_pr_)

    def LowStateHgHandler(self, msg: LowStateHG):
        self.low_state = msg
        self.mode_machine_ = self.low_state.mode_machine
        self.remote_controller.set(self.low_state.wireless_remote)

    def send_cmd(self, cmd: LowCmdHG):
        cmd.crc = CRC().Crc(cmd)
        self.lowcmd_publisher_.Write(cmd)

    def wait_for_low_state(self):
        while self.low_state.tick == 0:
            time.sleep(self.config.control_dt)
        print("Successfully connected to the robot.")

    def estimate_base_lin_vel(self, base_quat) -> np.ndarray:
        """机体系线速度估计（真机没有直接测量）。默认零占位——在此接入你的状态估计器。"""
        return np.zeros(3, dtype=np.float32)

    def _read_remote(self):
        rc = self.remote_controller
        if rc.is_button_pressed(KeyMap.F1):
            self.state_cmd.skill_cmd = FSMCommand.PASSIVE
        if rc.is_button_pressed(KeyMap.start):
            self.state_cmd.skill_cmd = FSMCommand.POS_RESET
        if rc.is_button_released(KeyMap.A) and rc.is_button_pressed(KeyMap.R1):
            self.state_cmd.skill_cmd = FSMCommand.STEER
        # 摇杆 → vel_cmd [fwd, left, turn]
        self.state_cmd.vel_cmd[0] = rc.ly
        self.state_cmd.vel_cmd[1] = rc.lx * -1
        self.state_cmd.vel_cmd[2] = rc.rx * -1

    def _read_state(self):
        q = np.zeros(NUM_JOINTS, dtype=np.float32)
        dq = np.zeros(NUM_JOINTS, dtype=np.float32)
        for i in range(NUM_JOINTS):
            q[i] = self.low_state.motor_state[i].q
            dq[i] = self.low_state.motor_state[i].dq
        base_quat = np.array(self.low_state.imu_state.quaternion, dtype=np.float64)  # pelvis (w,x,y,z)
        gyro = np.array(self.low_state.imu_state.gyroscope, dtype=np.float32)         # 机体系

        self.state_cmd.q = q
        self.state_cmd.dq = dq
        self.state_cmd.ang_vel = gyro.reshape(-1)[:3]
        self.state_cmd.projected_gravity = get_gravity_orientation(base_quat).astype(np.float32)
        self.state_cmd.base_lin_vel = self.estimate_base_lin_vel(base_quat)

    def run(self):
        t0 = time.time()
        self._read_remote()
        self._read_state()
        self.fsm.run()
        act, kps, kds = self.policy_output.actions, self.policy_output.kps, self.policy_output.kds
        for i in range(NUM_JOINTS):
            self.low_cmd.motor_cmd[i].q = float(act[i])
            self.low_cmd.motor_cmd[i].qd = 0
            self.low_cmd.motor_cmd[i].kp = float(kps[i])
            self.low_cmd.motor_cmd[i].kd = float(kds[i])
            self.low_cmd.motor_cmd[i].tau = 0
        self.send_cmd(self.low_cmd)
        dt = self.config.control_dt - (time.time() - t0)
        if dt > 0:
            time.sleep(dt)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("net", type=str, help="network interface (e.g. enp4s0)")
    p.add_argument("config", type=str, help="hardware yaml (configs/g1_smp.yaml)")
    p.add_argument("--policy", required=True, help="exported SMP policy.onnx")
    p.add_argument("--command-mode", default="steering", choices=["steering", "forward"])
    p.add_argument("--max-speed", type=float, default=2.0)
    args = p.parse_args()

    config = RealConfig(args.config)
    ChannelFactoryInitialize(0, args.net)
    controller = Controller(config, args.policy, args.command_mode, args.max_speed)

    print("默认阻尼保护(PassiveMode)。按 [start] 站到就绪位, 再按 [R1+A] 开始行走; "
          "[F1] 阻尼, [select] 退出。")
    while True:
        try:
            controller.run()
            if controller.remote_controller.is_button_pressed(KeyMap.select):
                break
        except KeyboardInterrupt:
            break
    create_damping_cmd(controller.low_cmd)
    controller.send_cmd(controller.low_cmd)
    print("Exit")
