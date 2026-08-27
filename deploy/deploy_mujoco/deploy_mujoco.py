"""deploy_mujoco.py — SMP Forward/Steering sim2sim（MuJoCo + PC 手柄 + FSM）。

数据流（每物理步 2ms）：
  PC 手柄 → StateAndCmd(技能切换 + 摇杆 vel_cmd)
  MuJoCo 传感器 → StateAndCmd(base 线/角速度、重力投影、关节角/速度)
  FSM.run() → SteeringMode 推理 → PolicyOutput(目标角/kp/kd)
  pd_control → d.ctrl → mj_step

控制：物理 500Hz，策略每 control_decimation=10 步(50Hz)推理一次。增益/默认角/action_scale/
关节序全部来自导出的 policy.onnx 元数据(common/onnx_policy.py)。

手柄映射（与真机 deploy_real.py 一致）：
  START → 就绪位   R1+A → 行走(SteeringMode)   L1 → 阻尼   SELECT → 退出
  左摇杆 → 行进方向+速度   右摇杆 X → 朝向(steering 模式)

用法：
  python deploy_mujoco/deploy_mujoco.py --policy policy.onnx --command-mode steering --max-speed 2.0
依赖：mujoco, pygame 或 evdev。
"""

import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.absolute()))

import argparse

import mujoco
import mujoco.viewer
import numpy as np

from common.ctrlcomp import StateAndCmd, PolicyOutput
from common.fsm_utils import FSMCommand, get_gravity_orientation, quat_rotate_inverse
from common.joystick import JoystickButton, make_joystick
from FSM.FSM import FSM


def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd


def main():
    p = argparse.ArgumentParser(description="SMP G1 sim2sim (MuJoCo + gamepad + FSM).")
    p.add_argument("--policy", required=True, help="exported SMP policy.onnx (scripts/export_onnx.py)")
    p.add_argument("--xml", default=str(Path(__file__).parent.parent / "unitree_description/mjcf/g1_liao.xml"))
    p.add_argument("--command-mode", default="steering", choices=["steering", "forward"])
    p.add_argument("--max-speed", type=float, default=2.0, help="Forward=5.0, Steering=2.0")
    p.add_argument("--joystick", default="auto", choices=["auto", "evdev", "pygame"])
    args = p.parse_args()

    simulation_dt = 0.002
    control_decimation = 10
    num_joints = 29

    m = mujoco.MjModel.from_xml_path(args.xml)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    state_cmd = StateAndCmd(num_joints)
    policy_output = PolicyOutput(num_joints)
    fsm = FSM(state_cmd, policy_output, args.policy,
              command_mode=args.command_mode, max_speed=args.max_speed)
    bundle = fsm.steering_policy.bundle

    # 初始站姿（默认关节角 + 站立高度）
    d.qpos[2] = 0.78
    d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    d.qpos[7:7 + num_joints] = bundle.default_motor
    mujoco.mj_forward(m, d)

    joystick = make_joystick(prefer=args.joystick)
    state_cmd.skill_cmd = FSMCommand.POS_RESET   # sim 友好：开机进就绪位站立

    policy_action = bundle.default_motor.astype(np.float32).copy()
    kps = np.zeros(num_joints, dtype=np.float32)
    kds = np.zeros(num_joints, dtype=np.float32)

    with mujoco.viewer.launch_passive(m, d) as viewer:
        running = True
        while viewer.is_running() and running:
            step_start = time.time()

            joystick.update()
            if joystick.is_button_pressed(JoystickButton.SELECT):
                running = False
            if joystick.is_button_released(JoystickButton.START):
                state_cmd.skill_cmd = FSMCommand.POS_RESET
            if joystick.is_button_released(JoystickButton.L1):
                state_cmd.skill_cmd = FSMCommand.PASSIVE
            if joystick.is_button_released(JoystickButton.A) and joystick.is_button_pressed(JoystickButton.R1):
                state_cmd.skill_cmd = FSMCommand.STEER
            # 摇杆 → vel_cmd [fwd, left, turn]（evdev 轴序 0=LX,1=LY,2=RY,3=RX）
            state_cmd.vel_cmd[0] = -joystick.get_axis_value(1)   # 左摇杆 Y：上=前
            state_cmd.vel_cmd[1] = -joystick.get_axis_value(0)   # 左摇杆 X：左=+左
            state_cmd.vel_cmd[2] = -joystick.get_axis_value(3)   # 右摇杆 X：转/朝向

            tau = pd_control(policy_action, d.qpos[7:7 + num_joints], kps,
                             np.zeros(num_joints), d.qvel[6:6 + num_joints], kds)
            d.ctrl[:] = tau
            mujoco.mj_step(m, d)
            fsm.sim_counter += 1

            if fsm.sim_counter % control_decimation == 0:
                base_quat = d.qpos[3:7].copy()                  # pelvis 四元数 (w,x,y,z)
                world_lin_vel = d.qvel[0:3].copy()              # 世界系线速度
                state_cmd.base_lin_vel = quat_rotate_inverse(base_quat, world_lin_vel).astype(np.float32)
                state_cmd.ang_vel = d.qvel[3:6].copy()          # free joint 角速度已是机体系
                state_cmd.projected_gravity = get_gravity_orientation(base_quat).astype(np.float32)
                state_cmd.q = d.qpos[7:7 + num_joints].copy()
                state_cmd.dq = d.qvel[6:6 + num_joints].copy()
                fsm.run()
                policy_action = policy_output.actions.copy()
                kps = policy_output.kps.copy()
                kds = policy_output.kds.copy()

            viewer.sync()
            dt = m.opt.timestep - (time.time() - step_start)
            if dt > 0:
                time.sleep(dt)

    print("\nExit")


if __name__ == "__main__":
    main()
