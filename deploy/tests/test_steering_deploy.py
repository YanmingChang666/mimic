"""test_steering_deploy.py — SMP 部署栈离线验证（用假 onnx，无需 mjlab/mujoco/手柄）。

覆盖：观测 101 维布局与顺序、电机序↔策略序重排、摇杆→steering 指令映射、FSM 状态切换。
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.absolute()
sys.path.append(str(ROOT))

import numpy as np

from tests._make_dummy_onnx import make_dummy_onnx, SEQ_JOINT_NAMES
from common.ctrlcomp import StateAndCmd, PolicyOutput
from common.onnx_policy import MOTOR_ORDER, load_policy_bundle
from common.fsm_utils import FSMCommand, FSMStateName
from FSM.FSM import FSM

ONNX = str(ROOT / "tests" / "_dummy_smp_policy.onnx")
make_dummy_onnx(ONNX)


def test_bundle_reorder():
    """电机序增益 = 按名字从 seq 序重排；往返一致。"""
    b = load_policy_bundle(ONNX)
    assert b.num_obs == 101 and b.num_actions == 29
    # stiffness_motor[m] 应等于 seq 序里 MOTOR_ORDER[m] 那一项
    name_to_seq = {n: i for i, n in enumerate(SEQ_JOINT_NAMES)}
    for m, jname in enumerate(MOTOR_ORDER):
        assert abs(b.stiffness_motor[m] - b.stiffness_seq[name_to_seq[jname]]) < 1e-6
    # 往返
    x = np.arange(29, dtype=np.float32)
    assert np.allclose(b.motor_to_seq(b.seq_to_motor(x)), x)
    print("[deploy] bundle 双向关节序重排  OK")


def test_obs_layout():
    """SteeringMode 组的 101 维观测各段位置/内容正确。"""
    sc, po = StateAndCmd(29), PolicyOutput(29)
    fsm = FSM(sc, po, ONNX, command_mode="steering", max_speed=2.0)
    steer = fsm.steering_policy
    b = steer.bundle

    rng = np.random.default_rng(1)
    q = rng.uniform(-0.4, 0.4, 29).astype(np.float32)
    dq = rng.uniform(-1, 1, 29).astype(np.float32)
    blv = np.array([0.3, -0.1, 0.05], dtype=np.float32)
    av = np.array([0.01, 0.02, -0.03], dtype=np.float32)
    pg = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    sc.q, sc.dq, sc.base_lin_vel, sc.ang_vel, sc.projected_gravity = q, dq, blv, av, pg
    sc.vel_cmd = np.array([1.0, 0.0, 0.0], dtype=np.float32)   # 全速前进
    steer.enter()
    steer.action_buffer = rng.uniform(-1, 1, 29).astype(np.float32)
    ab = steer.action_buffer.copy()
    steer.run()

    o = steer.last_obs
    assert np.allclose(o[0:3], blv, atol=1e-6)
    assert np.allclose(o[3:6], av, atol=1e-6)
    assert np.allclose(o[6:9], pg, atol=1e-6)
    assert np.allclose(o[9:38], b.motor_to_seq(q) - b.default_seq, atol=1e-5)
    assert np.allclose(o[38:67], b.motor_to_seq(dq), atol=1e-5)
    assert np.allclose(o[67:96], ab, atol=1e-6)
    # command：steering 模式全速前进 → dir=+x, speed≈max, face=+x
    assert np.allclose(o[96:98], [1.0, 0.0], atol=1e-5)
    assert abs(o[98] - 2.0) < 1e-5
    assert np.allclose(o[99:101], [1.0, 0.0], atol=1e-5)
    # 动作重排到电机序、缩放叠加默认角
    exp_motor = b.seq_to_motor(b.default_seq + steer.action_buffer * b.action_scale_seq)
    assert np.allclose(po.actions, exp_motor, atol=1e-5)
    print("[deploy] 观测 101 维布局 + 动作重排  OK")


def test_command_modes():
    sc, po = StateAndCmd(29), PolicyOutput(29)
    fsm_f = FSM(sc, po, ONNX, command_mode="forward", max_speed=5.0)
    s = fsm_f.steering_policy
    sc.vel_cmd = np.array([0.6, 0.9, 0.5], dtype=np.float32)  # forward 模式只看第 0 维
    c = s._build_command()
    assert np.allclose(c, [1.0, 0.0, 0.6 * 5.0, 1.0, 0.0], atol=1e-5), c

    sc2, po2 = StateAndCmd(29), PolicyOutput(29)
    fsm_s = FSM(sc2, po2, ONNX, command_mode="steering", max_speed=2.0)
    s2 = fsm_s.steering_policy
    sc2.vel_cmd = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # 静止
    c2 = s2._build_command()
    assert np.allclose(c2[:3], [1.0, 0.0, 0.0], atol=1e-5)   # 无输入 → +x, 0 速
    sc2.vel_cmd = np.array([0.0, 1.0, 0.0], dtype=np.float32)  # 纯左移
    c3 = s2._build_command()
    assert np.allclose(c3[:2], [0.0, 1.0], atol=1e-5) and abs(c3[2] - 2.0) < 1e-5
    print("[deploy] forward/steering 指令映射  OK")


def test_fsm_transitions():
    sc, po = StateAndCmd(29), PolicyOutput(29)
    fsm = FSM(sc, po, ONNX)
    assert fsm.cur_policy.name == FSMStateName.PASSIVE
    sc.skill_cmd = FSMCommand.POS_RESET; fsm.run(); fsm.run()
    assert fsm.cur_policy.name == FSMStateName.FIXEDPOSE
    sc.skill_cmd = FSMCommand.STEER; fsm.run(); fsm.run()
    assert fsm.cur_policy.name == FSMStateName.STEERING
    sc.skill_cmd = FSMCommand.PASSIVE; fsm.run(); fsm.run()
    assert fsm.cur_policy.name == FSMStateName.PASSIVE
    # passive 只响应 POS_RESET
    sc.skill_cmd = FSMCommand.STEER; fsm.run()
    assert fsm.cur_policy.name == FSMStateName.PASSIVE
    print("[deploy] passive<->fixedpose<->steering 切换  OK")


if __name__ == "__main__":
    test_bundle_reorder()
    test_obs_layout()
    test_command_modes()
    test_fsm_transitions()
    print("ALL SMP DEPLOY TESTS PASSED")
