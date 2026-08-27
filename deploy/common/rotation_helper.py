"""rotation_helper.py — IMU 姿态变换工具（真机用）。

从 deploy_real/common/rotation_helper.py 移植到项目级 common/。BeyondMimic 的 anchor 用 torso
系四元数：G1 的 IMU 装在 pelvis，需经三个 waist 关节把 pelvis 姿态前推到 torso。
"""

import numpy as np
from scipy.spatial.transform import Rotation as R


def get_gravity_orientation(quaternion):
    qw, qx, qy, qz = quaternion[0], quaternion[1], quaternion[2], quaternion[3]
    g = np.zeros(3)
    g[0] = 2 * (-qz * qx + qw * qy)
    g[1] = -2 * (qz * qy + qw * qx)
    g[2] = 1 - 2 * (qw * qw + qz * qz)
    return g


def transform_imu_data(waist_yaw, waist_yaw_omega, imu_quat, imu_omega):
    RzWaist = R.from_euler("z", waist_yaw).as_matrix()
    R_torso = R.from_quat([imu_quat[1], imu_quat[2], imu_quat[3], imu_quat[0]]).as_matrix()
    R_pelvis = np.dot(R_torso, RzWaist.T)
    w = np.dot(RzWaist, imu_omega[0]) - np.array([0, 0, waist_yaw_omega])
    return R.from_matrix(R_pelvis).as_quat()[[3, 0, 1, 2]], w


def transform_pelvis_to_torso_complete(waist_yaw, waist_roll, waist_pitch, pelvis_quat):
    """完整 pelvis→torso 姿态变换，含三个 waist 关节。输入/输出四元数均 (w,x,y,z)。"""
    R_waist_yaw = R.from_euler("z", waist_yaw)
    R_waist_roll = R.from_euler("x", waist_roll)
    R_waist_pitch = R.from_euler("y", waist_pitch)
    R_pelvis = R.from_quat([pelvis_quat[1], pelvis_quat[2], pelvis_quat[3], pelvis_quat[0]])
    R_torso = R_pelvis * R_waist_yaw * R_waist_roll * R_waist_pitch
    q = R_torso.as_quat()  # [x,y,z,w]
    return np.array([q[3], q[0], q[1], q[2]])
