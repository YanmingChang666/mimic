"""remote_controller.py — 解析 G1「内置无线遥控器」原始字节流（真机 sim2real 用）。

机器人把遥控器状态打包进 LowState.wireless_remote 的 40 字节：前 2 字节是 16 个按键的位掩码，
后面是 4 路摇杆轴 float。本类解出按键状态 + 摇杆值，并维护「按下/松开」边沿检测（供暂停等
一次性触发用）。这与 PC 端 joystick.py(pygame/evdev) 是两套不同来源，按键编号也不同（见 KeyMap）。

相对旧 deploy_real/common/remote_controller.py：补上 is_button_pressed/is_button_released
边沿检测（照 RoboMimicDeploy_G1 版本），FSM 层才能做「组合键切状态」+「按一下暂停」。
"""

import struct


class KeyMap:
    # 各按键在位掩码中的 bit 位置（与 Unitree 遥控器协议一致）
    R1 = 0
    L1 = 1
    start = 2
    select = 3
    R2 = 4
    L2 = 5
    F1 = 6
    F2 = 7
    A = 8
    B = 9
    X = 10
    Y = 11
    up = 12
    right = 13
    down = 14
    left = 15


class RemoteController:
    def __init__(self):
        self.lx = 0.0
        self.ly = 0.0
        self.rx = 0.0
        self.ry = 0.0
        self.button = [0] * 16

        self.button_states = [False] * 16
        self.button_released = [False] * 16

    def set(self, data):
        """解析遥控器原始字节流（每帧由 DDS 回调传入）。"""
        keys = struct.unpack("H", data[2:4])[0]       # 16 个按键位掩码
        for i in range(16):
            self.button[i] = (keys & (1 << i)) >> i
        # 4 路摇杆轴（float）：左X、右X、右Y、左Y
        self.lx = struct.unpack("f", data[4:8])[0]
        self.rx = struct.unpack("f", data[8:12])[0]
        self.ry = struct.unpack("f", data[12:16])[0]
        self.ly = struct.unpack("f", data[20:24])[0]

        # 边沿检测：本帧由按下→松开 记为 released
        self.button_released = [False] * 16
        for i in range(16):
            current_state = self.button[i] == 1
            if self.button_states[i] and not current_state:
                self.button_released[i] = True
            self.button_states[i] = current_state

    def is_button_pressed(self, button_id):
        if 0 <= button_id < 16:
            return self.button_states[button_id]
        return False

    def is_button_released(self, button_id):
        if 0 <= button_id < 16:
            return self.button_released[button_id]
        return False
