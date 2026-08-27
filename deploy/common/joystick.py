"""joystick.py — PC 端 Xbox/PS 手柄读取（pygame 优先，evdev 回退），sim2sim 用。

与真机的 remote_controller.py 区分：这套读的是插在电脑上的 USB/蓝牙手柄；真机读的是机器人
内置无线遥控器。两者对外都提供 update()/is_button_pressed()/is_button_released()/
get_axis_value()/get_hat_direction()，deploy 层按同一套逻辑映射按键。

从 RoboMimicDeploy_G1/common/joystick.py 移植（去掉对 common.path_config 的耦合）。
"""

import pygame
from enum import IntEnum, unique


@unique
class JoystickButton(IntEnum):
    # Standard PlayStation/Xbox Layout
    A = 0
    B = 1
    X = 2
    Y = 3
    L1 = 4
    R1 = 5
    SELECT = 6
    START = 7
    L3 = 8
    R3 = 9
    HOME = 10
    UP = 11
    DOWN = 12
    LEFT = 13
    RIGHT = 14


class JoyStick:
    """pygame 后端。"""
    def __init__(self):
        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            raise RuntimeError("No joystick connected!")
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()

        self.button_count = self.joystick.get_numbuttons()
        self.button_states = [False] * self.button_count
        self.button_released = [False] * self.button_count

        self.axis_count = self.joystick.get_numaxes()
        self.axis_states = [0.0] * self.axis_count

        self.hat_count = self.joystick.get_numhats()
        self.hat_states = [(0, 0)] * self.hat_count

    def update(self):
        pygame.event.pump()
        self.button_released = [False] * self.button_count
        for i in range(self.button_count):
            current_state = self.joystick.get_button(i) == 1
            if self.button_states[i] and not current_state:
                self.button_released[i] = True
            self.button_states[i] = current_state
        for i in range(self.axis_count):
            self.axis_states[i] = self.joystick.get_axis(i)
        for i in range(self.hat_count):
            self.hat_states[i] = self.joystick.get_hat(i)

    def is_button_pressed(self, button_id):
        if 0 <= button_id < self.button_count:
            return self.button_states[button_id]
        return False

    def is_button_released(self, button_id):
        if 0 <= button_id < self.button_count:
            return self.button_released[button_id]
        return False

    def get_axis_value(self, axis_id):
        if 0 <= axis_id < self.axis_count:
            return self.axis_states[axis_id]
        return 0.0

    def get_hat_direction(self, hat_id=0):
        if 0 <= hat_id < self.hat_count:
            return self.hat_states[hat_id]
        return (0, 0)


class JoyStickEvdev:
    """evdev 后端，对外接口与 JoyStick 完全一致（pygame 识别不了手柄时用）。"""
    def __init__(self):
        import threading
        from common.gamepad import GamepadState, GamepadHandler, make_handle, list_gamepads

        found = list_gamepads()
        if not found:
            raise RuntimeError("evdev 未发现任何手柄设备（/dev/input 下无匹配项）。")
        print("检测到手柄设备：")
        for p, n in found:
            print(f"  {p}  ->  {n}")

        self.rc = GamepadState()
        self.handler = GamepadHandler()
        self.handler.reconnect()
        threading.Thread(target=self.handler.listen,
                         args=(make_handle(self.rc),), daemon=True).start()

        n = max(int(b) for b in JoystickButton) + 1
        self.button_states = [False] * n
        self.button_released = [False] * n

    def _read(self, b):
        rc = self.rc
        return {
            JoystickButton.A: rc.A,        JoystickButton.B: rc.B,
            JoystickButton.X: rc.X,        JoystickButton.Y: rc.Y,
            JoystickButton.L1: rc.LB,      JoystickButton.R1: rc.RB,
            JoystickButton.SELECT: rc.BACK, JoystickButton.START: rc.START,
            JoystickButton.L3: rc.L3,      JoystickButton.R3: rc.R3,
            JoystickButton.HOME: rc.HOME,
            JoystickButton.UP: rc.DPAD_Y < 0,    JoystickButton.DOWN: rc.DPAD_Y > 0,
            JoystickButton.LEFT: rc.DPAD_X < 0,  JoystickButton.RIGHT: rc.DPAD_X > 0,
        }.get(b, False)

    def update(self):
        for b in JoystickButton:
            i = int(b)
            now = self._read(b)
            self.button_released[i] = self.button_states[i] and not now
            self.button_states[i] = now

    def is_button_pressed(self, button_id):
        i = int(button_id)
        return self.button_states[i] if 0 <= i < len(self.button_states) else False

    def is_button_released(self, button_id):
        i = int(button_id)
        return self.button_released[i] if 0 <= i < len(self.button_released) else False

    def get_axis_value(self, axis_id):
        # 轴编号沿用 pygame 习惯：0=左X, 1=左Y, 2=右Y, 3=右X
        rc = self.rc
        return {0: rc.LEFT_X, 1: rc.LEFT_Y, 2: rc.RIGHT_Y, 3: rc.RIGHT_X}.get(axis_id, 0.0)

    def get_hat_direction(self, hat_id=0):
        # pygame hat: y 向上为 +1；evdev DPAD_Y 向上为 -1 → 取反对齐
        return (self.rc.DPAD_X, -self.rc.DPAD_Y)


def make_joystick(prefer="auto"):
    """创建手柄读取器，自动选可用后端。

    prefer: "auto" 先 evdev 后 pygame；"evdev" 强制 evdev；"pygame" 强制 pygame。
    """
    if prefer in ("auto", "evdev"):
        try:
            js = JoyStickEvdev()
            print("[joystick] 使用 evdev 后端。")
            return js
        except Exception as e:                       # noqa: BLE001
            if prefer == "evdev":
                raise
            print(f"[joystick] evdev 不可用（{e}），回退到 pygame。")
    js = JoyStick()
    print(f"[joystick] 使用 pygame 后端：{js.joystick.get_name()}")
    return js
