"""gamepad.py — 基于 evdev 的 Linux 手柄读取（USB 接收器 / 蓝牙手柄均适用）。

【为什么需要它？为什么不只用 pygame？】
  pygame(SDL) 依赖一份「游戏手柄映射数据库」，对很多通过 USB 接收器 / 蓝牙 连接的手柄
  （北通 BEITONG、部分 Xbox 兼容手柄等）识别不稳定或映射错乱，典型现象是「手柄插上了、
  系统能识别，但程序里按键无反应」。evdev 直接读 /dev/input/eventX 原始事件，更可靠。

【依赖与权限】
  1) pip install evdev
  2) 读 /dev/input/eventX 需权限：sudo usermod -aG input $USER  （之后重新登录）

从 RoboMimicDeploy_G1/common/gamepad.py 移植（BeyondMimic 部署自包含，避免跨仓库依赖）。
"""

import os
import json
import time
import threading

try:
    from evdev import InputDevice, list_devices, ecodes
    _HAS_EVDEV = True
except Exception:                       # noqa: BLE001  evdev 未安装
    _HAS_EVDEV = False

CALIBRATION_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "gamepad_calibration.json")


def load_calibration(path=CALIBRATION_FILE):
    """读取按键标定文件，返回 {设备名: {键码int: 逻辑名}}。无文件返回 {}。"""
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception:                   # noqa: BLE001  无标定文件属正常
        return {}
    profiles = {}
    if isinstance(data.get("profiles"), dict):
        for dev, m in data["profiles"].items():
            profiles[dev] = {int(k): v for k, v in m.items()}
    elif isinstance(data.get("buttons"), dict):
        profiles[data.get("device", "")] = {int(k): v for k, v in data["buttons"].items()}
    return profiles


GAMEPAD_KEYWORDS = ("x-box", "xbox", "sony", "beitong", "dongle",
                    "gamepad", "controller", "joystick", "pad")


def _build_code_to_btn():
    if not _HAS_EVDEV:
        return {}
    return {
        ecodes.BTN_SOUTH: "A",   ecodes.BTN_EAST: "B",
        ecodes.BTN_NORTH: "Y",   ecodes.BTN_WEST: "X",   # NORTH=Y, WEST=X
        ecodes.BTN_TL: "LB",     ecodes.BTN_TR: "RB",
        ecodes.BTN_SELECT: "BACK", ecodes.BTN_START: "START", ecodes.BTN_MODE: "HOME",
        ecodes.BTN_THUMBL: "L3", ecodes.BTN_THUMBR: "R3",
        ecodes.KEY_MENU: "START", ecodes.KEY_BACK: "BACK",
    }

_CODE_TO_BTN = _build_code_to_btn()


class GamepadState:
    """手柄当前状态快照（由后台线程持续更新，主线程随时读取）。"""
    def __init__(self):
        self.A = self.B = self.X = self.Y = False
        self.BACK = self.START = self.HOME = False
        self.LB = self.RB = False
        self.L3 = self.R3 = False
        self.LT = 0.0
        self.RT = 0.0
        self.LEFT_X = 0.0
        self.LEFT_Y = 0.0
        self.RIGHT_X = 0.0
        self.RIGHT_Y = 0.0
        self.DPAD_X = 0
        self.DPAD_Y = 0

    def __repr__(self):
        return (f"A={int(self.A)} B={int(self.B)} X={int(self.X)} Y={int(self.Y)} "
                f"LB={int(self.LB)} RB={int(self.RB)} BACK={int(self.BACK)} START={int(self.START)} "
                f"L3={int(self.L3)} R3={int(self.R3)} "
                f"LEFT=({self.LEFT_X:+.2f},{self.LEFT_Y:+.2f}) "
                f"RIGHT=({self.RIGHT_X:+.2f},{self.RIGHT_Y:+.2f}) "
                f"DPAD=({self.DPAD_X},{self.DPAD_Y}) LT={self.LT:.2f} RT={self.RT:.2f}")


def list_gamepads():
    """返回 [(设备路径, 设备名)]，列出所有「看起来像手柄」的输入设备。"""
    if not _HAS_EVDEV:
        return []
    out = []
    for path in list_devices():
        try:
            dev = InputDevice(path)
        except Exception:
            continue
        name = dev.name or ""
        caps = dev.capabilities()
        has_btn = ecodes.EV_KEY in caps and any(
            c in caps[ecodes.EV_KEY] for c in (ecodes.BTN_SOUTH, ecodes.BTN_A, ecodes.BTN_GAMEPAD)
        )
        if any(k in name.lower() for k in GAMEPAD_KEYWORDS) or has_btn:
            out.append((path, name))
    return out


def make_handle(rc: GamepadState):
    """生成回调，把 GamepadHandler 内部 state 同步到 GamepadState 对象 rc。"""
    def handle(state):
        btns = state["buttons"]
        rc.A = "A" in btns;   rc.B = "B" in btns
        rc.X = "X" in btns;   rc.Y = "Y" in btns
        rc.LB = "LB" in btns; rc.RB = "RB" in btns
        rc.BACK = "BACK" in btns; rc.START = "START" in btns; rc.HOME = "HOME" in btns
        rc.L3 = "L3" in btns; rc.R3 = "R3" in btns
        rc.LT = state["triggers"]["LT"]; rc.RT = state["triggers"]["RT"]
        rc.LEFT_X, rc.LEFT_Y = state["left_stick"]
        rc.RIGHT_X, rc.RIGHT_Y = state["right_stick"]
        rc.DPAD_X, rc.DPAD_Y = state["dpad"]
    return handle


class GamepadHandler:
    """后台线程读 evdev 事件流，维护合并的 state 字典（同时打开所有像手柄的设备）。"""

    def __init__(self, device_path=None, deadzone=0.08):
        if not _HAS_EVDEV:
            raise RuntimeError("未安装 evdev，请先 `pip install evdev`。")
        self.deadzone = deadzone
        self.devices = {}
        self._absinfo = {}
        self._axis_role = {}
        self._code_map = {}
        self._cal_profiles = load_calibration()
        if self._cal_profiles:
            print(f"[gamepad] 已加载按键标定，设备档：{list(self._cal_profiles.keys())}")
        self.state = {
            "buttons": set(),
            "left_stick": [0.0, 0.0],
            "right_stick": [0.0, 0.0],
            "dpad": [0, 0],
            "triggers": {"LT": 0.0, "RT": 0.0},
        }

    def _open(self, path):
        dev = InputDevice(path)
        absinfo = {}
        for code, ai in dev.capabilities().get(ecodes.EV_ABS, []):
            absinfo[code] = (ai.min, ai.max)
        self.devices[path] = dev
        self._absinfo[path] = absinfo
        self._axis_role[path] = self._build_axis_roles(absinfo)
        profile = self._cal_profiles.get(dev.name)
        self._code_map[path] = profile if profile else dict(_CODE_TO_BTN)
        tag = "（标定）" if profile else "（默认映射）"
        print(f"✅ 接入手柄: {dev.name}  @ {path} {tag}")

    def open_all(self):
        n = 0
        for path, _name in list_gamepads():
            if path in self.devices:
                continue
            try:
                self._open(path)
                n += 1
            except PermissionError:
                print(f"⚠️ 无权限读取 {path}：执行 `sudo usermod -aG input $USER` 后重新登录。")
            except Exception as e:           # noqa: BLE001
                print(f"⚠️ 打开 {path} 失败: {e}")
        return n

    @staticmethod
    def _build_axis_roles(absinfo):
        a = set(absinfo.keys())
        role = {
            ecodes.ABS_X: ("stick", "LX"),
            ecodes.ABS_Y: ("stick", "LY"),
            ecodes.ABS_HAT0X: ("dpad", "X"),
            ecodes.ABS_HAT0Y: ("dpad", "Y"),
        }
        if ecodes.ABS_RX in a:
            role[ecodes.ABS_RX] = ("stick", "RX")
            role[ecodes.ABS_RY] = ("stick", "RY")
        else:
            role[ecodes.ABS_Z]  = ("stick", "RX")
            role[ecodes.ABS_RZ] = ("stick", "RY")
        if ecodes.ABS_GAS in a and ecodes.ABS_BRAKE in a:
            role[ecodes.ABS_GAS]   = ("trig", "RT")
            role[ecodes.ABS_BRAKE] = ("trig", "LT")
        else:
            role[ecodes.ABS_Z]  = ("trig", "LT")
            role[ecodes.ABS_RZ] = ("trig", "RT")
        return role

    def reconnect(self):
        print("🔄 正在查找并打开手柄...")
        while not self.devices:
            if self.open_all() == 0:
                time.sleep(1)

    def _norm(self, absinfo, code, value):
        lo, hi = absinfo.get(code, (-32768, 32767))
        if hi == lo:
            return 0.0
        v = (value - lo) / (hi - lo) * 2.0 - 1.0
        return 0.0 if abs(v) < self.deadzone else round(v, 3)

    def _norm_trigger(self, absinfo, code, value):
        lo, hi = absinfo.get(code, (0, 255))
        if hi == lo:
            return 0.0
        return round((value - lo) / (hi - lo), 3)

    def process_event(self, path, event):
        if event.type == ecodes.EV_KEY:
            btn = self._code_map.get(path, _CODE_TO_BTN).get(event.code)
            if btn is not None:
                if event.value:
                    self.state["buttons"].add(btn)
                else:
                    self.state["buttons"].discard(btn)
        elif event.type == ecodes.EV_ABS:
            code, val = event.code, event.value
            role = self._axis_role.get(path, {}).get(code)
            if role is None:
                return
            absinfo = self._absinfo.get(path, {})
            kind, which = role
            if kind == "stick":
                v = self._norm(absinfo, code, val)
                if which == "LX":   self.state["left_stick"][0] = v
                elif which == "LY": self.state["left_stick"][1] = v
                elif which == "RX": self.state["right_stick"][0] = v
                elif which == "RY": self.state["right_stick"][1] = v
            elif kind == "trig":
                self.state["triggers"][which] = self._norm_trigger(absinfo, code, val)
            elif kind == "dpad":
                if which == "X":    self.state["dpad"][0] = int(val)
                else:               self.state["dpad"][1] = int(val)

    def _drop(self, path):
        for d in (self.devices, self._absinfo, self._axis_role, self._code_map):
            d.pop(path, None)

    def listen(self, callback=None):
        import select
        last_scan = 0.0
        while True:
            now = time.time()
            if now - last_scan > 1.0:
                self.open_all()
                last_scan = now
            if not self.devices:
                time.sleep(0.5)
                continue
            fdmap = {d.fd: (p, d) for p, d in self.devices.items()}
            try:
                r, _, _ = select.select(list(fdmap.keys()), [], [], 0.5)
            except (OSError, ValueError):
                for p in list(self.devices):
                    try:
                        self.devices[p].capabilities()
                    except Exception:        # noqa: BLE001
                        print(f"❌ 手柄断开: {p}")
                        self._drop(p)
                continue
            for fd in r:
                path, dev = fdmap[fd]
                try:
                    for event in dev.read():
                        self.process_event(path, event)
                        if callback:
                            callback(self.state)
                except OSError:
                    print(f"❌ 手柄断开: {path}")
                    self._drop(path)


if __name__ == "__main__":
    if not _HAS_EVDEV:
        print("未安装 evdev：pip install evdev")
        raise SystemExit(1)
    print("检测到的手柄设备：")
    for p, n in list_gamepads():
        print(f"  {p}  ->  {n}")
    rc = GamepadState()
    h = GamepadHandler()
    threading.Thread(target=h.listen, args=(make_handle(rc),), daemon=True).start()
    print("开始打印手柄状态（Ctrl+C 退出）...")
    try:
        while True:
            print(rc)
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
