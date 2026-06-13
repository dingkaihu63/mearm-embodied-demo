"""
MeArm 工作台 — 串口桥接 (ArmSerial)
==================================
从原 workbench_server.py 提取，不做任何修改。
"""

from __future__ import annotations

import threading
import time
from typing import Optional

from .config import BAUD_RATE, SERIAL_TIMEOUT, SAFE_MODE, SAFE_JOINT_DELAY, log
from .shared_state import state

# ─── 可选串口 ──────────────────────────────────────────────────────────────────
try:
    import serial
    import serial.tools.list_ports
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False


# ══════════════════════════════════════════════════════════════════════════════
# 串口桥接
# ══════════════════════════════════════════════════════════════════════════════

class ArmSerial:
    """线程安全的串口接口.

    协议 (与 Arduino 固件匹配):
      - 发送 "joint:angle,..." → 立即回复 ACK:MOVE
      - 所有舵机到达目标后 → 回复 ACK:DONE
      - 发送 "home" → 回复 ACK:HOME
      - 发送 "status" → 回复 STATUS:base:X,left:Y,right:Z,claw:W
    """

    def __init__(self, port: Optional[str]):
        self._lock = threading.Lock()
        self._ser = None
        if port and HAS_SERIAL:
            try:
                self._ser = serial.Serial(port, BAUD_RATE, timeout=SERIAL_TIMEOUT)
                time.sleep(2.0)
                self._ser.reset_input_buffer()
                # 读取 READY 启动消息
                deadline = time.time() + 3.0
                while time.time() < deadline:
                    if self._ser.in_waiting:
                        line = self._ser.readline().decode("ascii", errors="ignore").strip()
                        if line:
                            state.add_serial_rx(line)
                            state.add_log(f"Arduino: {line}")
                        if "READY" in line:
                            break
                state.serial_connected = True
                state.add_log(f"✅ 串口已连接: {port} @ {BAUD_RATE}")
            except Exception as e:
                state.add_log(f"⚠️ 串口连接失败 ({port}): {e} — 使用模拟模式")
        else:
            state.add_log("ℹ️ 模拟模式 (无串口连接)")

        if SAFE_MODE:
            state.add_log("🛡️ 安全模式: 单舵机顺序运动, USB 限流")

    def send(self, command: str, wait_ack: bool = True) -> str:
        """发送原始命令, 返回第一个响应行."""
        state.add_serial_tx(command)
        if not self._ser:
            state.add_serial_rx("OK (模拟)")
            return "OK (模拟)"

        with self._lock:
            try:
                cmd = command.strip() + "\n"
                self._ser.write(cmd.encode("ascii"))
                if wait_ack:
                    deadline = time.time() + SERIAL_TIMEOUT
                    while time.time() < deadline:
                        if self._ser.in_waiting:
                            resp = self._ser.readline().decode("ascii", errors="ignore").strip()
                            state.add_serial_rx(resp)
                            return resp
                    state.add_serial_rx("(超时)")
                    return "(超时)"
            except Exception as e:
                state.add_serial_rx(f"ERR: {e}")
                return f"ERR: {e}"
        return ""

    def read_line(self, timeout: float = SERIAL_TIMEOUT) -> str:
        """读取一行串口响应 (非阻塞等待)."""
        if not self._ser:
            return ""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._ser.in_waiting:
                resp = self._ser.readline().decode("ascii", errors="ignore").strip()
                if resp:
                    state.add_serial_rx(resp)
                    return resp
            time.sleep(0.01)
        return ""

    def wait_done(self, timeout: float = 5.0) -> bool:
        """等待 ACK:DONE (运动完成). 返回 True 表示收到."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            line = self.read_line(timeout=min(0.5, timeout))
            if "DONE" in line:
                return True
            if "ERR" in line:
                log.warning(f"Arduino 错误: {line}")
                return False
        return False

    def move(self, joints: dict[str, int]) -> str:
        """发送关节指令. 安全模式下逐个关节发送.

        返回第一个 ACK (ACK:MOVE), 运动完成时 Arduino 会异步发送 ACK:DONE.
        """
        if not SAFE_MODE or len(joints) <= 1:
            tokens = [f"{j}:{v}" for j, v in joints.items()]
            return self.send(",".join(tokens))

        # 安全模式: 逐个发送, 降低瞬时电流
        results = []
        for joint, angle in joints.items():
            r = self.send(f"{joint}:{angle}")
            results.append(r)
            time.sleep(SAFE_JOINT_DELAY)
        return "; ".join(results)

    def home(self) -> str:
        return self.send("home")

    def close(self):
        if self._ser and self._ser.is_open:
            self._ser.close()
            state.serial_connected = False
