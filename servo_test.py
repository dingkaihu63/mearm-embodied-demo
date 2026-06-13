"""
MeArm 单舵机逐一测试 (USB 供电安全模式)
==========================================
每个舵机单独测试，小幅度缓慢移动，确保 USB 不超载。

用法:
  python servo_test.py              # 自动检测 COM 口
  python servo_test.py --port COM3  # 指定端口
"""

import argparse
import sys
import time
import serial
import serial.tools.list_ports

BAUD = 115200

# 测试参数: (关节名, 起始角度, 测试目标, 步长, 间隔秒)
# 新起始位: base=30, left=150, right=150, claw=90
TESTS = [
    # (关节, 回零位, 测试角度1, 测试角度2, 步长, 每步间隔ms)
    # 夹爪 — 电流最小, 先测 (开合范围 0-90)
    ("claw",     90,  45,  90,  5, 150),
    # 右侧 (原肘部) — 从150向下测
    ("right",    150, 130, 150, 3, 200),
    # 左侧 (原肩膀) — 从150向下测, 负载大用小幅度
    ("left",     150, 135, 150, 2, 250),
    # 底座 — 从30向右测
    ("base",     30,  60,  30,  3, 200),
]


def find_port() -> str:
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        if "USB" in p.description or "Arduino" in p.description:
            return p.device
    if ports:
        return ports[0].device
    return ""


def smooth_move(ser, joint: str, start: int, end: int, step: int, interval_ms: float):
    """从 start 到 end 逐步发送，每步只动一个舵机."""
    step = step if start < end else -step
    angles = list(range(start, end + (1 if step > 0 else -1), step))
    if angles[-1] != end:
        angles.append(end)

    for i, angle in enumerate(angles):
        cmd = f"{joint}:{angle}\n"
        ser.write(cmd.encode("ascii"))
        ser.flush()
        # 读 ACK
        time.sleep(0.05)
        while ser.in_waiting:
            resp = ser.readline().decode("ascii", errors="ignore").strip()
            if resp:
                print(f"  [{joint}] {angle}° → {resp}")
        bar_len = 20
        done = int((i + 1) / len(angles) * bar_len)
        print(f"\r  {joint}: {'█' * done}{'░' * (bar_len - done)} {angle}°", end="")
        time.sleep(interval_ms / 1000.0)
    print()


def main():
    parser = argparse.ArgumentParser(description="MeArm 单舵机安全测试")
    parser.add_argument("--port", default=None, help="串口 (如 COM3)")
    args = parser.parse_args()

    port = args.port or find_port()
    if not port:
        print("❌ 未找到 Arduino 串口，请用 --port 指定")
        sys.exit(1)

    print(f"\n{'='*50}")
    print(f"  MeArm 单舵机逐一测试")
    print(f"  端口: {port}")
    print(f"  ⚠️  USB 供电模式 — 每次只动一个舵机")
    print(f"{'='*50}\n")

    try:
        ser = serial.Serial(port, BAUD, timeout=1.0)
        time.sleep(2.0)
        ser.reset_input_buffer()
    except Exception as e:
        print(f"❌ 无法打开 {port}: {e}")
        sys.exit(1)

    # 读取 READY
    time.sleep(0.3)
    while ser.in_waiting:
        line = ser.readline().decode("ascii", errors="ignore").strip()
        if line:
            print(f"  Arduino: {line}")

    print("\n🏠 先回零...")
    ser.write(b"home\n")
    time.sleep(1.5)
    while ser.in_waiting:
        line = ser.readline().decode("ascii", errors="ignore").strip()
        if line:
            print(f"  {line}")

    for joint, home_angle, test1, test2, step, interval in TESTS:
        print(f"\n{'─'*40}")
        print(f"🔧 测试 {joint.upper()} (其它舵机不动)")
        print(f"{'─'*40}")

        # 确保从回零位开始
        ser.write(f"{joint}:{home_angle}\n".encode("ascii"))
        time.sleep(0.3)

        # 等待 DONE
        deadline = time.time() + 3.0
        while time.time() < deadline:
            while ser.in_waiting:
                resp = ser.readline().decode("ascii", errors="ignore").strip()
                if "DONE" in resp:
                    deadline = 0
                elif resp:
                    print(f"  {resp}")

        # 朝着测试角度 1 移动
        print(f"  → 移动到 {test1}° ...")
        smooth_move(ser, joint, home_angle, test1, step, interval)

        # 回到测试角度 2
        print(f"  → 移动到 {test2}° ...")
        smooth_move(ser, joint, test1, test2, step, interval)

        print(f"  ✅ {joint} 测试完成")

    # 全部测完回零
    print(f"\n{'='*50}")
    print("🏠 全部测试完成，回零...")
    ser.write(b"home\n")
    time.sleep(2.0)
    while ser.in_waiting:
        resp = ser.readline().decode("ascii", errors="ignore").strip()
        if resp:
            print(f"  {resp}")

    ser.close()
    print("✅ 安全测试结束，串口已释放\n")


if __name__ == "__main__":
    main()
