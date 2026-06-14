"""
MeArm 工作台 — 线程安全共享状态
===============================
从原 workbench_server.py 提取，不做任何修改。
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .config import HOME_ANGLES, JOINT_LIMITS, HSV_RANGES
from . import config as _config


# ══════════════════════════════════════════════════════════════════════════════
# 线程安全共享状态
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Detection:
    color: str
    cx: int
    cy: int
    area: float
    x_mm: float = 0.0
    y_mm: float = 0.0
    # ── YOLO 扩展字段 ──────────────────────────────────────────────
    class_name: str = ""          # 英文类名 (e.g. "cup")
    class_cn: str = ""            # 中文类名 (e.g. "杯子")
    confidence: float = 1.0       # 检测置信度 (HSV 默认 1.0)
    source: str = "hsv"           # "yolo" | "hsv" | "merged"
    bbox: tuple = ()              # YOLO 边界框 (x1, y1, x2, y2), HSV 为空


class SharedState:
    """所有子系统共享的线程安全状态."""

    def __init__(self):
        self._lock = threading.Lock()

        # 关节角度 (当前指令值)
        self.joint_angles: dict[str, int] = dict(HOME_ANGLES)

        # 摄像头帧
        self.raw_frame: Optional[np.ndarray] = None
        self.mask_frame: Optional[np.ndarray] = None
        self.annotated_frame: Optional[np.ndarray] = None

        # 检测结果
        self.detections: list[Detection] = []

        # 事件日志 (最近 200 条)
        self.log_entries: deque[str] = deque(maxlen=200)

        # 串口消息 (最近 100 条)
        self.serial_tx: deque[str] = deque(maxlen=100)
        self.serial_rx: deque[str] = deque(maxlen=100)

        # 连接状态
        self.serial_connected: bool = False
        self.camera_connected: bool = False
        self.camera_source: str = ""
        self.arm_busy: bool = False
        self.current_gesture: str = ""
        self.vision_fps: float = 0.0

        # 手势识别状态
        self.hand_detected: bool = False
        self.hand_gesture: str = ""
        self.hand_landmarks: list = []  # 21 个 MediaPipe 关键点 (x, y, z)
        self.gesture_recog_enabled: bool = True       # 手势识别开关
        self.gesture_paused_until: float = 0.0        # 暂停识别直到的时间戳

        # HSV 范围 (运行时可调)
        self.hsv_ranges: dict = dict(HSV_RANGES)

        # 语音识别最近文本 (供 learner 使用)
        self.last_voice_text: str = ""

        # 动作完成后自动暂停 (防止过度识别和重复运动)
        # True = 暂停等待用户按"继续", 语音/手势被阻塞
        # False = 正常监听
        self.action_paused: bool = True

    def update_joint(self, name: str, angle: int):
        with self._lock:
            lo, hi = JOINT_LIMITS.get(name, (0, 180))
            self.joint_angles[name] = max(lo, min(hi, int(angle)))

    def update_joints(self, angles: dict[str, int]):
        with self._lock:
            for name, angle in angles.items():
                lo, hi = JOINT_LIMITS.get(name, (0, 180))
                self.joint_angles[name] = max(lo, min(hi, int(angle)))

    def update_frames(self, raw: np.ndarray, mask: np.ndarray, annotated: np.ndarray):
        with self._lock:
            self.raw_frame = raw.copy()
            self.mask_frame = mask.copy()
            self.annotated_frame = annotated.copy()

    def update_detections(self, dets: list[Detection]):
        with self._lock:
            self.detections = list(dets)

    def add_log(self, msg: str):
        with self._lock:
            self.log_entries.append(f"[{time.strftime('%H:%M:%S')}] {msg}")
        import logging
        logging.getLogger("workbench").info(msg)

    def add_serial_tx(self, msg: str):
        with self._lock:
            self.serial_tx.append(f"[{time.strftime('%H:%M:%S')}] TX: {msg}")

    def add_serial_rx(self, msg: str):
        with self._lock:
            self.serial_rx.append(f"[{time.strftime('%H:%M:%S')}] RX: {msg}")

    def get_state_dict(self) -> dict:
        with self._lock:
            return {
                "joints": dict(self.joint_angles),
                "detections": [{"color": d.color, "cx": d.cx, "cy": d.cy,
                                "area": int(d.area), "x_mm": round(d.x_mm, 1),
                                "y_mm": round(d.y_mm, 1),
                                "class_name": d.class_name,
                                "class_cn": d.class_cn,
                                "confidence": round(d.confidence, 2),
                                "source": d.source,
                                "bbox": list(d.bbox) if d.bbox else []}
                               for d in self.detections],
                "serial_connected": self.serial_connected,
                "camera_connected": self.camera_connected,
                "camera_source": self.camera_source,
                "arm_busy": self.arm_busy,
                "current_gesture": self.current_gesture,
                "vision_fps": round(self.vision_fps, 1),
                "hand_detected": self.hand_detected,
                "hand_gesture": self.hand_gesture,
                "gesture_recog_enabled": self.gesture_recog_enabled,
                "action_paused": self.action_paused,
            }

    def get_logs(self) -> list[str]:
        with self._lock:
            return list(self.log_entries)

    def get_serial_msgs(self) -> dict:
        with self._lock:
            return {"tx": list(self.serial_tx), "rx": list(self.serial_rx)}


# 全局状态实例
state = SharedState()
