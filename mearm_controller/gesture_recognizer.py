"""
MeArm 工作台 — 手势识别 (GestureRecognizer)
==========================================
MediaPipe HandLandmarker (Tasks API) 实时手势分类。

稳定性优化:
  - 多帧确认: 同一手势需连续 N 帧才触发 (避免单帧误判)
  - 退出迟滞: 手势消失后需连续 M 帧 "none" 才确认退出
  - 提高 MediaPipe 检测/跟踪置信度阈值
  - 挥手检测: 更长的历史窗口 + 更多穿越次数
  - 加长冷却时间
"""

from __future__ import annotations

import os
import time
from collections import deque
from typing import Optional

import cv2
import numpy as np

from .config import (
    MP_WRIST, MP_THUMB_TIP, MP_INDEX_TIP, MP_MIDDLE_TIP, MP_RING_TIP, MP_PINKY_TIP,
    MP_INDEX_PIP, MP_MIDDLE_PIP, MP_RING_PIP, MP_PINKY_PIP,
    MP_INDEX_MCP, MP_MIDDLE_MCP, MP_RING_MCP, MP_PINKY_MCP,
    HAND_GESTURE_CN, GESTURE_COOLDOWN,
    GESTURE_MIN_CONSECUTIVE, GESTURE_EXIT_HYSTERESIS,
    WAVE_MIN_CROSSES, WAVE_HISTORY_FRAMES,
    log,
)
from .shared_state import state


def _distance(a, b):
    """两个 landmark 之间的欧氏距离."""
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


class GestureRecognizer:
    """使用 MediaPipe HandLandmarker (Tasks API) 实时识别手势.

    process_frame(frame) → 返回手势名称字符串.
    通过 state.hand_detected / state.hand_gesture 同步状态.

    稳定性机制:
      - 同一手势需连续 GESTURE_MIN_CONSECUTIVE 帧确认才触发
      - 手势消失后需连续 GESTURE_EXIT_HYSTERESIS 帧 "none" 才重置
      - 挥手需要 WAVE_MIN_CROSSES 次穿越 + WAVE_HISTORY_FRAMES 帧历史
    """

    MODEL_FILE = "hand_landmarker.task"

    FINGERS = {
        "thumb":  (MP_THUMB_TIP,  MP_INDEX_MCP, MP_WRIST),
        "index":  (MP_INDEX_TIP,  MP_INDEX_PIP, MP_INDEX_MCP),
        "middle": (MP_MIDDLE_TIP, MP_MIDDLE_PIP, MP_MIDDLE_MCP),
        "ring":   (MP_RING_TIP,   MP_RING_PIP, MP_RING_MCP),
        "pinky":  (MP_PINKY_TIP,  MP_PINKY_PIP, MP_PINKY_MCP),
    }

    # 需要多帧确认的手势 (所有手势都做多帧确认)
    CONFIRM_GESTURES = {"open_palm", "fist", "pointing", "thumbs_up", "peace", "wave"}

    def __init__(self):
        self._available = False
        self._detector = None

        # 挥手检测状态
        self._wrist_history: deque = deque(maxlen=WAVE_HISTORY_FRAMES)
        self._wave_frames = 0

        # 冷却计时
        self._last_gesture = ""
        self._last_gesture_time = 0.0

        # ── 多帧确认 ──────────────────────────────────────────────────────
        self._confirm_gesture = ""       # 当前正在确认的手势
        self._confirm_count = 0          # 连续帧数
        self._none_count = 0             # 连续 "none" 帧数 (退出迟滞)

        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision

            if not os.path.exists(self.MODEL_FILE):
                log.warning(f"MediaPipe 模型文件未找到: '{self.MODEL_FILE}'")
                state.add_log("⚠️ 手势识别模型缺失 (hand_landmarker.task)")
                return

            base_options = mp_python.BaseOptions(model_asset_path=self.MODEL_FILE)
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                num_hands=1,
                min_hand_detection_confidence=0.8,   # 提高: 0.7 → 0.8
                min_tracking_confidence=0.7,          # 提高: 0.5 → 0.7
            )
            self._detector = vision.HandLandmarker.create_from_options(options)
            self._mp = mp
            self._mp_python = mp_python
            self._available = True
            state.add_log("✋ MediaPipe 手势识别已就绪 (高稳定模式)")
        except ImportError:
            log.warning("mediapipe 未安装 — 手势识别不可用")
        except Exception as e:
            log.warning(f"MediaPipe 初始化失败: {e}")

    def _is_finger_extended(self, landmarks, name: str) -> bool:
        """判断一根手指是否伸展."""
        tip_idx, pip_idx, mcp_idx = self.FINGERS[name]
        tip = np.array([landmarks[tip_idx].x, landmarks[tip_idx].y, landmarks[tip_idx].z])
        pip = np.array([landmarks[pip_idx].x, landmarks[pip_idx].y, landmarks[pip_idx].z])
        mcp = np.array([landmarks[mcp_idx].x, landmarks[mcp_idx].y, landmarks[mcp_idx].z])

        if name == "thumb":
            wrist = np.array([landmarks[MP_WRIST].x, landmarks[MP_WRIST].y])
            index_mcp = np.array([landmarks[MP_INDEX_MCP].x, landmarks[MP_INDEX_MCP].y])
            thumb_tip = np.array([landmarks[MP_THUMB_TIP].x, landmarks[MP_THUMB_TIP].y])
            return np.linalg.norm(thumb_tip - wrist) > np.linalg.norm(index_mcp - wrist) * 1.1

        return _distance(tip, mcp) > _distance(pip, mcp) * 1.2

    def _classify_on_frame(self, detection_result) -> tuple[str, list]:
        """在单帧内分类手势. 返回 (手势名, 关键点列表)."""
        if not detection_result.hand_landmarks:
            return "none", []

        landmarks = detection_result.hand_landmarks[0]
        h, w = 480, 640

        pts = [(int(l.x * w), int(l.y * h), l.z) for l in landmarks]

        ext = {name: self._is_finger_extended(landmarks, name) for name in self.FINGERS}
        fingers_up = sum(1 for v in ext.values() if v)

        if fingers_up == 5:
            gesture = "open_palm"
        elif fingers_up == 0:
            gesture = "fist"
        elif fingers_up == 1 and ext["index"]:
            gesture = "pointing"
        elif fingers_up == 1 and ext["thumb"]:
            gesture = "thumbs_up"
        elif fingers_up == 2 and ext["index"] and ext["middle"]:
            gesture = "peace"
        elif fingers_up == 2 and ext["index"] and ext["thumb"]:
            gesture = "pointing"
        else:
            gesture = "none"

        # 挥手检测 (更严格)
        if gesture == "open_palm":
            wrist_x = landmarks[MP_WRIST].x
            self._wrist_history.append(wrist_x)
            self._wave_frames += 1
            if self._wave_frames >= WAVE_HISTORY_FRAMES and len(self._wrist_history) >= WAVE_HISTORY_FRAMES:
                crosses = 0
                direction = 0
                hist_list = list(self._wrist_history)
                prev = hist_list[0]
                for x in hist_list[1:]:
                    d = 1 if x > prev else -1
                    if d != direction and direction != 0:
                        crosses += 1
                    direction = d
                    prev = x
                if crosses >= WAVE_MIN_CROSSES:
                    gesture = "wave"
        else:
            self._wrist_history.clear()
            self._wave_frames = 0

        return gesture, pts

    def process_frame(self, frame: np.ndarray) -> Optional[str]:
        """处理一帧, 返回检测到并确认的手势名称.

        稳定性逻辑:
          1. 单帧分类 → 原始手势名
          2. 同一手势需连续 N 帧不变才"确认"
          3. 手势消失后需连续 M 帧 "none" 才重置状态
          4. 已确认的手势在冷却期内不再触发
        """
        if not self._available or self._detector is None:
            return None

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._detector.detect(mp_image)

        raw_gesture, pts = self._classify_on_frame(result)
        now = time.time()

        state.hand_detected = bool(result.hand_landmarks)
        state.hand_gesture = raw_gesture
        state.hand_landmarks = pts

        # ── 多帧确认 ──────────────────────────────────────────────────────
        if raw_gesture != "none" and raw_gesture == self._confirm_gesture:
            # 相同手势持续中
            self._confirm_count += 1
            self._none_count = 0
        elif raw_gesture != "none" and raw_gesture != self._confirm_gesture:
            # 手势变化 → 重置计数器
            self._confirm_gesture = raw_gesture
            self._confirm_count = 1
            self._none_count = 0
        elif raw_gesture == "none":
            # 无手势 → 累加退出迟滞
            self._none_count += 1
            if self._none_count >= GESTURE_EXIT_HYSTERESIS:
                # 足够多帧无手势, 确认退出
                self._confirm_gesture = ""
                self._confirm_count = 0
                self._none_count = 0

        # ── 确认触发检查 ──────────────────────────────────────────────────
        if self._confirm_count == GESTURE_MIN_CONSECUTIVE:
            # 刚好达到确认阈值 → 触发
            confirmed = self._confirm_gesture

            # 冷却检查
            if confirmed == self._last_gesture:
                if now - self._last_gesture_time < GESTURE_COOLDOWN:
                    # 冷却中, 不触发但保持确认状态
                    return None

            self._last_gesture = confirmed
            self._last_gesture_time = now
            return confirmed

        return None

    def release(self):
        if self._detector:
            self._detector.close()
