"""
MeArm 工作台 — 视觉管线 (VisionPipeline)
=======================================
从原 workbench_server.py 提取，不做任何修改。
"""

from __future__ import annotations

import sys
import time
from collections import deque
from typing import Optional

import cv2
import numpy as np

from .config import (
    HSV_RANGES, MIN_CONTOUR_AREA, CALIB_POINTS,
    FRAME_WIDTH, FRAME_HEIGHT, log,
)
from .shared_state import state, Detection


# ══════════════════════════════════════════════════════════════════════════════
# 视觉管线
# ══════════════════════════════════════════════════════════════════════════════

class VisionPipeline:
    """摄像头采集 + HSV 颜色检测 + 单应性变换.

    支持两种模式:
      1. 本地摄像头: VisionPipeline(cam_index=0)
      2. 手机 IP 摄像头: VisionPipeline(ip_cam_url="http://192.168.1.5:8080/video")
    """

    def __init__(self, cam_index: int = 0, ip_cam_url: Optional[str] = None):
        self._cap = None
        self._cam_index = cam_index
        self._ip_cam_url = ip_cam_url
        self._source_label = ""

        if ip_cam_url:
            # ── 手机 IP 摄像头模式 ──────────────────────────────────────────
            self._open_ip_camera(ip_cam_url)
        else:
            # ── 本地 USB/内置摄像头模式 ──────────────────────────────────────
            self._open_local_camera(cam_index)

        self._H_mat = self._build_homography()
        self._fps_counter = deque(maxlen=30)
        self._last_time = time.time()
        self._grab_failures = 0

    def _open_ip_camera(self, url: str):
        """通过 IP 摄像头 (手机) 的 MJPEG 流连接."""
        state.add_log(f"📱 正在连接手机摄像头: {url}")
        # cv2.VideoCapture 可以直接打开 MJPEG/RTSP 流
        cap = cv2.VideoCapture(url)
        # 给一点时间建立连接
        time.sleep(1.0)
        if cap.isOpened():
            self._cap = cap
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
            state.camera_connected = True
            self._source_label = f"手机 ({url})"
            state.camera_source = f"手机 IP ({url})"
            state.add_log(f"✅ 手机摄像头已连接: {url}")
        else:
            cap.release()
            state.add_log(f"⚠️ 无法连接手机摄像头: {url}")
            state.add_log("   请确认手机 IP 摄像头 App 已开启，且电脑和手机在同一 WiFi")
            state.camera_connected = False

    def _open_local_camera(self, cam_index: int):
        """打开本地 USB/内置摄像头."""
        # Windows 上优先尝试 DSHOW 后端 (兼容性更好), 再尝试 MSMF
        backends = []
        if sys.platform == "win32":
            backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        else:
            backends = [cv2.CAP_ANY]

        for backend in backends:
            cap = cv2.VideoCapture(cam_index, backend)
            if cap.isOpened():
                self._cap = cap
                break
            cap.release()

        # 如果指定索引打不开, 尝试 0, 1
        if self._cap is None:
            for idx in [0, 1]:
                if idx == cam_index:
                    continue
                for backend in backends:
                    cap = cv2.VideoCapture(idx, backend)
                    if cap.isOpened():
                        self._cap = cap
                        state.add_log(f"⚠️ 索引 {cam_index} 不可用, 回退到索引 {idx}")
                        break
                    cap.release()
                if self._cap is not None:
                    break

        if self._cap is None or not self._cap.isOpened():
            state.add_log(f"⚠️ 无法打开摄像头 (尝试了所有可用索引)")
            state.camera_connected = False
        else:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
            self._cap.set(cv2.CAP_PROP_FPS, 30)
            state.camera_connected = True
            self._source_label = f"本地摄像头 (索引 {cam_index})"
            state.camera_source = f"本地 (索引 {cam_index})"
            state.add_log(f"✅ 本地摄像头已打开 (索引 {cam_index})")

        self._H_mat = self._build_homography()
        self._fps_counter = deque(maxlen=30)
        self._last_time = time.time()
        self._grab_failures = 0

    def _build_homography(self) -> Optional[np.ndarray]:
        if len(CALIB_POINTS) < 4:
            return None
        px = np.float32([[p[0], p[1]] for p in CALIB_POINTS])
        mm = np.float32([[p[2], p[3]] for p in CALIB_POINTS])
        H, _ = cv2.findHomography(px, mm)
        return H

    def pixel_to_mm(self, px: int, py: int) -> tuple[float, float]:
        if self._H_mat is None:
            return 0.0, 0.0
        pt = np.array([[[float(px), float(py)]]], dtype=np.float32)
        out = cv2.perspectiveTransform(pt, self._H_mat)
        return float(out[0][0][0]), float(out[0][0][1])

    def detect_colors(self, frame: np.ndarray) -> list[Detection]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        results: list[Detection] = []

        with state._lock:
            ranges = dict(state.hsv_ranges)

        for color_name, (lo, hi) in ranges.items():
            if color_name == "red2":
                continue
            mask = cv2.inRange(hsv, np.array(lo), np.array(hi))
            if color_name == "red":
                lo2, hi2 = ranges.get("red2", ((170, 120, 70), (180, 255, 255)))
                mask |= cv2.inRange(hsv, np.array(lo2), np.array(hi2))

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < MIN_CONTOUR_AREA:
                    continue
                M = cv2.moments(cnt)
                if M["m00"] == 0:
                    continue
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                x_mm, y_mm = self.pixel_to_mm(cx, cy)
                results.append(Detection(color_name, cx, cy, area, x_mm, y_mm))

        # 每种颜色只保留最大的
        best: dict[str, Detection] = {}
        for d in results:
            if d.color not in best or d.area > best[d.color].area:
                best[d.color] = d
        return list(best.values())

    def annotate(self, frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
        COLOR_MAP = {
            "red": (0, 0, 220), "green": (0, 200, 0),
            "blue": (220, 80, 0), "yellow": (0, 200, 200),
        }
        out = frame.copy()
        for d in detections:
            bgr = COLOR_MAP.get(d.color, (180, 180, 180))
            cv2.circle(out, (d.cx, d.cy), 12, bgr, -1)
            cv2.circle(out, (d.cx, d.cy), 14, (255, 255, 255), 1)
            label = f"{d.color} ({d.x_mm:.0f},{d.y_mm:.0f}mm)"
            # 背景框
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(out, (d.cx + 14, d.cy - th - 10),
                          (d.cx + 14 + tw + 4, d.cy - 2), (40, 40, 40), -1)
            cv2.putText(out, label, (d.cx + 16, d.cy - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, bgr, 2)
        return out

    def build_mask_display(self, frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
        """构建所有颜色掩膜的合成显示."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        composite = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)

        color_bgr = {"red": (0, 0, 255), "green": (0, 255, 0),
                     "blue": (255, 0, 0), "yellow": (0, 255, 255)}

        with state._lock:
            ranges = dict(state.hsv_ranges)

        for color_name, (lo, hi) in ranges.items():
            if color_name == "red2":
                continue
            mask = cv2.inRange(hsv, np.array(lo), np.array(hi))
            if color_name == "red":
                lo2, hi2 = ranges.get("red2", ((170, 120, 70), (180, 255, 255)))
                mask |= cv2.inRange(hsv, np.array(lo2), np.array(hi2))

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            composite[mask > 0] = color_bgr.get(color_name, (128, 128, 128))

        # 叠加上检测框
        for d in detections:
            cv2.circle(composite, (d.cx, d.cy), 15, (255, 255, 255), 2)

        return composite

    def process_frame(self) -> bool:
        """处理一帧. 返回 True 表示成功."""
        if self._cap is None or not self._cap.isOpened():
            time.sleep(0.1)
            return False

        ret, frame = self._cap.read()
        if not ret or frame is None:
            self._grab_failures += 1
            time.sleep(0.05)
            return False

        self._grab_failures = 0

        # FPS 计算
        now = time.time()
        self._fps_counter.append(now - self._last_time)
        self._last_time = now
        if len(self._fps_counter) >= 10:
            state.vision_fps = 1.0 / (sum(self._fps_counter) / len(self._fps_counter))

        dets = self.detect_colors(frame)
        annotated = self.annotate(frame, dets)
        mask_display = self.build_mask_display(frame, dets)

        state.update_frames(frame, mask_display, annotated)
        state.update_detections(dets)
        return True

    def read_raw(self) -> Optional[np.ndarray]:
        if self._cap is None:
            return None
        ret, frame = self._cap.read()
        return frame if ret else None

    def release(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        state.camera_connected = False
