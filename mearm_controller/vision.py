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
    FRAME_WIDTH, FRAME_HEIGHT,
    YOLO_ENABLED, YOLO_FRAME_INTERVAL, YOLO_CONFIDENCE_THRESHOLD,
    YOLO_CLASS_CN,
    API_VISION_ENABLED, log,
)
from .shared_state import state, Detection
from .vision_yolo import YOLODetector


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

        # ── YOLO 物体检测 ──────────────────────────────────────────────────
        self._yolo: Optional[YOLODetector] = None
        self._yolo_frame_count = 0
        self._yolo_cached: list[Detection] = []
        if YOLO_ENABLED:
            try:
                self._yolo = YOLODetector()
                if self._yolo.is_available:
                    self._yolo.warmup()
                else:
                    state.add_log("ℹ️ YOLO 不可用, 仅使用 HSV 颜色检测")
            except Exception as e:
                state.add_log(f"⚠️ YOLO 加载失败: {e} — 仅使用 HSV 颜色检测")

        # ── API 视觉回退 (云端/本地 vision LLM) ──────────────────────────
        self._api_detector = None  # 由 server.py 通过 set_api_detector() 注入
        self._api_frame_count = 0
        self._api_cached: list[Detection] = []

    def set_api_detector(self, detector):
        """注入 API 检测器 (由 server.py 在 LLM 就绪后调用)."""
        self._api_detector = detector
        if detector and getattr(detector, 'is_available', False):
            state.add_log("☁️ API 视觉回退已启用")

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

    def _open_local_camera(self, cam_index):
        """打开本地 USB/内置摄像头.

        Args:
            cam_index: 摄像头索引 (int) 或 "auto" (自动搜索 0-5)
        """
        # Windows 上优先尝试 DSHOW 后端 (兼容性更好), 再尝试 MSMF
        backends = []
        if sys.platform == "win32":
            backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        else:
            backends = [cv2.CAP_ANY]

        # ── 确定要尝试的索引列表 ──────────────────────────────────────────
        if cam_index == "auto" or cam_index < 0:
            trial_indices = list(range(6))  # 0-5 自动搜索
            state.add_log("🔍 自动搜索摄像头 (索引 0-5)...")
        else:
            trial_indices = [int(cam_index)]

        # ── 按优先级尝试 ──────────────────────────────────────────────────
        used_index = -1
        for idx in trial_indices:
            for backend in backends:
                cap = cv2.VideoCapture(idx, backend)
                if cap.isOpened():
                    # 验证能读到画面
                    ret, test_frame = cap.read()
                    if ret and test_frame is not None and test_frame.size > 0:
                        self._cap = cap
                        used_index = idx
                        break
                    cap.release()
                else:
                    cap.release()
            if self._cap is not None:
                break

        if self._cap is None or not self._cap.isOpened():
            state.add_log(f"⚠️ 无法打开摄像头 (尝试了索引 {trial_indices})")
            state.add_log("   请确认摄像头已连接, 或尝试 --cam 0 / --cam 1")
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

    def detect_objects(self, frame: np.ndarray) -> list[Detection]:
        """合并 YOLO + HSV + API 检测结果.

        优先级: YOLO (本地) > HSV (颜色) > API (云端回退)
        """
        # ── YOLO 检测 (帧间隔) ─────────────────────────────────────────────
        yolo_dets: list[Detection] = []
        yolo_ok = self._yolo and self._yolo.is_available
        if yolo_ok:
            self._yolo_frame_count += 1
            if self._yolo_frame_count % YOLO_FRAME_INTERVAL == 1:
                yolo_dets = self._yolo.detect(frame)
                for d in yolo_dets:
                    d.x_mm, d.y_mm = self.pixel_to_mm(d.cx, d.cy)
                self._yolo_cached = yolo_dets
            else:
                yolo_dets = self._yolo_cached

        # ── HSV 检测 ───────────────────────────────────────────────────────
        hsv_dets = self.detect_colors(frame)

        # ── API 视觉回退 (YOLO不可用且HSV检测少时启用) ──────────────────
        api_dets: list[Detection] = []
        if not yolo_ok and self._api_detector and self._api_detector.is_available:
            self._api_frame_count += 1
            if self._api_frame_count % 15 == 1:  # 每15帧调一次API
                api_dets = self._api_detector.detect(frame)
                for d in api_dets:
                    d.x_mm, d.y_mm = self.pixel_to_mm(d.cx, d.cy)
                self._api_cached = api_dets
            else:
                api_dets = self._api_cached

        # ── 合并去重 ───────────────────────────────────────────────────────
        merged = self._merge_detections(yolo_dets, hsv_dets)
        # 只有 YOLO 没结果时才附加 API 检测 (避免重复)
        if not yolo_dets and api_dets:
            merged = api_dets + merged
        return merged

    @staticmethod
    def _merge_detections(
        yolo_dets: list[Detection],
        hsv_dets: list[Detection],
        iou_threshold: float = 0.3,
    ) -> list[Detection]:
        """合并 YOLO 和 HSV 检测, 去重.

        规则:
        - 如果 YOLO 边界框和 HSV 圆心重叠 (中心距离 < 阈值), 合并
          → 保留 YOLO 的 class_name + HSV 的 color, source="merged"
        - 不重叠的各自保留
        - YOLO 排前面 (通常更重要)
        """
        import math

        result: list[Detection] = []
        used_hsv: set[int] = set()

        for yd in yolo_dets:
            merged = False
            for i, hd in enumerate(hsv_dets):
                if i in used_hsv:
                    continue
                # 检查中心距离
                dist = math.sqrt((yd.cx - hd.cx) ** 2 + (yd.cy - hd.cy) ** 2)
                # 使用 HSV 检测的直径作为参考
                hsv_radius = math.sqrt(hd.area / 3.14159) if hd.area > 0 else 15
                if dist < hsv_radius * 1.5 or dist < 25:
                    # 合并: YOLO 的类别 + HSV 的颜色
                    yd.color = hd.color
                    yd.source = "merged"
                    used_hsv.add(i)
                    merged = True
                    break
            if not merged:
                yd.source = "yolo"
            result.append(yd)

        # 添加未被合并的 HSV 检测
        for i, hd in enumerate(hsv_dets):
            if i not in used_hsv:
                hd.source = "hsv"
                result.append(hd)

        return result

    def annotate(self, frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
        COLOR_MAP = {
            "red": (0, 0, 220), "green": (0, 200, 0),
            "blue": (220, 80, 0), "yellow": (0, 200, 200),
        }
        # YOLO 类别颜色 (伪彩色, 按类别名 hash)
        YOLO_COLORS: dict[str, tuple] = {}

        out = frame.copy()
        for d in detections:
            # ── YOLO/merged: 画矩形框 + 标签 ──────────────────────────
            if d.source in ("yolo", "merged") and d.bbox:
                x1, y1, x2, y2 = d.bbox
                # 按类名生成稳定颜色
                if d.class_name not in YOLO_COLORS:
                    h = hash(d.class_name) % 180
                    bgr = cv2.cvtColor(np.uint8([[[h, 220, 200]]]), cv2.COLOR_HSV2BGR)[0][0]
                    bgr = (int(bgr[0]), int(bgr[1]), int(bgr[2]))
                    YOLO_COLORS[d.class_name] = bgr
                bgr = YOLO_COLORS[d.class_name]

                cv2.rectangle(out, (x1, y1), (x2, y2), bgr, 2)

                # 标签: 中文类名 + 置信度
                label = f"{d.class_cn or d.class_name} {d.confidence:.2f}"
                if d.color and d.source == "merged":
                    label += f" [{d.color}]"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 6, y1), bgr, -1)
                cv2.putText(out, label, (x1 + 3, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # ── HSV: 画圆点 (保持原有样式) ──────────────────────────────
            if d.source in ("hsv", "merged"):
                bgr_dot = COLOR_MAP.get(d.color, (180, 180, 180))
                cv2.circle(out, (d.cx, d.cy), 12, bgr_dot, -1)
                cv2.circle(out, (d.cx, d.cy), 14, (255, 255, 255), 1)

            # ── 纯 YOLO (无颜色): 画小圆标记中心 ───────────────────────
            if d.source == "yolo":
                cv2.circle(out, (d.cx, d.cy), 5, (255, 255, 255), -1)
                cv2.circle(out, (d.cx, d.cy), 7, (0, 0, 0), 1)

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

        dets = self.detect_objects(frame)
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
        if self._yolo is not None:
            self._yolo.release()
            self._yolo = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        state.camera_connected = False
