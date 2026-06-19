"""
MeArm 工作台 — 视觉管线 (VisionPipeline)
=======================================
从原 workbench_server.py 提取，不做任何修改。
"""

from __future__ import annotations

import math
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
    YOLO_CLASS_CN, FACE_DETECTION_ENABLED, FACE_FRAME_INTERVAL, log,
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

        # ── HSV 资源缓存 (避免每帧重建 kernel/np.array) ──────────────────
        self._hsv_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        self._hsv_lo_cache: dict[str, np.ndarray] = {}
        self._hsv_hi_cache: dict[str, np.ndarray] = {}

        # ── 人脸检测 & 性别识别 ──────────────────────────────────────────
        self._face_detector = None
        self._face_frame_count = 0
        if FACE_DETECTION_ENABLED:
            try:
                from .face_detector import FaceDetector
                self._face_detector = FaceDetector()
                if self._face_detector.is_available:
                    state.add_log("👤 人脸检测已集成到视觉管线")
            except Exception as e:
                state.add_log(f"⚠️ 人脸检测加载失败: {e}")

        # ── 检测结果时序平滑 (减少抖动) ──────────────────────────────────
        # 保留最近 3 帧检测, 用位置加权平均稳定中心点
        self._det_history: deque = deque(maxlen=3)
        self._det_match_dist = 40  # 匹配距离阈值 (像素)

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

    def _get_hsv_arrays(self, color_name: str, lo: tuple, hi: tuple) -> tuple[np.ndarray, np.ndarray]:
        """获取缓存的 HSV lo/hi numpy 数组, 仅在范围变化时重建."""
        key = f"{color_name}_{lo}_{hi}"
        if key not in self._hsv_lo_cache:
            self._hsv_lo_cache[key] = np.array(lo)
            self._hsv_hi_cache[key] = np.array(hi)
        return self._hsv_lo_cache[key], self._hsv_hi_cache[key]

    def detect_colors(self, frame: np.ndarray) -> list[Detection]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        results: list[Detection] = []

        with state._lock:
            ranges = dict(state.hsv_ranges)

        kernel = self._hsv_kernel

        for color_name, (lo, hi) in ranges.items():
            if color_name == "red2":
                continue
            lo_arr, hi_arr = self._get_hsv_arrays(color_name, lo, hi)
            mask = cv2.inRange(hsv, lo_arr, hi_arr)
            if color_name == "red":
                lo2, hi2 = ranges.get("red2", ((170, 120, 70), (180, 255, 255)))
                lo2_arr, hi2_arr = self._get_hsv_arrays("red2", lo2, hi2)
                mask |= cv2.inRange(hsv, lo2_arr, hi2_arr)

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
        """合并 YOLO + HSV 检测结果 + 时序平滑.

        策略:
        1. YOLO 检测通用物体 (每 YOLO_FRAME_INTERVAL 帧跑一次)
        2. HSV 检测彩色物体 (每帧都跑)
        3. 去重: 如果 YOLO 和 HSV 检测到同一位置, 合并为一条 (source="merged")
        4. 时序平滑: 与历史帧匹配, 加权平均中心点减少抖动
        5. 返回合并后的 Detection 列表
        """
        # ── YOLO 检测 (帧间隔) ─────────────────────────────────────────────
        yolo_dets: list[Detection] = []
        yolo_ok = self._yolo and self._yolo.is_available
        if yolo_ok:
            self._yolo_frame_count += 1
            if self._yolo_frame_count % YOLO_FRAME_INTERVAL == 1:
                yolo_dets = self._yolo.detect(frame)
                # 填充世界坐标
                for d in yolo_dets:
                    d.x_mm, d.y_mm = self.pixel_to_mm(d.cx, d.cy)
                self._yolo_cached = yolo_dets
            else:
                yolo_dets = self._yolo_cached

        # ── HSV 检测 ───────────────────────────────────────────────────────
        hsv_dets = self.detect_colors(frame)

        # ── 合并去重 ───────────────────────────────────────────────────────
        merged = self._merge_detections(yolo_dets, hsv_dets)

        # ── 时序平滑: 减少检测中心抖动 ───────────────────────────────────
        smoothed = self._temporal_smooth(merged)
        return smoothed

    def _temporal_smooth(self, current: list[Detection]) -> list[Detection]:
        """时序平滑: 用最近几帧的检测结果加权平均中心点, 减少抖动.

        匹配规则: 同颜色/同类名的检测在 _det_match_dist 像素内视为同一物体.
        权重: 当前帧 0.6, 历史帧 0.4 (越新权重越高).
        """
        if not current:
            self._det_history.append([])
            return current

        # 当前帧加入历史
        self._det_history.append(list(current))

        # 如果只有 1 帧历史, 无法平滑
        if len(self._det_history) < 2:
            return current

        history = list(self._det_history)
        result: list[Detection] = []

        for d in current:
            # 在历史帧中找匹配
            matched_cx = [d.cx]
            matched_cy = [d.cy]
            for past_frame in history[:-1]:
                best_match = None
                best_dist = self._det_match_dist
                for pd in past_frame:
                    # 同颜色或同类名才匹配
                    same_id = (d.color and pd.color == d.color) or \
                              (d.class_name and pd.class_name == d.class_name)
                    if not same_id:
                        continue
                    dist = math.hypot(d.cx - pd.cx, d.cy - pd.cy)
                    if dist < best_dist:
                        best_dist = dist
                        best_match = pd
                if best_match:
                    matched_cx.append(best_match.cx)
                    matched_cy.append(best_match.cy)

            # 加权平均: 当前帧权重 0.6, 历史帧平分 0.4
            if len(matched_cx) > 1:
                n_hist = len(matched_cx) - 1
                hist_weight = 0.4 / n_hist
                cur_weight = 0.6
                total_w = cur_weight + hist_weight * n_hist
                smooth_cx = int((d.cx * cur_weight + sum(x * hist_weight for x in matched_cx[1:])) / total_w)
                smooth_cy = int((d.cy * cur_weight + sum(y * hist_weight for y in matched_cy[1:])) / total_w)
                # 更新世界坐标
                x_mm, y_mm = self.pixel_to_mm(smooth_cx, smooth_cy)
                result.append(Detection(
                    color=d.color, cx=smooth_cx, cy=smooth_cy, area=d.area,
                    x_mm=x_mm, y_mm=y_mm,
                    class_name=d.class_name, class_cn=d.class_cn,
                    confidence=d.confidence, source=d.source, bbox=d.bbox,
                ))
            else:
                result.append(d)

        return result

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
        result: list[Detection] = []
        used_hsv: set[int] = set()

        for yd in yolo_dets:
            merged = False
            for i, hd in enumerate(hsv_dets):
                if i in used_hsv:
                    continue
                # 检查中心距离
                dist = math.hypot(yd.cx - hd.cx, yd.cy - hd.cy)
                # 使用 HSV 检测的直径作为参考
                hsv_radius = math.sqrt(hd.area / math.pi) if hd.area > 0 else 15
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

        # ── 人脸检测 (帧间隔, 节省 CPU) ────────────────────────────────
        if self._face_detector and self._face_detector.is_available:
            self._face_frame_count += 1
            if self._face_frame_count % FACE_FRAME_INTERVAL == 1:
                self._face_detector.process_frame(frame, pixel_to_mm=self.pixel_to_mm)

        annotated = self.annotate(frame, dets)
        # 在人脸检测可用时, 叠加人脸标注
        if self._face_detector and self._face_detector.is_available:
            annotated = self._face_detector.annotate(annotated)
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
        self._face_detector = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        state.camera_connected = False
