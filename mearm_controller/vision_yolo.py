"""
MeArm 工作台 — YOLO 物体检测器 (YOLODetector)
=============================================
基于 ultralytics YOLOv8, 在 RTX 5060 (CUDA) 或 CPU 上运行。

输出与现有 HSV 管道兼容的 Detection 对象列表。
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from .config import (
    YOLO_MODEL, YOLO_CONFIDENCE_THRESHOLD, YOLO_DEVICE,
    YOLO_CLASS_CN, log,
)
from .shared_state import Detection

log_yolo = logging.getLogger("workbench.yolo")


class YOLODetector:
    """Ultralytics YOLO 检测器封装.

    使用方式:
        detector = YOLODetector()
        dets = detector.detect(frame)  # frame 是 BGR numpy array
        # dets 是 list[Detection], 每个 Detection 含 class_name, confidence, bbox
    """

    def __init__(
        self,
        model_name: str = YOLO_MODEL,
        confidence: float = YOLO_CONFIDENCE_THRESHOLD,
        device: str = YOLO_DEVICE,
    ):
        self._confidence = confidence
        self._device = device
        self._model = None
        self._available = False
        self._frame_count = 0

        try:
            import torch
            from ultralytics import YOLO

            # 自动检测设备
            if device == "cuda:0" and not torch.cuda.is_available():
                log_yolo.warning("CUDA 不可用, 回退到 CPU")
                device = "cpu"

            self._model = YOLO(model_name)
            self._model.to(device)
            self._available = True
            log.info(f"🔍 YOLO 检测器已就绪 (模型: {model_name}, 设备: {device})")
        except ImportError as e:
            log.warning(f"⚠️ YOLO 依赖缺失: {e} — 物体检测已禁用")
        except Exception as e:
            log.warning(f"⚠️ YOLO 初始化失败: {e} — 物体检测已禁用")

    @property
    def is_available(self) -> bool:
        return self._available

    @property
    def device(self) -> str:
        return self._device

    def warmup(self):
        """预热推理 (避免第一帧卡顿)."""
        if not self._available:
            return
        try:
            dummy = np.zeros((480, 640, 3), dtype=np.uint8)
            self._model(dummy, conf=self._confidence, verbose=False)
            log_yolo.debug("YOLO 预热完成")
        except Exception:
            pass

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """对一帧 BGR 图像做 YOLO 推理, 返回 Detection 列表.

        Args:
            frame: BGR 格式的 numpy 数组 (H, W, 3)

        Returns:
            list[Detection]: 检测到的物体, 按置信度降序排列
        """
        if not self._available or self._model is None:
            return []

        self._frame_count += 1
        try:
            results = self._model(
                frame,
                conf=self._confidence,
                verbose=False,
                device=self._device,
            )
        except Exception as e:
            log_yolo.error(f"YOLO 推理失败: {e}")
            return []

        detections: list[Detection] = []
        if not results or results[0].boxes is None:
            return detections

        boxes = results[0].boxes
        h, w = frame.shape[:2]

        for i in range(len(boxes)):
            conf = float(boxes.conf[i])
            if conf < self._confidence:
                continue

            cls_id = int(boxes.cls[i])
            class_name = results[0].names.get(cls_id, f"class_{cls_id}")
            class_cn = YOLO_CLASS_CN.get(class_name, class_name)

            # 边界框
            xyxy = boxes.xyxy[i].cpu().numpy()
            x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
            # 边界检查
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)

            # 中心点
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            area = float((x2 - x1) * (y2 - y1))

            detections.append(Detection(
                color="",           # YOLO 没有颜色信息, 留给 merge 阶段补充
                cx=cx, cy=cy,
                area=area,
                x_mm=0.0, y_mm=0.0,  # 世界坐标由 VisionPipeline.pixel_to_mm 填充
                class_name=class_name,
                class_cn=class_cn,
                confidence=round(conf, 3),
                source="yolo",
                bbox=(x1, y1, x2, y2),
            ))

        # 按置信度降序
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    def release(self):
        """释放模型资源."""
        if self._model is not None:
            del self._model
            self._model = None
        self._available = False
