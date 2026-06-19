"""
MeArm 工作台 — 人脸检测 & 性别识别 (FaceDetector)
=================================================
使用 OpenCV Haar Cascade 人脸检测 + Caffe DNN 性别分类。
性别模型可选 — 没有模型时仅检测人脸，性别返回空字符串。
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import cv2
import numpy as np

from .shared_state import state

log = logging.getLogger("workbench")

# ─── 性别模型文件路径 ──────────────────────────────────────────────────────
_GENDER_PROTO = "models/gender_deploy.prototxt"
_GENDER_MODEL = "models/gender_net.caffemodel"

# ─── 性别标签 (Caffe 模型的输出) ─────────────────────────────────────────────
_GENDER_LIST = ["male", "female"]


class FaceDetector:
    """人脸检测 + 性别识别.

    人脸检测: OpenCV Haar Cascade (内置, 无需额外模型)
    性别识别: Caffe DNN 模型 (可选, 约 3MB)
    """

    def __init__(self):
        self._available = False
        self._gender_available = False

        # ── 1. Haar Cascade 人脸检测 (内置) ──────────────────────────────
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        if not os.path.exists(cascade_path):
            log.warning("Haar Cascade 人脸检测模型不存在: %s", cascade_path)
            self._face_cascade = None
        else:
            self._face_cascade = cv2.CascadeClassifier(cascade_path)
            self._available = True
            log.info("👤 人脸检测已就绪 (Haar Cascade)")

        # ── 2. Caffe 性别分类模型 (可选) ────────────────────────────────
        self._gender_net = None
        if os.path.exists(_GENDER_PROTO) and os.path.exists(_GENDER_MODEL):
            try:
                self._gender_net = cv2.dnn.readNetFromCaffe(
                    _GENDER_PROTO, _GENDER_MODEL)
                self._gender_available = True
                log.info("👤 性别识别已就绪 (Caffe DNN)")
            except Exception as e:
                log.warning("性别模型加载失败: %s — 仅检测人脸", e)
        else:
            log.info("ℹ️ 性别模型未安装, 仅检测人脸位置。"
                     "下载 gender_deploy.prototxt + gender_net.caffemodel 到 models/ 即可启用。")

        # 防抖: 连续 N 帧未检测到人脸才清除状态
        self._face_lost_frames = 0
        self._face_lost_threshold = 15  # ~0.5 秒 @30fps

    @property
    def is_available(self) -> bool:
        return self._available

    @property
    def gender_available(self) -> bool:
        return self._gender_available

    def detect_faces(self, frame: np.ndarray) -> list[dict]:
        """检测帧中的所有人脸.

        Returns:
            [{"bbox": (x, y, w, h), "cx": int, "cy": int,
              "gender": "male"|"female"|"", "gender_conf": float}, ...]
            按人脸面积降序排列 (最大的人脸排第一).
        """
        if not self._available or self._face_cascade is None:
            return []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 直方图均衡化改善光照
        gray = cv2.equalizeHist(gray)

        faces = self._face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        results = []
        for (x, y, w, h) in faces:
            cx = x + w // 2
            cy = y + h // 2
            gender = ""
            gender_conf = 0.0

            # ── 性别分类 ────────────────────────────────────────────────
            if self._gender_net is not None:
                try:
                    face_roi = frame[y:y + h, x:x + w]
                    blob = cv2.dnn.blobFromImage(
                        face_roi, scalefactor=1.0, size=(227, 227),
                        mean=(78.42633776, 87.76891437, 114.89584775),
                        swapRB=False,
                    )
                    self._gender_net.setInput(blob)
                    preds = self._gender_net.forward()
                    idx = int(np.argmax(preds[0]))
                    gender_conf = float(preds[0][idx])
                    if gender_conf >= 0.5:
                        gender = _GENDER_LIST[idx]
                except Exception:
                    pass

            results.append({
                "bbox": (int(x), int(y), int(w), int(h)),
                "cx": int(cx),
                "cy": int(cy),
                "gender": gender,
                "gender_conf": round(gender_conf, 2),
            })

        # 按面积降序
        results.sort(key=lambda r: r["bbox"][2] * r["bbox"][3], reverse=True)
        return results

    def get_primary_face(self, frame: np.ndarray) -> Optional[dict]:
        """获取画面中主要的人脸 (最大的)."""
        faces = self.detect_faces(frame)
        return faces[0] if faces else None

    def process_frame(self, frame: np.ndarray,
                      pixel_to_mm=None) -> Optional[dict]:
        """处理一帧, 更新共享状态中的人脸信息.

        Args:
            frame: BGR 帧
            pixel_to_mm: 可选, 像素→世界坐标转换函数 (cx, cy) → (x_mm, y_mm)
        """
        face = self.get_primary_face(frame)
        if face is not None:
            self._face_lost_frames = 0
            state.face_detected = True
            state.face_gender = face.get("gender", "")
            state.face_confidence = face.get("gender_conf", 0.0)
            state.face_bbox = face.get("bbox", ())
            state.face_cx = face["cx"]
            state.face_cy = face["cy"]
            if pixel_to_mm is not None:
                x_mm, y_mm = pixel_to_mm(face["cx"], face["cy"])
                state.face_x_mm = x_mm
                state.face_y_mm = y_mm
        else:
            self._face_lost_frames += 1
            if self._face_lost_frames >= self._face_lost_threshold:
                state.face_detected = False
                state.face_gender = ""
                state.face_confidence = 0.0
                state.face_bbox = ()
                state.face_cx = 0
                state.face_cy = 0
                state.face_x_mm = 0.0
                state.face_y_mm = 0.0
        return face

    def annotate(self, frame: np.ndarray) -> np.ndarray:
        """在帧上标注人脸和性别."""
        out = frame.copy()
        faces = self.detect_faces(frame)
        for i, face in enumerate(faces):
            x, y, w, h = face["bbox"]
            color = (0, 255, 0) if i == 0 else (200, 200, 200)
            cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
            label_parts = ["face"]
            if face["gender"]:
                label_parts.append({"male": "M", "female": "F"}.get(
                    face["gender"], "?"))
            label = " ".join(label_parts)
            cv2.putText(out, label, (x, y - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return out
