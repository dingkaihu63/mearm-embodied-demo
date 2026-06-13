"""
MeArm 工作台 — API 视觉物体检测器 (APIDetector)
================================================
当 YOLO 不可用时 (无 GPU / 模型未下载)，使用云端/本地 LLM 视觉模型做物体检测。

Release 版: Ollama vision model (llava / minicpm-v / qwen2.5-vl)
Original 版: Kimi k2.6 multimodal

输出与 YOLODetector 统一的 list[Detection] 接口。
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

import numpy as np

from .config import (
    API_VISION_ENABLED, API_VISION_FRAME_INTERVAL,
    YOLO_CLASS_CN, log,
)
from .shared_state import Detection

log_api = logging.getLogger("workbench.api_vision")

# ─── 物体检测 Prompt ──────────────────────────────────────────────────────────
DETECT_PROMPT = """\
你是物体检测专家。请分析这张桌面工作区域的摄像头画面, 列出你能看到的物体。

要求:
1. 只列出画面中清晰可见、适合机械臂抓取的小型物体 (如杯子/瓶子/苹果/书本等)
2. 每个物体给出: 英文类名 (cup/bottle/apple/book/scissors/cell phone/mouse等)
3. 描述物体在画面中的大致位置 (左/中/右, 上/中/下)
4. 只输出 JSON, 不要其他文字

JSON 格式:
{
  "objects": [
    {"class_name": "cup", "position": "center-middle", "description": "白色陶瓷杯"},
    ...
  ]
}

如果没有适合抓取的物体, 返回 {"objects": []}
【绝对只输出 JSON】"""


class APIDetector:
    """调用 LLM 视觉 API 做物体检测.

    与 YOLODetector 接口一致:
        detector = APIDetector(llm_parser)
        dets = detector.detect(frame) → list[Detection]
    """

    def __init__(self, llm_parser, frame_interval: int = API_VISION_FRAME_INTERVAL):
        """
        Args:
            llm_parser: LLMIntentParser 实例 (需支持 parse_with_image)
            frame_interval: API 调用帧间隔
        """
        self._llm = llm_parser
        self._frame_interval = frame_interval
        self._frame_count = 0
        self._cached: list[Detection] = []
        self._available = False

        if not API_VISION_ENABLED:
            log_api.info("API 视觉回退已禁用")
            return

        if llm_parser and getattr(llm_parser, 'is_available', False):
            # Check if vision model is available
            if hasattr(llm_parser, '_vision_model') and llm_parser._vision_model:
                self._available = True
                log.info(f"☁️ API 视觉检测器已就绪 (模型: {llm_parser._vision_model})")
            elif hasattr(llm_parser, '_client') and not hasattr(llm_parser, '_vision_model'):
                # Original: Kimi k2.6 supports multimodal natively
                self._available = True
                log.info("☁️ API 视觉检测器已就绪 (Kimi 多模态)")
            else:
                log_api.warning("API 视觉回退: 未配置视觉模型, 不可用")
        else:
            log_api.info("API 视觉回退: LLM 不可用")

    @property
    def is_available(self) -> bool:
        return self._available

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """对一帧画面调用 API 检测物体.

        Args:
            frame: BGR numpy array (H, W, 3)

        Returns:
            list[Detection]: 检测到的物体列表
        """
        if not self._available:
            return []

        self._frame_count += 1
        if self._frame_count % self._frame_interval != 1:
            return self._cached

        try:
            raw = self._call_vision_api(frame)
            dets = self._parse_response(raw, frame)
            self._cached = dets
            return dets
        except Exception as e:
            log_api.error(f"API 视觉检测失败: {e}")
            return self._cached  # 返回上次缓存

    def _call_vision_api(self, frame: np.ndarray) -> str:
        """调用底层视觉 API, 返回原始文本."""
        import base64
        import cv2

        _, jpeg = cv2.imencode(".jpg", frame,
                               [cv2.IMWRITE_JPEG_QUALITY, 60])
        img_b64 = base64.b64encode(jpeg.tobytes()).decode("ascii")

        # Try parse_with_image first (release: Ollama vision, original: Kimi)
        if hasattr(self._llm, 'parse_with_image'):
            # Use the LLM's own multimodal method with detect prompt
            result = self._llm.parse_with_image(
                DETECT_PROMPT, frame,
                visible_colors=[]  # Not needed for detection
            )
            if result and isinstance(result, dict):
                # Return raw text for parsing
                return json.dumps(result)

        # Fallback: direct API call
        content = [
            {"type": "image_url",
             "image_url": {"url": f"data:image/jpeg;base64,{img_b64}",
                          "detail": "low"}},
            {"type": "text", "text": DETECT_PROMPT},
        ]
        resp = self._llm._client.chat.completions.create(
            model=getattr(self._llm, '_vision_model',
                         getattr(self._llm, '_model', 'qwen2.5:7b')),
            max_tokens=512,
            timeout=15.0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "你是物体检测专家。只输出 JSON。"},
                {"role": "user", "content": content},
            ],
            temperature=0.1,
        )
        return resp.choices[0].message.content.strip()

    @staticmethod
    def _parse_response(raw: str, frame: np.ndarray) -> list[Detection]:
        """解析 API 返回的 JSON, 转为 Detection 列表.

        物体位置由文字描述估算像素坐标。
        """
        h, w = frame.shape[:2]

        # 位置文字 → 像素坐标映射
        POS_X = {"left": w // 4, "center": w // 2, "middle": w // 2,
                 "right": 3 * w // 4}
        POS_Y = {"top": h // 4, "upper": h // 3, "middle": h // 2,
                 "center": h // 2, "bottom": 3 * h // 4, "lower": 2 * h // 3}

        # ── JSON 提取 ──────────────────────────────────────────────────
        data = None
        raw = raw.strip()
        # 直接解析
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            pass
        # Markdown 代码块
        if data is None:
            m = re.search(r'```(?:json)?\s*\n?(\{.*?\})\s*\n?```', raw, re.DOTALL)
            if m:
                try:
                    data = json.loads(m.group(1))
                except json.JSONDecodeError:
                    pass
        # 最外层 {...}
        if data is None:
            m = re.search(r'\{.*"objects"\s*:.*\}', raw, re.DOTALL)
            if m:
                try:
                    data = json.loads(m.group(0))
                except json.JSONDecodeError:
                    pass

        if data is None:
            log_api.warning(f"无法解析 API 返回: {raw[:200]}")
            return []

        objects = data.get("objects", [])
        if not isinstance(objects, list):
            return []

        dets: list[Detection] = []
        for obj in objects:
            class_name = obj.get("class_name", "").lower()
            if not class_name:
                continue
            pos = obj.get("position", "center-middle")
            parts = pos.split("-")
            x_part = parts[0] if len(parts) > 0 else "center"
            y_part = parts[1] if len(parts) > 1 else "middle"

            cx = POS_X.get(x_part, w // 2)
            cy = POS_Y.get(y_part, h // 2)
            class_cn = YOLO_CLASS_CN.get(class_name, class_name)

            dets.append(Detection(
                color="",
                cx=cx, cy=cy,
                area=800.0,  # 估算面积
                x_mm=0.0, y_mm=0.0,  # 由 vision pipeline 填充
                class_name=class_name,
                class_cn=class_cn,
                confidence=0.7,  # API 检测默认置信度
                source="api",
                bbox=(),
            ))

        log_api.info(f"API 检测到 {len(dets)} 个物体: "
                    f"{[d.class_cn for d in dets]}")
        return dets
