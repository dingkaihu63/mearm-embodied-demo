"""
MeArm 工作台 — LLM 意图解析器 (LLMIntentParser)
==============================================
从原 workbench_server.py 提取，不做任何修改。
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Optional

from .config import (
    LLM_SYSTEM_PROMPT,
    # 云端 API
    LLM_API_KEY, LLM_API_BASE_URL, LLM_API_MODEL,
    VISION_API_KEY, VISION_API_BASE_URL, VISION_API_MODEL,
    VISION_API_TEMPERATURE,
    # 本地 Ollama
    OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_VISION_MODEL,
    LLM_TIMEOUT, LLM_MAX_TOKENS, COLOR_CN, YOLO_CN_CLASS, log,
)
from .shared_state import state
from .speaker import Speaker


# ══════════════════════════════════════════════════════════════════════════════
# LLM 意图解析器 (本地 Ollama / 云端 API 自适应)
# ══════════════════════════════════════════════════════════════════════════════

class LLMIntentParser:
    """调用 LLM 将文本/图片解析为结构化动作.

    自动检测提供者:
    - 设置 LLM_API_KEY 环境变量 → 云端 API (DeepSeek/OpenAI 等)
    - 未设置 → 本地 Ollama

    多模态视觉:
    - 设置 VISION_API_KEY → 云端视觉 API (Kimi/GPT-4V 等)
    - 设置 OLLAMA_VISION_MODEL → Ollama 视觉模型 (llava 等)
    - 均未设置 → 纯文本模式
    """

    # 语音回复系统提示词 — 简短自然口语化
    REPLY_SYSTEM_PROMPT = (
        "你是 MeArm 桌面机械臂的语音助手。用户刚刚执行了一个操作，"
        "你需要用一句简短、自然、口语化的中文来回应。"
        "可以带一点俏皮或温暖的语气，但不要啰嗦（不超过20个字）。"
        "只输出这句话，不要加任何前缀、引号或解释。"
    )

    def __init__(self, system_prompt: Optional[str] = None):
        self._available = False
        self._system_prompt = system_prompt or LLM_SYSTEM_PROMPT

        # ── 检测 LLM 提供者: 云端 API > 本地 Ollama ─────────────────
        if LLM_API_KEY:
            self._provider = "cloud"
            self._model = LLM_API_MODEL
            self._base_url = LLM_API_BASE_URL
            self._api_key = LLM_API_KEY
        else:
            self._provider = "ollama"
            self._model = OLLAMA_MODEL
            self._base_url = OLLAMA_BASE_URL
            self._api_key = "ollama"  # Ollama 忽略 api_key 但 SDK 要求非空

        # ── 检测视觉提供者: 云端视觉 > Ollama 视觉 > 无 ────────────
        self._vision_client = None
        if VISION_API_KEY:
            self._vision_provider = "cloud"
            self._vision_model = VISION_API_MODEL
            self._vision_base_url = VISION_API_BASE_URL
            self._vision_api_key = VISION_API_KEY
        elif OLLAMA_VISION_MODEL:
            self._vision_provider = "ollama"
            self._vision_model = OLLAMA_VISION_MODEL
            self._vision_base_url = OLLAMA_BASE_URL
            self._vision_api_key = "ollama"
        else:
            self._vision_provider = None
            self._vision_model = None
            self._vision_base_url = None
            self._vision_api_key = None

        try:
            from openai import OpenAI
            self._client = OpenAI(
                base_url=self._base_url,
                api_key=self._api_key,
            )
            # 如果视觉提供者与文本提供者不同, 创建独立客户端
            if (self._vision_provider and
                    self._vision_base_url != self._base_url):
                self._vision_client = OpenAI(
                    base_url=self._vision_base_url,
                    api_key=self._vision_api_key,
                )
            else:
                self._vision_client = self._client
            self._available = True

            provider_label = {"cloud": "☁️ 云端", "ollama": "🖥️ Ollama"}
            state.add_log(
                f"🧠 LLM 已就绪 ({provider_label.get(self._provider, self._provider)}: {self._model})"
            )
            if self._vision_model:
                vp_label = {"cloud": "☁️ 云端视觉", "ollama": "🖥️ Ollama 视觉"}
                state.add_log(
                    f"👁️ 多模态视觉已就绪 ({vp_label.get(self._vision_provider, self._vision_provider)}: {self._vision_model})"
                )
        except ImportError:
            state.add_log("⚠️ openai 包未安装 — LLM 已禁用")
        except Exception as e:
            state.add_log(f"⚠️ LLM 初始化失败: {e}")

    @property
    def is_available(self) -> bool:
        return self._available

    # ── API 调用核心 ───────────────────────────────────────────────────

    def _call_api(self, content, use_vision: bool = False) -> Optional[dict]:
        """调用 LLM API, 返回解析后的意图 dict.

        Args:
            content: 消息内容 (字符串或 multimodal 列表)
            use_vision: 是否使用视觉模型 (自动选择客户端和 temperature)
        """
        client = self._vision_client if use_vision and self._vision_client else self._client
        model = self._vision_model if use_vision and self._vision_model else self._model
        temperature = VISION_API_TEMPERATURE if use_vision else 0.1
        try:
            resp = client.chat.completions.create(
                model=model,
                max_tokens=LLM_MAX_TOKENS,
                timeout=LLM_TIMEOUT,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": self._system_prompt},
                    {"role": "user", "content": content},
                ],
                temperature=temperature,
            )
            raw = resp.choices[0].message.content.strip()
            log.info(f"LLM 原始回复 ({model}): {raw[:200]}")
            return self._extract_json(raw)
        except Exception as e:
            log.error(f"LLM 调用失败 ({model}): {e}")
            return None

    # ── 纯文本解析 ────────────────────────────────────────────────────

    def parse(self, text: str, visible_colors: list[str],
             spatial_context: str = "") -> Optional[dict]:
        if not self._available:
            return None
        context_part = ""
        if spatial_context:
            context_part = f"\n{spatial_context}\n"
        user_msg = (
            f"命令: \"{text}\"\n"
            f"可见物体: {visible_colors if visible_colors else ['无']}\n"
            f"{context_part}"
            f"请返回 json 格式的意图解析结果."
        )
        return self._call_api(user_msg)

    # ── 多模态解析 (图片 + 文本, 需要视觉模型) ──────────────────────

    def parse_with_image(self, text: str, image_bgr,
                         visible_colors: list[str],
                         spatial_context: str = "") -> Optional[dict]:
        """多模态意图解析 — 将摄像头画面 + 文本发送给 Ollama 视觉模型.

        需要配置 OLLAMA_VISION_MODEL 环境变量 (如 llava, qwen2.5-vl:7b).
        未配置则返回 None, 由上层回退到纯文本解析.
        """
        if not self._available or not self._vision_model:
            return None
        import base64
        import cv2

        _, jpeg = cv2.imencode(".jpg", image_bgr,
                               [cv2.IMWRITE_JPEG_QUALITY, 60])
        img_b64 = base64.b64encode(jpeg.tobytes()).decode("ascii")

        context_part = ""
        if spatial_context:
            context_part = f"\n{spatial_context}\n"

        user_content = [
            {"type": "image_url",
             "image_url": {"url": f"data:image/jpeg;base64,{img_b64}",
                           "detail": "low"}},
            {"type": "text",
             "text": (
                 f"用户说: \"{text}\"\n"
                 f"HSV 检测到的颜色: {visible_colors if visible_colors else ['无']}\n"
                 f"{context_part}"
                 f"请观察画面中的物体和手势，结合语音和空间记忆，返回 json 意图."
             )},
        ]
        return self._call_api(user_content, use_vision=True)

    @staticmethod
    def _extract_json(raw: str) -> Optional[dict]:
        """从 LLM 回复中提取 JSON, 兼容以下情况:
        1. 纯 JSON: '{"action": "say", ...}'
        2. Markdown 代码块: '```json\\n{...}\\n```'
        3. 自然语言 + JSON 混合: '好的，以下是...\\n{...}'
        4. 空字符串
        """
        import re
        raw = raw.strip()
        if not raw:
            return None

        result = None

        # 尝试直接解析
        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            pass

        # 尝试从 markdown 代码块提取
        if result is None:
            m = re.search(r'```(?:json)?\s*\n?(\{.*?\})\s*\n?```', raw, re.DOTALL)
            if m:
                try:
                    result = json.loads(m.group(1))
                except json.JSONDecodeError:
                    pass

        # 尝试从文本中提取最外层 {...}
        if result is None:
            m = re.search(r'\{[^{}]*"action"\s*:\s*"[^"]*"[^{}]*\}', raw, re.DOTALL)
            if m:
                try:
                    result = json.loads(m.group(0))
                except json.JSONDecodeError:
                    pass

        if result is None:
            return None

        # ── 字段名规范化: 处理 LLM 返回的非标准格式 ─────────────────
        return LLMIntentParser._normalize_intent(result)

    @staticmethod
    def _normalize_intent(raw: dict) -> dict:
        """将各种非标准 JSON 格式规范化为标准意图 dict.

        处理的变体:
          {"intent": "other", "gesture": "point"} → action=say
          {"intent": "select", "target": "red"} → action=pick_and_place, color=red
          {"intent": "greeting"} → action=gesture, gesture=greet
        """
        action = raw.get("action", "")
        if not action:
            # 尝试从 intent 字段推断
            intent = raw.get("intent", "").lower()
            if intent in ("greet", "greeting", "hello", "hi"):
                action = "gesture"
                raw.setdefault("gesture", "greet")
            elif intent in ("pick", "pick_and_place", "grab", "fetch", "select", "get"):
                action = "pick_and_place"
                raw.setdefault("color", raw.get("target") or raw.get("color"))
            elif intent in ("wave", "挥手"):
                action = "gesture"
                raw.setdefault("gesture", "wave")
            elif intent in ("point", "指向", "pointing"):
                action = "gesture"
                raw.setdefault("gesture", "point")
            elif intent in ("nod", "点头"):
                action = "gesture"
                raw.setdefault("gesture", "nod")
            elif intent in ("home", "reset", "回零"):
                action = "home"
            elif intent in ("stop", "halt", "暂停"):
                action = "say"
            elif intent in ("open", "张开", "松手"):
                action = "claw_open"
            elif intent in ("close", "闭合", "抓住"):
                action = "claw_close"
            elif intent in ("move", "rotate", "旋转", "移动"):
                action = "move_joint"
            else:
                action = "say"

        # 确保必有字段 (透传 joint/direction 给 move_joint)
        result = {
            "action": action,
            "color": raw.get("color") or raw.get("target"),
            "class_name": raw.get("class_name"),  # YOLO 物体类名
            "gesture": raw.get("gesture"),
            "message": raw.get("message", ""),
            "confidence": float(raw.get("confidence", 0.5)),
        }
        # 透传 move_joint 专用字段
        if action == "move_joint":
            result["joint"] = raw.get("joint", "base")
            result["direction"] = int(raw.get("direction", 1))
        return result

    def reply(self, action_desc: str) -> Optional[str]:
        """为已执行的操作生成一句自然的语音回复（供 TTS 播报）。"""
        if not self._available:
            return None
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                max_tokens=60,
                timeout=5.0,
                messages=[
                    {"role": "system", "content": self.REPLY_SYSTEM_PROMPT},
                    {"role": "user", "content": f"操作：{action_desc}"},
                ],
                temperature=0.8,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            log.error(f"LLM reply 失败: {e}")
            return None

    # ─── 颜色名映射 (中文→英文) ──────────────────────────────────────────
    COLOR_MAP = {
        "红": "red", "红色": "red", "红的": "red", "red": "red",
        "绿": "green", "绿色": "green", "绿的": "green", "green": "green",
        "蓝": "blue", "蓝色": "blue", "蓝的": "blue", "blue": "blue",
        "黄": "yellow", "黄色": "yellow", "黄的": "yellow", "yellow": "yellow",
        "白": "white", "白色": "white", "white": "white",
        "黑": "black", "黑色": "black", "black": "black",
        "紫": "purple", "紫色": "purple", "橙": "orange", "橙色": "orange",
        "粉": "pink", "粉色": "pink",
    }

    # ─── 结构化关键词规则表 (优先级从高到低) ──────────────────────────────
    KEYWORD_RULES = [
        # ── 1. 停止/暂停 (最高优先级) ─────────────────────────────────
        (["停止", "暂停", "停", "停下", "别动", "stop", "halt", "halt", "住手"],
         {"action": "say", "message": "好的，已停止。", "confidence": 1.0}),

        # ── 2. 回零/复位 ──────────────────────────────────────────────
        (["回零", "归位", "复位", "初始位置", "回家", "回去", "home", "reset",
          "恢复", "归零", "回到原点", "回原位"],
         {"action": "home", "message": "收到，正在回到初始位置。", "confidence": 1.0}),

        # ── 3. 夹爪控制 ──────────────────────────────────────────────
        (["张开", "打开", "松手", "放手", "松开", "open", "张开夹爪", "松爪"],
         {"action": "claw_open", "message": "好的，张开夹爪。", "confidence": 0.95}),
        (["闭合", "抓住", "抓紧", "夹紧", "握住", "握紧", "合上", "close", "grip",
          "闭合夹爪", "夹住", "抓一下"],
         {"action": "claw_close", "message": "好的，夹爪已闭合。", "confidence": 0.95}),

        # ── 4. 问候/社交 ─────────────────────────────────────────────
        (["你好", "您好", "嗨", "hello", "hi", "嘿", "哈喽", "在吗", "在不在",
          "早上好", "晚上好", "下午好", "早啊", "好久不见", "来了"],
         {"action": "gesture", "gesture": "greet",
          "message": "你好呀，我是机械臂小助手，有什么可以帮你的？", "confidence": 0.95}),
        (["谢谢", "感谢", "多谢", "thanks", "thank", "辛苦了", "麻烦你了", "谢了"],
         {"action": "say", "message": "不客气，随时为你服务~", "confidence": 0.95}),
        (["再见", "拜拜", "bye", "回头见", "走了", "下次见", "晚安", "拜"],
         {"action": "gesture", "gesture": "wave",
          "message": "再见，下次见~", "confidence": 0.95}),
        (["你叫什么", "你是谁", "你的名字", "自我介绍", "介绍一下自己", "你是什么"],
         {"action": "say", "message": "我是 MeArm 桌面机械臂，叫我小臂就好~", "confidence": 0.95}),
        (["你怎么样", "你好吗", "how are you", "状态怎么样"],
         {"action": "say", "message": "我很好，随时待命！", "confidence": 0.9}),

        # ── 5. 手势触发 ──────────────────────────────────────────────
        (["挥手", "wave", "摇手", "摇摆", "摆摆手"],
         {"action": "gesture", "gesture": "wave", "message": "收到，挥手~", "confidence": 0.9}),
        (["点头", "nod", "鞠躬", "点点头"],
         {"action": "gesture", "gesture": "nod", "message": "好的，点头~", "confidence": 0.9}),
        # 指向 (方向感知)
        (["指左边", "指左", "指向左边", "向左指", "左边", "point left"],
         {"action": "gesture", "gesture": "point_left",
          "message": "好的，指向左边。", "confidence": 0.9}),
        (["指右边", "指右", "指向右边", "向右指", "右边", "point right"],
         {"action": "gesture", "gesture": "point_right",
          "message": "好的，指向右边。", "confidence": 0.9}),
        (["指前面", "指前", "指向前面", "向前指", "前面",
          "指向", "指着", "point", "那边", "那个方向", "指点"],
         {"action": "gesture", "gesture": "point",
          "message": "好的，指向那边~", "confidence": 0.9}),
        (["问候", "greet", "打招呼", "问好"],
         {"action": "gesture", "gesture": "greet", "message": "你好呀~", "confidence": 0.9}),
        (["点赞", "棒", "thumbs up", "厉害", "牛", "赞"],
         {"action": "gesture", "gesture": "greet", "message": "谢谢夸奖~", "confidence": 0.85}),
        (["耶", "peace", "剪刀手", "胜利"],
         {"action": "gesture", "gesture": "wave", "message": "耶~", "confidence": 0.85}),

        # ── 6. 底座旋转 ──────────────────────────────────────────────
        (["左转", "转左", "向左转", "逆时针", "turn left", "转到左边", "往左"],
         {"action": "move_joint", "joint": "base", "direction": -1,
          "message": "好的，向左转。", "confidence": 0.9}),
        (["右转", "转右", "向右转", "顺时针", "turn right", "转到右边", "往右"],
         {"action": "move_joint", "joint": "base", "direction": 1,
          "message": "好的，向右转。", "confidence": 0.9}),

        # ── 7. 手臂升降 ──────────────────────────────────────────────
        (["抬臂", "举高", "抬高", "抬起来", "raise", "往上抬", "举起来", "升高",
          "高一点", "再高点"],
         {"action": "move_joint", "joint": "lift", "direction": 1,
          "message": "好的，抬高手臂。", "confidence": 0.9}),
        (["放低", "降低", "放下来", "lower", "往下放", "低一点", "再低点",
          "降下来", "放下手臂"],
         {"action": "move_joint", "joint": "lift", "direction": -1,
          "message": "好的，放低手臂。", "confidence": 0.9}),

        # ── 8. 肘部伸缩 ──────────────────────────────────────────────
        (["前伸", "伸出去", "往前伸", "extend", "伸长", "伸出去点", "往前",
          "伸远点", "够过去"],
         {"action": "move_joint", "joint": "elbow", "direction": 1,
          "message": "好的，向前伸展。", "confidence": 0.9}),
        (["后缩", "收回来", "往后缩", "retract", "缩回来", "收回去", "往后",
          "缩一点", "收一下", "收回", "收缩"],
         {"action": "move_joint", "joint": "elbow", "direction": -1,
          "message": "好的，收缩回来。", "confidence": 0.9}),

        # ── 9. 速度控制 ──────────────────────────────────────────────
        (["快点", "加速", "快一点", "faster", "速度快点", "迅速"],
         {"action": "say", "message": "好的，加快速度。", "confidence": 0.85}),
        (["慢点", "减速", "慢一点", "slower", "速度慢点", "慢些"],
         {"action": "say", "message": "好的，放慢速度。", "confidence": 0.85}),

        # ── 10. 状态查询 ─────────────────────────────────────────────
        (["状态", "status", "怎么样", "在哪里", "什么位置", "当前位置", "报告状态"],
         {"action": "say", "message": "机械臂各关节正常，随时待命。", "confidence": 0.9}),
        (["看到什么", "有什么", "检测到什么", "能看到", "识别", "看到没"],
         {"action": "say", "message": "让我看看摄像头画面...", "confidence": 0.8}),

        # ── 11. 确认/否定 ────────────────────────────────────────────
        (["是的", "好的", "行", "可以", "没问题", "yes", "ok", "okay",
          "好呀", "好啊", "确认", "没错", "对的对的", "对的", "嗯嗯", "好嘞"],
         {"action": "say", "message": "好的，收到！", "confidence": 0.9}),
        (["不要", "别了", "算了", "取消", "no", "cancel", "不对", "不是",
          "不用了", "没事", "别这样", "不做了", "放弃"],
         {"action": "say", "message": "好的，已取消。", "confidence": 0.9}),

        # ── 12. 空间指示 ─────────────────────────────────────────────
        (["这里", "这儿", "here", "这个位置"],
         {"action": "say", "message": "看到这里了。", "confidence": 0.75}),
        (["那里", "那儿", "那边", "there", "那个位置"],
         {"action": "say", "message": "看到那边了。", "confidence": 0.75}),

        # ── 13. 自检/帮助 ─────────────────────────────────────────────
        (["帮助", "help", "怎么用", "有什么功能", "能做什么", "你会什么",
          "你的功能", "使用说明"],
         {"action": "say",
          "message": "我可以语音控制抓取物体、做手势、回零。试试说'你好'或'抓红色'~",
          "confidence": 0.9}),

        # ── 14. 空间记忆引用 ───────────────────────────────────────────
        (["刚才放的", "刚刚放的", "上次放的", "刚放的那个", "刚才那个", "刚刚那个",
          "上次那个", "上一个", "刚放的东西", "刚刚放的东西", "刚才放的东西"],
         {"action": "_spatial_lookup_last_placed", "confidence": 1.0,
          "message": "让我回想一下刚刚放了什么..."}),
    ]

    # ─── 抓取动词 ───────────────────────────────────────────────────────
    PICK_VERBS = ["抓", "捡", "拿", "取", "pick", "grab", "get", "take",
                  "搬运", "移动", "搬", "夹取", "抓取", "拾取", "捡起"]

    @staticmethod
    def keyword_fallback(text: str, visible_colors: list[str],
                         spatial=None) -> dict:
        """结构化关键词匹配.

        优先级:
        0. 空间记忆引用 (spatial memory lookup, 如果 spatial 可用)
        1. 颜色 + 抓取动词 → pick_and_place
        2. 结构化规则表 (KEYWORD_RULES, 按定义顺序)
        3. 部分关键词匹配 (回零, 停止等)
        4. 未知 → 返回低置信度, 交给上层 LLM

        Args:
            text: 用户输入文本
            visible_colors: 摄像头可见颜色列表
            spatial: 可选的空间记忆库 (SpatialMemory)
        """
        t = text.lower()

        # ── A. 颜色 + 抓取动词 (最高优先级, 精确匹配) ──────────────────
        for cn_color, en_color in LLMIntentParser.COLOR_MAP.items():
            if cn_color in t:
                if any(v in t for v in LLMIntentParser.PICK_VERBS):
                    col_name = COLOR_CN.get(en_color, en_color)
                    return {"action": "pick_and_place", "color": en_color,
                            "class_name": None,
                            "gesture": None,
                            "message": f"好的，正在抓取{col_name}物体。",
                            "confidence": 0.85}
                # 如果只说了颜色没说要抓, 提示一下
                if cn_color in ["红色", "绿色", "蓝色", "黄色", "红", "绿", "蓝", "黄"]:
                    # 给中等置信度, 可能用户想说抓取但没说完整
                    pass

        # ── A2. 物体名 + 抓取动词 ──────────────────────────────────────────
        for cn_name, en_name in YOLO_CN_CLASS.items():
            if cn_name in t:
                if any(v in t for v in LLMIntentParser.PICK_VERBS):
                    return {"action": "pick_and_place", "color": None,
                            "class_name": en_name,
                            "gesture": None,
                            "message": f"好的，正在抓取{cn_name}。",
                            "confidence": 0.85}
                break  # 一个文本最多匹配一个物体名

        # ── A3. 空间记忆引用 (spatial memory lookup) ──────────────────────
        if spatial is not None and not spatial.is_empty:
            take_back_keywords = ["拿回来", "取回来", "拿回去", "拿过来", "取过来",
                                  "放回去", "放回来", "拣回来", "捡回来",
                                  "拿回", "取回", "带回", "搬回"]
            has_take_back = any(k in t for k in take_back_keywords)

            if has_take_back:
                last = spatial.last_placed()
                if last:
                    col_name = COLOR_CN.get(last.color, last.color)
                    desc = last.object_desc or col_name
                    return {"action": "pick_and_place",
                            "color": last.color,
                            "class_name": last.class_name or None,
                            "gesture": None,
                            "message": f"好的，我去把{desc}拿回来。",
                            "confidence": 0.8}

            # 用户说 "刚才放的是什么" / "刚才放的XX"
            if any(k in t for k in ["刚才放的", "刚刚放的", "上次放的", "刚放的那个",
                                      "刚才那个", "刚刚那个", "上次那个"]):
                last = spatial.last_placed()
                if last:
                    col_name = COLOR_CN.get(last.color, last.color)
                    desc = last.object_desc or col_name
                    if any(v in t for v in LLMIntentParser.PICK_VERBS + ["拿", "取", "捡", "移动", "搬"]):
                        return {"action": "pick_and_place",
                                "color": last.color,
                                "class_name": last.class_name or None,
                                "gesture": None,
                                "message": f"好的，我来处理{desc}。",
                                "confidence": 0.8}
                    else:
                        return {"action": "say",
                                "color": None, "class_name": None,
                                "gesture": None,
                                "message": f"刚刚放的是{desc}，在({last.drop_x:.0f}, {last.drop_y:.0f})mm 处。",
                                "confidence": 0.9}

        # ── B. 遍历结构化规则表 ────────────────────────────────────────
        for keywords, intent in LLMIntentParser.KEYWORD_RULES:
            if any(k in t for k in keywords):
                result = dict(intent)  # 拷贝
                result.setdefault("color", None)
                result.setdefault("class_name", None)
                result.setdefault("gesture", None)

                # ── 处理空间记忆查找动作 ──────────────────────────────
                if result.get("action", "").startswith("_spatial_lookup"):
                    if spatial is not None and not spatial.is_empty:
                        last = spatial.last_placed()
                        if last:
                            col_name = COLOR_CN.get(last.color, last.color)
                            desc = last.object_desc or col_name
                            result = {
                                "action": "say",
                                "color": last.color,
                                "class_name": last.class_name or None,
                                "gesture": None,
                                "message": f"刚刚放的是{desc}，在({last.drop_x:.0f}, {last.drop_y:.0f})mm 处。",
                                "confidence": 0.9,
                            }
                            return result
                        else:
                            return {"action": "say", "color": None, "class_name": None,
                                    "gesture": None,
                                    "message": "我还没放过任何东西呢。", "confidence": 0.9}
                    else:
                        return {"action": "say", "color": None, "class_name": None,
                                "gesture": None,
                                "message": "空间记忆未启用，无法查询历史。", "confidence": 0.5}

                # 如果是 pick_and_place 且没指定颜色, 用 visible 中第一个
                if result.get("action") == "pick_and_place" and not result.get("color"):
                    if visible_colors:
                        result["color"] = visible_colors[0]
                return result

        # ── C. 模糊回零匹配 (单独处理, "回"字太短容易误触发) ──────────
        if any(k in t for k in ["回零", "回原位", "回去", "归位"]):
            return {"action": "home", "color": None, "class_name": None,
                    "gesture": None,
                    "message": "收到，正在回到初始位置。", "confidence": 1.0}

        # ── D. 未知指令 → 低置信度, 交给 Ollama LLM ───────────────────
        return {"action": "say", "color": None, "class_name": None,
                "gesture": None,
                "message": "抱歉，我没有理解您的指令。", "confidence": 0.0}


# ══════════════════════════════════════════════════════════════════════════════
# LLM 语音回复辅助
# ══════════════════════════════════════════════════════════════════════════════

def llm_speak(speaker: Speaker, llm: Optional[LLMIntentParser],
              action_desc: str, fallback: str = ""):
    """用 LLM 生成自然的语音回复并播报。

    流程: 后台线程调用 Ollama → speaker.speak() 播报。
    如果 LLM 不可用或失败，播报 fallback。
    """
    if llm and llm.is_available:
        def _worker():
            try:
                reply = llm.reply(action_desc)
                if reply:
                    speaker.speak(reply)
                elif fallback:
                    speaker.speak(fallback)
            except Exception as e:
                log.error(f"llm_speak 线程异常: {e}")
                if fallback:
                    speaker.speak(fallback)
        threading.Thread(target=_worker, daemon=True).start()
    elif fallback:
        speaker.speak(fallback)
