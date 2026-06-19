"""
MeArm 工作台 — LLM 意图解析器 (LLMIntentParser)
==============================================
Ollama / 云端 API 自适应 + 本地关键词库 + 降级规则。
"""

from __future__ import annotations

import json
import logging
import re
import threading
from typing import Optional

from .config import (
    LLM_SYSTEM_PROMPT, LLM_API_KEY, LLM_API_BASE_URL, LLM_API_MODEL,
    OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_VISION_MODEL,
    VISION_API_KEY, VISION_API_BASE_URL, VISION_API_MODEL, VISION_API_TEMPERATURE,
    COLOR_CN, YOLO_CN_CLASS, LLM_TIMEOUT, LLM_MAX_TOKENS, log,
)
from .shared_state import state
from .speaker import Speaker
from .keyword_library import get_library


# ── 预编译正则 (避免每次调用重新编译) ──────────────────────────────────────────
_RE_JSON_BLOCK = re.compile(r'```(?:json)?\s*\n?(\{.*?\})\s*\n?```', re.DOTALL)
_RE_JSON_INLINE = re.compile(r'\{[^{}]*"action"\s*:\s*"[^"]*"[^{}]*\}', re.DOTALL)


# ══════════════════════════════════════════════════════════════════════════════
# LLM 意图解析器
# ══════════════════════════════════════════════════════════════════════════════

class LLMIntentParser:
    """LLM 意图解析器 — 自动检测 Ollama / 云端 API 提供者."""

    REPLY_SYSTEM_PROMPT = (
        "你是 MeArm 桌面机械臂的语音助手。用户刚刚执行了一个操作，"
        "你需要用一句简短、自然、口语化的中文来回应。"
        "可以带一点俏皮或温暖的语气，但不要啰嗦（不超过20个字）。"
        "只输出这句话，不要加任何前缀、引号或解释。"
    )

    # ─── 中文颜色 → 英文映射 ──────────────────────────────────────────
    COLOR_MAP = {
        "红": "red", "红色": "red", "红的": "red", "red": "red",
        "绿": "green", "绿色": "green", "绿的": "green", "green": "green",
        "蓝": "blue", "蓝色": "blue", "蓝的": "blue", "blue": "blue",
        "黄": "yellow", "黄色": "yellow", "黄的": "yellow", "yellow": "yellow",
    }

    # ─── 抓取动词 ────────────────────────────────────────────────────
    PICK_VERBS = ["抓", "捡", "拿", "取", "pick", "grab", "get", "take",
                  "搬运", "移动", "搬", "夹取", "抓取", "拾取", "捡起",
                  "握", "夹", "捏", "提", "拎", "捞", "拾", "端"]

    # ─── 关键词索引 (类级构建, O(1) 查找替代 O(n) 遍历) ──────────────────
    _keyword_index: dict[str, list[tuple[int, dict]]] = {}

    @classmethod
    def _build_keyword_index(cls):
        """构建关键词 → 规则的反向索引 (仅构建一次)."""
        if cls._keyword_index:
            return
        for idx, (keywords, intent) in enumerate(cls.KEYWORD_RULES):
            for kw in keywords:
                kw_lower = kw.lower()
                if kw_lower not in cls._keyword_index:
                    cls._keyword_index[kw_lower] = []
                cls._keyword_index[kw_lower].append((idx, intent))

    def __init__(self):
        self._available = False
        self._provider = "none"
        self._vision_model = None
        self._vision_provider = "none"
        self._system_prompt = LLM_SYSTEM_PROMPT
        self._prompt_lock = threading.Lock()

        try:
            from openai import OpenAI
        except ImportError:
            state.add_log("⚠️ openai 包未安装 — LLM 已禁用")
            return

        # ── 优先检查云端 API ──────────────────────────────────────────
        if LLM_API_KEY:
            try:
                self._client = OpenAI(
                    api_key=LLM_API_KEY,
                    base_url=LLM_API_BASE_URL,
                    timeout=LLM_TIMEOUT,
                )
                self._available = True
                self._provider = "cloud"
                state.add_log(f"🧠 LLM: 云端 API ({LLM_API_MODEL})")
            except Exception as e:
                state.add_log(f"⚠️ 云端 LLM 初始化失败: {e}")
        else:
            # ── 回退到本地 Ollama ─────────────────────────────────────
            try:
                self._client = OpenAI(
                    api_key="ollama",
                    base_url=OLLAMA_BASE_URL,
                    timeout=LLM_TIMEOUT,
                )
                self._available = True
                self._provider = "ollama"
                state.add_log(f"🖥️ LLM: Ollama ({OLLAMA_MODEL})")
            except Exception as e:
                state.add_log(f"⚠️ Ollama 初始化失败: {e}")

        # ── 初始化视觉模型 ────────────────────────────────────────────
        if self._available:
            # 优先云端视觉 API
            if VISION_API_KEY:
                try:
                    self._vision_client = OpenAI(
                        api_key=VISION_API_KEY,
                        base_url=VISION_API_BASE_URL,
                        timeout=LLM_TIMEOUT,
                    )
                    self._vision_model = VISION_API_MODEL
                    self._vision_provider = "cloud"
                    state.add_log(f"👁️ 视觉: 云端 API ({VISION_API_MODEL})")
                except Exception:
                    pass
            elif OLLAMA_VISION_MODEL:
                self._vision_model = OLLAMA_VISION_MODEL
                self._vision_provider = "ollama"
                state.add_log(f"👁️ 视觉: Ollama ({OLLAMA_VISION_MODEL})")

    @property
    def is_available(self) -> bool:
        return self._available

    @property
    def system_prompt(self) -> str:
        with self._prompt_lock:
            return self._system_prompt

    def update_system_prompt(self, new_prompt: str) -> bool:
        with self._prompt_lock:
            if new_prompt and new_prompt != self._system_prompt:
                self._system_prompt = new_prompt
                return True
            return False

    def parse(self, text: str, visible_colors: list[str]) -> Optional[dict]:
        if not self._available:
            return None
        model = LLM_API_MODEL if self._provider == "cloud" else OLLAMA_MODEL
        user_msg = (
            f"命令: \"{text}\"\n"
            f"可见物体: {visible_colors if visible_colors else ['无']}\n"
            f"请返回 json 格式的意图解析结果."
        )
        try:
            resp = self._client.chat.completions.create(
                model=model,
                max_tokens=LLM_MAX_TOKENS,
                timeout=LLM_TIMEOUT,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": self._system_prompt},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.1,
            )
            raw = resp.choices[0].message.content.strip()
            log.info(f"LLM 原始回复: {raw[:200]}")
            return self._extract_json(raw)
        except Exception as e:
            log.error(f"LLM 调用失败 (text='{text[:30]}'): {e}")
            return None

    def parse_with_image(self, text: str, frame, visible_colors: list[str],
                         spatial_context: str = "") -> Optional[dict]:
        """发送图片 + 文本给视觉模型进行多模态推理."""
        import base64
        import cv2

        if not self._available or not self._vision_model:
            return None

        # 编码图片
        _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        img_b64 = base64.b64encode(jpeg).decode("utf-8")

        ctx = ""
        if spatial_context:
            ctx = f"\n空间记忆:\n{spatial_context}"

        user_content = [
            {"type": "text", "text": (
                f"用户说: \"{text}\"\n"
                f"可见物体颜色: {visible_colors if visible_colors else ['无']}{ctx}\n"
                f"请结合图片理解用户意图，输出 JSON。"
            )},
            {"type": "image_url", "image_url": {
                "url": f"data:image/jpeg;base64,{img_b64}",
                "detail": "low",
            }},
        ]

        client = getattr(self, '_vision_client', self._client)
        model = self._vision_model if self._vision_provider == "cloud" else self._vision_model
        try:
            resp = client.chat.completions.create(
                model=model,
                max_tokens=LLM_MAX_TOKENS,
                timeout=LLM_TIMEOUT,
                messages=[
                    {"role": "system", "content": self._system_prompt},
                    {"role": "user", "content": user_content},
                ],
                temperature=VISION_API_TEMPERATURE if self._vision_provider == "cloud" else 0.1,
            )
            raw = resp.choices[0].message.content.strip()
            log.info(f"视觉 LLM 回复: {raw[:200]}")
            return self._extract_json(raw)
        except Exception as e:
            log.error(f"视觉 LLM 调用失败: {e}")
            return None

    @staticmethod
    def _extract_json(raw: str) -> Optional[dict]:
        """从 LLM 回复中提取 JSON."""
        raw = raw.strip()
        if not raw:
            return None

        result = None
        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            pass

        if result is None:
            m = _RE_JSON_BLOCK.search(raw)
            if m:
                try:
                    result = json.loads(m.group(1))
                except json.JSONDecodeError:
                    pass

        if result is None:
            m = _RE_JSON_INLINE.search(raw)
            if m:
                try:
                    result = json.loads(m.group(0))
                except json.JSONDecodeError:
                    pass

        if result is None:
            return None

        # 字段名规范化
        action = result.get("action", "")
        if not action:
            intent = result.get("intent", "").lower()
            if intent in ("greet", "greeting", "hello", "hi"):
                result["action"] = "gesture"
                result.setdefault("gesture", "greet")
            elif intent in ("pick", "pick_and_place", "grab", "fetch", "select", "get"):
                result["action"] = "pick_and_place"
            elif intent in ("wave", "挥手"):
                result["action"] = "gesture"
                result.setdefault("gesture", "wave")
            elif intent in ("home", "reset", "回零"):
                result["action"] = "home"
            elif intent in ("stop", "halt", "暂停"):
                result["action"] = "say"
            else:
                result["action"] = "say"

        return result

    # ══════════════════════════════════════════════════════════════════════
    # 硬编码降级规则 (JSON 关键词库不可用时)
    # 注意: 此列表不含 talent_show/play_praise (开源版不包含视频功能)
    # ══════════════════════════════════════════════════════════════════════

    KEYWORD_RULES: list[tuple[list[str], dict]] = [
        # ── 1. 控制 ────────────────────────────────────────────────
        (["停止", "暂停", "停", "停下", "别动", "stop", "halt", "住手"],
         {"action": "say", "message": "好的，已停止。", "confidence": 1.0}),
        (["回零", "归位", "复位", "初始位置", "回家", "回去", "home", "reset", "恢复", "归零", "回到原点", "回原位"],
         {"action": "home", "message": "收到，正在回到初始位置。", "confidence": 1.0}),
        # ── 2. 夹爪 ────────────────────────────────────────────────
        (["张开", "打开", "松手", "放手", "松开", "open", "张开夹爪", "松爪"],
         {"action": "claw_open", "message": "好的，张开夹爪。", "confidence": 0.95}),
        (["闭合", "抓住", "抓紧", "夹紧", "握住", "握紧", "合上", "close", "grip", "闭合夹爪", "夹住", "抓一下"],
         {"action": "claw_close", "message": "好的，夹爪已闭合。", "confidence": 0.95}),
        # ── 3. 社交 ────────────────────────────────────────────────
        (["你好", "您好", "嗨", "hello", "hi", "嘿", "哈喽", "在吗", "在不在", "早上好", "晚上好", "下午好", "早啊", "好久不见", "来了"],
         {"action": "gesture", "gesture": "greet", "message": "你好呀，我是机械臂小助手，有什么可以帮你的？", "confidence": 0.95}),
        (["谢谢", "感谢", "多谢", "thanks", "thank", "辛苦了", "麻烦你了", "谢了"],
         {"action": "say", "message": "不客气，随时为你服务~", "confidence": 0.95}),
        (["再见", "拜拜", "bye", "回头见", "走了", "下次见", "晚安", "拜"],
         {"action": "gesture", "gesture": "wave", "message": "再见，下次见~", "confidence": 0.95}),
        (["你叫什么", "你是谁", "你的名字", "自我介绍", "介绍一下自己", "你是什么"],
         {"action": "say", "message": "我是 MeArm 桌面机械臂，叫我小臂就好~", "confidence": 0.95}),
        (["你怎么样", "你好吗", "how are you", "状态怎么样"],
         {"action": "say", "message": "我很好，随时待命！", "confidence": 0.9}),
        # ── 4. 手势 ────────────────────────────────────────────────
        (["挥手", "wave", "摇手", "摇摆", "摆摆手"],
         {"action": "gesture", "gesture": "wave", "message": "收到，挥手~", "confidence": 0.9}),
        (["点头", "nod", "鞠躬", "点点头"],
         {"action": "gesture", "gesture": "nod", "message": "好的，点头~", "confidence": 0.9}),
        (["指左边", "指左", "指向左边", "向左指", "左边", "point left"],
         {"action": "gesture", "gesture": "point_left", "message": "好的，指向左边。", "confidence": 0.9}),
        (["指右边", "指右", "指向右边", "向右指", "右边", "point right"],
         {"action": "gesture", "gesture": "point_right", "message": "好的，指向右边。", "confidence": 0.9}),
        (["指前面", "指前", "指向前面", "向前指", "前面", "指向", "指着", "point", "那边", "那个方向", "指点"],
         {"action": "gesture", "gesture": "point", "message": "好的，指向那边~", "confidence": 0.9}),
        (["问候", "greet", "打招呼", "问好"],
         {"action": "gesture", "gesture": "greet", "message": "你好呀~", "confidence": 0.9}),
        # ── 5. 关节运动 ────────────────────────────────────────────
        (["左转", "转左", "向左转", "逆时针", "turn left", "转到左边", "往左"],
         {"action": "move_joint", "joint": "base", "direction": -1, "message": "好的，向左转。", "confidence": 0.9}),
        (["右转", "转右", "向右转", "顺时针", "turn right", "转到右边", "往右"],
         {"action": "move_joint", "joint": "base", "direction": 1, "message": "好的，向右转。", "confidence": 0.9}),
        (["转回来", "回中", "转正", "居中", "看前面", "朝前"],
         {"action": "home", "message": "好的，转回来了。", "confidence": 0.85}),
        (["抬臂", "举高", "抬高", "抬起来", "raise", "往上抬", "举起来", "升高", "高一点", "再高点", "往上"],
         {"action": "move_joint", "joint": "lift", "direction": 1, "message": "好的，抬高手臂。", "confidence": 0.9}),
        (["放低", "降低", "放下来", "lower", "往下放", "低一点", "再低点", "降下来", "放下手臂", "往下", "下去"],
         {"action": "move_joint", "joint": "lift", "direction": -1, "message": "好的，放低手臂。", "confidence": 0.9}),
        (["前伸", "伸出去", "往前伸", "extend", "伸长", "伸出去点", "往前", "伸远点", "够过去"],
         {"action": "move_joint", "joint": "elbow", "direction": 1, "message": "好的，向前伸展。", "confidence": 0.9}),
        (["后缩", "收回来", "往后缩", "retract", "缩回来", "收回去", "往后", "缩一点", "收一下", "收回", "收缩", "拉回来"],
         {"action": "move_joint", "joint": "elbow", "direction": -1, "message": "好的，收缩回来。", "confidence": 0.9}),
        # ── 6. 速度控制 ────────────────────────────────────────────
        (["快点", "加速", "快一点", "faster", "速度快点", "迅速"],
         {"action": "say", "message": "好的，加快速度。", "confidence": 0.85}),
        (["慢点", "减速", "慢一点", "slower", "速度慢点", "慢些"],
         {"action": "say", "message": "好的，放慢速度。", "confidence": 0.85}),
        # ── 7. 状态查询 ────────────────────────────────────────────
        (["状态", "status", "怎么样", "在哪里", "什么位置", "当前位置", "报告状态"],
         {"action": "say", "message": "机械臂各关节正常，随时待命。", "confidence": 0.9}),
        (["看到什么", "有什么", "检测到什么", "能看到", "识别", "看到没", "看看", "瞅瞅"],
         {"action": "say", "message": "让我看看摄像头画面...", "confidence": 0.8}),
        # ── 8. 确认/否定 ───────────────────────────────────────────
        (["是的", "好的", "行", "可以", "没问题", "yes", "ok", "okay", "好呀", "好啊", "确认", "没错", "对的对的", "对的", "嗯嗯", "好嘞"],
         {"action": "say", "message": "好的，收到！", "confidence": 0.9}),
        (["不要", "别了", "算了", "取消", "no", "cancel", "不对", "不是", "不用了", "没事", "别这样", "不做了", "放弃"],
         {"action": "say", "message": "好的，已取消。", "confidence": 0.9}),
        # ── 9. 空间指示 ────────────────────────────────────────────
        (["这里", "这儿", "here", "这个位置"],
         {"action": "say", "message": "看到这里了。", "confidence": 0.75}),
        (["那里", "那儿", "那边", "there", "那个位置"],
         {"action": "say", "message": "看到那边了。", "confidence": 0.75}),
        # ── 10. 帮助 ───────────────────────────────────────────────
        (["帮助", "help", "怎么用", "有什么功能", "能做什么", "你会什么", "你的功能", "使用说明"],
         {"action": "say", "message": "你可以对我说：你好、抓红色、挥手、回零、握手等。也可以对着摄像头做手势哦~", "confidence": 0.9}),
        # ── 11. 握手 ───────────────────────────────────────────────
        (["握手", "握个手", "握握手", "handshake", "握一下", "握握手吧", "来握手", "握个爪"],
         {"action": "handshake", "message": "好的，握个手！", "confidence": 0.95}),
        # ── 12. 空间记忆引用 ───────────────────────────────────────
        (["刚才放的", "刚刚放的", "上次放的", "刚放的那个", "刚才那个", "刚刚那个", "上次那个", "上一个", "刚放的东西", "刚刚放的东西", "刚才放的东西"],
         {"action": "_spatial_lookup_last_placed", "confidence": 1.0, "message": "让我回想一下刚刚放了什么..."}),
    ]

    @staticmethod
    def keyword_fallback(text: str, visible_colors: list[str],
                         spatial=None) -> Optional[dict]:
        """结构化关键词匹配 (O(1) 索引查找).

        优先级:
        0. JSON 关键词库 (热加载, 主规则源)
        1. 空间记忆引用
        2. 颜色 + 抓取动词 → pick_and_place
        3. 结构化规则表 (KEYWORD_RULES, 硬编码降级) — 使用索引加速
        4. 未知 → 返回 None, 交给上层 LLM

        Args:
            text: 用户输入文本
            visible_colors: 摄像头可见颜色列表
            spatial: 可选的空间记忆库 (SpatialMemory)
        """
        # ── 优先尝试 JSON 关键词库 (热加载, 用户可编辑) ──────────────────
        try:
            lib = get_library()
            if lib.is_loaded:
                result = lib.match(text, visible_colors, spatial=spatial)
                if result is not None:
                    return result
        except Exception:
            pass

        t = text.lower()

        # ── A. 颜色 + 抓取动词 (最高优先级) ──────────────────────────────
        for cn_color, en_color in LLMIntentParser.COLOR_MAP.items():
            if cn_color in t:
                if any(v in t for v in LLMIntentParser.PICK_VERBS):
                    col_name = COLOR_CN.get(en_color, en_color)
                    return {"action": "pick_and_place", "color": en_color,
                            "class_name": None,
                            "gesture": None,
                            "message": f"好的，正在抓取{col_name}物体。",
                            "confidence": 0.85}

        # ── A2. 物体名 + 抓取动词 ──────────────────────────────────────
        for cn_name, en_name in YOLO_CN_CLASS.items():
            if cn_name in t:
                if any(v in t for v in LLMIntentParser.PICK_VERBS):
                    return {"action": "pick_and_place", "color": None,
                            "class_name": en_name,
                            "gesture": None,
                            "message": f"好的，正在抓取{cn_name}。",
                            "confidence": 0.85}
                break

        # ── B. 使用索引查找结构化规则 (O(1) 替代 O(n) 遍历) ────────────
        LLMIntentParser._build_keyword_index()

        matched: list[tuple[int, float, dict]] = []
        seen_rules: set[int] = set()

        for kw, rule_list in LLMIntentParser._keyword_index.items():
            if kw in t:
                for rule_idx, intent in rule_list:
                    if rule_idx in seen_rules:
                        continue
                    seen_rules.add(rule_idx)
                    matched.append((len(kw), intent.get("confidence", 0.5), intent))

        if matched:
            matched.sort(key=lambda x: (x[0], x[1]), reverse=True)
            intent = dict(matched[0][2])
            intent.setdefault("color", None)
            intent.setdefault("class_name", None)
            intent.setdefault("gesture", None)

            # 处理空间记忆查找动作
            if intent.get("action", "").startswith("_spatial_lookup"):
                if spatial is not None and not spatial.is_empty:
                    last = spatial.last_placed()
                    if last:
                        return {"action": "pick_and_place",
                                "color": last.color,
                                "class_name": last.class_name,
                                "gesture": None,
                                "message": f"好的，把刚刚放的{last.color or last.class_name or '东西'}拿回来。",
                                "confidence": 0.9}
                return {"action": "say",
                        "message": "抱歉，我记不得刚刚放了什么。",
                        "confidence": 0.5}

            return intent

        return None


# ══════════════════════════════════════════════════════════════════════════════
# llm_speak — LLM 润色语音回复
# ══════════════════════════════════════════════════════════════════════════════

def llm_speak(speaker: Speaker, llm: Optional[LLMIntentParser],
              context: str, fallback_message: str):
    """使用 LLM 生成自然语音回复, 如果 LLM 不可用则直接播报."""
    if llm and llm.is_available:
        try:
            resp = llm._client.chat.completions.create(
                model=LLM_API_MODEL if llm._provider == "cloud" else OLLAMA_MODEL,
                max_tokens=LLM_MAX_TOKENS,
                timeout=LLM_TIMEOUT,
                messages=[
                    {"role": "system", "content": LLMIntentParser.REPLY_SYSTEM_PROMPT},
                    {"role": "user", "content": f"场景: {context}\n请用口语化中文回复:"},
                ],
                temperature=0.7,
            )
            raw = resp.choices[0].message.content.strip()
            if raw and len(raw) <= 40:
                speaker.speak(raw)
                return
        except Exception:
            pass
    speaker.speak(fallback_message)
