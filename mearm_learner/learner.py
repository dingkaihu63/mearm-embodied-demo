"""
MeArm Learner — 技能学习器
==========================
从交互历史中发现用户习惯、偏好和模式，
生成个性化用户画像供 PromptAdapter 使用。
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Optional

from .memory import InteractionMemory

log = logging.getLogger("mearm_learner")

# 最少交互次数才启用学习
MIN_INTERACTIONS_FOR_LEARNING = 10


class SkillLearner:
    """从交互历史中学习用户行为模式.

    使用方式:
      learner = SkillLearner(memory)
      profile = learner.get_user_profile()
      hints = learner.get_personalized_hints()
    """

    def __init__(self, memory: InteractionMemory):
        self._memory = memory

    @property
    def ready(self) -> bool:
        """是否有足够的数据进行学习."""
        return self._memory.count >= MIN_INTERACTIONS_FOR_LEARNING

    def detect_preferences(self) -> dict:
        """检测用户偏好."""
        if not self.ready:
            return {}

        interactions = self._memory.recent(100)

        color_prefs: Counter = Counter()
        gesture_prefs: Counter = Counter()
        action_prefs: Counter = Counter()
        success_count = 0
        fail_count = 0

        for i in interactions:
            if i.success:
                success_count += 1
                intent = i.llm_intent
                if isinstance(intent, dict):
                    if intent.get("color"):
                        color_prefs[intent["color"]] += 1
                    if intent.get("gesture"):
                        gesture_prefs[intent["gesture"]] += 1
                    if intent.get("action"):
                        action_prefs[intent["action"]] += 1
            else:
                fail_count += 1

        return {
            "favorite_color": color_prefs.most_common(1)[0] if color_prefs else None,
            "top_colors": color_prefs.most_common(3),
            "favorite_gesture": gesture_prefs.most_common(1)[0] if gesture_prefs else None,
            "top_gestures": gesture_prefs.most_common(3),
            "top_actions": action_prefs.most_common(3),
            "success_count": success_count,
            "fail_count": fail_count,
            "total": len(interactions),
        }

    def learn_patterns(self) -> dict:
        """从历史中发现高级行为模式."""
        if not self.ready:
            return {"patterns_found": False, "reason": "insufficient_data"}

        prefs = self.detect_preferences()
        interactions = self._memory.recent(50)

        patterns = {
            "patterns_found": True,
            "preferences": prefs,
        }

        # 1. 检测常用命令类型
        cmd_types = Counter(i.input_type for i in interactions)
        patterns["primary_input"] = cmd_types.most_common(1)[0][0] if cmd_types else "unknown"

        # 2. 检测色彩 + 动作组合
        color_action_pairs = []
        for i in interactions:
            intent = i.llm_intent
            if isinstance(intent, dict) and intent.get("color") and intent.get("action") == "pick_and_place":
                color_action_pairs.append(intent["color"])
        color_freq = Counter(color_action_pairs)
        patterns["pick_color_pattern"] = color_freq.most_common(3)

        # 3. 检测手势偏好
        gesture_seqs = []
        for i in interactions:
            intent = i.llm_intent
            if isinstance(intent, dict) and intent.get("gesture") and intent.get("action") == "gesture":
                gesture_seqs.append(intent["gesture"])
        gesture_freq = Counter(gesture_seqs)
        patterns["gesture_pattern"] = gesture_freq.most_common(3)

        # 4. 时间模式 (简单: 最近 N 条的时间跨度)
        if len(interactions) >= 5:
            first_ts = interactions[0].timestamp
            last_ts = interactions[-1].timestamp
            span_minutes = (last_ts - first_ts) / 60.0 if last_ts > first_ts else 0
            if span_minutes > 0:
                patterns["session_duration_minutes"] = round(span_minutes, 1)
                patterns["interaction_rate_per_minute"] = round(len(interactions) / span_minutes, 2)

        return patterns

    def get_user_profile(self) -> dict:
        """生成简洁的用户画像摘要."""
        patterns = self.learn_patterns()
        if not patterns.get("patterns_found"):
            return {
                "available": False,
                "message": "数据不足, 需要至少 10 条交互记录",
                "current_count": self._memory.count,
            }

        prefs = patterns.get("preferences", {})
        lines = []

        if prefs.get("favorite_color"):
            color_name, count = prefs["favorite_color"]
            lines.append(f"- 最常抓取: {color_name} 物体 ({count} 次)")

        if prefs.get("favorite_gesture"):
            gesture_name, count = prefs["favorite_gesture"]
            lines.append(f"- 偏好手势: {gesture_name} ({count} 次)")

        if prefs.get("top_colors"):
            top_colors = [f"{c}({n}次)" for c, n in prefs["top_colors"][:2]]
            lines.append(f"- 关注颜色: {', '.join(top_colors)}")

        if patterns.get("primary_input"):
            input_map = {"voice": "语音", "text": "文本", "gesture": "手势"}
            lines.append(f"- 主要输入方式: {input_map.get(patterns['primary_input'], patterns['primary_input'])}")

        success_rate = round(prefs.get("success_count", 0) / max(prefs.get("total", 1), 1) * 100, 1)
        lines.append(f"- 操作成功率: {success_rate}%")

        return {
            "available": True,
            "total_interactions": self._memory.count,
            "summary_lines": lines,
            "full_profile": patterns,
        }

    def get_personalized_hints(self) -> str:
        """返回可注入 LLM prompt 的个性化提示文本."""
        profile = self.get_user_profile()
        if not profile.get("available"):
            return ""

        lines = ["\n[自学习用户画像]", "该用户历史行为:"]
        lines.extend(profile.get("summary_lines", []))

        # 附加行为建议
        prefs = profile.get("full_profile", {}).get("preferences", {})
        if prefs.get("favorite_color"):
            color_name, _ = prefs["favorite_color"]
            lines.append(f"- 当前可见 {color_name} 物体时优先考虑抓取动作")

        if prefs.get("favorite_gesture"):
            gesture_name, _ = prefs["favorite_gesture"]
            lines.append(f"- 社交互动时优先考虑 {gesture_name} 手势")

        return "\n".join(lines)
