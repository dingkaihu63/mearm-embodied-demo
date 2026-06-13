"""
MeArm Learner — LLM 提示词适配器
================================
根据自学习结果动态增强 LLM system prompt，
使机械臂的响应越来越贴合用户习惯。
"""

from __future__ import annotations

import logging
from typing import Optional

from .memory import InteractionMemory
from .learner import SkillLearner

log = logging.getLogger("mearm_learner")


class PromptAdapter:
    """动态调整 LLM system prompt，注入用户画像.

    使用方式:
      adapter = PromptAdapter(memory, base_prompt=LLM_SYSTEM_PROMPT)
      enhanced_prompt = adapter.get_augmented_prompt()
      llm = LLMIntentParser(api_key, system_prompt=enhanced_prompt)
    """

    def __init__(self, memory: InteractionMemory,
                 base_prompt: Optional[str] = None):
        self._memory = memory
        self._learner = SkillLearner(memory)
        self._base_prompt = base_prompt or ""
        self._last_profile_hash: str = ""
        self._cached_prompt: Optional[str] = None

    @property
    def base_prompt(self) -> str:
        return self._base_prompt

    @base_prompt.setter
    def base_prompt(self, value: str):
        self._base_prompt = value
        self._cached_prompt = None  # 使缓存失效

    def get_augmented_prompt(self) -> str:
        """返回增强后的完整 system prompt.

        如果学习数据不足，返回原始 prompt 不变。
        如果用户画像有更新，重新生成增强 prompt（带缓存）。
        """
        if not self._learner.ready:
            return self._base_prompt

        # 检查画像是否变化（通过摘要 hash）
        hints = self._learner.get_personalized_hints()
        if not hints:
            return self._base_prompt

        profile_hash = str(hash(hints))
        if profile_hash == self._last_profile_hash and self._cached_prompt is not None:
            return self._cached_prompt

        # 生成增强 prompt
        augmented = self._base_prompt + "\n" + hints
        self._last_profile_hash = profile_hash
        self._cached_prompt = augmented

        log.info(f"LLM prompt 已增强 (交互数: {self._memory.count})")
        return augmented

    @staticmethod
    def build_example_context(memory: InteractionMemory, limit: int = 3) -> str:
        """构建少样本上下文 (从历史中选取成功案例).

        将最近成功的交互作为 few-shot 示例注入 prompt，
        帮助 LLM 更好理解用户的表达风格。
        """
        recent = memory.recent(50)
        successes = [i for i in recent if i.success and i.llm_intent]

        if len(successes) < 2:
            return ""

        examples = successes[-limit:]
        lines = ["\n[历史成功示例]"]

        for ex in examples:
            intent = ex.llm_intent
            if isinstance(intent, dict):
                lines.append(
                    f"用户说: \"{ex.raw_input}\" → "
                    f"动作: {intent.get('action')}, "
                    f"颜色: {intent.get('color')}, "
                    f"手势: {intent.get('gesture')}"
                )

        return "\n".join(lines)
