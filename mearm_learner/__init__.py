"""
MeArm Learner — 本地自学习库
============================
结合语音识别、手势识别和大模型的交互历史，
实现渐进式个性化学习和 LLM 提示词增强。

模块:
  memory.py   — InteractionMemory 交互记忆存储与检索
  learner.py  — SkillLearner 从历史中学习用户模式
  adapter.py  — PromptAdapter 动态调整 LLM 提示词
"""

from .memory import InteractionMemory, Interaction
from .learner import SkillLearner
from .adapter import PromptAdapter
