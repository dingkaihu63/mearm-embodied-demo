"""
MeArm Learner — 交互记忆存储与检索
==================================
轻量级、离线、基于 JSON 文件的交互记忆库。
使用 TF-IDF 关键词匹配实现简单的语义检索。
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from collections import Counter
from dataclasses import dataclass, field, asdict
from typing import Optional

# ─── 默认存储路径 ─────────────────────────────────────────────────────────────
DEFAULT_MEMORY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "memory")
DEFAULT_MEMORY_FILE = "interactions.json"


@dataclass
class Interaction:
    """单条交互记录."""
    timestamp: float
    input_type: str          # "voice" | "text" | "gesture"
    raw_input: str           # 原始输入文本
    visible_colors: list[str]
    llm_intent: dict         # LLM 解析结果 (action, color, gesture, message, confidence)
    executed_action: str     # 实际执行的动作
    success: bool            # 执行是否成功
    user_feedback: str = "neutral"  # "positive" | "negative" | "neutral"
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Interaction":
        return cls(**d)


# ─── 简单中文分词 (基于字/词统计, 不依赖 jieba) ────────────────────────────────

# 常见中文停用词
_CN_STOP_WORDS = set(
    "的了吗呢吧啊呀嘛哈哦嗯喔哎喂嘿嗨呵哟哒啦捏么这那是会个有在和不"
    "了也都很要就看没说去走来过上对到把让被给从到以之为与及或但而"
    "因为所以如果虽然可以能够应该需要已经正在将要必须可能一定大概也许"
    "我你他她它我们你们他们它们自己这个那个哪个什么怎么怎样多少"
    "一个一下一点一些一种一次一回一遍一番一直一起一块一同"
)

# 中文标点
_CN_PUNCT = set("，。！？、；：「」『』（）【】《》…—～·　 ")


def _tokenize_cn(text: str) -> list[str]:
    """简单的中文分词: 2-gram + 单字过滤停用词."""
    # 去标点和空白
    cleaned = "".join(ch for ch in text if ch not in _CN_PUNCT and ch != " ")
    if not cleaned:
        return []

    tokens = []
    # 2-gram
    for i in range(len(cleaned) - 1):
        bigram = cleaned[i:i + 2]
        if bigram[0] not in _CN_STOP_WORDS or bigram[1] not in _CN_STOP_WORDS:
            tokens.append(bigram)
    # 单字 (非停用词)
    for ch in cleaned:
        if ch not in _CN_STOP_WORDS and ch.strip():
            tokens.append(ch)

    return tokens


def _tokenize_en(text: str) -> list[str]:
    """简单的英文分词."""
    return re.findall(r'[a-zA-Z]+', text.lower())


def tokenize(text: str) -> list[str]:
    """混合中英文分词."""
    return _tokenize_cn(text) + _tokenize_en(text)


# ─── TF-IDF 检索器 ────────────────────────────────────────────────────────────

class _TFIDFIndex:
    """内存中的 TF-IDF 索引, 用于快速语义检索."""

    def __init__(self):
        self._docs: list[list[str]] = []      # 每个文档的 token 列表
        self._doc_count = 0
        self._df: dict[str, int] = {}          # document frequency
        self._idf: dict[str, float] = {}       # 预计算的 IDF

    def add(self, tokens: list[str]):
        self._docs.append(tokens)
        self._doc_count += 1
        unique = set(tokens)
        for t in unique:
            self._df[t] = self._df.get(t, 0) + 1

    def _update_idf(self):
        for term, df in self._df.items():
            self._idf[term] = math.log((self._doc_count + 1) / (df + 1)) + 1.0

    def search(self, query_tokens: list[str], k: int = 5) -> list[tuple[int, float]]:
        """返回 (文档索引, 相似度分数) 的排序列表."""
        if self._doc_count == 0:
            return []

        self._update_idf()

        # 查询向量 (TF-IDF)
        q_counter = Counter(query_tokens)
        q_norm = math.sqrt(sum((q_counter[t] * self._idf.get(t, 1.0)) ** 2
                               for t in q_counter))
        if q_norm == 0:
            return []

        scores = []
        for idx, doc_tokens in enumerate(self._docs):
            d_counter = Counter(doc_tokens)
            d_norm = math.sqrt(sum((d_counter[t] * self._idf.get(t, 1.0)) ** 2
                                   for t in d_counter))
            if d_norm == 0:
                continue
            # 余弦相似度
            dot = sum(q_counter[t] * self._idf.get(t, 1.0) *
                      d_counter.get(t, 0) * self._idf.get(t, 1.0)
                      for t in q_counter if t in d_counter)
            score = dot / (q_norm * d_norm) if q_norm * d_norm > 0 else 0.0
            scores.append((idx, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]


# ══════════════════════════════════════════════════════════════════════════════
# InteractionMemory
# ══════════════════════════════════════════════════════════════════════════════

class InteractionMemory:
    """交互记忆库: 存储、检索、持久化.

    使用方式:
      mem = InteractionMemory()
      mem.add(Interaction(...))
      similar = mem.find_similar("抓红色")
      stats = mem.stats()
    """

    def __init__(self, storage_dir: Optional[str] = None):
        self._storage_dir = storage_dir or DEFAULT_MEMORY_DIR
        self._storage_path = os.path.join(self._storage_dir, DEFAULT_MEMORY_FILE)
        self._interactions: list[Interaction] = []
        self._index = _TFIDFIndex()
        self._modified = False

        # 自动加载已有记忆
        self.load()

    @property
    def count(self) -> int:
        return len(self._interactions)

    def add(self, interaction: Interaction):
        """追加一条交互记录并索引."""
        self._interactions.append(interaction)
        tokens = tokenize(interaction.raw_input)
        self._index.add(tokens)
        self._modified = True

    def recent(self, n: int = 20) -> list[Interaction]:
        """返回最近 N 条记录."""
        return self._interactions[-n:]

    def find_similar(self, text: str, k: int = 5) -> list[Interaction]:
        """语义检索: 找到与输入文本最相似的历史交互."""
        query_tokens = tokenize(text)
        results = self._index.search(query_tokens, k=k)
        return [self._interactions[idx] for idx, _ in results]

    def stats(self) -> dict:
        """统计概览."""
        if not self._interactions:
            return {"total": 0}

        total = len(self._interactions)
        successes = sum(1 for i in self._interactions if i.success)
        action_counts = Counter(i.executed_action for i in self._interactions)
        input_counts = Counter(i.input_type for i in self._interactions)
        color_counts: Counter = Counter()
        gesture_counts: Counter = Counter()

        for i in self._interactions:
            if i.llm_intent:
                intent = i.llm_intent
                if intent.get("color"):
                    color_counts[intent["color"]] += 1
                if intent.get("gesture"):
                    gesture_counts[intent["gesture"]] += 1

        # 常见命令 (聚合 raw_input)
        cmd_counter = Counter(i.raw_input for i in self._interactions)

        return {
            "total": total,
            "success_rate": round(successes / total * 100, 1) if total > 0 else 0,
            "top_actions": action_counts.most_common(5),
            "top_inputs": input_counts.most_common(3),
            "top_colors": color_counts.most_common(4),
            "top_gestures": gesture_counts.most_common(4),
            "top_commands": cmd_counter.most_common(5),
        }

    def save(self):
        """持久化到 JSON 文件."""
        os.makedirs(self._storage_dir, exist_ok=True)
        data = {
            "version": 1,
            "updated": time.time(),
            "interactions": [i.to_dict() for i in self._interactions],
        }
        tmp_path = self._storage_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, self._storage_path)
        self._modified = False

    def load(self):
        """从 JSON 文件加载."""
        if not os.path.exists(self._storage_path):
            return

        try:
            with open(self._storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self._interactions = []
            self._index = _TFIDFIndex()
            for item in data.get("interactions", []):
                interaction = Interaction.from_dict(item)
                self._interactions.append(interaction)
                self._index.add(tokenize(interaction.raw_input))

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            import logging
            logging.getLogger("mearm_learner").warning(f"加载交互记忆失败: {e}")
            self._interactions = []
            self._index = _TFIDFIndex()

    def auto_save(self):
        """如果已修改则自动保存."""
        if self._modified:
            self.save()

    def clear(self):
        """清空记忆 (谨慎使用)."""
        self._interactions = []
        self._index = _TFIDFIndex()
        self._modified = True
        self.save()
