"""
MeArm 工作台 — 本地关键词库 (KeywordLibrary)
============================================
独立 JSON 关键词库，支持热加载和快速匹配。

特性:
  - O(1) 关键词索引: 关键词 → 规则列表
  - 热加载: 文件 mtime 变化时自动重新加载 (每 5 秒检查)
  - 降级策略: JSON 缺失/损坏时回退硬编码规则
  - 优先级匹配: 颜色+动词 > 结构化规则 > 模糊匹配

用法:
  from .keyword_library import get_library
  lib = get_library()
  result = lib.match("你好", visible_colors=[])
"""

from __future__ import annotations

import json
import logging
import os
import time
import threading
from typing import Optional

from .config import COLOR_CN, YOLO_CN_CLASS, log as _log

_log = logging.getLogger("workbench.keyword_lib")


class KeywordLibrary:
    """本地关键词匹配引擎.

    每个实例跟踪 JSON 文件的 mtime，在 match() 调用时自动检测变更并热加载。
    线程安全: 加载/匹配使用 coarse-grained lock.
    """

    def __init__(self, json_path: str):
        self._path = json_path
        self._lock = threading.Lock()
        self._mtime: float = 0.0
        self._rules: list[dict] = []
        self._index: dict[str, list[dict]] = {}  # keyword → list of matching rules
        self._color_map: dict[str, str] = {}
        self._pick_verbs: list[str] = []
        self._spatial_take_back_keywords: list[str] = []
        self._loaded = False
        self._load()

    # ─── 加载 ──────────────────────────────────────────────────────────────

    def _load(self) -> bool:
        """加载 JSON 文件并构建关键词索引. 返回是否成功."""
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            _log.warning(f"关键词库文件未找到: {self._path}，使用内置规则")
            self._loaded = False
            return False
        except json.JSONDecodeError as e:
            _log.error(f"关键词库 JSON 解析失败: {e}，使用内置规则")
            self._loaded = False
            return False

        self._mtime = os.path.getmtime(self._path)
        self._rules = data.get("rules", [])
        self._color_map = data.get("color_map", {})
        self._pick_verbs = data.get("pick_verbs", [])
        self._spatial_take_back_keywords = data.get("spatial_take_back_keywords", [])

        # 构建反向索引: keyword → [rule, ...]
        self._index.clear()
        for rule in self._rules:
            for kw in rule.get("keywords", []):
                k = kw.lower()
                if k not in self._index:
                    self._index[k] = []
                self._index[k].append(rule)

        self._loaded = True
        _log.info(f"关键词库已加载: {len(self._rules)} 条规则, "
                  f"{len(self._index)} 个关键词索引, "
                  f"{len(self._color_map)} 个颜色映射")
        return True

    def _maybe_reload(self):
        """如果 JSON 文件已变更, 自动热加载."""
        try:
            new_mtime = os.path.getmtime(self._path)
            if new_mtime > self._mtime:
                with self._lock:
                    if new_mtime > self._mtime:  # double-check
                        _log.info("检测到关键词库更新，热加载中...")
                        self._load()
        except OSError:
            pass  # 文件暂时不可读, 下次再试

    # ─── 匹配 ──────────────────────────────────────────────────────────────

    def match(self, text: str, visible_colors: list[str],
              spatial=None) -> Optional[dict]:
        """匹配文本到意图.

        匹配优先级:
          1. 颜色 + 抓取动词 → pick_and_place
          2. 物体名 + 抓取动词 → pick_and_place
          3. 空间记忆引用 (take back / last placed)
          4. 结构化规则表 (按 JSON 中定义顺序)
          5. 未匹配 → None (交给上层 API)

        Args:
            text: 用户输入文本 (语音识别结果)
            visible_colors: HSV 检测到的颜色列表
            spatial: 可选的空间记忆库 (SpatialMemory)

        Returns:
            匹配到 → 意图 dict (action, message, confidence, ...)
            未匹配 → None
        """
        # 自动热加载
        self._maybe_reload()

        t = text.lower().strip()
        if not t:
            return None

        # ── A. 颜色 + 抓取动词 (最高优先级) ──────────────────────────────
        result = self._match_color_pick(t)
        if result:
            return result

        # ── B. 物体名 + 抓取动词 ──────────────────────────────────────────
        result = self._match_object_pick(t)
        if result:
            return result

        # ── C. 空间记忆引用 ──────────────────────────────────────────────
        if spatial is not None and not spatial.is_empty:
            result = self._match_spatial(t, spatial)
            if result:
                return result

        # ── D. 遍历结构化规则表 (关键词索引 O(1) 查找) ─────────────────
        # 先精确匹配 (规则的所有关键词中存在 t 中)
        matched_rules: list[tuple[dict, int]] = []  # (rule, match_score)

        for rule in self._rules:
            for kw in rule.get("keywords", []):
                if kw in t:
                    # match_score: 关键词长度 (越长约好, 避免"停"匹配"停止"中的"停")
                    matched_rules.append((rule, len(kw)))
                    break  # 一个规则只算一次

        if matched_rules:
            # 按 confidence 降序, 再按关键词长度降序
            matched_rules.sort(key=lambda x: (x[0].get("confidence", 0), x[1]),
                               reverse=True)
            best = matched_rules[0][0]

            result = {
                "action": best.get("action", "say"),
                "color": best.get("color"),
                "class_name": best.get("class_name"),
                "gesture": best.get("gesture"),
                "message": best.get("message", ""),
                "confidence": best.get("confidence", 0.5),
            }

            # 透传 move_joint 字段
            if result["action"] == "move_joint":
                result["joint"] = best.get("joint", "base")
                result["direction"] = int(best.get("direction", 1))

            # 处理空间记忆特殊动作
            if str(result.get("action", "")).startswith("_spatial"):
                return self._resolve_spatial_action(result, spatial)

            # pick_and_place 自动填颜色
            if result["action"] == "pick_and_place" and not result.get("color"):
                if visible_colors:
                    result["color"] = visible_colors[0]

            return result

        # ── E. 模糊回零匹配 (回零/归位等, 规则表中已覆盖但做兜底) ────
        if any(k in t for k in ["回零", "回原位", "回去", "归位"]):
            return {"action": "home", "color": None, "class_name": None,
                    "gesture": None,
                    "message": "收到，正在回到初始位置。", "confidence": 1.0}

        return None  # 未匹配, 交给上层 API

    def _match_color_pick(self, t: str) -> Optional[dict]:
        """颜色 + 抓取动词匹配."""
        for cn_color, en_color in self._color_map.items():
            if cn_color in t:
                if any(v in t for v in self._pick_verbs):
                    col_name = COLOR_CN.get(en_color, en_color)
                    return {"action": "pick_and_place", "color": en_color,
                            "class_name": None, "gesture": None,
                            "message": f"好的，正在抓取{col_name}物体。",
                            "confidence": 0.85}
        return None

    def _match_object_pick(self, t: str) -> Optional[dict]:
        """物体名 + 抓取动词匹配."""
        for cn_name, en_name in YOLO_CN_CLASS.items():
            if cn_name in t:
                if any(v in t for v in self._pick_verbs):
                    return {"action": "pick_and_place", "color": None,
                            "class_name": en_name, "gesture": None,
                            "message": f"好的，正在抓取{cn_name}。",
                            "confidence": 0.85}
                break
        return None

    def _match_spatial(self, t: str, spatial) -> Optional[dict]:
        """空间记忆引用匹配."""
        # "拿回来" 语义
        if any(k in t for k in self._spatial_take_back_keywords):
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

        # "刚才放的XX" 引用
        spatial_ref_keywords = ["刚才放的", "刚刚放的", "上次放的", "刚放的那个",
                                "刚才那个", "刚刚那个", "上次那个"]
        if any(k in t for k in spatial_ref_keywords):
            last = spatial.last_placed()
            if last:
                col_name = COLOR_CN.get(last.color, last.color)
                desc = last.object_desc or col_name
                if any(v in t for v in self._pick_verbs + ["拿", "取", "捡", "移动", "搬"]):
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
        return None

    def _resolve_spatial_action(self, result: dict, spatial) -> Optional[dict]:
        """解析 _spatial_lookup_last_placed 等特殊动作."""
        action = result.get("action", "")
        if action == "_spatial_lookup_last_placed":
            last = spatial.last_placed()
            if last:
                col_name = COLOR_CN.get(last.color, last.color)
                desc = last.object_desc or col_name
                return {"action": "say", "color": last.color,
                        "class_name": last.class_name or None,
                        "gesture": None,
                        "message": f"刚刚放的是{desc}，在({last.drop_x:.0f}, {last.drop_y:.0f})mm 处。",
                        "confidence": 0.9}
            else:
                return {"action": "say", "color": None, "class_name": None,
                        "gesture": None,
                        "message": "我还没放过任何东西呢。", "confidence": 0.9}
        return result

    # ─── 查询 ──────────────────────────────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def rule_count(self) -> int:
        return len(self._rules)

    def categories(self) -> list[str]:
        """返回所有类别名."""
        cats: set[str] = set()
        for rule in self._rules:
            cat = rule.get("category", "")
            if cat:
                cats.add(cat)
        return sorted(cats)

    def rules_by_category(self, category: str) -> list[dict]:
        """返回指定类别的所有规则."""
        return [r for r in self._rules if r.get("category") == category]


# ─── 全局单例 ─────────────────────────────────────────────────────────────

_library: Optional[KeywordLibrary] = None
_lock = threading.Lock()


def get_library(json_path: Optional[str] = None) -> KeywordLibrary:
    """获取全局关键词库单例.

    Args:
        json_path: JSON 文件路径, 默认使用与当前模块同目录的 keyword_library.json
    """
    global _library
    if _library is None:
        with _lock:
            if _library is None:  # double-check
                if json_path is None:
                    json_path = os.path.join(
                        os.path.dirname(os.path.abspath(__file__)),
                        "keyword_library.json"
                    )
                _library = KeywordLibrary(json_path)
    return _library


def reload_library():
    """强制重新加载关键词库 (用于手动刷新)."""
    global _library
    with _lock:
        if _library is not None:
            _library._load()
        else:
            get_library()
