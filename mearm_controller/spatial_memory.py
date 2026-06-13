"""
MeArm 工作台 — 空间记忆库 (SpatialMemory)
==========================================
记录每次 pick-and-place 操作的物理空间信息，
支持上下文查询（"刚刚放的东西"、"红色的在哪"等），
实现机械臂对物理世界状态的持续记忆。

每次操作后自动更新，LLM 调用前注入上下文。
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

from .config import COLOR_CN, YOLO_CLASS_CN

# ─── 存储路径 ─────────────────────────────────────────────────────────────────
DEFAULT_MEMORY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "memory")
SPATIAL_MEMORY_FILE = "spatial_memory.json"


@dataclass
class PlacementRecord:
    """单条空间记忆记录 — 一次物体移动的完整信息."""

    timestamp: float                    # Unix 时间戳
    time_str: str = ""                  # 人类可读时间 (自动填充)

    # ── 物体描述 ──────────────────────────────────────────────────────────
    object_desc: str = ""               # 人类可读描述 (如 "红色方块")
    color: str = ""                     # 颜色 (英文, "red"/"blue" 等)
    class_name: str = ""                # 物体类名 (英文, "cup"/"block" 等)
    class_cn: str = ""                  # 物体类名 (中文)

    # ── 空间坐标 (mm) ─────────────────────────────────────────────────────
    pick_x: float = 0.0                 # 从哪里抓取
    pick_y: float = 0.0
    pick_z: float = 0.0
    drop_x: float = 0.0                 # 放到哪里
    drop_y: float = 0.0
    drop_z: float = 0.0

    # ── 操作元信息 ────────────────────────────────────────────────────────
    action_type: str = "pick_and_place" # "pick_and_place" | "move" | "drop"
    success: bool = True
    input_type: str = ""                # "voice" | "gesture" | "text"
    raw_input: str = ""                 # 原始用户输入

    def to_dict(self) -> dict:
        d = asdict(self)
        # 移除空的可选字段以减小文件体积
        return {k: v for k, v in d.items() if v != "" and v != 0.0 or k in ("timestamp", "success")}

    @classmethod
    def from_dict(cls, d: dict) -> "PlacementRecord":
        # 兼容旧版本缺少的字段
        defaults = {
            "time_str": "", "object_desc": "", "color": "", "class_name": "",
            "class_cn": "", "pick_x": 0.0, "pick_y": 0.0, "pick_z": 0.0,
            "drop_x": 0.0, "drop_y": 0.0, "drop_z": 0.0,
            "action_type": "pick_and_place", "success": True,
            "input_type": "", "raw_input": "",
        }
        for k, v in defaults.items():
            d.setdefault(k, v)
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


class SpatialMemory:
    """空间记忆库 — 线程安全、自动持久化.

    使用方式:
      sp = SpatialMemory()
      sp.record_placement(color="red", class_name="cup", pick_x=50, pick_y=100,
                          drop_x=0, drop_y=120, input_type="voice", raw_input="抓红色杯子")
      ctx = sp.get_context_text()        # 获取供 LLM 使用的上下文
      last = sp.last_placed()            # 查询最近放置的物体
      found = sp.find_by_color("red")    # 查询红色物体最后位置
    """

    MAX_RECORDS = 100  # 最多保留记录数

    def __init__(self, storage_dir: Optional[str] = None):
        self._storage_dir = storage_dir or DEFAULT_MEMORY_DIR
        self._storage_path = os.path.join(self._storage_dir, SPATIAL_MEMORY_FILE)
        self._lock = threading.Lock()
        self._records: list[PlacementRecord] = []
        self._modified = False

        self.load()

    # ─── 属性 ──────────────────────────────────────────────────────────────
    @property
    def count(self) -> int:
        with self._lock:
            return len(self._records)

    @property
    def is_empty(self) -> bool:
        return self.count == 0

    # ─── 记录操作 ──────────────────────────────────────────────────────────
    def record_placement(
        self,
        color: str = "",
        class_name: str = "",
        pick_x: float = 0.0,
        pick_y: float = 0.0,
        pick_z: float = 0.0,
        drop_x: float = 0.0,
        drop_y: float = 0.0,
        drop_z: float = 0.0,
        success: bool = True,
        input_type: str = "",
        raw_input: str = "",
    ) -> PlacementRecord:
        """记录一次物体放置操作.

        Args:
            color: 物体颜色 (英文)
            class_name: 物体类名 (英文)
            pick_x, pick_y, pick_z: 抓取坐标 (mm)
            drop_x, drop_y, drop_z: 放置坐标 (mm)
            success: 操作是否成功
            input_type: 输入类型 ("voice"/"gesture"/"text")
            raw_input: 原始用户输入文本
        """
        now = time.time()
        time_str = time.strftime("%H:%M:%S", time.localtime(now))

        # 构建人类可读描述
        desc_parts = []
        if color:
            desc_parts.append(COLOR_CN.get(color, color))
        if class_name:
            cn = YOLO_CLASS_CN.get(class_name, class_name)
            desc_parts.append(cn)
        object_desc = "".join(desc_parts) if desc_parts else "未知物体"

        record = PlacementRecord(
            timestamp=now,
            time_str=time_str,
            object_desc=object_desc,
            color=color,
            class_name=class_name,
            class_cn=YOLO_CLASS_CN.get(class_name, class_name) if class_name else "",
            pick_x=pick_x, pick_y=pick_y, pick_z=pick_z,
            drop_x=drop_x, drop_y=drop_y, drop_z=drop_z,
            action_type="pick_and_place",
            success=success,
            input_type=input_type,
            raw_input=raw_input,
        )

        with self._lock:
            self._records.append(record)
            # 限制最大记录数
            if len(self._records) > self.MAX_RECORDS:
                self._records = self._records[-self.MAX_RECORDS:]
            self._modified = True

        self.auto_save()
        return record

    # ─── 查询接口 ──────────────────────────────────────────────────────────
    def last_placed(self) -> Optional[PlacementRecord]:
        """返回最近一次成功放置的记录."""
        with self._lock:
            for r in reversed(self._records):
                if r.success and r.action_type == "pick_and_place":
                    return r
        return None

    def last_action(self) -> Optional[PlacementRecord]:
        """返回最近一次操作记录 (无论类型)."""
        with self._lock:
            return self._records[-1] if self._records else None

    def recent_placements(self, n: int = 5) -> list[PlacementRecord]:
        """返回最近 N 次放置记录."""
        with self._lock:
            placements = [r for r in reversed(self._records)
                         if r.success and r.action_type == "pick_and_place"]
            return placements[:n]

    def find_by_color(self, color: str) -> Optional[PlacementRecord]:
        """查找指定颜色的物体最后被放到的位置."""
        with self._lock:
            for r in reversed(self._records):
                if r.color == color and r.success:
                    return r
        return None

    def find_by_class(self, class_name: str) -> Optional[PlacementRecord]:
        """查找指定类别的物体最后被放到的位置."""
        with self._lock:
            for r in reversed(self._records):
                if r.class_name == class_name and r.success:
                    return r
        return None

    def find_by_desc(self, desc: str) -> Optional[PlacementRecord]:
        """模糊查找: 匹配颜色或类别名称 (中文/英文)."""
        with self._lock:
            for r in reversed(self._records):
                if not r.success:
                    continue
                # 检查中文类名
                if r.class_cn and r.class_cn in desc:
                    return r
                # 检查中文颜色
                if r.color:
                    color_cn = COLOR_CN.get(r.color, "")
                    if color_cn and color_cn in desc:
                        return r
                # 检查物体描述
                if r.object_desc and r.object_desc in desc:
                    return r
        return None

    # ─── 上下文生成 (供 LLM) ───────────────────────────────────────────────
    def get_context_text(self, recent_n: int = 3) -> str:
        """生成空间记忆上下文文本，供注入 LLM prompt.

        格式:
          [空间记忆]
          - 12:30:15 把 红色方块 从 (50,100) 放到了 (0,120)
          - 12:30:45 把 蓝色杯子 从 (80,60) 放到了 (-30,120)
          - 最近放置: 蓝色杯子 @ (-30, 120)mm
        """
        with self._lock:
            records = [r for r in self._records if r.success][-recent_n:]

        if not records:
            return ""

        lines = ["[空间记忆 - 最近操作]"]
        for r in records:
            pick_str = f"({r.pick_x:.0f},{r.pick_y:.0f})"
            drop_str = f"({r.drop_x:.0f},{r.drop_y:.0f})"
            lines.append(
                f"- {r.time_str} 把 [{r.object_desc}] 从 {pick_str} 放到了 {drop_str}"
            )

        # 最近一次放置
        last = records[-1]
        lines.append(f"- 最近放置的物体: [{last.object_desc}] @ ({last.drop_x:.0f}, {last.drop_y:.0f})mm")

        # 当前已知物体位置汇总
        lines.append("- 当前工作台已知物体:")
        seen = {}
        for r in reversed(records):
            key = r.object_desc
            if key not in seen:
                seen[key] = r
        for desc, r in seen.items():
            lines.append(f"  [{desc}] @ ({r.drop_x:.0f}, {r.drop_y:.0f})mm")

        return "\n".join(lines)

    def get_context_for_prompt(self) -> str:
        """获取适合嵌入 LLM system prompt 的简短上下文."""
        last = self.last_placed()
        if not last:
            return ""

        lines = ["\n[工作台空间状态]"]
        lines.append(f"最近操作: {last.time_str} 将 {last.object_desc} "
                     f"放在了 ({last.drop_x:.0f}, {last.drop_y:.0f})mm 处")

        # 汇总所有已知物体
        with self._lock:
            seen = {}
            for r in reversed(self._records):
                if r.success and r.object_desc not in seen:
                    seen[r.object_desc] = r

        if seen:
            lines.append("当前工作台上的物体:")
            for desc, r in seen.items():
                lines.append(f"  - {desc} 在 ({r.drop_x:.0f}, {r.drop_y:.0f})mm")

        return "\n".join(lines)

    # ─── 持久化 ────────────────────────────────────────────────────────────
    def save(self):
        """持久化到 JSON 文件."""
        os.makedirs(self._storage_dir, exist_ok=True)
        with self._lock:
            data = {
                "version": 1,
                "updated": time.time(),
                "updated_str": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())),
                "total": len(self._records),
                "records": [r.to_dict() for r in self._records],
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

            with self._lock:
                self._records = []
                for item in data.get("records", []):
                    record = PlacementRecord.from_dict(item)
                    self._records.append(record)

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            import logging
            logging.getLogger("workbench").warning(f"加载空间记忆失败: {e}")
            self._records = []

    def auto_save(self):
        """如果已修改则自动保存."""
        if self._modified:
            self.save()

    def clear(self):
        """清空空间记忆."""
        with self._lock:
            self._records = []
            self._modified = True
        self.save()

    # ─── 统计 ──────────────────────────────────────────────────────────────
    def stats(self) -> dict:
        """空间记忆统计信息."""
        with self._lock:
            if not self._records:
                return {"total": 0, "is_empty": True}

            from collections import Counter
            color_counts = Counter(r.color for r in self._records if r.color)
            class_counts = Counter(r.class_name for r in self._records if r.class_name)

            return {
                "total": len(self._records),
                "is_empty": False,
                "last_time": self._records[-1].time_str,
                "last_object": self._records[-1].object_desc,
                "top_colors": color_counts.most_common(4),
                "top_objects": [(YOLO_CLASS_CN.get(c, c), n) for c, n in class_counts.most_common(4)],
            }


# ─── 全局单例 ─────────────────────────────────────────────────────────────────
spatial = SpatialMemory()
