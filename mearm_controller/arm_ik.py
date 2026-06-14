"""
MeArm 工作台 — 逆运动学 (ArmIK)
===============================
解耦平行四边形 (Pantograph) 逆运动学。

机械原理 (参见 左右臂控制原理.txt):
  - Left 舵机 (左侧): 连接主连杆, 控制 Y 轴 (前后伸缩)
    left=150 → 大臂竖直, Y≈0   (完全收缩)
    left= 90 → 大臂水平, Y≈中  (半伸出)
    left= 30 → 大臂前倾, Y≈max (完全伸出)

  - Right 舵机 (右侧): 连接辅助推拉杆, 控制 Z 轴 (上下升降)
    right=150 → 末端最高, Z≈max
    right= 30 → 末端最低, Z≈min

连杆效应: 平行四边形结构使夹爪在伸缩时保持水平姿态,
Y 轴和 Z 轴近似解耦 — left 管远近, right 管高低.

伺服角度约定:
  left  = 150 - (r / Y_MAX) * 120  (r = 水平距离, Y_MAX ≈ 130mm)
  right =  30 + (z / Z_SPAN) * 120 (z = 高度, Z_SPAN = Z_MAX - Z_MIN)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from .config import JOINT_LIMITS

log = logging.getLogger("workbench")


class ArmIK:
    # ── 解耦模型标定参数 ──────────────────────────────────────────────
    Y_MAX = 130.0     # left=30 时最大前伸距离 (mm)
    Z_MIN = 20.0      # right=30 时末端最低高度 (mm, 距台面)
    Z_MAX = 150.0     # right=150 时末端最高高度 (mm, 距台面)

    # ── 旧版遗留参数 (保留供标定模块兼容) ────────────────────────────
    L1 = 75.0         # deprecated: 解耦模型不再使用连杆长度
    L2 = 75.0
    L_TOTAL = L1 + L2
    H = 55.0
    ELBOW_SHOULDER_RATIO = 1.0
    ELBOW_OFFSET = 0

    @classmethod
    def solve(cls, x_mm: float, y_mm: float, z_mm: float) -> Optional[dict[str, int]]:
        """解耦 pantograph 逆运动学求解.

        Args:
            x_mm, y_mm: 目标水平坐标 (原点=底座中心, y 轴=正前方)
            z_mm: 目标距台面高度 (mm)

        Returns:
            {"base": int, "left": int, "right": int} 或 None (不可达)

        原理:
          平行四边形连杆将 Y(远近) 和 Z(高低) 解耦.
          Left 舵机 → Y 轴, Right 舵机 → Z 轴.
        """
        # 1. 底座旋转: 让臂平面朝向目标 (舵机已转 180°)
        r = np.sqrt(x_mm ** 2 + y_mm ** 2)   # 水平距离
        base_deg = 90 + int(np.degrees(np.arctan2(x_mm, y_mm)))

        # 2. 可达性检查
        if r > cls.Y_MAX:
            log.warning(f"目标 ({x_mm:.0f},{y_mm:.0f},{z_mm:.0f}) "
                        f"水平距离 {r:.0f}mm 超出最大前伸 {cls.Y_MAX:.0f}mm")
            return None
        if r < 5.0:
            log.warning("目标距底座过近")
            return None
        if not (cls.Z_MIN <= z_mm <= cls.Z_MAX):
            log.warning(f"目标高度 {z_mm:.0f}mm 超出范围 [{cls.Z_MIN:.0f}, {cls.Z_MAX:.0f}]")
            return None

        # 3. Left 舵机 → Y 轴 (前后伸缩)
        #    left=150 → r=0 (收缩), left=30 → r=Y_MAX (完全伸出)
        #    线性映射: r / Y_MAX ∈ [0, 1]
        left_deg = 150 - int(round(r * 120.0 / cls.Y_MAX))

        # 4. Right 舵机 → Z 轴 (上下升降)
        #    right=30 → z=Z_MIN (最低), right=150 → z=Z_MAX (最高)
        #    线性映射: (z - Z_MIN) / Z_SPAN ∈ [0, 1]
        z_span = cls.Z_MAX - cls.Z_MIN
        right_deg = 30 + int(round((z_mm - cls.Z_MIN) * 120.0 / z_span))

        # 5. 限位钳制
        result = {"base": base_deg, "left": left_deg, "right": right_deg}
        for joint, deg in list(result.items()):
            lo, hi = JOINT_LIMITS[joint]
            result[joint] = int(np.clip(deg, lo, hi))

        return result
