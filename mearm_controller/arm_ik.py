"""
MeArm 工作台 — 逆运动学 (ArmIK)
===============================
从原 workbench_server.py 提取，不做任何修改。
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from .config import JOINT_LIMITS

log = logging.getLogger("workbench")


# ══════════════════════════════════════════════════════════════════════════════
# 逆运动学 (与原有代码一致)
# ══════════════════════════════════════════════════════════════════════════════

class ArmIK:
    L1 = 75.0
    L2 = 75.0
    L_TOTAL = L1 + L2
    H = 55.0
    ELBOW_SHOULDER_RATIO = 1.0
    ELBOW_OFFSET = 0

    @classmethod
    def solve(cls, x_mm: float, y_mm: float, z_mm: float) -> Optional[dict[str, int]]:
        base_deg = 90 - int(np.degrees(np.arctan2(x_mm, y_mm)))
        reach_2d = np.sqrt(x_mm ** 2 + y_mm ** 2)
        dz = z_mm - cls.H
        dist = np.sqrt(reach_2d ** 2 + dz ** 2)

        if dist > cls.L_TOTAL - 5:
            log.warning(f"目标 ({x_mm:.0f},{y_mm:.0f},{z_mm:.0f}) 超出最远可达范围")
            return None
        if dist < 20:
            log.warning("目标距底座过近")
            return None

        shoulder_angle_rad = np.arctan2(dz, reach_2d)
        shoulder_deg = 90 + int(np.degrees(shoulder_angle_rad))
        elbow_deg = 90 + int(cls.ELBOW_SHOULDER_RATIO * (shoulder_deg - 90) + cls.ELBOW_OFFSET)

        result = {"base": base_deg, "left": shoulder_deg, "right": elbow_deg}
        for joint, deg in result.items():
            lo, hi = JOINT_LIMITS[joint]
            result[joint] = int(np.clip(deg, lo, hi))
        return result
