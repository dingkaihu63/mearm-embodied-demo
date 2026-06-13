"""
MeArm 工作台 — LLM 增强逆运动学
===============================
在解析 IK (arm_ik.py) 基础上，用 LLM 推理能力处理边缘情况：

1. solve_with_fallback()  — 目标超出范围时，LLM 推理可达替代坐标
2. plan_pick_sequence()   — LLM 规划抓取策略，替代硬编码序列
3. analyze_calibration()  — 分析历史执行偏差，LLM 建议参数修正

使用方式:
  from .ik_llm import IKLLMEnhancer
  enhancer = IKLLMEnhancer()
  result = enhancer.solve_with_fallback(x, y, z, visible_colors, llm)
  strategy = enhancer.plan_pick_sequence(target, visible_colors, llm)
"""

from __future__ import annotations

import json
import logging
from typing import Optional

import numpy as np

from .arm_ik import ArmIK
from .config import JOINT_LIMITS

log = logging.getLogger("workbench")

# ══════════════════════════════════════════════════════════════════════════════
# LLM 提示词模板
# ══════════════════════════════════════════════════════════════════════════════

FALLBACK_PROMPT = """\
你是 MeArm 机械臂的运动规划专家。机械臂参数:
- 底座高度 H=55mm (肩关节距台面)
- 摄像头在底座上方 240mm 处, 水平朝前拍摄 (非俯拍)
  画面中: 上方=远处, 下方=近处/台面, 物体 y 坐标越大越远
- 等效连杆总长 L1+L2=150mm (最大水平/垂直伸展约 145mm)
- 关节限位: base 30-150°, left 30-150°, right 30-150°, claw 10-73°
- 底座旋转中心在原点, Y 轴正方向为前方

用户想要到达一个目标坐标，但解析 IK 发现目标超出范围。
请推理出最接近的可达替代坐标 (closest_x, closest_y, closest_z)，
即在不改变方向的前提下，沿原方向缩短到可达范围内。

当前目标: ({x:.0f}, {y:.0f}, {z:.0f}) mm
当前关节角度: base={base}°, left={left}°, right={right}°

仅输出 JSON:
{{"reachable": true/false, "closest_x": 0.0, "closest_y": 0.0, "closest_z": 0.0,
  "strategy": "简短说明如何调整"}}
"""

PICK_PLAN_PROMPT = """\
你是 MeArm 机械臂的运动规划专家。你需要规划一个"抓取-放置"动作序列。

目标物体: {color}
世界坐标: ({x:.0f}, {y:.0f}, {z:.0f}) mm
可见物体: {visible_colors}
当前关节: base={base}°, left={left}°, right={right}°

机械臂参数与约束:
- 底座高度 55mm, 摄像头在底座上方 240mm, 水平朝前拍摄
  画面中上方像素=远处物体, 下方像素=近处/台面
- 物体 (x,y) 坐标由水平画面经单应性变换得到, z 需估算
- 等效连杆 L1+L2=150mm, 最大水平伸展 ~145mm
- 夹爪需从上方垂直接近 (approach_height 建议 30-80mm 高于物体)
- 抓取高度 (pick_height) 建议 15-25mm (小物体高度)
- 搬运高度 (carry_height) 建议 70-100mm
- 放置位置 (drop_x, drop_y) 建议在机械臂前方 (0, 80-130) 范围内
- 如果物体太远(>130mm)，建议"先推近再抓取"
- 如果物体在底座附近(<30mm)，建议用指尖拨动
- 摄像头分辨率 640x480, 水平朝向

请规划最优动作序列，仅输出 JSON:
{{
  "approach_height": 60, "pick_height": 20, "carry_height": 85,
  "drop_x": 0, "drop_y": 120,
  "pre_push": true/false,
  "push_direction": "forward/left/right/none",
  "grip_offset": 0,
  "notes": "简短说明策略理由"
}}
"""

CALIBRATION_PROMPT = """\
你是 MeArm 机械臂的标定专家。根据以下实际执行记录，
分析定位偏差模式，建议运动学参数修正。

实际执行记录 (目标坐标 → 关节角度):
{records}

当前运动学参数:
- L1=L2=75mm, H=55mm
- ELBOW_OFFSET=0 (肘部偏置角)

请分析偏差模式并输出修正建议，仅输出 JSON:
{{
  "elbow_offset_adjust": 0.0,
  "base_offset_adjust": 0.0,
  "confidence": 0.0,
  "notes": "分析说明"
}}
"""


# ══════════════════════════════════════════════════════════════════════════════
# 辅助: 水平摄像头 — 根据像素位置估算深度
# ══════════════════════════════════════════════════════════════════════════════

def estimate_depth_from_pixel_y(py: int, camera_height_mm: float = 240.0,
                                  table_vanish_y: int = 240) -> float:
    """水平摄像头: 根据物体在画面中的垂直像素位置估算其距底座的距离.

    摄像头在底座上方 240mm 水平朝前。画面中:
    - py 越小 (画面越上方) → 物体越远 (接近地平线)
    - py 越大 (画面越下方) → 物体越近 (靠近底座)
    - table_vanish_y: 台面在画面中消失线的 y 像素 (~画面中部)

    Args:
        py: 物体中心在画面中的 y 像素坐标
        camera_height_mm: 摄像头距台面高度
        table_vanish_y: 台面消失线 y 坐标 (默认 240=画面中部)

    Returns:
        估算的物体距底座深度 (mm)
    """
    # 远距离: py 接近消失线 → 深度大
    # 近距离: py 接近画面底部 → 深度小
    frame_bottom = 480.0
    if py >= frame_bottom - 10:
        return 20.0  # 画面最底部, 就在底座旁边
    if py <= table_vanish_y:
        # 在消失线或以上 → 远处 (100-300mm)
        ratio = (table_vanish_y - py) / table_vanish_y
        return round(100.0 + ratio * 200.0, 1)
    else:
        # 消失线以下 → 近处 (20-100mm)
        ratio = (py - table_vanish_y) / (frame_bottom - table_vanish_y)
        return round(100.0 - ratio * 80.0, 1)


# ══════════════════════════════════════════════════════════════════════════════
# IKLLMEnhancer
# ══════════════════════════════════════════════════════════════════════════════

class IKLLMEnhancer:
    """LLM 增强的逆运动学求解器.

    包装原有的 ArmIK 解析解，在边缘情况下用 LLM 推理补救。
    """

    @staticmethod
    def solve_with_fallback(
        x_mm: float, y_mm: float, z_mm: float,
        visible_colors: list[str],
        llm,  # LLMIntentParser instance
        current_joints: Optional[dict] = None,
    ) -> Optional[dict[str, int]]:
        """解析 IK + LLM 降级策略.

        1. 先尝试解析 IK
        2. 如果失败，用 LLM 推理可达替代坐标
        3. 用替代坐标重新求解
        4. 如果仍失败，用 LLM 最后一次尝试
        """
        # 第一层: 解析解
        result = ArmIK.solve(x_mm, y_mm, z_mm)
        if result is not None:
            log.debug(f"IK 解析解: {result}")
            return result

        # 超出范围 — 记录
        dist = np.sqrt(x_mm**2 + y_mm**2 + (z_mm - ArmIK.H)**2)
        log.info(f"目标 ({x_mm:.0f},{y_mm:.0f},{z_mm:.0f}) 超出范围 (dist={dist:.0f}mm, "
                 f"max={ArmIK.L_TOTAL-5:.0f}mm) — 尝试 LLM fallback")

        if llm is None or not getattr(llm, 'is_available', False):
            return None

        # 第二层: LLM 推理替代坐标
        joints = current_joints or {"base": 90, "left": 90, "right": 90}
        prompt = FALLBACK_PROMPT.format(
            x=x_mm, y=y_mm, z=z_mm,
            base=joints.get("base", 90),
            left=joints.get("left", 90),
            right=joints.get("right", 90),
        )

        try:
            fallback = llm._client.chat.completions.create(
                model=getattr(llm, '_model', 'qwen2.5:7b'),
                max_tokens=200,
                timeout=8.0,
                messages=[
                    {"role": "system", "content": "仅输出 JSON，不要其他内容。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
            )
            raw = fallback.choices[0].message.content.strip()
            plan = json.loads(raw)

            if plan.get("reachable", True):
                cx = plan.get("closest_x", x_mm * 0.85)
                cy = plan.get("closest_y", y_mm * 0.85)
                cz = plan.get("closest_z", z_mm * 0.85)
                log.info(f"LLM 建议替代坐标: ({cx:.0f}, {cy:.0f}, {cz:.0f}) — {plan.get('strategy', '')}")

                result = ArmIK.solve(cx, cy, cz)
                if result is not None:
                    log.info(f"LLM fallback 成功: {result}")
                    return result

        except Exception as e:
            log.warning(f"LLM fallback 失败: {e}")

        # 第三层: 沿方向缩回 15%
        scaled_x = x_mm * 0.85
        scaled_y = y_mm * 0.85
        scaled_z = z_mm * 0.85
        log.info(f"最终 fallback: 缩回 15% → ({scaled_x:.0f}, {scaled_y:.0f}, {scaled_z:.0f})")
        return ArmIK.solve(scaled_x, scaled_y, scaled_z)

    @staticmethod
    def plan_pick_sequence(
        target_x: float, target_y: float, target_z: float,
        color: str,
        visible_colors: list[str],
        llm,  # LLMIntentParser instance
        current_joints: Optional[dict] = None,
    ) -> dict:
        """LLM 规划抓取策略，返回可执行的参数.

        相比硬编码序列的优势:
        - 远处物体 → 先推近
        - 近处物体 → 指尖拨动
        - 根据物体位置调整放置点
        """
        joints = current_joints or {"base": 90, "left": 90, "right": 90}

        # 默认策略 (不依赖 LLM 也能工作)
        default_strategy = {
            "approach_height": 60.0,
            "pick_height": 20.0,
            "carry_height": 85.0,
            "drop_x": 0.0,
            "drop_y": 120.0,
            "pre_push": False,
            "push_direction": "none",
            "grip_offset": 0,
            "notes": "默认策略 (LLM 不可用)",
        }

        if llm is None or not getattr(llm, 'is_available', False):
            return default_strategy

        prompt = PICK_PLAN_PROMPT.format(
            color=color,
            x=target_x, y=target_y, z=target_z,
            visible_colors=visible_colors,
            base=joints.get("base", 90),
            left=joints.get("left", 90),
            right=joints.get("right", 90),
        )

        try:
            resp = llm._client.chat.completions.create(
                model=getattr(llm, '_model', 'qwen2.5:7b'),
                max_tokens=256,
                timeout=8.0,
                messages=[
                    {"role": "system", "content": "仅输出 JSON，不要其他内容。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
            )
            raw = resp.choices[0].message.content.strip()
            strategy = json.loads(raw)
            log.info(f"LLM 抓取策略: {strategy.get('notes', '')}")

            # 合并默认值（LLM 可能漏字段）
            for k, v in default_strategy.items():
                if k not in strategy:
                    strategy[k] = v
            return strategy

        except Exception as e:
            log.warning(f"LLM 抓取规划失败: {e} — 使用默认策略")
            return default_strategy

    @staticmethod
    def analyze_calibration(
        execution_records: list[dict],
        llm,  # LLMIntentParser instance
    ) -> Optional[dict]:
        """分析历史执行记录，用 LLM 推理标定参数修正.

        execution_records 格式:
          [{"target_xyz": [x,y,z], "result_joints": {base,left,right}, "success": bool}, ...]

        返回修正建议或 None.
        """
        if not execution_records or len(execution_records) < 3:
            return None
        if llm is None or not getattr(llm, 'is_available', False):
            return None

        # 格式化记录为可读文本
        lines = []
        for i, rec in enumerate(execution_records[-10:]):  # 最近 10 条
            tgt = rec.get("target_xyz", [0, 0, 0])
            joints = rec.get("result_joints", {})
            ok = "✓" if rec.get("success") else "✗"
            lines.append(
                f"  [{ok}] 目标({tgt[0]:.0f},{tgt[1]:.0f},{tgt[2]:.0f})mm → "
                f"关节 base={joints.get('base','?')}° "
                f"left={joints.get('left','?')}° "
                f"right={joints.get('right','?')}°"
            )

        prompt = CALIBRATION_PROMPT.format(records="\n".join(lines))

        try:
            resp = llm._client.chat.completions.create(
                model=getattr(llm, '_model', 'qwen2.5:7b'),
                max_tokens=200,
                timeout=8.0,
                messages=[
                    {"role": "system", "content": "仅输出 JSON，不要其他内容。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
            )
            raw = resp.choices[0].message.content.strip()
            calib = json.loads(raw)

            elbow_adj = calib.get("elbow_offset_adjust", 0.0)
            base_adj = calib.get("base_offset_adjust", 0.0)
            confidence = calib.get("confidence", 0.0)

            if confidence > 0.5:
                log.info(
                    f"LLM 标定建议: elbow_offset={elbow_adj:.1f}°, "
                    f"base_offset={base_adj:.1f}° (置信度={confidence:.1%})"
                )
            else:
                log.info(f"LLM 标定: 置信度过低 ({confidence:.1%}), 跳过自动修正")

            return calib

        except Exception as e:
            log.warning(f"LLM 标定分析失败: {e}")
            return None

    @staticmethod
    def apply_calibration(calib_result: dict):
        """将标定结果应用到 ArmIK 参数."""
        if not calib_result:
            return

        elbow_adj = calib_result.get("elbow_offset_adjust", 0.0)
        base_adj = calib_result.get("base_offset_adjust", 0.0)
        confidence = calib_result.get("confidence", 0.0)

        if confidence < 0.5:
            return

        if abs(elbow_adj) > 0.5:
            old = ArmIK.ELBOW_OFFSET
            ArmIK.ELBOW_OFFSET += elbow_adj
            ArmIK.ELBOW_OFFSET = round(ArmIK.ELBOW_OFFSET, 1)
            log.info(f"标定已应用: ELBOW_OFFSET {old} → {ArmIK.ELBOW_OFFSET}")
