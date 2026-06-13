"""
MeArm 工作台 — Flask 路由 & SocketIO 事件
========================================
从原 workbench_server.py 提取，不做任何修改。
"""

from __future__ import annotations

import base64
import json
import logging
import threading
import time
from typing import Optional

import cv2
import numpy as np

from flask import Flask, Response, render_template, request, jsonify
from flask_socketio import SocketIO, emit

from .config import (
    JOINT_LIMITS, HOME_ANGLES, GESTURES, COLOR_CN, JOINT_CN, GESTURE_CN,
    HAND_GESTURE_CN, FRAME_WIDTH, FRAME_HEIGHT, JPEG_QUALITY,
    SAFE_MODE, SAFE_GESTURE_DELAY, SAFE_PICK_DELAY,
    GESTURE_PAUSE_AFTER_ACTION,
    YOLO_CLASS_CN, log,
)
from .shared_state import state
from .arm_serial import ArmSerial
from .arm_ik import ArmIK
from .ik_llm import IKLLMEnhancer
from .llm_parser import LLMIntentParser, llm_speak
from .speaker import Speaker
from .voice_listener import VoiceListener
from .gesture_recognizer import GestureRecognizer

# ─── 自学习库 (可选) ──────────────────────────────────────────────────────────
try:
    from mearm_learner import InteractionMemory, PromptAdapter, Interaction
    HAS_LEARNER = True
except ImportError:
    HAS_LEARNER = False


# ══════════════════════════════════════════════════════════════════════════════
# Flask 应用 & 路由
# ══════════════════════════════════════════════════════════════════════════════

def create_app() -> Flask:
    app = Flask(__name__, template_folder="../templates")
    app.config["SECRET_KEY"] = "mearm-workbench-secret"
    return app


# ══════════════════════════════════════════════════════════════════════════════
# pick-and-place 辅助函数
# ══════════════════════════════════════════════════════════════════════════════

def _pick_and_place(arm: ArmSerial, x_mm: float, y_mm: float,
                   z_mm: float = 0.0,
                   llm: Optional[LLMIntentParser] = None,
                   color: str = "",
                   class_name: str = "",
                   visible_colors: Optional[list[str]] = None):
    """LLM 增强的 pick-and-place 序列.

    优先使用 LLM 规划策略，LLM 不可用时回退到硬编码默认值。
    """
    # ── 获取 LLM 策略 ──────────────────────────────────────────────────────
    strategy = IKLLMEnhancer.plan_pick_sequence(
        target_x=x_mm, target_y=y_mm, target_z=z_mm,
        color=color,
        visible_colors=visible_colors or [],
        llm=llm,
        current_joints=state.joint_angles,
    )

    pick_h = strategy.get("pick_height", 20.0)
    carry_h = strategy.get("carry_height", 90.0)
    drop_x = strategy.get("drop_x", 0.0)
    drop_y = strategy.get("drop_y", 120.0)
    pre_push = strategy.get("pre_push", False)

    move_delay = SAFE_PICK_DELAY if SAFE_MODE else 0.35
    claw_delay = 0.4 if SAFE_MODE else 0.2

    def _move(x, y, z):
        # 使用 LLM fallback: 解析 IK 失败时自动寻找替代坐标
        angles = IKLLMEnhancer.solve_with_fallback(
            x, y, z,
            visible_colors=visible_colors or [],
            llm=llm,
            current_joints=state.joint_angles,
        )
        if angles:
            state.update_joints(angles)
            arm.move(angles)
            time.sleep(move_delay)
        else:
            log.warning(f"_move 失败: 无法到达 ({x:.0f},{y:.0f},{z:.0f})")

    # ── 可选: 先推近 (远处物体策略) ────────────────────────────────────────
    if pre_push:
        push_dir = strategy.get("push_direction", "forward")
        log.info(f"先推近物体: 方向={push_dir}")
        push_x = x_mm + (20 if push_dir == "forward" else
                         -20 if push_dir == "back" else 0)
        push_y = y_mm + (20 if push_dir == "right" else
                         -20 if push_dir == "left" else 0)
        _move(push_x, push_y, pick_h + 10)
        _move(push_x, push_y, pick_h)
        # 推
        push_target_x = x_mm - 30
        push_target_y = y_mm - 30
        _move(push_target_x, push_target_y, pick_h)
        _move(push_target_x, push_target_y, carry_h)
        # 更新目标
        x_mm, y_mm = push_target_x, push_target_y
        state.add_log(f"  推近完成, 新目标: ({x_mm:.0f},{y_mm:.0f})mm")

    # ── 抓取序列 ───────────────────────────────────────────────────────────
    _move(x_mm, y_mm, carry_h)
    arm.move({"claw": JOINT_LIMITS["claw"][1]}); time.sleep(claw_delay)  # 张开
    _move(x_mm, y_mm, pick_h)
    arm.move({"claw": JOINT_LIMITS["claw"][0]}); time.sleep(claw_delay)  # 闭合
    _move(x_mm, y_mm, carry_h)
    _move(drop_x, drop_y, carry_h)
    _move(drop_x, drop_y, pick_h)
    arm.move({"claw": JOINT_LIMITS["claw"][1]}); time.sleep(claw_delay)  # 张开
    arm.home()
    state.update_joints(dict(HOME_ANGLES))
    state.add_log("✅ 抓取放置完成")


# ══════════════════════════════════════════════════════════════════════════════
# 路由注册
# ══════════════════════════════════════════════════════════════════════════════

def register_routes(app: Flask, socketio: SocketIO, arm: ArmSerial,
                    llm: Optional[LLMIntentParser], speaker: Speaker,
                    listener: VoiceListener, gesture_recog: Optional[GestureRecognizer] = None,
                    memory: Optional["InteractionMemory"] = None,
                    prompt_adapter: Optional["PromptAdapter"] = None):
    """注册所有 HTTP 路由和 SocketIO 事件.

    Args:
        memory: 自学习交互记忆库 (可选)
        prompt_adapter: LLM 提示词适配器 (可选)
    """

    # ── 物体查找辅助函数 ───────────────────────────────────────────────────────
    def _find_target(color: str = "", class_name: str = ""):
        """从当前检测结果中查找匹配的物体.

        优先级: class_name + color > class_name > color > 第一个检测
        返回 (Detection, label_cn) 或 (None, "")
        """
        dets = state.detections
        if not dets:
            return None, ""

        # 1. 精确匹配: class_name AND color
        if class_name and color:
            for d in dets:
                if d.class_name == class_name and d.color == color:
                    return d, f"{d.class_cn or class_name}({COLOR_CN.get(color, color)})"

        # 2. 仅 class_name
        if class_name:
            for d in dets:
                if d.class_name == class_name:
                    label = d.class_cn or class_name
                    if d.color:
                        label += f"({COLOR_CN.get(d.color, d.color)})"
                    return d, label

        # 3. 仅 color
        if color:
            for d in dets:
                if d.color == color:
                    label = COLOR_CN.get(color, color)
                    if d.class_cn:
                        label = f"{d.class_cn}({label})"
                    return d, label

        # 4. 返回置信度最高的检测
        best = max(dets, key=lambda d: d.confidence)
        label = best.class_cn or best.color or "物体"
        return best, label

    # ── 获取可见物体信息 (供 LLM 意图解析) ───────────────────────────────────
    def _get_visible_info() -> tuple[list[str], list[str]]:
        """返回 (visible_colors, visible_objects)."""
        colors = list({d.color for d in state.detections if d.color})
        objects = list({d.class_name for d in state.detections if d.class_name})
        return colors, objects

    # ── 文本/语音指令处理管道 ─────────────────────────────────────────────────
    def process_command(text: str):
        """完整指令管道: 文本 → LLM/关键词 → 意图 → 执行 → 语音回复"""
        state.last_voice_text = text
        visible_colors, visible_objects = _get_visible_info()

        # 解析意图
        if llm:
            intent = llm.parse(text, visible_colors)
            if intent is None:
                intent = LLMIntentParser.keyword_fallback(text, visible_colors)
        else:
            intent = LLMIntentParser.keyword_fallback(text, visible_colors)

        state.add_log(f"🧠 意图: {json.dumps(intent, ensure_ascii=False)}")

        # 语音回复 — 统一走 LLM 生成自然口语
        message = intent.get("message", "")
        action_desc = intent.get("action", "say")
        if message:
            llm_speak(speaker, llm,
                      f"用户指令意图: {action_desc}, 回复要点: {message}",
                      message)

        action = intent.get("action", "say")

        if action == "pick_and_place":
            color = intent.get("color") or ""
            class_name = intent.get("class_name") or ""
            target, label_cn = _find_target(color=color, class_name=class_name)

            if target is None:
                # 生成有意义的错误信息
                if class_name and color:
                    desc = f"{COLOR_CN.get(color, color)}{YOLO_CLASS_CN.get(class_name, class_name)}"
                elif class_name:
                    desc = YOLO_CLASS_CN.get(class_name, class_name)
                elif color:
                    desc = COLOR_CN.get(color, color)
                else:
                    desc = "指定"
                msg = f"抱歉，我现在看不到{desc}物体。"
                llm_speak(speaker, llm,
                          f"摄像头中没有检测到{desc}物体，无法抓取",
                          msg)
                state.add_log(f"❌ {msg}")
                emit("error", {"msg": msg})
            else:
                state.arm_busy = True
                state.add_log(f"🎯 语音抓取: {label_cn} @ ({target.x_mm:.0f},{target.y_mm:.0f})mm")
                emit("state_update", state.get_state_dict())

                def _pick():
                    _pick_and_place(arm, target.x_mm, target.y_mm,
                                    llm=llm, color=target.color or color,
                                    class_name=target.class_name or class_name,
                                    visible_colors=visible_colors)
                    state.arm_busy = False
                    state.gesture_paused_until = time.time() + GESTURE_PAUSE_AFTER_ACTION
                    llm_speak(speaker, llm,
                              f"{label_cn}物体抓取放置完成",
                              f"{label_cn}抓取完成")
                    socketio.emit("state_update", state.get_state_dict())

                threading.Thread(target=_pick, daemon=True).start()

        elif action == "gesture":
            gname = intent.get("gesture") or "wave"
            seq = GESTURES.get(gname)
            if seq:
                state.arm_busy = True
                state.current_gesture = gname
                state.add_log(f"🎭 语音手势: {gname}")
                emit("state_update", state.get_state_dict())

                def _gesture():
                    for step in seq:
                        state.update_joints(step)
                        arm.move(step)
                        time.sleep(SAFE_GESTURE_DELAY if SAFE_MODE else 0.4)
                    state.arm_busy = False
                    state.current_gesture = ""
                    state.gesture_paused_until = time.time() + GESTURE_PAUSE_AFTER_ACTION
                    socketio.emit("state_update", state.get_state_dict())

                threading.Thread(target=_gesture, daemon=True).start()

        elif action == "home":
            on_home()

        # 推送更新
        emit("state_update", state.get_state_dict())
        # 也推送语音回复到前端
        emit("voice_reply", {"text": message, "intent": intent})

        # ── 自学习: 记录文本指令交互 ──────────────────────────────────────
        if memory is not None and HAS_LEARNER:
            memory.add(Interaction(
                timestamp=time.time(),
                input_type="text",
                raw_input=text,
                visible_colors=visible_colors,
                llm_intent=intent,
                executed_action=action,
                success=(action != "say"),
            ))
            memory.auto_save()
            # 积累足够数据后增强 LLM
            if prompt_adapter and llm:
                enhanced = prompt_adapter.get_augmented_prompt()
                if enhanced and enhanced != llm._system_prompt:
                    llm._system_prompt = enhanced
                    state.add_log("📈 LLM 提示词已增强 (基于学习)")

    # ── 主页 ──────────────────────────────────────────────────────────────────
    @app.route("/")
    def index():
        return render_template("workbench.html")

    # ── MJPEG 流: 原始画面 ────────────────────────────────────────────────────
    def gen_raw():
        while True:
            with state._lock:
                frame = state.raw_frame.copy() if state.raw_frame is not None else None
            if frame is None:
                # 返回占位帧
                frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
                cv2.putText(frame, "Waiting for camera...", (120, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)
            _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")
            time.sleep(0.03)

    @app.route("/video/raw")
    def video_raw():
        return Response(gen_raw(), mimetype="multipart/x-mixed-replace; boundary=frame")

    # ── MJPEG 流: HSV 掩膜画面 ────────────────────────────────────────────────
    def gen_mask():
        while True:
            with state._lock:
                frame = state.mask_frame.copy() if state.mask_frame is not None else None
            if frame is None:
                frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
                cv2.putText(frame, "Waiting...", (180, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)
            _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")
            time.sleep(0.03)

    @app.route("/video/mask")
    def video_mask():
        return Response(gen_mask(), mimetype="multipart/x-mixed-replace; boundary=frame")

    # ── MJPEG 流: 检测标注画面 ────────────────────────────────────────────────
    def gen_annotated():
        while True:
            with state._lock:
                frame = state.annotated_frame.copy() if state.annotated_frame is not None else None
            if frame is None:
                frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
                cv2.putText(frame, "Waiting...", (180, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)
            _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")
            time.sleep(0.03)

    @app.route("/video/annotated")
    def video_annotated():
        return Response(gen_annotated(), mimetype="multipart/x-mixed-replace; boundary=frame")

    # ── API: 状态快照 ────────────────────────────────────────────────────────
    @app.route("/api/state")
    def api_state():
        return jsonify(state.get_state_dict())

    @app.route("/api/logs")
    def api_logs():
        return jsonify(state.get_logs())

    @app.route("/api/serial")
    def api_serial():
        return jsonify(state.get_serial_msgs())

    @app.route("/api/hsv")
    def api_hsv():
        with state._lock:
            return jsonify({
                color: {"lo": list(lo), "hi": list(hi)}
                for color, (lo, hi) in state.hsv_ranges.items()
                if color != "red2"
            })

    # ── SocketIO: 客户端连接 ──────────────────────────────────────────────────
    @socketio.on("connect")
    def on_connect():
        emit("state_update", state.get_state_dict())
        emit("logs", state.get_logs())
        emit("serial_update", state.get_serial_msgs())
        emit("voice_status", {"listening": listener._running if listener else False})
        log.info("浏览器客户端已连接")

    # ── SocketIO: 手动移动关节 ────────────────────────────────────────────────
    @socketio.on("move_joint")
    def on_move_joint(data: dict):
        joint = data.get("joint", "")
        angle = data.get("angle", 90)
        state.update_joint(joint, angle)
        arm.move({joint: int(angle)})
        jcn = JOINT_CN.get(joint, joint)
        llm_speak(speaker, llm,
                  f"{jcn}关节移动到了{int(angle)}度",
                  f"{jcn}已调到{int(angle)}度")

    # ── SocketIO: 全部关节移动 ────────────────────────────────────────────────
    @socketio.on("move_all")
    def on_move_all(data: dict):
        angles = data.get("angles", {})
        state.update_joints(angles)
        arm.move({k: int(v) for k, v in angles.items()})
        parts = [f"{JOINT_CN.get(k, k)}{int(v)}度" for k, v in angles.items()]
        desc = "所有关节已更新: " + ", ".join(parts)
        llm_speak(speaker, llm, desc, desc)

    # ── SocketIO: 回零 ────────────────────────────────────────────────────────
    @socketio.on("home")
    def on_home():
        state.update_joints(dict(HOME_ANGLES))
        state.arm_busy = False
        state.current_gesture = ""
        state.gesture_paused_until = time.time() + GESTURE_PAUSE_AFTER_ACTION
        arm.home()
        state.add_log("🏠 机械臂回零")
        llm_speak(speaker, llm, "机械臂已回到初始位置", "已回到初始位置")
        emit("state_update", state.get_state_dict())

    # ── SocketIO: 张开夹爪 ────────────────────────────────────────────────────
    @socketio.on("claw_open")
    def on_claw_open():
        max_open = JOINT_LIMITS["claw"][1]
        state.update_joint("claw", max_open)
        arm.move({"claw": max_open})
        state.add_log("🖐 夹爪张开")
        llm_speak(speaker, llm, "夹爪已张开", "夹爪已张开")

    # ── SocketIO: 闭合夹爪 ────────────────────────────────────────────────────
    @socketio.on("claw_close")
    def on_claw_close():
        min_close = JOINT_LIMITS["claw"][0]
        state.update_joint("claw", min_close)
        arm.move({"claw": min_close})
        state.add_log("✊ 夹爪闭合")
        llm_speak(speaker, llm, "夹爪已闭合", "夹爪已闭合")

    # ── SocketIO: 执行手势 ────────────────────────────────────────────────────
    @socketio.on("gesture")
    def on_gesture(data: dict):
        name = data.get("name", "wave")
        seq = GESTURES.get(name)
        if not seq:
            emit("error", {"msg": f"未知手势: {name}"})
            return
        state.arm_busy = True
        state.current_gesture = name
        state.add_log(f"🎭 执行手势: {name}")
        gcn = GESTURE_CN.get(name, name)
        llm_speak(speaker, llm, f"开始执行{gcn}手势", f"正在执行{gcn}手势")

        def _run():
            delay = SAFE_GESTURE_DELAY if SAFE_MODE else 0.4
            for step in seq:
                state.update_joints(step)
                arm.move(step)
                time.sleep(delay)
            state.arm_busy = False
            state.current_gesture = ""
            state.gesture_paused_until = time.time() + GESTURE_PAUSE_AFTER_ACTION
            llm_speak(speaker, llm, f"{gcn}手势执行完成", f"{gcn}手势完成")
            socketio.emit("state_update", state.get_state_dict())

        threading.Thread(target=_run, daemon=True).start()
        emit("state_update", state.get_state_dict())

    # ── SocketIO: 抓取颜色物体 ────────────────────────────────────────────────
    @socketio.on("pick_color")
    def on_pick_color(data: dict):
        color = data.get("color", "")
        class_name = data.get("class_name", "")
        target, label_cn = _find_target(color=color, class_name=class_name)

        if target is None:
            desc = label_cn or color or class_name or "物体"
            state.add_log(f"❌ 未找到 {desc}")
            llm_speak(speaker, llm,
                      f"摄像头中没有检测到{desc}，无法抓取",
                      f"抱歉，我现在看不到{desc}")
            emit("error", {"msg": f"未找到 {desc}"})
            return

        state.arm_busy = True
        state.add_log(f"🎯 抓取 {label_cn} @ ({target.x_mm:.0f}, {target.y_mm:.0f})mm")
        llm_speak(speaker, llm,
                  f"开始抓取{label_cn}",
                  f"正在抓取{label_cn}")

        def _pick():
            _pick_and_place(arm, target.x_mm, target.y_mm,
                            llm=llm, color=target.color or color,
                            class_name=target.class_name or class_name,
                            visible_colors=[d.color for d in state.detections])
            state.arm_busy = False
            state.gesture_paused_until = time.time() + GESTURE_PAUSE_AFTER_ACTION
            llm_speak(speaker, llm,
                      f"{label_cn}抓取放置完成",
                      f"{label_cn}抓取完成")
            socketio.emit("state_update", state.get_state_dict())

        threading.Thread(target=_pick, daemon=True).start()

    # ── SocketIO: 更新 HSV 范围 ───────────────────────────────────────────────
    @socketio.on("update_hsv")
    def on_update_hsv(data: dict):
        color = data.get("color", "")
        lo = data.get("lo", [0, 0, 0])
        hi = data.get("hi", [180, 255, 255])
        with state._lock:
            state.hsv_ranges[color] = (tuple(lo), tuple(hi))
        state.add_log(f"🎨 HSV 范围已更新: {color} lo={lo} hi={hi}")

    # ── SocketIO: 截图请求 ────────────────────────────────────────────────────
    @socketio.on("snapshot")
    def on_snapshot():
        with state._lock:
            frame = state.annotated_frame.copy() if state.annotated_frame is not None else None
        if frame is not None:
            _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            b64 = base64.b64encode(jpeg.tobytes()).decode("ascii")
            emit("snapshot_data", {"image": f"data:image/jpeg;base64,{b64}"})
            state.add_log("📸 截图已保存")
            llm_speak(speaker, llm, "当前画面截图已保存", "截图已保存")

    # ── SocketIO: 文本指令 (从网页输入框发送) ─────────────────────────────────
    @socketio.on("text_command")
    def on_text_command(data: dict):
        text = data.get("text", "").strip()
        if not text:
            return
        state.add_log(f"💬 收到指令: '{text}'")
        process_command(text)

    # ── SocketIO: 语音识别开关 ────────────────────────────────────────────────
    @socketio.on("toggle_voice")
    def on_toggle_voice(data: dict):
        enable = data.get("enable", False)
        if enable:
            listener.start()
            emit("voice_status", {"listening": True})
        else:
            listener.stop()
            emit("voice_status", {"listening": False})

    # ── SocketIO: 获取语音状态 ────────────────────────────────────────────────
    @socketio.on("voice_status")
    def on_voice_status():
        emit("voice_status", {"listening": listener._running if listener else False})

    # ── SocketIO: 紧急停止运动 ─────────────────────────────────────────────────
    @socketio.on("stop_motion")
    def on_stop_motion():
        state.arm_busy = False
        state.current_gesture = ""
        state.update_joints(dict(HOME_ANGLES))
        arm.home()
        state.add_log("🛑 紧急停止 — 已回零")
        llm_speak(speaker, llm, "机械臂已紧急停止，回到初始位置", "已停止，回零了")
        emit("state_update", state.get_state_dict())

    # ── SocketIO: 开关手势识别 ─────────────────────────────────────────────────
    @socketio.on("toggle_gesture_recog")
    def on_toggle_gesture_recog(data: dict):
        enable = data.get("enable", True)
        state.gesture_recog_enabled = enable
        status = "已开启" if enable else "已暂停"
        state.add_log(f"✋ 手势识别{status}")
        llm_speak(speaker, llm, f"手势识别{status}", f"手势识别{status}")
        emit("state_update", state.get_state_dict())
