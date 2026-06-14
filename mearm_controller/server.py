"""
MeArm 工作台 — 主入口 & 后台线程
===============================
从原 workbench_server.py 提取，不做任何修改。

用法:
  python -m mearm_controller.server                        # 模拟模式 (无串口)
  python -m mearm_controller.server --port COM3            # 连接 Arduino
  python -m mearm_controller.server --port COM3 --cam 1    # 指定摄像头
  python workbench_server.py                               # 兼容旧入口

然后打开浏览器访问: http://localhost:5000
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import queue
import sys
import threading
import time
from typing import Optional

import cv2

from flask_socketio import SocketIO

from .config import (
    SAFE_MODE,

    HAND_GESTURE_CN, COLOR_CN, GESTURES, HOME_ANGLES,
    SAFE_GESTURE_DELAY, GESTURE_PAUSE_AFTER_ACTION,
    LLM_SYSTEM_PROMPT, log,
)
from .shared_state import state
from .arm_serial import ArmSerial
from .vision import VisionPipeline
from .voice_listener import VoiceListener
from .speaker import Speaker
from .gesture_recognizer import GestureRecognizer
from .llm_parser import LLMIntentParser, llm_speak
from .routes import create_app, register_routes, _pick_and_place
from .spatial_memory import spatial

# ─── 自学习库 (可选) ──────────────────────────────────────────────────────────
try:
    from mearm_learner import InteractionMemory, PromptAdapter, Interaction
    HAS_LEARNER = True
except ImportError:
    HAS_LEARNER = False


# ══════════════════════════════════════════════════════════════════════════════
# 后台视觉线程
# ══════════════════════════════════════════════════════════════════════════════

def vision_loop(vision: VisionPipeline, gesture_recog: Optional[GestureRecognizer],
                socketio: SocketIO, gesture_queue: queue.Queue):
    """在后台线程中持续运行视觉处理 + 手势识别."""
    log.info("视觉 + 手势处理线程已启动")
    while True:
        if vision._cap is None or not vision._cap.isOpened():
            time.sleep(0.5)
            continue
        ok = vision.process_frame()
        if not ok:
            time.sleep(0.05)
            continue

        # 手势识别 (检查开关和暂停状态)
        if gesture_recog and state.gesture_recog_enabled:
            now = time.time()
            if state.arm_busy or state.action_paused or now < state.gesture_paused_until:
                time.sleep(0.05)
                continue
            with state._lock:
                frame = state.raw_frame.copy() if state.raw_frame is not None else None
            if frame is not None:
                gesture = gesture_recog.process_frame(frame)
                if gesture and gesture != "none":
                    state.add_log(f"✋ 检测到手势: {HAND_GESTURE_CN.get(gesture, gesture)}")
                    gesture_queue.put(gesture)
                    socketio.emit("gesture_event", {
                        "gesture": gesture,
                        "name_cn": HAND_GESTURE_CN.get(gesture, gesture),
                    })
    log.info("视觉处理线程已退出")


# ══════════════════════════════════════════════════════════════════════════════
# 状态广播线程
# ══════════════════════════════════════════════════════════════════════════════

def broadcast_loop(socketio: SocketIO):
    """定期向所有客户端广播状态更新."""
    log.info("状态广播线程已启动")
    while True:
        time.sleep(0.15)  # ~7 Hz
        try:
            socketio.emit("state_update", state.get_state_dict())
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════════════
# Arduino 状态轮询线程
# ══════════════════════════════════════════════════════════════════════════════

def status_poll_loop(arm: ArmSerial):
    """每秒查询 Arduino 实际关节角度，同步到 shared state.

    注意: arm_busy 时跳过, 避免与 wait_done() 抢串口响应.
    """
    log.info("Arduino 状态轮询线程已启动")
    while True:
        time.sleep(1.0)
        if state.arm_busy:
            continue
        if not arm._ser or not arm._ser.is_open:
            continue
        try:
            resp = arm.send("status", wait_ack=True)
            if resp.startswith("STATUS:"):
                # 格式: STATUS:base:X,left:Y,right:Z,claw:W
                parts = resp.replace("STATUS:", "").split(",")
                for part in parts:
                    if ":" in part:
                        joint, val = part.split(":")
                        try:
                            state.joint_angles[joint] = int(val)
                        except ValueError:
                            pass
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════════════
# 主入口
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="MeArm 工作台服务器")
    parser.add_argument("--port", default=None, help="Arduino 串口 (如 COM3)")
    parser.add_argument("--cam", default=0, help="摄像头索引 (默认 0, 可用 'auto' 自动搜索)")
    parser.add_argument("--ip-cam", default=None, help="手机 IP 摄像头 URL (如 http://192.168.1.5:8080/video)")
    parser.add_argument("--host", default="0.0.0.0", help="绑定地址 (默认 0.0.0.0)")
    parser.add_argument("--web-port", type=int, default=5000, help="Web 端口 (默认 5000)")
    parser.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    parser.add_argument("--ollama-url", default=None,
                    help="Ollama API 地址 (默认 http://localhost:11434/v1)")
    parser.add_argument("--api-key", default=None,
                    help="云端 LLM API Key (OpenAI 兼容, 如 DeepSeek)")
    parser.add_argument("--api-base-url", default=None,
                    help="云端 LLM API 地址 (默认 https://api.deepseek.com)")
    parser.add_argument("--vision-api-key", default=None,
                    help="云端多模态视觉 API Key (如 Moonshot/Kimi)")
    parser.add_argument("--no-llm", action="store_true", help="仅关键词模式, 跳过 LLM")
    parser.add_argument("--no-voice", action="store_true", help="禁用语音识别")
    parser.add_argument("--unsafe", action="store_true", help="关闭安全模式 (需外接电源)")
    args = parser.parse_args()

    # ── 安全模式 (必须在 ArmSerial 构造前设置, 因为 __init__ 读取此标志) ──
    if args.unsafe:
        SAFE_MODE = False
        state.add_log("⚡ 安全模式已关闭 — 需要外接电源!")

    # 检查 Flask 依赖
    try:
        from flask import Flask
        from flask_socketio import SocketIO
    except ImportError:
        print("请安装 Flask: pip install flask flask-socketio")
        sys.exit(1)

    print("""
    ╔══════════════════════════════════════════════╗
    ║      🤖 MeArm 工作台 v1.0                    ║
    ║      实时调试与观测仪表盘                       ║
    ╚══════════════════════════════════════════════╝
    """)

    # ── LLM 初始化 (Ollama / 云端 API 自适应) ──────────────────────────
    import mearm_controller.config as cfg
    if args.ollama_url:
        cfg.OLLAMA_BASE_URL = args.ollama_url
    if args.api_key:
        cfg.LLM_API_KEY = args.api_key
    if args.api_base_url:
        cfg.LLM_API_BASE_URL = args.api_base_url
    if args.vision_api_key:
        cfg.VISION_API_KEY = args.vision_api_key
    api_key = True  # LLMIntentParser auto-detects provider

    # ── 初始化子系统 ──────────────────────────────────────────────────────────
    arm = ArmSerial(args.port)
    cam_arg = args.cam
    if cam_arg == "auto":
        cam_arg = _auto_detect_cam()
        state.add_log(f"🔍 自动检测到摄像头索引 {cam_arg}")
    vision = VisionPipeline(cam_index=int(cam_arg) if cam_arg != "auto" else 0, ip_cam_url=args.ip_cam)

    # 语音队列 (Vosk → 指令处理)
    voice_q: queue.Queue = queue.Queue()
    listener = VoiceListener(voice_q)
    speaker = Speaker()
    llm = LLMIntentParser() if not args.no_llm else None

    # ── API 视觉回退 (YOLO 不可用时) ─────────────────────────────────────
    if llm and llm.is_available and not vision._yolo:
        try:
            from .vision_api import APIDetector
            api_detector = APIDetector(llm)
            if api_detector.is_available:
                vision.set_api_detector(api_detector)
        except Exception as e:
            state.add_log(f"⚠️ API 视觉回退初始化失败: {e}")

    # 手势识别器
    gesture_recog = GestureRecognizer()
    gesture_q: queue.Queue = queue.Queue()  # 手势事件 → LLM 处理

    # ── 自学习库初始化 ──────────────────────────────────────────────────────
    memory = None
    prompt_adapter = None
    if HAS_LEARNER:
        memory = InteractionMemory()
        prompt_adapter = PromptAdapter(memory, base_prompt=LLM_SYSTEM_PROMPT)
        state.add_log(f"🧠 自学习库已启用 (历史交互: {memory.count} 条)")
        # 如果已有足够数据，切换为增强 prompt
        if prompt_adapter:
            enhanced = prompt_adapter.get_augmented_prompt()
            if enhanced and llm:
                llm.update_system_prompt(enhanced)
                state.add_log("📈 LLM 提示词已根据历史学习增强")

    # 启动语音
    if not args.no_voice:
        listener.start()
    speaker.start()

    # 创建 Flask 应用
    app = create_app()
    socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading",
                        ping_interval=5, ping_timeout=30)

    # 注册路由 (传入语音/LLM/自学习/空间记忆组件)
    register_routes(app, socketio, arm, llm, speaker, listener, gesture_recog,
                    memory=memory, prompt_adapter=prompt_adapter, spatial=spatial)

    # 启动视觉 + 手势线程
    vis_thread = threading.Thread(
        target=vision_loop, args=(vision, gesture_recog, socketio, gesture_q), daemon=True)
    vis_thread.start()

    # 启动广播线程
    bcast_thread = threading.Thread(target=broadcast_loop, args=(socketio,), daemon=True)
    bcast_thread.start()

    # 启动 Arduino 状态轮询
    status_thread = threading.Thread(target=status_poll_loop, args=(arm,), daemon=True)
    status_thread.start()

    # ── 手势事件处理循环 ──────────────────────────────────────────────────────
    def gesture_poll_loop():
        """从手势队列中取出事件，通过 LLM 解析为动作."""
        log.info("手势事件处理循环已启动")
        while True:
            try:
                gesture_name = gesture_q.get(timeout=0.5)
            except queue.Empty:
                continue
            except Exception:
                break

            if state.action_paused:
                continue

            with app.app_context():
                # 将手势转为自然语言描述，交给 LLM 决策
                gcn = HAND_GESTURE_CN.get(gesture_name, gesture_name)
                gesture_text = f"用户做了手势: {gcn} (gesture: {gesture_name})"
                visible_colors = [d.color for d in state.detections]

                # ── 空间记忆上下文 ──────────────────────────────────────
                spatial_context = spatial.get_context_text() if not spatial.is_empty else ""

                # ── 分层意图解析: 关键词(含空间记忆) → Kimi(视觉优先) → DeepSeek ──
                intent = LLMIntentParser.keyword_fallback(
                    gesture_name, visible_colors, spatial=spatial)

                if (intent is None or intent.get("confidence", 0) < 0.5) and llm:
                    frame = state.raw_frame
                    if frame is not None and llm._vision_model:
                        llm_intent = llm.parse_with_image(gesture_text, frame, visible_colors,
                                                          spatial_context=spatial_context)
                    # else fall through to text-only below
                    if llm_intent and llm_intent.get("confidence", 0) >= 0.5:
                        intent = llm_intent
                        state.add_log("🎯 视觉解析生效 (手势)")

                if (intent is None or intent.get("confidence", 0) < 0.5) and llm:
                    context_text = gesture_text
                    if spatial_context:
                        context_text = f"{spatial_context}\n{gesture_text}"
                    llm_intent = llm.parse(context_text, visible_colors)
                    if llm_intent:
                        intent = llm_intent

                if intent is None:
                    intent = {"action": "say", "color": None, "gesture": None,
                              "message": "抱歉，我没有理解这个手势。", "confidence": 0.0}

                state.add_log(f"✋ 手势意图: {json.dumps(intent, ensure_ascii=False)}")

                message = intent.get("message", "")
                action_desc = f"用户做了{HAND_GESTURE_CN.get(gesture_name, gesture_name)}手势"
                if message:
                    llm_speak(speaker, llm,
                              f"{action_desc}, 回复要点: {message}",
                              message)

                action = intent.get("action", "say")

                # ── 忙态保护: arm_busy 时只允许 say/home ──
                _BUSY_SAFE = {"say", "home"}
                if state.arm_busy and action not in _BUSY_SAFE:
                    state.add_log(f"⚠️ 机械臂正忙, 手势动作被拦截: {action}")
                    state.gesture_paused_until = time.time() + 1.5
                    socketio.emit("state_update", state.get_state_dict())
                    socketio.emit("voice_reply",
                                  {"text": "机械臂正忙，手势已忽略。", "intent": intent})
                    # ── 仍记录交互 ──────────────────────────────────────
                    if memory is not None:
                        memory.add(Interaction(
                            timestamp=time.time(),
                            input_type="gesture",
                            raw_input=gesture_name,
                            visible_colors=visible_colors,
                            llm_intent=intent,
                            executed_action="blocked_busy",
                            success=False,
                        ))
                        memory.auto_save()
                    continue

                if action == "pick_and_place":
                    color = intent.get("color")
                    target = next((d for d in state.detections if d.color == color), None)
                    ccn = COLOR_CN.get(color, color)
                    if target is None:
                        llm_speak(speaker, llm,
                                  f"摄像头中没有检测到{ccn}物体",
                                  f"想抓{ccn}但没看到诶")
                        state.add_log(f"❌ 手势抓取失败: 看不到 {color}")
                        # 动作未执行，缩短暂停
                        state.gesture_paused_until = time.time() + 1.5
                    else:
                        state.arm_busy = True
                        state.add_log(f"✋ 手势抓取: {color}")
                        socketio.emit("state_update", state.get_state_dict())

                        def _g_pick():
                            _pick_and_place(arm, target.x_mm, target.y_mm,
                                            llm=llm, color=color,
                                            visible_colors=visible_colors,
                                            spatial=spatial,
                                            input_type="gesture", raw_input=gesture_name)
                            state.arm_busy = False
                            state.action_paused = True
                            state.gesture_paused_until = time.time() + GESTURE_PAUSE_AFTER_ACTION
                            llm_speak(speaker, llm,
                                      f"{ccn}物体抓取放置完成",
                                      f"{ccn}抓好啦")
                            socketio.emit("state_update", state.get_state_dict())

                        threading.Thread(target=_g_pick, daemon=True).start()

                elif action == "gesture":
                    gname = intent.get("gesture") or "wave"
                    seq = GESTURES.get(gname)
                    if seq:
                        state.arm_busy = True
                        state.current_gesture = gname
                        state.add_log(f"✋ 手势触发机械臂: {gname}")
                        socketio.emit("state_update", state.get_state_dict())

                        def _g_gesture():
                            for step in seq:
                                state.update_joints(step)
                                arm.move(step)
                                arm.wait_done(timeout=2.0)
                                time.sleep(SAFE_GESTURE_DELAY if SAFE_MODE else 0.4)
                            state.arm_busy = False
                            state.current_gesture = ""
                            state.action_paused = True
                            state.gesture_paused_until = time.time() + GESTURE_PAUSE_AFTER_ACTION
                            socketio.emit("state_update", state.get_state_dict())

                        threading.Thread(target=_g_gesture, daemon=True).start()

                elif action == "home":
                    state.update_joints(dict(HOME_ANGLES))
                    state.arm_busy = False
                    state.current_gesture = ""
                    state.action_paused = False
                    state.gesture_paused_until = time.time() + GESTURE_PAUSE_AFTER_ACTION
                    arm.home()
                    state.add_log("🏠 手势触发回零")
                    llm_speak(speaker, llm, "机械臂已回到初始位置", "回零了")

                socketio.emit("state_update", state.get_state_dict())
                socketio.emit("voice_reply", {"text": message, "intent": intent})

                # ── 自学习: 记录手势交互 ──────────────────────────────────
                if memory is not None:
                    memory.add(Interaction(
                        timestamp=time.time(),
                        input_type="gesture",
                        raw_input=gesture_name,
                        visible_colors=visible_colors,
                        llm_intent=intent,
                        executed_action=action,
                        success=(action != "say"),
                    ))
                    memory.auto_save()

    gesture_thread = threading.Thread(target=gesture_poll_loop, daemon=True)
    gesture_thread.start()

    # ── 语音指令处理循环 ──────────────────────────────────────────────────────
    def voice_poll_loop():
        """从语音识别队列中取出文本，调用指令管道处理."""
        log.info("语音指令处理循环已启动")
        while True:
            try:
                text = voice_q.get(timeout=0.5)
            except queue.Empty:
                continue
            except Exception:
                break

            if state.action_paused:
                continue

            # 在 Flask 上下文中处理
            with app.app_context():
                state.last_voice_text = text
                visible_colors = [d.color for d in state.detections]

                # ── 空间记忆上下文 ──────────────────────────────────────
                spatial_context = spatial.get_context_text() if not spatial.is_empty else ""

                # ── 分层意图解析: 关键词(本地优先) → Kimi混合识别 → DeepSeek ──
                # 本地关键词优先 — 毫秒级响应:
                #   1. 关键词本地匹配 — 主解析层 (含空间记忆查询), 高置信度直接执行
                #   2. Kimi 多模态 — 视觉+语音联合理解, 关键词未命中时兜底
                #   3. DeepSeek 纯文本 — 最终兜底
                intent = None

                # 1. 关键词优先 (本地毫秒级, 不经过网络)
                intent = LLMIntentParser.keyword_fallback(text, visible_colors, spatial=spatial)
                if intent and intent.get("confidence", 0) >= 0.8:
                    state.add_log(f"📝 关键词匹配: '{text}' → {intent.get('action', '?')} (本地)")
                elif intent and intent.get("confidence", 0) >= 0.5:
                    state.add_log(f"📝 关键词低置信度: '{text}' → {intent.get('action', '?')}, 尝试 API 增强")
                else:
                    intent = None  # 低于 0.5 不采纳

                # 2. Kimi 混合识别: 关键词未命中时, 用视觉+语音联合理解
                if (intent is None or intent.get("confidence", 0) < 0.5) and llm:
                    frame = state.raw_frame
                    if frame is not None and llm._vision_model:
                        llm_intent = llm.parse_with_image(text, frame, visible_colors,
                                                          spatial_context=spatial_context)
                    # else fall through to text-only below

                    if llm_intent and llm_intent.get("confidence", 0) >= 0.5:
                        intent = llm_intent
                        state.add_log(f"🎯 视觉解析: '{text}' → {kimi_intent.get('action', '?')}")

                # 3. LLM 纯文本兜底
                if (intent is None or intent.get("confidence", 0) < 0.5) and llm:
                    context_text = text
                    if spatial_context:
                        context_text = f"{spatial_context}\n用户说: {text}"
                    llm_intent = llm.parse(context_text, visible_colors)
                    if llm_intent:
                        intent = llm_intent
                        state.add_log(f"🧠 LLM 解析: '{text}' → {llm_intent.get('action', '?')}")

                if intent is None:
                    intent = {"action": "say", "color": None, "gesture": None,
                              "message": "抱歉，我没有理解您的指令。", "confidence": 0.0}

                state.add_log(f"🧠 语音意图: {json.dumps(intent, ensure_ascii=False)}")

                message = intent.get("message", "")
                action_type = intent.get("action", "say")
                if message:
                    llm_speak(speaker, llm,
                              f"用户说了: {text}, 意图: {action_type}, 回复要点: {message}",
                              message)

                action = intent.get("action", "say")

                # ── 忙态保护: arm_busy 时只允许 say/home ──
                _BUSY_SAFE = {"say", "home"}
                if state.arm_busy and action not in _BUSY_SAFE:
                    state.add_log(f"⚠️ 机械臂正忙, 语音动作被拦截: {action}")
                    llm_speak(speaker, llm,
                              "机械臂正在执行上一个动作，请稍等一下",
                              "我还在忙，请等一下再试")
                    socketio.emit("state_update", state.get_state_dict())
                    socketio.emit("voice_reply",
                                  {"text": "机械臂正忙，请稍后再试。", "intent": intent})
                    # ── 仍记录交互 (自学习) ──────────────────────────────
                    if memory is not None:
                        memory.add(Interaction(
                            timestamp=time.time(),
                            input_type="voice",
                            raw_input=text,
                            visible_colors=visible_colors,
                            llm_intent=intent,
                            executed_action="blocked_busy",
                            success=False,
                        ))
                        memory.auto_save()
                    continue

                if action == "pick_and_place":
                    color = intent.get("color")
                    target = next((d for d in state.detections if d.color == color), None)
                    ccn = COLOR_CN.get(color, color)
                    if target is None:
                        llm_speak(speaker, llm,
                                  f"摄像头中没有检测到{ccn}物体",
                                  f"抱歉，我现在看不到{ccn}物体")
                        state.add_log(f"❌ 看不到 {color}")
                    else:
                        state.arm_busy = True
                        state.add_log(f"🎯 语音抓取: {color}")
                        socketio.emit("state_update", state.get_state_dict())

                        def _pick():
                            _pick_and_place(arm, target.x_mm, target.y_mm,
                                            llm=llm, color=color,
                                            visible_colors=visible_colors,
                                            spatial=spatial,
                                            input_type="voice", raw_input=text)
                            state.arm_busy = False
                            state.action_paused = True
                            state.gesture_paused_until = time.time() + GESTURE_PAUSE_AFTER_ACTION
                            llm_speak(speaker, llm,
                                      f"{ccn}物体抓取放置完成",
                                      f"{ccn}物体抓取完成")
                            socketio.emit("state_update", state.get_state_dict())

                        threading.Thread(target=_pick, daemon=True).start()

                elif action == "gesture":
                    gname = intent.get("gesture") or "wave"
                    seq = GESTURES.get(gname)
                    if seq:
                        state.arm_busy = True
                        state.current_gesture = gname
                        socketio.emit("state_update", state.get_state_dict())

                        def _gesture():
                            for step in seq:
                                state.update_joints(step)
                                arm.move(step)
                                arm.wait_done(timeout=2.0)
                                time.sleep(SAFE_GESTURE_DELAY if SAFE_MODE else 0.4)
                            state.arm_busy = False
                            state.current_gesture = ""
                            state.action_paused = True
                            state.gesture_paused_until = time.time() + GESTURE_PAUSE_AFTER_ACTION
                            socketio.emit("state_update", state.get_state_dict())

                        threading.Thread(target=_gesture, daemon=True).start()

                elif action == "claw_open":
                    state.arm_busy = True
                    socketio.emit("state_update", state.get_state_dict())
                    state.update_joint("claw", 90)
                    arm.move({"claw": 90})
                    state.arm_busy = False
                    state.action_paused = True
                    state.add_log("🖐️ 语音: 张开夹爪")

                elif action == "claw_close":
                    state.arm_busy = True
                    socketio.emit("state_update", state.get_state_dict())
                    state.update_joint("claw", 0)
                    arm.move({"claw": 0})
                    state.arm_busy = False
                    state.action_paused = True
                    state.add_log("🖐️ 语音: 闭合夹爪")

                elif action == "move_joint":
                    joint_name = intent.get("joint", "base")
                    direction = intent.get("direction", 1)
                    step = 15  # 每步角度
                    # 映射关节名到实际舵机
                    joint_map = {
                        "base": ["base"],
                        "lift": ["left", "right"],    # 肩关节 (双臂同步)
                        "elbow": ["left", "right"],   # 肘关节 (双臂同步)
                    }
                    targets = joint_map.get(joint_name, ["base"])
                    state.arm_busy = True
                    socketio.emit("state_update", state.get_state_dict())
                    for j in targets:
                        current = state.joint_angles.get(j, 90)
                        new_angle = current + direction * step
                        state.update_joint(j, new_angle)
                        arm.move({j: new_angle})
                    state.arm_busy = False
                    state.action_paused = True
                    jcn = {"base": "底座", "lift": "手臂", "elbow": "肘部"}.get(joint_name, joint_name)
                    dir_cn = "↑" if direction > 0 else "↓"
                    state.add_log(f"🔧 语音关节运动: {jcn} {dir_cn}")

                elif action == "home":
                    state.update_joints(dict(HOME_ANGLES))
                    state.arm_busy = False
                    state.current_gesture = ""
                    state.action_paused = False
                    state.gesture_paused_until = time.time() + GESTURE_PAUSE_AFTER_ACTION
                    arm.home()
                    state.add_log("🏠 语音回零")

                socketio.emit("state_update", state.get_state_dict())
                socketio.emit("voice_reply", {"text": message, "intent": intent})

                # ── 自学习: 记录语音交互 ──────────────────────────────────
                if memory is not None:
                    memory.add(Interaction(
                        timestamp=time.time(),
                        input_type="voice",
                        raw_input=text,
                        visible_colors=visible_colors,
                        llm_intent=intent,
                        executed_action=action,
                        success=(action != "say"),
                    ))
                    memory.auto_save()
                    # 积累足够数据后，尝试增强 LLM prompt
                    if prompt_adapter and llm:
                        enhanced = prompt_adapter.get_augmented_prompt()
                        if enhanced and enhanced != llm._system_prompt:
                            llm._system_prompt = enhanced
                            state.add_log("📈 LLM 提示词已增强 (基于学习)")

    voice_thread = threading.Thread(target=voice_poll_loop, daemon=True)
    voice_thread.start()

    # 状态日志
    state.add_log("🚀 MeArm 工作台已启动")
    if llm:
        provider = getattr(llm, '_provider', 'ollama')
        provider_label = {"cloud": "☁️ 云端 API", "ollama": "🖥️ Ollama"}
        state.add_log(f"🧠 LLM 模式 ({provider_label.get(provider, provider)})")
        if getattr(llm, '_vision_model', None):
            vp = getattr(llm, '_vision_provider', 'ollama')
            vp_label = {"cloud": "☁️ 云端视觉", "ollama": "🖥️ Ollama 视觉"}
            state.add_log(f"👁️ 多模态视觉 ({vp_label.get(vp, vp)}: {llm._vision_model})")
    else:
        state.add_log("📝 关键词模式")

    # 自动打开浏览器
    if not args.no_browser:
        import webbrowser
        threading.Timer(1.5, lambda: webbrowser.open(f"http://localhost:{args.web_port}")).start()

    print(f"\n  🌐 浏览器访问: http://localhost:{args.web_port}\n")
    print("  按 Ctrl+C 退出\n")

    try:
        socketio.run(app, host=args.host, port=args.web_port, debug=False, allow_unsafe_werkzeug=True)
    except KeyboardInterrupt:
        print("\n正在关闭...")
    finally:
        listener.stop()
        speaker.stop()
        vision.release()
        arm.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
