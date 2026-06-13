"""
MeArm 工作台 — 全局配置常量
============================
从原 workbench_server.py 提取，不做任何修改。
"""

from __future__ import annotations

import logging
import os

try:
    from dotenv import load_dotenv
    load_dotenv()  # 加载 .env 文件 (如果存在)
except ImportError:
    pass  # python-dotenv 未安装, 跳过

# ─── 日志 ─────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("workbench")

# ══════════════════════════════════════════════════════════════════════════════
# 配置
# ══════════════════════════════════════════════════════════════════════════════

BAUD_RATE = 115_200
SERIAL_TIMEOUT = 2.0

JOINT_LIMITS = {
    "base": (30, 150),
    "left": (30, 150),
    "right": (30, 150),
    "claw": (0, 90),
}
# 起始位置: 底座逆时针90°侧, 左右臂向上90°, 夹爪全开
HOME_ANGLES = {"base": 30, "left": 150, "right": 150, "claw": 90}

HSV_RANGES: dict[str, tuple] = {
    "red": ((0, 120, 70), (10, 255, 255)),
    "red2": ((170, 120, 70), (180, 255, 255)),
    "green": ((40, 60, 40), (85, 255, 255)),
    "blue": ((100, 80, 40), (130, 255, 255)),
    "yellow": ((20, 100, 100), (35, 255, 255)),
}
MIN_CONTOUR_AREA = 800

CALIB_POINTS: list[tuple] = [
    (100, 400, -80, 80),
    (540, 400, 80, 80),
    (100, 100, -80, 160),
    (540, 100, 80, 160),
]

# ─── 摄像头安装位置 ───────────────────────────────────────────────────────────
# 摄像头在机械臂底座上方约 240mm，水平朝前拍摄工作区域
CAMERA_HEIGHT_MM = 240.0   # 摄像头距台面的高度 (mm)
CAMERA_OFFSET_X_MM = 0.0   # 摄像头相对于底座中心的 X 偏移
CAMERA_OFFSET_Y_MM = 0.0   # 摄像头相对于底座中心的 Y 偏移
CAMERA_DIRECTION = "horizontal"  # 水平朝前 (非俯拍)

# 手势序列 (适配新起始位: base=30, left=150, right=150, claw=90)
# 指向类手势只有1步, 停留在目标位置不自动回零
GESTURES: dict[str, list[dict]] = {
    "wave": [          # 大幅左右摇摆
        {"base": 30, "left": 150, "right": 150, "claw": 90},
        {"base": 120, "left": 150, "right": 150, "claw": 90},
        {"base": 30, "left": 150, "right": 150, "claw": 90},
        {"base": 120, "left": 150, "right": 150, "claw": 90},
        {"base": 30, "left": 150, "right": 150, "claw": 90},
    ],
    "point": [         # 指向前方 (base居中, 手臂下放, 爪半合)
        {"base": 90, "left": 120, "right": 120, "claw": 45},
    ],
    "point_left": [    # 指向左边
        {"base": 30, "left": 120, "right": 120, "claw": 45},
    ],
    "point_right": [   # 指向右边
        {"base": 150, "left": 120, "right": 120, "claw": 45},
    ],
    "nod": [
        {"base": 30, "left": 130, "right": 130, "claw": 90},
        {"base": 30, "left": 150, "right": 150, "claw": 90},
        {"base": 30, "left": 130, "right": 130, "claw": 90},
        {"base": 30, "left": 150, "right": 150, "claw": 90},
    ],
    "greet": [         # 大弧度问候
        {"base": 30, "left": 150, "right": 150, "claw": 90},
        {"base": 100, "left": 130, "right": 130, "claw": 90},
        {"base": 30, "left": 150, "right": 150, "claw": 90},
        {"base": 100, "left": 130, "right": 130, "claw": 90},
        {"base": 30, "left": 150, "right": 150, "claw": 90},
    ],
}

# 中英文名称映射 (用于语音播报)
JOINT_CN = {"base": "底座", "left": "左侧", "right": "右侧", "claw": "夹爪"}
GESTURE_CN = {"wave": "挥手", "point": "指向", "nod": "点头", "greet": "问候"}
COLOR_CN = {"red": "红色", "green": "绿色", "blue": "蓝色", "yellow": "黄色"}

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
JPEG_QUALITY = 60  # MJPEG 压缩质量 (降低以提高帧率)

# ─── 安全模式 ─────────────────────────────────────────────────────────────────
SAFE_MODE = True          # 默认开启安全模式
SAFE_JOINT_DELAY = 0.08   # 单关节发送间隔 (秒)
SAFE_GESTURE_DELAY = 0.6  # 手势步间间隔 (秒)
SAFE_PICK_DELAY = 0.5     # 抓取步间间隔 (秒)

# ─── 手势识别 ─────────────────────────────────────────────────────────────────
# MediaPipe 手部关键点索引
MP_WRIST = 0
MP_THUMB_TIP, MP_INDEX_TIP, MP_MIDDLE_TIP, MP_RING_TIP, MP_PINKY_TIP = 4, 8, 12, 16, 20
MP_INDEX_PIP, MP_MIDDLE_PIP, MP_RING_PIP, MP_PINKY_PIP = 6, 10, 14, 18
MP_INDEX_MCP, MP_MIDDLE_MCP, MP_RING_MCP, MP_PINKY_MCP = 5, 9, 13, 17

# 手势名称 → 中文
HAND_GESTURE_CN = {
    "open_palm": "五指张开",
    "fist": "握拳",
    "pointing": "食指指向",
    "thumbs_up": "竖大拇指",
    "peace": "剪刀手",
    "wave": "挥手",
    "none": "",
}

# 手势触发冷却 (秒) — 防止连续重复触发 (提高以避免误触发)
GESTURE_COOLDOWN = 3.0
# 动作完成后暂停手势识别的时长 (秒)
GESTURE_PAUSE_AFTER_ACTION = 4.0

# ─── 手势识别稳定性 ─────────────────────────────────────────────────────
# 同一手势需要连续检测到的帧数才触发 (降低灵敏度)
GESTURE_MIN_CONSECUTIVE = 5
# 手势消失后需要连续 "none" 帧数才确认退出 (迟滞)
GESTURE_EXIT_HYSTERESIS = 8
# 挥手检测: 需要的最小穿越次数
WAVE_MIN_CROSSES = 5
# 挥手检测: 手腕历史帧数
WAVE_HISTORY_FRAMES = 20

# ─── 语音识别稳定性 ─────────────────────────────────────────────────────
# Vosk 小模型 (cn/en) 对短词的置信度通常在 0.35-0.55，门槛太高会导致
# "你好"等短指令完全无法识别。
# 中文模型: 0.35 (小模型对短词的典型下限)
VOICE_MIN_CONFIDENCE_CN = 0.35
# 英文模型: 0.45 (略高，因为非中文环境下英文更可能是噪音)
VOICE_MIN_CONFIDENCE_EN = 0.45
# 向后兼容: 通用门槛取中值
VOICE_MIN_CONFIDENCE = 0.40
# 最短识别文本长度 (字符数, 过滤噪音短片段)
VOICE_MIN_TEXT_LENGTH = 1
# 同文本去重窗口 (秒)
VOICE_DEDUP_WINDOW = 2.0
# 英文模型固定置信度 (小模型不返回 confidence 时使用)
# 设为低于 VOICE_MIN_CONFIDENCE_EN，避免空白英文结果抢走中文识别
VOICE_EN_DEFAULT_CONFIDENCE = 0.30

# ─── 语音播报 ─────────────────────────────────────────────────────────────────
# Edge-TTS 中文语音选项 (按自然度排序)
EDGE_VOICE = "zh-CN-XiaoxiaoNeural"   # 女声, 温暖自然 ← 默认
# EDGE_VOICE = "zh-CN-YunxiNeural"    # 男声, 沉稳
# EDGE_VOICE = "zh-CN-XiaoyiNeural"   # 女声, 活泼可爱
EDGE_TTS_RATE = "+10%"                # 语速稍快
EDGE_TTS_PITCH = "+0Hz"               # 音高不变

# ─── YOLO 物体检测 ─────────────────────────────────────────────────────────────
YOLO_MODEL = "yolov8s.pt"         # ultralytics 自动下载, 也可用 yolov8n.pt (更快)
YOLO_CONFIDENCE_THRESHOLD = 0.5    # 最低置信度
YOLO_DEVICE = "cuda:0"            # 有 GPU 用 cuda:0, 否则 "cpu"
YOLO_ENABLED = True               # 总开关 (无 GPU 会自动回退 API)
# YOLO 检测帧间隔: 每 N 帧跑一次, 其余帧复用上次结果 (1=每帧都跑)
YOLO_FRAME_INTERVAL = 3
# API 视觉回退 (YOLO 不可用时, 用 Ollama vision model 做检测)
API_VISION_ENABLED = True
API_VISION_FRAME_INTERVAL = 15     # API 调用间隔 (帧), 避免频繁调用
# COCO 80类中常用物体的中文名
YOLO_CLASS_CN: dict[str, str] = {
    "cup": "杯子", "bottle": "瓶子", "wine glass": "酒杯",
    "bowl": "碗", "spoon": "勺子", "fork": "叉子", "knife": "刀",
    "apple": "苹果", "orange": "橙子", "banana": "香蕉",
    "carrot": "胡萝卜", "broccoli": "西兰花", "pizza": "披萨",
    "cake": "蛋糕", "donut": "甜甜圈", "sandwich": "三明治", "hot dog": "热狗",
    "book": "书", "scissors": "剪刀", "cell phone": "手机",
    "mouse": "鼠标", "keyboard": "键盘", "remote": "遥控器",
    "clock": "钟表", "vase": "花瓶", "teddy bear": "泰迪熊",
    "toothbrush": "牙刷", "hair drier": "吹风机",
    "laptop": "笔记本电脑", "tv": "电视", "tvmonitor": "显示器",
    "tennis racket": "网球拍", "baseball bat": "棒球棒",
    "baseball glove": "棒球手套", "sports ball": "球",
    "backpack": "背包", "umbrella": "雨伞", "handbag": "手提包",
    "suitcase": "行李箱", "tie": "领带",
}
YOLO_CN_CLASS: dict[str, str] = {v: k for k, v in YOLO_CLASS_CN.items()}

# ─── LLM ──────────────────────────────────────────────────────────────────────
LLM_SYSTEM_PROMPT = """\
你是 MeArm V1.0 机械臂的 AI 大脑. \
机械臂有 4 个关节: 底座(旋转)、肩(上下)、肘(伸缩)、爪(抓取). \
你会收到用户的命令/手势和当前可见的物体列表 (含颜色 + YOLO 类名).

【严格要求】只输出一行合法的 JSON 对象, 不要有任何前言、解释、markdown 标记或后缀.

JSON 格式:
{"action": "<pick_and_place|gesture|home|say>", "color": null, "class_name": null, "gesture": null, "message": "中文回复", "confidence": 0.95}

约束:
- action: pick_and_place | gesture | home | say
- color: 仅限 visible_objects 中列出的英文颜色名, 否则 null
- class_name: 若用户说了物体名(杯子/苹果/瓶子/书等), 填英文类名, 否则 null
- gesture: wave | point | nod | greet, 否则 null
- message: 简短口语化中文 (≤20字)
- confidence: 0.0–1.0
- 问候/你好/嗨/早上好 → action="gesture", gesture="greet", message="你好呀~"
- 谢谢/感谢 → action="say", message="不客气~"
- 再见/拜拜 → action="gesture", gesture="wave", message="再见~"
- 抓取 + 颜色 → action="pick_and_place", color=颜色名
- 抓取 + 物体名 → action="pick_and_place", class_name=物体英文名
- 回零/回家/reset → action="home"
- 无法理解 → action="say", message="抱歉，我不太明白"
- 【绝对不要输出 json 以外的任何文字】"""

# ─── 云端 LLM API (OpenAI 兼容, 可选) ───────────────────────────────────
# 设置 LLM_API_KEY 后自动使用云端 API；未设置则回退到本地 Ollama
# 支持 DeepSeek, OpenAI, 或其他 OpenAI 兼容 API
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_API_BASE_URL = os.getenv("LLM_API_BASE_URL", "https://api.deepseek.com")
LLM_API_MODEL = os.getenv("LLM_API_MODEL", "deepseek-chat")

# ─── 云端多模态视觉 API (OpenAI 兼容, 可选) ─────────────────────────────
# 设置后用于摄像头画面理解 (图片+文本联合推理)
# 支持 Kimi/Moonshot, GPT-4V, 或其他 OpenAI 兼容多模态 API
VISION_API_KEY = os.getenv("VISION_API_KEY", "")
VISION_API_BASE_URL = os.getenv("VISION_API_BASE_URL", "https://api.moonshot.cn/v1")
VISION_API_MODEL = os.getenv("VISION_API_MODEL", "kimi-k2.6")
# 视觉模型 temperature (Kimi 等模型要求 temperature=1)
VISION_API_TEMPERATURE = float(os.getenv("VISION_API_TEMPERATURE", "1.0"))

# ─── Ollama (本地 LLM) ────────────────────────────────────────────────────
# 安装 Ollama: https://ollama.com
# 拉取模型: ollama pull qwen2.5:7b
# 可选视觉模型: ollama pull llava  (用于多模态画面理解)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
# 视觉模型 (空字符串表示不启用多模态)
OLLAMA_VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", "")
LLM_TIMEOUT = 15.0
LLM_MAX_TOKENS = 512
