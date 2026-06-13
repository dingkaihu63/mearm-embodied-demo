# 🤖 MeArm V1.0 工作台系统手册

> 桌面机械臂实时调试与智能交互仪表盘  
> 更新时间: 2026-06-13

---

## 目录

1. [系统概述](#1-系统概述)
2. [快速启动](#2-快速启动)
3. [输入通道](#3-输入通道)
4. [智能决策](#4-智能决策)
5. [语音反馈](#5-语音反馈)
6. [视觉管线](#6-视觉管线)
7. [手势识别](#7-手势识别)
8. [机械臂控制](#8-机械臂控制)
9. [安全与保护](#9-安全与保护)
10. [Web 仪表盘](#10-web-仪表盘)
11. [技术栈与开源项目](#11-技术栈与开源项目)
12. [常见问题](#12-常见问题)
13. [系统架构图](#13-系统架构图)

---

## 1. 系统概述

MeArm 工作台是一个桌面机械臂的全功能调试与交互系统。通过 **摄像头 + 麦克风 + LLM + 语音合成**，实现了"说人话、看手势、做动作"的自然交互体验。

### 核心能力

- ⌨️ 文本输入 → LLM 理解 → 动作 + 语音回复
- 🎤 中英双语语音 → 离线识别 → LLM → 动作 + 语音回复
- ✋ 手部手势 → 实时分类 → LLM → 动作 + 语音回复
- 🎚️ 手动滑块/按钮 → 直接控制 + 语音确认反馈

### 硬件配置

| 组件 | 规格 |
|------|------|
| 开发板 | Arduino Uno R3 |
| 舵机 | 4 路: Base(D11), Left(D10), Right(D9), Claw(D6) |
| 波特率 | 115200 |
| 关节限位 | base/left/right: 30-150°, claw: 0-90° |
| 回零位 | base=30, left=150, right=150, claw=90 |

---

## 2. 快速启动

### 环境要求

- Python 3.10 (推荐使用 conda 环境)
- Ollama (本地 LLM, 默认方案)
- Windows 11 / Linux / macOS

### 启动步骤

```bash
# 1. 进入项目目录
cd .

# 2. 安装依赖
pip install -r requirements.txt

# 3. 安装 Ollama 并拉取模型
ollama pull qwen2.5:7b

# 4. 下载 Vosk/MediaPipe 模型 (详见 DEPENDENCIES.md)

# 5. 纯软件测试模式（不连硬件，全功能可用）
python workbench_server.py

# 6. 连 Arduino 安全模式（USB 供电，单舵机顺序运动）
python workbench_server.py --port COM3

# 7. 自动搜索摄像头 + Arduino
python workbench_server.py --port COM3 --cam auto

# 8. 外接电源全速模式
python workbench_server.py --port COM3 --unsafe

# 9. 使用手机 IP 摄像头
python workbench_server.py --ip-cam http://192.168.x.x:8080/video
```

### 可选参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--port` | Arduino 串口号 | 模拟模式 |
| `--cam` | 本地摄像头索引 | 0 |
| `--ip-cam` | 手机 IP 摄像头 URL | — |
| `--host` | Web 绑定地址 | 0.0.0.0 |
| `--web-port` | Web 端口 | 5000 |
| `--api-key` | 云端 LLM API Key | — |
| `--api-base-url` | 云端 LLM API 地址 | — |
| `--ollama-url` | Ollama API 地址 | http://localhost:11434/v1 |
| `--vision-api-key` | 云端视觉 API Key | — |
| `--no-llm` | 禁用 LLM，仅关键词模式 | — |
| `--no-voice` | 禁用语音识别 | — |
| `--no-browser` | 不自动打开浏览器 | — |
| `--unsafe` | 关闭安全模式 | — |

启动后浏览器访问: **http://localhost:5000**

---

## 3. 输入通道

### 3.1 文本指令

Web 界面底部输入框，支持中英文自然语言。

**示例**:
- "抓红色物体" / "pick red"
- "你好" / "hello"
- "挥手" / "wave"
- "回零" / "home"

### 3.2 双语语音识别

**技术**: Vosk 离线语音识别，双模型并行运行

| 语言 | 模型 | 大小 |
|------|------|------|
| 英文 | vosk-model-small-en-us-0.15 | 68 MB |
| 中文 | vosk-model-small-cn | 66 MB |

**工作原理**: 同一段麦克风音频同时送入中英文识别器，取置信度最高的结果输出。完全离线，无需联网。

**使用**: 点击 🎤 按钮变绿后，对着麦克风说话。

### 3.3 手势识别

**技术**: Google MediaPipe Hands，21 个手部关键点实时检测

**可识别手势**（本地毫秒级分类）:

| 手势 | 判断条件 | LLM 决策动作 |
|------|----------|-------------|
| 🖐 五指张开 | 五指全部伸展 | 俏皮回应 |
| ✊ 握拳 | 五指全部卷曲 | 抓取最近颜色物体 |
| 👆 食指指向 | 仅食指伸展 | 指向物体 → pick-and-place |
| 👍 竖大拇指 | 仅拇指伸展 | greet 问候手势 |
| ✌️ 剪刀手 | 食指+中指伸展 | wave 挥手 |
| 👋 挥手 | 五指张开 + 手腕规律摆动 | wave 挥手 |

**防重复机制**:
- 同一手势 2 秒冷却期
- 动作执行后额外暂停 4 秒
- `arm_busy` 期间完全跳过手势处理
- 可用 ✋ 按钮随时开关

### 3.4 手动控制

Web 界面提供:
- 四关节独立滑块 + 实时角度显示
- 一键回零、张开/闭合夹爪
- 手势库按钮 (wave/point/nod/greet)
- 颜色抓取按钮 (红/绿/蓝/黄)

---

## 4. 智能决策

### 4.1 LLM 意图解析

**模型**: Ollama / 云端 API 自适应  
**默认**: qwen2.5:7b (本地)  
**云端**: 设置 `LLM_API_KEY` 后自动切换

所有输入（文本/语音/手势）统一走 LLM 解析，输出结构化 JSON:

```json
{
  "action": "pick_and_place | gesture | home | say",
  "color": "red | green | blue | yellow | null",
  "gesture": "wave | point | nod | greet | null",
  "message": "简短口语化中文回复",
  "confidence": 0.0–1.0
}
```

### 4.2 关键词回退

LLM 不可用时（离线/超时/API 故障），自动切换到规则匹配模式:
- 关键词: pick/grab/抓/拿 + 颜色 → pick_and_place
- 关键词: wave/point/nod/greet/挥手/指/点头/问候 → gesture
- 关键词: home/reset/回 → home

### 4.3 LLM 语音回复生成

**所有语音回复统一调用 LLM 生成**，而非使用 hardcode 固定文本。

系统提示词:
> 你是 MeArm 桌面机械臂的语音助手。用一句简短、自然、口语化的中文回应。可以带一点俏皮或温暖的语气，但不要啰嗦（不超过 20 个字）。

同一个"回零"操作，LLM 可能生成 "归位啦~"、"已复位，随时待命"、"回到出发点了" 等多样化表达。

---

## 5. 语音反馈

### 5.1 Edge-TTS 神经网络语音合成

**技术**: Microsoft Edge TTS (免费，无需 API Key)

| 配置项 | 值 |
|--------|-----|
| 默认语音 | zh-CN-XiaoxiaoNeural (女声, 温暖自然) |
| 可选男声 | zh-CN-YunxiNeural (沉稳大气) |
| 可选活泼 | zh-CN-XiaoyiNeural (可爱) |
| 语速 | +10% |
| 音高 | ±0Hz |

### 5.2 语音播报覆盖范围

| 触发场景 | 是否播报 |
|----------|---------|
| 文本指令执行 | ✅ |
| 语音指令执行 | ✅ |
| 手势触发动作 | ✅ |
| 滑块移动关节 | ✅ |
| 手动按钮操作 | ✅ |
| 动作完成 | ✅ |
| 错误/失败 | ✅ |

**回退策略**: edge-tts 不可用时降级到 pyttsx3 (Windows SAPI5)。

---

## 6. 视觉管线

### 6.1 摄像头

- 支持本地 USB/内置摄像头 (DSHOW/MSMF 后端)
- 支持手机 IP 摄像头 (MJPEG/RTSP 流)
- 分辨率: 640×480
- 三路 MJPEG 输出流

### 6.2 HSV 颜色检测

| 颜色 | 色相范围 | 最小面积 |
|------|----------|---------|
| 🔴 红色 | 0-10° + 170-180° | 800 px² |
| 🟢 绿色 | 40-85° | 800 px² |
| 🔵 蓝色 | 100-130° | 800 px² |
| 🟡 黄色 | 20-35° | 800 px² |

每种颜色只保留面积最大的轮廓，Web 界面可在线调节 HSV 范围。

### 6.3 单应性变换

通过 4 个标定点将像素坐标 (px, py) 转换为世界坐标 (x_mm, y_mm)，用于 IK 逆运动学计算。

### 6.4 逆运动学 (IK)

```
输入: 目标世界坐标 (x_mm, y_mm, z_mm)
输出: 关节角度 {base, left, right}

L1 = L2 = 75mm (连杆长度)
H = 55mm (底座高度)
```

---

## 7. 手势识别

### 7.1 核心技术

**MediaPipe HandLandmarker** (Tasks API, 0.10.35)

- 模型文件: `hand_landmarker.task` (7.5 MB)
- 21 个手部关键点 (x, y, z)
- 检测置信度: 0.7, 跟踪置信度: 0.5
- 最大手数: 1

### 7.2 手指伸展判断

```
拇指: 指尖-手腕距离 > 食指MCP-手腕距离 × 1.1
其他: 指尖-MCP距离 > PIP-MCP距离 × 1.2
```

### 7.3 挥手检测

- 缓冲 15 帧手腕 x 坐标
- 统计左右摆动跨越次数 ≥ 3
- 配合五指张开状态联合判断

### 7.4 流程图

```
摄像头帧 → BGR→RGB → mp.Image → detector.detect()
  → HandLandmarkerResult.hand_landmarks[0] (21 个 NormalizedLandmark)
  → 每指伸展判断 → fingers_up 计数
  → 手势分类 (open_palm/fist/pointing/thumbs_up/peace/wave)
  → 2 秒冷却检查 → gesture_queue → LLM 决策
```

---

## 8. 机械臂控制

### 8.1 串口通信

```
Python → Arduino: "joint:angle,..." | "home" | "status"
Arduino → Python: "ACK:MOVE" | "ACK:DONE" | "ACK:HOME" | "STATUS:..." | "ERR:..."
```

### 8.2 安全模式

USB 供电时默认开启:
- 多关节命令拆分为单关节逐一发送
- 关节间隔: 80ms
- 手势步间隔: 600ms
- 抓取步间隔: 500ms

**警告**: USB 供电下绝对不要用 `--unsafe`，否则 4 个舵机同时运动会烧 USB 口。外接 5V 2A+ 电源后才可关闭安全模式。

### 8.3 Pick-and-Place 序列

```
1. 移动到物体上方 (CARRY_H=90mm)
2. 张开夹爪
3. 下降到抓取高度 (PICK_H=20mm)
4. 闭合夹爪
5. 抬起到搬运高度
6. 移动到投放点 (0, 120mm)
7. 下降到投放高度
8. 张开夹爪释放
9. 回零位
```

---

## 9. 安全与保护

| 机制 | 说明 |
|------|------|
| 🛡️ 安全模式 | 单舵机顺序运动，限流保护 |
| 🛑 急停回零 | 一键停止所有运动并归位 |
| ⏸️ 手势暂停 | 动作完成后自动暂停手势识别 4 秒 |
| 🔒 关节限位 | 所有角度钳制在安全范围内 |
| ⚠️ 超范围检测 | IK 目标超出可达范围时拒绝执行 |

---

## 10. Web 仪表盘

**地址**: http://localhost:5000

### 布局

```
┌─────────────────────┬──────────┐
│                     │ 角度仪表 │
│   摄像头 (三路)      │ 盘 +     │
│   原始/HSV/标注      │ 滑块控制 │
│                     │          │
├─────────────────────┼──────────┤
│                     │ 快捷控制 │
│   颜色检测           │ 按钮面板 │
│   结果面板           │          │
├─────────────────────┼──────────┤
│ 串口监视器           │ 手势状态 │
│ TX/RX 实时          │ + 日志   │
└─────────────────────┴──────────┘
```

### 新增按钮

- 🛑 **急停回零**: 立即停止运动，回到初始位置
- ✋ **手势识别: 开/关**: 暂停/恢复手势识别

---

## 11. 技术栈与开源项目

### 核心依赖

| 类别 | 库 | 用途 |
|------|-----|------|
| Web 框架 | **Flask** + **Flask-SocketIO** | HTTP 服务 + WebSocket 实时通信 |
| 计算机视觉 | **OpenCV** (opencv-python) | 摄像头采集、HSV 颜色空间、形态学操作 |
| 科学计算 | **NumPy** | 矩阵运算、逆运动学 |
| 串口通信 | **PySerial** | Arduino 双向通信 |
| LLM | **OpenAI Python SDK** | DeepSeek API 调用 |

### 语音相关

| 类别 | 项目 | 用途 |
|------|------|------|
| 语音识别 | **Vosk** | 离线语音识别，支持中英文双模型 |
| 音频采集 | **PyAudio** | 麦克风实时音频流 |
| 语音合成 | **Edge-TTS** | 微软免费神经网络 TTS |
| 音频播放 | **Pygame** | MP3 音频播放 |

### 手势识别

| 类别 | 项目 | 用途 |
|------|------|------|
| 手势检测 | **MediaPipe** (Google) | 手部 21 关键点实时检测 |
| 模型 | hand_landmarker.task | MediaPipe 预训练手势模型 |

### 开源项目一览

| 项目 | 官网/仓库 | 许可证 |
|------|-----------|--------|
| MediaPipe | https://github.com/google-ai-edge/mediapipe | Apache 2.0 |
| Vosk | https://github.com/alphacep/vosk-api | Apache 2.0 |
| Edge-TTS | https://github.com/rany2/edge-tts | GPLv3 |
| OpenCV | https://github.com/opencv/opencv | Apache 2.0 |
| Flask-SocketIO | https://github.com/miguelgrinberg/Flask-SocketIO | MIT |
| DeepSeek | https://api.deepseek.com | 商业 API |

---

## 12. 常见问题

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| 串口已连接但无响应 | 残留旧 Python 进程占用 | `pkill -9 -f python` 全杀后重启 |
| COM3 PermissionError | Arduino IDE 等程序占用 | 关闭所有串口工具，拔插 USB |
| MediaPipe 无法启动 | NumPy 2.x 不兼容 | `pip install "numpy<2"` |
| Vosk 模型下载极慢 | alphacephei.com 德国服务器 | 从 hf-mirror.com 下载 |
| 手势无法触发 | arm_busy 或暂停中 | 等动作完成 4 秒后再试 |
| 语音无回复 | Edge-TTS 网络问题 | 检查网络，会自动降级到 pyttsx3 |
| 摄像头无画面 | 多进程抢摄像头 | 确认只有一个 Python 进程 |
| pip 安装慢 | 默认 PyPI 源 | `-i https://pypi.tuna.tsinghua.edu.cn/simple` |

---

## 13. 系统架构图

```
┌──────────────────────────────────────────────────────────────┐
│                         输入层                                │
│                                                              │
│  ⌨️ 文本输入      🎤 麦克风          📷 摄像头               │
│  (Web 输入框)     (PyAudio)         (OpenCV)                 │
│       │               │                  │                    │
│       ▼               ▼                  ▼                    │
│  SocketIO        Vosk 双语         MediaPipe Hands           │
│  text_command    中+英并行识别      21 关键点 → 手势分类      │
│       │               │                  │                    │
│       └───────────────┼──────────────────┘                    │
│                       ▼                                       │
├──────────────────────────────────────────────────────────────┤
│                         决策层                                │
│                                                              │
│              🧠 DeepSeek LLM (deepseek-v4-flash)              │
│              ┌────────────────────────────────┐              │
│              │ 文本/语音/手势 → JSON 意图     │              │
│              │ {action, color, gesture, msg}   │              │
│              └────────────────────────────────┘              │
│                       │                                       │
│              📝 关键词规则回退 (LLM 不可用时)                  │
│                                                              │
├──────────────────────────────────────────────────────────────┤
│                         执行层                                │
│                                                              │
│     ┌──────────────┐    ┌──────────────┐    ┌──────────────┐ │
│     │  机械臂动作   │    │  语音回复     │    │  Web 推送    │ │
│     │              │    │              │    │              │ │
│     │ 🔌 PySerial  │    │ 🔊 Edge-TTS  │    │ 📡 SocketIO  │ │
│     │ Arduino 通信  │    │ 神经网络     │    │ 实时状态同步  │ │
│     │ IK 逆运动学   │    │ 自然语音     │    │              │ │
│     │ 安全模式限流   │    │ LLM 动态生成 │    │              │ │
│     └──────────────┘    └──────────────┘    └──────────────┘ │
│                                                              │
├──────────────────────────────────────────────────────────────┤
│                         监控层                                │
│                                                              │
│  📊 角度仪表盘   🎨 HSV 调节   🔌 串口监视器   📋 事件日志   │
│  🛑 急停按钮     ✋ 手势开关    📸 截图        ⏸️ 手势暂停   │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## 文件清单

```
.
├── workbench_server.py          ← 主后端 (~1900 行)
├── templates/
│   └── workbench.html           ← Web 前端
├── hand_landmarker.task         ← MediaPipe 手势模型 (7.5MB)
├── vosk-model-small-en-us-0.15/ ← 英文语音模型 (68MB)
├── vosk-model-small-cn/         ← 中文语音模型 (66MB)
├── main.ino                     ← Arduino 固件
├── servo_test.py                ← 舵机测试脚本
├── run_workbench.bat            ← Windows 启动脚本
├── requirements.txt             ← Python 依赖
├── .env.example                 ← 环境变量模板
├── README.md                    ← 项目主页
├── DEPENDENCIES.md              ← 模型下载详解
├── 技术文档.md                  ← 技术设计文档
├── 测试教程.md                   ← 首次测试教程
└── MeArm工作台系统手册.md        ← 本文档
```

---

*Generated on 2026-06-10 · MeArm Workbench v1.0*
