<!--
  MeArm Embodied Intelligence Demo
  Perception -> Reasoning -> Action | Voice · Gesture · Vision · LLM · Arduino · TTS
-->
<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.11%2B-cu128?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/CUDA-12.8-green?logo=nvidia&logoColor=white" alt="CUDA">
  <img src="https://img.shields.io/badge/Ultralytics-8.4-purple?logo=yolo&logoColor=white" alt="Ultralytics">
  <img src="https://img.shields.io/badge/Arduino-Uno%20R3-teal?logo=arduino&logoColor=white" alt="Arduino">
  <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License">
  <img src="https://img.shields.io/badge/Platform-Win11-blue?logo=windows&logoColor=white" alt="Platform">
</p>

<h1 align="center">🦾 MeArm 具身智能演示台</h1>

<p align="center">
  <b>感知 → 推理 → 行动</b>：全栈运行的具身智能最小闭环<br>
  Voice &nbsp;·&nbsp; Gesture &nbsp;·&nbsp; Vision &nbsp;·&nbsp; LLM &nbsp;·&nbsp; Serial &nbsp;·&nbsp; TTS
</p>

<p align="center">
  <i>💬 "抓红色那个" "拿杯子" "你好" "再见" &nbsp;|&nbsp; ✋ 握拳 · 张开手掌 · 食指指向</i>
</p>

---

## 📖 目录

- [💡 项目简介](#-项目简介)
- [✨ 功能矩阵](#-功能矩阵)
- [🧱 硬件搭建](#-硬件搭建)
- [📦 环境要求](#-环境要求)
- [🚀 快速开始](#-快速开始)
- [🧠 LLM 配置](#-llm-配置)
- [⌨️ 命令行参数](#️-命令行参数)
- [🎮 使用指南](#-使用指南)
- [🏗️ 系统架构](#️-系统架构)
- [🗂️ 项目结构](#️-项目结构)
- [🔧 配置参考](#-配置参考)
- [🐛 故障排查](#-故障排查)
- [📄 许可 & 致谢](#-许可--致谢)

---

## 💡 项目简介

**具身智能 (Embodied Intelligence)** 的核心命题：智能不能只活在抽象符号里——它需要一具身体，去**感知**物理世界、**理解**上下文、并**付诸行动**。

本项目用不到 **300 元** 的硬件，搭建了完整的具身智能最小闭环：

| 环节 | 含义 | 实现 | 位置 |
|:---:|------|------|:---:|
| 👁️ **感知** | 看见、听见、察觉手势 | 摄像头 + 麦克风 + MediaPipe | 本地 |
| 🧠 **推理** | 理解意图、做出决策 | Ollama / 云端 LLM | 本地/云端 |
| 🦾 **行动** | 执行物理动作 | Arduino + 4×SG90 舵机 | 本地 |
| 🔊 **反馈** | 自然的语音回应 | Edge-TTS 神经网络合成 | 本地 |

> 🎯 **定位**：可运行、可修改、可理解的动手 Demo——适合用来理解「AI × 机器人」的工作方式。

---

## ✨ 功能矩阵

### 🧠 智能交互

| 功能 | 说明 |
|------|------|
| **LLM 意图解析** | Ollama / 云端 API 自适应，NL → 结构化动作 |
| **三层意图架构** | 关键词快匹配 → 多模态视觉 → 纯文本 LLM |
| **离线语音识别** | Vosk 中/英文双模型并行，无需联网 |
| **手势控制** | MediaPipe 21 点手部关键点，6 种手势 |
| **语音合成播报** | Edge-TTS 神经网络语音，自然口语回复 |
| **自学习闭环** | TF-IDF 交互记忆 → 用户画像 → Prompt 增强 |

### 👁️ 视觉感知

| 功能 | 说明 |
|------|------|
| **YOLOv8s 检测** | 80 类物体，CUDA 加速 ~5ms/帧 |
| **HSV 颜色定位** | 红/绿/蓝/黄四色，始终运行，不依赖 GPU |
| **API 视觉回退** | 无 GPU 时自动切换到视觉 LLM |
| **三路 MJPEG 流** | 检测标注 / 原始画面 / HSV 掩膜 |

### 🛡️ 安全 & 控制

| 功能 | 说明 |
|------|------|
| **安全模式** | 逐关节步进，防 USB 过载重启 |
| **LLM 增强 IK** | 目标超出范围 → LLM 推理替代坐标 → 自动缩回 |
| **紧急停止** | 一键回零，即时中断所有运动 |
| **冷却 & 迟滞** | 手势/语音双重去重，避免误触发 |

---

## 🧱 硬件搭建

### 物料清单

> **图纸参考**：[Instructables: MeArm V1.0](https://www.instructables.com/MeArm-Robot-Arm-Your-Robot-V10/) — 含激光切割 DXF 图纸和装配步骤。

| 类别 | 物料 | 数量 | 备注 |
|------|------|:---:|------|
| 🔩 结构 | MeArm V1.0 亚克力套件 | 1 | 3mm 激光切割 |
| 🔌 主控 | Arduino Uno R3 / Nano | 1 | 或兼容板 |
| ⚙️ 舵机 | SG90 9g 微型舵机 | 4 | 底座/左臂/右臂/夹爪 |
| 📷 摄像头 | USB 摄像头 | 1 | 笔记本自带或手机 IP 摄像头 |
| 🎤 麦克风 | USB / 3.5mm | 1 | 语音识别用 |
| 🔋 电源 | 5-6V DC / 2A+ | 1 | **强烈建议外接** |
| 📎 紧固件 | M3 螺丝+螺母 | 若干 | M3×6/8/10/12/20mm |
| 🔧 可选 | Arduino 传感器扩展板 | 1 | 简化接线 |

### 舵机接线

| 舵机 | 功能 | Arduino 引脚 |
|------|------|:-----------:|
| 底座 (base) | 水平旋转 | **D11** |
| 左侧 (left) | 肩关节升降 | **D10** |
| 右侧 (right) | 肘关节伸缩 | **D9** |
| 夹爪 (claw) | 开合抓取 | **D6** |

> ⚠️ **组装前先校准舵机**：运行 `servo_test.py` 将每个舵机调到中位，再安装臂杆，避免卡死或扫齿。

### ⚡ 电源警告

```
Arduino Uno USB 5V 只能提供 ~500mA。
四个 SG90 同时启动峰值 > 1A → USB 供电会导致 Uno 重启或舵机无力。

✅ 必须外接 5-6V DC / 2A+ 电源，与 Arduino GND 共地。
   安全模式下逐关节步进可降低瞬时电流，但仍建议外接电源。
```

---

## 📦 环境要求

### 软件版本

| 组件 | 最低 | 推荐 | 说明 |
|------|:---:|:---:|------|
| 🐍 Python | 3.10 | **3.10** (Conda) | 虚拟环境推荐 |
| 🔥 PyTorch | 2.0 | **2.11+cu128** | CUDA 12.8, RTX 5060 |
| 🎯 Ultralytics | 8.0 | **8.4** | YOLOv8s |
| 🔧 CUDA | 11.8 | **12.8** | GPU 加速 (可选) |
| 🖥️ Ollama | latest | latest | 本地 LLM |
| 📦 Arduino IDE | 1.8+ | 2.x | 烧录固件 |

### Python 依赖

```bash
# 核心 (必装)
flask>=3.0.0 flask-socketio>=5.3.0   # Web 服务 + WebSocket
openai>=1.0.0 python-dotenv>=1.0.0   # LLM 客户端 + 配置

# 视觉 (必装)
opencv-python>=4.8.0 numpy>=1.24.0

# 串口 (必装)
pyserial>=3.5

# 语音识别 (必装)
vosk>=0.3.45 pyaudio>=0.2.14

# 语音合成 (必装)
edge-tts>=6.1.0 pygame>=2.5.0

# 手势识别 (必装)
mediapipe>=0.10.0

# YOLO 物体检测 (推荐，有 GPU 时)
torch>=2.0.0 ultralytics>=8.0.0

# 自学习 (可选)
scikit-learn>=1.3.0
```

### 模型文件清单

> 模型文件**不包含在仓库中**，请按指引下载。

| 模型 | 大小 | 用途 | 必装 | 安装 |
|------|------|------|:---:|------|
| **Vosk 中文语音** | ~44 MB | 中文离线 ASR | ✅ | `wget` 解压到 `models/` |
| **Vosk 英文语音** | ~41 MB | 英文离线 ASR | ✅ | `wget` 解压到 `models/` |
| **MediaPipe 手势** | ~7.5 MB | 21 关键点检测 | ✅ | `wget` 到 `models/` |
| **Ollama LLM** | ~4-8 GB | 意图解析+回复 | ✅ | `ollama pull qwen2.5:7b` |
| **YOLOv8s** | ~22 MB | 80 类物体检测 | ⭐ | 首次运行自动下载 |
| **Ollama 视觉** | ~4-7 GB | 多模态理解 | ○ | `ollama pull llava` |
| **Vosk 中文大** | ~1.4 GB | 高精度中文 ASR | ○ | `wget` 解压 |

> ✅ 必装 &nbsp; ⭐ 推荐 &nbsp; ○ 可选

---

## 🚀 快速开始

### 1. 克隆 & 安装

```bash
git clone https://github.com/dingkaihu63/mearm-embodied-demo.git
cd mearm-embodied-demo

# 推荐 Conda 虚拟环境
conda create -n py310 python=3.10 -y
conda activate py310
pip install -r requirements.txt

# 有 NVIDIA GPU 时 (推荐)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

### 2. 下载模型

```bash
mkdir -p models

# Vosk 中文语音 (必装)
wget https://alphacephei.com/vosk/models/vosk-model-small-cn-0.22.zip
unzip vosk-model-small-cn-0.22.zip -d models/ && rm vosk-model-small-cn-0.22.zip

# Vosk 英文语音 (必装)
wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
unzip vosk-model-small-en-us-0.15.zip -d models/ && rm vosk-model-small-en-us-0.15.zip

# MediaPipe 手势 (必装)
wget -O models/hand_landmarker.task   https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

### 3. 安装 Ollama & 拉取模型

从 [ollama.com](https://ollama.com) 下载安装，然后：

```bash
ollama pull qwen2.5:7b        # 必装 — 中文意图解析

# 可选视觉模型
ollama pull llava             # 多模态画面理解 (无 GPU 回退)
```

### 4. 烧录 Arduino 固件

Arduino IDE 打开 `main.ino` → 选择 **Arduino Uno** + COM 端口 → 上传。

### 5. 配置 (可选)

```bash
cp .env.example .env
# 编辑 .env 按需修改 (LLM 提供者/模型路径/串口)
```

### 6. 启动

```bash
# 模拟模式 (无硬件)
python workbench_server.py

# 连接机械臂
python workbench_server.py --port COM3

# 完整模式
python workbench_server.py --port COM3 --cam auto

# 关键词模式 (无需 LLM)
python workbench_server.py --no-llm
```

浏览器自动打开 → **http://localhost:5000**

---

## 🧠 LLM 配置

### 方案 A：本地 Ollama (默认)

```bash
# .env
OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_MODEL=qwen2.5:7b
OLLAMA_VISION_MODEL=llava:latest   # 可选
```

### 方案 B：云端 API

```bash
# .env
LLM_API_KEY=sk-your-deepseek-key
LLM_API_BASE_URL=https://api.deepseek.com
LLM_API_MODEL=deepseek-chat
```

### 方案 C：混合模式

```bash
# .env
LLM_API_KEY=sk-your-deepseek-key    # 文本用 DeepSeek
OLLAMA_VISION_MODEL=llava:latest    # 视觉用 Ollama
```

### 云端多模态视觉

```bash
# .env
VISION_API_KEY=sk-your-moonshot-key
VISION_API_BASE_URL=https://api.moonshot.cn/v1
VISION_API_MODEL=kimi-k2.6
```

### 意图解析三层架构

```
用户输入 (语音/手势/文本)
        │
        ▼
┌─ 第一层：关键词快匹配 ──────────────────┐
│  30 条规则 · 13 类别 · < 1ms · 零 API    │
│  命中 (置信度 ≥0.5) → 直接执行 ✅         │
└──────────┬──────────────────────────────┘
           │ 未命中
           ▼
┌─ 第二层：多模态视觉解析 ─────────────────┐
│  摄像头画面 + 语音 → 视觉 LLM            │
│  命中 → 执行 ｜ 未配置 → 跳过            │
└──────────┬──────────────────────────────┘
           │ 未命中
           ▼
┌─ 第三层：纯文本 LLM 解析 ────────────────┐
│  自然语言 → JSON 意图 → 执行             │
│  失败 → 友好回复                         │
└──────────────────────────────────────────┘
```

---

## ⌨️ 命令行参数

```
用法: python workbench_server.py [选项]

硬件连接:
  --port PORT          Arduino 串口 (如 COM3)
  --cam INDEX          摄像头索引 (0=默认, auto=自动搜索)
  --ip-cam URL         手机 IP 摄像头 URL

LLM 配置:
  --ollama-url URL     Ollama API 地址
  --api-key KEY        云端 LLM API Key
  --api-base-url URL   云端 LLM API 地址
  --vision-api-key KEY 云端视觉 API Key
  --no-llm             仅关键词规则模式

服务:
  --host HOST          绑定地址 (默认 0.0.0.0)
  --web-port PORT      Web 端口 (默认 5000)
  --no-browser         不自动打开浏览器

其他:
  --no-voice           禁用语音识别
  --unsafe             关闭安全模式 (需外接电源)
```

---

## 🎮 使用指南

### 🎤 语音指令

| 类别 | 示例 | 效果 |
|------|------|------|
| 🙋 社交 | `"你好"` `"早上好"` | 打招呼 |
| 🙏 礼貌 | `"谢谢"` `"辛苦了"` | 语音回复 |
| 👋 告别 | `"再见"` `"拜拜"` `"bye"` | 挥手告别 |
| 🎯 颜色抓取 | `"抓红色"` `"拿蓝色那个"` `"捡绿色"` | 抓取指定颜色 |
| 🎯 物体抓取 | `"抓杯子"` `"拿苹果"` `"捡瓶子"` | YOLO 识别抓取 |
| 🔄 旋转 | `"左转"` `"右转"` | 底座旋转 |
| 📐 升降 | `"抬高"` `"放低"` `"举起来"` | 手臂升降 |
| ↔️ 伸缩 | `"前伸"` `"后缩"` `"收回来"` | 肘部伸缩 |
| ✋ 夹爪 | `"张开"` `"闭合"` `"抓住"` | 夹爪开合 |
| 🏠 回零 | `"回家"` `"回零"` `"home"` | 回到初始位 |
| 🛑 停止 | `"停止"` `"暂停"` `"停"` | 停止运动 |
| 📊 状态 | `"你怎么样"` `"看到什么"` `"状态"` | 状态查询 |
| ❓ 帮助 | `"帮助"` `"你能做什么"` | 功能说明 |

> 💡 30 条关键词规则覆盖 13 个类别，断网也能用。

### ✋ 手势控制

| 手势 | 触发动作 |
|------|------|
| 🖐️ 五指张开 | 问候打招呼 |
| ✊ 握拳 | 抓取 (配合语音指定) |
| ☝️ 食指指向 | 指向方向 |
| 👍 竖大拇指 | 点赞回应 |
| ✌️ 剪刀手 | 庆祝 |
| 👋 挥手 | 再见 |

### 🌐 Web 仪表盘

| 面板 | 功能 |
|------|------|
| 📷 三路摄像头 | 检测标注 / 原始 / HSV 掩膜 — 标签页切换 |
| 🎚️ 关节滑块 | base/left/right/claw 实时控制 |
| 🎯 检测结果 | 物体列表 + 一键抓取按钮 |
| 🕹️ 快捷按钮 | 回零 · 夹爪 · 手势 · 急停 |
| 🎨 HSV 调节 | 弹窗在线调参，实时预览 |
| 🔌 串口监视 | TX(蓝)/RX(绿) 双向收发 |
| 📋 事件日志 | 最近 200 条，可展开 |
| 💬 语音交互 | 文本输入框 + 语音开关 |
| ✋ 手势开关 | 一键启停 |

---

## 🏗️ 系统架构

### 感知 → 推理 → 行动 闭环

```
┌──────────────────────────────────────────────────┐
│                   感知 Perception                  │
│  🎤 Vosk 语音  │  ✋ MediaPipe 手势  │  👁️ YOLO+HSV+API │
└──────────────────────┬───────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────┐
│                   推理 Reasoning                   │
│  📝 关键词快匹配 (30条)  →  🧠 LLM 意图解析        │
│  📐 LLM 增强逆运动学 (解析解 + 三层降级)           │
└──────────────────────┬───────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────┐
│                   行动 Action                      │
│  🦾 Arduino 四舵机  →  🔊 Edge-TTS 语音回复       │
│  🌐 Flask WebSocket 实时推送                      │
└──────────────────────┬───────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────┐
│                   学习 Learning                    │
│  💾 TF-IDF 交互记忆 → 👤 用户画像 → 📈 Prompt 增强 │
└──────────────────────────────────────────────────┘
```

### 视觉三层检测

```
摄像头画面 (640×480)
    │
    ├─ 第 1 层：YOLOv8s (CUDA ~5ms, 每 3 帧)
    │   └─ 可用 → 80 类检测 ✅
    │
    ├─ 第 2 层：HSV 颜色检测 (始终运行)
    │   └─ 红/绿/蓝/黄 四色定位
    │
    └─ 第 3 层：API 视觉回退 (无 GPU, 每 15 帧)
        └─ Ollama vision / 云端视觉 API
```

### 通信协议

```
Python ── TX (115200bps) ──▶ Arduino
       base:90,left:120,right:80
       home / status

Arduino ── RX ────────────▶ Python
       READY
       ACK:MOVE → ACK:DONE
       STATUS:base:85,left:90,right:92,claw:50
       ERR:SYNTAX / ERR:BUF_OVERFLOW
```

---

## 🗂️ 项目结构

```
mearm-embodied-demo/
│
├── mearm_controller/          # 🧠 核心控制 (15 模块)
│   ├── server.py              #   主入口 & 后台线程
│   ├── config.py              #   全局配置 & 环境变量
│   ├── shared_state.py        #   线程安全共享状态
│   ├── llm_parser.py          #   LLM 意图解析 (Ollama/云端自适应)
│   ├── ik_llm.py              #   LLM 增强逆运动学
│   ├── arm_ik.py              #   解析 IK 求解器
│   ├── arm_serial.py          #   Arduino 串口桥接
│   ├── vision.py              #   视觉管线
│   ├── vision_yolo.py         #   YOLOv8s 检测器
│   ├── vision_api.py          #   API 视觉回退
│   ├── voice_listener.py      #   Vosk 语音识别
│   ├── gesture_recognizer.py  #   MediaPipe 手势识别
│   ├── speaker.py             #   Edge-TTS 语音合成
│   ├── routes.py              #   Flask 路由 & SocketIO
│   └── __init__.py
│
├── mearm_learner/             # 🧠 自学习模块
│   ├── memory.py              #   TF-IDF 交互记忆
│   ├── learner.py             #   用户画像分析
│   ├── adapter.py             #   Prompt 动态增强
│   └── __init__.py
│
├── templates/
│   └── workbench.html         # 🌐 Web 仪表盘
│
├── models/                    # 📦 模型文件 (自行下载)
├── memory/                    # 💾 交互记忆 (运行时生成)
│
├── main.ino                   # 🔌 Arduino 固件
├── workbench_server.py        # 🚀 入口脚本
├── servo_test.py              # 🔧 舵机校准工具
├── run_workbench.bat          # 🪟 Windows 启动
│
├── requirements.txt           # 📦 Python 依赖
├── .env.example               # ⚙️ 环境变量模板
├── .gitignore
│
├── README.md                  # 📖 本文件
├── DEPENDENCIES.md            # 📥 模型下载详解
├── MeArm工作台系统手册.md     # 📘 系统操作手册
├── 技术文档.md                # 📕 技术设计文档
└── 测试教程.md                # 📗 首次测试教程
```

---

## 🔧 配置参考

| 变量 | 默认值 | 说明 |
|------|--------|------|
| **本地 Ollama** | | |
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1` | API 地址 |
| `OLLAMA_MODEL` | `qwen2.5:7b` | 文本模型 |
| `OLLAMA_VISION_MODEL` | (空) | 视觉模型 (`llava`) |
| **云端 API** | | |
| `LLM_API_KEY` | (空) | 设置后自动切换云端 |
| `LLM_API_BASE_URL` | `https://api.deepseek.com` | API 地址 |
| `LLM_API_MODEL` | `deepseek-chat` | 文本模型名 |
| `VISION_API_KEY` | (空) | 视觉 API Key |
| `VISION_API_BASE_URL` | `https://api.moonshot.cn/v1` | 视觉 API 地址 |
| `VISION_API_MODEL` | `kimi-k2.6` | 视觉模型名 |
| `VISION_API_TEMPERATURE` | `1.0` | 视觉 temperature |
| **Vosk** | | |
| `VOSK_MODEL_CN` | `models/vosk-model-small-cn` | 中文 ASR |
| `VOSK_MODEL_EN` | `models/vosk-model-small-en-us-0.15` | 英文 ASR |
| **YOLO** | | |
| `YOLO_MODEL` | `yolov8s.pt` | 模型文件 |
| `YOLO_DEVICE` | `cuda:0` | 推理设备 |
| `YOLO_ENABLED` | `True` | 启用开关 |
| **其他** | | |
| `FLASK_SECRET_KEY` | (自动生成) | Flask 密钥 |
| `SERIAL_PORT` | (空) | 默认串口 |

---

## 🐛 故障排查

| 现象 | 可能原因 | 解决方法 |
|------|---------|---------|
| 🔌 串口连不上 | IDE 占用/端口号错 | 关闭 Arduino IDE，确认 COM 号 |
| ⚡ 机械臂不动 | USB 供电不足 | **外接 5V 2A+ 电源** |
| 📷 摄像头黑屏 | 索引错误 | `--cam auto` 自动搜索 |
| 🎯 YOLO 不工作 | 缺 torch/ultralytics | `pip install torch ultralytics` |
| 🎤 语音无反应 | 麦克风未接 | 检查录音设备 |
| 🤖 LLM 不响应 | Ollama 未启动 | `ollama serve` |
| ⏱️ 舵机抖动 | 角度超限 | 控制在 30-150° |
| 🔴 ERR:SYNTAX | 固件不匹配 | 重新烧录 `main.ino` |
| 🟡 识别不到物体 | 光线/HSV 偏差 | Web UI → HSV 弹窗调参 |
| 🟠 手势误触发 | 光照变化 | 调整摄像头角度 |
| 🔵 ERR:BUF_OVERFLOW | 发送过快 | 增大 `SAFE_JOINT_DELAY` |
| 🔑 云端 API 不工作 | Key/网络问题 | 检查 Key 和代理设置 |

---

## 📄 许可 & 致谢

### 许可证

**MIT License** — 自由使用、修改和分发。

### 开源组件

| 项目 | 用途 | 许可 |
|------|------|------|
| [OpenCV](https://opencv.org) | 图像处理 | Apache 2.0 |
| [MediaPipe](https://developers.google.com/mediapipe) | 手势检测 | Apache 2.0 |
| [Vosk](https://alphacephei.com/vosk) | 语音识别 | Apache 2.0 |
| [Ultralytics](https://docs.ultralytics.com) | YOLO 检测 | AGPL-3.0 |
| [PyTorch](https://pytorch.org) | 深度学习 | BSD |
| [Ollama](https://ollama.com) | 本地 LLM | MIT |
| [Edge-TTS](https://github.com/rany2/edge-tts) | 语音合成 | GPL 3.0 |
| [Flask](https://flask.palletsprojects.com) | Web 框架 | BSD |
| [MeArm](https://mearm.com) | 机械臂设计 | CC-BY-SA |

### 模型来源

| 模型 | 来源 |
|------|------|
| Vosk 语音模型 | [alphacephei.com/vosk/models](https://alphacephei.com/vosk/models) |
| MediaPipe 手势 | [Google MediaPipe](https://storage.googleapis.com/mediapipe-models) |
| YOLOv8s | [Ultralytics](https://docs.ultralytics.com) (首次运行自动下载) |

---

<p align="center">
  <br>
  <i>🦾 Built with curiosity · Powered by open source · 2026</i>
</p>
