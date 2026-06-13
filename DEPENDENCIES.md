# 依赖模型清单 / Dependency Models Checklist

本项目使用以下预训练模型，由于体积较大，不包含在代码库中，请自行下载：

## 必装模型 / Required

| 模型 | 文件目录 | 大小 | 用途 | 下载地址 |
|------|---------|------|------|----------|
| Vosk 中文语音 | `models/vosk-model-small-cn/` | ~44MB | 中文离线语音识别 | [下载](https://alphacephei.com/vosk/models/vosk-model-small-cn-0.22.zip) |
| Vosk 英文语音 | `models/vosk-model-small-en-us-0.15/` | ~41MB | 英文离线语音识别 | [下载](https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip) |
| MediaPipe 手势 | `models/hand_landmarker.task` | ~7.5MB | 手势关键点检测 | [下载](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task) |
| Ollama LLM | 通过 Ollama 拉取 | ~4-8GB | 意图解析 | `ollama pull qwen2.5:7b` |

## 可选模型 / Optional

| 模型 | 文件目录 | 大小 | 用途 | 下载地址 |
|------|---------|------|------|----------|
| Vosk 中文大模型 | `models/vosk-model-cn-0.22/` | ~1.4GB | 更高精度的中文语音识别 | [下载](https://alphacephei.com/vosk/models/vosk-model-cn-0.22.zip) |
| Ollama 视觉模型 | 通过 Ollama 拉取 | ~4-7GB | 多模态画面理解 | `ollama pull llava` |

## 安装步骤 / Setup Instructions

### 1. 下载并解压 Vosk 模型

```bash
# 中文小模型 (必装)
wget https://alphacephei.com/vosk/models/vosk-model-small-cn-0.22.zip
unzip vosk-model-small-cn-0.22.zip -d models/
rm vosk-model-small-cn-0.22.zip

# 英文小模型 (必装)
wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
unzip vosk-model-small-en-us-0.15.zip -d models/
rm vosk-model-small-en-us-0.15.zip
```

### 2. 下载 MediaPipe 手势模型

```bash
wget -O models/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

### 3. 安装 Ollama 并拉取模型

从 [ollama.com](https://ollama.com) 下载安装 Ollama，然后：

```bash
ollama pull qwen2.5:7b        # 必装 — 中文意图解析
ollama pull llava             # 可选 — 多模态视觉理解
```

### 4. 也可通过环境变量自定义模型路径

```ini
# .env
VOSK_MODEL_EN=models/vosk-model-small-en-us-0.15
VOSK_MODEL_CN=models/vosk-model-small-cn
OLLAMA_MODEL=qwen2.5:7b
OLLAMA_VISION_MODEL=llava
```
