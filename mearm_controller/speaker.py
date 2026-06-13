"""
MeArm 工作台 — 语音播报 (Speaker)
================================
从原 workbench_server.py 提取，不做任何修改。
"""

from __future__ import annotations

import logging
import os
import queue
import tempfile
import threading
import time
from typing import Optional

from .config import EDGE_VOICE, EDGE_TTS_RATE, EDGE_TTS_PITCH, log
from .shared_state import state


# ══════════════════════════════════════════════════════════════════════════════
# 语音播报 (Edge-TTS 神经网络语音 + pygame 播放)
# ══════════════════════════════════════════════════════════════════════════════

class Speaker:
    """使用 Edge-TTS (微软免费神经网络语音) 生成自然语音, pygame 播放.

    回退策略: edge-tts 不可用 → pyttsx3 (SAPI5)
    """

    def __init__(self):
        self._queue = queue.Queue()
        self._running = False
        self._thread = None
        self._engine = None  # 'edge' | 'pyttsx3' | None

        # 优先用 edge-tts
        try:
            import edge_tts
            import pygame
            # 初始化 pygame mixer (只用一次)
            pygame.mixer.init(frequency=24000, size=-16, channels=1)
            self._engine = "edge"
            log.info("🔊 Edge-TTS 神经网络语音已就绪 (zh-CN-XiaoxiaoNeural)")
        except ImportError:
            log.warning("edge-tts 或 pygame 未安装 — 尝试 pyttsx3 回退")
            try:
                import pyttsx3
                self._engine = "pyttsx3"
                log.info("🔊 pyttsx3 (SAPI5) 语音已就绪")
            except ImportError:
                log.warning("pyttsx3 也未安装 — TTS 不可用")

    def _synthesize_edge(self, text: str) -> Optional[bytes]:
        """用 edge-tts 合成语音, 返回 MP3 字节. (在线, 免费)"""
        import edge_tts
        import asyncio

        async def _synth():
            communicate = edge_tts.Communicate(
                text, EDGE_VOICE,
                rate=EDGE_TTS_RATE,
                pitch=EDGE_TTS_PITCH,
            )
            # 收集所有音频块
            chunks = []
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    chunks.append(chunk["data"])
            return b"".join(chunks)

        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            data = loop.run_until_complete(_synth())
            loop.close()
            return data if data else None
        except Exception as e:
            log.error(f"Edge-TTS 合成失败: {e}")
            return None

    def _play_mp3_bytes(self, mp3_data: bytes):
        """用 pygame 播放内存中的 MP3 数据."""
        import pygame
        tmp_path = ""
        try:
            # 使用唯一临时文件名 (避免多线程并发冲突)
            tmp_fd, tmp_path = tempfile.mkstemp(suffix=".mp3", prefix="_mearm_tts_")
            os.close(tmp_fd)
            with open(tmp_path, "wb") as f:
                f.write(mp3_data)
            pygame.mixer.music.load(tmp_path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.05)
        except Exception as e:
            log.error(f"pygame 播放失败: {e}")
        finally:
            if tmp_path:
                try:
                    time.sleep(0.1)  # 确保 pygame 释放文件句柄
                    os.remove(tmp_path)
                except Exception:
                    pass

    def _worker(self):
        while self._running or not self._queue.empty():
            try:
                text = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if self._engine == "edge":
                mp3_data = self._synthesize_edge(text)
                if mp3_data:
                    self._play_mp3_bytes(mp3_data)
                else:
                    # edge-tts 失败, 静默丢弃 (日志已记录)
                    pass
            elif self._engine == "pyttsx3":
                import pyttsx3
                try:
                    engine = pyttsx3.init()
                    voices = engine.getProperty('voices')
                    for voice in voices:
                        if any(k in voice.name.lower()
                               for k in ['chinese', 'zh', 'huihui', 'mandarin']):
                            engine.setProperty('voice', voice.id)
                            break
                    engine.setProperty('rate', 160)
                    engine.say(text)
                    engine.runAndWait()
                except Exception as e:
                    log.error(f"pyttsx3 TTS 错误: {e}")

    def start(self):
        if self._engine is None:
            state.add_log("⚠️ TTS 不可用 (请安装: pip install edge-tts pygame)")
            return
        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        voice_name = "Edge-TTS 神经网络" if self._engine == "edge" else "pyttsx3 SAPI5"
        state.add_log(f"🔊 语音播报已就绪 ({voice_name})")

    def stop(self):
        self._running = False

    def speak(self, text: str):
        if text and self._engine is not None:
            self._queue.put(text)
