"""
MeArm 工作台 — 语音识别 (VoiceListener)
======================================
Vosk 双模型并行: 中文 + 英文离线识别。

稳定性优化:
  - 最低置信度门槛 (VOICE_MIN_CONFIDENCE)
  - 最短文本长度过滤
  - 去重窗口 (同文本短期内不再发送)
  - 部分结果积累 → 最终结果合并
"""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
from typing import Optional

from .config import (
    log,
    VOICE_MIN_CONFIDENCE,
    VOICE_MIN_CONFIDENCE_CN,
    VOICE_MIN_CONFIDENCE_EN,
    VOICE_MIN_TEXT_LENGTH,
    VOICE_DEDUP_WINDOW,
    VOICE_EN_DEFAULT_CONFIDENCE,
)
from .shared_state import state


class VoiceListener:
    """双语言语音识别: 中文 + 英文 Vosk 模型并行, 综合评分取最优.

    优化点:
      - 语言分阈值: 中文 0.35 / 英文 0.45 (适配小模型)
      - 综合评分: 置信度 + 文本长度 + 中文偏好, 防止英文噪音抢走中文
      - 只保留最新 partial (不累积膨胀), 超时 2s 自动清理
      - 跳过短于 VOICE_MIN_TEXT_LENGTH 字符的噪音片段
      - VOICE_DEDUP_WINDOW 秒内相同文本不重复发送
    """

    MODEL_EN = os.getenv("VOSK_MODEL_EN", "models/vosk-model-small-en-us-0.15")
    MODEL_CN = os.getenv("VOSK_MODEL_CN", "models/vosk-model-small-cn")

    def __init__(self, text_queue: queue.Queue, sample_rate: int = 16000):
        self._q = text_queue
        self._sr = sample_rate
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._available = False
        self._langs: list[str] = []

        # ── 去重状态 ──────────────────────────────────────────────────────
        self._last_text = ""
        self._last_text_time = 0.0

        # ── 部分结果累积 (只保留最新 partial, 超时自动清理) ─────────────
        self._partials: dict[str, list[str]] = {}  # lang → [latest_partial]

        try:
            import vosk
            import pyaudio
            self._available = True
        except ImportError:
            log.warning("Vosk 或 PyAudio 未安装 — 语音识别不可用")

    def _worker(self):
        import vosk
        import pyaudio

        models: dict[str, vosk.Model] = {}
        recs: dict[str, vosk.KaldiRecognizer] = {}

        for lang, path in [("en", self.MODEL_EN), ("cn", self.MODEL_CN)]:
            try:
                models[lang] = vosk.Model(path)
                recs[lang] = vosk.KaldiRecognizer(models[lang], self._sr)
                self._langs.append(lang)
                log.info(f"Vosk 模型已加载: {path}")
            except Exception:
                log.warning(f"Vosk 模型未找到: '{path}' (跳过 {lang})")

        if not models:
            log.error("没有可用的语音模型")
            state.add_log("⚠️ 语音识别不可用 (模型缺失)")
            return

        pa = pyaudio.PyAudio()
        stream = pa.open(format=pyaudio.paInt16, channels=1, rate=self._sr,
                         input=True, frames_per_buffer=4096)
        state.add_log(f"🎤 语音监听已启动 (语言: {'+'.join(self._langs)})")
        log.info(f"语音监听已启动 — 请说话 (语言: {self._langs})")

        # ── partial 超时清理 (秒) ─────────────────────────────────────────
        PARTIAL_TIMEOUT = 2.0
        _last_partial_time: dict[str, float] = {}

        while self._running:
            try:
                data = stream.read(4096, exception_on_overflow=False)
            except Exception as e:
                log.error(f"语音读取错误: {e}")
                break

            best_text = ""
            best_score = -1.0  # 综合评分 (置信度 + 文本长度加权)
            best_conf_raw = -1.0  # 原始置信度 (用于阈值检查)
            best_lang = ""
            now_ts = time.time()

            # ── 清理过期的 partial ─────────────────────────────────────────
            for lang in list(self._partials.keys()):
                if now_ts - _last_partial_time.get(lang, 0) > PARTIAL_TIMEOUT:
                    self._partials[lang] = []

            for lang, rec in recs.items():
                if rec.AcceptWaveform(data):
                    # ── 最终结果 ──────────────────────────────────────────────
                    raw = rec.Result()
                    try:
                        result = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    text = result.get("text", "").strip()
                    # 中文模型有 confidence; 英文小模型无, 用固定默认值
                    conf = result.get("confidence", VOICE_EN_DEFAULT_CONFIDENCE if lang == "en" else VOICE_MIN_CONFIDENCE)

                    # 只取最新 partial (最完整), 不用累积的所有
                    prev_partials = self._partials.get(lang, [])
                    latest_partial = prev_partials[-1] if prev_partials else ""

                    if latest_partial and len(latest_partial) > len(text):
                        # partial 比 final 更完整 (Vosk 常见现象)
                        combined = latest_partial
                    elif text:
                        combined = text
                    else:
                        combined = ""

                    self._partials[lang] = []
                    _last_partial_time.pop(lang, None)

                    if combined.strip():
                        # 综合评分: 置信度 + 短文本惩罚 (防止2字以下的噪音),
                        #            + 语言偏好 (中文优先, 因为用户说中文)
                        text_len_bonus = min(len(combined) / 10.0, 0.15)  # 最长词给 +0.15
                        lang_bonus = 0.10 if lang == "cn" else 0.0
                        score = conf + text_len_bonus + lang_bonus
                        if score > best_score:
                            best_text = combined.strip()
                            best_score = score
                            best_conf_raw = conf
                            best_lang = lang
                else:
                    # ── 部分结果 ──────────────────────────────────────────────
                    partial_raw = rec.PartialResult()
                    try:
                        partial = json.loads(partial_raw)
                    except json.JSONDecodeError:
                        continue
                    ptext = partial.get("partial", "").strip()
                    if ptext:
                        # 只保留最新 partial (覆盖而非追加)
                        self._partials[lang] = [ptext]
                        _last_partial_time[lang] = now_ts

            # ── 发送最终结果 (带过滤) ───────────────────────────────────────
            if best_text:
                if self._should_accept(best_text, best_lang, best_conf_raw):
                    state.add_log(f"🎤 听到: '{best_text}' (conf={best_conf_raw:.2f}, score={best_score:.2f}, lang={best_lang})")
                    self._q.put(best_text)
                    self._last_text = best_text
                    self._last_text_time = time.time()
                else:
                    log.info(f"🎤 过滤: '{best_text}' (conf={best_conf_raw:.2f}, lang={best_lang}, 不满足门槛)")

        stream.stop_stream()
        stream.close()
        pa.terminate()
        state.add_log("🎤 语音监听已停止")

    def _should_accept(self, text: str, lang: str = "", raw_conf: float = -1.0) -> bool:
        """综合判断是否接受此识别结果.

        Args:
            text: 识别的文本
            lang: 来源语言 ("cn" | "en" | ""), 用于选择对应阈值
            raw_conf: 原始置信度 (不含 score 加成)
        """
        now = time.time()

        # 1. 语言特定的置信度门槛 (用原始置信度)
        if lang == "cn":
            min_conf = VOICE_MIN_CONFIDENCE_CN
        elif lang == "en":
            min_conf = VOICE_MIN_CONFIDENCE_EN
        else:
            min_conf = VOICE_MIN_CONFIDENCE

        if raw_conf >= 0 and raw_conf < min_conf:
            return False

        # 2. 最短文本长度 (过滤单字噪音)
        if len(text) < VOICE_MIN_TEXT_LENGTH:
            return False

        # 3. 去重: 相同文本在 DEDUP_WINDOW 秒内不重复
        if text == self._last_text and (now - self._last_text_time) < VOICE_DEDUP_WINDOW:
            log.debug(f"🎤 去重: 跳过重复文本 '{text}'")
            return False

        return True

    def start(self):
        if not self._available:
            state.add_log("⚠️ 语音识别不可用 (缺失 vosk/pyaudio)")
            return
        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
