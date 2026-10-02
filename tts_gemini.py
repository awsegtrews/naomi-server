"""Живий голос Наомі — Gemini TTS (безкоштовний ключ Google AI Studio): емоції, звуки <pff>, <giggle>…

Говорить потоком: перший звук ~1.5 с, далі синтез удвічі швидший за мовлення. Gemini віддає PCM 24 кГц —
переводимо в 16 кГц для Cardputer. Якщо Gemini недоступний (ліміт, мережа) — device.py говорить голосом Edge.
"""
import base64
import json
import logging
import os
import re
import time

import httpx
import numpy as np

from textutil import clean

log = logging.getLogger("naomi.tts")
KEY = os.environ.get("GEMINI_API_KEY", "")
MODEL = os.environ.get("NAOMI_TTS_MODEL", "gemini-3.8-flash-tts")
VOICE = os.environ.get("NAOMI_GEMINI_VOICE", "Leda")
USE_FOR = os.environ.get("NAOMI_GEMINI_FOR", "all")  # all — завжди, tsundere — лише для цундере, none — ніколи
URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:streamGenerateContent"

# звуки, які модель може вставляти у відповідь (решту тегів прибираємо)
SOUNDS = ["pff", "tsk", "giggle", "chuckle", "laugh", "sigh", "gasp", "phew", "groan", "grr", "whispers",
          "short pause", "long pause"]
_SOUND = re.compile(r"<\s*(" + "|".join(SOUNDS) + r")\s*>", re.I)
_ANY_TAG = re.compile(r"<[^<>]{1,24}>")

STYLES = {  # характер -> настрій відповіді -> як говорити
    "tsundere": {
        "angry": "a tsundere anime girl, annoyed and pouty, huffy, speaking a bit faster",
        "shy": "a tsundere anime girl, flustered and embarrassed, soft and hesitant",
        "kiss": "a tsundere anime girl, shy and affectionate, soft and warm, a little embarrassed",
        "happy": "a tsundere anime girl in a good mood, lively, playful and teasing",
        "sad": "gentle, caring and sincere, calm and warm",
        "default": "a tsundere anime girl, lively, a bit grumpy but warm",
    },
    "normal": {
        "happy": "a friendly young woman, warm, cheerful and lively",
        "sad": "gentle, caring and sincere, calm",
        "default": "a friendly young woman, warm and natural, conversational",
    },
}

_cooldown_until = 0.0  # після ліміту якийсь час говоримо звичайним голосом, щоб не чекати марно


class Unavailable(Exception):
    pass


def enabled(style: str) -> bool:
    return (bool(KEY) and (USE_FOR == "all" or (USE_FOR == "tsundere" and style == "tsundere"))
            and time.time() >= _cooldown_until)


def strip_tags(text: str) -> str:
    """Для екрана й звичайного голосу: без <звуків>."""
    return re.sub(r"\s{2,}", " ", _ANY_TAG.sub(" ", text)).strip()


def prepare(text: str) -> str:
    """Для Gemini: чистий текст + дозволені звуки в кутових дужках."""
    out, pos = [], 0
    for m in _ANY_TAG.finditer(text):
        out.append(clean(text[pos:m.start()]))
        sound = _SOUND.fullmatch(m.group(0))
        if sound:
            out.append(f"<{sound.group(1).lower()}>")
        pos = m.end()
    out.append(clean(text[pos:]))
    return " ".join(p for p in out if p).strip()


def style_for(style: str, mood: str) -> str:
    table = STYLES.get(style, STYLES["normal"])
    return table.get(mood, table["default"]) + "; speaking Ukrainian"


class _Resampler:
    """24 кГц -> 16 кГц потоково: фільтр нижніх частот (~7 кГц, щоб не було свисту) + інтерполяція 3:2."""

    def __init__(self) -> None:
        n = np.arange(31) - 15
        h = 2 * 0.3 * np.sinc(2 * 0.3 * n) * np.hamming(31)
        self.h = (h / h.sum()).astype(np.float32)
        self.hist = np.zeros(30, np.float32)
        self.buf = np.zeros(0, np.float32)
        self.pos = 0.0

    def feed(self, pcm: np.ndarray) -> np.ndarray:
        x = np.concatenate([self.hist, pcm.astype(np.float32)])
        self.hist = x[-30:]
        self.buf = np.concatenate([self.buf, np.convolve(x, self.h, mode="valid")])
        if len(self.buf) - 1 < self.pos:
            return np.zeros(0, np.int16)
        n = int((len(self.buf) - 1 - self.pos) // 1.5) + 1
        idx = self.pos + 1.5 * np.arange(n)
        i0 = idx.astype(np.int64)
        frac = (idx - i0).astype(np.float32)
        out = self.buf[i0] * (1 - frac) + self.buf[np.minimum(i0 + 1, len(self.buf) - 1)] * frac
        nxt = self.pos + 1.5 * n
        self.buf, self.pos = self.buf[int(nxt):], nxt - int(nxt)
        return np.clip(out, -32768, 32767).astype(np.int16)


async def stream(text: str, style: str, mood: str):
    """Віддає PCM16 16 кГц шматками, щойно Gemini їх генерує. Unavailable — якщо не вдалося."""
    global _cooldown_until
    body = {
        "contents": [{"role": "user", "parts": [{"text": text, "speech_metadata": {"style": style_for(style, mood)}}]}],
        "generationConfig": {"responseModalities": ["AUDIO"], "speechConfig": {"voiceConfig": {"voice": VOICE}}},
    }
    rs = _Resampler()
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(20.0)) as client:
            async with client.stream("POST", URL, params={"alt": "sse"}, headers={"x-goog-api-key": KEY},
                                     json=body) as r:
                if r.status_code != 200:
                    detail = (await r.aread())[:400].decode(errors="replace")
                    if r.status_code == 429:  # ліміт: денний — чекаємо довше
                        _cooldown_until = time.time() + (3 * 3600 if "PerDay" in detail else 600)
                    raise Unavailable(f"{r.status_code}: {detail}")
                async for line in r.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    d = json.loads(line[5:])
                    for p in ((d.get("candidates") or [{}])[0].get("content") or {}).get("parts", []):
                        data = (p.get("inlineData") or {}).get("data")
                        if data:
                            out = rs.feed(np.frombuffer(base64.b64decode(data), "<i2"))
                            if len(out):
                                yield out
    except httpx.HTTPError as e:
        raise Unavailable(type(e).__name__) from e
