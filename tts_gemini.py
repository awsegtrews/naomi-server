"""Живий голос Наомі — Gemini TTS (безкоштовний ключ Google AI Studio): емоції, звуки <pff>, <giggle>…

Говорить потоком: перший звук ~1.5 с, далі синтез удвічі швидший за мовлення. Gemini віддає PCM 24 кГц —
переводимо в 16 кГц для Cardputer.

Безкоштовно Google дає ~10 фраз на добу КОЖНІЙ TTS-моделі, тож перебираємо моделі по черзі (вичерпана
відпочиває, скільки скаже Google). Далі — Live-моделі (ліміти окремі): той самий голос читає текст дослівно.
Коли вичерпано все — device.py говорить звичайним голосом Edge.
"""
import asyncio
import base64
import json
import logging
import os
import re
import time

import httpx
import numpy as np
import websockets

from textutil import clean

log = logging.getLogger("naomi.tts")
KEY = os.environ.get("GEMINI_API_KEY", "")
# черга голосів: першим — Live 3.1 (обрав власник, найшвидший), далі TTS-моделі, далі інші Live
ORDER = os.environ.get("NAOMI_VOICE_ORDER", "gemini-3.1-flash-live-preview,gemini-3.8-flash-tts,"
                       "gemini-3.8-flash-lite-tts,gemini-3.1-flash-tts-preview,gemini-2.5-flash-preview-tts,"
                       "gemini-3.8-live,gemini-2.5-flash-native-audio-latest").split(",")


def _is_live(model: str) -> bool:
    return "live" in model or "native-audio" in model
LIVE_URL = ("wss://generativelanguage.googleapis.com/ws/"
            "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent")
VOICE = os.environ.get("NAOMI_GEMINI_VOICE", "Leda")
USE_FOR = os.environ.get("NAOMI_GEMINI_FOR", "all")  # all — завжди, tsundere — лише для цундере, none — ніколи
NEW_API = ("gemini-3.8",)  # ці моделі розуміють speech_metadata.style і звуки <pff>; старші — опис стилю словами

# звуки, які модель може вставляти у відповідь (решту тегів прибираємо)
SOUNDS = ["pff", "tsk", "giggle", "chuckle", "laugh", "sigh", "gasp", "phew", "groan", "grr", "whispers",
          "short pause", "long pause"]
_SOUND = re.compile(r"<\s*(" + "|".join(SOUNDS) + r")\s*>", re.I)
_ANY_TAG = re.compile(r"<[^<>]{1,24}>")
# Live-моделі тегів не розуміють — кажемо звук словом
_SOUND_WORDS = {"pff": "Пф!", "tsk": "Тц.", "giggle": "Хі-хі.", "chuckle": "Хе-хе.", "laugh": "Ха-ха!",
                "sigh": "Ех…", "gasp": "Ах!", "phew": "Фух!", "groan": "Ох…", "grr": "Р-р-р!",
                "short pause": "…", "long pause": "…", "whispers": ""}

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

_resting: dict[str, float] = {}  # модель -> до коли не пробувати (вичерпала ліміт)


class Unavailable(Exception):
    pass


class _ModelRefused(Exception):
    pass


def enabled(style: str) -> bool:
    if not KEY or not (USE_FOR == "all" or (USE_FOR == "tsundere" and style == "tsundere")):
        return False
    now = time.time()
    return any(now >= _resting.get(m, 0) for m in ORDER)


def sounds_as_words(text: str) -> str:
    return re.sub(r"\s{2,}", " ", _SOUND.sub(lambda m: _SOUND_WORDS[m.group(1).lower()], text)).strip()


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


def _body(model: str, text: str, how: str) -> dict:
    if model.startswith(NEW_API):
        return {"contents": [{"role": "user", "parts": [{"text": text, "speech_metadata": {"style": how}}]}],
                "generationConfig": {"responseModalities": ["AUDIO"],
                                     "speechConfig": {"voiceConfig": {"voice": VOICE}}}}
    return {"contents": [{"role": "user", "parts": [{"text": f"Say it as {how}: {strip_tags(text)}"}]}],
            "generationConfig": {"responseModalities": ["AUDIO"],
                                 "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": VOICE}}}}}


def _rest(model: str, status: int, detail: str) -> None:
    """Скільки моделі відпочивати: Google сам підказує retryDelay; невідома помилка — кілька хвилин."""
    seconds = 300
    try:
        for d in json.loads(detail).get("error", {}).get("details", []):
            if "retryDelay" in d:
                seconds = int(float(d["retryDelay"].rstrip("s"))) + 60
    except (ValueError, AttributeError):
        pass
    if status == 400:  # модель не приймає такий запит — не мучимо її до перезапуску
        seconds = 24 * 3600
    _resting[model] = time.time() + seconds
    log.warning("%s: %s — відпочиває %d хв", model, status, seconds // 60)


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


async def _stream_model(client: httpx.AsyncClient, model: str, text: str, how: str):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent"
    rs = _Resampler()
    async with client.stream("POST", url, params={"alt": "sse"}, headers={"x-goog-api-key": KEY},
                             json=_body(model, text, how)) as r:
        if r.status_code != 200:
            detail = (await r.aread()).decode(errors="replace")
            _rest(model, r.status_code, detail)
            raise _ModelRefused(f"{model} {r.status_code}")
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


LIVE_TONES = {  # для Live — лише тон голосу, без ролі персонажа (з роллю модель починає імпровізувати)
    "tsundere": {
        "angry": "annoyed, pouty and huffy, a bit faster", "shy": "flustered, embarrassed and hesitant, soft",
        "kiss": "shy and affectionate, soft and warm, a little embarrassed", "happy": "cheerful, playful, teasing",
        "sad": "gentle, caring and sincere, calm", "default": "lively and expressive, a bit teasing and grumpy",
    },
    "normal": {
        "happy": "cheerful and warm", "sad": "gentle, caring and calm", "default": "warm, natural and lively",
    },
}


async def _stream_live(model: str, text: str, style: str, mood: str):
    """Live-модель як рушій озвучення: читає текст дослівно тим самим голосом."""
    tones = LIVE_TONES.get(style, LIVE_TONES["normal"])
    system = ("You are a text-to-speech engine. Speak the user's message aloud word for word in Ukrainian, then stop. "
              "Say nothing else: no greetings, no comments, no extra words before or after, no translation. "
              f"Tone of voice: {tones.get(mood, tones['default'])}.")
    rs = _Resampler()
    try:
        async with websockets.connect(LIVE_URL, additional_headers={"x-goog-api-key": KEY}, max_size=None,
                                      open_timeout=10) as ws:
            await ws.send(json.dumps({"setup": {
                "model": f"models/{model}",
                "generationConfig": {"responseModalities": ["AUDIO"],
                                     "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": VOICE}}}},
                "systemInstruction": {"parts": [{"text": system}]}}}))
            ready = json.loads(await asyncio.wait_for(ws.recv(), 10))
            if "setupComplete" not in ready:
                _rest(model, 400, "{}")
                raise _ModelRefused(f"{model} setup")
            await ws.send(json.dumps({"clientContent": {
                "turns": [{"role": "user", "parts": [{"text": sounds_as_words(text)}]}], "turnComplete": True}}))
            while True:
                msg = json.loads(await asyncio.wait_for(ws.recv(), 20))
                content = msg.get("serverContent", {})
                for p in content.get("modelTurn", {}).get("parts", []):
                    data = (p.get("inlineData") or {}).get("data")
                    if data:
                        out = rs.feed(np.frombuffer(base64.b64decode(data), "<i2"))
                        if len(out):
                            yield out
                if content.get("turnComplete") or "goAway" in msg:
                    return
    except websockets.ConnectionClosed as e:
        reason = (e.rcvd.reason if e.rcvd else "") or type(e).__name__
        # вичерпаний ліміт Live закриває з'єднання (1011 «Resource has been exhausted»)
        _resting[model] = time.time() + (3600 if "exhaust" in reason.lower() or "quota" in reason.lower() else 300)
        log.warning("%s: закрито (%s) — відпочиває", model, reason[:120])
        raise _ModelRefused(f"{model} {reason[:60]}") from e
    except (OSError, asyncio.TimeoutError, websockets.InvalidHandshake) as e:
        _resting[model] = time.time() + 300
        raise _ModelRefused(f"{model} {type(e).__name__}") from e


async def stream(text: str, style: str, mood: str):
    """Віддає PCM16 16 кГц шматками, щойно Gemini їх генерує. Unavailable — якщо жодна модель не змогла."""
    how, refused = style_for(style, mood), []
    async with httpx.AsyncClient(timeout=httpx.Timeout(20.0)) as client:
        for model in ORDER:
            if time.time() < _resting.get(model, 0):
                continue
            started = False
            source = (_stream_live(model, text, style, mood) if _is_live(model)
                      else _stream_model(client, model, text, how))
            try:
                async for pcm in source:
                    started = True
                    yield pcm
                return
            except _ModelRefused as e:
                if started:  # Live обірвався посеред фрази — договорювати іншою моделлю не будемо
                    raise Unavailable(str(e)) from e
                refused.append(str(e))  # нічого не прозвучало — пробуємо наступну модель
            except (httpx.HTTPError, json.JSONDecodeError) as e:
                if started:
                    raise Unavailable(f"{model} обірвався: {type(e).__name__}") from e
                refused.append(f"{model} {type(e).__name__}")
    raise Unavailable("; ".join(refused) or "усі моделі відпочивають")
