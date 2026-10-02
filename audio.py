"""Звук: розпаковка µ-law від Cardputer, WAV для Whisper, голос Наомі по реченнях.

Голос по реченнях — Google Chirp 3 HD (той самий Leda, що й живий голос Gemini, без ліміту на добу),
а без його ключа чи при збої — Edge (Поліна). Живий голос з емоціями — у tts_gemini.py.
"""
import base64
import io
import logging
import os
import re
import time
import wave

import edge_tts
import httpx
import miniaudio
import numpy as np

log = logging.getLogger("naomi.audio")
SAMPLE_RATE = 16000
VOICE = os.environ.get("NAOMI_VOICE", "uk-UA-PolinaNeural")
VOICE_RATE = os.environ.get("NAOMI_VOICE_RATE", "+8%")
CHIRP_KEY = os.environ.get("GOOGLE_TTS_KEY", "")
CHIRP_VOICE = os.environ.get("NAOMI_CHIRP_VOICE", "uk-UA-Chirp3-HD-Leda")
_chirp_resting = 0.0  # після помилки якийсь час не пробуємо


def _mulaw_table() -> np.ndarray:
    u = ~np.arange(256, dtype=np.int32) & 0xFF
    sign = u & 0x80
    exponent = (u >> 4) & 0x07
    mantissa = u & 0x0F
    sample = (((mantissa << 3) + 0x84) << exponent) - 0x84
    return np.where(sign != 0, -sample, sample).astype(np.int16)


_MULAW = _mulaw_table()


def mulaw_to_pcm16(data: bytes) -> np.ndarray:
    return _MULAW[np.frombuffer(data, dtype=np.uint8)]


def pcm16_to_wav(pcm: np.ndarray, rate: int = SAMPLE_RATE) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.astype("<i2").tobytes())
    return buf.getvalue()


# вигуки, які синтезатор читає по літерах («Хм» -> «Ха-ем»), — заміняємо на ті, що звучать
_INTERJECTIONS = [(re.compile(r"\b([Хх])м+\b"), lambda m: "Гм" if m.group(1) == "Х" else "гм"),
                  (re.compile(r"\b([Пп])ф+\b"), lambda m: "Тю" if m.group(1) == "П" else "тю")]


async def _chirp(text: str) -> bytes | None:
    """Google Cloud Chirp 3 HD -> PCM 16 кГц; None — немає ключа або збій (тоді говорить Edge)."""
    global _chirp_resting
    if not CHIRP_KEY or time.time() < _chirp_resting:
        return None
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.post("https://texttospeech.googleapis.com/v1/text:synthesize",
                                  headers={"x-goog-api-key": CHIRP_KEY}, json={
                                      "input": {"text": text},
                                      "voice": {"languageCode": CHIRP_VOICE[:5], "name": CHIRP_VOICE},
                                      "audioConfig": {"audioEncoding": "LINEAR16", "sampleRateHertz": SAMPLE_RATE}})
        if r.status_code != 200:
            _chirp_resting = time.time() + (3600 if r.status_code in (400, 401, 403) else 120)
            log.warning("Chirp %s: %s", r.status_code, r.text[:300])
            return None
        with wave.open(io.BytesIO(base64.b64decode(r.json()["audioContent"]))) as w:  # LINEAR16 приходить з WAV-заголовком
            return w.readframes(w.getnframes())
    except (httpx.HTTPError, wave.Error, KeyError, ValueError) as e:
        log.warning("Chirp: %s", type(e).__name__)
        return None


async def synthesize(text: str) -> bytes:
    """Текст -> сирий PCM 16 біт, 16 кГц, моно (саме те, що грає Cardputer)."""
    if not text.strip():
        return b""
    for pattern, repl in _INTERJECTIONS:
        text = pattern.sub(repl, text)
    pcm = await _chirp(text)
    if pcm is not None:
        return pcm
    mp3 = bytearray()
    async for chunk in edge_tts.Communicate(text, VOICE, rate=VOICE_RATE).stream():
        if chunk["type"] == "audio":
            mp3 += chunk["data"]
    decoded = miniaudio.decode(
        bytes(mp3),
        output_format=miniaudio.SampleFormat.SIGNED16,
        nchannels=1,
        sample_rate=SAMPLE_RATE,
    )
    return decoded.samples.tobytes()


def normalize(pcm: np.ndarray, max_gain: float = 8.0) -> np.ndarray:
    """Прибирає постійну складову і підсилює тихий запис (мікрофон Cardputer доволі тихий)."""
    x = pcm.astype(np.float32)
    x -= x.mean()
    peak = float(np.abs(x).max()) or 1.0
    gain = min(max_gain, 0.9 * 32767 / peak)
    return np.clip(x * gain, -32768, 32767).astype(np.int16)


def pcm16_to_mulaw(pcm: np.ndarray) -> bytes:
    """PCM 16 біт -> µ-law (G.711), так само кодує й пристрій."""
    x = pcm.astype(np.int32)
    sign = (x < 0).astype(np.int32) << 7
    x = np.minimum(np.abs(x), 32635) + 0x84
    exponent = np.clip(np.floor(np.log2(x)).astype(np.int32) - 7, 0, 7)
    mantissa = (x >> (exponent + 3)) & 0x0F
    return (~(sign | (exponent << 4) | mantissa) & 0xFF).astype(np.uint8).tobytes()
