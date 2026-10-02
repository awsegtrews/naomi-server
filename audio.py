"""Звук: розпаковка µ-law від Cardputer, WAV для Whisper, голос Наомі (Edge TTS)."""
import io
import os
import wave

import edge_tts
import miniaudio
import numpy as np

SAMPLE_RATE = 16000
VOICE = os.environ.get("NAOMI_VOICE", "uk-UA-PolinaNeural")
VOICE_RATE = os.environ.get("NAOMI_VOICE_RATE", "+8%")


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


async def synthesize(text: str) -> bytes:
    """Текст -> сирий PCM 16 біт, 16 кГц, моно (саме те, що грає Cardputer)."""
    if not text.strip():
        return b""
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
