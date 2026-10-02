"""Інтернет-радіо: сервер декодує потік і шле його на Cardputer тим самим µ-law, що й голос."""
import logging
import queue
import threading

import miniaudio
import numpy as np

from audio import SAMPLE_RATE, pcm16_to_mulaw

log = logging.getLogger("naomi.radio")

STATIONS = {
    "Хіт FM": "https://online.hitfm.ua/HitFM",
    "Kiss FM": "https://online.kissfm.ua/KissFM",
    "Radio ROKS": "https://online.radioroks.ua/RadioROKS",
    "Мелодія FM": "https://online.melodiafm.ua/MelodiaFM",
    "Наше радіо": "https://online.nasheradio.ua/NasheRadio",
    "Radio Relax": "https://online.radiorelax.ua/RadioRelax",
    "Radio Jazz": "https://online.radiojazz.ua/RadioJazz",
}
ALIASES = {"хіт": "Хіт FM", "hit": "Хіт FM", "кіс": "Kiss FM", "kiss": "Kiss FM", "рокс": "Radio ROKS",
           "roks": "Radio ROKS", "рок": "Radio ROKS", "мелод": "Мелодія FM", "наше": "Наше радіо",
           "релакс": "Radio Relax", "relax": "Radio Relax", "спок": "Radio Relax", "джаз": "Radio Jazz",
           "jazz": "Radio Jazz"}


def find_station(name: str) -> str:
    n = (name or "").lower()
    for key, title in ALIASES.items():
        if key in n:
            return title
    for title in STATIONS:
        if title.lower() in n or n in title.lower():
            return title
    return "Хіт FM"


class RadioStream:
    """Потік у фоновому потоці: MP3 з інтернету -> PCM 16 кГц -> µ-law шматками."""

    def __init__(self, url: str, gain: float = 0.8) -> None:
        self.url, self.gain = url, gain
        self.chunks: queue.Queue = queue.Queue(maxsize=48)
        self.stop = threading.Event()
        threading.Thread(target=self._run, daemon=True).start()

    def _put(self, item) -> None:
        while not self.stop.is_set():
            try:
                self.chunks.put(item, timeout=0.5)
                return
            except queue.Full:
                continue

    def _run(self) -> None:
        client = None
        try:
            client = miniaudio.IceCastClient(self.url)
            stream = miniaudio.stream_any(client, source_format=client.audio_format,
                                          output_format=miniaudio.SampleFormat.SIGNED16, nchannels=1,
                                          sample_rate=SAMPLE_RATE, frames_to_read=2048)
            for frames in stream:
                if self.stop.is_set():
                    break
                pcm = (np.asarray(frames, dtype=np.float32) * self.gain).astype(np.int16)
                self._put(pcm16_to_mulaw(pcm))
        except Exception as e:
            log.warning("радіо зупинилось: %s", e)
        finally:
            if client:
                client.close()
            try:
                self.chunks.put_nowait(None)
            except queue.Full:
                pass

    def get(self, timeout: float = 10):
        return self.chunks.get(timeout=timeout)

    def close(self) -> None:
        self.stop.set()
