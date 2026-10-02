"""«Великий екран» Наомі (рамка з Raspberry Pi, телефон, ПК): трансляція подій розмови.

Екрани лише дивляться: відкривають /screen, підключаються до /screen/ws, надсилають ключ SCREEN_KEY
і отримують портрет (лише після ключа — малюнок не публічний) та події: слухає/думає/каже
(з «ритмом» голосу для рота), нагадування, тривоги, погоду, тему.
"""
import asyncio
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
from fastapi import WebSocket

log = logging.getLogger("naomi.screens")
HERE = Path(__file__).resolve().parent
ASSETS = HERE / "screen_assets.json"
PAGE = HERE / "static" / "screen.html"
ENV_FPS = 25  # скільки значень гучності на секунду для анімації рота


def envelope(pcm: np.ndarray, rate: int = 16000) -> list[int]:
    """Гучність голосу по кадрах (0..100, як рівень рота на Cardputer) — щоб рот рухався в такт."""
    step = rate // ENV_FPS
    n = len(pcm) // step
    if n == 0:
        return []
    frames = pcm[: n * step].astype(np.float32).reshape(n, step)
    rms = np.sqrt((frames ** 2).mean(axis=1)) / 7000.0
    return [int(min(1.0, v) * 100) for v in rms]


class Screens:
    def __init__(self) -> None:
        self.clients: set[WebSocket] = set()
        self.state: dict = {}  # останні значення — щоб новий екран одразу все показав
        self._assets: str | None = None

    def assets(self) -> str | None:
        if self._assets is None and ASSETS.exists():
            page = hashlib.sha1(PAGE.read_bytes()).hexdigest()[:12] if PAGE.exists() else ""  # оновили — екран перезавантажиться
            self._assets = json.dumps({"t": "assets", "page": page, **json.loads(ASSETS.read_text(encoding="utf-8"))})
        return self._assets

    async def serve(self, ws: WebSocket) -> None:
        assets = self.assets()
        if assets is None:
            log.warning("немає %s — екрану нічого показати", ASSETS.name)
            await ws.close(code=4404)
            return
        await ws.send_text(assets)
        for event in list(self.state.values()):
            await ws.send_text(json.dumps({**event, "replay": True}, ensure_ascii=False))
        self.clients.add(ws)
        log.info("екран підключився (усього %d)", len(self.clients))
        try:
            while True:
                await ws.receive_text()  # екран шле лише «я тут» раз на хвилину
        finally:
            self.clients.discard(ws)
            log.info("екран відключився (усього %d)", len(self.clients))

    def send(self, event: dict, remember: str | None = None) -> None:
        """Розіслати подію всім екранам (не чекаючи). remember — ключ для стану при підключенні."""
        if remember:
            self.state[remember] = event
        if not self.clients:
            return
        text = json.dumps(event, ensure_ascii=False)
        for ws in list(self.clients):
            asyncio.get_running_loop().create_task(self._safe_send(ws, text))

    async def _safe_send(self, ws: WebSocket, text: str) -> None:
        try:
            await ws.send_text(text)
        except Exception:
            self.clients.discard(ws)


SCREENS = Screens()
