"""Сервер «Наомі». Локально:  uvicorn app:app --host 0.0.0.0 --port 7860

/device — WebSocket для Cardputer (протокол описано в device.py)
/home   — WebSocket для ПК-агента та домашнього мосту (pc_link.py)
/       — стан сервера
"""
import asyncio
import hmac
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket

import storage

storage.restore()  # у хмарі: забрати пам'ять, списки й нагадування з приватного сховища до старту

from brains import Brain  # noqa: E402
from device import DeviceSession  # noqa: E402
from pc_link import HomeLink  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
log = logging.getLogger("naomi")

DEVICE_TOKEN = os.environ.get("NAOMI_DEVICE_TOKEN", "")
HOME_TOKEN = os.environ.get("NAOMI_HOME_TOKEN", "")
if not DEVICE_TOKEN or not HOME_TOKEN:
    raise SystemExit("Задай NAOMI_DEVICE_TOKEN і NAOMI_HOME_TOKEN (довгі випадкові рядки).")

home = HomeLink()
brain = Brain(home)


async def deliver_reminder(text: str) -> bool:
    session = DeviceSession.current
    return session is not None and await session.try_alarm(text)


async def on_home_event(role: str, msg: dict) -> None:
    if msg.get("event") == "claude_done":
        text = str(msg.get("text", ""))
        log.info("Claude закінчив (%s): %s", "ok" if msg.get("ok") else "помилка", text[:200])
        brain.announce(("Готово! " if msg.get("ok") and msg.get("mode") == "build" else "") + text,
                       "happy" if msg.get("ok") else "sad")


async def inbox_loop() -> None:
    while True:
        await asyncio.sleep(1)
        session = DeviceSession.current
        if session is not None and brain.inbox:
            await session.try_announce()


async def notify_air(region: str, active: bool) -> None:
    session = DeviceSession.current
    if session is not None:
        await session.air_alert(region, active)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    home.on_event = on_home_event
    tasks = [asyncio.create_task(brain.reminders.run(deliver_reminder)),
             asyncio.create_task(brain.alerts.run(notify_air)),
             asyncio.create_task(inbox_loop())]
    yield
    for t in tasks:
        t.cancel()


app = FastAPI(title="Naomi", lifespan=lifespan)


def _token_ok(given: str | None, expected: str) -> bool:
    return hmac.compare_digest((given or "").encode(), expected.encode())


@app.get("/")
def status():
    return {"name": "Naomi", "pc_online": home.online("pc"), "bridge_online": home.online("bridge"),
            "device_online": DeviceSession.current is not None}


@app.websocket("/device")
async def device_socket(ws: WebSocket):
    if not _token_ok(ws.headers.get("x-token"), DEVICE_TOKEN):
        await ws.close(code=4401)
        return
    await ws.accept()
    log.info("Cardputer підключився")
    try:
        await DeviceSession(ws, brain, home).run()
    except Exception:
        log.exception("сесія пристрою")
    log.info("Cardputer відключився")


@app.websocket("/home")
async def home_socket(ws: WebSocket):
    role = ws.query_params.get("role", "")
    if role not in ("pc", "bridge") or not _token_ok(ws.headers.get("x-token"), HOME_TOKEN):
        await ws.close(code=4401)
        return
    await ws.accept()
    try:
        await home.serve(ws, role)
    except Exception:
        pass
