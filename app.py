"""Сервер «Наомі». Локально:  uvicorn app:app --host 0.0.0.0 --port 7860

/device — WebSocket для Cardputer (протокол описано в device.py)
/home   — WebSocket для ПК-агента та домашнього мосту (pc_link.py)
/screen — «великий екран» (рамка з Raspberry Pi, телефон): /screen#SCREEN_KEY, події — /screen/ws
/       — стан сервера
"""
import asyncio
import hmac
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse

import storage

storage.restore()  # у хмарі: забрати пам'ять, списки й нагадування з приватного сховища до старту

from brains import CITY, Brain  # noqa: E402
from device import DeviceSession  # noqa: E402
from pc_link import HomeLink  # noqa: E402
from screens import SCREENS  # noqa: E402
from weather import short as short_weather  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
log = logging.getLogger("naomi")

DEVICE_TOKEN = os.environ.get("NAOMI_DEVICE_TOKEN", "")
HOME_TOKEN = os.environ.get("NAOMI_HOME_TOKEN", "")
SCREEN_KEY = os.environ.get("SCREEN_KEY", "")  # порожній — великий екран вимкнений
SCREEN_PAGE = Path(__file__).resolve().parent / "static" / "screen.html"
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
    SCREENS.send({"t": "air", "on": active, "region": region}, remember="air")  # екран — навіть без Cardputer
    session = DeviceSession.current
    if session is not None:
        await session.air_alert(region, active)


_weather_at = 0.0


async def screen_weather(force: bool = False) -> None:
    """Погода для екрана — і тоді, коли Cardputer вимкнений (не частіше ніж раз на 30 хв)."""
    global _weather_at
    if not force and time.time() - _weather_at < 1800:
        return
    _weather_at = time.time()
    try:
        temp, desc = await short_weather(CITY)
        SCREENS.send({"t": "widget", "weather": temp, "desc": desc}, remember="widget")
    except Exception as e:
        log.debug("погода для екрана: %s", e)


async def screen_loop() -> None:
    while True:
        await asyncio.sleep(60)
        if SCREENS.clients:
            await screen_weather()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    home.on_event = on_home_event
    tasks = [asyncio.create_task(brain.reminders.run(deliver_reminder)),
             asyncio.create_task(brain.alerts.run(notify_air)),
             asyncio.create_task(inbox_loop()),
             asyncio.create_task(screen_loop())]
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


@app.get("/screen")
def screen_page():
    return FileResponse(SCREEN_PAGE, media_type="text/html; charset=utf-8", headers={"Cache-Control": "no-cache"})


@app.websocket("/screen/ws")
async def screen_socket(ws: WebSocket):
    """Ключ приходить першим повідомленням (а не в адресі), щоб не осідати в журналах сервера."""
    await ws.accept()
    try:
        first = json.loads(await asyncio.wait_for(ws.receive_text(), 15))
    except Exception:
        await ws.close(code=4400)
        return
    if not SCREEN_KEY or not _token_ok(str(first.get("key", "")), SCREEN_KEY):
        await ws.close(code=4401)
        return
    if "timers" not in SCREENS.state:
        SCREENS.state["timers"] = {"t": "timers", "items": [{"due": int(i["due"]), "text": i["text"]}
                                                            for i in brain.reminders.upcoming()]}
    try:
        await asyncio.wait_for(screen_weather(force="widget" not in SCREENS.state), 6)
    except asyncio.TimeoutError:
        pass
    try:
        await SCREENS.serve(ws)
    except Exception:
        pass
