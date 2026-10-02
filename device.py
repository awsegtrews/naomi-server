"""Сесія Cardputer по WebSocket /device.

Пристрій -> сервер:
  {"t":"hello","brain":"groq"}            при підключенні
  {"t":"start","brain":"groq"}            почав говорити (перебиває відповідь, якщо вона звучить)
  <бінарні кадри>                          голос: µ-law, 16 кГц, моно
  {"t":"stop"}                             відпустив кнопку — обробляй
  {"t":"cancel"} / {"t":"reset"}           скасувати / нова розмова
  {"t":"ack","b":N}                        відтворив ще N байт (керування потоком)
Сервер -> пристрій:
  {"t":"heard","text":...}  {"t":"say","text":...,"mood":"happy|calm|sad|surprised|thinking"}
  {"t":"cmd","volume":0..10,"bright":1..10,"theme":0..3}   {"t":"alarm","text":...} (+ голос)
  {"t":"audio","bytes":≈N,"rate":16000} + бінарні кадри µ-law (+ {"t":"total","bytes":N}) + {"t":"end"}
  {"t":"error","text":...}  {"t":"status","pc":true|false}
"""
import asyncio
import json
import logging
import re
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
from fastapi import WebSocket

from audio import SAMPLE_RATE, mulaw_to_pcm16, normalize, pcm16_to_mulaw, pcm16_to_wav, synthesize
from brains import CITY, OWNER_VOC, Brain
from radio import STATIONS, RadioStream
from weather import short as short_weather
from pc_link import HomeLink
from textutil import clean

log = logging.getLogger("naomi.device")

FRAME = 1024              # 64 мс звуку
WINDOW = 12 * 1024        # стільки байтів може бути «в дорозі» без підтвердження (буфер пристрою 32 КБ)
MAX_RECORD = SAMPLE_RATE * 60
GREET_EVERY = 2 * 3600    # вітатися голосом не частіше ніж раз на 2 години
_last_greeting = 0.0


async def greeting() -> str:
    h = datetime.now(ZoneInfo("Europe/Kyiv")).hour
    part = ("Доброго ранку" if 5 <= h < 11 else "Доброго дня" if h < 17 else
            "Доброго вечора" if h < 23 else "Доброї ночі")
    text = f"{part}, {OWNER_VOC}! Я на зв'язку."
    if 5 <= h < 11:  # зранку — одразу погода
        try:
            temp, desc = await short_weather(CITY)
            text += f" У місті {CITY} зараз {temp.replace('+', 'плюс ').replace('-', 'мінус ')}, {desc}."
        except Exception:
            pass
    return text


class DeviceSession:
    current: "DeviceSession | None" = None  # одночасно працює лише остання сесія Cardputer
    alive: list["DeviceSession"] = []       # усі відкриті — якщо остання закриється, повертаємось до попередньої

    def __init__(self, ws: WebSocket, brain: Brain, home: HomeLink) -> None:
        self.ws, self.brain, self.home = ws, brain, home
        self.provider = "groq"
        self.audio = bytearray()
        self.recording = False
        self.task: asyncio.Task | None = None
        self.sent = self.acked = 0
        self.acked_event = asyncio.Event()
        self.mute = False  # тихий режим: відповіді лише текстом на екрані

    async def send(self, obj: dict) -> None:
        await self.ws.send_text(json.dumps(obj, ensure_ascii=False))

    async def run(self) -> None:
        old, DeviceSession.current = DeviceSession.current, self
        DeviceSession.alive.append(self)
        if old is not None:
            await old.cancel()
        self.brain.device_cmd = self.send_cmd
        self.brain.confirm = self.ask_confirm
        self.confirms: dict[int, asyncio.Future] = {}
        self.confirm_id = 0
        self.brain.reminders.on_change = self.push_timers
        self.battery_warned = False
        widget = asyncio.create_task(self.widget_loop())
        await self.send({"t": "status", "pc": self.home.online("pc")})
        try:
            while True:
                msg = await self.ws.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                if msg.get("bytes") is not None:
                    if self.recording and len(self.audio) < MAX_RECORD:
                        self.audio += msg["bytes"]
                elif msg.get("text"):
                    await self.on_text(json.loads(msg["text"]))
        finally:
            widget.cancel()
            await self.cancel()
            DeviceSession.alive.remove(self)
            if DeviceSession.current is self:
                DeviceSession.current = DeviceSession.alive[-1] if DeviceSession.alive else None
                if DeviceSession.current:  # повертаємо гачки «мозку» попередній сесії
                    prev = DeviceSession.current
                    self.brain.device_cmd, self.brain.confirm = prev.send_cmd, prev.ask_confirm
                    self.brain.reminders.on_change = prev.push_timers

    def push_timers(self) -> None:
        items = [{"due": int(i["due"]), "text": i["text"]} for i in self.brain.reminders.upcoming()]
        asyncio.get_running_loop().create_task(self.send({"t": "timers", "items": items}))

    async def widget_loop(self) -> None:
        """Погода біля годинника на екрані — щопівгодини."""
        while True:
            try:
                temp, desc = await short_weather(CITY)
                await self.send({"t": "widget", "weather": temp, "desc": desc})
            except Exception as e:
                log.debug("віджет погоди: %s", e)
            await asyncio.sleep(1800)

    async def play_radio(self, title: str) -> None:
        log.info("радіо: %s", title)
        await self.send({"t": "say", "text": f"Грає {title}. Щоб вимкнути — натисни G0.", "mood": "happy"})
        stream = RadioStream(STATIONS[title])
        await self.send({"t": "audio", "bytes": 0, "rate": SAMPLE_RATE})
        self.sent = self.acked = 0
        try:
            while True:
                chunk = await asyncio.to_thread(stream.get, 15)
                if chunk is None:
                    break
                for i in range(0, len(chunk), FRAME):
                    while self.sent - self.acked >= WINDOW:
                        self.acked_event.clear()
                        await asyncio.wait_for(self.acked_event.wait(), timeout=15)
                    part = chunk[i:i + FRAME]
                    await self.ws.send_bytes(part)
                    self.sent += len(part)
        finally:
            stream.close()
        await self.send({"t": "end"})

    async def on_battery(self, level: int) -> None:
        self.brain.device_state["battery"] = level
        if level > 30:
            self.battery_warned = False
        elif level <= 15 and not self.battery_warned and not self.busy():
            self.battery_warned = True
            self.task = asyncio.create_task(
                self.speak(f"У мене лишилось {level} відсотків заряду. Постав мене, будь ласка, на зарядку.", "sad"))

    async def greet(self) -> None:
        await self.speak(await greeting(), "happy")

    async def ask_confirm(self, text: str, timeout: float = 60) -> bool:
        self.confirm_id += 1
        cid = self.confirm_id
        fut = asyncio.get_running_loop().create_future()
        self.confirms[cid] = fut
        await self.send({"t": "confirm", "id": cid, "text": text})
        try:
            return await asyncio.wait_for(fut, timeout)
        except asyncio.TimeoutError:
            return False
        finally:
            self.confirms.pop(cid, None)

    async def try_announce(self) -> bool:
        """Сказати перше повідомлення з «вхідних», якщо пристрій нічим не зайнятий."""
        if not self.brain.inbox or self.busy():
            return False
        text, mood = self.brain.inbox.pop(0)
        self.task = asyncio.create_task(self.speak(text, mood))
        return True

    async def send_cmd(self, cmd: dict) -> None:
        await self.send({"t": "cmd", **cmd})

    def busy(self) -> bool:
        return self.recording or (self.task is not None and not self.task.done())

    async def try_alarm(self, text: str) -> bool:
        """Нагадування від планувальника: лише коли пристрій нічим не зайнятий."""
        if self.busy():
            return False
        self.task = asyncio.create_task(self.alarm(text))
        return True

    async def alarm(self, text: str) -> None:
        await self.send({"t": "alarm", "text": text})
        await asyncio.sleep(1.6)  # пристрій грає мелодію-сигнал
        try:
            await self.speak("Нагадую: " + text, "happy")
        except TimeoutError:
            log.warning("нагадування не дограло — пристрій зник")

    async def on_text(self, m: dict) -> None:
        t = m.get("t")
        if t == "confirm_reply":
            fut = self.confirms.pop(int(m.get("id", -1)), None)
            if fut and not fut.done():
                fut.set_result(bool(m.get("ok")))
            return
        if t == "text":  # написали з клавіатури Cardputer
            await self.cancel()
            self.task = asyncio.create_task(self.answer(str(m.get("text", ""))[:500], typed=True))
            return
        if t == "mute":
            self.mute = bool(m.get("on"))
            return
        if t == "hello":
            self.mute = bool(m.get("mute", False))
            global _last_greeting
            self.brain.device_state = {k: m[k] for k in ("volume", "bright", "theme", "battery") if k in m}
            self.push_timers()
            if time.time() - _last_greeting > GREET_EVERY and not self.busy():
                _last_greeting = time.time()
                self.task = asyncio.create_task(self.greet())
        elif t == "battery":
            await self.on_battery(int(m.get("level", -1)))
        if t != "ack":
            log.info("<- %s (аудіо в буфері: %d байт)", m, len(self.audio))
        if t in ("hello", "start"):
            self.provider = "claude" if m.get("brain") == "claude" else "groq"
        if t == "start":
            await self.cancel()
            self.audio.clear()
            self.recording = True
        elif t == "stop":
            self.recording = False
            self.task = asyncio.create_task(self.process(bytes(self.audio)))
        elif t == "cancel":
            self.recording = False
            await self.cancel()
        elif t == "reset":
            self.brain.reset()
        elif t == "ack":
            self.acked += int(m.get("b", 0))
            self.acked_event.set()

    async def cancel(self) -> None:
        if self.task and not self.task.done():
            self.task.cancel()
            try:
                await self.task
            except (asyncio.CancelledError, Exception):
                pass
        self.task = None

    async def process(self, ulaw: bytes) -> None:
        try:
            pcm = mulaw_to_pcm16(ulaw)
            if len(pcm) < SAMPLE_RATE * 0.35:
                await self.speak("Я нічого не почула. Тримай кнопку, поки говориш.")
                return
            heard = await self.brain.transcribe(pcm16_to_wav(normalize(pcm)))
            log.info("почула [%s]: %s", self.provider, heard)
            if not heard:
                await self.speak("Я не розчула, повтори, будь ласка.")
                return
            await self.send({"t": "heard", "text": clean(heard)})
            await self.answer(heard)
        except asyncio.CancelledError:
            raise
        except TimeoutError:
            log.warning("пристрій перестав підтверджувати звук (зник зв'язок?) — зупиняю відповідь")
        except Exception:
            log.exception("помилка обробки")
            await self.speak("Ой, у мене щось зламалось на сервері. Спробуй ще раз.")

    async def answer(self, text: str, typed: bool = False) -> None:
        try:
            if not text.strip():
                return
            answer = await self.brain.reply(self.provider, text)
            log.info("відповідь%s [%s]: %s", " (текстом)" if typed else "", self.brain.last_mood, answer)
            await self.speak(answer, self.brain.last_mood)
            if self.brain.pending_radio:
                title, self.brain.pending_radio = self.brain.pending_radio, None
                await self.play_radio(title)
        except asyncio.CancelledError:
            raise
        except TimeoutError:
            log.warning("пристрій перестав підтверджувати звук")
        except Exception:
            log.exception("помилка відповіді")
            await self.speak("Ой, у мене щось зламалось на сервері. Спробуй ще раз.", "sad")

    async def air_alert(self, region: str, active: bool) -> None:
        """Тривога важливіша за все, крім моменту, коли власник саме говорить."""
        for _ in range(120):
            if not self.recording:
                break
            await asyncio.sleep(0.5)
        await self.cancel()  # перериваємо радіо чи відповідь
        self.task = asyncio.create_task(self._air(region, active))

    async def _air(self, region: str, active: bool) -> None:
        await self.send({"t": "air", "on": active, "region": region})
        await asyncio.sleep(3.2 if active else 1.2)  # сирена / м'який сигнал на пристрої
        text = (f"Увага! Повітряна тривога: {region}. Будь ласка, пройди в укриття." if active
                else f"Відбій повітряної тривоги: {region}.")
        try:
            await self.speak(text, "sad" if active else "happy", force_voice=True)
        except TimeoutError:
            log.warning("тривогу не дограно — пристрій зник")

    async def speak(self, text: str, mood: str = "calm", force_voice: bool = False) -> None:
        """Озвучує по реченнях: перше речення звучить, поки синтезуються наступні."""
        text = clean(text) or "…"
        log.info("говорю [%s]: %s", mood, text)
        if self.mute and not force_voice:  # тихий режим — лише текст
            await self.send({"t": "say", "text": text, "mood": mood, "silent": True})
            return
        await self.send({"t": "say", "text": text, "mood": mood})
        parts = [p for p in re.split(r"(?<=[.!?…])\s+", text) if p.strip()] or [text]
        jobs = [asyncio.create_task(synthesize(p)) for p in parts]
        # орієнтовна тривалість (~14 символів/с), точну надішлемо, коли синтез завершиться
        await self.send({"t": "audio", "bytes": int(len(text) * SAMPLE_RATE / 14), "rate": SAMPLE_RATE})
        self.sent = self.acked = 0
        try:
            for job in jobs:
                ulaw = pcm16_to_mulaw(np.frombuffer(await job, dtype="<i2"))
                if job is jobs[-1]:
                    await self.send({"t": "total", "bytes": self.sent + len(ulaw)})
                for i in range(0, len(ulaw), FRAME):
                    while self.sent - self.acked >= WINDOW:
                        self.acked_event.clear()
                        await asyncio.wait_for(self.acked_event.wait(), timeout=15)
                    chunk = ulaw[i:i + FRAME]
                    await self.ws.send_bytes(chunk)
                    self.sent += len(chunk)
        finally:
            for job in jobs:
                job.cancel()
        await self.send({"t": "end"})
        await self.send({"t": "status", "pc": self.home.online("pc")})
