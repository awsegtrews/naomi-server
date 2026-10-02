"""Зв'язок хмари з домом: ПК-агент і домашній міст підключаються сюди по WebSocket.

Вони самі тримають з'єднання до сервера (вихідне), тому не треба відкривати
порти на роутері чи мати білу IP-адресу.
"""
import asyncio
import itertools
import json
import logging

from fastapi import WebSocket

log = logging.getLogger("naomi.link")


class HomeLink:
    def __init__(self) -> None:
        self.clients: dict[str, WebSocket] = {}  # "pc" | "bridge" -> сокет
        self.pending: dict[int, asyncio.Future] = {}
        self._ids = itertools.count(1)
        self.on_event = None  # async (role, msg) -> None: події, які ПК надсилає сам

    def online(self, role: str) -> bool:
        return role in self.clients

    async def serve(self, ws: WebSocket, role: str) -> None:
        old = self.clients.get(role)
        if old is not None:
            await old.close(code=4000)
        self.clients[role] = ws
        log.info("%s підключився", role)
        try:
            while True:
                msg = json.loads(await ws.receive_text())
                if "event" in msg:
                    if self.on_event:
                        await self.on_event(role, msg)
                    continue
                fut = self.pending.pop(msg.get("id"), None)
                if fut and not fut.done():
                    fut.set_result(msg)
        finally:
            if self.clients.get(role) is ws:
                del self.clients[role]
            log.info("%s відключився", role)

    async def call(self, role: str, action: str, arg: str = "", timeout: float = 20) -> str:
        ws = self.clients.get(role)
        if ws is None:
            return "OFFLINE"
        req_id = next(self._ids)
        fut = asyncio.get_running_loop().create_future()
        self.pending[req_id] = fut
        try:
            await ws.send_text(json.dumps({"id": req_id, "action": action, "arg": arg}, ensure_ascii=False))
            msg = await asyncio.wait_for(fut, timeout)
        except asyncio.TimeoutError:
            return "Помилка: ПК не відповів вчасно."
        finally:
            self.pending.pop(req_id, None)
        prefix = "" if msg.get("ok") else "Помилка: "
        return prefix + str(msg.get("result", ""))
