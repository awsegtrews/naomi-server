"""Нагадування й таймери. Сервер сам «будить» Cardputer, коли настає час."""
import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import storage

log = logging.getLogger("naomi.reminders")
KYIV = ZoneInfo("Europe/Kyiv")
FILE = Path(__file__).with_name("reminders.json")


class Reminders:
    def __init__(self) -> None:
        self.items: list[dict] = []
        self.next_id = 1
        self.on_change = None  # callable() -> None: показати на пристрої
        try:
            saved = json.loads(FILE.read_text(encoding="utf-8"))
            self.items, self.next_id = saved["items"], saved["next_id"]
        except (OSError, ValueError, KeyError):
            pass

    def upcoming(self) -> list[dict]:
        return sorted(self.items, key=lambda i: i["due"])[:3]

    def _save(self) -> None:
        if self.on_change:
            self.on_change()
        try:
            FILE.write_text(json.dumps({"items": self.items, "next_id": self.next_id}, ensure_ascii=False),
                            encoding="utf-8")
            storage.changed(FILE)
        except OSError:
            log.warning("не вдалося зберегти нагадування")

    def add(self, text: str, minutes: float | None = None, at: str | None = None, seconds: float | None = None) -> str:
        now = datetime.now(KYIV)
        if at:
            h, m = (int(x) for x in at.replace(".", ":").split(":")[:2])
            due = now.replace(hour=h, minute=m, second=0, microsecond=0)
            if due <= now:
                due += timedelta(days=1)
        elif (minutes or 0) > 0 or (seconds or 0) > 0:
            due = now + timedelta(minutes=float(minutes or 0), seconds=float(seconds or 0))
        else:
            return "Скажи, через скільки хвилин або о котрій нагадати."
        item = {"id": self.next_id, "due": due.timestamp(), "text": text or "час!"}
        self.next_id += 1
        self.items.append(item)
        self._save()
        left = (due - now).total_seconds()
        if left < 3600:  # короткий таймер — кажемо «через скільки», а не «о котрій»
            mins, secs = int(left // 60), int(round(left % 60))
            when = "через " + (f"{mins} хв " if mins else "") + (f"{secs} с" if secs else "")
            return f"Таймер №{item['id']} поставлено {when.strip()}: {item['text']}"
        when = due.strftime("%H:%M") + ("" if due.date() == now.date() else " завтра")
        return f"Нагадування №{item['id']} поставлено на {when}: {item['text']}"

    def list(self) -> str:
        if not self.items:
            return "Нагадувань немає."
        rows = []
        for it in sorted(self.items, key=lambda i: i["due"]):
            rows.append(f"№{it['id']} о {datetime.fromtimestamp(it['due'], KYIV):%H:%M} — {it['text']}")
        return "; ".join(rows)

    def cancel(self, rid: int | None) -> str:
        if not rid:
            n = len(self.items)
            self.items = []
            self._save()
            return f"Скасувала всі нагадування ({n})."
        before = len(self.items)
        self.items = [i for i in self.items if i["id"] != rid]
        self._save()
        return "Скасувала." if len(self.items) < before else f"Нагадування №{rid} не знайшла."

    async def run(self, deliver) -> None:
        """deliver(text) -> bool: True, якщо пристрій прийняв нагадування."""
        while True:
            await asyncio.sleep(1)
            now = time.time()
            for it in [i for i in self.items if i["due"] <= now]:
                try:
                    if await deliver(it["text"]):
                        self.items.remove(it)
                        self._save()
                except Exception:
                    log.exception("доставка нагадування")
