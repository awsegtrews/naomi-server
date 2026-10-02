"""Повітряні тривоги з відкритих джерел — Наомі сама попереджає про початок і відбій.

Це ДОДАТКОВЕ попередження: дані неофіційні й можуть запізнюватись. Офіційний застосунок
«Повітряна тривога» залишається головним.
"""
import asyncio
import logging

import httpx

log = logging.getLogger("naomi.alerts")
UA = {"User-Agent": "NaomiVoiceAssistant/0.1"}


async def fetch_states() -> dict[str, bool]:
    async with httpx.AsyncClient(timeout=10, headers=UA) as c:
        try:
            d = (await c.get("https://ubilling.net.ua/aerialalerts/")).json()
            return {k: bool(v["alertnow"]) for k, v in d["states"].items()}
        except Exception as e:
            log.debug("ubilling: %s — пробую alerts.com.ua", e)
            d = (await c.get("https://alerts.com.ua/api/states")).json()
            return {s["name"]: bool(s["alert"]) for s in d["states"]}


def match_region(states: dict[str, bool], region: str) -> str | None:
    r = region.lower().replace("область", "").strip()
    if r in ("київ", "києві", "м. київ", "місто київ"):
        return "м. Київ" if "м. Київ" in states else None
    for name in states:
        if r and r[:5] in name.lower():
            return name
    return None


class AlertWatcher:
    def __init__(self, region: str = "м. Київ") -> None:
        self.region = region
        self.active: bool | None = None  # невідомо до першого опитування

    async def status(self, region: str | None = None) -> str:
        states = await fetch_states()
        name = match_region(states, region or self.region)
        if not name:
            return f"Не знайшла регіон «{region or self.region}»."
        others = [k for k, v in states.items() if v and k != name and k != "Севастополь"]
        main = f"У регіоні «{name}» зараз {'ПОВІТРЯНА ТРИВОГА' if states[name] else 'тривоги немає'}."
        return main + (f" Тривога також: {', '.join(others)}." if others else "")

    async def run(self, notify) -> None:
        """notify(region, active) — викликається при зміні стану в регіоні користувача."""
        while True:
            try:
                states = await fetch_states()
                name = match_region(states, self.region)
                if name is not None:
                    now = states[name]
                    if self.active is not None and now != self.active:
                        log.info("тривога %s: %s", name, now)
                        await notify(name, now)
                    self.active = now
            except Exception as e:
                log.debug("опитування тривог: %s", e)
            await asyncio.sleep(20)
