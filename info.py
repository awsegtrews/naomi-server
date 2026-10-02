"""Корисні джерела без ключів: курси НБУ і свіжі новини (RSS Суспільного)."""
import html
import re
import time

import httpx

UA = {"User-Agent": "NaomiVoiceAssistant/0.1"}
_cache: dict[str, tuple[float, object]] = {}


async def _get(url: str, ttl: float) -> str:
    hit = _cache.get(url)
    if hit and time.time() - hit[0] < ttl:
        return hit[1]
    async with httpx.AsyncClient(timeout=10, follow_redirects=True, headers=UA) as c:
        text = (await c.get(url)).text
    _cache[url] = (time.time(), text)
    return text


NAMES = {"USD": "долар", "EUR": "євро", "PLN": "злотий", "GBP": "фунт", "CHF": "франк", "CZK": "чеська крона",
         "CNY": "юань", "JPY": "єна", "CAD": "канадський долар", "TRY": "турецька ліра"}


async def currency(codes: list[str] | None = None) -> str:
    import json
    data = json.loads(await _get("https://bank.gov.ua/NBUStatService/v1/statdirectory/exchange?json", 3600))
    rates = {d["cc"]: d for d in data}
    codes = [c.upper() for c in (codes or ["USD", "EUR", "PLN"])]
    parts = []
    for c in codes:
        d = rates.get(c)
        if d:
            parts.append(f"{NAMES.get(c, d['txt'])} {d['rate']:.2f} грн".replace(".", ","))
    if not parts:
        return "Не знайшла таких валют у НБУ."
    date = next(iter(rates.values()))["exchangedate"]
    return f"Офіційний курс НБУ на {date}: " + ", ".join(parts) + "."


async def news(count: int = 6) -> str:
    rss = await _get("https://suspilne.media/rss/all.rss", 600)
    titles = re.findall(r"<item>.*?<title>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</title>", rss, re.S)
    titles = [html.unescape(t).strip() for t in titles if t.strip()][:count]
    if not titles:
        return "Не вдалося отримати новини."
    return ("Свіжі заголовки Суспільного: " + " | ".join(titles) +
            ". Перекажи 2–3 найважливіші одним-двома реченнями.")
