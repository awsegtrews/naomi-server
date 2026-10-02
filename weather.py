"""Погода з Open-Meteo (безкоштовно, без ключа)."""
import httpx

WMO = {
    0: "ясно", 1: "переважно ясно", 2: "мінлива хмарність", 3: "хмарно", 45: "туман", 48: "туман з памороззю",
    51: "легка мряка", 53: "мряка", 55: "сильна мряка", 56: "крижана мряка", 57: "сильна крижана мряка",
    61: "невеликий дощ", 63: "дощ", 65: "сильний дощ", 66: "крижаний дощ", 67: "сильний крижаний дощ",
    71: "невеликий сніг", 73: "сніг", 75: "сильний сніг", 77: "снігова крупа", 80: "короткочасний дощ",
    81: "зливи", 82: "сильні зливи", 85: "снігопад", 86: "сильний снігопад", 95: "гроза", 96: "гроза з градом",
    99: "сильна гроза з градом",
}


_places: dict[str, dict] = {}


async def short(city: str) -> tuple[str, str]:
    """Коротко для екрана й привітання: ("+15°", "ясно")."""
    async with httpx.AsyncClient(timeout=8) as c:
        key = city.strip().lower()
        place = _places.get(key)
        if place is None:
            g = (await c.get("https://geocoding-api.open-meteo.com/v1/search",
                             params={"name": city, "count": 1, "language": "uk"})).json()
            place = _places[key] = g["results"][0]
        f = (await c.get("https://api.open-meteo.com/v1/forecast", params={
            "latitude": place["latitude"], "longitude": place["longitude"],
            "current": "temperature_2m,weather_code", "timezone": "auto"})).json()
    t = round(f["current"]["temperature_2m"])
    return f"{t:+d}°", WMO.get(f["current"]["weather_code"], "")


async def weather(city: str) -> str:
    async with httpx.AsyncClient(timeout=8) as c:
        key = city.strip().lower()
        place = _places.get(key)
        if place is None:
            g = (await c.get("https://geocoding-api.open-meteo.com/v1/search",
                             params={"name": city, "count": 1, "language": "uk"})).json()
            if not g.get("results"):
                return f"Не знайшла місто «{city}»."
            place = _places[key] = g["results"][0]
        f = (await c.get("https://api.open-meteo.com/v1/forecast", params={
            "latitude": place["latitude"], "longitude": place["longitude"],
            "current": "temperature_2m,apparent_temperature,weather_code,wind_speed_10m",
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max,weather_code",
            "timezone": "auto", "forecast_days": 2, "wind_speed_unit": "ms",
        })).json()
    cur, d = f["current"], f["daily"]
    r = round
    return (f"{place['name']}: зараз {r(cur['temperature_2m'])}°, відчувається як {r(cur['apparent_temperature'])}°, "
            f"{WMO.get(cur['weather_code'], '')}, вітер {r(cur['wind_speed_10m'])} м/с. "
            f"Сьогодні від {r(d['temperature_2m_min'][0])}° до {r(d['temperature_2m_max'][0])}°, "
            f"ймовірність опадів {d['precipitation_probability_max'][0]}%. "
            f"Завтра від {r(d['temperature_2m_min'][1])}° до {r(d['temperature_2m_max'][1])}°, {WMO.get(d['weather_code'][1], '')}.")
