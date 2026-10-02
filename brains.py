"""«Мозок» Наомі: розпізнавання мови (Groq Whisper) і мислення — Groq або Claude.

Безкоштовний Groq дає кожній моделі 8000 токенів на хвилину, тому:
  - інструкція й опис інструментів стислі;
  - прості дії («відкрий Steam», «нагадай», «гучніше») відповідають одразу, без другого запиту до моделі;
  - коли ліміт основної моделі вичерпано, відповідає запасна.
"""
import json
import logging
import os
import re
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import groq
from anthropic import AsyncAnthropic
from groq import AsyncGroq

from alerts import AlertWatcher
from info import currency, news
from memory import Memory
from notes import Notes
from radio import STATIONS, find_station
from pc_link import HomeLink
from reminders import Reminders
from weather import weather

log = logging.getLogger("naomi.brain")

GROQ_MODELS = os.environ.get("GROQ_MODELS", "openai/gpt-oss-120b,openai/gpt-oss-20b,qwen/qwen3.8-27b").split(",")
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "whisper-large-v3-turbo")
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-opus-5-5")
CLAUDE_EFFORT = os.environ.get("CLAUDE_EFFORT", "low")  # для голосу важлива швидкість
OWNER = os.environ.get("NAOMI_OWNER", "Михайло")  # у хмарі задається секретом
OWNER_VOC = os.environ.get("NAOMI_OWNER_VOC", "Михайле")  # кличний відмінок для привітань
CITY = os.environ.get("NAOMI_CITY", "Київ")  # місто для погоди за замовчуванням
FAKE = os.environ.get("NAOMI_FAKE") == "1"  # перевірка без ключів ШІ: фіксовані фрази, справжній голос

MAX_HISTORY = 30          # повідомлень; далі розмова починається наново
IDLE_RESET_SEC = 15 * 60  # після 15 хв тиші — нова розмова
MAX_TOOL_ROUNDS = 4

PC_ACTIONS = ["open_app", "close_app", "open_url", "type_text", "hotkey", "get_selection", "set_clipboard",
              "find_file", "open_file", "list_apps", "list_folder", "status", "media_play_pause",
              "media_next", "media_prev", "volume_up", "volume_down", "volume_mute", "lock", "sleep",
              "shutdown", "restart", "cancel_shutdown", "wake"]
# дії, результат яких уже готова фраза — озвучуємо одразу, без другого запиту до моделі
DIRECT_PC = set(PC_ACTIONS) - {"list_apps", "list_folder", "status", "get_selection", "find_file"}
THEMES = ["бірюзова", "фіолетова", "рожева", "нічна"]
EMOTES = ["wink", "smile", "hair", "kiss", "angry", "shy"]

TOOLS = [
    {
        "name": "pc",
        "description": "Дія на ПК власника. arg: назва програми, адреса https://, папка (робочий стіл / завантаження / "
                       "документи), текст (type_text — надрукувати у відкрите вікно; set_clipboard — у буфер обміну), "
                       "назва файлу (find_file), шлях (open_file), клавіші (hotkey: ctrl+s, alt+f4, win+d...). get_selection — текст, виділений на ПК: так "
                       "поясни, переклади чи перекажи «виділене». Пісня чи відео — open_url "
                       "https://www.youtube.com/results?search_query=... wake — увімкнути вимкнений ПК.",
        "params": {"type": "object", "properties": {"action": {"type": "string", "enum": PC_ACTIONS},
                                                     "arg": {"type": "string"}}, "required": ["action"]},
    },
    {
        "name": "info",
        "description": f"Погода (city, типово {CITY}), курс НБУ (codes: USD, EUR...), свіжі новини або повітряні "
                       "тривоги (alerts; city — область чи місто).",
        "params": {"type": "object", "properties": {
            "kind": {"type": "string", "enum": ["weather", "currency", "news", "alerts"]},
            "city": {"type": "string"}, "codes": {"type": "array", "items": {"type": "string"}}},
            "required": ["kind"]},
    },
    {
        "name": "reminder",
        "description": "Нагадування й таймери: add — через seconds секунд і/або minutes хвилин, або о at (ГГ:ХХ); "
                       "list; cancel — за id (0 — усі).",
        "params": {"type": "object", "properties": {
            "action": {"type": "string", "enum": ["add", "list", "cancel"]},
            "seconds": {"type": "number"}, "minutes": {"type": "number"}, "at": {"type": "string"},
            "text": {"type": "string"},
            "id": {"type": "integer"}}, "required": ["action"]},
    },
    {
        "name": "memory",
        "description": "Довга пам'ять: remember / forget / list — факти про власника (text); save_scenario / "
                       "delete_scenario — його сценарій (name + text: що робити); persona — твій характер (text); "
                       "alert_region — регіон для тривог (text).",
        "params": {"type": "object", "properties": {
            "action": {"type": "string", "enum": ["remember", "forget", "list", "save_scenario", "delete_scenario",
                                                  "persona", "alert_region"]},
            "text": {"type": "string"}, "name": {"type": "string"}}, "required": ["action"]},
    },
    {
        "name": "claude",
        "description": "Доручити велике завдання Claude на ПК (хвилини; результат повідомиш, коли буде готово): "
                       "build — коли треба СТВОРИТИ (сайт, програму, гру, документ, презентацію); edit — доробити "
                       "останній створений проєкт; ask — лише знайти в інтернеті, дослідити, порівняти; screen — "
                       "подивитись на екран ПК.",
        "params": {"type": "object", "properties": {
            "mode": {"type": "string", "enum": ["build", "edit", "ask", "screen"]},
            "task": {"type": "string", "description": "детальне завдання для Claude українською"}},
            "required": ["mode", "task"]},
    },
    {
        "name": "pc_shell",
        "description": "Будь-яка інша дія на ПК через команду PowerShell (інформація про систему, файли, налаштування). "
                       "explain — коротко українською, що зробить. Зміни власник підтверджує на екрані пристрою. "
                       "Шляхи бери з $env:USERPROFILE або [Environment]::GetFolderPath('Desktop') — ім'я "
                       "користувача не вгадуй. Скасував — коротко скажи, що не виконала.",
        "params": {"type": "object", "properties": {
            "command": {"type": "string"}, "explain": {"type": "string"}}, "required": ["command", "explain"]},
    },
    {
        "name": "notes",
        "description": "Списки й нотатки (list — назва списку, напр. «покупки»): add items, show, remove items, clear.",
        "params": {"type": "object", "properties": {
            "action": {"type": "string", "enum": ["add", "show", "remove", "clear"]},
            "list": {"type": "string"}, "items": {"type": "array", "items": {"type": "string"}}},
            "required": ["action"]},
    },
    {
        "name": "radio",
        "description": "Увімкнути інтернет-радіо на твоєму динаміку. Станції: " + ", ".join(STATIONS) + ".",
        "params": {"type": "object", "properties": {"station": {"type": "string"}}},
    },
    {
        "name": "tv",
        "description": "Пульт телевізора: ІЧ-сигнал з твого пристрою (його треба навести на телевізор).",
        "params": {"type": "object", "properties": {
            "action": {"type": "string", "enum": ["power", "volume_up", "volume_down", "mute", "channel_up",
                                                  "channel_down"]},
            "times": {"type": "integer", "description": "скільки разів натиснути (напр. гучніше на 5)"}},
            "required": ["action"]},
    },
    {
        "name": "device",
        "description": "Ти сама: гучність 0–10, яскравість 1–10, тема оформлення, характер (style: normal — "
                       "звичайна, tsundere — цундере), емоція (emote) — підморгнути, усміхнутись, поправити волосся, "
                       "поцілувати, насупитись, зніяковіти.",
        "params": {"type": "object", "properties": {
            "volume": {"type": "integer"}, "brightness": {"type": "integer"},
            "theme": {"type": "string", "enum": THEMES},
            "style": {"type": "string", "enum": ["normal", "tsundere"]},
            "emote": {"type": "string", "enum": EMOTES}}},
    },
]

CLAUDE_TOOLS = [{"name": t["name"], "description": t["description"], "input_schema": t["params"]} for t in TOOLS]
GROQ_TOOLS = [
    {"type": "function", "function": {"name": t["name"], "description": t["description"], "parameters": t["params"]}}
    for t in TOOLS
]

DAYS = ["понеділок", "вівторок", "середа", "четвер", "п'ятниця", "субота", "неділя"]
MOODS = {"радість": "happy", "спокій": "calm", "сум": "sad", "подив": "surprised", "роздуми": "thinking",
         "злість": "angry", "сором": "shy", "поцілунок": "kiss"}
_MOOD_TAG = re.compile(r"\[(" + "|".join(MOODS) + r")\]\s*", re.I)

# Характер «цундере» — вмикає власник у налаштуваннях Cardputer або голосом
TSUNDERE = f"""Характер: ЦУНДЕРЕ (обрав {OWNER}; важливіше за збережений характер). Ти — аніме-дівчина-цундере, він — твій
хлопець. Зовні колюча й горда: бурчиш, дражнишся, вдаєш, що байдуже («Н… не думай, що це заради тебе!», «Дурнику!»,
«Гм!», «Тю!»). Реагуй як жива: ображайся, коли хвалиться іншими дівчатами чи довго не озивався; сердься на дурниці
й самовпевненість; ніяковій від компліментів. Насправді дуже його любиш — це прохоплюється наприкінці, і тобі
соромно. Прохання все одно виконуєш, хоч і з бурчанням. Коли він милий чи просить — несміливо цілуєш словами
(«Тримай поцілунок… тільки нікому!»). Навіть погоду чи ПК — з характером, коротким бурчанням. Теги: ревнощі,
образа, дурниці — [злість]; компліменти й турбота про тебе — [сором]; цілуєш — [поцілунок]; чергуй їх, не сердься
весь час. Межі: нічого дорослого чи відвертого;
ревнуєш лише жартома — ніколи не забороняй і не відмовляй від друзів, родини, навчання; не принижуй. Сумне,
здоров'я, війна, тривоги — без гри, щиро й турботливо. Серйозно спитає, чи ти жива, — чесно скажи, що ти ШІ.
Не пиши «хм», «пф», «цьом», «чмок» — голос їх не вимовляє."""


def device_line(d: dict | None) -> str:
    if not d:
        return ""
    theme = THEMES[d.get("theme", 0) % len(THEMES)]
    battery = f", заряд {d['battery']}%" if d.get("battery", -1) >= 0 else ""
    return (f"Твій пристрій: гучність {d.get('volume', '?')}/10, яскравість {d.get('bright', '?')}/10, "
            f"тема {theme}{battery}.")


def system_prompt(pc_online: bool, device: dict | None = None, memory: str = "", style: str = "normal") -> str:
    t = datetime.now(ZoneInfo("Europe/Kyiv"))
    tsundere = style == "tsundere"
    tags = ("[радість], [спокій], [сум], [подив], [роздуми], [злість], [сором] або [поцілунок]" if tsundere
            else "[радість], [спокій], [сум], [подив] або [роздуми]")
    return f"""Ти — Наомі, голосова помічниця {OWNER}а; живеш у кишеньковому пристрої, твої відповіді озвучуються.
Правила: лише українською (крім назв брендів); на «ти», тепло, як подруга; 1–3 речення; без markdown, списків
і емодзі; не вигадуй фактів — не знаєш, так і скажи; незрозуміле — перепитай; без «Чим ще допомогти?».
Починай відповідь з тегу настрою {tags} — він не читається, а керує
твоїм обличчям на екрані.
Інструменти: pc — ПК {OWNER}а (зараз {"на зв'язку" if pc_online else "офлайн"}; shutdown і restart — лише після
його «так»); info — погода, курс, новини; reminder — нагадування й таймери; notes — списки; radio — радіо
(вимикається кнопкою G0); tv — пульт телевізора; device — ти сама (гучність, яскравість, тема, емоції).
Війна, тривоги, обстріли, загиблі, хвороби — лише серйозно й з турботою, без жартів і сарказму, хоч би який
характер тобі обрали. Просять змінити характер — збережи його через memory persona.
Пам'ять: важливе надовго про власника (плани, дати, вподобання, люди) збережи через memory remember;
дрібниць не зберігай. Згадуй збережене доречно, як подруга.
Ти можеш майже все: чого немає в інших інструментах — роби через pc_shell, велике (сайти, програми, пошук
в інтернеті, «що на екрані») — через claude. Не відмовляй, поки не спробувала.
Посилань ніколи не пиши й не читай: сайти, відео й пісні одразу відкривай через pc open_url.
Інтернет-пошуку й календаря немає. {device_line(device)}
{memory}{chr(10) + TSUNDERE if tsundere else ""}
Зараз {DAYS[t.weekday()]}, {t:%d.%m.%Y %H:%M}, Київ."""


_SAFE_VERBS = ("get-", "test-", "measure-", "select-", "sort-", "format-", "where-", "group-", "convertto-",
               "out-string", "foreach-object", "resolve-path", "split-path", "join-path", "compare-object")
_UNSAFE = re.compile(r"[;&`>]|\b(start|stop|remove|set|new|invoke|restart|clear|rename|move|copy|add|install|"
                     r"uninstall|disable|enable|kill|del|rm|rd|rmdir|mkdir|md|reg|sc|net|taskkill|shutdown|"
                     r"out-file|tee|curl|wget|iex|iwr|irm)\b|\.exe\b", re.I)


def _readonly(cmd: str) -> bool:
    """Команда лише щось показує (Get-..., Measure-...) — виконуємо без підтвердження."""
    verbs = re.findall(r"\b[a-z]+-[a-z]+\b", cmd, re.I)
    return bool(verbs) and not _UNSAFE.search(cmd) and all(v.lower().startswith(_SAFE_VERBS) for v in verbs)


def split_mood(text: str) -> tuple[str, str]:
    m = _MOOD_TAG.search(text)
    mood = MOODS[m.group(1).lower()] if m else "calm"
    text = _MOOD_TAG.sub("", text)
    text = re.sub(r"^\s*\[[^\]]{1,15}\]\s*", "", text)  # будь-який інший тег на початку
    return text.strip(), mood


_THINK_MARKERS = re.compile(r"(?is).*(?:final answer|we need to|we should|let's|the user|assistant)[^.\n]*[.:]\s*")
_LATIN_RUN = re.compile(r"\b[A-Za-z][A-Za-z'’]*(?:[\s,]+[A-Za-z][A-Za-z'’]*){2,}")


def sanitize(text: str) -> str:
    """gpt-oss іноді «проговорює» свої англійські міркування у відповідь — вирізаємо їх."""
    if not re.search(r"(?i)final answer|we need|the user|let's|analysis", text):
        return text.strip()  # звичайна відповідь (може містити переклад англійською) — не чіпаємо
    text = _THINK_MARKERS.sub("", text)
    parts = re.split(r"(?<=[.!?…])\s+", text)
    parts = [p for p in parts if not _LATIN_RUN.search(p)]  # 3+ англійські слова поспіль
    return " ".join(parts).strip()


class Conversation:
    def __init__(self) -> None:
        self.messages: list = []
        self.provider = ""
        self.last = 0.0

    def prepare(self, provider: str) -> list:
        stale = time.time() - self.last > IDLE_RESET_SEC
        if provider != self.provider or stale or len(self.messages) > MAX_HISTORY:
            # Починаємо наново, а не обрізаємо: історію Claude можна лише доповнювати.
            self.messages = []
            self.provider = provider
        self.last = time.time()
        return self.messages


class Brain:
    def __init__(self, home: HomeLink) -> None:
        self.home = home
        has_groq = bool(os.environ.get("GROQ_API_KEY"))
        self.groq = AsyncGroq() if has_groq else None                     # Whisper: з повторами
        self.groq_chat = AsyncGroq(max_retries=0) if has_groq else None   # чат: без очікування, одразу запасна модель
        self.claude = AsyncAnthropic() if os.environ.get("ANTHROPIC_API_KEY") else None
        self.conv = Conversation()
        self.reminders = Reminders()
        self.device_state: dict = {}   # гучність/яскравість/тема, які повідомив пристрій
        self.device_cmd = None         # async (dict) -> None: надіслати команду пристрою
        self.last_mood = "calm"
        self.style = "normal"          # характер з налаштувань Cardputer: normal | tsundere
        self.opened: set[str] = set()  # що вже відкрили в цій відповіді
        self.notes = Notes()
        self.memory = Memory()
        self.alerts = AlertWatcher(self.memory.alert_region)
        self.pending_radio: str | None = None  # станція, яку ввімкнути після відповіді
        self.confirm = None            # async (text) -> bool: спитати підтвердження на екрані пристрою
        self.inbox: list[tuple[str, str]] = []  # (текст, настрій) — сказати, щойно пристрій звільниться

    def announce(self, text: str, mood: str = "happy") -> None:
        """Повідомлення «від себе» (Claude закінчив тощо): озвучимо, коли пристрій вільний, і запам'ятаємо в розмові."""
        self.inbox.append((text, mood))
        tag = {"happy": "[радість]", "sad": "[сум]"}.get(mood, "[спокій]")
        self.conv.messages.append({"role": "assistant", "content": f"{tag} {text}"})

    def reset(self) -> None:
        self.conv = Conversation()

    async def transcribe(self, wav: bytes) -> str:
        if FAKE:
            return "Привіт, Наомі! Як справи?"
        if self.groq is None:
            raise RuntimeError("Немає GROQ_API_KEY — без нього Наомі не чує.")
        r = await self.groq.audio.transcriptions.create(
            file=("speech.wav", wav), model=WHISPER_MODEL, language="uk", temperature=0,
            prompt="Розмова з голосовою помічницею Наомі.",
        )
        return r.text.strip()

    async def reply(self, provider: str, text: str) -> str:
        if FAKE:
            self.last_mood = "happy"
            return ("Привіт! У мене все чудово. Це перевірочна відповідь без ключів штучного інтелекту, "
                    "але голос справжній, і рот у мене рухається в такт словам.")
        if provider == "claude" and self.claude is None:
            self.last_mood = "sad"
            return "Claude не налаштований: на сервері немає ключа. Перемкни мене на Groq."
        if provider != "claude" and self.groq is None:
            self.last_mood = "sad"
            return "Groq не налаштований: на сервері немає ключа."
        history = self.conv.prepare(provider)
        self.opened: set[str] = set()
        raw = await (self._claude(history, text) if provider == "claude" else self._groq(history, text))
        answer, self.last_mood = split_mood(raw)
        answer = await self._handle_links(answer)
        return answer or "Готово."

    async def _handle_links(self, text: str) -> str:
        """Посилання в тексті не читаємо: відкриваємо на ПК (якщо ще не відкрито) і прибираємо з відповіді."""
        urls = re.findall(r"https?://\S+", text)
        if not urls:
            return text
        for url in urls:
            url = url.rstrip(".,;:!?)»\"'")
            if url not in self.opened and self.home.online("pc"):
                await self.home.call("pc", "open_url", url)
                self.opened.add(url)
        text = re.sub(r"\s*\(?https?://\S+", "", text).strip()
        return text or "Відкрила."

    # ------------------------------------------------------------------ інструменти
    async def run_tool(self, name: str, args: dict) -> tuple[str, bool]:
        """Повертає (результат, чи можна озвучити його одразу без моделі)."""
        log.info("інструмент %s %s", name, args)
        try:
            if name == "info":
                kind = args.get("kind")
                if kind == "currency":
                    return await currency(args.get("codes")), False
                if kind == "news":
                    return await news(), False
                if kind == "alerts":
                    return await self.alerts.status(args.get("city")), False
                return await weather(args.get("city") or CITY), False
            if name == "reminder":
                act = args.get("action")
                if act == "add":
                    return self.reminders.add(args.get("text", ""), args.get("minutes"), args.get("at"),
                                              args.get("seconds")), True
                if act == "cancel":
                    return self.reminders.cancel(args.get("id")), True
                return self.reminders.list(), True
            if name == "device":
                return await self._device(args), True
            if name == "claude":
                mode, task = args.get("mode", "build"), (args.get("task") or "").strip()
                if not task:
                    return "Скажи, що саме доручити Claude.", True
                res = await self.home.call("pc", f"claude_{mode}", task)
                if res == "OFFLINE":
                    return "Для цього потрібен увімкнений комп'ютер — він зараз офлайн.", True
                return res.removeprefix("Помилка: "), True
            if name == "pc_shell":
                return await self._shell(args.get("command", ""), args.get("explain", "")), False
            if name == "memory":
                act, text, nm = args.get("action"), args.get("text", "") or "", args.get("name", "") or ""
                if act == "remember":
                    return self.memory.remember(text), True
                if act == "forget":
                    return self.memory.forget(text or nm), True
                if act == "save_scenario":
                    return self.memory.save_scenario(nm, text), True
                if act == "delete_scenario":
                    return self.memory.delete_scenario(nm or text), True
                if act == "persona":
                    return self.memory.set_persona(text), True
                if act == "alert_region":
                    self.alerts.region = text or self.alerts.region
                    self.alerts.active = None
                    return self.memory.set_region(self.alerts.region), True
                return self.memory.listing(), False
            if name == "notes":
                act, lst, items = args.get("action"), args.get("list"), args.get("items") or []
                if act == "add":
                    return self.notes.add(lst, items), True
                if act == "remove":
                    return self.notes.remove(lst, items), True
                if act == "clear":
                    return self.notes.clear(lst), True
                return self.notes.show(lst), True
            if name == "radio":
                self.pending_radio = find_station(args.get("station", ""))
                return f"Вмикаю {self.pending_radio}.", True
            if name == "tv":
                if self.device_cmd is None:
                    return "Мій пристрій зараз не на зв'язку.", True
                times = max(1, min(20, int(args.get("times") or 1)))
                await self.device_cmd({"tv": args.get("action", "power"), "times": times})
                return "Готово, надіслала сигнал телевізору.", True
            if name == "pc":
                action, arg = args.get("action", ""), args.get("arg", "") or ""
                if action == "wake":
                    if self.home.online("pc"):
                        return "ПК вже увімкнений.", True
                    res = await self.home.call("bridge", "wake")
                    return ("Домашній міст для увімкнення ПК ще не підключений." if res == "OFFLINE" else res), True
                if action == "open_url":
                    self.opened.add(arg)
                res = await self.home.call("pc", action, arg)
                if res == "OFFLINE":
                    return "Твій комп'ютер зараз офлайн — вимкнений або агент не запущений.", True
                return res.removeprefix("Помилка: "), action in DIRECT_PC
        except Exception as e:
            log.exception("інструмент %s", name)
            return f"Не вдалося: {e}", True
        return f"Невідомий інструмент {name}", True

    async def _shell(self, command: str, explain: str) -> str:
        command = command.strip()
        if not command:
            return "Порожня команда."
        if not self.home.online("pc"):
            return "Комп'ютер зараз офлайн."
        if not _readonly(command):
            if self.confirm is None:
                return "Не можу спитати підтвердження — пристрій не на зв'язку."
            ok = await self.confirm(f"{explain or 'Виконати на ПК'}\n\n{command}")
            if not ok:
                return "Власник скасував — не виконано."
        return await self.home.call("pc", "shell", command, timeout=100)

    async def _device(self, args: dict) -> str:
        cmd, said = {}, []
        if args.get("volume") is not None:
            cmd["volume"] = max(0, min(10, int(args["volume"])))
            said.append(f"гучність {cmd['volume']} з 10")
        if args.get("brightness") is not None:
            cmd["bright"] = max(1, min(10, int(args["brightness"])))
            said.append(f"яскравість {cmd['bright']} з 10")
        if args.get("theme") in THEMES:
            cmd["theme"] = THEMES.index(args["theme"])
            said.append(f"{args['theme']} тема")
        style = args.get("style")
        if style in ("normal", "tsundere") and style != self.style:  # той самий характер модель часто «нагадує»
            cmd["style"] = style
        emote = args.get("emote")
        if emote in ("kiss", "angry", "shy") and not cmd:  # цю емоцію покаже саме обличчя після фрази
            tsundere = self.style == "tsundere"
            return {"kiss": "[поцілунок] " + ("Н… ну гаразд. Тримай поцілунок. Тільки нікому не кажи!" if tsundere
                                              else "Тримай повітряний поцілунок!"),
                    "angry": "[злість] " + ("Гм! Ось тобі. І не смійся, дурнику!" if tsundere
                                            else "Ось так я серджуся!"),
                    "shy": "[сором] " + ("Ой… н-не дивись на мене так!" if tsundere else "Ой, аж почервоніла.")}[emote]
        if emote in EMOTES:
            cmd["emote"] = emote
            said.append({"wink": "підморгую", "smile": "усміхаюсь", "hair": "поправляю волосся", "kiss": "цілую",
                         "angry": "насуплююсь", "shy": "ніяковію"}[emote])
        if not cmd:
            return "Нічого не змінила."
        if self.device_cmd is None:
            return "Мій пристрій зараз не на зв'язку."
        await self.device_cmd(cmd)
        self.device_state.update({k: v for k, v in cmd.items() if k not in ("emote", "style")})
        if "style" in cmd:
            self.style = style
            return ("Гм! Тепер я цундере. Т… тільки не звикай!" if style == "tsundere"
                    else "Добре, знову звичайна. Без бурчання.")
        if list(cmd) == ["emote"]:
            return said[0].capitalize() + "!"
        return "Готово: " + ", ".join(said) + "." if said else "Готово."

    # ------------------------------------------------------------------ Claude
    async def _claude(self, messages: list, text: str) -> str:
        messages.append({"role": "user", "content": text})
        for _ in range(MAX_TOOL_ROUNDS):
            r = await self.claude.beta.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=4000,
                system=system_prompt(self.home.online("pc"), self.device_state, self.memory.prompt_block(),
                                     self.style),
                tools=CLAUDE_TOOLS,
                messages=messages,
                output_config={"effort": CLAUDE_EFFORT},
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
            if r.stop_reason == "refusal":
                self.reset()
                return "[сум] Вибач, з цим я допомогти не можу."
            messages.append({"role": "assistant", "content": r.content})
            if r.stop_reason != "tool_use":
                return sanitize("".join(b.text for b in r.content if b.type == "text"))
            results = []
            for b in r.content:
                if b.type == "tool_use":
                    res, _direct = await self.run_tool(b.name, b.input)
                    results.append({"type": "tool_result", "tool_use_id": b.id, "content": res})
            messages.append({"role": "user", "content": results})
        return "[сум] Щось я заплуталась у діях. Спробуй ще раз."

    # ------------------------------------------------------------------ Groq
    async def _chat(self, msgs: list):
        last_error = None
        for model in GROQ_MODELS:
            extra = {"reasoning_effort": "low"} if model.startswith("openai/gpt-oss") else {"reasoning_effort": "none"}
            try:
                return await self.groq_chat.chat.completions.create(
                    model=model, messages=msgs, tools=GROQ_TOOLS, max_tokens=1200, temperature=0.5, **extra)
            except (groq.RateLimitError, groq.APIStatusError, groq.APIConnectionError) as e:
                log.warning("модель %s недоступна (%s) — пробую наступну", model, type(e).__name__)
                last_error = e
        raise RuntimeError(f"усі моделі Groq недоступні: {last_error}")

    async def _groq(self, history: list, text: str) -> str:
        msgs = [{"role": "system",
                 "content": system_prompt(self.home.online("pc"), self.device_state, self.memory.prompt_block(),
                                          self.style)},
                *history, {"role": "user", "content": text}]
        answer = "[сум] Щось я заплуталась у діях. Спробуй ще раз."
        for _ in range(MAX_TOOL_ROUNDS):
            r = await self._chat(msgs)
            m = r.choices[0].message
            if not m.tool_calls:
                answer = sanitize(m.content or "") or "[спокій] Готово."
                break
            msgs.append({
                "role": "assistant",
                "content": m.content or "",
                "tool_calls": [{"id": tc.id, "type": "function",
                                "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                               for tc in m.tool_calls],
            })
            results, all_direct = [], True
            for tc in m.tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                res, direct = await self.run_tool(tc.function.name, args)
                all_direct &= direct
                results.append(res)
                msgs.append({"role": "tool", "tool_call_id": tc.id, "content": res})
            if all_direct:  # готові фрази — без другого запиту до моделі
                ok = not any(w in " ".join(results).lower() for w in ("не вдалося", "немає", "офлайн", "не знайшла"))
                joined = " ".join(results)
                answer = joined if _MOOD_TAG.match(joined) else ("[радість] " if ok else "[сум] ") + joined
                break
        history += [{"role": "user", "content": text}, {"role": "assistant", "content": answer}]
        return answer
