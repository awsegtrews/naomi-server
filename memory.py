"""Довга пам'ять Наомі: факти про власника, його сценарії й обраний характер. Зберігається у файлі."""
import json
import logging
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import storage

log = logging.getLogger("naomi.memory")
FILE = Path(__file__).with_name("memory.json")
MAX_FACTS = 40


class Memory:
    def __init__(self) -> None:
        self.facts: list[dict] = []
        self.scenarios: dict[str, str] = {}
        self.persona = ""
        self.alert_region = "м. Київ"
        try:
            d = json.loads(FILE.read_text(encoding="utf-8"))
            self.facts = d.get("facts", [])
            self.scenarios = d.get("scenarios", {})
            self.persona = d.get("persona", "")
            self.alert_region = d.get("alert_region", self.alert_region)
        except (OSError, ValueError):
            pass

    def _save(self) -> None:
        try:
            FILE.write_text(json.dumps({"facts": self.facts, "scenarios": self.scenarios, "persona": self.persona,
                                        "alert_region": self.alert_region}, ensure_ascii=False, indent=1),
                            encoding="utf-8")
            storage.changed(FILE)
        except OSError:
            log.warning("не вдалося зберегти пам'ять")

    # ---- факти
    def remember(self, text: str) -> str:
        text = text.strip()
        if not text:
            return "Що саме запам'ятати?"
        self.facts.append({"text": text, "date": datetime.now(ZoneInfo("Europe/Kyiv")).strftime("%d.%m.%Y")})
        self.facts = self.facts[-MAX_FACTS:]
        self._save()
        return f"Запам'ятала: {text}"

    def forget(self, text: str) -> str:
        words = [w for w in text.lower().split() if len(w) > 3] or [text.lower()]
        before = len(self.facts)
        self.facts = [f for f in self.facts if not any(w in f["text"].lower() for w in words)]
        self._save()
        return "Забула." if len(self.facts) < before else "Такого не пам'ятаю."

    # ---- сценарії
    def save_scenario(self, name: str, steps: str) -> str:
        name = name.strip().lower()
        if not name or not steps.strip():
            return "Скажи назву сценарію і що в ньому робити."
        self.scenarios[name] = steps.strip()
        self._save()
        return f"Сценарій «{name}» збережено. Скажи «{name}», і я все зроблю."

    def delete_scenario(self, name: str) -> str:
        return "Видалила сценарій." if self.scenarios.pop(name.strip().lower(), None) else "Такого сценарію немає."

    # ---- характер
    def set_persona(self, text: str) -> str:
        self.persona = "" if text.strip().lower() in ("", "звичайна", "звичайний", "як раніше") else text.strip()
        self._save()
        return "Добре, буду собою." if not self.persona else "Добре, так і буде."

    def set_region(self, region: str) -> str:
        self.alert_region = region.strip()
        self._save()
        return f"Тепер стежу за тривогами: {self.alert_region}."

    def listing(self) -> str:
        parts = []
        if self.facts:
            parts.append("Факти: " + "; ".join(f["text"] for f in self.facts))
        if self.scenarios:
            parts.append("Сценарії: " + "; ".join(self.scenarios))
        return " ".join(parts) or "Поки нічого не пам'ятаю."

    def prompt_block(self) -> str:
        lines = []
        if self.facts:
            lines.append("Що ти пам'ятаєш про власника: " + "; ".join(f"{f['text']} ({f['date']})" for f in self.facts))
        if self.scenarios:
            lines.append("Його сценарії (коли він називає сценарій — виконай усі кроки інструментами): " +
                         "; ".join(f"«{k}»: {v}" for k, v in self.scenarios.items()))
        if self.persona:
            lines.append(f"Твій характер зараз: {self.persona}.")
        return "\n".join(lines)
