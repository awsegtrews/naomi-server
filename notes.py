"""Списки й нотатки: «запиши в покупки молоко», «що в списку покупок?»."""
import json
import logging
from pathlib import Path

import storage

log = logging.getLogger("naomi.notes")
FILE = Path(__file__).with_name("notes.json")


class Notes:
    def __init__(self) -> None:
        try:
            self.lists: dict[str, list[str]] = json.loads(FILE.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self.lists = {}

    def _save(self) -> None:
        try:
            FILE.write_text(json.dumps(self.lists, ensure_ascii=False, indent=1), encoding="utf-8")
            storage.changed(FILE)
        except OSError:
            log.warning("не вдалося зберегти нотатки")

    @staticmethod
    def _name(name: str | None) -> str:
        return (name or "нотатки").strip().lower()

    def add(self, name: str | None, items: list[str]) -> str:
        n = self._name(name)
        items = [i.strip() for i in items if i and i.strip()]
        if not items:
            return "Що саме записати?"
        self.lists.setdefault(n, []).extend(items)
        self._save()
        return f"Записала в «{n}»: {', '.join(items)}."

    def show(self, name: str | None) -> str:
        n = self._name(name)
        items = self.lists.get(n)
        if not items:
            others = ", ".join(f"«{k}»" for k, v in self.lists.items() if v)
            return f"Список «{n}» порожній." + (f" Є списки: {others}." if others else "")
        return f"У списку «{n}»: {', '.join(items)}."

    def remove(self, name: str | None, items: list[str]) -> str:
        n = self._name(name)
        have = self.lists.get(n, [])
        gone = [h for h in have if any(i.lower() in h.lower() for i in items)]
        self.lists[n] = [h for h in have if h not in gone]
        self._save()
        return f"Викреслила: {', '.join(gone)}." if gone else "Такого в списку не знайшла."

    def clear(self, name: str | None) -> str:
        n = self._name(name)
        self.lists.pop(n, None)
        self._save()
        return f"Очистила список «{n}»."
