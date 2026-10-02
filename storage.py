"""Збереження пам'яті, списків і нагадувань між перезапусками хмари.

У безкоштовному Hugging Face Space файли контейнера зникають після перезапуску, тому дані
дублюються в приватний датасет того ж акаунта (NAOMI_DATA_REPO, токен HF_TOKEN).
Локально (без цих змінних) просто нічого не робить.
"""
import logging
import os
import threading
from pathlib import Path

log = logging.getLogger("naomi.storage")
HERE = Path(__file__).resolve().parent
REPO = os.environ.get("NAOMI_DATA_REPO", "")
TOKEN = os.environ.get("HF_TOKEN", "")
FILES = ["memory.json", "notes.json", "reminders.json"]
READ_ONLY = ["screen_assets.json"]  # портрет для великого екрана: лише в приватному сховищі, не в публічному коді
DELAY = 15  # секунд: кілька змін поспіль — одне збереження

_pending: set[str] = set()
_lock = threading.Lock()
_timer: threading.Timer | None = None


def enabled() -> bool:
    return bool(REPO and TOKEN)


def restore() -> None:
    """Перед стартом: забрати збережені файли з датасету."""
    if not enabled():
        return
    from huggingface_hub import hf_hub_download
    for name in FILES + READ_ONLY:
        try:
            hf_hub_download(REPO, name, repo_type="dataset", token=TOKEN, local_dir=str(HERE))
            log.info("відновлено %s", name)
        except Exception as e:  # файлу ще немає — це нормально
            log.info("%s у сховищі немає (%s)", name, type(e).__name__)


def _flush() -> None:
    global _timer
    with _lock:
        names, _timer = sorted(_pending), None
        _pending.clear()
    try:
        from huggingface_hub import CommitOperationAdd, HfApi
        ops = [CommitOperationAdd(path_in_repo=n, path_or_fileobj=str(HERE / n)) for n in names if (HERE / n).exists()]
        if ops:
            HfApi(token=TOKEN).create_commit(REPO, ops, repo_type="dataset", commit_message="Наомі: збереження")
            log.info("збережено в сховище: %s", ", ".join(names))
    except Exception:
        log.exception("не вдалося зберегти в сховище")


def changed(path: Path) -> None:
    """Викликається після кожного запису файлу даних."""
    global _timer
    if not enabled():
        return
    with _lock:
        _pending.add(Path(path).name)
        if _timer is None:
            _timer = threading.Timer(DELAY, _flush)
            _timer.daemon = True
            _timer.start()
