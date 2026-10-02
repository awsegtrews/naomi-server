"""Чистка тексту для екрана Cardputer (його шрифт: латиниця, кирилиця, базова пунктуація) і для голосу."""
import re
import unicodedata

_EXTRA = set("«»—–…’‘“”№°·")


def _ok(ch: str) -> bool:
    o = ord(ch)
    return 0x20 <= o < 0x7F or 0x400 <= o <= 0x45F or ch in "ҐґЄєІіЇї" or ch in _EXTRA


def clean(text: str) -> str:
    text = re.sub(r"[*_#`>|~]+", "", text)              # markdown
    text = re.sub(r"^\s*[-•]\s+", "", text, flags=re.M)  # пункти списків
    text = re.sub(r"\s+", " ", text).strip()
    out = []
    for ch in text:
        if _ok(ch):
            out.append(ch)
            continue
        base = unicodedata.normalize("NFKD", ch)          # é -> e
        out.extend(c for c in base if _ok(c))
    return re.sub(r"\s{2,}", " ", "".join(out)).strip()
