"""A line OCR read a little differently from one translated before (a voiced mark lost, a small kana read large,
ロ for 口, a stray 「 or 、) is the same line: its translation is used again instead of asking the engine, which may
word it differently - and costs a call. Only the misreadings OCR makes are forgiven, never an added or dropped word."""
from __future__ import annotations

import hashlib
import unicodedata

# characters that drop out of the comparison: OCR adds, loses and confuses these often, and they carry no words
NOISE = set(" 　\t「」『』【】（）()[]〈〉《》、。，．,.・･…‥!?！？~〜～_\"'“”‘’♪☆★♡")
# characters OCR reads as one another, each mapped to one of them (katakana is then compared as hiragana, so
# へ/ヘ, り/リ, ペ/ぺ need nothing here; a long-vowel mark is kept - ビル and ビール are two words - but its lookalikes
# are one with it)
LOOKALIKE = {**dict.fromkeys("口□", "ロ"), **dict.fromkeys("一—―‐-－", "ー"), "力": "カ", "工": "エ", "夕": "タ",
             "二": "ニ", "卜": "ト", "八": "ハ", "才": "オ", "千": "チ"}
SMALL = dict(zip("ぁぃぅぇぉっゃゅょゎゕゖ", "あいうえおつやゆよわかけ"))
MIN_LEN = 8  # (shorter text: one character more or less is a different word too often)


def _chars(text: str) -> list[tuple[str, str]]:
    """(character as read, its skeleton) for each character that counts."""
    out = []
    for c in unicodedata.normalize("NFKC", text):
        if c in NOISE:
            continue
        s = LOOKALIKE.get(c, c)
        s = "".join(x for x in unicodedata.normalize("NFD", s) if x not in "゙゚")  # (voiced marks)
        s = LOOKALIKE.get(s, s)
        if "ァ" <= s <= "ヶ":
            s = chr(ord(s) - 0x60)  # katakana -> hiragana
        out.append((c, SMALL.get(s, s)))
    return out


def fuzzy_key(scope: str, text: str) -> str | None:
    """Key the line shares with every line it could be a misreading of; None for a line too short to judge."""
    ch = _chars(text)
    if len(ch) < MIN_LEN:
        return None
    return hashlib.sha1(f"{scope}|{''.join(s for _, s in ch)}".encode()).hexdigest()


def differences(a: str, b: str) -> int | None:
    """How many characters two lines with the same key read differently (None if they are not alike at all)."""
    ca, cb = _chars(a), _chars(b)
    if len(ca) != len(cb) or any(x[1] != y[1] for x, y in zip(ca, cb)):
        return None
    if [c for c, _ in ca if c.isdigit()] != [c for c, _ in cb if c.isdigit()]:
        return None  # (numbers are never a misreading to forgive: "3" and "8" read alike, mean otherwise)
    return sum(x[0] != y[0] for x, y in zip(ca, cb))


def close_enough(a: str, b: str) -> bool:
    """b is a, read a little differently: at most one character in ten read otherwise (at least one allowed)."""
    d = differences(a, b)
    return d is not None and d <= max(1, len(_chars(a)) // 10)
