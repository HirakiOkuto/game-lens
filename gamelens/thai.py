"""Thai has no spaces between words and Qt (without ICU) will not break a Thai line at all. Windows 10+ ships ICU
as icu.dll; its line-break iterator knows Thai words, so we insert zero-width spaces at its break points and Qt
wraps there."""
from __future__ import annotations

import ctypes
import functools
import re

ZWSP = "​"
UBRK_LINE = 2
_THAI = re.compile(r"[฀-๿]")
_THAI_CHARS = set(map(chr, range(0x0E00, 0x0E80)))
# signs that belong to the letter before them: above/below vowels and tone marks, and the vowels written after it
_FOLLOWS = set(map(chr, [0x0E30, 0x0E31, 0x0E32, 0x0E33, *range(0x0E34, 0x0E3B), 0x0E45, 0x0E46, *range(0x0E47, 0x0E4F)]))
_LEADING = re.compile(r"(?<=[ก-๎])(?=[เ-ไ])")  # before เ แ โ ใ ไ (not at the start)

try:
    _icu = ctypes.CDLL("icu.dll")
    _icu.ubrk_open.restype = ctypes.c_void_p
    _icu.ubrk_open.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p, ctypes.c_int32, ctypes.POINTER(ctypes.c_int)]
    _icu.ubrk_first.argtypes = [ctypes.c_void_p]
    _icu.ubrk_next.argtypes = [ctypes.c_void_p]
    _icu.ubrk_close.argtypes = [ctypes.c_void_p]
except (OSError, AttributeError):
    _icu = None


def segments(text: str) -> list[str]:
    """Line-break opportunities as a list of pieces (whole text if ICU is unavailable)."""
    if not text or _icu is None or max(map(ord, text)) > 0xFFFF:  # (ICU counts in UTF-16 units)
        return [text] if text else []
    status = ctypes.c_int(0)
    # the iterator reads the text where it lies, while it runs: a buffer of our own that lives until it is closed
    # (a str passed straight in is copied to a temporary that is gone after the call - the breaks came out of
    # whatever was there then, often letter by letter)
    buf = ctypes.create_unicode_buffer(text)
    it = _icu.ubrk_open(UBRK_LINE, b"th", ctypes.addressof(buf), len(text), ctypes.byref(status))
    if not it or status.value > 0:  # > 0 is an ICU error; negative values are warnings
        return [text]
    try:
        pos = [_icu.ubrk_first(it)]
        while (n := _icu.ubrk_next(it)) != -1:
            pos.append(n)
    finally:
        _icu.ubrk_close(it)
    return [text[a:b] for a, b in zip(pos, pos[1:])]


@functools.lru_cache(maxsize=2048)
def breakable(text: str) -> str:
    """Text with zero-width spaces between Thai words (unchanged if it has no Thai)."""
    if not _THAI.search(text or ""):
        return text
    # ICU does not know loanwords and breaks them letter by letter (เ|ค|วสต์, อ|น|ิม่า): a lone letter goes with
    # what follows it, and a sign that belongs to the letter before it (a vowel sign, a tone mark, ์) stays there
    parts, carry = [], ""
    for p in segments(text):
        p, carry = carry + p, ""
        if parts and (p[0] in _FOLLOWS or p[1:2] == "์"):
            parts[-1] += p
        elif len(p) == 1 and p in _THAI_CHARS:
            carry = p
        else:
            parts.append(p)
    if carry:
        parts.append(carry)
    # no syllable starts with a consonant and then a leading vowel: that consonant ends the one before (ICU: เท|มเควสต์)
    for i in range(1, len(parts)):
        p = parts[i]
        if len(p) >= 2 and "ก" <= p[0] <= "ฮ" and "เ" <= p[1] <= "ไ" and parts[i - 1][-1:] in _THAI_CHARS:
            parts[i - 1], parts[i] = parts[i - 1] + parts[i][0], parts[i][1:]
    # a long chunk ICU does not know (a loanword: ไอเทมเควสต์) may still break before a leading vowel, which always
    # starts a syllable - otherwise a narrow label wraps anywhere inside it (ไอเทมเค / วสต์)
    parts = [q for p in parts for q in (_LEADING.split(p) if sum(c in _THAI_CHARS for c in p) >= 6 else [p]) if q]
    out = []
    for i, p in enumerate(parts):
        out.append(p)
        if i + 1 < len(parts) and not p.endswith((" ", "\n", ZWSP)):
            out.append(ZWSP)
    return "".join(out)
