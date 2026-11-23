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

try:
    _icu = ctypes.CDLL("icu.dll")
    _icu.ubrk_open.restype = ctypes.c_void_p
    _icu.ubrk_open.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_wchar_p, ctypes.c_int32, ctypes.POINTER(ctypes.c_int)]
    _icu.ubrk_first.argtypes = [ctypes.c_void_p]
    _icu.ubrk_next.argtypes = [ctypes.c_void_p]
    _icu.ubrk_close.argtypes = [ctypes.c_void_p]
except (OSError, AttributeError):
    _icu = None


def segments(text: str) -> list[str]:
    """Line-break opportunities as a list of pieces (whole text if ICU is unavailable)."""
    if not text or _icu is None:
        return [text] if text else []
    status = ctypes.c_int(0)
    it = _icu.ubrk_open(UBRK_LINE, b"th", text, len(text), ctypes.byref(status))
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
    parts = segments(text)
    out = []
    for i, p in enumerate(parts):
        out.append(p)
        if i + 1 < len(parts) and not p.endswith((" ", "\n", ZWSP)):
            out.append(ZWSP)
    return "".join(out)
