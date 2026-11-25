"""Reading typeset text off a game screen: RapidOCR (PP-OCR on onnxruntime), detection + recognition.

Japanese uses the PP-OCRv6 `japan` recogniser. Other languages are routed like this:
PP-OCRv6 covers Chinese and Latin-script languages; PP-OCRv5 adds Korean, Thai, Cyrillic, Arabic and more.
"""
from __future__ import annotations

import os
import threading

import cv2
import numpy as np
from PIL import Image

_rapid: dict[tuple[str, str], object] = {}
_lock = threading.Lock()

_V5 = {
    "ko": "korean", "th": "th", "el": "el", "ta": "ta", "te": "te",
    "ru": "eslav", "uk": "eslav", "be": "eslav",
    "bg": "cyrillic", "sr": "cyrillic", "mk": "cyrillic", "kk": "cyrillic", "mn": "cyrillic",
    "ar": "arabic", "fa": "arabic", "ur": "arabic",
    "hi": "devanagari", "mr": "devanagari", "ne": "devanagari",
}
_V6 = {"ja": "japan", "zh": "ch", "zh-cn": "ch", "zh-hk": "chinese_cht", "zh-tw": "chinese_cht"}
NO_SPACE = {"ja", "zh", "zh-cn", "zh-hk", "zh-tw", "th"}  # scripts written without spaces between words


def normalize_lang(lang: str) -> str:
    return (lang or "ja").lower().replace("_", "-")


def route(lang: str) -> tuple[str, str]:
    """(PP-OCR version, recognition language) for a source language."""
    lang = normalize_lang(lang)
    if lang in _V5:
        return ("PP-OCRv5", _V5[lang])
    if lang in _V6:
        return ("PP-OCRv6", _V6[lang])
    return ("PP-OCRv6", "en")  # multilingual Latin model (en, es, fr, de, pt, id, vi, ...)


def _get_rapid(version: str, rec_lang: str):
    key = (version, rec_lang)
    with _lock:
        if key not in _rapid:
            from rapidocr import LangDet, LangRec, ModelType, OCRVersion, RapidOCR

            v5 = version == "PP-OCRv5"
            _rapid[key] = RapidOCR(params={
                "Global.log_level": "warning",
                "Global.use_cls": False,  # game text is upright
                "EngineConfig.onnxruntime.use_cuda": os.environ.get("OCR_CUDA") == "1",  # CPU by default: ~0.2 s a box
                "Det.ocr_version": OCRVersion(version),
                "Det.lang_type": LangDet.CH if v5 else LangDet.EN,
                "Det.model_type": ModelType.MOBILE if v5 else ModelType.SMALL,
                "Rec.ocr_version": OCRVersion(version),
                "Rec.lang_type": LangRec(rec_lang),
                "Rec.model_type": ModelType.MOBILE if v5 else ModelType.SMALL,
            })
        return _rapid[key]


def _drop_ruby(lines: list[dict]) -> list[dict]:
    """Furigana: a line much thinner than a line right below it that it overlaps horizontally."""
    keep = []
    for l in lines:
        x, y, w, h = l["bbox"]
        ruby = False
        for m in lines:
            mx, my, mw, mh = m["bbox"]
            if m is l or mh < 1.6 * h:
                continue
            gap = my - (y + h)  # ruby sits just above the base line
            overlap = min(x + w, mx + mw) - max(x, mx)
            if -0.3 * mh <= gap <= 0.35 * mh and overlap > 0.5 * w:
                ruby = True
                break
        if not ruby:
            keep.append(l)
    return keep


def _rows(lines: list[dict]) -> list[dict]:
    """Top to bottom; pieces of one row left to right."""
    rows: list[list[dict]] = []
    for l in sorted(lines, key=lambda t: t["bbox"][1]):
        y, h = l["bbox"][1], l["bbox"][3]
        for row in rows:
            r0 = min(r["bbox"][1] for r in row)
            r1 = max(r["bbox"][1] + r["bbox"][3] for r in row)
            if min(r1, y + h) - max(r0, y) > 0.5 * min(h, r1 - r0):
                row.append(l)
                break
        else:
            rows.append([l])
    return [l for row in rows for l in sorted(row, key=lambda t: t["bbox"][0])]


def _drop_repeats(lines: list[dict]) -> list[dict]:
    """The detector sometimes returns a second box for a part of a line it already found: keep the better-read
    of two boxes that cover the same pixels (the longer text when they are about equally sure)."""
    keep: list[dict] = []
    for l in sorted(lines, key=lambda l: (-l["bbox"][2] * l["bbox"][3], -l["conf"])):
        x, y, w, h = l["bbox"]
        for k in keep:
            kx, ky, kw, kh = k["bbox"]
            iw, ih = min(x + w, kx + kw) - max(x, kx), min(y + h, ky + kh) - max(y, ky)
            if iw > 0 and ih > 0 and iw * ih > 0.6 * w * h:
                break
            # a ghost box hanging off the line it repeats: less overlap, but the text is a piece of that line's
            if iw > 0.8 * w and ih > 0.2 * h and _is_piece_of(l["text"], k["text"]):
                break
        else:
            keep.append(l)
    return keep


def _is_piece_of(part: str, whole: str) -> bool:
    import difflib

    if not 3 <= len(part) < len(whole):
        return False
    blocks = difflib.SequenceMatcher(None, part, whole, autojunk=False).get_matching_blocks()
    return sum(b.size for b in blocks) >= 0.75 * len(part)


def _tighten(img: np.ndarray, bbox: list[int]) -> list[int]:
    """PP-OCR's detector pads its boxes (and small crops are padded further): shrink a line box to its ink, so
    the box height is the glyph height (the client sizes the translation from it)."""
    x, y, w, h = bbox
    crop = img[y : y + h, x : x + w]
    if crop.size == 0 or w < 4 or h < 4:
        return bbox
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ink = th > 0 if (th > 0).mean() < 0.5 else th == 0
    rows = np.flatnonzero(ink.mean(1) > 0.01)
    cols = np.flatnonzero(ink.mean(0) > 0.02)
    if len(rows) < 2 or len(cols) < 2:
        return bbox
    y0, y1 = max(0, rows[0] - 1), min(h, rows[-1] + 2)
    x0, x1 = max(0, cols[0] - 1), min(w, cols[-1] + 2)
    if (y1 - y0) < 0.3 * h:
        return bbox  # most likely an underline or a frame line, not the text: keep the detector's box
    return [int(x + x0), int(y + y0), int(x1 - x0), int(y1 - y0)]


def read_lines(img: np.ndarray, lang: str = "ja", min_score: float = 0.5) -> list[dict]:
    """Horizontal lines of typeset text (game dialogue boxes, menus): PP-OCR detection + recognition on the
    whole image. Returns [{bbox:[x,y,w,h], text, conf}] in reading order (rows top to bottom, left to right)."""
    lang = normalize_lang(lang)
    version, rl = route(lang)
    engine = _get_rapid(version, rl)
    H, W = img.shape[:2]
    # a small crop (a speaker-name box) is enlarged: PP-OCR's detector misses short text near the image edge
    scale = 1.0 if min(H, W) >= 96 else min(3.0, 96 / max(1, min(H, W)))
    work = img if scale == 1.0 else np.array(Image.fromarray(img).resize((int(W * scale), int(H * scale)), Image.Resampling.LANCZOS))
    pad = 32
    padded = np.pad(work, ((pad, pad), (pad, pad), (0, 0)), mode="edge")  # game text sits on dark boxes too
    res = engine(padded, use_det=True, use_cls=False, use_rec=True)
    if res is None or res.txts is None or len(res.txts) == 0:
        if H * scale <= 160:  # one short line the detector did not find: read the whole crop as that line
            r = engine(work, use_det=False, use_cls=False, use_rec=True)
            if r is not None and r.txts and r.scores[0] >= max(min_score, 0.6) and r.txts[0].strip():
                return [{"bbox": [0, 0, W, H], "text": r.txts[0].strip(), "conf": round(float(r.scores[0]), 3)}]
        return []
    lines = []
    for box, txt, score in zip(res.boxes, res.txts, res.scores):
        if score < min_score or not txt.strip():
            continue
        xs, ys = (box[:, 0] - pad) / scale, (box[:, 1] - pad) / scale
        x0, y0 = max(0, int(xs.min())), max(0, int(ys.min()))
        x1, y1 = min(W, int(np.ceil(xs.max()))), min(H, int(np.ceil(ys.max())))
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue
        lines.append({"bbox": [x0, y0, x1 - x0, y1 - y0], "text": txt.strip(), "conf": round(float(score), 3)})
    lines = _drop_repeats([dict(l, bbox=_tighten(img, l["bbox"])) for l in lines])
    if lang in NO_SPACE:
        lines = _drop_ruby(lines)
    lines = _rows(lines)
    return lines


def _name_variants(img: np.ndarray):
    """Pictures of a name box with the scene behind the letters pushed away, twice the size (the first reading
    is of the picture as it is). Bright text on a bright scene is where the plain picture fails."""
    big = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(big, cv2.COLOR_RGB2GRAY)
    hi = float(np.percentile(gray, 99))
    levels = np.clip((gray.astype(np.float32) - (hi - 90)) / 90 * 255, 0, 255).astype(np.uint8)
    yield cv2.cvtColor(levels, cv2.COLOR_GRAY2RGB)
    top = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
    yield cv2.cvtColor(255 - cv2.normalize(top, None, 0, 255, cv2.NORM_MINMAX), cv2.COLOR_GRAY2RGB)


def read_name(img: np.ndarray, lang: str, find, fold) -> list[dict]:
    """A speaker-name box. find(text) -> the known name it is (as the glossary writes it) or None; fold(text)
    normalises a reading for comparison. The plain reading if it is a known name; otherwise the box is read two
    more ways and the first known name wins. A name nobody knows is believed only when two readings agree;
    otherwise [] (better to leave the original on screen than to translate a misreading)."""
    lines = read_lines(img, lang)
    readings = [("".join(l["text"] for l in lines), lines)]
    if (known := find(readings[0][0])):
        return _named(lines, known)
    for pic in _name_variants(img):
        ls = [dict(l, bbox=[v // 2 for v in l["bbox"]]) for l in read_lines(pic, lang)]
        text = "".join(l["text"] for l in ls)
        if (known := find(text)):
            return _named(ls, known)
        readings.append((text, ls))
    for i, (text, ls) in enumerate(readings):
        if text and any(fold(text) == fold(other) for j, (other, _) in enumerate(readings) if j != i):
            return ls
    return []


def _named(lines: list[dict], name: str) -> list[dict]:
    """One line carrying the known name, covering everything that was read."""
    x0, y0 = min(l["bbox"][0] for l in lines), min(l["bbox"][1] for l in lines)
    x1 = max(l["bbox"][0] + l["bbox"][2] for l in lines)
    y1 = max(l["bbox"][1] + l["bbox"][3] for l in lines)
    return [{"bbox": [x0, y0, x1 - x0, y1 - y0], "text": name, "conf": max(l["conf"] for l in lines)}]
