"""Reading typeset text off a game screen: RapidOCR (PP-OCR on onnxruntime), detection + recognition.

Japanese uses the PP-OCRv6 `japan` recogniser. Other languages are routed like this:
PP-OCRv6 covers Chinese and Latin-script languages; PP-OCRv5 adds Korean, Thai, Cyrillic, Arabic and more.
"""
from __future__ import annotations

import logging
import os
import re
import threading

import cv2
import numpy as np
from PIL import Image

log = logging.getLogger("uvicorn.error")  # (the server log shows what uvicorn logs)

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
            params = {
                "Global.log_level": "warning",
                "Global.use_cls": False,  # game text is upright
                # on the graphics card through DirectML (any DirectX 12 card): a whole screen in ~0.3 s instead of
                # ~2 s, read the same. OCR_GPU=0 keeps it on the processor.
                "EngineConfig.onnxruntime.use_dml": os.environ.get("OCR_GPU", "1") != "0",
                # 4 threads read as fast as all of them and leave the rest of the CPU to the game
                "EngineConfig.onnxruntime.intra_op_num_threads": int(os.environ.get("OCR_THREADS", "4")),
                # read at the picture's own size: RapidOCR's default blows a wide dialogue strip up several times,
                # which takes three times as long and reads worse
                "Det.limit_type": "max",
                "Det.limit_side_len": 2000,
                "Det.ocr_version": OCRVersion(version),
                "Det.lang_type": LangDet.CH if v5 else LangDet.EN,
                "Det.model_type": ModelType.MOBILE if v5 else ModelType.SMALL,
                "Rec.ocr_version": OCRVersion(version),
                "Rec.lang_type": LangRec(rec_lang),
                # the medium recogniser tells voiced kana apart (しっぽ / しっほ, まず / ます) where small does not
                "Rec.model_type": ModelType.MOBILE if v5 else ModelType.MEDIUM,
            }
            # the models load on first use: read a small picture now, so a graphics driver DirectML cannot use
            # shows here (then the processor reads)
            probe = np.full((48, 240, 3), 255, np.uint8)
            cv2.putText(probe, "OCR 123", (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            try:
                engine = RapidOCR(params=params)
                engine(probe)
            except Exception as e:  # noqa: BLE001
                log.warning("text reader on the graphics card failed (%s); using the processor", e)
                engine = RapidOCR(params={**params, "EngineConfig.onnxruntime.use_dml": False})
                engine(probe)
            _rapid[key] = engine
            log.info("text reader %s/%s on %s", version, rec_lang, _provider(engine))
        return _rapid[key]


def _provider(engine) -> str:
    """Which device the text reader runs on (for the log)."""
    try:
        return engine.text_det.session.session.get_providers()[0]
    except Exception:  # noqa: BLE001 - rapidocr's insides may change; this is only for the log
        return "?"


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


_KATA = r"[ァ-ヺー]"
_FIXES = [  # look-alikes the recogniser mixes up, decided by their neighbours
    (re.compile(r"(?<=[ぁ-ゖァ-ヺ])[一—―](?=[ぁ-ゖァ-ヺ、。！？!?～〜…])"), "ー"),
    (re.compile(rf"(?<={_KATA})べ|べ(?={_KATA})"), "ベ"),
    (re.compile(rf"(?<={_KATA})ぺ|ぺ(?={_KATA})"), "ペ"),
    (re.compile(rf"(?<={_KATA})へ(?={_KATA})"), "ヘ"),
    (re.compile(r"(?<=[ぁ-ゖ])ベ(?=[ぁ-ゖ])"), "べ"),
]


def fix_kana(text: str) -> str:
    for pat, to in _FIXES:
        text = pat.sub(to, text)
    return text


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
                return [{"bbox": [0, 0, W, H], "dbox": [0, 0, W, H], "text": r.txts[0].strip(),
                         "conf": round(float(r.scores[0]), 3)}]
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
    # dbox is the detector's own box: steady in height and position even over a busy scene, where tightening
    # (bbox, right for text on a panel) is fooled by what lies behind the glyphs
    lines = _drop_repeats([dict(l, bbox=_tighten(img, l["bbox"]), dbox=l["bbox"]) for l in lines])
    if lang in NO_SPACE:
        lines = _drop_ruby(lines)
    lines = _rows(lines)
    if lang == "ja":
        for l in lines:
            l["text"] = fix_kana(l["text"])
    return lines


def _levels(gray: np.ndarray, pct: float = 99) -> np.ndarray:
    """Everything but the brightest pixels pushed to black."""
    hi = float(np.percentile(gray, pct))
    return np.clip((gray.astype(np.float32) - (hi - 90)) / 90 * 255, 0, 255).astype(np.uint8)


def _centred_rows(lines: list[dict], width: int) -> list[dict]:
    """The lines of rows that are set around the middle. A row is judged as a whole: a gap inside it (名付けて
    ベタベタ弾！) makes the detector return pieces of it."""
    rows: list[list[dict]] = []
    for l in sorted(lines, key=lambda l: l["dbox"][1] + l["dbox"][3] / 2):
        c = l["dbox"][1] + l["dbox"][3] / 2
        if rows and abs(c - (rows[-1][0]["dbox"][1] + rows[-1][0]["dbox"][3] / 2)) < 0.5 * l["dbox"][3]:
            rows[-1].append(l)
        else:
            rows.append([l])
    out = []
    for row in rows:
        x0 = min(l["dbox"][0] for l in row)
        x1 = max(l["dbox"][0] + l["dbox"][2] for l in row)
        if abs((x0 + x1) / 2 - width / 2) <= max(0.035 * width, 1.2 * row[0]["dbox"][3]):
            out += row
    return [l for l in lines if any(l is o for o in out)]


def read_dialog(img: np.ndarray, lang: str, centred: bool, has_text) -> list[dict]:
    """A dialogue region. centred: the game sets its dialogue around the region's middle, so a line anywhere else
    is not dialogue (a menu that happens to be under the region) or is not complete yet, and is ignored.
    Over a bright scene the plain reading loses parts of a line; then the picture is read again with the scene
    pushed away and the fuller reading wins. has_text(text): is it in the game's script at all."""
    W = img.shape[1]

    def good(lines):
        lines = [l for l in lines if l["conf"] >= 0.6]
        if centred:
            lines = _centred_rows(lines, W)
        jp = [l for l in lines if has_text(l["text"])]
        # marks standing apart ("！" after a gap) belong to a row that has text
        return [l for l in lines if l in jp or any(abs(l["dbox"][1] - k["dbox"][1]) < 0.5 * k["dbox"][3] for k in jp)]

    plain = read_lines(img, lang)
    best = good(plain)
    score = lambda ls: sum(len(l["text"]) * l["conf"] for l in ls)  # noqa: E731
    doubtful = not best or len(best) < len([l for l in plain if has_text(l["text"])]) or min(l["conf"] for l in best) < 0.93
    if doubtful:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        again = good(read_lines(cv2.cvtColor(_levels(gray, 99.3), cv2.COLOR_GRAY2RGB), lang))
        best = again if score(again) > 1.03 * score(best) else best
    if sum(len(l["text"]) for l in best) <= 3:
        # a word or two (うん。 응.): the detector can miss a lone glyph, or half of it; bigger, it finds it
        k = 1.5
        big = read_lines(cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC), lang)
        big = good([dict(l, bbox=[round(v / k) for v in l["bbox"]], dbox=[round(v / k) for v in l["dbox"]]) for l in big])
        best = big if score(big) > 1.03 * score(best) else best
    return best


def _name_variants(img: np.ndarray):
    """Pictures of a name box with the scene behind the letters pushed away, twice the size (the first reading
    is of the picture as it is). Bright text on a bright scene is where the plain picture fails."""
    big = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(big, cv2.COLOR_RGB2GRAY)
    yield cv2.cvtColor(_levels(gray), cv2.COLOR_GRAY2RGB)
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
        ls = [dict(l, bbox=[v // 2 for v in l["bbox"]], dbox=[v // 2 for v in l["dbox"]]) for l in read_lines(pic, lang)]
        text = "".join(l["text"] for l in ls)
        if (known := find(text)):
            return _named(ls, known)
        readings.append((text, ls))
    for i, (text, ls) in enumerate(readings):
        if text and any(fold(text) == fold(other) for j, (other, _) in enumerate(readings) if j != i):
            return ls
    return []


def _named(lines: list[dict], name: str) -> list[dict]:
    """One line carrying the known name, where its longest piece was read (specks read beside it do not count)."""
    main = max(lines, key=lambda l: len(l["text"]))
    return [{**main, "text": name, "conf": max(l["conf"] for l in lines)}]
