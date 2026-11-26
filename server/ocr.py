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


def _wide(c: str) -> bool:
    return "⺀" <= c <= "鿿" or "가" <= c <= "힣" or "！" <= c <= "｠"


def _tighten(img: np.ndarray, bbox: list[int], text: str = "") -> list[int]:
    """PP-OCR's detector pads its boxes (and small crops are padded further): shrink a line box to its ink, so
    the box height is the glyph height (the client sizes the translation from it). text: what was read there -
    a box too narrow for its letters lost some of them to the threshold."""
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
    # A box too narrow for the letters read in it lost some to the threshold: a selected tab's glow made its
    # first two letters look like background, and the translation went over the last two only
    # (Japanese letters are at least 0.6 as wide as they are high, Latin ones a quarter; only a cut of about a
    # letter counts, and only on a line set across)
    gh = y1 - y0
    need = gh * sum(0.6 if _wide(c) else 0.25 for c in text.strip())
    lost = w > h and (x1 - x0) < need and w - (x1 - x0) >= 0.8 * gh
    if w - (x1 - x0) > max(0.08 * w, 1.5 * h) or lost:
        # much more cut off than the detector's padding: one threshold does not tell the letters from what is
        # behind them (light text over the 3D scene: its coloured words and the end of the line were lost).
        # The detector's own width stays; the height is still the letters'
        x0, x1 = 0, w
    return [int(x + x0), int(y + y0), int(x1 - x0), int(y1 - y0)]


_ICON_AS_TEXT = re.compile(r"^[0-9０-９?？!！](?=[぀-ヿ一-鿿])")


def _drop_icon(img: np.ndarray, line: dict) -> dict:
    """An icon before a label, read as its first letter (a help button's book with a ? on it: 2ヘルプ). It is a solid
    block about as wide as the letters are high, apart from them; a digit is a thin stroke (01キボ小屋 stays)."""
    m = _ICON_AS_TEXT.match(line["text"])
    if not m:
        return line
    x, y, w, h = line["bbox"]
    crop = img[y : y + h, x : x + w]
    if crop.size == 0 or h < 6:
        return line
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ink = th > 0 if (th > 0).mean() < 0.5 else th == 0
    rows = np.flatnonzero(ink.mean(1) > 0.01)
    cols = np.flatnonzero(ink.mean(0) > 0.02)
    if len(rows) < 2 or len(cols) < 2:
        return line
    gh = rows[-1] - rows[0] + 1
    breaks = np.flatnonzero(np.diff(cols) > 1)
    if not len(breaks):
        return line
    a, b, c = cols[0], cols[breaks[0]] + 1, cols[breaks[0] + 1]  # the first block [a, b), the next one from c
    fill = ink[rows[0] : rows[-1] + 1, a:b].mean()
    if (b - a) < 0.75 * gh or fill < 0.65 or (c - b) < 0.3 * gh:
        return line
    dx, dy, dw, dh = line["dbox"]
    cut = x + c - 1
    return dict(line, text=line["text"][m.end():].lstrip(), bbox=[int(cut), y, int(x + w - cut), h],
                dbox=[int(max(dx, cut - 2)), dy, int(dx + dw - max(dx, cut - 2)), dh])


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
        line = {"bbox": [x0, y0, x1 - x0, y1 - y0], "text": txt.strip(), "conf": round(float(score), 3)}
        # text set at an angle (a title across a picture): its box is much taller than its letters. The angle
        # (degrees, clockwise) and the line's own length and height along it go with it
        top = box[1].astype(float) - box[0].astype(float)
        tilt = float(np.degrees(np.arctan2(top[1], top[0])))
        if 3 <= abs(tilt) <= 45:
            line.update(tilt=round(tilt, 1), tw=round(float(np.hypot(*top)) / scale),
                        th=round(float(np.hypot(*(box[3].astype(float) - box[0].astype(float)))) / scale))
        lines.append(line)
    # dbox is the detector's own box: steady in height and position even over a busy scene, where tightening
    # (bbox, right for text on a panel) is fooled by what lies behind the glyphs
    lines = _drop_repeats([_drop_icon(img, dict(l, bbox=_tighten(img, l["bbox"], l["text"]), dbox=l["bbox"])) for l in lines])
    if lang in NO_SPACE:
        lines = _drop_ruby(lines)
    lines = _rows(lines)
    if lang == "ja":
        for l in lines:
            l["text"] = fix_kana(l["text"])
    return lines


def _script_chars(text: str, lang: str) -> int:
    """Letters of the language's own script in a reading (icons are read as stray marks and Latin letters)."""
    if lang == "ja" or lang.startswith("zh"):
        return sum("぀" <= c <= "ヿ" or "一" <= c <= "鿿" for c in text)
    if lang == "ko":
        return sum("가" <= c <= "힣" for c in text)
    return sum(c.isalpha() for c in text)


def _read_new_boxes(img: np.ndarray, lang: str, engine, lines: list[dict], boxes, most: int, enlarge: float = 1.0,
                    lone: bool = False, sure: float = 0.9, grow: bool = False) -> list[dict]:
    """Of the boxes a second look found, those where no line is yet: read from the picture itself (enlarged by
    `enlarge` first: small thin letters read better). Only readings in the language's own letters, read surely,
    count (icons come out as stray marks). lone: a single letter may count too (遠 / 近 on round buttons) when it is
    read all but certainly. grow: the box may have lost the end of the line to the scene behind it
    (テンプレ|ート over bright grass): it is widened while the reading grows and stays as sure."""

    def read(x, y, w, h):
        crop = img[y:y + h, x:x + w]
        if enlarge != 1.0:
            crop = cv2.resize(crop, None, fx=enlarge, fy=enlarge, interpolation=cv2.INTER_CUBIC)
        r = engine(crop, use_det=False, use_cls=False, use_rec=True)
        return (r.txts[0].strip(), float(r.scores[0])) if r is not None and r.txts else ("", 0.0)

    H, W = img.shape[:2]
    found = []
    for x0, y0, x1, y1 in boxes:
        x0, y0, x1, y1 = max(0, int(x0)), max(0, int(y0)), min(W, int(np.ceil(x1))), min(H, int(np.ceil(y1)))
        w, h = x1 - x0, y1 - y0
        if h < 8 or w < (0.8 if lone else 1.5) * h:
            continue
        covered = max((max(0, min(x1, l["dbox"][0] + l["dbox"][2]) - max(x0, l["dbox"][0]))
                       * max(0, min(y1, l["dbox"][1] + l["dbox"][3]) - max(y0, l["dbox"][1])) for l in lines), default=0)
        if covered <= 0.2 * w * h:
            found.append((x0, y0, w, h))
    out: list[dict] = []
    need = 2 if lang in NO_SPACE or lang == "ko" else 3
    for x, y, w, h in sorted(found, key=lambda b: -b[3])[:most]:
        text, conf = read(x, y, w, h)
        while grow and text and x + w < W:
            step = min(W - x - w, max(4, round(0.8 * h)))
            t2, c2 = read(x, y, w + step, h)
            if len(t2) <= len(text) or not t2.startswith(text[:-1]) or c2 < max(sure, conf - 0.05):
                break
            text, conf, w = t2, max(conf, c2), w + step
        if conf < sure:
            continue
        n = _script_chars(text, lang)
        if n >= need or (lone and n == 1 and len(text) == 1 and conf >= 0.98):
            line = {"bbox": _tighten(img, [x, y, w, h], text), "dbox": [x, y, w, h],
                    "text": fix_kana(text) if lang == "ja" else text, "conf": round(conf, 3)}
            if not any(_overlap(line["dbox"], o["dbox"]) > 0.5 for o in out):  # (tiles overlap: found twice)
                out.append(line)
    return out


def _overlap(a: list[int], b: list[int]) -> float:
    """How much of the smaller of two boxes the other covers."""
    iw = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
    ih = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
    return max(0, iw) * max(0, ih) / max(1, min(a[2] * a[3], b[2] * b[3]))


def faint_lines(img: np.ndarray, lang: str, lines: list[dict], most: int = 16) -> list[dict]:
    """Lines of a whole screen that the detector missed: dim text on a panel of nearly its own colour (a tab that is
    not selected, grey on grey). The screen's contrast is raised piece by piece (CLAHE) and only the detector
    looks again."""
    lang = normalize_lang(lang)
    engine = _get_rapid(*route(lang))
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 9)).apply(lab[..., 0])
    pad = 32
    res = engine(np.pad(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB), ((pad, pad), (pad, pad), (0, 0)), mode="edge"),
                 use_det=True, use_cls=False, use_rec=False)
    if res is None or res.boxes is None:
        return []
    boxes = [(b[:, 0].min() - pad, b[:, 1].min() - pad, b[:, 0].max() - pad, b[:, 1].max() - pad) for b in res.boxes]
    return _read_new_boxes(img, lang, engine, lines, boxes, most)


def busy(img: np.ndarray) -> float:
    """Share of a screen that is fine detail (the open world's grass and leaves; a menu's panels are plain)."""
    g = cv2.cvtColor(cv2.resize(img, (480, 270), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY).astype(np.float32)
    return float((np.abs(cv2.Laplacian(g, cv2.CV_32F)) > 12).mean())


def scene_lines(img: np.ndarray, lang: str, lines: list[dict], most: int = 24) -> list[dict]:
    """Lines over the moving world that the detector does not see at the picture's own size: small thin light
    letters on bright grass (photo mode's 表情 / テンプレート / シーン score nothing at all). Colour is taken out
    (the letters are grey, the grass is green and yellow), each half of the screen is looked at twice as big, and
    what is found where no line is gets read from the picture, enlarged. About 0.8 s on the graphics card, so only
    for a busy screen with little text found (see read_screen)."""
    lang = normalize_lang(lang)
    engine = _get_rapid(*route(lang))
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    s = hsv[..., 1].astype(np.float32) / 255
    grey = cv2.cvtColor(np.clip(hsv[..., 2] * (1 - s) ** 2, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    H, W = img.shape[:2]
    k, pad, over = 2.0, 32, 40
    boxes = []
    for x0, x1 in ((0, W // 2 + over), (W // 2 - over, W)):
        big = cv2.resize(grey[:, x0:x1], None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC)
        res = engine(np.pad(big, ((pad, pad), (pad, pad), (0, 0)), mode="edge"), use_det=True, use_cls=False, use_rec=False)
        if res is not None and res.boxes is not None:
            boxes += [((b[:, 0].min() - pad) / k + x0, (b[:, 1].min() - pad) / k, (b[:, 0].max() - pad) / k + x0,
                       (b[:, 1].max() - pad) / k) for b in res.boxes]
    return _read_new_boxes(img, lang, engine, lines, boxes, most, enlarge=2.0, lone=True, sure=0.8, grow=True)


SCENE_BUSY = 0.4  # busy(): menus 0.02-0.3, the open world 0.55


def read_screen(img: np.ndarray, lang: str = "ja", min_score: float = 0.5, deep: bool = True) -> list[dict]:
    """A whole screen: its lines, and the faint ones the detector misses at first. deep: over a busy scene where
    little text was found, look for the small letters on it too (slow; a quick second look leaves it out)."""
    lines = read_lines(img, lang, min_score)
    if min(img.shape[:2]) >= 400:
        lines = _rows(lines + faint_lines(img, lang, lines))
        if deep and len(lines) < 8 and busy(img) >= SCENE_BUSY:
            lines = _rows(lines + scene_lines(img, lang, lines))
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
