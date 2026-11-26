"""Whole-screen translation (menus, settings, quest lists): which OCR lines to translate and where to draw them.
Kept free of app state so tools/render_screen.py can show exactly what the overlay would draw on a screenshot."""
from __future__ import annotations

import re
from collections import deque

import cv2
import numpy as np

from .overlay import Item, text_rect_for, to_qimage
from .render import SIZE_FACTOR, Line, analyze_line, erase_lines, learn_strokes, luminance, outline_for, over_scene
from .watcher import plain_text, similar

# characters of the source script: a line without any is a number, a resolution, "OFF", "TAA"... and stays as it is
SCRIPT = {
    "ja": re.compile(r"[぀-ヿ㐀-鿿]"),
    "zh": re.compile(r"[㐀-鿿]"),
    "ko": re.compile(r"[가-힯ᄀ-ᇿ]"),
}


def in_any(bbox: list[int], rects: list) -> bool:
    """True when the centre of bbox lies inside one of rects (x, y, w, h)."""
    cx, cy = bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2
    return any(x <= cx <= x + w and y <= cy <= y + h for x, y, w, h in rects)


def has_source_text(text: str, src_lang: str) -> bool:
    pat = SCRIPT.get((src_lang or "").lower().split("-")[0])
    return bool(pat.search(text)) if pat else any(c.isalpha() for c in text)


def lines_to_translate(lines: list[dict], skip: list, src_lang: str = "ja", min_conf: float = 0.75) -> list[dict]:
    """Lines worth translating: in the source script, read with some confidence (an emblem or a pattern "read"
    as a character scores low), and not inside a region that translates itself."""
    return [l for l in lines if has_source_text(l["text"], src_lang) and l.get("conf", 1.0) >= min_conf
            and not in_any(l["bbox"], skip)]


def text_marks(frame: np.ndarray, boxes, texts=None) -> list:
    """What the text in each box looks like (its edges), and about how many letters it has (box width / height
    unless the texts are given): kept to tell later whether that text is still there."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    out = []
    for i, (x, y, w, h) in enumerate(boxes):
        x0, y0 = max(0, int(x)), max(0, int(y))
        crop = gray[y0 : y0 + int(h), x0 : x0 + int(w)]
        if crop.size:
            n = len(texts[i]) if texts else max(1, int(w) // max(1, int(h)))
            out.append(((x0, y0, crop.shape[1], crop.shape[0]), cv2.Canny(crop, 60, 160) > 0, n))
    return out


def text_still_there(frame: np.ndarray, marks: list) -> bool:
    """Is the text that was translated still on screen as it was? A box whose edges are mostly gone has other text
    now (or none). A short label or two may flicker over something moving; a sentence that changed (the next tip
    in a bubble), or a good part of all the text, means the translation no longer fits."""
    if not marks:
        return True
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    gone = 0
    total = sum(n for _, _, n in marks)
    for (x, y, w, h), edges, n in marks:
        crop = gray[y : y + h, x : x + w]
        if crop.shape != edges.shape:
            continue
        now = cv2.dilate((cv2.Canny(crop, 60, 160) > 0).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        e = int(edges.sum())
        if e and (edges & now).sum() < 0.6 * e:
            if n >= 8:
                return False
            gone += n
    return gone < 0.2 * total


def common_heights(heights: list[int]) -> list[float]:
    """Labels of one menu are set in one size, but their measured glyph heights differ by a few pixels with the
    characters in them. Each height becomes the median of the heights close to it, so they come out equal."""
    out = []
    for h in heights:
        peers = sorted(v for v in heights if abs(v - h) <= 0.25 * max(v, h))
        out.append(float(peers[len(peers) // 2]))
    return out


def place(line: Line, size: float) -> tuple[tuple[int, int, int, int], str]:
    """(rect, align) for the translation of one line: it starts where the original starts and may run over the
    empty background beside it. Text with equal room on both sides sits centred in something (a button, a
    drop-down), so its translation is centred there too; text hard against the right is right-aligned."""
    x, _, w, _ = line.box
    L, R = line.free_left, line.free_right
    cy = (line.glyph_y[0] + line.glyph_y[1]) / 2
    tall = int(max(1.9 * line.glyph_h, 1.6 * size))
    y0 = int(cy - tall / 2)
    margin = 4
    if min(L, R) >= max(8, 0.6 * line.glyph_h) and abs(L - R) <= max(8, 0.3 * max(L, R)):
        half = w / 2 + min(L, R) - margin
        return (int(x + w / 2 - half), y0, int(2 * half), tall), "center"
    if R < 0.5 * w and L > 3 * max(R, margin):
        return (x - L + margin, y0, w + L - margin, tall), "right"
    return (x, y0, w + max(0, R - margin), tall), "left"


def paragraphs(lines: list[dict]) -> list[list[dict]]:
    """Group lines that belong to one block of running text: stacked tightly (closer than rows of a list are),
    the same size, either starting at the same left edge or centred on the same middle, and the line above long
    enough to have wrapped. Short labels stacked in a table or under an icon stay apart."""
    groups: list[list[dict]] = []
    for l in sorted(lines, key=lambda l: (l["bbox"][1], l["bbox"][0])):
        x, y, w, h = l["bbox"]
        for g in groups:
            px, py, pw, ph = g[-1]["bbox"]
            gap = y - (py + ph)
            same_size = abs(h - ph) <= 0.3 * max(h, ph)
            same_left = abs(x - g[0]["bbox"][0]) <= 0.6 * h
            same_middle = abs((x + w / 2) - (g[0]["bbox"][0] + g[0]["bbox"][2] / 2)) <= 0.6 * h
            wrapped = len(plain_text(g[-1]["text"])) >= 8
            if -0.2 * h <= gap <= 0.75 * h and same_size and (same_left or same_middle) and wrapped:
                g.append(l)
                break
        else:
            groups.append([l])
    return groups


def requests(groups: list[list[dict]], src_lang: str = "ja") -> list[dict]:
    """What to send for translation: one entry per group, its lines joined (id = the first line's)."""
    sep = "" if (src_lang or "").lower().split("-")[0] in ("ja", "zh") else " "
    return [{"id": g[0]["id"], "src": sep.join(l["text"] for l in g), "kind": "ui"} for g in groups]


THAI_MARKS = set(chr(c) for c in [0x0E31, *range(0x0E34, 0x0E3B), *range(0x0E47, 0x0E4F)])  # above / below a letter


def room_for(rect, align: str, text: str, size: float, frame_w: int, own=None, others=(), look: Line | None = None):
    """A label's translation on one line where it fits on one line at its size: a box squeezed between icons would
    otherwise break it letter by letter. Widened around its middle (or from its start), within the picture, but
    never over other text of the screen (`others`: boxes of every line read, `own` among them) - there the font
    gets smaller instead, nor off the plain panel or button it is on. A left-aligned label (a table's) only grows
    to the right, a right-aligned one to the left; a centred one (under an icon) on both sides."""
    x, y, w, h = rect
    need = int(0.55 * size * sum(c not in THAI_MARKS for c in text)) + 6
    if w >= need:
        return rect
    lo, hi = 0, frame_w
    if look is not None and look.calm:  # on a panel or a button: stay on its own background
        lo, hi = look.box[0] - look.free_left, look.box[0] + look.box[2] + look.free_right
    ox = own[0] + own[2] / 2 if own else x + w / 2
    for b in others:
        if b is own or b[1] >= y + h or b[1] + b[3] <= y:
            continue  # another row
        if b[0] + b[2] <= ox:
            lo = max(lo, b[0] + b[2] + 4)
        elif b[0] >= ox:
            hi = min(hi, b[0] - 4)
    if align == "left":
        lo = x
    elif align == "right":
        hi = x + w
    new = min(need, max(3 * w, 220), hi - lo)
    if new <= w:
        return rect
    nx = x + (w - new) // 2 if align == "center" else x - (new - w) if align == "right" else x
    nx = int(max(lo, min(nx, hi - new)))
    return (nx, y, int(new), h)


def place_block(looks: list[Line], size: float) -> tuple[tuple[int, int, int, int], str]:
    """(rect, align) for the translation of a paragraph: the block the original lines fill, a little deeper
    (Thai lines need more height), and as wide as the empty background beside the longest line allows."""
    x0 = min(a.box[0] for a in looks)
    x1 = max(a.box[0] + a.box[2] for a in looks)
    top, bottom = looks[0].glyph_y[0], looks[-1].glyph_y[1]
    widest = max(looks, key=lambda a: a.box[2])
    lefts = [a.box[0] for a in looks]
    mids = [a.box[0] + a.box[2] / 2 for a in looks]
    gh = max(1, looks[0].glyph_h)
    y = int(top - 0.3 * size)  # a line box has headroom above the letters
    tall = int(bottom - top + 0.9 * size)
    if max(lefts) - min(lefts) > 0.8 * gh and max(mids) - min(mids) <= 0.8 * gh:
        half = (x1 - x0) / 2 + max(0, min(widest.free_left, widest.free_right) - 4)
        mid = sum(mids) / len(mids)
        return (int(mid - half), y, int(2 * half), tall), "center"
    return (x0, y, (x1 - x0) + max(0, widest.free_right - 4), tall), "left"


def screen_items(frame: np.ndarray, groups: list[list[dict]], translations: dict[str, str], font_scale: float,
                 others: list | None = None) -> list[Item]:
    """Overlay items for the groups (see `paragraphs`) that got a translation. Text without one is left untouched.
    others: boxes of every line on the screen (numbers too), which a widened translation must not cover."""
    others = others if others is not None else [l["bbox"] for g in groups for l in g]
    done = []
    for g in groups:
        tl = translations.get(g[0]["id"], "").strip()
        looks = [a for l in g if (a := analyze_line(frame, l["bbox"])) is not None]
        if tl and looks:
            done.append((tl, looks, g[0]["bbox"]))
    if not done:
        return []
    heights = common_heights([sorted(a.glyph_h for a in looks)[len(looks) // 2] for _, looks, _ in done])
    items = []
    for (tl, looks, own), gh in zip(done, heights):
        size = max(11.0, gh * SIZE_FACTOR * font_scale)
        a = looks[0]
        if len(looks) == 1:
            rect, align = place(a, size)
            rect = room_for(rect, align, tl, size, frame.shape[1], own, others, a)
        else:
            rect, align = place_block(looks, size)
        items.append(Item(rect=rect, text=tl, fg=a.fg, outline=outline_for(a.fg, a.bg, all(k.calm for k in looks)),
                          max_px=size, align=align, vcenter=len(looks) == 1))
        items[-1]._boxes = [k.box for k in looks]  # where the original text is (to notice when it changes)
    patch = erase_lines(frame, [a for _, looks, _ in done for a in looks])
    if patch is not None:
        items[0].patch, items[0].patch_pos = to_qimage(patch), (0, 0)
    return items


# glyph height / height of the detector's box around a line (a name box is enlarged before it is read, which
# gives its box relatively more margin)
DET_TO_GLYPH = {"name": 0.62, "dialog": 0.9, "subtitle": 0.9}


class TextSize:
    """A game sets all its dialogue in one size. Single measurements jump about over a busy scene, so a region's
    size is the median of the last ones; until there are a few, implausible ones are held down. Whether the region
    lies over the game scene is remembered as well: a scene can be one dark colour for a moment."""

    def __init__(self):
        self.seen: deque = deque(maxlen=25)
        self.over_scene: deque = deque(maxlen=12)

    def settle(self, height: float, most: float, sure: bool = True) -> float:
        """sure: the line was long enough to be measured well (over a busy scene short ones come out too tall)."""
        if not sure:
            return sorted([*self.seen, min(height, most)])[len(self.seen) // 2]
        self.seen.append(min(height, most))
        return sorted(self.seen)[(len(self.seen) - 1) // 2]


def remember(context, src: str, tl: str):
    """Add a line to the context for the next translations. A line that went on replaces its own beginning."""
    if context and (src.startswith(context[-1]["src"][: max(3, len(context[-1]["src"]) - 3)]) or similar(context[-1]["src"], src) >= 0.6):
        context.pop()
    context.append({"src": src, "tl": tl})


def rows_of(boxes: list[list[int]]) -> list[float]:
    """Middle (y) of each row of text; pieces side by side are one row."""
    rows: list[list[float]] = []
    for _, y, _, h in sorted(boxes, key=lambda b: b[1] + b[3] / 2):
        c = y + h / 2
        if rows and abs(c - sum(rows[-1]) / len(rows[-1])) < 0.5 * h:
            rows[-1].append(c)
        else:
            rows.append([c])
    return [sum(r) / len(r) for r in rows]


def region_item(crop: np.ndarray, px: tuple[int, int, int, int], lines: list[dict], text: str, kind: str, align: str,
                font_scale: float, strokes: np.ndarray | None = None, ink: str = "any",
                sizes: TextSize | None = None) -> Item | None:
    """Overlay item for one watched region (a dialogue box, a speaker name): crop is the region's picture, px its
    place in the game's client area, lines the OCR lines inside the crop, text the translation, strokes the
    watcher's map of strokes that have been standing still (any size; it is scaled to the crop)."""
    if not text or not lines:
        return None
    if strokes is not None and strokes.shape != crop.shape[:2]:
        strokes = cv2.resize(strokes.astype(np.uint8), (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    scene = over_scene(crop)
    if sizes is not None:
        sizes.over_scene.append(scene)
        scene = any(sizes.over_scene)
    boxes = [l["dbox"] if scene and l.get("dbox") else l["bbox"] for l in lines]
    looks = [a for b in boxes if (a := analyze_line(crop, b, strokes, ink, scene)) is not None]
    if not looks:
        return None
    patch = erase_lines(crop, looks)
    fg = looks[0].fg
    found = [(l.get("dbox") or l["bbox"])[3] for l in lines if len(l["text"]) >= 2] or [(lines[0].get("dbox") or lines[0]["bbox"])[3]]
    if scene or (sizes is not None and all(l.get("dbox") for l in lines)):
        # the lowest of the boxes: ruby above a line, or a detail of the scene, makes a box taller
        height = DET_TO_GLYPH.get(kind, 0.8) * min(found)
    else:
        height = max(a.glyph_h for a in looks)
    if sizes is not None:
        height = sizes.settle(height, *((0.65 * px[3],) if kind == "name" else (0.3 * px[3], max(len(l["text"]) for l in lines) >= 8)))
    size = max(12.0, min(height * SIZE_FACTOR, 0.8 * px[3]) * font_scale)  # never taller than the region
    if scene:  # what is behind the text keeps changing, so the outline follows the text alone
        outline = "#101010" if luminance(fg) > 128 else "#f4f4f4"
    else:
        outline = outline_for(fg, looks[0].bg, all(a.plain for a in looks))
    rows = rows_of(boxes if scene else [[a.box[0], a.glyph_y[0], a.box[2], a.glyph_h] for a in looks])
    item = Item(rect=text_rect_for(px, boxes, kind, align, size), text=text, fg=fg, outline=outline, max_px=size,
                vcenter=kind != "dialog", align=align, bold=kind == "name", rows=[px[1] + r for r in rows],
                patch=to_qimage(patch) if patch is not None else None, patch_pos=(px[0], px[1]))
    item._scale = font_scale
    item._lines = looks  # kept so the patch can be made again from a later frame
    item._live = True
    item._seen = crop if not scene else None
    return item


def ocr_region(api, crop: np.ndarray, kind: str, align: str, profile, frame_h: int = 0) -> dict | None:
    """Read a region: {"text", "lines"}, or None when there is nothing in the game's language (a name already in
    Latin letters, like the player's own, stays as it is). Worker thread; the replay tool uses it too."""
    ocr = api.ocr_lines(crop, profile.src_lang, kind, profile.series, profile.target, align, frame_h)
    lines = [l for l in ocr["lines"] if l["text"].strip()]
    text = ocr["text"].strip()
    if not lines or not has_source_text(text, profile.src_lang):
        return None
    return {"text": text, "lines": lines, "ms": ocr.get("ms", 0)}


def looks_like_labels(lines: list[dict]) -> bool:
    """Separate labels side by side (a menu's buttons under the dialogue region), not a row of dialogue: pieces of
    one row far apart. A row of dialogue that the reader split (名付けて ベタベタ弾！) has a gap of a letter or so."""
    rows: list[list[dict]] = []
    for l in sorted(lines, key=lambda l: l["bbox"][1] + l["bbox"][3] / 2):
        c, h = l["bbox"][1] + l["bbox"][3] / 2, l["bbox"][3]
        if rows and abs(c - (rows[-1][0]["bbox"][1] + rows[-1][0]["bbox"][3] / 2)) < 0.5 * h:
            rows[-1].append(l)
        else:
            rows.append([l])
    for row in rows:
        row.sort(key=lambda l: l["bbox"][0])
        h = sorted(l["bbox"][3] for l in row)[len(row) // 2]
        gaps = [b["bbox"][0] - (a["bbox"][0] + a["bbox"][2]) for a, b in zip(row, row[1:])]
        if gaps and min(gaps) > 2.5 * h:
            return True
    return False


def choice_items(crop: np.ndarray, px: tuple[int, int, int, int], lines: list[dict], translations: dict[str, str],
                 font_scale: float) -> list[Item]:
    """Overlay items for a list of answers to pick from: each one is laid over its own button, like a menu."""
    groups = [[l] for l in sorted(lines, key=lambda l: l["bbox"][1])]
    items = screen_items(crop, groups, translations, font_scale)
    for it in items:
        x, y, w, h = it.rect
        it.rect = (x + px[0], y + px[1], w, h)
    if items and items[0].patch is not None:
        items[0].patch_pos = (px[0], px[1])
        # buttons are often see-through: the patch follows the scene behind them, like a region's over the scene
        items[0]._lines = [a for g in groups if (a := analyze_line(crop, g[0]["bbox"])) is not None]
        items[0]._live, items[0]._seen = True, crop
    return items


def translate_region(api, crop: np.ndarray, px: tuple[int, int, int, int], reading: dict, kind: str, align: str, ink: str,
                     profile, context: list[dict], strokes, sizes: TextSize, allow_cloud: bool) -> dict:
    """Translate a complete reading and lay the translation out. Worker thread; the replay tool uses it too."""
    labels = kind in ("dialog", "subtitle") and looks_like_labels(reading["lines"])
    if kind == "choice" or labels:  # each piece on its own: answers to pick from, or a menu's labels
        lines = [dict(l, text=l["text"].strip("・·•･.。 ")) for l in reading["lines"]]
        lines = [l for l in lines if has_source_text(l["text"], profile.src_lang)]
        res = api.translate_lines([{"id": str(i), "src": l["text"], "kind": "ui" if labels else "dialog"}
                                   for i, l in enumerate(lines)], series=profile.series, target=profile.target,
                                  src_lang=profile.src_lang, engine=profile.live_engine or profile.engine, allow_cloud=allow_cloud,
                                  context=[] if labels else context)
        for i, l in enumerate(lines):
            l["id"] = str(i)
        if labels:
            kind = "labels"
        tls = res["lines"]
        items = choice_items(crop, px, lines, tls, profile.font_scale)
        return {"src": reading["text"], "tl": " / ".join(tls.get(str(i), "") for i in range(len(lines))), "items": items,
                "engine": res.get("engine", ""), "kind": kind, "cached": bool(res.get("cached"))}
    res = api.translate_lines([{"id": "a", "src": reading["text"], "kind": kind}], series=profile.series,
                              target=profile.target, src_lang=profile.src_lang, engine=profile.live_engine or profile.engine,
                              allow_cloud=allow_cloud, context=context if kind != "name" else [])
    tl = res["lines"].get("a", "")
    item = region_item(crop, px, reading["lines"], tl, kind, align, profile.font_scale, strokes, ink, sizes)
    return {"src": reading["text"], "tl": tl, "items": [item] if item else [], "engine": res.get("engine", ""),
            "kind": kind, "cached": bool(res.get("cached"))}


def refresh_patch(item: Item, crop: np.ndarray):
    """Text drawn over the moving game scene: paint the original glyphs over again from the current picture, so
    what covers them moves with the scene instead of standing still on top of it."""
    lines = getattr(item, "_lines", None)
    if not lines or item.patch is None or crop.shape[:2] != (item.patch.height(), item.patch.width()):
        return
    seen = getattr(item, "_seen", None)
    if seen is not None and seen.shape == crop.shape:  # a dialogue box: nothing to do while it does not change
        if float(cv2.absdiff(seen[::2, ::2], crop[::2, ::2]).mean()) < 1.0:
            return
        item._seen = crop.copy()
    item._looks = getattr(item, "_looks", 0) + 1
    if item._looks % 2 == 0:  # ten times a second, for as long as it takes to see all of it
        for l in lines:
            learn_strokes(l, crop)
    patch = erase_lines(crop, lines)
    if patch is not None:
        item.patch = to_qimage(patch)


def extend_item(item: Item, crop: np.ndarray, lines: list[dict], ink: str, strokes: np.ndarray | None = None):
    """The same text was read again: cover what its rows have grown by since (dots still being typed, an end that
    was not seen before)."""
    have = getattr(item, "_lines", None)
    if not have or len(have) > 12:
        return
    if strokes is not None and strokes.shape != crop.shape[:2]:
        strokes = cv2.resize(strokes.astype(np.uint8), (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    for l in lines:
        x, y, w, h = l.get("dbox") or l["bbox"]
        row = [k for k in have if abs(y + h / 2 - (k.box[1] + k.box[3] / 2)) < 0.5 * h]
        if not row:
            continue
        left, right = min(k.box[0] for k in row), max(k.box[0] + k.box[2] for k in row)
        for a, b in ((x, left), (right, x + w)):
            if b - a > 0.9 * h and (look := analyze_line(crop, [a, y, b - a, h], strokes, ink, item._seen is None)) and look.mask.any():
                have.append(look)


def begins(old: dict, new: dict) -> bool:
    """Is what `old` read still on screen as whole rows of what `new` read (more rows were typed below, or a letter
    was misread), so that a translation laid over it covers nothing half? (readings as from ocr_region)"""
    def rows(reading):
        out: list[list[dict]] = []
        for l in sorted(reading["lines"], key=lambda l: l["bbox"][1] + l["bbox"][3] / 2):
            c, h = l["bbox"][1] + l["bbox"][3] / 2, l["bbox"][3]
            if out and abs(c - (out[-1][0]["bbox"][1] + out[-1][0]["bbox"][3] / 2)) < 0.5 * h:
                out[-1].append(l)
            else:
                out.append([l])
        return [plain_text("".join(l["text"] for l in sorted(r, key=lambda l: l["bbox"][0]))) for r in out]

    a, b = rows(old), rows(new)
    return 0 < len(a) <= len(b) and all(abs(len(x) - len(y)) <= 2 and similar(x, y) >= 0.75 for x, y in zip(a, b))


def merge_glossary(base: dict, server: dict) -> dict:
    """The glossary file (base) wins for every name it has; names only the server knows (found from the game's
    own text with the history window's button) are kept after them."""
    out = {}
    for part in ("characters", "terms"):
        mine = [e for e in base.get(part, []) if e.get("src")]
        have = {e["src"] for e in mine}
        out[part] = mine + [e for e in server.get(part, []) if e.get("src") and e["src"] not in have]
    return out
