"""Whole-screen translation (menus, settings, quest lists): which OCR lines to translate and where to draw them.
Kept free of app state so tools/render_screen.py can show exactly what the overlay would draw on a screenshot."""
from __future__ import annotations

import re

import cv2
import numpy as np

from .overlay import Item, text_rect_for, to_qimage
from .render import SIZE_FACTOR, Line, analyze_line, erase_lines, outline_for

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
    the same size, and either starting at the same left edge or centred on the same middle."""
    groups: list[list[dict]] = []
    for l in sorted(lines, key=lambda l: (l["bbox"][1], l["bbox"][0])):
        x, y, w, h = l["bbox"]
        for g in groups:
            px, py, pw, ph = g[-1]["bbox"]
            gap = y - (py + ph)
            same_size = abs(h - ph) <= 0.3 * max(h, ph)
            same_left = abs(x - g[0]["bbox"][0]) <= 0.6 * h
            same_middle = abs((x + w / 2) - (g[0]["bbox"][0] + g[0]["bbox"][2] / 2)) <= 0.6 * h
            if -0.2 * h <= gap <= 0.75 * h and same_size and (same_left or same_middle):
                g.append(l)
                break
        else:
            groups.append([l])
    return groups


def requests(groups: list[list[dict]], src_lang: str = "ja") -> list[dict]:
    """What to send for translation: one entry per group, its lines joined (id = the first line's)."""
    sep = "" if (src_lang or "").lower().split("-")[0] in ("ja", "zh") else " "
    return [{"id": g[0]["id"], "src": sep.join(l["text"] for l in g), "kind": "ui"} for g in groups]


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


def screen_items(frame: np.ndarray, groups: list[list[dict]], translations: dict[str, str], font_scale: float) -> list[Item]:
    """Overlay items for the groups (see `paragraphs`) that got a translation. Text without one is left untouched."""
    done = []
    for g in groups:
        tl = translations.get(g[0]["id"], "").strip()
        looks = [a for l in g if (a := analyze_line(frame, l["bbox"])) is not None]
        if tl and looks:
            done.append((tl, looks))
    if not done:
        return []
    heights = common_heights([sorted(a.glyph_h for a in looks)[len(looks) // 2] for _, looks in done])
    items = []
    for (tl, looks), gh in zip(done, heights):
        size = max(11.0, gh * SIZE_FACTOR * font_scale)
        a = looks[0]
        if len(looks) == 1:
            rect, align = place(a, size)
        else:
            rect, align = place_block(looks, size)
        items.append(Item(rect=rect, text=tl, fg=a.fg, outline=outline_for(a.fg, a.bg, all(k.calm for k in looks)),
                          max_px=size, align=align, vcenter=len(looks) == 1))
    patch = erase_lines(frame, [a for _, looks in done for a in looks])
    if patch is not None:
        items[0].patch, items[0].patch_pos = to_qimage(patch), (0, 0)
    return items


def region_item(crop: np.ndarray, px: tuple[int, int, int, int], boxes: list[list[int]], text: str, kind: str,
                align: str, font_scale: float, strokes: np.ndarray | None = None) -> Item | None:
    """Overlay item for one watched region (a dialogue box, a speaker name): crop is the region's picture, px its
    place in the game's client area, boxes the OCR line boxes inside the crop, text the translation, strokes the
    watcher's map of strokes that have been standing still (any size; it is scaled to the crop)."""
    if not text:
        return None
    if strokes is not None and strokes.shape != crop.shape[:2]:
        strokes = cv2.resize(strokes.astype(np.uint8), (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    looks = [a for b in boxes if (a := analyze_line(crop, b, strokes)) is not None]
    patch = erase_lines(crop, looks)
    fg = looks[0].fg if looks else "#ffffff"
    heights = sorted(a.glyph_h for a in looks) or sorted(b[3] for b in boxes)
    size = min(heights[len(heights) // 2] * SIZE_FACTOR, 0.8 * px[3])  # never taller than the region
    plain = all(a.plain for a in looks)
    calm = all(a.calm for a in looks)
    item = Item(rect=text_rect_for(px, boxes, kind, align), text=text, fg=fg,
                outline=outline_for(fg, looks[0].bg, calm) if looks else "",
                max_px=max(12.0, size * font_scale), vcenter=kind != "dialog", align=align, bold=kind == "name",
                patch=to_qimage(patch) if patch is not None else None, patch_pos=(px[0], px[1]))
    item._scale = font_scale
    item._lines = looks  # kept so the patch can be made again from a later frame
    item._live = not plain
    return item


def refresh_patch(item: Item, crop: np.ndarray):
    """Text drawn over the moving game scene: paint the original glyphs over again from the current picture, so
    what covers them moves with the scene instead of standing still on top of it."""
    lines = getattr(item, "_lines", None)
    if not lines or item.patch is None or crop.shape[:2] != (item.patch.height(), item.patch.width()):
        return
    patch = erase_lines(crop, lines)
    if patch is not None:
        item.patch = to_qimage(patch)
