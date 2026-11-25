"""Looking at one line of on-screen text: its colours, which pixels are glyphs, how tall the glyphs are and how
much empty background lies to its left and right. From that: a patch that covers only the original glyphs
(inpainted from their surroundings, transparent elsewhere) and the room the translation may take."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Line:
    box: tuple[int, int, int, int]  # x, y, w, h as given (clipped to the image)
    fg: str  # text colour
    bg: str  # colour of the background the text sits on
    plain: bool  # the background is one colour (a box, a button), not a picture or the game scene
    calm: bool  # ...or nearly: a soft gradient or paper texture, where an outline in its colour still blends in
    glyph_y: tuple[int, int]  # top and bottom of the glyphs (tighter than the box)
    free_left: int  # pixels of plain background left of the text before something else starts
    free_right: int
    mask: np.ndarray  # uint8 mask of glyph pixels, for the area `mask_box`
    mask_box: tuple[int, int, int, int]  # x0, y0, x1, y1

    @property
    def glyph_h(self) -> int:
        return self.glyph_y[1] - self.glyph_y[0]


# Font pixel size for a translation, from the height of the original glyphs. Japanese glyphs fill their line;
# Thai and Latin letters are about half as tall as their font size, so the same size would look much smaller.
SIZE_FACTOR = 1.3


def _hex(c) -> str:
    return "#%02x%02x%02x" % tuple(int(max(0, min(255, round(float(v))))) for v in c[::-1])  # BGR -> #rrggbb


def luminance(hex_color: str) -> float:
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
    return 0.299 * r + 0.587 * g + 0.114 * b


def contrast_outline(fg_hex: str) -> str:
    """Dark around light text, light around dark text."""
    return "#101010" if luminance(fg_hex) > 128 else "#f4f4f4"


def outline_for(fg_hex: str, bg_hex: str, plain: bool = True) -> str:
    """Outline in the background's own colour: invisible on a plain background, and it hides what is left of the
    original glyphs' edges. Over a picture or the game scene: dark around text that is lighter than what is
    behind it, light around darker text, so it stays readable whatever moves behind it."""
    lf, lb = luminance(fg_hex), luminance(bg_hex)
    if abs(lf - lb) < 40:  # text and background too close to tell which is which
        return contrast_outline(fg_hex)
    if plain:
        return bg_hex
    return "#101010" if lf > lb else "#f4f4f4"


def _free_run(frame: np.ndarray, b0: int, b1: int, start: int, step: int, limit: int = 700) -> int:
    """How far plain background continues from column `start` going left (step -1) or right (+1): until the
    colour jumps between neighbouring columns (an edge, another text) or has drifted far from where it began."""
    W = frame.shape[1]
    if step > 0:
        cols = frame[b0:b1, start : min(W, start + limit)]
    else:
        cols = frame[b0:b1, max(0, start - limit + 1) : start + 1][:, ::-1]
    if cols.shape[1] < 2:
        return 0
    med = np.median(cols, 0).astype(np.int16)  # one colour per column
    step = np.abs(np.diff(med, axis=0)).max(1)
    jump = step > max(12.0, 5 * float(np.median(step)))
    drift = np.abs(med[1:] - med[0]).max(1) > 80
    stop = np.flatnonzero(jump | drift)
    return int(stop[0]) if len(stop) else cols.shape[1] - 1


def _thin_strokes(area_bgr: np.ndarray, h: int) -> tuple[np.ndarray, np.ndarray]:
    """(mask, strength) of thin strokes in a picture: how text looks when it is drawn over the game scene or a
    textured panel, where no single background colour exists to compare against. Light strokes or dark ones,
    whichever there are more of: the strokes of a text outnumber the gaps they enclose."""
    gray = cv2.cvtColor(area_bgr, cv2.COLOR_BGR2GRAY)
    size = max(5, int(0.4 * h)) | 1
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    top = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k)
    black = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k)
    light, dark = (top > 40) & (gray > 130), black > 35
    return (light, top) if light.sum() >= dark.sum() else (dark, black)


def analyze_line(frame: np.ndarray, bbox, strokes: np.ndarray | None = None) -> Line | None:
    """frame: BGR image; bbox: [x, y, w, h] of one horizontal text line. strokes: optional mask the size of the
    frame of where text strokes have been standing still (from the watcher): for text over a moving scene it
    tells the glyphs from thin bright details of the scene."""
    H, W = frame.shape[:2]
    x, y, w, h = (int(v) for v in bbox)
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    w, h = x1 - x0, y1 - y0
    b0 = y0 + int(0.2 * h)
    b1 = max(b0 + 1, y0 + int(0.8 * h))  # the middle rows: clear of a bar's or button's own edges

    # Background colour: just left and just right of the text, and the box's own outline. The one most of the
    # box's middle rows agree with wins: the background is what a text line mostly consists of. (A box that
    # starts at the very edge of a dark bar, or pokes out of it, has the bar on one side only.)
    sw = max(3, h // 2)
    cands = []
    for sx0, sx1 in ((x0 - 2 - sw, x0 - 2), (x1 + 2, x1 + 2 + sw)):
        sx0, sx1 = max(0, sx0), min(W, sx1)
        if sx1 - sx0 >= 2:
            cands.append(np.median(frame[b0:b1, sx0:sx1].reshape(-1, 3), 0))
    box = frame[y0:y1, x0:x1].astype(np.int16)
    cands.append(np.median(np.concatenate([box[0], box[-1], box[:, 0], box[:, -1]]), 0))
    middle = frame[b0:b1, x0:x1].reshape(-1, 3).astype(np.int16)
    bg = max(cands, key=lambda c: int((np.abs(middle - c).max(1) < 30).sum()))

    g = max(2, round(0.15 * h))
    ex0, ey0, ex1, ey1 = max(0, x0 - g), max(0, y0 - g), min(W, x1 + g), min(H, y1 + g)
    area = frame[ey0:ey1, ex0:ex1].astype(np.int16)
    dist = np.abs(area - bg).max(2)
    inner = dist[y0 - ey0 : y1 - ey0, x0 - ex0 : x1 - ex0]
    top = float(np.percentile(inner, 97))
    every = dist > max(24.0, 0.4 * top)
    # How even is the background? Measured on the frame of pixels just around the text box, which holds almost no
    # glyph pixels (the 75th percentile shrugs off the few that poke into it).
    around = np.ones(dist.shape, bool)
    around[y0 - ey0 : y1 - ey0, x0 - ex0 : x1 - ex0] = False
    spread = float(np.percentile(dist[around], 75)) if around.any() else 0.0
    plain = spread < 14
    if not plain:  # over a picture, a texture or the scene: no one background colour, look for strokes instead
        every, dist = _thin_strokes(frame[ey0:ey1, ex0:ex1], h)
        if strokes is not None:
            still = cv2.dilate(strokes[ey0:ey1, ex0:ex1].astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
            if (every & still).sum() >= 0.3 * every.sum():
                every &= still
    # Glyphs are connected shapes that lie inside the line. A shape that runs out of the area around the line
    # is something else that merely reaches into it: a border, the rest of an icon, a different background
    # (the box poking out of a bar). A very long one is an underline. Those are neither measured nor erased.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(every.astype(np.uint8), connectivity=8)
    AH, AW = every.shape
    glyphish = np.zeros(n, bool)
    solid = np.zeros(n, bool)
    for i in range(1, n):
        cx, cy = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
        cw, ch, size = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT], stats[i, cv2.CC_STAT_AREA]
        leaves = ((cx == 0 and ex0 > 0) or (cy == 0 and ey0 > 0) or (cx + cw == AW and ex1 < W) or (cy + ch == AH and ey1 < H))
        glyphish[i] = not leaves and cw <= 6 * h
        solid[i] = glyphish[i] and size >= 4
    ink = glyphish[labels]
    if not ink.any():  # nothing of the kind (letters joined into one shape across the line): take everything
        ink, solid = every, np.ones(n, bool)
    real = solid[labels] & ink
    rows, cols = np.flatnonzero(real.any(1)), np.flatnonzero(real.any(0))
    gy = (ey0 + int(rows[0]), ey0 + int(rows[-1]) + 1) if len(rows) >= 2 else (y0, y1)
    if gy[1] - gy[0] < 0.35 * h:
        gy = (y0, y1)
    if len(cols) >= 2:  # where the text really starts and ends (the given box may take in part of an icon)
        x0, x1 = ex0 + int(cols[0]), ex0 + int(cols[-1]) + 1
        w = x1 - x0
    if not ink.any():
        fg = 255 - bg
    else:  # the colour at the heart of the strokes: their soft edges are a blend with the background
        fg = np.median(area[ink & (dist >= np.percentile(dist[ink], 80))], 0)
    b0 = gy[0] + int(0.2 * (gy[1] - gy[0]))
    b1 = max(b0 + 1, gy[0] + int(0.8 * (gy[1] - gy[0])))
    grow = max(1, round((0.07 if plain else 0.12) * h))  # over a scene: take the glyphs' shadow too
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1))
    mask = cv2.dilate(ink.astype(np.uint8) * 255, k)
    return Line(box=(x0, y0, w, h), fg=_hex(fg), bg=_hex(bg), plain=plain, calm=spread < 35, glyph_y=gy,
                free_left=_free_run(frame, b0, b1, x0 - 3, -1) if x0 >= 4 else 0,
                free_right=_free_run(frame, b0, b1, x1 + 2, 1) if x1 + 3 < W else 0,
                mask=mask, mask_box=(ex0, ey0, ex1, ey1))


def erase_lines(frame: np.ndarray, lines: list[Line]) -> np.ndarray | None:
    """BGRA image the size of the frame: each line's glyphs painted over with their surroundings (inpainted at
    full resolution, line by line), transparent everywhere else."""
    lines = [l for l in lines if l is not None and l.mask.any()]
    if not lines:
        return None
    H, W = frame.shape[:2]
    out = np.zeros((H, W, 4), np.uint8)
    for l in lines:
        ex0, ey0, ex1, ey1 = l.mask_box
        m = 6  # context around the glyphs for the inpainting
        cx0, cy0, cx1, cy1 = max(0, ex0 - m), max(0, ey0 - m), min(W, ex1 + m), min(H, ey1 + m)
        crop = frame[cy0:cy1, cx0:cx1]
        mask = np.zeros(crop.shape[:2], np.uint8)
        mask[ey0 - cy0 : ey1 - cy0, ex0 - cx0 : ex1 - cx0] = l.mask
        filled = cv2.inpaint(crop, mask, 3, cv2.INPAINT_TELEA)
        alpha = cv2.GaussianBlur(mask, (0, 0), 0.8)
        alpha[mask > 0] = 255
        dst = out[cy0:cy1, cx0:cx1]
        keep = alpha > dst[..., 3]
        dst[keep] = np.dstack([filled, alpha])[keep]
    return out
