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
    run: np.ndarray | None = None  # for how many looks in a row each pixel has been part of a stroke (learn_strokes)

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


def _thin_strokes(area_bgr: np.ndarray, h: int, ink: str = "any") -> tuple[np.ndarray, np.ndarray]:
    """(mask, strength) of thin strokes in a picture: how text looks when it is drawn over the game scene or a
    textured panel, where no single background colour exists to compare against. Light strokes or dark ones,
    whichever there are more of: the strokes of a text outnumber the gaps they enclose."""
    gray = cv2.cvtColor(area_bgr, cv2.COLOR_BGR2GRAY)
    size = max(5, int(0.4 * h)) | 1
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    top = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k)
    black = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k)
    light, dark = (top > 40) & (gray > 130), black > 35
    if ink == "light" or (ink == "any" and light.sum() >= dark.sum()):
        # white text on a nearly white scene barely stands out: lower the bar until there is as much ink as text has
        for bar in (28, 18):
            if light.mean() >= 0.08:
                break
            light = (top > bar) & (gray > 150)
        return light, top
    return dark, black


def over_scene(crop: np.ndarray) -> bool:
    """Does a watched region lie over the game scene (or a picture) rather than in a dialogue box? Judged on the
    whole region, so that all its lines are treated alike: a box is one colour or a soft shade, a scene is not."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    size = max(9, gray.shape[0] // 4) | 1
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    # opening removes thin light strokes, closing thin dark ones: what is left is the background
    bg = cv2.morphologyEx(cv2.morphologyEx(gray, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)
    return float(np.percentile(bg, 90) - np.percentile(bg, 10)) > 35


def analyze_line(frame: np.ndarray, bbox, strokes: np.ndarray | None = None, ink_kind: str = "any",
                 scene: bool | None = None, _dense: bool = False) -> Line | None:
    """frame: BGR image; bbox: [x, y, w, h] of one horizontal text line. strokes: optional mask the size of the
    frame of where text strokes have been standing still (from the watcher): for text over a moving scene it
    tells the glyphs from thin bright details of the scene. ink_kind: light / dark when it is known which the
    text is. scene: whether the text is known to lie over the scene (a watched region, see `over_scene`); None
    judges it from the line itself (menus)."""
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
    # The box's own first and last columns: in front of and behind the letters. Where they are what lies beside the
    # box too, that is the background, also when bold letters fill most of the line (a small tag's label). Where
    # they differ, the box hugs a small tag or button whose fill they are, and the fill joins the candidates.
    end_px = np.concatenate([box[b0 - y0 : b1 - y0, :2], box[b0 - y0 : b1 - y0, -2:]], 1).reshape(-1, 3)
    ends = np.median(end_px, 0)
    solid = (np.abs(end_px - ends).max(1) < 20).mean() >= 0.85  # one colour there, not cut-through letters
    sides = cands[:-1]
    # ...and further out too: light letters with a glow have the glow's colour at their ends and just beside them,
    # but the picture they stand on beyond it
    far = []
    for fx0, fx1 in ((x0 - 2 - 3 * sw, x0 - 2 - 2 * sw), (x1 + 2 + 2 * sw, x1 + 2 + 3 * sw)):
        fx0, fx1 = max(0, fx0), min(W, fx1)
        if fx1 - fx0 >= 2:
            far.append(np.median(frame[b0:b1, fx0:fx1].reshape(-1, 3), 0))
    if solid and sides and w >= 3 * h and all(np.abs(c - ends).max() < 30 for c in sides + far):
        bg = ends
    else:
        if solid:
            cands.append(ends)
        bg = max(cands, key=lambda c: int((np.abs(middle - c).max(1) < 30).sum()))

    g = max(2, round(0.15 * h))
    ex0, ey0, ex1, ey1 = max(0, x0 - g), max(0, y0 - g), min(W, x1 + g), min(H, y1 + g)
    area = frame[ey0:ey1, ex0:ex1].astype(np.int16)
    dist = np.abs(area - bg).max(2)
    inner = dist[y0 - ey0 : y1 - ey0, x0 - ex0 : x1 - ex0]
    top = float(np.percentile(inner, 97))
    every = dist > max(24.0, 0.4 * top)
    # How even is the background? What is neither text nor anything else that stands out from it.
    rest = area[~cv2.dilate(every.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)]
    spread = 0.0 if len(rest) < 8 else float(np.abs(rest - bg).max(1).mean())
    if scene is not None:
        spread = 99.0 if scene else 0.0
    plain = spread < 30
    if plain and scene is None:
        # a background that changes along the line (a highlight's glow, a gradient): measured against the
        # background right around each letter, the strokes the single colour missed are found too
        k = max(3, int(1.2 * h)) | 1
        local = cv2.medianBlur(frame[ey0:ey1, ex0:ex1], min(k, 255)).astype(np.int16)
        near = np.abs(area - local).max(2)
        top_near = float(np.percentile(near[y0 - ey0 : y1 - ey0, x0 - ex0 : x1 - ex0], 97))
        found = near > max(24.0, 0.4 * top_near)
        # the one colour took in the glow as well: much "ink" also in the middle of the line (not just the bands above
        # and below a dark bar the box pokes out of)
        mid = every[y0 - ey0 + h // 4 : y1 - ey0 - h // 4]
        glow = mid.size and mid.mean() > 0.35 and found.mean() < 0.6 * every.mean()
        if found.mean() < 0.4 and glow:
            every = found  # (the text colour is still measured against the one background colour)
        else:
            # the background changes colour part of the way along (a dark tree behind the middle of a line over a
            # blurred scene): there the one colour marks every pixel, the letters drown in one big shape and are
            # neither measured nor erased. Those columns take the letters found against what is around them
            rows = every[y0 - ey0 + h // 4 : y1 - ey0 - h // 4]
            if rows.size:
                solid = cv2.blur(rows.mean(0, keepdims=True).astype(np.float32), (max(3, h // 2), 1))[0] > 0.7
                # (a stretch of another background is broad and leaves most of the line as it was; bold letters on
                # a tag fill their columns too, but all along the line)
                runs = np.diff(np.concatenate([[0], solid.astype(np.int8), [0]]))
                longest = (np.flatnonzero(runs == -1) - np.flatnonzero(runs == 1)).max(initial=0)
                # (and it is a background: it fills the margin above and below the line there too. Dense bold
                # letters - 絆心, 報酬 - fill their columns' middle as well, but not the margin)
                margin = np.concatenate([every[: y0 - ey0], every[y1 - ey0 :]])
                beyond = margin.size and margin[:, solid].mean() > 0.6 if solid.any() else False
                if longest >= 2 * h and solid.mean() <= 0.6 and beyond:
                    every[:, solid] = found[:, solid]
            # one end of the line lies on a lighter or darker patch (a panel's gradient, a highlight): there the
            # one colour marks the patch from top to bottom, the letters over it drown in one shape that runs out of
            # the area and is taken for something else (絆心, 報酬 were neither erased nor measured). Over such a
            # patch the letters found against what is right around them count instead
            n_, lab_, st_, _ = cv2.connectedComponentsWithStats(every.astype(np.uint8), connectivity=8)
            AH_, AW_ = every.shape
            for i in range(1, n_):
                cx, cy, cw, ch = st_[i, 0], st_[i, 1], st_[i, 2], st_[i, 3]
                at_end = (cx == 0 and ex0 > 0) or (cx + cw == AW_ and ex1 < W)
                if at_end and ch >= 0.9 * AH_ and cw >= 0.5 * h and cw <= 0.6 * AW_:
                    cols = slice(cx, cx + cw)
                    every[:, cols] = found[:, cols]
    on_scene = False
    if scene is None:
        # light game text with its soft shadow straight over the 3D scene (a hint, a quest): the scene is light
        # here and dark there, so no one colour and no one threshold finds the letters. Pixels brighter than what
        # is around them, with their shadow close by, do
        lt, rise = _light_on_scene(frame[ey0:ey1, ex0:ex1], h)
        if 0.03 <= lt.mean() <= 0.3:
            lum = cv2.cvtColor(frame[ey0:ey1, ex0:ex1], cv2.COLOR_BGR2GRAY)
            inside = (slice(y0 - ey0 + h // 4, y1 - ey0 - h // 4), slice(x0 - ex0, x1 - ex0))
            cols = lt[y0 - ey0 : y1 - ey0, x0 - ex0 : x1 - ex0].any(0).mean()
            # (and they are light letters: much brighter than the line's area as a whole. Next to dark letters on a
            # light chip it finds the chip itself, as bright as the rest of it)
            lighter = float(np.median(lum[lt])) - float(np.median(lum))
            # (letters are thin: the light found filling a third of the line is the tag between bold dark letters)
            # And what is behind them is a picture: detail everywhere, not the flat fill of a button or a tag
            lap = np.abs(cv2.Laplacian(lum, cv2.CV_16S, ksize=3))
            behind = ~cv2.dilate(lt.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
            texture = float((lap[behind] > 10).mean()) if behind.any() else 0.0
            # (_dense: the other ways found next to no letters. Then dense kanji, which fill the line's middle too,
            # over a scene smooth in part - photo mode's 表情 / 詳細 over sand and sky - count as well, when the light
            # found is in pieces no wider than a letter each, which a chip or a bar is not)
            n_, _, st_, _ = cv2.connectedComponentsWithStats(lt.astype(np.uint8), connectivity=8)
            apart = _dense and n_ > 2 and int(st_[1:, cv2.CC_STAT_WIDTH].max()) <= 1.2 * h
            dense_ok = lt[inside].mean() <= 0.4 or (apart and lt[inside].mean() <= 0.7)
            textured = texture >= 0.58 or (apart and texture >= 0.45)
            if textured and cols >= 0.5 and float(np.median(lum[lt])) >= 150 and lighter >= 35 and dense_ok:
                every, dist, on_scene, plain, spread = lt, rise, True, False, 99.0
                bg = np.median(frame[ey0:ey1, ex0:ex1][~lt], 0)
    if not plain and not on_scene:  # over a picture, a texture or the scene: no one background colour, strokes
        every, dist = _thin_strokes(frame[ey0:ey1, ex0:ex1], h, ink_kind)
        if strokes is not None:
            still = cv2.dilate(strokes[ey0:ey1, ex0:ex1].astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
            if (every & still).sum() >= 0.3 * every.sum():
                # whole strokes: one that stood still in good part counts all along, also where it got lost for a moment
                n, labels = cv2.connectedComponents(every.astype(np.uint8), connectivity=8)
                share = np.bincount(labels[still], minlength=n) / np.maximum(1, np.bincount(labels.ravel(), minlength=n))
                whole = share >= 0.4
                whole[0] = False
                every = (every & still) | whole[labels]
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
    if not ink.any() and every.mean() < 0.35:  # letters joined into one shape across the line: take everything
        ink, solid = every, np.ones(n, bool)  # (much more than a third of the area is not text at all)
    real = solid[labels] & ink
    if scene is None and not on_scene and not _dense and real.sum() < 0.03 * w * h:
        # next to nothing found of the letters (light kanji over sand: 表情 was neither erased nor measured, its
        # translation took the water's blue): look once more for light letters on the scene, less strictly
        again = analyze_line(frame, bbox, strokes, ink_kind, scene, _dense=True)
        if again is not None and (again.mask > 0).sum() > 0:
            return again
    rows, cols = np.flatnonzero(real.any(1)), np.flatnonzero(real.any(0))
    gy = (ey0 + int(rows[0]), ey0 + int(rows[-1]) + 1) if len(rows) >= 2 else (y0, y1)
    if on_scene and real.any():
        # over the scene a blade of grass or a glint reaches above or below the letters: the letters are the rows
        # most of the line's strokes cross
        density = real.sum(1).astype(np.float32)
        dense = np.flatnonzero(density >= 0.2 * float(density.max()))
        if len(dense) >= 2:
            gy = (ey0 + int(dense[0]), ey0 + int(dense[-1]) + 1)
    # (letters are within the reader's box, give or take a pixel or two: more is a glow or the scene behind)
    slack = max(1, round(0.06 * h))
    gy = (max(gy[0], y0 - slack), min(gy[1], y1 + slack))
    if gy[1] - gy[0] < 0.35 * h:
        gy = (y0, y1)
    if len(cols) >= 2 and cols[-1] - cols[0] + 1 >= 0.25 * w:  # where the text really starts and ends (the given box
        x0, x1 = ex0 + int(cols[0]), ex0 + int(cols[-1]) + 1  # may take in part of an icon); a much narrower find is
        w = x1 - x0  # a piece of the text only (faint glyphs on a light panel), not where it ends
    if not ink.any():
        fg = 255 - bg
    else:  # the colour at the heart of the strokes: their soft edges are a blend with the background
        fg = np.median(area[ink & (dist >= np.percentile(dist[ink], 80))], 0)
    if not plain and ink_kind in ("light", "dark"):  # a scene can fool the measurement: keep text light / dark
        lum = float(0.114 * fg[0] + 0.587 * fg[1] + 0.299 * fg[2])
        if ink_kind == "light" and lum < 170:
            fg = np.array([250.0, 250.0, 250.0])
        elif ink_kind == "dark" and lum > 110:
            fg = np.array([40.0, 40.0, 40.0])
    b0 = gy[0] + int(0.2 * (gy[1] - gy[0]))
    b1 = max(b0 + 1, gy[0] + int(0.8 * (gy[1] - gy[0])))
    # in a watched region: take the glyphs' shadow too
    grow = max(1, round((0.07 if scene is None and plain else 0.12) * h))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1))
    cover = ink
    if not plain and strokes is not None and every.mean() < 0.35:
        cover = every  # strokes that stood still are text, also where they run into a detail of the scene
    if not plain and scene is None and len(cols) >= 2:
        # over the scene the last letters can touch a blade of grass that runs out of the area, and are taken for the
        # scene (テンプレー|ト was left standing): the reader's box says the text goes on, so in the letters' rows up
        # to its end the strokes are erased too
        bx1 = min(W, int(bbox[0]) + int(bbox[2]))
        last = ex0 + int(cols[-1]) + 1
        if bx1 - last > 0.5 * h:
            tail = np.zeros(cover.shape, bool)
            tail[max(0, gy[0] - ey0) : gy[1] - ey0, last - ex0 : bx1 - ex0] = True
            cover = cover | (every & tail)
    mask = cv2.dilate(cover.astype(np.uint8) * 255, k)
    return Line(box=(x0, y0, w, h), fg=_hex(fg), bg=_hex(bg), plain=plain, calm=spread < 35, glyph_y=gy,
                free_left=_free_run(frame, b0, b1, x0 - 3, -1) if x0 >= 4 else 0,
                free_right=_free_run(frame, b0, b1, x1 + 2, 1) if x1 + 3 < W else 0,
                mask=mask, mask_box=(ex0, ey0, ex1, ey1))


def _light_on_scene(area: np.ndarray, h: int) -> tuple[np.ndarray, np.ndarray]:
    """Light letters over a busy picture: pixels clearly brighter than the picture around them (a median over about
    one and a half letters), in shapes that have a much darker pixel close by (the letters' shadow). Returns the
    mask and how much brighter each pixel is."""
    lum = cv2.cvtColor(area, cv2.COLOR_BGR2GRAY)
    k = max(3, int(1.6 * h)) | 1
    rise = lum.astype(np.int16) - cv2.medianBlur(lum, min(k, 255)).astype(np.int16)
    bright = (rise > max(25.0, 0.35 * float(np.percentile(rise, 99)))) & (lum > 140)
    r = max(2, round(0.12 * h))
    darkest = cv2.erode(lum, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    edge = (lum.astype(np.int16) - darkest.astype(np.int16)) > 70
    n, labels = cv2.connectedComponents(bright.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    keep[np.unique(labels[bright & edge])] = True
    keep[0] = False
    return keep[labels], rise


def learn_strokes(line: Line, frame: np.ndarray, need: int = 4) -> bool:
    """Text over the scene stands still while the scene moves: whatever keeps being a stroke there, look after look,
    is text too, and is covered from now on (the first look may have missed it). True when something was added."""
    x0, y0, x1, y1 = line.mask_box
    h = line.box[3]
    m = max(2, int(0.3 * h))
    ax0, ay0 = max(0, x0 - m), max(0, y0 - m)
    strokes, _ = _thin_strokes(frame[ay0 : y1 + m, ax0 : x1 + m], h, "light" if luminance(line.fg) > 128 else "dark")
    strokes = strokes[y0 - ay0 : y1 - ay0, x0 - ax0 : x1 - ax0]
    if line.run is None:
        line.run = np.zeros(line.mask.shape, np.uint8)
    line.run = np.where(strokes, np.minimum(line.run, 250) + 1, 0).astype(np.uint8)
    new = (line.run >= need) & (line.mask == 0)
    if new.sum() < 6 or (new | (line.mask > 0)).mean() > 0.85:
        return False
    g = max(1, round(0.08 * h))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * g + 1, 2 * g + 1))
    line.mask = line.mask | cv2.dilate(new.astype(np.uint8) * 255, k)
    return True


def pull_push(img: np.ndarray, hole: np.ndarray) -> np.ndarray:
    """The hole filled by weighted multiresolution interpolation: from coarser and coarser averages of the known
    pixels, each level trusted where enough of it is known (float32 result)."""
    known = (hole == 0).astype(np.float32)
    c = img.astype(np.float32) * known[..., None]
    w = known
    levels = []
    while True:
        levels.append((c, w))
        hh, ww = w.shape
        if (w > 0).all() or min(hh, ww) <= 1:
            break
        size = (max(1, (ww + 1) // 2), max(1, (hh + 1) // 2))
        c = cv2.resize(c, size, interpolation=cv2.INTER_AREA).reshape(size[1], size[0], -1)  # (keeps one channel as one)
        w = cv2.resize(w, size, interpolation=cv2.INTER_AREA)
    c, w = levels[-1]
    col = c / np.maximum(w, 1e-6)[..., None]
    for c, w in reversed(levels[:-1]):
        up = cv2.resize(col, (w.shape[1], w.shape[0]), interpolation=cv2.INTER_LINEAR).reshape(w.shape[0], w.shape[1], -1)
        a = np.clip(w * 4, 0, 1)[..., None]  # (a quarter known is enough to trust the average at this level)
        col = np.where(w[..., None] > 0, a * (c / np.maximum(w, 1e-6)[..., None]) + (1 - a) * up, up)
    out = img.astype(np.float32)
    out[hole > 0] = col[hole > 0]
    return out


def harmonic_fill(img: np.ndarray, hole: np.ndarray, steps: int = 10) -> np.ndarray:
    """pull_push, then steps of Laplace's equation inside the hole: a slope (a panel's gradient, a glow) is carried
    across the hole as a slope, where inpainting from the edge leaves a flat bar or streaks. (Ten steps do nearly all
    sixty would - smooth panels 1.25 / 6.8 against 1.19 / 6.7 - at the cost of Telea's inpainting, ~1 ms a line.)"""
    x = pull_push(img, hole)
    m = hole > 0
    for _ in range(steps):
        x[m] = cv2.blur(x, (3, 3))[m]
    return x


def inpaint(crop: np.ndarray, mask: np.ndarray, h: int, full: bool = False) -> np.ndarray:
    """crop with the pixels of mask filled in from around them.

    Measured on 380 real glyph masks laid over clean background from our screenshots (colour error / gradient
    error, lower is better): Telea's method 6.2 / 38.4, Navier-Stokes 5.5 / 36.4, the harmonic fill 6.1 / 35.0 -
    on smooth panels 1.25 / 6.8 against Telea's 1.8 / 9.1. Navier-Stokes wins the numbers but, seen on the screen,
    draws streaks across a wide hole ("4306" on a dark pill), where the harmonic fill stays smooth. So a one-off
    erase (full) is harmonic; the repaint every tick for text over the scene stays with Telea, taller lines (1080p
    and up) filled at a fraction of their size."""
    if full:
        return np.clip(harmonic_fill(crop, mask), 0, 255).astype(np.uint8)
    k = 1 if h <= 22 else 2 if h <= 44 else 3
    if k == 1:
        return cv2.inpaint(crop, mask, 3, cv2.INPAINT_TELEA)
    H, W = mask.shape
    size = (max(1, W // k), max(1, H // k))
    # averaged over the pixels that are not text, so nothing of it bleeds into the small picture
    keep = (mask == 0).astype(np.float32)
    weight = cv2.resize(keep, size, interpolation=cv2.INTER_AREA)
    small = cv2.resize(crop.astype(np.float32) * keep[..., None], size, interpolation=cv2.INTER_AREA) / np.maximum(weight, 0.05)[..., None]
    hole = (weight < 0.4).astype(np.uint8) * 255
    small = cv2.inpaint(np.clip(small, 0, 255).astype(np.uint8), hole, 3, cv2.INPAINT_TELEA)
    return cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)


def erase_lines(frame: np.ndarray, lines: list[Line], full: bool = True) -> np.ndarray | None:
    """BGRA image the size of the frame: each line's glyphs painted over with their surroundings (inpainted line by
    line), transparent everywhere else. full=False for the repaint every tick (see inpaint)."""
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
        filled = inpaint(crop, mask, l.box[3], full)
        alpha = cv2.GaussianBlur(mask, (0, 0), 0.8)
        alpha[mask > 0] = 255
        dst = out[cy0:cy1, cx0:cx1]
        keep = alpha > dst[..., 3]
        dst[keep] = np.dstack([filled, alpha])[keep]
    return out
