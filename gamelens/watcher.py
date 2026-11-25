"""Deciding when a watched region has new, settled text worth reading.

Game dialogue is often drawn straight over the moving 3D scene, so "did the pixels change" is useless: they always
do. What tells text from scene is that glyph strokes are thin, bright (or dark) and stay exactly where they are.
`stroke_map` keeps thin strokes only; `RegionWatcher` keeps the strokes that survived the last few frames and
follows how many there are: rising while a line is being typed out, level once it is complete, gone when the
line is replaced.
"""
from __future__ import annotations

from collections import deque

import cv2
import numpy as np

THUMB_W = 320
STROKE_W = 480


def ink_map(region_bgr: np.ndarray, ink: str = "light") -> np.ndarray:
    """Coarse picture of a whole frame, for "has the screen changed" (menus): light / dark pixels, or grey levels."""
    h, w = region_bgr.shape[:2]
    gray = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2GRAY) if region_bgr.ndim == 3 else region_bgr
    size = (THUMB_W, max(1, int(h * THUMB_W / w))) if w > THUMB_W else None
    if ink in ("light", "dark"):
        # threshold at full size, then shrink: averaging first would wash thin glyph strokes out
        m = (gray >= 190) if ink == "light" else (gray <= 70)
        if size is None:
            return m
        return cv2.resize(m.astype(np.uint8) * 255, size, interpolation=cv2.INTER_AREA) >= 40
    if size is not None:
        gray = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
    return (gray // 32).astype(np.uint8)  # 'any': coarse grey levels


def changed_pixels(a: np.ndarray, b: np.ndarray) -> int:
    if a.shape != b.shape:
        return a.size
    return int(np.count_nonzero(a != b))


def stroke_map(region_bgr: np.ndarray, ink: str = "light") -> np.ndarray:
    """Thin strokes in a region: light ones (white text), dark ones, or both ('any'). Large bright or dark areas
    (sky, mist, a white dress) are not strokes and drop out."""
    gray = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2GRAY) if region_bgr.ndim == 3 else region_bgr
    h, w = gray.shape[:2]
    if w > STROKE_W:
        gray = cv2.resize(gray, (STROKE_W, max(1, int(h * STROKE_W / w))), interpolation=cv2.INTER_AREA)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    light = dark = None
    if ink in ("light", "any"):
        light = (cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k) > 50) & (gray > 150)
    if ink in ("dark", "any"):
        dark = (cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k) > 50) & (gray < 110)
    return light if dark is None else (dark if light is None else light | dark)


def has_text(strokes: np.ndarray | None, min_frac: float = 0.003) -> bool:
    return strokes is not None and float(strokes.mean()) >= min_frac


class RegionWatcher:
    """feed(stroke_map, now) returns
      'changed' - the text that was read is gone, replaced or being added to: hide its translation;
      'stable'  - text is on screen and has stopped growing for `stable_ms`: read it now;
      None      - nothing to do.
    A line that never stops moving is read after `max_wait_ms` anyway."""

    def __init__(self, stable_ms: int = 400, max_wait_ms: int = 4000, frames: int = 4):
        self.stable_s = stable_ms / 1000
        self.max_wait_s = max_wait_ms / 1000
        self.hist: deque = deque(maxlen=frames)
        self.persistent: np.ndarray | None = None  # strokes present in every one of the last frames
        self.ref: np.ndarray | None = None  # the strokes of the text that was read
        self.ref_n, self.ref_at = 0, -9.0
        self.level = 0  # stroke count when it last moved
        self.last_move = 0.0
        self.dirty_since: float | None = None
        self.started = False

    def reset(self):
        self.hist.clear()
        self.ref, self.dirty_since, self.started = None, None, False

    def force(self, now: float):
        """Read again as soon as the text is still (e.g. the server was not ready)."""
        self.ref, self.last_move, self.dirty_since = None, now, now

    def feed(self, strokes: np.ndarray, now: float) -> str | None:
        if self.hist and self.hist[-1].shape != strokes.shape:
            self.reset()
        self.hist.append(strokes)
        if len(self.hist) < self.hist.maxlen:
            if not self.started:
                self.started, self.last_move, self.dirty_since = True, now, now
                return "changed"
            return None
        p = np.logical_and.reduce(list(self.hist))
        n = int(p.sum())
        self.persistent = p
        if self.ref is not None:
            kept = int((p & self.ref).sum()) >= 0.6 * self.ref_n
            grown = n > 1.25 * self.ref_n + 20
            if kept and not grown:
                return None  # the text that was read is still there
            if kept and grown and now - self.ref_at < 1.2 and self.stable_s < 1.0:
                # the same line went on typing right after it was read: this game pauses mid-line, wait longer
                self.stable_s = min(1.0, self.stable_s + 0.1)
            self.ref, self.level, self.last_move, self.dirty_since = None, n, now, now
            return "changed"
        if n < max(12, int(0.003 * p.size)):  # no text here
            self.level, self.last_move, self.dirty_since = n, now, None
            return None
        if abs(n - self.level) > max(6, 0.05 * max(n, self.level)):  # still being typed out (or fading)
            self.level, self.last_move = n, now
        if self.dirty_since is None:
            self.dirty_since = now
        if now - self.last_move >= self.stable_s or now - self.dirty_since >= self.max_wait_s:
            self.ref, self.ref_n, self.ref_at, self.dirty_since = p, n, now, None
            return "stable"
        return None


def similar(a: str, b: str) -> float:
    """0..1 similarity of two OCR results (OCR jitters by a character or two between reads)."""
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    import difflib

    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()
