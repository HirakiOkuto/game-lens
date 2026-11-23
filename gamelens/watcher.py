"""Per-region change detection: decides when a dialogue box has new, settled text worth reading.

Frames are reduced to a small binary "ink" map (light text on a dark box by default), so a moving 3D scene behind
a semi-transparent box changes little while a new glyph from a typewriter effect does.
"""
from __future__ import annotations

import cv2
import numpy as np

THUMB_W = 320


def ink_map(region_bgr: np.ndarray, ink: str = "light") -> np.ndarray:
    """Binary map of text-like pixels at a reduced size."""
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


class RegionWatcher:
    """feed() returns 'changed' when text starts changing (hide the old translation), 'stable' when it has been
    still for `stable_ms` (read it now), else None. Continuous change (an animated background) still yields
    'stable' after `max_wait_ms`, so a line is never stuck untranslated."""

    def __init__(self, stable_ms: int = 250, max_wait_ms: int = 1500, min_changed: int = 3, frac: float = 0.0015):
        self.stable_s = stable_ms / 1000
        self.max_wait_s = max_wait_ms / 1000
        self.min_changed = min_changed
        self.frac = frac
        self.prev: np.ndarray | None = None
        self.dirty_since: float | None = None
        self.last_change = 0.0

    def reset(self):
        self.prev, self.dirty_since = None, None

    def feed(self, ink: np.ndarray, now: float) -> str | None:
        if self.prev is None:
            self.prev, self.dirty_since, self.last_change = ink, now, now
            return "changed"
        n = changed_pixels(self.prev, ink)
        self.prev = ink
        if n >= max(self.min_changed, self.frac * ink.size):
            self.last_change = now
            if self.dirty_since is None:
                self.dirty_since = now
                return "changed"
            if now - self.dirty_since >= self.max_wait_s:
                # never still (animated text effect / background): read anyway, stay "dirty" so the overlay is not
                # cleared again; the same text read twice is not translated twice
                self.dirty_since = now
                return "stable"
            return None
        if self.dirty_since is not None and (now - self.last_change >= self.stable_s or now - self.dirty_since >= self.max_wait_s):
            self.dirty_since = None
            return "stable"
        return None

    def force(self, now: float):
        """Treat the region as freshly changed (e.g. after the game window regained focus)."""
        self.dirty_since, self.last_change = now, now


def has_text(ink: np.ndarray, min_frac: float = 0.002) -> bool:
    """A nearly empty ink map means the box is gone or blank: no need to OCR it."""
    if ink.dtype == bool:
        return ink.mean() >= min_frac
    return True


def similar(a: str, b: str) -> float:
    """0..1 similarity of two OCR results (OCR jitters by a character or two between reads)."""
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    import difflib

    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()
