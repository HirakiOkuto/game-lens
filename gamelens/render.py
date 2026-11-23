"""Erasing the original text: a patch that covers only the Japanese glyphs (inpainted from their surroundings),
transparent everywhere else, so the live game stays visible around the Thai text."""
from __future__ import annotations

import cv2
import numpy as np


def glyph_mask(crop_bgr: np.ndarray, boxes: list[list[int]]) -> np.ndarray:
    """Text pixels inside the line boxes (Otsu per line: the smaller class is the ink), grown past the box so
    anti-aliased edges, outlines and drop shadows are covered too (a thin halo left behind shows as a ghost)."""
    h, w = crop_bgr.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    for x, y, bw, bh in boxes:
        x0, y0, x1, y1 = max(0, x), max(0, y), min(w, x + bw), min(h, y + bh)
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue
        _, th = cv2.threshold(gray[y0:y1, x0:x1], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        ink = th > 0 if (th > 0).mean() < 0.5 else th == 0
        grow = max(3, int(0.18 * bh))
        # dilate on a canvas larger than the box, so the growth is not cut off at the box edge
        gx0, gy0, gx1, gy1 = max(0, x0 - grow), max(0, y0 - grow), min(w, x1 + grow), min(h, y1 + grow)
        canvas = np.zeros((gy1 - gy0, gx1 - gx0), np.uint8)
        canvas[y0 - gy0 : y1 - gy0, x0 - gx0 : x1 - gx0] = ink.astype(np.uint8) * 255
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1))
        mask[gy0:gy1, gx0:gx1] = np.maximum(mask[gy0:gy1, gx0:gx1], cv2.dilate(canvas, k))
    return mask


def erase_patch(crop_bgr: np.ndarray, boxes: list[list[int]]) -> np.ndarray | None:
    """BGRA image the size of the crop: inpainted pixels where the glyphs were, alpha feathered at the edges."""
    if not boxes:
        return None
    mask = glyph_mask(crop_bgr, boxes)
    if not mask.any():
        return None
    # inpaint at reduced size for speed on big boxes, then scale back
    h, w = crop_bgr.shape[:2]
    scale = min(1.0, 640 / max(w, 1))
    if scale < 1.0:
        small = cv2.resize(crop_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        sm = cv2.resize(mask, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
        filled = cv2.resize(cv2.inpaint(small, sm, 3, cv2.INPAINT_TELEA), (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        filled = cv2.inpaint(crop_bgr, mask, 3, cv2.INPAINT_TELEA)
    alpha = cv2.GaussianBlur(mask, (0, 0), 1.5)
    return np.dstack([filled, alpha])


def contrast_outline(fg_hex: str) -> str:
    """Outline colour for Thai text: dark around light text, light around dark text."""
    r, g, b = (int(fg_hex[i : i + 2], 16) for i in (1, 3, 5))
    return "#101010" if 0.299 * r + 0.587 * g + 0.114 * b > 128 else "#f4f4f4"
