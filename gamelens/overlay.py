"""The overlay: a borderless, transparent, click-through, always-on-top window laid exactly over the game's client
area. Each item erases the Japanese glyphs (patch) and draws the Thai text fitted into the same place."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from PySide6.QtCore import QPointF, QRect, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QFontMetricsF, QGuiApplication, QImage, QPainter, QPainterPath, QPen, QTextLayout, QTextOption, QTransform
from PySide6.QtWidgets import QWidget

from .render import contrast_outline
from .thai import ZWSP, breakable

FADE_S = 0.12


def to_qimage(bgra: np.ndarray) -> QImage:
    h, w = bgra.shape[:2]
    img = QImage(np.ascontiguousarray(bgra).data, w, h, 4 * w, QImage.Format.Format_ARGB32)
    return img.copy()  # own the pixels (the numpy buffer goes away)


@dataclass
class Item:
    rect: tuple[int, int, int, int]  # text box in client pixels (physical)
    text: str
    fg: str = "#ffffff"
    max_px: float = 28
    align: str = "left"  # left / center / right
    outline: str = ""  # outline colour ('' = one that contrasts with fg)
    bold: bool = False
    vcenter: bool = False  # one line, centred vertically in rect (names, subtitles, UI labels)
    rows: list[float] | None = None  # middle (y) of each original line: the translation sits on the same lines
    angle: float = 0.0  # degrees clockwise around the middle of rect (text the game sets at an angle)
    patch: QImage | None = None
    patch_pos: tuple[int, int] = (0, 0)
    born: float = field(default_factory=time.monotonic)


def fit_lines(text: str, family: str, bold: bool, w: float, h: float, max_px: float, min_px: float = 9):
    """Largest font size (<= max_px) at which the wrapped text fits w x h. Returns (font, [(line, x_off, width)], line_h)."""
    text = breakable(text)
    lo, hi, best = min_px, max(min_px, max_px), None
    for _ in range(9):  # binary search on the pixel size
        px = (lo + hi) / 2
        f = QFont(family)
        f.setPixelSize(max(1, int(px)))
        f.setBold(bold)
        lines, lh, widest = _layout(text, f, w)
        if len(lines) * lh <= h and widest <= w + 1:
            best, lo = (f, lines, lh), px
        else:
            hi = px
    if best is None:  # does not fit even at min size: use it anyway (it overflows downwards)
        f = QFont(family)
        f.setPixelSize(int(min_px))
        f.setBold(bold)
        lines, lh, _ = _layout(text, f, w)
        best = (f, lines, lh)
    return best


def _layout(text: str, font: QFont, width: float):
    lay = QTextLayout(text, font)
    opt = QTextOption()
    opt.setWrapMode(QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
    lay.setTextOption(opt)
    lay.beginLayout()
    out, widest = [], 0.0
    while True:
        line = lay.createLine()
        if not line.isValid():
            break
        line.setLineWidth(width)
        s = text[line.textStart() : line.textStart() + line.textLength()].replace(ZWSP, "").rstrip()
        nw = line.naturalTextWidth()
        widest = max(widest, nw)
        out.append((s, nw))
    lay.endLayout()
    return out, QFontMetricsF(font).lineSpacing(), widest


class Overlay(QWidget):
    def __init__(self, family: str = "Bai Jamjuree"):
        super().__init__(None)
        self.family = family
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Tool
            | Qt.WindowType.WindowTransparentForInput | Qt.WindowType.WindowDoesNotAcceptFocus
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setStyleSheet("background: transparent;")  # the app theme must not paint the overlay
        self.items: dict[str, list[Item]] = {}
        self.pending: dict[str, tuple[int, int]] = {}  # key -> anchor (client px): "translating…" dots
        self.toast_text, self.toast_until = "", 0.0
        self.client = (0, 0, 1, 1)
        self.dpr = 1.0
        self._fade = QTimer(self, interval=16, timeout=self._fade_tick)

    # --- geometry ---------------------------------------------------------------------------------------------
    def follow(self, client: tuple[int, int, int, int]):
        """Place the overlay over the game's client area (physical pixels)."""
        if client == self.client and self.isVisible():
            return
        self.client = client
        x, y, w, h = client
        screen = QGuiApplication.screenAt(QRect(x, y, 1, 1).center()) or QGuiApplication.primaryScreen()
        self.dpr = screen.devicePixelRatio() if screen else 1.0
        d = self.dpr
        self.setGeometry(int(x / d), int(y / d), max(1, int(round(w / d))), max(1, int(round(h / d))))
        self.update()

    # --- content ----------------------------------------------------------------------------------------------
    def set_items(self, key: str, items: list[Item]):
        now = time.monotonic()
        for it in items:
            it.born = now
        self.items[key] = items
        self._fade.start()
        self.update()

    def clear(self, key: str | None = None):
        if key is None:
            self.items.clear()
            self.pending.clear()
        else:
            self.items.pop(key, None)
        self.update()

    def set_pending(self, key: str, anchor: tuple[int, int] | None):
        """Small animated dots at `anchor` while a translation for `key` is on its way."""
        if anchor is None:
            self.pending.pop(key, None)
        else:
            self.pending[key] = anchor
            self._fade.start()
        self.update()

    def toast(self, text: str, seconds: float = 2.5):
        self.toast_text, self.toast_until = text, time.monotonic() + seconds
        self._fade.start()
        self.update()

    def _fade_tick(self):
        now = time.monotonic()
        fading = any(now - it.born < FADE_S for its in self.items.values() for it in its)
        if not (fading or self.pending or now < self.toast_until + 0.3):
            self._fade.stop()
        self.update()

    def _draw_hud(self, p: QPainter, now: float):
        f = QFont(self.family)
        f.setPixelSize(max(13, int(15 * self.dpr)))
        f.setBold(True)
        if self.pending:
            p.setOpacity(1.0)
            r = 4 * self.dpr
            for x, y in self.pending.values():
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QColor(0, 0, 0, 150))
                p.drawRoundedRect(QRectF(x - 46 * self.dpr, y, 42 * self.dpr, 16 * self.dpr), 8 * self.dpr, 8 * self.dpr)
                for i in range(3):
                    phase = (now * 3 - i * 0.35) % 1.0
                    p.setBrush(QColor(255, 255, 255, int(90 + 165 * max(0.0, 1 - abs(phase - 0.5) * 2))))
                    p.drawEllipse(QPointF(x - 36 * self.dpr + i * 11 * self.dpr, y + 8 * self.dpr), r, r)
        if now < self.toast_until + 0.3 and self.toast_text:
            p.setOpacity(max(0.0, min(1.0, (self.toast_until + 0.3 - now) / 0.3)))
            fm = QFontMetricsF(f)
            w = fm.horizontalAdvance(self.toast_text) + 28 * self.dpr
            h = fm.height() + 14 * self.dpr
            x = (self.client[2] - w) / 2
            y = 18 * self.dpr
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(20, 22, 28, 215))
            p.drawRoundedRect(QRectF(x, y, w, h), h / 2, h / 2)
            p.setPen(QColor("#ffffff"))
            p.setFont(f)
            p.drawText(QRectF(x, y, w, h), Qt.AlignmentFlag.AlignCenter, self.toast_text)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        p.scale(1 / self.dpr, 1 / self.dpr)  # draw in physical pixels, like the capture
        now = time.monotonic()
        for its in self.items.values():
            for it in its:
                p.setOpacity(min(1.0, (now - it.born) / FADE_S))
                if it.patch is not None:
                    p.drawImage(QPointF(*it.patch_pos), it.patch)
                self._draw_text(p, it)
        self._draw_hud(p, now)
        p.end()

    def _draw_text(self, p: QPainter, it: Item):
        x, y, w, h = it.rect
        if not it.text or w < 4 or h < 4:
            return
        key = (it.text, w, h, round(it.max_px, 1), self.family)
        if getattr(it, "_fit_key", None) != key:  # lay out once, not on every repaint
            it._fit_key, it._fit = key, fit_lines(it.text, self.family, it.bold, w, h, it.max_px)
        font, lines, lh = it._fit
        fm = QFontMetricsF(font)
        path = QPainterPath()
        top = y + max(0.0, (h - len(lines) * lh) / 2) if it.vcenter or (len(lines) == 1 and it.align == "center") else y
        if it.rows:
            # as many lines or fewer: around the middle of the original block; more: from its first line down
            mid = sum(it.rows) / len(it.rows) if len(lines) <= len(it.rows) else it.rows[0] + (len(lines) - 1) * lh / 2
            top = mid - len(lines) * lh / 2 - 0.1 * font.pixelSize()
        elif it.vcenter:
            # a line box keeps headroom for marks above the letters, so the letters themselves sit below its
            # middle: lift them to where the original text's middle was
            top -= 0.1 * font.pixelSize()
        for i, (s, nw) in enumerate(lines):
            lx = x + (w - nw) / 2 if it.align == "center" else (x + w - nw if it.align == "right" else x)
            path.addText(QPointF(lx, top + i * lh + fm.ascent()), font, s)
        if it.angle:
            turn = QTransform().translate(x + w / 2, y + h / 2).rotate(it.angle).translate(-(x + w / 2), -(y + h / 2))
            path = turn.map(path)
        outline = QColor(it.outline or contrast_outline(it.fg))
        pen = QPen(outline, max(1.6, font.pixelSize() / 8))
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        p.strokePath(path, pen)
        p.fillPath(path, QColor(it.fg))


def text_rect_for(region_px: tuple[int, int, int, int], line_boxes: list[list[int]], kind: str,
                  align: str = "left", size: float = 0) -> tuple[int, int, int, int]:
    """The room a region's translation may take. Dialogue: from its first line down to the region's end (and at
    least two lines of `size`, so a short box does not shrink the text), as wide as the region allows. A name or
    subtitle: one line where the original is, with room above and below for Thai vowels and tone marks."""
    rx, ry, rw, rh = region_px
    if not line_boxes:
        return region_px
    x0 = min(b[0] for b in line_boxes)
    y0 = min(b[1] for b in line_boxes)
    x1 = max(b[0] + b[2] for b in line_boxes)
    y1 = max(b[1] + b[3] for b in line_boxes)
    x, w = (rx + 4, rw - 8) if align == "center" else (rx + x0, max(x1 - x0, rw - x0 - 4))
    if kind == "dialog":
        return x, ry + y0, w, int(max(y1 - y0, rh - y0 - 4, 2.8 * size))
    tall = int(max(1.9 * (y1 - y0), 1.6 * size))
    return x, ry + (y0 + y1) // 2 - tall // 2, w, tall
