"""Marking the screen regions to watch, on a frozen frame of the game.

game-lens first finds the text on screen and outlines it (dashed); one click on the dialogue makes a region.
Drag on empty space for a region by hand; drag a region to move it, its corner to resize it.
1 / 2 / 3 set the kind of the selected region, Delete removes it, Enter saves, Esc cancels."""
from __future__ import annotations

import threading
import uuid

import numpy as np
from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen
from PySide6.QtWidgets import QHBoxLayout, QMenu, QPushButton, QWidget

from .profile import KIND_LABELS, KINDS, Region

COLORS = {"dialog": "#4fc3f7", "name": "#ffb74d", "subtitle": "#aed581"}
HANDLE = 14


def cluster_lines(lines: list[dict]) -> list[dict]:
    """OCR lines -> text blocks: pieces of one row, and rows stacked close together with overlapping x.
    Each block: {bbox, lines (rows), text, lh (line height)}."""
    blocks: list[dict] = []
    for l in sorted(lines, key=lambda l: (l["bbox"][1], l["bbox"][0])):
        x, y, w, h = l["bbox"]
        for b in blocks:
            bx, by, bw, bh = b["bbox"]
            lh = b["lh"]
            if abs(h - lh) > 0.5 * max(h, lh):
                continue  # a different text size: a different element
            row_y = b["last_y"]
            same_row = min(y + h, row_y + lh) - max(y, row_y) > 0.5 * min(h, lh) and x - (bx + bw) < 2.5 * lh
            next_row = 0 <= y - (row_y + lh) < 1.3 * h and min(x + w, bx + bw) - max(x, bx) > -0.5 * h
            if same_row or next_row:
                nx, ny = min(x, bx), min(y, by)
                b["bbox"] = [nx, ny, max(x + w, bx + bw) - nx, max(y + h, by + bh) - ny]
                if next_row and not same_row:
                    b["lines"] += 1
                    b["last_y"] = y
                b["text"] += l["text"]
                break
        else:
            blocks.append({"bbox": [x, y, w, h], "lines": 1, "text": l["text"], "lh": h, "last_y": y})
    return blocks


def is_name_block(block: dict, blocks: list[dict]) -> bool:
    """A speaker name: one short line sitting just above a bigger text block."""
    if block["lines"] > 1 or len(block["text"]) > 12:
        return False
    x, y, w, h = block["bbox"]
    for b in blocks:
        if b is block:
            continue
        bx, by, bw, bh = b["bbox"]
        below = 0 <= by - (y + h) < 4 * max(h, b["lh"])
        if below and (bw > 1.5 * w or b["lines"] > 1) and min(x + w, bx + bw) - max(x, bx) > -2 * h:
            return True
    return False


def suggest_region(block: dict, fw: int, fh: int, blocks: list[dict] | None = None) -> tuple[str, list[float]]:
    """A region for a clicked text block: a speaker name gets a snug box; dialogue gets a box widened and deepened
    so longer lines and up to three lines still fall inside it."""
    x, y, w, h = block["bbox"]
    lh = block["lh"]
    if is_name_block(block, blocks or [block]):
        kind, pad = "name", int(0.35 * lh)
        rect = [x - pad, y - pad, w + 2 * pad + int(2 * lh), h + 2 * pad]
    else:
        kind, pad = "dialog", int(0.5 * lh)
        x1 = max(x + w + pad, min(fw - pad, x + int(0.6 * fw)))
        rect = [x - pad, y - pad, x1 - (x - pad), max(h, 3 * lh + 2 * int(0.45 * lh)) + 2 * pad]
    rx, ry = max(0, rect[0]), max(0, rect[1])
    rw, rh = min(fw - rx, rect[2]), min(fh - ry, rect[3])
    return kind, [rx / fw, ry / fh, rw / fw, rh / fh]


def snap_to_box(frame: np.ndarray, block: dict, rect: list[float]) -> list[float]:
    """Grow a dialogue region to the edges of the box the text sits in: walk right and down from the text while
    the background stays the box's colour (a dialogue box is a flat or softly shaded panel)."""
    fh, fw = frame.shape[:2]
    x, y, w, h = block["bbox"]
    lh = block["lh"]
    band = frame[y : y + h, :].astype(np.int16)
    if band.size == 0:
        return rect
    ref = np.median(band[:, max(0, x - int(0.3 * lh)) : x].reshape(-1, 3), 0) if x > 2 else np.median(band.reshape(-1, 3), 0)

    def same(col_pixels):
        return np.linalg.norm(np.median(col_pixels.reshape(-1, 3), 0) - ref) < 28

    right = x + w
    while right < fw - 1 and same(band[:, right : right + 1]):
        right += 1
    col = frame[:, max(0, x - 2) : x + min(w, int(4 * lh))].astype(np.int16)
    bottom = y + h
    while bottom < fh - 1 and same(col[bottom : bottom + 1]):
        bottom += 1
    pad = int(0.25 * lh)
    nx, ny = rect[0] * fw, rect[1] * fh
    nw = max(rect[2] * fw, right - pad - nx)
    nh = max(rect[3] * fh, bottom - pad - ny) if bottom - y < 8 * lh else rect[3] * fh
    nw = min(nw, fw - nx)
    nh = min(nh, fh - ny)
    return [nx / fw, ny / fh, nw / fw, nh / fh]


class RegionEditor(QWidget):
    saved = Signal(list)
    cancelled = Signal()
    _suggested = Signal(list)

    def __init__(self, frame_bgr: np.ndarray, regions: list[Region], geometry: QRect, find_text=None):
        super().__init__(None)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Tool)
        self.setGeometry(geometry)
        h, w = frame_bgr.shape[:2]
        rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
        self.img = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
        self.fw, self.fh = w, h
        self.frame = frame_bgr
        self.regions = [Region(**vars(r)) for r in regions]
        self.selected: Region | None = None
        self.blocks: list[dict] = []
        self.status = "กำลังหาข้อความบนจอ…" if find_text else ""
        self.mode = None  # 'new' / 'move' / 'resize'
        self.press: QPoint | None = None
        self.cursor_pos: QPoint | None = None
        self.orig: list[float] | None = None
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        bar = QWidget(self)
        bl = QHBoxLayout(bar)
        bl.setContentsMargins(0, 0, 0, 0)
        for text, slot, primary in (("ยกเลิก (Esc)", self._cancel, False), ("บันทึก (Enter)", self._save, True)):
            b = QPushButton(text)
            b.setObjectName("primary" if primary else "")
            b.clicked.connect(slot)
            bl.addWidget(b)
        bar.adjustSize()
        self.bar = bar

        self._suggested.connect(self._got_blocks)
        if find_text:
            threading.Thread(target=lambda: self._suggested.emit(self._find(find_text, frame_bgr)), daemon=True).start()

    @staticmethod
    def _find(find_text, frame) -> list:
        try:
            return cluster_lines(find_text(frame))
        except Exception:  # noqa: BLE001 - suggestions are optional
            return []

    def _got_blocks(self, blocks: list):
        self.blocks = blocks
        self.status = ("คลิกที่บทพูด (เส้นประ) เพื่อสร้างกรอบ" if blocks else "ไม่พบข้อความบนจอ — ลากกรอบเองได้เลย")
        self.update()

    def resizeEvent(self, _):
        self.bar.move(self.width() - self.bar.width() - 16, self.height() - self.bar.height() - 16)

    # widget <-> normalised <-> frame
    def _to_widget(self, n: list[float]) -> QRect:
        W, H = self.width(), self.height()
        return QRect(int(n[0] * W), int(n[1] * H), int(n[2] * W), int(n[3] * H))

    def _block_rect(self, b: dict) -> QRect:
        x, y, w, h = b["bbox"]
        return self._to_widget([x / self.fw, y / self.fh, w / self.fw, h / self.fh])

    def paintEvent(self, _):
        p = QPainter(self)
        p.drawImage(self.rect(), self.img)
        p.fillRect(self.rect(), QColor(0, 0, 0, 110))
        f = QFont("Leelawadee UI")
        f.setPixelSize(15)
        p.setFont(f)
        hover = self._block_at(self.cursor_pos) if self.cursor_pos and self.mode is None else None
        for b in self.blocks:
            q = self._block_rect(b).adjusted(-4, -4, 4, 4)
            p.setPen(QPen(QColor("#ffffff" if b is hover else "#c8d2e0"), 2 if b is hover else 1, Qt.PenStyle.DashLine))
            p.setBrush(QColor(255, 255, 255, 40 if b is hover else 0))
            p.drawRect(q)
        p.setBrush(Qt.BrushStyle.NoBrush)
        for r in self.regions:
            q = self._to_widget(r.rect)
            c = QColor(COLORS.get(r.kind, "#ffffff"))
            p.fillRect(q, QColor(c.red(), c.green(), c.blue(), 60 if r is self.selected else 35))
            p.setPen(QPen(c, 3 if r is self.selected else 2))
            p.drawRect(q)
            tag = QRect(q.x(), q.y() - 24, max(130, len(KIND_LABELS[r.kind]) * 9), 22)
            p.fillRect(tag, c)
            p.setPen(QColor("#101318"))
            p.drawText(tag.adjusted(6, 0, 0, 0), Qt.AlignmentFlag.AlignVCenter, KIND_LABELS.get(r.kind, r.kind))
            p.fillRect(QRect(q.right() - HANDLE + 1, q.bottom() - HANDLE + 1, HANDLE, HANDLE), c)
        if self.mode == "new" and self.press and self.cursor_pos:
            p.setPen(QPen(QColor("#ffffff"), 1, Qt.PenStyle.DashLine))
            p.drawRect(QRect(self.press, self.cursor_pos).normalized())
        # instructions
        p.fillRect(QRect(0, 0, self.width(), 58), QColor(10, 12, 16, 210))
        p.setPen(QColor("#ffffff"))
        f.setPixelSize(17)
        f.setBold(True)
        p.setFont(f)
        p.drawText(QRect(16, 6, self.width() - 32, 24), Qt.AlignmentFlag.AlignVCenter,
                   self.status or "ลากเพื่อสร้างกรอบรอบกล่องบทพูด")
        f.setPixelSize(13)
        f.setBold(False)
        p.setFont(f)
        p.setPen(QColor("#b9c2d0"))
        p.drawText(QRect(16, 30, self.width() - 32, 22), Qt.AlignmentFlag.AlignVCenter,
                   "ลากที่ว่าง = กรอบใหม่ • ลากกรอบ = ย้าย • ลากมุมขวาล่าง = ปรับขนาด • "
                   "1 บทพูด  2 ชื่อผู้พูด  3 ซับ • Delete ลบ • คลิกขวา = ตัวเลือก")
        p.end()

    # --- hit tests --------------------------------------------------------------------------------------------
    def _region_at(self, pos: QPoint) -> Region | None:
        for r in reversed(self.regions):
            if self._to_widget(r.rect).adjusted(-2, -2, 2, 2).contains(pos):
                return r
        return None

    def _block_at(self, pos: QPoint | None) -> dict | None:
        if pos is None or self._region_at(pos):
            return None
        for b in self.blocks:
            if self._block_rect(b).adjusted(-6, -6, 6, 6).contains(pos):
                return b
        return None

    def _on_handle(self, r: Region, pos: QPoint) -> bool:
        q = self._to_widget(r.rect)
        return abs(pos.x() - q.right()) <= HANDLE and abs(pos.y() - q.bottom()) <= HANDLE

    # --- mouse ------------------------------------------------------------------------------------------------
    def mousePressEvent(self, e):
        pos = e.position().toPoint()
        r = self._region_at(pos)
        if e.button() == Qt.MouseButton.RightButton:
            if r:
                self.selected = r
                self._menu(r, e.globalPosition().toPoint())
            return
        self.press = pos
        if r:
            self.selected = r
            self.mode = "resize" if self._on_handle(r, pos) else "move"
            self.orig = list(r.rect)
        elif (b := self._block_at(pos)) is not None:
            kind, rect = suggest_region(b, self.fw, self.fh, self.blocks)
            if kind == "dialog":
                rect = snap_to_box(self.frame, b, rect)
            self.selected = Region(id=uuid.uuid4().hex[:8], kind=kind, rect=rect)
            self.regions.append(self.selected)
            self.mode = None
        else:
            self.selected = None
            self.mode = "new"
        self.update()

    def mouseMoveEvent(self, e):
        pos = e.position().toPoint()
        self.cursor_pos = pos
        W, H = max(1, self.width()), max(1, self.height())
        if self.mode in ("move", "resize") and self.selected and self.press:
            dx, dy = (pos.x() - self.press.x()) / W, (pos.y() - self.press.y()) / H
            x, y, w, h = self.orig
            if self.mode == "move":
                self.selected.rect = [min(max(0, x + dx), 1 - w), min(max(0, y + dy), 1 - h), w, h]
            else:
                self.selected.rect = [x, y, max(0.02, min(1 - x, w + dx)), max(0.015, min(1 - y, h + dy))]
        r = self._region_at(pos)
        if r and self._on_handle(r, pos):
            self.setCursor(Qt.CursorShape.SizeFDiagCursor)
        elif r:
            self.setCursor(Qt.CursorShape.SizeAllCursor)
        elif self._block_at(pos):
            self.setCursor(Qt.CursorShape.PointingHandCursor)
        else:
            self.setCursor(Qt.CursorShape.CrossCursor)
        self.update()

    def mouseReleaseEvent(self, e):
        if self.mode == "new" and self.press:
            q = QRect(self.press, e.position().toPoint()).normalized()
            if q.width() > 12 and q.height() > 8:
                W, H = max(1, self.width()), max(1, self.height())
                self.selected = Region(id=uuid.uuid4().hex[:8], kind="dialog",
                                       rect=[q.x() / W, q.y() / H, q.width() / W, q.height() / H])
                self.regions.append(self.selected)
        self.mode, self.press = None, None
        self.update()

    def _menu(self, r: Region, at):
        m = QMenu(self)
        for i, k in enumerate(KINDS, 1):
            a = m.addAction(f"{i}  {KIND_LABELS[k]}")
            a.setCheckable(True)
            a.setChecked(r.kind == k)
            a.triggered.connect(lambda _=False, k=k: setattr(r, "kind", k))
        m.addSeparator()
        ink = m.addMenu("สีตัวอักษรในกรอบ")
        for v, label in (("light", "สว่าง (ขาวบนพื้นเข้ม)"), ("dark", "เข้ม (ดำบนพื้นสว่าง)"), ("any", "ไม่แน่นอน / หลายสี")):
            a = ink.addAction(label)
            a.setCheckable(True)
            a.setChecked(r.ink == v)
            a.triggered.connect(lambda _=False, v=v: setattr(r, "ink", v))
        al = m.addAction("จัดข้อความกึ่งกลาง")
        al.setCheckable(True)
        al.setChecked(r.align == "center")
        al.triggered.connect(lambda on: setattr(r, "align", "center" if on else "left"))
        m.addSeparator()
        m.addAction("ลบกรอบนี้ (Delete)").triggered.connect(lambda: self._delete(r))
        m.exec(at)
        self.update()

    def _delete(self, r: Region):
        if r in self.regions:
            self.regions.remove(r)
        if self.selected is r:
            self.selected = None
        self.update()

    def keyPressEvent(self, e):
        k = e.key()
        if k in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self._save()
        elif k == Qt.Key.Key_Escape:
            self._cancel()
        elif k in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace) and self.selected:
            self._delete(self.selected)
        elif self.selected and Qt.Key.Key_1 <= k <= Qt.Key.Key_3:
            self.selected.kind = KINDS[k - Qt.Key.Key_1]
            self.update()

    def _save(self):
        self.saved.emit(self.regions)
        self.close()

    def _cancel(self):
        self.cancelled.emit()
        self.close()
