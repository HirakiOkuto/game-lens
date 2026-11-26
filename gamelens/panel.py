"""The main window: a 3-step readiness checklist, one big start button, engine + key setup and display options.
Closing it only hides it (game-lens keeps running in the tray)."""
from __future__ import annotations

import webbrowser

from PySide6.QtCore import QMargins, Qt, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFrame, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
                               QScrollArea, QSlider, QVBoxLayout, QWidget)

from .profile import GAME_LANGS, ENGINE_BY_KEY, ENGINE_INFO
from .ui_style import BAD, IDLE, OK, WARN

COLORS = {"ok": OK, "warn": WARN, "bad": BAD, "idle": IDLE}


def repolish(w: QWidget):
    """Re-apply the style sheet after an object-name change."""
    w.style().unpolish(w)
    w.style().polish(w)


def card(title: str) -> tuple[QFrame, QVBoxLayout]:
    f = QFrame()
    f.setObjectName("card")
    lay = QVBoxLayout(f)
    lay.setContentsMargins(14, 12, 14, 12)
    lay.setSpacing(6)
    if title:
        t = QLabel(title)
        t.setObjectName("h2")
        lay.addWidget(t)
    return f, lay


class Step(QWidget):
    """● 1. title — detail ......... [action]"""

    def __init__(self, n: int, title: str, action: str):
        super().__init__()
        self.dot = QLabel("●")
        self.dot.setFixedWidth(18)
        self.title = QLabel(f"{n}. {title}")
        self.title.setObjectName("h2")
        self.detail = QLabel("")
        self.detail.setObjectName("muted")
        self.detail.setWordWrap(True)
        self.button = QPushButton(action)
        text = QVBoxLayout()
        text.setSpacing(1)
        text.addWidget(self.title)
        text.addWidget(self.detail)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 4, 0, 4)
        row.addWidget(self.dot, 0, Qt.AlignmentFlag.AlignTop)
        row.addLayout(text, 1)
        row.addWidget(self.button, 0, Qt.AlignmentFlag.AlignVCenter)
        self.set("idle", "")

    def set(self, state: str, detail: str, action: str | None = None, primary: bool = False):
        self.dot.setStyleSheet(f"color: {COLORS[state]}; font-size: 14pt;")
        self.detail.setText(detail)
        if action is None:
            self.button.hide()
        else:
            self.button.setText(action)
            self.button.setObjectName("primary" if primary else "")
            repolish(self.button)
            self.button.show()


class Panel(QWidget):
    toggle = Signal()
    fix_server = Signal()
    pick_game = Signal()
    edit_regions = Signal()
    engine_changed = Signal(str)
    model_size_changed = Signal(str)
    save_keys = Signal(dict)
    test_engine = Signal()
    font_scale_changed = Signal(float)
    target_changed = Signal(str)
    src_lang_changed = Signal(str)
    toasts_changed = Signal(bool)
    auto_screen_changed = Signal(bool)
    show_history = Signal()
    make_shortcut = Signal()
    quit_app = Signal()

    def __init__(self, hotkeys: dict, peek_key: str):
        super().__init__(None)
        self.setWindowTitle("game-lens")
        self.setObjectName("window")
        self.setMinimumWidth(500)
        self.ready = False

        head = QHBoxLayout()
        title = QLabel("訳  game-lens")
        title.setObjectName("title")
        sub = QLabel("แปลเกมญี่ปุ่นเป็นไทย ทับบนจอเกม")
        sub.setObjectName("muted")
        tcol = QVBoxLayout()
        tcol.setSpacing(0)
        tcol.addWidget(title)
        tcol.addWidget(sub)
        head.addLayout(tcol)

        # 1-2-3 checklist
        steps, sl = card("เตรียมพร้อม")
        self.s_server = Step(1, "ระบบแปล", "เปิด")
        self.s_game = Step(2, "เกม", "เลือกเกม")
        self.s_regions = Step(3, "กรอบข้อความในเกม", "ตั้งกรอบ")
        self.s_server.button.clicked.connect(self.fix_server.emit)
        self.s_game.button.clicked.connect(self.pick_game.emit)
        self.s_regions.button.clicked.connect(self.edit_regions.emit)
        for s in (self.s_server, self.s_game, self.s_regions):
            sl.addWidget(s)

        self.big = QPushButton("▶  เริ่มแปล")
        self.big.setObjectName("big")
        self.big.clicked.connect(self.toggle.emit)
        self.big_hint = QLabel("")
        self.big_hint.setObjectName("hint")
        self.big_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # engine
        eng, el = card("เครื่องแปล")
        self.engine = QComboBox()
        for e in ENGINE_INFO:
            self.engine.addItem(e["name"], e["key"])
        self.engine.currentIndexChanged.connect(self._engine_picked)
        self.engine_desc = QLabel("")
        self.engine_desc.setObjectName("muted")
        self.engine_desc.setWordWrap(True)
        # which size of the local model (only for Hy-MT2)
        self.size_box = QWidget()
        szl = QHBoxLayout(self.size_box)
        szl.setContentsMargins(0, 0, 0, 0)
        szl.addWidget(QLabel("รุ่นโมเดล"))
        self.size = QComboBox()
        for label, key in (("อัตโนมัติตามการ์ดจอ", "auto"),
                           ("Index-Translate 9B — แปลเป็นธรรมชาติที่สุด • การ์ดจอ ~5.5 GB • โหลด 5.8 GB", "index"),
                           ("Hy-MT2 7B — การ์ดจอ ~5.5 GB • โหลด 4.6 GB", "7b"),
                           ("Hy-MT2 1.8B — เร็ว การ์ดจอ ~2 GB หรือ CPU • แปลด้อยกว่า • โหลด 1.1 GB", "1.8b")):
            self.size.addItem(label, key)
        self.size.currentIndexChanged.connect(lambda _: self.model_size_changed.emit(self.size.currentData()))
        szl.addWidget(self.size, 1)
        self.key_box = QWidget()
        kl = QGridLayout(self.key_box)
        kl.setContentsMargins(0, 2, 0, 2)
        kl.setVerticalSpacing(8)
        self.key_edits: dict[str, QLineEdit] = {}
        self.key_labels: dict[str, QLabel] = {}
        row = 0
        for e in ENGINE_INFO:
            for name, label in e.get("keys", []):
                lab = QLabel(label)
                ed = QLineEdit()
                ed.setEchoMode(QLineEdit.EchoMode.Password if "REGION" not in name else QLineEdit.EchoMode.Normal)
                kl.addWidget(lab, row, 0)
                kl.addWidget(ed, row, 1)
                self.key_edits[name], self.key_labels[name] = ed, lab
                row += 1
        krow = QHBoxLayout()
        self.signup = QPushButton("สมัคร / ขอ key ↗")
        self.signup.setObjectName("link")
        self.signup.clicked.connect(lambda: webbrowser.open(ENGINE_BY_KEY[self.engine.currentData()].get("signup", "")))
        save = QPushButton("บันทึก key")
        save.clicked.connect(self._save_keys)
        krow.addWidget(self.signup)
        krow.addStretch()
        krow.addWidget(save)
        kl.addLayout(krow, row, 0, 1, 2)
        trow = QHBoxLayout()
        self.test_btn = QPushButton("ทดสอบ")
        self.test_btn.clicked.connect(self.test_engine.emit)
        self.engine_status = QLabel("")
        self.engine_status.setObjectName("muted")
        self.engine_status.setWordWrap(True)
        trow.addWidget(self.test_btn)
        trow.addWidget(self.engine_status, 1)
        el.addWidget(self.engine)
        el.addWidget(self.engine_desc)
        el.addWidget(self.size_box)
        el.addWidget(self.key_box)
        el.addLayout(trow)

        # display
        disp, dl = card("การแสดงผล")
        g = QGridLayout()
        g.addWidget(QLabel("ขนาดตัวอักษร"), 0, 0)
        self.font = QSlider(Qt.Orientation.Horizontal, minimum=60, maximum=130)
        self.font.valueChanged.connect(lambda v: (self.font_val.setText(f"{v}%"), self.font_scale_changed.emit(v / 100)))
        self.font_val = QLabel("")
        self.font_val.setFixedWidth(44)
        g.addWidget(self.font, 0, 1)
        g.addWidget(self.font_val, 0, 2)
        g.addWidget(QLabel("ภาษาในเกม"), 1, 0)
        self.src = QComboBox()
        for code, label in GAME_LANGS:
            self.src.addItem(label, code)
        self.src.currentIndexChanged.connect(lambda _: self.src_lang_changed.emit(self.src.currentData()))
        g.addWidget(self.src, 1, 1, 1, 2)
        g.addWidget(QLabel("แปลเป็น"), 2, 0)
        self.target = QComboBox()
        self.target.addItem("ภาษาไทย", "th")
        self.target.addItem("English", "en")
        self.target.currentIndexChanged.connect(lambda _: self.target_changed.emit(self.target.currentData()))
        g.addWidget(self.target, 2, 1, 1, 2)
        self.toasts = QCheckBox("แสดงข้อความสถานะสั้นๆ บนจอเกม")
        self.toasts.toggled.connect(self.toasts_changed.emit)
        self.auto = QCheckBox("แปลเมนู / เควสต์เองเมื่อหน้าจอนิ่ง (ไม่ต้องกด F9)")
        self.auto.setToolTip("เมื่อภาพทั้งจอหยุดนิ่ง เช่น เปิดเมนู จะแปลข้อความทั้งหน้าให้เอง\n"
                             "ใช้เครื่องแปลมากขึ้น ถ้าใช้ API ที่มีโควตาควรระวัง")
        self.auto.toggled.connect(self.auto_screen_changed.emit)
        dl.addLayout(g)
        dl.addWidget(self.auto)
        dl.addWidget(self.toasts)

        # hotkeys
        keys, kl2 = card("ปุ่มลัดระหว่างเล่น")
        grid = QGridLayout()
        rows = [(peek_key, "กดค้าง = ดูต้นฉบับ"), (hotkeys.get("toggle", ""), "หยุด / เริ่มแปล"),
                (hotkeys.get("screen", ""), "แปลทั้งจอ (เมนู)"), (hotkeys.get("regions", ""), "ตั้งกรอบข้อความ"),
                (hotkeys.get("history", ""), "ประวัติคำแปล")]
        for i, (k, what) in enumerate(rows):
            kk = QLabel(k)
            kk.setStyleSheet("background:#262a33; border:1px solid #343a46; border-radius:6px; padding:1px 7px;")
            w = QLabel(what)
            w.setObjectName("muted")
            grid.addWidget(kk, i // 2, (i % 2) * 2, Qt.AlignmentFlag.AlignLeft)
            grid.addWidget(w, i // 2, (i % 2) * 2 + 1)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)
        grid.setHorizontalSpacing(8)
        kl2.addLayout(grid)

        foot = QHBoxLayout()
        hist = QPushButton("ประวัติคำแปล")
        hist.clicked.connect(self.show_history.emit)
        sc = QPushButton("สร้างไอคอนบน Desktop")
        sc.clicked.connect(self.make_shortcut.emit)
        q = QPushButton("ปิดโปรแกรม")
        q.clicked.connect(self.quit_app.emit)
        foot.addWidget(hist)
        foot.addWidget(sc)
        foot.addStretch()
        foot.addWidget(q)

        body = QWidget()
        body.setObjectName("window")
        lay = QVBoxLayout(body)
        lay.setContentsMargins(16, 14, 16, 14)
        lay.setSpacing(10)
        lay.addLayout(head)
        lay.addWidget(steps)
        lay.addWidget(self.big)
        lay.addWidget(self.big_hint)
        lay.addWidget(eng)
        lay.addWidget(disp)
        lay.addWidget(keys)
        lay.addLayout(foot)
        # a narrow window: drop-downs shrink to it (their lists still open as wide as their longest entry)
        for combo in body.findChildren(QComboBox):
            combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(10)
            combo.view().setMinimumWidth(combo.view().sizeHintForColumn(0) + 24)
        scroll = QScrollArea()
        scroll.setWidget(body)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)
        self._body = body

    def sizeHint(self):
        h = self._body.sizeHint()
        return h.expandedTo(self.minimumSize()).grownBy(QMargins(0, 0, 12, 0))

    # --- from the controller ----------------------------------------------------------------------------------
    def load(self, engine: str, font_scale: float, target: str, toasts: bool, auto_screen: bool = False,
             model_size: str = "auto", auto_label: str = "", src_lang: str = "ja"):
        self.auto.blockSignals(True)
        self.auto.setChecked(auto_screen)
        self.auto.blockSignals(False)
        self.size.blockSignals(True)
        if auto_label:
            self.size.setItemText(0, f"อัตโนมัติ — เครื่องนี้: {auto_label}")
        self.size.setCurrentIndex(max(0, self.size.findData(model_size)))
        self.size.blockSignals(False)
        for w in (self.engine, self.font, self.target, self.toasts, self.src):
            w.blockSignals(True)
        self.src.setCurrentIndex(max(0, self.src.findData(src_lang)))
        self.engine.setCurrentIndex(max(0, self.engine.findData(engine)))
        self.font.setValue(int(round(font_scale * 100)))
        self.font_val.setText(f"{self.font.value()}%")
        self.target.setCurrentIndex(max(0, self.target.findData(target)))
        self.toasts.setChecked(toasts)
        for w in (self.engine, self.font, self.target, self.toasts, self.src):
            w.blockSignals(False)
        self._show_engine(engine)

    def set_running(self, running: bool, ready: bool, hint: str):
        running = running and ready  # "stop" only means something once it is actually translating
        self.big.setText("■  หยุดแปล" if running else "▶  เริ่มแปล")
        self.big.setObjectName("bigstop" if running else "big")
        repolish(self.big)
        self.big.setEnabled(ready or running)
        self.big_hint.setText(hint)

    def set_keys(self, masked: dict):
        for names in masked.values():
            for n, v in names.items():
                if n in self.key_edits and not self.key_edits[n].hasFocus():
                    self.key_edits[n].clear()
                    self.key_edits[n].setPlaceholderText(f"ตั้งไว้แล้ว ({v})" if v else "วาง key ที่นี่")

    def set_engine_status(self, text: str, state: str = "idle"):
        self.engine_status.setText(text)
        self.engine_status.setStyleSheet(f"color: {COLORS[state] if state != 'idle' else '#9aa3b2'};")

    # --- internal ---------------------------------------------------------------------------------------------
    def _engine_picked(self):
        key = self.engine.currentData()
        self._show_engine(key)
        self.engine_changed.emit(key)

    def _show_engine(self, key: str):
        info = ENGINE_BY_KEY[key]
        self.engine_desc.setText(info["desc"])
        self.size_box.setVisible(key == "hymt")
        mine = {n for n, _ in info.get("keys", [])}
        for n, ed in self.key_edits.items():
            ed.setVisible(n in mine)
            self.key_labels[n].setVisible(n in mine)
        self.key_box.setVisible(bool(mine))
        self.signup.setVisible(bool(info.get("signup")))
        self.set_engine_status("")

    def _save_keys(self):
        info = ENGINE_BY_KEY[self.engine.currentData()]
        vals = {n: self.key_edits[n].text().strip() for n, _ in info.get("keys", []) if self.key_edits[n].text().strip()}
        if vals:
            self.save_keys.emit(vals)

    def closeEvent(self, e):
        e.ignore()
        self.hide()
