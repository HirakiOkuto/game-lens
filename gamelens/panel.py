"""The main window: a 3-step readiness checklist, one big start button, engine + key setup and display options.
Closing it only hides it (game-lens keeps running in the tray)."""
from __future__ import annotations

import webbrowser

from PySide6.QtCore import QMargins, Qt, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFrame, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
                               QScrollArea, QSlider, QVBoxLayout, QWidget)

from .i18n import LANGS, lang, t
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
        head = QLabel(title)
        head.setObjectName("h2")
        lay.addWidget(head)
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
    english_keys_changed = Signal(bool)
    ui_lang_changed = Signal(str)
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
        sub = QLabel(t("Game text translated into Thai, drawn over the game"))
        sub.setObjectName("muted")
        tcol = QVBoxLayout()
        tcol.setSpacing(0)
        tcol.addWidget(title)
        tcol.addWidget(sub)
        head.addLayout(tcol)

        # 1-2-3 checklist
        steps, sl = card(t("Getting ready"))
        self.s_server = Step(1, t("Translation system"), t("Start"))
        self.s_game = Step(2, t("Game"), t("Choose game"))
        self.s_regions = Step(3, t("Text regions in the game"), t("Set regions"))
        self.s_server.button.clicked.connect(self.fix_server.emit)
        self.s_game.button.clicked.connect(self.pick_game.emit)
        self.s_regions.button.clicked.connect(self.edit_regions.emit)
        for s in (self.s_server, self.s_game, self.s_regions):
            sl.addWidget(s)

        self.big = QPushButton(t("▶  Start translating"))
        self.big.setObjectName("big")
        self.big.clicked.connect(self.toggle.emit)
        self.big_hint = QLabel("")
        self.big_hint.setObjectName("hint")
        self.big_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # engine
        eng, el = card(t("Translator"))
        self.engine = QComboBox()
        for e in ENGINE_INFO:
            self.engine.addItem(t(e["name"]), e["key"])
        self.engine.currentIndexChanged.connect(self._engine_picked)
        self.engine_desc = QLabel("")
        self.engine_desc.setObjectName("muted")
        self.engine_desc.setWordWrap(True)
        # which size of the local model (only for Hy-MT2)
        self.size_box = QWidget()
        szl = QHBoxLayout(self.size_box)
        szl.setContentsMargins(0, 0, 0, 0)
        szl.addWidget(QLabel(t("Model")))
        self.size = QComboBox()
        for label, key in (("Automatic, by graphics card", "auto"),
                           ("Index-Translate 9B — most natural • ~5.5 GB graphics memory • 5.8 GB download", "index"),
                           ("Hy-MT2 7B — ~5.5 GB graphics memory • 4.6 GB download", "7b"),
                           ("Hy-MT2 1.8B — fast, ~2 GB graphics memory or CPU • lower quality • 1.1 GB download", "1.8b")):
            self.size.addItem(t(label), key)
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
                lab = QLabel(t(label))
                ed = QLineEdit()
                ed.setEchoMode(QLineEdit.EchoMode.Password if "REGION" not in name else QLineEdit.EchoMode.Normal)
                kl.addWidget(lab, row, 0)
                kl.addWidget(ed, row, 1)
                self.key_edits[name], self.key_labels[name] = ed, lab
                row += 1
        krow = QHBoxLayout()
        self.signup = QPushButton(t("Sign up / get a key ↗"))
        self.signup.setObjectName("link")
        self.signup.clicked.connect(lambda: webbrowser.open(ENGINE_BY_KEY[self.engine.currentData()].get("signup", "")))
        save = QPushButton(t("Save key"))
        save.clicked.connect(self._save_keys)
        krow.addWidget(self.signup)
        krow.addStretch()
        krow.addWidget(save)
        kl.addLayout(krow, row, 0, 1, 2)
        trow = QHBoxLayout()
        self.test_btn = QPushButton(t("Test"))
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
        disp, dl = card(t("Display"))
        g = QGridLayout()
        g.addWidget(QLabel(t("App language")), 3, 0)
        self.ui_lang = QComboBox()
        for code, label in LANGS:
            self.ui_lang.addItem(label, code)
        self.ui_lang.setCurrentIndex(max(0, self.ui_lang.findData(lang())))
        self.ui_lang.currentIndexChanged.connect(lambda _: self.ui_lang_changed.emit(self.ui_lang.currentData()))
        g.addWidget(self.ui_lang, 3, 1, 1, 2)
        g.addWidget(QLabel(t("Text size")), 0, 0)
        self.font = QSlider(Qt.Orientation.Horizontal, minimum=60, maximum=130)
        self.font.valueChanged.connect(lambda v: (self.font_val.setText(f"{v}%"), self.font_scale_changed.emit(v / 100)))
        self.font_val = QLabel("")
        self.font_val.setFixedWidth(44)
        g.addWidget(self.font, 0, 1)
        g.addWidget(self.font_val, 0, 2)
        g.addWidget(QLabel(t("Game language")), 1, 0)
        self.src = QComboBox()
        for code, label in GAME_LANGS:
            self.src.addItem(t(label), code)
        self.src.currentIndexChanged.connect(lambda _: self.src_lang_changed.emit(self.src.currentData()))
        g.addWidget(self.src, 1, 1, 1, 2)
        g.addWidget(QLabel(t("Translate into")), 2, 0)
        self.target = QComboBox()
        self.target.addItem(t("Thai"), "th")
        self.target.addItem(t("English"), "en")
        self.target.currentIndexChanged.connect(lambda _: self.target_changed.emit(self.target.currentData()))
        g.addWidget(self.target, 2, 1, 1, 2)
        self.toasts = QCheckBox(t("Show short status messages on the game screen"))
        self.toasts.toggled.connect(self.toasts_changed.emit)
        self.english = QCheckBox(t("Switch the keyboard to English while in the game"))
        self.english.setToolTip(t("A Thai keyboard makes the game show its key hints as boxes □"))
        self.english.toggled.connect(self.english_keys_changed.emit)
        self.auto = QCheckBox(t("Translate menus automatically when the screen is still"))
        self.auto.setToolTip(t("When the whole screen stands still, e.g. a menu is open, all its text is translated.\n"
                               "This uses the translator more: mind the quota of a paid API."))
        self.auto.toggled.connect(self.auto_screen_changed.emit)
        dl.addLayout(g)
        dl.addWidget(self.auto)
        dl.addWidget(self.toasts)
        dl.addWidget(self.english)

        # hotkeys
        keys, kl2 = card(t("Keys while playing"))
        grid = QGridLayout()
        rows = [(peek_key, t("hold: show the original")), (hotkeys.get("toggle", ""), t("start / stop")),
                (hotkeys.get("screen", ""), t("translate the screen")),
                (hotkeys.get("regions", ""), t("set regions")), (hotkeys.get("history", ""), t("history"))]
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
        hist = QPushButton(t("Translation history"))
        hist.clicked.connect(self.show_history.emit)
        sc = QPushButton(t("Desktop icon"))
        sc.clicked.connect(self.make_shortcut.emit)
        q = QPushButton(t("Quit"))
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
             model_size: str = "auto", auto_label: str = "", src_lang: str = "ja", english_keys: bool = True):
        self.auto.blockSignals(True)
        self.auto.setChecked(auto_screen)
        self.auto.blockSignals(False)
        self.size.blockSignals(True)
        if auto_label:
            self.size.setItemText(0, t("Automatic — this computer: {label}", label=auto_label))
        self.size.setCurrentIndex(max(0, self.size.findData(model_size)))
        self.size.blockSignals(False)
        for w in (self.engine, self.font, self.target, self.toasts, self.src, self.english):
            w.blockSignals(True)
        self.english.setChecked(english_keys)
        self.src.setCurrentIndex(max(0, self.src.findData(src_lang)))
        self.engine.setCurrentIndex(max(0, self.engine.findData(engine)))
        self.font.setValue(int(round(font_scale * 100)))
        self.font_val.setText(f"{self.font.value()}%")
        self.target.setCurrentIndex(max(0, self.target.findData(target)))
        self.toasts.setChecked(toasts)
        for w in (self.engine, self.font, self.target, self.toasts, self.src, self.english):
            w.blockSignals(False)
        self._show_engine(engine)

    def set_running(self, running: bool, ready: bool, hint: str):
        running = running and ready  # "stop" only means something once it is actually translating
        self.big.setText(t("■  Stop translating") if running else t("▶  Start translating"))
        self.big.setObjectName("bigstop" if running else "big")
        repolish(self.big)
        self.big.setEnabled(ready or running)
        self.big_hint.setText(hint)

    def set_keys(self, masked: dict):
        for names in masked.values():
            for n, v in names.items():
                if n in self.key_edits and not self.key_edits[n].hasFocus():
                    self.key_edits[n].clear()
                    self.key_edits[n].setPlaceholderText(t("set ({value})", value=v) if v else t("paste the key here"))

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
        self.engine_desc.setText(t(info["desc"]))
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
