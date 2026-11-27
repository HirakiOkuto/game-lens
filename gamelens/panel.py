"""The main window: a 3-step readiness checklist, one big start button, engine + key setup and display options.
Closing it only hides it (game-lens keeps running in the tray)."""
from __future__ import annotations

import webbrowser

from PySide6.QtCore import QMargins, Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QFont, QFontDatabase, QGuiApplication
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFontComboBox, QFrame, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
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
        self.state = state
        self.dot.setStyleSheet(f"color: {COLORS[state]}; font-size: 14pt;")
        self.detail.setText(detail)
        if action is None:
            self.button.hide()
        else:
            self.button.setText(action)
            self.button.setObjectName("primary" if primary else "")
            repolish(self.button)
            self.button.show()


def usable_font(family: str, target: str) -> str:
    """The font to draw in: the chosen one if it is on this computer with the letters of the language translated into,
    else the one that comes with game-lens, else Windows' own."""
    ws = QFontDatabase.WritingSystem.Thai if target == "th" else QFontDatabase.WritingSystem.Latin
    have = set(QFontDatabase.families(ws))
    return next((f for f in (family, "Bai Jamjuree", "Leelawadee UI", "Tahoma") if f in have), family)


# keys a shortcut can be: Qt key -> the name win32.parse_hotkey knows
KEY_NAMES = {**{getattr(Qt.Key, f"Key_F{i}"): f"F{i}" for i in range(1, 13)},
             **{getattr(Qt.Key, f"Key_{c}"): c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"},
             Qt.Key.Key_Insert: "Insert", Qt.Key.Key_Home: "Home", Qt.Key.Key_End: "End", Qt.Key.Key_PageUp: "PageUp",
             Qt.Key.Key_PageDown: "PageDown", Qt.Key.Key_Pause: "Pause", Qt.Key.Key_Space: "Space", Qt.Key.Key_Tab: "Tab"}
ALONE_OK = {"F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10", "F11", "F12", "Insert", "Home", "End", "PageUp",
            "PageDown", "Pause"}


class KeyButton(QPushButton):
    """Shows a shortcut; clicked, it takes the next key press (with Ctrl / Alt / Shift) as the new one.
    plain: one key without modifiers only (the key held to see the original is read as one key)."""

    changed = Signal(str, str)  # name, new key
    capturing = Signal(bool)

    def __init__(self, name: str, spec: str, plain: bool = False):
        super().__init__(spec)
        self.name, self.spec, self.plain, self.taking = name, spec, plain, False
        self.setToolTip(t("Click, then press the new key"))
        self.clicked.connect(self._start)

    def set_spec(self, spec: str):
        self.spec = spec
        if not self.taking:
            self.setText(spec)

    def _start(self):
        if self.taking:
            return
        self.taking = True
        self.setText(t("Press a key… (Esc: cancel)"))
        self.grabKeyboard()
        self.capturing.emit(True)

    def _stop(self, new: str = ""):
        self.taking = False
        self.releaseKeyboard()
        self.setText(new or self.spec)
        if new and new != self.spec:
            self.spec = new
            self.changed.emit(self.name, new)
        self.capturing.emit(False)

    def keyPressEvent(self, e):
        if not self.taking:
            return super().keyPressEvent(e)
        k = e.key()
        if k == Qt.Key.Key_Escape:
            return self._stop()
        name = KEY_NAMES.get(k)
        if name is None:
            return  # a modifier on its own (wait for the key), or a key that cannot be a shortcut
        mods = e.modifiers()
        parts = [] if self.plain else [m for m, f in (("Ctrl", Qt.KeyboardModifier.ControlModifier),
                                                      ("Alt", Qt.KeyboardModifier.AltModifier),
                                                      ("Shift", Qt.KeyboardModifier.ShiftModifier)) if mods & f]
        if not self.plain and not parts and name not in ALONE_OK:
            self.setText(t("A letter needs Ctrl or Alt (the game uses it)"))
            return
        self._stop("+".join(parts + [name]))

    def focusOutEvent(self, e):
        if self.taking:
            self._stop()
        super().focusOutEvent(e)


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
    font_family_changed = Signal(str)
    target_changed = Signal(str)
    src_lang_changed = Signal(str)
    toasts_changed = Signal(bool)
    english_keys_changed = Signal(bool)
    hotkey_changed = Signal(str, str)
    taking_key = Signal(bool)
    closed = Signal()  # the window was closed (game-lens keeps running in the tray)
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
        self.sub = sub = QLabel("")
        sub.setObjectName("muted")
        tcol = QVBoxLayout()
        tcol.setSpacing(0)
        tcol.addWidget(title)
        tcol.addWidget(sub)
        head.addLayout(tcol)
        head.addStretch()
        # the app's language, where one looks for it: top right
        self.ui_lang = QComboBox()
        for code, label in LANGS:
            self.ui_lang.addItem("🌐  " + label, code)
        self.ui_lang.setToolTip(t("App language"))
        self.ui_lang.setCurrentIndex(max(0, self.ui_lang.findData(lang())))
        self.ui_lang.currentIndexChanged.connect(lambda _: self.ui_lang_changed.emit(self.ui_lang.currentData()))
        head.addWidget(self.ui_lang, 0, Qt.AlignmentFlag.AlignTop)

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
        # once all three are green they fold into one line (they take room that is needed for nothing then)
        self.summary = QWidget()
        sml = QHBoxLayout(self.summary)
        sml.setContentsMargins(0, 2, 0, 2)
        dot = QLabel("●")
        dot.setFixedWidth(18)
        dot.setStyleSheet(f"color: {OK}; font-size: 14pt;")
        self.summary_text = QLabel("")
        self.summary_text.setObjectName("muted")
        self.summary_text.setWordWrap(True)
        self.more = QPushButton("")
        self.more.setObjectName("link")
        self.more.clicked.connect(self._toggle_steps)
        sml.addWidget(dot, 0, Qt.AlignmentFlag.AlignTop)
        sml.addWidget(self.summary_text, 1)
        sml.addWidget(self.more, 0, Qt.AlignmentFlag.AlignVCenter)
        sl.addWidget(self.summary)
        self.steps_open = False

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
        szv = QVBoxLayout(self.size_box)
        szv.setContentsMargins(0, 0, 0, 0)
        szv.setSpacing(2)
        szl = QHBoxLayout()
        szl.addWidget(QLabel(t("Model")))
        self.size = QComboBox()
        # short names in the list (it is narrow); what each needs is written under it
        for label, key in (("Automatic, by graphics card", "auto"), ("Index-Translate 9B", "index"),
                           ("Hy-MT2 7B", "7b"), ("Hy-MT2 1.8B", "1.8b")):
            self.size.addItem(t(label), key)
        self.size.currentIndexChanged.connect(lambda _: (self._size_note(), self.model_size_changed.emit(self.size.currentData())))
        szl.addWidget(self.size, 1)
        szv.addLayout(szl)
        self.size_note = QLabel("")
        self.size_note.setObjectName("muted")
        self.size_note.setWordWrap(True)
        szv.addWidget(self.size_note)
        self.auto_label = ""
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

        # languages
        langs, ll = card(t("Languages"))
        lg = QGridLayout()
        # display
        disp, dl = card(t("Display"))
        g = QGridLayout()
        g.addWidget(QLabel(t("Text size")), 0, 0)
        self.font = QSlider(Qt.Orientation.Horizontal, minimum=60, maximum=130)
        self.font.valueChanged.connect(lambda v: (self.font_val.setText(f"{v}%"), self._preview(),
                                                  self.font_scale_changed.emit(v / 100)))
        self.font_val = QLabel("")
        self.font_val.setFixedWidth(44)
        g.addWidget(self.font, 0, 1)
        g.addWidget(self.font_val, 0, 2)
        g.addWidget(QLabel(t("Font")), 1, 0)
        self.family = QFontComboBox()
        self.family.setToolTip(t("The font of the translations drawn over the game (only fonts that have the letters "
                                 "of the language translated into are listed)"))
        self.family.currentFontChanged.connect(lambda f: (self._preview(), self.font_note.hide(),
                                                          self.font_family_changed.emit(f.family())))
        g.addWidget(self.family, 1, 1, 1, 2)
        self.font_note = QLabel("")
        self.font_note.setObjectName("muted")
        self.font_note.setStyleSheet(f"color: {WARN};")
        self.font_note.setWordWrap(True)
        self.font_note.hide()
        g.addWidget(self.font_note, 3, 0, 1, 3)
        # what the translations look like in that font and size, without going to the game
        self.sample = QLabel("")
        self.sample.setWordWrap(True)
        self.sample.setMinimumHeight(44)
        g.addWidget(self.sample, 2, 0, 1, 3)
        lg.addWidget(QLabel(t("Game language")), 0, 0)
        self.src = QComboBox()
        for code, label in GAME_LANGS:
            self.src.addItem(t(label), code)
        self.src.currentIndexChanged.connect(lambda _: self.src_lang_changed.emit(self.src.currentData()))
        lg.addWidget(self.src, 0, 1)
        lg.addWidget(QLabel(t("Translate into")), 1, 0)
        self.target = QComboBox()
        self.target.addItem(t("Thai"), "th")
        self.target.addItem(t("English"), "en")
        self.target.currentIndexChanged.connect(lambda _: (self._fonts_for(self.target.currentData()),
                                                           self._for_target(self.target.currentData()),
                                                           self.target_changed.emit(self.target.currentData())))
        lg.addWidget(self.target, 1, 1)
        lg.setColumnStretch(1, 1)
        ll.addLayout(lg)
        self.toasts = QCheckBox(t("Show short status messages on the game screen"))
        self.toasts.toggled.connect(self.toasts_changed.emit)
        self.english = QCheckBox(t("Switch the keyboard to English while in the game"))
        self.english.setToolTip(t("A Thai keyboard makes the game show its key hints as boxes □"))
        self.english.toggled.connect(self.english_keys_changed.emit)
        self.english_note = QWidget()
        enl = QVBoxLayout(self.english_note)
        enl.setContentsMargins(26, 0, 0, 0)
        enl.setSpacing(2)
        note = QLabel(t("Not possible with this game (it runs as administrator): set it once in Windows instead — "
                        "nothing to do if done already."))
        note.setObjectName("muted")
        note.setWordWrap(True)
        go = QPushButton(t("Open the Windows keyboard settings ↗"))
        go.setObjectName("link")
        go.clicked.connect(lambda: QDesktopServices.openUrl(QUrl("ms-settings:typing")))
        go.setToolTip(t("Advanced keyboard settings → Override for default input method: English, and tick “Let me use "
                        "a different input method for each app window”. Then open the game again."))
        enl.addWidget(note)
        enl.addWidget(go)
        self.english_note.hide()
        self.auto = QCheckBox(t("Translate menus automatically when the screen is still"))
        self.auto.setToolTip(t("When the whole screen stands still, e.g. a menu is open, all its text is translated.\n"
                               "This uses the translator more: mind the quota of a paid API."))
        self.auto.toggled.connect(self.auto_screen_changed.emit)
        dl.addLayout(g)
        dl.addWidget(self.toasts)
        opts, ol = card(t("While playing"))
        ol.addWidget(self.auto)
        ol.addWidget(self.english)
        ol.addWidget(self.english_note)

        # hotkeys
        keys, kl2 = card(t("Keys while playing"))
        grid = QGridLayout()
        rows = [("peek", peek_key, t("hold: show the original")), ("toggle", hotkeys.get("toggle", ""), t("start / stop")),
                ("screen", hotkeys.get("screen", ""), t("translate the screen")),
                ("regions", hotkeys.get("regions", ""), t("set regions")), ("history", hotkeys.get("history", ""), t("history"))]
        self.key_buttons: dict[str, KeyButton] = {}
        for i, (name, k, what) in enumerate(rows):
            kk = KeyButton(name, k, plain=name == "peek")
            kk.changed.connect(self.hotkey_changed.emit)
            kk.capturing.connect(self.taking_key.emit)
            self.key_buttons[name] = kk
            w = QLabel(what)
            w.setObjectName("muted")
            grid.addWidget(kk, i // 2, (i % 2) * 2, Qt.AlignmentFlag.AlignLeft)
            grid.addWidget(w, i // 2, (i % 2) * 2 + 1)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)
        grid.setHorizontalSpacing(8)
        kl2.addLayout(grid)
        khint = QLabel(t("Click a key to change it"))
        khint.setObjectName("muted")
        kl2.addWidget(khint)
        self.keys_card = keys
        keys.hide()

        foot = QHBoxLayout()
        hist = QPushButton(t("Translation history"))
        hist.clicked.connect(self.show_history.emit)
        self.keys_btn = QPushButton(t("Keys"))
        self.keys_btn.setCheckable(True)
        self.keys_btn.toggled.connect(self.keys_card.setVisible)
        sc = QPushButton(t("Desktop icon"))
        sc.clicked.connect(self.make_shortcut.emit)
        q = QPushButton(t("Quit"))
        q.clicked.connect(self.quit_app.emit)
        foot.addWidget(hist)
        foot.addWidget(self.keys_btn)
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
        lay.addWidget(langs)
        lay.addWidget(disp)
        lay.addWidget(opts)
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
    def _fonts_for(self, target: str):
        """List only the fonts that can write the language translated into."""
        ws = QFontDatabase.WritingSystem.Thai if target == "th" else QFontDatabase.WritingSystem.Latin
        if self.family.writingSystem() != ws:
            self.family.blockSignals(True)
            keep = self.family.currentFont()
            self.family.setWritingSystem(ws)
            self.family.setCurrentFont(keep)
            self.family.blockSignals(False)

    def load(self, engine: str, font_scale: float, target: str, toasts: bool, auto_screen: bool = False,
             model_size: str = "auto", auto_label: str = "", src_lang: str = "ja", english_keys: bool = True,
             font_family: str = ""):
        self.auto.blockSignals(True)
        self.auto.setChecked(auto_screen)
        self.auto.blockSignals(False)
        self.size.blockSignals(True)
        self.auto_label = auto_label
        self.size.setCurrentIndex(max(0, self.size.findData(model_size)))
        self.size.blockSignals(False)
        self._size_note()
        for w in (self.engine, self.font, self.target, self.toasts, self.src, self.english):
            w.blockSignals(True)
        self.english.setChecked(english_keys)
        self.src.setCurrentIndex(max(0, self.src.findData(src_lang)))
        self.engine.setCurrentIndex(max(0, self.engine.findData(engine)))
        self.font.setValue(int(round(font_scale * 100)))
        self.font_val.setText(f"{self.font.value()}%")
        self.target.setCurrentIndex(max(0, self.target.findData(target)))
        self._fonts_for(target)
        self._for_target(target)
        if font_family:
            self.family.blockSignals(True)
            self.family.setCurrentFont(QFont(font_family))
            self.family.blockSignals(False)
            # not on this computer (or without the letters of the language): the list cannot show it; say so
            missing = font_family not in QFontDatabase.families(self.family.writingSystem())
            self.font_note.setText(t("The font {name} is not on this computer (or has no letters of this language): "
                                     "{used} is used instead. Choose one from the list.", name=font_family,
                                     used=usable_font(font_family, target)) if missing else "")
            self.font_note.setVisible(missing)
            if missing:  # the list (and the sample) show the font that is used
                self.family.blockSignals(True)
                self.family.setCurrentFont(QFont(usable_font(font_family, target)))
                self.family.blockSignals(False)
        self.toasts.setChecked(toasts)
        for w in (self.engine, self.font, self.target, self.toasts, self.src, self.english):
            w.blockSignals(False)
        self._show_engine(engine)
        self._preview()

    def set_running(self, running: bool, ready: bool, hint: str):
        running = running and ready  # "stop" only means something once it is actually translating
        self.big.setText(t("■  Stop translating") if running else t("▶  Start translating"))
        self.big.setObjectName("bigstop" if running else "big")
        repolish(self.big)
        self.big.setEnabled(ready or running)
        self.big_hint.setText(hint)
        self._fold_steps()

    def set_keys(self, masked: dict):
        for names in masked.values():
            for n, v in names.items():
                if n in self.key_edits and not self.key_edits[n].hasFocus():
                    self.key_edits[n].clear()
                    self.key_edits[n].setPlaceholderText(t("set ({value})", value=v) if v else t("paste the key here"))

    def set_hotkeys(self, hotkeys: dict, peek_key: str):
        for name, b in self.key_buttons.items():
            b.set_spec(peek_key if name == "peek" else hotkeys.get(name, ""))

    def set_english_available(self, ok: bool, checked: bool):
        """The keyboard switch greys out where Windows does not let it work (a game running as administrator)."""
        self.english.blockSignals(True)
        self.english.setEnabled(ok)
        self.english.setChecked(checked and ok)
        self.english.setToolTip(t("A Thai keyboard makes the game show its key hints as boxes □"))
        self.english_note.setVisible(not ok)
        self.english.blockSignals(False)

    def set_engine_status(self, text: str, state: str = "idle"):
        self.engine_status.setText(text)
        self.engine_status.setStyleSheet(f"color: {COLORS[state] if state != 'idle' else '#9aa3b2'};")

    # --- internal ---------------------------------------------------------------------------------------------
    SAMPLES = {"th": "ผู้บัญชาการคะ ไปเดินเล่นกันไหม? HP +120",
               "en": "Commander, shall we go for a walk? HP +120"}

    SIZE_NOTES = {"index": "Most natural • ~5.5 GB graphics memory • 5.8 GB download",
                  "7b": "~5.5 GB graphics memory • 4.6 GB download",
                  "1.8b": "Fast • ~2 GB graphics memory, or the CPU • lower quality • 1.1 GB download"}

    def _size_note(self):
        key = self.size.currentData()
        if key == "auto":
            self.size_note.setText(t("This computer: {label}", label=self.auto_label) if self.auto_label else "")
        else:
            self.size_note.setText(t(self.SIZE_NOTES[key]))
        self.size_note.setVisible(bool(self.size_note.text()))

    def _for_target(self, target: str):
        name = self.target.itemText(max(0, self.target.findData(target)))
        self.sub.setText(t("Game text translated into {language}, drawn over the game", language=name))
        self._preview()

    def _preview(self):
        """A line in the chosen font and size, as it would be drawn over a game's dark text box."""
        if not hasattr(self, "sample"):
            return
        px = max(10, round(24 * self.font.value() / 100))
        fam = self.family.currentFont().family().replace("'", "")
        self.sample.setStyleSheet(f"background: #2c313b; border-radius: 8px; padding: 6px 10px; color: #f4f4f4; "
                                  f"font-family: '{fam}'; font-size: {px}px;")
        self.sample.setText(self.SAMPLES.get(self.target.currentData(), self.SAMPLES["en"]))

    def _fold_steps(self):
        steps = (self.s_server, self.s_game, self.s_regions)
        ready = all(getattr(s, "state", "") == "ok" for s in steps)
        folded = ready and not self.steps_open
        for s in steps:
            s.setVisible(not folded)
        self.summary.setVisible(ready)
        if ready:
            regions = self.s_regions.detail.text().split(":")[0]
            self.summary_text.setText(f"{self.s_server.detail.text()}\n{self.s_game.detail.text()} • {regions}"
                                      if folded else "")
            self.summary_text.setVisible(folded)
            self.more.setText(t("Details") if folded else t("Hide details"))

    def _toggle_steps(self):
        self.steps_open = not self.steps_open
        self._fold_steps()

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
        if QGuiApplication.instance().isSavingSession():  # (Windows signing out or shutting down: let it)
            e.accept()
            return
        e.ignore()
        self.hide()
        self.closed.emit()
