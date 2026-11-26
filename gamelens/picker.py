"""Choosing the game window: from a list, or by just switching to the game for a moment."""
from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QPushButton, QVBoxLayout

from . import win32
from .i18n import t

SKIP_EXE = {"nvidia overlay.exe", "nvidia share.exe", "gamebar.exe", "gamebarftserver.exe", "explorer.exe", "python.exe", "pythonw.exe", "textinputhost.exe", "applicationframehost.exe",
            "systemsettings.exe", "shellexperiencehost.exe", "searchapp.exe", "startmenuexperiencehost.exe"}


class GamePicker(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(t("Choose the game window"))
        self.resize(560, 500)
        self.result_window: dict | None = None
        self.own = {int(parent.winId())} if parent else set()
        self.own.add(int(self.winId()))

        title = QLabel(t("Which window is your game?"))
        title.setObjectName("h2")
        auto = QPushButton(t("⚡ Detect — press, then switch to the game within 5 seconds"))
        auto.setObjectName("primary")
        auto.clicked.connect(self.start_auto)
        self.auto_label = QLabel("")
        self.auto_label.setObjectName("muted")
        self.search = QLineEdit(placeholderText=t("Search window titles…"))
        self.search.textChanged.connect(self.fill)
        self.list = QListWidget()
        self.list.itemDoubleClicked.connect(lambda _: self.accept_selected())
        ok = QPushButton(t("Use this window"))
        ok.setObjectName("primary")
        ok.clicked.connect(self.accept_selected)
        refresh = QPushButton(t("Refresh"))
        refresh.clicked.connect(self.fill)
        cancel = QPushButton(t("Cancel"))
        cancel.clicked.connect(self.reject)

        lay = QVBoxLayout(self)
        lay.setSpacing(10)
        lay.addWidget(title)
        lay.addWidget(auto)
        lay.addWidget(self.auto_label)
        or_label = QLabel(t("or choose from the list:"))
        or_label.setObjectName("muted")
        lay.addWidget(or_label)
        lay.addWidget(self.search)
        lay.addWidget(self.list, 1)
        row = QHBoxLayout()
        row.addWidget(refresh)
        row.addStretch()
        row.addWidget(cancel)
        row.addWidget(ok)
        lay.addLayout(row)

        self.countdown = 0
        self.timer = QTimer(self, interval=1000, timeout=self.tick)
        self.fill()

    def windows(self) -> list[dict]:
        out = []
        for w in win32.list_windows():
            if w["exe"].lower() in SKIP_EXE or w["hwnd"] in self.own:
                continue
            r = win32.client_rect(w["hwnd"])
            if not r or r[2] < 320 or r[3] < 200:
                continue  # games are big windows
            w["size"] = r[2:]
            out.append(w)
        return sorted(out, key=lambda w: -w["size"][0] * w["size"][1])

    def fill(self):
        q = self.search.text().lower()
        self.list.clear()
        for w in self.windows():
            if q and q not in w["title"].lower() and q not in w["exe"].lower():
                continue
            it = QListWidgetItem(f"{w['title']}\n{w['exe']}  •  {w['size'][0]}×{w['size'][1]}")
            it.setData(Qt.ItemDataRole.UserRole, w)
            self.list.addItem(it)
        if self.list.count():
            self.list.setCurrentRow(0)

    def accept_selected(self):
        it = self.list.currentItem()
        if it:
            self.result_window = it.data(Qt.ItemDataRole.UserRole)
            self.accept()

    def start_auto(self):
        self.countdown = 5
        self.auto_label.setText(t("Switch to the game window now… {n}", n=5))
        self.timer.start()

    def tick(self):
        self.countdown -= 1
        if self.countdown > 0:
            self.auto_label.setText(t("Switch to the game window now… {n}", n=self.countdown))
            return
        self.timer.stop()
        fg = win32.foreground()
        if not fg or fg in self.own:
            self.auto_label.setText(t("You did not switch to the game. Try again, or choose from the list"))
            return
        exe = win32.process_exe(fg)
        if exe.lower() in SKIP_EXE:
            self.auto_label.setText(t("That window is not a game ({exe}). Try again", exe=exe))
            return
        self.result_window = {"hwnd": fg, "title": win32.window_title(fg), "exe": exe}
        self.accept()
        self.raise_()
