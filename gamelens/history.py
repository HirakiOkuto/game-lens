"""Everything translated in a session: a JSONL log on disk and a window to scroll back through it."""
from __future__ import annotations

import html
import json
import time
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QPushButton, QTextBrowser, QVBoxLayout, QWidget

from .i18n import t


class HistoryLog:
    def __init__(self, folder: Path, series: str):
        folder.mkdir(parents=True, exist_ok=True)
        self.path = folder / f"{series}-{time.strftime('%Y%m%d')}.jsonl"
        self.entries: list[dict] = []
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                try:
                    self.entries.append(json.loads(line))
                except ValueError:
                    pass

    def add(self, kind: str, src: str, tl: str, engine: str) -> dict:
        e = {"t": time.time(), "kind": kind, "src": src, "tl": tl, "engine": engine}
        self.entries.append(e)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
        return e

    def source_text(self, limit_chars: int = 20000) -> str:
        """Unique source lines, newest last, for glossary extraction."""
        seen, out = set(), []
        for e in self.entries:
            if e["src"] not in seen:
                seen.add(e["src"])
                out.append(e["src"])
        text = "\n".join(out)
        return text[-limit_chars:]


class HistoryWindow(QWidget):
    build_glossary = Signal()

    def __init__(self, log: HistoryLog):
        super().__init__(None)
        self.log = log
        self.setWindowTitle(t("game-lens — translation history"))
        self.setObjectName("window")
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)
        self.resize(560, 640)
        self.view = QTextBrowser()
        self.view.setStyleSheet("QTextBrowser { background: #1e2128; border: 1px solid #2a2e37; border-radius: 8px;"
                                " padding: 6px; font-family: 'Leelawadee UI', 'Yu Gothic UI'; font-size: 14px; }")
        btn = QPushButton(t("Build / update the character-name glossary from the history"))
        btn.clicked.connect(self.build_glossary.emit)
        row = QHBoxLayout()
        row.addWidget(btn)
        lay = QVBoxLayout(self)
        lay.addWidget(self.view)
        lay.addLayout(row)
        for e in log.entries[-300:]:
            self.append(e)

    def append(self, e: dict):
        tag = {"name": "👤 ", "ui": "▫ "}.get(e["kind"], "")
        self.view.append(
            f"<p style='margin:6px 0'><span style='color:#8b93a3'>{tag}{html.escape(e['src'])}</span><br>"
            f"<b>{html.escape(e['tl'])}</b></p>"
        )
        sb = self.view.verticalScrollBar()
        sb.setValue(sb.maximum())
