"""Everything translated in a session: a JSONL log on disk and a window to scroll back through it."""
from __future__ import annotations

import html
import json
import time
from pathlib import Path

from PySide6.QtCore import Qt, QUrl, Signal
from PySide6.QtWidgets import (QComboBox, QDialog, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
                               QTextBrowser, QVBoxLayout, QWidget)

from .i18n import t


class HistoryLog:
    def __init__(self, folder: Path, series: str):
        folder.mkdir(parents=True, exist_ok=True)
        self.series = series
        self.path = folder / f"{series}-{time.strftime('%Y%m%d')}.jsonl"
        self.entries: list[dict] = []
        self.seen: set = set()  # menu text read again and again (the same screen) is logged once
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                try:
                    self.entries.append(json.loads(line))
                    if self.entries[-1].get("kind") == "ui":
                        self.seen.add((self.entries[-1]["src"], self.entries[-1]["tl"]))
                except ValueError:
                    pass

    def add(self, kind: str, src: str, tl: str, engine: str) -> dict | None:
        if kind == "ui":
            if (src, tl) in self.seen:
                return None
            self.seen.add((src, tl))
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


# what a correction is: a term, or a character's name (with the character's gender for ครับ / ค่ะ); or a fix of
# what the text reader keeps misreading, or of a word the translations keep getting wrong
FIX_KINDS = (("term", "A word, a thing or a whole line"), ("female", "A character's name (female)"),
             ("male", "A character's name (male)"), ("neutral", "A character's name (gender not given)"),
             ("ocr", "A misreading of the original (fixed before translating)"),
             ("tlfix", "A wrong word in the translation (replaced in every translation)"))
# per kind: the two fields' labels, the hint, and which text each field starts with
FIX_FORMS = {
    "ocr": ("Read as", "Should read", "Keep only the misread part of the original and write it as it should read. "
            "It is put right before translating, wherever it is read so.", "src", "src"),
    "tlfix": ("In the translation", "Replace with", "Keep only the wrong word of the translation and write the right "
              "one. It is replaced in every translation, those made before too.", "tl", "tl"),
}
FIX_FORM = ("Original", "Translate as", "Keep only the word or name to correct (e.g. a name in the line), or the whole "
            "line to fix the whole line. It is translated this way from now on, also in other lines.", "src", "tl")


class FixDialog(QDialog):
    """Correct a translation: the Japanese word (or the whole line) and how it is to be translated from now on."""

    def __init__(self, src: str, tl: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(t("Correct a translation"))
        self.setMinimumWidth(480)
        orig = QLabel(f"<span style='color:#8b93a3'>{html.escape(src)}</span><br><b>{html.escape(tl)}</b>")
        orig.setWordWrap(True)
        self.texts = {"src": src, "tl": tl}
        self.hint = QLabel()
        self.hint.setObjectName("muted")
        self.hint.setWordWrap(True)
        self.src = QLineEdit(src)
        self.tl = QLineEdit(tl)
        self.src_label, self.tl_label = QLabel(), QLabel()
        self.kind = QComboBox()
        for key, label in FIX_KINDS:
            self.kind.addItem(t(label), key)
        self.form = None
        self.kind.currentIndexChanged.connect(lambda _: self._show_form())
        g = QGridLayout()
        g.addWidget(QLabel(t("What it is")), 0, 0)
        g.addWidget(self.kind, 0, 1)
        g.addWidget(self.src_label, 1, 0)
        g.addWidget(self.src, 1, 1)
        g.addWidget(self.tl_label, 2, 0)
        g.addWidget(self.tl, 2, 1)
        cancel = QPushButton(t("Cancel"))
        cancel.clicked.connect(self.reject)
        self.save = QPushButton(t("Save"))
        self.save.setObjectName("primary")
        self.save.setDefault(True)
        self.save.clicked.connect(self.accept)
        for e in (self.src, self.tl):
            e.textChanged.connect(lambda _: self.save.setEnabled(bool(self.src.text().strip() and self.tl.text().strip())))
        row = QHBoxLayout()
        row.addStretch()
        row.addWidget(cancel)
        row.addWidget(self.save)
        lay = QVBoxLayout(self)
        lay.addWidget(orig)
        lay.addWidget(self.hint)
        lay.addLayout(g)
        lay.addLayout(row)
        self._show_form()

    def _show_form(self):
        form = FIX_FORMS.get(self.kind.currentData(), FIX_FORM)
        if form is self.form:
            return  # (one name kind for another: what was typed stays)
        self.form = form
        a, b, hint, start_a, start_b = form
        self.src_label.setText(t(a))
        self.tl_label.setText(t(b))
        self.hint.setText(t(hint))
        self.src.setText(self.texts[start_a])
        self.tl.setText(self.texts[start_b])
        self.tl.setFocus()
        self.tl.selectAll()

    def value(self) -> tuple[str, str, str]:
        return self.src.text().strip(), self.tl.text().strip(), self.kind.currentData()


class HistoryWindow(QWidget):
    build_glossary = Signal()
    fix = Signal(str, str, str)  # source, translation, kind (FIX_KINDS)

    def __init__(self, log: HistoryLog):
        super().__init__(None)
        self.log = log
        self.setWindowTitle(t("game-lens — translation history"))
        self.setObjectName("window")
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)
        self.resize(560, 640)
        self.view = QTextBrowser()
        self.view.setOpenLinks(False)
        self.view.anchorClicked.connect(self._fix_clicked)
        self.shown: list[dict] = []
        tip = QLabel(t("✎ next to a translation: correct it (it is translated that way from now on)"))
        tip.setObjectName("muted")
        tip.setWordWrap(True)
        self.view.setStyleSheet("QTextBrowser { background: #1e2128; border: 1px solid #2a2e37; border-radius: 8px;"
                                " padding: 6px; font-family: 'Leelawadee UI', 'Yu Gothic UI'; font-size: 14px; }")
        btn = QPushButton(t("Build / update the character-name glossary from the history"))
        btn.clicked.connect(self.build_glossary.emit)
        row = QHBoxLayout()
        row.addWidget(btn)
        lay = QVBoxLayout(self)
        lay.addWidget(tip)
        lay.addWidget(self.view)
        lay.addLayout(row)
        for e in log.entries[-300:]:
            self.append(e)

    def append(self, e: dict | None):
        if not e:
            return
        self.shown.append(e)
        tag = {"name": "👤 ", "ui": "▫ "}.get(e["kind"], "")
        self.view.append(
            f"<p style='margin:6px 0'><span style='color:#8b93a3'>{tag}{html.escape(e['src'])}</span><br>"
            f"<b>{html.escape(e['tl'])}</b> &nbsp;<a href='fix:{len(self.shown) - 1}' "
            f"style='color:#6a9dff; text-decoration:none'>✎</a></p>"
        )
        sb = self.view.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _fix_clicked(self, url: QUrl):
        s = url.toString()
        if not s.startswith("fix:"):
            return
        e = self.shown[int(s[4:])]
        d = FixDialog(e["src"], e["tl"], self)
        if d.exec() == QDialog.DialogCode.Accepted:
            self.fix.emit(*d.value())
