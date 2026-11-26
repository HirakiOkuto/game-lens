"""A stand-in "game" for trying game-lens before a real game is installed: a window with a scene that keeps
moving and a semi-transparent dialogue box whose Japanese lines appear with a typewriter effect.
Run:  .venv\\Scripts\\python.exe tools\\fake_game.py      (Space / click = next line)"""
import math
import sys

from PySide6.QtCore import QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QLinearGradient, QPainter
from PySide6.QtWidgets import QApplication, QWidget

LINES = [
    ("ラフィー", "指揮官、おはよう……今日はいい天気だね。"),
    ("ラフィー", "一緒に港を散歩しない？ラフィー、眠いけど頑張る。"),
    ("ジャベリン", "指揮官！新しい任務が届きましたよ！早く確認しましょう！"),
    ("ラフィー", "……ちょっとだけ、休憩してもいい？"),
]


MENU = ["装備", "スキル", "編成", "ショップ", "設定"]
QUEST = ["メインクエスト", "港の倉庫で指揮官を探す", "報酬：ダイヤ 50個"]
POPUP = ["ステータス", "会心ダメージ　150.0%", "回復効果増加　0.0%", "火属性ダメージ増幅　0.0%"]


class FakeGame(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FakeGame - game-lens test")
        self.resize(1280, 720)
        self.i, self.shown, self.t = 0, 0, 0.0
        self.menu = "--menu" in sys.argv
        self.popup = False  # P toggles a pop-up over the menu (--popup N: opens it after N seconds)
        if "--popup" in sys.argv:
            QTimer.singleShot(int(float(sys.argv[sys.argv.index("--popup") + 1]) * 1000), self.toggle_popup)
        QTimer(self, interval=33, timeout=self.step).start()
        if "--auto" in sys.argv:  # advance by itself every N seconds (for unattended tests)
            secs = float(sys.argv[sys.argv.index("--auto") + 1])
            QTimer(self, interval=int(secs * 1000), timeout=self.next).start()

    def step(self):
        self.t += 0.033
        name, text = LINES[self.i]
        if self.shown < len(text) and int(self.t * 30) % 2 == 0:
            self.shown += 1  # ~15 characters a second
        self.update()

    def next(self):
        self.i = (self.i + 1) % len(LINES)
        self.shown = 0

    def keyPressEvent(self, e):
        if e.key() == Qt.Key.Key_Space:
            self.next()
        elif e.key() == Qt.Key.Key_M:
            self.menu = not self.menu
        elif e.key() == Qt.Key.Key_P:
            self.toggle_popup()

    def toggle_popup(self):
        self.popup = not self.popup
        self.update()

    def mousePressEvent(self, _):
        self.next()

    def paint_menu(self, p, w, h):
        """A still menu screen (--menu, or M to toggle): buttons on the left, a quest panel on the right."""
        p.fillRect(self.rect(), QColor(24, 30, 52))
        f = QFont("Yu Gothic UI")
        f.setPixelSize(int(h * 0.04))
        p.setFont(f)
        for i, t in enumerate(MENU):
            r = QRectF(w * 0.08, h * (0.16 + i * 0.13), w * 0.24, h * 0.09)
            p.fillRect(r, QColor(52, 64, 104))
            p.setPen(QColor("#f5f5f5"))
            p.drawText(r.adjusted(24, 0, 0, 0), Qt.AlignmentFlag.AlignVCenter, t)
        panel = QRectF(w * 0.5, h * 0.16, w * 0.42, h * 0.4)
        p.fillRect(panel, QColor(236, 232, 220))
        for i, t in enumerate(QUEST):
            f.setBold(i == 0)
            f.setPixelSize(int(h * (0.042 if i == 0 else 0.034)))
            p.setFont(f)
            p.setPen(QColor("#2a2a35"))
            p.drawText(QRectF(panel.x() + 28, panel.y() + 24 + i * h * 0.085, panel.width() - 40, h * 0.07),
                       Qt.AlignmentFlag.AlignVCenter, t)
        if self.popup:
            box = QRectF(w * 0.2, h * 0.3, w * 0.5, h * 0.45)
            p.fillRect(box, QColor(245, 245, 240))
            for i, t in enumerate(POPUP):
                f.setBold(i == 0)
                f.setPixelSize(int(h * 0.036))
                p.setFont(f)
                p.setPen(QColor("#30303a"))
                p.drawText(QRectF(box.x() + 30, box.y() + 20 + i * h * 0.09, box.width() - 50, h * 0.07),
                           Qt.AlignmentFlag.AlignVCenter, t)

    def paintEvent(self, _):
        p = QPainter(self)
        w, h = self.width(), self.height()
        if self.menu:
            self.paint_menu(p, w, h)
            p.end()
            return
        g = QLinearGradient(0, 0, 0, h)
        hue = (math.sin(self.t * 0.3) + 1) / 2
        g.setColorAt(0, QColor.fromHsvF(0.55 + 0.05 * hue, 0.5, 0.9))
        g.setColorAt(1, QColor.fromHsvF(0.33, 0.6, 0.5))
        p.fillRect(self.rect(), g)
        for k in range(6):  # drifting "scenery"
            x = (k * 230 + self.t * 40) % (w + 200) - 100
            p.fillRect(QRectF(x, h * 0.35 + 30 * math.sin(self.t + k), 120, 160), QColor(255, 255, 255, 60))
        box = QRectF(w * 0.12, h * 0.72, w * 0.76, h * 0.22)
        p.fillRect(box, QColor(15, 20, 45, 200))
        name, text = LINES[self.i]
        f = QFont("Yu Gothic UI")
        f.setPixelSize(int(h * 0.035))
        f.setBold(True)
        p.setFont(f)
        p.setPen(QColor("#ffd479"))
        p.drawText(QRectF(box.x() + 24, box.y() - h * 0.06, w * 0.3, h * 0.05), Qt.AlignmentFlag.AlignVCenter, name)
        f.setBold(False)
        f.setPixelSize(int(h * 0.038))
        p.setFont(f)
        p.setPen(QColor("#f5f5f5"))
        p.drawText(box.adjusted(28, 22, -28, -18), Qt.TextFlag.TextWordWrap, text[: self.shown])
        p.end()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    g = FakeGame()
    g.show()
    sys.exit(app.exec())
