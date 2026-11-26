"""One dark theme for every game-lens window."""

ACCENT = "#4f8cff"
OK, WARN, BAD, IDLE = "#3ecf8e", "#f5b942", "#ff6b6b", "#6b7280"

QSS = f"""
QWidget {{ color: #e7e9ee; font-family: 'Leelawadee UI', 'Segoe UI'; font-size: 10.5pt; }}
QWidget#window, QDialog, QMenu {{ background: #15171c; }}
QLabel, QCheckBox {{ background: transparent; }}
QMenu {{ border: 1px solid #2a2e37; padding: 4px; }}
QMenu::item {{ padding: 6px 18px; border-radius: 6px; }}
QMenu::item:selected {{ background: {ACCENT}; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: #3a404c; border-radius: 4px; min-height: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QScrollBar:horizontal {{ height: 0; }}
QFrame#card {{ background: #1e2128; border: 1px solid #2a2e37; border-radius: 12px; }}
QLabel#title {{ font-size: 17pt; font-weight: 600; }}
QLabel#h2 {{ font-size: 11.5pt; font-weight: 600; color: #ffffff; }}
QLabel#muted {{ color: #9aa3b2; font-size: 9.5pt; }}
QLabel#hint {{ color: #9aa3b2; font-size: 9pt; }}
QPushButton {{ background: #2a2f3a; border: 1px solid #343a46; border-radius: 8px; padding: 6px 14px; min-height: 22px; }}
QPushButton:hover {{ background: #333a47; }}
QPushButton:disabled {{ color: #5d6472; background: #20232a; }}
QPushButton#primary {{ background: {ACCENT}; border: none; color: white; font-weight: 600; }}
QPushButton#primary:hover {{ background: #6a9dff; }}
QPushButton#big {{ background: {ACCENT}; border: none; color: white; font-size: 13pt; font-weight: 600; padding: 12px; border-radius: 10px; }}
QPushButton#big:hover {{ background: #6a9dff; }}
QPushButton#big:disabled {{ background: #26314a; color: #7f8aa3; }}
QPushButton#bigstop {{ background: #3a2f33; border: 1px solid #5a3a40; color: #ffb4b4; font-size: 13pt; font-weight: 600; padding: 12px; border-radius: 10px; }}
QPushButton#link {{ background: transparent; border: none; color: {ACCENT}; padding: 0; text-align: left; }}
QComboBox, QLineEdit {{ background: #262a33; border: 1px solid #343a46; border-radius: 8px; padding: 6px 8px; }}
QComboBox QAbstractItemView {{ background: #262a33; selection-background-color: {ACCENT}; }}
QListWidget {{ background: #1e2128; border: 1px solid #2a2e37; border-radius: 8px; }}
QListWidget::item {{ padding: 8px; }}
QListWidget::item:selected {{ background: {ACCENT}; color: white; border-radius: 6px; }}
QSlider::groove:horizontal {{ height: 6px; background: #2a2f3a; border-radius: 3px; }}
QSlider::handle:horizontal {{ background: {ACCENT}; width: 16px; margin: -6px 0; border-radius: 8px; }}
QCheckBox::indicator {{ width: 16px; height: 16px; }}
QCheckBox:disabled {{ color: #5d6470; }}
QCheckBox::indicator:disabled {{ background: #2a2e37; border: 1px solid #3a3f4a; border-radius: 3px; }}
QToolTip {{ background: #262a33; color: #e7e9ee; border: 1px solid #343a46; }}
"""

