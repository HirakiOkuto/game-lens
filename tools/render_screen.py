"""Show what whole-screen translation (F9) would draw on a screenshot, without the game.

  .venv\\Scripts\\python.exe tools\\render_screen.py shot.png out.png [--engine hymt]

The game-lens server must be running (open game-lens first). The screenshot should be the game's client area.
Prints every OCR line with its translation; out.png is the screenshot with the overlay drawn on it.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
from PySide6.QtCore import QPoint  # noqa: E402
from PySide6.QtGui import QImage, QPainter  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from gamelens.api import LensClient  # noqa: E402
from gamelens.overlay import Overlay  # noqa: E402
from gamelens.profile import REMOTE, Profile, Settings  # noqa: E402
from gamelens.screen import lines_to_translate, paragraphs, requests, screen_items  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("out")
    ap.add_argument("--engine", default="")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    app = QApplication([])  # noqa: F841
    st = Settings.load()
    prof = Profile.load(st.resolve(st.profile))
    engine = a.engine or prof.engine
    api = LensClient(st.server, st.resolve(st.token_file))
    frame = cv2.imread(a.image)
    ocr = api.ocr_lines(frame, prof.src_lang, frame_h=frame.shape[0])
    lines = lines_to_translate(ocr["lines"], [], prof.src_lang)
    groups = paragraphs(lines)
    res = api.translate_lines(requests(groups, prof.src_lang), series=prof.series, target=prof.target,
                              src_lang=prof.src_lang, engine=engine, allow_cloud=engine in REMOTE, context=[])
    sent = {l["id"] for l in lines}
    print(f"OCR {ocr['ms']} ms, {len(ocr['lines'])} lines, {len(lines)} to translate in {len(groups)} blocks, by {res.get('engine')}")
    for l in ocr["lines"]:
        print(f"  {l['bbox']!s:24} {l['conf']:.2f} {l['text']!r}" + ("" if l["id"] in sent else "   (left as it is)"))
    print()
    for r in requests(groups, prof.src_lang):
        print(f"  {r['src']}\n    -> {res['lines'].get(r['id'], '')}")
    items = screen_items(frame, groups, res["lines"], prof.font_scale)
    h, w = frame.shape[:2]
    ov = Overlay(prof.font_family)
    ov.client, ov.dpr = (0, 0, w, h), 1.0
    ov.resize(w, h)
    ov.items["screen"] = items
    for it in items:
        it.born -= 10  # past the fade-in
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).convertToFormat(QImage.Format.Format_ARGB32)
    p = QPainter(img)
    ov.render(p, QPoint(0, 0))
    p.end()
    img.save(a.out)


if __name__ == "__main__":
    main()
