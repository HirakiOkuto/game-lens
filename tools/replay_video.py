"""Play a recording of a game through game-lens's region pipeline (watcher -> OCR -> translation -> overlay),
without the game: to see which lines get caught, how late the translation appears and what it looks like.

  .venv\\Scripts\\python.exe tools\\replay_video.py video.mp4 --start 26:36 --end 27:02 ^
      --region name:center:760,825,400,50 --region dialog:center:560,890,800,95 --out out_dir [--save-video]

Regions are kind:align:x,y,w,h in the video's own pixels. The game-lens server must be running (open game-lens).
Time is the video's: a translation shows up as late after "text is still" as OCR + translation really took.
Writes out_dir/events.txt, a before/after picture for every translated line, and with --save-video the whole clip
with the overlay drawn in (out_dir/replay.mp4).
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PySide6.QtCore import QPoint  # noqa: E402
from PySide6.QtGui import QImage, QPainter  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from gamelens.api import LensClient  # noqa: E402
from gamelens.overlay import Overlay  # noqa: E402
from gamelens.profile import REMOTE, Profile, Settings  # noqa: E402
from gamelens.screen import refresh_patch, region_item  # noqa: E402
from gamelens.watcher import RegionWatcher, similar, stroke_map  # noqa: E402

TICK = 0.05  # the app looks at the game 20 times a second


def seconds(spec: str) -> float:
    parts = [float(p) for p in spec.split(":")]
    return sum(v * 60 ** i for i, v in enumerate(reversed(parts)))


def compose(frame: np.ndarray, ov: Overlay) -> np.ndarray:
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).convertToFormat(QImage.Format.Format_ARGB32)
    p = QPainter(img)
    ov.render(p, QPoint(0, 0))
    p.end()
    buf = np.frombuffer(img.constBits(), np.uint8).reshape(h, img.bytesPerLine())[:, : w * 4].reshape(h, w, 4)
    return buf[:, :, :3].copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--start", default="0")
    ap.add_argument("--end", default="")
    ap.add_argument("--region", action="append", required=True, help="kind:align:x,y,w,h (video pixels)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--engine", default="")
    ap.add_argument("--ink", default="light")
    ap.add_argument("--save-video", action="store_true", help="also write the clip with the overlay drawn in")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    app = QApplication([])  # noqa: F841
    st = Settings.load()
    prof = Profile.load(st.resolve(st.profile))
    engine = a.engine or prof.engine
    api = LensClient(st.server, st.resolve(st.token_file))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    regions = []
    for i, spec in enumerate(a.region):
        kind, align, rect = spec.split(":")
        regions.append({"id": f"r{i}", "kind": kind, "align": align, "px": tuple(int(v) for v in rect.split(","))})
    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 60
    t0 = seconds(a.start)
    t1 = seconds(a.end) if a.end else cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    cap.set(cv2.CAP_PROP_POS_MSEC, t0 * 1000)
    W, H = int(cap.get(3)), int(cap.get(4))
    ov = Overlay(prof.font_family)
    ov.client, ov.dpr = (0, 0, W, H), 1.0
    ov.resize(W, H)
    writer = None
    if a.save_video:
        writer = cv2.VideoWriter(str(out / "replay.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 1 / TICK, (W // 2, H // 2))

    state = {r["id"]: {"w": RegionWatcher(prof.stable_ms, prof.max_wait_ms), "gen": 0, "pending": None, "shown": None,
                       "changed_at": None} for r in regions}
    context: list[dict] = []
    stats = {"read": 0, "shown": 0, "too_late": 0, "same": 0}
    events, shots = [], []
    log = open(out / "events.txt", "w", encoding="utf-8")

    def say(msg):
        print(msg)
        log.write(msg + "\n")

    next_tick, n_frames = t0, 0
    while True:
        ok, frame = cap.read()
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        if not ok or t > t1:
            break
        if t < next_tick:
            continue
        next_tick += TICK
        n_frames += 1
        for r in regions:
            s = state[r["id"]]
            x, y, w, h = r["px"]
            crop = frame[y : y + h, x : x + w]
            ev = s["w"].feed(stroke_map(crop, a.ink), t)
            if ev == "changed":
                s["gen"] += 1
                if s["pending"]:  # the line was replaced before its translation was ready
                    stats["too_late"] += 1
                    say(f"[{int(t)//60}:{t%60:05.2f}] {r['kind']:6} MISSED: gone {t - s['pending']['stable_at']:.1f}s after it was "
                        f"complete, the translation needed {s['pending']['took']:.1f}s | {s['pending']['src']}")
                s["pending"] = None
                if s["shown"]:
                    events.append((r["kind"], "hidden", t, s["shown"]["src"]))
                    s["shown"]["until"] = t
                ov.items.pop(r["id"], None)
                s["changed_at"] = t
            elif ev == "stable":
                wall = time.perf_counter()
                ocr = api.ocr_lines(crop, prof.src_lang)
                text = ocr["text"].strip()
                lines = [l for l in ocr["lines"] if l["text"].strip()]
                if not text:
                    continue
                prev = s["shown"]
                if prev and similar(prev["src"], text) >= 0.92:
                    stats["same"] += 1
                    if prev["item"] is not None:
                        ov.items[r["id"]] = [prev["item"]]
                    continue
                stats["read"] += 1
                ctx = context[-6:] if r["kind"] != "name" else []
                res = api.translate_lines([{"id": "a", "src": text, "kind": r["kind"]}], series=prof.series,
                                          target=prof.target, src_lang=prof.src_lang, engine=engine,
                                          allow_cloud=engine in REMOTE, context=ctx)
                took = time.perf_counter() - wall
                tl = res["lines"].get("a", "")
                item = region_item(crop, r["px"], [l["bbox"] for l in lines], tl, r["kind"], r["align"], prof.font_scale,
                                   s["w"].persistent)
                s["pending"] = {"ready": t + took, "gen": s["gen"], "item": item, "src": text, "tl": tl, "took": took,
                                "stable_at": t, "since_change": t - (s["changed_at"] or t),
                                "cached": bool(res.get("cached")), "engine": res.get("engine")}
            pend = s["pending"]
            if pend and t >= pend["ready"] and pend["gen"] == s["gen"]:
                s["pending"] = None
                s["shown"] = pend
                stats["shown"] += 1
                if pend["item"] is not None:
                    pend["item"].born -= 10
                    ov.items[r["id"]] = [pend["item"]]
                    shots.append((t + 0.3, r, pend))
                if r["kind"] != "name" and pend["tl"]:
                    context.append({"src": pend["src"], "tl": pend["tl"]})
                say(f"[{int(t)//60}:{t%60:05.2f}] {r['kind']:6} typed {pend['since_change']:.1f}s, still at "
                    f"{pend['stable_at']%60:05.2f}, shown +{pend['took']:.2f}s ({pend['engine']}{', cache' if pend['cached'] else ''})"
                    f"\n          {pend['src']}\n          -> {pend['tl']}")
            # text over the moving scene: paint the covered glyphs again from this frame
            for it in ov.items.get(r["id"], []):
                if getattr(it, "_live", False):
                    refresh_patch(it, crop)
        while shots and t >= shots[0][0]:
            _, r, pend = shots.pop(0)
            if ov.items.get(r["id"]):
                xs = [q["px"][0] for q in regions] + [q["px"][0] + q["px"][2] for q in regions]
                ys = [q["px"][1] for q in regions] + [q["px"][1] + q["px"][3] for q in regions]
                x0, x1, y0, y1 = max(0, min(xs) - 60), min(W, max(xs) + 60), max(0, min(ys) - 40), min(H, max(ys) + 40)
                both = np.vstack([frame[y0:y1, x0:x1], compose(frame, ov)[y0:y1, x0:x1]])
                cv2.imwrite(str(out / f"line_{int(t)//60:03d}_{int(t%60):02d}_{r['kind']}.png"), both)
        if writer is not None:
            writer.write(cv2.resize(compose(frame, ov) if ov.items else frame, (W // 2, H // 2), interpolation=cv2.INTER_AREA))
    if writer is not None:
        writer.release()
    say(f"{t1 - t0:.0f} s of video: {stats['read']} texts read, {stats['shown']} translations shown, "
        f"{stats['too_late']} replaced before the translation was ready, {stats['same']} re-reads of the same text")
    log.close()


if __name__ == "__main__":
    main()
