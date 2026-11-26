"""Play a recording of a game through game-lens's region pipeline (watcher -> OCR -> translation -> overlay),
without the game: which lines get caught, how late the translation appears, what it looks like, what goes wrong.
It runs the app's own code (watcher, screen); only capture and the overlay window are left out.

  .venv\\Scripts\\python.exe tools\\replay_video.py video.mp4 --start 26:36 --end 27:02 ^
      --region name:center:768,826,384,45 --region dialog:center:307,893,1305,135 --out out_dir [--save-video]

Regions are kind:align:x,y,w,h in the video's own pixels. The game-lens server must be running (open game-lens).
Time is the video's: a translation shows up as late after "text is still" as OCR + translation really took.
--segments FILE plays many sections ("12:30 - 13:05" a line). --no-pictures skips the before/after pictures.
Writes out_dir/events.txt, events.json and the pictures; every picture whose line has a problem is also in
out_dir/problems. Problems checked: original text left uncovered, untranslated characters, text that had to
shrink or runs out of its room, and text that stood on screen without a translation.
"""
import argparse
import json
import re
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
from gamelens.screen import SCRIPT, TextSize, begins, extend_item, ocr_region, refresh_patch, remember, translate_region  # noqa: E402
from gamelens.watcher import TextFollower, goes_on, has_text, similar, stroke_map  # noqa: E402

TICK = 0.05  # the app looks at the game 20 times a second


def seconds(spec: str) -> float:
    return sum(float(p) * 60 ** i for i, p in enumerate(reversed(spec.split(":"))))


def clock(t: float) -> str:
    return f"{int(t) // 60}:{t % 60:05.2f}"


def compose(frame: np.ndarray, ov: Overlay) -> np.ndarray:
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).convertToFormat(QImage.Format.Format_ARGB32)
    p = QPainter(img)
    ov.render(p, QPoint(0, 0))
    p.end()
    buf = np.frombuffer(img.constBits(), np.uint8).reshape(h, img.bytesPerLine())[:, : w * 4].reshape(h, w, 4)
    return buf[:, :, :3].copy()


class Replay:
    def __init__(self, a, prof, api, out: Path):
        self.a, self.prof, self.api, self.out = a, prof, api, out
        self.regions = []
        for i, spec in enumerate(a.region):
            kind, align, rect = spec.split(":")
            self.regions.append({"id": f"r{i}", "kind": kind, "align": align, "px": tuple(int(v) for v in rect.split(","))})
        self.cap = cv2.VideoCapture(a.video)
        self.W, self.H = int(self.cap.get(3)), int(self.cap.get(4))
        self.size = None
        if a.size:  # as if the game ran in a smaller (or bigger) window; regions stay in the video's pixels
            self.size = tuple(int(v) for v in a.size.lower().split("x"))
            fx, fy = self.size[0] / self.W, self.size[1] / self.H
            for r in self.regions:
                x, y, w, h = r["px"]
                r["px"] = (round(x * fx), round(y * fy), round(w * fx), round(h * fy))
            self.W, self.H = self.size
        self.ov = Overlay(prof.font_family)
        self.ov.client, self.ov.dpr = (0, 0, self.W, self.H), 1.0
        self.ov.resize(self.W, self.H)
        self.script = SCRIPT.get(prof.src_lang.split("-")[0], re.compile(r"[^\W\d_]"))
        self.events: list[dict] = []
        self.refresh_ms: list[float] = []
        self.reads = 0
        self.times = [float(v) for v in a.times.split(",")] if a.times else None
        self.log = open(out / "events.txt", "w", encoding="utf-8")
        self.writer = None
        if a.save_video:
            self.writer = cv2.VideoWriter(str(out / "replay.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 1 / TICK, (self.W // 2, self.H // 2))

    def say(self, msg):
        print(msg, flush=True)
        self.log.write(msg + "\n")

    def play(self, t0: float, t1: float):
        prof, api = self.prof, self.api
        st = {r["id"]: {"w": TextFollower(), "reading": None, "pending": None, "older": None, "shown": None, "read": None,
                        "sizes": TextSize(), "bare_since": None, "bare_told": False} for r in self.regions}
        self.ov.items.clear()
        context: list[dict] = []
        shots = []
        self.cap.set(cv2.CAP_PROP_POS_MSEC, t0 * 1000)
        next_tick = t0
        while True:
            if not self.cap.grab():
                break
            t = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            if t > t1:
                break
            if t < next_tick:
                continue
            next_tick += TICK
            ok, frame = self.cap.retrieve()
            if not ok:
                break
            if self.size:
                frame = cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA)
            for r in self.regions:
                s = st[r["id"]]
                x, y, w, h = r["px"]
                crop = frame[y : y + h, x : x + w]
                w_ = s["w"]
                ev = w_.feed(stroke_map(crop, self.a.ink), t)
                if ev == "hide":
                    self.missed(s, r, t)
                    self.ov.items.pop(r["id"], None)
                    self.gone(s, t)
                elif ev == "read":
                    wall = time.perf_counter()
                    got = ocr_region(api, crop.copy(), r["kind"], r["align"], prof, self.H)
                    s["reading"] = {"ready": t + (self.times[0] if self.times else time.perf_counter() - wall), "got": got, "crop": crop.copy()}
                    self.reads += 1
                rd = s["reading"]
                if rd and t >= rd["ready"]:
                    s["reading"] = None
                    got = rd["got"]
                    what = w_.on_read(got["text"] if got else "", t)
                    if got:
                        s["read"] = got
                    if s["pending"] and s["pending"]["seq"] != w_.seq:  # replaced, or it went on
                        s["older"] = s["pending"] if s["pending"]["line"] == w_.line else None
                        self.missed(s, r, t)
                    if what is None and got and w_.settled:
                        for it in self.ov.items.get(r["id"], ()):
                            extend_item(it, rd["crop"], got["lines"], self.a.ink, w_.persistent)
                    if what == "hide":
                        self.ov.items.pop(r["id"], None)
                        self.gone(s, t)
                    elif what == "translate":
                        prev = s["shown"]
                        if w_.corrected:
                            self.events.append({"ev": "corrected", "kind": r["kind"], "t": t, "src": got["text"], "was": prev["src"] if prev else ""})
                            self.say(f"[{clock(t)}] {r['kind']:6} READ ANEW: {prev['src'] if prev else ''} => {got['text']}")
                        if prev and similar(prev["src"], got["text"]) >= 0.92 and not w_.corrected:
                            self.events.append({"ev": "same", "kind": r["kind"], "t": t, "src": got["text"]})
                            if prev["items"] and r["id"] not in self.ov.items:
                                self.ov.items[r["id"]] = prev["items"]
                                s["on"] = (t, prev["src"], r["kind"])
                        else:
                            if prev and not w_.corrected and not goes_on(prev["src"], got["text"]):
                                self.ov.items.pop(r["id"], None)
                                self.gone(s, t)
                            wall = time.perf_counter()
                            res = translate_region(api, rd["crop"], r["px"], got, r["kind"], r["align"], self.a.ink, prof,
                                                   context[-6:], w_.persistent, s["sizes"], prof.engine in REMOTE)
                            took = time.perf_counter() - wall
                            if self.times:
                                took = 0.05 if r["kind"] == "name" else self.times[1] + self.times[2] * len(got["text"])
                            s["pending"] = dict(res, ready=t + took, seq=w_.seq, line=w_.line, took=took, stable_at=t, got=got)
                old = s["older"]
                if old and (old["line"] != w_.line or s["shown"] and s["shown"].get("seq", 0) > old["seq"]):
                    s["older"] = old = None
                if old and t >= old["ready"]:
                    s["older"] = None
                    if old["items"] and begins(old["got"], s["read"]):
                        s["shown"] = old
                        for it in old["items"]:
                            it.born -= 10
                        self.ov.items[r["id"]] = old["items"]
                        self.gone(s, t)
                        s["on"] = (t, old["src"], r["kind"])
                        self.events.append({"ev": "part", "kind": r["kind"], "t": t, "src": old["src"]})
                        self.say(f"[{clock(t)}] {r['kind']:6} PART SHOWN: {old['src']}")
                pend = s["pending"]
                if pend and t >= pend["ready"] and pend["seq"] == w_.seq and not w_.in_doubt:
                    s["pending"], s["shown"] = None, pend
                    for it in pend["items"]:
                        it.born -= 10
                    if pend["items"]:
                        self.ov.items[r["id"]] = pend["items"]
                    if pend["kind"] not in ("name", "choice", "labels") and pend["tl"]:
                        remember(context, pend["src"], pend["tl"])
                    self.gone(s, t)
                    s["on"] = (t, pend["src"], r["kind"])
                    shots.append((t + 0.3, r, pend))
                # text standing on screen without a translation (and none on its way)
                bare = has_text(s["w"].persistent) and r["id"] not in self.ov.items and not s["pending"]
                if not bare:
                    s["bare_since"], s["bare_told"] = None, False
                elif s["bare_since"] is None:
                    s["bare_since"] = t
                elif t - s["bare_since"] > 5 and not s["bare_told"]:  # longer than any line takes to be typed
                    s["bare_told"] = True
                    self.events.append({"ev": "bare", "kind": r["kind"], "t": t})
                    if self.a.pictures:
                        cv2.imwrite(str(self.out / "problems" / f"bare_{int(t)//60:03d}_{int(t%60):02d}_{r['kind']}.png"), crop)
                for it in self.ov.items.get(r["id"], []):
                    if getattr(it, "_live", False):  # as the app does on every tick
                        c0 = time.perf_counter()
                        refresh_patch(it, crop)
                        self.refresh_ms.append((time.perf_counter() - c0) * 1000)
            while shots and t >= shots[0][0]:
                _, r, pend = shots.pop(0)
                self.check(frame, t, r, pend)
            if self.writer is not None:
                pic = compose(frame, self.ov) if self.ov.items else frame
                self.writer.write(cv2.resize(pic, (self.W // 2, self.H // 2), interpolation=cv2.INTER_AREA))

    def gone(self, s, t):
        on, s["on"] = s.get("on"), None
        if on:
            self.events.append({"ev": "gone", "kind": on[2], "t": t, "src": on[1], "for": round(t - on[0], 2)})

    def missed(self, s, r, t):
        p, s["pending"] = s["pending"], None
        if p and p.get("tl"):
            self.events.append({"ev": "missed", "kind": r["kind"], "t": t, "src": p["src"], "had": round(t - p["stable_at"], 2),
                                "needed": round(p["took"], 2)})
            self.say(f"[{clock(t)}] {r['kind']:6} GONE OR GREW: {t - p['stable_at']:.1f}s after it was taken as complete, the "
                     f"translation needed {p['took']:.1f}s | {p['src']}")

    def check(self, frame, t, r, pend):
        """Look at a translation that has just been put on screen."""
        e = {"ev": "shown", "kind": r["kind"], "t": round(t, 2), "src": pend["src"], "tl": pend["tl"], "engine": pend["engine"],
             "cached": pend["cached"], "took": round(pend["took"], 2), "problems": []}
        live = pend["items"] and self.ov.items.get(r["id"]) is pend["items"]
        full = compose(frame, self.ov)
        if pend["items"]:
            it = pend["items"][0]
            x, y, w, h = r["px"]
            fit = getattr(it, "_fit", None)
            if fit:
                font, lines, lh = fit
                e.update(px=font.pixelSize(), want=round(it.max_px, 1), lines=len(lines), rows=len(it.rows or []),
                         row0=round(it.rows[0] - y, 1) if it.rows else None)
                if font.pixelSize() < 0.8 * it.max_px:
                    e["problems"].append("shrunk")
                if max(nw for _, nw in lines) > it.rect[2] + 2:
                    e["problems"].append("too-wide")
                if it.rows and len(lines) > len(it.rows) and it.rows[0] + (len(lines) - 0.5) * lh > y + h + 1.2 * lh:
                    e["problems"].append("too-deep")
            if live and it.patch is not None:  # what is left to read once the original is painted over?
                text, it.text = it.text, ""
                bare = compose(frame, self.ov)[y : y + h, x : x + w]
                it.text = text
                left = "".join(l["text"] for l in self.api.ocr_lines(bare, self.prof.src_lang, frame_h=self.H)["lines"] if l["conf"] > 0.75)
                left = "".join(self.script.findall(left))
                e["left"] = left
                if len(left) >= 3 and len(left) > 0.25 * len(self.script.findall(pend["src"])):
                    e["problems"].append("not-erased")
        elif pend["tl"]:
            e["problems"].append("no-item")
        if self.script.search(pend["tl"]):
            e["problems"].append("untranslated")
        if not pend["tl"]:
            e["problems"].append("empty")
        self.events.append(e)
        self.say(f"[{clock(t)}] {r['kind']:6} +{e['took']:.2f}s ({e['engine']}{', cache' if e['cached'] else ''})"
                 f" {e.get('px', '-')}px{' ' + ','.join(e['problems']) if e['problems'] else ''}\n          {e['src']}\n          -> {e['tl']}")
        if self.a.pictures or e["problems"]:
            xs = [q["px"][0] for q in self.regions] + [q["px"][0] + q["px"][2] for q in self.regions]
            ys = [q["px"][1] for q in self.regions] + [q["px"][1] + q["px"][3] for q in self.regions]
            x0, x1, y0, y1 = max(0, min(xs) - 20), min(self.W, max(xs) + 20), max(0, min(ys) - 30), min(self.H, max(ys) + 30)
            both = np.vstack([frame[y0:y1, x0:x1], full[y0:y1, x0:x1]])
            name = f"line_{int(t)//60:03d}_{int(t%60):02d}_{r['kind']}.png"
            if self.a.pictures:
                cv2.imwrite(str(self.out / name), both)
            if e["problems"]:
                cv2.imwrite(str(self.out / "problems" / f"{'+'.join(e['problems'])}_{name}"), both)

    def finish(self, length: float):
        if self.writer is not None:
            self.writer.release()
        n = lambda ev, **kw: sum(1 for e in self.events if e["ev"] == ev and all(e.get(k) == v for k, v in kw.items()))  # noqa: E731
        bad: dict[str, int] = {}
        for e in self.events:
            for p in e.get("problems", []):
                bad[p] = bad.get(p, 0) + 1
        self.say(f"{length:.0f} s of video: {n('shown', kind='dialog')} dialogue and {n('shown', kind='name')} name translations shown, "
                 f"{n('missed')} gone (or typed on) before the translation was ready, {n('same')} re-reads of the same text, "
                 f"{n('bare')} times text stood untranslated, {self.reads / max(1, length):.2f} readings a second; problems: {bad or 'none'}")
        if self.refresh_ms:
            self.say(f"re-covering text over the scene: {np.median(self.refresh_ms):.1f} ms a tick (95%: {np.percentile(self.refresh_ms, 95):.1f})")
        (self.out / "events.json").write_text(json.dumps(self.events, ensure_ascii=False, indent=0), encoding="utf-8")
        self.log.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--start", default="0")
    ap.add_argument("--end", default="")
    ap.add_argument("--segments", default="", help="file with 'm:ss - m:ss' lines to play instead of --start/--end")
    ap.add_argument("--part", default="0/1", help="i/n: play every n-th segment only (to run several at once)")
    ap.add_argument("--region", action="append", required=True, help="kind:align:x,y,w,h (video pixels)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--engine", default="")
    ap.add_argument("--size", default="", help="play as if the game's window were this big, e.g. 1280x720")
    ap.add_argument("--ink", default="light")
    ap.add_argument("--times", help="pretend reading took this long and translating that long: READ,BASE,PER_CHARACTER "
                    "(seconds), e.g. 0.25,0.35,0.028; results then do not depend on what else the computer is doing")
    ap.add_argument("--no-pictures", dest="pictures", action="store_false")
    ap.add_argument("--save-video", action="store_true", help="also write the clip with the overlay drawn in")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    app = QApplication([])  # noqa: F841
    st = Settings.load()
    prof = Profile.load(st.resolve(st.profile))
    prof.engine = a.engine or prof.engine
    out = Path(a.out)
    (out / "problems").mkdir(parents=True, exist_ok=True)
    for stale in [*out.glob("line_*.png"), *(out / "problems").glob("*.png")]:  # would pass for this run's
        stale.unlink()
    rp = Replay(a, prof, LensClient(st.server, st.resolve(st.token_file)), out)
    if a.segments:
        spans = [(seconds(m[1]), seconds(m[2])) for line in open(a.segments, encoding="utf-8")
                 if (m := re.match(r"\s*([\d:.]+)\s*-\s*([\d:.]+)", line))]
        i, n = (int(v) for v in a.part.split("/"))
        spans = spans[i::n]
    else:
        spans = [(seconds(a.start), seconds(a.end) if a.end else rp.cap.get(cv2.CAP_PROP_FRAME_COUNT) / rp.cap.get(cv2.CAP_PROP_FPS))]
    for t0, t1 in spans:
        if len(spans) > 1:
            rp.say(f"=== {clock(t0)} - {clock(t1)}")
        rp.play(t0, t1)
    rp.finish(sum(b - x for x, b in spans))


if __name__ == "__main__":
    main()
