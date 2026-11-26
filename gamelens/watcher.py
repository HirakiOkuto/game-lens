"""Deciding when a watched region has new, settled text worth reading.

Game dialogue is often drawn straight over the moving 3D scene, so "did the pixels change" is useless: they always
do. What tells text from scene is that glyph strokes are thin, bright (or dark) and stay exactly where they are.
`stroke_map` keeps thin strokes only; `TextFollower` keeps the strokes that survived the last few frames, has them
read while they change, and notices when the line it knows is replaced.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter, deque

import cv2
import numpy as np

THUMB_W = 320
STROKE_W = 480


def ink_map(region_bgr: np.ndarray, ink: str = "light") -> np.ndarray:
    """Coarse picture of a whole frame, for "has the screen changed" (menus): light / dark pixels, or grey levels."""
    h, w = region_bgr.shape[:2]
    gray = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2GRAY) if region_bgr.ndim == 3 else region_bgr
    size = (THUMB_W, max(1, int(h * THUMB_W / w))) if w > THUMB_W else None
    if ink in ("light", "dark"):
        # threshold at full size, then shrink: averaging first would wash thin glyph strokes out
        m = (gray >= 190) if ink == "light" else (gray <= 70)
        if size is None:
            return m
        return cv2.resize(m.astype(np.uint8) * 255, size, interpolation=cv2.INTER_AREA) >= 40
    if size is not None:
        gray = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
    return (gray // 32).astype(np.uint8)  # 'any': coarse grey levels


def auto_pause(settled_kinds: set[str], empties: int) -> float:
    """How long the screen must have been still before it is read as a menu by itself. Reading a whole screen keeps
    the OCR busy for about a second, and a line of dialogue would have to wait for it: while one is on screen, and
    where nothing was found the last times (the player stands still in the world), take more time."""
    if "name" in settled_kinds and settled_kinds - {"name"}:
        need = 6.0  # someone is speaking
    else:
        need = 2.0 if settled_kinds else 0.7
    return max(need, min(8.0, 2.0 ** empties) if empties else 0.0)


def stroke_change(a: np.ndarray, b: np.ndarray, rows: int = 9, cols: int = 16) -> float:
    """How much of the screen's strokes are in parts of it that changed (0 = all the strokes stand still, 1 = none
    do). Judged tile by tile, so that a character moving in a menu leaves the menu's text "still", while the 3D
    world, whose strokes are everywhere, is not."""
    if a.shape != b.shape:
        return 1.0
    h, w = a.shape
    moved = total = 0
    for r in range(rows):
        for c in range(cols):
            ta = a[r * h // rows : (r + 1) * h // rows, c * w // cols : (c + 1) * w // cols]
            tb = b[r * h // rows : (r + 1) * h // rows, c * w // cols : (c + 1) * w // cols]
            n = max(int(ta.sum()), int(tb.sum()))
            total += n
            if n and (ta ^ tb).sum() > 0.25 * n:
                moved += n
    return moved / max(1, total)


def changed_pixels(a: np.ndarray, b: np.ndarray) -> int:
    if a.shape != b.shape:
        return a.size
    return int(np.count_nonzero(a != b))


def stroke_map(region_bgr: np.ndarray, ink: str = "light") -> np.ndarray:
    """Thin strokes in a region: light ones (white text), dark ones, or both ('any'). Large bright or dark areas
    (sky, mist, a white dress) are not strokes and drop out."""
    gray = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2GRAY) if region_bgr.ndim == 3 else region_bgr
    h, w = gray.shape[:2]
    if w > STROKE_W:
        gray = cv2.resize(gray, (STROKE_W, max(1, int(h * STROKE_W / w))), interpolation=cv2.INTER_AREA)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    light = dark = None
    if ink in ("light", "any"):
        light = (cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k) > 50) & (gray > 150)
    if ink in ("dark", "any"):
        dark = (cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k) > 50) & (gray < 110)
    return light if dark is None else (dark if light is None else light | dark)


def has_text(strokes: np.ndarray | None, min_frac: float = 0.0008) -> bool:
    """Could there be text at all? A short line (うん。) has few strokes; whether it is text is for the OCR to say."""
    return strokes is not None and int(strokes.sum()) >= max(12, min_frac * strokes.size)


# a line that ends like this is complete as far as reading can tell (not so after an ellipsis: "くっ……" goes on)
SENTENCE_END = re.compile(r"[^.．…‥・·•●･][。．.！!？?♪♡★」』）)\]】]$")
# how long after a line was taken as complete it is read again, in case it went on unnoticed (over a bright scene
# the strokes of what was added may not show) or was misread
CHECKS = (0.5, 1.2, 2.5, 5.0)
QUIET = 0.3  # nothing typed for so long: the line stands (typing adds letters far more often)
BACK = 2.0  # a line that is read again so soon after its strokes were lost had never been gone


class TextFollower:
    """Follows the text of one region. feed() every tick tells what to do now:
      'hide' - the text whose translation is shown (or on its way) is gone or replaced;
      'read' - read the region now, then pass the result to on_read();
      None   - nothing.
    on_read() says 'translate' when the text is complete, 'hide' when what is shown turns out to be wrong. A line
    being typed out is read again and again; it counts as complete when it ends like a sentence, when nothing was
    typed for a moment, or when two readings in a row are the same. If it goes on after all, it is read again.

    Reading costs CPU, so it only happens while there are strokes that stand still and something about them is new,
    and ever less often while nothing readable is found (the scene itself has thin strokes too).

    `seq` counts texts (a line that went on is a new one), `line` counts lines: a translation of an older seq but the
    same line is of the beginning of what is on screen now."""

    def __init__(self, interval: float = 0.25, max_interval: float = 1.2, frames: int = 3, confirm: bool = False):
        self.min_interval, self.max_interval = interval, max_interval
        self.confirm = confirm  # always want two equal readings (engines that are paid by the character)
        self.hist: deque = deque(maxlen=frames)
        self.persistent: np.ndarray | None = None  # strokes present in every one of the last frames
        self.ref: np.ndarray | None = None  # strokes of the text that was taken as complete
        self.ref_n = 0
        self.reading = False
        self.read_at = -9.0
        self.read_n = -1  # stroke count and strokes when the last reading was taken
        self.read_strokes: np.ndarray | None = None
        self.last_text = ""
        self.misses = 0
        self.seq = 0  # translations of an older text are out of date
        self.line = 0
        self.gone_text, self.gone_at = "", -9.0  # the line whose strokes were lost last
        self.unsure = False  # this line's strokes cannot be trusted: whether it is still there is for reading to say
        self.votes: Counter = Counter()  # how the line was read, each time
        self.corrected = False  # 'translate' is for the same line, misread at first
        self.done_at = 0.0
        self.checks = 0
        self.blank = 0
        self.differs = False  # the last reading of the line was not what it was taken for
        self.level, self.grew_at = 0, -9.0  # stroke count when letters were last added, and when that was

    def reset(self):
        self.__init__(self.min_interval, self.max_interval, self.hist.maxlen, self.confirm)

    @property
    def in_doubt(self) -> bool:
        """Not the moment to put a translation on screen: the next reading tells whether the line is still there."""
        return self.ref is not None and (self.blank > 0 or self.unsure and self.reading)

    @property
    def settled(self) -> bool:
        return self.ref is not None

    def feed(self, strokes: np.ndarray, now: float) -> str | None:
        if self.hist and self.hist[-1].shape != strokes.shape:
            seq, line = self.seq, self.line
            self.reset()
            self.seq, self.line = seq + 1, line + 1
        self.hist.append(strokes)
        if len(self.hist) < self.hist.maxlen:
            return None
        p = np.logical_and.reduce(list(self.hist))
        n = int(p.sum())
        self.persistent = p
        if n > 1.04 * self.level + 4:
            self.level, self.grew_at = n, now
        elif n < 0.75 * self.level:
            self.level = n
        if self.ref is not None:
            kept = int((p & self.ref).sum()) / self.ref_n
            grown = n > 1.2 * self.ref_n + 15
            if kept >= 0.3 or self.unsure:
                # Still there as far as one can tell. Strokes also get lost when the scene behind the text turns
                # bright, and new ones appear when it gets busy: when in doubt it is read again, not hidden.
                doubt = (kept < 0.6 or grown or self.differs or self.blank) and now - self.read_at >= self.min_interval
                due = self.checks < len(CHECKS) and now - self.done_at >= CHECKS[self.checks]
                if not self.reading and (doubt or due):
                    self.checks += due
                    self.reading, self.read_at, self.read_n, self.read_strokes = True, now, n, p
                    return "read"
                return None
            self.gone_text, self.gone_at = self.last_text, now
            self.ref = None
            self.seq += 1
            self.line += 1
            self.misses = 0
            self.read_n, self.read_strokes, self.last_text = -1, None, ""
            return "hide"
        if not has_text(p):
            if self.last_text or self.misses:
                self.last_text, self.misses = "", 0
            self.read_n, self.read_strokes = -1, None
            return None
        if self.reading:
            return None
        wait = min(self.max_interval, self.min_interval * 2 ** min(self.misses, 5))
        stopped = self.misses <= 4 and self.grew_at > self.read_at and now - self.grew_at >= 0.1 and self._grown(n)
        if stopped or self._jumped(p, n):  # letters were added and no more are coming, or something quite different
            wait = min(wait, self.min_interval)  # appeared: do not wait out the pauses of before
        if now - self.read_at < wait:
            return None
        if self.misses >= 2 and self.read_strokes is not None and not self._moved(p, n):
            return None  # the same strokes that were unreadable twice
        self.reading, self.read_at = True, now
        self.read_n, self.read_strokes = n, p
        return "read"

    def _moved(self, p, n) -> bool:
        return abs(n - self.read_n) > max(5, 0.04 * max(n, self.read_n)) or int((p & self.read_strokes).sum()) < 0.85 * min(n, self.read_n)

    def _grown(self, n) -> bool:
        return n > 1.12 * self.read_n + 10

    def _jumped(self, p, n) -> bool:
        if self.read_strokes is None or self.read_strokes.shape != p.shape:
            return True
        return n > 1.6 * self.read_n + 40 or int((p & self.read_strokes).sum()) < 0.4 * min(n, self.read_n)

    def on_read(self, text: str, now: float) -> str | None:
        """The reading asked for has arrived. 'translate': the text is complete (and new); 'hide': the text that is
        shown went on or was replaced, it will be read again."""
        self.reading = False
        if self.read_strokes is None:
            return None
        if self.ref is not None:
            return self._check(text, now)
        if not text:
            self.misses += 1
            self.last_text = ""
            return None
        self.misses = 0
        growing = self._grown(int(self.persistent.sum()))  # typed on while it was being read
        quiet = now - self.grew_at >= QUIET and not growing and not self._moved(self.persistent, int(self.persistent.sum()))
        back = now - self.gone_at <= BACK and bool(plain_text(text)) and same_text(self.gone_text, text)
        same = bool(self.last_text) and same_text(self.last_text, text)
        self.last_text = text
        if back:
            self.unsure = True
        elif not same_text(self.gone_text, text):
            self.unsure = False
        if same or back or ((quiet or SENTENCE_END.search(text) and not growing) and not self.confirm):
            self._take(now)
            return "translate"
        return None

    def _take(self, now: float):
        # what was read; what was typed since is to be noticed
        self.ref, self.ref_n = self.read_strokes, max(1, int(self.read_strokes.sum()))
        self.done_at, self.checks, self.blank = now, 0, 0
        self.votes, self.differs, self.corrected = Counter([self.last_text]), False, False

    def _check(self, text: str, now: float) -> str | None:
        """A text that was taken as complete, read once more."""
        old = self.last_text
        if not text:
            self.blank += 1
            kept = int((self.persistent & self.ref).sum()) / self.ref_n
            if self.blank < 3 and kept >= 0.3:  # (below that it is neither seen nor read any more)
                return None
            self.seq += 1  # gone for good
            self.line += 1
            self.gone_text, self.unsure = "", False
            self.ref, self.last_text, self.misses = None, "", 1
            return "hide"
        self.blank = 0
        alike = similar(old, text)
        longer = len(plain_text(text)) >= len(plain_text(old)) + 2 and (
            plain_text(text).startswith(plain_text(old)[: max(2, len(plain_text(old)) - 3)]) or alike >= 0.45)
        if not longer and alike >= 0.4:  # the same, read a little differently or only in part
            self.ref, self.ref_n = self.read_strokes, max(1, int(self.read_strokes.sum()))  # how it looks now
            self.votes[text] += 1
            self.differs = skeleton(text) != skeleton(old) and len(plain_text(text)) >= len(plain_text(old)) - 1 \
                and len(self.votes) < 4 and sum(self.votes.values()) < 7
            if self.differs and not self.corrected and not self.confirm and self.votes[text] > self.votes[old]:
                self.last_text, self.differs, self.corrected = text, False, True  # it was misread at first
                self.seq += 1
                return "translate"
            return None
        self.seq += 1
        if not longer:
            self.line += 1
            self.unsure = False
        self.last_text = text
        self.misses = 0
        if SENTENCE_END.search(text) and not self.confirm:
            self._take(now)
            return "translate"
        self.ref = None
        return "hide" if not longer else None  # a line that goes on keeps what is shown until the rest is there

    def again(self):
        """The text could not be handled (server not ready, an error): read it again."""
        self.ref, self.reading, self.misses = None, False, 0
        self.read_n, self.read_strokes, self.last_text = -1, None, ""


def plain_text(s: str) -> str:
    """Without what OCR reads differently from one time to the next: dots, dashes, spaces, small marks."""
    return re.sub(r"[\s.．。、，,…‥・·•●･ー一\-—―~～〜'\"「」『』:：]+", "", s)


def skeleton(s: str) -> str:
    """What is left when characters OCR often confuses, with little harm to the meaning, are treated alike: voicing
    marks (は ば ぱ), small kana (ィ イ), hiragana and katakana (へ ヘ), full and half width."""
    s = "".join(c for c in unicodedata.normalize("NFKD", plain_text(s)) if c not in "\u3099\u309a!?")
    s = "".join(chr(ord(c) + 0x60) if "\u3041" <= c <= "\u3096" else c for c in s)
    return s.translate(str.maketrans("ァィゥェォッャュョヮ", "アイウエオツヤユヨワ"))


def goes_on(old: str, new: str) -> bool:
    """Is `new` the text `old` with more of it typed out?"""
    po, pn = plain_text(old), plain_text(new)
    return len(pn) >= len(po) and (pn.startswith(po[: max(2, len(po) - 3)]) or similar(po, pn[: len(po)]) >= 0.8)


def same_text(a: str, b: str) -> bool:
    """Two readings of one unchanged text (as against one with more letters typed)."""
    pa, pb = plain_text(a), plain_text(b)
    return abs(len(pa) - len(pb)) <= (1 if len(pa) > 8 else 0) and similar(pa, pb) >= 0.9


def similar(a: str, b: str) -> float:
    """0..1 similarity of two OCR results (OCR jitters by a character or two between reads)."""
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    import difflib

    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()
