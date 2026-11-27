"""game-lens server: text lines off a game screen (OCR) and their translation.

Run:  .venv\\Scripts\\python.exe -m uvicorn app:app --app-dir server --host 127.0.0.1 --port 9960
game-lens starts and stops it by itself. Every endpoint except /health needs the header X-Lens-Token
(data/token.txt, created on first start). Screenshots are processed in memory only; the SQLite cache keeps
text and translations.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import threading
import time
import unicodedata
from pathlib import Path

import cv2
import numpy as np
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from pydantic import BaseModel

from ocr import NO_SPACE, normalize_lang, read_dialog, read_lines, read_name, read_screen, route
from store import Store
from fuzzy import close_enough, differences, fuzzy_key
from translate.engines import (REMOTE, STREAM, EngineError, clean_mt, extract_glossary, fold_name, get_engine,
                               glossary_terms, match_name, match_term, translate_blocks)

ROOT = Path(__file__).parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)
TOKEN_FILE = DATA / "token.txt"
if not TOKEN_FILE.exists():
    TOKEN_FILE.write_text(secrets.token_urlsafe(24))
TOKEN = TOKEN_FILE.read_text().strip()
# API keys entered in game-lens are kept here (this machine only) and override the environment
KEYS_FILE = DATA / "keys.json"
KEY_NAMES = {"deepl": ["DEEPL_API_KEY"], "azure": ["AZURE_TRANSLATOR_KEY", "AZURE_TRANSLATOR_REGION"],
             "google": ["GOOGLE_TRANSLATE_API_KEY"], "cloud": ["ANTHROPIC_API_KEY"]}


def _load_keys():
    import json

    try:
        for k, v in json.loads(KEYS_FILE.read_text(encoding="utf-8")).items():
            if v:
                os.environ[k] = v
    except (OSError, ValueError):
        pass


_load_keys()

SUPPORTED_TARGETS = ("th", "en")
store = Store(DATA / "cache.sqlite")
ocr_lock = threading.Lock()  # one OCR at a time

app = FastAPI(title="game-lens")


def auth(x_lens_token: str = Header(default="")):
    if not secrets.compare_digest(x_lens_token, TOKEN):
        raise HTTPException(401, "bad or missing X-Lens-Token")


def decode_image(data: bytes) -> np.ndarray:
    arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        raise HTTPException(400, "cannot decode image")
    return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)


def already_in(text: str, target: str, src: str = "ja") -> bool:
    """True when OCR'd text is written in the target language's script only (nothing to translate): an English
    "CONTENTS" heading on a Japanese page, a Thai sign in a Korean one. The script says nothing when the source
    language is written in the same script as the target: Indonesian, Spanish, Vietnamese... are Latin like English."""
    if target == "en" and route(src) == ("PP-OCRv6", "en"):
        return False  # a Latin-script source (src == target never gets here)
    t = unicodedata.normalize("NFKC", text)
    letters = [c for c in t if c.isalpha()]
    if not letters:
        return True
    if target == "en":
        return all(c.isascii() for c in letters)
    if target == "th":
        return all("฀" <= c <= "๿" for c in letters)
    return False


def glossary_key(series: str, target: str) -> str:
    return f"{series}|{target}"


@app.get("/ping")
def ping():
    """Is the server up? (/health also probes every engine, which takes a second or two.)"""
    return {"ok": True, "app": "game-lens"}


@app.get("/health")
def health():
    from concurrent.futures import ThreadPoolExecutor

    names = list(GAME_ENGINES)
    # probe in parallel: a refused localhost connection can take a second on Windows
    with ThreadPoolExecutor(len(names)) as pool:
        up = dict(zip(names, pool.map(lambda n: get_engine(n).available(), names)))
    return {"ok": True, "app": "game-lens", **up, "targets": SUPPORTED_TARGETS}


GAME_ENGINES = ("local", "cloud", "google", "tgemma", "plamo", "hymt", "index", "deepl", "azure", "google_free")


@app.post("/ocr/lines", dependencies=[Depends(auth)])
async def ocr_lines(
    image: UploadFile = File(...),
    src_lang: str = Form("ja"),
    min_score: float = Form(0.5),
    kind: str = Form(""),
    series: str = Form(""),
    target: str = Form("th"),
    align: str = Form("left"),
):
    """A screenshot or a crop of one (a dialogue box): its typeset horizontal lines, in reading order. For
    kind=name (a speaker-name box) the names of the series' glossary help to read it."""
    src = normalize_lang(src_lang)
    img = decode_image(await image.read())
    t0 = time.perf_counter()

    def work():
        with ocr_lock:
            if kind == "name" and series:
                terms = glossary_terms(store.get_glossary(glossary_key(series, target))[1])
                lines = read_name(img, src, lambda t: (match_term(t, terms) or (None,))[0], fold_name)
            elif kind == "dialog":
                lines = read_dialog(img, src, align == "center", lambda t: not already_in(t, "en", "ja") or src != "ja")
            elif kind in ("screen", "screen-look"):  # a whole screen (a menu): its faint text too
                # (a look every few seconds at a screen already read leaves out the slow look for small text over
                # the moving world: what the first read found of it is known already)
                lines = read_screen(img, src, min_score, deep=kind == "screen")
            else:
                lines = read_lines(img, src, min_score)
        for i, l in enumerate(lines):
            l["id"] = f"l{i}"
        return lines

    lines = await run_in_thread(work)
    sep = "" if src.split("-")[0] in NO_SPACE else " "
    h, w = img.shape[:2]
    return {"w": w, "h": h, "srcLang": src, "lines": lines, "text": sep.join(l["text"] for l in lines),
            "ms": round((time.perf_counter() - t0) * 1000)}


class GameLine(BaseModel):
    id: str
    src: str
    kind: str = "dialog"  # dialog / name / subtitle / ui


class GameContext(BaseModel):
    src: str
    tl: str


class TranslateLinesReq(BaseModel):
    lines: list[GameLine]
    series: str = ""
    target: str = "th"
    src_lang: str = "ja"
    engine: str = "local"
    allow_cloud: bool = False
    context: list[GameContext] = []


NAME_RE = re.compile(r"^[ァ-ヺー・＝=]{2,10}$")  # a name in katakana (ジャベリン, カーク)
LATIN_NAME_RE = re.compile(r"^[A-Z][A-Za-z'’-]*( [A-Z][A-Za-z'’-]*){0,2}$")  # Javelin, Han Youyou


def with_learned(glossary: dict, learned: list[tuple[str, str]]) -> dict:
    """The glossary with the speaker names learned so far; the glossary's own entry wins (a correction), a term too."""
    have = {c.get("src") for c in glossary.get("characters", [])} | {a for c in glossary.get("characters", []) for a in c.get("aliases", [])}
    have |= {x.get("src") for x in glossary.get("terms", [])}
    extra = [{"src": s, "tl": t, "source": "learned"} for s, t in learned if s not in have]
    return {**glossary, "characters": list(glossary.get("characters", [])) + extra} if extra else glossary


NOT_A_NAME: set[tuple[str, str]] = set()  # (glossary key, text) the engine gave no name for: not asked again


def apply_fixes(text: str, fixes: list) -> str:
    """text with each fix's "src" replaced by its "tl", the longest first (so a fix of a whole phrase is not undone by
    one of a word in it)."""
    for f in sorted((f for f in fixes or [] if isinstance(f, dict) and f.get("src")), key=lambda f: -len(f["src"])):
        text = text.replace(f["src"], f.get("tl", ""))
    return text


def line_scope(src: str, kind: str, series: str = "") -> str:
    return f"{src}|{'ui3' if kind == 'ui' else kind}|{series}"


def line_key(src: str, kind: str, text: str, series: str = "") -> str:
    # menu text is cached under "ui3": translations made while the labels before one were its context (which
    # copied their translations into it), or with the note for spoken dialogue (which added to descriptions),
    # are not used any more
    # (and per game: the same line in two games has each game's names and terms in it)
    kind = "ui3" if kind == "ui" else kind
    return hashlib.sha1((f"{src}|{kind}|{text}" + (f"|{series}" if series else "")).encode()).hexdigest()


class WarmReq(BaseModel):
    src_lang: str = "ja"
    engine: str = "local"


@app.post("/warmup", dependencies=[Depends(auth)])
async def warmup(req: WarmReq):
    """Do once what the first reading and the first translations would otherwise wait for: load the text reader
    (~1.3 s) and let a local translation model answer a few requests at a time (its first such round takes ~1 s
    longer). Nothing of it is kept."""
    src = normalize_lang(req.src_lang)
    done = {}

    def work():
        t0 = time.perf_counter()
        probe = np.full((64, 320, 3), 255, np.uint8)
        cv2.putText(probe, "OCR 123", (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
        with ocr_lock:
            read_lines(probe, src)
        done["ocr_ms"] = round((time.perf_counter() - t0) * 1000)
        if req.engine in ("hymt", "index"):
            eng = get_engine(req.engine)
            if eng.available():
                t0 = time.perf_counter()
                try:
                    eng.translate_page(["はい", "いいえ", "戻る", "決定"], src, "th", [], {}, style="game", kinds=["ui"] * 4)
                    done["model_ms"] = round((time.perf_counter() - t0) * 1000)
                except EngineError as e:
                    done["model_error"] = str(e)[:200]

    await run_in_thread(work)
    return done


# Menu text that differs from text translated before only in its numbers ("攻撃力が4%上昇" / "7.1%上昇", "Lv.3",
# "12/30") is not translated again: the earlier translation is kept as a pattern and the numbers are put in.
NUM_RE = re.compile(r"\d+(?:[.,]\d+)*")


def mask_numbers(text: str) -> tuple[str, list[str]]:
    """The text with its numbers taken out, and the numbers in order."""
    nums = NUM_RE.findall(text)
    return NUM_RE.sub("\x00", text), nums


def make_pattern(nums: list[str], tl: str) -> str | None:
    """The translation as a pattern for other numbers; None where that is not safe: the numbers must all differ
    and each stand in the translation exactly once, with no number besides them."""
    if not nums or len(set(nums)) != len(nums) or sorted(NUM_RE.findall(tl)) != sorted(nums):
        return None
    return json.dumps({"t": NUM_RE.sub(lambda m: f"\x00{nums.index(m.group())}\x00", tl), "n": nums}, ensure_ascii=False)


def fill_pattern(pattern: str, nums: list[str], target: str) -> str | None:
    try:
        p = json.loads(pattern)
        if len(p["n"]) != len(nums):
            return None
        # English words change with one / many ("1 item", "5 items"): only like for like
        if target == "en" and any((a.strip("0.,") == "1") != (b.strip("0.,") == "1") for a, b in zip(p["n"], nums)):
            return None
        return re.sub("\x00(\\d+)\x00", lambda m: nums[int(m.group(1))], p["t"])
    except (ValueError, KeyError, IndexError, TypeError):
        return None


@app.post("/translate/lines", dependencies=[Depends(auth)])
async def translate_lines(req: TranslateLinesReq):
    """Game text lines -> translations, cached per line (a line seen before comes back without an engine call)."""
    return await _translate_lines(req)


@app.post("/translate/stream", dependencies=[Depends(auth)])
async def translate_stream(req: TranslateLinesReq):
    """As /translate/lines, but the translation is sent as the model writes it: lines of JSON, {"partial": text so
    far} (a local model translating a line), then {"done": what /translate/lines answers} or {"error", "status"}.
    For a line of dialogue: its first words are on screen in a few tenths of a second, not after the whole line."""
    import asyncio

    from fastapi.responses import StreamingResponse

    loop = asyncio.get_running_loop()
    q: asyncio.Queue = asyncio.Queue()
    target, fixes = req.target, None
    last = [0.0, ""]

    def partial(text: str):  # (in the worker thread)
        text = clean_mt(text, target)
        text = apply_fixes(text, fixes or [])
        now = time.monotonic()
        if text and text != last[1] and now - last[0] >= 0.04:  # (a few updates a second are enough to see)
            last[0], last[1] = now, text
            loop.call_soon_threadsafe(q.put_nowait, {"partial": text})

    async def work():
        try:
            q.put_nowait({"done": await _translate_lines(req, partial)})
        except HTTPException as e:
            q.put_nowait({"error": str(e.detail), "status": e.status_code})
        except Exception as e:  # noqa: BLE001 - the stream must end with something
            q.put_nowait({"error": f"{type(e).__name__}: {e}", "status": 500})

    if req.series:
        _, g = store.get_glossary(glossary_key(req.series, req.target))
        fixes = g.get("tl_fixes")

    async def lines():
        task = asyncio.create_task(work())
        try:
            while True:
                msg = await q.get()
                yield json.dumps(msg, ensure_ascii=False) + "\n"
                if "partial" not in msg:
                    break
        finally:
            if not task.done():
                task.cancel()  # (the client went away)

    return StreamingResponse(lines(), media_type="application/x-ndjson")


async def _translate_lines(req: TranslateLinesReq, partial=None):
    if req.target not in SUPPORTED_TARGETS:
        raise HTTPException(400, f"target must be one of {SUPPORTED_TARGETS}")
    if req.engine not in GAME_ENGINES:
        raise HTTPException(400, f"engine must be one of {GAME_ENGINES}")
    src = normalize_lang(req.src_lang)
    t0 = time.perf_counter()
    gkey = glossary_key(req.series, req.target) if req.series else ""
    gver, glossary = store.get_glossary(gkey) if req.series else (0, {})
    if req.series:
        glossary = with_learned(glossary, store.get_learned(gkey))
    # the player's fixes: what OCR keeps misreading, put right before anything else looks at the line (the cache too)
    for l in req.lines:
        l.src = apply_fixes(l.src, glossary.get("ocr_fixes"))
    keys = {l.id: line_key(src, l.kind, l.src.strip(), gkey) for l in req.lines}
    cached = store.get_lines(sorted(set(keys.values())), req.target, gver, req.engine)
    masked = {l.id: mask_numbers(l.src.strip()) for l in req.lines if l.kind == "ui"}
    pkeys = {i: line_key(src, "ui#n", m, gkey) for i, (m, nums) in masked.items() if nums and keys[i] not in cached}
    patterns = store.get_lines(sorted(set(pkeys.values())), req.target, gver, req.engine) if pkeys else {}
    # a line OCR read a little differently from one translated before: that translation (fuzzy.py)
    fkeys = {l.id: f for l in req.lines if l.kind != "name" and keys[l.id] not in cached
             and (f := fuzzy_key(line_scope(src, l.kind, gkey), l.src.strip()))}
    seen = store.get_fuzzy(sorted(set(fkeys.values())))
    alike: dict[str, list[str]] = {}  # line id -> keys of the lines it may be a misreading of, nearest first
    for l in req.lines:
        got = [(differences(t, l.src.strip()), k) for k, t in seen.get(fkeys.get(l.id, ""), [])
               if close_enough(t, l.src.strip())]
        if got:
            alike[l.id] = [k for _, k in sorted(got)]
    near = store.get_lines(sorted({k for v in alike.values() for k in v}), req.target, gver, req.engine)
    # an earlier fallback is cached under the requested engine; retry once that engine is usable again
    remote_ok = req.engine in REMOTE and req.allow_cloud and get_engine(req.engine).available()
    local_ok = None  # (a local engine that failed once - loading, a timeout - is asked again once it is up)
    out: dict[str, str] = {}
    from_cache, from_alike, todo, engines_used = [], [], [], []
    names = glossary_terms(glossary)
    if req.series and req.target != "en":
        # a speaker the glossary does not have: written in Latin letters like the known names (the official Thai
        # site keeps them so), not made Thai by the engine - which did it differently line to line (シフォン came
        # out ซิฟอน and ชีฟอง) and translated some (ジャベリン became หอกยาว, "spear"). Asked once, then known
        new = [l for l in req.lines if l.kind == "name" and NAME_RE.match(l.src.strip()) and not match_name(l.src.strip(), names)
               and (gkey, l.src.strip()) not in NOT_A_NAME]
        if new:
            try:
                en, _, _ = await run_in_thread(translate_blocks, [{"id": l.id, "src": l.src.strip(), "kind": "name"} for l in new],
                                               {}, [], req.engine, req.allow_cloud, False, src, "en", True)
            except EngineError:
                en = {}
            for l in new:
                tl = (en.get(l.id) or "").strip().strip("\"'.。「」")
                if not LATIN_NAME_RE.match(tl):
                    if l.id in en:  # (the engine answered, with no name: not asked again for it)
                        NOT_A_NAME.add((gkey, l.src.strip()))
                    continue
                store.put_learned(gkey, l.src.strip(), tl)
                names.append((l.src.strip(), tl))
                glossary.setdefault("characters", []).append({"src": l.src.strip(), "tl": tl, "source": "learned"})
    for l in req.lines:
        text = l.src.strip()
        if not text or src.split("-")[0] == req.target or already_in(text, req.target, src):
            out[l.id] = ""
            continue
        # a line that is nothing but a known name (a speaker-name box) needs no engine; for name boxes this
        # also forgives the misreadings OCR makes on short text
        known = match_name(text, names) if l.kind == "name" else dict(names).get(text)
        if known:
            out[l.id] = known
            engines_used.append("glossary")
            continue
        hit = cached.get(keys[l.id])
        fallback = hit and hit[0] not in (req.engine, "none")
        if fallback and not remote_ok and req.engine not in REMOTE:
            if local_ok is None:
                local_ok = get_engine(req.engine).available()
            fallback = local_ok
        if hit and not (fallback and (remote_ok or local_ok)):
            out[l.id] = hit[1]
            from_cache.append(l.id)
            engines_used.append(hit[0])
            continue
        pat = patterns.get(pkeys.get(l.id, ""))
        filled = fill_pattern(pat[1], masked[l.id][1], req.target) if pat and not remote_ok else None
        if filled:
            out[l.id] = filled
            from_cache.append(l.id)
            engines_used.append(pat[0])
            continue
        # (only what the engine asked for made: a fallback's translation is not carried to a new line)
        twin = next((near[k] for k in alike.get(l.id, []) if k in near and near[k][0] == req.engine), None)
        if twin:
            out[l.id] = twin[1]
            from_cache.append(l.id)
            from_alike.append(l.id)
            engines_used.append(twin[0])
            continue
        todo.append({"id": l.id, "src": text, "kind": l.kind})
    metrics: dict = {}
    if todo:
        ctx = [c.model_dump() for c in req.context][-12:]
        try:
            tl, used, metrics = await run_in_thread(
                streamed, partial, todo, glossary, ctx, req.engine, req.allow_cloud, False, src, req.target, True,
            )
        except EngineError as e:
            raise HTTPException(503, f"translation failed: {e}")
        for b in todo:
            out[b["id"]] = tl.get(b["id"], "")
        # (not an empty answer: a model's slip would come back for the line from then on)
        store.put_lines([(keys[b["id"]], tl[b["id"]]) for b in todo if tl.get(b["id"])], req.target, gver, req.engine, used)
        # (only the lines the engine translated: a misreading kept as the line itself would let the next one drift)
        store.put_fuzzy([(fkeys[b["id"]], keys[b["id"]], b["src"]) for b in todo if b["id"] in fkeys and tl.get(b["id"])])
        new = [(pkeys[b["id"]], p) for b in todo if b["id"] in pkeys and tl.get(b["id"])
               and (p := make_pattern(masked[b["id"]][1], tl[b["id"]]))]
        if new:
            store.put_lines(new, req.target, gver, req.engine, used)
        engines_used.append(used)
    if from_alike:  # (and the misreading's own key, so it is found at once next time)
        store.put_lines([(keys[i], out[i]) for i in from_alike], req.target, gver, req.engine, req.engine)
    engine = max(set(engines_used), key=engines_used.count) if engines_used else "none"
    # ...and words the engine keeps getting wrong, put right in what goes back (not in the cache: a fix changed or
    # taken out is so at once, for lines translated before too)
    out = {i: apply_fixes(v, glossary.get("tl_fixes")) if v else v for i, v in out.items()}
    return {"lines": out, "engine": engine, "engineRequested": req.engine, "cached": from_cache, "alike": from_alike,
            "metrics": {**metrics, "totalMs": round((time.perf_counter() - t0) * 1000)}}


def _mask(v: str) -> str:
    return "" if not v else ("•" * 4 + v[-4:] if len(v) > 8 else "•" * len(v))


@app.get("/config/keys", dependencies=[Depends(auth)])
def keys_get():
    """Which API keys are set (masked; the keys themselves never leave the server)."""
    return {eng: {n: _mask(os.environ.get(n, "")) for n in names} for eng, names in KEY_NAMES.items()}


@app.put("/config/keys", dependencies=[Depends(auth)])
def keys_put(keys: dict[str, str]):
    """Set (or clear with "") API keys by env name, e.g. {"DEEPL_API_KEY": "...:fx"}; saved in data/keys.json."""
    import json

    from translate import engines as eng_mod

    allowed = {n for names in KEY_NAMES.values() for n in names}
    bad = set(keys) - allowed
    if bad:
        raise HTTPException(400, f"unknown key names: {sorted(bad)}")
    try:
        saved = json.loads(KEYS_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        saved = {}
    for n, v in keys.items():
        v = (v or "").strip()
        saved[n] = v
        if v:
            os.environ[n] = v
        else:
            os.environ.pop(n, None)
    KEYS_FILE.write_text(json.dumps(saved, indent=1), encoding="utf-8")
    for eng, names in KEY_NAMES.items():  # engines read their key when created: make them again
        if set(names) & set(keys):
            eng_mod._engines.pop(eng, None)
    return keys_get()


@app.post("/config/test/{engine}", dependencies=[Depends(auth)])
def engine_test(engine: str, target: str = "th"):
    """Translate one short line with exactly this engine (no fallback): shows whether a key works."""
    if engine not in GAME_ENGINES:
        raise HTTPException(400, f"engine must be one of {GAME_ENGINES}")
    eng = get_engine(engine)
    t0 = time.perf_counter()
    try:
        tl, used, m = translate_blocks([{"id": "t", "src": "こんにちは、指揮官。", "kind": "dialog"}], {}, [], engine, True,
                                       False, "ja", target, True)
    except EngineError as e:
        return {"ok": False, "error": str(e)[:300]}
    if used != engine:
        err = "; ".join(m.get("errors", [])) or f"used {used} instead"
        return {"ok": False, "error": err[:300], "available": eng.available()}
    return {"ok": True, "text": tl.get("t", ""), "ms": round((time.perf_counter() - t0) * 1000)}


@app.get("/usage", dependencies=[Depends(auth)])
def usage():
    """Characters used / limit of the DeepL API key this billing period (null when no key is set)."""
    eng = get_engine("deepl")
    if not eng.available():
        return {"deepl": None}
    try:
        return {"deepl": eng.usage()}
    except EngineError as e:
        return {"deepl": {"error": str(e)}}


class ExtractTextReq(BaseModel):
    text: str
    target: str = "th"
    src_lang: str = "ja"
    engine: str = "local"
    allow_cloud: bool = False


@app.post("/glossary/{series:path}/extract-text", dependencies=[Depends(auth)])
def glossary_extract_text(series: str, req: ExtractTextReq):
    """Same as /extract, from raw text (game-lens sends the lines it has seen); merges new names in."""
    if not req.text.strip():
        raise HTTPException(400, "no text")
    try:
        proposed = extract_glossary(req.text[-20000:], req.engine, req.allow_cloud, normalize_lang(req.src_lang), req.target)
    except EngineError as e:
        raise HTTPException(503, str(e))
    gkey = glossary_key(series, req.target)
    _, current = store.get_glossary(gkey)
    known_c = {c["src"] for c in current.get("characters", [])}
    known_t = {t["src"] for t in current.get("terms", [])}
    merged = {
        "characters": current.get("characters", []) + [c for c in proposed["characters"] if c["src"] not in known_c],
        "terms": current.get("terms", []) + [t for t in proposed["terms"] if t["src"] not in known_t],
    }
    return {"version": store.put_glossary(gkey, merged), "glossary": merged, "proposed": proposed}


@app.get("/glossary/{series:path}", dependencies=[Depends(auth)])
def glossary_get(series: str, target: str = "th"):
    ver, g = store.get_glossary(glossary_key(series, target))
    return {"version": ver, "glossary": g}


@app.put("/glossary/{series:path}", dependencies=[Depends(auth)])
def glossary_put(series: str, glossary: dict, target: str = "th"):
    return {"version": store.put_glossary(glossary_key(series, target), glossary)}


def streamed(partial, *args):
    """translate_blocks, with the text so far going to `partial` while a local model writes it (see STREAM)."""
    STREAM.sink = partial
    try:
        return translate_blocks(*args)
    finally:
        STREAM.sink = None


async def run_in_thread(fn, *args):
    import anyio

    return await anyio.to_thread.run_sync(lambda: fn(*args))
