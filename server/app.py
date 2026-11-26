"""game-lens server: text lines off a game screen (OCR) and their translation.

Run:  .venv\\Scripts\\python.exe -m uvicorn app:app --app-dir server --host 127.0.0.1 --port 9960
game-lens starts and stops it by itself. Every endpoint except /health needs the header X-Lens-Token
(data/token.txt, created on first start). Screenshots are processed in memory only; the SQLite cache keeps
text and translations.
"""
from __future__ import annotations

import hashlib
import os
import secrets
import threading
import time
import unicodedata
from pathlib import Path

import cv2
import numpy as np
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from pydantic import BaseModel

from ocr import NO_SPACE, normalize_lang, read_dialog, read_lines, read_name, route
from store import Store
from translate.engines import (REMOTE, EngineError, extract_glossary, fold_name, get_engine, glossary_terms, match_name,
                               match_term, translate_blocks)

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


def line_key(src: str, kind: str, text: str) -> str:
    return hashlib.sha1(f"{src}|{kind}|{text}".encode()).hexdigest()


@app.post("/translate/lines", dependencies=[Depends(auth)])
async def translate_lines(req: TranslateLinesReq):
    """Game text lines -> translations, cached per line (a line seen before comes back without an engine call)."""
    if req.target not in SUPPORTED_TARGETS:
        raise HTTPException(400, f"target must be one of {SUPPORTED_TARGETS}")
    if req.engine not in GAME_ENGINES:
        raise HTTPException(400, f"engine must be one of {GAME_ENGINES}")
    src = normalize_lang(req.src_lang)
    t0 = time.perf_counter()
    gver, glossary = store.get_glossary(glossary_key(req.series, req.target)) if req.series else (0, {})
    keys = {l.id: line_key(src, l.kind, l.src.strip()) for l in req.lines}
    cached = store.get_lines(sorted(set(keys.values())), req.target, gver, req.engine)
    # an earlier fallback is cached under the requested engine; retry once that engine is usable again
    remote_ok = req.engine in REMOTE and req.allow_cloud and get_engine(req.engine).available()
    out: dict[str, str] = {}
    from_cache, todo, engines_used = [], [], []
    names = glossary_terms(glossary)
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
        if hit and not (hit[0] not in (req.engine, "none") and remote_ok):
            out[l.id] = hit[1]
            from_cache.append(l.id)
            engines_used.append(hit[0])
            continue
        todo.append({"id": l.id, "src": text, "kind": l.kind})
    metrics: dict = {}
    if todo:
        ctx = [c.model_dump() for c in req.context][-12:]
        try:
            tl, used, metrics = await run_in_thread(
                translate_blocks, todo, glossary, ctx, req.engine, req.allow_cloud, False, src, req.target, True,
            )
        except EngineError as e:
            raise HTTPException(503, f"translation failed: {e}")
        for b in todo:
            out[b["id"]] = tl.get(b["id"], "")
        store.put_lines([(keys[b["id"]], tl[b["id"]]) for b in todo if b["id"] in tl], req.target, gver, req.engine, used)
        engines_used.append(used)
    engine = max(set(engines_used), key=engines_used.count) if engines_used else "none"
    return {"lines": out, "engine": engine, "engineRequested": req.engine, "cached": from_cache,
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


async def run_in_thread(fn, *args):
    import anyio

    return await anyio.to_thread.run_sync(lambda: fn(*args))
