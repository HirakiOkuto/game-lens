"""The glossary files (glossaries/<series>.<target>.json): names and terms that are always translated one way.
Corrections made in the history window go into them."""
from __future__ import annotations

import json
from pathlib import Path

ABOUT = ("Names and terms that are always translated this way. game-lens sends this file to its server at start; "
         "edit 'tl' to change how a name is written, or correct a translation in the history window (F7). "
         "ocr_fixes: what the text reader keeps misreading ('src') and what it should read ('tl'), put right before "
         "translating; tl_fixes: a word in the translations ('src') always replaced by 'tl'.")
FIX_LISTS = ("ocr_fixes", "tl_fixes")  # (each {"src", "tl"}: applied round the translation, not told the engine)


def dump_glossary(d: dict) -> str:
    """The file as people read and edit it: one name or term per line."""
    parts = []
    if "_about" in d:
        parts.append(f'  "_about": {json.dumps(d["_about"], ensure_ascii=False)}')
    for key in ("characters", "terms", *(k for k in FIX_LISTS if d.get(k))):
        rows = [f"    {json.dumps(x, ensure_ascii=False)}" for x in d.get(key, [])]
        parts.append(f'  "{key}": [' + ("\n" + ",\n".join(rows) + "\n  ]" if rows else "]"))
    return "{\n" + ",\n".join(parts) + "\n}\n"


def add_correction(path: Path, src: str, tl: str, kind: str) -> dict:
    """Make `src` translate as `tl` from now on. kind: "term", or a character's name with a gender: "female",
    "male", "neutral". An earlier entry of the same original is replaced (also of the other kind).
    kind "ocr": the text reader's `src` is to be read as `tl`; "tlfix": `src` in a translation becomes `tl`."""
    d = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"_about": ABOUT, "characters": [], "terms": []}
    if kind in ("ocr", "tlfix"):
        fixes = d.setdefault("ocr_fixes" if kind == "ocr" else "tl_fixes", [])
        fixes[:] = [x for x in fixes if x.get("src") != src] + [{"src": src, "tl": tl}]
        if "_about" in d:
            d["_about"] = ABOUT  # (an older file's note does not tell of the fixes)
        from .profile import write_safely

        path.parent.mkdir(parents=True, exist_ok=True)
        write_safely(path, dump_glossary(d))
        return d
    chars, terms = d.setdefault("characters", []), d.setdefault("terms", [])
    terms[:] = [x for x in terms if x.get("src") != src]
    old = next((c for c in chars if c.get("src") == src or src in c.get("aliases", [])), None)
    if kind == "term":
        chars[:] = [c for c in chars if c is not old]
        terms.append({"src": src, "reading": "", "tl": tl, "note": "", "source": "user"})
    else:
        gender = "" if kind == "neutral" else kind
        if old is not None:
            old.update(tl=tl, gender=gender, source="user")
        else:
            chars.append({"src": src, "aliases": [], "tl": tl, "gender": gender, "role": "", "source": "user"})
    path.parent.mkdir(parents=True, exist_ok=True)
    from .profile import write_safely

    write_safely(path, dump_glossary(d))
    return d
