"""SQLite cache: translated lines and glossaries. Never stores screenshots."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS line_tl (
  line_key TEXT NOT NULL, target TEXT NOT NULL, glossary_ver INTEGER NOT NULL, want TEXT NOT NULL, engine TEXT NOT NULL,
  tl TEXT NOT NULL, created REAL NOT NULL, PRIMARY KEY (line_key, target, glossary_ver, want));
CREATE TABLE IF NOT EXISTS glossaries (
  series_key TEXT PRIMARY KEY, version INTEGER NOT NULL, json TEXT NOT NULL, updated REAL NOT NULL);
"""


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.executescript(SCHEMA)
        self.lock = threading.Lock()

    def _q(self, sql, args=()):
        with self.lock:
            cur = self.db.execute(sql, args)
            rows = cur.fetchall()
            self.db.commit()
            return rows

    # lines: keyed by a hash of source language + kind + text, so a line seen again is instant.
    # `want` is the engine asked for; `engine` the one that produced the text (a fallback is cached under the
    # requested engine, so a broken engine is not retried for every line)
    def get_lines(self, keys: list[str], target: str, glossary_ver: int, want: str) -> dict[str, tuple[str, str]]:
        if not keys:
            return {}
        marks = ",".join("?" * len(keys))
        rows = self._q(
            f"SELECT line_key, engine, tl FROM line_tl WHERE target=? AND glossary_ver=? AND want=? AND line_key IN ({marks})",
            (target, glossary_ver, want, *keys),
        )
        return {k: (e, tl) for k, e, tl in rows}

    def put_lines(self, items: list[tuple[str, str]], target: str, glossary_ver: int, want: str, engine: str):
        now = time.time()
        with self.lock:
            self.db.executemany(
                "INSERT OR REPLACE INTO line_tl VALUES (?,?,?,?,?,?,?)",
                [(k, target, glossary_ver, want, engine, tl, now) for k, tl in items],
            )
            self.db.commit()

    def get_glossary(self, series_key: str) -> tuple[int, dict]:
        r = self._q("SELECT version, json FROM glossaries WHERE series_key=?", (series_key,))
        return (r[0][0], json.loads(r[0][1])) if r else (0, {"characters": [], "terms": []})

    def put_glossary(self, series_key: str, glossary: dict) -> int:
        ver, _ = self.get_glossary(series_key)
        self._q(
            "INSERT OR REPLACE INTO glossaries VALUES (?,?,?,?)",
            (series_key, ver + 1, json.dumps(glossary, ensure_ascii=False), time.time()),
        )
        return ver + 1

    def clear(self):
        self._q("DELETE FROM line_tl")
