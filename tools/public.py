"""What a build leaves out or adds (tools/build_portable.py and tools/make_zip.py use it).

"Google free" uses Google's unofficial endpoints with the API key built into Chrome, as LunaTranslator and others do;
builds include it. `--no-google-free` makes a build without it: the key taken out of the source and a PUBLIC_BUILD
file that hides the engine in the app.
"""
from __future__ import annotations

import importlib.metadata as md
import re
from pathlib import Path

KEY_RE = re.compile(rb'KEY = "AIza[0-9A-Za-z_\-]+"')
MARKER = "PUBLIC_BUILD"
MARKER_TEXT = "This copy of game-lens leaves out the unofficial \"Google free\" engine.\r\n".encode("utf-8")


def public_file(name: str, data: bytes) -> bytes:
    """A file of the app as it goes into a public build."""
    if name == "server/translate/web_engines.py":
        data, n = KEY_RE.subn(b'KEY = ""', data)
        assert n == 1, "Chrome's key not found in web_engines.py: check tools/public.py"
    return data


def has_key(data: bytes) -> bool:
    return b"AIza" in data


def package_licenses(site: Path) -> bytes:
    """Name, version and license of every Python package that goes along (in `site`, from their own metadata)."""
    rows = []
    for d in md.distributions(path=[str(site)]):
        m = d.metadata
        lic = m.get("License-Expression") or ""
        if not lic:
            classes = [c.split("::")[-1].strip() for c in (m.get_all("Classifier") or []) if c.startswith("License ::")]
            lic = "; ".join(classes) or (m.get("License") or "").split("\n")[0][:80]
        rows.append(f"{m['Name']:32} {d.version:16} {lic}")
    head = ("Python packages in this copy of game-lens and their licenses (each package's own license text is in its\r\n"
            "python/Lib/site-packages/<name>-<version>.dist-info folder). See THIRD_PARTY_NOTICES.md.\r\n\r\n")
    return (head + "\r\n".join(sorted(rows, key=str.lower)) + "\r\n").encode("utf-8")
