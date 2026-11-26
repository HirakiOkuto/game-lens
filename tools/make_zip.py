r"""Build a zip to give game-lens to someone else: what is committed, without tests, without anything of this
computer (keys, token, cache, logs, the Python environment). The model folder is the app's own `models/`.

    .venv\Scripts\python.exe tools\make_zip.py            -> dist\game-lens-<date>.zip
"""
from __future__ import annotations

import io
import json
import subprocess
import sys
import zipfile
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from public import MARKER, MARKER_TEXT, has_key, public_file  # noqa: E402
LEAVE_OUT = ("tests/", "server/tests/", "tools/make_zip.py", "experiments/")
PERSONAL = "--personal" in sys.argv  # keep "Google free" (Chrome's key): for your own computers only


def main():
    if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", ".", ":!settings.json"], cwd=ROOT).returncode:
        print("note: uncommitted changes are not in the zip (it is built from the last commit)")
    raw = subprocess.run(["git", "archive", "--format=zip", "HEAD"], cwd=ROOT, capture_output=True, check=True).stdout
    out = ROOT / "dist" / f"game-lens-{'personal-' if PERSONAL else ''}{date.today():%Y%m%d}.zip"
    out.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(raw)) as src, zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as dst:
        for info in src.infolist():
            name = info.filename
            if info.is_dir() or name.startswith(LEAVE_OUT):
                continue
            data = src.read(info)
            if name == "settings.json":  # models in the app's own folder, not this computer's shared one
                s = json.loads(data)
                s["models_dir"] = "models"
                data = json.dumps(s, ensure_ascii=False, indent=2).encode("utf-8")
            if not PERSONAL:
                data = public_file(name, data)
            dst.writestr("game-lens/" + name, data)
        if not PERSONAL:
            dst.writestr("game-lens/" + MARKER, MARKER_TEXT)
        dst.writestr("game-lens/models/วางไฟล์โมเดล .gguf ไว้ที่นี่.txt",
                     "game-lens ดาวน์โหลดโมเดลแปลภาษามาไว้ที่นี่เองเมื่อเปิดใช้ครั้งแรก\r\n".encode("utf-8"))
    names = zipfile.ZipFile(out).namelist()
    bad = [n for n in names if any(k in n for k in ("token", "keys.json", ".sqlite", ".venv/", "logs/"))]
    assert not bad, bad
    if not PERSONAL:
        assert not has_key(zipfile.ZipFile(out).read("game-lens/server/translate/web_engines.py")), "Chrome's key left in"
    print(f"{out}  ({out.stat().st_size / 1e6:.1f} MB, {len(names)} files)")


if __name__ == "__main__":
    sys.exit(main())
