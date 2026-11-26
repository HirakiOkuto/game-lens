"""Starts game-lens's own server (server/) and the local translation model in the background, without console
windows. Only processes started here are stopped on exit."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx

CREATE_NO_WINDOW = 0x08000000
SERVER_PORT = 9960
MODELS = {
    # engine -> (port, llama serve arguments); only one fits in VRAM next to the game
    "hymt": (9943, ["-m", "{models}/{hymt}", "--port", "9943", "-c", "8192", "--jinja", "--alias", "hymt"]),
    "index": (9944, ["-m", "{models}/Index-Translate-9B.Q4_K_M.gguf", "--port", "9944", "-c", "8192", "--jinja",
                     "--alias", "index"]),
    "local": (9940, ["-hf", "unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q4_K_M", "--no-mmproj", "--port", "9940", "-c", "16384",
                     "--jinja", "--alias", "gemma26b"]),
}
MODEL_NAMES = {"hymt": "Hy-MT2", "index": "Index-Translate 9B", "local": "Gemma 26B"}
INDEX_FILE = ("Index-Translate-9B.Q4_K_M.gguf",
              "https://huggingface.co/IndexTeam/Index-Translate-9B-GGUF/resolve/main/Index-Translate-9B.Q4_K_M.gguf")
# Hy-MT2 comes in two sizes: (name, file, where it is downloaded from by itself when it is not there)
HYMT = {
    "7b": ("Hy-MT2 7B", "Hy-MT2-7B-Q4_K_M.gguf",
           "https://huggingface.co/tencent/Hy-MT2-7B-GGUF/resolve/main/Hy-MT2-7B-Q4_K_M.gguf"),
    "1.8b": ("Hy-MT2 1.8B", "Hy-MT2-1.8B-Q4_K_M.gguf",
             "https://huggingface.co/tencent/Hy-MT2-1.8B-GGUF/resolve/main/Hy-MT2-1.8B-Q4_K_M.gguf"),
}
BIG_CARD_GB = 10  # 7B takes about 5.5 GB of graphics memory, and the game wants its share too
SMALL_CARD_GB = 3  # less than that (or an integrated chip): the model runs on the processor


def vram_gb() -> float:
    """Memory of the biggest graphics card, from what its driver writes in the registry (any maker). 0 if none."""
    try:
        import winreg

        best = 0
        key = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key) as k:
            for i in range(64):
                try:
                    sub = winreg.EnumKey(k, i)
                except OSError:
                    break
                try:
                    with winreg.OpenKey(k, sub) as s:
                        best = max(best, int(winreg.QueryValueEx(s, "HardwareInformation.qwMemorySize")[0]))
                except (OSError, ValueError, TypeError):
                    continue
        return best / 2**30
    except OSError:
        return 0.0


def pick_hymt(setting: str, vram: float) -> tuple[str, bool]:
    """(size, on the graphics card?) for the setting 'auto' / '7b' / '1.8b' and a card with `vram` GB."""
    gpu = vram >= SMALL_CARD_GB
    if setting in HYMT:
        return setting, gpu
    return ("7b" if vram >= BIG_CARD_GB else "1.8b"), gpu


def port_up(port: int, path: str = "/health") -> bool:
    try:
        return httpx.get(f"http://127.0.0.1:{port}{path}", timeout=2).status_code == 200
    except httpx.HTTPError:
        return False


class ServerManager:
    def __init__(self, root: Path, log_dir: Path, models_dir: Path | None = None, hymt_size: str = "auto"):
        self.root = Path(root).resolve()
        self.vram = vram_gb()
        self.hymt, self.hymt_gpu = pick_hymt(hymt_size, self.vram)
        self.models = Path(models_dir or self.root / "models")  # folder with the .gguf translation models
        self.logs = Path(log_dir)
        self.logs.mkdir(parents=True, exist_ok=True)
        self.procs: dict[str, subprocess.Popen] = {}
        self.lock = threading.Lock()
        self.state = "idle"  # idle / starting / ready / error
        self.detail = ""

    # --- queries ----------------------------------------------------------------------------------------------
    def server_up(self) -> bool:
        return port_up(SERVER_PORT, "/ping")  # /health probes every engine and can take seconds

    def model_up(self, engine: str) -> bool:
        return engine in MODELS and port_up(MODELS[engine][0])

    def python(self) -> Path:
        """The interpreter for the server: the one running game-lens (its .venv, or the portable build's own)."""
        exe = Path(sys.executable)
        console = exe.with_name("python.exe")
        if console.exists():
            return console
        return self.root / ".venv" / "Scripts" / "python.exe"

    def llama(self) -> str | None:
        """llama.cpp: the portable build's own copy, or one installed on this computer."""
        own = self.root / "llama" / "llama.exe"
        return str(own) if own.exists() else shutil.which("llama")

    def check(self, engine: str = "") -> list[str]:
        """Problems that would stop the setup from starting at all."""
        problems = []
        if not (self.root / "server" / "app.py").exists():
            problems.append(f"ไม่พบ server ของ game-lens ที่ {self.root / 'server'}")
        elif not self.python().exists():
            problems.append("game-lens ยังไม่ได้ติดตั้ง (.venv)")
        if engine in MODELS and not self.model_up(engine) and not self.llama():
            problems.append("ไม่พบโปรแกรม llama สำหรับโมเดลในเครื่อง — ติดตั้ง Llama for Windows หรือเลือกเครื่องแปลออนไลน์")
        return problems

    def set_hymt(self, setting: str) -> bool:
        """Choose the Hy-MT2 size again (auto / 7b / 1.8b). True when that changes which model runs; one this
        process started is stopped, so that the next ensure() starts (and if need be downloads) the other."""
        new = pick_hymt(setting, self.vram)
        if new == (self.hymt, self.hymt_gpu):
            return False
        self.hymt, self.hymt_gpu = new
        if "hymt" in self.procs:
            self.stop("hymt")
        return True

    def model_name(self, engine: str) -> str:
        """The model this engine runs here, as the user sees it ("Hy-MT2 1.8B (การ์ดจอ 8 GB)")."""
        if engine != "hymt":
            return MODEL_NAMES.get(engine, engine)
        where = f"การ์ดจอ {self.vram:.0f} GB" if self.hymt_gpu else "ใช้ CPU เพราะไม่มีการ์ดจอแยก"
        return f"{HYMT[self.hymt][0]} ({where})"

    def download_model(self, engine: str, say) -> bool:
        """Fetch the engine's model file if it is missing (resumes a broken download). say(text) reports progress."""
        if engine == "hymt":
            _, name, url = HYMT[self.hymt]
        elif engine == "index":
            name, url = INDEX_FILE
        else:
            return True
        dest = self.models / name
        if dest.exists():
            return True
        part = dest.with_name(name + ".part")
        try:
            self.models.mkdir(parents=True, exist_ok=True)
            have = part.stat().st_size if part.exists() else 0
            headers = {"Range": f"bytes={have}-"} if have else {}
            with httpx.stream("GET", url, headers=headers, follow_redirects=True, timeout=60) as r:
                if r.status_code == 200:
                    have = 0  # the server sends it all again
                elif r.status_code != 206:
                    raise RuntimeError(f"HTTP {r.status_code}")
                total = have + int(r.headers.get("content-length", 0))
                shown = -1
                with open(part, "ab" if have else "wb") as f:
                    for chunk in r.iter_bytes(1 << 20):
                        f.write(chunk)
                        have += len(chunk)
                        pct = int(100 * have / total) if total else 0
                        if pct != shown:
                            shown = pct
                            say(f"กำลังดาวน์โหลดโมเดลแปล {pct}% ({have / 1e9:.1f} / {total / 1e9:.1f} GB) ครั้งแรกครั้งเดียว")
            part.replace(dest)
            return True
        except (httpx.HTTPError, OSError, RuntimeError) as e:
            say(f"ดาวน์โหลดโมเดลไม่สำเร็จ ({e}) — เปิด game-lens ใหม่เพื่อโหลดต่อ หรือเลือกเครื่องแปลออนไลน์")
            return False

    # --- start / stop -----------------------------------------------------------------------------------------
    def _spawn(self, name: str, args: list[str], cwd: Path, env: dict | None = None) -> subprocess.Popen:
        log = open(self.logs / f"{name}.log", "ab")
        p = subprocess.Popen(args, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL, creationflags=CREATE_NO_WINDOW)
        self.procs[name] = p
        return p

    def start_model(self, engine: str):
        """Start the local model this engine needs; stop the other one if we started it (VRAM)."""
        if engine not in MODELS or self.model_up(engine):
            return
        for other in MODELS:
            if other != engine and other in self.procs:
                self.stop(other)
        port, args = MODELS[engine]
        args = [a.replace("{models}", self.models.as_posix()).replace("{hymt}", HYMT[self.hymt][1]) for a in args]
        if engine == "hymt" and not self.hymt_gpu:
            args += ["-ngl", "0"]  # no graphics card worth the name: all on the processor
        self._spawn(engine, [self.llama() or "llama", "serve", *args], self.root)

    def start_server(self):
        if self.server_up():
            return
        env = dict(os.environ, LLM_URL="http://127.0.0.1:9940/v1", LLM_MODEL="gemma26b",
                   HYMT_URL="http://127.0.0.1:9943", INDEX_URL="http://127.0.0.1:9944", PYTHONIOENCODING="utf-8")
        env["PYTHONNOUSERSITE"] = "1"
        py = self.python()
        self._spawn("server", [str(py), "-m", "uvicorn", "app:app", "--host", "127.0.0.1", "--port", str(SERVER_PORT)],
                    self.root / "server", env)

    def ensure(self, engine: str, on_change=None, timeout_s: float = 240):
        """Blocking (run it in a worker): bring the server and the engine's model up. on_change(state, detail)."""
        with self.lock:
            def say(state, detail):
                self.state, self.detail = state, detail
                if on_change:
                    on_change(state, detail)

            problems = self.check(engine)
            if problems:
                say("error", problems[0])
                return False
            if not self.server_up():
                say("starting", "กำลังเปิดระบบแปล…")
                self.start_server()
            if engine in ("hymt", "index") and not self.model_up(engine) and not self.download_model(engine, lambda t: say("starting", t)):
                say("error", self.detail)
                return False
            if engine in MODELS and not self.model_up(engine):
                say("starting", f"กำลังโหลด {self.model_name(engine)} (ครั้งแรกอาจใช้เวลาสักครู่)…")
                self.start_model(engine)
            t0 = time.monotonic()
            while time.monotonic() - t0 < timeout_s:
                srv = self.server_up()
                mdl = engine not in MODELS or self.model_up(engine)
                if srv and mdl:
                    say("ready", "พร้อม")
                    return True
                for name, p in list(self.procs.items()):
                    if p.poll() is not None:
                        self.procs.pop(name)
                        say("error", f"{name} ปิดตัวเอง (ดู logs/{name}.log)")
                        return False
                time.sleep(1)
            say("error", "เปิดระบบแปลไม่ทันเวลา (ดู logs/server.log)")
            return False

    def stop(self, name: str | None = None):
        for n in [name] if name else list(self.procs):
            p = self.procs.pop(n, None)
            if p and p.poll() is None:
                p.terminate()
                try:
                    p.wait(5)
                except subprocess.TimeoutExpired:
                    p.kill()
