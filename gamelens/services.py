"""Starts game-lens's own server (server/) and the local translation model in the background, without console
windows. Only processes started here are stopped on exit."""
from __future__ import annotations

import os
import subprocess
import threading
import time
from pathlib import Path

import httpx

CREATE_NO_WINDOW = 0x08000000
SERVER_PORT = 9960
MODELS = {
    # engine -> (port, llama serve arguments); only one fits in VRAM next to the game
    "hymt": (9943, ["-m", "{models}/Hy-MT2-7B-Q4_K_M.gguf", "--port", "9943", "-c", "8192", "--jinja",
                    "--alias", "hymt7b"]),
    "local": (9940, ["-hf", "unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q4_K_M", "--no-mmproj", "--port", "9940", "-c", "16384",
                     "--jinja", "--alias", "gemma26b"]),
}
MODEL_NAMES = {"hymt": "Hy-MT2 7B", "local": "Gemma 26B"}


def port_up(port: int, path: str = "/health") -> bool:
    try:
        return httpx.get(f"http://127.0.0.1:{port}{path}", timeout=2).status_code == 200
    except httpx.HTTPError:
        return False


class ServerManager:
    def __init__(self, root: Path, log_dir: Path, models_dir: Path | None = None):
        self.root = Path(root).resolve()
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

    def check(self, engine: str = "") -> list[str]:
        """Problems that would stop the setup from starting at all."""
        problems = []
        if not (self.root / "server" / "app.py").exists():
            problems.append(f"ไม่พบ server ของ game-lens ที่ {self.root / 'server'}")
        elif not (self.root / ".venv" / "Scripts" / "python.exe").exists():
            problems.append("game-lens ยังไม่ได้ติดตั้ง (.venv)")
        if engine == "hymt" and not self.model_up("hymt") and not (self.models / "Hy-MT2-7B-Q4_K_M.gguf").exists():
            problems.append(f"ไม่พบไฟล์โมเดล {self.models / 'Hy-MT2-7B-Q4_K_M.gguf'} — เลือกเครื่องแปลอื่น "
                            "หรือแก้ models_dir ใน settings.json")
        return problems

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
        args = [a.replace("{models}", self.models.as_posix()) for a in args]
        self._spawn(engine, ["llama", "serve", *args], self.root)

    def start_server(self):
        if self.server_up():
            return
        env = dict(os.environ, LLM_URL="http://127.0.0.1:9940/v1", LLM_MODEL="gemma26b",
                   HYMT_URL="http://127.0.0.1:9943", PYTHONIOENCODING="utf-8")
        py = self.root / ".venv" / "Scripts" / "python.exe"
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
            if engine in MODELS and not self.model_up(engine):
                say("starting", f"กำลังโหลด {MODEL_NAMES[engine]} (ครั้งแรกอาจใช้เวลาสักครู่)…")
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
