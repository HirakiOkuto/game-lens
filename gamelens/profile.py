"""Per-game profile (which window, which screen regions, which engine) and global settings, as JSON files."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KINDS = ("dialog", "name", "subtitle", "choice")
KIND_LABELS = {"dialog": "บทพูด (กล่องข้อความ)", "name": "ชื่อผู้พูด", "subtitle": "ซับระหว่างเล่น",
               "choice": "ตัวเลือกคำตอบ"}
# engines the server knows, in the order the user sees them; remote ones send text off this machine
ENGINE_INFO = [
    {"key": "hymt", "name": "Hy-MT2 — ในเครื่อง (แนะนำ)",
     "desc": "เร็ว ฟรี ไม่ต้องใช้เน็ต • เลือกรุ่นตามการ์ดจอให้เอง: 7B (การ์ดจอ 10 GB ขึ้นไป) หรือ 1.8B", "keys": []},
    {"key": "local", "name": "Gemma 26B — ในเครื่อง",
     "desc": "แปลสวยที่สุดในเครื่อง เข้าใจบริบท • ใช้การ์ดจอมาก อาจทำให้เกมกระตุก", "keys": []},
    {"key": "google_free", "name": "Google — ฟรี ไม่ต้องสมัคร",
     "desc": "ไม่ใช้การ์ดจอ • ช่องทางไม่เป็นทางการ อาจถูกจำกัดชั่วคราวถ้าใช้เยอะ", "keys": []},
    {"key": "deepl", "name": "DeepL",
     "desc": "ฟรี 500,000 ตัวอักษร/เดือน (สมัคร DeepL API Free) • ไม่ใช้การ์ดจอ",
     "keys": [("DEEPL_API_KEY", "API key")], "signup": "https://www.deepl.com/pro-api"},
    {"key": "azure", "name": "Microsoft Azure",
     "desc": "ฟรี 2,000,000 ตัวอักษร/เดือน (แผน F0) • ไม่ใช้การ์ดจอ",
     "keys": [("AZURE_TRANSLATOR_KEY", "Key"), ("AZURE_TRANSLATOR_REGION", "Region (เช่น southeastasia)")],
     "signup": "https://portal.azure.com/#create/Microsoft.CognitiveServicesTextTranslation"},
    {"key": "google", "name": "Google Cloud",
     "desc": "ฟรี 500,000 ตัวอักษร/เดือน • ไม่ใช้การ์ดจอ",
     "keys": [("GOOGLE_TRANSLATE_API_KEY", "API key")],
     "signup": "https://console.cloud.google.com/apis/library/translate.googleapis.com"},
    {"key": "cloud", "name": "Claude",
     "desc": "แปลสวยและเข้าใจบริบทมาก • คิดเงินตามการใช้งาน",
     "keys": [("ANTHROPIC_API_KEY", "API key")], "signup": "https://console.anthropic.com/"},
]
ENGINE_BY_KEY = {e["key"]: e for e in ENGINE_INFO}
ENGINES = {e["key"]: e["name"] for e in ENGINE_INFO}
REMOTE = {"google_free", "google", "deepl", "azure", "cloud"}


@dataclass
class Region:
    id: str
    kind: str = "dialog"
    rect: list[float] = field(default_factory=lambda: [0.2, 0.75, 0.6, 0.2])  # x, y, w, h as fractions of the client area
    ink: str = "light"  # text colour class for change detection: light / dark / any
    align: str = "left"  # left / center


@dataclass
class Profile:
    name: str = "game"
    series: str = "game"  # glossary key on the server
    exe: str = ""
    title: str = ""
    src_lang: str = "ja"
    target: str = "th"
    engine: str = "hymt"
    font_family: str = "Leelawadee UI"
    font_scale: float = 0.9  # Thai size relative to the Japanese line height
    # how the player's own character speaks (answers, lines without a name): neutral = no ครับ / ค่ะ at all, so it fits
    # whoever plays; female / male; "" = as translated
    player_gender: str = "neutral"
    regions: list[Region] = field(default_factory=list)
    path: str = ""

    @classmethod
    def load(cls, path: Path) -> "Profile":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        d["regions"] = [Region(**r) for r in d.get("regions", [])]
        d.pop("path", None)
        p = cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        p.path = str(path)
        return p

    def save(self):
        d = asdict(self)
        d.pop("path")
        Path(self.path).write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")


@dataclass
class Settings:
    server: str = "http://127.0.0.1:9960"
    token_file: str = "server/data/token.txt"
    profile: str = "profiles/azur-promilia.json"
    hotkeys: dict = field(default_factory=lambda: {"toggle": "Ctrl+F8", "screen": "F9", "regions": "F10", "history": "F7"})
    peek_key: str = "F8"  # hold to see the original text (polled, not captured: the game still gets the key)
    context_lines: int = 6
    models_dir: str = "D:/AI/models"  # shared folder with the .gguf translation models
    local_model: str = "auto"  # Hy-MT2 size: auto (by the graphics card's memory) / 7b / 1.8b
    auto_start_server: bool = True  # start the server + local model when game-lens starts
    stop_server_on_exit: bool = True  # ...and stop what we started when it quits
    start_hidden: bool = False  # open only the tray icon once setup is complete
    toasts: bool = True  # short status messages on the game screen
    auto_screen: bool = False  # translate the whole screen by itself when it stands still (menus, quests)

    def save(self, path: Path = ROOT / "settings.json"):
        path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path = ROOT / "settings.json") -> "Settings":
        if not path.exists():
            s = cls()
            path.write_text(json.dumps(asdict(s), ensure_ascii=False, indent=2), encoding="utf-8")
            return s
        d = json.loads(path.read_text(encoding="utf-8"))
        s = cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        s.server = os.environ.get("GAMELENS_SERVER", s.server)  # tests and tools can point at another server
        return s

    def resolve(self, rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else (ROOT / p).resolve()
