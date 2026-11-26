"""Per-game profile (which window, which screen regions, which engine) and global settings, as JSON files."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# languages a game can be in (code for the server, name in the window)
GAME_LANGS = [("ja", "ญี่ปุ่น"), ("ko", "เกาหลี"), ("zh", "จีนตัวย่อ (จีนแผ่นดินใหญ่)"), ("zh-tw", "จีนตัวเต็ม (ไต้หวัน / ฮ่องกง)"),
              ("en", "อังกฤษ")]
KINDS = ("dialog", "name", "subtitle", "choice")
KIND_LABELS = {"dialog": "บทพูด (กล่องข้อความ)", "name": "ชื่อผู้พูด", "subtitle": "ซับระหว่างเล่น",
               "choice": "ตัวเลือกคำตอบ"}
# engines the server knows, in the order the user sees them; remote ones send text off this machine
ENGINE_INFO = [
    {"key": "hymt", "name": "โมเดลแปลในเครื่อง (แนะนำ)", "desc": "ฟรี ไม่ต้องใช้เน็ต • เลือกรุ่นตามการ์ดจอให้เอง", "keys": []},
    {"key": "google_free", "name": "Google — ฟรี ไม่ต้องสมัคร", "desc": "ฟรี • ไม่ใช้การ์ดจอ", "keys": []},
    {"key": "deepl", "name": "DeepL", "desc": "ฟรี 500,000 ตัวอักษร/เดือน • ต้องสมัคร",
     "keys": [("DEEPL_API_KEY", "API key")], "signup": "https://www.deepl.com/pro-api"},
    {"key": "azure", "name": "Microsoft Azure", "desc": "ฟรี 2,000,000 ตัวอักษร/เดือน • ต้องสมัคร",
     "keys": [("AZURE_TRANSLATOR_KEY", "Key"), ("AZURE_TRANSLATOR_REGION", "Region (เช่น southeastasia)")],
     "signup": "https://portal.azure.com/#create/Microsoft.CognitiveServicesTextTranslation"},
    {"key": "google", "name": "Google Cloud", "desc": "ฟรี 500,000 ตัวอักษร/เดือน • ต้องสมัคร",
     "keys": [("GOOGLE_TRANSLATE_API_KEY", "API key")],
     "signup": "https://console.cloud.google.com/apis/library/translate.googleapis.com"},
    {"key": "cloud", "name": "Claude", "desc": "แปลสวยที่สุด • คิดเงินตามการใช้",
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
    live_engine: str = field(default="", repr=False, compare=False)  # set while running, never saved (see save)

    @classmethod
    def load(cls, path: Path) -> "Profile":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        d["regions"] = [Region(**r) for r in d.get("regions", [])]
        d.pop("path", None)
        d.pop("live_engine", None)
        p = cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        p.path = str(path)
        return p

    def save(self):
        d = asdict(self)
        d.pop("path")
        d.pop("live_engine")
        Path(self.path).write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")


def guess_lang(text: str, default: str = "ja") -> str:
    """The game's language from its window title: kana is Japanese, hangul Korean, Chinese characters alone Chinese
    (traditional when it has letters only traditional Chinese uses)."""
    import re

    if re.search(r"[\u3040-\u30ff]", text):
        return "ja"
    if re.search(r"[\uac00-\ud7af]", text):
        return "ko"
    if re.search(r"[\u4e00-\u9fff]", text):
        return "zh-tw" if re.search(r"[們個這會來說為與對時開發戰鬥國學點從後裡]", text) else "zh"
    return default


def find_profile(exe: str, folder: Path = ROOT / "profiles") -> Path | None:
    """The profile of the game whose program is `exe`, if there is one."""
    exe = (exe or "").lower()
    for p in sorted(folder.glob("*.json")):
        try:
            if exe and json.loads(p.read_text(encoding="utf-8")).get("exe", "").lower() == exe:
                return p
        except (OSError, ValueError):
            continue
    return None


def new_profile(exe: str, title: str, base: "Profile", folder: Path = ROOT / "profiles") -> "Profile":
    """A profile for a game game-lens has not seen: named after its program, the engine and display settings of
    the one in use, no regions yet, the language guessed from the window title."""
    import re

    stem = Path(exe).stem if exe else (title or "game")
    slug = re.sub(r"[^a-z0-9]+", "-", stem.lower()).strip("-") or "game"
    path, n = folder / f"{slug}.json", 2
    while path.exists():
        path, n = folder / f"{slug}-{n}.json", n + 1
    p = Profile(name=(title or stem)[:60], series=slug, exe=exe, src_lang=guess_lang(title or "", base.src_lang),
                target=base.target, engine=base.engine, font_family=base.font_family, font_scale=base.font_scale,
                player_gender=base.player_gender, path=str(path))
    folder.mkdir(parents=True, exist_ok=True)
    p.save()
    return p


@dataclass
class Settings:
    server: str = "http://127.0.0.1:9960"
    token_file: str = "server/data/token.txt"
    profile: str = "profiles/azur-promilia.json"
    hotkeys: dict = field(default_factory=lambda: {"toggle": "Ctrl+F8", "screen": "F9", "regions": "F10", "history": "F7"})
    peek_key: str = "F8"  # hold to see the original text (polled, not captured: the game still gets the key)
    context_lines: int = 6
    models_dir: str = "D:/AI/models"  # shared folder with the .gguf translation models
    local_model: str = "auto"  # the local translator's model: auto (by the graphics card) / index / 7b / 1.8b
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
