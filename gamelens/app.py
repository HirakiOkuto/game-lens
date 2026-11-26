"""game-lens controller: finds the game window, watches its regions, asks the server for OCR + translation and
puts the Thai text in place on the overlay. One main window (Panel) plus a tray icon."""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from ctypes import wintypes
from pathlib import Path

import numpy as np
from PySide6.QtCore import QAbstractNativeEventFilter, QObject, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QIcon, QPainter, QPixmap
from PySide6.QtNetwork import QLocalServer, QLocalSocket
from PySide6.QtWidgets import QApplication, QMenu, QSystemTrayIcon

from . import win32
from .api import LensClient, LensError
from .capture import GameCapture
from .history import HistoryLog, HistoryWindow
from .i18n import set_lang, t
from .overlay import Overlay
from .panel import Panel
from .picker import GamePicker
from .profile import GAME_LANGS, find_profile, guess_lang, new_profile, ENGINE_BY_KEY, KIND_LABELS, REMOTE, ROOT, Profile, Settings
from .region_editor import RegionEditor
from .screen import (TextSize, begins, text_marks, text_state, extend_item, lines_to_translate, merge_glossary, ocr_region, paragraphs, refresh_patch, remember,
                     requests, screen_items, translate_region)
from .services import HYMT, MODELS, ServerManager, pick_local
from .ui_style import QSS
from .speech import fit_speaker, gender_of, genders
from .watcher import TextFollower, auto_pause, changed_pixels, plain_text, stroke_change, goes_on, ink_map, similar, stroke_map

HK_TOGGLE, HK_SCREEN, HK_REGIONS, HK_HISTORY = 1, 2, 3, 4
INSTANCE_KEY = os.environ.get("GAMELENS_INSTANCE", "game-lens-ui")  # one running copy per key (tests use their own)
log = logging.getLogger("gamelens")


class HotkeyFilter(QAbstractNativeEventFilter):
    def __init__(self, on_hotkey):
        super().__init__()
        self.on_hotkey = on_hotkey

    def nativeEventFilter(self, event_type, message):
        if bytes(event_type) == b"windows_generic_MSG":
            msg = wintypes.MSG.from_address(int(message))
            if msg.message == win32.WM_HOTKEY:
                self.on_hotkey(int(msg.wParam))
                return True, 0
        return False, 0


class Bridge(QObject):
    """Worker threads -> UI thread."""
    region_read = Signal(str, object, object)
    region_done = Signal(str, object, object)
    screen_done = Signal(int, object)
    message = Signal(str, str)
    status = Signal(dict)
    server_state = Signal(str, str)
    engine_test = Signal(dict)
    keys = Signal(dict)


def app_icon() -> QIcon:
    pm = QPixmap(64, 64)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setBrush(QColor("#4f8cff"))
    p.setPen(Qt.PenStyle.NoPen)
    p.drawRoundedRect(2, 2, 60, 60, 14, 14)
    f = QFont("Yu Gothic UI")
    f.setPixelSize(38)
    f.setBold(True)
    p.setFont(f)
    p.setPen(QColor("#ffffff"))
    p.drawText(pm.rect(), Qt.AlignmentFlag.AlignCenter, "訳")
    p.end()
    return QIcon(pm)


def create_shortcut(folder: str | None = None) -> str:
    """A game-lens.lnk (no console window) on the Desktop, or in `folder`. Returns an error message or ''."""
    ico = ROOT / "logs" / "game-lens-icon.ico"  # (a new file name makes Windows drop a cached old picture)
    ico.parent.mkdir(exist_ok=True)
    app_icon().pixmap(64, 64).save(str(ico), "ICO")
    exe = ROOT / "game-lens.exe"  # the portable build's launcher
    target, args = (exe, "") if exe.exists() else (ROOT / ".venv" / "Scripts" / "pythonw.exe", "-m gamelens")
    where = f"'{folder}'" if folder else "[Environment]::GetFolderPath('Desktop')"
    ps = (
        f"$d={where};"
        "$s=(New-Object -ComObject WScript.Shell).CreateShortcut(\"$d\\game-lens.lnk\");"
        f"$s.TargetPath='{target}';$s.Arguments='{args}';$s.WorkingDirectory='{ROOT}';"
        f"$s.IconLocation='{ico}';$s.Description='game-lens';$s.Save()"
    )
    r = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True,
                       creationflags=0x08000000)
    return "" if r.returncode == 0 and not r.stderr.strip() else (r.stderr.strip() or "powershell failed")


def friendly_error(err: str) -> str:
    """A short Thai message for the game screen (details stay in logs/gamelens.log)."""
    e = err.lower()
    if "10061" in e or "cannot reach the server" in e or "ไม่สามารถเชื่อมต่อ server" in err or "connecterror" in e:
        return t("The translation system is not ready — open the game-lens window to see why")
    if "429" in e or "rate limit" in e:
        return t("The online translator is limited for now (429) — try another translator")
    if "quota" in e or "456" in e:
        return t("The translator's quota for this month is used up — choose another translator in the game-lens window")
    if "403" in e or "401" in e or "bad key" in e:
        return t("The translator's key is not valid — check it in the game-lens window")
    if "http 500" in e:
        return t("The translation system ran into a problem — see logs/server.log")
    if "unreachable" in e:
        return t("The translator is not running — open the game-lens window and press Load")
    return t("Translation failed: {error}", error=err.split(": ", 1)[-1][:80])


def region_px(rect: list[float], shape) -> tuple[int, int, int, int]:
    H, W = shape[:2]
    x, y = int(rect[0] * W), int(rect[1] * H)
    w, h = int(rect[2] * W), int(rect[3] * H)
    x, y = max(0, min(x, W - 1)), max(0, min(y, H - 1))
    return x, y, max(1, min(w, W - x)), max(1, min(h, H - y))


LANG_NAMES = dict(GAME_LANGS)
STILL = 0.35  # share of the screen's strokes in parts that moved, below which its text counts as standing still


class Controller(QObject):
    def __init__(self, app: QApplication):
        super().__init__()
        self.app = app
        self.settings = Settings.load()
        set_lang(self.settings.ui_lang)
        ppath = self.settings.resolve(os.environ.get("GAMELENS_PROFILE") or self.settings.profile)
        if not ppath.exists():
            ppath.parent.mkdir(parents=True, exist_ok=True)
            Profile(name=ppath.stem, series=ppath.stem, path=str(ppath)).save()
        self.profile = Profile.load(ppath)
        self.api = LensClient(self.settings.server, self.settings.resolve(self.settings.token_file))
        if self.profile.engine == "index":  # chosen in the window before it became a size of the local translator
            self.profile.engine, self.settings.local_model = "hymt", "index"
            self.profile.save()
            self.settings.save()
        if self.profile.engine not in ENGINE_BY_KEY:  # an engine no longer offered (Gemma): the local translator
            self.profile.engine = "hymt"
            self.profile.save()
        self.services = ServerManager(ROOT, ROOT / "logs", self.settings.resolve(self.settings.models_dir),
                                      self.settings.local_model)
        self.profile.live_engine = self.services.engine_for(self.profile.engine)
        self.pool = ThreadPoolExecutor(8)  # reading and translating several regions at once, status polls, screens
        self.bridge = Bridge()
        self.bridge.region_read.connect(self.on_region_read)
        self.bridge.region_done.connect(self.on_region_done)
        self.bridge.screen_done.connect(self.on_screen_done)
        self.bridge.message.connect(self._tray_message)
        self.bridge.status.connect(self.on_status)
        self.bridge.server_state.connect(lambda s, d: self.refresh_panel())
        self.bridge.engine_test.connect(self.on_engine_test)
        self.bridge.keys.connect(lambda k: self.panel.set_keys(k))

        self.overlay = Overlay(self.profile.font_family)
        self.overlay.winId()  # native window exists from the start: hotkeys are delivered to it
        self.history = HistoryLog(ROOT / "logs", self.profile.series)
        self.history_win: HistoryWindow | None = None
        self.editor: RegionEditor | None = None

        self.hwnd: int | None = None
        self.cap: GameCapture | None = None
        self.enabled = True
        self.health: dict | None = None
        self.watchers: dict[str, TextFollower] = {}
        self.shown: dict[str, dict] = {}
        self.shown_seq: dict[str, int] = {}
        self.region_active = 0.0  # when text in a region last changed
        self.auto_empties = 0  # times in a row that auto mode found nothing to translate
        self.genders: dict[str, str] | None = None  # speaker name -> gender, from the glossary file
        self.held: dict[str, tuple] = {}  # translations that wait for a line to be read once more
        self.readings: dict[str, dict] = {}  # what was read last in each region
        self.sizes: dict[str, TextSize] = {}
        self.context: deque = deque(maxlen=self.settings.context_lines)
        self.screen_gen = 0
        self.screen_ref: np.ndarray | None = None
        self.screen_frame: np.ndarray | None = None
        self.screen_marks: list = []  # what the translated text looked like (see text_state)
        self.screen_strokes = self.screen_last = None
        self.screen_quiet = 0.0
        self.screen_menu = False  # the screen translation is of a menu: the regions keep quiet meanwhile
        self.screen_auto = False
        self.screen_texts: frozenset = frozenset()  # what the last screen translation said
        self.next_keys, self.keys_told = 0.0, False  # see keep_english_keys
        self.keys_possible = True
        self.screen_busy = False
        self.next_attach = 0.0
        self.last_error = ("", 0.0)
        self.waiting_toast = -99.0
        # automatic whole-screen translation (menus, quest lists): when the whole picture stands still
        self.auto_screen = self.settings.auto_screen or os.environ.get("GAMELENS_AUTO_SCREEN") == "1"
        self.auto_prev: np.ndarray | None = None
        self.auto_done: np.ndarray | None = None  # the screen we last translated (or tried to)
        self.auto_still_since = 0.0
        self.peek_vk = win32.parse_hotkey(self.settings.peek_key)[1]
        self.always_show = os.environ.get("GAMELENS_ALWAYS_SHOW") == "1"  # debug: ignore which window has focus
        self.reset_watchers()

        # windows
        self.build_panel()

        self.tray = QSystemTrayIcon(app_icon())
        self.tray.setToolTip("game-lens")
        self.tray.activated.connect(lambda reason: self.show_panel() if reason == QSystemTrayIcon.ActivationReason.Trigger else None)
        self.build_menu()
        self.tray.show()

        self.hotkeys = HotkeyFilter(self.on_hotkey)
        app.installNativeEventFilter(self.hotkeys)
        hk = self.settings.hotkeys
        hwnd = int(self.overlay.winId())
        for hid, name in ((HK_TOGGLE, "toggle"), (HK_SCREEN, "screen"), (HK_REGIONS, "regions"), (HK_HISTORY, "history")):
            if hk.get(name) and not win32.register_hotkey(hwnd, hid, hk[name]):
                self.notify("game-lens", t("The key {key} is used by another program (change it in settings.json)", key=hk[name]))

        self.timer = QTimer(self, interval=50, timeout=self.tick)
        self.timer.start()
        self.status_timer = QTimer(self, interval=2000, timeout=lambda: self.pool.submit(self.poll_status))
        self.status_timer.start()
        self.pool.submit(self.poll_status)
        if self.settings.auto_start_server:
            self.start_services()
        self.refresh_panel()
        if not (self.settings.start_hidden and self.setup_complete()):
            self.show_panel()

    # --- small helpers ----------------------------------------------------------------------------------------
    def setup_complete(self) -> bool:
        return bool((self.profile.exe or self.profile.title) and self.profile.regions)

    def build_panel(self):
        """The main window, in the app's language (made again when that changes)."""
        self.panel = Panel(self.settings.hotkeys, self.settings.peek_key)
        self.panel.setWindowIcon(app_icon())
        pick = pick_local("auto", self.services.vram)[0]
        auto = "Index-Translate 9B" if pick == "index" else HYMT[pick][0]
        where = t("graphics card {gb} GB", gb=f"{self.services.vram:.0f}") if self.services.vram >= 1 else t("no graphics card")
        self.auto_label = f"{auto}, {where}"
        self.load_panel()
        self.panel.model_size_changed.connect(self.set_model_size)
        self.panel.auto_screen_changed.connect(self.set_auto_screen)
        self.panel.toggle.connect(self.toggle)
        self.panel.fix_server.connect(self.start_services)
        self.panel.pick_game.connect(self.pick_window)
        self.panel.edit_regions.connect(self.edit_regions)
        self.panel.engine_changed.connect(self.set_engine)
        self.panel.save_keys.connect(lambda v: self.pool.submit(self._save_keys, v))
        self.panel.test_engine.connect(lambda: self.pool.submit(self._test_engine))
        self.panel.font_scale_changed.connect(self.set_font_scale)
        self.panel.target_changed.connect(lambda code: self.set_profile(target=code))
        self.panel.src_lang_changed.connect(self.set_src_lang)
        self.panel.toasts_changed.connect(self.set_toasts)
        self.panel.english_keys_changed.connect(self.set_english_keys)
        self.panel.show_history.connect(self.show_history)
        self.panel.make_shortcut.connect(self.make_shortcut)
        self.panel.quit_app.connect(self.quit)
        self.panel.ui_lang_changed.connect(self.set_ui_lang)

    def set_ui_lang(self, code: str):
        self.settings.ui_lang = code
        self.settings.save()
        set_lang(code)
        old, geo, shown = self.panel, self.panel.geometry(), self.panel.isVisible()
        self.build_panel()
        self.panel.setGeometry(geo)
        old.hide()
        old.deleteLater()
        self.build_menu()
        self.refresh_panel()
        if shown:
            self.panel.show()

    def show_panel(self):
        if not self.panel.isVisible() and not getattr(self, "_placed", False):
            self._placed = True
            scr = self.app.primaryScreen().availableGeometry()
            h = min(self.panel.sizeHint().height(), scr.height() - 60)
            self.panel.resize(self.panel.sizeHint().width(), h)
            self.panel.move(scr.right() - self.panel.width() - 24, scr.top() + 24)  # beside the game, not over it
        self.panel.show()
        self.panel.raise_()
        self.panel.activateWindow()

    def notify(self, title: str, msg: str):
        self.bridge.message.emit(title, msg)

    def _tray_message(self, title: str, msg: str):
        self.tray.showMessage(title, msg, QSystemTrayIcon.MessageIcon.Information, 4000)

    def toast(self, text: str, seconds: float = 2.5):
        if self.settings.toasts:
            self.overlay.toast(text, seconds)

    def build_menu(self):
        m = QMenu()
        m.addAction(t("Open the game-lens window"), self.show_panel)
        self.toggle_action = m.addAction(t("Stop translating"), self.toggle)
        m.addAction(t("Translate the whole screen"), self.translate_screen)
        m.addAction(t("Set text regions"), self.edit_regions)
        m.addAction(t("Translation history"), self.show_history)
        m.addSeparator()
        m.addAction(t("Quit"), self.quit)
        self.menu = m
        self.tray.setContextMenu(m)

    # --- translation system (server + model) ------------------------------------------------------------------
    def start_services(self):
        engine = self.profile.live_engine
        threading.Thread(target=self.services.ensure, args=(engine, lambda s, d: self.bridge.server_state.emit(s, d)),
                         kwargs={"ocr_lang": self.profile.src_lang},
                         daemon=True).start()

    def poll_status(self):
        st = {"server": False, "health": None}
        try:
            st["health"] = self.api.health()
            st["server"] = True
            if not getattr(self, "_keys_loaded", False):
                self._keys_loaded = True  # first: two polls can overlap while the server is slow to answer
                self.bridge.keys.emit(self.api.keys())
                self.sync_glossary()
        except LensError:
            self._keys_loaded = False
        self.bridge.status.emit(st)

    def fit_speaker(self, payload: dict):
        """Polite particles of the speaker's gender (the model cannot tell who speaks): the name on screen now. The
        player's own character answers (choices) and speaks lines that come without a name (or with the player's
        own name, which is not in Japanese)."""
        if payload.get("kind") not in ("dialog", "subtitle", "choice") or not payload.get("tl"):
            return
        if self.genders is None:
            path = ROOT / "glossaries" / f"{self.profile.series}.{self.profile.target}.json"
            try:
                self.genders = genders(json.loads(path.read_text(encoding="utf-8"))) if path.exists() else {}
            except (ValueError, OSError):
                self.genders = {}
        names = [self.watchers[r.id].last_text for r in self.profile.regions if r.kind == "name" and r.id in self.watchers]
        if payload["kind"] == "choice" or not any(names):
            gender = self.profile.player_gender
        else:
            gender = next((g for n in names if (g := gender_of(n, self.genders))), "")
        if gender:
            payload["tl"] = fit_speaker(payload["tl"], gender)
            for it in payload.get("items", []):
                it.text = fit_speaker(it.text, gender)

    def sync_glossary(self):
        """Names and terms kept in glossaries/<series>.<target>.json go to the server whenever it comes up."""
        p = self.profile
        path = ROOT / "glossaries" / f"{p.series}.{p.target}.json"
        if not path.exists():
            return
        try:
            base = json.loads(path.read_text(encoding="utf-8"))
            current = self.api.glossary(p.series, p.target)
            merged = merge_glossary(base, current)
            if merged != current:
                ver = self.api.put_glossary(p.series, p.target, merged)
                log.info("glossary %s: %d names, %d terms (version %d)", path.name, len(merged["characters"]),
                         len(merged["terms"]), ver)
        except (LensError, ValueError, OSError) as e:
            log.warning("glossary %s not loaded: %s", path.name, e)

    def on_status(self, st: dict):
        self.health = st["health"]
        self.refresh_panel()

    def engine_ready(self) -> bool:
        return bool(self.health and self.health.get(self.profile.live_engine))

    def refresh_panel(self):
        p, eng = self.profile, self.profile.live_engine
        info = ENGINE_BY_KEY[self.profile.engine]
        # 1. translation system
        svc = self.services
        if self.health and self.health.get(eng):
            self.panel.s_server.set("ok", t("Ready — {name}", name=svc.model_name(eng) if eng in MODELS else t(info["name"])))
        elif svc.state == "starting":
            self.panel.s_server.set("warn", svc.detail)
        elif not self.health:
            msg = svc.detail if svc.state == "error" else t("The translation system is not running")
            self.panel.s_server.set("bad", msg, t("Start"), primary=True)
        elif not self.health.get(eng):
            if eng in MODELS:
                self.panel.s_server.set("warn", t("{name} is not loaded yet", name=svc.model_name(eng)), t("Load"), primary=True)
            elif info.get("keys"):
                self.panel.s_server.set("warn", t("Enter the key for {name} (below)", name=t(info["name"])))
            else:
                self.panel.s_server.set("warn", t("{name} is not ready", name=t(info["name"])))
        else:
            self.panel.s_server.set("ok", t("Ready — {name}", name=t(info["name"])))
        # 2. game
        if not (p.exe or p.title):
            self.panel.s_game.set("bad", t("No game window chosen yet"), t("Choose game"), primary=True)
        elif self.hwnd is None:
            self.panel.s_game.set("warn", t("Waiting for the game… ({game}) • set the game to Borderless / Windowed", game=p.exe or p.title), t("Change"))
        else:
            self.panel.s_game.set("ok", f"{win32.window_title(self.hwnd)[:50] or p.exe} • {t(LANG_NAMES.get(p.src_lang, p.src_lang))}",
                                  t("Change"))
        # 3. regions
        if not p.regions:
            if self.hwnd is None:
                self.panel.s_regions.set("idle", t("Open the game first, then go to a screen with dialogue"))
            else:
                self.panel.s_regions.set("bad", t("Go to a screen with dialogue, then press Set regions"), t("Set regions"), primary=True)
        else:
            kinds = ", ".join(sorted({t(KIND_LABELS[r.kind]).split(" (")[0] for r in p.regions}))
            self.panel.s_regions.set("ok", t("{n} regions: {kinds}", n=len(p.regions), kinds=kinds), t("Edit") if self.hwnd else None)
        ready = bool(self.health) and self.hwnd is not None and bool(p.regions)
        if self.enabled and ready:
            hint = t("Translating — you can switch to the game")
        elif not self.enabled:
            hint = t("Stopped")
        else:
            hint = t("Complete the steps that are not green yet; translating then starts by itself")
        self.panel.set_running(self.enabled, ready, hint)
        self.toggle_action.setText(t("Stop translating") if self.enabled else t("Start translating"))
        self.tray.setToolTip("game-lens — " + hint)

    def set_model_size(self, size: str):
        """The local translator's model picked in the window: kept in settings.json, and swapped now."""
        self.settings.local_model = size
        self.settings.save()
        if not self.services.set_hymt(size):
            self.refresh_panel()
            return
        self.profile.live_engine = self.services.engine_for(self.profile.engine)
        self.shown.clear()
        if self.services.model_up(self.profile.live_engine):  # still the old one: started outside game-lens (e.g. Llama for Windows)
            self.panel.set_engine_status(t("The old model is still running outside game-lens — close that program first, "
                                           "then press Start in step 1"), "warn")
        elif self.profile.engine == "hymt" and self.settings.auto_start_server:  # the local translator
            self.start_services()
        self.refresh_panel()

    def set_engine(self, key: str):
        self.set_profile(engine=key)
        self.profile.live_engine = self.services.engine_for(key)
        if key in MODELS and self.settings.auto_start_server:
            self.start_services()
        self.refresh_panel()

    def _save_keys(self, vals: dict):
        try:
            self.bridge.keys.emit(self.api.set_keys(vals))
        except LensError as e:
            self.bridge.engine_test.emit({"ok": False, "error": str(e)})
            return
        self._test_engine()

    def _test_engine(self):
        self.bridge.engine_test.emit({"pending": True})
        try:
            r = self.api.test_engine(self.profile.live_engine, self.profile.target)
            if r.get("ok") and self.profile.engine == "deepl":
                u = self.api.usage().get("deepl") or {}
                if "used" in u:
                    r["quota"] = t("{used:,} / {limit:,} characters used this month", used=u["used"], limit=u["limit"])
        except LensError as e:
            r = {"ok": False, "error": str(e)}
        self.bridge.engine_test.emit(r)
        self.poll_status()

    def on_engine_test(self, r: dict):
        if r.get("pending"):
            self.panel.set_engine_status(t("Testing…"))
        elif r.get("ok"):
            extra = f" • {r['quota']}" if r.get("quota") else ""
            self.panel.set_engine_status(t("✓ Works: “{text}” ({ms} ms){extra}", text=r["text"], ms=r["ms"], extra=extra), "ok")
        else:
            self.panel.set_engine_status(t("✗ {error}", error=r.get("error", t("does not work"))[:160]), "bad")

    # --- profile / display ------------------------------------------------------------------------------------
    def set_profile(self, **kw):
        for k, v in kw.items():
            setattr(self.profile, k, v)
        self.profile.save()
        self.shown_seq.clear()
        self.shown.clear()  # translate again with the new settings
        self.reset_watchers()

    def set_font_scale(self, v: float):
        self.profile.font_scale = v
        self.profile.save()
        for its in list(self.overlay.items.values()):  # live preview on what is on screen now
            for it in its:
                it.max_px = it.max_px / max(0.01, getattr(it, "_scale", v)) * v
                it._scale = v
        self.overlay.update()

    def set_src_lang(self, code: str):
        self.set_profile(src_lang=code)
        if self.settings.auto_start_server:
            self.start_services()  # (fetches the text reader for that language if it is not there yet)

    def set_english_keys(self, on: bool):
        self.settings.english_keys = on
        self.settings.save()

    def keep_english_keys(self, now: float):
        """While the game has the focus, its keyboard is English: with a Thai layout the game shows its key hints
        as boxes. Checked twice a second; asked for again only when it went back to another language."""
        if now < self.next_keys:
            return
        self.next_keys = now + 0.5
        if win32.keyboard_layout(self.hwnd) & 0x3FF == win32.LANG_ENGLISH:
            return
        hkl = win32.english_layout()
        if hkl and not win32.ask_layout(self.hwnd, hkl):
            self.keys_possible = False  # Windows keeps us out after all
            self.panel.set_english_available(False, self.settings.english_keys)
            log.info("keyboard: the game's window takes no messages from game-lens")
            return
        if hkl:
            if not self.keys_told:
                self.keys_told = True
                self.toast(t("Keyboard switched to English (this can be turned off in the game-lens window)"), 3)

    def set_toasts(self, on: bool):
        self.settings.toasts = on
        self.settings.save()

    def toggle(self):
        self.enabled = not self.enabled
        if not self.enabled:
            self.overlay.clear()
            self.toast(t("game-lens: stopped"))
        else:
            self.reset_watchers()
            self.toast(t("game-lens: translating"))
        self.refresh_panel()

    def quit(self):
        log.info("quitting")
        self.timer.stop()
        self.status_timer.stop()
        for hid in (HK_TOGGLE, HK_SCREEN, HK_REGIONS, HK_HISTORY):
            win32.unregister_hotkey(int(self.overlay.winId()), hid)
        if self.cap:
            self.cap.stop()
        if self.settings.stop_server_on_exit:
            self.services.stop()
        self.pool.shutdown(wait=False, cancel_futures=True)
        self.tray.hide()
        log.info("stopped; leaving")
        logging.shutdown()
        # Everything is saved and stopped. End the process here, without unloading Qt and the capture library
        # under worker threads that may still be inside a call: that teardown hangs or crashes.
        win32.kernel32.TerminateProcess(win32.kernel32.GetCurrentProcess(), 0)

    def on_hotkey(self, hid: int):
        {HK_TOGGLE: self.toggle, HK_SCREEN: self.translate_screen, HK_REGIONS: self.edit_regions,
         HK_HISTORY: self.show_history}.get(hid, lambda: None)()

    def make_shortcut(self):
        err = create_shortcut()
        if err:
            self.notify(t("The icon could not be made"), err[:200])
        else:
            self.notify("game-lens", t("The game-lens icon is on the Desktop — next time just double-click it"))

    # --- game window ------------------------------------------------------------------------------------------
    def load_panel(self):
        self.panel.load(self.profile.engine, self.profile.font_scale, self.profile.target, self.settings.toasts,
                        self.auto_screen, self.settings.local_model, self.auto_label, self.profile.src_lang,
                        self.settings.english_keys)
        if getattr(self, "hwnd", None) is not None:
            self.panel.set_english_available(getattr(self, "keys_possible", True), self.settings.english_keys)

    def switch_profile(self, path: Path):
        """Another game: its own regions, language and names. Remembered as the game to open next time."""
        self.detach()
        self.profile = Profile.load(path)
        if self.profile.engine not in ENGINE_BY_KEY:
            self.profile.engine = "hymt"
        self.profile.live_engine = self.services.engine_for(self.profile.engine)
        if not os.environ.get("GAMELENS_PROFILE"):
            try:
                self.settings.profile = path.relative_to(ROOT).as_posix()
            except ValueError:
                self.settings.profile = str(path)
            self.settings.save()
        self.genders = None
        self.shown.clear()
        self.shown_seq.clear()
        self.readings.clear()
        self.sizes.clear()
        self.context.clear()
        self.reset_watchers()
        self.load_panel()
        self.pool.submit(self.sync_glossary)
        log.info("profile %s (%s)", path.name, self.profile.src_lang)
        self.refresh_panel()

    def pick_window(self):
        dlg = GamePicker(self.panel)
        if dlg.exec() and dlg.result_window:
            w = dlg.result_window
            same = (w["exe"] or "").lower() == (self.profile.exe or "").lower()
            if not same and self.profile.exe:  # another game: its own profile (made now if it is new)
                found = find_profile(w["exe"])
                path = found or Path(new_profile(w["exe"], w.get("title", ""), self.profile).path)
                self.switch_profile(path)
                if not found:
                    self.toast(t("New profile for {name} — set the game language and the text regions (F10)", name=self.profile.name), 6)
            else:
                if not self.profile.exe and not self.profile.regions:
                    self.profile.src_lang = guess_lang(w.get("title", ""), self.profile.src_lang)
                    self.load_panel()
                self.profile.exe, self.profile.title = w["exe"], ""
                self.profile.save()
            self.detach()
            self.next_attach = 0.0
            self.attach()
            self.show_panel()
            self.refresh_panel()

    def detach(self):
        if self.cap:
            self.cap.stop()
        self.cap, self.hwnd = None, None
        self.overlay.clear()
        self.overlay.hide()
        self.refresh_panel()

    def attach(self):
        now = time.monotonic()
        if now < self.next_attach or not (self.profile.exe or self.profile.title):
            return
        self.next_attach = now + 1.0
        hwnd = win32.find_window(self.profile.exe, self.profile.title)
        if not hwnd:
            other = self.other_game_running(now)
            if other:
                self.switch_profile(other)
                self.next_attach = 0.0
            return
        try:
            self.cap = GameCapture(hwnd)
        except Exception as e:  # noqa: BLE001 - capture can refuse some windows
            log.warning("capture failed: %s", e)
            if not getattr(self, "_capture_warned", False):
                self._capture_warned = True
                self.notify(t("game-lens: the game window cannot be captured"), str(e)[:200])
            return
        self.hwnd = hwnd
        # Windows lets no message from a normal program into a game that runs as administrator
        self.keys_possible = not (win32.window_elevated(hwnd) and not win32.self_elevated())
        self.panel.set_english_available(self.keys_possible, self.settings.english_keys)
        self.reset_watchers()
        self.refresh_panel()
        if self.profile.regions:
            self.toast(t("game-lens is translating • hold F8 to see the original"), 4)
        else:
            self.show_panel()

    def other_game_running(self, now: float) -> Path | None:
        """The profile of another known game that is open while this profile's game is not (every few seconds)."""
        if os.environ.get("GAMELENS_PROFILE") or now < getattr(self, "next_other", 0.0):
            return None
        self.next_other = now + 5.0
        mine = Path(self.profile.path).resolve()
        for p in sorted((ROOT / "profiles").glob("*.json")):
            if p.resolve() == mine:
                continue
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if (d.get("exe") or d.get("title")) and win32.find_window(d.get("exe", ""), d.get("title", "")):
                return p
        return None

    def someone_speaking(self) -> bool:
        """A line of dialogue is on screen: a short name in a name region and text in a dialogue region. (On a menu,
        whatever lies in the regions is part of the menu.)"""
        def has(kind, short=False):
            return any(r.kind == kind and (w := self.watchers.get(r.id)) and w.last_text
                       and (not short or len(plain_text(w.last_text).strip("【】[]")) <= 10) for r in self.profile.regions)
        return has("name", short=True) and (has("dialog") or has("subtitle"))

    def reset_watchers_keep_overlay(self):
        paid = self.profile.engine in REMOTE and self.profile.engine != "google_free"
        self.watchers = {r.id: TextFollower(confirm=paid) for r in self.profile.regions}

    def reset_watchers(self):
        paid = self.profile.engine in REMOTE and self.profile.engine != "google_free"
        self.watchers = {r.id: TextFollower(confirm=paid) for r in self.profile.regions}
        self.overlay.clear()

    # --- main loop --------------------------------------------------------------------------------------------
    def tick(self):
        if self.hwnd is None or not win32.is_window(self.hwnd) or (self.cap and self.cap.closed):
            if self.hwnd is not None:
                self.detach()
            self.attach()
            return
        cr = win32.client_rect(self.hwnd)
        minimized = win32.is_minimized(self.hwnd) or not cr or cr[2] < 50
        fg = win32.foreground()
        own = {int(w.winId()) for w in (self.history_win,) if w is not None and w.isVisible()}
        peek = win32.key_down(self.peek_vk)
        focused = fg == self.hwnd or fg in own or self.always_show
        visible = not minimized and self.editor is None and focused and not peek
        if fg == self.hwnd and self.settings.english_keys and self.keys_possible:
            self.keep_english_keys(time.monotonic())
        if visible:
            self.overlay.follow(cr)
            if not self.overlay.isVisible():
                self.overlay.show()
        elif self.overlay.isVisible():
            self.overlay.hide()
        if not self.enabled or minimized or self.editor is not None:
            return
        frame, _ = self.cap.client_frame()
        if frame is None or frame.shape[0] != cr[3] or frame.shape[1] != cr[2]:
            return  # no frame yet, or the window is being resized
        now = time.monotonic()
        if self.screen_ref is not None:
            self.check_screen_dismiss(frame)
        if self.auto_screen:
            self.auto_screen_tick(frame, now)
        for r in self.profile.regions:
            if self.screen_menu and self.screen_ref is not None:
                break  # a menu translated as a whole: the regions would only translate pieces of it again
            px = region_px(r.rect, frame.shape)
            x, y, w, h = px
            crop = frame[y : y + h, x : x + w]
            wt = self.watchers[r.id]
            ev = wt.feed(stroke_map(crop, r.ink), now)
            if ev == "hide":
                self.region_active = now
                self.overlay.clear(r.id)  # never leave an old translation over new text
                self.overlay.set_pending(r.id, None)
            elif ev == "read":
                if not self.health or self.services.state == "starting":
                    wt.again()  # the server is still starting (or the model being downloaded / loaded)
                    wt.read_at = now + 0.5
                    if now - self.waiting_toast > 15:
                        self.waiting_toast = now
                        self.toast(t("Starting the translation system… please wait"), 3)
                    continue
                self.pool.submit(self.job_read, r, px, crop.copy(), wt, frame.shape[0])
            # text over the moving scene: paint the covered glyphs again from this frame, so what hides them moves
            # with the scene instead of standing still on top of it
            live = [it for it in self.overlay.items.get(r.id, ()) if getattr(it, "_live", False)]
            for it in live:
                refresh_patch(it, crop)
            if live:
                self.overlay.update()

    def job_read(self, r, px, crop: np.ndarray, wt: TextFollower, frame_h: int):
        """Worker: read a region. The result goes through the UI thread, which owns the follower."""
        try:
            got = ocr_region(self.api, crop, r.kind, r.align, self.profile, frame_h)
        except Exception as e:  # noqa: BLE001 - a worker must always report back
            log.warning("region %s read: %s", r.id, e) if isinstance(e, LensError) else log.exception("region %s read", r.id)
            got = {"error": str(e) if isinstance(e, LensError) else f"{type(e).__name__}: {e}"}
        self.bridge.region_read.emit(r.id, wt, (r, px, crop, got))

    def on_region_read(self, rid: str, wt: TextFollower, data):
        r, px, crop, got = data
        now = time.monotonic()
        if wt is not self.watchers.get(rid):
            return  # the regions were edited meanwhile
        if got and "error" in got:
            wt.again()
            wt.read_at = now + 2  # not again at once
            self.show_error(got["error"])
            return
        what = wt.on_read(got["text"] if got else "", now)
        if got:
            self.readings[rid] = got
        if what:
            self.region_active = now  # something is being said: no time to read the whole screen
        if what == "hide":
            self.overlay.clear(rid)
            self.overlay.set_pending(rid, None)
        if rid in self.held:
            self.on_region_done(rid, *self.held.pop(rid))
        if what is None and got and wt.settled:
            for it in self.overlay.items.get(rid, ()):
                extend_item(it, crop, got["lines"], r.ink, wt.persistent)
        if what != "translate" or not self.enabled:
            return
        prev = self.shown.get(rid)
        if prev and similar(prev["src"], got["text"]) >= 0.92 and not wt.corrected:
            if prev["items"] and self.overlay.items.get(rid) is not prev["items"]:
                self.overlay.set_items(rid, prev["items"])
            return
        if prev and not wt.corrected and not goes_on(prev["src"], got["text"]):
            self.overlay.clear(rid)  # another text: nothing old may stay over it
        log.info("region %s ocr %d ms: %r", rid, got["ms"], got["text"])
        self.overlay.set_pending(rid, (px[0] + px[2], px[1] + 4))  # dots: a translation is on its way
        self.pool.submit(self.job_translate, r, px, crop, got, wt, (wt.seq, wt.line), wt.persistent)

    def job_translate(self, r, px, crop: np.ndarray, got: dict, wt: TextFollower, seq: tuple[int, int], strokes):
        t0 = time.perf_counter()
        try:
            res = translate_region(self.api, crop, px, got, r.kind, r.align, r.ink, self.profile, list(self.context), strokes,
                                   self.sizes.setdefault(r.id, TextSize()), self.profile.engine in REMOTE)
            log.info("region %s %s %d ms%s: %r", r.id, res["engine"], (time.perf_counter() - t0) * 1000,
                     " (cache)" if res["cached"] else "", res["tl"])
        except Exception as e:  # noqa: BLE001
            log.warning("region %s: %s", r.id, e) if isinstance(e, LensError) else log.exception("region %s failed", r.id)
            res = {"error": str(e) if isinstance(e, LensError) else f"{type(e).__name__}: {e}"}
        self.bridge.region_done.emit(r.id, (wt, seq, got), res)

    def show_error(self, error: str):
        msg, at = self.last_error
        if error != msg or time.monotonic() - at > 20:  # do not repeat the same error every line
            self.last_error = (error, time.monotonic())
            self.toast(friendly_error(error), 4)

    def on_region_done(self, rid: str, which, payload: dict):
        wt, (seq, line), got = which
        if wt is not self.watchers.get(rid) or line != wt.line:
            return  # the text was replaced while it was being translated
        if wt.in_doubt:
            self.held[rid] = (which, payload)  # until the next reading has come in
            return
        self.fit_speaker(payload)
        if seq != wt.seq:  # more was typed meanwhile: better its beginning in the player's language than nothing
            if seq > self.shown_seq.get(rid, -1) and payload.get("items") and begins(got, self.readings[rid]):
                self.shown[rid], self.shown_seq[rid] = payload, seq
                self.overlay.set_items(rid, payload["items"])
            return
        self.overlay.set_pending(rid, None)
        if "error" in payload:
            wt.again()
            wt.read_at = time.monotonic() + 2
            self.show_error(payload["error"])
            return
        self.shown[rid], self.shown_seq[rid] = payload, seq
        self.overlay.set_items(rid, payload["items"])
        if payload["tl"]:
            e = self.history.add(payload["kind"], payload["src"], payload["tl"], payload["engine"])
            if self.history_win:
                self.history_win.append(e)
            if payload["kind"] not in ("name", "choice", "labels"):
                remember(self.context, payload["src"], payload["tl"])

    # --- whole-screen mode (menus, item descriptions) ---------------------------------------------------------
    def translate_screen(self):
        """F9: translate what is on screen now; again to remove it."""
        if self.screen_ref is not None:
            self.dismiss_screen()
            return
        if not self.cap:
            self.notify("game-lens", t("The game window was not found"))
            return
        frame, _ = self.cap.client_frame()
        if frame is None:
            return
        self.start_screen(frame, ink_map(frame, "any"), auto=False)

    def start_screen(self, frame: np.ndarray, ref: np.ndarray, auto: bool, regions: bool = True):
        """regions: leave the text inside the regions to them (not when they keep quiet for a menu)."""
        if self.screen_busy:
            return
        self.screen_auto = auto
        self.screen_busy = True
        self.screen_gen += 1
        self.screen_ref = ref
        self.screen_frame, self.screen_marks = frame, []
        self.screen_strokes, self.screen_last = stroke_map(frame, "any"), None  # to notice text that appears later
        self.auto_done = stroke_map(frame, "any")  # this text is handled: auto mode will not start on it again
        self.overlay.set_pending("screen", (frame.shape[1] // 2 + 21, 60))
        if not auto:
            self.toast(t("Translating the whole screen… (F9 / Esc to close)"), 2)
        # lines inside a region that is showing its own translation are left to that region
        # ...but only while someone speaks (a short name in a name region): on a menu, the text that happens to lie
        # in a region is part of the menu
        speaking = self.someone_speaking()
        skip = [region_px(r.rect, frame.shape) for r in self.profile.regions
                if regions and speaking and (self.overlay.items.get(r.id) or self.watchers[r.id].settled or self.watchers[r.id].last_text)]
        self.pool.submit(self.job_screen, frame, self.screen_gen, skip, auto)

    def auto_screen_tick(self, frame: np.ndarray, now: float):
        """Menus and quest lists stand still while gameplay moves: translate the screen once its text has been
        still for a moment and it is not the picture we already handled. The text, not the whole picture: a menu
        often has a character moving in it. Text is thin strokes that stay put; the 3D world's are everywhere and
        never still."""
        s = stroke_map(frame, "any")
        prev, self.auto_prev = self.auto_prev, s
        if prev is None or prev.shape != s.shape or stroke_change(prev, s) > STILL:
            self.auto_still_since = now
            return
        if self.screen_busy or self.screen_ref is not None or not self.health or self.services.state == "starting":
            return
        if self.auto_done is not None and self.auto_done.shape == s.shape:
            changed = stroke_change(self.auto_done, s)
            if changed <= STILL:
                return  # handled already
            if changed > 0.6:
                self.auto_empties = 0  # quite a different screen
        wts = {r.kind: self.watchers[r.id] for r in self.profile.regions}
        settled = {k for k, w in wts.items() if w.settled and not (k == "name" and len(plain_text(w.last_text)) > 10)}
        still = now - max(self.auto_still_since, self.region_active)
        if still < auto_pause(settled, self.auto_empties) or any(w.reading for w in wts.values()):
            return
        self.start_screen(frame, ink_map(frame, "any"), auto=True)

    def set_auto_screen(self, on: bool):
        self.auto_screen = self.settings.auto_screen = on
        self.settings.save()
        self.auto_done = None
        if not on:
            self.dismiss_screen()

    def job_screen(self, frame: np.ndarray, gen: int, skip: list, auto: bool):
        p = self.profile
        try:
            ocr = self.api.ocr_lines(frame, p.src_lang, frame_h=frame.shape[0])
            lines = lines_to_translate(ocr["lines"], skip, p.src_lang)
            log.info("screen%s: %d lines", " (auto)" if auto else "", len(lines))
            if not lines:
                self.bridge.screen_done.emit(gen, None if auto else [])
                return
            groups = paragraphs(lines)
            res = self.api.translate_lines(requests(groups, p.src_lang), series=p.series, target=p.target,
                                           src_lang=p.src_lang, engine=p.live_engine or p.engine, allow_cloud=p.engine in REMOTE,
                                           context=[])
            items = screen_items(frame, groups, res["lines"], p.font_scale, [l["bbox"] for l in ocr["lines"]])
            log.info("screen: %d translated by %s", len(items), res.get("engine"))
            self.bridge.screen_done.emit(gen, items if items or not auto else None)
        except Exception as e:  # noqa: BLE001
            log.warning("screen: %s", e)
            self.bridge.screen_done.emit(gen, {"error": str(e)})

    def on_screen_done(self, gen: int, items):
        self.screen_busy = False
        self.overlay.set_pending("screen", None)
        if gen != self.screen_gen or self.screen_ref is None:
            return
        if isinstance(items, dict):
            self.show_error(items["error"])
            self.screen_ref = None
            self.auto_done = None  # not handled after all: auto mode tries again in a while
            self.auto_still_since = time.monotonic() + 2
            return
        # nothing found, or the same labels as the last time (the HUD while walking): wait longer before trying again
        texts = frozenset(it.text for it in items) if isinstance(items, list) else frozenset()
        same = bool(texts) and texts == self.screen_texts and self.screen_auto
        # read again for nothing (the screen only shimmered): do not let that happen every second
        self.screen_quiet = time.monotonic() + 8 if texts and texts == self.screen_texts else 0.0
        self.screen_texts = texts or self.screen_texts
        self.auto_empties = 0 if items and not same else self.auto_empties + 1
        if items is None:  # auto mode found nothing to translate: stay quiet
            self.dismiss_screen()  # (and take away what an earlier reading of this screen put there)
            return
        if not items:
            self.toast(t("No text on the screen"), 2)
        self.overlay.set_items("screen", items)
        # a menu (many labels; or asked for with F9): the regions' own translations would be pieces of it
        self.screen_menu = len(items) >= 8 or not self.screen_auto or not self.someone_speaking()
        if self.screen_menu:
            for r in self.profile.regions:
                self.overlay.clear(r.id)
                self.overlay.set_pending(r.id, None)
            self.shown.clear()
            self.reset_watchers_keep_overlay()
        self.screen_marks = text_marks(self.screen_frame, [b for it in items for b in getattr(it, "_boxes", [])])

    def check_screen_dismiss(self, frame: np.ndarray):
        if win32.key_down(win32.VK["ESC"]):
            self.dismiss_screen()
            return
        if self.screen_marks:
            state = text_state(frame, self.screen_marks)
            strokes = stroke_map(frame, "any")
            last, self.screen_last = self.screen_last, strokes
            # text the translation does not cover appeared (a pop-up, another tab): its strokes differ from the
            # picture that was read, tile by tile, though the screen itself has come to rest
            new_text = stroke_change(self.screen_strokes, strokes) > STILL and time.monotonic() > self.screen_quiet
            if state == "gone":  # another page, a closed menu
                self.dismiss_screen()
                self.auto_done = None
            elif state == "changed" or new_text:
                if last is None or stroke_change(last, strokes) > STILL:
                    return  # still moving (a pop-up opening): wait until it stands
                # read the screen again; what is shown now may lie over other text, so it goes first
                auto, menu = self.screen_auto, self.screen_menu
                self.dismiss_screen()
                self.screen_menu = menu
                self.start_screen(frame, ink_map(frame, "any"), auto=auto, regions=not menu)
            return
        now_map = ink_map(frame, "any")  # still being read: only a quite different screen counts
        if changed_pixels(self.screen_ref, now_map) > 0.25 * now_map.size:
            self.dismiss_screen()

    def dismiss_screen(self):
        self.screen_ref, self.screen_marks, self.screen_menu = None, [], False
        self.screen_gen += 1
        self.overlay.set_pending("screen", None)
        self.overlay.clear("screen")

    # --- windows ----------------------------------------------------------------------------------------------
    def edit_regions(self):
        if self.editor is not None:
            return
        if not self.cap or self.hwnd is None:
            self.show_panel()
            self.notify("game-lens", t("Open the game and choose its window first"))
            return
        frame, _ = self.cap.client_frame()
        cr = win32.client_rect(self.hwnd)
        if frame is None or not cr:
            return
        self.overlay.follow(cr)  # sets the geometry of the game's client area (logical pixels)
        self.overlay.hide()
        self.panel.hide()
        p = self.profile
        find = (lambda f: self.api.ocr_lines(f, p.src_lang, frame_h=f.shape[0])["lines"]) if self.health else None
        self.editor = RegionEditor(frame, p.regions, self.overlay.geometry(), find)
        self.editor.setStyleSheet(QSS)

        def done(regions=None):
            if regions is not None:
                self.profile.regions = regions
                self.profile.save()
                self.shown.clear()
                self.shown_seq.clear()
                self.reset_watchers()
                if regions:
                    self.toast(t("{n} regions saved — translating", n=len(regions)), 3)
            self.editor = None
            self.refresh_panel()
            if not self.profile.regions:
                self.show_panel()

        self.editor.saved.connect(done)
        self.editor.cancelled.connect(lambda: done(None))
        self.editor.show()
        self.editor.activateWindow()
        self.editor.setFocus()

    def show_history(self):
        if self.history_win is None:
            self.history_win = HistoryWindow(self.history)
            self.history_win.setWindowIcon(app_icon())
            self.history_win.build_glossary.connect(lambda: self.pool.submit(self.build_glossary))
        if self.history_win.isVisible():
            self.history_win.hide()
        else:
            self.history_win.show()
            self.history_win.raise_()

    def build_glossary(self):
        p = self.profile
        text = self.history.source_text()
        if not text:
            self.notify("game-lens", t("No history to build a glossary from yet"))
            return
        engine = "cloud" if p.engine == "cloud" else "local"  # needs a general LLM, not an MT model
        self.notify("game-lens", t("Looking for character names in the history… (needs Gemma or Claude)"))
        try:
            r = self.api.extract_glossary(p.series, text, p.target, engine, engine == "cloud")
        except LensError as e:
            self.notify(t("The glossary could not be built"), str(e)[:200])
            return
        n = len(r["proposed"].get("characters", [])) + len(r["proposed"].get("terms", []))
        self.notify(t("Glossary updated"), t("{n} names / terms found — they are translated the same way every time", n=n))


def main():
    (ROOT / "logs").mkdir(exist_ok=True)
    logging.basicConfig(filename=ROOT / "logs" / "gamelens.log", level=logging.INFO, encoding="utf-8",
                        format="%(asctime)s %(levelname)s %(message)s")
    sys.excepthook = lambda *exc: log.critical("uncaught", exc_info=exc)
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    app.setApplicationName("game-lens")
    app.setWindowIcon(app_icon())
    # one instance: starting game-lens again just brings the running one's window up
    # `python -m gamelens --quit` closes the running one (and the server / model it started)
    command = b"quit" if "--quit" in sys.argv else b"show"
    sock = QLocalSocket()
    sock.connectToServer(INSTANCE_KEY)
    if sock.waitForConnected(1000):
        sock.write(command)
        sock.flush()
        sock.waitForReadyRead(5000)  # the running instance answers "ok" once it has the command
        return
    if command == b"quit":
        return
    QLocalServer.removeServer(INSTANCE_KEY)
    server = QLocalServer()
    server.listen(INSTANCE_KEY)
    app.setStyleSheet(QSS)
    ctl = Controller(app)

    def on_connection():
        conn = server.nextPendingConnection()

        def handle():
            data = bytes(conn.readAll())
            if not data:
                return
            log.info("command from another launch: %r", data)
            conn.write(b"ok")
            conn.flush()
            if data == b"quit":
                QTimer.singleShot(0, ctl.quit)
            else:
                ctl.show_panel()

        conn.readyRead.connect(handle)
        handle()  # the command may already be here

    server.newConnection.connect(on_connection)
    sys.exit(app.exec())
