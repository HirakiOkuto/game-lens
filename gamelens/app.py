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

import numpy as np
from PySide6.QtCore import QAbstractNativeEventFilter, QObject, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QIcon, QPainter, QPixmap
from PySide6.QtNetwork import QLocalServer, QLocalSocket
from PySide6.QtWidgets import QApplication, QMenu, QSystemTrayIcon

from . import win32
from .api import LensClient, LensError
from .capture import GameCapture
from .history import HistoryLog, HistoryWindow
from .overlay import Overlay
from .panel import Panel
from .picker import GamePicker
from .profile import ENGINE_BY_KEY, KIND_LABELS, REMOTE, ROOT, Profile, Settings
from .region_editor import RegionEditor
from .screen import (TextSize, begins, extend_item, lines_to_translate, merge_glossary, ocr_region, paragraphs, refresh_patch, remember,
                     requests, screen_items, translate_region)
from .services import MODEL_NAMES, MODELS, ServerManager
from .ui_style import QSS
from .watcher import TextFollower, auto_pause, changed_pixels, goes_on, ink_map, similar, stroke_map

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
    pyw = ROOT / ".venv" / "Scripts" / "pythonw.exe"
    where = f"'{folder}'" if folder else "[Environment]::GetFolderPath('Desktop')"
    ps = (
        f"$d={where};"
        "$s=(New-Object -ComObject WScript.Shell).CreateShortcut(\"$d\\game-lens.lnk\");"
        f"$s.TargetPath='{pyw}';$s.Arguments='-m gamelens';$s.WorkingDirectory='{ROOT}';"
        f"$s.IconLocation='{ico}';$s.Description='game-lens';$s.Save()"
    )
    r = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True,
                       creationflags=0x08000000)
    return "" if r.returncode == 0 and not r.stderr.strip() else (r.stderr.strip() or "powershell failed")


def friendly_error(err: str) -> str:
    """A short Thai message for the game screen (details stay in logs/gamelens.log)."""
    e = err.lower()
    if "10061" in e or "ต่อ server ไม่ได้" in err or "connecterror" in e:
        return "ระบบแปลยังไม่พร้อม — เปิดหน้าต่าง game-lens เพื่อดูสถานะ"
    if "429" in e or "rate limit" in e:
        return "เครื่องแปลออนไลน์ถูกจำกัดชั่วคราว (429) — ลองเปลี่ยนเครื่องแปล"
    if "quota" in e or "456" in e:
        return "โควตาเครื่องแปลหมดเดือนนี้ — เปลี่ยนเครื่องแปลในหน้าต่าง game-lens"
    if "403" in e or "401" in e or "bad key" in e:
        return "key ของเครื่องแปลไม่ถูกต้อง — ตรวจในหน้าต่าง game-lens"
    if "http 500" in e:
        return "ระบบแปลขัดข้อง — ดูรายละเอียดใน logs/server.log"
    if "unreachable" in e:
        return "เครื่องแปลยังไม่ได้เปิด — เปิดหน้าต่าง game-lens แล้วกด โหลด"
    return "แปลไม่สำเร็จ: " + err.split(": ", 1)[-1][:80]


def region_px(rect: list[float], shape) -> tuple[int, int, int, int]:
    H, W = shape[:2]
    x, y = int(rect[0] * W), int(rect[1] * H)
    w, h = int(rect[2] * W), int(rect[3] * H)
    x, y = max(0, min(x, W - 1)), max(0, min(y, H - 1))
    return x, y, max(1, min(w, W - x)), max(1, min(h, H - y))


class Controller(QObject):
    def __init__(self, app: QApplication):
        super().__init__()
        self.app = app
        self.settings = Settings.load()
        ppath = self.settings.resolve(os.environ.get("GAMELENS_PROFILE") or self.settings.profile)
        if not ppath.exists():
            ppath.parent.mkdir(parents=True, exist_ok=True)
            Profile(name=ppath.stem, series=ppath.stem, path=str(ppath)).save()
        self.profile = Profile.load(ppath)
        self.api = LensClient(self.settings.server, self.settings.resolve(self.settings.token_file))
        self.services = ServerManager(ROOT, ROOT / "logs", self.settings.resolve(self.settings.models_dir))
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
        self.held: dict[str, tuple] = {}  # translations that wait for a line to be read once more
        self.readings: dict[str, dict] = {}  # what was read last in each region
        self.sizes: dict[str, TextSize] = {}
        self.context: deque = deque(maxlen=self.settings.context_lines)
        self.screen_gen = 0
        self.screen_ref: np.ndarray | None = None
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
        self.panel = Panel(self.settings.hotkeys, self.settings.peek_key)
        self.panel.setWindowIcon(app_icon())
        self.panel.load(self.profile.engine, self.profile.font_scale, self.profile.target, self.settings.toasts,
                        self.auto_screen)
        self.panel.auto_screen_changed.connect(self.set_auto_screen)
        self.panel.toggle.connect(self.toggle)
        self.panel.fix_server.connect(self.start_services)
        self.panel.pick_game.connect(self.pick_window)
        self.panel.edit_regions.connect(self.edit_regions)
        self.panel.engine_changed.connect(self.set_engine)
        self.panel.save_keys.connect(lambda v: self.pool.submit(self._save_keys, v))
        self.panel.test_engine.connect(lambda: self.pool.submit(self._test_engine))
        self.panel.font_scale_changed.connect(self.set_font_scale)
        self.panel.target_changed.connect(lambda t: self.set_profile(target=t))
        self.panel.toasts_changed.connect(self.set_toasts)
        self.panel.show_history.connect(self.show_history)
        self.panel.make_shortcut.connect(self.make_shortcut)
        self.panel.quit_app.connect(self.quit)

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
                self.notify("game-lens", f"ปุ่มลัด {hk[name]} ถูกโปรแกรมอื่นใช้อยู่ (แก้ได้ใน settings.json)")

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
        m.addAction("เปิดหน้าต่าง game-lens", self.show_panel)
        self.toggle_action = m.addAction("หยุดแปล", self.toggle)
        m.addAction("แปลทั้งหน้าจอ", self.translate_screen)
        m.addAction("ตั้งกรอบข้อความ", self.edit_regions)
        m.addAction("ประวัติคำแปล", self.show_history)
        m.addSeparator()
        m.addAction("ปิดโปรแกรม", self.quit)
        self.menu = m
        self.tray.setContextMenu(m)

    # --- translation system (server + model) ------------------------------------------------------------------
    def start_services(self):
        engine = self.profile.engine
        threading.Thread(target=self.services.ensure, args=(engine, lambda s, d: self.bridge.server_state.emit(s, d)),
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
        return bool(self.health and self.health.get(self.profile.engine))

    def refresh_panel(self):
        p, eng = self.profile, self.profile.engine
        info = ENGINE_BY_KEY[eng]
        # 1. translation system
        svc = self.services
        if self.health and self.health.get(eng):
            self.panel.s_server.set("ok", f"พร้อม — {info['name']}")
        elif svc.state == "starting":
            self.panel.s_server.set("warn", svc.detail)
        elif not self.health:
            msg = svc.detail if svc.state == "error" else "ระบบแปลยังไม่เปิด"
            self.panel.s_server.set("bad", msg, "เปิด", primary=True)
        elif not self.health.get(eng):
            if eng in MODELS:
                self.panel.s_server.set("warn", f"ยังไม่ได้โหลด {MODEL_NAMES[eng]}", "โหลด", primary=True)
            elif info.get("keys"):
                self.panel.s_server.set("warn", f"ยังไม่ได้ใส่ key ของ {info['name']} (ช่องด้านล่าง)")
            else:
                self.panel.s_server.set("warn", f"{info['name']} ยังไม่พร้อม")
        else:
            self.panel.s_server.set("ok", f"พร้อม — {info['name']}")
        # 2. game
        if not (p.exe or p.title):
            self.panel.s_game.set("bad", "ยังไม่ได้เลือกหน้าต่างเกม", "เลือกเกม", primary=True)
        elif self.hwnd is None:
            self.panel.s_game.set("warn", f"รอเกมเปิด… ({p.exe or p.title}) • ตั้งเกมเป็น Borderless/Windowed", "เปลี่ยน")
        else:
            self.panel.s_game.set("ok", win32.window_title(self.hwnd)[:60] or p.exe, "เปลี่ยน")
        # 3. regions
        if not p.regions:
            if self.hwnd is None:
                self.panel.s_regions.set("idle", "เปิดเกมก่อน แล้วไปหน้าที่มีบทพูด")
            else:
                self.panel.s_regions.set("bad", "ไปหน้าที่มีบทพูดในเกม แล้วกดตั้งกรอบ", "ตั้งกรอบ", primary=True)
        else:
            kinds = ", ".join(sorted({KIND_LABELS[r.kind].split(" (")[0] for r in p.regions}))
            self.panel.s_regions.set("ok", f"{len(p.regions)} กรอบ: {kinds}", "แก้ไข" if self.hwnd else None)
        ready = bool(self.health) and self.hwnd is not None and bool(p.regions)
        if self.enabled and ready:
            hint = "กำลังแปล — สลับไปเล่นเกมได้เลย"
        elif not self.enabled:
            hint = "หยุดอยู่"
        else:
            hint = "ทำขั้นตอนที่ยังไม่เป็นสีเขียวให้ครบ แล้วจะเริ่มแปลเอง"
        self.panel.set_running(self.enabled, ready, hint)
        self.toggle_action.setText("หยุดแปล" if self.enabled else "เริ่มแปล")
        self.tray.setToolTip("game-lens — " + hint)

    def set_engine(self, key: str):
        self.set_profile(engine=key)
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
            r = self.api.test_engine(self.profile.engine, self.profile.target)
            if r.get("ok") and self.profile.engine == "deepl":
                u = self.api.usage().get("deepl") or {}
                if "used" in u:
                    r["quota"] = f"ใช้ไป {u['used']:,} / {u['limit']:,} ตัวอักษรเดือนนี้"
        except LensError as e:
            r = {"ok": False, "error": str(e)}
        self.bridge.engine_test.emit(r)
        self.poll_status()

    def on_engine_test(self, r: dict):
        if r.get("pending"):
            self.panel.set_engine_status("กำลังทดสอบ…")
        elif r.get("ok"):
            extra = f" • {r['quota']}" if r.get("quota") else ""
            self.panel.set_engine_status(f"✓ ใช้ได้: “{r['text']}” ({r['ms']} ms){extra}", "ok")
        else:
            self.panel.set_engine_status(f"✗ {r.get('error', 'ใช้ไม่ได้')[:160]}", "bad")

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

    def set_toasts(self, on: bool):
        self.settings.toasts = on
        self.settings.save()

    def toggle(self):
        self.enabled = not self.enabled
        if not self.enabled:
            self.overlay.clear()
            self.toast("game-lens: หยุดแปล")
        else:
            self.reset_watchers()
            self.toast("game-lens: แปลต่อ")
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
            self.notify("สร้างไอคอนไม่สำเร็จ", err[:200])
        else:
            self.notify("game-lens", "สร้างไอคอน game-lens บน Desktop แล้ว — ครั้งต่อไปดับเบิลคลิกได้เลย")

    # --- game window ------------------------------------------------------------------------------------------
    def pick_window(self):
        dlg = GamePicker(self.panel)
        if dlg.exec() and dlg.result_window:
            w = dlg.result_window
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
            return
        try:
            self.cap = GameCapture(hwnd)
        except Exception as e:  # noqa: BLE001 - capture can refuse some windows
            log.warning("capture failed: %s", e)
            if not getattr(self, "_capture_warned", False):
                self._capture_warned = True
                self.notify("game-lens: จับภาพหน้าต่างเกมไม่ได้", str(e)[:200])
            return
        self.hwnd = hwnd
        self.reset_watchers()
        self.refresh_panel()
        if self.profile.regions:
            self.toast("game-lens พร้อมแปล • กด F8 ค้างเพื่อดูต้นฉบับ", 4)
        else:
            self.show_panel()

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
                if not self.health:
                    wt.again()  # the server is still starting
                    wt.read_at = now + 0.5
                    if now - self.waiting_toast > 15:
                        self.waiting_toast = now
                        self.toast("กำลังเปิดระบบแปล… รอสักครู่", 3)
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
        if what or (got and not wt.settled):
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
            if payload["kind"] not in ("name", "choice"):
                remember(self.context, payload["src"], payload["tl"])

    # --- whole-screen mode (menus, item descriptions) ---------------------------------------------------------
    def translate_screen(self):
        """F9: translate what is on screen now; again to remove it."""
        if self.screen_ref is not None:
            self.dismiss_screen()
            return
        if not self.cap:
            self.notify("game-lens", "ยังไม่พบหน้าต่างเกม")
            return
        frame, _ = self.cap.client_frame()
        if frame is None:
            return
        self.start_screen(frame, ink_map(frame, "any"), auto=False)

    def start_screen(self, frame: np.ndarray, ref: np.ndarray, auto: bool):
        if self.screen_busy:
            return
        self.screen_busy = True
        self.screen_gen += 1
        self.screen_ref = ref
        self.auto_done = ref  # this picture is handled: auto mode will not start on it again
        self.overlay.set_pending("screen", (frame.shape[1] // 2 + 21, 60))
        if not auto:
            self.toast("กำลังแปลทั้งหน้าจอ… (F9 / Esc ปิด)", 2)
        # lines inside a region that is showing its own translation are left to that region
        skip = [region_px(r.rect, frame.shape) for r in self.profile.regions
                if self.overlay.items.get(r.id) or self.watchers[r.id].settled or self.watchers[r.id].last_text]
        self.pool.submit(self.job_screen, frame, self.screen_gen, skip, auto)

    def auto_screen_tick(self, frame: np.ndarray, now: float):
        """Menus and quest lists stand still while gameplay moves: translate the screen once it has been still
        for a moment and is not the picture we already handled."""
        m = ink_map(frame, "any")
        prev, self.auto_prev = self.auto_prev, m
        if prev is None or changed_pixels(prev, m) > 0.01 * m.size:
            self.auto_still_since = now
            return
        if self.screen_busy or self.screen_ref is not None or not self.health:
            return
        if self.auto_done is not None:
            changed = changed_pixels(self.auto_done, m)
            if changed <= 0.03 * m.size:
                return
            if changed > 0.25 * m.size:
                self.auto_empties = 0  # quite a different screen
        wts = {r.kind: self.watchers[r.id] for r in self.profile.regions}
        still = now - max(self.auto_still_since, self.region_active)
        if still < auto_pause({k for k, w in wts.items() if w.settled}, self.auto_empties) or any(w.reading for w in wts.values()):
            return
        self.start_screen(frame, m, auto=True)

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
                                           src_lang=p.src_lang, engine=p.engine, allow_cloud=p.engine in REMOTE,
                                           context=[])
            items = screen_items(frame, groups, res["lines"], p.font_scale)
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
        self.auto_empties = 0 if items else self.auto_empties + 1
        if items is None:  # auto mode found nothing to translate: stay quiet
            self.screen_ref = None
            return
        if not items:
            self.toast("ไม่พบข้อความบนจอ", 2)
        self.overlay.set_items("screen", items)

    def check_screen_dismiss(self, frame: np.ndarray):
        if win32.key_down(win32.VK["ESC"]):
            self.dismiss_screen()
            return
        now_map = ink_map(frame, "any")
        if changed_pixels(self.screen_ref, now_map) > 0.03 * now_map.size:
            self.dismiss_screen()  # the menu changed: the labels no longer match

    def dismiss_screen(self):
        self.screen_ref = None
        self.screen_gen += 1
        self.overlay.set_pending("screen", None)
        self.overlay.clear("screen")

    # --- windows ----------------------------------------------------------------------------------------------
    def edit_regions(self):
        if self.editor is not None:
            return
        if not self.cap or self.hwnd is None:
            self.show_panel()
            self.notify("game-lens", "เปิดเกมและเลือกหน้าต่างเกมก่อน")
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
                    self.toast(f"บันทึก {len(regions)} กรอบแล้ว — เริ่มแปล", 3)
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
            self.notify("game-lens", "ยังไม่มีประวัติให้สร้าง glossary")
            return
        engine = "cloud" if p.engine == "cloud" else "local"  # needs a general LLM, not an MT model
        self.notify("game-lens", "กำลังหาชื่อตัวละครจากประวัติ… (ต้องใช้ Gemma หรือ Claude)")
        try:
            r = self.api.extract_glossary(p.series, text, p.target, engine, engine == "cloud")
        except LensError as e:
            self.notify("สร้าง glossary ไม่สำเร็จ", str(e)[:200])
            return
        n = len(r["proposed"].get("characters", [])) + len(r["proposed"].get("terms", []))
        self.notify("glossary อัปเดตแล้ว", f"พบ {n} ชื่อ/ศัพท์ — ชื่อจะถูกแปลเหมือนกันทุกครั้ง")


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
