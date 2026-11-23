"""game-lens controller: finds the game window, watches its regions, asks the server for OCR + translation and
puts the Thai text in place on the overlay. One main window (Panel) plus a tray icon."""
from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
import time
from collections import defaultdict, deque
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
from .overlay import Item, Overlay, text_rect_for, to_qimage
from .panel import Panel
from .picker import GamePicker
from .profile import ENGINE_BY_KEY, KIND_LABELS, REMOTE, ROOT, Profile, Settings
from .region_editor import RegionEditor
from .render import erase_patch
from .services import MODEL_NAMES, MODELS, ServerManager
from .ui_style import QSS
from .watcher import RegionWatcher, changed_pixels, has_text, ink_map, similar

HK_TOGGLE, HK_SCREEN, HK_REGIONS, HK_HISTORY = 1, 2, 3, 4
INSTANCE_KEY = "game-lens-ui"
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
    region_done = Signal(str, int, object)
    screen_done = Signal(int, object)
    message = Signal(str, str)
    status = Signal(dict)
    server_state = Signal(str, str)
    engine_test = Signal(dict)
    keys = Signal(dict)
    pending = Signal(str, int, object)


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


def is_continuation(before: np.ndarray | None, now: np.ndarray) -> bool:
    """True when the text read a moment ago is still there with more added (a typewriter effect that paused),
    False when it was replaced by a new line."""
    if before is None or before.dtype != bool or now.dtype != bool or before.shape != now.shape:
        return False
    n = int(before.sum())
    return n > 0 and int((before & now).sum()) / n > 0.8 and int(now.sum()) > n


def in_any(bbox: list[int], rects: list) -> bool:
    """True when the centre of bbox lies inside one of rects (x, y, w, h)."""
    cx, cy = bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2
    return any(x <= cx <= x + w and y <= cy <= y + h for x, y, w, h in rects)


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
        self.pool = ThreadPoolExecutor(3)
        self.bridge = Bridge()
        self.bridge.region_done.connect(self.on_region_done)
        self.bridge.screen_done.connect(self.on_screen_done)
        self.bridge.message.connect(self._tray_message)
        self.bridge.status.connect(self.on_status)
        self.bridge.server_state.connect(lambda s, d: self.refresh_panel())
        self.bridge.engine_test.connect(self.on_engine_test)
        self.bridge.keys.connect(lambda k: self.panel.set_keys(k))
        self.bridge.pending.connect(lambda rid, gen, at: self.overlay.set_pending(rid, at) if gen == self.gen[rid] else None)

        self.overlay = Overlay(self.profile.font_family)
        self.overlay.winId()  # native window exists from the start: hotkeys are delivered to it
        self.history = HistoryLog(ROOT / "logs", self.profile.series)
        self.history_win: HistoryWindow | None = None
        self.editor: RegionEditor | None = None

        self.hwnd: int | None = None
        self.cap: GameCapture | None = None
        self.enabled = True
        self.health: dict | None = None
        self.watchers: dict[str, RegionWatcher] = {}
        self.gen: dict[str, int] = defaultdict(int)
        self.busy: set[str] = set()
        self.shown: dict[str, dict] = {}
        self.context: deque = deque(maxlen=self.settings.context_lines)
        self.screen_gen = 0
        self.screen_ref: np.ndarray | None = None
        self.screen_busy = False
        self.next_attach = 0.0
        self.last_error = ("", 0.0)
        self.last_stable: dict[str, float] = {}
        self.stable_ink: dict[str, np.ndarray] = {}
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
                self.bridge.keys.emit(self.api.keys())
                self._keys_loaded = True
        except LensError:
            self._keys_loaded = False
        self.bridge.status.emit(st)

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
        self.watchers = {r.id: RegionWatcher(self.profile.stable_ms, self.profile.max_wait_ms) for r in self.profile.regions}
        for r in self.profile.regions:
            self.gen[r.id] += 1
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
            ink = ink_map(crop, r.ink)
            ev = self.watchers[r.id].feed(ink, now)
            if ev == "changed":
                self.gen[r.id] += 1
                self.overlay.clear(r.id)  # never leave an old translation over new text
                self.overlay.set_pending(r.id, None)
                wt = self.watchers[r.id]
                if (now - self.last_stable.get(r.id, -9) < 1.2 and wt.stable_s < 1.0
                        and is_continuation(self.stable_ink.get(r.id), ink)):
                    # the same line went on typing right after we read it: this game pauses mid-line, so wait
                    # a little longer before reading from now on
                    wt.stable_s = min(1.0, wt.stable_s + 0.1)
                    log.info("region %s: stable time now %d ms", r.id, wt.stable_s * 1000)
            elif ev == "stable":
                self.last_stable[r.id] = now
                self.stable_ink[r.id] = ink
                if not has_text(ink):
                    self.shown.pop(r.id, None)
                    continue
                if not self.health:
                    self.watchers[r.id].force(now)  # the server is still starting: read it once it is up
                    if now - self.waiting_toast > 15:
                        self.waiting_toast = now
                        self.toast("กำลังเปิดระบบแปล… รอสักครู่", 3)
                    continue
                if r.id in self.busy:
                    self.watchers[r.id].force(now)  # read again once the running request is back
                    continue
                self.busy.add(r.id)
                self.pool.submit(self.job_region, r, px, crop.copy(), self.gen[r.id])

    def job_region(self, r, px, crop: np.ndarray, gen: int):
        try:
            payload = self._region_payload(r, px, crop, gen)
        except LensError as e:
            log.warning("region %s: %s", r.id, e)
            payload = {"error": str(e)}
        except Exception as e:  # noqa: BLE001 - a worker must always report back
            log.exception("region %s failed", r.id)
            payload = {"error": f"{type(e).__name__}: {e}"}
        self.bridge.region_done.emit(r.id, gen, payload)

    def _region_payload(self, r, px, crop, gen) -> dict | None:
        p = self.profile
        t0 = time.perf_counter()
        ocr = self.api.ocr_lines(crop, p.src_lang)
        lines = [l for l in ocr["lines"] if l["text"].strip()]
        text = ocr["text"].strip()
        log.info("region %s ocr %d ms: %r", r.id, (time.perf_counter() - t0) * 1000, text)
        if not lines or not text:
            return None
        prev = self.shown.get(r.id)
        if prev and similar(prev["src"], text) >= 0.92:
            return {"reuse": True}
        # tell the UI a translation is on its way (dots at the region's top-right corner)
        self.bridge.pending.emit(r.id, gen, (px[0] + px[2], px[1] + 4))
        t1 = time.perf_counter()
        ctx = list(self.context) if r.kind != "name" else []
        res = self.api.translate_lines([{"id": "a", "src": text, "kind": r.kind}], series=p.series, target=p.target,
                                       src_lang=p.src_lang, engine=p.engine, allow_cloud=p.engine in REMOTE, context=ctx)
        tl = res["lines"].get("a", "")
        log.info("region %s %s %d ms%s: %r", r.id, res.get("engine"), (time.perf_counter() - t1) * 1000,
                 " (cache)" if res.get("cached") else "", tl)
        boxes = [l["bbox"] for l in lines]
        patch = erase_patch(crop, boxes)
        heights = sorted(b[3] for b in boxes)
        # line boxes are tight around the glyphs: the font's pixel size is a little more than the ink height
        size = min(heights[len(heights) // 2] * 1.15, 0.8 * px[3])  # never taller than the region
        item = Item(rect=text_rect_for(px, boxes, r.kind), text=tl, fg=lines[0].get("fg", "#ffffff"),
                    max_px=max(12.0, size * p.font_scale), vcenter=r.kind != "dialog", align=r.align, bold=r.kind == "name",
                    patch=to_qimage(patch) if patch is not None and tl else None, patch_pos=(px[0], px[1]))
        item._scale = p.font_scale
        return {"src": text, "tl": tl, "items": [item] if tl else [], "engine": res.get("engine", ""), "kind": r.kind}

    def on_region_done(self, rid: str, gen: int, payload):
        self.busy.discard(rid)
        self.overlay.set_pending(rid, None)
        if isinstance(payload, dict) and "error" in payload:
            msg, at = self.last_error
            if payload["error"] != msg or time.monotonic() - at > 20:  # do not repeat the same error every line
                self.last_error = (payload["error"], time.monotonic())
                self.toast(friendly_error(payload["error"]), 4)
            return
        if gen != self.gen[rid]:
            return  # the text changed while we were reading it; a newer read is coming
        if payload is None:
            self.shown.pop(rid, None)
            return
        if payload.get("reuse"):
            prev = self.shown.get(rid)
            if prev:
                self.overlay.set_items(rid, prev["items"])
            return
        self.shown[rid] = payload
        self.overlay.set_items(rid, payload["items"])
        if payload["tl"]:
            e = self.history.add(payload["kind"], payload["src"], payload["tl"], payload["engine"])
            if self.history_win:
                self.history_win.append(e)
            if payload["kind"] != "name":
                self.context.append({"src": payload["src"], "tl": payload["tl"]})

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
        skip = [region_px(r.rect, frame.shape) for r in self.profile.regions if self.overlay.items.get(r.id)]
        self.pool.submit(self.job_screen, frame, self.screen_gen, skip, auto)

    def auto_screen_tick(self, frame: np.ndarray, now: float):
        """Menus and quest lists stand still while gameplay moves: translate the screen once it has been still
        for a moment and is not the picture we already handled."""
        m = ink_map(frame, "any")
        prev, self.auto_prev = self.auto_prev, m
        if prev is None or changed_pixels(prev, m) > 0.01 * m.size:
            self.auto_still_since = now
            return
        if now - self.auto_still_since < 0.7 or self.screen_busy or self.screen_ref is not None or not self.health:
            return
        if self.auto_done is not None and changed_pixels(self.auto_done, m) <= 0.03 * m.size:
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
            ocr = self.api.ocr_lines(frame, p.src_lang)
            lines = [l for l in ocr["lines"] if any(c.isalpha() for c in l["text"]) and not in_any(l["bbox"], skip)]
            log.info("screen%s: %d lines", " (auto)" if auto else "", len(lines))
            if not lines:
                self.bridge.screen_done.emit(gen, None if auto else [])
                return
            res = self.api.translate_lines([{"id": l["id"], "src": l["text"], "kind": "ui"} for l in lines], series=p.series,
                                           target=p.target, src_lang=p.src_lang, engine=p.engine,
                                           allow_cloud=p.engine in REMOTE, context=[])
            patch = erase_patch(frame, [l["bbox"] for l in lines])
            items = []
            for l in lines:
                tl = res["lines"].get(l["id"], "")
                if not tl:
                    continue
                x, y, w, h = l["bbox"]
                pad = int(0.45 * h)
                items.append(Item(rect=(x, y - pad, int(w * 1.25) + 8, h + 2 * pad), text=tl, fg=l.get("fg", "#ffffff"),
                                  max_px=max(11.0, h * 1.15 * p.font_scale), vcenter=True))
            if patch is not None and items:
                items[0].patch, items[0].patch_pos = to_qimage(patch), (0, 0)
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
            self.toast(friendly_error(items["error"]), 4)
            self.screen_ref = None
            return
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
        find = (lambda f: self.api.ocr_lines(f, p.src_lang)["lines"]) if self.health else None
        self.editor = RegionEditor(frame, p.regions, self.overlay.geometry(), find)
        self.editor.setStyleSheet(QSS)

        def done(regions=None):
            if regions is not None:
                self.profile.regions = regions
                self.profile.save()
                self.shown.clear()
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
