"""Small ctypes wrappers around user32/dwmapi/kernel32. Read-only queries about other windows, nothing injected."""
from __future__ import annotations

import ctypes
import os
from ctypes import wintypes

user32 = ctypes.WinDLL("user32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
dwmapi = ctypes.WinDLL("dwmapi")

WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
user32.EnumWindows.argtypes = [WNDENUMPROC, wintypes.LPARAM]
user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
user32.IsWindowVisible.argtypes = [wintypes.HWND]
user32.IsWindow.argtypes = [wintypes.HWND]
user32.IsIconic.argtypes = [wintypes.HWND]
user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
user32.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
user32.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.POINT)]
user32.GetForegroundWindow.restype = wintypes.HWND
user32.GetAsyncKeyState.argtypes = [ctypes.c_int]
user32.GetAsyncKeyState.restype = ctypes.c_short
user32.RegisterHotKey.argtypes = [wintypes.HWND, ctypes.c_int, wintypes.UINT, wintypes.UINT]
user32.UnregisterHotKey.argtypes = [wintypes.HWND, ctypes.c_int]
user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
kernel32.OpenProcess.restype = wintypes.HANDLE
kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
kernel32.QueryFullProcessImageNameW.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)]
kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
kernel32.GetCurrentProcess.restype = wintypes.HANDLE
kernel32.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
dwmapi.DwmGetWindowAttribute.argtypes = [wintypes.HWND, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD]

PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
DWMWA_EXTENDED_FRAME_BOUNDS = 9
WM_HOTKEY = 0x0312
MOD_ALT, MOD_CONTROL, MOD_SHIFT, MOD_NOREPEAT = 0x1, 0x2, 0x4, 0x4000

VK = {f"F{i}": 0x6F + i for i in range(1, 13)}
VK.update({"ESC": 0x1B, "ESCAPE": 0x1B, "TAB": 0x09, "SPACE": 0x20, "INSERT": 0x2D, "HOME": 0x24, "END": 0x23,
           "PAGEUP": 0x21, "PAGEDOWN": 0x22, "PAUSE": 0x13})
VK.update({chr(c): c for c in range(ord("A"), ord("Z") + 1)})
VK.update({str(d): 0x30 + d for d in range(10)})


def parse_hotkey(spec: str) -> tuple[int, int]:
    """'Ctrl+Shift+F8' -> (modifiers, vk)."""
    mods, vk = 0, 0
    for part in spec.replace(" ", "").split("+"):
        p = part.upper()
        if p in ("CTRL", "CONTROL"):
            mods |= MOD_CONTROL
        elif p == "ALT":
            mods |= MOD_ALT
        elif p == "SHIFT":
            mods |= MOD_SHIFT
        elif p in VK:
            vk = VK[p]
        else:
            raise ValueError(f"unknown key {part!r} in hotkey {spec!r}")
    if not vk:
        raise ValueError(f"hotkey {spec!r} has no key")
    return mods, vk


def window_title(hwnd: int) -> str:
    n = user32.GetWindowTextLengthW(hwnd)
    buf = ctypes.create_unicode_buffer(n + 1)
    user32.GetWindowTextW(hwnd, buf, n + 1)
    return buf.value


def process_exe(hwnd: int) -> str:
    pid = wintypes.DWORD()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    h = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid.value)
    if not h:
        return ""
    try:
        size = wintypes.DWORD(1024)
        buf = ctypes.create_unicode_buffer(size.value)
        if kernel32.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(size)):
            return os.path.basename(buf.value)
        return ""
    finally:
        kernel32.CloseHandle(h)


GWL_EXSTYLE = -20
WS_EX_TRANSPARENT, WS_EX_TOOLWINDOW, WS_EX_NOACTIVATE = 0x20, 0x80, 0x08000000
DWMWA_CLOAKED = 14
user32.GetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int]
user32.GetWindowLongW.restype = ctypes.c_long


def is_app_window(hwnd: int) -> bool:
    """A window a user could be playing in: not an invisible overlay, tool window or cloaked UWP shell."""
    ex = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
    if ex & (WS_EX_TRANSPARENT | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE):
        return False
    cloaked = wintypes.DWORD()
    dwmapi.DwmGetWindowAttribute(hwnd, DWMWA_CLOAKED, ctypes.byref(cloaked), ctypes.sizeof(cloaked))
    return not cloaked.value


def list_windows() -> list[dict]:
    """Visible top-level application windows with a title: [{hwnd, title, exe}]."""
    out = []

    def cb(hwnd, _):
        if user32.IsWindowVisible(hwnd) and user32.GetWindowTextLengthW(hwnd) > 0 and is_app_window(hwnd):
            out.append({"hwnd": int(hwnd), "title": window_title(hwnd), "exe": process_exe(hwnd)})
        return True

    user32.EnumWindows(WNDENUMPROC(cb), 0)
    return out


def find_window(exe: str = "", title: str = "") -> int | None:
    """First visible window whose process is `exe` (case-insensitive) and/or whose title contains `title`."""
    exe, title = exe.lower(), title.lower()
    best = None
    for w in list_windows():
        if exe and w["exe"].lower() != exe:
            continue
        if title and title not in w["title"].lower():
            continue
        # a game usually has one big window; prefer the largest
        r = client_rect(w["hwnd"])
        area = r[2] * r[3] if r else 0
        if best is None or area > best[0]:
            best = (area, w["hwnd"])
    return best[1] if best and best[0] > 0 else None


def is_window(hwnd: int) -> bool:
    return bool(hwnd and user32.IsWindow(hwnd))


def is_minimized(hwnd: int) -> bool:
    return bool(user32.IsIconic(hwnd))


def foreground() -> int:
    return int(user32.GetForegroundWindow() or 0)


def client_rect(hwnd: int) -> tuple[int, int, int, int] | None:
    """Client area in physical screen pixels: (x, y, w, h)."""
    rc = wintypes.RECT()
    if not user32.GetClientRect(hwnd, ctypes.byref(rc)):
        return None
    pt = wintypes.POINT(0, 0)
    user32.ClientToScreen(hwnd, ctypes.byref(pt))
    return pt.x, pt.y, rc.right - rc.left, rc.bottom - rc.top


def frame_bounds(hwnd: int) -> tuple[int, int, int, int] | None:
    """The window's visible bounds (what Windows Graphics Capture captures), physical pixels."""
    rc = wintypes.RECT()
    if dwmapi.DwmGetWindowAttribute(hwnd, DWMWA_EXTENDED_FRAME_BOUNDS, ctypes.byref(rc), ctypes.sizeof(rc)) != 0:
        if not user32.GetWindowRect(hwnd, ctypes.byref(rc)):
            return None
    return rc.left, rc.top, rc.right - rc.left, rc.bottom - rc.top


def key_down(vk: int) -> bool:
    return bool(user32.GetAsyncKeyState(vk) & 0x8000)


def register_hotkey(hwnd: int, hid: int, spec: str) -> bool:
    mods, vk = parse_hotkey(spec)
    return bool(user32.RegisterHotKey(hwnd, hid, mods | MOD_NOREPEAT, vk))


def unregister_hotkey(hwnd: int, hid: int):
    user32.UnregisterHotKey(hwnd, hid)
