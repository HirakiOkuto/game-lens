"""Windows Graphics Capture of one game window (the same API OBS uses): nothing is injected into the game.
Window capture sees only the game's own pixels, never our overlay drawn on top of it."""
from __future__ import annotations

import sys
import threading
import time

import numpy as np

from . import win32


class GameCapture:
    def __init__(self, hwnd: int, min_interval_ms: int = 40):
        from windows_capture import WindowsCapture

        self.hwnd = hwnd
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._ts = 0.0
        self.closed = False
        win11 = sys.getwindowsversion().build >= 22000
        cap = WindowsCapture(
            cursor_capture=False,
            draw_border=False if win11 else None,  # the yellow border can only be turned off on Windows 11
            minimum_update_interval=min_interval_ms if win11 else None,  # Windows 10 cannot throttle
            window_hwnd=hwnd,
        )

        @cap.event
        def on_frame_arrived(frame, control):
            if self.closed:
                control.stop()
                return
            buf = frame.frame_buffer  # BGRA view valid only inside this callback
            with self._lock:
                if self._frame is None or self._frame.shape != buf.shape:
                    self._frame = np.empty(buf.shape, np.uint8)
                np.copyto(self._frame, buf)
                self._ts = time.monotonic()

        @cap.event
        def on_closed():
            self.closed = True

        self._control = cap.start_free_threaded()

    def stop(self):
        # Only raise the flag: the capture thread stops itself on its next frame (see on_frame_arrived).
        # Calling the native stop() here joins that thread while we hold the GIL it needs for the callback,
        # which deadlocks the UI thread.
        self.closed = True

    def client_frame(self) -> tuple[np.ndarray | None, float]:
        """The client area (no title bar / borders) as BGR, plus the frame time."""
        with self._lock:
            if self._frame is None:
                return None, 0.0
            full, ts = self._frame.copy(), self._ts
        cr, fb = win32.client_rect(self.hwnd), win32.frame_bounds(self.hwnd)
        if cr and fb and cr[2] > 0 and cr[3] > 0:
            sx = full.shape[1] / max(1, fb[2])  # 1.0 unless the window is DPI-virtualised
            sy = full.shape[0] / max(1, fb[3])
            x0, y0 = int(round((cr[0] - fb[0]) * sx)), int(round((cr[1] - fb[1]) * sy))
            x1, y1 = x0 + int(round(cr[2] * sx)), y0 + int(round(cr[3] * sy))
            x0, y0 = max(0, x0), max(0, y0)
            if x1 <= full.shape[1] and y1 <= full.shape[0] and x1 > x0 and y1 > y0:
                full = full[y0:y1, x0:x1]
        return np.ascontiguousarray(full[:, :, :3]), ts
