"""HTTP client for the game-lens server (server/: OCR + translation)."""
from __future__ import annotations

from pathlib import Path

import cv2
import httpx
import numpy as np


class LensError(RuntimeError):
    pass


class LensClient:
    def __init__(self, base: str, token_file: Path, transport: httpx.BaseTransport | None = None):
        self.base = base.rstrip("/")
        self.token_file = Path(token_file)
        self.http = httpx.Client(timeout=httpx.Timeout(60, connect=3), transport=transport)

    def _headers(self) -> dict:
        try:
            return {"X-Lens-Token": self.token_file.read_text().strip()}
        except OSError as e:
            raise LensError(f"อ่าน token ไม่ได้: {self.token_file} (ระบบแปลยังไม่เคยเปิด)") from e

    def _call(self, method: str, path: str, **kw) -> dict:
        try:
            r = self.http.request(method, self.base + path, headers=self._headers(), **kw)
        except httpx.HTTPError as e:
            raise LensError(f"ต่อ server ไม่ได้ ({self.base}): {e}") from e
        if r.status_code != 200:
            try:
                detail = r.json().get("detail", r.text[:200])
            except Exception:
                detail = r.text[:200]
            raise LensError(f"server HTTP {r.status_code}: {detail}")
        return r.json()

    def health(self) -> dict:
        try:
            return self.http.get(self.base + "/health", timeout=5).json()
        except (httpx.HTTPError, ValueError) as e:
            raise LensError(f"ต่อ server ไม่ได้ ({self.base}): {e}") from e

    def ocr_lines(self, bgr: np.ndarray, src_lang: str = "ja", kind: str = "", series: str = "", target: str = "th",
                  align: str = "left", frame_h: int = 0) -> dict:
        """frame_h: height of the whole game picture. The text of a small window (720p) is enlarged to the size it
        has at 1080p before it is read, which reads clearly better; boxes come back in the picture's own pixels."""
        k = min(1.5, 1080 / frame_h) if 0 < frame_h < 1000 else 1.0
        if k > 1.0:
            bgr = cv2.resize(bgr, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC)
        ok, png = cv2.imencode(".png", bgr, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        if not ok:
            raise LensError("encode image failed")
        files = {"image": ("r.png", png.tobytes(), "image/png")}
        got = self._call("POST", "/ocr/lines", files=files, data={"src_lang": src_lang, "kind": kind, "series": series, "target": target, "align": align})
        if k > 1.0:
            for l in got["lines"]:
                for b in ("bbox", "dbox"):
                    if b in l:
                        l[b] = [round(v / k) for v in l[b]]
        return got

    def translate_lines(self, lines: list[dict], *, series: str, target: str, src_lang: str, engine: str,
                        allow_cloud: bool, context: list[dict]) -> dict:
        body = {"lines": lines, "series": series, "target": target, "src_lang": src_lang, "engine": engine,
                "allow_cloud": allow_cloud, "context": context}
        return self._call("POST", "/translate/lines", json=body)

    def usage(self) -> dict:
        return self._call("GET", "/usage")

    def extract_glossary(self, series: str, text: str, target: str, engine: str, allow_cloud: bool) -> dict:
        body = {"text": text, "target": target, "engine": engine, "allow_cloud": allow_cloud}
        return self._call("POST", f"/glossary/{series}/extract-text", json=body)

    def glossary(self, series: str, target: str) -> dict:
        return self._call("GET", f"/glossary/{series}", params={"target": target})["glossary"]

    def put_glossary(self, series: str, target: str, glossary: dict) -> int:
        return self._call("PUT", f"/glossary/{series}", params={"target": target}, json=glossary)["version"]

    # --- settings ---------------------------------------------------------------------------------------------
    def keys(self) -> dict:
        return self._call("GET", "/config/keys")

    def set_keys(self, keys: dict[str, str]) -> dict:
        return self._call("PUT", "/config/keys", json=keys)

    def test_engine(self, engine: str, target: str = "th") -> dict:
        return self._call("POST", f"/config/test/{engine}", params={"target": target})
