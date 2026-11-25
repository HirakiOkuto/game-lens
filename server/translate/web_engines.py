"""Sentence-level web translation engines: the DeepL API, Azure Translator, and Google's keyless `gtx`
endpoint (the one LunaTranslator and browser extensions use).

DeepL API Free (500k characters/month) and Azure Translator F0 (2M characters/month) are free with a key.
The keyless endpoint is unofficial: it rate-limits per IP (HTTP 429) and can change at any time.
(Checked 2026-11-23: Edge's anonymous Bing token endpoint answers 404 and DeepL's web JSON-RPC answers
"older version of DeepL", so neither is implemented.) Every failure raises
EngineError, so `engine_order` falls back to the local LLM instead of leaving a line untranslated.

None of these take a glossary for Thai (DeepL glossaries exclude Thai; the others have none), so names are
locked on our side (`lock` / `unlock`): a glossary name in the source is swapped for its translation, wrapped
in a no-translate tag where the engine honours one, and the tags are stripped from the answer.
"""
from __future__ import annotations

import html
import os
import re
from concurrent.futures import ThreadPoolExecutor

import httpx

from .engines import EngineError, glossary_terms, terms_in_text

_PUA = 0xE000  # private-use code points: placeholders no source text contains


def terms_in(texts: list[str], glossary: dict) -> list[tuple[str, str]]:
    """Glossary (as written in the texts, translation) pairs that occur in any of the texts, longest first."""
    terms = glossary_terms(glossary or {})
    used: dict[str, str] = {}
    for t in texts:
        for surface, tl in terms_in_text(t, terms):
            used.setdefault(surface, tl)
    return sorted(used.items(), key=lambda t: -len(t[0]))


def lock(text: str, terms: list[tuple[str, str]], mode: str) -> str:
    """mode: 'xml' (DeepL API, <k> ignore tag), 'html' (Google/Bing, translate=no span), 'inline' (no tags)."""
    slots: list[str] = []
    for src, tl in terms:
        if src in text:
            text = text.replace(src, chr(_PUA + len(slots)))
            slots.append(tl)
    if mode in ("xml", "html"):
        text = html.escape(text, quote=False)
    for i, tl in enumerate(slots):
        if mode == "xml":
            rep = f"<k>{html.escape(tl, quote=False)}</k>"
        elif mode == "html":
            rep = f'<span translate="no" class="notranslate">{html.escape(tl, quote=False)}</span>'
        else:
            rep = tl
        text = text.replace(chr(_PUA + i), rep)
    return text


def unlock(text: str, mode: str) -> str:
    if mode == "xml":
        return html.unescape(re.sub(r"</?k>", "", text)).strip()
    if mode == "html":
        return html.unescape(re.sub(r"</?span[^>]*>", "", text)).strip()
    return text.strip()


def _base(code: str) -> str:
    return (code or "").lower().split("-")[0]


# --- DeepL API (key) -------------------------------------------------------------------------------------------
class DeepLEngine:
    """DeepL API v2 with DEEPL_API_KEY (keys ending in ':fx' are Free-plan keys and use api-free.deepl.com).

    The previous lines go in `context`, which DeepL does not bill. Up to 50 lines per request.
    """

    name = "deepl"
    TARGETS = {"th": "TH", "en": "EN-US"}

    def __init__(self, transport: httpx.BaseTransport | None = None):
        self.key = os.environ.get("DEEPL_API_KEY", "")
        self.base = "https://api-free.deepl.com" if self.key.endswith(":fx") else "https://api.deepl.com"
        self.model_type = os.environ.get("DEEPL_MODEL_TYPE", "")  # e.g. latency_optimized / quality_optimized
        self.http = httpx.Client(timeout=30, transport=transport)

    def available(self) -> bool:
        return bool(self.key)

    def _headers(self):
        return {"Authorization": f"DeepL-Auth-Key {self.key}"}

    @staticmethod
    def _src(code: str) -> str:
        b = _base(code)
        return {"zh": "ZH", "ja": "JA", "ko": "KO", "en": "EN"}.get(b, b.upper())

    def _post(self, body: dict) -> list[str]:
        try:
            r = self.http.post(f"{self.base}/v2/translate", json=body, headers=self._headers())
        except httpx.HTTPError as e:
            raise EngineError(f"DeepL unreachable: {e}") from e
        if r.status_code != 200:
            try:
                msg = r.json().get("message") or r.text[:200]
            except Exception:
                msg = r.text[:200]
            hint = {403: " (bad key)", 456: " (quota exceeded)", 429: " (rate limited)"}.get(r.status_code, "")
            raise _DeepLHttp(r.status_code, f"DeepL HTTP {r.status_code}{hint}: {msg}")
        return [t["text"] for t in r.json()["translations"]]

    def translate_page(self, texts: list[str], src: str, target: str, context: list[dict], glossary: dict,
                       style: str = "manga") -> list[str]:
        if not self.key:
            raise EngineError("DeepL: DEEPL_API_KEY is not set")
        terms = terms_in(texts, glossary)
        ctx = "\n".join(c["src"] for c in context[-8:] if c.get("src"))
        out: list[str] = []
        for i in range(0, len(texts), 50):
            chunk = texts[i : i + 50]
            mode = "xml" if terms else "inline"
            body = {"text": [lock(t, terms, mode) for t in chunk], "source_lang": self._src(src),
                    "target_lang": self.TARGETS.get(target, target.upper())}
            if ctx:
                body["context"] = ctx
            if self.model_type:
                body["model_type"] = self.model_type
            if mode == "xml":
                body.update(tag_handling="xml", ignore_tags=["k"])
            try:
                res = self._post(body)
            except _DeepLHttp as e:
                if e.status != 400 or mode != "xml":
                    raise EngineError(str(e)) from e
                # the language pair may not support tag handling: put the names inline instead
                mode = "inline"
                body["text"] = [lock(t, terms, mode) for t in chunk]
                body.pop("tag_handling"), body.pop("ignore_tags")
                try:
                    res = self._post(body)
                except _DeepLHttp as e2:
                    raise EngineError(str(e2)) from e2
            out += [unlock(t, mode) for t in res]
        return out

    def usage(self) -> dict:
        if not self.key:
            raise EngineError("DeepL: DEEPL_API_KEY is not set")
        try:
            r = self.http.get(f"{self.base}/v2/usage", headers=self._headers())
        except httpx.HTTPError as e:
            raise EngineError(f"DeepL unreachable: {e}") from e
        if r.status_code != 200:
            raise EngineError(f"DeepL HTTP {r.status_code}")
        d = r.json()
        return {"used": d.get("character_count", 0), "limit": d.get("character_limit", 0)}


class _DeepLHttp(EngineError):
    def __init__(self, status: int, msg: str):
        super().__init__(msg)
        self.status = status


# --- Google, keyless (gtx) -------------------------------------------------------------------------------------
class GoogleFreeEngine:
    """translate.googleapis.com's `client=gtx` endpoint (what browser extensions and LunaTranslator use)."""

    name = "google_free"
    URL = "https://translate.googleapis.com/translate_a/single"
    LANGS = {"zh": "zh-CN", "zh-cn": "zh-CN", "zh-hk": "zh-TW", "zh-tw": "zh-TW"}

    def __init__(self, transport: httpx.BaseTransport | None = None):
        self.http = httpx.Client(timeout=15, transport=transport, headers={"User-Agent": "Mozilla/5.0"})

    def available(self) -> bool:
        return True

    def _lang(self, code: str) -> str:
        c = (code or "").lower()
        return self.LANGS.get(c, c.split("-")[0])

    def _one(self, text: str, src: str, target: str) -> str:
        params = {"client": "gtx", "sl": self._lang(src), "tl": self._lang(target), "dt": "t", "q": text}
        try:
            r = self.http.get(self.URL, params=params)
        except httpx.HTTPError as e:
            raise EngineError(f"Google (free) unreachable: {e}") from e
        if r.status_code != 200:
            raise EngineError(f"Google (free) HTTP {r.status_code}{' (rate limited)' if r.status_code == 429 else ''}")
        try:
            return "".join(seg[0] for seg in r.json()[0] if seg and seg[0])
        except (ValueError, TypeError, IndexError) as e:
            raise EngineError("Google (free) returned an unexpected answer") from e

    def translate_page(self, texts: list[str], src: str, target: str, context: list[dict], glossary: dict,
                       style: str = "manga") -> list[str]:
        terms = terms_in(texts, glossary)
        locked = [lock(t, terms, "inline") for t in texts]
        with ThreadPoolExecutor(min(4, max(1, len(texts)))) as pool:
            return [t.strip() for t in pool.map(lambda t: self._one(t, src, target), locked)]


# --- Microsoft Azure Translator (key; the F0 tier is free up to 2M characters a month) ------------------------
class AzureTranslatorEngine:
    """Azure AI Translator v3 with AZURE_TRANSLATOR_KEY (+ AZURE_TRANSLATOR_REGION for regional resources).

    Names are locked with class="notranslate" spans (textType=html). Up to 25 lines per request.
    """

    name = "azure"
    URL = "https://api.cognitive.microsofttranslator.com/translate"
    LANGS = {"zh": "zh-Hans", "zh-cn": "zh-Hans", "zh-hk": "zh-Hant", "zh-tw": "zh-Hant"}

    def __init__(self, transport: httpx.BaseTransport | None = None):
        self.key = os.environ.get("AZURE_TRANSLATOR_KEY", "")
        self.region = os.environ.get("AZURE_TRANSLATOR_REGION", "")
        self.http = httpx.Client(timeout=20, transport=transport)

    def available(self) -> bool:
        return bool(self.key)

    def _lang(self, code: str) -> str:
        c = (code or "").lower()
        return self.LANGS.get(c, c.split("-")[0])

    def _call(self, texts: list[str], src: str, target: str) -> list[str]:
        headers = {"Ocp-Apim-Subscription-Key": self.key}
        if self.region:
            headers["Ocp-Apim-Subscription-Region"] = self.region
        params = {"api-version": "3.0", "from": self._lang(src), "to": self._lang(target), "textType": "html"}
        try:
            r = self.http.post(self.URL, params=params, json=[{"Text": t} for t in texts], headers=headers)
        except httpx.HTTPError as e:
            raise EngineError(f"Azure Translator unreachable: {e}") from e
        if r.status_code != 200:
            try:
                msg = r.json()["error"]["message"]
            except Exception:
                msg = r.text[:200]
            hint = {401: " (bad key or region)", 403: " (quota exceeded)", 429: " (rate limited)"}.get(r.status_code, "")
            raise EngineError(f"Azure Translator HTTP {r.status_code}{hint}: {msg}")
        return [item["translations"][0]["text"] for item in r.json()]

    def translate_page(self, texts: list[str], src: str, target: str, context: list[dict], glossary: dict,
                       style: str = "manga") -> list[str]:
        if not self.key:
            raise EngineError("Azure Translator: AZURE_TRANSLATOR_KEY is not set")
        terms = terms_in(texts, glossary)
        out: list[str] = []
        for i in range(0, len(texts), 25):
            out += [unlock(t, "html") for t in self._call([lock(t, terms, "html") for t in texts[i : i + 25]], src, target)]
        return out
