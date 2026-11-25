"""Translation engines for game text.

LlamaCppEngine (a general LLM on a local llama.cpp server) and ClaudeEngine answer chat_json(system, user, schema);
the translation models (Hy-MT, PLaMo, TranslateGemma) and the web engines translate line by line.
"""
from __future__ import annotations

import json
import os
import re
import time

import httpx

from . import prompts


class EngineError(RuntimeError):
    pass


class LlamaCppEngine:
    name = "local"

    def __init__(self):
        self.url = os.environ.get("LLM_URL", "http://127.0.0.1:9940/v1")
        self.model = os.environ.get("LLM_MODEL", "gemma26b")
        self.timeout = float(os.environ.get("LLM_TIMEOUT_S", "600"))

    def available(self) -> bool:
        try:
            return httpx.get(f"{self.url}/models", timeout=0.8).status_code == 200
        except httpx.HTTPError:
            return False

    def chat_json(self, system: str, user: str, schema: dict, max_tokens: int = 4096) -> dict:
        body = {
            "model": self.model,
            "temperature": 0.3,
            "max_tokens": max_tokens,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            # Gemma 4 thinks by default; for translation we want the answer only.
            "chat_template_kwargs": {"enable_thinking": False},
            "response_format": {"type": "json_schema", "json_schema": {"name": "out", "strict": True, "schema": schema}},
        }
        try:
            r = httpx.post(f"{self.url}/chat/completions", json=body, timeout=self.timeout)
        except httpx.HTTPError as e:
            raise EngineError(f"llama.cpp unreachable at {self.url}: {e}") from e
        if r.status_code != 200:
            raise EngineError(f"llama.cpp HTTP {r.status_code}: {r.text[:300]}")
        content = r.json()["choices"][0]["message"].get("content") or ""
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            raise EngineError(f"llama.cpp returned invalid JSON ({len(content)} chars)") from e


class ClaudeEngine:
    name = "cloud"

    def __init__(self):
        import anthropic

        self._anthropic = anthropic
        self.model = os.environ.get("CLOUD_MODEL", "claude-opus-5-5")
        self.effort = os.environ.get("CLOUD_EFFORT", "low")
        self._client = None

    def available(self) -> bool:
        return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN") or os.environ.get("ANTHROPIC_PROFILE"))

    @property
    def client(self):
        if self._client is None:
            self._client = self._anthropic.Anthropic()
        return self._client

    def chat_json(self, system: str, user: str, schema: dict, max_tokens: int = 16000) -> dict:
        a = self._anthropic
        try:
            resp = self.client.beta.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
                output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": schema}},
                # on a policy decline the API re-runs the request on a fallback model inside the same call
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except a.RateLimitError as e:
            raise EngineError(f"Claude rate limited: {e}") from e
        except a.APIStatusError as e:
            raise EngineError(f"Claude HTTP {e.status_code}: {e.message}") from e
        except a.APIConnectionError as e:
            raise EngineError(f"Claude unreachable: {e}") from e
        if resp.stop_reason == "refusal":
            raise EngineError(f"Claude declined ({resp.stop_details.category if resp.stop_details else 'unknown'})")
        if resp.stop_reason == "max_tokens":
            raise EngineError("Claude hit max_tokens")
        text = "".join(b.text for b in resp.content if b.type == "text")
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise EngineError("Claude returned invalid JSON") from e


class GoogleTranslateEngine:
    """Google Cloud Translation API (v2 / Basic) with an API key.

    Machine translation, one text box at a time: fast and cheap, but it sees no page context, uses no glossary
    and does not repair OCR mistakes, so dialogue quality is below the LLM engines.
    """

    name = "google"
    URL = "https://translation.googleapis.com/language/translate/v2"
    # MangaDex-style codes -> Google codes (others: the part before '-')
    LANGS = {"zh": "zh-CN", "zh-cn": "zh-CN", "zh-hk": "zh-TW", "zh-tw": "zh-TW", "ja-ro": "ja", "ko-ro": "ko"}

    def __init__(self, transport: httpx.BaseTransport | None = None):
        self.key = os.environ.get("GOOGLE_TRANSLATE_API_KEY", "")
        self.http = httpx.Client(timeout=30, transport=transport)

    def available(self) -> bool:
        return bool(self.key)

    def _lang(self, code: str) -> str:
        c = (code or "").lower()
        return self.LANGS.get(c, c.split("-")[0])

    def translate(self, texts: list[str], src: str, target: str, glossary: dict | None = None) -> list[str]:
        if not self.key:
            raise EngineError("Google Translate: GOOGLE_TRANSLATE_API_KEY is not set")
        from .web_engines import lock, terms_in, unlock

        # glossary names go in as their translation inside a translate="no" span (html mode); plain text otherwise
        terms = terms_in(texts, glossary or {})
        fmt = "html" if terms else "text"
        if terms:
            texts = [lock(t, terms, "html") for t in texts]
        out: list[str] = []
        for i in range(0, len(texts), 100):  # the API takes at most 128 segments per request
            body = {"q": texts[i : i + 100], "target": self._lang(target), "source": self._lang(src), "format": fmt}
            try:
                r = self.http.post(self.URL, params={"key": self.key}, json=body)
            except httpx.HTTPError as e:
                raise EngineError(f"Google Translate unreachable: {e}") from e
            if r.status_code != 200:
                try:
                    msg = r.json()["error"]["message"]
                except Exception:
                    msg = r.text[:200]
                raise EngineError(f"Google Translate HTTP {r.status_code}: {msg}")
            out += [t["translatedText"] for t in r.json()["data"]["translations"]]
        return [unlock(t, "html") for t in out] if terms else out


class TranslateGemmaEngine:
    """Google's TranslateGemma (open translation model, Gemma 3 based) on a local llama.cpp server.

    Offline and free, like Google Translate but on this machine. It translates one text at a time with its
    official prompt (taken from the model's chat template), so it sees no page context and no glossary.
    Start it with:  llama serve -m models/translategemma-12b-it.Q4_K_M.gguf --port 9941 -c 8192 -np 4 --jinja
    """

    name = "tgemma"
    LANGS = {"zh-cn": "zh", "zh-hk": "zh-TW", "zh-tw": "zh-TW", "es-la": "es", "pt-br": "pt", "ja-ro": "ja", "ko-ro": "ko"}
    NAMES = {"ja": "Japanese", "ko": "Korean", "zh": "Chinese", "zh-TW": "Chinese", "en": "English", "th": "Thai",
             "vi": "Vietnamese", "id": "Indonesian", "es": "Spanish", "pt": "Portuguese", "fr": "French",
             "de": "German", "it": "Italian", "ru": "Russian", "uk": "Ukrainian", "pl": "Polish", "tr": "Turkish",
             "ar": "Arabic", "ms": "Malay"}

    def __init__(self):
        self.url = os.environ.get("TGEMMA_URL", "http://127.0.0.1:9941")
        self.parallel = int(os.environ.get("TGEMMA_PARALLEL", "4"))  # match llama serve -np

    def available(self) -> bool:
        try:
            return httpx.get(f"{self.url}/health", timeout=0.8).status_code == 200
        except httpx.HTTPError:
            return False

    def _code(self, lang: str) -> str:
        c = (lang or "").lower()
        return self.LANGS.get(c, c.split("-")[0])

    def _prompt(self, text: str, src: str, tgt: str) -> str:
        s, t = self._code(src), self._code(tgt)
        S, T = self.NAMES.get(s, s), self.NAMES.get(t, t)
        return (
            f"<start_of_turn>user\nYou are a professional {S} ({s}) to {T} ({t}) translator. Your goal is to accurately "
            f"convey the meaning and nuances of the original {S} text while adhering to {T} grammar, vocabulary, and "
            f"cultural sensitivities.\nProduce only the {T} translation, without any additional explanations or "
            f"commentary. Please translate the following {S} text into {T}:\n\n\n{text.strip()}<end_of_turn>\n"
            f"<start_of_turn>model\n"
        )

    def _one(self, text: str, src: str, tgt: str) -> str:
        body = {"prompt": self._prompt(text, src, tgt), "n_predict": 512, "temperature": 0, "stop": ["<end_of_turn>"],
                "cache_prompt": True}
        try:
            r = httpx.post(f"{self.url}/completion", json=body, timeout=120)
        except httpx.HTTPError as e:
            raise EngineError(f"TranslateGemma unreachable at {self.url}: {e}") from e
        if r.status_code != 200:
            raise EngineError(f"TranslateGemma HTTP {r.status_code}: {r.text[:200]}")
        return r.json().get("content", "").strip()

    def translate(self, texts: list[str], src: str, target: str, glossary: dict | None = None) -> list[str]:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(self.parallel) as pool:
            return list(pool.map(lambda t: self._one(t, src, target), texts))


class PlamoTranslateEngine:
    """Preferred Networks' PLaMo 3 Translate on a local llama.cpp server, prompted in its PFML3 format.

    Unlike TranslateGemma it takes translation history and a glossary: every box is translated with the
    previous boxes (this page and the pages before) as input/output pairs, and series names as glossary entries.
    Start it with:  llama serve -m models/plamo-3-translate-8b-Q4_K_M.gguf --port 9942 -c 16384 --jinja
    """

    name = "plamo"
    LANGS = {"zh": "zh-hans", "zh-cn": "zh-hans", "zh-hk": "zh-hant", "zh-tw": "zh-hant", "es-la": "es",
             "pt-br": "pt", "ja-ro": "ja", "ko-ro": "ko"}
    TAG, MSG, KEY, VAL = "<|plamo:tag|>", "<|plamo:msg|>", "<|plamo:key|>", "<|plamo:val|>"
    MAX_HISTORY = 16  # input/output pairs kept as context

    def __init__(self):
        self.url = os.environ.get("PLAMO_URL", "http://127.0.0.1:9942")

    def available(self) -> bool:
        try:
            return httpx.get(f"{self.url}/health", timeout=0.8).status_code == 200
        except httpx.HTTPError:
            return False

    def _code(self, lang: str) -> str:
        c = (lang or "").lower()
        return self.LANGS.get(c, c.split("-")[0])

    def _clean(self, s: str) -> str:
        # the PFML3 control tokens must never appear inside a body
        return re.sub(r"<\|plamo:[a-z]+\|>", "", s).strip()

    def _prompt(self, text: str, history: list[tuple[str, str]], glossary: list[tuple[str, str]], src: str, tgt: str,
                style: str = "manga") -> str:
        T, M, K, V = self.TAG, self.MSG, self.KEY, self.VAL
        s, t = self._code(src), self._code(tgt)
        p = f"{T}dataset{M}translation"
        what = ("Dialogue, speaker names and UI text from a video game, in order." if style == "game"
                else "Speech bubbles and captions from one manga chapter, in reading order.")
        p += f"{T}context{M}{what} Keep each line short and conversational."
        for a, b in history[-self.MAX_HISTORY :]:
            p += f"{T}input{K}lang{V}{s}{M}{self._clean(a)}{T}output{K}lang{V}{t}{M}{self._clean(b)}"
        if glossary:
            p += f"{T}glossary" + "".join(f"<|plamo:index|>{self._clean(a)}<|plamo:item|>{self._clean(b)}" for a, b in glossary)
        p += f"{T}input{K}lang{V}{s}{M}{self._clean(text)}{T}output{K}lang{V}{t}{M}"
        return p

    def translate_page(self, texts: list[str], src: str, target: str, context: list[dict], glossary: dict,
                       style: str = "manga") -> list[str]:
        terms = glossary_terms(glossary)
        history = [(c["src"], c["tl"]) for c in context if c.get("src") and c.get("tl")]
        out = []
        for text in texts:  # sequential: each box sees the translations of the boxes before it
            used = terms_in_text(text, terms)  # only terms that occur in this box
            body = {"prompt": self._prompt(text, history, used, src, target, style), "n_predict": 512, "temperature": 0,
                    "stop": [self.TAG, "<|plamo:bos|>"], "cache_prompt": True}
            try:
                r = httpx.post(f"{self.url}/completion", json=body, timeout=120)
            except httpx.HTTPError as e:
                raise EngineError(f"PLaMo Translate unreachable at {self.url}: {e}") from e
            if r.status_code != 200:
                raise EngineError(f"PLaMo Translate HTTP {r.status_code}: {r.text[:200]}")
            tl = echo_guard(self._clean(r.json().get("content", "")), text, history)
            out.append(tl)
            if tl:
                history.append((text, tl))
        return out


# OCR often reads a small kana as its full-size twin (ミィティ -> ミイテイ, メーツァ -> メーツア)
_SMALL = str.maketrans("ぁぃぅぇぉっゃゅょゎァィゥェォッャュョヮヵヶ", "あいうえおつやゆよわアイウエオツヤユヨワカケ")


def fold_kana(s: str) -> str:
    """Small kana as full-size ones: same length, so positions in the folded text are positions in the text."""
    return s.translate(_SMALL)


def fold_name(s: str) -> str:
    """For comparing whole names: small kana folded, voicing marks dropped (テララ / デララ), spaces removed."""
    import unicodedata

    plain = "".join(c for c in unicodedata.normalize("NFD", fold_kana(s)) if unicodedata.category(c) != "Mn")
    return "".join(plain.split())


def terms_in_text(text: str, terms: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Glossary terms that occur in the text, longest first, as (the way it is written in the text, translation).
    Small kana are compared as full-size ones. A short name that only shows up inside a longer term (the name
    シャル inside the place シャルル村) is not counted."""
    used, rest = [], fold_kana(text)
    for a, b in sorted(terms, key=lambda t: -len(t[0])):
        fa = fold_kana(a)
        i = rest.find(fa) if fa else -1
        if i < 0:
            continue
        while i >= 0:
            surface = text[i : i + len(fa)]
            if (surface, b) not in used:
                used.append((surface, b))
            rest = rest[:i] + "\x00" * len(fa) + rest[i + len(fa) :]
            i = rest.find(fa)
    return used


def match_term(text: str, terms: list[tuple[str, str]]) -> tuple[str, str] | None:
    """The glossary entry (source, translation) for a text that is nothing but a known name (a speaker-name box), allowing for the misreadings
    OCR makes on short text: small kana, voicing marks, or one wrong character in a name of four or more."""
    import difflib

    want = fold_name(text)
    if not want:
        return None
    best, score = None, 0.0
    for a, b in terms:
        fa = fold_name(a)
        if fa == want:
            return a, b
        if len(fa) >= 4 and abs(len(fa) - len(want)) <= 1:
            r = difflib.SequenceMatcher(None, fa, want, autojunk=False).ratio()
            if r > score:
                best, score = (a, b), r
    return best if score >= 0.75 else None


def match_name(text: str, terms: list[tuple[str, str]]) -> str | None:
    """Its translation, or None."""
    hit = match_term(text, terms)
    return hit[1] if hit else None


def glossary_terms(glossary: dict) -> list[tuple[str, str]]:
    terms = [(c["src"], c["tl"]) for c in glossary.get("characters", []) if c.get("src") and c.get("tl")]
    terms += [(a, c["tl"]) for c in glossary.get("characters", []) for a in c.get("aliases", []) if a and c.get("tl")]
    terms += [(x["src"], x["tl"]) for x in glossary.get("terms", []) if x.get("src") and x.get("tl")]
    return terms


class HyMtEngine:
    """Tencent Hy-MT2 (translation model, 33 languages incl. Thai) on a local llama.cpp server.

    Uses the model card's "background information" and "reference translations" prompts: the previous boxes of the
    page / chapter go in as background, series names as reference translations.
    Start it with:  llama serve -m models/Hy-MT2-7B-Q4_K_M.gguf --port 9943 -c 8192 --jinja
    """

    name = "hymt"
    NAMES = {"ja": "Japanese", "ko": "Korean", "zh": "Chinese", "zh-cn": "Chinese", "zh-hk": "Traditional Chinese",
             "zh-tw": "Traditional Chinese", "en": "English", "th": "Thai", "vi": "Vietnamese", "id": "Indonesian",
             "es": "Spanish", "es-la": "Spanish", "pt": "Portuguese", "pt-br": "Portuguese", "fr": "French",
             "de": "German", "it": "Italian", "ru": "Russian", "ar": "Arabic", "ms": "Malay", "tl": "Filipino"}
    MAX_HISTORY = 12

    def __init__(self):
        self.url = os.environ.get("HYMT_URL", "http://127.0.0.1:9943")

    def available(self) -> bool:
        try:
            return httpx.get(f"{self.url}/health", timeout=0.8).status_code == 200
        except httpx.HTTPError:
            return False

    def _prompt(self, text: str, history: list[tuple[str, str]], glossary: list[tuple[str, str]], target: str,
                style: str = "manga") -> str:
        tl = self.NAMES.get(target, target)
        if style == "game":
            bg = ["These lines are dialogue, speaker names and menu text from a video game, in order. "
                  f"Write natural, spoken {tl} that sounds like an official game localisation."]
        else:
            bg = ["These lines are speech bubbles and captions from a manga, in reading order. "
                  f"Write natural, casual spoken {tl} that fits a speech bubble."]
        if history:
            bg.append("Earlier lines and their translations:")
            bg += [f"{a} → {b}" for a, b in history[-self.MAX_HISTORY :]]
        # the model card's own section for terminology: names listed under "background" get transliterated anyway
        ref = ""
        if glossary:
            ref = "[Reference Translations]\n" + "\n".join(f"{a} translates to {b}" for a, b in glossary) + "\n\n"
        return (
            "[Background Information]\n" + "\n".join(bg) + "\n\n" + ref +
            f"Please translate the following text into {tl}, taking the provided background information into "
            "consideration. Only output the translated result without any additional explanation.\n\n"
            f"[Source Text]\n{text.strip()}"
        )

    def _ask(self, prompt: str) -> str:
        body = {"messages": [{"role": "user", "content": prompt}], "temperature": 0, "max_tokens": 512, "cache_prompt": True}
        try:
            r = httpx.post(f"{self.url}/v1/chat/completions", json=body, timeout=120)
        except httpx.HTTPError as e:
            raise EngineError(f"Hy-MT unreachable at {self.url}: {e}") from e
        if r.status_code != 200:
            raise EngineError(f"Hy-MT HTTP {r.status_code}: {r.text[:200]}")
        return (r.json()["choices"][0]["message"].get("content") or "").strip()

    def _with_slots(self, text: str, history, used: list[tuple[str, str]], target: str, style: str) -> str:
        """Translate with N1, N2... standing in for the names, then put the names back. '' if a slot got lost."""
        slots = {}
        for i, (a, b) in enumerate(used, 1):
            text = text.replace(a, f"N{i}")
            slots[f"N{i}"] = b
        out = self._ask(self._prompt(text, history, [], target, style))
        for k in sorted(slots, key=len, reverse=True):  # N10 before N1
            if k not in out:
                return ""
            out = out.replace(k, slots[k])
        return out

    def translate_page(self, texts: list[str], src: str, target: str, context: list[dict], glossary: dict,
                       style: str = "manga") -> list[str]:
        terms = glossary_terms(glossary)
        history = [(c["src"], c["tl"]) for c in context if c.get("src") and c.get("tl")]
        out = []
        for text in texts:
            used = terms_in_text(text, terms)
            tl = echo_guard(self._ask(self._prompt(text, history, used, target, style)), text, history)
            if tl and any(b not in tl for _, b in used):
                # a name came out transliterated: once more with stand-ins (these lose some meaning, so only now)
                tl = self._with_slots(text, history, used, target, style) or tl
            out.append(tl)
            if tl:
                history.append((text, tl))
        return out


NOISE_RE = re.compile(r"[A-Za-z]{1,2}[\s?!.•·]*")  # a lone Latin letter or two from OCR ("E", "O") is noise


def echo_guard(tl: str, text: str, history: list[tuple[str, str]]) -> str:
    """Small translation models sometimes answer a short box by repeating an earlier line from the history."""
    if tl and any(tl == b and a != text for a, b in history):
        return ""
    return tl


CHATTER = re.compile(
    r"ขออภัย|ไม่สามารถแปล|กรุณา(ให้|ระบุ)|โปรดระบุ|ข้อความที่คุณต้องการ|I'm sorry|I cannot|I can't translate|please provide",
    re.IGNORECASE,
)


def drop_untranslated(out: dict[str, str], blocks: list[dict]) -> dict[str, str]:
    """A translation model hands OCR garbage ('E', '7nl') back unchanged; an unchanged box has nothing to show."""
    src = {b["id"]: re.sub(r"\W", "", b["src"]).lower() for b in blocks}
    return {i: ("" if t and re.sub(r"\W", "", t).lower() == src.get(i) else t) for i, t in out.items()}


def clean_mt(text: str, target: str) -> str:
    """Tidy sentence-level MT output for a speech bubble.

    Translation models fed OCR garbage or a lone syllable sometimes answer like an assistant ("Sorry, I cannot
    translate this"), list alternatives ("A / B / C") or both Thai politeness endings ("ค่ะ/ครับ").
    """
    t = text.strip().strip('"“”')
    if CHATTER.search(t):
        return ""  # an apology or request is never a translation of a bubble
    t = re.sub(r"\s*\((เลือก|choose|depending)[^)]*\)", "", t)  # "(เลือกตามเพศของผู้พูด)"
    t = re.sub(r"(ค่ะ|คะ|ครับ)\s*/\s*(ค่ะ|คะ|ครับ)", r"\1", t)  # keep one politeness ending
    # alternatives "A / B", "A | B", "A /\nB", trailing "A /": keep the first
    t = re.split(r"\s+[/|¦]\s+|\s*[/|¦]\s*(?:\n|$)", t)[0]
    t = re.sub(r"\s*\((ขึ้นอยู่กับ|หรือ อาจ|or )[^)]*\)", "", t)  # "(ขึ้นอยู่กับเพศของผู้พูด)", "(หรือ อาจหมายถึง ...)"
    t = t.replace("。", "").replace("؟", "?").rstrip("|").strip()
    if target == "th":
        t = re.sub(r"(?<=[฀-๿])\.$", "", t)  # Thai does not end sentences with a full stop
    return t.strip()


_engines: dict[str, object] = {}
WEB = {"deepl", "azure", "google_free"}  # translate/web_engines.py
REMOTE = {"cloud", "google"} | WEB  # engines that send text off this machine (need per-site permission)
MT = {"google", "tgemma"}  # sentence-level machine translation: one text box at a time, no prompt/glossary
NO_GLOSSARY_EXTRACTION = MT | WEB | {"plamo", "hymt"}  # translation-only models cannot extract a glossary


def get_engine(name: str):
    if name not in _engines:
        if name in WEB:
            from . import web_engines

            _engines[name] = {"deepl": web_engines.DeepLEngine, "azure": web_engines.AzureTranslatorEngine,
                              "google_free": web_engines.GoogleFreeEngine}[name]()
            return _engines[name]
        _engines[name] = {"local": LlamaCppEngine, "cloud": ClaudeEngine, "google": GoogleTranslateEngine,
                          "tgemma": TranslateGemmaEngine, "plamo": PlamoTranslateEngine,
                          "hymt": HyMtEngine}[name]()
    return _engines[name]


def engine_order(pref: str, allow_cloud: bool) -> tuple[list[str], str | None]:
    """Engines to try, plus a note when the user's choice could not be honoured."""
    if pref in REMOTE and not allow_cloud:
        return ["local"], "remote engine not allowed for this site; used the local LLM"
    order = [pref]
    if pref in REMOTE:
        order.append("hymt")  # a web engine failing (quota, 429, network): the fast local model first, if it runs
    if pref != "local":
        order.append("local")  # ... then the local LLM
    elif allow_cloud:
        order.append("cloud")
    return order, None


SCRIPT_RE = {
    "cjk": re.compile(r"[぀-ヿ一-鿿]"),
    "hangul": re.compile(r"[가-힯ᄀ-ᇿ]"),
    "thai": re.compile(r"[฀-๿]"),
    "cyrillic": re.compile(r"[Ѐ-ӿ]"),
    "arabic": re.compile(r"[؀-ۿ]"),
}


def _leak_re(src: str, target: str):
    """Regex for source-script characters that must not survive in the translation (None if not checkable)."""
    s = (src or "").lower()
    script = "cjk" if s in ("ja",) or s.startswith("zh") else {"ko": "hangul", "th": "thai", "ru": "cyrillic", "uk": "cyrillic", "ar": "arabic", "fa": "arabic"}.get(s)
    if script is None or (script == "thai" and target == "th"):
        return None
    return SCRIPT_RE[script]


NOVEL_BATCH_CHARS = int(os.environ.get("NOVEL_BATCH_CHARS", "140"))


def translate_blocks(
    blocks: list[dict], glossary: dict, context: list[dict], engine_pref: str = "local", allow_cloud: bool = False,
    novel: bool = False, src: str = "ja", target: str = "th", game: bool = False,
) -> tuple[dict[str, str], str, dict]:
    """A light-novel page is several times longer than a manga page, and the local LLM decodes slowly; its server
    runs several slots whose throughput adds up, so a long page goes out as a few batches of paragraphs at once."""
    total = sum(len(b["src"]) for b in blocks)
    if not (novel and engine_pref == "local" and total > NOVEL_BATCH_CHARS * 1.5):
        return _translate_blocks(blocks, glossary, context, engine_pref, allow_cloud, novel, src, target, game)
    from concurrent.futures import ThreadPoolExecutor

    batches, cur, n = [], [], 0
    for b in blocks:
        if cur and n + len(b["src"]) > NOVEL_BATCH_CHARS:
            batches.append(cur)
            cur, n = [], 0
        cur.append(b)
        n += len(b["src"])
    if cur:
        batches.append(cur)
    t0 = time.perf_counter()
    # the first batch goes alone; its translation then serves as context for the rest, which run at once, so the
    # narrator's pronouns and the characters' speech stay consistent across batches
    first = _translate_blocks(batches[0], glossary, context, engine_pref, allow_cloud, novel, src, target)
    ctx = context + [{"src": b["src"], "tl": first[0][b["id"]]} for b in batches[0] if first[0].get(b["id"])]
    with ThreadPoolExecutor(max(1, len(batches) - 1)) as pool:
        results = [first] + list(pool.map(
            lambda bs: _translate_blocks(bs, glossary, ctx, engine_pref, allow_cloud, novel, src, target), batches[1:],
        ))
    out, metrics = {}, {"missing": [], "jaLeaked": [], "errors": [], "batches": len(batches)}
    for tl, _, m in results:
        out.update(tl)
        for k in ("missing", "jaLeaked", "errors"):
            metrics[k] += m.get(k, [])
    metrics["ms"] = round((time.perf_counter() - t0) * 1000)
    names = [name for _, name, _ in results if name != "none"]
    return out, (max(set(names), key=names.count) if names else "none"), metrics


def _translate_blocks(
    blocks: list[dict], glossary: dict, context: list[dict], engine_pref: str = "local", allow_cloud: bool = False,
    novel: bool = False, src: str = "ja", target: str = "th", game: bool = False,
) -> tuple[dict[str, str], str, dict]:
    """Translate [{id, src, kind}] -> ({id: tl}, engine_name, metrics).

    Tries the preferred engine, retries once on a bad/incomplete answer, then falls back to cloud if allowed.
    """
    todo = [b for b in blocks if b["src"].strip() and not NOISE_RE.fullmatch(b["src"].strip())]
    if not todo:
        return {}, "none", {}
    if game:
        system = prompts.game_system(glossary, src, target)
    else:
        system = prompts.novel_system(glossary, src, target) if novel else prompts.manga_system(glossary, src, target)
    leak = _leak_re(src, target)
    order, note = engine_order(engine_pref, allow_cloud)
    errors = [note] if note else []
    for name in order:
        eng = get_engine(name)
        if hasattr(eng, "translate_page"):  # page-aware translation models (PLaMo, Hy-MT)
            t0 = time.perf_counter()
            try:
                tls = eng.translate_page([b["src"] for b in todo], src, target, context, glossary or {},
                                         style="game" if game else "manga")
            except EngineError as e:
                errors.append(f"{name}: {e}")
                continue
            out = drop_untranslated({b["id"]: clean_mt(t, target) for b, t in zip(todo, tls)}, todo)
            ms = round((time.perf_counter() - t0) * 1000)
            return out, name, {"ms": ms, "missing": [], "jaLeaked": [i for i, t in out.items() if leak and leak.search(t)], "errors": errors}
        if name in MT:
            t0 = time.perf_counter()
            try:
                tls = eng.translate([b["src"] for b in todo], src, target, glossary=glossary or {})
            except EngineError as e:
                errors.append(f"{name}: {e}")
                continue
            out = drop_untranslated({b["id"]: clean_mt(t, target) for b, t in zip(todo, tls)}, todo)
            ms = round((time.perf_counter() - t0) * 1000)
            return out, name, {"ms": ms, "missing": [], "jaLeaked": [i for i, t in out.items() if leak and leak.search(t)], "errors": errors}
        for attempt in range(2):
            ids = [b["id"] for b in todo]
            schema = prompts.translation_schema(ids, strict_counts=(name == "local"))
            t0 = time.perf_counter()
            try:
                data = eng.chat_json(system, prompts.translation_user(todo, context, novel, target, game), schema)
            except EngineError as e:
                errors.append(f"{name}#{attempt}: {e}")
                if "unreachable" in str(e):
                    break  # no point retrying a dead server
                continue
            out = {l["id"]: l["tl"].strip() for l in data.get("lines", []) if l.get("id") in ids}
            missing = [i for i in ids if i not in out]
            leaked = [i for i, tl in out.items() if leak and leak.search(tl)]
            if missing and attempt == 0:
                errors.append(f"{name}#{attempt}: missing ids {missing}")
                continue
            ms = round((time.perf_counter() - t0) * 1000)
            return out, name, {"ms": ms, "missing": missing, "jaLeaked": leaked, "errors": errors}
    raise EngineError("; ".join(errors) or "no engine available")


def extract_glossary(text: str, engine_pref: str = "local", allow_cloud: bool = False, src: str = "ja", target: str = "th") -> dict:
    order, _ = engine_order(engine_pref, allow_cloud)
    order = [n for n in order if n not in NO_GLOSSARY_EXTRACTION] or ["local"]  # needs a general LLM
    last = None
    for name in order:
        try:
            return get_engine(name).chat_json(prompts.glossary_system(src, target), text, prompts.glossary_schema, max_tokens=8192)
        except EngineError as e:
            last = e
    raise last or EngineError("no engine available")
