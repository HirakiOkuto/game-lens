"""game-lens server: endpoints and translation engines; no network, models or GPU (all stubbed or mocked).
Run (from game-lens/): .venv\\Scripts\\python.exe -m pytest server/tests -q
"""
import io
import json
import os
import sys
from pathlib import Path

import cv2
import httpx
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from translate import engines, prompts, web_engines  # noqa: E402

class FakeEngine:
    def __init__(self, answers):
        self.answers, self.calls = list(answers), 0

    def chat_json(self, system, user, schema, max_tokens=4096):
        self.calls += 1
        a = self.answers.pop(0)
        if isinstance(a, Exception):
            raise a
        return a


G = {"characters": [{"src": "ラフィー", "aliases": ["ラフィ"], "tl": "ลาฟีย์", "gender": "female", "role": ""}],
     "terms": [{"src": "指揮官", "tl": "ผู้บัญชาการ"}]}


def test_lock_and_unlock_names():
    terms = web_engines.terms_in(["ラフィー、指揮官が<来た>"], G)
    assert terms[0][0] == "ラフィー"  # longest first: the alias ラフィ must not split ラフィー
    x = web_engines.lock("ラフィー、指揮官が<来た>", terms, "xml")
    assert x == "<k>ลาฟีย์</k>、<k>ผู้บัญชาการ</k>が&lt;来た&gt;"
    assert web_engines.unlock("<k>ลาฟีย์</k> &lt;มา&gt;", "xml") == "ลาฟีย์ <มา>"
    h = web_engines.lock("ラフィ!", web_engines.terms_in(["ラフィ!"], G), "html")
    assert h == '<span translate="no" class="notranslate">ลาฟีย์</span>!'
    assert web_engines.unlock('<span translate="no" class="notranslate">ลาฟีย์</span> พูด&#39;', "html") == "ลาฟีย์ พูด'"
    assert web_engines.lock("ラフィー", terms, "inline") == "ลาฟีย์"
    assert web_engines.terms_in(["こんにちは"], G) == []


def _deepl(handler, monkeypatch, key="abc:fx"):
    monkeypatch.setenv("DEEPL_API_KEY", key)
    e = web_engines.DeepLEngine(transport=httpx.MockTransport(handler))
    monkeypatch.setitem(engines._engines, "deepl", e)
    return e


def test_deepl_engine_context_tags_and_free_host(monkeypatch):
    seen = []

    def handler(req: httpx.Request):
        seen.append((str(req.url), req.headers["authorization"], json.loads(req.content)))
        body = seen[-1][2]
        return httpx.Response(200, json={"translations": [{"text": f"T[{t}]"} for t in body["text"]]})

    _deepl(handler, monkeypatch)
    out, name, _ = engines.translate_blocks(
        [{"id": "a", "src": "ラフィー、おはよう", "kind": "dialog"}, {"id": "b", "src": "行こう", "kind": "dialog"}],
        G, [{"src": "前の台詞", "tl": "ก่อน"}], "deepl", allow_cloud=True, game=True,
    )
    url, auth, body = seen[0]
    assert url.startswith("https://api-free.deepl.com/v2/translate") and auth == "DeepL-Auth-Key abc:fx"
    assert body["source_lang"] == "JA" and body["target_lang"] == "TH" and body["context"] == "前の台詞"
    assert body["tag_handling"] == "xml" and body["ignore_tags"] == ["k"] and body["text"][0] == "<k>ลาฟีย์</k>、おはよう"
    assert name == "deepl" and out == {"a": "T[ลาฟีย์、おはよう]", "b": "T[行こう]"}


def test_deepl_retries_inline_when_tags_are_refused(monkeypatch):
    bodies = []

    def handler(req):
        b = json.loads(req.content)
        bodies.append(b)
        if "tag_handling" in b:
            return httpx.Response(400, json={"message": "tag handling not supported"})
        return httpx.Response(200, json={"translations": [{"text": t} for t in b["text"]]})

    e = _deepl(handler, monkeypatch, key="pro-key")
    assert e.base == "https://api.deepl.com"
    assert e.translate_page(["ラフィー"], "ja", "th", [], G) == ["ลาฟีย์"]
    assert len(bodies) == 2 and bodies[1]["text"] == ["ลาฟีย์"]


class DeadPageEngine:
    def translate_page(self, *a, **kw):
        raise engines.EngineError("Hy-MT unreachable")


class PageEngine:
    def translate_page(self, texts, *a, **kw):
        return [f"hy:{t}" for t in texts]


def test_web_engine_failure_falls_back_to_hymt_then_local(monkeypatch):
    _deepl(lambda req: httpx.Response(456, json={"message": "Quota exceeded"}), monkeypatch)
    local = FakeEngine([{"lines": [{"id": "a", "tl": "ก"}]}])
    monkeypatch.setitem(engines._engines, "local", local)
    monkeypatch.setitem(engines._engines, "hymt", DeadPageEngine())
    out, name, m = engines.translate_blocks([{"id": "a", "src": "あ"}], {}, [], "deepl", allow_cloud=True, game=True)
    assert name == "local" and out == {"a": "ก"} and "quota" in m["errors"][0] and "Hy-MT" in m["errors"][1]
    monkeypatch.setitem(engines._engines, "hymt", PageEngine())
    out, name, _ = engines.translate_blocks([{"id": "a", "src": "あ"}], {}, [], "deepl", allow_cloud=True, game=True)
    assert name == "hymt" and out == {"a": "hy:あ"}
    assert engines.engine_order("hymt", False)[0] == ["hymt", "local"]
    assert engines.engine_order("google_free", True)[0] == ["google_free", "hymt", "local"]


def test_deepl_usage(monkeypatch):
    e = _deepl(lambda req: httpx.Response(200, json={"character_count": 10, "character_limit": 500000}), monkeypatch)
    assert e.usage() == {"used": 10, "limit": 500000}


def test_google_official_locks_names_in_html_mode(monkeypatch):
    seen = {}

    def handler(req):
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, json={"data": {"translations": [{"translatedText": q.replace("おはよう", "อรุณสวัสดิ์")} for q in seen["body"]["q"]]}})

    monkeypatch.setenv("GOOGLE_TRANSLATE_API_KEY", "k")
    g = engines.GoogleTranslateEngine(transport=httpx.MockTransport(handler))
    assert g.translate(["ラフィー、おはよう"], "ja", "th", G) == ["ลาฟีย์、อรุณสวัสดิ์"]
    assert seen["body"]["format"] == "html" and 'translate="no"' in seen["body"]["q"][0]


def test_google_free_engine(monkeypatch):
    seen = []

    def handler(req):
        seen.append(dict(req.url.params))
        return httpx.Response(200, json=[[["สวัสดี ", req.url.params["q"], None], ["ครับ", "x", None]], None, "ja"])

    e = web_engines.GoogleFreeEngine(transport=httpx.MockTransport(handler))
    assert e.translate_page(["ラフィー"], "ja", "th", [], G) == ["สวัสดี ครับ"]
    assert seen[0]["client"] == "gtx" and seen[0]["sl"] == "ja" and seen[0]["tl"] == "th" and seen[0]["q"] == "ลาฟีย์"
    e429 = web_engines.GoogleFreeEngine(transport=httpx.MockTransport(lambda r: httpx.Response(429)))
    with pytest.raises(engines.EngineError, match="rate limited"):
        e429.translate_page(["あ"], "ja", "th", [], {})


def test_azure_translator_engine(monkeypatch):
    seen = {}

    def handler(req):
        seen.update(key=req.headers.get("ocp-apim-subscription-key"), region=req.headers.get("ocp-apim-subscription-region"),
                    params=dict(req.url.params), body=json.loads(req.content))
        return httpx.Response(200, json=[{"translations": [{"text": i["Text"] + " ไทย", "to": "th"}]} for i in seen["body"]])

    monkeypatch.setenv("AZURE_TRANSLATOR_KEY", "az")
    monkeypatch.setenv("AZURE_TRANSLATOR_REGION", "southeastasia")
    e = web_engines.AzureTranslatorEngine(transport=httpx.MockTransport(handler))
    assert e.translate_page(["ラフィー"], "ja", "th", [], G) == ["ลาฟีย์ ไทย"]
    assert seen["key"] == "az" and seen["region"] == "southeastasia"
    assert seen["params"]["textType"] == "html" and seen["params"]["to"] == "th" and seen["params"]["from"] == "ja"
    assert 'class="notranslate"' in seen["body"][0]["Text"]
    bad = web_engines.AzureTranslatorEngine(transport=httpx.MockTransport(lambda r: httpx.Response(401, json={"error": {"message": "denied"}})))
    with pytest.raises(engines.EngineError, match="bad key"):
        bad.translate_page(["あ"], "ja", "th", [], {})


def test_game_prompts():
    s = prompts.game_system(G, "ja", "th")
    assert "[dialog]" in s and "ラフィー" in s and "ลาฟีย์" in s
    u = prompts.translation_user([{"id": "a", "src": "行こう"}], [{"src": "前", "tl": "ก่อน"}], target="th", game=True)
    assert "บทพูดก่อนหน้า" in u and "a\t[dialog]\t行こう" in u
    assert "video game" in engines.HyMtEngine()._prompt("x", [], [], "th", "game")


def test_read_lines_drops_ruby_and_orders_rows():
    from ocr import _drop_ruby, _rows

    base = {"bbox": [100, 50, 300, 30], "text": "指揮官"}
    ruby = {"bbox": [110, 36, 60, 12], "text": "しきかん"}
    second = {"bbox": [100, 100, 200, 30], "text": "二行目"}
    right = {"bbox": [420, 52, 80, 28], "text": "右"}
    assert _drop_ruby([base, ruby, second]) == [base, second]
    assert [l["text"] for l in _rows([second, right, base])] == ["指揮官", "右", "二行目"]


def test_line_colors_dark_box_white_text():
    from ocr import line_colors

    img = np.full((40, 200, 3), (20, 30, 60), np.uint8)
    cv2.putText(img, "TEXT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (250, 250, 250), 2)
    c = line_colors(img, [0, 0, 200, 40])
    assert c["bg"] == "#141e3c" and int(c["fg"][1:3], 16) > 200


def _client(monkeypatch, tmp_path):
    import app as app_mod
    from fastapi.testclient import TestClient
    from store import Store

    monkeypatch.setattr(app_mod, "store", Store(tmp_path / "game.sqlite"))
    return app_mod, TestClient(app_mod.app), {"X-Lens-Token": app_mod.TOKEN}


def test_ocr_lines_endpoint(monkeypatch, tmp_path):
    app_mod, client, h = _client(monkeypatch, tmp_path)
    monkeypatch.setattr(app_mod, "read_lines", lambda img, src, ms: [
        {"bbox": [5, 5, 50, 20], "text": "こんにちは", "conf": 0.9}, {"bbox": [5, 30, 50, 20], "text": "指揮官", "conf": 0.8}])
    ok, png = cv2.imencode(".png", np.zeros((60, 80, 3), np.uint8))
    files = {"image": ("r.png", io.BytesIO(png.tobytes()), "image/png")}
    assert client.post("/ocr/lines", files=files).status_code == 401
    files = {"image": ("r.png", io.BytesIO(png.tobytes()), "image/png")}
    d = client.post("/ocr/lines", files=files, headers=h).json()
    assert d["text"] == "こんにちは指揮官" and d["lines"][1]["id"] == "l1" and "fg" in d["lines"][0] and d["w"] == 80


def test_translate_lines_cache_glossary_and_skips(monkeypatch, tmp_path):


    app_mod, client, h = _client(monkeypatch, tmp_path)
    fake = FakeEngine([{"lines": [{"id": "n", "tl": "ลาฟีย์"}, {"id": "d", "tl": "อรุณสวัสดิ์ ผู้บัญชาการ"}]},
                       {"lines": [{"id": "d", "tl": "ใหม่"}]}])
    seen = []
    orig = fake.chat_json
    fake.chat_json = lambda system, user, schema, max_tokens=4096: (seen.append((system, user)), orig(system, user, schema))[1]
    monkeypatch.setitem(engines._engines, "local", fake)
    client.put("/glossary/azur-promilia?target=th", json=G, headers=h)
    req = {"series": "azur-promilia", "engine": "local",
           "lines": [{"id": "n", "src": "ラフィー", "kind": "name"}, {"id": "d", "src": "おはよう、指揮官", "kind": "dialog"},
                     {"id": "x", "src": "OK", "kind": "ui"}, {"id": "e", "src": "  ", "kind": "ui"}],
           "context": [{"src": "前", "tl": "ก่อน"}]}
    d = client.post("/translate/lines", json=req, headers=h).json()
    assert d["lines"] == {"n": "ลาฟีย์", "d": "อรุณสวัสดิ์ ผู้บัญชาการ", "x": "", "e": ""}
    assert d["engine"] == "local" and d["cached"] == []
    system, user = seen[0]
    assert "นักแปลเกม" in system and "ลาฟีย์" in system and "n\t[name]\tラフィー" in user and "บทพูดก่อนหน้า" in user
    # same lines again: all from cache, no engine call
    d2 = client.post("/translate/lines", json=req, headers=h).json()
    assert d2["lines"] == d["lines"] and sorted(d2["cached"]) == ["d", "n"] and fake.calls == 1
    # the kind is part of the key: the same text as a UI label is translated separately
    r3 = client.post("/translate/lines", json={**req, "lines": [{"id": "d", "src": "おはよう、指揮官", "kind": "ui"}]}, headers=h)
    assert r3.json()["lines"] == {"d": "ใหม่"} and fake.calls == 2
    assert client.post("/translate/lines", json={**req, "engine": "nope"}, headers=h).status_code == 400
    assert client.post("/translate/lines", json={**req, "target": "fr"}, headers=h).status_code == 400


def test_translate_lines_remote_needs_permission(monkeypatch, tmp_path):


    app_mod, client, h = _client(monkeypatch, tmp_path)
    monkeypatch.setitem(engines._engines, "local", FakeEngine([{"lines": [{"id": "a", "tl": "ก"}]}]))
    calls = []
    monkeypatch.setitem(engines._engines, "google_free", web_engines.GoogleFreeEngine(
        transport=httpx.MockTransport(lambda r: (calls.append(1), httpx.Response(200, json=[[["ข", "x"]]]))[1])))
    req = {"engine": "google_free", "allow_cloud": False, "lines": [{"id": "a", "src": "あ"}]}
    d = client.post("/translate/lines", json=req, headers=h).json()
    assert d["engine"] == "local" and calls == []
    # allowed now: the cached local fallback is replaced by the requested engine
    d = client.post("/translate/lines", json={**req, "allow_cloud": True}, headers=h).json()
    assert d["engine"] == "google_free" and d["lines"] == {"a": "ข"} and calls == [1]


def test_usage_and_health_list_new_engines(monkeypatch, tmp_path):
    app_mod, client, h = _client(monkeypatch, tmp_path)
    monkeypatch.delenv("DEEPL_API_KEY", raising=False)
    monkeypatch.setitem(engines._engines, "deepl", web_engines.DeepLEngine())
    assert client.get("/usage", headers=h).json() == {"deepl": None}
    for n in ("local", "cloud", "tgemma", "plamo", "hymt"):
        monkeypatch.setitem(engines._engines, n, type("X", (), {"available": lambda self: False})())
    monkeypatch.delenv("AZURE_TRANSLATOR_KEY", raising=False)
    monkeypatch.setitem(engines._engines, "azure", web_engines.AzureTranslatorEngine())
    d = client.get("/health").json()
    assert d["deepl"] is False and d["azure"] is False and d["google_free"] is True


def test_config_keys_roundtrip_and_engine_reset(monkeypatch, tmp_path):
    app_mod, client, h = _client(monkeypatch, tmp_path)
    monkeypatch.setattr(app_mod, "KEYS_FILE", tmp_path / "keys.json")
    monkeypatch.delenv("DEEPL_API_KEY", raising=False)
    monkeypatch.setitem(engines._engines, "deepl", web_engines.DeepLEngine())
    assert client.get("/config/keys", headers=h).json()["deepl"]["DEEPL_API_KEY"] == ""
    d = client.put("/config/keys", json={"DEEPL_API_KEY": "abcdefgh1234:fx"}, headers=h).json()
    assert d["deepl"]["DEEPL_API_KEY"] == "••••4:fx" and "abcdefgh" not in str(d)
    assert "deepl" not in engines._engines  # re-created with the new key on next use
    assert engines.get_engine("deepl").key == "abcdefgh1234:fx"
    assert json.loads((tmp_path / "keys.json").read_text())["DEEPL_API_KEY"] == "abcdefgh1234:fx"
    assert client.put("/config/keys", json={"PATH": "x"}, headers=h).status_code == 400
    client.put("/config/keys", json={"DEEPL_API_KEY": ""}, headers=h)
    assert "DEEPL_API_KEY" not in os.environ
    monkeypatch.setitem(engines._engines, "deepl", web_engines.DeepLEngine())


def test_engine_test_endpoint_reports_fallback_as_failure(monkeypatch, tmp_path):
    app_mod, client, h = _client(monkeypatch, tmp_path)
    monkeypatch.setitem(engines._engines, "google_free", web_engines.GoogleFreeEngine(
        transport=httpx.MockTransport(lambda r: httpx.Response(429))))
    monkeypatch.setitem(engines._engines, "hymt", PageEngine())
    d = client.post("/config/test/google_free", headers=h).json()
    assert d["ok"] is False and "429" in d["error"]
    monkeypatch.setitem(engines._engines, "google_free", web_engines.GoogleFreeEngine(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json=[[["สวัสดี", "x"]]]))))
    d = client.post("/config/test/google_free", headers=h).json()
    assert d["ok"] is True and d["text"] == "สวัสดี"


def test_tighten_shrinks_padded_box_to_glyphs():
    from ocr import _tighten

    img = np.full((60, 200, 3), (70, 140, 100), np.uint8)
    img[20:42, 30:120] = (255, 212, 121)  # "text" 22 px high inside a 60 px box
    x, y, w, h = _tighten(img, [0, 0, 200, 60])
    assert 18 <= y <= 20 and 22 <= h <= 25 and 28 <= x <= 30 and 90 <= w <= 93
    assert all(type(v) is int for v in (x, y, w, h))  # JSON-serialisable
    assert _tighten(img, [0, 0, 3, 3]) == [0, 0, 3, 3]


def test_read_lines_real_pipeline_returns_json_ready_boxes(monkeypatch):
    """read_lines end to end with a stub PP-OCR engine (no models): boxes are plain ints, tightened, ordered."""
    import ocr

    class Res:
        def __init__(self, boxes, txts, scores):
            self.boxes, self.txts, self.scores = boxes, txts, scores

    def engine(img, use_det=True, use_cls=False, use_rec=True):
        # one detection around the text, padded like PP-OCR does (coordinates in the padded, scaled image)
        return Res([np.array([[40, 30], [300, 30], [300, 110], [40, 110]], float)], ["ラフィー"], [0.95])

    monkeypatch.setattr(ocr, "_get_rapid", lambda v, l: engine)
    img = np.full((43, 384, 3), (70, 140, 100), np.uint8)
    img[10:32, 20:100] = (255, 212, 121)
    lines = ocr.read_lines(img, "ja")
    assert lines and json.dumps(lines)
    x, y, w, h = lines[0]["bbox"]
    assert h <= 26 and y >= 8
