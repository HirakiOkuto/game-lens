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
    monkeypatch.setitem(engines._engines, "index", DeadPageEngine())
    out, name, m = engines.translate_blocks([{"id": "a", "src": "あ"}], {}, [], "deepl", allow_cloud=True, game=True)
    assert name == "local" and out == {"a": "ก"} and "quota" in m["errors"][0] and len(m["errors"]) == 3
    monkeypatch.setitem(engines._engines, "hymt", PageEngine())
    out, name, _ = engines.translate_blocks([{"id": "a", "src": "あ"}], {}, [], "deepl", allow_cloud=True, game=True)
    assert name == "hymt" and out == {"a": "hy:あ"}
    assert engines.engine_order("hymt", False)[0] == ["hymt", "local"]
    assert engines.engine_order("google_free", True)[0] == ["google_free", "index", "hymt", "local"]


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
    sent, seen = [], []

    def chrome(req):  # translate-pa: every line in one request, names in translate="no" spans
        if req.url.path.endswith("translateHtml"):
            body = json.loads(req.content)
            sent.append((body, req.headers.get("x-goog-api-key")))
            return httpx.Response(200, json=[[t.replace("ラフィー", "") + " สวัสดี &amp; ok" for t in body[0][0]]])
        return httpx.Response(500)

    e = web_engines.GoogleFreeEngine(transport=httpx.MockTransport(chrome))
    out = e.translate_page(["ラフィー", "あ"], "ja", "th", [], G)
    assert out == ["ลาฟีย์ สวัสดี & ok", "あ สวัสดี & ok"] and len(sent) == 1  # one request for the page
    body, key = sent[0]
    assert body[0][1:] == ["ja", "th"] and body[1] == "wt_lib" and key == web_engines.GoogleFreeEngine.KEY
    assert 'translate="no"' in body[0][0][0] and "ลาฟีย์" in body[0][0][0]

    def old(req):  # refused there: the older gtx endpoint, line by line
        if req.url.path.endswith("translateHtml"):
            return httpx.Response(403)
        seen.append(dict(req.url.params))
        return httpx.Response(200, json=[[["สวัสดี ", req.url.params["q"], None], ["ครับ", "x", None]], None, "ja"])

    e = web_engines.GoogleFreeEngine(transport=httpx.MockTransport(old))
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
    assert d["text"] == "こんにちは指揮官" and d["lines"][1]["id"] == "l1" and d["w"] == 80


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
           "lines": [{"id": "n", "src": "村長", "kind": "name"}, {"id": "d", "src": "おはよう、指揮官", "kind": "dialog"},
                     {"id": "x", "src": "OK", "kind": "ui"}, {"id": "e", "src": "  ", "kind": "ui"}],
           "context": [{"src": "前", "tl": "ก่อน"}]}
    d = client.post("/translate/lines", json=req, headers=h).json()
    assert d["lines"] == {"n": "ลาฟีย์", "d": "อรุณสวัสดิ์ ผู้บัญชาการ", "x": "", "e": ""}
    assert d["engine"] == "local" and d["cached"] == []
    system, user = seen[0]
    assert "นักแปลเกม" in system and "ลาฟีย์" in system and "n\t[name]\t村長" in user and "บทพูดก่อนหน้า" in user
    # same lines again: all from cache, no engine call
    d2 = client.post("/translate/lines", json=req, headers=h).json()
    assert d2["lines"] == d["lines"] and sorted(d2["cached"]) == ["d", "n"] and fake.calls == 1
    # the kind is part of the key: the same text as a UI label is translated separately
    r3 = client.post("/translate/lines", json={**req, "lines": [{"id": "d", "src": "おはよう、指揮官", "kind": "ui"}]}, headers=h)
    assert r3.json()["lines"] == {"d": "ใหม่"} and fake.calls == 2
    assert client.post("/translate/lines", json={**req, "engine": "nope"}, headers=h).status_code == 400
    assert client.post("/translate/lines", json={**req, "target": "fr"}, headers=h).status_code == 400


def test_menu_text_differing_only_in_numbers_is_not_translated_again(monkeypatch, tmp_path):
    app_mod, client, h = _client(monkeypatch, tmp_path)
    fake = FakeEngine([{"lines": [{"id": "a", "tl": "พลังโจมตีเพิ่มขึ้น 4% และ HP เพิ่มขึ้น 1%"}, {"id": "b", "tl": "เลเวล 3"},
                                  {"id": "c", "tl": "สามชิ้น"}, {"id": "d", "tl": "อรุณสวัสดิ์ 5"}]},
                       {"lines": [{"id": "c", "tl": "เจ็ดชิ้น"}, {"id": "d", "tl": "อรุณสวัสดิ์ 6"}, {"id": "e", "tl": "1/1"}]}])
    monkeypatch.setitem(engines._engines, "local", fake)
    ui = lambda i, s: {"id": i, "src": s, "kind": "ui"}  # noqa: E731
    req = {"engine": "local", "lines": [ui("a", "攻撃力が4%上昇し、HPが1%増加する。"), ui("b", "Lv.3"), ui("c", "3個"),
                                        {"id": "d", "src": "おはよう5", "kind": "dialog"}]}
    client.post("/translate/lines", json=req, headers=h)
    req2 = {"engine": "local", "lines": [ui("a", "攻撃力が7.1%上昇し、HPが2%増加する。"), ui("b", "Lv.12"), ui("c", "7個"),
                                         {"id": "d", "src": "おはよう6", "kind": "dialog"}, ui("e", "1/1")]}
    d = client.post("/translate/lines", json=req2, headers=h).json()
    assert d["lines"]["a"] == "พลังโจมตีเพิ่มขึ้น 7.1% และ HP เพิ่มขึ้น 2%" and d["lines"]["b"] == "เลเวล 12"
    assert sorted(d["cached"]) == ["a", "b"]  # the numbers put into the earlier translations
    assert d["lines"]["c"] == "เจ็ดชิ้น"  # its number was written out as a word: translated itself
    assert d["lines"]["d"] == "อรุณสวัสดิ์ 6"  # dialogue is always translated itself
    assert app_mod.make_pattern(["1", "1"], "1/1") is None  # the same number twice: which is which?
    assert app_mod.fill_pattern(app_mod.make_pattern(["5"], "5 items"), ["1"], "en") is None  # "1 items"
    assert app_mod.fill_pattern(app_mod.make_pattern(["5"], "5 items"), ["8"], "en") == "8 items"


def test_translate_lines_remote_needs_permission(monkeypatch, tmp_path):


    app_mod, client, h = _client(monkeypatch, tmp_path)
    monkeypatch.setitem(engines._engines, "local", FakeEngine([{"lines": [{"id": "a", "tl": "ก"}]}]))
    calls = []
    monkeypatch.setitem(engines._engines, "google_free", web_engines.GoogleFreeEngine(
        transport=httpx.MockTransport(lambda r: (calls.append(1), httpx.Response(200, json=[["ข"]]))[1])))
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
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json=[["สวัสดี"]]))))
    d = client.post("/config/test/google_free", headers=h).json()
    assert d["ok"] is True and d["text"] == "สวัสดี"


def test_tighten_shrinks_padded_box_to_glyphs():
    from ocr import _tighten

    img = np.full((60, 200, 3), (70, 140, 100), np.uint8)
    img[20:42, 30:120] = (255, 212, 121)  # "text" 22 px high inside a 60 px box
    x, y, w, h = _tighten(img, [24, 0, 102, 60])  # the detector's padding: a few pixels each side
    assert 18 <= y <= 20 and 22 <= h <= 25 and 28 <= x <= 30 and 90 <= w <= 93
    # much more cut off than any padding (one threshold lost part of the line over a busy picture): width kept
    x, y, w, h = _tighten(img, [0, 0, 200, 60])
    assert x == 0 and w == 200 and 22 <= h <= 25
    assert all(type(v) is int for v in (x, y, w, h))  # JSON-serialisable
    assert _tighten(img, [0, 0, 3, 3]) == [0, 0, 3, 3]
    # a selected tab: its first letters dimmed by a glow fall to the other side of the threshold. The box is too
    # narrow for the four letters read in it, so it keeps the detector's width (the translation covers them all)
    tab = np.full((24, 70, 3), 30, np.uint8)
    tab[:, :35] = 150  # the glow behind the first two
    for i, v in enumerate((200, 200, 250, 250)):
        tab[4:21, 3 + i * 16 : 16 + i * 16] = v
    cut = _tighten(tab, [0, 0, 70, 24])
    assert cut[0] >= 30  # the threshold alone keeps only the bright half
    x, y, w, h = _tighten(tab, [0, 0, 70, 24], "アイテム")
    assert x == 0 and w == 70
    assert _tighten(tab, [0, 0, 70, 24], "OK") == cut  # two Latin letters do fit there


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


def test_icon_read_as_the_first_letter_is_dropped():
    """A help button: a solid icon (a book with a ? on it, read as 2) apart from ヘルプ. A digit is thin strokes."""
    from ocr import _drop_icon

    img = np.full((40, 160, 3), 60, np.uint8)
    img[5:35, 5:31] = 235  # the icon: a light block as high as the letters
    img[12:28, 15:21] = 60  # (its ? cut into it)
    for x0 in (50, 80, 110):  # three letters of thin strokes
        img[5:35, x0 : x0 + 4] = 235
        img[18:22, x0 : x0 + 20] = 235
    line = {"bbox": [3, 3, 130, 34], "dbox": [0, 0, 136, 40], "text": "2ヘルプ", "conf": 0.99}
    got = _drop_icon(img, line)
    assert got["text"] == "ヘルプ" and 45 <= got["bbox"][0] <= 50 and got["bbox"][0] + got["bbox"][2] == 133
    assert got["dbox"][0] + got["dbox"][2] == 136 and json.dumps(got)
    # a thin digit before the letters is text (1キボ), and so is a line that does not start like that
    img[:, :40] = 60  # the icon gone, a "1" in its place
    img[5:35, 10:14] = 235
    assert _drop_icon(img, dict(line, text="1キボ"))["text"] == "1キボ"
    assert _drop_icon(img, dict(line, text="01キボ小屋"))["text"] == "01キボ小屋"


def test_faint_lines_reads_only_new_boxes_in_the_languages_letters(monkeypatch):
    """A whole screen looked at again with its contrast raised: boxes where no line is are read from the picture
    itself; a box over a line already read, and icons read as stray marks, are not taken."""
    import ocr

    class Res:
        def __init__(self, boxes=None, txts=None, scores=None):
            self.boxes, self.txts, self.scores = boxes, txts, scores

    box = lambda x, y, w, h: np.array([[x + 32, y + 32], [x + w + 32, y + 32], [x + w + 32, y + h + 32], [x + 32, y + h + 32]], float)
    reads = {44: ("アニマ", 0.99), 30: ("十", 0.95), 60: ("アイテム", 0.6)}  # by the width of the crop read

    def engine(img, use_det=True, use_cls=False, use_rec=True):
        if use_det:  # the boosted screen: a known line, a dim tab label, an icon, a doubtful reading
            return Res([box(155, 269, 58, 18), box(162, 573, 44, 15), box(300, 100, 30, 15), box(300, 200, 60, 15)])
        t, s = reads[img.shape[1]]
        return Res(txts=[t], scores=[s])

    monkeypatch.setattr(ocr, "_get_rapid", lambda v, l: engine)
    img = np.full((1080, 1920, 3), 90, np.uint8)
    known = [{"bbox": [155, 269, 58, 18], "dbox": [152, 266, 64, 24], "text": "アイテム", "conf": 1.0}]
    got = ocr.faint_lines(img, "ja", known)
    assert [l["text"] for l in got] == ["アニマ"] and got[0]["dbox"] == [162, 573, 44, 15]
    assert json.dumps(got)


def test_read_screen_looks_closer_only_at_a_busy_scene_with_little_text(monkeypatch):
    """The slow look for small letters on the moving world (photo mode) runs for a busy screen where little was found,
    not for a menu, nor for the quick look every few seconds; a box that lost the end of its line is widened."""
    import ocr

    calls = []
    monkeypatch.setattr(ocr, "read_lines", lambda img, lang, ms=0.5: [])
    monkeypatch.setattr(ocr, "faint_lines", lambda img, lang, lines: [])
    monkeypatch.setattr(ocr, "scene_lines", lambda img, lang, lines: calls.append(1) or [
        {"bbox": [20, 735, 96, 25], "dbox": [20, 735, 96, 25], "text": "テンプレート", "conf": 0.95}])
    rng = np.random.default_rng(1)
    grass = rng.integers(0, 255, (1080, 1920, 3), dtype=np.uint8)  # detail everywhere
    menu = np.full((1080, 1920, 3), 60, np.uint8)
    assert ocr.busy(grass) >= ocr.SCENE_BUSY > ocr.busy(menu)
    assert [l["text"] for l in ocr.read_screen(grass, "ja")] == ["テンプレート"] and calls == [1]
    assert ocr.read_screen(grass, "ja", deep=False) == [] and ocr.read_screen(menu, "ja") == [] and calls == [1]

    # widening: テンプレ (0.83) grows into テンプレー, テンプレート; one more step reads past the end and is less sure
    class Res:
        def __init__(self, t, s):
            self.txts, self.scores = [t], [s]

    reads = {61: ("テンプレ", 0.83), 81: ("テンプレー", 0.92), 101: ("テンプレート", 0.95), 121: ("テンプレートラ", 0.86)}
    engine = lambda img, use_det=False, use_cls=False, use_rec=True: Res(*reads[round(img.shape[1] / 2)])  # noqa: E731
    img = np.full((1080, 1920, 3), 90, np.uint8)
    got = ocr._read_new_boxes(img, "ja", engine, [], [(15, 735, 76, 760)], 8, enlarge=2.0, lone=True, sure=0.8, grow=True)
    assert [(l["text"], l["dbox"][2]) for l in got] == [("テンプレート", 101)]


def test_translate_lines_answers_a_bare_glossary_name_without_an_engine(monkeypatch, tmp_path):
    app_mod, client, h = _client(monkeypatch, tmp_path)
    fake = FakeEngine([{"lines": [{"id": "d", "tl": "Laffey ง่วงแล้ว"}]}])
    monkeypatch.setitem(engines._engines, "local", fake)
    g = {"characters": [{"src": "ラフィー", "aliases": ["ラフィ"], "tl": "Laffey"}], "terms": []}
    client.put("/glossary/game?target=th", json=g, headers=h)
    req = {"series": "game", "engine": "local", "lines": [{"id": "n", "src": "ラフィー", "kind": "name"},
                                                           {"id": "a", "src": " ラフィ ", "kind": "name"}]}
    d = client.post("/translate/lines", json=req, headers=h).json()
    assert d["lines"] == {"n": "Laffey", "a": "Laffey"} and d["engine"] == "glossary" and fake.calls == 0
    d = client.post("/translate/lines", json={**req, "lines": [{"id": "d", "src": "ラフィー、眠い", "kind": "dialog"}]}, headers=h).json()
    assert d["lines"] == {"d": "Laffey ง่วงแล้ว"} and fake.calls == 1  # a sentence still goes to the engine


def test_an_unknown_speaker_is_learned_in_latin_letters_once(monkeypatch, tmp_path):
    """A speaker the glossary does not have came out in Thai, differently line to line (シフォン: ซิฟอน / ชีฟอง), or
    translated (ジャベリン: หอกยาว). Now the engine is asked once for it in English, the name is kept, and both name
    boxes and dialogue use it from then on; a reading that is no name (村民たち, a sentence) is translated as before."""
    app_mod, client, h = _client(monkeypatch, tmp_path)
    calls = []

    def fake_blocks(blocks, glossary, ctx, engine, allow_cloud, novel, src, target, game):
        calls.append((target, [x["src"] for x in blocks], [c["tl"] for c in glossary.get("characters", [])]))
        if target == "en":
            return {x["id"]: {"シフォン": "Chiffon.", "カーク": "the spear"}.get(x["src"], "") for x in blocks}, "local", {}
        return {x["id"]: "ชาวบ้าน" if x["src"] == "村民たち" else "พูด" for x in blocks}, "local", {}

    monkeypatch.setattr(app_mod, "translate_blocks", fake_blocks)
    client.put("/glossary/game?target=th", json={"characters": [{"src": "テララ", "tl": "Terara"}], "terms": []}, headers=h)
    req = {"series": "game", "engine": "local", "lines": [{"id": "a", "src": "シフォン", "kind": "name"},
                                                           {"id": "b", "src": "カーク", "kind": "name"},
                                                           {"id": "c", "src": "村民たち", "kind": "name"},
                                                           {"id": "d", "src": "テララ", "kind": "name"}]}
    d = client.post("/translate/lines", json=req, headers=h).json()["lines"]
    assert d == {"a": "Chiffon", "b": "พูด", "c": "ชาวบ้าน", "d": "Terara"}  # (no name in "the spear": Thai as before)
    assert calls[0][:2] == ("en", ["シフォン", "カーク"])
    calls.clear()
    d = client.post("/translate/lines", json={**req, "lines": [{"id": "a", "src": "シフオン", "kind": "name"}]}, headers=h).json()
    assert d["lines"] == {"a": "Chiffon"} and d["engine"] == "glossary" and calls == []  # (misread small kana: still it)
    client.post("/translate/lines", json={**req, "lines": [{"id": "x", "src": "シフォン、こっち！", "kind": "dialog"}]}, headers=h)
    assert calls and calls[-1][0] == "th" and "Chiffon" in calls[-1][2]  # dialogue gets the name too


def test_terms_in_text_prefers_the_longer_term():
    terms = [("シャル", "Shalle"), ("シャルル村", "หมู่บ้าน Charlulu"), ("テララ", "Terara")]
    assert engines.terms_in_text("シャルル村へ行こう、テララ", terms) == [("シャルル村", "หมู่บ้าน Charlulu"), ("テララ", "Terara")]
    assert engines.terms_in_text("シャル、シャルル村だよ", terms) == [("シャルル村", "หมู่บ้าน Charlulu"), ("シャル", "Shalle")]
    assert engines.terms_in_text("こんにちは", terms) == []


def test_glossary_names_survive_ocr_misreadings():
    terms = [("ミィティ", "Mitty"), ("メーツァ", "Metsa"), ("ファランタ", "Faridah"), ("テララ", "Terara"), ("ノノ", "Nono")]
    for seen, want in (("ミイテイ", "Mitty"), ("ミイティ", "Mitty"), ("メーッア", "Metsa"), ("フアランタ", "Faridah"),
                       ("クアランタ", "Faridah"), ("デララ", "Terara"), ("テララ", "Terara"), ("ノノ", "Nono")):
        assert engines.match_name(seen, terms) == want, seen
    for other in ("三行", "ララ", "Alehero", "ミィティの父", ""):
        assert engines.match_name(other, terms) is None, other
    # inside a sentence: small kana are forgiven, and the engine is told the name the way the text has it
    assert engines.terms_in_text("ミイティの父が来た", terms) == [("ミイティ", "Mitty")]
    assert web_engines.lock("ミイティ、行こう", web_engines.terms_in(["ミイティ、行こう"], {"characters": [{"src": "ミィティ", "tl": "Mitty"}]}), "inline") == "Mitty、行こう"


def test_hymt_retries_with_stand_ins_when_a_name_is_transliterated(monkeypatch):
    sent = []

    class Resp:
        status_code = 200

        def __init__(self, c):
            self._c = c

        def json(self):
            return {"choices": [{"message": {"content": self._c}}]}

    def fake_post(url, json, timeout):
        p = json["messages"][0]["content"]
        sent.append(p)
        src = p.split("[Source Text]\n")[1]
        return Resp({"ミイティの父": "พ่อของมิตตี้", "N1の父": "พ่อของ N1", "ノノ": "Nono จ้ะ", "行こう": "ไปกัน"}[src])

    monkeypatch.setattr(engines, "http_post", fake_post)
    g = {"characters": [{"src": "ミィティ", "tl": "Mitty"}, {"src": "ノノ", "tl": "Nono"}], "terms": []}
    e = engines.HyMtEngine()
    assert e.translate_page(["ミイティの父"], "ja", "th", [], g, style="game") == ["พ่อของ Mitty"] and len(sent) == 2
    assert "[Reference Translations]\nミイティ translates to Mitty" in sent[0] and "Reference" not in sent[1]
    sent.clear()
    assert e.translate_page(["ノノ", "行こう"], "ja", "th", [], g, style="game") == ["Nono จ้ะ", "ไปกัน"] and len(sent) == 2
    assert "ไปกัน" not in sent[0] and "Nono จ้ะ" in sent[1]  # dialogue: the line before is context
    sent.clear()
    ctx = [{"src": "おはよう", "tl": "อรุณสวัสดิ์"}]
    out = e.translate_page(["行こう", "ノノ", "行こう"], "ja", "th", ctx, g, style="game", kinds=["ui", "ui", "dialog"])
    assert out == ["ไปกัน", "Nono จ้ะ", "ไปกัน"]  # menu labels several at a time, still in their places
    labels = [p for p in sent if "อรุณสวัสดิ์" not in p]  # (the dialogue line has the context)
    assert len(labels) == 2 and not any("Nono จ้ะ" in p or "ไปกัน" in p for p in labels)  # labels: each on its own
    # menu text is not spoken: faithful and concise, not in a character's voice (that added to descriptions)
    assert all("faithfully" in p and "spoken" not in p for p in labels)
    assert all("spoken" in p and "faithfully" not in p for p in sent if p not in labels)
    idx = engines.IndexTranslateEngine()
    assert "忠实于原文" in idx._prompt("証。", [], [], "th", "game-ui") and "口语化" not in idx._prompt("証。", [], [], "th", "game-ui")
    assert "口语化" in idx._prompt("行こう", [], [], "th", "game")
    # a stand-in that gets lost: keep the first answer rather than a broken one
    monkeypatch.setattr(engines.HyMtEngine, "_ask", lambda self, p: "พ่อของมิตตี้")
    assert e.translate_page(["ミイティの父"], "ja", "th", [], g, style="game") == ["พ่อของมิตตี้"]


def test_overlapping_ocr_boxes_are_read_once():
    from ocr import _drop_repeats

    whole = {"bbox": [100, 50, 600, 30], "text": "とても立派な方だったんだね。", "conf": 0.98}
    part = {"bbox": [300, 52, 390, 27], "text": "も立派な方たったんだね", "conf": 0.9}
    below = {"bbox": [100, 90, 500, 30], "text": "二行目", "conf": 0.9}
    assert sorted(l["text"] for l in _drop_repeats([part, whole, below])) == sorted([whole["text"], below["text"]])


def test_ghost_line_that_repeats_a_piece_of_another_is_dropped():
    from ocr import _drop_repeats

    line = {"bbox": [374, 0, 556, 44], "text": "ミィティのお父さん……とても立派な方だったんだね。", "conf": 0.99}
    ghost = {"bbox": [665, 8, 248, 88], "text": "も立派な方たったんだね", "conf": 0.94}
    second = {"bbox": [400, 46, 500, 44], "text": "本当にそう思うよ。", "conf": 0.95}
    assert [l["text"] for l in _drop_repeats([line, ghost, second])] == [line["text"], second["text"]]


def test_read_name_lets_the_glossary_pick_the_right_reading(monkeypatch):
    import ocr

    terms = [("ミィティ", "Mitty"), ("テララ", "Terara")]
    find = lambda t: (engines.match_term(t, terms) or (None,))[0]  # noqa: E731
    img = np.zeros((45, 384, 3), np.uint8)

    def readings(*texts):
        it = iter(texts)
        monkeypatch.setattr(ocr, "read_lines", lambda pic, lang: [
            {"bbox": [100, 10, 80, 20], "dbox": [96, 4, 88, 32], "text": t, "conf": 0.8}] if (t := next(it)) else [])

    readings("テララ")  # known at once: read only once
    assert ocr.read_name(img, "ja", find, engines.fold_name)[0]["text"] == "テララ"
    readings("三行", "ミイテイ")  # misread over a bright scene; the second reading is a known name
    got = ocr.read_name(img, "ja", find, engines.fold_name)
    assert got[0]["text"] == "ミィティ" and got[0]["bbox"] == [50, 5, 40, 10]  # glossary spelling, box in the original's pixels
    readings("ミィティの父", "ミイティの父", "77")  # not in the glossary, but two readings agree: a real name
    assert ocr.read_name(img, "ja", find, engines.fold_name)[0]["text"] == "ミィティの父"
    readings("三行", "美", "")  # nobody knows it and the readings disagree: leave the original on screen
    assert ocr.read_name(img, "ja", find, engines.fold_name) == []


def test_fix_kana_lookalikes():
    from ocr import fix_kana

    assert fix_kana("ほ一ら、こ一いうとき") == "ほーら、こーいうとき" and fix_kana("この一言") == "この一言"
    assert fix_kana("このべリーの") == "このベリーの" and fix_kana("すベては") == "すべては" and fix_kana("食べる") == "食べる"
    assert fix_kana("ベタべタする") == "ベタベタする" and fix_kana("一つ") == "一つ" and fix_kana("もう一度") == "もう一度"


def test_read_dialog_ignores_text_that_is_not_centred_and_rereads_doubtful_lines(monkeypatch):
    import ocr

    img = np.zeros((106, 1305, 3), np.uint8)
    jp = lambda t: any("぀" <= c <= "鿿" for c in t)  # noqa: E731
    L = lambda x, w, t, c=0.98, y=6: {"bbox": [x, y, w, 30], "dbox": [x, y, w, 36], "text": t, "conf": c}  # noqa: E731
    calls = []

    def readings(*pages):
        it = iter(pages)
        calls.clear()
        monkeypatch.setattr(ocr, "read_lines", lambda pic, lang: (calls.append(1), next(it))[1])

    two = [L(402, 500, "ああ、これは天道の塔だ。"), L(352, 600, "谷に存在している。", y=46)]
    readings(two)
    assert ocr.read_dialog(img, "ja", True, jp) == two and len(calls) == 1  # sure and centred: read once
    pieces = [L(480, 100, "名付けて"), L(620, 150, "ベタベタ弾"), L(800, 20, "！")]
    readings(pieces)
    assert ocr.read_dialog(img, "ja", True, jp) == pieces  # one centred row in three pieces
    readings([L(900, 160, "購入制限：0")], [L(900, 160, "購入制限：0")], [L(1350, 240, "購入制限：0")])
    assert ocr.read_dialog(img, "ja", True, jp) == []  # a shop's text under the region is not dialogue
    readings([L(900, 160, "購入制限：0")])
    assert ocr.read_dialog(img, "ja", False, jp)  # ...a region with a left-aligned box takes it
    part, whole = [L(650, 250, "る……これは……？", 0.8)], [L(402, 500, "くっ……体がベタベタする……これは……？", 0.95)]
    readings(part, whole)
    assert ocr.read_dialog(img, "ja", True, jp) == whole and len(calls) == 2  # half a line over a bright scene: read again
    lone = [L(620, 60, "うん。")]
    readings([L(640, 30, "ん。")], [{**lone[0], "bbox": [930, 9, 90, 45], "dbox": [930, 9, 90, 54]}])
    got = ocr.read_dialog(img, "ja", True, jp)
    assert [l["text"] for l in got] == ["うん。"] and got[0]["bbox"] == [620, 6, 60, 30] and len(calls) == 2  # a word: read bigger


def test_index_translate_gets_names_as_a_hard_constraint(monkeypatch):
    sent = []

    class Resp:
        status_code = 200

        def json(self):
            return {"choices": [{"message": {"content": "ไปกันเถอะ Mitty�"}}]}

    def fake_post(url, json, timeout):
        sent.append(json)
        return Resp()

    monkeypatch.setattr(engines, "http_post", fake_post)
    e = engines.IndexTranslateEngine()
    out = e.translate_page(["ミイティ、行こう"], "ja", "th", [{"src": "おはよう", "tl": "อรุณสวัสดิ์"}],
                           {"characters": [{"src": "ミィティ", "tl": "Mitty"}]}, style="game")
    assert out == ["ไปกันเถอะ Mitty"]  # broken bytes dropped
    p = sent[0]["messages"][0]["content"]
    assert "翻译成泰语" in p and "【源文】\nミイティ、行こう" in p and "ミイティ -> Mitty" in p and "おはよう => อรุณสวัสดิ์" in p
    assert sent[0]["chat_template_kwargs"] == {"enable_thinking": False} and sent[0]["temperature"] == 0
    assert engines.engine_order("index", False)[0] == ["index", "hymt", "local"]


def test_google_free_is_no_engine_without_its_key(monkeypatch):
    """A build given to others has Chrome's key taken out: the engine says it is not there instead of trying."""
    monkeypatch.setattr(web_engines.GoogleFreeEngine, "KEY", "")
    e = web_engines.GoogleFreeEngine(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=[["x"]])))
    assert e.available() is False
    with pytest.raises(engines.EngineError):
        e.translate_page(["こんにちは"], "ja", "th", [], {})
