"""Client logic without a game, server or GPU. Run: .venv\\Scripts\\python.exe -m pytest tests -q"""
import json
import os
import sys
from pathlib import Path

import cv2
import httpx
import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gamelens import thai, win32  # noqa: E402
from gamelens.watcher import RegionWatcher, has_text, ink_map, similar  # noqa: E402


def box(text_px=0, w=400, h=100):
    img = np.full((h, w, 3), (40, 30, 20), np.uint8)
    if text_px:
        img[40:60, 20 : 20 + text_px] = 255
    return img


def test_watcher_typewriter_then_stable():
    w = RegionWatcher(stable_ms=250)
    t = 0.0
    assert w.feed(ink_map(box(0)), t) == "changed"  # first frame
    t += 0.3
    assert w.feed(ink_map(box(0)), t) == "stable"
    # a typewriter effect: a few new glyph columns every 50 ms
    events = []
    for i in range(1, 8):
        t += 0.05
        events.append(w.feed(ink_map(box(i * 30)), t))
    assert events[0] == "changed" and set(events[1:]) == {None}
    t += 0.1
    assert w.feed(ink_map(box(7 * 30)), t) is None  # not yet still for 250 ms
    t += 0.2
    assert w.feed(ink_map(box(7 * 30)), t) == "stable"
    t += 0.5
    assert w.feed(ink_map(box(7 * 30)), t) is None  # nothing new


def test_watcher_ignores_background_motion_and_forces_after_max_wait():
    rng = np.random.default_rng(0)
    w = RegionWatcher(stable_ms=250, max_wait_ms=1000)
    t = 0.0
    w.feed(ink_map(box(200)), t)
    # dark noise behind light text: the light-ink map does not move
    for _ in range(5):
        t += 0.05
        img = box(200)
        img[70:] = rng.integers(0, 120, img[70:].shape, dtype=np.uint8)
        w.feed(ink_map(img), t)
    assert w.dirty_since is None or t - w.dirty_since < 1  # settled, not stuck
    # constantly changing light pixels: still read after max_wait
    w2 = RegionWatcher(stable_ms=250, max_wait_ms=600)
    got, t = [], 0.0
    for i in range(40):
        t += 0.05
        got.append(w2.feed(ink_map(box(20 + (i % 2) * 200)), t))
    assert "stable" in got


def test_has_text_and_similarity():
    assert not has_text(ink_map(box(0)))
    assert has_text(ink_map(box(200)))
    assert similar("指揮官、おはよう", "指揮官、おはよう") == 1.0
    assert similar("指揮官、おはよう。", "指揮官、おはよう") > 0.92
    assert similar("はい", "いいえ、違います") < 0.5


def test_thai_breakable_inserts_zero_width_spaces():
    s = thai.breakable("ผู้บัญชาการคะวันนี้อากาศดีมาก")
    assert thai.ZWSP in s and s.replace(thai.ZWSP, "") == "ผู้บัญชาการคะวันนี้อากาศดีมาก"
    assert thai.breakable("Hello world") == "Hello world"


def test_overlay_fit_wraps_thai_and_shrinks():
    from PySide6.QtWidgets import QApplication

    from gamelens.overlay import fit_lines

    QApplication.instance() or QApplication([])
    text = "ผู้บัญชาการคะ วันนี้อากาศดีมากเลยนะคะ ไปเดินเล่นที่ท่าเรือด้วยกันไหมคะ"
    font, lines, lh = fit_lines(text, "Leelawadee UI", False, 300, 120, 40)
    assert len(lines) >= 2 and len(lines) * lh <= 120 and font.pixelSize() <= 40
    assert all(w <= 301 for _, w in lines)
    assert "".join(s for s, _ in lines).replace(" ", "") == text.replace(" ", "")
    big, _, _ = fit_lines("ลาฟีย์", "Leelawadee UI", True, 300, 40, 30)
    assert big.pixelSize() >= 25  # a short name keeps (almost) the original size


def test_erase_patch_covers_only_glyphs():
    from gamelens.render import contrast_outline, erase_patch

    img = np.full((80, 300, 3), (60, 40, 30), np.uint8)
    cv2.putText(img, "TEXT", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)
    tight = erase_patch(img, [[18, 24, 100, 34]])  # a box tight around the glyphs: the cover still grows past it
    assert tight[20, 60, 3] > 0 or tight[60, 60, 3] > 0
    p = erase_patch(img, [[10, 15, 150, 50]])
    assert p.shape == (80, 300, 4)
    assert p[5, 290, 3] == 0 and p[:, :, 3].max() == 255  # transparent away from the text
    painted = p[:, :, 3] > 200
    assert np.abs(p[painted][:, :3].astype(int) - [60, 40, 30]).mean() < 40  # glyphs filled with the box colour
    assert contrast_outline("#ffffff") == "#101010" and contrast_outline("#000000") == "#f4f4f4"


def test_text_rect_for_dialog_and_name():
    from gamelens.overlay import text_rect_for

    region = (100, 500, 800, 150)
    boxes = [[20, 10, 400, 30], [20, 50, 300, 30]]
    x, y, w, h = text_rect_for(region, boxes, "dialog")
    assert (x, y) == (120, 510) and w >= 400 and y + h <= 650
    nx, ny, nw, nh = text_rect_for(region, [[10, 5, 80, 20]], "name")
    assert nx == 110 and nw >= 80 and nh > 30 and ny + nh // 2 == 500 + 15  # same start, same centre line


def test_parse_hotkey():
    assert win32.parse_hotkey("Ctrl+F8") == (win32.MOD_CONTROL, 0x77)
    assert win32.parse_hotkey("F9") == (0, 0x78)
    with pytest.raises(ValueError):
        win32.parse_hotkey("Ctrl+")


def test_region_px_clamps():
    from gamelens.app import region_px

    assert region_px([0.5, 0.5, 0.8, 0.8], (100, 200, 3)) == (100, 50, 100, 50)


def test_api_client_sends_token_and_image(tmp_path):
    from gamelens.api import LensClient, LensError

    tok = tmp_path / "token.txt"
    tok.write_text("secret\n")
    seen = {}

    def handler(req: httpx.Request):
        seen[req.url.path] = req
        if req.url.path == "/ocr/lines":
            return httpx.Response(200, json={"lines": [], "text": ""})
        if req.url.path == "/translate/lines":
            return httpx.Response(200, json={"lines": {"a": "ก"}, "engine": "hymt"})
        return httpx.Response(503, json={"detail": "translation failed: x"})

    c = LensClient("http://lens", tok, transport=httpx.MockTransport(handler))
    c.ocr_lines(np.zeros((10, 10, 3), np.uint8))
    assert seen["/ocr/lines"].headers["x-lens-token"] == "secret" and b"image/png" in seen["/ocr/lines"].content
    r = c.translate_lines([{"id": "a", "src": "あ", "kind": "dialog"}], series="s", target="th", src_lang="ja",
                          engine="hymt", allow_cloud=False, context=[])
    assert r["lines"]["a"] == "ก" and json.loads(seen["/translate/lines"].content)["series"] == "s"
    with pytest.raises(LensError, match="503"):
        c.usage()
    with pytest.raises(LensError, match="token"):
        LensClient("http://lens", tmp_path / "missing.txt", transport=httpx.MockTransport(handler)).usage()


def test_profile_roundtrip(tmp_path):
    from gamelens.profile import Profile, Region

    p = Profile(name="x", series="x", path=str(tmp_path / "p.json"), regions=[Region(id="r1", kind="name")])
    p.save()
    q = Profile.load(tmp_path / "p.json")
    assert q.regions[0].kind == "name" and q.engine == "hymt" and q.path.endswith("p.json")


def test_ink_map_keeps_thin_strokes_when_shrinking():
    img = np.full((160, 980, 3), (45, 30, 20), np.uint8)
    cv2.putText(img, "thin text line here", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (245, 245, 245), 1)
    m = ink_map(img, "light")
    assert m.shape[1] == 320 and has_text(m)


def test_cluster_lines_and_suggested_regions():
    from gamelens.region_editor import cluster_lines, suggest_region

    lines = [  # a speaker name, two dialogue lines, and an unrelated HUD label far away
        {"bbox": [180, 470, 70, 25], "text": "ラフィー"},
        {"bbox": [180, 540, 450, 34], "text": "指揮官、おはよう。"},
        {"bbox": [180, 584, 380, 34], "text": "今日はいい天気だね。"},
        {"bbox": [1100, 20, 120, 20], "text": "メニュー"},
    ]
    lines.append({"bbox": [650, 540, 120, 34], "text": "ね"})  # a second piece on the first dialogue row
    blocks = cluster_lines(lines)
    assert len(blocks) == 3
    dialog = next(b for b in blocks if b["lines"] == 2)
    assert dialog["text"] == "指揮官、おはよう。ね今日はいい天気だね。" and dialog["bbox"][2] == 590
    kind, rect = suggest_region(dialog, 1280, 720, blocks)
    assert kind == "dialog" and rect[0] < 180 / 1280 and rect[2] > 450 / 1280 and rect[1] + rect[3] <= 1.0
    name = next(b for b in blocks if b["text"] == "ラフィー")
    kind, rect = suggest_region(name, 1280, 720, blocks)
    assert kind == "name" and rect[2] < 0.3
    # one short dialogue line with nothing below it is dialogue, not a name
    one = cluster_lines([{"bbox": [180, 540, 200, 34], "text": "はい"}])
    assert suggest_region(one[0], 1280, 720, one)[0] == "dialog"


def test_panel_builds_and_switches_engine_fields():
    from PySide6.QtWidgets import QApplication

    from gamelens.panel import Panel

    QApplication.instance() or QApplication([])
    p = Panel({"toggle": "Ctrl+F8", "screen": "F9", "regions": "F10", "history": "F7"}, "F8")
    got = []
    p.engine_changed.connect(got.append)
    p.load("hymt", 0.9, "th", True)
    assert p.key_box.isHidden() and p.font.value() == 90
    p.engine.setCurrentIndex(p.engine.findData("azure"))
    assert got == ["azure"] and not p.key_box.isHidden()
    assert not p.key_edits["AZURE_TRANSLATOR_REGION"].isHidden() and p.key_edits["DEEPL_API_KEY"].isHidden()
    saved = []
    p.save_keys.connect(saved.append)
    p.key_edits["AZURE_TRANSLATOR_KEY"].setText(" k1 ")
    p._save_keys()
    assert saved == [{"AZURE_TRANSLATOR_KEY": "k1"}]
    p.set_keys({"azure": {"AZURE_TRANSLATOR_KEY": "••••abcd", "AZURE_TRANSLATOR_REGION": ""}})
    assert "abcd" in p.key_edits["AZURE_TRANSLATOR_KEY"].placeholderText()
    p.s_game.set("bad", "ยังไม่ได้เลือก", "เลือกเกม", primary=True)
    assert p.s_game.button.text() == "เลือกเกม"
    p.set_running(False, False, "x")
    assert not p.big.isEnabled()


def test_friendly_errors_are_thai_and_short():
    from gamelens.app import friendly_error

    assert "ยังไม่พร้อม" in friendly_error("ต่อ server ไม่ได้ (http://127.0.0.1:9960): [WinError 10061] refused")
    assert "429" in friendly_error("server HTTP 503: translation failed: google_free: Google (free) HTTP 429 (rate limited)")
    assert "โควตา" in friendly_error("server HTTP 503: translation failed: deepl: DeepL HTTP 456 (quota exceeded): x")
    assert len(friendly_error("x: " + "y" * 500)) < 100


def test_snap_dialog_region_to_its_box():
    from gamelens.region_editor import cluster_lines, snap_to_box, suggest_region

    frame = np.full((720, 1280, 3), (60, 140, 70), np.uint8)
    frame[518:677, 154:1126] = (55, 45, 27)  # the dialogue box
    frame[545:575, 183:640] = (245, 245, 245)  # its text
    blocks = cluster_lines([{"bbox": [183, 545, 457, 30], "text": "指揮官、おはよう……今日はいい天気だね。"}])
    kind, rect = suggest_region(blocks[0], 1280, 720, blocks)
    x, y, w, h = snap_to_box(frame, blocks[0], rect)
    assert kind == "dialog" and abs((x + w) * 1280 - 1120) < 12 and abs((y + h) * 720 - 670) < 12


def test_is_continuation_tells_typing_from_a_new_line():
    from gamelens.app import is_continuation

    half, full, other = ink_map(box(100)), ink_map(box(300)), ink_map(box(0))
    other[10:20, 150:300] = True
    assert is_continuation(half, full)  # same line, more text
    assert not is_continuation(full, other)  # replaced by a different line
    assert not is_continuation(full, half) and not is_continuation(None, full)


def test_in_any_skips_lines_inside_regions():
    from gamelens.app import in_any

    assert in_any([110, 110, 20, 10], [(100, 100, 200, 50)])
    assert not in_any([10, 10, 20, 10], [(100, 100, 200, 50)]) and not in_any([10, 10, 20, 10], [])
