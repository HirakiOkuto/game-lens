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
from gamelens.watcher import TextFollower, has_text, ink_map, similar, stroke_map  # noqa: E402


TEXT = "The quick brown fox jumps over"


def scene(chars=0, text=TEXT, seed=None, w=640, h=90):
    """A dialogue region: `chars` characters of white text typed so far. With a seed, the background is a
    moving scene: big bright shapes that are somewhere else in every frame."""
    img = np.full((h, w, 3), (60, 45, 30), np.uint8)
    if seed is not None:
        rng = np.random.default_rng(seed)
        for _ in range(6):
            x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
            cv2.circle(img, (x, y), int(rng.integers(25, 60)), (170, 160, 150), -1)  # bright, but text stays readable
    if chars:
        cv2.putText(img, text[:chars], (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (250, 250, 250), 2, cv2.LINE_AA)
    return img


def follow(w, frames, t=0.0, dt=0.05, ocr=0.1, shown_of=None):
    """Play frames [(picture, text an OCR would read)]; OCR takes `ocr` seconds. Returns (log, time)."""
    log, due = [], getattr(w, "_due", None)  # a reading may be under way from the frames before
    for pic, text in frames:
        t += dt
        ev = w.feed(stroke_map(pic), t)
        if ev:
            log.append((ev, round(t, 2)))
        if ev == "read":
            due = (t + ocr, text)
        if due and t >= due[0]:
            text_read, due = due[1], None
            what = w.on_read(text_read, t)
            if what:
                log.append((what, text_read))
    w._due = due
    return log, t


def typed(text, start=2, step=2, **kw):
    return [(scene(n, text, **{k: (v + n if k == "seed" else v) for k, v in kw.items()}), text[:n]) for n in range(start, len(text) + 1, step)]


def still(text, n, **kw):
    return [(scene(len(text), text, **{k: (v + 1000 + i if k == "seed" else v) for k, v in kw.items()}), text) for i in range(n)]


def wants(log):
    return [x for e, x in log if e == "translate"]


LINE = "The quick brown fox jumps ove."  # ends like a sentence


def test_stroke_map_keeps_text_and_drops_big_bright_areas():
    assert not has_text(stroke_map(scene(0)))
    assert has_text(stroke_map(scene(len(TEXT))))
    blobs = stroke_map(scene(0, seed=1))
    assert blobs.mean() < 0.3 * stroke_map(scene(len(TEXT), seed=1)).mean()  # the bright shapes are not strokes
    dark = np.full((90, 640, 3), 230, np.uint8)
    cv2.putText(dark, TEXT, (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (40, 40, 40), 2, cv2.LINE_AA)
    assert has_text(stroke_map(dark, "dark")) and has_text(stroke_map(dark, "any"))  # dark text on a light box
    assert stroke_map(np.zeros((300, 1920, 3), np.uint8)).shape[1] == 480


def test_follower_translates_as_soon_as_the_sentence_is_complete():
    w = TextFollower()
    log, t = follow(w, [(scene(0), "")] * 8)
    assert log == []  # an empty region is never read
    log, t = follow(w, typed(LINE), t)
    assert wants(log) == []  # half a line is not translated, however often it is read
    done = t
    log, t = follow(w, still(LINE, 16), t)
    assert wants(log) == [LINE] and w.settled
    first = next(i for i, (e, _) in enumerate(log) if e == "translate")
    assert log[first - 1][0] == "read" and log[first - 1][1] - done <= 0.5  # within half a second of the last letter
    log, t = follow(w, still(LINE, 40), t)
    assert [e for e, _ in log] == ["read"] * len(log) and len(log) <= 4  # it stays: a few checks, nothing else
    other = "A completely different line!!"
    log, t = follow(w, still(other, 16), t)
    assert wants(log) == [other]  # replaced: noticed by its strokes or by reading it again at once
    log, t = follow(w, [(scene(0), "")] * 10, t)
    assert "hide" in [e for e, _ in log] and not w.settled


def test_follower_without_sentence_end_wants_two_equal_readings():
    w = TextFollower()
    line = "The quick brown fox jumps over"
    log, t = follow(w, typed(line) + still(line, 24))
    assert wants(log) == [line] and [e for e, _ in log].count("read") >= 2
    careful = TextFollower(confirm=True)  # engines paid by the character: two equal readings in any case
    log, _ = follow(careful, still(LINE, 4) + still(LINE, 20))
    reads_before = [e for e, _ in log[: [e for e, _ in log].index("translate")]].count("read")
    assert wants(log) == [LINE] and reads_before >= 2


def test_follower_line_that_goes_on_after_a_pause():
    """The game stops typing at "Oh no!" for a moment, then goes on: the first reading is dropped in time."""
    w = TextFollower()
    whole = "Oh no! The quick brown fox runs."
    half = [(scene(6, whole), whole[:6])] * 14
    log, t = follow(w, half)
    assert wants(log) == ["Oh no!"]
    seq = w.seq
    log, t = follow(w, typed(whole, 8) + still(whole, 16), t)
    assert w.seq > seq  # what was on its way for "Oh no!" is out of date
    assert wants(log) == [whole]


def test_follower_over_a_moving_scene():
    w = TextFollower()
    log, t = follow(w, [(scene(0, seed=i), "") for i in range(10)])
    assert wants(log) == []
    log, t = follow(w, typed(LINE, seed=100) + still(LINE, 20, seed=200), t)
    assert wants(log) == [LINE]
    log, t = follow(w, still(LINE, 30, seed=5000), t)
    assert {e for e, _ in log} <= {"read"}  # the scene alone changes nothing (a check or two)
    other = "Another line arrives here now."
    log, t = follow(w, still(other, 20, seed=9000), t)
    assert wants(log) == [other]


def test_follower_reads_short_lines():
    w = TextFollower()
    log, _ = follow(w, [(scene(0), "")] * 5 + [(scene(3, "Un."), "Un.")] * 14)
    assert wants(log) == ["Un."]


def test_follower_reads_ever_less_often_what_it_cannot_read():
    w = TextFollower()
    frames = [(scene(6 + (i // 6) % 9 * 2), "") for i in range(400)]  # strokes that change but are never text
    log, t = follow(w, frames)
    reads = [x for e, x in log if e == "read"]
    assert wants(log) == [] and len(reads) < 0.06 * len(frames)  # 20 s: not more than about one a second
    same = TextFollower()
    log, _ = follow(same, [(scene(12), "")] * 300)  # strokes that stand still and cannot be read
    assert [e for e, _ in log].count("read") <= 3


def test_follower_again_after_an_error():
    w = TextFollower()
    log, t = follow(w, still(LINE, 14))
    assert wants(log) == [LINE]
    w.again()  # the server was not ready
    log, t = follow(w, still(LINE, 14), t)
    assert wants(log) == [LINE]


def test_similarity():
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


def _ui_frame():
    """A light panel with a dark label on the left, a white pill with a centred value, and a dark bar heading."""
    img = np.full((200, 600, 3), (225, 215, 205), np.uint8)
    cv2.putText(img, "Label", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (90, 80, 70), 1, cv2.LINE_AA)
    cv2.rectangle(img, (360, 40), (560, 72), (252, 252, 252), -1)
    cv2.putText(img, "Val", (444, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (90, 80, 70), 1, cv2.LINE_AA)
    cv2.rectangle(img, (30, 110), (570, 136), (50, 45, 40), -1)
    cv2.putText(img, "Heading", (34, 129), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1, cv2.LINE_AA)
    return img


def test_analyze_line_colours_room_and_glyphs():
    from gamelens.render import analyze_line, luminance

    img = _ui_frame()
    label = analyze_line(img, [38, 44, 56, 20])
    assert luminance(label.fg) < 120 < luminance(label.bg)  # dark text on the light panel
    assert label.free_right > 200 and label.free_right < 330  # runs up to the pill, not through it
    assert 10 <= label.glyph_h <= 20
    val = analyze_line(img, [442, 46, 36, 20])
    assert luminance(val.bg) > 240 and abs(val.free_left - val.free_right) < 20  # centred in the pill
    head = analyze_line(img, [32, 112, 84, 22])  # the box starts at the very edge of the dark bar
    assert luminance(head.bg) < 80 < luminance(head.fg) and head.free_right > 300


def test_erase_lines_covers_only_the_glyphs():
    from gamelens.render import analyze_line, contrast_outline, erase_lines, outline_for

    img = _ui_frame()
    label = analyze_line(img, [38, 44, 56, 20])
    p = erase_lines(img, [label])
    assert p.shape == (200, 600, 4) and p[150, 300, 3] == 0 and p[:, :, 3].max() == 255
    painted = p[:, :, 3] > 200
    assert np.abs(p[painted][:, :3].astype(int) - [225, 215, 205]).max() < 25  # filled with the panel colour
    assert p[40:76, 360:560, 3].max() == 0  # the pill next to it is untouched
    assert erase_lines(img, []) is None
    assert contrast_outline("#ffffff") == "#101010" and contrast_outline("#000000") == "#f4f4f4"
    assert outline_for("#303030", "#f0f0f0") == "#f0f0f0" and outline_for("#808080", "#909090") == "#f4f4f4"
    # over a picture: light text gets a dark outline, text darker than its background a light one
    assert outline_for("#f5f5f5", "#405060", False) == "#101010" and outline_for("#a09080", "#f8ecd5", False) == "#f4f4f4"


def test_screen_layout_keeps_one_size_and_uses_the_room():
    from gamelens.screen import common_heights, has_source_text, lines_to_translate, paragraphs, place, screen_items

    assert has_source_text("画質", "ja") and has_source_text("オリジナル", "ja")
    assert not has_source_text("90FPS", "ja") and not has_source_text("1280×720", "ja") and has_source_text("OFF", "en")
    lines = [{"id": "a", "bbox": [0, 0, 10, 10], "text": "TAA"}, {"id": "b", "bbox": [0, 0, 10, 10], "text": "垂直同期"}]
    assert [l["id"] for l in lines_to_translate(lines, [], "ja")] == ["b"]
    unsure = [{"id": "e", "bbox": [0, 0, 40, 40], "text": "米", "conf": 0.66}]  # an emblem "read" as a character
    assert lines_to_translate(unsure, [], "ja") == []
    assert common_heights([13, 16, 17, 18, 27]) == [16.0, 17.0, 17.0, 17.0, 27.0]

    img = _ui_frame()
    ocr = [{"id": "l", "bbox": [38, 44, 56, 20], "text": "x"}, {"id": "v", "bbox": [442, 46, 36, 20], "text": "x"},
           {"id": "n", "bbox": [32, 112, 84, 22], "text": "x"}]
    assert [len(g) for g in paragraphs(ocr)] == [1, 1, 1]  # a label, a value, a heading: no running text here
    items = screen_items(img, paragraphs(ocr), {"l": "ป้ายที่ยาวกว่าต้นฉบับมาก", "v": "ค่า", "n": ""}, 0.9)
    assert len(items) == 2  # the line without a translation is left alone
    label, val = items
    assert label.align == "left" and 38 <= label.rect[0] <= 44 and label.rect[2] > 200  # may run over the empty panel
    assert val.align == "center" and abs((val.rect[0] + val.rect[2] / 2) - 460) < 12  # centred in the pill
    assert label.patch is not None and label.patch.pixelColor(50, 125).alpha() == 0  # untranslated heading not erased
    assert label.outline and label.fg != label.outline


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
    # a 720p window's text is enlarged before it is read, and the boxes come back in the picture's own pixels
    boxed = LensClient("http://lens", tok, transport=httpx.MockTransport(
        lambda req: httpx.Response(200, json={"lines": [{"bbox": [30, 15, 300, 30], "dbox": [27, 12, 306, 36], "text": "x"}], "text": "x"})))
    got = boxed.ocr_lines(np.zeros((71, 870, 3), np.uint8), frame_h=720)["lines"][0]
    assert got["bbox"] == [20, 10, 200, 20] and got["dbox"] == [18, 8, 204, 24]
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
    assert m.shape[1] == 320 and m.mean() > 0.002


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
    kind, rect, align = suggest_region(dialog, 1280, 720, blocks)
    assert kind == "dialog" and rect[0] < 180 / 1280 and rect[2] > 450 / 1280 and rect[1] + rect[3] <= 1.0
    name = next(b for b in blocks if b["text"] == "ラフィー")
    assert align == "left"
    kind, rect, _ = suggest_region(name, 1280, 720, blocks)
    assert kind == "name" and rect[2] < 0.3
    # subtitle-style dialogue set around the middle of the screen: a centred region, wide enough for long lines
    mid = cluster_lines([{"bbox": [520, 602, 240, 17], "text": "ああ、これは天道の塔だ。"}, {"bbox": [600, 555, 80, 20], "text": "ファランタ"}])
    line = next(b for b in mid if b["text"].startswith("ああ"))
    kind, rect, align = suggest_region(line, 1280, 720, mid)
    assert (kind, align) == ("dialog", "center") and abs(rect[0] + rect[2] / 2 - 0.5) < 0.01 and rect[2] > 0.6
    who = next(b for b in mid if b["text"] == "ファランタ")
    kind, rect, align = suggest_region(who, 1280, 720, mid)
    assert (kind, align) == ("name", "center") and abs(rect[0] + rect[2] / 2 - 0.5) < 0.01
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
    kind, rect, _ = suggest_region(blocks[0], 1280, 720, blocks)
    x, y, w, h = snap_to_box(frame, blocks[0], rect)
    assert kind == "dialog" and abs((x + w) * 1280 - 1120) < 12 and abs((y + h) * 720 - 670) < 12


def test_in_any_skips_lines_inside_regions():
    from gamelens.screen import in_any

    assert in_any([110, 110, 20, 10], [(100, 100, 200, 50)])
    assert not in_any([10, 10, 20, 10], [(100, 100, 200, 50)]) and not in_any([10, 10, 20, 10], [])


def test_region_item_for_a_dialogue_box():
    from gamelens.screen import region_item

    crop = np.full((160, 900, 3), (55, 45, 27), np.uint8)  # a dark dialogue box with two lines of light text
    cv2.putText(crop, "first line of text", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (245, 245, 245), 2, cv2.LINE_AA)
    cv2.putText(crop, "second line", (30, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (245, 245, 245), 2, cv2.LINE_AA)
    boxes = [{"bbox": [28, 26, 290, 32], "dbox": [24, 20, 298, 44], "text": "一行目です"}, {"bbox": [28, 71, 190, 32], "dbox": [24, 65, 198, 44], "text": "二行目"}]
    it = region_item(crop, (150, 520, 900, 160), boxes, "บรรทัดแรกของข้อความ บรรทัดที่สอง", "dialog", "left", 0.9)
    assert it.rect[0] == 178 and it.rect[1] == 546 and it.rect[2] > 800  # from the first line, across the box
    assert int(it.fg[1:3], 16) > 200 and int(it.outline[1:3], 16) < 90  # light text, outline in the box colour
    assert 18 <= it.max_px <= 34 and it.patch is not None and it.patch_pos == (150, 520)
    assert it.patch.pixelColor(40, 40).alpha() > 200 and it.patch.pixelColor(600, 140).alpha() == 0
    assert region_item(crop, (150, 520, 900, 160), boxes, "", "dialog", "left", 0.9) is None



def test_paragraphs_join_running_text_but_not_list_rows():
    from gamelens.screen import paragraphs, place_block, requests, screen_items

    # a description of three tightly stacked lines, then list rows spaced a full line apart
    para = [{"id": f"p{i}", "bbox": [80, 100 + i * 27, 380 - i * 40, 20], "text": t}
            for i, t in enumerate(["また、桃の花が咲く季節が来たね", "最近みんなはどんな悩みを", "抱えているのかな？"])]
    rows = [{"id": f"r{i}", "bbox": [80, 300 + i * 38, 90, 20], "text": t} for i, t in enumerate(["攻撃力", "物理防御力", "魔法防御力"])]
    groups = paragraphs(rows + para)
    assert sorted(len(g) for g in groups) == [1, 1, 1, 3]
    req = requests(groups, "ja")
    block = next(r for r in req if r["id"] == "p0")
    assert block["src"] == "また、桃の花が咲く季節が来たね最近みんなはどんな悩みを抱えているのかな？"
    # two centred subtitle lines of different lengths are one block too
    centred = paragraphs([{"id": "a", "bbox": [300, 500, 400, 24], "text": "x"}, {"id": "b", "bbox": [200, 532, 600, 24], "text": "y"}])
    assert [len(g) for g in centred] == [2]

    img = np.full((420, 600, 3), (238, 233, 228), np.uint8)
    for i, w in enumerate((380, 340, 300)):
        cv2.putText(img, "x" * (w // 19), (80, 117 + i * 27), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (90, 80, 70), 1, cv2.LINE_AA)
    items = screen_items(img, [para], {"p0": "อีกครั้งแล้วนะ ถึงฤดูที่ดอกพีชบานอีกแล้ว ช่วงนี้ทุกคนมีเรื่องกังวลอะไรกันบ้างนะ"}, 0.9)
    assert len(items) == 1
    it = items[0]
    assert it.align == "left" and not it.vcenter and 76 <= it.rect[0] <= 84  # one block, from the paragraph's left edge
    assert it.rect[2] >= 380 and it.rect[3] >= 70  # as wide as the longest line at least, and all its lines deep
    for y in (110, 137, 164):
        assert it.patch.pixelColor(100, y).alpha() > 0  # every original line is covered


def test_merge_glossary_file_wins_and_server_only_names_stay():
    from gamelens.screen import merge_glossary

    base = {"characters": [{"src": "テララ", "tl": "Terara"}], "terms": [{"src": "キボ", "tl": "Kibo"}]}
    server = {"characters": [{"src": "テララ", "tl": "เทราระ"}, {"src": "ファランタ", "tl": "ฟาลันตา"}], "terms": []}
    m = merge_glossary(base, server)
    assert [c["tl"] for c in m["characters"]] == ["Terara", "ฟาลันตา"] and m["terms"] == [{"src": "キボ", "tl": "Kibo"}]
    assert merge_glossary(base, m) == m  # nothing to send the second time


def test_region_item_light_text_over_a_bright_scene():
    from gamelens.render import luminance
    from gamelens.screen import region_item

    crop = np.full((50, 384, 3), (60, 130, 90), np.uint8)  # grass, with bright skin and cloth in front of it
    crop[:, :170] = (170, 190, 220)
    cv2.circle(crop, (250, 20), 22, (200, 200, 205), -1)
    cv2.putText(crop, "Name", (150, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (250, 250, 250), 2, cv2.LINE_AA)
    it = region_item(crop, (768, 826, 384, 50), [{"bbox": [0, 0, 384, 50], "dbox": [0, 0, 384, 50], "text": "シンフォーリア"}], "Symphoria", "name", "center", 0.9, None, "light")
    assert luminance(it.fg) > 170 and it.outline == "#101010"  # stays light text with a dark outline
    if it.patch is not None:  # the cover is the glyphs, not a slab over the whole box
        covered = sum(it.patch.pixelColor(x, y).alpha() > 128 for x in range(0, 384, 4) for y in range(0, 50, 4))
        assert covered < 0.4 * 96 * 13


def test_heading_whose_box_pokes_out_of_its_bar_stays_plain():
    """The settings screen's section headings: OCR boxes taller than the dark bar the text sits on."""
    from gamelens.render import analyze_line, luminance

    img = np.full((120, 600, 3), (215, 205, 190), np.uint8)
    cv2.rectangle(img, (30, 42), (570, 65), (63, 64, 66), -1)
    cv2.putText(img, "Custom setting", (45, 59), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (215, 214, 213), 1, cv2.LINE_AA)
    a = analyze_line(img, [32, 38, 150, 31])
    assert a.plain and luminance(a.bg) < 90 and luminance(a.fg) > 160
    assert a.glyph_h <= 18 and 40 <= a.box[0] <= 48  # the glyphs, not the bar's edges


def test_follower_notices_a_line_that_went_on_unseen():
    """Over a bright scene the strokes of what is added may not show: a later check reads the longer line."""
    w = TextFollower()
    log, t = follow(w, still("Really, is that so?", 12))
    assert wants(log) == ["Really, is that so?"]
    longer = [(scene(19, "Really, is that so?"), "Really, is that so? I am sorry.")] * 20  # the same strokes
    log, t = follow(w, longer, t)
    assert wants(log) == ["Really, is that so? I am sorry."] and w.settled
    log, t = follow(w, [(scene(19, "Really, is that so?"), "Really, is tha")] * 30, t)  # read only in part: ignored
    assert {e for e, _ in log} <= {"read"}


def test_follower_does_not_trust_an_ellipsis():
    w = TextFollower()
    whole = "Well... the fox is tired now."
    log, t = follow(w, [(scene(7, whole), whole[:7])] * 4)
    assert wants(log) == []  # "Well..." may go on: wait for a second equal reading


def test_text_size_settles_and_context_replaces_a_lines_beginning():
    from gamelens.screen import TextSize, remember

    z = TextSize()
    assert z.settle(90, 32) == 32  # an implausible first measurement is held down
    for h in (23, 24, 22, 23, 60, 24):
        last = z.settle(h, 32)
    assert 22 <= last <= 25
    ctx = []
    remember(ctx, "おはよう。", "อรุณสวัสดิ์")
    remember(ctx, "なにがあったのか", "เกิดอะไร")
    remember(ctx, "なにがあったのかはわかりません。", "ไม่รู้ว่าเกิดอะไรขึ้น")
    assert [c["src"] for c in ctx] == ["おはよう。", "なにがあったのかはわかりません。"]


def test_same_text_and_goes_on():
    from gamelens.watcher import goes_on, same_text

    assert same_text("体につくとなかなか取れないのが厄介なところだが……", "体につくとなかなか取れないのが厄介なところだが…")  # dots are not counted
    assert not same_text("この建物……すごく", "この建物……すごく高い") and not same_text("うん", "うんう")
    assert goes_on("そんなことがあったんですね……", "そんなことがあったんですね……ミイティ、ごめんなさい。")
    assert not goes_on("そんなことがあったんですね……", "ミイティ、だれにだって怖いものはあります。")


def test_follower_does_not_hide_a_line_whose_strokes_fade_for_a_moment():
    """The scene behind the text turns bright: half the strokes are lost, the text is the same."""
    w = TextFollower()
    log, t = follow(w, still(LINE, 14))
    assert wants(log) == [LINE]
    faded = scene(len(LINE), LINE)
    faded[:, 330:] = (235, 235, 235)
    log, t = follow(w, [(faded, LINE)] * 20 + still(LINE, 20), t)
    assert {e for e, _ in log} <= {"read"} and w.settled  # read again, never hidden


def test_follower_takes_a_line_that_ends_with_dots_once_typing_has_stopped():
    from gamelens.watcher import same_text

    assert same_text("飛翔の儀はまだだけど• • • •", "飛翔の儀はまだだけど……")
    w = TextFollower()
    line = "The quick brown fox jumps...."
    log, t = follow(w, typed(line, step=3), ocr=0.1)
    assert wants(log) == []
    done = t
    log, t = follow(w, still(line, 24), t)
    assert wants(log) == [line]


def test_follower_notices_what_was_typed_while_it_was_being_read():
    w = TextFollower()
    whole = "Oh no! The quick brown fox runs."
    frames = [(scene(6, whole), whole[:6])] * 8 + typed(whole, 7, step=2) + still(whole, 16)
    log, _ = follow(w, frames, ocr=0.25)
    assert wants(log)[-1] == whole


def test_suggested_dialogue_region_has_room_for_three_rows():
    from gamelens.region_editor import cluster_lines, suggest_region

    # 1080p, rows 40 px apart (the game's own measure): one row on screen when the region is set
    blocks = cluster_lines([{"bbox": [730, 903, 462, 23], "text": "な、なによ。アタシはもう子どもじゃないわ！"}])
    kind, rect, align = suggest_region(blocks[0], 1920, 1080, blocks)
    assert (kind, align) == ("dialog", "center")
    assert rect[1] * 1080 <= 903 and (rect[1] + rect[3]) * 1080 >= 903 + 2 * 40 + 26


def test_follower_trusts_reading_once_strokes_have_failed_a_line():
    """Over a busy, bright scene the strokes come and go although the line stands: it must not blink."""
    w = TextFollower()
    line = "Oh no."
    gone = scene(0)
    log, t = follow(w, still(line, 14))
    assert wants(log) == [line] and not w.unsure
    flicker = ([(gone, line)] * 4 + still(line, 8)) * 8  # the OCR still reads it all the time
    log, t = follow(w, flicker, t)
    assert [e for e, _ in log].count("hide") == 1 and w.unsure and w.settled
    log, t = follow(w, [(gone, "")] * 30, t)  # now it is really gone
    assert "hide" in [e for e, _ in log] and not w.settled


def test_follower_reads_anew_a_line_that_was_misread():
    from gamelens.watcher import skeleton

    assert skeleton("ますは、ミイティ") == skeleton("まずは、ミィティ") and skeleton("おしい") != skeleton("おい")
    w = TextFollower()
    log, t = follow(w, still(LINE, 10))
    line, seq = w.line, w.seq
    better = LINE.replace("quick", "brisk")
    log, t = follow(w, [(p, better) for p, _ in still(LINE, 60)], t)
    assert wants(log) == [better] and w.corrected and w.line == line and w.seq > seq
    assert "hide" not in [e for e, _ in log]


def test_follower_counts_lines_apart_from_texts():
    w = TextFollower()
    whole = "Oh no! The quick brown fox runs."
    log, t = follow(w, [(scene(6, whole), whole[:6])] * 14)
    line, seq = w.line, w.seq
    log, t = follow(w, typed(whole, 8) + still(whole, 16), t)
    assert w.line == line and w.seq > seq  # it went on: the translation of "Oh no!" may still be shown
    log, t = follow(w, still("A completely different line!!", 16), t)
    assert w.line > line


def test_begins_wants_whole_rows():
    from gamelens.screen import begins

    row = lambda y, t: {"bbox": [100, y, 20 * len(t), 24], "text": t}
    first = {"lines": [row(10, "そうだな。")]}
    assert begins(first, {"lines": [row(11, "そうだな。"), row(50, "皆を集めてくれ。")]})
    assert not begins(first, {"lines": [row(11, "そうだな。皆を集めてくれ。")]})  # typed on in the same row
    assert begins(first, {"lines": [row(10, "そうだね。")]}) and not begins(first, {"lines": [row(10, "ちがう！")]})


def test_regions_remember_that_they_lie_over_the_scene():
    from gamelens.screen import TextSize, region_item, refresh_patch

    rng = np.random.default_rng(1)
    busy = cv2.resize(rng.integers(0, 255, (12, 60, 3), np.uint8), (640, 100), interpolation=cv2.INTER_CUBIC)
    dark = np.full((100, 640, 3), (30, 60, 20), np.uint8)
    line = [{"bbox": [190, 25, 260, 30], "dbox": [188, 22, 264, 36], "text": "ああああああ"}]
    z = TextSize()
    for pic in (busy, dark):
        cv2.putText(pic, "Oh no, no!", (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    on_box = region_item(dark, (0, 0, 640, 100), line, "ไม่นะ", "dialog", "center", 1.0, None, "light", TextSize())
    item = region_item(busy, (0, 0, 640, 100), line, "ไม่นะ", "dialog", "center", 1.0, None, "light", z)
    item = region_item(dark, (0, 0, 640, 100), line, "ไม่นะ", "dialog", "center", 1.0, None, "light", z)
    assert item.outline == "#101010" and item._seen is None and abs(item.max_px - on_box.max_px) < 1
    before = item.patch
    refresh_patch(item, busy)  # the camera cuts to something else: what covers the text follows
    assert item.patch is not before
    same = on_box.patch
    refresh_patch(on_box, dark)
    assert on_box.patch is same  # a dialogue box that does not change costs nothing
    refresh_patch(on_box, busy)
    assert on_box.patch is not same


def test_strokes_that_were_missed_are_learned_and_rows_extended():
    """The first look missed part of the line; it is covered once it has stood still for a few looks."""
    from gamelens.render import analyze_line, learn_strokes
    from gamelens.screen import TextSize, extend_item, region_item

    def pic(seed, text="Oh no, no!"):
        rng = np.random.default_rng(seed)
        im = cv2.resize(rng.integers(0, 150, (6, 30, 3), np.uint8), (640, 100), interpolation=cv2.INTER_CUBIC)
        cv2.putText(im, text, (200, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        return im

    first = pic(0)
    line = analyze_line(first, [195, 33, 165, 34], None, "light", True)
    whole = line.mask.copy()
    line.mask[:, 80:] = 0
    for i in range(1, 8):
        learn_strokes(line, pic(i))
    assert (line.mask[whole > 0] > 0).mean() > 0.9  # the scene behind changed at every look, the text did not
    assert line.mask.mean() < 1.5 * whole.mean() + 10  # and little else was taken for text

    read = [{"bbox": [195, 33, 165, 34], "dbox": [195, 33, 165, 34], "text": "ああああああああ"}]
    item = region_item(first, (0, 0, 640, 100), read, "ไม่นะ", "dialog", "center", 1.0, None, "light", TextSize())
    n = len(item._lines)
    extend_item(item, first, read, "light", None)
    assert len(item._lines) == n
    more = [dict(read[0], dbox=[195, 33, 260, 34])]
    extend_item(item, pic(0, "Oh no, no! Yes"), more, "light", None)
    assert len(item._lines) == n + 1 and item._lines[-1].box[0] >= 340


def test_auto_screen_gives_way_to_dialogue():
    from gamelens.watcher import auto_pause

    assert auto_pause(set(), 0) < 1  # a menu: at once
    assert auto_pause({"name", "dialog"}, 0) >= 5  # someone is speaking
    assert 1 < auto_pause({"dialog"}, 0) < 5
    assert auto_pause(set(), 1) < auto_pause(set(), 3) <= 8 and auto_pause(set(), 9) <= 8
