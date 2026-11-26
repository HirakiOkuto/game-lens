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
    if thai._icu is None:
        return
    parts = lambda t: thai.breakable(t).split(thai.ZWSP)  # noqa: E731
    # words, every time (the iterator once read a buffer that was gone, and broke letter by letter at random)
    for _ in range(20):
        assert parts("ภาษาไทยเป็นภาษาที่สวยงาม" + " " * _)[:3] == ["ภาษา", "ไทย", "เป็น"]
    # a loanword ICU does not know breaks between syllables, never inside one or before a vowel sign
    assert parts("ไอเทมเควสต์") == ["ไอ", "เทม", "เควสต์"]
    for w in ("อนิม่า", "อุปกรณ์", "สตาร์ลิงก์", "แท่นหินแห่งเสียงสะท้อน"):
        assert all(q and q[0] not in thai._FOLLOWS and q[-1] not in "เแโใไ" for q in parts(w)), parts(w)


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


def test_analyze_line_small_tags():
    """The reader's box hugs a small tag: bold letters fill most of a white tag's middle, and a filled tag's own
    colour is not what lies beside it. Both must still come out as the tag's colour behind its letters."""
    from gamelens.render import analyze_line, luminance

    img = cv2.imread(str(Path(__file__).parent / "data" / "kibo_tags.png"))  # from Azur Promilia's Kibo page
    white = analyze_line(img, [23, 35, 87, 18])  # 異生類·翼形族: bold blue letters on a white tag
    assert luminance(white.fg) < 160 and luminance(white.bg) > 200
    filled = analyze_line(img, [53, 10, 34, 17])  # 小型: white letters on a blue-grey tag
    assert luminance(filled.fg) > 200 and luminance(filled.bg) < 170


def test_analyze_line_light_labels_over_photo_mode_grass():
    """Photo mode's labels: thin light letters straight on the moving world. Dense kanji over sand (表情) were neither
    found nor measured - left standing, the translation in the water's blue; the end of テンプレート touches a blade of
    grass that runs out of the line and was left standing."""
    from gamelens.render import analyze_line, luminance

    img = cv2.imread(str(Path(__file__).parent / "data" / "photo_labels.png"))  # from Azur Promilia's photo mode
    face = analyze_line(img, [47, 30, 34, 22])  # 表情
    assert not face.plain and luminance(face.fg) > 170 and (face.mask > 0).sum() > 300
    tpl = analyze_line(img, [20, 165, 96, 25])  # テンプレート
    x0, y0, x1, y1 = tpl.mask_box
    to = tpl.mask[:, 100 - x0 : 113 - x0]  # where ト is
    assert (to > 0).sum() > 40


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


def test_friendly_errors_in_the_app_language_and_short():
    from gamelens.app import friendly_error
    from gamelens.i18n import set_lang

    try:
        set_lang("th")
        assert "ยังไม่พร้อม" in friendly_error("ไม่สามารถเชื่อมต่อ server ได้ (http://127.0.0.1:9960): [WinError 10061] refused")
        assert "429" in friendly_error("server HTTP 503: translation failed: google_free: Google (free) HTTP 429 (rate limited)")
        assert "โควตา" in friendly_error("server HTTP 503: translation failed: deepl: DeepL HTTP 456 (quota exceeded): x")
        set_lang("en")
        assert "not ready" in friendly_error("Cannot reach the server (http://127.0.0.1:9960): [WinError 10061] refused")
        assert "quota" in friendly_error("server HTTP 503: translation failed: deepl: DeepL HTTP 456 (quota exceeded): x")
        assert len(friendly_error("x: " + "y" * 500)) < 110
    finally:
        set_lang("en")


def test_every_text_of_the_app_has_its_thai():
    """Every t("...") in the app has an entry in the Thai table, and its {values} match."""
    import ast
    import re
    from pathlib import Path

    from gamelens.i18n import TH
    from gamelens.profile import ENGINE_INFO, GAME_LANGS, KIND_LABELS

    used = set()
    for f in Path("gamelens").glob("*.py"):
        for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "t" and node.args
                    and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)):
                used.add(node.args[0].value)
    names_shown = {"Local translation model (recommended)", "Google — free, no sign-up"}  # the others are brand names
    used |= names_shown | {e["desc"] for e in ENGINE_INFO} | {n for _, n in GAME_LANGS} | set(KIND_LABELS.values())
    missing = sorted(u for u in used if u not in TH)
    assert missing == [], missing
    for en, th in TH.items():
        assert set(re.findall(r"\{(\w+)", en)) == set(re.findall(r"\{(\w+)", th)), en


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
    centred = paragraphs([{"id": "a", "bbox": [300, 500, 400, 24], "text": "ステータスはここでチェック！"},
                          {"id": "b", "bbox": [200, 532, 600, 24], "text": "みんな唯一無二の能力を持ってるのだ。"}])
    assert [len(g) for g in centred] == [2]
    # labels of a table, stacked as tightly as running text, stay apart (天性 over 愉快·優良, 攻撃 over ブレイク値)
    table = paragraphs([{"id": "a", "bbox": [670, 523, 24, 13], "text": "攻撃"}, {"id": "b", "bbox": [670, 545, 59, 13], "text": "ブレイク値"},
                        {"id": "c", "bbox": [670, 567, 39, 14], "text": "EX強度"}])
    assert [len(g) for g in table] == [1, 1, 1]
    # a label under an icon wraps after a few letters in its narrow tab (bag: キボのタ / マゴ, クエスト / アイテム);
    # the next tab's label further down stays apart
    tabs = paragraphs([{"id": "a", "bbox": [155, 672, 57, 16], "text": "キボのタ"}, {"id": "b", "bbox": [169, 691, 30, 15], "text": "マゴ"},
                       {"id": "c", "bbox": [155, 773, 58, 15], "text": "クエスト"}, {"id": "d", "bbox": [155, 792, 58, 15], "text": "アイテム"},
                       {"id": "e", "bbox": [169, 472, 30, 16], "text": "装備"}])
    assert sorted("".join(l["text"] for l in g) for g in tabs) == ["キボのタマゴ", "クエストアイテム", "装備"]
    # a card's text set so tightly that the reader's boxes overlap and come out in different heights (Kibo factors)
    card = paragraphs([{"id": "a", "bbox": [387, 489, 129, 13], "text": "エネミーがブレイクした"},
                       {"id": "c", "bbox": [387, 514, 80, 15], "text": "75%增加する。"},
                       {"id": "b", "bbox": [387, 498, 135, 19], "text": "時、自身のEXダメージが"},
                       {"id": "d", "bbox": [575, 496, 141, 15], "text": "自身の攻撃力が4%上昇し、"},
                       {"id": "e", "bbox": [575, 507, 102, 15], "text": "HPが1%増加する。"}])
    assert [[l["id"] for l in g] for g in card] == [["a", "b", "c"], ["d", "e"]]

    img = np.full((420, 600, 3), (238, 233, 228), np.uint8)
    for i, w in enumerate((380, 340, 300)):
        cv2.putText(img, "x" * (w // 9), (80, 117 + i * 27), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (90, 80, 70), 1, cv2.LINE_AA)  # (as wide as its box, as text read off a screen is)
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


def test_choices_are_laid_over_their_own_buttons():
    from gamelens.screen import choice_items, refresh_patch

    pic = np.full((200, 400, 3), (200, 180, 170), np.uint8)
    lines = []
    for i, (y, text) in enumerate(((40, "Lend me a hand?"), (110, "Blast magic?"))):
        cv2.rectangle(pic, (20, y - 10), (380, y + 34), (240, 238, 236), -1)
        cv2.putText(pic, text, (60, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (110, 100, 90), 2, cv2.LINE_AA)
        lines.append({"id": str(i), "bbox": [56, y, 210, 28], "text": text})
    items = choice_items(pic, (500, 300, 400, 200), lines, {"0": "ช่วยหน่อยได้ไหม", "1": "เวทระเบิดเหรอ"}, 1.0)
    assert [it.text for it in items] == ["ช่วยหน่อยได้ไหม", "เวทระเบิดเหรอ"]
    assert all(it.rect[0] >= 500 and it.rect[1] >= 300 for it in items)  # placed in the game's window, not the crop
    assert items[0].rect[1] < items[1].rect[1] and items[0].patch_pos == (500, 300)
    same = items[0].patch
    refresh_patch(items[0], pic)
    assert items[0].patch is same  # nothing moved behind the buttons


def test_polite_particles_follow_the_speaker():
    from gamelens.speech import fit_speaker, gender_of, genders

    known = genders({"characters": [{"src": "シンフォーリア", "tl": "Symphoria", "gender": "female", "aliases": []},
                                    {"src": "イェクウ", "tl": "Yekuu", "gender": "male", "aliases": ["エクウ"]},
                                    {"src": "ルオチン", "tl": "Luoqing", "gender": ""}]})
    assert gender_of("シンフォーリア", known) == "female" and gender_of("エクウ", known) == "male"
    assert gender_of("ルオチン", known) == "" and gender_of("", known) == ""
    she = lambda s: fit_speaker(s, "female")
    assert she("ยังลุกได้ไหมครับ?") == "ยังลุกได้ไหมคะ?"
    assert she("ไม่ครับ นี่มัน……") == "ไม่ค่ะ นี่มัน……"
    assert she("ระวังนะครับ!") == "ระวังนะคะ!" and she("ขอบคุณครับผม") == "ขอบคุณค่ะ"
    assert she("คุณ Starborn ครับ ร่างกายเป็นอย่างไรบ้างครับ?") == "คุณ Starborn คะ ร่างกายเป็นอย่างไรบ้างคะ?"
    assert she("แล้ว ผมสนใจเรื่องนี้") == "แล้ว ฉันสนใจเรื่องนี้" and she("ผมยาวสวยจัง") == "ผมยาวสวยจัง"
    he = lambda s: fit_speaker(s, "male")
    assert he("ได้คะแนนเยอะค่ะ") == "ได้คะแนนเยอะครับ" and he("จริงเหรอคะ?") == "จริงเหรอครับ?"
    assert fit_speaker("ไปกันครับ", "") == "ไปกันครับ"  # nobody known: as translated
    me = lambda s: fit_speaker(s, "neutral")  # the player's own character: no particles at all
    assert me("ช่วยหน่อยได้ไหมครับ?") == "ช่วยหน่อยได้ไหม?" and me("ไม่ครับ นี่มัน……") == "ไม่ นี่มัน……"
    assert me("ขอบคุณครับผม") == "ขอบคุณ" and me("ครับ") == "อืม" and me("ระวังนะคะ!") == "ระวังนะ!"
    assert me("ผมสนใจเรื่องนี้") == "ฉันสนใจเรื่องนี้" and me("ได้คะแนนเยอะ") == "ได้คะแนนเยอะ"


def test_menu_labels_under_the_dialogue_region_are_told_from_dialogue():
    from gamelens.screen import looks_like_labels

    row = lambda x, t, y=100, h=14: {"bbox": [x, y, 14 * len(t), h], "text": t}
    menu = [row(60, "・ヘルプ"), row(190, "バッグ"), row(320, "星源の庭"), row(450, "実績")]
    assert looks_like_labels(menu)
    split = [row(100, "名付けて"), row(170, "ベタベタ弾！")]  # one row of dialogue read in two pieces
    assert not looks_like_labels(split)
    assert not looks_like_labels([row(100, "そうだな。"), row(100, "皆を集めてくれ。", y=140)])


def test_local_model_size_follows_the_graphics_card():
    from gamelens.services import HYMT, ServerManager, pick_hymt

    assert pick_hymt("auto", 16) == ("7b", True) and pick_hymt("auto", 12) == ("7b", True)
    assert pick_hymt("auto", 8) == ("1.8b", True) and pick_hymt("auto", 4) == ("1.8b", True)
    assert pick_hymt("auto", 0) == ("1.8b", False) and pick_hymt("auto", 1.5) == ("1.8b", False)  # no card: CPU
    assert pick_hymt("7b", 6) == ("7b", True) and pick_hymt("1.8b", 16) == ("1.8b", True)  # chosen by hand
    assert set(HYMT) == {"7b", "1.8b"} and all(f.endswith(".gguf") for _, f, _ in HYMT.values())
    from gamelens.profile import ROOT

    m = ServerManager(ROOT, ROOT / "logs", ROOT / "models", "1.8b")
    assert m.model_name("hymt").startswith("Hy-MT2 1.8B") and m.model_name("local") == "Gemma 26B"


def test_choosing_another_model_size_in_the_window():
    from gamelens.profile import ROOT
    from gamelens.services import ServerManager

    m = ServerManager(ROOT, ROOT / "logs", ROOT / "models", "auto")
    m.vram, m.local, m.hymt, m.hymt_gpu = 8.0, "1.8b", "1.8b", True  # as on an 8 GB card
    assert m.set_hymt("1.8b") is False  # what auto picked already
    assert m.set_hymt("7b") is True and m.hymt == "7b" and m.model_name("hymt").startswith("Hy-MT2 7B")
    assert m.set_hymt("auto") is True and m.hymt == "1.8b"


def test_index_translate_is_a_local_engine_that_downloads_itself():
    from gamelens.profile import ROOT
    from gamelens.services import INDEX_FILE, MODELS, ServerManager, pick_local

    assert "index" in MODELS and MODELS["index"][0] == 9944
    assert INDEX_FILE[0] in " ".join(MODELS["index"][1]) and INDEX_FILE[1].endswith(INDEX_FILE[0])
    # the default: Index-Translate where it fits next to a game, Hy-MT2 1.8B below that
    assert pick_local("auto", 16) == ("index", True) and pick_local("auto", 8) == ("1.8b", True)
    assert pick_local("auto", 0) == ("1.8b", False) and pick_local("7b", 16) == ("7b", True)
    m = ServerManager(ROOT, ROOT / "logs", ROOT / "models", "auto")
    m.vram, m.local, m.hymt, m.hymt_gpu = 16.0, "index", "7b", True
    assert m.engine_for("hymt") == "index" and m.engine_for("deepl") == "deepl"
    assert m.model_name("index").startswith("Index-Translate 9B")
    assert m.set_hymt("1.8b") and m.engine_for("hymt") == "hymt" and m.hymt == "1.8b"
    assert m.set_hymt("index") and m.engine_for("hymt") == "index"
    from gamelens.profile import Profile
    p = Profile(engine="hymt")
    p.live_engine = "index"
    import dataclasses, json
    d = dataclasses.asdict(p); d.pop("path"); d.pop("live_engine")
    assert "live_engine" not in json.dumps(d)


def test_each_game_gets_its_own_profile(tmp_path):
    from gamelens.profile import Profile, find_profile, guess_lang, new_profile

    assert guess_lang("アズールプロミリア") == "ja" and guess_lang("메이플스토리") == "ko"
    assert guess_lang("原神") == "zh" and guess_lang("原神 開發中") == "zh-tw" and guess_lang("Game", "ko") == "ko"
    base = Profile(name="Azur", series="azur", exe="AzurPromiliaJP.exe", engine="hymt", font_scale=0.8,
                   path=str(tmp_path / "azur.json"))
    base.save()
    assert find_profile("azurpromiliajp.exe", tmp_path) == tmp_path / "azur.json"  # case does not matter
    assert find_profile("Other.exe", tmp_path) is None
    p = new_profile("BlueArchiveKR.exe", "블루 아카이브", base, tmp_path)
    assert p.src_lang == "ko" and p.series == "bluearchivekr" and p.regions == [] and p.font_scale == 0.8
    assert find_profile("BlueArchiveKR.exe", tmp_path) == tmp_path / "bluearchivekr.json"
    again = new_profile("BlueArchiveKR.exe", "", base, tmp_path)
    assert again.path.endswith("bluearchivekr-2.json") and again.src_lang == base.src_lang


def test_a_translated_screen_stays_while_something_else_moves():
    from gamelens.screen import text_marks, text_still_there

    rng = np.random.default_rng(3)
    pic = cv2.resize(rng.integers(40, 90, (30, 50, 3), np.uint8), (1000, 600), interpolation=cv2.INTER_CUBIC)
    boxes = []
    for i, text in enumerate(["HP 1217", "ATK 24", "Look, you can see its power now!"]):
        y = 100 + 60 * i
        cv2.putText(pic, text, (500, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (240, 240, 240), 2, cv2.LINE_AA)
        boxes.append([496, y, 30 * len(text) // 2 + 40, 30])
    marks = text_marks(pic, boxes)
    moving = pic.copy()
    moving[50:500, 50:400] = np.roll(pic[50:500, 50:400], 12, axis=0)  # a character bobbing beside the text
    assert text_still_there(pic, marks) and text_still_there(moving, marks)
    assert text_still_there(cv2.convertScaleAbs(pic, alpha=1.1, beta=5), marks)  # a light pulse
    tip = pic.copy()
    tip[220:250, 490:1000] = 60  # the long line (a tip) was replaced
    assert not text_still_there(tip, marks)
    assert not text_still_there(cv2.resize(rng.integers(40, 90, (30, 50, 3), np.uint8), (1000, 600)), marks)  # another page


def test_screen_text_changed_or_gone():
    from gamelens.screen import text_marks, text_state

    pic = np.full((600, 800, 3), 70, np.uint8)
    boxes = []
    texts = ["Attack power", "Defence power", "Critical rate", "Break value", "EX strength", "Tap to go on now"]
    for i, text in enumerate(texts):  # a menu of labels, and a tip (the last line)
        cv2.putText(pic, text, (60, 60 + 70 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (240, 240, 240), 2, cv2.LINE_AA)
        boxes.append([56, 38 + 70 * i, 17 * len(text) + 10, 30])
    marks = text_marks(pic, boxes, texts)
    assert text_state(pic, marks) == "same"
    tip = pic.copy()
    tip[38 + 350 - 4 : 38 + 350 + 34, :] = 70  # the tip went (the next one comes)
    assert text_state(tip, marks) == "changed"
    assert text_state(np.full_like(pic, 70), marks) == "gone"  # another screen


def test_text_standing_still_while_a_character_moves_counts_as_still():
    from gamelens.watcher import stroke_change

    rng = np.random.default_rng(5)
    def frame(shift):
        pic = np.full((540, 960, 3), (120, 160, 110), np.uint8)
        cv2.circle(pic, (480, 250 + shift), 90, (235, 235, 230), -1)  # a soft round character bobbing
        cv2.circle(pic, (470, 240 + shift), 20, (60, 60, 60), 3)
        for i, t in enumerate(["HP 1217", "ATK 24", "DEF 30", "CRIT 29.9%"]):
            cv2.putText(pic, t, (300, 400 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (250, 250, 250), 1, cv2.LINE_AA)
        return pic
    a, b = stroke_map(frame(0), "any"), stroke_map(frame(6), "any")
    from gamelens.app import STILL

    assert stroke_change(a, b) <= STILL  # the menu's text stands still: it can be read
    grass = cv2.resize(rng.integers(0, 255, (60, 100, 3), np.uint8), (960, 540), interpolation=cv2.INTER_NEAREST)
    grass2 = cv2.resize(rng.integers(0, 255, (60, 100, 3), np.uint8), (960, 540), interpolation=cv2.INTER_NEAREST)
    assert stroke_change(stroke_map(grass, "any"), stroke_map(grass2, "any")) > STILL  # the world moving: not still


def test_the_text_reader_files_for_each_game_language():
    from gamelens.services import ocr_files

    ja = [p.name for p, _, _ in ocr_files("ja")]
    assert "PP-OCRv6_det_small.onnx" in ja and "PP-OCRv6_rec_medium.onnx" in ja
    assert [p.name for p, _, _ in ocr_files("zh-tw")] == ja  # one multilingual model for Japanese and Chinese
    ko = [p.name for p, _, _ in ocr_files("ko")]
    assert "korean_PP-OCRv5_rec_mobile.onnx" in ko and "PP-OCRv6_rec_medium.onnx" not in ko
    assert all(u.startswith("https://") and len(s) == 64 for _, u, s in ocr_files("ja"))


def test_a_correction_goes_into_the_glossary_file(tmp_path):
    from gamelens.glossary import add_correction

    path = tmp_path / "game.th.json"
    path.write_text(json.dumps({"_about": "x", "characters": [{"src": "アニス", "aliases": ["アニ"], "tl": "Anis",
                                                                "gender": "", "role": "", "source": "fan"}],
                                "terms": [{"src": "マウント", "reading": "", "tl": "ติดตั้ง", "note": "", "source": "x"}]},
                               ensure_ascii=False), encoding="utf-8")
    add_correction(path, "マウント", "พาหนะ", "term")  # replaces the term
    add_correction(path, "アニ", "Anais", "female")  # an alias: corrects that character
    add_correction(path, "キャベッコ", "Cabbecco", "neutral")  # a new name
    d = json.loads(path.read_text(encoding="utf-8"))
    assert [x["tl"] for x in d["terms"]] == ["พาหนะ"]
    assert d["characters"][0]["tl"] == "Anais" and d["characters"][0]["gender"] == "female"
    assert d["characters"][1] == {"src": "キャベッコ", "aliases": [], "tl": "Cabbecco", "gender": "", "role": "", "source": "user"}
    add_correction(path, "キャベッコ", "แคเบ็กโกะ", "term")  # now a term: no longer a character
    d = json.loads(path.read_text(encoding="utf-8"))
    assert [c["src"] for c in d["characters"]] == ["アニス"] and d["terms"][-1]["tl"] == "แคเบ็กโกะ"
    assert path.read_text(encoding="utf-8").count("\n") == 10  # one entry per line, as people edit it
    add_correction(tmp_path / "new.en.json", "総合評価", "Overall", "term")  # no file yet
    assert json.loads((tmp_path / "new.en.json").read_text(encoding="utf-8"))["terms"][0]["tl"] == "Overall"


def test_menu_text_is_logged_once_and_items_know_their_line(tmp_path):
    from gamelens.history import HistoryLog
    from gamelens.screen import screen_items

    h = HistoryLog(tmp_path, "g")
    assert h.add("ui", "整理", "จัดระเบียบ", "index") and h.add("ui", "整理", "จัดระเบียบ", "index") is None
    assert h.add("dialog", "はい", "ครับ", "index") and h.add("dialog", "はい", "ครับ", "index")  # dialogue repeats
    img = np.full((200, 400, 3), (225, 215, 205), np.uint8)
    cv2.putText(img, "Label", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (90, 80, 70), 1, cv2.LINE_AA)
    cv2.putText(img, "Other", (40, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (90, 80, 70), 1, cv2.LINE_AA)
    groups = [[{"id": "a", "bbox": [38, 44, 56, 20], "text": "ラベル"}], [{"id": "b", "bbox": [38, 134, 56, 20], "text": "他"}]]
    items = screen_items(img, groups, {"b": "อื่น"}, 0.9)
    assert [it._id for it in items] == ["b"]


def test_a_translation_goes_only_when_its_own_text_changed():
    from gamelens.screen import marks_unchanged, text_marks

    img = np.full((200, 400, 3), (225, 215, 205), np.uint8)
    cv2.putText(img, "Label", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (90, 80, 70), 1, cv2.LINE_AA)
    cv2.putText(img, "Other", (40, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (90, 80, 70), 1, cv2.LINE_AA)
    top, bottom = text_marks(img, [[38, 44, 60, 20]]), text_marks(img, [[38, 134, 60, 20]])
    popup = img.copy()
    cv2.rectangle(popup, (20, 120), (300, 190), (245, 245, 240), -1)  # a pop-up over the lower one
    assert marks_unchanged(popup, top) and not marks_unchanged(popup, bottom)


def test_a_key_button_takes_the_next_key():
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication

    from gamelens.panel import KeyButton

    QApplication.instance() or QApplication([])
    got = []
    b = KeyButton("screen", "F9")
    b.changed.connect(lambda n, s: got.append((n, s)))
    b.show()
    b.click()
    QTest.keyClick(b, Qt.Key.Key_Q)  # a letter alone: the game needs it
    QTest.keyClick(b, Qt.Key.Key_Q, Qt.KeyboardModifier.ControlModifier)
    peek = KeyButton("peek", "F8", plain=True)
    peek.changed.connect(lambda n, s: got.append((n, s)))
    peek.show()
    peek.click()
    QTest.keyClick(peek, Qt.Key.Key_F11, Qt.KeyboardModifier.AltModifier)  # held, read as one key
    b.click()
    QTest.keyClick(b, Qt.Key.Key_Escape)
    assert got == [("screen", "Ctrl+Q"), ("peek", "F11")] and b.text() == "Ctrl+Q"


def test_a_font_that_is_not_there_is_said_and_another_used():
    from PySide6.QtWidgets import QApplication

    from gamelens.app import load_fonts
    from gamelens.panel import Panel, usable_font

    QApplication.instance() or QApplication([])
    load_fonts()
    assert usable_font("Kanit", "th") == "Kanit" and usable_font("No Such Font", "th") == "Bai Jamjuree"
    p = Panel({"toggle": "Ctrl+F8", "screen": "F9", "regions": "F10", "history": "F7"}, "F8")
    closed = []
    p.closed.connect(lambda: closed.append(1))
    p.load("hymt", 0.9, "th", True, font_family="No Such Font")
    assert not p.font_note.isHidden() and "Bai Jamjuree" in p.font_note.text()
    assert p.family.currentFont().family() == "Bai Jamjuree"  # what is used is what the list shows
    p.load("hymt", 0.9, "th", True, font_family="Kanit")
    assert p.font_note.isHidden() and p.family.currentFont().family() == "Kanit"
    p.show()
    p.close()
    assert closed == [1] and not p.isVisible()


def test_a_screen_is_translated_in_parts_shown_as_they_come():
    from gamelens.app import Controller
    from gamelens.profile import Profile

    img = np.full((300, 400, 3), (225, 215, 205), np.uint8)
    groups = []
    for i in range(10):
        cv2.putText(img, "Label", (40, 25 + i * 27), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 80, 70), 1, cv2.LINE_AA)
        groups.append([{"id": f"l{i}", "bbox": [38, 12 + i * 27, 50, 18], "text": f"ラベル{i}"}])
    reqs = [{"id": g[0]["id"], "src": g[0]["text"], "kind": "ui"} for g in groups]
    asked, shown = [], []

    class Api:
        def translate_lines(self, part, **kw):
            asked.append([r["id"] for r in part])
            return {"lines": {r["id"]: "ป้าย" for r in part}, "engine": "index"}

    class Fake:
        PART = Controller.PART
        translate_in_parts = Controller.translate_in_parts
        api, profile, screen_gen = Api(), Profile(), 7
        bridge = type("B", (), {"screen_part": type("S", (), {"emit": staticmethod(lambda gen, items: shown.append(len(items)))})()})()

    res = Fake().translate_in_parts(reqs, groups, img, [g[0]["bbox"] for g in groups], 7)
    assert asked == [["l0", "l1", "l2", "l3"], ["l4", "l5", "l6", "l7"], ["l8", "l9"]]  # from the top, a part at a time
    assert shown == [4, 8] and len(res["lines"]) == 10  # (the last part comes with the whole screen)


def test_labels_stay_on_their_panel_and_off_each_other():
    from gamelens.overlay import Item
    from gamelens.render import analyze_line
    from gamelens.screen import keep_apart, room_for

    # a short centred label near the right edge of a light card: its long translation moves left, inside the card
    img = np.full((120, 600, 3), (120, 130, 110), np.uint8)
    cv2.rectangle(img, (100, 20), (500, 100), (240, 238, 232), -1)
    cv2.putText(img, "abc", (440, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (90, 80, 70), 1, cv2.LINE_AA)
    look = analyze_line(img, [438, 48, 40, 24])
    x, y, w, h = room_for((438, 40, 44, 40), "center", "กำลังดำเนินการอยู่นะ", 20, 600, [438, 48, 40, 24], [[438, 48, 40, 24]], look)
    assert w > 100 and x >= 100 and x + w <= 500 and x < 438
    # left-aligned with the card's empty middle on its left (a status at the card's right edge): also to the left
    x, y, w, h = room_for((438, 40, 44, 40), "left", "กำลังดำเนินการอยู่นะ", 20, 600, [438, 48, 40, 24], [[438, 48, 40, 24]], look)
    assert w > 100 and x < 438 and x >= 100
    # a skill's name with its icon right beside it on the left: never over the icon
    cv2.circle(img, (422, 60), 12, (90, 160, 120), -1)
    look = analyze_line(img, [438, 48, 40, 24])
    x, y, w, h = room_for((438, 40, 44, 40), "left", "กำลังดำเนินการอยู่นะ", 20, 600, [438, 48, 40, 24], [[438, 48, 40, 24]], look)
    assert x == 438
    # two translations of one row given the same room: they share it
    a, b = Item(rect=(100, 10, 400, 40), text="ก" * 30), Item(rect=(300, 10, 250, 40), text="ข" * 20)
    a._boxes, b._boxes = [(100, 14, 120, 30)], [(420, 14, 60, 30)]
    keep_apart([a, b])
    assert a.rect[0] + a.rect[2] <= b.rect[0] and a.rect[2] >= 120 and b.rect[0] + b.rect[2] == 550


def test_text_set_at_an_angle_is_drawn_at_that_angle_in_its_own_size():
    from gamelens.screen import screen_items

    img = np.full((300, 600, 3), (233, 225, 205), np.uint8)
    layer = np.zeros((60, 260, 3), np.uint8)
    cv2.putText(layer, "TITLE", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3, cv2.LINE_AA)
    m = cv2.getRotationMatrix2D((130, 30), -12, 1.0)  # (12 degrees clockwise on screen)
    turned = cv2.warpAffine(layer, m, (260, 120), borderValue=(0, 0, 0))
    ys, xs = np.nonzero(turned[:, :, 0] > 100)
    img[100 + ys, 150 + xs] = (70, 86, 96)
    box = [150 + int(xs.min()), 100 + int(ys.min()), int(xs.max() - xs.min()), int(ys.max() - ys.min())]
    line = {"id": "a", "bbox": box, "text": "題名", "tilt": 12.0, "tw": 170, "th": 44}
    flat = dict(line, id="b")
    for k in ("tilt", "tw", "th"):
        flat.pop(k)
    tilted = screen_items(img, [[line]], {"a": "หัวข้อ"}, 1.0)[0]
    plain = screen_items(img, [[flat]], {"b": "หัวข้อ"}, 1.0)[0]
    assert tilted.angle == 12.0 and plain.angle == 0
    assert tilted.max_px < 0.9 * plain.max_px  # the tall box of the slanted line is not its letters' height
    cx = tilted.rect[0] + tilted.rect[2] / 2
    assert abs(cx - (box[0] + box[2] / 2)) <= 2 and tilted.align == "center"


def test_light_text_over_the_scene_is_found_whole():
    """A hint drawn straight over the 3D world (grass light and dark behind it, a coloured word, a key icon): all of
    the line is found and erased, its letters measured light, its height the letters' (not the box's)."""
    from gamelens.render import analyze_line, luminance

    img = cv2.imread(str(Path(__file__).parent / "data" / "scene_hint.png"))  # from Azur Promilia, 1080p
    a = analyze_line(img, [54, 17, 462, 33])  # を押して、キボのスキルを使って謎解きする
    assert not a.plain and luminance(a.fg) > 200
    assert a.glyph_h <= 26  # the letters are ~20 px: grass reaching through the box is not counted
    cols = (a.mask > 0).any(0)
    x0 = a.mask_box[0]
    # letters are found from the first (the coloured 押して) to the last (謎解きする), not only in the middle
    assert cols[90 - x0 : 140 - x0].any() and cols[470 - x0 : 510 - x0].any()


def test_new_text_is_told_from_a_screen_that_moves_by_itself():
    from gamelens.watcher import SteadyTiles

    rng = np.random.default_rng(1)
    base = np.zeros((90, 160), bool)
    base[10:14, 20:60] = True  # the menu's text
    looks = []
    for i in range(40):  # grass waving in the lower half: different strokes every look
        f = base.copy()
        f[50:90] = rng.random((40, 160)) < 0.2
        looks.append(f)
    st = SteadyTiles(looks[0])
    assert max(st.feed(f) for f in looks[1:]) <= 0.5
    popup = looks[-1].copy()
    popup[20:45, 40:120] = rng.random((25, 80)) < 0.3  # a pop-up opens over the still part, then stands
    vals = []
    for _ in range(25):
        g = popup.copy()
        g[50:90] = rng.random((40, 160)) < 0.2
        vals.append(st.feed(g))
    assert vals[-1] > 0.5


def test_letters_at_the_end_of_a_line_on_a_gradient_are_found():
    """News list items on a panel whose shade changes along the line: the first, dense letters (絆心) lie on the
    lighter end. They are found (and so erased), not taken for a patch of background running out of the area."""
    from gamelens.render import analyze_line

    img = cv2.imread(str(Path(__file__).parent / "data" / "news_list.png"))  # from Azur Promilia's news
    a = analyze_line(img, [47, 18, 184, 18])  # 絆心の観測「花霞を胸に」
    assert a.box[0] <= 49  # starts where the line starts, not after 絆心
    x0 = a.mask_box[0]
    assert (a.mask[:, 47 - x0 : 80 - x0] > 0).sum() > 120  # 絆心 is covered


def test_a_title_read_differently_each_look_does_not_reread_the_screen_forever():
    """A list whose last row is half under its fading edge: each look reads those titles differently (真紅, 紀行,
    ダンハレイ紀行, then nothing), and every look had set off a whole new reading and translation, every few seconds."""
    from types import SimpleNamespace

    from gamelens.app import Controller
    from gamelens.screen import known

    assert known("紀行", ["ダンハレイ紀行"]) and known("ダンハレイ紀行", ["紀行"]) and known("ダンハレイ紀行", ["ダンハレイ紀行。"])
    assert not known("真紅", ["ダンハレイ紀行", "所持合計数"]) and not known("真紅", [])

    looks = iter([["真紅", "ダンハレイ紀行"], ["紀行"], ["真紅"], ["新しい報酬"]])
    sent = []
    c = SimpleNamespace(profile=SimpleNamespace(src_lang="ja"), screen_read=["ダンハレイ紀行"], rechecking=True,
                        api=SimpleNamespace(ocr_lines=lambda *a, **k: {"lines": [
                            {"text": t, "bbox": [0, 0, 40, 16], "conf": 0.99} for t in next(looks)]}),
                        bridge=SimpleNamespace(screen_more=SimpleNamespace(emit=lambda gen, new: sent.append(new))))
    frame = np.zeros((40, 40, 3), np.uint8)
    Controller.job_recheck(c, frame, 1)
    assert sent == [["真紅"]] and c.rechecking is False
    c.screen_read += ["真紅"]  # (what set off a reading is known after it, read again or not: see on_screen_more)
    Controller.job_recheck(c, frame, 1)
    Controller.job_recheck(c, frame, 1)
    assert sent == [["真紅"]]  # a piece of a known title, a title already tried: nothing new
    Controller.job_recheck(c, frame, 1)
    assert sent == [["真紅"], ["新しい報酬"]]  # text that really is new still counts

    # what set off a reading is remembered for this screen, and a screen is read again so at most twice
    started = []
    m = SimpleNamespace(screen_gen=5, screen_ref=object(), screen_busy=False, screen_auto=True, screen_menu=True,
                        screen_tried=[], screen_more_n=0, screen_marks=[],
                        cap=SimpleNamespace(client_frame=lambda: (frame, None)),
                        start_screen=lambda *a, **k: started.append(k))
    Controller.on_screen_more(m, 5, ["真紅"])
    assert m.screen_tried == ["真紅"] and m.screen_more_n == 1 and started[-1]["more"] is True


def test_auto_mode_backs_off_on_a_slowly_changing_scene():
    """Standing in the world where nothing is to be read: a scene that changes slowly (light, water, an idle
    character) had auto mode read the whole screen every 0.75 s for many minutes. Only a sudden change (another
    screen, the camera turned) ends the longer and longer wait."""
    from collections import deque
    from types import SimpleNamespace

    from gamelens.app import Controller
    from gamelens.watcher import ScreenCalm

    rng = np.random.default_rng(3)
    base = rng.integers(40, 200, (270, 480, 3)).astype(np.float32)
    c = SimpleNamespace(auto_prev=None, auto_calm=ScreenCalm(), auto_thumbs=deque(), auto_empties=3, auto_still_since=0.0,
                        screen_busy=True, screen_ref=None, health=True, auto_done=None)
    now = 0.0
    for i in range(60):  # 6 s of a picture brightening slowly
        now += 0.1
        Controller.auto_screen_tick(c, np.clip(base + 0.4 * i, 0, 255).astype(np.uint8), now)
    assert c.auto_empties == 3
    now += 0.1
    Controller.auto_screen_tick(c, (base * 0.3).astype(np.uint8), now)  # a menu darkens the screen
    assert c.auto_empties == 0


def test_auto_mode_reads_a_still_empty_world_only_now_and_then():
    """The player stands in the world for a minute, nothing to read: the wait after empty readings counted from the
    moment the screen went still, so after 8 s every tick read the screen again (77 readings a minute, until the
    count of empty readings overflowed 2 ** n)."""
    from collections import deque
    from types import SimpleNamespace

    from gamelens.app import Controller
    from gamelens.watcher import ScreenCalm, auto_pause

    assert auto_pause(set(), 5000) == 8.0  # (no overflow, however many)
    rng = np.random.default_rng(4)
    frame = rng.integers(0, 255, (270, 480, 3), dtype=np.uint8)
    reads = []

    def start_screen(f, ref, auto=True, **k):  # a reading that finds nothing (see on_screen_done)
        reads.append(c.now)
        c.auto_empties = min(c.auto_empties + 1, 8)
        c.auto_read_at = c.now

    c = SimpleNamespace(auto_prev=None, auto_calm=ScreenCalm(), auto_thumbs=deque(), auto_empties=0, auto_still_since=0.0,
                        auto_read_at=0.0, region_active=0.0, screen_busy=False, screen_ref=None, health=True, auto_done=None,
                        services=SimpleNamespace(state="ready"), profile=SimpleNamespace(regions=[]), watchers={},
                        start_screen=start_screen, now=0.0)
    for _ in range(600):  # a minute at 10 looks a second
        c.now += 0.1
        Controller.auto_screen_tick(c, frame, c.now)
    assert 4 <= len(reads) <= 12, len(reads)
