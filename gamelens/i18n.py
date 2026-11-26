"""The app's own words in the language the player picked: English (the default) or Thai.

t("English text", **values) gives the text in the current language; the English text is the key, so a missing
Thai entry shows the English one instead of nothing."""
from __future__ import annotations

LANGS = [("en", "English"), ("th", "ไทย")]
_lang = "en"


def set_lang(code: str):
    global _lang
    _lang = code if code in dict(LANGS) else "en"


def lang() -> str:
    return _lang


def t(text: str, **values) -> str:
    s = TH.get(text, text) if _lang == "th" else text
    return s.format(**values) if values else s


TH = {
    # --- main window -------------------------------------------------------------------------------------------
    "Game text translated into Thai, drawn over the game": "แปลข้อความในเกมเป็นภาษาไทย แสดงทับบนหน้าจอเกม",
    "Getting ready": "การเตรียมพร้อม",
    "Translation system": "ระบบแปลภาษา",
    "Game": "เกม",
    "Text regions in the game": "กรอบข้อความในเกม",
    "Start": "เปิด",
    "Choose game": "เลือกเกม",
    "Set regions": "ตั้งกรอบ",
    "Change": "เปลี่ยน",
    "Edit": "แก้ไข",
    "Load": "โหลด",
    "▶  Start translating": "▶  เริ่มการแปล",
    "■  Stop translating": "■  หยุดการแปล",
    "Translator": "เครื่องแปล",
    "Model": "รุ่นโมเดล",
    "Automatic, by graphics card": "อัตโนมัติตามการ์ดจอ",
    "Automatic — this computer: {label}": "อัตโนมัติ — เครื่องนี้: {label}",
    "Index-Translate 9B — most natural • ~5.5 GB graphics memory • 5.8 GB download":
        "Index-Translate 9B — แปลเป็นธรรมชาติที่สุด • ใช้หน่วยความจำการ์ดจอประมาณ 5.5 GB • ดาวน์โหลด 5.8 GB",
    "Hy-MT2 7B — ~5.5 GB graphics memory • 4.6 GB download":
        "Hy-MT2 7B — ใช้หน่วยความจำการ์ดจอประมาณ 5.5 GB • ดาวน์โหลด 4.6 GB",
    "Hy-MT2 1.8B — fast, ~2 GB graphics memory or CPU • lower quality • 1.1 GB download":
        "Hy-MT2 1.8B — รวดเร็ว ใช้หน่วยความจำการ์ดจอประมาณ 2 GB หรือ CPU • คุณภาพการแปลต่ำกว่า • ดาวน์โหลด 1.1 GB",
    "Sign up / get a key ↗": "สมัคร / ขอ key ↗",
    "Save key": "บันทึก key",
    "Test": "ทดสอบ",
    "Display": "การแสดงผล",
    "Text size": "ขนาดตัวอักษร",
    "Game language": "ภาษาในเกม",
    "Translate into": "แปลเป็น",
    "App language": "ภาษาของโปรแกรม",
    "Thai": "ภาษาไทย",
    "Show short status messages on the game screen": "แสดงข้อความสถานะบนหน้าจอเกม",
    "Switch the keyboard to English while in the game": "เปลี่ยนแป้นพิมพ์เป็นภาษาอังกฤษโดยอัตโนมัติขณะอยู่ในเกม",
    "A Thai keyboard makes the game show its key hints as boxes □":
        "แป้นพิมพ์ภาษาไทยทำให้ป้ายปุ่มในเกมแสดงเป็นรูปสี่เหลี่ยม □",
    "Translate menus automatically when the screen is still":
        "แปลเมนูและภารกิจโดยอัตโนมัติเมื่อหน้าจอนิ่ง (ไม่ต้องกด F9)",
    "When the whole screen stands still, e.g. a menu is open, all its text is translated.\n"
    "This uses the translator more: mind the quota of a paid API.":
        "เมื่อหน้าจอหยุดนิ่ง เช่น ขณะเปิดเมนู ระบบจะแปลข้อความทั้งหน้าจอให้\n"
        "การตั้งค่านี้ใช้เครื่องแปลมากขึ้น โปรดระวังโควตาหากใช้ API ที่มีค่าใช้จ่าย",
    "Not possible with this game: it runs as administrator. Set Windows to keep an English keyboard for it "
    "(see README)":
        "ใช้ไม่ได้กับเกมนี้ เนื่องจากเกมทำงานในสิทธิ์ผู้ดูแลระบบ โปรดตั้งค่า Windows ให้ใช้แป้นพิมพ์ภาษาอังกฤษกับเกมนี้ "
        "(ดูวิธีใน README)",
    "Keys while playing": "ปุ่มลัดระหว่างเล่น",
    "hold: show the original": "กดค้าง = แสดงต้นฉบับ",
    "start / stop": "หยุด / เริ่มการแปล",
    "translate the screen": "แปลทั้งหน้าจอ (เมนู)",
    "set regions": "ตั้งกรอบข้อความ",
    "history": "ประวัติการแปล",
    "Translation history": "ประวัติการแปล",
    "Desktop icon": "สร้างไอคอนบนเดสก์ท็อป",
    "Quit": "ปิดโปรแกรม",
    "set ({value})": "ตั้งค่าแล้ว ({value})",
    "paste the key here": "วาง key ที่นี่",
    # --- engines -----------------------------------------------------------------------------------------------
    "Local translation model (recommended)": "โมเดลแปลภาษาในเครื่อง (แนะนำ)",
    "Free, works offline • model picked by your graphics card": "ฟรี ไม่ต้องใช้อินเทอร์เน็ต • เลือกรุ่นตามการ์ดจอโดยอัตโนมัติ",
    "Google — free, no sign-up": "Google — ฟรี ไม่ต้องสมัครสมาชิก",
    "Free • no graphics card needed": "ฟรี • ไม่ใช้การ์ดจอ",
    "Free 500,000 characters a month • sign-up needed": "ฟรี 500,000 ตัวอักษรต่อเดือน • ต้องสมัครสมาชิก",
    "Free 2,000,000 characters a month • sign-up needed": "ฟรี 2,000,000 ตัวอักษรต่อเดือน • ต้องสมัครสมาชิก",
    "Best translations • paid per use": "คุณภาพการแปลสูงที่สุด • คิดค่าบริการตามการใช้งาน",
    "Region (e.g. southeastasia)": "Region (เช่น southeastasia)",
    # --- kinds of regions, game languages -----------------------------------------------------------------------
    "Dialogue (text box)": "บทพูด (กล่องข้อความ)",
    "Speaker name": "ชื่อผู้พูด",
    "Subtitles while playing": "คำบรรยายระหว่างเล่น",
    "Answer choices": "ตัวเลือกคำตอบ",
    "Japanese": "ภาษาญี่ปุ่น",
    "Korean": "ภาษาเกาหลี",
    "Chinese (simplified)": "ภาษาจีนตัวย่อ",
    "Chinese (traditional: Taiwan / Hong Kong)": "ภาษาจีนตัวเต็ม (ไต้หวัน / ฮ่องกง)",
    "English": "ภาษาอังกฤษ",
    # --- status --------------------------------------------------------------------------------------------------
    "Ready — {name}": "พร้อมใช้งาน — {name}",
    "The translation system is not running": "ระบบแปลภาษายังไม่ทำงาน",
    "{name} is not loaded yet": "ยังไม่ได้โหลด {name}",
    "Enter the key for {name} (below)": "โปรดใส่ key ของ {name} (ช่องด้านล่าง)",
    "{name} is not ready": "{name} ยังไม่พร้อมใช้งาน",
    "No game window chosen yet": "ยังไม่ได้เลือกหน้าต่างเกม",
    "Waiting for the game… ({game}) • set the game to Borderless / Windowed":
        "กำลังรอเกม… ({game}) • โปรดตั้งเกมเป็นแบบ Borderless หรือ Windowed",
    "Open the game first, then go to a screen with dialogue": "โปรดเปิดเกมก่อน แล้วไปยังหน้าที่มีบทพูด",
    "Go to a screen with dialogue, then press Set regions": "โปรดไปยังหน้าที่มีบทพูดในเกม แล้วกด ตั้งกรอบ",
    "{n} regions: {kinds}": "{n} กรอบ: {kinds}",
    "Translating — you can switch to the game": "กำลังแปล — สามารถกลับไปเล่นเกมได้",
    "Stopped": "หยุดการแปลอยู่",
    "Complete the steps that are not green yet; translating then starts by itself":
        "โปรดดำเนินการในขั้นตอนที่ยังไม่เป็นสีเขียวให้ครบ ระบบจะเริ่มแปลโดยอัตโนมัติ",
    "Stop translating": "หยุดการแปล",
    "Start translating": "เริ่มการแปล",
    "The old model is still running outside game-lens — close that program first, then press Start in step 1":
        "โมเดลเดิมยังทำงานอยู่นอก game-lens — โปรดปิดโปรแกรมนั้นก่อน แล้วกด เปิด ในขั้นตอนที่ 1",
    "Testing…": "กำลังทดสอบ…",
    "✓ Works: “{text}” ({ms} ms){extra}": "✓ ใช้งานได้: “{text}” ({ms} ms){extra}",
    "✗ {error}": "✗ {error}",
    "does not work": "ใช้งานไม่ได้",
    "{used:,} / {limit:,} characters used this month": "ใช้ไปแล้ว {used:,} / {limit:,} ตัวอักษรในเดือนนี้",
    "graphics card {gb} GB": "การ์ดจอ {gb} GB",
    "no graphics card": "ไม่มีการ์ดจอแยก",
    "on the CPU: no graphics card": "ใช้ CPU เนื่องจากไม่มีการ์ดจอแยก",
    "Starting the translation system…": "กำลังเปิดระบบแปลภาษา…",
    "Loading {name} (the first time takes a while)…": "กำลังโหลด {name} (ครั้งแรกอาจใช้เวลาสักครู่)…",
    "Ready": "พร้อมใช้งาน",
    "{name} stopped (see logs/{name}.log)": "{name} หยุดทำงาน (ดูรายละเอียดใน logs/{name}.log)",
    "The translation system did not start in time (see logs/server.log)":
        "ระบบแปลภาษาเริ่มทำงานไม่ทันเวลา (ดูรายละเอียดใน logs/server.log)",
    "Downloading the translation model {pct}% ({have} / {total} GB) — only the first time":
        "กำลังดาวน์โหลดโมเดลแปลภาษา {pct}% ({have} / {total} GB) — เฉพาะครั้งแรกเท่านั้น",
    "The model download failed ({error}) — open game-lens again to continue, or choose an online translator":
        "ดาวน์โหลดโมเดลไม่สำเร็จ ({error}) — โปรดเปิด game-lens อีกครั้งเพื่อดาวน์โหลดต่อ หรือเลือกเครื่องแปลออนไลน์",
    "game-lens's server was not found at {path}": "ไม่พบ server ของ game-lens ที่ {path}",
    "game-lens is not installed (.venv)": "game-lens ยังไม่ได้ติดตั้ง (.venv)",
    "llama (for the local models) was not found — install Llama for Windows or choose an online translator":
        "ไม่พบโปรแกรม llama สำหรับโมเดลในเครื่อง — โปรดติดตั้ง Llama for Windows หรือเลือกเครื่องแปลออนไลน์",
    # --- messages on the game screen and from the tray -------------------------------------------------------------
    "The translation system is not ready — open the game-lens window to see why":
        "ระบบแปลภาษายังไม่พร้อม — โปรดเปิดหน้าต่าง game-lens เพื่อตรวจสอบสถานะ",
    "The online translator is limited for now (429) — try another translator":
        "เครื่องแปลออนไลน์ถูกจำกัดการใช้งานชั่วคราว (429) — โปรดเปลี่ยนเครื่องแปล",
    "The translator's quota for this month is used up — choose another translator in the game-lens window":
        "โควตาของเครื่องแปลในเดือนนี้หมดแล้ว — โปรดเปลี่ยนเครื่องแปลในหน้าต่าง game-lens",
    "The translator's key is not valid — check it in the game-lens window":
        "key ของเครื่องแปลไม่ถูกต้อง — โปรดตรวจสอบในหน้าต่าง game-lens",
    "The translation system ran into a problem — see logs/server.log":
        "ระบบแปลภาษาขัดข้อง — ดูรายละเอียดใน logs/server.log",
    "The translator is not running — open the game-lens window and press Load":
        "เครื่องแปลยังไม่ได้เปิดใช้งาน — โปรดเปิดหน้าต่าง game-lens แล้วกด โหลด",
    "Translation failed: {error}": "การแปลไม่สำเร็จ: {error}",
    "The key {key} is used by another program (change it in settings.json)":
        "ปุ่มลัด {key} ถูกใช้งานโดยโปรแกรมอื่น (แก้ไขได้ใน settings.json)",
    "Open the game-lens window": "เปิดหน้าต่าง game-lens",
    "Translate the whole screen": "แปลทั้งหน้าจอ",
    "Set text regions": "ตั้งกรอบข้อความ",
    "The keyboard cannot be switched: the game runs as administrator — run game-lens as administrator too":
        "ไม่สามารถเปลี่ยนแป้นพิมพ์ได้ เนื่องจากเกมทำงานในสิทธิ์ผู้ดูแลระบบ — โปรดเปิด game-lens ในสิทธิ์ผู้ดูแลระบบด้วย",
    "Keyboard switched to English (this can be turned off in the game-lens window)":
        "เปลี่ยนแป้นพิมพ์เป็นภาษาอังกฤษแล้ว (ปิดการตั้งค่านี้ได้ในหน้าต่าง game-lens)",
    "game-lens: stopped": "game-lens: หยุดการแปล",
    "game-lens: translating": "game-lens: แปลต่อ",
    "The icon could not be made": "สร้างไอคอนไม่สำเร็จ",
    "The game-lens icon is on the Desktop — next time just double-click it":
        "สร้างไอคอน game-lens บนเดสก์ท็อปแล้ว — ครั้งต่อไปสามารถดับเบิลคลิกเพื่อเปิดได้ทันที",
    "New profile for {name} — set the game language and the text regions (F10)":
        "สร้างโปรไฟล์ใหม่สำหรับ {name} แล้ว — โปรดตั้งภาษาในเกมและกรอบข้อความ (F10)",
    "game-lens: the game window cannot be captured": "game-lens: ไม่สามารถจับภาพหน้าต่างเกมได้",
    "game-lens is translating • hold F8 to see the original": "game-lens พร้อมแปล • กด F8 ค้างไว้เพื่อดูต้นฉบับ",
    "Starting the translation system… please wait": "กำลังเปิดระบบแปลภาษา… โปรดรอสักครู่",
    "The game window was not found": "ไม่พบหน้าต่างเกม",
    "Translating the whole screen… (F9 / Esc to close)": "กำลังแปลทั้งหน้าจอ… (กด F9 หรือ Esc เพื่อปิด)",
    "No text on the screen": "ไม่พบข้อความบนหน้าจอ",
    "Open the game and choose its window first": "โปรดเปิดเกมและเลือกหน้าต่างเกมก่อน",
    "{n} regions saved — translating": "บันทึก {n} กรอบแล้ว — เริ่มการแปล",
    "No history to build a glossary from yet": "ยังไม่มีประวัติการแปลสำหรับสร้าง glossary",
    "Looking for character names in the history… (needs Gemma or Claude)":
        "กำลังค้นหาชื่อตัวละครจากประวัติการแปล… (ต้องใช้ Gemma หรือ Claude)",
    "The glossary could not be built": "สร้าง glossary ไม่สำเร็จ",
    "Glossary updated": "อัปเดต glossary แล้ว",
    "{n} names / terms found — they are translated the same way every time":
        "พบชื่อและคำศัพท์ {n} รายการ — จะแปลเหมือนกันทุกครั้ง",
    # --- choosing the game window --------------------------------------------------------------------------------
    "Choose the game window": "เลือกหน้าต่างเกม",
    "Which window is your game?": "หน้าต่างใดคือเกมของคุณ?",
    "⚡ Detect — press, then switch to the game within 5 seconds":
        "⚡ ตรวจจับอัตโนมัติ — กดแล้วสลับไปยังเกมภายใน 5 วินาที",
    "Search window titles…": "ค้นหาชื่อหน้าต่าง…",
    "Use this window": "ใช้หน้าต่างนี้",
    "Refresh": "รีเฟรช",
    "Cancel": "ยกเลิก",
    "or choose from the list:": "หรือเลือกจากรายการ:",
    "Switch to the game window now… {n}": "โปรดสลับไปยังหน้าต่างเกม… {n}",
    "You did not switch to the game. Try again, or choose from the list":
        "ยังไม่ได้สลับไปยังเกม โปรดลองอีกครั้งหรือเลือกจากรายการ",
    "That window is not a game ({exe}). Try again": "หน้าต่างนั้นไม่ใช่เกม ({exe}) โปรดลองอีกครั้ง",
    # --- setting regions ---------------------------------------------------------------------------------------
    "Looking for text on the screen…": "กำลังค้นหาข้อความบนหน้าจอ…",
    "Cancel (Esc)": "ยกเลิก (Esc)",
    "Save (Enter)": "บันทึก (Enter)",
    "Click the dialogue (dashed) to make a region": "คลิกที่บทพูด (เส้นประ) เพื่อสร้างกรอบ",
    "No text found on the screen — drag a region yourself": "ไม่พบข้อความบนหน้าจอ — สามารถลากเพื่อสร้างกรอบเองได้",
    "Drag to make a region around the dialogue box": "ลากเพื่อสร้างกรอบรอบกล่องบทพูด",
    "Drag empty space = new region • drag a region = move • drag its lower right corner = resize • "
    "1 dialogue  2 speaker  3 subtitles  4 choices • Delete removes • right click = options":
        "ลากที่ว่าง = สร้างกรอบใหม่ • ลากกรอบ = ย้าย • ลากมุมขวาล่าง = ปรับขนาด • "
        "1 บทพูด  2 ชื่อผู้พูด  3 คำบรรยาย  4 ตัวเลือก • Delete = ลบ • คลิกขวา = ตัวเลือกเพิ่มเติม",
    "Text colour in the region": "สีตัวอักษรในกรอบ",
    "Light (white on dark)": "สว่าง (ตัวขาวบนพื้นเข้ม)",
    "Dark (black on light)": "เข้ม (ตัวดำบนพื้นสว่าง)",
    "Varies / several colours": "ไม่แน่นอน / หลายสี",
    "Centre the text": "จัดข้อความกึ่งกลาง",
    "Remove this region (Delete)": "ลบกรอบนี้ (Delete)",
    # --- history -----------------------------------------------------------------------------------------------
    "game-lens — translation history": "game-lens — ประวัติการแปล",
    "Build / update the character-name glossary from the history":
        "สร้าง / อัปเดต glossary ชื่อตัวละครจากประวัติการแปล",
    # --- the server ---------------------------------------------------------------------------------------------
    "The token cannot be read: {path} (the translation system has never run)":
        "ไม่สามารถอ่าน token ได้: {path} (ระบบแปลภาษายังไม่เคยเปิดใช้งาน)",
    "Cannot reach the server ({url}): {error}": "ไม่สามารถเชื่อมต่อ server ได้ ({url}): {error}",
}
