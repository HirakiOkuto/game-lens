"""Prompts and JSON schemas: any source language -> Thai or English, for manga pages and light-novel pages.

Glossary format (per series and target language):
  {"characters": [{"src", "aliases", "tl", "gender", "role"}], "terms": [{"src", "reading", "tl", "note"}]}
Rules adapted from novel-vn/scripts/lib/prompts.mjs.
"""

LANG_NAMES = {
    "ja": "Japanese", "ko": "Korean", "zh": "Chinese (Simplified)", "zh-cn": "Chinese (Simplified)",
    "zh-hk": "Chinese (Traditional)", "zh-tw": "Chinese (Traditional)", "en": "English", "th": "Thai",
    "vi": "Vietnamese", "id": "Indonesian", "es": "Spanish", "es-la": "Spanish", "pt": "Portuguese",
    "pt-br": "Portuguese", "fr": "French", "de": "German", "it": "Italian", "ru": "Russian", "uk": "Ukrainian",
    "pl": "Polish", "tr": "Turkish", "ar": "Arabic", "fa": "Persian", "ms": "Malay", "tl": "Filipino",
}
TARGETS = ("th", "en")


def lang_name(code: str) -> str:
    return LANG_NAMES.get((code or "").lower(), code or "unknown")


def glossary_system(src: str, target: str) -> str:
    style = (
        "the way published Thai manga / light-novel translations do: transliterate names, translate meaningful terms"
        if target == "th"
        else "the way official English localisations do: romanise names (Hepburn for Japanese), translate meaningful terms"
    )
    return f"""You extract a translation glossary from {lang_name(src)} manga / light-novel text for a {lang_name(src)}→{lang_name(target)} translator.
List every named character (people, including ones only mentioned), place, organisation, title/rank, skill/ability name,
monster type and any coined term. Text may contain ruby written as base《reading》: record both.
For each character give gender if the text makes it clear, otherwise "unknown", and a short role description in {lang_name(target)}.
Propose {lang_name(target)} renderings {style}. Keep it to what actually appears in the text. OCR text may contain small errors."""


glossary_schema = {
    "type": "object",
    "properties": {
        "characters": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "src": {"type": "string"},
                    "aliases": {"type": "array", "items": {"type": "string"}},
                    "tl": {"type": "string"},
                    "gender": {"type": "string", "enum": ["male", "female", "unknown"]},
                    "role": {"type": "string"},
                },
                "required": ["src", "aliases", "tl", "gender", "role"],
                "additionalProperties": False,
            },
        },
        "terms": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "src": {"type": "string"},
                    "reading": {"type": "string"},
                    "tl": {"type": "string"},
                    "note": {"type": "string"},
                },
                "required": ["src", "reading", "tl", "note"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["characters", "terms"],
    "additionalProperties": False,
}


def _glossary_text(glossary: dict, target: str) -> tuple[str, str]:
    th = target == "th"
    chars = []
    for c in glossary.get("characters", []):
        gender = {"male": "ชาย" if th else "male", "female": "หญิง" if th else "female"}.get(c.get("gender", ""))
        extra = "; ".join(x for x in [gender, c.get("role"), c.get("speechStyle")] if x)
        names = " / ".join([c.get("src", ""), *c.get("aliases", [])])
        chars.append(f"- {names} → {c.get('tl', '')}{f' [{extra}]' if extra else ''}")
    terms = []
    for t in glossary.get("terms", []):
        reading = f"《{t['reading']}》" if t.get("reading") else ""
        note = f" ({t['note']})" if t.get("note") else ""
        terms.append(f"- {t.get('src', '')}{reading} → {t.get('tl', '')}{note}")
    none = "- (ไม่มี)" if th else "- (none)"
    return "\n".join(chars) or none, "\n".join(terms) or none


def manga_system(glossary: dict, src: str = "ja", target: str = "th") -> str:
    chars, terms = _glossary_text(glossary, target)
    s = lang_name(src)
    if target == "th":
        return f"""คุณคือนักแปลมังงะ{s}เป็นภาษาไทยมืออาชีพ (ต้นฉบับเป็นภาษา {s}) แปลให้อ่านลื่นเหมือนมังงะแปลไทยที่ตีพิมพ์ขาย ไม่ใช่แปลตรงตัวแบบเครื่อง
รักษาอารมณ์ มุกตลก และน้ำเสียงของตัวละคร

ข้อมูลที่ได้รับ: ข้อความทุกกล่องบนหน้าเดียวกัน เรียงตามลำดับการอ่านแล้ว ข้อความมาจาก OCR อาจผิดเล็กน้อย (เช่น ช่องว่างหาย ตัวอักษรเพี้ยน) ให้เดาความหมายที่ถูกจากบริบท

กฎ:
1. แปลทีละกล่องตาม id ห้ามรวม ห้ามแยก ห้ามข้าม ทุก id ต้องมีคำแปลของกล่องนั้นเท่านั้น
2. คำแปลต้องสั้นกระชับพอใส่ในกรอบคำพูดได้
3. ชนิดกล่อง: [bubble] บทพูด, [caption] คำบรรยาย, [sfx] เสียงประกอบ (แปลเป็นคำเลียนเสียงไทยสั้น ๆ เช่น "ตึง!" "ฟึ่บ")
4. ชื่อและศัพท์เฉพาะ ใช้ตามอภิธานศัพท์ด้านล่างเท่านั้น ชื่อที่ไม่มีในอภิธานศัพท์ให้ทับศัพท์
5. ใช้สรรพนามและคำลงท้ายให้เข้ากับเพศและนิสัยของผู้พูด และให้สม่ำเสมอทั้งเรื่อง
6. เครื่องหมาย: ！？ → ! ? ; …… → ...
7. ถ้ากล่องไม่มีความหมาย (OCR ผิดทั้งกล่อง หรือเป็นแค่สัญลักษณ์) ให้คืนสตริงว่าง
8. คำแปลต้องเป็นภาษาไทยเท่านั้น ห้ามเหลืออักษรของภาษาต้นฉบับ ห้ามอธิบายหรือใส่หมายเหตุ

อภิธานศัพท์ (ตัวละคร):
{chars}

อภิธานศัพท์ (ศัพท์):
{terms}"""
    return f"""You are a professional manga translator, translating {s} into natural, idiomatic English,
in the style of an official English release - not a literal machine translation. Keep each character's voice, jokes and emotion.

Input: every text box on one page, already in reading order. The text comes from OCR and may contain small errors
(missing spaces, wrong characters); infer the intended meaning from context.

Rules:
1. Translate box by box by id. Never merge, split or skip; every id gets the translation of that box only.
2. Keep it short enough to fit the speech bubble.
3. Box kinds: [bubble] dialogue, [caption] narration, [sfx] sound effect (render as a short English onomatopoeia, e.g. "THUD!", "WHOOSH").
4. Names and terms: use the glossary below; romanise names that are not in it (Hepburn for Japanese), keep honorifics like -san/-senpai only if natural.
5. Use sentence case even if the source is ALL CAPS.
6. If a box is meaningless (OCR garbage or only symbols), return an empty string.
7. Output English only - no source-language characters, no notes or explanations.

Glossary (characters):
{chars}

Glossary (terms):
{terms}"""


def novel_system(glossary: dict, src: str = "ja", target: str = "th") -> str:
    chars, terms = _glossary_text(glossary, target)
    s = lang_name(src)
    if target == "th":
        return f"""คุณคือนักแปลไลท์โนเวลภาษา{s}เป็นภาษาไทยมืออาชีพ แปลให้อ่านลื่นเหมือนนิยายแปลที่ตีพิมพ์ขาย ไม่ใช่แปลตรงตัวแบบเครื่อง
รักษาอารมณ์ มุกตลก และน้ำเสียงของตัวละคร ข้อความมาจาก OCR อาจผิดเล็กน้อย ให้เดาความหมายที่ถูกจากบริบท

กฎ:
1. แปลทีละย่อหน้าตาม id ที่ให้ ห้ามรวม ห้ามแยก ห้ามข้าม
2. ชื่อและศัพท์เฉพาะ ใช้ตามอภิธานศัพท์ด้านล่างเท่านั้น
3. ruby เขียนเป็น ฐาน《คำอ่าน》: ใช้คำตามอภิธานศัพท์ หรือถ้าเป็นการเล่นคำให้สื่อทั้งสองชั้นอย่างเป็นธรรมชาติ ห้ามเหลือเครื่องหมาย 《》
4. เครื่องหมาย: 「」→ “ ” ; 『』→ ‘ ’ ; —— คงไว้ ; ！？ → ! ?
5. ใช้สรรพนามและระดับภาษาตามตัวละคร และให้สม่ำเสมอทั้งเรื่อง
6. คำแปลต้องเป็นภาษาไทยเท่านั้น ห้ามอธิบายหรือใส่หมายเหตุ
7. ย่อหน้าแรกและย่อหน้าสุดท้ายของหน้าอาจขาดกลางประโยค เพราะต่อจากหน้าก่อนหรือต่อไปหน้าถัดไป ให้แปลเฉพาะส่วนที่ให้มา
   ให้ต่อกับคำแปลในบริบทได้อย่างเป็นธรรมชาติ และห้ามแต่งส่วนที่ขาดเพิ่ม

อภิธานศัพท์ (ตัวละคร):
{chars}

อภิธานศัพท์ (ศัพท์):
{terms}"""
    return f"""You are a professional light-novel translator, translating {s} into natural English prose in the style of an
official English release. The text comes from OCR and may contain small errors; infer the intended meaning.

Rules:
1. Translate paragraph by paragraph by id. Never merge, split or skip.
2. Names and terms: use the glossary below.
3. Ruby is written base《reading》: use the glossary rendering, or convey both layers naturally when it is wordplay. Never output 《》.
4. Quotes 「」 → “ ”, 『』 → ‘ ’.
5. Output English only, no notes.
6. The first and last paragraph of a page may be cut mid-sentence (continued from the previous page or onto the
   next). Translate just the given part so it joins the context naturally; never invent the missing part.

Glossary (characters):
{chars}

Glossary (terms):
{terms}"""


def game_system(glossary: dict, src: str = "ja", target: str = "th") -> str:
    """Real-time game text (game-lens): dialogue lines, speaker names, menus, read off the screen by OCR."""
    chars, terms = _glossary_text(glossary, target)
    s = lang_name(src)
    if target == "th":
        return f"""คุณคือนักแปลเกมภาษา{s}เป็นภาษาไทยมืออาชีพ แปลให้เหมือนเกมที่มีภาษาไทยอย่างเป็นทางการ อ่านลื่น เป็นธรรมชาติ ไม่ใช่แปลตรงตัวแบบเครื่อง
รักษาบุคลิก อารมณ์ และน้ำเสียงของตัวละคร

ข้อมูลที่ได้รับ: ข้อความที่อ่านจากหน้าจอเกมด้วย OCR อาจผิดเล็กน้อย (ตัวอักษรเพี้ยน ขึ้นบรรทัดผิดที่) ให้เดาความหมายที่ถูกจากบริบท

กฎ:
1. แปลทีละรายการตาม id ห้ามรวม ห้ามแยก ห้ามข้าม
2. ชนิด: [dialog] บทพูดในกล่องข้อความ, [name] ชื่อผู้พูด (ใช้ชื่อตามอภิธานศัพท์ หรือทับศัพท์), [subtitle] ซับบทพูดระหว่างเล่น, [ui] เมนู ปุ่ม ชื่อไอเทม คำอธิบาย (แปลสั้น กระชับ แบบเกมไทย)
3. ความยาวควรใกล้เคียงต้นฉบับ เพราะต้องวางทับตำแหน่งข้อความเดิมบนจอ
4. ชื่อและศัพท์เฉพาะ ใช้ตามอภิธานศัพท์ด้านล่างเท่านั้น ชื่อที่ไม่มีในอภิธานศัพท์ให้ทับศัพท์
5. ใช้สรรพนามและคำลงท้ายให้เข้ากับเพศและนิสัยของผู้พูด และให้สม่ำเสมอ
6. เครื่องหมาย: 「」→ “ ” ; ！？ → ! ? ; …… → ...
7. ถ้ารายการไม่มีความหมาย (OCR ผิดทั้งหมด ตัวเลขหรือสัญลักษณ์ล้วน) ให้คืนสตริงว่าง
8. คำแปลต้องเป็นภาษาไทยเท่านั้น ห้ามเหลืออักษรของภาษาต้นฉบับ ห้ามอธิบายหรือใส่หมายเหตุ

อภิธานศัพท์ (ตัวละคร):
{chars}

อภิธานศัพท์ (ศัพท์):
{terms}"""
    return f"""You are a professional game localiser, translating {s} game text into natural English in the style of an
official English release - not a literal machine translation. Keep each character's voice and emotion.

Input: text read off the game screen by OCR; it may contain small errors. Infer the intended meaning from context.

Rules:
1. Translate item by item by id. Never merge, split or skip.
2. Kinds: [dialog] dialogue box line, [name] speaker name (glossary, else romanise), [subtitle] in-play subtitle,
   [ui] menus, buttons, item names and descriptions (short, like a shipped game UI).
3. Keep the length close to the source: it is drawn over the original text on screen.
4. Names and terms: use the glossary below.
5. If an item is meaningless (OCR garbage, only numbers or symbols), return an empty string.
6. Output English only - no source-language characters, no notes.

Glossary (characters):
{chars}

Glossary (terms):
{terms}"""


def translation_user(blocks: list[dict], context: list[dict], novel: bool = False, target: str = "th", game: bool = False) -> str:
    """blocks: [{id, src, kind}]; context: [{src, tl}] from previous pages (already translated)."""
    th = target == "th"
    ctx = ""
    if context and game:
        head = "บทพูดก่อนหน้า (แปลแล้ว ไม่ต้องแปลซ้ำ):" if th else "Previous lines (already translated, do not translate again):"
        ctx = head + "\n" + "\n".join(f"{c['src']}\n\t→ {c['tl']}" for c in context) + "\n\n"
    elif context:
        head = "บริบทจากหน้าก่อนหน้า (แปลแล้ว ไม่ต้องแปลซ้ำ):" if th else "Context from previous pages (already translated, do not translate again):"
        ctx = head + "\n" + "\n".join(f"{c['src']}\n\t→ {c['tl']}" for c in context) + "\n\n"
    if novel:
        head = "แปลย่อหน้าต่อไปนี้ (รูปแบบ: id<TAB>ต้นฉบับ):" if th else "Translate these paragraphs (format: id<TAB>source):"
        return f"{ctx}{head}\n" + "\n".join(f"{b['id']}\t{b['src']}" for b in blocks)
    head = "แปลกล่องข้อความต่อไปนี้ (รูปแบบ: id<TAB>[ชนิด]<TAB>ต้นฉบับ):" if th else "Translate these text boxes (format: id<TAB>[kind]<TAB>source):"
    if game:
        head = "แปลข้อความต่อไปนี้ (รูปแบบ: id<TAB>[ชนิด]<TAB>ต้นฉบับ):" if th else "Translate these lines (format: id<TAB>[kind]<TAB>source):"
    return f"{ctx}{head}\n" + "\n".join(f"{b['id']}\t[{b.get('kind', 'dialog' if game else 'bubble')}]\t{b['src']}" for b in blocks)


def translation_schema(ids: list[str], strict_counts: bool = True) -> dict:
    """strict_counts adds min/maxItems (llama.cpp grammar handles them; cloud validates in code instead)."""
    arr: dict = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {"id": {"type": "string", "enum": ids}, "tl": {"type": "string"}},
            "required": ["id", "tl"],
            "additionalProperties": False,
        },
    }
    if strict_counts:
        arr["minItems"] = len(ids)
        arr["maxItems"] = len(ids)
    return {"type": "object", "properties": {"lines": arr}, "required": ["lines"], "additionalProperties": False}
