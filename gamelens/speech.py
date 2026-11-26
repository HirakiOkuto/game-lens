"""Who is speaking: Thai ends polite sentences with ครับ (a man) or ค่ะ / คะ (a woman), and a translation model that
sees one line at a time cannot tell which. The speaker's name is on screen, and the glossary knows the characters."""
from __future__ import annotations

import re

from .watcher import skeleton

END = r"(?=$|[\s!?！？.…,、。~〜」』)\"'])"  # the particle ends a word (not คะแนน)
# a question word just before the particle: the woman's form is then คะ, not ค่ะ
QUESTION = re.compile(r"(ไหม|มั้ย|หรือ|เหรอ|หรอ|อะไร|ไหน|ใคร|ยังไง|อย่างไร|เท่าไร|เท่าไหร่|ทำไม|บ้าง|เปล่า|มั้ง|ล่ะ|สินะ)\s*$")
# ผม is also "hair": only taken for "I" where no word about hair follows
HAIR = ("ยาว", "สั้น", "ร่วง", "หงอก", "เปีย", "หยิก", "ตรง", "สี", "ของ", "เผ้า", "ม้า", "ทรง", "ที่", "นุ่ม")


def genders(glossary: dict) -> dict[str, str]:
    """skeleton of every name and alias in the glossary -> 'female' / 'male' (characters without one are left out)."""
    out = {}
    for c in glossary.get("characters", []):
        g = (c.get("gender") or "").lower()
        if g in ("female", "male"):
            for name in [c.get("src", ""), c.get("tl", ""), *c.get("aliases", [])]:
                if name:
                    out[skeleton(name).lower()] = g
    return out


def gender_of(speaker: str, known: dict[str, str]) -> str:
    return known.get(skeleton(speaker or "").lower(), "") if speaker else ""


def fit_speaker(tl: str, gender: str) -> str:
    """The translation with the polite particles (and the word for "I") of the speaker's gender."""
    if gender == "female":
        tl = re.sub("ครับผม" + END, "ค่ะ", tl)

        def particle(m):
            before, after = m.string[: m.start()], m.string[m.end():]
            asks = after.lstrip()[:1] in ("?", "？") or QUESTION.search(before) or before.endswith("นะ")
            asks = asks or re.search(r"(คุณ|ท่าน)?\s*[A-Za-z][\w.]*\s*$", before)  # calling someone: "Starborn คะ"
            return "คะ" if asks else "ค่ะ"

        tl = re.sub("ครับ" + END, particle, tl)
        tl = re.sub(r"(^|[\s\"'「(…])ผม(?!" + "|".join(HAIR) + ")", r"\1ฉัน", tl)
    elif gender == "male":
        tl = re.sub("(ค่ะ|คะ|ค่า|จ้ะ)" + END, "ครับ", tl)
        tl = re.sub(r"(^|[\s\"'「(…])ดิฉัน", r"\1ผม", tl)
    return tl
