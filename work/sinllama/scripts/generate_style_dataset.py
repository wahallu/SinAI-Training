#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FRESH SINHALA NEWS STYLE DATASET GENERATOR

Input:
    /home/jovyan/style_rewriter/data/train1.jsonl

Output:
    /home/jovyan/style_rewriter/data/style_dataset_fresh.jsonl

IMPORTANT:
- Does NOT use previous rewritten datasets.
- Every source article is independently rewritten into all 5 styles.
- Results are appended immediately after successful generation.
- If the process stops, already-generated rows remain in the output file.
- Existing output is NOT used to generate content.
- Strict factual preservation.
- Strong protection against hallucination.
- Strong truncation detection.
"""

import os
import json
import time
import argparse
import requests
import re
import random
import threading
from pathlib import Path
from collections import Counter
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime


# ================================================================
# CONFIGURATION
# ================================================================

# DO NOT hard-code your NVIDIA API key in the source code.
# Set it before running:
#
# export NVIDIA_API_KEY="YOUR_KEY"
#
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "").strip()

MODEL_NAME = "nvidia/llama-3.3-nemotron-super-49b-v1.5"

INVOKE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

INPUT_PATH = "/home/jovyan/style_rewriter/data/train1.jsonl"

OUTPUT_PATH = "/home/jovyan/style_rewriter/data/style_dataset_fresh.jsonl"


ALL_STYLES = [
    "style_1_formal_news",
    "style_2_editorial",
    "style_3_sports",
    "style_4_youth",
    "style_5_feature",
]


# ================================================================
# GLOBAL SYSTEM PROMPT
# ================================================================

SYSTEM_PROMPT = r"""
ඔබ ඉතා උසස් සිංහල පුවත්පත් ලේඛකයෙකි.

ඔබට ලබාදෙනු ලබන්නේ සැබෑ පුවත් ලිපියකි.
ඔබගේ කාර්යය වන්නේ එම ලිපියේ අර්ථය, කරුණු සහ තොරතුරු
වෙනස් නොකර, ලබාදී ඇති විශේෂිත ලේඛන ශෛලියට ගැළපෙන ලෙස
නැවත ලිවීමයි.

============================================================
අනිවාර්ය කරුණු සංරක්ෂණ නීති
============================================================

1. මුල් ලිපියේ ඇති සියලු ප්‍රධාන කරුණු සංරක්ෂණය කරන්න.

2. මුල් ලිපියේ නැති කිසිදු කරුණක් එකතු නොකරන්න.

3. කිසිදු පුද්ගලයෙකු, ආයතනයක්, ස්ථානයක්, දිනයක්,
   සංඛ්‍යාවක්, මුදලක් හෝ සිදුවීමක් නිර්මාණය නොකරන්න.

4. මුල් ලිපියේ ඇති සංඛ්‍යා වෙනස් නොකරන්න.

5. පුද්ගල නාම වෙනස් නොකරන්න.

6. ස්ථාන නාම වෙනස් නොකරන්න.

7. ආයතන / සංවිධාන නාම වෙනස් නොකරන්න.

8. මුල් ලිපියේ ඇති දිනයන් සහ කාල සම්බන්ධතා සංරක්ෂණය කරන්න.

9. මුල් ලිපියේ යම් කරුණක් "කියන", "බවට", "සඳහන් කරයි",
   "චෝදනා කරයි", "වාර්තා වේ" වැනි අවිනිශ්චිත හෝ attribution
   ආකාරයකින් තිබේ නම්, එය සත්‍යයක් ලෙස වෙනස් නොකරන්න.

10. "චෝදනා කළේය" → "සිදු කළේය" වැනි අර්ථ වෙනසක් නොකරන්න.

11. "භාරයට ගත්" → "අත්අඩංගුවට ගත්" වැනි නීතිමය අර්ථය
    වෙනස් කරන වචන භාවිත නොකරන්න.

12. මුල් ලිපියේ නැති quotes, statements, opinions හෝ
    conclusions එකතු නොකරන්න.

13. මුල් ලිපියේ නැති පුද්ගලික අත්දැකීම්, හැඟීම්,
    කාලගුණය, පරිසර විස්තර, සංවාද හෝ සිද්ධි එකතු නොකරන්න.

14. style එක වෙනස් විය හැකි නමුත් FACTS වෙනස් නොවිය යුතුය.

============================================================
සම්පූර්ණත්ව නීති
============================================================

15. මුල් ලිපිය කිසිසේත් කෙටි නොකරන්න.

16. මුල් ලිපියේ සියලුම තොරතුරු rewritten article එකේ
    තිබිය යුතුය.

17. වාක්‍යයක් මැද නතර නොකරන්න.

18. rewritten article එක අවසානයේ සම්පූර්ණ වාක්‍යයක් තිබිය යුතුය.

19. "..." භාවිත කරමින් ලිපිය කෙටි නොකරන්න.

20. ලිපියේ අවසාන කොටස කිසිසේත් අත්හැර නොදමන්න.

============================================================
ප්‍රතිදාන නීති
============================================================

21. rewritten article එක පමණක් output කරන්න.

22. "මෙන්න නැවත ලියූ ලිපිය", "Rewritten article:",
    "Here is..." වැනි meta text නොලියන්න.

23. Markdown heading, bullet list හෝ bold title එකතු නොකරන්න.

24. මුල් title එක rewritten_text තුළ නැවත නොලියන්න.

25. මුල් article එකේ paragraph structure එක හැකි තරම්
    ස්වභාවිකව පවත්වා ගන්න.

26. සිංහල භාෂාවේ ස්වාභාවික සහ නිවැරදි ව්‍යාකරණ භාවිත කරන්න.

27. Artificial phrases, filler phrases සහ meaningless
    dramatic phrases භාවිත නොකරන්න.

28. එකම වචනය හෝ වාක්‍ය ඛණ්ඩය අනවශ්‍ය ලෙස නැවත නැවත භාවිත නොකරන්න.

29. Article එකේ source facts වලට වඩා වැඩි claims නොකරන්න.

30. Style instruction එකක් factual accuracy සමඟ ගැටෙන්නේ නම්,
    factual accuracy ප්‍රමුඛ කරගන්න.
"""


# ================================================================
# STYLE PROMPTS
# ================================================================

STYLE_INSTRUCTIONS = {

    "style_1_formal_news": r"""
ශෛලිය: FORMAL NEWS

මෙය සාම්ප්‍රදායික සිංහල ප්‍රධාන පුවත් වාර්තාවක් ලෙස ලියන්න.

නීති:

- වෛෂයික සහ නිෂ්පාක්ෂික භාෂාව භාවිත කරන්න.
- වැදගත්ම කරුණ මුලින් දක්වන්න.
- නිවැරදි පුවත් වාර්තාකරණ වාක්‍ය රටා භාවිත කරන්න.
- "ඇත", "තිබේ", "වේ", "විය", "කරයි", "සඳහන් කළේය"
  වැනි සුදුසු ක්‍රියාපද භාවිත කරන්න.
- passive construction අවශ්‍ය අවස්ථාවලදී "විසින්" නිවැරදිව භාවිත කරන්න.
- "එසේම", "තවද", "කෙසේ වෙතත්", "මේ අතර" වැනි
  සම්බන්ධක වචන අවශ්‍ය තැන්වල පමණක් භාවිත කරන්න.
- අදහස් හෝ විශ්ලේෂණ එකතු නොකරන්න.
- sensational language භාවිත නොකරන්න.
- source article එකේ සියලු තොරතුරු පවත්වා ගන්න.
- article එක කෙටි නොකරන්න.

මෙම ලිපිය formal news style එකට නැවත ලියන්න.
""",

    "style_2_editorial": r"""
ශෛලිය: EDITORIAL / ANALYTICAL

ලිපිය editorial / analytical newspaper style එකකට
නැවත ලියන්න.

නීති:

- කරුණු අතර logical flow එකක් ඇති කරන්න.
- හේතු සහ ප්‍රතිඵල source article එකෙන් පැහැදිලිව
  ලබාගත හැකි අවස්ථාවලදී පමණක් සම්බන්ධ කරන්න.
- analytical language භාවිත කළ හැකි නමුත්
  source එකේ නැති opinion එකක් එකතු නොකරන්න.
- "සැලකිය යුතුය", "පෙනී යයි", "මේ අනුව" වැනි
  expressions අවශ්‍ය තැන්වල පමණක් භාවිත කරන්න.
- "අපි", "අපට" වැනි first-person expressions
  අධික ලෙස භාවිත නොකරන්න.
- source එකේ facts වලින් beyond conclusions
  ලබා නොදෙන්න.
- කිසිදු political, social හෝ moral opinion එකක්
  model විසින් නිර්මාණය නොකරන්න.
- forced conclusion එකක් එකතු නොකරන්න.
- විශේෂයෙන් පහත වාක්‍යය සෑම article එකකටම දමන්න එපා:
  "මෙම සියලු කරුණු සමස්තයක් ලෙස සලකන කල..."

Editorial tone එක භාවිත කළත් article එකේ factual
content සම්පූර්ණයෙන්ම පවත්වා ගන්න.

මෙම ලිපිය editorial style එකට නැවත ලියන්න.
""",

    "style_3_sports": r"""
ශෛලිය: SPORTS JOURNALISM

මෙය sports journalism style එකකින් ලියන්න.

ඉතා වැදගත්:

SOURCE ARTICLE එක sports article එකක් නොවේ නම්,
එය sports article එකක් බවට පත් නොකරන්න.

උදාහරණයක් ලෙස Local, Politics, Business, International
හෝ වෙනත් article එකක් ලබාදුන්නේ නම්:

- sports terminology එකතු නොකරන්න.
- තරග, ජයග්‍රහණ, ක්‍රීඩකයින්, විනිසුරුවන්,
  score, match වැනි දේවල් නිර්මාණය නොකරන්න.
- source article එකේ subject එක එලෙසම තබා,
  sports journalism එකේ energetic, concise,
  action-oriented sentence style පමණක් භාවිත කරන්න.

Source article එක සැබවින්ම sports article එකක් නම්:

- ප්‍රධාන ක්‍රීඩා සිදුවීම මුලින් දක්වන්න.
- ක්‍රියාකාරී සහ වේගවත් වාක්‍ය භාවිත කරන්න.
- තරගකාරී බව source එකේ තිබේ නම් පමණක්
  energetic language භාවිත කරන්න.
- source එකේ නැති score, result, performance,
  victory හෝ dramatic event එකතු නොකරන්න.

කිසිදු නව sports fact එකක් නිර්මාණය නොකරන්න.

මෙම ලිපිය sports journalism style එකට නැවත ලියන්න.
""",

    "style_4_youth": r"""
ශෛලිය: YOUTH NEWS

තරුණ පාඨකයන්ට පහසුවෙන් කියවිය හැකි modern Sinhala
news style එකකට article එක නැවත ලියන්න.

නීති:

- සරල සහ ස්වභාවික සිංහල භාවිත කරන්න.
- වාක්‍ය සාපේක්ෂව කෙටි සහ පැහැදිලි විය හැක.
- conversational tone සුළු වශයෙන් භාවිත කළ හැක.
- "දන්නවද?" වැනි expressions අවශ්‍ය නම් උපරිම එක් වරක්
  පමණක් භාවිත කළ හැක.
- "යාලුවනේ" වැනි filler words අධික ලෙස භාවිත නොකරන්න.
- social-media style එකකට පත් නොකරන්න.
- emojis භාවිත නොකරන්න.
- slang අධික ලෙස භාවිත නොකරන්න.
- source article එකේ facts සියල්ල තබන්න.
- source article එකේ නැති excitement, emotion,
  opinion හෝ conclusion එකතු නොකරන්න.
- forced ending එකක් භාවිත නොකරන්න.
- "ඒ නිසා යාලුවනේ, මේ ගැන අනිවාර්යයෙන්ම දැනගන්න!"
  වැනි fixed ending භාවිත නොකරන්න.

Youth-friendly වීම යනු facts වෙනස් කිරීම නොවේ.

මෙම ලිපිය youth news style එකට නැවත ලියන්න.
""",

    "style_5_feature": r"""
ශෛලිය: FEATURE JOURNALISM

ලිපිය feature journalism style එකකින් නැවත ලියන්න.

නමුත් මෙය factual news dataset එකක් බැවින් පහත නීති
අතිශයින් වැදගත් වේ:

- source එකේ නැති scene එකක් නිර්මාණය නොකරන්න.
- source එකේ නැති weather එකක් එකතු නොකරන්න.
- source එකේ නැති atmosphere එකක් එකතු නොකරන්න.
- source එකේ නැති emotions එකතු නොකරන්න.
- source එකේ නැති personal experience එකක් එකතු නොකරන්න.
- source එකේ නැති dialogue එකක් එකතු නොකරන්න.
- source එකේ නැති human-interest detail එකක් එකතු නොකරන්න.
- fictional storytelling නොකරන්න.

Feature style එක ලබාදිය යුත්තේ:

- වඩාත් smooth narrative flow එකකින්,
- සිද්ධියට context එකක් ලබාදෙන ආකාරයෙන්,
- විවිධ දිගින් යුතු ස්වභාවික වාක්‍ය භාවිතයෙන්,
- source එකේ තිබෙන මානව හෝ පසුබිම් කරුණු
  ඉස්මතු කිරීමෙන් පමණි.

Source එකේ facts පමණක් භාවිත කරන්න.

Forced dramatic opening භාවිත නොකරන්න.

"අඳුරු උදෑසනක..."
"නිහඬ පරිසරය තුළ..."
"මෙම අභිමාණීය..."
වැනි source එකේ නැති descriptive phrases
අවශ්‍යතාවයකින් තොරව එකතු නොකරන්න.

Forced ending එකක් භාවිත නොකරන්න.

"මෙම කතාව අනාගත පරම්පරාවට ද ආදර්ශයක් වනු ඇත"
වැනි fixed sentence එකක් සෑම article එකකටම දමන්න එපා.

මෙම ලිපිය factual feature journalism style එකට
නැවත ලියන්න.
"""
}


# ================================================================
# API
# ================================================================

def call_api(prompt, max_tokens=4096, temperature=0.10, max_retries=5):

    if not NVIDIA_API_KEY:
        raise RuntimeError(
            "NVIDIA_API_KEY is not set. "
            "Run: export NVIDIA_API_KEY='YOUR_KEY'"
        )

    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],

        # Larger output limit prevents truncation.
        "max_tokens": max_tokens,

        # Lower temperature = better factual consistency.
        "temperature": temperature,

        "top_p": 0.90,

        "stream": False,
    }

    for attempt in range(max_retries):

        try:

            response = requests.post(
                INVOKE_URL,
                headers=headers,
                json=payload,
                timeout=240,
            )

            if response.status_code == 200:

                data = response.json()

                choice = data["choices"][0]

                message = choice.get("message", {})

                text = message.get("content", "").strip()

                if not text:
                    raise RuntimeError("Empty API response")

                return text

            if response.status_code == 429:

                wait = min(2 ** (attempt + 1), 30)

                print(
                    f"\n⚠️ Rate limit. Waiting {wait}s...",
                    flush=True
                )

                time.sleep(wait)

                continue

            if response.status_code >= 500:

                wait = min(2 ** (attempt + 1), 20)

                print(
                    f"\n⚠️ Server error {response.status_code}. "
                    f"Retrying in {wait}s...",
                    flush=True
                )

                time.sleep(wait)

                continue

            raise RuntimeError(
                f"API {response.status_code}: "
                f"{response.text[:500]}"
            )

        except requests.exceptions.Timeout:

            if attempt < max_retries - 1:

                time.sleep(3)

                continue

            raise

        except requests.exceptions.ConnectionError:

            if attempt < max_retries - 1:

                time.sleep(3)

                continue

            raise

    raise RuntimeError(
        f"API failed after {max_retries} attempts"
    )


# ================================================================
# CLEAN RESPONSE
# ================================================================

def clean_response(text):
    """
    Remove model artifacts without destroying article content.
    """

    if not text:
        return ""

    text = text.strip()

    # Remove code fences.
    text = re.sub(
        r"^```(?:text|sinhala)?\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"\s*```$",
        "",
        text
    )

    lines = text.splitlines()

    cleaned = []

    for line in lines:

        stripped = line.strip()

        if not stripped:
            continue

        # Remove obvious meta responses.
        if re.match(
            r"^(rewritten article|rewritten text|here is|here's|"
            r"මෙන්න නැවත|නැවත ලියූ ලිපිය)",
            stripped,
            flags=re.IGNORECASE
        ):
            continue

        # Remove markdown heading markers.
        stripped = re.sub(
            r"^#{1,6}\s*",
            "",
            stripped
        )

        # Remove bold wrappers.
        stripped = stripped.replace("**", "")

        cleaned.append(stripped)

    result = "\n".join(cleaned).strip()

    return result


# ================================================================
# TITLE REMOVAL
# ================================================================

def remove_duplicate_title(title, rewritten):
    """
    Remove title if model repeats source title at beginning.
    """

    if not title or not rewritten:
        return rewritten

    title = title.strip()

    text = rewritten.strip()

    # Exact title.
    if text.startswith(title):

        remainder = text[len(title):].lstrip(" :-–—\n")

        if remainder:
            return remainder

    # Bold title.
    bold_title = f"**{title}**"

    if text.startswith(bold_title):

        remainder = text[len(bold_title):].lstrip(" :-–—\n")

        if remainder:
            return remainder

    return rewritten


# ================================================================
# TEXT NORMALIZATION
# ================================================================

def normalize_text(text):

    if not text:
        return ""

    text = text.replace("\u00ad", "")

    text = text.replace("\r\n", "\n")

    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    return text.strip()


# ================================================================
# TRUNCATION DETECTION
# ================================================================

def looks_truncated(original, rewritten):

    if not rewritten:
        return True

    text = rewritten.strip()

    # Obvious truncation.
    if text.endswith("..."):
        return True

    if text.endswith(".."):
        return True

    # Sinhala / English sentence-ending punctuation.
    valid_endings = (
        ".",
        "!",
        "?",
        "।",
        "යි",
        "වේ",
        "බව",
        "තිබේ",
        "කරයි",
        "විය",
        "ඇත",
        "තිබුණි",
        "තිබුණා",
        "කළේය",
        "කර තිබේ",
        "වූ බව",
    )

    if not text.endswith(valid_endings):
        # Do not immediately reject very short news articles.
        if len(original) > 500:
            return True

    # Compare character lengths.
    original_len = len(original)

    rewritten_len = len(rewritten)

    # A rewrite that suddenly becomes extremely short is suspicious.
    if original_len >= 500:

        ratio = rewritten_len / original_len

        if ratio < 0.55:
            return True

    return False


# ================================================================
# REPETITION DETECTION
# ================================================================

def has_excessive_repetition(text):

    words = re.findall(
        r"[\u0D80-\u0DFF]+",
        text
    )

    if len(words) < 20:
        return False

    counts = Counter(words)

    for word, count in counts.items():

        if len(word) > 3 and count >= 12:

            # Common Sinhala particles can naturally repeat.
            ignored = {
                "සඳහන්",
                "විසින්",
                "සඳහා",
                "ලෙස",
                "අතර",
                "පිළිබඳ",
                "කිරීම",
                "තිබේ",
            }

            if word not in ignored:
                return True

    # Repeated identical 3-word sequence.
    normalized = re.sub(
        r"\s+",
        " ",
        text
    )

    if re.search(
        r"(\S+\s+\S+\s+\S+).*\1",
        normalized
    ):
        return True

    return False


# ================================================================
# SINHALA RATIO
# ================================================================

def Sinhala_ratio(text):

    sinhala = len(
        re.findall(
            r"[\u0D80-\u0DFF]",
            text
        )
    )

    letters = len(
        re.findall(
            r"[A-Za-z\u0D80-\u0DFF]",
            text
        )
    )

    if letters == 0:
        return 0

    return sinhala / letters


# ================================================================
# BASIC FACT PRESERVATION
# ================================================================

def extract_numbers(text):

    return re.findall(
        r"\d+(?:[.,]\d+)*",
        text
    )


def validate_numbers(original, rewritten):

    original_numbers = Counter(
        extract_numbers(original)
    )

    rewritten_numbers = Counter(
        extract_numbers(rewritten)
    )

    # Every number appearing in source must remain.
    for number, count in original_numbers.items():

        if rewritten_numbers[number] < count:

            return False

    return True


# ================================================================
# VALIDATION
# ================================================================

def validate_rewrite(
    title,
    original,
    rewritten,
    style
):

    issues = []

    original = normalize_text(original)

    rewritten = normalize_text(rewritten)

    if not rewritten:

        issues.append("empty")

        return issues

    if len(rewritten) < 50:

        issues.append("too_short")

    # ------------------------------------------------------------
    # Length
    # ------------------------------------------------------------

    ratio = len(rewritten) / max(len(original), 1)

    if ratio < 0.55:

        issues.append("too_short_relative_to_source")

    if ratio > 1.80:

        issues.append("too_long")

    # ------------------------------------------------------------
    # Sinhala dominance
    # ------------------------------------------------------------

    ratio_sinhala = Sinhala_ratio(rewritten)

    if ratio_sinhala < 0.60:

        issues.append("low_sinhala_ratio")

    # ------------------------------------------------------------
    # Truncation
    # ------------------------------------------------------------

    if looks_truncated(original, rewritten):

        issues.append("possible_truncation")

    # ------------------------------------------------------------
    # Numbers
    # ------------------------------------------------------------

    if not validate_numbers(original, rewritten):

        issues.append("missing_source_number")

    # ------------------------------------------------------------
    # Repetition
    # ------------------------------------------------------------

    if has_excessive_repetition(rewritten):

        issues.append("excessive_repetition")

    # ------------------------------------------------------------
    # Markdown
    # ------------------------------------------------------------

    if "```" in rewritten:

        issues.append("markdown_artifact")

    if rewritten.startswith("#"):

        issues.append("markdown_heading")

    # ------------------------------------------------------------
    # Meta language
    # ------------------------------------------------------------

    meta_patterns = [
        "මෙන්න",
        "නැවත ලියූ ලිපිය",
        "rewritten article",
        "here is the rewritten",
    ]

    lower = rewritten.lower()

    for pattern in meta_patterns:

        if pattern.lower() in lower:

            issues.append("meta_text")

            break

    # ------------------------------------------------------------
    # Obvious hallucination / generic filler
    # ------------------------------------------------------------

    dangerous_phrases = [

        "ඔබේ ජයග්‍රහණය",

        "අවසන් විසිල්",

        "දැඩි තරගකාරීත්වය තවත්",

        "යාලුවනේ, මේ ගැන අනිවාර්යයෙන්ම",

        "අනාගත පරම්පරාවට ද ආදර්ශයක්",

        "අඳුරු උදෑසනක",

        "නිහඬ පරිසරය තුළ",

        "මෙම අභිමාණීය",

        "මෙම සුවිශේෂී මොහොත",

        "මෙම ඓතිහාසික අවස්ථාව",

    ]

    for phrase in dangerous_phrases:

        if phrase in rewritten:

            issues.append("generic_or_hallucinated_phrase")

            break

    return list(set(issues))


# ================================================================
# GENERATION TOKEN CALCULATION
# ================================================================

def calculate_max_tokens(content):

    """
    Sinhala text can require a relatively large token count.

    Never use the previous tiny 2048-ish limit for long articles.
    """

    chars = len(content)

    # Roughly generous allowance.
    estimated = int(chars / 1.4) + 1200

    # Minimum.
    estimated = max(
        estimated,
        3000
    )

    # Maximum supported generation.
    estimated = min(
        estimated,
        8192
    )

    return estimated


# ================================================================
# PROCESS ONE
# ================================================================

_worker_lock = threading.Lock()

_last_call_time = {}


def process_one(
    title,
    content,
    url,
    category,
    date_published,
    style,
    worker_id=0
):

    if not content or not content.strip():

        return None

    content = normalize_text(content)

    title = normalize_text(title)

    # Small per-worker delay.
    with _worker_lock:

        now = time.time()

        last = _last_call_time.get(
            worker_id,
            0
        )

        gap = 0.10 - (now - last)

        if gap > 0:

            time.sleep(gap)

        _last_call_time[worker_id] = time.time()

    style_instruction = STYLE_INSTRUCTIONS[style]

    prompt = f"""
{style_instruction}

============================================================
SOURCE ARTICLE
============================================================

Title:
{title}

Category:
{category}

Article:
{content}

============================================================
FINAL REQUIREMENT
============================================================

Return ONLY the rewritten article body.

Do NOT return the title.

Do NOT explain what you changed.

Do NOT summarize.

Do NOT omit any part of the source.

Do NOT invent information.

Rewrite the complete article now.
"""

    max_tokens = calculate_max_tokens(content)

    # ------------------------------------------------------------
    # Retry generation if validation fails.
    # ------------------------------------------------------------

    for attempt in range(3):

        try:

            raw = call_api(
                prompt,
                max_tokens=max_tokens,
                temperature=0.10
            )

            rewritten = clean_response(raw)

            rewritten = remove_duplicate_title(
                title,
                rewritten
            )

            rewritten = normalize_text(
                rewritten
            )

            issues = validate_rewrite(
                title,
                content,
                rewritten,
                style
            )

            serious_issues = {
                "empty",
                "too_short",
                "too_short_relative_to_source",
                "possible_truncation",
                "missing_source_number",
                "low_sinhala_ratio",
                "excessive_repetition",
                "markdown_artifact",
                "meta_text",
                "generic_or_hallucinated_phrase",
            }

            serious = [
                issue
                for issue in issues
                if issue in serious_issues
            ]

            if not serious:

                return {
                    "title": title,
                    "content": content,
                    "category": category or "",
                    "url": url or "",
                    "date_published": date_published or "",
                    "style": style,
                    "rewritten_text": rewritten,
                }

            print(
                f"\n⚠️ Retry [{style}] "
                f"attempt={attempt + 1} "
                f"issues={issues}",
                flush=True
            )

            # Strengthen prompt after failed validation.
            prompt += f"""

IMPORTANT CORRECTION:

Your previous response failed validation because:
{", ".join(issues)}

Generate the COMPLETE article again.

It is especially important that:
- no sentence is truncated
- all source facts remain
- all source numbers remain
- no new facts are introduced
- no title is included
- no generic filler is used
"""

            time.sleep(
                0.5 * (attempt + 1)
            )

        except Exception as exc:

            if attempt < 2:

                time.sleep(
                    1.5 * (attempt + 1)
                )

                continue

            print(
                f"\n❌ Failed [{style}] "
                f"{url[:60]}: {exc}",
                flush=True
            )

            return None

    return None


# ================================================================
# JSONL LOADING
# ================================================================

def load_source_articles(path):

    """
    Reads train1.jsonl.

    IMPORTANT:
    This reads ONLY the source dataset.
    It does NOT read previous rewritten datasets.
    """

    articles = []

    print(
        f"\n📂 Reading source dataset:\n   {path}",
        flush=True
    )

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        for line_number, line in enumerate(
            f,
            start=1
        ):

            line = line.strip()

            if not line:
                continue

            try:

                row = json.loads(line)

            except json.JSONDecodeError:

                print(
                    f"\n⚠️ Invalid JSON at line "
                    f"{line_number}. Skipping.",
                    flush=True
                )

                continue

            content = row.get(
                "content",
                ""
            )

            if not content:
                continue

            articles.append({
                "title": row.get("title", ""),
                "content": content,
                "category": row.get("category", ""),
                "url": row.get("url", ""),
                "date_published": row.get(
                    "date_published",
                    ""
                ),
            })

    return articles


# ================================================================
# WORKER
# ================================================================

def worker(
    worker_id,
    task_queue,
    output_file,
    output_lock,
    counters,
    counter_lock,
    pbar
):

    while True:

        try:

            task = task_queue.get(
                timeout=3
            )

        except Empty:

            break

        try:

            (
                title,
                content,
                url,
                category,
                date_pub,
                style
            ) = task

            result = process_one(
                title,
                content,
                url,
                category,
                date_pub,
                style,
                worker_id
            )

            if result:

                # ------------------------------------------------
                # WRITE IMMEDIATELY
                # ------------------------------------------------
                with output_lock:

                    output_file.write(
                        json.dumps(
                            result,
                            ensure_ascii=False
                        ) + "\n"
                    )

                    output_file.flush()

                with counter_lock:

                    counters["generated"] += 1

                    counters[style] += 1

            else:

                with counter_lock:

                    counters["failed"] += 1

        except Exception as exc:

            with counter_lock:

                counters["failed"] += 1

            print(
                f"\n❌ Worker {worker_id}: {exc}",
                flush=True
            )

        finally:

            pbar.update(1)

            task_queue.task_done()


# ================================================================
# MAIN
# ================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Fresh Sinhala Style Dataset Generator"
        )
    )

    parser.add_argument(
        "--input",
        default=INPUT_PATH,
        help="Source JSONL dataset"
    )

    parser.add_argument(
        "--output",
        default=OUTPUT_PATH,
        help="Fresh output JSONL dataset"
    )

    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Number of API workers"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit source articles"
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing output before starting"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    args = parser.parse_args()

    random.seed(
        args.seed
    )

    input_path = Path(
        args.input
    )

    output_path = Path(
        args.output
    )

    # ============================================================
    # HEADER
    # ============================================================

    print(
        "\n" + "=" * 72
    )

    print(
        "  FRESH SINHALA STYLE DATASET GENERATOR"
    )

    print(
        "=" * 72
    )

    print(
        f"\n📂 Source dataset:"
    )

    print(
        f"   {input_path}"
    )

    print(
        f"\n📄 NEW output dataset:"
    )

    print(
        f"   {output_path}"
    )

    print(
        "\n⚠️ Previous rewritten datasets are NOT used."
    )

    print(
        "⚠️ Every article is freshly rewritten."
    )

    print(
        "⚠️ Each successful row is written immediately."
    )

    # ============================================================
    # API KEY
    # ============================================================

    if not NVIDIA_API_KEY:

        print(
            "\n❌ NVIDIA_API_KEY is not set."
        )

        print(
            "\nRun:"
        )

        print(
            "export NVIDIA_API_KEY='YOUR_API_KEY'"
        )

        return

    # ============================================================
    # INPUT
    # ============================================================

    if not input_path.exists():

        print(
            f"\n❌ Input file not found:"
            f"\n{input_path}"
        )

        return

    # ============================================================
    # OUTPUT
    # ============================================================

    if args.overwrite and output_path.exists():

        print(
            f"\n🗑️ Removing existing output:"
            f"\n{output_path}"
        )

        output_path.unlink()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # ============================================================
    # LOAD ARTICLES
    # ============================================================

    articles = load_source_articles(
        input_path
    )

    if args.limit:

        articles = articles[
            :args.limit
        ]

    article_count = len(
        articles
    )

    total_tasks = (
        article_count *
        len(ALL_STYLES)
    )

    print(
        f"\n📰 Source articles: "
        f"{article_count:,}"
    )

    print(
        f"🎨 Styles per article: "
        f"{len(ALL_STYLES)}"
    )

    print(
        f"🔢 Total generations: "
        f"{total_tasks:,}"
    )

    # ============================================================
    # CREATE TASK QUEUE
    # ============================================================

    tasks = []

    for article in articles:

        for style in ALL_STYLES:

            tasks.append((
                article["title"],
                article["content"],
                article["url"],
                article["category"],
                article["date_published"],
                style,
            ))

    random.shuffle(
        tasks
    )

    task_queue = Queue()

    for task in tasks:

        task_queue.put(
            task
        )

    # ============================================================
    # COUNTERS
    # ============================================================

    counters = {
        "generated": 0,
        "failed": 0,
    }

    for style in ALL_STYLES:

        counters[style] = 0

    counter_lock = threading.Lock()

    output_lock = threading.Lock()

    # ============================================================
    # PROGRESS BAR
    # ============================================================

    try:

        from tqdm import tqdm

        pbar = tqdm(
            total=total_tasks,
            desc="⚡ Generating",
            unit="article"
        )

    except ImportError:

        class SimpleProgress:

            def __init__(self, total):

                self.total = total

                self.current = 0

            def update(self, n):

                self.current += n

                print(
                    f"\r⚡ "
                    f"{self.current:,}/"
                    f"{self.total:,}",
                    end="",
                    flush=True
                )

            def close(self):

                print()

        pbar = SimpleProgress(
            total_tasks
        )

    # ============================================================
    # START
    # ============================================================

    print(
        "\n" + "=" * 72
    )

    print(
        f"🚀 Starting generation "
        f"with {args.concurrency} workers"
    )

    print(
        "=" * 72
    )

    print(
        "\n💾 Results are being written "
        "to the output file immediately."
    )

    print(
        "👀 You can open/tail the file while generation is running."
    )

    print(
        "\nExample:"
    )

    print(
        f"tail -f {output_path}"
    )

    print()

    start_time = time.time()

    # ============================================================
    # OPEN OUTPUT ONCE
    # ============================================================

    with open(
        output_path,
        "a",
        encoding="utf-8"
    ) as output_file:

        with ThreadPoolExecutor(
            max_workers=args.concurrency
        ) as executor:

            futures = []

            for worker_id in range(
                args.concurrency
            ):

                future = executor.submit(
                    worker,
                    worker_id,
                    task_queue,
                    output_file,
                    output_lock,
                    counters,
                    counter_lock,
                    pbar
                )

                futures.append(
                    future
                )

            # Wait for every queued task.
            task_queue.join()

            # Ensure worker exceptions are surfaced.
            for future in futures:

                try:

                    future.result()

                except Exception as exc:

                    print(
                        f"\n❌ Worker exception: {exc}",
                        flush=True
                    )

    pbar.close()

    elapsed = (
        time.time() -
        start_time
    )

    generated = counters[
        "generated"
    ]

    failed = counters[
        "failed"
    ]

    speed = (
        generated /
        max(elapsed, 1)
    ) * 60

    # ============================================================
    # FINAL REPORT
    # ============================================================

    print(
        "\n" + "=" * 72
    )

    print(
        "  ✅ GENERATION COMPLETE"
    )

    print(
        "=" * 72
    )

    print(
        f"\n📰 Source articles:"
        f" {article_count:,}"
    )

    print(
        f"🔢 Requested generations:"
        f" {total_tasks:,}"
    )

    print(
        f"✅ Successfully generated:"
        f" {generated:,}"
    )

    print(
        f"❌ Failed:"
        f" {failed:,}"
    )

    print(
        f"⏱️ Time:"
        f" {elapsed / 60:.1f} minutes"
    )

    print(
        f"⚡ Speed:"
        f" {speed:.1f} rows/min"
    )

    print(
        f"\n📄 Output:"
        f"\n   {output_path}"
    )

    print(
        "\n📊 Generated per style:"
    )

    for style in ALL_STYLES:

        print(
            f"   {style:<28}"
            f" {counters[style]:>7,}"
        )

    print(
        "\n" + "=" * 72
    )

    print(
        "  IMPORTANT"
    )

    print(
        "=" * 72
    )

    print(
        "\nThe output is a completely fresh dataset."
    )

    print(
        "It was generated directly from train1.jsonl."
    )

    print(
        "No old rewritten dataset was used."
    )

    print(
        "\nBefore training, inspect a random sample "
        "for factual preservation and grammar."
    )


# ================================================================
# ENTRY POINT
# ================================================================

if __name__ == "__main__":

    main()