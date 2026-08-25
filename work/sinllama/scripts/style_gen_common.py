#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
style_gen_common.py
====================
Shared engine behind the 5 generate_style_v2_<style>.py scripts.

Each of those 5 scripts targets ONE style and calls run(STYLE_ID,
SHARD_INDEX) below. Splitting the 5 styles into 5 disjoint shards of
train1.jsonl (by SHARD_INDEX out of SHARD_COUNT) means all 5 scripts
can be launched in parallel, in separate terminals, with zero risk of
two scripts rewriting the same source article.

Why a fresh generator instead of reusing Correct_style_dataset.py's
approach: clean_style_dataset.py's own docstring documents that the
DeepSeek correction pass which produced style_dataset3_corrected.jsonl
hard-forced an exact opening/closing sentence into nearly every
editorial/youth/feature row, and separately corrupted 4-11% of rows
with invalid Sinhala grapheme sequences. The generation prompts below
(ported from generate_style_dataset.py, which never had either defect)
explicitly forbid forced fixed endings per style. This script does not
run a second "correction" pass at all - one careful generation call
per article, validated and retried on the spot, is the fix.

Output schema (matches what train_style.py's convert_record() expects,
see work/sinllama/scripts/train_style.py):
    {title, content, category, url, date_published, style, rewritten_text}

Usage (from within a style-specific thin script):
    export NVIDIA_API_KEY='...'
    python generate_style_v2_formal.py --concurrency 8

Resumable: rows already present in the output file (by url) are
skipped and counted toward the target on restart.
"""

import os
import re
import json
import time
import random
import argparse
import threading
from pathlib import Path
from collections import Counter
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor

import requests


# ================================================================
# CONFIGURATION
# ================================================================

# DO NOT hard-code the API key here. Set it before running:
#   export NVIDIA_API_KEY="YOUR_KEY"
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "").strip()

BASE_URL = "https://integrate.api.nvidia.com/v1"
INVOKE_URL = f"{BASE_URL}/chat/completions"
MODEL_NAME = "openai/gpt-oss-120b"

INPUT_PATH = "/home/jovyan/style_rewriter/data/train1.jsonl"
DATA_DIR = "/home/jovyan/style_rewriter/data"

SHARD_COUNT = 5          # one shard per style script, kept disjoint
DEFAULT_TARGET = 2000    # rows per style
CANDIDATE_MULTIPLIER = 3  # candidate pool = target * this, headroom for validation failures

# Set to False process-wide the first time the deployment 400s on it
# (see call_api). Module-level so the cost of a wrong guess is one
# wasted call per process, not one per article.
_reasoning_effort_supported = True


# ================================================================
# SYSTEM PROMPT (shared fact-preservation contract, all styles)
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
31. කිසිදු ශෛලියක් සඳහා fixed/forced opening sentence හෝ
    fixed/forced closing sentence නොයොදන්න - එය මුල් article එකේ
    ස්වභාවික අන්තර්ගතයට ගැළපෙන විට පමණක්, ස්වභාවිකව පැන නගින
    එකක් විය යුතුය.
"""


# ================================================================
# PER-STYLE INSTRUCTIONS
# (verbatim from generate_style_dataset.py - proven not to force
# fixed opening/closing sentences)
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
""",
}


# ================================================================
# ADAPTIVE RATE LIMITER
#
# Each of the 5 generate_style_v2_*.py scripts runs as its own OS
# process (no shared memory), so this limiter is per-process. It
# starts conservative and self-tunes: ramps RPM up slowly on a
# success streak, cuts it hard (and honors Retry-After) on a 429.
# Running 5 of these at once means the *combined* request rate still
# adapts to whatever the account's real ceiling is - each process
# just discovers its own ~1/5 share of it independently, instead of
# every worker retrying on the same fixed backoff schedule and
# re-colliding in lockstep (which is what a bare exponential backoff
# with no jitter does under this much concurrency).
# ================================================================

class RateLimiter:
    def __init__(self, rpm, max_rpm=None, min_rpm=2,
                 ramp_every=15, ramp_factor=1.15, backoff_factor=0.5):
        self.rpm = max(rpm, 1)
        self.max_rpm = max_rpm or (rpm * 6)
        self.min_rpm = min_rpm
        self.ramp_every = ramp_every
        self.ramp_factor = ramp_factor
        self.backoff_factor = backoff_factor
        self.success_streak = 0
        self.min_interval = 60.0 / self.rpm
        self.lock = threading.Lock()
        self.next_allowed = 0.0
        self.pause_until = 0.0

    def wait(self):
        while True:
            with self.lock:
                now = time.monotonic()
                deadline = max(self.next_allowed, self.pause_until)
                gap = deadline - now
                if gap <= 0:
                    # Small jitter so concurrent workers in this process
                    # don't all wake and fire on the exact same tick.
                    self.next_allowed = now + self.min_interval + random.uniform(0, 0.15)
                    return
            time.sleep(min(gap, 1.0))

    def report_success(self):
        with self.lock:
            self.success_streak += 1
            if self.success_streak >= self.ramp_every and self.rpm < self.max_rpm:
                old = self.rpm
                self.rpm = min(self.max_rpm, self.rpm * self.ramp_factor)
                self.min_interval = 60.0 / self.rpm
                self.success_streak = 0
                if int(self.rpm) != int(old):
                    print(f"\n[rate] ramping up: {old:.1f} -> {self.rpm:.1f} rpm", flush=True)

    def report_429(self, retry_after):
        with self.lock:
            self.pause_until = max(self.pause_until, time.monotonic() + retry_after)
            old = self.rpm
            self.rpm = max(self.min_rpm, self.rpm * self.backoff_factor)
            self.min_interval = 60.0 / self.rpm
            self.success_streak = 0
            print(
                f"\n[rate] 429 -> pausing {retry_after:.1f}s, "
                f"cutting rate {old:.1f} -> {self.rpm:.1f} rpm",
                flush=True,
            )


# ================================================================
# API CALL
# ================================================================

def call_api(prompt, rate_limiter, max_tokens=4096, temperature=0.25, max_retries=6):

    global _reasoning_effort_supported

    if not NVIDIA_API_KEY:
        raise RuntimeError(
            "NVIDIA_API_KEY is not set. Run: export NVIDIA_API_KEY='YOUR_KEY'"
        )

    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.90,
        # Discourage the literal phrase loops that were dominating retries
        # (see has_phrase_repetition) at the source, rather than only
        # detecting and re-generating after the fact.
        "frequency_penalty": 0.3,
        "presence_penalty": 0.2,
        "stream": False,
    }

    if _reasoning_effort_supported:
        # gpt-oss-120b is a reasoning model; a lower reasoning budget cuts
        # latency per call substantially. Self-disables process-wide the
        # first time a deployment rejects the field (see 400 handling
        # below), so a wrong guess only costs one wasted call, not one
        # per article.
        payload["reasoning_effort"] = "low"

    for attempt in range(max_retries):
        rate_limiter.wait()
        try:
            response = requests.post(
                INVOKE_URL, headers=headers, json=payload, timeout=240,
            )

            if response.status_code == 200:
                data = response.json()
                message = data["choices"][0].get("message", {})

                # openai/gpt-oss-120b is a reasoning model; NIM deployments
                # normally return only the final answer in `content`, but
                # fall back to `reasoning_content` defensively in case a
                # given deployment splits them.
                text = (message.get("content") or "").strip()
                if not text:
                    text = (message.get("reasoning_content") or "").strip()

                if not text:
                    raise RuntimeError(
                        f"Empty API response: {json.dumps(data)[:500]}"
                    )

                rate_limiter.report_success()
                return text

            if response.status_code == 400 and "reasoning_effort" in payload:
                print(
                    f"\n[warn] {MODEL_NAME} rejected reasoning_effort - "
                    f"disabling it for the rest of this run: "
                    f"{response.text[:200]}",
                    flush=True,
                )
                payload.pop("reasoning_effort", None)
                _reasoning_effort_supported = False
                continue

            if response.status_code == 429:
                retry_after = None
                header_val = response.headers.get("Retry-After")
                if header_val:
                    try:
                        retry_after = float(header_val)
                    except ValueError:
                        pass
                if retry_after is None:
                    retry_after = min(5 * (attempt + 1), 30)
                rate_limiter.report_429(retry_after)
                continue

            if response.status_code >= 500:
                wait = min(2 ** (attempt + 1), 20) + random.uniform(0, 1)
                print(
                    f"\n[warn] Server error {response.status_code}. "
                    f"Retrying in {wait:.1f}s...",
                    flush=True,
                )
                time.sleep(wait)
                continue

            raise RuntimeError(
                f"API {response.status_code}: {response.text[:500]}"
            )

        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                time.sleep(3 + random.uniform(0, 1))
                continue
            raise

        except requests.exceptions.ConnectionError:
            if attempt < max_retries - 1:
                time.sleep(3 + random.uniform(0, 1))
                continue
            raise

    raise RuntimeError(f"API failed after {max_retries} attempts")


# ================================================================
# TEXT CLEANUP (ported from generate_style_dataset.py)
# ================================================================

def clean_response(text):
    if not text:
        return ""

    text = text.strip()
    text = re.sub(r"^```(?:text|sinhala)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)

    cleaned = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if re.match(
            r"^(rewritten article|rewritten text|here is|here's|"
            r"මෙන්න නැවත|නැවත ලියූ ලිපිය)",
            stripped,
            flags=re.IGNORECASE,
        ):
            continue
        stripped = re.sub(r"^#{1,6}\s*", "", stripped)
        stripped = stripped.replace("**", "")
        cleaned.append(stripped)

    return "\n".join(cleaned).strip()


def remove_duplicate_title(title, rewritten):
    if not title or not rewritten:
        return rewritten

    title = title.strip()
    text = rewritten.strip()

    if text.startswith(title):
        remainder = text[len(title):].lstrip(" :-–—\n")
        if remainder:
            return remainder

    bold_title = f"**{title}**"
    if text.startswith(bold_title):
        remainder = text[len(bold_title):].lstrip(" :-–—\n")
        if remainder:
            return remainder

    return rewritten


def normalize_text(text):
    if not text:
        return ""
    text = text.replace("­", "")
    text = text.replace("\r\n", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def looks_truncated(original, rewritten):
    if not rewritten:
        return True

    text = rewritten.strip()

    if text.endswith("...") or text.endswith(".."):
        return True

    valid_endings = (
        ".", "!", "?", "।", "යි", "වේ", "බව", "තිබේ", "කරයි", "විය",
        "ඇත", "තිබුණි", "තිබුණා", "කළේය", "කර තිබේ", "වූ බව",
    )

    if not text.endswith(valid_endings):
        if len(original) > 500:
            return True

    original_len = len(original)
    rewritten_len = len(rewritten)

    if original_len >= 500:
        ratio = rewritten_len / original_len
        if ratio < 0.55:
            return True

    return False


def has_word_repetition(text):
    words = re.findall(r"[඀-෿]+", text)

    if len(words) < 20:
        return False

    counts = Counter(words)
    ignored = {
        "සඳහන්", "විසින්", "සඳහා", "ලෙස", "අතර", "පිළිබඳ", "කිරීම", "තිබේ",
    }

    for word, count in counts.items():
        if len(word) > 3 and count >= 12 and word not in ignored:
            return True

    return False


def has_phrase_repetition(text):
    """Flags genuine degenerate loops (a run of words the model repeats
    verbatim later on), while trying not to fire on ordinary journalistic
    Sinhala, which often reuses short 3-word attribution phrases (dates,
    "... බව පොලිසිය සඳහන් කරයි" style constructions) coincidentally.
    A 4-token verbatim repeat is a much rarer coincidence than a 3-token
    one, so this checks 4 tokens rather than the 3 used previously."""
    normalized = re.sub(r"\s+", " ", text)
    if re.search(r"(\S+\s+\S+\s+\S+\s+\S+).*\1", normalized):
        return True
    return False


def sinhala_ratio(text):
    sinhala = len(re.findall(r"[඀-෿]", text))
    letters = len(re.findall(r"[A-Za-z඀-෿]", text))
    if letters == 0:
        return 0
    return sinhala / letters


def extract_numbers(text):
    return re.findall(r"\d+(?:[.,]\d+)*", text)


def validate_numbers(original, rewritten):
    original_numbers = Counter(extract_numbers(original))
    rewritten_numbers = Counter(extract_numbers(rewritten))
    for number, count in original_numbers.items():
        if rewritten_numbers[number] < count:
            return False
    return True


DANGEROUS_PHRASES = [
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
    # forced correction-pass templates (see clean_style_dataset.py) -
    # never allow a fresh generation to reproduce these either.
    "විශ්ලේෂණය කරන විට",
    "ඉදිරි ක්‍රියාමාර්ග පිළිබඳව සාකච්ඡා කළ යුතු කාලය එළඹ ඇත",
    "දන්නවද?",
    "ඇහුවද?",
    "මේක අහන්න!",
    "අනිවාර්යයෙන්ම දැනගන්න",
]


def validate_rewrite(title, original, rewritten, style):
    issues = []

    original = normalize_text(original)
    rewritten = normalize_text(rewritten)

    if not rewritten:
        return ["empty"]

    if len(rewritten) < 50:
        issues.append("too_short")

    ratio = len(rewritten) / max(len(original), 1)
    if ratio < 0.55:
        issues.append("too_short_relative_to_source")
    if ratio > 1.80:
        issues.append("too_long")

    if sinhala_ratio(rewritten) < 0.60:
        issues.append("low_sinhala_ratio")

    if looks_truncated(original, rewritten):
        issues.append("possible_truncation")

    if not validate_numbers(original, rewritten):
        issues.append("missing_source_number")

    if has_word_repetition(rewritten):
        issues.append("word_repetition")

    if has_phrase_repetition(rewritten):
        issues.append("phrase_repetition")

    if "```" in rewritten:
        issues.append("markdown_artifact")

    if rewritten.startswith("#"):
        issues.append("markdown_heading")

    meta_patterns = ["මෙන්න", "නැවත ලියූ ලිපිය", "rewritten article", "here is the rewritten"]
    lower = rewritten.lower()
    for pattern in meta_patterns:
        if pattern.lower() in lower:
            issues.append("meta_text")
            break

    for phrase in DANGEROUS_PHRASES:
        if phrase in rewritten:
            issues.append("generic_hallucinated_or_forced_template_phrase")
            break

    return list(set(issues))


SERIOUS_ISSUES = {
    "empty",
    "too_short",
    "too_short_relative_to_source",
    "possible_truncation",
    "missing_source_number",
    "low_sinhala_ratio",
    "word_repetition",
    "phrase_repetition",
    "markdown_artifact",
    "meta_text",
    "generic_hallucinated_or_forced_template_phrase",
}


def calculate_max_tokens(content):
    chars = len(content)
    estimated = int(chars / 1.4) + 1200
    estimated = max(estimated, 3000)
    estimated = min(estimated, 8192)
    return estimated


# ================================================================
# SOURCE LOADING + SHARDING
# ================================================================

def load_source_articles(path):
    articles = []
    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            content = row.get("content", "")
            if not content or not content.strip():
                continue

            articles.append({
                "title": row.get("title", ""),
                "content": content,
                "category": row.get("category", ""),
                "url": row.get("url", ""),
                "date_published": row.get("date_published", ""),
            })

    return articles


def select_shard_candidates(articles, shard_index, shard_count, candidate_count, seed):
    """Disjoint slice of `articles` for this shard, then locally shuffled
    so a style script doesn't just process consecutive scrape-order rows."""
    shard = articles[shard_index::shard_count]
    rng = random.Random(seed)
    rng.shuffle(shard)
    return shard[:candidate_count]


def load_already_done(output_path):
    done_urls = set()
    valid_count = 0
    if not Path(output_path).exists():
        return done_urls, valid_count

    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            rewritten = row.get("rewritten_text", "")
            if row.get("url") and rewritten and rewritten.strip():
                done_urls.add(row["url"])
                valid_count += 1

    return done_urls, valid_count


# ================================================================
# PER-ARTICLE PROCESSING
# ================================================================

def process_one(title, content, url, category, date_published, style, rate_limiter):
    content = normalize_text(content)
    title = normalize_text(title)

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

    for attempt in range(3):
        try:
            raw = call_api(prompt, rate_limiter, max_tokens=max_tokens)
            rewritten = clean_response(raw)
            rewritten = remove_duplicate_title(title, rewritten)
            rewritten = normalize_text(rewritten)

            issues = validate_rewrite(title, content, rewritten, style)
            serious = [issue for issue in issues if issue in SERIOUS_ISSUES]

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

            print(f"\n[retry] [{style}] attempt={attempt + 1} issues={issues}", flush=True)

            prompt += f"""

IMPORTANT CORRECTION:

Your previous response failed validation because:
{", ".join(issues)}

Generate the COMPLETE article again. It is especially important that:
- no sentence is truncated
- all source facts remain
- all source numbers remain
- no new facts are introduced
- no title is included
- no forced/fixed opening or closing sentence is used
- no generic filler is used
"""
            time.sleep(0.5 * (attempt + 1))

        except Exception as exc:
            if attempt < 2:
                time.sleep(1.5 * (attempt + 1))
                continue
            print(f"\n[fail] [{style}] {url[:60]}: {exc}", flush=True)
            return None

    return None


# ================================================================
# WORKER
# ================================================================

def worker(task_queue, output_file, output_lock, counters, counter_lock, target, rate_limiter, pbar):
    while True:
        with counter_lock:
            if counters["written"] >= target:
                break

        try:
            task = task_queue.get(timeout=3)
        except Empty:
            break

        try:
            title, content, url, category, date_pub, style = task

            result = process_one(title, content, url, category, date_pub, style, rate_limiter)

            with counter_lock:
                target_already_met = counters["written"] >= target

            if result and target_already_met:
                # Target already met by another worker while we were
                # mid-flight; discard this extra row (finally: below
                # still runs task_done() exactly once for this item).
                pass

            elif result:
                with output_lock:
                    output_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                    output_file.flush()

                with counter_lock:
                    counters["written"] += 1
                    counters["generated"] += 1
                    pbar.update(1)
            else:
                with counter_lock:
                    counters["failed"] += 1

        except Exception as exc:
            with counter_lock:
                counters["failed"] += 1
            print(f"\n[error] worker: {exc}", flush=True)

        finally:
            task_queue.task_done()


# ================================================================
# MAIN ENTRY POINT (called by each thin generate_style_v2_*.py script)
# ================================================================

def run(style_id, shard_index, default_target=DEFAULT_TARGET, shard_count=SHARD_COUNT):
    if style_id not in STYLE_INSTRUCTIONS:
        raise ValueError(f"Unknown style_id: {style_id}")

    parser = argparse.ArgumentParser(
        description=f"Generate {style_id} rewrites via {MODEL_NAME} (NVIDIA-hosted)."
    )
    parser.add_argument("--input", default=INPUT_PATH)
    parser.add_argument(
        "--output",
        default=str(Path(DATA_DIR) / f"style_dataset_v2_{style_id}.jsonl"),
    )
    parser.add_argument("--target", type=int, default=default_target)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--seed", type=int, default=hash(style_id) % (2**31))
    parser.add_argument(
        "--rpm", type=int, default=6,
        help="Starting requests/minute for this script's adaptive rate "
             "limiter. Kept low by default because all 5 style scripts "
             "typically run at once against the same account - each "
             "ramps up on its own once it sees the real ceiling isn't "
             "being hit.",
    )
    parser.add_argument(
        "--max-rpm", type=int, default=None,
        help="Hard ceiling for this script's rate limiter (default: rpm*6).",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Delete existing output before starting (loses resume progress).",
    )
    args = parser.parse_args()

    if not NVIDIA_API_KEY:
        print("\n[error] NVIDIA_API_KEY is not set.")
        print("Run: export NVIDIA_API_KEY='YOUR_API_KEY'")
        return

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"\n[error] Input file not found: {input_path}")
        return

    if args.overwrite and output_path.exists():
        output_path.unlink()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"  STYLE DATASET GENERATOR v2  -  {style_id}")
    print(f"  model={MODEL_NAME}  base_url={BASE_URL}")
    print("=" * 72)

    done_urls, already_valid = load_already_done(output_path)
    if already_valid:
        print(f"\nResuming: {already_valid}/{args.target} rows already in {output_path.name}")

    remaining = args.target - already_valid
    if remaining <= 0:
        print(f"\nTarget already met ({already_valid} >= {args.target}). Nothing to do.")
        return

    print(f"\nLoading source articles from {input_path} ...")
    all_articles = load_source_articles(input_path)
    print(f"Loaded {len(all_articles):,} usable source articles total.")

    candidate_count = remaining * CANDIDATE_MULTIPLIER
    candidates = select_shard_candidates(
        all_articles, shard_index, shard_count, candidate_count, args.seed
    )
    candidates = [c for c in candidates if c["url"] not in done_urls]

    print(
        f"Shard {shard_index}/{shard_count}: "
        f"{len(candidates):,} candidate articles queued "
        f"(need {remaining:,} more successes)."
    )

    task_queue = Queue()
    for article in candidates:
        task_queue.put((
            article["title"], article["content"], article["url"],
            article["category"], article["date_published"], style_id,
        ))

    counters = {"written": already_valid, "generated": 0, "failed": 0}
    counter_lock = threading.Lock()
    output_lock = threading.Lock()
    rate_limiter = RateLimiter(args.rpm, max_rpm=args.max_rpm)

    print(
        f"Rate limiter: starting {args.rpm} rpm "
        f"(ceiling {args.max_rpm or args.rpm * 6} rpm), "
        f"concurrency={args.concurrency}"
    )

    try:
        from tqdm import tqdm
        pbar = tqdm(total=args.target, initial=already_valid, desc=f"{style_id}", unit="row")
    except ImportError:
        class SimpleProgress:
            def __init__(self, total, initial):
                self.total = total
                self.current = initial
            def update(self, n):
                self.current += n
                print(f"\r{self.current:,}/{self.total:,}", end="", flush=True)
            def close(self):
                print()
        pbar = SimpleProgress(args.target, already_valid)

    start_time = time.time()

    with open(output_path, "a", encoding="utf-8") as output_file:
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = [
                executor.submit(
                    worker, task_queue, output_file, output_lock,
                    counters, counter_lock, args.target, rate_limiter, pbar,
                )
                for _ in range(args.concurrency)
            ]

            task_queue.join()

            for future in futures:
                try:
                    future.result()
                except Exception as exc:
                    print(f"\n[error] worker exception: {exc}", flush=True)

    pbar.close()

    elapsed = time.time() - start_time

    print("\n" + "=" * 72)
    print(f"  DONE  -  {style_id}")
    print("=" * 72)
    print(f"Rows in output now : {counters['written']:,} / {args.target:,}")
    print(f"Newly generated    : {counters['generated']:,}")
    print(f"Failed             : {counters['failed']:,}")
    print(f"Elapsed            : {elapsed / 60:.1f} min")
    print(f"Output             : {output_path}")

    if counters["written"] < args.target:
        print(
            f"\n[warn] Short of target by {args.target - counters['written']:,} rows. "
            "Re-run this script (it resumes automatically) - candidate pool may "
            "have been exhausted; widen --target's candidate pool via a smaller "
            "shard_count or check the [fail]/[retry] log lines above for a "
            "systemic issue."
        )
