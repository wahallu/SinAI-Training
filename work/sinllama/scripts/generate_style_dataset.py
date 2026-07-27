import os
import json
import time
import random
import argparse
import requests
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
import threading
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor
os.environ["NVIDIA_API_KEY"] = "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"

load_dotenv()

INVOKE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
DEFAULT_MODEL = "qwen/qwen3-next-80b-a3b-instruct"

SYSTEM_PROMPT = """You are an expert Sinhala journalism writer who rewrites news articles
into different stylistic registers while preserving every fact exactly.
This is a REWRITE task, not a summarization task: the output must cover every
fact and be approximately the same length as the source article — never
compress it into a shorter summary.
Respond only with the rewritten article text — no title, no preamble, no
explanation, no markdown, no labels like "Article:" or "Rewritten:"."""

# Junk patterns that sometimes leak into scraped article text (CMS embed
# shortcodes, leftover HTML, etc.) and should never reach the model.
import re

_JUNK_PATTERNS = [
    re.compile(r"\[ot-video[^\]]*\]?", re.IGNORECASE),
    re.compile(r"\[/?ot-video[^\]]*\]?", re.IGNORECASE),
    re.compile(r"<[^>]+>"),  # stray HTML tags
    re.compile(r"\u00ad"),  # soft hyphen
]


def clean_content(text: str) -> str:
    for pattern in _JUNK_PATTERNS:
        text = pattern.sub("", text)
    return text.strip()

# -----------------------------------------------------------------------
# Canonical style instructions. Keys MUST stay byte-for-byte identical to
# whatever is used at inference/training time (build_prompt / style tag).
# -----------------------------------------------------------------------
STYLE_INSTRUCTIONS = {
    "style_1_formal_news": """
ඔබ සිංහල ප්‍රධාන පුවත් වාර්තාකරණ විශේෂඥයෙකි.

පහත නීති අනිවාර්යයෙන්ම අනුගමනය කරන්න:
1. ලිපියේ ඇති කරුණු පමණක් භාවිතා කරන්න. නව තොරතුරු එකතු නොකරන්න.
2. පුද්ගලයන්, ස්ථාන, සංඛ්‍යා නිවැරදිව තබා ගන්න. කිසිවක් වෙනස් නොකරන්න.
3. සියලු වාක්‍ය වෛෂයික (objective) විය යුතුය - කිසිදු අදහසක් හෝ විශ්ලේෂණයක් එක් නොකරන්න.
4. හැකි සෑම තැනකම කර්තෘ විරහිත වාක්‍ය (passive voice) භාවිතා කරන්න.
5. උපුල්ලංකෘත පිරමිඩ ව්‍යුහය අනුගමනය කරන්න - වැදගත්ම කරුණ මුලින්ම දක්වන්න.
6. මුල් ලිපියේ ඇති සියලුම කරුණු එලෙසින්ම තබා ගන්න.
7. නැවත ලියූ ලිපියේ දිග මුල් ලිපියේ දිගට ආසන්න විය යුතුය (කෙටි කිරීම හෝ දිගු කිරීම නොකරන්න).
8. මුල් ලිපියේ ඉංග්‍රීසි වචන නොතිබුනේ නම්, ඉංග්‍රීසි වචන එකතු නොකරන්න.
9. අකුරු වැරදි නොකරන්න. සියලු වචන නිවැරදිව අක්ෂර වින්‍යාසයෙන් ලියන්න.
10. වාක්‍ය සම්පූර්ණ විය යුතුය - අතරමගදී කපා නොදමන්න.
{shared_rules}
Article:
{content}
""",

    "style_2_editorial": """
ඔබ සිංහල කතුවැකි/විශ්ලේෂණ විශේෂඥයෙකි.

පහත නීති අනිවාර්යයෙන්ම අනුගමනය කරන්න:
1. "විශ්ලේෂණය කරන විට" යන වදනින් ලිපිය ආරම්භ කරන්න.
2. පළමු පුරුෂ බහුවචනය භාවිතා කරන්න: "අපි", "අපට", "අප" යනාදිය.
3. විශ්ලේෂණාත්මක භාෂාවක් භාවිතා කරන්න: "සැලකිය යුතුය", "සමස්තයක් ලෙස", "විශේෂයෙන්ම", "එබැවින්" වැනි.
4. මුල් ලිපියේ කරුණු මත පදනම්ව දෘෂ්ටිකෝණයක් ප්‍රකාශ කරන්න.
5. "මෙම සියලු කරුණු සමස්තයක් ලෙස සලකන කල, ඉදිරි ක්‍රියාමාර්ග පිළිබඳව සාකච්ඡා කළ යුතු කාලය එළඹ ඇත" යන වදනින් ලිපිය අවසන් කරන්න.
6. මුල් ලිපියේ ඇති සියලුම කරුණු නිවැරදිව තබා ගන්න - කිසිදු නව තොරතුරක් එකතු නොකරන්න.
7. නැවත ලියූ ලිපියේ දිග මුල් ලිපියේ දිගට ආසන්න විය යුතුය.
8. අකුරු වැරදි නොකරන්න. සියලු වචන නිවැරදිව අක්ෂර වින්‍යාසයෙන් ලියන්න.
9. වාක්‍ය සම්පූර්ණ විය යුතුය - අතරමගදී කපා නොදමන්න.
{shared_rules}
Article:
{content}
""",

    "style_3_sports": """
ඔබ සිංහල ක්‍රීඩා පුවත්පත් කලාවේ විශේෂඥයෙකි.

පහත නීති අනිවාර්යයෙන්ම අනුගමනය කරන්න:
1. වඩාත්ම නාටකීය කරුණ මුලින්ම ගෙන එන්න - එය ලිපියේ පළමු වාක්‍යය විය යුතුය.
2. ඉහළ ශක්තියෙන් යුත් භාෂාව භාවිතා කරන්න. උදාහරණ: "ක්‍රීඩාංගණය උණුසුම් වාතාවරණයක් ගෙන ආවා!", "තීරණාත්මක ජයග්රහණයක්!", "විශිෂ්ට දක්ෂතාවයක්!"
3. ක්‍රියාකාරී ක්‍රියා පද භාවිතා කරන්න: "පහර දුන්නා", "ජයග්‍රහණය කළා", "ප්‍රහාරය එල්ල කළා", "තක්කඩි ලෙස පැන ගත්තා" වැනි.
4. වේගවත් රිද්මයක් සහ උද්වේගකර භාෂාවක් පවත්වා ගන්න.
5. මුල් ලිපියේ ඇති සියලුම කරුණු, නම්, ස්ථාන, සංඛ්‍යා නිවැරදිව තබා ගන්න. කිසිවක් වෙනස් නොකරන්න.
6. උද්ගාමී බව පෙන්වීමට උපරිම වශයෙන් එක් විස්මයන් සලකුණක් ( ! ) පමණක් භාවිතා කරන්න. එකට වඩා භාවිතා නොකරන්න.
7. මුල් ලිපියේ ඉංග්‍රීසි වචන නොතිබුනේ නම්, ඉංග්‍රීසි වචන එකතු නොකරන්න.
8. නැවත ලියූ ලිපියේ දිග මුල් ලිපියේ දිගට ආසන්න විය යුතුය.
9. අකුරු වැරදි නොකරන්න. සියලු වචන නිවැරදිව අක්ෂර වින්‍යාසයෙන් ලියන්න.
10. වාක්‍ය සම්පූර්ණ විය යුතුය - අතරමගදී කපා නොදමන්න.
{shared_rules}
Article:
{content}
""",

    "style_4_youth": """
ඔබ තරුණ පාඨකයන් සඳහා ලියන සිංහල ලේඛකයෙකි.

පහත නීති අනිවාර්යයෙන්ම අනුගමනය කරන්න:
1. සෘජු ආමන්ත්‍රණයකින් ලිපිය ආරම්භ කරන්න: "දන්නවද?" හෝ "ඇහුවද?" හෝ "මේක අහන්න!"
2. අවිධිමත් සිංහල භාෂාවක් භාවිතා කරන්න: "ගොඩක්", "ටිකක්", "හිතෙනවා", "නෙවෙයි", "ඉන්නවා" වැනි.
3. කෙටි, තියුණු වාක්‍ය භාවිතා කරන්න. එක් වාක්‍යයක අදහස් 2-3කට වඩා නොදමන්න.
4. "ඒ නිසා යාලුවනේ, මේ ගැන අනිවාර්යයෙන්ම දැනගන්න!" යන වදනින් ලිපිය අවසන් කරන්න.
5. මුල් ලිපියේ ඇති සියලුම කරුණු, නම්, ස්ථාන, සංඛ්‍යා නිවැරදිව තබා ගන්න.
6. නැවත ලියූ ලිපියේ දිග මුල් ලිපියේ දිගට ආසන්න විය යුතුය - දැඩි ලෙස කෙටි නොකරන්න.
7. මුල් ලිපියේ ඉංග්‍රීසි වචන නොතිබුනේ නම්, ඉංග්‍රීසි වචන එකතු නොකරන්න.
8. අකුරු වැරදි නොකරන්න. සියලු වචන නිවැරදිව අක්ෂර වින්‍යාසයෙන් ලියන්න.
9. වාක්‍ය සම්පූර්ණ විය යුතුය - අතරමගදී කපා නොදමන්න.
{shared_rules}
Article:
{content}
""",

    "style_5_feature": """
ඔබ සිංහල විශේෂාංග/කතාන්දර ලේඛන විශේෂඥයෙකි.

පහත නීති අනිවාර්යයෙන්ම අනුගමනය කරන්න:
1. කතාන්දර ආරම්භයකින් ලිපිය ආරම්භ කරන්න. උදා: "එක් දිනක, මෙම සිදුවීම සිදු විය" හෝ "අඳුරු උදෑසනක, ..." හෝ "සුළඟ හමා යන විට, ..."
2. දෘශ්‍ය විස්තර භාවිතා කරන්න - ස්ථානය, කාලගුණය, වාතාවරණය පිළිබඳ විස්තර එකතු කරන්න.
3. මානුෂීය දෘෂ්ටිකෝණයක් ගෙන එන්න - සිදුවීම මිනිසුන්ට බලපෑ ආකාරය විස්තර කරන්න.
4. හැකි සෑම තැනකම වර්තමාන කාල ක්‍රියාපද භාවිතා කරන්න.
5. "මෙම කතාව අනාගත පරම්පරාවට ද ආදර්ශයක් වනු ඇත" යන වදනින් ලිපිය අවසන් කරන්න.
6. මුල් ලිපියේ ඇති සියලුම කරුණු, නම්, ස්ථාන, සංඛ්‍යා නිවැරදිව තබා ගන්න.
7. කිසිදු නව තොරතුරක් එකතු නොකරන්න.
8. නැවත ලියූ ලිපියේ දිග මුල් ලිපියේ දිගට ආසන්න විය යුතුය.
9. අකුරු වැරදි නොකරන්න. සියලු වචන නිවැරදිව අක්ෂර වින්‍යාසයෙන් ලියන්න.
10. වාක්‍ය සම්පූර්ණ විය යුතුය - අතරමගදී කපා නොදමන්න.
{shared_rules}
Article:
{content}
""",
}

# ✅ NEW: appended to EVERY style's rule list, addressing specific
# correctness failures found by manual review of v09 outputs:
#   - dropped/flipped gender honorifics (මහත්මිය -> මහතා on a real person)
#   - quoted text altered, including an English word inserted into a
#     direct Sinhala quote
#   - non-Sinhala symbols (e.g. "&") appearing where none existed
#   - a fabricated specific fact (an invented "වසර දහයක්" duration)
#   - one style's mandated opening/closing phrase bleeding into another
#     style's output for the same article
SHARED_ADDITIONAL_RULES = """
11. මුල් ලිපියේ ඇති පුද්ගලයන්ගේ ගරු නාම (මහතා/මහත්මිය/මිය) සහ ස්ත්‍රී-පුරුෂ භාවය නිවැරදිව එලෙසින්ම තබා ගන්න - කිසිවිටක වෙනස් නොකරන්න.
12. උද්ධෘත ලකුණු ඇතුළත ("...") ඇති වචන මුල් ලිපියේ පරිදි එලෙසින්ම, කිසිදු වෙනසක් නොකර, පරිවර්තනයක් නොකර තබා ගන්න.
13. මුල් ලිපියේ නොතිබූ කිසිදු සංකේතයක් (& % # වැනි) හෝ ඉලක්කමක්, කාල සීමාවක්, දිනයක් අලුතින් එකතු නොකරන්න.
14. මෙම විශේෂිත style එකට පමණක් අදාළ ආරම්භක/අවසාන වදන් පමණක් භාවිතා කරන්න - වෙනත් style එකකට අයත් ආරම්භක හෝ අවසාන වදන් මෙහි භාවිතා නොකරන්න.
"""

STYLE_INSTRUCTIONS = {
    style: template.format(shared_rules=SHARED_ADDITIONAL_RULES, content="{content}")
    for style, template in STYLE_INSTRUCTIONS.items()
}

ALL_STYLES = list(STYLE_INSTRUCTIONS.keys())


def load_api_key():
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError("NVIDIA_API_KEY not found in environment.")
    return api_key


def load_processed_pairs(output_path: Path) -> set:
    """Returns a set of (url, style) tuples already written to output."""
    processed = set()
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if "url" in data and "style" in data:
                        processed.add((data["url"], data["style"]))
                except Exception:
                    continue
    return processed


def build_prompt(style: str, content: str) -> str:
    template = STYLE_INSTRUCTIONS[style]
    return template.format(content=content)


# Required opening/closing markers per style, used to detect truncation.
# Only styles with a mandated fixed phrase are checked; formal/sports don't
# have one, so they're skipped (None).
REQUIRED_CLOSING = {
    "style_1_formal_news": None,
    "style_2_editorial": "ඉදිරි ක්‍රියාමාර්ග පිළිබඳව සාකච්ඡා කළ යුතු කාලය එළඹ ඇත",
    "style_3_sports": None,
    "style_4_youth": "අනිවාර්යයෙන්ම දැනගන්න",
    "style_5_feature": "අනාගත පරම්පරාවට ද ආදර්ශයක් වනු ඇත",
}

# Catches a short run of characters immediately repeated (e.g. "වලවල",
# "දීදී", "රරර") - a stutter/duplication artifact seen in generation,
# not a legitimate Sinhala spelling pattern.
_STUTTER_PATTERN = re.compile(r"([\u0D80-\u0DFF]{2,4})\1")

# ✅ NEW: detectors for correctness failures found by manual review of
# actual outputs - these are cheap heuristics, not perfect, but turn
# "eyeball a few examples" into "measure across the whole dataset".

# Gender-marked honorifics - if the source uses one and the rewrite
# uses a DIFFERENT one, that's a misgendering of a real, named person.
_HONORIFIC_GROUPS = [
    {"මහත්මිය", "මහත්මියගේ", "මහත්මියට"},   # female honorific forms
    {"මහතා", "මහතාගේ", "මහතාට"},             # male honorific forms
]

# Foreign symbols that should never appear unless already in the source.
_FOREIGN_SYMBOL_PATTERN = re.compile(r"[&%#@]")

# All 5 styles' mandated opening/closing phrases, used to detect one
# style's marker bleeding into another style's output for the same article.
_ALL_STYLE_MARKERS = {
    "style_2_editorial": ["විශ්ලේෂණය කරන විට", "ඉදිරි ක්‍රියාමාර්ග පිළිබඳව සාකච්ඡා කළ යුතු කාලය එළඹ ඇත"],
    "style_4_youth": ["දන්නවද", "ඇහුවද", "මේක අහන්න", "අනිවාර්යයෙන්ම දැනගන්න"],
    "style_5_feature": ["එක් දිනක, මෙම සිදුවීම සිදු විය", "අනාගත පරම්පරාවට ද ආදර්ශයක් වනු ඇත"],
}

_SINHALA_DIGIT_WORDS = [
    "එක", "දෙක", "තුන", "හතර", "පහ", "හය", "හත", "අට", "නවය", "දහය",
    "විස්ස", "තිහ", "හතළිහ", "පනහ", "සියය", "දහස", "ලක්ෂ", "කෝටි",
]


def check_quality(style: str, rewritten: str, original: str = "") -> list:
    """Returns a list of issue strings (empty if none found). Doesn't
    reject anything automatically - just flags rows so bad generations
    can be filtered or re-run afterward instead of discovered by eye.

    ✅ EXPANDED: added checks beyond the original truncation/stutter
    detectors, based on specific correctness failures found by manually
    reading v09 outputs (see conversation history) - a misgendered real
    person, an English word inserted into a direct quote, a fabricated
    duration, and a wrong style's marker phrase bleeding into this one."""
    issues = []

    required_close = REQUIRED_CLOSING.get(style)
    if required_close and required_close not in rewritten:
        issues.append("missing_required_closing")

    if _STUTTER_PATTERN.search(rewritten):
        issues.append("possible_stutter_duplication")

    if len(rewritten) < 20:
        issues.append("suspiciously_short")

    # Honorific/gender mismatch: source uses one gendered honorific group
    # but the rewrite only uses the OTHER group - e.g. source has
    # "මහත්මිය" (Mrs.) but rewrite only has "මහතා" (Mr.) forms.
    if original:
        for i, group_a in enumerate(_HONORIFIC_GROUPS):
            source_has_a = any(term in original for term in group_a)
            if not source_has_a:
                continue
            rewrite_has_a = any(term in rewritten for term in group_a)
            other_group = _HONORIFIC_GROUPS[1 - i]
            rewrite_has_other_only = (
                not rewrite_has_a and any(term in rewritten for term in other_group)
            )
            if rewrite_has_other_only:
                issues.append("honorific_gender_mismatch")
                break

    # Foreign symbols appearing that weren't in the source at all.
    if _FOREIGN_SYMBOL_PATTERN.search(rewritten) and not _FOREIGN_SYMBOL_PATTERN.search(original):
        issues.append("foreign_symbol_added")

    # Cross-style marker leakage: does this output contain another
    # style's mandated opening/closing phrase?
    for other_style, markers in _ALL_STYLE_MARKERS.items():
        if other_style == style:
            continue
        if any(marker in rewritten for marker in markers):
            issues.append("wrong_style_marker_present")
            break

    # Very rough numeric-hallucination check: a spelled-out Sinhala
    # number word or duration term appears in the rewrite but not
    # anywhere in the source. Imperfect (misses digit-based numbers,
    # can false-positive on common words), but catches cases like an
    # invented "වසර දහයක්" (ten years) with no basis in the source.
    if original:
        for word in _SINHALA_DIGIT_WORDS:
            if word in rewritten and word not in original:
                # only flag if it's paired with a time/duration word
                # nearby, to reduce false positives from unrelated uses
                if any(t in rewritten for t in ["වසර", "මාස", "දින", "සති", "පැය"]):
                    issues.append("possible_numeric_hallucination")
                    break

    return issues


class RateLimiter:
    """
    ✅ UPGRADED: was a fixed-rpm limiter requiring you to manually guess
    the right --rpm (too low wastes capacity, too high triggers 429s and
    wastes time on failed/retried requests). This version starts
    conservative and adaptively probes the real ceiling:
      - Every `ramp_every` successful requests with no errors, rpm
        increases by `ramp_factor` (capped at max_rpm).
      - On any 429, rpm is immediately cut by `backoff_factor` (in
        addition to the existing pause_until backoff), so it settles
        just below whatever the real limit turns out to be instead of
        needing you to find that number by trial and error.
    Shared across all worker threads via the same lock-protected state
    as before.
    """

    def __init__(self, rpm: int, max_rpm: int = None, min_rpm: int = 5,
                 ramp_every: int = 15, ramp_factor: float = 1.15,
                 backoff_factor: float = 0.6):
        self.rpm = max(rpm, 1)
        self.max_rpm = max_rpm or (rpm * 8)  # generous ceiling, still bounded
        self.min_rpm = min_rpm
        self.ramp_every = ramp_every
        self.ramp_factor = ramp_factor
        self.backoff_factor = backoff_factor
        self.success_streak = 0
        self.min_interval = 60.0 / self.rpm
        self.lock = threading.Lock()
        self.next_allowed = 0.0
        # if the API tells us to back off globally (e.g. after a 429),
        # every thread checks this before its next request.
        self.pause_until = 0.0

    def wait(self):
        while True:
            with self.lock:
                now = time.monotonic()
                wait_for = max(self.next_allowed, self.pause_until) - now
                if wait_for <= 0:
                    self.next_allowed = now + self.min_interval
                    return
            time.sleep(wait_for)

    def report_success(self):
        with self.lock:
            self.success_streak += 1
            if self.success_streak >= self.ramp_every and self.rpm < self.max_rpm:
                old_rpm = self.rpm
                self.rpm = min(self.max_rpm, self.rpm * self.ramp_factor)
                self.min_interval = 60.0 / self.rpm
                self.success_streak = 0
                if int(self.rpm) != int(old_rpm):
                    print(f"\n[RateLimiter] No errors for a while - "
                          f"ramping up: {old_rpm:.1f} -> {self.rpm:.1f} rpm")

    def report_429(self, retry_after: float):
        with self.lock:
            self.pause_until = max(self.pause_until, time.monotonic() + retry_after)
            old_rpm = self.rpm
            self.rpm = max(self.min_rpm, self.rpm * self.backoff_factor)
            self.min_interval = 60.0 / self.rpm
            self.success_streak = 0
            print(f"\n[RateLimiter] 429 received - cutting rate: "
                  f"{old_rpm:.1f} -> {self.rpm:.1f} rpm")


def worker(api_key, input_queue, output_file, lock, pbar, model_name, rate_limiter):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }

    while True:
        try:
            record, style = input_queue.get(timeout=5)
        except Empty:
            break

        try:
            content = clean_content(record.get("content", ""))
            if not content:
                with lock:
                    with open(output_file, "a", encoding="utf-8") as out_f:
                        error_record = record.copy()
                        error_record.pop("title", None)
                        error_record["style"] = style
                        error_record["error"] = "empty content"
                        out_f.write(json.dumps(error_record, ensure_ascii=False) + "\n")
                        out_f.flush()
                pbar.update(1)
                continue

            prompt = build_prompt(style, content)

            # Styles that add framing text (opening cue + mandatory closing
            # sentence) need MORE tokens than the source article, not the
            # same amount - editorial/feature were getting truncated mid-
            # sentence, cutting off the required closing line entirely.
            # Sinhala also tends to need more tokens/word than a crude
            # word-count heuristic assumes, so the multiplier and floor
            # are both raised generously here.
            style_multiplier = {
                "style_1_formal_news": 3.0,
                "style_2_editorial": 3.8,   # adds intro + closing sentence
                "style_3_sports": 3.2,
                "style_4_youth": 3.2,       # adds opening + closing line
                "style_5_feature": 3.8,     # adds scene-setting + closing
            }.get(style, 3.2)

            approx_len_tokens = max(768, int(len(content.split()) * style_multiplier))
            max_tokens = min(approx_len_tokens, 4096)

            payload = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": max_tokens,
                "temperature": 0.3,
                "top_p": 0.9,
                "frequency_penalty": 0.4,
                "presence_penalty": 0.2,
                "stream": False,
            }

            success = False
            retries = 0

            while not success and retries < 5:
                rate_limiter.wait()
                try:
                    response = requests.post(
                        INVOKE_URL,
                        headers=headers,
                        json=payload,
                        # ✅ FIXED: was 180s, but observed completions were
                        # taking 200-275s for this model/max_tokens combo -
                        # meaning most requests were timing out and
                        # RESTARTING from scratch just before they would
                        # have succeeded, wasting the time already spent
                        # waiting. Raised well above observed latency.
                        timeout=400,
                    )

                    if response.status_code == 200:
                        res_json = response.json()

                        rewritten = ""
                        try:
                            message = res_json["choices"][0]["message"]
                            rewritten = message.get("content")
                            if rewritten is None:
                                rewritten = ""
                            rewritten = rewritten.strip()
                        except Exception:
                            rewritten = str(res_json)

                        result = record.copy()
                        result.pop("title", None)
                        result["content"] = content  # cleaned version, not raw scrape
                        result["style"] = style
                        result["rewritten_text"] = rewritten
                        result["teacher_model"] = model_name
                        qc_issues = check_quality(style, rewritten, original=content)
                        if qc_issues:
                            result["qc_issues"] = qc_issues

                        with lock:
                            with open(output_file, "a", encoding="utf-8") as out_f:
                                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                                out_f.flush()
                                os.fsync(out_f.fileno())

                        success = True
                        rate_limiter.report_success()

                    else:
                        retry_after = None
                        if response.status_code == 429:
                            # Prefer the server's own guidance if it gives one.
                            header_val = response.headers.get("Retry-After")
                            if header_val:
                                try:
                                    retry_after = float(header_val)
                                except ValueError:
                                    retry_after = None
                            if retry_after is None:
                                try:
                                    body = response.json()
                                    retry_after = float(body.get("retryDelay", 0)) or None
                                except Exception:
                                    retry_after = None
                            retry_after = retry_after or (10 * (retries + 1))
                            print(f"\n[429] Backing off ALL threads for {retry_after:.1f}s")
                            rate_limiter.report_429(retry_after)
                        else:
                            print(f"\n[Error {response.status_code}] {response.text[:500]}")

                        retries += 1
                        time.sleep((2 ** retries) + 2)

                except Exception as e:
                    print(f"\n[Exception] {repr(e)}")
                    retries += 1
                    time.sleep(3)

            if not success:
                with lock:
                    with open(output_file, "a", encoding="utf-8") as out_f:
                        error_record = record.copy()
                        error_record.pop("title", None)
                        error_record["style"] = style
                        error_record["status"] = "failed"
                        error_record["error"] = f"failed after {retries} retries"
                        out_f.write(json.dumps(error_record, ensure_ascii=False) + "\n")
                        out_f.flush()
                        os.fsync(out_f.fileno())

        finally:
            input_queue.task_done()
            pbar.update(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="/home/jovyan/style_rewriter/data/train.jsonl")
    parser.add_argument("--output", default="/home/jovyan/style_rewriter/data/style_dataset2_dub.jsonl")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=20,
        help="Number of worker threads. With this model taking 150-275s per "
             "response, low concurrency (e.g. 3) badly under-uses your --rpm "
             "budget - too few requests are in-flight at once to sustain the "
             "target rate. Rough rule of thumb: concurrency should be roughly "
             "(typical latency in seconds) / (60 / rpm), e.g. 200s latency "
             "at 15 rpm (4s between starts) needs ~20+ concurrent threads to "
             "actually hit that rate.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--styles",
        default=",".join(ALL_STYLES),
        help=f"Comma-separated list of styles to generate. Choices: {', '.join(ALL_STYLES)}",
    )
    parser.add_argument(
        "--rpm",
        type=int,
        default=15,
        help="STARTING requests-per-minute budget (shared across all threads) - "
             "the rate limiter now adapts automatically from here: it ramps up "
             "when requests are succeeding cleanly, and cuts back hard on any "
             "429. You no longer need to guess the exact right number - this "
             "just needs to be a safe starting point.",
    )
    parser.add_argument(
        "--max-rpm",
        type=int,
        default=None,
        help="Ceiling the adaptive rate limiter won't ramp past, even if "
             "everything is succeeding. Defaults to 8x --rpm if not set.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N articles (file order, no shuffling). "
             "Use --sample instead if you want a randomized subset.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Randomly sample N articles (seeded, reproducible) instead of "
             "processing the whole corpus. Recommended over --limit for a "
             "representative style-rewriter training subset. Ignored if "
             "--limit is also set.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for --sample, so the same subset is picked every run.",
    )
    parser.add_argument(
        "--time-budget-hours",
        type=float,
        default=None,
        help="Auto-calculate how many articles to sample so the run finishes "
             "within this many hours at the given --rpm, with a safety margin "
             "for retries/429 backoff. Ignored if --sample or --limit is set explicitly.",
    )
    parser.add_argument(
        "--safety-margin",
        type=float,
        default=0.75,
        help="Fraction of the theoretical max throughput to actually plan for, "
             "since retries and 429 backoff eat into real throughput. "
             "Only used with --time-budget-hours. Default 0.75 (use 75%% of "
             "the theoretical rpm-based capacity).",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.touch(exist_ok=True)
    print(f"Saving to: {output_path.resolve()}")

    if not input_path.exists():
        print(f"Input file {input_path} not found.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    requested_styles = [s.strip() for s in args.styles.split(",") if s.strip()]
    invalid = [s for s in requested_styles if s not in STYLE_INSTRUCTIONS]
    if invalid:
        print(f"Unknown style(s): {invalid}. Valid choices: {ALL_STYLES}")
        return

    api_key = load_api_key()
    processed_pairs = load_processed_pairs(output_path)
    if processed_pairs:
        print(f"Resuming: found {len(processed_pairs)} (url, style) pairs already "
              f"in {output_path} - these will be skipped automatically.")

    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    if args.limit is not None:
        records = records[: args.limit]
    elif args.sample is not None:
        rng = random.Random(args.seed)
        if args.sample < len(records):
            records = rng.sample(records, args.sample)
    elif args.time_budget_hours is not None:
        # Work backwards from the time budget: how many total (article,
        # style) requests can we realistically fit, given the rpm cap and
        # a safety margin for retries/429 backoff eating into real
        # throughput? Then convert that into an article count.
        budget_minutes = args.time_budget_hours * 60
        theoretical_pairs = budget_minutes * args.rpm
        safe_pairs = int(theoretical_pairs * args.safety_margin)
        auto_sample_size = max(1, safe_pairs // len(requested_styles))

        print(
            f"--time-budget-hours {args.time_budget_hours} at --rpm {args.rpm} "
            f"(safety margin {args.safety_margin}) => sampling {auto_sample_size} articles "
            f"({auto_sample_size * len(requested_styles)} total pairs)"
        )

        if auto_sample_size < len(records):
            rng = random.Random(args.seed)
            records = rng.sample(records, auto_sample_size)
        print(f"Sampled {len(records)} articles (seed={args.seed})")

    # Build the full (record, style) work list, skipping anything already done.
    to_process = []
    for r in records:
        url = r.get("url")
        for style in requested_styles:
            if (url, style) not in processed_pairs:
                to_process.append((r, style))

    print(f"Model: {args.model}")
    print(f"Styles: {requested_styles}")
    print(f"Total articles: {len(records)}")
    print(f"Total (article, style) pairs: {len(records) * len(requested_styles)}")
    print(f"Already processed: {len(processed_pairs)}")
    print(f"Remaining: {len(to_process)}")
    print(f"Concurrency: {args.concurrency}")
    effective_max_rpm = args.max_rpm or (args.rpm * 8)
    print(f"Rate limit: starting at {args.rpm} rpm, adaptively ramping up to "
          f"{effective_max_rpm} rpm if no errors occur (cuts back on any 429)")

    est_minutes_slow = len(to_process) / args.rpm
    est_minutes_fast = len(to_process) / effective_max_rpm
    print(f"Estimated time: {est_minutes_fast:.0f}-{est_minutes_slow:.0f} min "
          f"(~{est_minutes_fast / 60:.1f}-{est_minutes_slow / 60:.1f} hours) "
          f"depending on how high the rate ramps before hitting a real limit")

    if not to_process:
        print("Everything already processed.")
        return

    input_queue = Queue()
    for item in to_process:
        input_queue.put(item)

    lock = threading.Lock()
    rate_limiter = RateLimiter(args.rpm, max_rpm=args.max_rpm)

    with tqdm(total=len(to_process), desc="Rewriting") as pbar:
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            for _ in range(args.concurrency):
                executor.submit(worker, api_key, input_queue, output_path, lock, pbar, args.model, rate_limiter)

            input_queue.join()

    print(f"\nComplete. Results: {output_path}")


if __name__ == "__main__":
    main()