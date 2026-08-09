"""
SinhalaJournal-LLM | Shared teacher-data quality checks
--------------------------------------------------------
Two defect classes confirmed to slip past every existing validation stage in
the summarizer pipeline (7_multilength_summary_generator.validate_summary,
7_train_summarizer.summary_is_clean, 7_test_summarizer's eval loop — the
v06-era predecessors of these three scripts had no equivalent checks at all):

1. Word-spacing glue: Sinhala words concatenated with no whitespace between
   them (a rare teacher-model decoding glitch, not a tokenizer/training
   artifact — confirmed present in a handful of raw teacher records).
2. Numeric-unit confusion: the article and summary agree on a digit string
   but disagree on its scale word (ලක්ෂ/මිලියන/කෝටි/බිලියන), e.g. article
   "මිලියන 476" vs. summary "ලක්ෂ 476" — a ~10x factual error. This survives
   6_multilength_summary_generator's existing digits_in() hallucination
   check because that check only compares raw digit strings, not units.

Shared across 4 sibling scripts in this directory (generator, trainer,
tester, and the one-off cleaning script) — imported directly since every
script here runs as `python abstractive/whatever.py` and Python auto-adds
the script's own directory to sys.path[0].
"""

import re

# ── Word-spacing glue detection ────────────────────────────────────────────
# Sinhala block (U+0D80-U+0DFF) + ASCII digits + zero-width joiner/non-joiner
# (U+200D/U+200C, required to correctly tokenize conjunct clusters like
# ශ්‍රී — omitting them artificially truncates real long words and corrupts
# threshold calibration).
GLUE_TOKEN_RE = re.compile(r'[඀-෿0-9‌‍]+')

# Calibrated against the full 6_multilength_summaries.jsonl corpus
# (35,569 records): swept 20-26 chars, hand-verified every hit. Legitimate
# long single tokens exist up to 23 chars (transliterated proper nouns with
# a case suffix glued on, e.g. ග්ලැක්සෝස්මිත්ක්ලයින් "GlaxoSmithKline";
# native compounds with case suffixes, e.g. සාමාන්‍යාධිකාරිවරයාගෙන් "from
# the General Manager"). 24 is the first threshold with zero false positives
# on this corpus while still catching the one confirmed glue defect. Re-sweep
# if the corpus grows substantially or a new teacher model is introduced.
MAX_GLUE_TOKEN_CHARS = 24


def detect_word_glue(text: str) -> str | None:
    """Flags a contiguous Sinhala-script/digit run (incl. ZWJ/ZWNJ conjuncts)
    at or above MAX_GLUE_TOKEN_CHARS as probable multi-word concatenation
    with no inter-word whitespace. Returns a rejection reason, or None."""
    if not text:
        return None
    for tok in GLUE_TOKEN_RE.findall(text):
        if len(tok) >= MAX_GLUE_TOKEN_CHARS:
            return f"word_glue:{tok[:40]}"
    return None


# ── Numeric-unit consistency ───────────────────────────────────────────────
UNIT_SCALE = {
    "ලක්ෂ": 100_000,
    "මිලියන": 1_000_000,
    "කෝටි": 10_000_000,
    "බිලියන": 1_000_000_000,
}

# "දශ ලක්ෂ" ("dasha-laksha", ten-lakh) is a common compound numeral in
# Sinhala financial reporting equal to 1,000,000 — i.e. synonymous with
# මිලියන, NOT a variant of bare ලක්ෂ (100,000). It must be tried before the
# bare ලක්ෂ alternative below, or the regex matches just the ලක්ෂ suffix and
# silently mis-scales the value 10x low, producing false unit-mismatch
# positives on this very common idiom (confirmed against real corpus
# records where article "රුපියල් දශ ලක්ෂ 15,000" and summary "රු. 15,000
# මිලියන" describe the identical figure — 15,000 million — and would
# otherwise be wrongly flagged as a mismatch).
_DASHA_LAKH = r'දශ\s*ලක්ෂ'
_UNIT_ALT = "|".join([_DASHA_LAKH] + list(UNIT_SCALE))
_NUM = r'[0-9][0-9,]*(?:\.[0-9]+)?'

# Both orderings occur at real volume in the corpus ("රුපියල් බිලියන 3.68ක්"
# and "USD 571.99 මිලියන" are both common patterns), so both must be matched.
NUMBER_UNIT_RE = re.compile(
    rf'(?:(?P<num_a>{_NUM})\s*(?P<unit_a>{_UNIT_ALT}))'
    rf'|(?:(?P<unit_b>{_UNIT_ALT})\s*\.?\s*(?P<num_b>{_NUM}))'
)


def _num_key(raw: str) -> int:
    return round(float(raw.replace(",", "")))


def _canonical_unit(raw_unit: str) -> str:
    return "මිලියන" if raw_unit.startswith("දශ") else raw_unit


def extract_number_unit_pairs(text: str) -> list[tuple[int, str]]:
    pairs = []
    for m in NUMBER_UNIT_RE.finditer(text):
        num = m.group("num_a") or m.group("num_b")
        unit = m.group("unit_a") or m.group("unit_b")
        pairs.append((_num_key(num), _canonical_unit(unit)))
    return pairs


def check_numeric_unit_consistency(summary: str, article: str) -> str | None:
    """Flags a (number, unit) pair in the summary where the same number
    appears in the article, but never with that unit — catches order-of-
    magnitude scale errors (e.g. article "මිලියන 476", summary "ලක්ෂ 476")
    that survive a raw-digit-string hallucination check because the digit
    string itself is reused correctly; only the unit is wrong. Returns a
    rejection reason, or None."""
    article_units_by_num: dict[int, set] = {}
    for num, unit in extract_number_unit_pairs(article):
        article_units_by_num.setdefault(num, set()).add(unit)
    if not article_units_by_num:
        return None
    for num, unit in extract_number_unit_pairs(summary):
        seen_units = article_units_by_num.get(num)
        if seen_units and unit not in seen_units:
            return f"unit_mismatch:{num}:{unit}_not_in:{sorted(seen_units)}"
    return None
