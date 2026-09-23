"""
SinhalaJournal-LLM | Shared teacher-data quality checks
--------------------------------------------------------
Two defect classes confirmed to slip past validation stages in the summarizer pipeline:

1. Word-spacing glue: Sinhala words concatenated with no whitespace between
   them across virama/hal-akuru boundaries (e.g., 'හෙරොයින්කිලෝවක්',
   'අනෙකුත්දෙදෙනාගේ', 'ඔවුන්සතුව', 'ඔවුන්රැගෙන', 'දැන්දිගටම', 'හෙරොයින්ග්රෑම්'),
   as well as long contiguous script runs (>= MAX_GLUE_TOKEN_CHARS).
   Sub-24-character glue defects are detected via virama-boundary analysis
   while preserving legitimate conjuncts (Rakaransaya, Yansaya, Bandi-akuru),
   inflected words (e.g. 'ඔවුන්ගේ', 'දැන්ම'), and valid single words
   (e.g., 'අත්අඩංගුවට', 'ප්රදේශයේ', 'මහේස්ත්රාත්', 'ශ්රී', 'කාර්යාලය', 'වාර්තාව').

2. Numeric & Unit consistency:
   - Scale words: order-of-magnitude currency/number scale errors (ලක්ෂ/මිලියන/කෝටි/බිලියන).
   - Metric mass/weight units: metric mismatches where source contains 'ග්රෑම්'/'මිලිග්රෑම්'
     but summary introduces 'කිලෝ'/'කිලෝග්රෑම්' without mathematical basis, or
     swaps units on the same number (e.g., 460g -> 460kg, 150g -> 150kg).

Shared across generator, trainer, tester, and evaluation scripts.
"""

from __future__ import annotations

import re

try:
    from sinhala_degluer import is_virama_glued, deglue_virama_boundaries
except ImportError:
    try:
        from tasks.sinhala_degluer import is_virama_glued, deglue_virama_boundaries
    except ImportError:
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).resolve().parent))
        from sinhala_degluer import is_virama_glued, deglue_virama_boundaries

# ── Word-spacing glue detection ────────────────────────────────────────────
# Sinhala block (U+0D80-U+0DFF) + ASCII digits + zero-width joiner/non-joiner
GLUE_TOKEN_RE = re.compile(r'[඀-෿0-9‌‍]+')

# 24 is the threshold for general long token runs (proper nouns, inflected compounds)
MAX_GLUE_TOKEN_CHARS = 24


def detect_word_glue(text: str) -> str | None:
    """Flags word concatenation defects in Sinhala text:
    1. Contiguous runs at or above MAX_GLUE_TOKEN_CHARS (>= 24 chars).
    2. Sub-24-character virama-boundary word concatenation defects
       (e.g., 'හෙරොයින්කිලෝවක්', 'අනෙකුත්දෙදෙනාගේ', 'ඔවුන්සතුව', 'ඔවුන්රැගෙන',
       'දැන්දිගටම', 'හෙරොයින්ග්රෑම්').
    Preserves legitimate conjuncts, inflected forms, and valid single words.
    Returns a rejection reason, or None."""
    if not text:
        return None
    for tok in GLUE_TOKEN_RE.findall(text):
        if len(tok) >= MAX_GLUE_TOKEN_CHARS:
            return f"word_glue:{tok[:40]}"
        if is_virama_glued(tok):
            return f"word_glue:{tok[:40]}"
    return None


# ── Scale word consistency (Currency / Large numbers) ──────────────────────
UNIT_SCALE = {
    "ලක්ෂ": 100_000,
    "මිලියන": 1_000_000,
    "කෝටි": 10_000_000,
    "බිලියන": 1_000_000_000,
}

_DASHA_LAKH = r'දශ\s*ලක්ෂ'
_UNIT_ALT = "|".join([_DASHA_LAKH] + list(UNIT_SCALE))
_NUM = r'[0-9][0-9,]*(?:\.[0-9]+)?'

NUMBER_UNIT_RE = re.compile(
    rf'(?:(?P<num_a>{_NUM})\s*(?P<unit_a>{_UNIT_ALT}))'
    rf'|(?:(?P<unit_b>{_UNIT_ALT})\s*\.?\s*(?P<num_b>{_NUM}))'
)


def _num_key(raw: str) -> int:
    clean = raw.rstrip("ක්ක").replace(",", "").strip()
    return round(float(clean))


def _canonical_unit(raw_unit: str) -> str:
    return "මිලියන" if raw_unit.startswith("දශ") else raw_unit


def extract_number_unit_pairs(text: str) -> list[tuple[int, str]]:
    pairs = []
    for m in NUMBER_UNIT_RE.finditer(text):
        num = m.group("num_a") or m.group("num_b")
        unit = m.group("unit_a") or m.group("unit_b")
        pairs.append((_num_key(num), _canonical_unit(unit)))
    return pairs


def _check_scale_word_consistency(summary: str, article: str) -> str | None:
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


# ── Metric Mass/Weight unit consistency ────────────────────────────────────
# Negative lookahead on 'කිලෝ' ensures metric distance/power/volume units
# (කිලෝමීටර්, කිලෝලීටර්, කිලෝවොට්, etc.) are NOT mistaken for kilograms.
# Bound (?<![a-zA-Z])kg(?![a-zA-Z]) ensures English words like 'background'
# are not mistaken for kilograms.
_KILO_PATTERNS = (
    r'(?:කිලෝ\s*ග්‍රෑම්|කිලෝ\s*ග්රෑම්|කිලෝග්‍රෑම්|කිලෝග්රෑම්|'
    r'කිලෝව(?:ක්|ක|ට)?(?![඀-෿])|'
    r'කිලෝ(?!\s*(?:මීටර්|මීටර|ලීටර්|ලීටර|වොට්|වෝල්ට්|බයිට්|බිට්|කැලරි|හර්ට්ස්|ජූල්|පැස්කල්|ඇම්පියර්|ඕම්|මී\.|ලී\.))(?![඀-෿])|'
    r'(?i:(?<![a-zA-Z])kg(?![a-zA-Z]))|කි\.ග්‍රෑ\.|කි\.ග්රෑ\.)'
)
# Negative lookbehinds prevent matching non-mass loanwords (ටෙලිග්‍රෑම්, ඉන්ස්ටග්‍රෑම්, ඉන්ස්ටාග්‍රෑම්, ප්‍රෝග්‍රෑම්, ඩයග්‍රෑම්)
# and prevent matching inside kilograms (කිලෝග්‍රෑම්, කිලෝ ග්‍රෑම්).
# Single-letter 'g' only matches lowercase 'g' to prevent colliding with uppercase cellular '5G'/'4G'.
_GRAM_PATTERNS = (
    r'(?:(?<!කිලෝ\s)(?<!කිලෝ)(?<!ටෙලි)(?<!ඉන්ස්ට)(?<!ඉන්ස්ටා)(?<!ප්‍රෝ)(?<!ප්රෝ)(?<!ඩය)(?<!හොලෝ)(?<!මොනෝ)(?:ග්‍රෑම්|ග්රෑම්|ග්‍රෑ\.|ග්රෑ\.)|'
    r'(?:(?<![a-zA-Z])g(?![a-zA-Z])))'
)
_MILLI_PATTERNS = (
    r'(?:මිලි\s*ග්‍රෑම්|මිලි\s*ග්රෑම්|මිලිග්‍රෑම්|මිලිග්රෑම්|මි\.ග්‍රෑ\.|මි\.ග්රෑ\.|(?i:(?<![a-zA-Z])mg(?![a-zA-Z])))'
)

_MASS_UNIT_ALT = rf'(?:{_KILO_PATTERNS}|{_MILLI_PATTERNS}|{_GRAM_PATTERNS})'

# Sinhala numeral words and fractions representing numbers
SINHALA_WORD_NUMBERS = {
    "එක": 1.0, "එකක්": 1.0,
    "දෙක": 2.0, "දෙකක්": 2.0,
    "තුන": 3.0, "තුනක්": 3.0,
    "හතර": 4.0, "හතරක්": 4.0,
    "පහ": 5.0, "පහක්": 5.0,
    "හය": 6.0, "හයක්": 6.0,
    "හත": 7.0, "හතක්": 7.0,
    "අට": 8.0, "අටක්": 8.0,
    "නවය": 9.0, "නවයක්": 9.0,
    "දහය": 10.0, "දහයක්": 10.0,
    "සියයක්": 100.0, "සිය": 100.0,
    "දහසක්": 1000.0, "දහස": 1000.0, "දහස්": 1000.0,
    "දෙදහසක්": 2000.0,
    "භාගයක්": 0.5, "භාග": 0.5,
    "අඩක්": 0.5, "අඩ": 0.5,
    "කාලක්": 0.25, "කාල": 0.25,
}

_WORD_NUMS_ALT = "|".join(sorted(SINHALA_WORD_NUMBERS.keys(), key=len, reverse=True))
_NUM = r'[0-9][0-9,]*(?:\.[0-9]+)?(?:ක්|ක)?'
_NUM_OR_WORD = rf'(?:{_NUM}|{_WORD_NUMS_ALT})'

MASS_NUMBER_UNIT_RE = re.compile(
    rf'(?:(?P<num_a>{_NUM_OR_WORD})\s*(?P<unit_a>{_MASS_UNIT_ALT}))'
    rf'|(?:(?P<unit_b>{_MASS_UNIT_ALT})\s*\.?\s*(?P<num_b>{_NUM_OR_WORD}))'
    rf'|(?P<standalone_kilo>කිලෝව(?:ක්|ක|ට)?(?![඀-෿]))'
)

_KILO_RE = re.compile(_KILO_PATTERNS)
_GRAM_OR_MILLI_RE = re.compile(rf'(?:{_GRAM_PATTERNS}|{_MILLI_PATTERNS})')


def _canonical_mass_unit(raw_unit: str) -> str:
    raw = raw_unit.strip()
    if _KILO_RE.search(raw):
        return "කිලෝග්‍රෑම්"
    if re.search(r'මිලි|(?i:mg)|මි\.', raw):
        return "මිලිග්‍රෑම්"
    return "ග්‍රෑම්"


def extract_mass_unit_pairs(text: str) -> list[tuple[float, str]]:
    pairs = []
    for m in MASS_NUMBER_UNIT_RE.finditer(text):
        if m.group("standalone_kilo"):
            pairs.append((1.0, "කිලෝග්‍රෑම්"))
            continue
        num_str = (m.group("num_a") or m.group("num_b") or "").strip()
        unit_str = (m.group("unit_a") or m.group("unit_b") or "").strip()
        if num_str and unit_str:
            try:
                clean_num_str = num_str.rstrip("ක්ක").strip()
                if num_str in SINHALA_WORD_NUMBERS:
                    num = SINHALA_WORD_NUMBERS[num_str]
                elif clean_num_str in SINHALA_WORD_NUMBERS:
                    num = SINHALA_WORD_NUMBERS[clean_num_str]
                else:
                    num = float(clean_num_str.replace(",", ""))
                pairs.append((num, _canonical_mass_unit(unit_str)))
            except ValueError:
                continue
    return pairs


def has_kilo_units(text: str) -> bool:
    return bool(_KILO_RE.search(text))


def has_gram_or_milli_units(text: str) -> bool:
    return bool(_GRAM_OR_MILLI_RE.search(text))


def check_mass_unit_consistency(summary: str, article: str) -> str | None:
    """Flags metric mass/weight unit mismatches:
    1. Summary introduces kilograms ('කිලෝ'/'කිලෝග්රෑම්'/'කිලෝවක්'/'kg') when source article
       only mentions grams/milligrams without mathematical basis, or introduces kilograms
       when source article contains no mass units at all.
    2. Summary attaches kilograms to a number that appears in the article as grams/milligrams
       (e.g., 460g -> 460kg, 150g -> 150kg)."""
    if not summary or not article:
        return None

    # 1. Number-unit collision (e.g., 460kg in summary vs 460g in article)
    article_mass_by_num: dict[float, set[str]] = {}
    for num, unit in extract_mass_unit_pairs(article):
        article_mass_by_num.setdefault(num, set()).add(unit)

    if article_mass_by_num:
        for num, unit in extract_mass_unit_pairs(summary):
            seen_units = article_mass_by_num.get(num)
            if seen_units and unit not in seen_units:
                num_display = int(num) if num.is_integer() else num
                return f"unit_mismatch:{num_display}:{unit}_not_in:{sorted(seen_units)}"

    summary_has_kilo = has_kilo_units(summary)
    article_has_kilo = has_kilo_units(article)
    article_has_sub_kilo = has_gram_or_milli_units(article)

    # 2. Fabricated kilograms when article does not have kilograms
    if summary_has_kilo and not article_has_kilo:
        if article_has_sub_kilo:
            summary_pairs = extract_mass_unit_pairs(summary)
            article_pairs = extract_mass_unit_pairs(article)
            has_basis = False
            if summary_pairs and article_pairs:
                kilo_count = 0
                valid_kilo_count = 0
                for s_num, s_unit in summary_pairs:
                    if s_unit == "කිලෝග්‍රෑම්":
                        kilo_count += 1
                        equiv_grams = s_num * 1000.0
                        equiv_mg = s_num * 1_000_000.0
                        for a_num, a_unit in article_pairs:
                            if a_unit == "ග්‍රෑම්" and abs(a_num - equiv_grams) < 0.01:
                                valid_kilo_count += 1
                                break
                            if a_unit == "මිලිග්‍රෑම්" and abs(a_num - equiv_mg) < 0.01:
                                valid_kilo_count += 1
                                break
                if kilo_count > 0 and valid_kilo_count == kilo_count:
                    has_basis = True
            if not has_basis:
                return "unit_mismatch:fabricated_kilo:summary_introduces_kilograms_without_basis"
        else:
            return "unit_mismatch:fabricated_kilo:summary_introduces_kilograms_without_basis"

    return None


def check_numeric_unit_consistency(summary: str, article: str) -> str | None:
    """Flags a numeric/unit defect in summary against article:
    1. Order-of-magnitude currency/number scale errors (e.g. article 'මිලියන 476',
       summary 'ලක්ෂ 476').
    2. Metric mass/weight unit mismatches (e.g. source contains 'ග්රෑම්'/'මිලිග්රෑම්'
       but summary introduces 'කිලෝ'/'කිලෝග්රෑම්' without mathematical basis,
       or swaps units on the same number).
    Returns a rejection reason string starting with 'unit_mismatch:', or None."""
    # 1. Scale word consistency
    scale_defect = _check_scale_word_consistency(summary, article)
    if scale_defect:
        return scale_defect

    # 2. Metric mass unit consistency
    mass_defect = check_mass_unit_consistency(summary, article)
    if mass_defect:
        return mass_defect

    return None
