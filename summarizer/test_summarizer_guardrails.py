#!/usr/bin/env python3
"""
Automated Test Suite: Sinhala LLM Summarizer Guardrails
=======================================================
Verifies all acceptance criteria defined in ORIGINAL_REQUEST.md:
1. Word Boundary Separation:
   - Glued test pairs ('හෙරොයින්කිලෝවක්', 'අනෙකුත්දෙදෙනාගේ', 'ඔවුන්සතුව',
     'ඔවුන්රැගෙන', 'දැන්දිගටම', 'හෙරොයින්ග්රෑම්', 'හෙරොයින්ග්‍රෑම්') are automatically separated.
   - Realistic multi-word and compound concatenations are separated (e.g. 'ඔවුන්විසින්සොයාගත්', 'රුපියල්මිලියන').
   - Valid Sinhala ligatures and words with internal viramas ('අත්අඩංගුවට',
     'ප්රදේශයේ', 'මහේස්ත්රාත්', 'ශ්රී', 'තත්ත්ව', 'නිෂ්පාදන') are never split.
   - Grammatical inflections ('ඔවුන්ගේ', 'තමන්ගේ', 'සැකකරුවන්ගේ', 'දැන්ම') and
     everyday vocabulary ('කාර්යාලය', 'වාර්තාව', 'අවස්ථාව', 'විස්තරය', 'දැන්වීම', 'බස්නාහිර', 'පොලිස්පති')
     are strictly preserved and never split.
2. Numeric & Unit Factual Guardrails:
   - detect_word_glue() flags sub-24-character glued tokens.
   - detect_word_glue() never falsely flags clean text with valid inflections and vocabulary.
   - check_numeric_unit_consistency() flags fabricated kilograms ('කිලෝ',
     'කිලෝග්රෑම්', 'කිලෝවක්', '1kg', '1KG') when source only mentions grams/milligrams.
   - check_numeric_unit_consistency() does NOT falsely flag distance units ('කිලෝමීටර්')
     or communication app loanwords ('ටෙලිග්‍රෑම්').
   - check_numeric_unit_consistency() catches mass number-unit collisions (e.g. 460g vs 460kg).
   - check_numeric_unit_consistency() preserves existing currency/scale checks.
3. Benchmark Article Verification:
   - Police drug raid article produces verified Short, Medium, Long summaries
     accurately citing true quantities (~77.26g heroin, Rs. 2.6M+ cash) without 1kg hallucination.
"""

import sys
import json
import unittest
from pathlib import Path

# Add abstractive and work/tasks directories to sys.path
BASE_DIR = Path(__file__).resolve().parent
ABSTRACTIVE_DIR = BASE_DIR / "abstractive"
TASKS_DIR = BASE_DIR.parent / "work" / "tasks"
BENCHMARK_PATH = BASE_DIR / "data" / "benchmark_drug_raid.json"

for p in (str(ABSTRACTIVE_DIR), str(TASKS_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from sinhala_degluer import deglue_virama_boundaries, deglue_token, is_virama_glued, heal_sinhala_text
from data_quality_checks import (
    detect_word_glue,
    check_numeric_unit_consistency,
    check_mass_unit_consistency,
    extract_mass_unit_pairs,
    has_kilo_units,
    has_gram_or_milli_units,
)


class TestSinhalaWordBoundaryDegluing(unittest.TestCase):
    """AC 1: Word Boundary Separation across virama/hal-akuru boundaries."""

    def test_glued_acceptance_pairs(self):
        """All mandatory glued test pairs from ORIGINAL_REQUEST.md are separated with whitespace."""
        test_cases = [
            ("හෙරොයින්කිලෝවක්", "හෙරොයින් කිලෝවක්"),
            ("අනෙකුත්දෙදෙනාගේ", "අනෙකුත් දෙදෙනාගේ"),
            ("ඔවුන්සතුව", "ඔවුන් සතුව"),
            ("ඔවුන්රැගෙන", "ඔවුන් රැගෙන"),
            ("දැන්දිගටම", "දැන් දිගටම"),
            ("හෙරොයින්ග්රෑම්", "හෙරොයින් ග්රෑම්"),
            ("හෙරොයින්ග්‍රෑම්", "හෙරොයින් ග්‍රෑම්"),
            # Realistic news compounds and multi-word concatenations
            ("රුපියල්මිලියන", "රුපියල් මිලියන"),
            ("සැකකරුවන්සතුව", "සැකකරුවන් සතුව"),
            ("ඔවුන්විසින්සොයාගත්", "ඔවුන් විසින් සොයාගත්"),
            # Conjunction and connective glues
            ("හෙරොයින්සහ", "හෙරොයින් සහ"),
            ("ඔවුන්සහ", "ඔවුන් සහ"),
            # Hal-akuru words glued to ASCII digits
            ("රුපියල්2.6", "රුපියල් 2.6"),
            ("ග්‍රෑම්77", "ග්‍රෑම් 77"),
            ("හෙරොයින්100", "හෙරොයින් 100"),
            # Indefinite noun prefixes and police news compounds
            ("මුදලක්සොයාගත්", "මුදලක් සොයාගත්"),
            ("පැකට්ටුවක්සොයාගත්", "පැකට්ටුවක් සොයාගත්"),
            ("පොලිස්නිලධාරීන්", "පොලිස් නිලධාරීන්"),
            ("පොලිස්විමර්ශන", "පොලිස් විමර්ශන"),
            ("දැන්වැඩිදුර", "දැන් වැඩිදුර"),
        ]
        for glued, expected in test_cases:
            with self.subTest(glued=glued):
                separated = deglue_token(glued)
                self.assertEqual(separated, expected, f"Expected '{glued}' to be separated into '{expected}', got '{separated}'")
                # Also verify in full text context
                text_input = f"පොලීසිය {glued} සොයාගෙන ඇත."
                text_expected = f"පොලීසිය {expected} සොයාගෙන ඇත."
                self.assertEqual(deglue_virama_boundaries(text_input), text_expected)

    def test_negative_valid_ligatures_and_single_words(self):
        """Valid Sinhala ligatures, inflections, and words with internal viramas are NEVER incorrectly split."""
        protected_words = [
            # Mandatory acceptance criteria words
            "අත්අඩංගුවට",
            "ප්රදේශයේ",
            "ප්‍රදේශයේ",
            "මහේස්ත්රාත්",
            "මහේස්ත්‍රාත්",
            "ශ්රී",
            "ශ්‍රී",
            "නිෂ්පාදන",
            "තත්ත්ව",
            "විශ්ලේෂණ",
            "සම්බන්ධ",
            "ආයෝජන",
            "ලක්ෂ",
            # Grammatical case inflections on virama-ending pronouns/nouns
            "ඔවුන්ගේ",
            "තමන්ගේ",
            "සැකකරුවන්ගේ",
            "පුද්ගලයින්ගේ",
            "නිලධාරීන්ගේ",
            "කාන්තාවන්ගේ",
            "මන්ත්‍රීවරුන්ගේ",
            "දැන්ම",
            "ඔවුන්ම",
            "ඔවුන්ට",
            "ඔවුන්ගෙන්",
            # Noun plural, oblique, and enclitic suffixes
            "හෙරොයින්වලින්",
            "හෙරොයින්වලට",
            "බස්වල",
            "බස්වලින්",
            "බස්වලට",
            "ඔවුන්ගෙන්ද",
            "ඔවුන්ටත්",
            "එයින්දීම",
            # Standard Sinhala vocabulary containing internal viramas/clusters
            "දැන්වීම",
            "දැන්වීම්",
            "දැන්වීය",
            "බස්නාහිර",
            "පොලිස්පති",
            "කාර්යාලය",
            "වාර්තාව",
            "අවස්ථාව",
            "විස්තරය",
            "පුස්තකාලය",
            "ශාස්ත්‍රාලය",
            "විශ්වාසය",
            "ධාර්මික",
            "ආර්ථිකය",
            "මාර්ගය",
            "පාර්ලිමේන්තුව",
            "විශ්වවිද්‍යාලය",
            "අමාත්‍යාංශය",
            "ජාත්‍යන්තර",
            "ත්‍රස්තවාදී",
            "ප්‍රජාතන්ත්‍රවාදී",
            "සෞඛ්‍ය",
            "අයිස්ක්‍රීම්",
            "බස්රථ",
        ]
        for word in protected_words:
            with self.subTest(word=word):
                self.assertFalse(is_virama_glued(word), f"'{word}' should NOT be considered virama-glued")
                result = deglue_token(word)
                self.assertEqual(result, word, f"Protected word '{word}' was incorrectly split into '{result}'")
                text_input = f"මෙම {word} පරීක්ෂා කරන ලදී."
                self.assertEqual(deglue_virama_boundaries(text_input), text_input)

    def test_subword_bpe_healing(self):
        """heal_sinhala_text repairs BPE splits AND virama glue while preserving valid inflected words."""
        raw_text = "කොළඹ ප්‍ර දේශයේදී හෙරොයින්කිලෝවක් සහ රුපියල් මිලියන 2.6ක් සමඟ ඔවුන්ගේ සහ ඔවුන්සතුව තිබූ මුදල් සොයාගෙන ඇත."
        healed = heal_sinhala_text(raw_text)
        self.assertIn("ප්‍රදේශයේදී", healed)
        self.assertIn("හෙරොයින් කිලෝවක්", healed)
        self.assertIn("ඔවුන්ගේ", healed)       # Must preserve genitive inflection
        self.assertIn("ඔවුන් සතුව", healed)   # Must de-glue concatenated words


class TestNumericAndUnitFactualGuardrails(unittest.TestCase):
    """AC 2: Metric mass/weight unit mismatches and sub-24 char glue detection."""

    def test_detect_word_glue_sub_24_char(self):
        """detect_word_glue flags sub-24 char virama-glued tokens and passes clean text."""
        glued_tokens = [
            "හෙරොයින්කිලෝවක්",
            "අනෙකුත්දෙදෙනාගේ",
            "ඔවුන්සතුව",
            "ඔවුන්රැගෙන",
            "දැන්දිගටම",
            "හෙරොයින්ග්රෑම්",
        ]
        for tok in glued_tokens:
            with self.subTest(tok=tok):
                self.assertLess(len(tok), 24, f"Test token '{tok}' should be sub-24 chars")
                reason = detect_word_glue(tok)
                self.assertIsNotNone(reason, f"detect_word_glue failed to catch sub-24 char glued token '{tok}'")
                self.assertTrue(reason.startswith("word_glue:"))

        # Negative check: valid words with internal viramas and inflections are not flagged
        clean_text = (
            "කොළඹ බස්නාහිර ප්‍රදේශයේදී අත්අඩංගුවට ගත් සැකකරුවන්ගේ වාර්තාව "
            "පොලිස්පති කාර්යාලය මගින් මහේස්ත්‍රාත් හමුවට ඉදිරිපත් කිරීමට දැන්වීමක් නිකුත් කළේය."
        )
        self.assertIsNone(detect_word_glue(clean_text), "detect_word_glue falsely flagged clean text")

        # Deglued text passes check
        for tok in glued_tokens:
            healed = heal_sinhala_text(tok)
            self.assertIsNone(detect_word_glue(healed), f"Healed token '{healed}' should pass detect_word_glue")

        # Hal-akuru words glued directly to ASCII digits are detected
        digit_glued = ["රුපියල්2.6", "ග්‍රෑම්77"]
        for dg in digit_glued:
            with self.subTest(dg=dg):
                reason = detect_word_glue(dg)
                self.assertIsNotNone(reason, f"detect_word_glue failed to catch hal-akuru digit glue '{dg}'")
                self.assertTrue(reason.startswith("word_glue:"))
                healed_dg = heal_sinhala_text(dg)
                self.assertIsNone(detect_word_glue(healed_dg), f"Healed '{healed_dg}' should pass detect_word_glue")

        # Clean text containing noun plural/oblique inflections (-වලින්, -වලට) is not flagged
        clean_oblique = "හෙරොයින්වලින් උපයාගත් මුදල් බස්වලින් ප්‍රවාහනය කළ සැකකරුවන් අත්අඩංගුවට ගෙන ඇත."
        self.assertIsNone(detect_word_glue(clean_oblique), "detect_word_glue falsely flagged clean text with oblique suffixes")

    def test_fabricated_kilograms_guardrail(self):
        """Quality check flags summaries that fabricate kilograms when article only mentions grams/milligrams,
        or when the article contains no mass units at all."""
        article = "පොලිස් වැටලීමේදී හෙරොයින් ග්‍රෑම් 77යි මිලිග්‍රෑම් 260ක් (ග්‍රෑම් 77.26ක්) සොයාගෙන ඇත."

        # Hallucination 1: 'හෙරොයින් කිලෝවක්' (a kilo of heroin)
        bad_summary_1 = "වැටලීමේදී හෙරොයින් කිලෝවක් සොයාගෙන තිබේ."
        defect_1 = check_numeric_unit_consistency(bad_summary_1, article)
        self.assertIsNotNone(defect_1, "Failed to catch fabricated 'කිලෝවක්'")
        self.assertIn("fabricated_kilo", defect_1)

        # Hallucination 2: 'හෙරොයින් කිලෝග්‍රෑම් 1ක්'
        bad_summary_2 = "පොලීසිය හෙරොයින් කිලෝග්‍රෑම් 1ක් අත්අඩංගුවට ගත්තේය."
        defect_2 = check_numeric_unit_consistency(bad_summary_2, article)
        self.assertIsNotNone(defect_2, "Failed to catch fabricated 'කිලෝග්‍රෑම්'")
        self.assertIn("fabricated_kilo", defect_2)

        # Hallucination 3: '1kg'
        bad_summary_3 = "පොලීසිය හෙරොයින් 1kg සොයාගෙන ඇත."
        defect_3 = check_numeric_unit_consistency(bad_summary_3, article)
        self.assertIsNotNone(defect_3, "Failed to catch fabricated '1kg'")

        # Hallucination 4: '1KG' (uppercase Latin)
        bad_summary_4 = "පොලීසිය හෙරොයින් 1KG සොයාගෙන ඇත."
        defect_4 = check_numeric_unit_consistency(bad_summary_4, article)
        self.assertIsNotNone(defect_4, "Failed to catch fabricated '1KG'")

        # Hallucination 5: Sinhala word numerals ('කිලෝ එකක්', 'කිලෝග්‍රෑම් දෙකක්')
        bad_summary_5 = "පොලීසිය හෙරොයින් කිලෝ එකක් සොයාගෙන ඇත."
        defect_5 = check_numeric_unit_consistency(bad_summary_5, article)
        self.assertIsNotNone(defect_5, "Failed to catch fabricated 'කිලෝ එකක්'")
        self.assertIn("fabricated_kilo", defect_5)

        bad_summary_5b = "පොලීසිය හෙරොයින් කිලෝග්‍රෑම් දෙකක් සොයාගෙන ඇත."
        defect_5b = check_numeric_unit_consistency(bad_summary_5b, article)
        self.assertIsNotNone(defect_5b, "Failed to catch fabricated 'කිලෝග්‍රෑම් දෙකක්'")
        self.assertIn("fabricated_kilo", defect_5b)

        # Hallucination 6: Fabricated kilograms when article contains NO mass units at all
        article_no_mass = "කොළඹදී රුපියල් මිලියන 5ක මුදල් කොල්ලකා සැකකරුවන් පලාගොස් ඇත."
        bad_summary_no_mass = "කොළඹදී හෙරොයින් කිලෝවක් සහ රුපියල් මිලියන 5ක් කොල්ලකා ඇත."
        defect_no_mass = check_numeric_unit_consistency(bad_summary_no_mass, article_no_mass)
        self.assertIsNotNone(defect_no_mass, "Failed to catch fabricated kilograms when article lacks mass units")
        self.assertIn("fabricated_kilo", defect_no_mass)

        # Faithful summary: mentions grams faithfully
        good_summary = "වැටලීමේදී හෙරොයින් ග්‍රෑම් 77.26ක් සොයාගෙන ඇත."
        self.assertIsNone(check_numeric_unit_consistency(good_summary, article), "Falsely flagged faithful summary")

    def test_negative_metric_non_mass_units(self):
        """Non-mass metric units (e.g. කිලෝමීටර්, කිලෝලීටර්, කිලෝවෝල්ට්) and loanwords (ටෙලිග්‍රෑම්, ප්රෝග්රෑම්) are never confused with kilograms/grams."""
        article = "කොළඹ නගරයේ සිට කිලෝමීටර් 10ක් ඈතින් පිහිටි ස්ථානයකදී හෙරොයින් ග්‍රෑම් 50ක් සොයාගෙන ඇත."

        # Summary cites 50g faithfully and mentions kilometers: should PASS
        valid_summary = "කිලෝමීටර් 10ක් ඔබ්බෙන් වූ ප්‍රදේශයකදී හෙරොයින් ග්‍රෑම් 50ක් සොයාගැනිණි."
        self.assertIsNone(
            check_numeric_unit_consistency(valid_summary, article),
            "Falsely flagged summary mentioning kilometers ('කිලෝමීටර්') as fabricated kilograms"
        )

        # Non-mass metric units: kiloliters, kilovolts, kilobytes
        article_non_mass = "ගබඩාවේ කිලෝලීටර් 10ක ඉන්ධන සහ කිලෝවෝල්ට් 33ක ට්‍රාන්ස්ෆෝමරයක් ඇත."
        self.assertFalse(has_kilo_units(article_non_mass), "Non-mass metric units falsely detected as kilograms")

        # English word 'background' must not match 'kg'
        self.assertFalse(has_kilo_units("The background of this case was reviewed."), "'background' falsely detected as kilograms")

        # Kilograms must not be falsely detected as grams
        article_kilo_only = "පොලීසිය හෙරොයින් කිලෝග්‍රෑම් 5ක් සහ කිලෝ ග්‍රෑම් 2ක් සොයාගෙන ඇත."
        self.assertFalse(has_gram_or_milli_units(article_kilo_only), "Kilograms falsely detected as grams/milligrams")

        # Loanwords like 'ටෙලිග්‍රෑම්' (Telegram), 'ඉන්ස්ටාග්‍රෑම්' (Instagram), and non-ZWJ 'ප්රෝග්රෑම්' (Program) are not treated as grams
        article_telegram = "ජාවාරම්කරුවන් ටෙලිග්‍රෑම් සහ ඉන්ස්ටාග්‍රෑම් මගින් මුදල් ගනුදෙනු සිදුකර ඇත."
        self.assertFalse(
            has_gram_or_milli_units(article_telegram),
            "'ටෙලිග්‍රෑම්' or 'ඉන්ස්ටාග්‍රෑම්' was falsely detected as metric mass unit 'ග්‍රෑම්'"
        )

        article_program = "රූපවාහිනී ප්රෝග්රෑම් විකාශය අත්හිටුවීමට තීරණය කළේය."
        self.assertFalse(
            has_gram_or_milli_units(article_program),
            "'ප්රෝග්රෑම්' (Program without ZWJ) was falsely detected as metric mass unit 'ග්‍රෑම්'"
        )

        # Telecom 5G/4G network generations must not be detected as grams
        article_5g = "ශ්‍රී ලංකා ටෙලිකොම් ආයතනය 5G තාක්ෂණය සහ 4G සබඳතා ව්‍යාප්ත කරයි."
        self.assertFalse(has_gram_or_milli_units(article_5g), "5G/4G falsely detected as grams")

    def test_mass_number_unit_swaps(self):
        """Catches when a number stated as grams/milligrams is swapped to kilograms."""
        article = "ගංජා ග්‍රෑම් 460ක් සහ හෙරොයින් ග්රෑම් 150ක් සොයාගෙන ඇත."

        # 460kg instead of 460g
        bad_summary = "ගංජා කිලෝග්‍රෑම් 460ක් සොයාගෙන ඇත."
        defect = check_numeric_unit_consistency(bad_summary, article)
        self.assertIsNotNone(defect, "Failed to catch 460g -> 460kg unit swap")
        self.assertTrue(defect.startswith("unit_mismatch:460"))

        # 150kg instead of 150g
        bad_summary_2 = "හෙරොයින් කිලෝ 150ක් සොයාගෙන ඇත."
        defect_2 = check_numeric_unit_consistency(bad_summary_2, article)
        self.assertIsNotNone(defect_2, "Failed to catch 150g -> 150kg unit swap")

    def test_mass_mathematical_conversions_and_sinhala_numerals(self):
        """Valid mathematical unit conversions (1000g = 1kg) expressed via digits or Sinhala numeral words pass."""
        article_1000g = "වැටලීමේදී හෙරොයින් ග්‍රෑම් 1000ක් සොයාගෙන ඇත."
        summary_1kg_digits = "වැටලීමේදී හෙරොයින් කිලෝ 1ක් සොයාගෙන ඇත."
        self.assertIsNone(check_numeric_unit_consistency(summary_1kg_digits, article_1000g))

        summary_1kg_words = "වැටලීමේදී හෙරොයින් කිලෝ එකක් සොයාගෙන ඇත."
        self.assertIsNone(check_numeric_unit_consistency(summary_1kg_words, article_1000g))

        article_1000g_words = "වැටලීමේදී හෙරොයින් ග්‍රෑම් දහසක් සොයාගෙන ඇත."
        self.assertIsNone(check_numeric_unit_consistency(summary_1kg_words, article_1000g_words))

        # Numeral ordering and inflected Arabic digits
        pairs_rev = extract_mass_unit_pairs("එකක් කිලෝ")
        self.assertEqual(pairs_rev, [(1.0, "කිලෝග්‍රෑම්")])

        pairs_arabic_inflected = extract_mass_unit_pairs("1000ක් ග්‍රෑම්")
        self.assertEqual(pairs_arabic_inflected, [(1000.0, "ග්‍රෑම්")])

        pairs_half_kilo = extract_mass_unit_pairs("කිලෝ අඩක්")
        self.assertEqual(pairs_half_kilo, [(0.5, "කිලෝග්‍රෑම්")])

        # Multi-kilo summary where only one has mathematical basis: MUST be caught
        bad_partial_basis = "වැටලීමේදී හෙරොයින් කිලෝ 1ක් සහ අයිස් කිලෝ 50ක් සොයාගෙන ඇත."
        defect_partial = check_numeric_unit_consistency(bad_partial_basis, article_1000g)
        self.assertIsNotNone(defect_partial, "Failed to catch fabricated kilo when another kilo had mathematical basis")
        self.assertIn("fabricated_kilo", defect_partial)

        # Swapping 1kg to 1g with word numerals is caught
        article_kilo = "වැටලීමේදී හෙරොයින් කිලෝ එකක් සොයාගෙන ඇත."
        summary_swapped_gram = "වැටලීමේදී හෙරොයින් ග්‍රෑම් එකක් සොයාගෙන ඇත."
        self.assertIsNotNone(check_numeric_unit_consistency(summary_swapped_gram, article_kilo))

    def test_currency_scale_consistency_preserved(self):
        """Existing currency scale words check (ලක්ෂ vs මිලියන) remains fully operational."""
        article = "රුපියල් මිලියන 476ක වත්කම් උපයා ඇත."
        bad_summary = "රුපියල් ලක්ෂ 476ක වත්කම් උපයා ඇත."
        good_summary = "රුපියල් මිලියන 476ක වත්කම් සොයාගෙන ඇත."

        self.assertIsNotNone(check_numeric_unit_consistency(bad_summary, article))
        self.assertIsNone(check_numeric_unit_consistency(good_summary, article))


class TestPoliceDrugRaidBenchmark(unittest.TestCase):
    """AC 3: Police drug raid benchmark article verification."""

    def setUp(self):
        self.assertTrue(BENCHMARK_PATH.exists(), f"Benchmark fixture not found at {BENCHMARK_PATH}")
        self.benchmark_data = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))

    def test_benchmark_article_verified_summaries(self):
        """Short, Medium, and Long summaries cite true quantities without hallucinating 1kg."""
        article = self.benchmark_data["content"].strip()
        summaries = self.benchmark_data["verified_summaries"]

        # Expected true facts
        forbidden_hallucinations = ["1kg", "1 kg", "කිලෝ 1", "කිලෝග්‍රෑම් 1", "කිලෝවක්", "1KG", "1 Kg", "1Kg", "1.0kg", "කිලෝ 1ක්", "කිලෝ එකක්"]

        for bucket in ("short", "medium", "long"):
            summary = summaries[bucket]
            with self.subTest(bucket=bucket):
                # 1. No word glue defects
                glue_defect = detect_word_glue(summary)
                self.assertIsNone(glue_defect, f"{bucket} summary contains glue defect: {glue_defect}")

                # 2. No unit consistency defects
                unit_defect = check_numeric_unit_consistency(summary, article)
                self.assertIsNone(unit_defect, f"{bucket} summary failed unit consistency: {unit_defect}")

                # 3. No 1kg hallucinations
                for forbidden in forbidden_hallucinations:
                    self.assertNotIn(
                        forbidden,
                        summary,
                        f"{bucket} summary contains hallucinated 1kg term '{forbidden}'"
                    )

                # 4. Contains accurate quantities (~77.26g heroin and Rs. 2.6M+ cash)
                self.assertTrue(
                    ("77.26" in summary or "77" in summary),
                    f"{bucket} summary missing ~77.26g heroin quantity"
                )
                self.assertTrue(
                    ("මිලියන 2.6" in summary or "2,650,000" in summary),
                    f"{bucket} summary missing Rs. 2.6M+ cash quantity"
                )

    def test_benchmark_rejects_hallucinated_1kg_summary(self):
        """The quality guardrail reliably rejects an adulterated 1kg summary for the benchmark article."""
        article = self.benchmark_data["content"].strip()

        hallucination_variants = [
            "කොළඹදී සිදුකළ වැටලීමකදී හෙරොයින් කිලෝවක් (1kg) සහ රුපියල් මිලියන 2.6ක මුදල් සමඟ සැකකරුවන් තිදෙනෙකු අත්අඩංගුවට ගෙන ඇත.",
            "කොළඹදී සිදුකළ වැටලීමකදී හෙරොයින් කිලෝ 1ක් සහ රුපියල් මිලියන 2.6ක මුදල් සොයාගෙන ඇත.",
            "කොළඹදී සිදුකළ වැටලීමකදී හෙරොයින් 1Kg සොයාගෙන ඇත.",
            "කොළඹදී සිදුකළ වැටලීමකදී හෙරොයින් කිලෝ එකක් සොයාගෙන ඇත.",
        ]
        for bad_summary in hallucination_variants:
            with self.subTest(bad_summary=bad_summary):
                defect = check_numeric_unit_consistency(bad_summary, article)
                self.assertIsNotNone(defect, f"Failed to reject benchmark summary: '{bad_summary}'")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  RUNNING SINHALA LLM SUMMARIZER GUARDRAIL TEST SUITE")
    print("=" * 70 + "\n")
    unittest.main(verbosity=2)
