"""
SinhalaJournal-LLM | Shared teacher-data quality checks
--------------------------------------------------------
Forwarder module for data_quality_checks in the summarizer parent directory.
"""

from __future__ import annotations

import sys
from pathlib import Path

_abstractive_dir = Path(__file__).resolve().parent / "abstractive"
if str(_abstractive_dir) not in sys.path:
    sys.path.insert(0, str(_abstractive_dir))

from data_quality_checks import (
    GLUE_TOKEN_RE,
    MAX_GLUE_TOKEN_CHARS,
    UNIT_SCALE,
    NUMBER_UNIT_RE,
    SINHALA_WORD_NUMBERS,
    MASS_NUMBER_UNIT_RE,
    detect_word_glue,
    extract_number_unit_pairs,
    extract_mass_unit_pairs,
    has_kilo_units,
    has_gram_or_milli_units,
    check_mass_unit_consistency,
    check_numeric_unit_consistency,
)
