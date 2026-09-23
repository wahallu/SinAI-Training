#!/usr/bin/env python3
"""
Automated Test Suite: Sinhala LLM Summarizer Guardrails
=======================================================
Mirror runner in abstractive directory.
"""

import sys
import unittest
from pathlib import Path

# Add parent directory to sys.path so the main test suite can be imported
parent_dir = Path(__file__).resolve().parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

import test_summarizer_guardrails
from test_summarizer_guardrails import *

if __name__ == "__main__":
    unittest.main(module=test_summarizer_guardrails, verbosity=2)
