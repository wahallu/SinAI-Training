"""
Extractive summarizer task specs for task_registry and serve_sinai gateway.
"""

import sys
from pathlib import Path

# Add summarizer package to path if needed
_SUMMARIZER_DIR = Path(__file__).resolve().parent.parent.parent / "summarizer"
if str(_SUMMARIZER_DIR) not in sys.path:
    sys.path.insert(0, str(_SUMMARIZER_DIR))

from extractive import (
    extractive_summarize,
    tfidf_summarize,
    textrank_summarize,
    rake_summarize,
    yake_summarize,
    keybert_summarize,
)


def prompt_extractive(text: str, **_) -> str:
    """Pass-through prompt builder for extractive summarization."""
    return text


def max_new_tokens(raw_text: str, prompt_token_len: int) -> int:
    return 180


REPETITION_PENALTY = 1.0


def run_extractive(method: str, text: str, n: int = 3) -> str:
    """Executes the specified extractive algorithm on the input text."""
    return extractive_summarize(text, method=method, n=n)
