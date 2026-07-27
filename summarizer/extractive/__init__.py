"""
Extractive Summarization Package for Sinhala.

Provides extractive summarizers:
  - TF-IDF
  - TextRank
  - RAKE
  - YAKE
  - KeyBERT
"""

from .tfidf import tfidf_summarize
from .textrank import textrank_summarize
from .rake import rake_summarize
from .yake import yake_summarize
from .keybert import keybert_summarize

EXTRACTIVE_METHODS = {
    "tfidf": tfidf_summarize,
    "textrank": textrank_summarize,
    "rake": rake_summarize,
    "yake": yake_summarize,
    "keybert": keybert_summarize,
}


def extractive_summarize(text: str, method: str = "textrank", n: int = 3) -> str:
    """Dispatches to the requested extractive summarizer method.
    
    Supported methods: 'tfidf', 'textrank', 'rake', 'yake', 'keybert'
    """
    method_norm = method.lower().replace("extractive_", "").strip()
    fn = EXTRACTIVE_METHODS.get(method_norm, textrank_summarize)
    return fn(text, n=n)


__all__ = [
    "tfidf_summarize",
    "textrank_summarize",
    "rake_summarize",
    "yake_summarize",
    "keybert_summarize",
    "extractive_summarize",
    "EXTRACTIVE_METHODS",
]
