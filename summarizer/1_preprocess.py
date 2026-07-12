"""
SinhalaJournal-LLM | Step 1: Dataset Preprocessing
----------------------------------------------------
Cleans and filters your scraped Sinhala news JSON dataset.
Output: A filtered JSONL file ready for training.

Usage:
    python 01_preprocess.py --input /path/to/your_dataset.json --output cleaned_dataset.jsonl
"""

import json
import re
import argparse
import unicodedata
from pathlib import Path
from collections import defaultdict


# ──────────────────────────────────────────────
# CONFIG — adjust these thresholds if needed
# ──────────────────────────────────────────────
MIN_CONTENT_CHARS = 300     # discard very short articles
MAX_CONTENT_CHARS = 10_000  # discard extremely long articles (outliers)
MIN_TITLE_CHARS   = 10      # discard one-word or empty titles
MAX_TITLE_CHARS   = 200     # discard unusually long titles


# ──────────────────────────────────────────────
# SINHALA UNICODE RANGE
# U+0D80 – U+0DFF
# ──────────────────────────────────────────────
SINHALA_PATTERN = re.compile(r'[\u0D80-\u0DFF]')


def clean_text(text: str) -> str:
    """Remove HTML tags, extra whitespace, and normalize Unicode."""
    if not text:
        return ""
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    # Remove special control characters but keep Sinhala Unicode
    text = unicodedata.normalize('NFC', text)
    # Collapse multiple whitespace/newlines
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def is_sinhala(text: str, threshold: float = 0.3) -> bool:
    """Return True if at least `threshold` fraction of chars are Sinhala."""
    if not text:
        return False
    sinhala_chars = len(SINHALA_PATTERN.findall(text))
    return (sinhala_chars / len(text)) >= threshold


def is_valid(article: dict) -> tuple[bool, str]:
    """Validate a single article. Returns (valid, reason)."""
    title   = article.get("title", "") or ""
    content = article.get("content", "") or ""

    if not title or not content:
        return False, "missing_field"

    if len(content) < MIN_CONTENT_CHARS:
        return False, "content_too_short"

    if len(content) > MAX_CONTENT_CHARS:
        return False, "content_too_long"

    if len(title) < MIN_TITLE_CHARS:
        return False, "title_too_short"

    if len(title) > MAX_TITLE_CHARS:
        return False, "title_too_long"

    if not is_sinhala(content):
        return False, "not_sinhala_content"

    if not is_sinhala(title):
        return False, "not_sinhala_title"

    return True, "ok"


def load_json_dataset(path: str) -> list[dict]:
    """Load JSON — handles both a list [] and newline-delimited JSONL."""
    path = Path(path)
    articles = []

    with open(path, "r", encoding="utf-8") as f:
        first_char = f.read(1)
        f.seek(0)

        if first_char == "[":
            # Standard JSON array
            articles = json.load(f)
        else:
            # Newline-delimited JSON
            for line in f:
                line = line.strip()
                if line:
                    articles.append(json.loads(line))

    return articles


def main(input_path: str, output_path: str):
    print(f"\n📂 Loading dataset from: {input_path}")
    articles = load_json_dataset(input_path)
    print(f"   Total raw articles: {len(articles):,}")

    # ── Deduplication by URL ─────────────────────
    seen_urls = set()
    unique_articles = []
    for a in articles:
        url = a.get("url") or a.get("URL") or ""
        if url and url in seen_urls:
            continue
        seen_urls.add(url)
        unique_articles.append(a)

    print(f"   After URL deduplication: {len(unique_articles):,}")

    # ── Clean + Filter ───────────────────────────
    stats = defaultdict(int)
    cleaned = []

    for article in unique_articles:
        article["title"]   = clean_text(article.get("title", ""))
        article["content"] = clean_text(article.get("content", ""))

        valid, reason = is_valid(article)
        stats[reason] += 1

        if valid:
            cleaned.append({
                "title":          article["title"],
                "content":        article["content"],
                "category":       article.get("mapped_category", ""),
                "url":            article.get("url", ""),
                "date_published": article.get("date_published", ""),
            })

    # ── Write Output ─────────────────────────────
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for item in cleaned:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # ── Report ───────────────────────────────────
    print(f"\n✅ Preprocessing complete!")
    print(f"   Kept  : {stats['ok']:,} articles → {output_path}")
    print(f"\n   Dropped breakdown:")
    for reason, count in sorted(stats.items()):
        if reason != "ok":
            print(f"     {reason:<25} {count:>8,}")
    print(f"\n   Final dataset size: {stats['ok']:,} article-title pairs")
    print(f"   Ready for: 02_build_dataset.py\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Sinhala news dataset")
    parser.add_argument("--input",  required=True, help="Path to raw JSON dataset")
    parser.add_argument("--output", default="data/cleaned_dataset.jsonl",
                        help="Output JSONL file path")
    args = parser.parse_args()
    main(args.input, args.output)
