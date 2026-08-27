"""
SinhalaJournal-LLM | Step 8b: Split-contamination audit
-------------------------------------------------------------------
Measures exactly how much of the new frozen test set / eval subset overlaps
the training data the EXISTING v06 and v07 adapters were actually trained
on — so the discount applied to their existing eval numbers is a measured
fact, not a qualitative "leakage risk" gesture.

How: replays 6_train_summarizer.py's and 7_train_summarizer.py's
build_samples() + random.seed(SEED) + random.shuffle() + 85/15 split logic
verbatim (same data files, same filters, same seed), but tracks (url,
bucket) identity instead of building the actual training prompt strings.
random.shuffle() only depends on list length/order, not element content, so
shuffling (url, bucket) tuples in the same order the original code would
have shuffled its formatted-prompt strings reproduces the exact same
permutation — i.e. this reconstructs precisely which articles/buckets each
adapter's original training run put in "train" vs "val", without needing to
load the model or unsloth (CPU-only, just the tokenizer, for the same
token-count filters the original scripts apply).

Usage:
    python abstractive/8_audit_split_contamination.py

Requires: abstractive/8_freeze_dataset_split.py must have already been run
(reads data/summarization_frozen_test.jsonl and ..._eval_subset.jsonl).
"""

import json
import random
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

from transformers import AutoTokenizer

from data_quality_checks import detect_word_glue, check_numeric_unit_consistency

# ──────────────────────────────────────────────
# PATHS (mirrors 6_train_summarizer.py / 7_train_summarizer.py exactly)
# ──────────────────────────────────────────────
SINLLAMA_BASE = "/home/jovyan/work/sinllama/models/SinLLaMA-merged-base"
DATA_DIR = Path("/home/jovyan/summarizer/data")

V06_TRAIN_DATA_PATH = DATA_DIR / "6_multilength_summaries.jsonl"
V07_TRAIN_DATA_PATH = DATA_DIR / "6_multilength_summaries_clean.jsonl"
QWEN_SUPPLEMENT_PATH = DATA_DIR / "5_qwen_summaries.jsonl"

FROZEN_TEST_PATH = DATA_DIR / "summarization_frozen_test.jsonl"
FROZEN_EVAL_SUBSET_PATH = DATA_DIR / "summarization_frozen_eval_subset.jsonl"

RESULTS_DIR = Path("/home/jovyan/summarizer/6_eval_results")

# ──────────────────────────────────────────────
# CONFIG (must match 6/7_train_summarizer.py exactly)
# ──────────────────────────────────────────────
SEED = 42
TRAIN_SPLIT = 0.85
MIN_ARTICLE_TOKENS = 50
MIN_SUMMARY_TOKENS = 10
BUCKET_FILTERS = {
    "short":  {"min_ratio": 0.04, "max_ratio": 0.18, "max_summary_tokens": 70},
    "medium": {"min_ratio": 0.12, "max_ratio": 0.32, "max_summary_tokens": 120},
    "long":   {"min_ratio": 0.22, "max_ratio": 0.55, "max_summary_tokens": 190},
}


def load_jsonl(path: Path) -> list:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


# ── Quality gates, verbatim from 6_train_summarizer.py / 7_train_summarizer.py ──
def summary_is_clean_v06(summary: str) -> bool:
    if not summary:
        return False
    if "�" in summary:
        return False
    if summary.startswith("#") or "\n#" in summary:
        return False
    if unicodedata.combining(summary[0]):
        return False
    return True


def summary_is_clean_v07(summary: str, article: str) -> bool:
    if not summary_is_clean_v06(summary):
        return False
    if detect_word_glue(summary):
        return False
    if check_numeric_unit_consistency(summary, article):
        return False
    return True


def build_sample_identities(train_data_path: Path, quality_fn, tokenizer, needs_article: bool) -> list:
    """Replays build_samples()'s filtering/order exactly, but returns
    (url, bucket) identities instead of formatted prompt strings — enough to
    reproduce the identical shuffle permutation and measure overlap."""
    samples = []
    seen_pairs = set()

    def try_add(article: str, summary: str, bucket: str, url: str):
        key = (url, bucket)
        if url and key in seen_pairs:
            return
        cfg = BUCKET_FILTERS[bucket]

        clean = quality_fn(summary, article) if needs_article else quality_fn(summary)
        if not article or not clean:
            return

        article_tokens = len(tokenizer.encode(article, add_special_tokens=False))
        summary_tokens = len(tokenizer.encode(summary, add_special_tokens=False))

        if article_tokens < MIN_ARTICLE_TOKENS:
            return
        if summary_tokens < MIN_SUMMARY_TOKENS or summary_tokens > cfg["max_summary_tokens"]:
            return
        ratio = summary_tokens / article_tokens
        if ratio < cfg["min_ratio"] or ratio > cfg["max_ratio"]:
            return

        samples.append((url, bucket))
        seen_pairs.add(key)

    for rec in load_jsonl(train_data_path):
        article = rec.get("content", "").strip()
        url = rec.get("url", "")
        for bucket in BUCKET_FILTERS:
            summary = rec.get(f"summary_{bucket}", "").strip()
            if summary:
                try_add(article, summary, bucket, url)

    if QWEN_SUPPLEMENT_PATH.exists():
        for rec in load_jsonl(QWEN_SUPPLEMENT_PATH):
            article = rec.get("content", "").strip()
            summary = rec.get("qwen_summary", "").strip()
            if article and summary:
                try_add(article, summary, "long", rec.get("url", ""))

    return samples


def reconstruct_train_val(sample_identities: list) -> tuple:
    """Byte-for-byte reproduction of main()'s random.seed(SEED) -> shuffle
    -> 85/15 split, applied to (url, bucket) identities instead of prompt
    strings. build_samples() consumes no random-module calls in the
    original scripts, so the random state at shuffle-time is identical to
    seeding immediately before shuffling here."""
    ordered = list(sample_identities)
    random.seed(SEED)
    random.shuffle(ordered)
    n_train = int(len(ordered) * TRAIN_SPLIT)
    return ordered[:n_train], ordered[n_train:]


def summarize_adapter(name: str, train_pairs: list, val_pairs: list, frozen_test_urls: set, frozen_eval_urls: set) -> dict:
    train_urls = {u for u, _ in train_pairs}
    val_urls = {u for u, _ in val_pairs}
    straddling = train_urls & val_urls  # articles whose buckets split across the adapter's own train/val

    test_overlap = train_urls & frozen_test_urls
    eval_overlap = train_urls & frozen_eval_urls

    report = {
        "adapter": name,
        "total_samples": len(train_pairs) + len(val_pairs),
        "train_samples": len(train_pairs),
        "val_samples": len(val_pairs),
        "train_unique_articles": len(train_urls),
        "own_split_straddling_articles": len(straddling),
        "own_split_straddling_pct_of_train_articles": round(100 * len(straddling) / max(1, len(train_urls)), 2),
        "frozen_test_size": len(frozen_test_urls),
        "frozen_test_contaminated_articles": len(test_overlap),
        "frozen_test_contamination_pct": round(100 * len(test_overlap) / max(1, len(frozen_test_urls)), 2),
        "frozen_eval_subset_size": len(frozen_eval_urls),
        "frozen_eval_subset_contaminated_articles": len(eval_overlap),
        "frozen_eval_subset_contamination_pct": round(100 * len(eval_overlap) / max(1, len(frozen_eval_urls)), 2),
    }
    return report


def main():
    print("\n" + "=" * 60)
    print("  SinhalaJournal-LLM | Split-contamination audit")
    print("=" * 60)

    for p in (FROZEN_TEST_PATH, FROZEN_EVAL_SUBSET_PATH):
        if not p.exists():
            raise SystemExit(f"Missing {p} — run abstractive/8_freeze_dataset_split.py first.")

    frozen_test_urls = {r["url"] for r in load_jsonl(FROZEN_TEST_PATH)}
    frozen_eval_urls = {r["url"] for r in load_jsonl(FROZEN_EVAL_SUBSET_PATH)}
    print(f"\nFrozen test set: {len(frozen_test_urls):,} articles")
    print(f"Frozen eval subset: {len(frozen_eval_urls):,} articles")

    print("\nLoading tokenizer (CPU, no model/unsloth needed)...")
    tokenizer = AutoTokenizer.from_pretrained(SINLLAMA_BASE, local_files_only=True)

    print(f"\nReplaying v06 build_samples() on {V06_TRAIN_DATA_PATH.name} ...")
    v06_samples = build_sample_identities(V06_TRAIN_DATA_PATH, summary_is_clean_v06, tokenizer, needs_article=False)
    v06_train, v06_val = reconstruct_train_val(v06_samples)
    print(f"  samples: {len(v06_samples):,}  train: {len(v06_train):,}  val: {len(v06_val):,}")

    print(f"\nReplaying v07 build_samples() on {V07_TRAIN_DATA_PATH.name} ...")
    v07_samples = build_sample_identities(V07_TRAIN_DATA_PATH, summary_is_clean_v07, tokenizer, needs_article=True)
    v07_train, v07_val = reconstruct_train_val(v07_samples)
    print(f"  samples: {len(v07_samples):,}  train: {len(v07_train):,}  val: {len(v07_val):,}")

    v06_report = summarize_adapter("summarization_sinllama_v06", v06_train, v06_val, frozen_test_urls, frozen_eval_urls)
    v07_report = summarize_adapter("summarization_sinllama_v07", v07_train, v07_val, frozen_test_urls, frozen_eval_urls)

    print("\n" + "-" * 60)
    for report in (v06_report, v07_report):
        print(f"\n{report['adapter']}:")
        print(f"  train articles                 : {report['train_unique_articles']:,}")
        print(f"  own-split straddling articles  : {report['own_split_straddling_articles']:,} "
              f"({report['own_split_straddling_pct_of_train_articles']}% of train articles)")
        print(f"  frozen TEST contamination      : {report['frozen_test_contaminated_articles']:,} / "
              f"{report['frozen_test_size']:,} ({report['frozen_test_contamination_pct']}%)")
        print(f"  frozen EVAL SUBSET contamination: {report['frozen_eval_subset_contaminated_articles']:,} / "
              f"{report['frozen_eval_subset_size']:,} ({report['frozen_eval_subset_contamination_pct']}%)")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"split_contamination_audit_{stamp}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "seed": SEED,
                "train_split": TRAIN_SPLIT,
                "frozen_test_path": str(FROZEN_TEST_PATH),
                "frozen_eval_subset_path": str(FROZEN_EVAL_SUBSET_PATH),
                "v06": v06_report,
                "v07": v07_report,
            },
            f, ensure_ascii=False, indent=2,
        )
    print(f"\n✅ Wrote {out_path}")


if __name__ == "__main__":
    main()
