"""
SinhalaJournal-LLM | Step 8a: Frozen article-level dataset split
-------------------------------------------------------------------
Fixes the train/eval leakage in 6_train_summarizer.py / 7_train_summarizer.py:
both explode each article into up to 3 samples (short/medium/long), shuffle
the *flat sample list*, then split 85/15 — so different length-variants of
the same article can land in both train and val/eval.

Split from the RAW corpus (6_multilength_summaries.jsonl, 35,569 rows,
the full article universe v06 trains on), not the cleaned 35,547-row file.
This matters for retraining v06 and v07 on comparable frozen partitions:
v06's dataset and v07's dataset (raw minus 22 rows dropped for word-glue/
number-unit/unrelated-content defects — pure row removal, verified
byte-identical content on the 35,547 shared rows) are DIFFERENT datasets,
and that difference is the actual thing being compared. Building the split
from the raw superset means both recipes share identical split boundaries
(which articles are held out is decided once, on the full universe); v07's
own quality filter (data_quality_checks.py, applied at training time in
7_train_summarizer.py's build_samples()) then naturally drops the ~22 bad
rows (and similar cases) from whatever lands in ITS training partition,
without needing a separate materialized "v07 split". Building the split
from the already-cleaned file instead (as an earlier version of this script
did) would silently retrain "v06" on v07's dataset, erasing the variable
under test.

The raw corpus already has exactly one row per article (verified: row count
== unique `url` count), with summary_short/medium/long as columns on that
one row. So an article-level split needs no grouping step — splitting the
rows themselves is already article-level, by construction.

This script performs that split ONCE, with a fixed seed, and persists the
result to disk. The output files must not be regenerated with a different
seed/ratio once anything has been evaluated or trained against them — that
would silently break comparability across runs, exactly the failure mode
this script exists to prevent.

Also draws a fixed, bounded evaluation subset from the test partition (see
EVAL_SUBSET_SIZE) so that repeated evaluation runs don't keep resampling —
"frozen" means frozen, not "resampled every run from a frozen pool".

Usage:
    python abstractive/8_freeze_dataset_split.py

Output:
    data/summarization_frozen_train.jsonl
    data/summarization_frozen_val.jsonl
    data/summarization_frozen_test.jsonl
    data/summarization_frozen_eval_subset.jsonl
    data/summarization_frozen_split_manifest.json
"""

import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
DATA_DIR = Path("/home/jovyan/summarizer/data")
SOURCE_PATH = DATA_DIR / "6_multilength_summaries.jsonl"  # raw superset, not the cleaned file — see module docstring

TRAIN_PATH = DATA_DIR / "summarization_frozen_train.jsonl"
VAL_PATH = DATA_DIR / "summarization_frozen_val.jsonl"
TEST_PATH = DATA_DIR / "summarization_frozen_test.jsonl"
EVAL_SUBSET_PATH = DATA_DIR / "summarization_frozen_eval_subset.jsonl"
MANIFEST_PATH = DATA_DIR / "summarization_frozen_split_manifest.json"

SEED = 42  # same project-wide seed used by 6/7_train_summarizer.py
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
# TEST_RATIO is the remainder (0.10)

# Fixed size of the persisted evaluation subset drawn from the test
# partition. Bounds GPU/API cost for repeated evaluation runs. The subset is
# the first EVAL_SUBSET_SIZE rows of the (already globally shuffled) test
# partition, so it's a fixed, reproducible slice — not a fresh sample.
EVAL_SUBSET_SIZE = 300


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_urls(urls: list) -> str:
    h = hashlib.sha256()
    for u in sorted(urls):
        h.update(u.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def load_jsonl(path: Path) -> list:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: list) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main():
    print("\n" + "=" * 60)
    print("  SinhalaJournal-LLM | Frozen article-level split")
    print("=" * 60)

    for existing in (TRAIN_PATH, VAL_PATH, TEST_PATH, EVAL_SUBSET_PATH, MANIFEST_PATH):
        if existing.exists():
            raise SystemExit(
                f"Refusing to overwrite existing frozen split file: {existing}\n"
                "The whole point of a frozen split is that it's written once. "
                "If you deliberately need to rebuild it, delete the existing "
                "summarization_frozen_*.jsonl / manifest files first, and "
                "understand that this invalidates comparability with any "
                "training/eval run that already used the old split."
            )

    print(f"\nSource: {SOURCE_PATH}")
    records = load_jsonl(SOURCE_PATH)
    n = len(records)
    print(f"  rows: {n:,}")

    urls = [r.get("url", "") for r in records]
    if len(set(urls)) != n or any(not u for u in urls):
        raise SystemExit(
            "Source file is not one row per article (duplicate or missing "
            "`url` values found) — the article-level split assumption this "
            "script depends on does not hold. Aborting rather than silently "
            "producing a leaky split."
        )
    print("  url uniqueness check passed (rows == unique urls)")

    source_hash = sha256_file(SOURCE_PATH)
    print(f"  sha256: {source_hash}")

    rng = random.Random(SEED)
    shuffled = records[:]
    rng.shuffle(shuffled)

    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    train_records = shuffled[:n_train]
    val_records = shuffled[n_train:n_train + n_val]
    test_records = shuffled[n_train + n_val:]

    eval_subset = test_records[:min(EVAL_SUBSET_SIZE, len(test_records))]

    print(f"\n  train : {len(train_records):,}")
    print(f"  val   : {len(val_records):,}")
    print(f"  test  : {len(test_records):,}")
    print(f"  eval_subset (fixed, first {EVAL_SUBSET_SIZE} of test): {len(eval_subset):,}")

    write_jsonl(TRAIN_PATH, train_records)
    write_jsonl(VAL_PATH, val_records)
    write_jsonl(TEST_PATH, test_records)
    write_jsonl(EVAL_SUBSET_PATH, eval_subset)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_path": str(SOURCE_PATH),
        "source_sha256": source_hash,
        "source_row_count": n,
        "seed": SEED,
        "ratios": {"train": TRAIN_RATIO, "val": VAL_RATIO, "test": round(1 - TRAIN_RATIO - VAL_RATIO, 4)},
        "eval_subset_size_requested": EVAL_SUBSET_SIZE,
        "splits": {
            "train": {
                "path": str(TRAIN_PATH),
                "count": len(train_records),
                "url_list_sha256": sha256_urls([r["url"] for r in train_records]),
            },
            "val": {
                "path": str(VAL_PATH),
                "count": len(val_records),
                "url_list_sha256": sha256_urls([r["url"] for r in val_records]),
            },
            "test": {
                "path": str(TEST_PATH),
                "count": len(test_records),
                "url_list_sha256": sha256_urls([r["url"] for r in test_records]),
            },
            "eval_subset": {
                "path": str(EVAL_SUBSET_PATH),
                "count": len(eval_subset),
                "url_list_sha256": sha256_urls([r["url"] for r in eval_subset]),
            },
        },
    }
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Wrote frozen split + manifest to {DATA_DIR}")
    print(f"   {MANIFEST_PATH.name}")
    print("\n   This split is now frozen. Do not regenerate it with a "
          "different seed/ratio without invalidating prior comparisons.")


if __name__ == "__main__":
    main()
