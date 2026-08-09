"""
SinhalaJournal-LLM | One-off cleaning pass for 6_multilength_summaries.jsonl
------------------------------------------------------------------------------
Drops records whose summary_short/medium/long fields contain a
word-spacing-glue or numeric-unit-mismatch defect (see
data_quality_checks.py), producing a *_clean.jsonl sibling file for
7_train_summarizer.py to train on — same "fix the data, not the model" move
as the sibling headline pipeline's clean_headline_dataset.py.

This is a pure local re-scan + filter: no API calls, no regeneration of
dropped records. The same two checks are also enforced going forward at
generation time (7_multilength_summary_generator.py) and train time
(7_train_summarizer.py) for defense in depth.

Run against the full 35,569-record 6_multilength_summaries.jsonl corpus,
this drops 22 records (21 genuine unit-scale/word-glue defects, manually
verified against full article context — mostly 100x-1000x errors like
"455 billion" rendered as "455 million"; 1 accepted false positive from
coincidental digit collision between two unrelated but individually-correct
figures in a dense article — a known limitation of the rounding-based
numeric-unit check, not worth chasing further for a 1-in-35k cost).

Usage:
    python abstractive/clean_multilength_dataset.py
    python abstractive/clean_multilength_dataset.py \
        --input data/6_multilength_summaries.jsonl \
        --output data/6_multilength_summaries_clean.jsonl
"""

import json
import argparse
from pathlib import Path
from collections import Counter

from data_quality_checks import detect_word_glue, check_numeric_unit_consistency

BUCKETS = ("summary_short", "summary_medium", "summary_long")


def find_defect(rec: dict) -> tuple[str, str] | None:
    """Returns (bucket, reason) for the first defective summary field found
    in this record, or None if the record is clean."""
    article = rec.get("content", "").strip()
    for bucket in BUCKETS:
        summary = rec.get(bucket, "").strip()
        if not summary:
            continue
        reason = detect_word_glue(summary)
        if reason is None and article:
            reason = check_numeric_unit_consistency(summary, article)
        if reason:
            return bucket, reason
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Drop word-glue / numeric-unit-mismatch records from the multi-length summary dataset"
    )
    parser.add_argument("--input", default="/home/jovyan/summarizer/data/6_multilength_summaries.jsonl")
    parser.add_argument("--output", default="/home/jovyan/summarizer/data/6_multilength_summaries_clean.jsonl")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    read = 0
    kept = 0
    dropped = []
    reason_counts = Counter()

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            read += 1

            defect = find_defect(rec)
            if defect is None:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                kept += 1
            else:
                bucket, reason = defect
                dropped.append((line_no, rec.get("url", ""), bucket, reason))
                reason_kind = reason.split(":", 1)[0]
                reason_counts[reason_kind] += 1

    print("=" * 70)
    print(" Multi-length summary dataset cleaning")
    print("=" * 70)
    print(f" Input          : {input_path}")
    print(f" Output         : {output_path}")
    print(f" Records read   : {read:,}")
    print(f" Records kept   : {kept:,}")
    print(f" Records dropped: {len(dropped):,}  ({dict(reason_counts)})")
    print("=" * 70)

    if dropped:
        print("\nDropped record detail:")
        for line_no, url, bucket, reason in dropped:
            print(f"  line {line_no:<7} bucket={bucket:<14} reason={reason}  url={url}")

    print(f"\nOutput written to: {output_path}")


if __name__ == "__main__":
    main()
