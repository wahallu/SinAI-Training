#!/usr/bin/env python3
"""Rescore saved grammar-evaluation transcripts against old and repaired gold.

The original evaluator prints one INPUT/PREDICT/EXPECTED block per example.
This script extracts the saved predictions, aligns them by stage and row index,
and applies the evaluator's exact-match rule: stripped prediction equals
stripped reference. It does not run a model or regenerate predictions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from itertools import combinations
from pathlib import Path
from typing import Any


STAGES = ("stage2", "stage3", "stage4", "stage5")
STAGE_RE = re.compile(r"^EVALUATION — (stage\d+)")
FIELD_RE = re.compile(r"^\s+(INPUT|PREDICT|EXPECTED)\s*: (.*)$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_transcript(path: Path) -> dict[str, list[dict[str, str]]]:
    rows: dict[str, list[dict[str, str]]] = {stage: [] for stage in STAGES}
    stage: str | None = None
    current: dict[str, str] = {}

    for line in path.read_text(encoding="utf-8").splitlines():
        stage_match = STAGE_RE.match(line)
        if stage_match:
            stage = stage_match.group(1)
            if stage not in rows:
                raise ValueError(f"Unsupported stage in {path}: {stage}")
            continue

        field_match = FIELD_RE.match(line)
        if not field_match:
            continue
        if stage is None:
            raise ValueError(f"Found an example before a stage header in {path}")

        field = field_match.group(1).lower()
        current[field] = field_match.group(2)
        if field == "expected":
            missing = {"input", "predict", "expected"} - current.keys()
            if missing:
                raise ValueError(
                    f"Incomplete example in {path} {stage}: missing {sorted(missing)}"
                )
            rows[stage].append(current)
            current = {}

    if current:
        raise ValueError(f"Trailing incomplete example in {path}")
    return rows


def ratio(correct: int, total: int) -> dict[str, int | float]:
    return {
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total else 0.0,
        "accuracy_percent": 100.0 * correct / total if total else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--old-gold-dir",
        type=Path,
        required=True,
        help="Directory containing grammar_test_stageN.jsonl.pre-v10-bak files",
    )
    parser.add_argument(
        "--repaired-gold-dir",
        type=Path,
        required=True,
        help="Directory containing repaired grammar_test_stageN.jsonl files",
    )
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        metavar="NAME=TRANSCRIPT",
        help="Model label and saved evaluation transcript; repeat for each model",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON output path")
    args = parser.parse_args()

    model_paths: dict[str, Path] = {}
    for value in args.model:
        if "=" not in value:
            parser.error(f"--model must be NAME=TRANSCRIPT, received {value!r}")
        name, raw_path = value.split("=", 1)
        if not name or name in model_paths:
            parser.error(f"Model names must be non-empty and unique: {name!r}")
        model_paths[name] = Path(raw_path)

    old: dict[str, list[dict[str, Any]]] = {}
    repaired: dict[str, list[dict[str, Any]]] = {}
    source_files: dict[str, dict[str, str]] = {}
    for stage in STAGES:
        old_path = args.old_gold_dir / f"grammar_test_{stage}.jsonl.pre-v10-bak"
        repaired_path = args.repaired_gold_dir / f"grammar_test_{stage}.jsonl"
        old[stage] = load_jsonl(old_path)
        repaired[stage] = load_jsonl(repaired_path)
        if len(old[stage]) != len(repaired[stage]):
            raise ValueError(
                f"Gold row-count mismatch for {stage}: "
                f"{len(old[stage])} old vs {len(repaired[stage])} repaired"
            )
        source_files[f"old_{stage}"] = {
            "path": str(old_path),
            "sha256": sha256(old_path),
        }
        source_files[f"repaired_{stage}"] = {
            "path": str(repaired_path),
            "sha256": sha256(repaired_path),
        }

    flat_gold: list[dict[str, Any]] = []
    for stage in STAGES:
        for index, (old_row, repaired_row) in enumerate(
            zip(old[stage], repaired[stage], strict=True)
        ):
            flat_gold.append(
                {
                    "stage": stage,
                    "index": index,
                    "old_input": old_row["input"].strip(),
                    "repaired_input": repaired_row["input"].strip(),
                    "old_gold": old_row["output"].strip(),
                    "repaired_gold": repaired_row["output"].strip(),
                }
            )

    gold_changed = [r for r in flat_gold if r["old_gold"] != r["repaired_gold"]]
    gold_unchanged = [r for r in flat_gold if r["old_gold"] == r["repaired_gold"]]
    input_changed = [
        r for r in flat_gold if r["old_input"] != r["repaired_input"]
    ]

    result: dict[str, Any] = {
        "method": {
            "prediction_source": "saved evaluation transcripts; no inference run",
            "alignment": "stage plus zero-based row index",
            "exact_match": "prediction.strip() == gold.strip()",
        },
        "benchmark": {
            "total": len(flat_gold),
            "gold_unchanged": len(gold_unchanged),
            "gold_repaired": len(gold_changed),
            "inputs_unchanged": len(flat_gold) - len(input_changed),
            "inputs_repaired": len(input_changed),
            "gold_repaired_with_same_input": sum(
                r["old_input"] == r["repaired_input"] for r in gold_changed
            ),
            "gold_repaired_with_repaired_input": sum(
                r["old_input"] != r["repaired_input"] for r in gold_changed
            ),
        },
        "sources": source_files,
        "models": {},
        "pairwise_comparisons": {},
    }

    model_records: dict[str, list[dict[str, Any]]] = {}
    for model, transcript_path in model_paths.items():
        transcript = parse_transcript(transcript_path)
        predictions: list[dict[str, Any]] = []
        for stage in STAGES:
            expected_count = len(old[stage])
            if len(transcript[stage]) != expected_count:
                raise ValueError(
                    f"{model} {stage}: transcript has {len(transcript[stage])} "
                    f"examples; gold has {expected_count}"
                )
            for gold_row, saved_row in zip(
                (r for r in flat_gold if r["stage"] == stage),
                transcript[stage],
                strict=True,
            ):
                predictions.append(
                    {
                        **gold_row,
                        "prediction": saved_row["predict"].strip(),
                        "report_input": saved_row["input"].strip(),
                        "report_expected": saved_row["expected"].strip(),
                    }
                )
        model_records[model] = predictions

        def score(rows: list[dict[str, Any]], key: str) -> dict[str, int | float]:
            return ratio(sum(r["prediction"] == r[key] for r in rows), len(rows))

        changed_same_input = [
            r
            for r in predictions
            if r["old_gold"] != r["repaired_gold"]
            and r["old_input"] == r["repaired_input"]
        ]
        changed_repaired_input = [
            r
            for r in predictions
            if r["old_gold"] != r["repaired_gold"]
            and r["old_input"] != r["repaired_input"]
        ]
        unchanged = [
            r for r in predictions if r["old_gold"] == r["repaired_gold"]
        ]
        changed = [
            r for r in predictions if r["old_gold"] != r["repaired_gold"]
        ]

        result["models"][model] = {
            "transcript": {
                "path": str(transcript_path),
                "sha256": sha256(transcript_path),
                "examples": len(predictions),
                "report_input_matches_old": sum(
                    r["report_input"] == r["old_input"] for r in predictions
                ),
                "report_input_matches_repaired": sum(
                    r["report_input"] == r["repaired_input"] for r in predictions
                ),
                "report_expected_matches_old": sum(
                    r["report_expected"] == r["old_gold"] for r in predictions
                ),
                "report_expected_matches_repaired": sum(
                    r["report_expected"] == r["repaired_gold"] for r in predictions
                ),
            },
            "overall": {
                "old_gold": score(predictions, "old_gold"),
                "repaired_gold": score(predictions, "repaired_gold"),
            },
            "gold_unchanged_group": {
                "old_gold": score(unchanged, "old_gold"),
                "repaired_gold": score(unchanged, "repaired_gold"),
            },
            "gold_repaired_group": {
                "old_gold": score(changed, "old_gold"),
                "repaired_gold": score(changed, "repaired_gold"),
            },
            "gold_repaired_same_input_group": {
                "old_gold": score(changed_same_input, "old_gold"),
                "repaired_gold": score(changed_same_input, "repaired_gold"),
            },
            "gold_and_input_repaired_group": {
                "old_gold": score(changed_repaired_input, "old_gold"),
                "repaired_gold": score(changed_repaired_input, "repaired_gold"),
            },
        }

    for left_name, right_name in combinations(model_paths, 2):
        left = model_records[left_name]
        right = model_records[right_name]
        aligned: list[dict[str, Any]] = []
        for left_row, right_row in zip(left, right, strict=True):
            if (left_row["stage"], left_row["index"]) != (
                right_row["stage"],
                right_row["index"],
            ):
                raise ValueError(
                    f"Pairwise alignment failed for {left_name} and {right_name}"
                )
            aligned.append(
                {
                    **left_row,
                    "left_prediction": left_row["prediction"],
                    "right_prediction": right_row["prediction"],
                    "left_report_input": left_row["report_input"],
                    "right_report_input": right_row["report_input"],
                }
            )

        same_input = [
            r
            for r in aligned
            if r["left_report_input"] == r["right_report_input"]
        ]
        same_input_unchanged_gold = [
            r for r in same_input if r["old_gold"] == r["repaired_gold"]
        ]
        same_input_repaired_gold = [
            r for r in same_input if r["old_gold"] != r["repaired_gold"]
        ]

        def pair_score(
            rows: list[dict[str, Any]], prediction_key: str, gold_key: str
        ) -> dict[str, int | float]:
            return ratio(
                sum(r[prediction_key] == r[gold_key] for r in rows), len(rows)
            )

        def pair_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
            return {
                left_name: {
                    "old_gold": pair_score(rows, "left_prediction", "old_gold"),
                    "repaired_gold": pair_score(
                        rows, "left_prediction", "repaired_gold"
                    ),
                },
                right_name: {
                    "old_gold": pair_score(rows, "right_prediction", "old_gold"),
                    "repaired_gold": pair_score(
                        rows, "right_prediction", "repaired_gold"
                    ),
                },
            }

        result["pairwise_comparisons"][f"{left_name}_vs_{right_name}"] = {
            "same_transcript_input_all": pair_group(same_input),
            "same_transcript_input_gold_unchanged": pair_group(
                same_input_unchanged_gold
            ),
            "same_transcript_input_gold_repaired": pair_group(
                same_input_repaired_gold
            ),
        }

    output = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(output, encoding="utf-8")
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
