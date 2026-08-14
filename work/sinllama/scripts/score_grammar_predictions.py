#!/usr/bin/env python3
"""Score frozen Sinhala grammar predictions against private gold JSONL.

This program performs no model inference. Keep private Stage 6 gold on the
offline scoring machine and submit only prediction JSONL files from GPU runs.
"""

from __future__ import annotations

import argparse
import collections
import difflib
import hashlib
import json
import random
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


Edit = tuple[int, int, tuple[str, ...], tuple[str, ...]]
Pair = tuple[tuple[str, ...], tuple[str, ...]]
NUMBER_RE = re.compile(r"(?<!\w)[+-]?(?:\d[\d,.:/-]*\d|\d)(?!\w)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, help="JSONL with id and prediction")
    parser.add_argument(
        "--prediction-field",
        default="prediction",
        help="Prediction JSON field to score (default: prediction; use raw_prediction for raw mT5)",
    )
    parser.add_argument("--gold", required=True, help="Gold JSONL; may be Stage 6 private gold")
    parser.add_argument("--train-data", required=True, help="Training JSONL for unseen-pair analysis")
    parser.add_argument("--output-prefix", required=True, help="Path prefix for .json and .md")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def normalize(text: str) -> str:
    return unicodedata.normalize("NFC", text).strip()


def tokens(text: str) -> list[str]:
    return re.findall(r"\S+", normalize(text))


def edit_set(source: str, target: str) -> set[Edit]:
    old, new = tokens(source), tokens(target)
    edits: set[Edit] = set()
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(
        None, old, new, autojunk=False
    ).get_opcodes():
        if tag != "equal":
            edits.add((i1, i2, tuple(old[i1:i2]), tuple(new[j1:j2])))
    return edits


def pair_set(source: str, target: str) -> set[Pair]:
    return {(old, new) for _, _, old, new in edit_set(source, target)}


def detection_set(source: str, target: str) -> set[tuple[int, int, tuple[str, ...]]]:
    return {(i1, i2, old) for i1, i2, old, _ in edit_set(source, target)}


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSON at {path}:{line_number}: {exc}") from exc
    return rows


def indexed(rows: list[dict], kind: str) -> dict[str, dict]:
    result = {}
    for index, row in enumerate(rows, 1):
        row_id = str(row.get("id", f"row-{index:06d}"))
        if row_id in result:
            raise ValueError(f"Duplicate {kind} ID: {row_id}")
        result[row_id] = row
    return result


def training_pairs(path: Path) -> set[Pair]:
    pairs: set[Pair] = set()
    for row in load_jsonl(path):
        source, target = normalize(str(row["input"])), normalize(str(row["output"]))
        pairs.update(pair_set(source, target))
    return pairs


@dataclass
class ExampleResult:
    id: str
    document_id: str
    category: str
    source: str
    target: str
    prediction: str
    exact: bool
    changed: bool
    clean_preserved: bool
    undercorrected: bool
    wrong_corrected: bool
    gold_edits: set[Edit] = field(default_factory=set)
    predicted_edits: set[Edit] = field(default_factory=set)
    gold_detections: set = field(default_factory=set)
    predicted_detections: set = field(default_factory=set)
    unseen_edits: set[Edit] = field(default_factory=set)
    predicted_pairs: set[Pair] = field(default_factory=set)
    unseen_lemma_pairs: set[Pair] = field(default_factory=set)
    protected_mutation: bool = False
    number_mutation: bool = False
    contextual: bool = False


def protected_mutated(source: str, prediction: str, metadata: list[dict]) -> bool:
    for span in metadata:
        text = normalize(str(span.get("text", "")))
        if text and source.count(text) != prediction.count(text):
            return True
    return False


def numbers(text: str) -> collections.Counter[str]:
    return collections.Counter(NUMBER_RE.findall(normalize(text)))


def metadata_unseen_lemma_pairs(row: dict) -> set[Pair]:
    result = set()
    for edit in row.get("edits", []):
        if edit.get("seen_lemma_family") is False:
            source = tuple(tokens(str(edit.get("source", ""))))
            target = tuple(tokens(str(edit.get("target", ""))))
            if source or target:
                result.add((source, target))
    return result


def build_results(
    gold_rows: list[dict],
    prediction_rows: list[dict],
    taught: set[Pair],
    prediction_field: str = "prediction",
) -> list[ExampleResult]:
    gold = indexed(gold_rows, "gold")
    predictions = indexed(prediction_rows, "prediction")
    missing = sorted(set(gold) - set(predictions))
    extra = sorted(set(predictions) - set(gold))
    if missing or extra:
        raise ValueError(
            f"ID mismatch: {len(missing)} missing predictions and {len(extra)} unknown predictions. "
            f"Examples: missing={missing[:3]}, extra={extra[:3]}"
        )

    results = []
    for row_id, row in gold.items():
        pred_row = predictions[row_id]
        source = normalize(str(row.get("input", "")))
        target = normalize(str(row.get("output", "")))
        if prediction_field not in pred_row:
            raise ValueError(
                f"Prediction row {row_id} is missing field {prediction_field!r}"
            )
        prediction = normalize(str(pred_row[prediction_field]))
        if not source or not target:
            raise ValueError(f"Gold row {row_id} has an empty input/output")
        submitted_input = normalize(str(pred_row.get("input", source)))
        if submitted_input != source:
            raise ValueError(f"Prediction input does not match gold input for {row_id}")

        gold_edits = edit_set(source, target)
        predicted_edits = edit_set(source, prediction)
        predicted_pairs = {(old, new) for _, _, old, new in predicted_edits}
        changed = source != target
        exact = prediction == target
        category = str(row.get("category", "unlabelled"))
        results.append(
            ExampleResult(
                id=row_id,
                document_id=str(row.get("source_document_id", row_id)),
                category=category,
                source=source,
                target=target,
                prediction=prediction,
                exact=exact,
                changed=changed,
                clean_preserved=(not changed and prediction == source),
                undercorrected=(changed and prediction == source),
                wrong_corrected=(changed and prediction not in {source, target}),
                gold_edits=gold_edits,
                predicted_edits=predicted_edits,
                gold_detections=detection_set(source, target),
                predicted_detections=detection_set(source, prediction),
                unseen_edits={
                    edit for edit in gold_edits if (edit[2], edit[3]) not in taught
                },
                predicted_pairs=predicted_pairs,
                unseen_lemma_pairs=metadata_unseen_lemma_pairs(row),
                protected_mutation=protected_mutated(
                    source, prediction, list(row.get("protected_spans", []))
                ),
                number_mutation=numbers(source) != numbers(prediction),
                contextual=(
                    bool(row.get("context_required"))
                    or category == "contextual-real-word"
                ),
            )
        )
    return results


def ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def f_beta(precision: float, recall: float, beta: float) -> float:
    beta_sq = beta * beta
    denominator = beta_sq * precision + recall
    return (1 + beta_sq) * precision * recall / denominator if denominator else 0.0


def aggregate(results: Iterable[ExampleResult]) -> dict[str, float | int]:
    rows = list(results)
    changed = [row for row in rows if row.changed]
    clean = [row for row in rows if not row.changed]
    contextual = [row for row in rows if row.contextual]
    gold_edits = sum(len(row.gold_edits) for row in rows)
    predicted_edits = sum(len(row.predicted_edits) for row in rows)
    matched_edits = sum(len(row.gold_edits & row.predicted_edits) for row in rows)
    gold_detections = sum(len(row.gold_detections) for row in rows)
    predicted_detections = sum(len(row.predicted_detections) for row in rows)
    matched_detections = sum(
        len(row.gold_detections & row.predicted_detections) for row in rows
    )
    unseen = sum(len(row.unseen_edits) for row in rows)
    unseen_hit = sum(len(row.unseen_edits & row.predicted_edits) for row in rows)
    unseen_lemmas = sum(len(row.unseen_lemma_pairs) for row in rows)
    unseen_lemma_hit = sum(
        len(row.unseen_lemma_pairs & row.predicted_pairs) for row in rows
    )
    edit_precision = ratio(matched_edits, predicted_edits)
    edit_recall = ratio(matched_edits, gold_edits)
    detection_precision = ratio(matched_detections, predicted_detections)
    detection_recall = ratio(matched_detections, gold_detections)
    return {
        "N": len(rows),
        "change_N": len(changed),
        "clean_N": len(clean),
        "exact_match": ratio(sum(row.exact for row in rows), len(rows)),
        "change_needed_exact": ratio(sum(row.exact for row in changed), len(changed)),
        "clean_preservation": ratio(sum(row.clean_preserved for row in clean), len(clean)),
        "overcorrection_rate": ratio(sum(not row.clean_preserved for row in clean), len(clean)),
        "undercorrection_rate": ratio(sum(row.undercorrected for row in changed), len(changed)),
        "wrong_correction_rate": ratio(sum(row.wrong_corrected for row in changed), len(changed)),
        "edit_precision": edit_precision,
        "edit_recall": edit_recall,
        "edit_f0.5": f_beta(edit_precision, edit_recall, 0.5),
        "detection_precision": detection_precision,
        "detection_recall": detection_recall,
        "detection_f1": f_beta(detection_precision, detection_recall, 1.0),
        "unseen_pair_N": unseen,
        "unseen_pair_recall": ratio(unseen_hit, unseen),
        "unseen_lemma_N": unseen_lemmas,
        "unseen_lemma_recall": ratio(unseen_lemma_hit, unseen_lemmas),
        "contextual_N": len(contextual),
        "contextual_exact": ratio(sum(row.exact for row in contextual), len(contextual)),
        "protected_mutation_rate": ratio(sum(row.protected_mutation for row in rows), len(rows)),
        "number_mutation_rate": ratio(sum(row.number_mutation for row in rows), len(rows)),
    }


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    weight = position - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def clustered_bootstrap(
    results: list[ExampleResult], samples: int, seed: int
) -> dict[str, dict[str, float]]:
    if samples <= 0:
        return {}
    by_document: dict[str, list[ExampleResult]] = collections.defaultdict(list)
    for row in results:
        by_document[row.document_id].append(row)
    document_ids = sorted(by_document)
    rng = random.Random(seed)
    keys = [
        "exact_match",
        "change_needed_exact",
        "clean_preservation",
        "overcorrection_rate",
        "edit_precision",
        "edit_recall",
        "edit_f0.5",
        "unseen_pair_recall",
        "contextual_exact",
        "protected_mutation_rate",
        "number_mutation_rate",
    ]
    distributions: dict[str, list[float]] = {key: [] for key in keys}
    for _ in range(samples):
        chosen = [rng.choice(document_ids) for _ in document_ids]
        sampled = [row for doc_id in chosen for row in by_document[doc_id]]
        metrics = aggregate(sampled)
        for key in keys:
            distributions[key].append(float(metrics[key]))
    return {
        key: {
            "low": round(percentile(values, 0.025), 6),
            "high": round(percentile(values, 0.975), 6),
        }
        for key, values in distributions.items()
    }


def per_category(results: list[ExampleResult]) -> dict[str, dict]:
    groups: dict[str, list[ExampleResult]] = collections.defaultdict(list)
    for row in results:
        groups[row.category].append(row)
    return {name: aggregate(rows) for name, rows in sorted(groups.items())}


def pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def markdown_report(report: dict) -> str:
    metrics = report["metrics"]
    ci = report["confidence_intervals_95"]

    def display(key: str) -> str:
        value = pct(float(metrics[key]))
        if key in ci:
            value += f" ({pct(ci[key]['low'])}–{pct(ci[key]['high'])})"
        return value

    lines = [
        "# Sinhala grammar prediction evaluation",
        "",
        f"- predictions: `{report['predictions_path']}`",
        f"- predictions SHA-256: `{report['predictions_sha256']}`",
        f"- prediction field: `{report['prediction_field']}`",
        f"- gold: `{report['gold_path']}`",
        f"- gold SHA-256: `{report['gold_sha256']}`",
        f"- examples: **{metrics['N']}** ({metrics['change_N']} change / {metrics['clean_N']} clean)",
        f"- bootstrap: {report['bootstrap_samples']} document-clustered samples",
        "",
        "## Primary metrics",
        "",
        "| Metric | Result |",
        "|---|---:|",
        f"| Edit precision | {display('edit_precision')} |",
        f"| Edit recall | {display('edit_recall')} |",
        f"| Edit F0.5 | {display('edit_f0.5')} |",
        f"| Detection precision | {display('detection_precision')} |",
        f"| Detection recall | {display('detection_recall')} |",
        f"| Detection F1 | {display('detection_f1')} |",
        f"| Unseen-pair recall (N={metrics['unseen_pair_N']}) | {display('unseen_pair_recall')} |",
        f"| Unseen-lemma recall (N={metrics['unseen_lemma_N']}) | {display('unseen_lemma_recall')} |",
        f"| Contextual exact (N={metrics['contextual_N']}) | {display('contextual_exact')} |",
        "",
        "## Restraint and exactness",
        "",
        "| Metric | Result |",
        "|---|---:|",
        f"| Overall exact match | {display('exact_match')} |",
        f"| Change-needed exact | {display('change_needed_exact')} |",
        f"| Clean preservation | {display('clean_preservation')} |",
        f"| Over-correction | {display('overcorrection_rate')} |",
        f"| Under-correction | {display('undercorrection_rate')} |",
        f"| Wrong correction | {display('wrong_correction_rate')} |",
        f"| Protected-span mutation | {display('protected_mutation_rate')} |",
        f"| Number mutation | {display('number_mutation_rate')} |",
        "",
        "## Per category",
        "",
        "| Category | N | Exact | Edit P | Edit R | F0.5 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for category, values in report["per_category"].items():
        lines.append(
            f"| {category} | {values['N']} | {pct(values['exact_match'])} | "
            f"{pct(values['edit_precision'])} | {pct(values['edit_recall'])} | "
            f"{pct(values['edit_f0.5'])} |"
        )
    lines.extend(
        [
            "",
            "Confidence intervals are document-clustered. If the gold has no "
            "`source_document_id`, each example is treated as its own cluster.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    prediction_path = Path(args.predictions)
    gold_path = Path(args.gold)
    train_path = Path(args.train_data)
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    gold_rows = load_jsonl(gold_path)
    prediction_rows = load_jsonl(prediction_path)
    taught = training_pairs(train_path)
    results = build_results(
        gold_rows, prediction_rows, taught, prediction_field=args.prediction_field
    )
    metrics = aggregate(results)
    report = {
        "predictions_path": str(prediction_path.resolve()),
        "predictions_sha256": hashlib.sha256(prediction_path.read_bytes()).hexdigest(),
        "prediction_field": args.prediction_field,
        "gold_path": str(gold_path.resolve()),
        "gold_sha256": hashlib.sha256(gold_path.read_bytes()).hexdigest(),
        "train_path": str(train_path.resolve()),
        "train_sha256": hashlib.sha256(train_path.read_bytes()).hexdigest(),
        "bootstrap_samples": args.bootstrap_samples,
        "metrics": metrics,
        "confidence_intervals_95": clustered_bootstrap(
            results, args.bootstrap_samples, args.seed
        ),
        "per_category": per_category(results),
        "failures": [
            {
                "id": row.id,
                "document_id": row.document_id,
                "category": row.category,
                "input": row.source,
                "prediction": row.prediction,
                "expected": row.target,
            }
            for row in results
            if not row.exact
        ],
    }
    json_path = Path(str(output_prefix) + ".json")
    md_path = Path(str(output_prefix) + ".md")
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    md_path.write_text(markdown_report(report), encoding="utf-8")
    print(f"Scored {metrics['N']} examples")
    print(f"Edit F0.5: {metrics['edit_f0.5']:.4f}")
    print(f"Unseen-pair recall: {metrics['unseen_pair_recall']:.4f} (N={metrics['unseen_pair_N']})")
    print(f"Report: {md_path}")
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()
