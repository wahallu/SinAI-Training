#!/usr/bin/env python3
"""Compute Stage 6 Sinhala grammar metrics from saved predictions.

This script performs no model inference. It aligns a prediction JSONL with a
gold JSONL by ID, validates the submitted input text, computes the same
continuous metrics used by ``test_grammar.py``, and writes a Markdown report.

Only the Python standard library is required.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        required=True,
        type=Path,
        help="Prediction JSONL containing id, input, and prediction fields",
    )
    parser.add_argument(
        "--gold",
        required=True,
        type=Path,
        help="Gold JSONL containing id, input, and output fields",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Markdown report path",
    )
    parser.add_argument(
        "--prediction-field",
        default="prediction",
        help="Prediction field to score (default: prediction)",
    )
    parser.add_argument(
        "--stage-name",
        default="stage6",
        help="Stage label displayed in the report (default: stage6)",
    )
    parser.add_argument(
        "--system-name",
        default="SinLLaMA v27 grammar adapter",
        help="System label displayed in the report",
    )
    return parser.parse_args()


def normalize(text: object) -> str:
    return unicodedata.normalize("NFC", str(text)).strip()


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}")
            rows.append(row)
    if not rows:
        raise ValueError(f"No JSONL records found in {path}")
    return rows


def index_by_id(rows: Iterable[dict], kind: str) -> dict[str, dict]:
    indexed: dict[str, dict] = {}
    for line_number, row in enumerate(rows, 1):
        if "id" not in row:
            raise ValueError(f"{kind} row {line_number} has no 'id' field")
        row_id = str(row["id"])
        if row_id in indexed:
            raise ValueError(f"Duplicate {kind} ID: {row_id}")
        indexed[row_id] = row
    return indexed


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sinhala_tokenize(text: str) -> list[str]:
    """Match test_grammar.py's Unicode combining-mark tokenization."""

    result: list[str] = []
    chars = list(text)
    index = 0
    while index < len(chars):
        cluster = chars[index]
        index += 1
        while index < len(chars) and unicodedata.combining(chars[index]):
            cluster += chars[index]
            index += 1
        if cluster.strip():
            result.append(cluster)
    return result


def token_prf(prediction: str, reference: str) -> tuple[float, float, float]:
    """Grapheme-token multiset precision, recall, and F1."""

    predicted = sinhala_tokenize(prediction)
    expected = sinhala_tokenize(reference)
    common = sum((Counter(predicted) & Counter(expected)).values())
    precision = common / len(predicted) if predicted else 0.0
    recall = common / len(expected) if expected else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return precision, recall, f1


def char_f1(prediction: str, reference: str) -> float:
    """Code-point multiset F1, matching test_grammar.py."""

    common = sum((Counter(prediction) & Counter(reference)).values())
    precision = common / len(prediction) if prediction else 0.0
    recall = common / len(reference) if reference else 0.0
    return (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )


def ngrams(tokens: Sequence[str], size: int) -> Counter[tuple[str, ...]]:
    return Counter(
        tuple(tokens[index : index + size])
        for index in range(len(tokens) - size + 1)
    )


def everygrams(tokens: Sequence[str], minimum: int = 1, maximum: int = 4) -> Counter:
    result: Counter[tuple[str, ...]] = Counter()
    for size in range(minimum, maximum + 1):
        result.update(ngrams(tokens, size))
    return result


def sentence_gleu(prediction: str, reference: str) -> float:
    """NLTK-compatible sentence GLEU over grapheme tokens (orders 1-4)."""

    predicted = everygrams(sinhala_tokenize(prediction))
    expected = everygrams(sinhala_tokenize(reference))
    denominator = max(sum(predicted.values()), sum(expected.values()))
    if not denominator:
        return 0.0
    return sum((predicted & expected).values()) / denominator


def lcs_length(left: Sequence[str], right: Sequence[str]) -> int:
    previous = [0] * (len(right) + 1)
    current = [0] * (len(right) + 1)
    for left_token in left:
        for right_index, right_token in enumerate(right, 1):
            if left_token == right_token:
                current[right_index] = previous[right_index - 1] + 1
            else:
                current[right_index] = max(
                    previous[right_index], current[right_index - 1]
                )
        previous, current = current, [0] * (len(right) + 1)
    return previous[-1]


def overlap_f1(overlap: int, predicted_count: int, reference_count: int) -> float:
    precision = overlap / predicted_count if predicted_count else 0.0
    recall = overlap / reference_count if reference_count else 0.0
    return (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )


def rouge_scores(prediction: str, reference: str) -> dict[str, float]:
    """ROUGE-1/2/L F1 over grapheme tokens, matching test_grammar.py."""

    predicted = sinhala_tokenize(prediction)
    expected = sinhala_tokenize(reference)
    if not predicted or not expected:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    predicted_unigrams = ngrams(predicted, 1)
    expected_unigrams = ngrams(expected, 1)
    unigram_overlap = sum((predicted_unigrams & expected_unigrams).values())

    predicted_bigrams = ngrams(predicted, 2)
    expected_bigrams = ngrams(expected, 2)
    bigram_overlap = sum((predicted_bigrams & expected_bigrams).values())

    return {
        "rouge1": overlap_f1(unigram_overlap, len(predicted), len(expected)),
        "rouge2": overlap_f1(
            bigram_overlap,
            max(len(predicted) - 1, 1),
            max(len(expected) - 1, 1),
        ),
        "rougeL": overlap_f1(
            lcs_length(predicted, expected), len(predicted), len(expected)
        ),
    }


def aligned_examples(
    gold_rows: list[dict], prediction_rows: list[dict], prediction_field: str
) -> list[dict]:
    gold = index_by_id(gold_rows, "gold")
    predictions = index_by_id(prediction_rows, "prediction")
    missing = sorted(set(gold) - set(predictions))
    extra = sorted(set(predictions) - set(gold))
    if missing or extra:
        raise ValueError(
            f"ID mismatch: {len(missing)} missing and {len(extra)} extra predictions; "
            f"missing examples={missing[:3]}, extra examples={extra[:3]}"
        )

    aligned: list[dict] = []
    for row_id, gold_row in gold.items():
        prediction_row = predictions[row_id]
        for field in ("input", "output"):
            if field not in gold_row:
                raise ValueError(f"Gold row {row_id} is missing field {field!r}")
        if prediction_field not in prediction_row:
            raise ValueError(
                f"Prediction row {row_id} is missing field {prediction_field!r}"
            )

        source = normalize(gold_row["input"])
        reference = normalize(gold_row["output"])
        prediction = normalize(prediction_row[prediction_field])
        submitted_input = normalize(prediction_row.get("input", source))
        if submitted_input != source:
            raise ValueError(f"Prediction input does not match gold input for {row_id}")

        aligned.append(
            {
                "id": row_id,
                "source": source,
                "reference": reference,
                "prediction": prediction,
                "category": str(gold_row.get("category", "unlabelled")),
            }
        )
    return aligned


def safe_ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def evaluate(examples: list[dict]) -> dict:
    per_example = []
    for row in examples:
        prediction = row["prediction"]
        reference = row["reference"]
        rouge = rouge_scores(prediction, reference)
        token_precision, token_recall, token_f1 = token_prf(prediction, reference)
        per_example.append(
            {
                **row,
                "needs_change": row["source"] != reference,
                "exact": prediction == reference,
                "rouge1": rouge["rouge1"],
                "rouge2": rouge["rouge2"],
                "rougeL": rouge["rougeL"],
                "gleu": sentence_gleu(prediction, reference),
                "char_f1": char_f1(prediction, reference),
                "token_precision": token_precision,
                "token_recall": token_recall,
                "token_f1": token_f1,
            }
        )

    changed = [row for row in per_example if row["needs_change"]]
    clean = [row for row in per_example if not row["needs_change"]]
    exact_count = sum(row["exact"] for row in per_example)
    changed_exact = sum(row["exact"] for row in changed)
    clean_exact = sum(row["exact"] for row in clean)
    overcorrected = sum(not row["exact"] for row in clean)

    continuous_keys = (
        "rouge1",
        "rouge2",
        "rougeL",
        "gleu",
        "char_f1",
        "token_precision",
        "token_recall",
        "token_f1",
    )
    metrics = {
        "N": len(per_example),
        "exact_count": exact_count,
        "change_N": len(changed),
        "change_exact_count": changed_exact,
        "clean_N": len(clean),
        "clean_exact_count": clean_exact,
        "overcorrection_count": overcorrected,
        "overall_accuracy": safe_ratio(exact_count, len(per_example)),
        "change_accuracy": safe_ratio(changed_exact, len(changed)),
        "nochange_accuracy": safe_ratio(clean_exact, len(clean)),
        "overcorrection_rate": safe_ratio(overcorrected, len(clean)),
    }
    metrics.update({key: mean(row[key] for row in per_example) for key in continuous_keys})

    category_rows: dict[str, list[dict]] = defaultdict(list)
    for row in per_example:
        category_rows[row["category"]].append(row)
    categories = {
        category: {
            "N": len(rows),
            "exact_count": sum(row["exact"] for row in rows),
            "exact_accuracy": safe_ratio(sum(row["exact"] for row in rows), len(rows)),
            "rougeL": mean(row["rougeL"] for row in rows),
            "char_f1": mean(row["char_f1"] for row in rows),
            "gleu": mean(row["gleu"] for row in rows),
            "token_f1": mean(row["token_f1"] for row in rows),
        }
        for category, rows in sorted(category_rows.items())
    }
    return {"metrics": metrics, "categories": categories}


def percent(value: float | None, decimals: int = 1) -> str:
    return "n/a" if value is None else f"{100 * value:.{decimals}f}%"


def warning_lines(metrics: dict) -> list[str]:
    warnings: list[str] = []
    if metrics["nochange_accuracy"] is not None and metrics["nochange_accuracy"] < 0.5:
        warnings.append(
            "No-change accuracy is low. The model changes too many already-correct inputs."
        )
    if metrics["change_accuracy"] is not None and metrics["change_accuracy"] < 0.5:
        warnings.append(
            "Change accuracy is low. Review failed corrections, dataset quality, "
            "and decoding settings."
        )
    if metrics["overcorrection_rate"] is not None and metrics["overcorrection_rate"] > 0.25:
        warnings.append(
            f"Over-correction is high ({percent(metrics['overcorrection_rate'])})."
        )
    if metrics["rougeL"] < 0.8:
        warnings.append("ROUGE-L is below 0.80.")
    if metrics["char_f1"] < 0.85:
        warnings.append("Char-F1 is below 0.85.")
    return warnings


def markdown_report(args: argparse.Namespace, evaluation: dict) -> str:
    metrics = evaluation["metrics"]
    warnings = warning_lines(metrics)
    command = (
        "python3 work/sinllama/scripts/score_grammar_stage6_full_metrics.py "
        f"--predictions {json.dumps(str(args.predictions))} "
        f"--gold {json.dumps(str(args.gold))} "
        f"--output {json.dumps(str(args.output))} "
        f"--stage-name {json.dumps(args.stage_name)} "
        f"--system-name {json.dumps(args.system_name)}"
    )

    lines = [
        f"# {args.system_name} — {args.stage_name} full metrics",
        "",
        "## Evaluation evidence",
        "",
        f"- Predictions: `{args.predictions.resolve()}`",
        f"- Predictions SHA-256: `{sha256(args.predictions)}`",
        f"- Gold: `{args.gold.resolve()}`",
        f"- Gold SHA-256: `{sha256(args.gold)}`",
        f"- Prediction field: `{args.prediction_field}`",
        f"- Evaluated samples: **{metrics['N']}**",
        "",
        f"## EXACT-MATCH RESULTS — {args.stage_name}",
        "",
        "```text",
        "Overall accuracy      : "
        f"{metrics['exact_count']}/{metrics['N']}  →  "
        f"{percent(metrics['overall_accuracy'])}",
        "Change-needed accuracy: "
        f"{metrics['change_exact_count']}/{metrics['change_N']}  →  "
        f"{percent(metrics['change_accuracy'])}",
        "No-change accuracy    : "
        f"{metrics['clean_exact_count']}/{metrics['clean_N']}  →  "
        f"{percent(metrics['nochange_accuracy'])}",
        "Over-correction rate  : "
        f"{metrics['overcorrection_count']}/{metrics['clean_N']}  →  "
        f"{percent(metrics['overcorrection_rate'])}  "
        "(changed correct sentences)",
        "```",
        "",
        f"## CONTINUOUS METRICS — {args.stage_name}",
        "",
        "Macro-average over all samples:",
        "",
        "```text",
        f"ROUGE-1   (grapheme): {metrics['rouge1']:.4f}",
        f"ROUGE-2   (grapheme): {metrics['rouge2']:.4f}",
        f"ROUGE-L   (grapheme): {metrics['rougeL']:.4f}",
        f"Sentence GLEU       : {metrics['gleu']:.4f}",
        f"Char-level F1       : {metrics['char_f1']:.4f}",
        f"Token Precision     : {metrics['token_precision']:.4f}",
        f"Token Recall        : {metrics['token_recall']:.4f}",
        f"Token F1            : {metrics['token_f1']:.4f}",
        "```",
        "",
        "## STAGE SUMMARY",
        "",
        f"| Metric | {args.stage_name} |",
        "|---|---:|",
        f"| Overall accuracy | {percent(metrics['overall_accuracy'])} |",
        f"| Change accuracy | {percent(metrics['change_accuracy'])} |",
        f"| No-change accuracy | {percent(metrics['nochange_accuracy'])} |",
        f"| Over-correction rate | {percent(metrics['overcorrection_rate'])} |",
        f"| ROUGE-L | {metrics['rougeL']:.4f} |",
        f"| Char-F1 | {metrics['char_f1']:.4f} |",
        f"| GLEU | {metrics['gleu']:.4f} |",
        f"| Token F1 | {metrics['token_f1']:.4f} |",
        "",
        "## CATEGORY BREAKDOWN",
        "",
        "| Category | N | Exact | Accuracy | ROUGE-L | Char-F1 | GLEU | Token F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for category, values in evaluation["categories"].items():
        lines.append(
            f"| {category} | {values['N']} | {values['exact_count']}/{values['N']} | "
            f"{percent(values['exact_accuracy'])} | {values['rougeL']:.4f} | "
            f"{values['char_f1']:.4f} | {values['gleu']:.4f} | "
            f"{values['token_f1']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## METRIC GUIDE FOR SINHALA GRAMMAR CORRECTION",
            "",
            "| Metric | Poor | OK | Good |",
            "|---|---:|---:|---:|",
            "| ROUGE-L | < 0.80 | 0.80–0.93 | > 0.93 |",
            "| Char-F1 | < 0.85 | 0.85–0.95 | > 0.95 |",
            "| GLEU | < 0.50 | 0.50–0.80 | > 0.80 |",
            "| Token F1 | < 0.80 | 0.80–0.93 | > 0.93 |",
            "| Over-correction | > 30% | 10–30% | < 10% |",
            "",
            "## WARNINGS",
            "",
        ]
    )
    if warnings:
        lines.extend(f"- ⚠️ {warning}" for warning in warnings)
    else:
        lines.append("- No threshold warnings were triggered.")

    lines.extend(
        [
            "",
            "## INTERPRETATION NOTES",
            "",
            "- Exact-match accuracy requires the full prediction to equal the "
            "supplied gold output.",
            "- Continuous metrics use the historical `test_grammar.py` definitions: "
            "ROUGE and GLEU use its Unicode combining-mark tokens; Char-F1 uses "
            "code-point multiset overlap; the reported token metrics use "
            "grapheme-token multiset overlap.",
            "- ROUGE, GLEU, Char-F1, and Token-F1 can remain high when only a small "
            "part of a long sentence is incorrect. They must not replace "
            "change-needed accuracy or over-correction reporting.",
            "- Scores measure agreement with the supplied Stage 6 automatic gold, "
            "not independent human adjudication of every valid Sinhala correction.",
            "",
            "## REPRODUCTION COMMAND",
            "",
            "```bash",
            command,
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    gold_rows = load_jsonl(args.gold)
    prediction_rows = load_jsonl(args.predictions)
    examples = aligned_examples(gold_rows, prediction_rows, args.prediction_field)
    evaluation = evaluate(examples)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown_report(args, evaluation), encoding="utf-8")

    metrics = evaluation["metrics"]
    print(f"Scored {metrics['N']} examples")
    print(
        f"Overall accuracy: {metrics['exact_count']}/{metrics['N']} "
        f"({percent(metrics['overall_accuracy'])})"
    )
    print(f"ROUGE-L: {metrics['rougeL']:.4f}")
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
