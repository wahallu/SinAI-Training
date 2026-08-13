#!/usr/bin/env python3
"""Create, validate, apply, and score compact grammar edit scripts.

The representation is deliberately strict and reversible:

    KEEP
    [{"s":12,"e":18,"o":"කෙරුනි","n":"කෙරුණි"}]

Offsets are Python Unicode character offsets into the NFC-normalized input.
Malformed or unsafe scripts are rejected to KEEP by callers.
"""

from __future__ import annotations

import collections
import difflib
import json
import re
import unicodedata
from dataclasses import dataclass


KEEP = "KEEP"
NUMBER_RE = re.compile(r"\d[\d,.:/-]*")
LATIN_RE = re.compile(r"[A-Za-z]+(?:['’-][A-Za-z]+)*")
URL_RE = re.compile(r"(?:https?://|www\.)\S+|\b\S+@\S+\.\S+\b", re.IGNORECASE)
QUOTED_RE = re.compile(r"[\"'“”‘’«»][^\"'“”‘’«»]+[\"'“”‘’«»]")


def normalize(text: str) -> str:
    return unicodedata.normalize("NFC", text).strip()


def make_edit_script(source: str, target: str) -> str:
    """Return a deterministic character-level edit script for source -> target."""
    source, target = normalize(source), normalize(target)
    if source == target:
        return KEEP
    operations = []
    matcher = difflib.SequenceMatcher(None, source, target, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "equal":
            operations.append({"s": i1, "e": i2, "o": source[i1:i2], "n": target[j1:j2]})
    return json.dumps(operations, ensure_ascii=False, separators=(",", ":"))


def parse_edit_script(source: str, script: str, max_operations: int = 32) -> list[dict]:
    """Parse and structurally validate a script; raise ValueError on any ambiguity."""
    source, script = normalize(source), script.strip()
    if script == KEEP:
        return []
    try:
        operations = json.loads(script)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid_json:{exc.msg}") from exc
    if not isinstance(operations, list) or not operations:
        raise ValueError("script_must_be_KEEP_or_nonempty_list")
    if len(operations) > max_operations:
        raise ValueError("too_many_operations")

    validated = []
    previous_end = -1
    previous_start = -1
    for operation in operations:
        if not isinstance(operation, dict) or set(operation) != {"s", "e", "o", "n"}:
            raise ValueError("operation_schema")
        start, end = operation["s"], operation["e"]
        old, new = operation["o"], operation["n"]
        if isinstance(start, bool) or isinstance(end, bool) or not isinstance(start, int) or not isinstance(end, int):
            raise ValueError("offset_type")
        if not isinstance(old, str) or not isinstance(new, str):
            raise ValueError("text_type")
        if start < 0 or end < start or end > len(source):
            raise ValueError("offset_range")
        if start < previous_end or start == previous_start:
            raise ValueError("operations_overlap_or_share_start")
        if source[start:end] != old:
            raise ValueError("old_text_mismatch")
        if old == new:
            raise ValueError("no_op")
        validated.append({"s": start, "e": end, "o": old, "n": new})
        previous_start, previous_end = start, end
    return validated


def _counter(pattern: re.Pattern[str], text: str) -> collections.Counter[str]:
    return collections.Counter(pattern.findall(text))


def _format_controls(text: str) -> collections.Counter[str]:
    return collections.Counter(char for char in text if unicodedata.category(char) == "Cf")


def _has_repetition_loop(text: str, width: int = 4, repeats: int = 3) -> bool:
    words = text.split()
    if len(words) < width * repeats:
        return False
    counts = collections.Counter(tuple(words[i : i + width]) for i in range(len(words) - width + 1))
    return any(count >= repeats for count in counts.values())


@dataclass(frozen=True)
class ApplyResult:
    text: str
    status: str
    reasons: tuple[str, ...]
    operation_count: int


def apply_edit_script(
    source: str,
    script: str,
    *,
    max_operations: int = 32,
    min_byte_length_ratio: float = 0.75,
    max_byte_length_ratio: float = 1.25,
    max_changed_fraction: float = 0.50,
    generation_finished: bool = True,
) -> ApplyResult:
    """Apply a script or safely return the source when validation fails."""
    source = normalize(source)
    try:
        operations = parse_edit_script(source, script, max_operations=max_operations)
    except ValueError as exc:
        return ApplyResult(source, "INVALID", (str(exc),), 0)
    if not operations:
        return ApplyResult(source, "KEEP", (), 0)

    candidate = source
    for operation in reversed(operations):
        candidate = candidate[: operation["s"]] + operation["n"] + candidate[operation["e"] :]
    candidate = normalize(candidate)

    reasons = []
    if not generation_finished:
        reasons.append("generation_cap_or_missing_eos")
    source_bytes = max(1, len(source.encode("utf-8")))
    length_ratio = len(candidate.encode("utf-8")) / source_bytes
    if not min_byte_length_ratio <= length_ratio <= max_byte_length_ratio:
        reasons.append("byte_length_ratio")
    similarity = difflib.SequenceMatcher(None, source, candidate, autojunk=False).ratio()
    if 1.0 - similarity > max_changed_fraction:
        reasons.append("edit_coverage")
    if _has_repetition_loop(candidate):
        reasons.append("repetition_loop")
    if _counter(NUMBER_RE, source) != _counter(NUMBER_RE, candidate):
        reasons.append("number_mutation")
    if _counter(LATIN_RE, source) != _counter(LATIN_RE, candidate):
        reasons.append("latin_span_mutation")
    if _counter(URL_RE, source) != _counter(URL_RE, candidate):
        reasons.append("url_or_email_mutation")
    if _counter(QUOTED_RE, source) != _counter(QUOTED_RE, candidate):
        reasons.append("quoted_span_mutation")
    if _format_controls(source) != _format_controls(candidate):
        reasons.append("unicode_format_control_mutation")

    if reasons:
        return ApplyResult(source, "REJECTED", tuple(sorted(set(reasons))), len(operations))
    return ApplyResult(candidate, "APPLIED", (), len(operations))


def _token_edits(source: str, target: str) -> set[tuple]:
    old, new = source.split(), target.split()
    edits = set()
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, old, new, autojunk=False).get_opcodes():
        if tag != "equal":
            edits.add((i1, i2, tuple(old[i1:i2]), tuple(new[j1:j2])))
    return edits


def correction_metrics(sources: list[str], targets: list[str], predictions: list[str]) -> dict[str, float]:
    """Compute generated development metrics using the offline scorer's edit definition."""
    if not (len(sources) == len(targets) == len(predictions)):
        raise ValueError("metric inputs have different lengths")
    changed_indexes = [i for i, (source, target) in enumerate(zip(sources, targets)) if source != target]
    clean_indexes = [i for i in range(len(sources)) if i not in set(changed_indexes)]
    gold_edits = [_token_edits(source, target) for source, target in zip(sources, targets)]
    predicted_edits = [_token_edits(source, prediction) for source, prediction in zip(sources, predictions)]
    gold_count = sum(map(len, gold_edits))
    predicted_count = sum(map(len, predicted_edits))
    matched = sum(len(gold & predicted) for gold, predicted in zip(gold_edits, predicted_edits))
    precision = matched / predicted_count if predicted_count else 0.0
    recall = matched / gold_count if gold_count else 0.0
    denominator = 0.25 * precision + recall
    f05 = 1.25 * precision * recall / denominator if denominator else 0.0
    exact = sum(prediction == target for prediction, target in zip(predictions, targets))
    change_exact = sum(predictions[i] == targets[i] for i in changed_indexes)
    clean_kept = sum(predictions[i] == sources[i] for i in clean_indexes)
    return {
        "edit_precision": precision,
        "edit_recall": recall,
        "edit_f0_5": f05,
        "exact_match": exact / len(sources) if sources else 0.0,
        "change_exact": change_exact / len(changed_indexes) if changed_indexes else 0.0,
        "clean_preservation": clean_kept / len(clean_indexes) if clean_indexes else 0.0,
    }
