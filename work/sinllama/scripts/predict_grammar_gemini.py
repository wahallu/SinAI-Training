#!/usr/bin/env python3
"""Run the frozen Stage 6 Sinhala grammar inputs through Gemini on a CPU.

The script uses Google's HTTPS Interactions API directly, so it needs no GPU
and no third-party Python packages. It appends and fsyncs one JSONL prediction
at a time, then safely resumes from a validated prefix if a run is interrupted.

Keep the API key out of source files and shell history:

    read -s GEMINI_API_KEY
    export GEMINI_API_KEY
    python3 scripts/predict_grammar_gemini.py \
      --input-data data/grammar_stage6_inputs.jsonl \
      --output Tested_results/Stage6/gemini_3_7_flash_stage6_predictions.jsonl
    unset GEMINI_API_KEY

Use --dry-run first to validate the dataset and planned request without making
API calls. API calls may incur charges under the Google project owning the key.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
import unicodedata
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


INSTRUCTION = (
    "Correct the grammar of the Sinhala sentence. "
    "ONLY fix errors. "
    "If the sentence is already correct, return it EXACTLY unchanged — "
    "do not rephrase, reorder, or change tense."
)
FROZEN_STAGE6_INPUT_SHA256 = (
    "e6dfdf31a8d75264a313450d9ce727949cf4a82916e411c972b998060066a66e"
)
DEFAULT_API_URL = "https://generativelanguage.googleapis.com/v1beta/interactions"
RETRYABLE_HTTP_CODES = {408, 429, 500, 502, 503, 504}
INSTRUCTION_SHA256 = hashlib.sha256(INSTRUCTION.encode("utf-8")).hexdigest()


class GeminiAPIError(RuntimeError):
    """A Gemini API failure with an optional HTTP status and retry delay."""

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.retry_after = retry_after


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-data", required=True, help="Frozen input JSONL")
    parser.add_argument("--output", required=True, help="Prediction JSONL to create/resume")
    parser.add_argument("--model", default="gemini-3.7-flash")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help=argparse.SUPPRESS)
    parser.add_argument("--api-key-env", default="GEMINI_API_KEY")
    parser.add_argument("--max-output-tokens", type=int, default=512)
    parser.add_argument("--thinking-level", choices=("low", "medium", "high"), default="low")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--requests-per-minute",
        type=float,
        default=10.0,
        help="Client-side rate cap; set it at or below the key's actual quota",
    )
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="Smoke-test the first N rows")
    parser.add_argument(
        "--expected-input-sha256",
        default=FROZEN_STAGE6_INPUT_SHA256,
        help="Fail if the complete input file does not have this SHA-256",
    )
    parser.add_argument(
        "--skip-input-hash-check",
        action="store_true",
        help="Allow a non-frozen input file (not suitable for the final comparison)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate only; make no API calls")
    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.max_output_tokens < 1:
        parser.error("--max-output-tokens must be at least 1")
    if args.requests_per_minute <= 0:
        parser.error("--requests-per-minute must be greater than 0")
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be greater than 0")
    if args.max_retries < 0:
        parser.error("--max-retries cannot be negative")
    return args


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def normalize(text: object) -> str:
    return unicodedata.normalize("NFC", str(text)).strip()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_inputs(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            source = normalize(row.get("input", ""))
            if not source:
                raise ValueError(f"Missing input at {path}:{line_number}")
            rows.append(
                {
                    "id": str(row.get("id", f"row-{len(rows) + 1:06d}")),
                    "input": source,
                }
            )
    if not rows:
        raise ValueError(f"No input rows found in {path}")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Input IDs are not unique")
    return rows


def load_resume_prefix(
    output_path: Path, expected_rows: list[dict[str, str]]
) -> list[dict[str, Any]]:
    if not output_path.exists():
        return []

    completed: list[dict[str, Any]] = []
    with output_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Cannot resume: invalid JSON at {output_path}:{line_number}. "
                    "Repair or move the output file before retrying."
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(f"Cannot resume: row {line_number} is not an object")
            completed.append(row)

    if len(completed) > len(expected_rows):
        raise ValueError("Cannot resume: output contains more rows than this run requests")

    seen_ids: set[str] = set()
    for index, existing in enumerate(completed):
        expected = expected_rows[index]
        if existing.get("id") != expected["id"]:
            raise ValueError(
                f"Cannot resume: output row {index + 1} has ID {existing.get('id')!r}; "
                f"expected {expected['id']!r}"
            )
        if normalize(existing.get("input", "")) != expected["input"]:
            raise ValueError(
                f"Cannot resume: input mismatch for {expected['id']} at output row {index + 1}"
            )
        if not normalize(existing.get("prediction", "")):
            raise ValueError(f"Cannot resume: missing prediction for {expected['id']}")
        if expected["id"] in seen_ids:
            raise ValueError(f"Cannot resume: duplicate output ID {expected['id']}")
        seen_ids.add(expected["id"])
    return completed


def validate_resume_config(
    completed: list[dict[str, Any]], args: argparse.Namespace
) -> None:
    expected_generation_config = {
        "max_output_tokens": args.max_output_tokens,
        "seed": args.seed,
        "thinking_level": args.thinking_level,
    }
    for row in completed:
        row_id = row.get("id", "unknown row")
        if row.get("model_requested") != args.model:
            raise ValueError(
                f"Cannot resume: {row_id} used model {row.get('model_requested')!r}, "
                f"not {args.model!r}"
            )
        if row.get("instruction_sha256") != INSTRUCTION_SHA256:
            raise ValueError(f"Cannot resume: {row_id} used a different instruction")
        if row.get("generation_config") != expected_generation_config:
            raise ValueError(f"Cannot resume: {row_id} used a different generation config")


def request_payload(args: argparse.Namespace, source: str) -> dict[str, Any]:
    return {
        "model": args.model,
        "input": source,
        "system_instruction": INSTRUCTION,
        "store": False,
        "generation_config": {
            "max_output_tokens": args.max_output_tokens,
            "seed": args.seed,
            "thinking_level": args.thinking_level,
        },
    }


def parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None


def safe_error_detail(raw_body: bytes) -> str:
    """Return a short API error without including request headers or secrets."""
    try:
        body = json.loads(raw_body.decode("utf-8", errors="replace"))
        detail = body.get("error", {}).get("message", "")
        if detail:
            return " ".join(str(detail).split())[:500]
    except (json.JSONDecodeError, AttributeError):
        pass
    return "Gemini returned an error without a readable message"


def call_gemini(
    *,
    api_url: str,
    api_key: str,
    payload: dict[str, Any],
    timeout_seconds: float,
) -> dict[str, Any]:
    request = urllib.request.Request(
        api_url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        raw_error = exc.read()
        raise GeminiAPIError(
            f"Gemini HTTP {exc.code}: {safe_error_detail(raw_error)}",
            status=exc.code,
            retry_after=parse_retry_after(exc.headers.get("Retry-After")),
        ) from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise GeminiAPIError(f"Gemini network error: {exc}") from exc

    try:
        result = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise GeminiAPIError("Gemini returned a non-JSON response") from exc
    if not isinstance(result, dict):
        raise GeminiAPIError("Gemini returned an unexpected JSON response")
    return result


def extract_prediction(response: dict[str, Any]) -> tuple[str, str]:
    if response.get("status") != "completed":
        raise GeminiAPIError(
            f"Gemini interaction did not complete (status={response.get('status')!r})"
        )

    model_outputs = [
        step
        for step in response.get("steps", [])
        if isinstance(step, dict) and step.get("type") == "model_output"
    ]
    if not model_outputs:
        raise GeminiAPIError("Gemini response contained no model_output step")

    text_parts = [
        part.get("text", "")
        for part in model_outputs[-1].get("content", [])
        if isinstance(part, dict) and part.get("type") == "text"
    ]
    raw_prediction = "".join(str(part) for part in text_parts).strip()
    if not raw_prediction:
        raise GeminiAPIError("Gemini returned an empty text response")

    # Match the existing SinLLaMA exporter: score the first non-empty response
    # line while retaining the complete raw response for auditability.
    prediction = normalize(raw_prediction.splitlines()[0])
    if not prediction:
        raise GeminiAPIError("Gemini's first response line was empty")
    return prediction, raw_prediction


def request_with_retries(
    *,
    args: argparse.Namespace,
    api_key: str,
    source: str,
    row_id: str,
) -> tuple[dict[str, Any], int]:
    payload = request_payload(args, source)
    for attempt in range(args.max_retries + 1):
        try:
            return (
                call_gemini(
                    api_url=args.api_url,
                    api_key=api_key,
                    payload=payload,
                    timeout_seconds=args.timeout_seconds,
                ),
                attempt,
            )
        except GeminiAPIError as exc:
            retryable = exc.status is None or exc.status in RETRYABLE_HTTP_CODES
            if not retryable or attempt >= args.max_retries:
                raise GeminiAPIError(f"{row_id}: {exc}", status=exc.status) from exc
            exponential = min(300.0, 2.0 * (2**attempt))
            delay = exc.retry_after if exc.retry_after is not None else exponential
            delay += random.uniform(0.0, min(1.0, delay * 0.1))
            print(
                f"{row_id}: transient API failure; retry {attempt + 1}/"
                f"{args.max_retries} in {delay:.1f}s ({exc})",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)
    raise AssertionError("retry loop exited unexpectedly")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_manifest(
    *,
    args: argparse.Namespace,
    input_path: Path,
    input_sha256: str,
    output_path: Path,
    rows: list[dict[str, Any]],
    requested_rows: int,
    resumed_rows: int,
    started_at: str,
    session_seconds: float,
) -> Path:
    usage_totals = {
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "total_thought_tokens": 0,
        "total_tokens": 0,
    }
    for row in rows:
        usage = row.get("usage", {})
        for field in usage_totals:
            value = usage.get(field, 0) if isinstance(usage, dict) else 0
            if isinstance(value, int):
                usage_totals[field] += value

    manifest = {
        "system": "Google Gemini grammar baseline",
        "api": "Gemini Interactions API v1beta",
        "model_requested": args.model,
        "models_returned": sorted(
            {str(row["model_returned"]) for row in rows if row.get("model_returned")}
        ),
        "input_path": str(input_path.resolve()),
        "input_sha256": input_sha256,
        "frozen_stage6_input_sha256": FROZEN_STAGE6_INPUT_SHA256,
        "input_hash_check_skipped": args.skip_input_hash_check,
        "output_path": str(output_path.resolve()),
        "output_sha256": file_sha256(output_path),
        "rows": len(rows),
        "requested_rows": requested_rows,
        "complete": len(rows) == requested_rows,
        "resumed_rows": resumed_rows,
        "prompt_instruction": INSTRUCTION,
        "instruction_sha256": INSTRUCTION_SHA256,
        "generation_config": {
            "max_output_tokens": args.max_output_tokens,
            "seed": args.seed,
            "thinking_level": args.thinking_level,
            "store": False,
        },
        "client": {
            "implementation": "Python standard library HTTPS",
            "requests_per_minute": args.requests_per_minute,
            "timeout_seconds": args.timeout_seconds,
            "max_retries": args.max_retries,
        },
        "usage": usage_totals,
        "started_at_utc": started_at,
        "completed_at_utc": utc_now(),
        "session_seconds": round(session_seconds, 3),
        "sum_row_latency_ms": round(
            sum(float(row.get("latency_ms", 0.0)) for row in rows), 3
        ),
    }
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    temporary_path = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_path, manifest_path)
    return manifest_path


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_data)
    output_path = Path(args.output)

    if not input_path.is_file():
        raise SystemExit(f"Input file not found: {input_path}")
    input_sha256 = file_sha256(input_path)
    if not args.skip_input_hash_check and input_sha256 != args.expected_input_sha256:
        raise SystemExit(
            "Frozen input hash mismatch:\n"
            f"  expected: {args.expected_input_sha256}\n"
            f"  actual:   {input_sha256}\n"
            "Refusing to run a non-comparable Stage 6 evaluation."
        )

    all_rows = load_inputs(input_path)
    rows = all_rows[: args.limit] if args.limit is not None else all_rows
    completed = load_resume_prefix(output_path, rows)
    validate_resume_config(completed, args)
    resumed_rows = len(completed)

    print(f"Model:          {args.model}")
    print(f"Input:          {input_path.resolve()}")
    print(f"Input SHA-256:  {input_sha256}")
    print(f"Rows requested: {len(rows)}")
    print(f"Already saved:  {len(completed)}")
    print(f"Output:         {output_path.resolve()}")
    print(f"Rate cap:       {args.requests_per_minute:g} requests/minute")

    if args.dry_run:
        print("Dry run passed; no API calls or file changes were made.")
        return

    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        raise SystemExit(
            f"Missing API key. Set it in the {args.api_key_env} environment variable."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    min_interval = 60.0 / args.requests_per_minute
    started_at = utc_now()
    session_started = time.perf_counter()
    last_request_started: float | None = None

    try:
        for index in range(len(completed), len(rows)):
            row = rows[index]
            if last_request_started is not None:
                wait = min_interval - (time.perf_counter() - last_request_started)
                if wait > 0:
                    time.sleep(wait)

            request_started = time.perf_counter()
            last_request_started = request_started
            try:
                response, retries = request_with_retries(
                    args=args,
                    api_key=api_key,
                    source=row["input"],
                    row_id=row["id"],
                )
                prediction, raw_prediction = extract_prediction(response)
            except GeminiAPIError as exc:
                raise SystemExit(
                    f"Stopped at {row['id']} after saving {len(completed)}/"
                    f"{len(rows)} rows: {exc}\nRun the same command to resume."
                ) from exc
            result = {
                "id": row["id"],
                "input": row["input"],
                "prediction": prediction,
                "raw_prediction": raw_prediction,
                "model_requested": args.model,
                "model_returned": response.get("model"),
                "instruction_sha256": INSTRUCTION_SHA256,
                "generation_config": {
                    "max_output_tokens": args.max_output_tokens,
                    "seed": args.seed,
                    "thinking_level": args.thinking_level,
                },
                "interaction_id": response.get("id"),
                "usage": response.get("usage", {}),
                "latency_ms": round((time.perf_counter() - request_started) * 1000, 3),
                "retries": retries,
            }
            append_jsonl(output_path, result)
            completed.append(result)
            print(f"Predicted {index + 1}/{len(rows)} ({row['id']})", flush=True)
    except KeyboardInterrupt:
        print(
            f"\nInterrupted after {len(completed)}/{len(rows)} rows. "
            "Run the same command to resume.",
            file=sys.stderr,
        )
        raise SystemExit(130)

    manifest_path = write_manifest(
        args=args,
        input_path=input_path,
        input_sha256=input_sha256,
        output_path=output_path,
        rows=completed,
        requested_rows=len(rows),
        resumed_rows=resumed_rows,
        started_at=started_at,
        session_seconds=time.perf_counter() - session_started,
    )
    print(f"Predictions: {output_path}")
    print(f"Manifest:    {manifest_path}")


if __name__ == "__main__":
    main()
