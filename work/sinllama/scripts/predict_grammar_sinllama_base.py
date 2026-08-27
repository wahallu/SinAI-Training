#!/usr/bin/env python3
"""Run the frozen Stage 6 grammar set on base SinLLaMA with no task LoRA.

This is the ablation condition:

    SinLLaMA-merged-base + the unchanged grammar prompt + no PEFT adapter

The default run accepts only the canonical 286-row Stage 6 input hash. It
appends and fsyncs one prediction at a time, safely resumes a matching prefix,
and records an audit manifest proving that no task adapter was requested or
loaded.

Run from ``work/sinllama`` on the GPU machine:

    python scripts/predict_grammar_sinllama_base.py \
      --base-model models/SinLLaMA-merged-base \
      --input-data data/grammar_stage6_inputs.jsonl \
      --output Tested_results/sinllama_base_stage6_predictions.jsonl

Use ``--dry-run`` before loading the model. ``--limit`` is only for a smoke
test; rerun the same command without it to resume and complete all 286 rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


INSTRUCTION = (
    "Correct the grammar of the Sinhala sentence. "
    "ONLY fix errors. "
    "If the sentence is already correct, return it EXACTLY unchanged — "
    "do not rephrase, reorder, or change tense."
)
PROMPT_TEMPLATE = (
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n"
)
FROZEN_STAGE6_INPUT_SHA256 = (
    "e6dfdf31a8d75264a313450d9ce727949cf4a82916e411c972b998060066a66e"
)
FROZEN_STAGE6_ROWS = 286
INSTRUCTION_SHA256 = hashlib.sha256(INSTRUCTION.encode("utf-8")).hexdigest()
PROMPT_TEMPLATE_SHA256 = hashlib.sha256(
    PROMPT_TEMPLATE.format(instruction=INSTRUCTION, input="{input}").encode("utf-8")
).hexdigest()
CONDITION = "sinllama_merged_base_no_task_lora"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-model",
        default="models/SinLLaMA-merged-base",
        help="Local merged SinLLaMA base directory; must not be a PEFT adapter",
    )
    parser.add_argument("--input-data", required=True, help="Frozen Stage 6 input JSONL")
    parser.add_argument("--output", required=True, help="Prediction JSONL to create/resume")
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None, help="Smoke-test the first N rows")
    parser.add_argument(
        "--expected-input-sha256",
        default=FROZEN_STAGE6_INPUT_SHA256,
        help="Fail unless the complete input file has this SHA-256",
    )
    parser.add_argument(
        "--skip-input-hash-check",
        action="store_true",
        help="Allow a changed 286-row file for diagnostics; not valid for final comparison",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the dataset and experiment configuration without loading the model",
    )
    args = parser.parse_args()

    if args.max_seq_length < 1:
        parser.error("--max-seq-length must be at least 1")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be at least 1")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    return args


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def normalize(text: object) -> str:
    return unicodedata.normalize("NFC", str(text)).strip()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
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
    if len(rows) != FROZEN_STAGE6_ROWS:
        raise ValueError(
            f"Stage 6 must contain exactly {FROZEN_STAGE6_ROWS} rows; found {len(rows)}"
        )
    return rows


def build_prompt(text: str) -> str:
    return PROMPT_TEMPLATE.format(instruction=INSTRUCTION, input=text)


def generation_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "do_sample": False,
        "temperature": 1.0,
        "repetition_penalty": 1.0,
        "max_seq_length": args.max_seq_length,
        "max_new_tokens": args.max_new_tokens,
        "stop_at_newline": True,
    }


def assert_base_model_has_no_adapter(base_path: Path) -> None:
    if not base_path.is_dir():
        raise FileNotFoundError(f"Base model directory not found: {base_path}")

    forbidden = [base_path / "adapter_config.json"]
    forbidden.extend(base_path.glob("adapter_model*"))
    present = sorted(path.name for path in forbidden if path.exists())
    if present:
        raise RuntimeError(
            "Refusing the no-LoRA baseline because the base-model directory contains "
            f"adapter artifacts: {', '.join(present)}"
        )


def install_tokenizer_compatibility_alias(base_path: Path) -> bool:
    """Let Transformers 4.x load tokenizer metadata written by 5.x."""
    config_path = base_path / "tokenizer_config.json"
    if not config_path.is_file():
        return False
    try:
        tokenizer_class = json.loads(
            config_path.read_text(encoding="utf-8")
        ).get("tokenizer_class")
    except (OSError, json.JSONDecodeError):
        return False
    if tokenizer_class != "TokenizersBackend":
        return False

    from transformers import PreTrainedTokenizerFast
    from transformers.models.auto import tokenization_auto

    original_resolver = tokenization_auto.tokenizer_class_from_name
    if original_resolver("TokenizersBackend") is not None:
        return False

    def compatible_resolver(class_name: str):
        if class_name == "TokenizersBackend":
            return PreTrainedTokenizerFast
        return original_resolver(class_name)

    tokenization_auto.tokenizer_class_from_name = compatible_resolver
    return True


def load_resume_prefix(
    output_path: Path,
    expected_rows: list[dict[str, str]],
    args: argparse.Namespace,
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
                    "Repair or move the output before retrying."
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(f"Cannot resume: row {line_number} is not an object")
            completed.append(row)

    if len(completed) > len(expected_rows):
        raise ValueError("Cannot resume: output has more rows than this run requests")

    expected_config = generation_config(args)
    seen_ids: set[str] = set()
    for index, existing in enumerate(completed):
        expected = expected_rows[index]
        row_id = expected["id"]
        if existing.get("id") != row_id:
            raise ValueError(
                f"Cannot resume: output row {index + 1} has ID "
                f"{existing.get('id')!r}; expected {row_id!r}"
            )
        if normalize(existing.get("input", "")) != expected["input"]:
            raise ValueError(f"Cannot resume: input mismatch for {row_id}")
        if not normalize(existing.get("prediction", "")):
            raise ValueError(f"Cannot resume: missing prediction for {row_id}")
        if row_id in seen_ids:
            raise ValueError(f"Cannot resume: duplicate output ID {row_id}")
        if existing.get("condition") != CONDITION:
            raise ValueError(f"Cannot resume: {row_id} came from another condition")
        if existing.get("base_model_requested") != args.base_model:
            raise ValueError(f"Cannot resume: {row_id} used another base model")
        if existing.get("adapter") is not None:
            raise ValueError(f"Cannot resume: {row_id} records a task adapter")
        if existing.get("instruction_sha256") != INSTRUCTION_SHA256:
            raise ValueError(f"Cannot resume: {row_id} used another instruction")
        if existing.get("prompt_template_sha256") != PROMPT_TEMPLATE_SHA256:
            raise ValueError(f"Cannot resume: {row_id} used another prompt template")
        if existing.get("generation_config") != expected_config:
            raise ValueError(f"Cannot resume: {row_id} used another decoding config")
        seen_ids.add(row_id)
    return completed


def append_prediction(handle, row: dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def model_metadata(base_path: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for filename in (
        "config.json",
        "generation_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "model.safetensors.index.json",
    ):
        path = base_path / filename
        if path.is_file():
            metadata[filename] = {
                "sha256": file_sha256(path),
                "bytes": path.stat().st_size,
            }
    return metadata


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_data)
    output_path = Path(args.output)
    base_path = Path(args.base_model)

    input_sha256 = file_sha256(input_path)
    if not args.skip_input_hash_check and input_sha256 != args.expected_input_sha256:
        raise SystemExit(
            "Frozen Stage 6 hash mismatch:\n"
            f"  expected: {args.expected_input_sha256}\n"
            f"  actual:   {input_sha256}\n"
            "Use the canonical frozen file. --skip-input-hash-check is diagnostics-only."
        )

    all_rows = load_inputs(input_path)
    rows = all_rows[: args.limit] if args.limit is not None else all_rows
    complete_frozen_run = (
        args.limit is None
        and not args.skip_input_hash_check
        and input_sha256 == FROZEN_STAGE6_INPUT_SHA256
        and len(rows) == FROZEN_STAGE6_ROWS
    )

    print(f"Condition: {CONDITION}")
    print(f"Base model: {args.base_model}")
    print("Task adapter: NONE")
    print(f"Input rows available: {len(all_rows)}")
    print(f"Rows requested: {len(rows)}")
    print(f"Input SHA-256: {input_sha256}")
    print(f"Instruction SHA-256: {INSTRUCTION_SHA256}")
    print(f"Prompt template SHA-256: {PROMPT_TEMPLATE_SHA256}")
    print(f"Complete frozen comparison: {complete_frozen_run}")

    completed = load_resume_prefix(output_path, rows, args)
    if completed:
        print(f"Resume prefix validated: {len(completed)}/{len(rows)}")

    if args.dry_run:
        print("Dry run complete: model was not loaded and no predictions were written")
        return

    assert_base_model_has_no_adapter(base_path)

    # Unsloth must patch Transformers before Transformers is imported.
    from unsloth import FastLanguageModel

    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU not found")

    class NewlineStoppingCriteria(StoppingCriteria):
        def __init__(self, tokenizer, prompt_length: int):
            self.prompt_length = prompt_length
            self.stop_ids = set()
            for text in ("\n", "\n\n", "###"):
                token_ids = tokenizer.encode(text, add_special_tokens=False)
                if token_ids:
                    self.stop_ids.add(token_ids[0])

        def __call__(self, input_ids, scores, **kwargs) -> bool:
            del scores, kwargs
            return (
                input_ids.shape[1] > self.prompt_length
                and input_ids[0, -1].item() in self.stop_ids
            )

    if install_tokenizer_compatibility_alias(base_path):
        print(
            "Tokenizer compatibility: mapped Transformers 5 "
            "TokenizersBackend metadata to the Transformers 4 fast tokenizer"
        )

    print(f"Loading merged base model only: {base_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(base_path),
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
        local_files_only=True,
    )
    if tokenizer is None:
        raise RuntimeError("Unsloth loaded the model but did not return a tokenizer")
    if getattr(model, "peft_config", None):
        raise RuntimeError(
            "Refusing the baseline: the loaded model exposes a non-empty PEFT config"
        )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    FastLanguageModel.for_inference(model)
    model.eval()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    generated_this_run = 0
    config = generation_config(args)

    with output_path.open("a", encoding="utf-8") as output_handle:
        with torch.inference_mode():
            for index, row in enumerate(rows[len(completed):], len(completed) + 1):
                prompt = build_prompt(row["input"])
                encoded = tokenizer(prompt, return_tensors="pt").to("cuda")
                prompt_length = encoded["input_ids"].shape[1]
                if prompt_length >= args.max_seq_length:
                    raise RuntimeError(
                        f"{row['id']} prompt needs {prompt_length} tokens; "
                        "raise --max-seq-length instead of truncating"
                    )

                stop = StoppingCriteriaList(
                    [NewlineStoppingCriteria(tokenizer, prompt_length)]
                )
                item_started = time.perf_counter()
                generated = model.generate(
                    **encoded,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    temperature=1.0,
                    repetition_penalty=1.0,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.eos_token_id,
                    stopping_criteria=stop,
                    use_cache=True,
                )
                raw_prediction = tokenizer.decode(
                    generated[0][prompt_length:], skip_special_tokens=True
                ).strip()
                prediction = normalize(raw_prediction.split("\n")[0])
                fallback_reason = None
                if not prediction or len(prediction) < 2:
                    prediction = row["input"]
                    fallback_reason = "empty_or_too_short_first_line"

                prediction_row = {
                    "id": row["id"],
                    "input": row["input"],
                    "prediction": prediction,
                    "raw_prediction": normalize(raw_prediction),
                    "fallback_applied": fallback_reason is not None,
                    "fallback_reason": fallback_reason,
                    "latency_ms": round(
                        (time.perf_counter() - item_started) * 1000, 3
                    ),
                    "condition": CONDITION,
                    "base_model_requested": args.base_model,
                    "adapter": None,
                    "instruction_sha256": INSTRUCTION_SHA256,
                    "prompt_template_sha256": PROMPT_TEMPLATE_SHA256,
                    "generation_config": config,
                }
                append_prediction(output_handle, prediction_row)
                completed.append(prediction_row)
                generated_this_run += 1

                if index % 10 == 0 or index == len(rows):
                    print(f"Predicted {index}/{len(rows)}", flush=True)

    wall_seconds = time.perf_counter() - started
    inference_seconds = sum(
        float(row.get("latency_ms", 0.0)) for row in completed
    ) / 1000.0
    manifest = {
        "created_at_utc": utc_now(),
        "condition": CONDITION,
        "system": "SinLLaMA merged base baseline",
        "base_model": str(base_path.resolve()),
        "base_model_requested": args.base_model,
        "base_model_metadata": model_metadata(base_path),
        "adapter": None,
        "task_adapter_loaded": False,
        "input_path": str(input_path.resolve()),
        "input_sha256": input_sha256,
        "expected_input_sha256": args.expected_input_sha256,
        "input_hash_check_skipped": args.skip_input_hash_check,
        "input_rows_available": len(all_rows),
        "rows": len(completed),
        "complete_frozen_comparison": complete_frozen_run and len(completed) == 286,
        "output_path": str(output_path.resolve()),
        "output_sha256": file_sha256(output_path),
        "instruction": INSTRUCTION,
        "instruction_sha256": INSTRUCTION_SHA256,
        "prompt_template": PROMPT_TEMPLATE,
        "prompt_template_sha256": PROMPT_TEMPLATE_SHA256,
        "unicode_normalization": "NFC",
        "generation": config,
        "fallback_rows": sum(bool(row.get("fallback_applied")) for row in completed),
        "resume_rows": len(completed) - generated_this_run,
        "generated_rows_this_run": generated_this_run,
        "wall_seconds_this_run": round(wall_seconds, 3),
        "recorded_inference_seconds": round(inference_seconds, 3),
        "recorded_examples_per_second": (
            round(len(completed) / inference_seconds, 4) if inference_seconds else None
        ),
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
        "software": {
            "torch": torch.__version__,
        },
    }
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Predictions: {output_path}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
