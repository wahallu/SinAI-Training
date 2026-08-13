#!/usr/bin/env python3
"""Generate and safely apply ByT5 v02 grammar edit scripts."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import time
from pathlib import Path

from grammar_edit_script import apply_edit_script, normalize
from predict_grammar_byt5 import load_inputs


PREFIX = "grammar edits: "


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--input-data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-source-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=640)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--allow-cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    if not torch.cuda.is_available() and not args.allow_cpu:
        raise SystemExit("CUDA GPU not found. Pass --allow-cpu only for diagnostics.")
    input_path, output_path = Path(args.input_data), Path(args.output)
    rows = load_inputs(input_path, args.limit)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model, dtype=dtype)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    generated_rows, statuses, rejection_reasons = [], collections.Counter(), collections.Counter()
    with torch.inference_mode():
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start : start + args.batch_size]
            lengths = [
                len(tokenizer(PREFIX + row["input"], add_special_tokens=True)["input_ids"])
                for row in batch
            ]
            if any(length > args.max_source_length for length in lengths):
                raise RuntimeError(
                    f"An input needs {max(lengths)} tokens; raise --max-source-length instead of truncating."
                )
            encoded = tokenizer(
                [PREFIX + row["input"] for row in batch], padding=True, truncation=True,
                max_length=args.max_source_length, return_tensors="pt",
            ).to(device)
            batch_started = time.perf_counter()
            generated = model.generate(
                **encoded, do_sample=False, num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens, early_stopping=args.num_beams > 1,
            )
            elapsed = time.perf_counter() - batch_started
            scripts = tokenizer.batch_decode(generated, skip_special_tokens=True)
            for row, script, token_ids in zip(batch, scripts, generated):
                eos_id = model.generation_config.eos_token_id
                generation_finished = eos_id is None or bool((token_ids == eos_id).any().item())
                result = apply_edit_script(
                    row["input"], normalize(script), generation_finished=generation_finished
                )
                statuses[result.status] += 1
                rejection_reasons.update(result.reasons)
                generated_rows.append({
                    "id": row["id"], "input": row["input"], "prediction": result.text,
                    "raw_edit_script": normalize(script), "script_status": result.status,
                    "safety_reasons": list(result.reasons), "operation_count": result.operation_count,
                    "generation_finished": generation_finished,
                    "latency_ms_batch_average": round(elapsed * 1000 / len(batch), 3),
                })
            print(f"Predicted {min(start + len(batch), len(rows))}/{len(rows)}", flush=True)

    with output_path.open("w", encoding="utf-8") as handle:
        for row in generated_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    total_seconds = time.perf_counter() - started
    manifest = {
        "model": str(Path(args.model).resolve()),
        "input_path": str(input_path.resolve()),
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "output_path": str(output_path.resolve()),
        "output_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "rows": len(rows), "prefix": PREFIX, "unicode_normalization": "NFC",
        "decoding": {"do_sample": False, "num_beams": args.num_beams,
                     "max_source_length": args.max_source_length,
                     "max_new_tokens": args.max_new_tokens},
        "safety": {"status_counts": dict(statuses), "rejection_reasons": dict(rejection_reasons)},
        "total_seconds": round(total_seconds, 3),
        "examples_per_second": round(len(rows) / total_seconds, 4),
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
    }
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Predictions: {output_path}")
    print(f"Manifest: {manifest_path}")
    print(f"Safety statuses: {dict(statuses)}")


if __name__ == "__main__":
    main()
