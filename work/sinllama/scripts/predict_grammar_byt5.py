#!/usr/bin/env python3
"""Generate deterministic ByT5 grammar predictions from input-only or gold JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import unicodedata
from pathlib import Path


PREFIX = "grammar: "


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Trained ByT5 directory")
    parser.add_argument("--input-data", required=True, help="JSONL with id/input or input/output")
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-source-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--allow-cpu", action="store_true")
    return parser.parse_args()


def normalize(text: str) -> str:
    return unicodedata.normalize("NFC", text).strip()


def load_inputs(path: Path, limit: int | None) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            source = normalize(str(row.get("input", "")))
            if not source:
                raise ValueError(f"Missing input near line {index + 1}")
            rows.append({"id": str(row.get("id", f"row-{len(rows) + 1:06d}")), "input": source})
            if limit is not None and len(rows) >= limit:
                break
    if not rows:
        raise ValueError("No input rows found")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Input IDs are not unique")
    return rows


def main() -> None:
    args = parse_args()
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    if not torch.cuda.is_available() and not args.allow_cpu:
        raise SystemExit("CUDA GPU not found. Pass --allow-cpu only for diagnostics.")

    input_path = Path(args.input_data)
    output_path = Path(args.output)
    rows = load_inputs(input_path, args.limit)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float32
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model, torch_dtype=dtype)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    source_sha = hashlib.sha256(input_path.read_bytes()).hexdigest()
    started = time.perf_counter()
    generated_rows = []
    with torch.inference_mode():
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start : start + args.batch_size]
            raw_lengths = [
                len(tokenizer(PREFIX + row["input"], add_special_tokens=True)["input_ids"])
                for row in batch
            ]
            if any(length > args.max_source_length for length in raw_lengths):
                longest = max(raw_lengths)
                raise RuntimeError(
                    f"An input needs {longest} tokens but --max-source-length is "
                    f"{args.max_source_length}; raise the limit instead of truncating"
                )
            encoded = tokenizer(
                [PREFIX + row["input"] for row in batch],
                padding=True,
                truncation=True,
                max_length=args.max_source_length,
                return_tensors="pt",
            ).to(device)
            batch_started = time.perf_counter()
            output_ids = model.generate(
                **encoded,
                do_sample=False,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                early_stopping=args.num_beams > 1,
            )
            elapsed = time.perf_counter() - batch_started
            predictions = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            per_item_ms = elapsed * 1000 / len(batch)
            for row, prediction in zip(batch, predictions):
                generated_rows.append(
                    {
                        "id": row["id"],
                        "input": row["input"],
                        "prediction": normalize(prediction),
                        "latency_ms_batch_average": round(per_item_ms, 3),
                    }
                )
            print(f"Predicted {min(start + len(batch), len(rows))}/{len(rows)}", flush=True)

    with output_path.open("w", encoding="utf-8") as handle:
        for row in generated_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    total_seconds = time.perf_counter() - started
    manifest = {
        "model": str(Path(args.model).resolve()),
        "input_path": str(input_path.resolve()),
        "input_sha256": source_sha,
        "output_path": str(output_path.resolve()),
        "rows": len(rows),
        "prefix": PREFIX,
        "unicode_normalization": "NFC",
        "decoding": {
            "do_sample": False,
            "num_beams": args.num_beams,
            "max_source_length": args.max_source_length,
            "max_new_tokens": args.max_new_tokens,
        },
        "total_seconds": round(total_seconds, 3),
        "examples_per_second": round(len(rows) / total_seconds, 4),
        "peak_gpu_memory_bytes": (
            torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        ),
    }
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Predictions: {output_path}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
