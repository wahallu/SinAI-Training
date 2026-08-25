#!/usr/bin/env python3
"""Export deterministic SinLLaMA grammar predictions for input-only JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import unicodedata
from pathlib import Path


INSTRUCTION = (
    "Correct the grammar of the Sinhala sentence. "
    "ONLY fix errors. "
    "If the sentence is already correct, return it EXACTLY unchanged — "
    "do not rephrase, reorder, or change tense."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="models/SinLLaMA-merged-base")
    parser.add_argument(
        "--adapter", default=None,
        help="LoRA adapter dir. Omit to run the bare base model (no adapter attached).",
    )
    parser.add_argument("--input-data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def normalize(text: object) -> str:
    return unicodedata.normalize("NFC", str(text)).strip()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_inputs(path: Path, limit: int | None) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            source = normalize(row.get("input", ""))
            if not source:
                raise ValueError(f"Missing input at line {line_number}")
            rows.append({
                "id": str(row.get("id", f"row-{len(rows) + 1:06d}")),
                "input": source,
            })
            if limit is not None and len(rows) >= limit:
                break
    if not rows:
        raise ValueError("No input rows found")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Input IDs are not unique")
    return rows


def build_prompt(text: str) -> str:
    return (
        f"### Instruction:\n{INSTRUCTION}\n\n"
        f"### Input:\n{text}\n\n"
        "### Response:\n"
    )


def install_tokenizer_compatibility_alias(base_path: Path) -> bool:
    """Let Transformers 4.x read tokenizer metadata written by 5.x.

    Transformers 5 renamed the generic fast-tokenizer implementation to
    ``TokenizersBackend``.  Older 4.x installations cannot resolve that class
    name even though they can read the same ``tokenizer.json`` through
    ``PreTrainedTokenizerFast``.  Install an in-memory alias so neither the
    model directory nor its tokenizer metadata has to be edited.
    """
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


def main() -> None:
    args = parse_args()
    # Unsloth must patch Transformers before Transformers or PEFT is imported.
    from unsloth import FastLanguageModel

    import torch
    from peft import PeftModel
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
            return (
                input_ids.shape[1] > self.prompt_length
                and input_ids[0, -1].item() in self.stop_ids
            )

    input_path = Path(args.input_data)
    output_path = Path(args.output)
    base_path = Path(args.base_model)
    adapter_path = Path(args.adapter) if args.adapter else None
    rows = load_inputs(input_path, args.limit)

    if install_tokenizer_compatibility_alias(base_path):
        print(
            "Tokenizer compatibility: mapped Transformers 5 "
            "TokenizersBackend metadata to the Transformers 4 fast tokenizer"
        )

    print(f"Loading base model and tokenizer: {base_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(base_path),
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
        local_files_only=True,
    )
    if tokenizer is None:
        raise RuntimeError("Unsloth loaded the model but did not return a tokenizer")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    if adapter_path is not None:
        print(f"Loading adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path, local_files_only=True)
    else:
        print("No adapter given — running the bare base model.")
    FastLanguageModel.for_inference(model)
    model.eval()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions = []
    started = time.perf_counter()
    with torch.inference_mode():
        for index, row in enumerate(rows, 1):
            prompt = build_prompt(row["input"])
            encoded = tokenizer(prompt, return_tensors="pt").to("cuda")
            prompt_length = encoded["input_ids"].shape[1]
            if prompt_length >= args.max_seq_length:
                raise RuntimeError(
                    f"{row['id']} prompt needs {prompt_length} tokens; raise --max-seq-length"
                )
            stop = StoppingCriteriaList([
                NewlineStoppingCriteria(tokenizer, prompt_length)
            ])
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
            prediction = tokenizer.decode(
                generated[0][prompt_length:], skip_special_tokens=True
            ).strip().split("\n")[0].strip()
            if not prediction or len(prediction) < 2:
                prediction = row["input"]
            predictions.append({
                "id": row["id"],
                "input": row["input"],
                "prediction": normalize(prediction),
                "latency_ms": round((time.perf_counter() - item_started) * 1000, 3),
            })
            if index % 10 == 0 or index == len(rows):
                print(f"Predicted {index}/{len(rows)}", flush=True)

    with output_path.open("w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    elapsed = time.perf_counter() - started
    manifest = {
        "system": "SinLLaMA grammar adapter",
        "base_model": str(base_path.resolve()),
        "adapter": str(adapter_path.resolve()) if adapter_path is not None else None,
        "input_path": str(input_path.resolve()),
        "input_sha256": file_sha256(input_path),
        "output_path": str(output_path.resolve()),
        "output_sha256": file_sha256(output_path),
        "rows": len(predictions),
        "prompt_instruction": INSTRUCTION,
        "decoding": {
            "do_sample": False,
            "max_seq_length": args.max_seq_length,
            "max_new_tokens": args.max_new_tokens,
            "stop_at_newline": True,
        },
        "total_seconds": round(elapsed, 3),
        "examples_per_second": round(len(predictions) / elapsed, 4),
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
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
