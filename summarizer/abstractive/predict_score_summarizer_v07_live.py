"""Frozen-set predict+score for the ORIGINAL (live-serving) v07 summarizer
adapter, Stage-6-style rigor.

KNOWN ISSUE -- READ BEFORE TRUSTING THE NUMBERS THIS PRODUCES:
`adapters/summarization_sinllama_v07` (this script's target) was trained
before the article-level frozen split existed. `8_audit_split_contamination.py`
measured that ~82.67% of the 300-article `summarization_frozen_eval_subset.jsonl`
this script scores against was ALREADY present in this exact adapter's
original training data (SUMMARIZATION_NEXT_STEPS.md, section 4). A high
score here is expected and does not demonstrate generalization -- it mostly
demonstrates memorization. This script exists to make that number visible
and comparable, not to claim it as a fair benchmark. The leakage-free
comparison already exists: `summarization_sinllama_v07_frozensplit`, scored
by `8_evaluate_summarizer.py --adapter v07` against the same eval subset,
with `frozensplit_v07_eval_20260826_043135.json` already committed to
`6_eval_results/`. If you want a trustworthy v07 number, use that one; this
script is for characterizing what the model users are ACTUALLY getting
today, contamination and all.

Reuses `8_evaluate_summarizer.py`'s prompt template (Llama-3 chat format --
NOT Alpaca, see CLAUDE.md), bucket token budgets, decoding config
(repetition_penalty=1.15, no no_repeat_ngram_size, greedy), and metrics
(ROUGE-1/2/L via native grapheme-cluster tokenization, length-band
adherence, clean-ending rate, word-glue / numeric-unit checks from
data_quality_checks.py) unchanged, so results are directly comparable to
the frozensplit_v06/v07 numbers already committed. What this script adds,
matching the grammar Stage 6 baseline script's rigor:

  * Every (article, bucket) generation is a separately appended, fsynced
    JSONL row -- safe to interrupt and resume (rerun the same command;
    already-completed rows are skipped).
  * A full manifest: adapter identity, input file SHA-256 (computed and
    recorded -- not pre-verified against a known-good hash, since this
    script was written without direct access to the GPU-side file; if you
    want that stronger guarantee, run `sha256sum` on the eval subset once
    and add it as EXPECTED_INPUT_SHA256 below), decoding config, timing,
    peak GPU memory, and the contamination caveat above baked in as a field
    so it travels with the results, not just this docstring.

Run from ``summarizer/`` on the GPU machine:

    python abstractive/predict_score_summarizer_v07_live.py \
      --input-data data/summarization_frozen_eval_subset.jsonl \
      --output 6_eval_results/v07_live_predictions.jsonl

Use ``--dry-run`` to validate the input file without loading the model.
``--limit`` restricts to the first N articles for a smoke test; rerun
without it to complete all 300 (273 usable after the completeness filter,
same as 8_evaluate_summarizer.py).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import unicodedata
import warnings
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from data_quality_checks import detect_word_glue, check_numeric_unit_consistency

warnings.filterwarnings("ignore")

SINLLAMA_BASE_DEFAULT = "/home/jovyan/work/sinllama/models/SinLLaMA-merged-base"
ADAPTER_DEFAULT = "/home/jovyan/work/sinllama/models/adapters/summarization_sinllama_v07"
INPUT_DEFAULT = "/home/jovyan/summarizer/data/summarization_frozen_eval_subset.jsonl"

MAX_SEQ_LENGTH = 2048

# Identical to 6/7_train_summarizer.BUCKET_FILTERS and 8_evaluate_summarizer.py.
BUCKET_BANDS = {
    "short": {"min_ratio": 0.04, "max_ratio": 0.18},
    "medium": {"min_ratio": 0.12, "max_ratio": 0.32},
    "long": {"min_ratio": 0.22, "max_ratio": 0.55},
}
BUCKET_TOKEN_BUDGETS = {"short": 80, "medium": 130, "long": 200}
LENGTH_LINES = {
    "short": "සාරාංශය ඉතා කෙටි විය යුතුය — මුල් ලිපියේ දිගෙන් 10%ක් පමණ.",
    "medium": "සාරාංශය මධ්‍යම දිගකින් විය යුතුය — මුල් ලිපියේ දිගෙන් 20%ක් පමණ.",
    "long": "සාරාංශය සවිස්තරාත්මක විය යුතුය — මුල් ලිපියේ දිගෙන් 35%ක් පමණ.",
}
ASSISTANT_HEADER = "<|start_header_id|>assistant<|end_header_id|>\n\n"

# From SUMMARIZATION_NEXT_STEPS.md section 4 (contamination audit against
# this exact frozen eval subset, for the original v07 checkpoint).
KNOWN_CONTAMINATION = {
    "source": "SUMMARIZATION_NEXT_STEPS.md section 4 (8_audit_split_contamination.py)",
    "frozen_eval_subset_contamination_pct": 82.67,
    "frozen_test_contamination_pct": 80.80,
    "caveat": (
        "This adapter was trained before the frozen split existed. ~82.67% of "
        "the 300-article eval subset scored here was already in this adapter's "
        "own training data. High scores reflect memorization, not generalization. "
        "See summarization_sinllama_v07_frozensplit for the leakage-free comparison."
    ),
}


def build_prompt(article: str, bucket: str) -> str:
    # Byte-identical to 8_evaluate_summarizer.py / 6/7_train_summarizer.format_prompt.
    return f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

පහත සිංහල පුවත් ලිපිය සාරාංශ කරන්න.

ලිපියේ ඇති තොරතුරු පමණක් භාවිතා කරන්න.
{LENGTH_LINES[bucket]}
අමතර අදහස්, විශ්ලේෂණ හෝ නව තොරතුරු එකතු නොකරන්න.

Article:
{article}

<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""


def sinhala_tokenize(text: str) -> list[str]:
    tokens = []
    chars = list(text)
    i = 0
    while i < len(chars):
        cluster = chars[i]
        i += 1
        while i < len(chars) and unicodedata.combining(chars[i]):
            cluster += chars[i]
            i += 1
        if cluster.strip():
            tokens.append(cluster)
    return tokens


def rouge_scores(pred: str, ref: str) -> dict[str, float]:
    def ngrams(tokens, n):
        return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))

    def lcs_length(a, b):
        m, n = len(a), len(b)
        prev = [0] * (n + 1)
        curr = [0] * (n + 1)
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                curr[j] = prev[j - 1] + 1 if a[i - 1] == b[j - 1] else max(prev[j], curr[j - 1])
            prev, curr = curr, [0] * (n + 1)
        return prev[n]

    pred_toks, ref_toks = sinhala_tokenize(pred), sinhala_tokenize(ref)
    if not pred_toks or not ref_toks:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    out = {}
    for n, key in ((1, "rouge1"), (2, "rouge2")):
        pn, rn = ngrams(pred_toks, n), ngrams(ref_toks, n)
        c = sum((pn & rn).values())
        prec = c / max(len(pred_toks) - n + 1, 1)
        rec = c / max(len(ref_toks) - n + 1, 1)
        out[key] = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    lcs = lcs_length(pred_toks, ref_toks)
    prec, rec = lcs / len(pred_toks), lcs / len(ref_toks)
    out["rougeL"] = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return out


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_eval_records(path: Path, limit: int | None) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if all(rec.get(f"summary_{b}", "").strip() for b in BUCKET_BANDS) and rec.get("content", "").strip():
                records.append(rec)
    if limit is not None:
        records = records[:limit]
    return records


def load_resume_ids(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    seen = set()
    with output_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                seen.add(json.loads(line)["id"])
    return seen


def append_row(handle, row: dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-model", default=SINLLAMA_BASE_DEFAULT)
    parser.add_argument("--adapter", default=ADAPTER_DEFAULT,
                         help="Must be the ORIGINAL live adapter, not a *_frozensplit staging path")
    parser.add_argument("--input-data", default=INPUT_DEFAULT)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=None, help="Smoke-test the first N articles")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if "_frozensplit" in args.adapter:
        parser.error(
            "This script is explicitly for the ORIGINAL live v07 adapter. "
            "Use 8_evaluate_summarizer.py --adapter v07 for the frozensplit adapter instead."
        )
    return args


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_data)
    output_path = Path(args.output)

    if not input_path.exists():
        raise SystemExit(f"{input_path} not found -- run abstractive/8_freeze_dataset_split.py first")

    records = load_eval_records(input_path, args.limit)
    input_sha256 = file_sha256(input_path)
    work_items = [(idx, bucket) for idx in range(len(records)) for bucket in BUCKET_BANDS]

    print(f"Adapter (ORIGINAL, live-serving): {args.adapter}")
    print(f"Input SHA-256: {input_sha256}")
    print(f"Usable articles: {len(records)} (of up to 300 in the frozen eval subset)")
    print(f"Generations planned: {len(work_items)} ({len(records)} articles x 3 buckets)")
    print(f"\n*** KNOWN ISSUE: {KNOWN_CONTAMINATION['caveat']} ***\n")

    if args.dry_run:
        print("Dry run complete: model was not loaded and no predictions were written")
        return

    already = load_resume_ids(output_path)
    if already:
        print(f"Resuming: {len(already)}/{len(work_items)} rows already present")

    import torch
    from transformers import AutoTokenizer
    from unsloth import FastLanguageModel
    from peft import PeftModel

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU not found")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading SinLLaMA base (4-bit)...")
    model, _ = FastLanguageModel.from_pretrained(
        model_name=args.base_model, max_seq_length=MAX_SEQ_LENGTH,
        dtype=torch.bfloat16, load_in_4bit=True,
        local_files_only=True, attn_implementation="eager",
    )
    print(f"Attaching summarization LoRA adapter (original v07): {args.adapter}")
    model = PeftModel.from_pretrained(model, args.adapter, local_files_only=True)
    FastLanguageModel.for_inference(model)
    model.eval()
    print("Model ready\n")

    def generate_summary(article: str, bucket: str) -> tuple[str, float]:
        prompt = build_prompt(article, bucket)
        inputs = tokenizer(prompt, return_tensors="pt", max_length=1800, truncation=True, padding=False).to("cuda")
        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=BUCKET_TOKEN_BUDGETS[bucket],
                do_sample=False,
                num_beams=1,
                repetition_penalty=1.15,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )
        elapsed = time.perf_counter() - t0
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
        if ASSISTANT_HEADER in full_text:
            summary = full_text.split(ASSISTANT_HEADER, 1)[1]
            summary = summary.split("<|eot_id|>")[0]
            summary = summary.split("�")[0].strip()
        else:
            new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
            summary = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return summary, elapsed

    output_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    generated_this_run = 0

    with output_path.open("a", encoding="utf-8") as handle:
        for idx, bucket in work_items:
            row_id = f"{idx:04d}-{bucket}"
            if row_id in already:
                continue
            rec = records[idx]
            article = rec["content"].strip()
            article_tokens = len(tokenizer.encode(article, add_special_tokens=False))
            reference = rec[f"summary_{bucket}"].strip()

            pred, elapsed = generate_summary(article, bucket)
            pred_tokens = len(tokenizer.encode(pred, add_special_tokens=False)) if pred else 0
            ratio = pred_tokens / article_tokens if article_tokens else 0.0
            band = BUCKET_BANDS[bucket]
            in_band = band["min_ratio"] <= ratio <= band["max_ratio"]
            ends_clean = pred.rstrip().endswith((".", "?", "!")) if pred else False
            glue_hit = detect_word_glue(pred) is not None if pred else False
            unit_hit = check_numeric_unit_consistency(pred, article) is not None if pred else False
            scores = rouge_scores(pred, reference)

            row = {
                "id": row_id,
                "article_index": idx,
                "bucket": bucket,
                "url": rec.get("url", ""),
                "prediction": pred,
                "reference": reference,
                "ratio": round(ratio, 4),
                "in_band": in_band,
                "ends_clean": ends_clean,
                "glue": glue_hit,
                "unit_mismatch": unit_hit,
                "rouge1": round(scores["rouge1"], 4),
                "rouge2": round(scores["rouge2"], 4),
                "rougeL": round(scores["rougeL"], 4),
                "latency_ms": round(elapsed * 1000, 3),
                "adapter": args.adapter,
            }
            append_row(handle, row)
            already.add(row_id)
            generated_this_run += 1
            print(f"[{generated_this_run:>4}] {bucket:<7} ratio={ratio:.2f} "
                  f"{'in-band' if in_band else 'out-of-band':<11} R-L={scores['rougeL']:.3f} ({elapsed:.1f}s)")

    wall_seconds = time.perf_counter() - started

    all_rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_bucket: dict[str, list[dict]] = {b: [] for b in BUCKET_BANDS}
    for row in all_rows:
        by_bucket[row["bucket"]].append(row)

    def mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    summary = {}
    for bucket, rows in by_bucket.items():
        if not rows:
            continue
        summary[bucket] = {
            "n": len(rows),
            "rougeL": mean([r["rougeL"] for r in rows]),
            "rouge1": mean([r["rouge1"] for r in rows]),
            "rouge2": mean([r["rouge2"] for r in rows]),
            "mean_ratio": mean([r["ratio"] for r in rows]),
            "in_band_pct": mean([r["in_band"] for r in rows]) * 100,
            "clean_end_pct": mean([r["ends_clean"] for r in rows]) * 100,
            "glue_pct": mean([r["glue"] for r in rows]) * 100,
            "unit_mismatch_pct": mean([r["unit_mismatch"] for r in rows]) * 100,
        }

    manifest = {
        "created_at_utc": utc_now(),
        "condition": "summarization_sinllama_v07_live_on_frozen_eval_subset",
        "known_contamination": KNOWN_CONTAMINATION,
        "adapter_requested": args.adapter,
        "base_model_requested": args.base_model,
        "input_path": str(input_path.resolve()),
        "input_sha256": input_sha256,
        "output_path": str(output_path.resolve()),
        "output_sha256": file_sha256(output_path),
        "rows": len(all_rows),
        "generated_rows_this_run": generated_this_run,
        "generation_config": {
            "do_sample": False, "num_beams": 1, "repetition_penalty": 1.15,
            "bucket_token_budgets": BUCKET_TOKEN_BUDGETS,
        },
        "summary": summary,
        "wall_seconds_this_run": round(wall_seconds, 3),
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
    }
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("\n" + "=" * 76)
    print("  v07 (LIVE, CONTAMINATED) -- FROZEN EVAL SUBSET SUMMARY")
    print("=" * 76)
    print(f"{'bucket':<8} {'R-L':>6} {'R-1':>6} {'R-2':>6} {'ratio':>6} {'in-band':>8} {'clean-end':>10} {'glue':>6} {'unit':>6}")
    for bucket, s in summary.items():
        print(f"{bucket:<8} {s['rougeL']:>6.3f} {s['rouge1']:>6.3f} {s['rouge2']:>6.3f} {s['mean_ratio']:>6.2f} "
              f"{s['in_band_pct']:>7.0f}% {s['clean_end_pct']:>9.0f}% {s['glue_pct']:>5.0f}% {s['unit_mismatch_pct']:>5.0f}%")
    print(f"\n*** {KNOWN_CONTAMINATION['frozen_eval_subset_contamination_pct']}% of this eval subset was in this adapter's training data. ***")
    print(f"\nPredictions: {output_path}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
