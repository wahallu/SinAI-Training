#!/usr/bin/env python3
"""Frozen-set predict+score for the headline adapter (v19), Stage-6-style rigor.

Reuses the exact evaluation this component has always used — same input
file, same 300-article/seed-42 subset, same bands, same artifact regex, same
own-band ROUGE/BLEU as test_headline_v18.py / test_headline_v19.py (see
CLAUDE.md's Headline section for the v18->v19->v20 table this is comparable
to). What this script adds on top, matching the grammar Stage 6 baseline
script's rigor:

  * Every (article, band) generation is a separately appended, fsynced JSONL
    row, so a crash partway through is safe to resume (rerun the same
    command; already-completed rows are skipped, not regenerated).
  * A full manifest is written: adapter identity, input file SHA-256 (
    computed and recorded, NOT pre-verified against a known-good hash --
    unlike grammar's Stage 6 set, headline_dataset_48k_balanced_val_clean.jsonl
    lives only on the GPU box and was never copied into this repo, so there
    is no independently-known "correct" hash to check against yet. If you
    want that stronger guarantee, run `sha256sum` on the file once and add
    it as EXPECTED_INPUT_SHA256 below).
  * Decoding config, timing, and peak GPU memory are all recorded.

Unlike grammar's Stage 6 script, there is no private-gold-vs-GPU split here
-- the reference headline already lives in the same input file test_headline
scripts have always used on the GPU, so this script predicts AND scores in
one pass (matching how test_headline_v19.py itself works), instead of a
separate offline scoring step.

Run from ``work/sinllama`` on the GPU machine:

    python scripts/predict_score_headline_v19.py \
      --adapter models/adapters/headline_sinllama_v19 \
      --input-data data/headline_dataset_48k_balanced_val_clean.jsonl \
      --output Tested_results/headline_v19_predictions.jsonl

Use ``--dry-run`` to validate the input file and sampling without loading
the model. ``--limit`` restricts to the first N articles for a smoke test;
rerun without it to complete all 300.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import time
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SINLLAMA_BASE_DEFAULT = "models/SinLLaMA-merged-base"
ADAPTER_DEFAULT = "models/adapters/headline_sinllama_v19"

SAMPLE_SIZE = 300  # matches test_headline_v18.py / test_headline_v19.py
SEED = 42
MAX_SEQ_LENGTH = 768
MAX_ARTICLE_CHARS = 2000
MIN_HEADLINE_CHARS = 5

# Byte-identical to HEADLINE_LENGTHS in train_headline_v18.py / v19 /
# tasks/headline.py / backend-api's prompts.py.
HEADLINE_LENGTHS = {
    "short": {"min_words": 3, "max_words": 5},
    "medium": {"min_words": 6, "max_words": 7},
    "long": {"min_words": 8, "max_words": 10},
}
TOKENS_PER_WORD_CEILING = 4.0
TOKENS_PER_WORD_FLOOR = 1.7
MIN_NEW_TOKENS_BASELINE = 5

# Trailing scraper artifacts (Hiru/ITN media tags) -- identical to
# test_headline_v18.py / v19.
ARTIFACT = re.compile(
    r"(වීඩියෝ|ජායාරූප|VIDEO|PHOTOS?|Video|PICTURES?|Interview)",
    re.IGNORECASE,
)

# Reference point from CLAUDE.md's Headline section (N=300, seed 42, same
# subset selection): v19 measured in-band short 89.7% / medium 74.3% /
# long 75.0% / overall 79.7%; artifact short 0.0% / medium 0.3% / long 3.0%
# / overall 1.1%. This run should land close to those numbers -- if it
# doesn't, something about the adapter, data file, or decoding drifted.
V19_REFERENCE = {
    "in_band": {"short": 0.897, "medium": 0.743, "long": 0.750, "overall": 0.797},
    "artifact": {"short": 0.000, "medium": 0.003, "long": 0.030, "overall": 0.011},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-model", default=SINLLAMA_BASE_DEFAULT)
    parser.add_argument("--adapter", default=ADAPTER_DEFAULT)
    parser.add_argument("--input-data", required=True, help="headline_dataset_48k_balanced_val_clean.jsonl")
    parser.add_argument("--output", required=True, help="Prediction JSONL to create/resume")
    parser.add_argument("--sample-size", type=int, default=SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--limit", type=int, default=None, help="Smoke-test the first N articles")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def normalize_sinhala(text: str) -> str:
    return unicodedata.normalize("NFC", text.strip()) if text else ""


def has_sinhala(text: str) -> bool:
    return any("඀" <= ch <= "෿" for ch in text)


def band_for(headline: str) -> str | None:
    words = len(headline.split())
    for name, band in HEADLINE_LENGTHS.items():
        if band["min_words"] <= words <= band["max_words"]:
            return name
    return None


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_val_samples(path: Path, sample_size: int, seed: int) -> list[dict]:
    samples = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            inp = item["input"]
            cat, art = "General", ""
            if "Category:" in inp:
                parts = inp.split("\n", 1)
                cat = parts[0].replace("Category:", "").strip()
                if len(parts) > 1 and "Article:" in parts[1]:
                    art = parts[1].split("Article:", 1)[1].strip()
            headline = normalize_sinhala(item["output"])
            samples.append({
                "article": normalize_sinhala(art),
                "category": cat,
                "expected": headline,
                "ref_band": band_for(headline),
            })
    if not samples:
        raise ValueError(f"No rows found in {path}")
    random.seed(seed)
    random.shuffle(samples)
    if sample_size:
        samples = samples[:sample_size]
    return samples


def build_prompt(article: str, length: str) -> str:
    article = normalize_sinhala(article)[:MAX_ARTICLE_CHARS]
    band = HEADLINE_LENGTHS[length]
    return (
        "### Instruction:\n"
        "Generate a concise Sinhala news headline for the article below.\n\n"
        "Rules:\n"
        "- Use formal Sinhala journalism style matching the article category\n"
        f"- Between {band['min_words']} and {band['max_words']} words"
        f" -- never fewer than {band['min_words']}\n"
        "- Capture the key person, event, number, or outcome\n"
        "- Output ONLY the headline, nothing else\n\n"
        f"### Input:\n{article}\n\n"
        "### Response:\n"
    )


def clean_output(result: str) -> str:
    result = result.split("\n")[0].strip()
    for marker in ["###", "Instruction:", "Input:", "Response:", "Category:", "Article:", "Rules:"]:
        if marker in result:
            result = result.split(marker)[0].strip()
    return normalize_sinhala(result.lstrip("-• ").strip())


def rouge_1(ref: str, hyp: str) -> float:
    r, h = set(ref.split()), set(hyp.split())
    if not r or not h:
        return 0.0
    ov = r & h
    p = len(ov) / len(h)
    rec = len(ov) / len(r)
    return (2 * p * rec) / (p + rec) if (p + rec) > 0 else 0.0


def rouge_l(ref: str, hyp: str) -> float:
    rw, hw = ref.split(), hyp.split()
    if not rw or not hw:
        return 0.0
    m, n = len(rw), len(hw)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i - 1][j - 1] + 1 if rw[i - 1] == hw[j - 1] else max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]
    p = lcs / n
    rec = lcs / m
    return (2 * p * rec) / (p + rec) if (p + rec) > 0 else 0.0


def bleu_score(ref: str, hyp: str, max_n: int = 4) -> float:
    ref_t, hyp_t = ref.split(), hyp.split()
    if not hyp_t or not ref_t:
        return 0.0
    bp = 1.0 if len(hyp_t) >= len(ref_t) else math.exp(1 - len(ref_t) / len(hyp_t))
    log_avg = 0.0
    for n in range(1, max_n + 1):
        def ngrams(t, n):
            return Counter(tuple(t[i:i + n]) for i in range(len(t) - n + 1))
        rng, hng = ngrams(ref_t, n), ngrams(hyp_t, n)
        if not hng:
            return 0.0
        ov = sum(min(rng[ng], hng.get(ng, 0)) for ng in hng)
        tot = sum(hng.values())
        prec = ov / tot if tot > 0 else 0.0
        if prec == 0:
            return 0.0
        log_avg += math.log(prec) / max_n
    return bp * math.exp(log_avg)


def load_resume_ids(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    seen = set()
    with output_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            seen.add(row["id"])
    return seen


def append_row(handle, row: dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_data)
    output_path = Path(args.output)

    samples = load_val_samples(input_path, args.sample_size, args.seed)
    if args.limit is not None:
        samples = samples[: args.limit]
    input_sha256 = file_sha256(input_path)

    work_items = [(idx, band) for idx in range(len(samples)) for band in HEADLINE_LENGTHS]

    print(f"Adapter: {args.adapter}")
    print(f"Input rows sampled: {len(samples)} (seed={args.seed}, of full file)")
    print(f"Input SHA-256: {input_sha256}")
    print(f"Generations planned: {len(work_items)} ({len(samples)} articles x 3 bands)")

    if args.dry_run:
        print("Dry run complete: model was not loaded and no predictions were written")
        return

    already = load_resume_ids(output_path)
    if already:
        print(f"Resuming: {len(already)}/{len(work_items)} rows already present")

    from unsloth import FastLanguageModel
    import torch
    from transformers import AutoTokenizer

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU not found")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("Loading pre-merged SinLLaMA base...")
    model, _ = FastLanguageModel.from_pretrained(
        model_name=args.base_model, max_seq_length=MAX_SEQ_LENGTH,
        dtype="bfloat16", load_in_4bit=True,
        local_files_only=True, attn_implementation="sdpa",
    )
    print(f"Loading headline adapter: {args.adapter}")
    model.load_adapter(args.adapter)
    FastLanguageModel.for_inference(model)
    model.eval()
    print("Model ready\n")

    def generate_headline(article_text: str, length: str) -> str:
        band = HEADLINE_LENGTHS[length]
        max_new_tokens = int(band["max_words"] * TOKENS_PER_WORD_CEILING) + 12
        min_new_tokens = max(MIN_NEW_TOKENS_BASELINE, int(band["min_words"] * TOKENS_PER_WORD_FLOOR))
        prompt = build_prompt(article_text, length)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                do_sample=True,
                temperature=0.3,
                top_p=0.9,
                repetition_penalty=1.1,
                no_repeat_ngram_size=2,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        result = clean_output(tokenizer.decode(generated_ids, skip_special_tokens=True).strip())
        if not has_sinhala(result) or len(result) < MIN_HEADLINE_CHARS:
            return ""
        return result

    output_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    generated_this_run = 0

    with output_path.open("a", encoding="utf-8") as handle:
        for idx, band_name in work_items:
            row_id = f"{idx:04d}-{band_name}"
            if row_id in already:
                continue
            item = samples[idx]
            item_started = time.perf_counter()
            gen = generate_headline(item["article"], band_name)
            band = HEADLINE_LENGTHS[band_name]
            words = len(gen.split()) if gen else 0
            in_band = bool(gen) and band["min_words"] <= words <= band["max_words"]
            artifact_hit = bool(gen) and bool(ARTIFACT.search(gen))
            own_band = band_name == item["ref_band"]
            row = {
                "id": row_id,
                "article_index": idx,
                "band": band_name,
                "category": item["category"],
                "reference": item["expected"],
                "reference_band": item["ref_band"],
                "prediction": gen,
                "words": words,
                "in_band": in_band,
                "artifact": artifact_hit,
                "empty": gen == "",
                "own_band_rouge1": rouge_1(item["expected"], gen) if own_band and gen else None,
                "own_band_rougeL": rouge_l(item["expected"], gen) if own_band and gen else None,
                "own_band_bleu": bleu_score(item["expected"], gen) if own_band and gen else None,
                "latency_ms": round((time.perf_counter() - item_started) * 1000, 3),
                "adapter": args.adapter,
            }
            append_row(handle, row)
            already.add(row_id)
            generated_this_run += 1
            if generated_this_run % 50 == 0:
                print(f"Generated {generated_this_run}/{len(work_items) - len(already) + generated_this_run}", flush=True)

    wall_seconds = time.perf_counter() - started

    # ── Aggregate from the full resumable file (covers rows from prior runs too) ──
    all_rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    band_stats = {b: {"n": 0, "in_band": 0, "artifact": 0, "empty": 0} for b in HEADLINE_LENGTHS}
    own_band_scores = []
    for row in all_rows:
        s = band_stats[row["band"]]
        s["n"] += 1
        s["in_band"] += row["in_band"]
        s["artifact"] += row["artifact"]
        s["empty"] += row["empty"]
        if row["own_band_rougeL"] is not None:
            own_band_scores.append(row)

    summary = {}
    for b, s in band_stats.items():
        summary[b] = {
            "n": s["n"],
            "in_band_rate": s["in_band"] / s["n"] if s["n"] else 0.0,
            "artifact_rate": s["artifact"] / s["n"] if s["n"] else 0.0,
            "empty_rate": s["empty"] / s["n"] if s["n"] else 0.0,
        }
    tot_n = sum(s["n"] for s in band_stats.values())
    tot_ok = sum(s["in_band"] for s in band_stats.values())
    tot_junk = sum(s["artifact"] for s in band_stats.values())
    summary["overall"] = {
        "n": tot_n,
        "in_band_rate": tot_ok / tot_n if tot_n else 0.0,
        "artifact_rate": tot_junk / tot_n if tot_n else 0.0,
    }
    if own_band_scores:
        n = len(own_band_scores)
        summary["own_band_rouge"] = {
            "n": n,
            "rouge1": sum(r["own_band_rouge1"] for r in own_band_scores) / n,
            "rougeL": sum(r["own_band_rougeL"] for r in own_band_scores) / n,
            "bleu": sum(r["own_band_bleu"] for r in own_band_scores) / n,
        }

    manifest = {
        "created_at_utc": utc_now(),
        "condition": "headline_sinllama_v19",
        "adapter_requested": args.adapter,
        "base_model_requested": args.base_model,
        "input_path": str(input_path.resolve()),
        "input_sha256": input_sha256,
        "sample_size": args.sample_size,
        "seed": args.seed,
        "output_path": str(output_path.resolve()),
        "output_sha256": file_sha256(output_path),
        "rows": len(all_rows),
        "generated_rows_this_run": generated_this_run,
        "generation_config": {
            "do_sample": True, "temperature": 0.3, "top_p": 0.9,
            "repetition_penalty": 1.1, "no_repeat_ngram_size": 2,
        },
        "summary": summary,
        "reference_v19_measurement_claude_md": V19_REFERENCE,
        "wall_seconds_this_run": round(wall_seconds, 3),
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
    }
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("\n" + "=" * 70)
    print(f"{'band':8s} {'in-band':>10s} {'artifact':>10s}")
    for b in HEADLINE_LENGTHS:
        s = summary[b]
        print(f"{b:8s} {s['in_band_rate']*100:9.1f}% {s['artifact_rate']*100:9.1f}%")
    print(f"{'overall':8s} {summary['overall']['in_band_rate']*100:9.1f}% {summary['overall']['artifact_rate']*100:9.1f}%")
    print(f"\nPredictions: {output_path}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
