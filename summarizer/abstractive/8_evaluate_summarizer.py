"""
SinhalaJournal-LLM | Step 8b: Frozen-split summarizer evaluation (Phase 3)
----------------------------------------------------------------------------
Evaluates ONE frozen-split adapter (summarization_sinllama_v06_frozensplit or
_v07_frozensplit, produced by 8_train_summarizer_v0{6,7}_frozensplit.py)
against data/summarization_frozen_eval_subset.jsonl — the fixed, leakage-free
300-article subset carved out by 8_freeze_dataset_split.py. Run it once per
adapter; the two output JSON files are then directly comparable, because:

  * Same 300 articles, same order, for both adapters (frozen file, no
    resampling) — unlike 6/7_test_summarizer.py, which draw a fresh
    random.sample() from the full corpus each run.
  * Same metric set for both — ROUGE-1/2/L (native Sinhala grapheme-cluster
    implementation; the standard rouge_score library breaks on Sinhala
    Unicode), length-band adherence, clean-ending rate, AND the v07-only
    glue/unit-mismatch checks from 7_test_summarizer.py. Applying the glue
    check to v06 too is deliberate: this is the first evaluation where that
    comparison is actually meaningful, since 6_test_summarizer.py never
    reported it. If v06 also glues/mismatches at some baseline rate, that
    tells you whether v07's data-quality filtering at train time actually
    changed anything or whether the earlier 6-vs-7 ROUGE comparison was
    just measuring the split leak.
  * Same generation params, matching serving (work/tasks/summarizer.py):
    repetition_penalty=1.15, no no_repeat_ngram_size, greedy decoding.

Neither adapter was trained on any article in this eval subset — v06's
frozensplit script trains on summarization_frozen_train.jsonl (a disjoint
partition of the same raw-file shuffle), and v07's additionally intersects
that partition with the clean-URL set. Both exclude the frozen val/test/eval
partitions by construction. See SUMMARIZATION_NEXT_STEPS.md Phase 1-2 for the
audit that established this (contrast with the ~81% contamination the OLD
v06/v07 adapters have against this same frozen test set).

Usage:
    python abstractive/8_evaluate_summarizer.py --adapter v06
    python abstractive/8_evaluate_summarizer.py --adapter v07
    python abstractive/8_evaluate_summarizer.py --adapter v06 --samples 20   # smoke test
"""

import json
import time
import argparse
import warnings
import unicodedata
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter

import torch
import numpy as np
from transformers import AutoTokenizer
from unsloth import FastLanguageModel
from peft import PeftModel

from data_quality_checks import detect_word_glue, check_numeric_unit_consistency

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# PATHS / CONFIG
# ──────────────────────────────────────────────
SINLLAMA_BASE = "/home/jovyan/work/sinllama/models/SinLLaMA-merged-base"
ADAPTERS_STAGING = Path("/home/jovyan/work/sinllama/models/adapters_staging")
ADAPTER_PATHS = {
    "v06": ADAPTERS_STAGING / "summarization_sinllama_v06_frozensplit",
    "v07": ADAPTERS_STAGING / "summarization_sinllama_v07_frozensplit",
}

EVAL_DATASET = Path("/home/jovyan/summarizer/data/summarization_frozen_eval_subset.jsonl")
MANIFEST_PATH = Path("/home/jovyan/summarizer/data/summarization_frozen_split_manifest.json")
OUTPUT_DIR = Path("/home/jovyan/summarizer/6_eval_results")

MAX_SEQ_LENGTH = 2048

# Trained compression bands per bucket (token-ratio space) — identical in
# 6_train_summarizer.BUCKET_FILTERS and 7_train_summarizer.BUCKET_FILTERS,
# unchanged by the frozensplit retrain (only the split boundary changed).
BUCKET_BANDS = {
    "short":  {"min_ratio": 0.04, "max_ratio": 0.18},
    "medium": {"min_ratio": 0.12, "max_ratio": 0.32},
    "long":   {"min_ratio": 0.22, "max_ratio": 0.55},
}

# Serving-side token budgets per bucket (work/tasks/summarizer.py).
BUCKET_TOKEN_BUDGETS = {"short": 80, "medium": 130, "long": 200}

LENGTH_LINES = {
    "short":  "සාරාංශය ඉතා කෙටි විය යුතුය — මුල් ලිපියේ දිගෙන් 10%ක් පමණ.",
    "medium": "සාරාංශය මධ්‍යම දිගකින් විය යුතුය — මුල් ලිපියේ දිගෙන් 20%ක් පමණ.",
    "long":   "සාරාංශය සවිස්තරාත්මක විය යුතුය — මුල් ලිපියේ දිගෙන් 35%ක් පමණ.",
}

ASSISTANT_HEADER = "<|start_header_id|>assistant<|end_header_id|>\n\n"


def build_prompt(article: str, bucket: str) -> str:
    # Byte-identical to 6/7_train_summarizer.format_prompt minus the response
    # (both recipes share this template — only the training data differs).
    return f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

පහත සිංහල පුවත් ලිපිය සාරාංශ කරන්න.

ලිපියේ ඇති තොරතුරු පමණක් භාවිතා කරන්න.
{LENGTH_LINES[bucket]}
අමතර අදහස්, විශ්ලේෂණ හෝ නව තොරතුරු එකතු නොකරන්න.

Article:
{article}

<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""


# ──────────────────────────────────────────────
# NATIVE SINHALA ROUGE (grapheme clusters — rouge_score library breaks on
# Sinhala Unicode; do not replace with the standard library)
# ──────────────────────────────────────────────
def sinhala_tokenize(text: str) -> list:
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


def rouge_scores(pred: str, ref: str) -> dict:
    def ngrams(tokens, n):
        return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1))

    def lcs_length(a, b):
        m, n = len(a), len(b)
        prev = [0] * (n + 1)
        curr = [0] * (n + 1)
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if a[i-1] == b[j-1]:
                    curr[j] = prev[j-1] + 1
                else:
                    curr[j] = max(prev[j], curr[j-1])
            prev, curr = curr, [0] * (n + 1)
        return prev[n]

    pred_toks = sinhala_tokenize(pred)
    ref_toks = sinhala_tokenize(ref)
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
    prec = lcs / len(pred_toks)
    rec = lcs / len(ref_toks)
    out["rougeL"] = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return out


# ──────────────────────────────────────────────
# MODEL
# ──────────────────────────────────────────────
def load_model(adapter_path: Path, adapter_label: str):
    print("🔹 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(SINLLAMA_BASE, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token

    print("🔹 Loading SinLLaMA base (4-bit)...")
    model, _ = FastLanguageModel.from_pretrained(
        model_name=SINLLAMA_BASE,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=torch.bfloat16,
        load_in_4bit=True,
        local_files_only=True,
        attn_implementation="eager",
    )

    print(f"🔹 Attaching summarization LoRA adapter ({adapter_label})...")
    model = PeftModel.from_pretrained(model, str(adapter_path), local_files_only=True)
    FastLanguageModel.for_inference(model)
    model.eval()
    print("   Model ready ✅\n")
    return model, tokenizer


def generate_summary(model, tokenizer, article: str, bucket: str) -> tuple[str, float]:
    prompt = build_prompt(article, bucket)
    inputs = tokenizer(
        prompt, return_tensors="pt", max_length=1800, truncation=True, padding=False,
    ).to("cuda")

    t0 = time.time()
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=BUCKET_TOKEN_BUDGETS[bucket],
            do_sample=False,
            num_beams=1,
            repetition_penalty=1.15,   # matches serving
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.time() - t0

    # Boundary-safe decode: full sequence split on the assistant header, so a
    # prompt/response boundary that falls mid-token can't corrupt the first
    # Sinhala grapheme cluster.
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    if ASSISTANT_HEADER in full_text:
        summary = full_text.split(ASSISTANT_HEADER, 1)[1]
        summary = summary.split("<|eot_id|>")[0]
        summary = summary.split("�")[0].strip()
    else:
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        summary = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return summary, elapsed


# ──────────────────────────────────────────────
# EVALUATION
# ──────────────────────────────────────────────
def load_eval_records(path: Path, limit: int | None) -> list:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if all(rec.get(f"summary_{b}", "").strip() for b in BUCKET_BANDS) and rec.get("content", "").strip():
                records.append(rec)
    if limit is not None:
        records = records[:limit]
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", choices=["v06", "v07"], required=True,
                         help="Which frozen-split adapter to evaluate.")
    parser.add_argument("--samples", type=int, default=None,
                         help="Evaluate only the first N eval-subset articles "
                              "(smoke test). Default: all 300.")
    args = parser.parse_args()

    if not EVAL_DATASET.exists():
        raise SystemExit(
            f"{EVAL_DATASET} not found — run abstractive/8_freeze_dataset_split.py first."
        )
    if MANIFEST_PATH.exists():
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        print(f"Frozen split manifest: seed={manifest['seed']} "
              f"source_sha256={manifest['source_sha256'][:12]}...")

    adapter_path = ADAPTER_PATHS[args.adapter]
    if not adapter_path.exists():
        raise SystemExit(
            f"{adapter_path} not found — run "
            f"abstractive/8_train_summarizer_{args.adapter}_frozensplit.py first."
        )

    records = load_eval_records(EVAL_DATASET, args.samples)
    print(f"Frozen eval-subset articles: {len(records):,} (of 300 total in the frozen file)")
    print(f"Evaluating {len(records)} articles x 3 lengths = {len(records) * 3} generations\n")

    model, tokenizer = load_model(adapter_path, args.adapter)

    results = defaultdict(list)
    details = []

    for i, rec in enumerate(records, 1):
        article = rec["content"].strip()
        article_tokens = len(tokenizer.encode(article, add_special_tokens=False))

        for bucket in BUCKET_BANDS:
            reference = rec[f"summary_{bucket}"].strip()
            pred, elapsed = generate_summary(model, tokenizer, article, bucket)

            pred_tokens = len(tokenizer.encode(pred, add_special_tokens=False))
            ratio = pred_tokens / article_tokens if article_tokens else 0
            band = BUCKET_BANDS[bucket]
            in_band = band["min_ratio"] <= ratio <= band["max_ratio"]
            ends_clean = pred.rstrip().endswith((".", "?", "!"))
            glue_hit = detect_word_glue(pred) is not None
            unit_hit = check_numeric_unit_consistency(pred, article) is not None

            scores = rouge_scores(pred, reference)
            results[bucket].append({
                "rougeL": scores["rougeL"],
                "rouge1": scores["rouge1"],
                "rouge2": scores["rouge2"],
                "ratio": ratio,
                "in_band": in_band,
                "ends_clean": ends_clean,
                "glue": glue_hit,
                "unit_mismatch": unit_hit,
                "latency": elapsed,
            })
            details.append({
                "sample": i, "bucket": bucket, "url": rec.get("url", ""),
                "prediction": pred, "reference": reference,
                "ratio": round(ratio, 3), "in_band": in_band,
                "ends_clean": ends_clean, "glue": glue_hit, "unit_mismatch": unit_hit,
                **{k: round(v, 4) for k, v in scores.items()},
            })
            print(f"[{i:>3}/{len(records)}] {bucket:<7} ratio={ratio:.2f} "
                  f"{'✓band' if in_band else '✗band'} "
                  f"{'✓end' if ends_clean else '✗end'} "
                  f"{'✓clean' if not glue_hit else '✗glue'} "
                  f"{'✓unit' if not unit_hit else '✗unit'} "
                  f"R-L={scores['rougeL']:.3f} ({elapsed:.1f}s)")

    # ── Report ──
    print("\n" + "=" * 76)
    print(f"  {args.adapter}_frozensplit — FROZEN EVAL SUBSET SUMMARY (leakage-controlled)")
    print("=" * 76)
    print(f"{'bucket':<8} {'R-L':>6} {'R-1':>6} {'R-2':>6} {'ratio':>6} "
          f"{'in-band':>8} {'clean-end':>10} {'glue':>6} {'unit':>6}")
    summary = {}
    for bucket, rows in results.items():
        rl = np.mean([r["rougeL"] for r in rows])
        r1 = np.mean([r["rouge1"] for r in rows])
        r2 = np.mean([r["rouge2"] for r in rows])
        ratio = np.mean([r["ratio"] for r in rows])
        in_band = np.mean([r["in_band"] for r in rows]) * 100
        clean = np.mean([r["ends_clean"] for r in rows]) * 100
        glue_pct = np.mean([r["glue"] for r in rows]) * 100
        unit_pct = np.mean([r["unit_mismatch"] for r in rows]) * 100
        print(f"{bucket:<8} {rl:>6.3f} {r1:>6.3f} {r2:>6.3f} {ratio:>6.2f} "
              f"{in_band:>7.0f}% {clean:>9.0f}% {glue_pct:>5.0f}% {unit_pct:>5.0f}%")
        summary[bucket] = {
            "rougeL": float(rl), "rouge1": float(r1), "rouge2": float(r2),
            "mean_ratio": float(ratio), "in_band_pct": float(in_band),
            "clean_end_pct": float(clean), "glue_pct": float(glue_pct),
            "unit_mismatch_pct": float(unit_pct),
        }

    mean_ratios = {b: np.mean([r["ratio"] for r in rows]) for b, rows in results.items()}
    separated = mean_ratios["short"] < mean_ratios["medium"] < mean_ratios["long"]
    print(f"\nLength modes separated (short < medium < long): {'YES ✓' if separated else 'NO ✗'}")
    print("  " + "  ".join(f"{b}={mean_ratios[b]:.2f}" for b in ("short", "medium", "long")))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUT_DIR / f"frozensplit_{args.adapter}_eval_{stamp}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "adapter": args.adapter,
            "adapter_path": str(adapter_path),
            "eval_dataset": str(EVAL_DATASET),
            "n_articles": len(records),
            "summary": summary,
            "details": details,
        }, f, ensure_ascii=False, indent=2)
    print(f"\nDetailed results: {out_path}")


if __name__ == "__main__":
    main()
