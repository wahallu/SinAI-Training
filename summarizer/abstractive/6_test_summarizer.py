"""
SinhalaJournal-LLM | Step 6: Length-conditioned summarizer evaluation
----------------------------------------------------------------------
Evaluates summarization_sinllama_v06 across all three length modes.

Differences from 4_test_summarizer.py (deliberate, all traced to serving
bugs found while debugging v02-v05 on the Model Comparison page):
  * Generation params MATCH SERVING (work/tasks/summarizer.py):
    repetition_penalty=1.15 and NO no_repeat_ngram_size. The old script's
    no_repeat_ngram_size=3 scans the whole input_ids including the prompt
    (= the article), which blocks the summary from opening with the
    article's own opening phrase and corrupts the first grapheme cluster
    ("හිටපු" -> "ිටපු"). Evaluating with settings serving doesn't use
    produces scores that don't transfer.
  * Per-length-bucket evaluation: each test article is summarized at
    short/medium/long, and length-adherence (achieved compression ratio
    inside the trained band) is reported as a first-class metric — that is
    the new capability v06 exists to provide.
  * Token budget per bucket mirrors the serving-side budget.

Usage:
    python abstractive/6_test_summarizer.py
    python abstractive/6_test_summarizer.py --samples 20 --seed 99
"""

import json
import time
import random
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

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# PATHS / CONFIG
# ──────────────────────────────────────────────
SINLLAMA_BASE = "/home/jovyan/work/sinllama/models/SinLLaMA-merged-base"
SUMM_ADAPTER  = "/home/jovyan/work/sinllama/models/adapters/summarization_sinllama_v06"
TEST_DATASET  = "/home/jovyan/summarizer/data/6_multilength_summaries.jsonl"
OUTPUT_DIR    = Path("/home/jovyan/summarizer/6_eval_results")

MAX_SEQ_LENGTH  = 2048
DEFAULT_SAMPLES = 15
SEED = 42

# Trained compression bands per bucket (token-ratio space, must match
# 6_train_summarizer.BUCKET_FILTERS) — adherence is measured against these.
BUCKET_BANDS = {
    "short":  {"min_ratio": 0.04, "max_ratio": 0.18},
    "medium": {"min_ratio": 0.12, "max_ratio": 0.32},
    "long":   {"min_ratio": 0.22, "max_ratio": 0.55},
}

# Serving-side token budgets per bucket (mirror of what
# work/tasks/summarizer.py will use once v06 deploys).
BUCKET_TOKEN_BUDGETS = {"short": 80, "medium": 130, "long": 200}

LENGTH_LINES = {
    "short":  "සාරාංශය ඉතා කෙටි විය යුතුය — මුල් ලිපියේ දිගෙන් 10%ක් පමණ.",
    "medium": "සාරාංශය මධ්‍යම දිගකින් විය යුතුය — මුල් ලිපියේ දිගෙන් 20%ක් පමණ.",
    "long":   "සාරාංශය සවිස්තරාත්මක විය යුතුය — මුල් ලිපියේ දිගෙන් 35%ක් පමණ.",
}

ASSISTANT_HEADER = "<|start_header_id|>assistant<|end_header_id|>\n\n"


def build_prompt(article: str, bucket: str) -> str:
    # Byte-identical to 6_train_summarizer.format_prompt minus the response.
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
def load_model():
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

    print("🔹 Attaching summarization LoRA adapter (v06)...")
    model = PeftModel.from_pretrained(model, SUMM_ADAPTER, local_files_only=True)
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
            repetition_penalty=1.15,   # matches serving; see module docstring
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
def load_test_records(path: str) -> list:
    """Records that carry all three reference summaries."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if all(rec.get(f"summary_{b}", "").strip() for b in BUCKET_BANDS) and rec.get("content", "").strip():
                records.append(rec)
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    records = load_test_records(TEST_DATASET)
    print(f"Records with all 3 reference lengths: {len(records):,}")
    random.seed(args.seed)
    chosen = random.sample(records, min(args.samples, len(records)))
    print(f"Evaluating {len(chosen)} articles x 3 lengths = {len(chosen) * 3} generations\n")

    model, tokenizer = load_model()

    results = defaultdict(list)
    details = []

    for i, rec in enumerate(chosen, 1):
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

            scores = rouge_scores(pred, reference)
            results[bucket].append({
                "rougeL": scores["rougeL"],
                "rouge1": scores["rouge1"],
                "rouge2": scores["rouge2"],
                "ratio": ratio,
                "in_band": in_band,
                "ends_clean": ends_clean,
                "latency": elapsed,
            })
            details.append({
                "sample": i, "bucket": bucket, "url": rec.get("url", ""),
                "prediction": pred, "reference": reference,
                "ratio": round(ratio, 3), "in_band": in_band,
                "ends_clean": ends_clean, **{k: round(v, 4) for k, v in scores.items()},
            })
            print(f"[{i:>2}/{len(chosen)}] {bucket:<7} ratio={ratio:.2f} "
                  f"{'✓band' if in_band else '✗band'} "
                  f"{'✓end' if ends_clean else '✗end'} "
                  f"R-L={scores['rougeL']:.3f} ({elapsed:.1f}s)")

    # ── Report ──
    print("\n" + "=" * 64)
    print("  v06 LENGTH-CONDITIONED EVALUATION SUMMARY")
    print("=" * 64)
    print(f"{'bucket':<8} {'R-L':>6} {'R-1':>6} {'R-2':>6} {'ratio':>6} {'in-band':>8} {'clean-end':>10}")
    for bucket, rows in results.items():
        rl = np.mean([r["rougeL"] for r in rows])
        r1 = np.mean([r["rouge1"] for r in rows])
        r2 = np.mean([r["rouge2"] for r in rows])
        ratio = np.mean([r["ratio"] for r in rows])
        in_band = np.mean([r["in_band"] for r in rows]) * 100
        clean = np.mean([r["ends_clean"] for r in rows]) * 100
        print(f"{bucket:<8} {rl:>6.3f} {r1:>6.3f} {r2:>6.3f} {ratio:>6.2f} {in_band:>7.0f}% {clean:>9.0f}%")

    # Length-mode separation: short must actually be shorter than long.
    mean_ratios = {b: np.mean([r["ratio"] for r in rows]) for b, rows in results.items()}
    separated = mean_ratios["short"] < mean_ratios["medium"] < mean_ratios["long"]
    print(f"\nLength modes separated (short < medium < long): {'YES ✓' if separated else 'NO ✗'}")
    print("  " + "  ".join(f"{b}={mean_ratios[b]:.2f}" for b in ("short", "medium", "long")))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUT_DIR / f"v06_eval_{stamp}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": {b: {
            "rougeL": float(np.mean([r["rougeL"] for r in rows])),
            "mean_ratio": float(np.mean([r["ratio"] for r in rows])),
            "in_band_pct": float(np.mean([r["in_band"] for r in rows]) * 100),
            "clean_end_pct": float(np.mean([r["ends_clean"] for r in rows]) * 100),
        } for b, rows in results.items()}, "details": details}, f, ensure_ascii=False, indent=2)
    print(f"\nDetailed results: {out_path}")


if __name__ == "__main__":
    main()
