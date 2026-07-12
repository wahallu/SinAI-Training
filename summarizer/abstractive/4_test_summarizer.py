"""
SinhalaJournal-LLM | Step 6: Summarizer Evaluation
----------------------------------------------------
Evaluates the Sinhala summarization model against a dedicated
test dataset (test_summarization_dataset.jsonl) that contains
ground-truth summaries produced alongside training data.

Usage:
    python test_summarizer.py
    python test_summarizer.py --samples 20
    python test_summarizer.py --samples 20 --seed 99
    python test_summarizer.py --all
"""

import json
import random
import argparse
import warnings
import time
import unicodedata
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from transformers import AutoTokenizer
from unsloth import FastLanguageModel
from peft import PeftModel

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────
# NATIVE SINHALA ROUGE (grapheme clusters — rouge_score library
# breaks on Sinhala Unicode)
# ──────────────────────────────────────────────────────────────
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
    ref_toks  = sinhala_tokenize(ref)
    if not pred_toks or not ref_toks:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    p1 = ngrams(pred_toks, 1)
    r1 = ngrams(ref_toks,  1)
    c1 = sum((p1 & r1).values())
    prec1 = c1 / len(pred_toks)
    rec1  = c1 / len(ref_toks)
    f1_1  = 2*prec1*rec1/(prec1+rec1) if (prec1+rec1) else 0.0

    p2 = ngrams(pred_toks, 2)
    r2 = ngrams(ref_toks,  2)
    c2 = sum((p2 & r2).values())
    prec2 = c2 / max(len(pred_toks)-1, 1)
    rec2  = c2 / max(len(ref_toks)-1,  1)
    f1_2  = 2*prec2*rec2/(prec2+rec2) if (prec2+rec2) else 0.0

    lcs = lcs_length(pred_toks, ref_toks)
    precL = lcs / len(pred_toks)
    recL  = lcs / len(ref_toks)
    f1_L  = 2*precL*recL/(precL+recL) if (precL+recL) else 0.0

    return {"rouge1": f1_1, "rouge2": f1_2, "rougeL": f1_L}


# ──────────────────────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────────────────────
SINLLAMA_BASE = "/home/jovyan/work/sinllama/models/SinLLaMA-merged-base"
SUMM_ADAPTER  = "/home/jovyan/work/sinllama/models/adapters/summarization_sinllama_v04"

TEST_DATASET = "/home/jovyan/summarizer/test_summarization_dataset.jsonl"
OUTPUT_DIR   = Path("/home/jovyan/summarizer/eval_results")

MAX_SEQ_LENGTH = 2048
DEFAULT_SAMPLES = 15
SEED = 42

STRONG_MATCH_THRESHOLD = 0.50
EXACT_MATCH_THRESHOLD = 0.90


# ──────────────────────────────────────────────────────────────
# PROMPT
# ──────────────────────────────────────────────────────────────
def build_prompt(article: str) -> str:
    return f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

පහත සිංහල පුවත් ලිපිය සාරාංශ කරන්න.

ලිපියේ ඇති තොරතුරු පමණක් භාවිතා කරන්න.
සාරාංශය මුල් ලිපියේ දිගෙන් 10% ත් 30% ත් අතර විය යුතුය.
අමතර අදහස්, විශ්ලේෂණ හෝ නව තොරතුරු එකතු නොකරන්න.

Article:
{article}

<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""


# ──────────────────────────────────────────────────────────────
# MODEL
# ──────────────────────────────────────────────────────────────
def load_model():
    print("🔹 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(SINLLAMA_BASE, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    print(f"   Vocab size : {len(tokenizer):,} tokens")

    print("🔹 Loading SinLLaMA base (4-bit)...")
    model, _ = FastLanguageModel.from_pretrained(
        model_name=SINLLAMA_BASE,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=torch.bfloat16,
        load_in_4bit=True,
        local_files_only=True,
        attn_implementation="eager",
    )

    print("🔹 Attaching summarization LoRA adapter (v02)...")
    model = PeftModel.from_pretrained(model, SUMM_ADAPTER, local_files_only=True)
    FastLanguageModel.for_inference(model)
    model.eval()
    print("   Model ready ✅\n")
    return model, tokenizer


def generate_summary(model, tokenizer, article: str) -> tuple[str, float]:
    """Deterministic summary generation for evaluation."""
    word_count = len(article.split())
    # A loose cap, still driven by article size.
    dynamic_limit = max(40, min(120, int(word_count * 0.25)))

    prompt = build_prompt(article)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=1800,
        truncation=True,
        padding=False,
    ).to("cuda")

    t0 = time.time()
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=dynamic_limit,
            do_sample=False,
            num_beams=1,
            repetition_penalty=1.15,
            length_penalty=0.8,
            no_repeat_ngram_size=3,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.time() - t0

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    summary = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return summary, elapsed


# ──────────────────────────────────────────────────────────────
# DATASET
# ──────────────────────────────────────────────────────────────
def load_test_dataset(path: str) -> list[dict]:
    if not Path(path).exists():
        print(f"\n❌  Test dataset not found: {path}")
        raise SystemExit(1)

    records = []
    skipped = 0
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"   ⚠️  Line {line_no} — JSON parse error: {e}")
                skipped += 1
                continue

            content = rec.get("content", "").strip()
            summary = rec.get("summary", "").strip()
            if not content or not summary:
                skipped += 1
                continue

            records.append({
                "title": rec.get("title", "").strip(),
                "content": content,
                "summary": summary,
                "category": rec.get("category", "Unknown").strip(),
                "url": rec.get("url", "").strip(),
            })

    print(f"   Loaded   : {len(records):,} valid records")
    if skipped:
        print(f"   Skipped  : {skipped} (missing content or summary)")
    return records


def sample_records(records: list[dict], n: int, seed: int) -> list[dict]:
    random.seed(seed)
    chosen = random.sample(records, min(n, len(records)))
    print(f"   Sampled  : {len(chosen)} (seed={seed})\n")
    return chosen


# ──────────────────────────────────────────────────────────────
# METRICS
# ──────────────────────────────────────────────────────────────
def compute_rouge(predictions: list[str], references: list[str]) -> dict:
    scores = defaultdict(list)
    per_sample = []

    for pred, ref in zip(predictions, references):
        s = rouge_scores(pred, ref)
        for key in ("rouge1", "rouge2", "rougeL"):
            scores[key].append(s[key])
        per_sample.append({k: round(s[k], 4) for k in ("rouge1", "rouge2", "rougeL")})

    return {
        "mean": {k: float(np.mean(v)) for k, v in scores.items()},
        "std":  {k: float(np.std(v)) for k, v in scores.items()},
        "min":  {k: float(np.min(v)) for k, v in scores.items()},
        "max":  {k: float(np.max(v)) for k, v in scores.items()},
        "per_sample": per_sample,
    }


def compute_rouge_by_category(samples, predictions) -> dict:
    cat_scores = defaultdict(list)
    for rec, pred in zip(samples, predictions):
        s = rouge_scores(pred, rec["summary"])
        cat_scores[rec["category"]].append(s["rougeL"])
    return {cat: round(float(np.mean(v)), 4) for cat, v in sorted(cat_scores.items())}


def compute_length_stats(articles, predictions, references) -> dict:
    art_lens = [len(a.split()) for a in articles]
    pred_lens = [len(p.split()) for p in predictions]
    ref_lens = [len(r.split()) for r in references]
    ratios = [p / a * 100 if a else 0 for p, a in zip(pred_lens, art_lens)]

    return {
        "article_words_mean": round(float(np.mean(art_lens)), 1),
        "pred_words_mean": round(float(np.mean(pred_lens)), 1),
        "ref_words_mean": round(float(np.mean(ref_lens)), 1),
        "compression_ratio_mean": round(float(np.mean(ratios)), 1),
        "compression_ratio_std": round(float(np.std(ratios)), 1),
        "per_sample_lengths": [
            {"article": a, "predicted": p, "reference": r, "ratio_pct": round(rt, 1)}
            for a, p, r, rt in zip(art_lens, pred_lens, ref_lens, ratios)
        ],
    }


def count_matches(rouge_l_scores: list[float]) -> dict:
    exact = sum(1 for s in rouge_l_scores if s >= EXACT_MATCH_THRESHOLD)
    strong = sum(1 for s in rouge_l_scores if STRONG_MATCH_THRESHOLD <= s < EXACT_MATCH_THRESHOLD)
    partial = sum(1 for s in rouge_l_scores if s < STRONG_MATCH_THRESHOLD)
    return {
        "exact_match": exact,
        "strong_match": strong,
        "partial_match": partial,
        "total": len(rouge_l_scores),
    }


# ──────────────────────────────────────────────────────────────
# CHARTS
# ──────────────────────────────────────────────────────────────
def build_charts(rouge_data, length_data, match_data, latencies, category_scores, output_dir: Path) -> str:
    sns.set_theme(style="darkgrid", palette="muted")
    fig = plt.figure(figsize=(20, 16))
    fig.patch.set_facecolor("#0f1117")
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.48, wspace=0.35)

    TEXT_COLOR = "#e8eaf6"
    BAR_COLORS = ["#5c6bc0", "#26a69a", "#ffa726"]

    def _ax(row, col, colspan=1):
        ax = fig.add_subplot(gs[row, col] if colspan == 1 else gs[row, col:col + colspan])
        ax.set_facecolor("#1a1d27")
        ax.tick_params(colors=TEXT_COLOR, labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333650")
        return ax

    per = rouge_data["per_sample"]
    r1 = [x["rouge1"] for x in per]
    r2 = [x["rouge2"] for x in per]
    rl = [x["rougeL"] for x in per]
    idx = list(range(1, len(r1) + 1))

    ax1 = _ax(0, 0)
    means = [rouge_data["mean"]["rouge1"], rouge_data["mean"]["rouge2"], rouge_data["mean"]["rougeL"]]
    stds = [rouge_data["std"]["rouge1"], rouge_data["std"]["rouge2"], rouge_data["std"]["rougeL"]]
    labels = ["ROUGE-1", "ROUGE-2", "ROUGE-L"]
    bars = ax1.bar(labels, means, color=BAR_COLORS, width=0.5, zorder=3)
    ax1.errorbar(labels, means, yerr=stds, fmt="none", color="white", capsize=6, linewidth=1.5, zorder=4)
    for bar, val in zip(bars, means):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{val:.3f}", ha="center", va="bottom", color=TEXT_COLOR, fontsize=10, fontweight="bold")
    ax1.set_ylim(0, 1.15)
    ax1.set_title("ROUGE Scores (mean ± std)", color=TEXT_COLOR, fontsize=11, pad=10)
    ax1.set_ylabel("Score", color=TEXT_COLOR)

    ax2 = _ax(0, 1)
    ax2.plot(idx, rl, color="#7986cb", linewidth=2, marker="o", markersize=5, zorder=3, label="ROUGE-L")
    ax2.axhline(STRONG_MATCH_THRESHOLD, color="#ffa726", linewidth=1.2, linestyle="--",
                label=f"Strong ({STRONG_MATCH_THRESHOLD})")
    ax2.axhline(EXACT_MATCH_THRESHOLD, color="#ef5350", linewidth=1.2, linestyle="--",
                label=f"Exact ({EXACT_MATCH_THRESHOLD})")
    ax2.set_xlim(0.5, len(idx) + 0.5)
    ax2.set_ylim(0, 1.05)
    ax2.set_title("ROUGE-L per Sample", color=TEXT_COLOR, fontsize=11, pad=10)
    ax2.set_xlabel("Sample #", color=TEXT_COLOR)
    ax2.legend(fontsize=8, facecolor="#1a1d27", labelcolor=TEXT_COLOR)

    ax3 = _ax(0, 2)
    pie_vals = [match_data["exact_match"], match_data["strong_match"], match_data["partial_match"]]
    pie_labels = [f"Exact\n(≥{EXACT_MATCH_THRESHOLD})", f"Strong\n(≥{STRONG_MATCH_THRESHOLD})", "Partial"]
    pie_colors = ["#4caf50", "#ffa726", "#ef5350"]
    _, _, autotexts = ax3.pie(
        pie_vals, labels=pie_labels, colors=pie_colors, autopct="%1.0f%%", startangle=90,
        textprops={"color": TEXT_COLOR, "fontsize": 9},
        wedgeprops={"linewidth": 2, "edgecolor": "#0f1117"},
    )
    for at in autotexts:
        at.set_color("white")
        at.set_fontweight("bold")
    ax3.set_title("Match Distribution", color=TEXT_COLOR, fontsize=11, pad=10)

    ax4 = _ax(1, 0, colspan=2)
    hm = np.array([r1, r2, rl])
    im = ax4.imshow(hm, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1, interpolation="nearest")
    ax4.set_yticks([0, 1, 2])
    ax4.set_yticklabels(["ROUGE-1", "ROUGE-2", "ROUGE-L"], color=TEXT_COLOR)
    ax4.set_xticks(range(len(idx)))
    ax4.set_xticklabels([str(i) for i in idx], color=TEXT_COLOR, fontsize=8)
    ax4.set_title("ROUGE Scores Heatmap (per Sample)", color=TEXT_COLOR, fontsize=11, pad=10)
    ax4.set_xlabel("Sample #", color=TEXT_COLOR)
    cbar = plt.colorbar(im, ax=ax4, fraction=0.03, pad=0.02)
    cbar.ax.tick_params(colors=TEXT_COLOR)
    for i in range(3):
        for j in range(len(idx)):
            val = hm[i, j]
            ax4.text(j, i, f"{val:.2f}", ha="center", va="center",
                     color="black" if val > 0.5 else "white", fontsize=7)

    ax5 = _ax(1, 2)
    if category_scores:
        cats = list(category_scores.keys())
        cvals = list(category_scores.values())
        ax5.barh(cats, cvals, color="#7986cb", edgecolor="#0f1117", zorder=3)
        for i, v in enumerate(cvals):
            ax5.text(v + 0.005, i, f"{v:.3f}", va="center", color=TEXT_COLOR, fontsize=8)
        ax5.set_xlim(0, 1.1)
        ax5.set_title("ROUGE-L by Category", color=TEXT_COLOR, fontsize=11, pad=10)
        ax5.set_xlabel("ROUGE-L Mean", color=TEXT_COLOR)
    else:
        ax5.text(0.5, 0.5, "No category data", ha="center", va="center",
                 color=TEXT_COLOR, transform=ax5.transAxes)
        ax5.set_title("ROUGE-L by Category", color=TEXT_COLOR, fontsize=11, pad=10)

    ax6 = _ax(2, 0, colspan=2)
    lengths = length_data["per_sample_lengths"]
    art_lens = [l["article"] for l in lengths]
    pred_lens = [l["predicted"] for l in lengths]
    ref_lens = [l["reference"] for l in lengths]
    x = np.arange(len(idx))
    w = 0.28
    ax6.bar(x - w, art_lens, width=w, label="Article", color="#5c6bc0", zorder=3)
    ax6.bar(x, ref_lens, width=w, label="Reference", color="#26a69a", zorder=3)
    ax6.bar(x + w, pred_lens, width=w, label="Predicted", color="#ffa726", zorder=3)
    ax6.set_xticks(x)
    ax6.set_xticklabels([str(i) for i in idx], color=TEXT_COLOR)
    ax6.set_title("Word Count: Article vs Reference vs Predicted", color=TEXT_COLOR, fontsize=11, pad=10)
    ax6.set_ylabel("Words", color=TEXT_COLOR)
    ax6.legend(fontsize=8, facecolor="#1a1d27", labelcolor=TEXT_COLOR)

    ax7 = _ax(2, 2)
    ax7.bar(idx, latencies, color="#7986cb", edgecolor="#0f1117", zorder=3)
    ax7.axhline(np.mean(latencies), color="#ffa726", linewidth=2, linestyle="--",
                label=f"Mean {np.mean(latencies):.1f}s")
    ax7.set_title("Inference Latency per Sample", color=TEXT_COLOR, fontsize=11, pad=10)
    ax7.set_xlabel("Sample #", color=TEXT_COLOR)
    ax7.set_ylabel("Seconds", color=TEXT_COLOR)
    ax7.legend(fontsize=8, facecolor="#1a1d27", labelcolor=TEXT_COLOR)

    fig.suptitle(
        "SinhalaJournal-LLM  |  Summarization Evaluation  |  Test Dataset",
        color=TEXT_COLOR, fontsize=14, fontweight="bold", y=0.98,
    )

    chart_path = output_dir / "evaluation_charts2.png"
    plt.savefig(chart_path, dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    return str(chart_path)


# ──────────────────────────────────────────────────────────────
# TERMINAL REPORT
# ──────────────────────────────────────────────────────────────
SEP = "─" * 64
SEP2 = "═" * 64

def print_header(dataset_path: str, n_total: int):
    print(f"\n{SEP2}")
    print("  SinhalaJournal-LLM  |  Summarizer Evaluation")
    print(f"{SEP2}")
    print(f"  Dataset : {dataset_path}")
    print(f"  Records : {n_total:,} total in test set")
    print(f"{SEP2}\n")


def print_sample(i, total, rec, predicted, scores, lengths, latency):
    label = (
        "🟢 EXACT" if scores["rougeL"] >= EXACT_MATCH_THRESHOLD else
        "🟡 STRONG" if scores["rougeL"] >= STRONG_MATCH_THRESHOLD else
        "🔴 PARTIAL"
    )
    print(f"\n{SEP}")
    print(f"  [{i}/{total}]  [{rec['category']}]  {rec['title'][:50]}")
    print(f"{SEP}")
    snippet = " ".join(rec["content"].split()[:50])
    print(f"  📰 Article ({lengths['article']} words):")
    print(f"     {snippet}{'...' if lengths['article'] > 50 else ''}")
    print(f"\n  ✅ Reference ({lengths['reference']} words):")
    print(f"     {rec['summary']}")
    print(f"\n  🤖 Predicted ({lengths['predicted']} words | {lengths['ratio_pct']}% | {latency:.1f}s):")
    print(f"     {predicted}")
    print(f"\n  📊 R1={scores['rouge1']:.3f}  R2={scores['rouge2']:.3f}  RL={scores['rougeL']:.3f}  {label}")


def print_summary_report(rouge_data, length_data, match_data, category_scores, latencies, n):
    print(f"\n\n{SEP2}")
    print("  AGGREGATE RESULTS")
    print(f"{SEP2}")

    print(f"\n  ROUGE Scores (n={n})")
    print(f"  {'Metric':<12} {'Mean':>8}  {'Std':>8}  {'Min':>8}  {'Max':>8}")
    print(f"  {SEP}")
    for key, label in [("rouge1", "ROUGE-1"), ("rouge2", "ROUGE-2"), ("rougeL", "ROUGE-L")]:
        m = rouge_data["mean"][key]
        s = rouge_data["std"][key]
        mn = rouge_data["min"][key]
        mx = rouge_data["max"][key]
        print(f"  {label:<12} {m:>8.3f}  {s:>8.3f}  {mn:>8.3f}  {mx:>8.3f}  {'█' * int(m * 20)}")

    print(f"\n  Length & Compression")
    print(f"  {'Metric':<32}  {'Value':>10}")
    print(f"  {SEP}")
    for label, val in [
        ("Avg article length (words)", f"{length_data['article_words_mean']}"),
        ("Avg predicted length (words)", f"{length_data['pred_words_mean']}"),
        ("Avg reference length (words)", f"{length_data['ref_words_mean']}"),
        ("Avg compression ratio", f"{length_data['compression_ratio_mean']}%"),
        ("Compression std", f"±{length_data['compression_ratio_std']}%"),
    ]:
        print(f"  {label:<32}  {val:>10}")

    if category_scores:
        print(f"\n  ROUGE-L by Category")
        print(f"  {'Category':<20}  {'ROUGE-L':>8}")
        print(f"  {SEP}")
        for cat, score in category_scores.items():
            print(f"  {cat:<20}  {score:>8.3f}")

    print(f"\n  Match Quality  (total={match_data['total']})")
    print(f"  {'Category':<32}  {'Count':>5}  {'%':>6}")
    print(f"  {SEP}")
    total = match_data["total"]
    for label, key in [
        (f"Exact match  (ROUGE-L ≥ {EXACT_MATCH_THRESHOLD})", "exact_match"),
        (f"Strong match (ROUGE-L ≥ {STRONG_MATCH_THRESHOLD})", "strong_match"),
        ("Partial match", "partial_match"),
    ]:
        cnt = match_data[key]
        pct = cnt / total * 100 if total else 0
        print(f"  {label:<32}  {cnt:>5}  {pct:>5.1f}%")

    print(f"\n  Inference Latency")
    print(f"  Mean : {np.mean(latencies):.2f}s  |  Min : {np.min(latencies):.2f}s"
          f"  |  Max : {np.max(latencies):.2f}s  |  Total : {sum(latencies):.1f}s")
    print(f"\n{SEP2}\n")


# ──────────────────────────────────────────────────────────────
# SAVE JSON
# ──────────────────────────────────────────────────────────────
def save_results(samples, predictions, rouge_data, length_data, match_data, category_scores, latencies, output_dir: Path) -> str:
    results = {
        "meta": {
            "timestamp": datetime.now().isoformat(),
            "n_samples": len(samples),
            "dataset_path": TEST_DATASET,
            "adapter": SUMM_ADAPTER,
            "thresholds": {
                "exact_match": EXACT_MATCH_THRESHOLD,
                "strong_match": STRONG_MATCH_THRESHOLD,
            },
        },
        "aggregate": {
            "rouge": rouge_data["mean"],
            "rouge_by_category": category_scores,
            "lengths": {
                "article_mean": length_data["article_words_mean"],
                "predicted_mean": length_data["pred_words_mean"],
                "reference_mean": length_data["ref_words_mean"],
                "compression_pct": length_data["compression_ratio_mean"],
            },
            "matches": match_data,
            "latency": {
                "mean_s": round(float(np.mean(latencies)), 2),
                "total_s": round(float(sum(latencies)), 2),
            },
        },
        "samples": [
            {
                "id": i + 1,
                "title": s["title"],
                "category": s["category"],
                "url": s["url"],
                "article": s["content"],
                "reference": s["summary"],
                "predicted": predictions[i],
                "rouge": rouge_data["per_sample"][i],
                "lengths": length_data["per_sample_lengths"][i],
                "latency_s": round(latencies[i], 2),
            }
            for i, s in enumerate(samples)
        ],
    }

    json_path = output_dir / "eval_results-2.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    return str(json_path)


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────
def main(n_samples: int = DEFAULT_SAMPLES, seed: int = SEED, use_all: bool = False):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model()

    print(f"📂 Loading test dataset: {TEST_DATASET}")
    all_records = load_test_dataset(TEST_DATASET)
    print_header(TEST_DATASET, len(all_records))

    if use_all:
        samples = all_records
        print(f"   Mode     : full evaluation ({len(samples)} records)\n")
    else:
        samples = sample_records(all_records, n_samples, seed)

    articles = []
    references = []
    predictions = []
    latencies = []

    for i, rec in enumerate(samples):
        predicted, elapsed = generate_summary(model, tokenizer, rec["content"])

        articles.append(rec["content"])
        references.append(rec["summary"])
        predictions.append(predicted)
        latencies.append(elapsed)

        s = rouge_scores(predicted, rec["summary"])
        per_scores = {k: round(s[k], 4) for k in ("rouge1", "rouge2", "rougeL")}
        pred_words = len(predicted.split())
        art_words = len(rec["content"].split())
        ref_words = len(rec["summary"].split())
        ratio = round(pred_words / art_words * 100 if art_words else 0, 1)

        print_sample(
            i + 1, len(samples), rec, predicted, per_scores,
            {"article": art_words, "predicted": pred_words, "reference": ref_words, "ratio_pct": ratio},
            elapsed,
        )

    rouge_data = compute_rouge(predictions, references)
    length_data = compute_length_stats(articles, predictions, references)
    match_data = count_matches([x["rougeL"] for x in rouge_data["per_sample"]])
    category_scores = compute_rouge_by_category(samples, predictions)

    print_summary_report(rouge_data, length_data, match_data, category_scores, latencies, len(samples))

    print("📊 Generating evaluation charts...")
    chart_path = build_charts(rouge_data, length_data, match_data, latencies, category_scores, OUTPUT_DIR)
    print(f"   Saved → {chart_path}")

    json_path = save_results(samples, predictions, rouge_data, length_data, match_data, category_scores, latencies, OUTPUT_DIR)
    print(f"   Saved → {json_path}")

    print("\n✅ Evaluation complete!")
    print(f"   Results dir : {OUTPUT_DIR}")
    print(f"   Charts      : evaluation_charts2.png")
    print(f"   Raw data    : eval_results-2.json\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SinhalaJournal-LLM summarizer on test dataset.")
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES,
                        help=f"Random samples to evaluate (default: {DEFAULT_SAMPLES})")
    parser.add_argument("--seed", type=int, default=SEED,
                        help=f"Random seed (default: {SEED})")
    parser.add_argument("--all", action="store_true",
                        help="Evaluate every record in the test dataset (ignores --samples)")
    args = parser.parse_args()
    main(n_samples=args.samples, seed=args.seed, use_all=args.all)
