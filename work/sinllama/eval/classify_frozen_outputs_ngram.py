#!/usr/bin/env python3
"""
Style Rewriter frozen-holdout evaluation -- Step 4 classification,
REPLACEMENT approach.

WHY THE PREVIOUS TWO APPROACHES WERE ABANDONED (both tested, both
rejected -- not fabricated, tested on 10-16 real examples first):

1. Zero-shot classification with plain SinLLaMA-merged-base (no
   adapter): the model ignored the classification instruction in
   83-87% of cases across 1000 items and simply continued generating
   news-style text instead of answering. See style_classification_
   results.json's "validity_warning" (previous run, now superseded).

2. Zero-shot classification with style_sinllama_v13 ADAPTER attached,
   using a deliberately different (non-rewrite-shaped) prompt: tested
   on 15 examples spread across all 5 styles. It answered with a
   parseable label far more often (13/15), but the labels were NOT
   discriminating on content -- 3/15 (20%) correct, statistically
   indistinguishable from random guessing among 5 classes. The model
   (fine-tuned only for REWRITING, never for classification) mostly
   produced "FOMAL" (a consistent misspelling), or echoed the label
   list itself ("FOMAL, EDITORIAL, SPORTS, Y..."), regardless of the
   input's actual style. A classifier that "answers" but doesn't
   discriminate is not meaningfully more usable than one that fails to
   answer at all -- so this path was also rejected rather than run at
   full scale and reported as if it worked.

REPLACEMENT: a lightweight, LLM-free embedding/similarity classifier,
using the character-trigram cosine-similarity function ALREADY
implemented in test_style_viva.py (char_ngrams / cosine_similarity_
counter -- the same function that computes "content_similarity" in
Step 3's scoring, reused here rather than reimplemented). For each of
the 5 styles, a reference profile is built from ~20-30 real, human-
accepted rewritten_text examples of that style from the ACTUAL
training data (style_rewriter/data/test_words_fixed.jsonl, the
7,555-row accepted set -- disjoint from the frozen holdout articles,
so no leakage). Each of the 1000 holdout outputs is then classified by
nearest cosine similarity to the 5 reference profiles.

LIMITATIONS, stated honestly:
- This measures surface-level character/vocabulary similarity to a
  small reference sample per style, NOT deep stylistic/semantic
  judgment. It is a coarse proxy, not a validated style classifier.
- Reference profiles are built from only ~20-30 examples per style
  (limited by feature-5's small pool -- 539 rows total in the source
  file), so profiles for underrepresented styles are noisier.
- This has never been validated against human judgments of style
  correctness, so its absolute accuracy numbers should be read as a
  weak, disclosed signal -- not as ground truth.
"""
import json
import random
import importlib.util
from pathlib import Path
from collections import Counter, defaultdict

EVAL_DIR = Path("/home/jovyan/work/sinllama/eval")
ADAPTER_OUT_PATH = EVAL_DIR / "style_frozen_adapter_outputs.json"
BASELINE_OUT_PATH = EVAL_DIR / "style_frozen_baseline_outputs.json"
RESULT_PATH = EVAL_DIR / "style_classification_results.json"
TRAINING_DATA_PATH = Path("/home/jovyan/style_rewriter/data/test_words_fixed.jsonl")

N_REFERENCE_PER_STYLE = 25
SEED = 42

STYLE_ID_TO_PUB = {
    "style_1_formal_news": "formal", "style_2_editorial": "editorial",
    "style_3_sports": "sports", "style_4_youth": "youth", "style_5_feature": "feature",
}
LABELS = ["FORMAL", "EDITORIAL", "SPORTS", "YOUTH", "FEATURE"]
PUB_TO_LABEL = {v: k.upper() for k, v in {
    "formal": "formal", "editorial": "editorial", "sports": "sports",
    "youth": "youth", "feature": "feature",
}.items()}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


viva = load_module("test_style_viva", "/home/jovyan/work/sinllama/scripts/test_style_viva.py")


def build_reference_profiles():
    by_style = defaultdict(list)
    with open(TRAINING_DATA_PATH, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            pub = STYLE_ID_TO_PUB.get(rec.get("style"))
            text = (rec.get("rewritten_text") or "").strip()
            if pub and text:
                by_style[pub].append(text)

    rng = random.Random(SEED)
    profiles = {}
    counts = {}
    for pub, texts in by_style.items():
        rng.shuffle(texts)
        sample = texts[:N_REFERENCE_PER_STYLE]
        counts[pub] = len(sample)
        combined = " ".join(sample)
        profiles[pub] = viva.char_ngrams(combined, 3)
    return profiles, counts


def classify(text, profiles):
    text_ngrams = viva.char_ngrams(text, 3)
    sims = {pub: viva.cosine_similarity_counter(text_ngrams, prof) for pub, prof in profiles.items()}
    best_pub = max(sims, key=sims.get)
    return best_pub.upper(), sims


def compute_prf(y_true, y_pred, labels):
    per_label = {}
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_label[label] = {
            "precision": round(precision * 100, 2), "recall": round(recall * 100, 2),
            "f1": round(f1 * 100, 2), "support": sum(1 for t in y_true if t == label),
        }
    n = len(y_true)
    accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / n * 100 if n else 0.0
    macro_p = sum(v["precision"] for v in per_label.values()) / len(labels)
    macro_r = sum(v["recall"] for v in per_label.values()) / len(labels)
    macro_f1 = sum(v["f1"] for v in per_label.values()) / len(labels)
    return {"accuracy": round(accuracy, 2), "precision": round(macro_p, 2),
            "recall": round(macro_r, 2), "f1": round(macro_f1, 2), "per_style": per_label}


def main():
    profiles, ref_counts = build_reference_profiles()
    print("Reference example counts per style:", ref_counts)

    with open(ADAPTER_OUT_PATH, encoding="utf-8") as f:
        adapter_data = json.load(f)
    with open(BASELINE_OUT_PATH, encoding="utf-8") as f:
        baseline_data = json.load(f)

    result = {
        "method": (
            "Lightweight, LLM-free character-trigram cosine-similarity "
            "classifier (reusing test_style_viva.py's char_ngrams()/"
            "cosine_similarity_counter(), the same function behind Step 3's "
            "content_similarity metric). Reference profiles built from "
            f"{N_REFERENCE_PER_STYLE} real human-accepted training examples "
            "per style (style_rewriter/data/test_words_fixed.jsonl, seed=42 "
            "sample, disjoint from the frozen holdout). Each output is "
            "classified by nearest cosine similarity to the 5 reference "
            "profiles. Replaces two rejected zero-shot LLM approaches (see "
            "'superseded_approaches' below) that were tested on 10-16 real "
            "examples each and found non-discriminating or non-answering "
            "before being abandoned -- not run at full scale and reported "
            "anyway."
        ),
        "reference_example_counts_per_style": ref_counts,
        "superseded_approaches": [
            {
                "approach": "Zero-shot classification, plain SinLLaMA-merged-base (no adapter)",
                "outcome": "REJECTED -- 83-87% of 1000 completions ignored the instruction and continued generating article text instead of answering.",
            },
            {
                "approach": "Zero-shot classification, style_sinllama_v13 ADAPTER attached, non-rewrite-shaped prompt",
                "outcome": "REJECTED after a 15-example pretest (3/5 styles represented, all 5 tested) -- 13/15 produced a parseable label, but only 3/15 (20%) were correct, statistically indistinguishable from random guessing among 5 classes. The adapter (fine-tuned only for rewriting) does not discriminate style from content when repurposed as a classifier; it defaults toward 'FORMAL' or echoes the label list regardless of the input's actual style.",
            },
        ],
        "limitations": (
            "This is a coarse, surface-level (character n-gram) similarity "
            "proxy, not a validated style classifier, and has not been "
            "checked against human judgments. Reference profiles for the "
            "smallest-represented style (feature, 539 total training rows) "
            "are the noisiest. Treat these accuracy numbers as a weak, "
            "disclosed signal for relative comparison (adapter vs baseline), "
            "not as ground truth about style quality."
        ),
        "labels": LABELS,
    }

    for system_name, data in [("adapter", adapter_data), ("baseline", baseline_data)]:
        y_true, y_pred = [], []
        for key, rec in data["generations"].items():
            text = rec["output"]
            if not text:
                continue
            pred, _sims = classify(text, profiles)
            y_true.append(rec["style"].upper())
            y_pred.append(pred)
        r = compute_prf(y_true, y_pred, LABELS)
        r["n"] = len(y_true)
        result[system_name] = r

    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {RESULT_PATH}")
    for system_name in ("adapter", "baseline"):
        r = result[system_name]
        print(f"\n{system_name}: n={r['n']} accuracy={r['accuracy']} precision={r['precision']} recall={r['recall']} f1={r['f1']}")
        for style, m in r["per_style"].items():
            print(f"  {style:10s} precision={m['precision']:6.2f} recall={m['recall']:6.2f} f1={m['f1']:6.2f} support={m['support']}")


if __name__ == "__main__":
    main()
