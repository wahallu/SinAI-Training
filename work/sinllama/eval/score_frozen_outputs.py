#!/usr/bin/env python3
"""
Style Rewriter frozen-holdout evaluation -- Step 3 (scoring).

Reuses the EXISTING scoring functions implemented for
viva_style_evaluation.json (work/sinllama/scripts/test_style_viva.py)
rather than reimplementing them -- imported directly via importlib.

Two of the eight requested metric names -- critical_fact_preservation
and full_fact_preservation -- come from test_style_viva.py's
VIVA_FACTS / CRITICAL_FACTS checklist, which is HAND-AUTHORED for the
one fixed viva article (specific facts like "50% tax", "2026 May 16").
That checklist cannot be mechanically applied to 100 new, unseen
articles without hand-writing 100 new fact checklists, which would
defeat the purpose of an automated frozen-holdout eval. This script
therefore adapts, rather than reimplements from scratch:

  - full_fact_preservation     -> test_style_viva.score_fact_tokens()
                                   (the module's own GENERIC important-
                                   token coverage score) used AS-IS,
                                   unmodified.
  - critical_fact_preservation -> a numbers + long-token ( >=5 chars,
                                   non-stopword) coverage score, built
                                   directly from the module's own
                                   extract_numbers() / tokenize_sinhala()
                                   / STOP_WORDS primitives (imported,
                                   not copied), using the same multiset
                                   coverage algorithm score_fact_tokens()
                                   already uses. Numbers/dates/long
                                   content words are exactly what made
                                   up the original hand-authored
                                   CRITICAL_FACTS list (tax rate, date,
                                   legal basis, authority, vehicle
                                   categories), so this is the closest
                                   automatable proxy for "the same idea,
                                   at scale" -- explicitly flagged here,
                                   not silently presented as identical.

All other metrics (number_preservation, content_similarity,
length_score, style_divergence, verbatim_copy,
diagnostic_style_signal_score) are the module's existing functions,
called unmodified.
"""
import json
import importlib.util
from pathlib import Path
from collections import Counter, defaultdict

EVAL_DIR = Path("/home/jovyan/work/sinllama/eval")
# Run A (authoritative, see build_outputs_from_run_a.py) was generated
# against its own holdout (train1.jsonl-based), not this session's
# style_frozen_holdout_v1.json (665,887-corpus-based) -- source article
# text must come from the population actually used for generation.
RUN_A_ARTICLES = Path("/home/jovyan/work/sinllama/data/style_frozen_holdout/frozen_holdout_articles.jsonl")
ADAPTER_OUT_PATH = EVAL_DIR / "style_frozen_adapter_outputs.json"
BASELINE_OUT_PATH = EVAL_DIR / "style_frozen_baseline_outputs.json"
SCORED_OUT_PATH = EVAL_DIR / "style_frozen_scored_results.json"

STYLE_ID_MAP = {
    "formal": "style_1_formal_news",
    "editorial": "style_2_editorial",
    "sports": "style_3_sports",
    "youth": "style_4_youth",
    "feature": "style_5_feature",
}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


viva = load_module("test_style_viva", "/home/jovyan/work/sinllama/scripts/test_style_viva.py")


def coverage(source_tokens, output_tokens):
    """Same multiset-coverage algorithm as viva.score_fact_tokens(), generalized
    to an arbitrary token set (needed for the critical-facts proxy since
    score_fact_tokens() hardcodes important_tokens() as its extractor)."""
    if not source_tokens:
        return 100.0
    out_counter = Counter(output_tokens)
    matched = 0
    for t in source_tokens:
        if out_counter[t] > 0:
            matched += 1
            out_counter[t] -= 1
    return matched / len(source_tokens) * 100.0


def critical_tokens(text):
    tokens = viva.tokenize_sinhala(text)
    result = []
    for token in tokens:
        clean = token.strip()
        if not clean:
            continue
        if any(ch.isdigit() for ch in clean):
            result.append(clean)
            continue
        if clean in viva.STOP_WORDS:
            continue
        if len(clean) >= 5:
            result.append(clean)
    return result


def score_pair(style_pub, source, output):
    style_id = STYLE_ID_MAP[style_pub]

    number_pres, _, _ = viva.score_numbers(source, output)
    full_fact, _, _ = viva.score_fact_tokens(source, output)
    critical_fact = coverage(critical_tokens(source), critical_tokens(output))
    content_sim = viva.score_content_similarity(source, output)
    length_sc = viva.score_length(source, output)
    divergence, verbatim = viva.score_style_divergence(source, output)
    signal_sc, hits = viva.score_style_signals(style_id, output)

    return {
        "critical_fact_preservation": round(critical_fact, 2),
        "full_fact_preservation": round(full_fact, 2),
        "number_preservation": round(number_pres, 2),
        "content_similarity": round(content_sim, 2),
        "length_score": round(length_sc, 2),
        "style_divergence": round(divergence, 2),
        "verbatim_copy": verbatim,
        "diagnostic_style_signal_score": round(signal_sc, 2),
        "style_signal_hits": hits,
    }


ZERO_SCORE = {
    "critical_fact_preservation": 0.0, "full_fact_preservation": 0.0,
    "number_preservation": 0.0, "content_similarity": 0.0, "length_score": 0.0,
    "style_divergence": 0.0, "verbatim_copy": False,
    "diagnostic_style_signal_score": 0.0, "style_signal_hits": [],
}

NUMERIC_METRICS = [
    "critical_fact_preservation", "full_fact_preservation", "number_preservation",
    "content_similarity", "length_score", "style_divergence", "diagnostic_style_signal_score",
]


def main():
    article_text_by_id = {}
    with open(RUN_A_ARTICLES, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            article_text_by_id[rec["url"]] = rec["content"]
    holdout = {"articles": [{"article_id": u} for u in article_text_by_id], "sampling_seed": 42}

    with open(ADAPTER_OUT_PATH, encoding="utf-8") as f:
        adapter_data = json.load(f)
    with open(BASELINE_OUT_PATH, encoding="utf-8") as f:
        baseline_data = json.load(f)

    per_style_records = {"adapter": defaultdict(list), "baseline": defaultdict(list)}

    for system_name, data in [("adapter", adapter_data), ("baseline", baseline_data)]:
        for key, rec in data["generations"].items():
            style = rec["style"]
            article_id = rec["article_id"]
            source = article_text_by_id.get(article_id, "")
            output = rec["output"]
            if not source:
                continue
            if not output:
                score = dict(ZERO_SCORE)
            else:
                score = score_pair(style, source, output)
            per_style_records[system_name][style].append({
                "article_id": article_id, **score,
            })

    per_style_results = {}
    anomalies = []
    for style in STYLE_ID_MAP:
        adapter_recs = per_style_records["adapter"][style]
        baseline_recs = per_style_records["baseline"][style]

        def aggregate(records):
            n = len(records)
            agg = {}
            for m in NUMERIC_METRICS:
                agg[m] = round(sum(r[m] for r in records) / n, 2) if n else None
            agg["verbatim_copy_rate_pct"] = round(
                100.0 * sum(1 for r in records if r["verbatim_copy"]) / n, 2
            ) if n else None
            agg["n"] = n
            return agg

        adapter_agg = aggregate(adapter_recs)
        baseline_agg = aggregate(baseline_recs)

        for m in NUMERIC_METRICS:
            if adapter_agg[m] is not None and baseline_agg[m] is not None:
                if m == "style_divergence":
                    continue  # baseline is EXPECTED to diverge more (no fine-tuning = doesn't respect the format); not an adapter-quality axis
                if baseline_agg[m] > adapter_agg[m]:
                    anomalies.append({
                        "style": style, "metric": m,
                        "adapter": adapter_agg[m], "baseline": baseline_agg[m],
                        "note": "baseline scores HIGHER than adapter -- flagged for manual review",
                    })

        per_style_results[style] = {
            "adapter": adapter_agg,
            "no_adapter_baseline": baseline_agg,
        }

    out = {
        "holdout_size": len(holdout["articles"]),
        "sampling_seed": holdout["sampling_seed"],
        "scoring_method_note": (
            "Reuses work/sinllama/scripts/test_style_viva.py's existing scoring "
            "functions (score_numbers, score_content_similarity, score_length, "
            "score_style_divergence, score_style_signals, score_fact_tokens) "
            "unmodified. critical_fact_preservation is an explicit, disclosed "
            "adaptation (numbers + long-token coverage, built from the module's "
            "own extract_numbers()/tokenize_sinhala()/STOP_WORDS primitives) "
            "since the module's only per-article 'critical facts' concept "
            "(VIVA_FACTS/CRITICAL_FACTS) is hand-authored for one fixed viva "
            "article and cannot mechanically scale to 100 new articles. See "
            "this script's module docstring for full detail."
        ),
        "per_style_results": per_style_results,
        "baseline_beats_adapter_anomalies": anomalies,
    }

    with open(SCORED_OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"Saved {SCORED_OUT_PATH}")
    print(json.dumps({"anomalies": anomalies}, ensure_ascii=False, indent=2))
    for style, res in per_style_results.items():
        print(f"\n{style}: n_adapter={res['adapter']['n']} n_baseline={res['no_adapter_baseline']['n']}")
        for m in NUMERIC_METRICS:
            print(f"  {m:32s} adapter={res['adapter'][m]:>7} baseline={res['no_adapter_baseline'][m]:>7}")


if __name__ == "__main__":
    main()
