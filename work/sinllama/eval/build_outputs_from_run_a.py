#!/usr/bin/env python3
"""
Step A.5 -- build the two required, clean output files from the chosen
authoritative run.

DECISION (documented per Step A.3/A.4): Run A (progress.txt / job 7311c77e,
started 11:27, completed 13:55, 1000/1000 records) is authoritative.
Verification: Run A's inline build_prompt() was diffed byte-for-byte
against the verified-correct production prompt builder
(work/tasks/style.py::prompt_style()) for all 5 styles using a test
article -- 5/5 matched exactly (see prompt_consistency_check.json for
the training-vs-serving verification, and this script's companion check
for the peer-script-vs-serving verification). Run B (this session's own
generate_frozen_outputs.py run, started 13:56, stopped at 200/1000
records once this was discovered) is DISCARDED -- its partial checkpoint
is not used for anything downstream.

Caveat carried forward explicitly (not silently dropped): Run A's
holdout population was sampled from
/home/jovyan/style_rewriter/data/train1.jsonl (521,980 rows), NOT from
the 665,887-article corpus (summarizer/all_articles_merged.json) that
this session's Step 1 (style_frozen_holdout_v1.json) was built from.
Both used seed=42 and the identical 11,867-URL exclusion set. This
script uses Run A's own holdout (frozen_holdout_articles.jsonl) as the
source-of-truth article population for the outputs below, since that's
what was actually generated against.
"""
import json
from pathlib import Path
from collections import defaultdict

RUN_A_CHECKPOINT = Path("/home/jovyan/work/sinllama/data/style_frozen_holdout/frozen_holdout_generations.jsonl")
RUN_A_ARTICLES = Path("/home/jovyan/work/sinllama/data/style_frozen_holdout/frozen_holdout_articles.jsonl")
RUN_A_IDS_META = Path("/home/jovyan/work/sinllama/data/style_frozen_holdout/frozen_holdout_ids.json")

EVAL_DIR = Path("/home/jovyan/work/sinllama/eval")
ADAPTER_OUT_PATH = EVAL_DIR / "style_frozen_adapter_outputs.json"
BASELINE_OUT_PATH = EVAL_DIR / "style_frozen_baseline_outputs.json"

BASE_MODEL = "/home/jovyan/work/sinllama/models/SinLLaMA-merged-base"
ADAPTER_PATH = "/home/jovyan/work/sinllama/models/adapters/style_sinllama_v13"

STYLE_ID_TO_PUB = {
    "style_1_formal_news": "formal",
    "style_2_editorial": "editorial",
    "style_3_sports": "sports",
    "style_4_youth": "youth",
    "style_5_feature": "feature",
}


def main():
    n_records = sum(1 for _ in open(RUN_A_CHECKPOINT, encoding="utf-8"))
    print(f"Run A checkpoint: {n_records} records")
    assert n_records == 1000, f"expected 1000 records, found {n_records}"

    with open(RUN_A_IDS_META, encoding="utf-8") as f:
        ids_meta = json.load(f)

    by_system = {"baseline": {}, "adapter": {}}
    with open(RUN_A_CHECKPOINT, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            sysname = "baseline" if rec["system"] == "no_adapter" else "adapter"
            style_pub = STYLE_ID_TO_PUB[rec["style"]]
            key = f"{rec['url']}::{style_pub}"
            by_system[sysname][key] = {
                "article_id": rec["url"],
                "style": style_pub,
                "category": rec.get("category"),
                "source_publication": None,  # not tracked by Run A's holdout metadata
                "output": rec["output"],
            }

    common_meta = {
        "run_provenance": (
            "Run A (job 7311c77e / run_frozen_holdout_eval.py), started "
            "2026-08-28 11:27, completed 13:55, 1000/1000 records. Chosen "
            "authoritative over this session's own Run B (generate_frozen_"
            "outputs.py, discarded at 200/1000 records) after verifying "
            "Run A's inline build_prompt() is byte-for-byte identical to "
            "work/tasks/style.py::prompt_style() (the verified production "
            "serving prompt) for all 5 styles."
        ),
        "holdout_population_caveat": (
            "Run A's 100-article holdout was sampled from "
            "/home/jovyan/style_rewriter/data/train1.jsonl (521,980 rows), "
            "seed=42, excluding the same 11,867 URLs used anywhere in style "
            "training/candidate generation -- NOT from the 665,887-article "
            "corpus (summarizer/all_articles_merged.json) that this "
            "session's own style_frozen_holdout_v1.json was built from. "
            f"excluded_used_urls_count={ids_meta['excluded_used_urls_count']}."
        ),
        "adapter_path": ADAPTER_PATH,
        "base_model": BASE_MODEL,
        "prompt_source": "Run A's own build_prompt(), verified byte-identical to work/tasks/style.py::prompt_style()",
    }

    adapter_out = {
        "description": "style_sinllama_v13 adapter outputs on Run A's frozen 100-article holdout, 5 styles each (500 total).",
        **common_meta,
        "count": len(by_system["adapter"]),
        "generations": by_system["adapter"],
    }
    baseline_out = {
        "description": "Plain SinLLaMA-merged-base, NO adapter, same holdout, same prompts (zero-shot baseline).",
        **common_meta,
        "count": len(by_system["baseline"]),
        "generations": by_system["baseline"],
    }

    with open(ADAPTER_OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(adapter_out, f, ensure_ascii=False, indent=2)
    with open(BASELINE_OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(baseline_out, f, ensure_ascii=False, indent=2)

    print(f"Wrote {ADAPTER_OUT_PATH} ({adapter_out['count']} records)")
    print(f"Wrote {BASELINE_OUT_PATH} ({baseline_out['count']} records)")


if __name__ == "__main__":
    main()
