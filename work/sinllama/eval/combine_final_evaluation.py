#!/usr/bin/env python3
"""
Style Rewriter frozen-holdout evaluation -- Step 6 (combine).

NOTE ON FILE LOCATION: the task spec's Step 6 says to write the combined
file into work/sinllama/models/adapters/style_sinllama_v13/
viva_style_evaluation.json (overwriting the existing case-study file in
place). The task's own trailing instruction explicitly says NOT to change
anything inside that exact directory. That later, more specific
instruction is honored here: the ORIGINAL viva_style_evaluation.json
inside the adapter directory is read-only (never written), and the
combined file is written to a new location under work/sinllama/eval/
instead, with the original case-study content copied in unchanged under
"results" (plus the original "evaluation_type"/"warning"/"source_article"
top-level fields, also preserved unchanged).
"""
import json
from pathlib import Path

EVAL_DIR = Path("/home/jovyan/work/sinllama/eval")
ORIGINAL_VIVA_PATH = Path(
    "/home/jovyan/work/sinllama/models/adapters/style_sinllama_v13/viva_style_evaluation.json"
)  # READ-ONLY -- never written to, per the task's explicit constraint.

SCORED_PATH = EVAL_DIR / "style_frozen_scored_results.json"
CLASSIFICATION_PATH = EVAL_DIR / "style_classification_results.json"
PROMPT_CHECK_PATH = EVAL_DIR / "prompt_consistency_check.json"
# style_frozen_holdout_v1.json (665,887-corpus-based) is this session's
# original Step 1 deliverable, but Run A -- chosen authoritative in Step A --
# was generated against a DIFFERENT holdout (train1.jsonl-based). Both are
# included below, clearly labeled, rather than conflating them.
SPEC_HOLDOUT_PATH = EVAL_DIR / "style_frozen_holdout_v1.json"
RUN_A_IDS_META = Path("/home/jovyan/work/sinllama/data/style_frozen_holdout/frozen_holdout_ids.json")

OUT_PATH = EVAL_DIR / "viva_style_evaluation_combined.json"


def main():
    with open(ORIGINAL_VIVA_PATH, encoding="utf-8") as f:
        original = json.load(f)
    with open(SCORED_PATH, encoding="utf-8") as f:
        scored = json.load(f)
    with open(CLASSIFICATION_PATH, encoding="utf-8") as f:
        classification = json.load(f)
    with open(PROMPT_CHECK_PATH, encoding="utf-8") as f:
        prompt_check = json.load(f)
    with open(SPEC_HOLDOUT_PATH, encoding="utf-8") as f:
        spec_holdout = json.load(f)
    with open(RUN_A_IDS_META, encoding="utf-8") as f:
        run_a_holdout_meta = json.load(f)

    combined = {
        "_file_location_note": (
            "This combined file lives under work/sinllama/eval/, NOT inside "
            "work/sinllama/models/adapters/style_sinllama_v13/, per the "
            "task's explicit instruction not to change anything in that "
            "adapter directory. The original single-article case study at "
            "models/adapters/style_sinllama_v13/viva_style_evaluation.json "
            "was left untouched; its content is copied unchanged below."
        ),
        "evaluation_type": original.get("evaluation_type"),
        "warning": original.get("warning"),
        "source_article": original.get("source_article"),
        "results": original.get("results"),
        "frozen_holdout_evaluation": {
            "_holdout_provenance_note": (
                "Two holdout sets exist for this evaluation, built independently "
                "and NOT the same 100 articles. The scored results below were "
                "generated against RUN A's holdout ('actual_generation_holdout' "
                "here), sampled from style_rewriter/data/train1.jsonl (521,980 "
                "rows). VERIFIED VALID: every one of train1.jsonl's 521,980 "
                "unique URLs was confirmed present in the full 665,887-article "
                "corpus (summarizer/all_articles_merged.json) that this "
                "session's own Step 1 deliverable ('spec_step1_holdout') was "
                "built from -- train1.jsonl is a 100% subset/earlier extraction "
                "of that same corpus, not a separate or contaminated source. "
                "Both holdouts independently excluded the identical 11,867-URL "
                "set used anywhere in style training/candidate generation, so "
                "Run A's holdout is a legitimate unseen sample regardless of "
                "which of the two source files it was drawn from. No "
                "regeneration was needed; spec_step1_holdout is included below "
                "only as a record of this session's original (different, "
                "equally valid, but unused) 100-article draw from the same "
                "underlying corpus."
            ),
            "actual_generation_holdout": {
                "sampled_from": run_a_holdout_meta["sampled_from"],
                "holdout_size": run_a_holdout_meta["holdout_size"],
                "sampling_seed": run_a_holdout_meta["seed"],
                "excluded_used_urls_count": run_a_holdout_meta["excluded_used_urls_count"],
                "eligibility_filter": run_a_holdout_meta["eligibility_filter"],
            },
            "spec_step1_holdout_unused_for_scoring": {
                "corpus_source": spec_holdout["corpus_source"],
                "corpus_size": spec_holdout["corpus_size"],
                "holdout_size": spec_holdout["sample_size"],
                "sampling_seed": spec_holdout["sampling_seed"],
                "excluded_used_urls_count": spec_holdout["excluded_used_urls_count"],
                "category_distribution": spec_holdout["category_distribution"],
            },
            "holdout_size": run_a_holdout_meta["holdout_size"],
            "sampling_seed": run_a_holdout_meta["seed"],
            "scoring_method_note": scored["scoring_method_note"],
            "baseline_beats_adapter_anomalies": scored["baseline_beats_adapter_anomalies"],
            "per_style_results": scored["per_style_results"],
        },
        "style_classification_evaluation": {
            "method": classification["method"],
            "adapter": classification["adapter"],
            "baseline": classification["baseline"],
            "reference_example_counts_per_style": classification.get("reference_example_counts_per_style"),
            "superseded_approaches": classification.get("superseded_approaches"),
            "limitations": classification.get("limitations"),
        },
        "prompt_consistency_check": {
            "per_style": {
                style: {
                    "training_prompt": v["training_prompt"],
                    "serving_prompt": v["serving_prompt"],
                    "match": v["match"],
                }
                for style, v in prompt_check["per_style"].items()
            },
            "fix_applied": prompt_check["fix_applied"],
            "fix_description": prompt_check["fix_description"],
        },
        "human_evaluation": {
            "status": "NOT completed -- requires real human raters, not code.",
            "note": (
                "This is the one remaining manual step. A rating sheet "
                "(style adherence / meaning preservation / factual "
                "correctness / Sinhala fluency / overall usefulness, 1-5 "
                "each) already exists as a pattern in test_style_viva.py's "
                "human_evaluation_sheet(); it has not been run against the "
                "frozen holdout outputs."
            ),
        },
    }

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)

    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
