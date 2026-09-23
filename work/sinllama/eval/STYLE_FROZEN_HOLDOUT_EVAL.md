# Style Adapter (`style_sinllama_v13`) — Frozen-Holdout Evaluation

This folder holds the frozen-holdout evaluation pipeline and results for the
**style rewriting adapter** (`style_sinllama_v13`). It answers: on articles
the adapter never saw during training, how much better is
`SinLLaMA-merged-base + style_sinllama_v13` than the plain base model at
rewriting news articles into 5 target styles (formal / editorial / sports /
youth / feature), while preserving facts?

All result files below are kept as-is (nothing in this folder was deleted or
overwritten as part of writing this doc).

## Pipeline (scripts, in run order)

| Step | Script | Purpose |
|---|---|---|
| 1 | `generate_frozen_outputs.py` | Generates rewrites for the 100 frozen-holdout articles × 5 styles × 2 systems (adapter, no-adapter baseline), using the exact production prompt (`work/tasks/style.py::prompt_style()`). Resumable via a JSONL checkpoint. |
| 2 | `build_outputs_from_run_a.py` | Builds the final `style_frozen_adapter_outputs.json` / `style_frozen_baseline_outputs.json` from the chosen authoritative generation run ("Run A" — see **Provenance** below). |
| 3 | `score_frozen_outputs.py` | Scores every output (facts, numbers, content similarity, length, style divergence, style signals, verbatim-copy rate) by reusing `scripts/test_style_viva.py`'s existing scoring functions. → `style_frozen_scored_results.json` |
| 4 | `classify_frozen_outputs.py` | **Abandoned approach**: zero-shot LLM style classification (base model, then adapter). Both rejected — see **Style classification** below. |
| 4′ | `classify_frozen_outputs_ngram.py` | **Replacement approach**: character-trigram cosine-similarity classifier. → `style_classification_results.json` |
| 5 | *(prompt audit)* | `prompt_consistency_check.json` — byte-for-byte diff of the training prompt (`scripts/train_style.py`) vs. the serving prompt (`work/tasks/style.py`) for all 5 styles. |
| 6 | `combine_final_evaluation.py` | Merges the original single-article viva case study + the frozen-holdout scores + the classification results + the prompt audit into one file. → `viva_style_evaluation_combined.json` |

## Result files in this folder

| File | What it is |
|---|---|
| `viva_style_evaluation_combined.json` | **The combined final report** — original viva case study + frozen-holdout scoring + classification + prompt audit + human-eval status, all in one place. Start here. |
| `style_frozen_scored_results.json` | Step 3 output: per-style, per-system (adapter/baseline) averaged scores over 100 articles. |
| `style_classification_results.json` | Step 4′ output: n-gram classifier accuracy/precision/recall/F1, adapter vs. baseline. |
| `prompt_consistency_check.json` | Step 5 output: training-vs-serving prompt diff, all 5 styles. |
| `style_frozen_holdout_v1.json` | This session's own 100-article holdout draw (665,887-article corpus, seed 42) — **built but not the one scoring used**; see Provenance. |
| `style_frozen_adapter_outputs.json` / `style_frozen_baseline_outputs.json` | Raw generations (100 articles × 5 styles each) for the adapter and the no-adapter baseline — the inputs to Step 3/4′. |
| `style_frozen_generations_checkpoint.jsonl` | Resumable checkpoint written during Step 1/generation. |
| `style_classification_checkpoint.jsonl` | Resumable checkpoint written during classification. |
| `generation_progress.txt` / `generation_stdout.log` | Run log for the generation step (timing: ~1406s per style per system). |
| `classification_progress.txt` / `classification_stdout.log` | Run log for the classification step. |
| `generate_frozen_outputs.py`, `build_outputs_from_run_a.py`, `score_frozen_outputs.py`, `classify_frozen_outputs.py`, `classify_frozen_outputs_ngram.py`, `combine_final_evaluation.py` | Pipeline scripts (see table above). |

### Provenance note: two holdout sets exist

`style_frozen_holdout_v1.json` (this folder) and `frozen_holdout_articles.jsonl`
(under `work/sinllama/data/style_frozen_holdout/`, "Run A") are **two
independently-built, non-identical 100-article draws**, both seed=42, both
excluding the same 11,867 URLs already used anywhere in style
training/candidate generation:

- `style_frozen_holdout_v1.json` — sampled from the 665,887-article corpus
  (`summarizer/all_articles_merged.json`).
- Run A's holdout — sampled from `style_rewriter/data/train1.jsonl`
  (521,980 rows), which was verified to be a 100% subset of that same
  665,887-article corpus (every URL confirmed present).

**Run A is the authoritative run actually used for all scoring below** — its
prompt builder was diffed byte-for-byte against the verified-correct
production prompt (`work/tasks/style.py::prompt_style()`) and matched 5/5
styles. A second attempt at generation in this session (started 13:56, using
`style_frozen_holdout_v1.json`) was stopped at 200/1000 records once Run A
was found valid and complete (1000/1000); that partial run was discarded.
`style_frozen_holdout_v1.json` is kept in this folder as a record of that
original draw, not because it was used.

---

## Results

### 1. Frozen-holdout scoring — adapter vs. no-adapter baseline (N=100 articles/style)

Source: `style_frozen_scored_results.json` / `viva_style_evaluation_combined.json` → `frozen_holdout_evaluation`.
All values are percent (0–100) except `verbatim_copy_rate_pct`, which was 0.0 everywhere (no output was a verbatim copy of its source).

| Style | System | Critical fact pres. | Full fact pres. | Number pres. | Content sim. | Length score | Style divergence | Style signal score |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| **Formal** | adapter | 85.36 | 84.09 | 94.26 | 91.08 | 86.71 | 26.38 | 19.86 |
| | baseline | 17.50 | 17.83 | 31.02 | 32.51 | 3.92 | 94.44 | 24.86 |
| **Editorial** | adapter | 74.91 | 74.46 | 90.61 | 82.57 | 52.05 | 47.32 | 23.33 |
| | baseline | 16.69 | 17.91 | 30.42 | 29.58 | 2.12 | 94.59 | 6.00 |
| **Sports** | adapter | 83.47 | 82.92 | 93.16 | 90.53 | 78.21 | 31.37 | 7.00 |
| | baseline | 18.05 | 19.12 | 28.29 | 32.86 | 3.24 | 94.60 | 8.14 |
| **Youth** | adapter | 76.29 | 73.89 | 94.51 | 82.35 | 75.49 | 41.67 | 18.00 |
| | baseline | 17.24 | 17.98 | 29.22 | 31.46 | 3.91 | 94.52 | 19.17 |
| **Feature** | adapter | 73.48 | 72.83 | 91.90 | 81.19 | 47.50 | 53.87 | 3.57 |
| | baseline | 17.77 | 18.99 | 27.62 | 33.99 | 3.21 | 94.89 | 4.29 |

**Read:** the adapter beats the no-adapter baseline by a wide margin on
every fact/content/length metric, in every style — e.g. critical-fact
preservation 73–85% (adapter) vs. 17–18% (baseline); length score 48–87%
(adapter) vs. 2–4% (baseline). Style divergence (lower = more different from
the plain-formal source) is also far lower for the adapter (26–54%) than the
baseline (94%+), i.e. the baseline barely writes in any target style at all.

**Anomaly, flagged for manual review in the source data:** on
`diagnostic_style_signal_score` only (surface keyword/phrase hits specific to
each style), the **baseline scores higher than the adapter** in 4/5 styles
(formal, sports, youth, feature — editorial is the one exception, where the
adapter wins 23.33 vs 6.00). This one metric is a coarse keyword-hit count,
not the fact/content/length metrics above, and is called out explicitly as
worth checking rather than silently reported as a win.

### 2. Style classification — adapter vs. baseline (N=500 outputs/system, 100/style)

Source: `style_classification_results.json`. **Method**: character-trigram
cosine-similarity to 5 style reference profiles (25 real accepted training
examples per style) — not an LLM judge (two zero-shot LLM classifier attempts
were tried first and rejected; see **Rejected approaches** below). Treat as a
weak, disclosed relative signal, not ground truth.

| System | Accuracy | Precision | Recall | F1 |
|---|--:|--:|--:|--:|
| **Adapter** | **65.4%** | 72.17% | 65.4% | 66.19% |
| Baseline (no adapter) | 23.8% | 25.33% | 23.8% | 22.0% |

Per-style (adapter / baseline), F1:

| Style | Adapter F1 | Baseline F1 | Adapter precision | Adapter recall |
|---|--:|--:|--:|--:|
| Formal | 61.99 | 25.10 | 49.12 | 84.00 |
| Editorial | 73.42 | 17.78 | 100.00 | 58.00 |
| Sports | 50.69 | 33.10 | 47.01 | 55.00 |
| Youth | 92.47 | 17.33 | 100.00 | 86.00 |
| Feature | 52.38 | 16.67 | 64.71 | 44.00 |

The adapter is correctly classified into its intended style ~2.7× more often
than the un-adapted base model (65.4% vs 23.8% accuracy) — its rewrites are
substantially more style-distinguishable, with youth (92.47 F1) and editorial
(73.42 F1) the strongest, sports (50.69 F1) and feature (52.38 F1) the
weakest.

**Rejected classification approaches (tested, not just assumed to fail):**
1. Zero-shot classification with the plain base model — 83–87% of 1000
   completions ignored the instruction and kept generating article text
   instead of a label.
2. Zero-shot classification with the `style_sinllama_v13` adapter attached
   (fine-tuned for rewriting, not classifying) — pretested on 15 examples:
   13/15 gave a parseable label but only 3/15 (20%) were correct,
   indistinguishable from random guessing over 5 classes; it defaulted
   toward "FORMAL" or echoed the label list.

### 3. Prompt consistency audit

Source: `prompt_consistency_check.json`. Compared the fully rendered prompt
(instruction + 16-point FACT PRESERVATION RULES block + scaffold) from the
canonical training source (`scripts/train_style.py::format_prompt()`)
against the serving source (`work/tasks/style.py::prompt_style()`, used by
`serve_sinai.py`), for all 5 styles.

**Result: `all_match: true`** — training and serving prompts are
byte-for-byte identical for formal, editorial, sports, youth, and feature.
(A prior mismatch existed for `style_sinllama_v12` per `work/tasks/style.py`'s
docstring; this audit re-confirms that fix still holds for v13, with no
further code change needed.)

### 4. Single-article viva case study (adapter only, illustrative — not a benchmark)

Source: `viva_style_evaluation_combined.json` → `results` (original content,
unchanged, copied from `models/adapters/style_sinllama_v13/viva_style_evaluation.json`).
One fixed source article (vehicle-import tax policy), rewritten into all 5
styles by the adapter:

| Style | Critical fact pres. | Full fact pres. | Content sim. | Length score | Style divergence | Style signal score |
|---|--:|--:|--:|--:|--:|--:|
| Formal news | 100.0 | 100.0 | 96.60 | 98.61 | 28.86 | 71.43 |
| Editorial | 100.0 | 89.47 | 92.51 | 67.97 | 13.86 | 66.67 |
| Sports | 100.0 | 89.47 | 93.89 | 89.42 | 32.02 | 14.29 |
| Youth | 100.0 | 94.74 | 78.14 | 74.93 | 64.22 | 33.33 |
| Feature | 87.5 | 89.47 | 89.87 | 61.28 | 45.52 | 14.29 |

No verbatim copying in any style; no human reference available for this
article, so ROUGE is null throughout.

### 5. Human evaluation

**Status: not completed.** Per `viva_style_evaluation_combined.json` →
`human_evaluation`: a rating sheet (style adherence / meaning preservation /
factual correctness / Sinhala fluency / overall usefulness, 1–5 each)
already exists as a pattern in `scripts/test_style_viva.py`'s
`human_evaluation_sheet()`, but it has not yet been run against the frozen
holdout outputs. This is the one remaining manual step in the pipeline.

---

## Bottom line

On a 100-article, article-level frozen holdout (no train/test leakage),
`style_sinllama_v13` clearly outperforms the plain base model at style
rewriting: 4–5× higher fact preservation, ~20–25× higher length-band
adherence, far lower style divergence from the source, and ~2.7× higher
style-classification accuracy. The training and serving prompts are
confirmed identical, so there's no prompt-drift explanation for the gap.
The one open item is human evaluation (rating sheet exists, not yet run),
and one metric (`diagnostic_style_signal_score`) is flagged as an anomaly
worth a closer look rather than being papered over.
