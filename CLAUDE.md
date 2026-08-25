# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This workspace develops **SinLlama** — a Sinhala-specialized LLM built on Meta Llama-3-8B, fine-tuned for four NLP tasks: grammar correction, headline generation, article summarization, and style rewriting. The inference API is served via FastAPI.

All core work lives in `work/sinllama/`. The summarizer's active training+eval pipeline is in `summarizer/abstractive/` (see "Summarizer" section below — despite the folder name, this is not a separate/legacy system).

---

## Key Commands

### First-time model setup (run once)
```bash
# Download base model, SinLlama adapter, and extended tokenizer from HuggingFace
cd work/sinllama && python download_model.py

# Merge base + SinLlama_v01 adapter into a single reusable model
python work/sinllama/prepare_sinllama_base.py
```

### Training (run from repo root or `work/sinllama/`)
```bash
python work/sinllama/scripts/train_grammar.py
python work/sinllama/scripts/train_headline.py       # v17, fixed "4-7 words" prompt
python work/sinllama/scripts/train_headline_v18.py   # v18, length-conditioned (see Headline section)
python work/sinllama/scripts/clean_headline_dataset.py  # run once before v19: strips scraper tags from references
python work/sinllama/scripts/train_headline_v19.py   # v19, length-conditioned + artifact-cleaned data
python work/sinllama/scripts/clean_headline_dataset_v20.py  # run once before v20: wider tags + cleans article INPUT too
python work/sinllama/scripts/train_headline_v20.py   # v20, cleans both sides of the training pair (see Headline section)
python work/sinllama/scripts/train_style.py
python work/sinllama/scripts/train_summarizer.py
```

### Evaluation / Testing
```bash
python work/sinllama/scripts/test_grammar.py
python work/sinllama/scripts/test_headline.py         # v17 only: fixed 4-7 word band, not length-aware
python work/sinllama/scripts/test_headline_v18.py     # v18: per-band in-band rate + artifact rate + own-band ROUGE
python work/sinllama/scripts/test_headline_v19.py     # v19: same, against the artifact-cleaned adapter
python work/sinllama/scripts/test_headline_v20.py     # v20: same, against the input+output-cleaned adapter
python work/sinllama/scripts/test_style.py
python summarizer/abstractive/6_test_summarizer.py   # latest (v06, length-conditioned); {2,3,4,5}_test_summarizer.py for earlier adapters
```

### Inference server
```bash
# serve_sinai.py lives at work/serve_sinai.py and uses bare imports
# (`from task_registry import ...`, `from tasks.summarizer import ...`),
# so it must be run with work/ as the working directory, not repo root.
cd work && uvicorn serve_sinai:app --host 0.0.0.0 --port 8000

# API endpoints
POST /generate   # body: {"prompt": "...", "task": "grammar|headline|summarizer|style", "style": "formal|sports|youth|editorial|feature", "length": "short|medium|long"}
                 # length applies to summarizer (compression band) and headline (word band)
GET  /tasks      # lists all tasks, styles, and example prompts
GET  /health
```

### Interactive terminal inference
```bash
python work/sinllama/sinllama_run.py
```

---

## Architecture

### Model loading chain

There are **two different model-loading patterns** in this codebase — understanding the difference is critical:

**Legacy (sinllama_run.py):** loads the raw Llama-3-8B + SinLlama_v01 LoRA adapter at runtime, resizes embeddings on CPU before moving to CUDA, then merges. Used for interactive inference only.

**Current (all train/test scripts and serve_sinai.py):** loads `SinLLaMA-merged-base`, a pre-merged snapshot produced by `prepare_sinllama_base.py`. All task-specific LoRA adapters layer on top of this merged base. This eliminates the 4-step chain at runtime.

### Task adapter structure

Each NLP task has its own LoRA adapter stored under `models/adapters/`. `serve_sinai.py`'s `find_latest_adapters()` auto-selects the highest-numbered folder present per task at startup (e.g. multiple `summarization_sinllama_v0N/` folders can coexist; the latest wins unless none are found, in which case it falls back to a hardcoded path):
```
models/
├── SinLLaMA-merged-base/    ← shared base for all tasks
├── adapters/
│   ├── grammar_sinllama_v13/
│   ├── headline_sinllama_v17/
│   ├── style_sinllama_v07/
│   └── summarization_sinllama_v02/ ... v06/  (see Summarizer section)
```

`serve_sinai.py` loads `SinLLaMA-merged-base` plus all four task adapters as
named PEFT adapters (adapter name == task name), switching between them per
request via `model.set_adapter(task)` — see `ADAPTER_PATHS` at the top of the
file. Training/test scripts instead load the base and attach a single
adapter via `PeftModel.from_pretrained(model, ADAPTER_PATH)`.

### Prompt format (Alpaca-style, with one exception)

Grammar, headline, style, and base use the same Alpaca-style template. The `### Input:` block is task-dependent:
```
### Instruction:
{sinhala instruction text}

### Input:  (or Article: / Text:)
{content}

### Response:
```

**Exception: the summarizer task uses a Llama-3 chat template** (`<|begin_of_text|><|start_header_id|>...`), not Alpaca — see `work/tasks/summarizer.py`. Do not assume the two are interchangeable.

`serve_sinai.py` auto-detects pre-formed Alpaca prompts by checking for `"### Instruction:"` in the request body; raw text is wrapped by task-specific `prompt_*` builders.

### Dataset naming convention

Datasets follow a stage-based progression: `*_stage1.jsonl`, `*_stage2.jsonl`, etc. The highest-numbered stage is the current training set. Manual datasets (`grammar_manual_dataset_stage*.jsonl`) are hand-curated; auto-generated ones have no `manual_` prefix.

---

## Critical Constraints

- **Always `load_in_4bit=True`** — prevents OOM on standard GPU hardware.
- **`lora_dropout=0.05` minimum** (not 0.0) — setting dropout to 0.0 activates Unsloth's fast CUDA LoRA kernel, which breaks on 4-bit quantized weights due to dtype mismatches in backward passes.
- **Embedding resize must happen on CPU** — `model.to("cpu")` before `resize_token_embeddings()`, then back to CUDA (only applies to the legacy chain in `sinllama_run.py`; the merged base already has correct dimensions).
- **`NCCL_P2P_DISABLE=1` and `NCCL_IB_DISABLE=1`** are set at the top of `serve_sinai.py` — required for single-GPU environments.
- **`rouge_score` library breaks on Sinhala Unicode** — `test_grammar.py` implements ROUGE natively on grapheme clusters. Do not replace with the standard library.

---

## Headline

**Length bands (short 3-5 / medium 6-7 / long 8-10 words).** Non-overlapping,
so a word count maps to exactly one band. Defined in three places that must
stay in sync: `work/tasks/headline.py` (`HEADLINE_LENGTHS`, token budgets),
`SinhalaJournalLLM/apps/backend-api/app/core/prompts.py` (the prompt the live
web app actually sends), and `scripts/train_headline_v18.py` (training).

`prompt_headline()` takes `length`; passing `None` reproduces the
pre-length-control prompt and the 60/5 token budget byte-for-byte, and nothing
defaults it on the caller's behalf — so `/compare` and any raw-text client are
unaffected unless they ask for a band. When `length` is set, the band drives
both the prompt line and per-band `max_new_tokens` / `min_new_tokens` (via
`TaskSpec.length_aware` and `TaskSpec.min_new_tokens_for`, which replaced the
old task-name branching in `run_generation()`).

**v17 is not length-conditioned.** All 48K of its training examples carried the
same fixed "Between 4 and 7 words" line, so the band line is a nudge it only
partly obeys. Measured against v17 on the live server: short 4/4 and medium 5/5
land in-band, long 1/5 — it stops at ~7 words regardless of the instruction.
Two things address that:

1. `min_new_tokens` per band (13 for long vs. the flat 5 before) blocks EOS
   until the band's lower edge is plausible. Sized from a measured
   tokens-per-word ratio on real v17 output (min 1.20, median 1.71, p90 2.67 —
   the spread is why no token floor can *guarantee* a word count).
2. `train_headline_v18.py` buckets the dataset by each reference headline's
   actual word count and trains the band line as a real condition. This is the
   headline equivalent of the summarizer's v02-v05 → v06 move below.

The hard guarantee on the upper bound lives outside this repo, in
backend-api's `headline_service.py`: out-of-band candidates are regenerated
with a corrective hint (2 rounds), then anything still over the ceiling is
trimmed to it.

**v18 is trained and live** (auto-selected by `find_latest_adapters()`, no
config change needed). The real read on it: `scripts/test_headline_v18.py` run
at N=300 articles x 3 bands (900 generations total) measured:

| band   | in-band       | artifact rate |
|--------|---------------|---------------|
| short  | 266/300 88.7% | 1/300   0.3%  |
| medium | 228/300 76.0% | 33/300  11.0% |
| long   | 234/300 78.0% | 67/300  22.3% |
| **all**| 728/900 80.9% | 101/900 11.2% |

Length conditioning works — 81% in-band on bands the model had zero training
signal for before v18 is a real result. But artifact rate scales with band
length: nearly 1 in 4 long headlines carries a trailing scraper tag. Root
cause is the training data, not generation — a meaningful share of the 48K
references still carry their original Hiru/ITN scrape tags
(`(වීඩියෝ)` / `VIDEO` / `PHOTOS` / `Interview`), those tags are extra words,
so tagged references cluster into exactly the medium/long buckets v18 was
asked to fill.

**v19 fixes the data, not the model.** `scripts/clean_headline_dataset.py`
strips trailing artifact tags from every reference headline (regex-based,
anchored to the end of the string so mid-sentence mentions are untouched),
writing `headline_dataset_48k_balanced_{train,val}_clean.jsonl`. Word bands
are recomputed from the cleaned text, so a headline that shrinks from "long"
to "medium" once its tag is gone lands in the band it actually belongs to.
`train_headline_v19.py` is byte-identical to v18 except it points at the
cleaned files — data cleanliness is the only variable. Run
`clean_headline_dataset.py` once before training v19, then
`test_headline_v19.py` after, to confirm artifact rate drops without in-band
rate regressing (if it regresses, the cleaning step over-stripped real
content, not just tags).

**v19 measured (N=300, run against the cleaned val set):**

| band   | in-band (v18 → v19)   | artifact (v18 → v19) |
|--------|------------------------|------------------------|
| short  | 88.7% → 89.7%          | 0.3% → 0.0%            |
| medium | 76.0% → 74.3%          | 11.0% → 0.3%           |
| long   | 78.0% → 75.0%          | 22.3% → 3.0%           |
| **all**| 80.9% → 79.7%          | 11.2% → 1.1%           |

Artifact rate dropped ~10x (11.2% → 1.1%) with in-band rate essentially flat
(within noise for N=300) — data cleaning fixed the leak without costing
length conditioning. ROUGE-1/L also ticked up (0.124→0.134 / 0.121→0.130)
since references are no longer mismatched by trailing junk. v19 was deployed
(`find_latest_adapters()` rescans on restart, highest version wins).

**v19 wasn't zero, and live-server testing on fresh articles surfaced why.**
1.1% overall wasn't the floor — reports came in of tags still showing up
occasionally, e.g. `ඇපල් AppStore හි යළිත් වැඩ පෙන්වන්නේ Telegram ය! (වීඩියෝ)`,
plus new variants v19's word list didn't cover: `[video]`, `[photo]`, and a
bare `- photo` with no brackets at all. Two root causes:

1. `clean_headline_dataset.py`'s separator class (`[\s\-–—:()!]*`) didn't
   include square brackets, so `[video]`-style tags survived cleaning.
2. It only cleaned the reference **headline** (`output`), never the article
   (`input`). Scraped articles routinely carry an inline tag right next to
   the sentence a headline gets built from — `...Telegram ය. (Video) මේ
   පිළිබඳව...` — and the model can copy that straight into a generated
   headline regardless of how clean the training label was, because the tag
   is sitting in its *input* context, not something it memorized from a
   label.

**Two-track fix, shipped separately because they have very different
lead times:**

- **Immediate, retrain-independent (already deployed):** backend-api gained
  `app/core/text_cleaning.py` with `strip_article_media_tags()` (strips
  inline tags from the article before it's ever put in a prompt — wired into
  `prompts.py`'s `prompt_headline()`) and `strip_headline_artifacts()`
  (strips a trailing tag from every generated candidate before the
  word-count/band logic runs — wired into `headline_service.py`, so it's the
  actual hard guarantee that no tag reaches a caller, independent of adapter
  version). This took effect the moment backend-api redeployed; no GPU-box
  change needed.
- **Root-cause, needs a retrain:** `scripts/clean_headline_dataset_v20.py`
  widens the artifact word list/separator class (square brackets, bare
  dash-prefixed words) **and** cleans the article `input` field too —
  stripping inline tags anywhere in the body, not just a trailing tag on the
  headline. `train_headline_v20.py` trains on the result
  (`headline_dataset_48k_balanced_{train,val}_clean_v20.jsonl`); everything
  else is unchanged from v19 so cleaning breadth is the only variable. Run
  `test_headline_v20.py` after and compare against v19's measured baseline
  (in its docstring) — artifact rate should drop further, in-band rate
  should hold roughly steady.

Note `work/tasks/headline.py`'s raw-text path (used by `/compare` and
`/tasks`) does **not** get the backend-api cleanup — it's a dev/testing
surface, not the production web app. A `/compare` result can show a tag the
real product would have already stripped; don't take it as representative of
what a user sees.

**v20 was trained and measured — not a clear win, not deployed.** N=300
result:

| band   | in-band (v19 → v20)   | artifact (v19 → v20) |
|--------|------------------------|------------------------|
| short  | 89.7% → 84.7% (↓)      | 0.0% → 0.3%            |
| medium | 74.3% → 75.3%          | 0.3% → 1.0%            |
| long   | 75.0% → 79.7% (↑)      | 3.0% → 2.3%            |
| **all**| 79.7% → 79.9%          | 1.1% → **1.2%** (↑)    |

Artifact rate did not drop — the opposite of what v20's article-input
cleaning was meant to fix — and the short band regressed 5pp. Root cause of
the flat artifact number: `test_headline_v20.py` evaluates against
`headline_dataset_48k_balanced_val_clean_v20.jsonl`, the *already-cleaned*
validation set, so the eval has nothing left in the article for the model to
copy a tag from — it can't measure what v20's input-cleaning was actually
built to test. The ~1-2% residual on both v19 and v20 here is the model
occasionally producing a tag-like pattern from general newswriting-style
exposure, not from copying a visible tag, and retraining on cleaner input
doesn't touch that. The short-band regression is more likely N=300 sampling
variance (`do_sample=True`) than a real effect.

**Decision: v19 stays deployed.** v20 isn't a measured improvement and
regressed the short band for no offsetting gain. If this gets revisited, the
eval needs redoing against a validation set that still has real, dirty
(uncleaned) article text — otherwise the comparison structurally can't see
the effect being tested for. In the meantime, backend-api's
`app/core/text_cleaning.py` (`strip_article_media_tags()` +
`strip_headline_artifacts()`) is already deployed and is a hard guarantee
independent of adapter version — that's what's actually keeping tags out of
what users see today.

## Summarizer

`summarizer/` holds two things: an early mT5-base pipeline (predates the SinLlama approach, now used as a comparison "teacher" model in `serve_sinai.py`) and the active SinLLaMA summarizer pipeline (`summarizer/abstractive/`, adapters v02–v07), which trains the same `summarization_sinllama_v*` adapters served by `work/serve_sinai.py`. Despite the folder name, `summarizer/abstractive/` is not legacy — it's the current summarizer training+eval code. Dependencies are in `summarizer/requirements.txt`.

**v07 status (as of 2026-08-22):** v07 (`abstractive/7_train_summarizer.py`, trained on the 35,547-row cleaned corpus) is fully trained and has a tracked eval (`6_eval_results/v07_eval_20260811_094434.json`) — it is not experimental/unfinished. `find_latest_adapters()` in `serve_sinai.py` auto-serves it in production today ahead of v06, purely because it version-sorts higher, independent of quality. On the one existing comparable evaluation (same 45-output protocol as v06's), v07 currently scores *worse* than v06 on ROUGE-L across all three length bands. **Don't trust that comparison** — a split-contamination audit (`abstractive/8_audit_split_contamination.py`) found that both `6_train_summarizer.py` and `7_train_summarizer.py` split their flat short/medium/long sample list at the sample level rather than the article level, and as a result ~81% of a newly built article-level frozen test set (`data/summarization_frozen_test.jsonl`, seed 42, manifest at `data/summarization_frozen_split_manifest.json`) was already present in *each* adapter's original training data. A leakage-controlled re-evaluation on the frozen split is planned but not yet complete; see `summarizer/SUMMARIZATION_NEXT_STEPS.md` for the full status and phased plan.

**Length conditioning (v06+):** `work/tasks/summarizer.py`'s `prompt_summarizer()` accepts a `length` param (`short`/`medium`/`long`) that selects v06's length-conditioned prompt; omitting it builds the legacy v02-v05 fixed-instruction prompt instead. `work/serve_sinai.py` auto-defaults `length` to `"medium"` per-request when the resolved adapter's version is >= `SUMMARIZER_LENGTH_CONDITIONED_FROM_VERSION` (currently 6.0, see `run_generation()`), so callers that don't send `length` still get a matching prompt for v06+, and v02-v05 behavior is unaffected. `/compare` checks this per adapter in the comparison set, so mixed v02-v05/v06+ comparisons get the correct prompt for each. Update `SUMMARIZER_LENGTH_CONDITIONED_FROM_VERSION` if a future summarizer version changes the prompt format again.
