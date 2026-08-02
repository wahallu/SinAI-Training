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
python work/sinllama/scripts/train_style.py
python work/sinllama/scripts/train_summarizer.py
```

### Evaluation / Testing
```bash
python work/sinllama/scripts/test_grammar.py
python work/sinllama/scripts/test_headline.py         # v17 only: fixed 4-7 word band, not length-aware
python work/sinllama/scripts/test_headline_v18.py     # v18: per-band in-band rate + artifact rate + own-band ROUGE
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
config change needed). A 9-sample smoke test against the live server (3
articles x 3 bands, identical prompts on both adapters) showed v18 landing
8/9 in-band vs. v17's 6/9, with the long band moving from 2/3 to 3/3 and fewer
trailing scraper artifacts (1/9 vs. 2/9). For a real read on in-band rate at
scale, run `scripts/test_headline_v18.py` (see Evaluation / Testing above) on
the GPU box — it generates each val article once per band and reports in-band
rate, artifact rate, and own-band ROUGE, rather than the single fixed-band
score `test_headline.py` gives.

## Summarizer

`summarizer/` holds two things: an early mT5-base pipeline (predates the SinLlama approach, now used as a comparison "teacher" model in `serve_sinai.py`) and the active SinLLaMA summarizer pipeline (`summarizer/abstractive/`, adapters v02–v06), which trains the same `summarization_sinllama_v*` adapters served by `work/serve_sinai.py`. Despite the folder name, `summarizer/abstractive/` is not legacy — it's the current summarizer training+eval code. Dependencies are in `summarizer/requirements.txt`.

**Length conditioning (v06+):** `work/tasks/summarizer.py`'s `prompt_summarizer()` accepts a `length` param (`short`/`medium`/`long`) that selects v06's length-conditioned prompt; omitting it builds the legacy v02-v05 fixed-instruction prompt instead. `work/serve_sinai.py` auto-defaults `length` to `"medium"` per-request when the resolved adapter's version is >= `SUMMARIZER_LENGTH_CONDITIONED_FROM_VERSION` (currently 6.0, see `run_generation()`), so callers that don't send `length` still get a matching prompt for v06+, and v02-v05 behavior is unaffected. `/compare` checks this per adapter in the comparison set, so mixed v02-v05/v06+ comparisons get the correct prompt for each. Update `SUMMARIZER_LENGTH_CONDITIONED_FROM_VERSION` if a future summarizer version changes the prompt format again.
