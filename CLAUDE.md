# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This workspace develops **SinLlama** — a Sinhala-specialized LLM built on Meta Llama-3-8B, fine-tuned for four NLP tasks: grammar correction, headline generation, article summarization, and style rewriting. The inference API is served via FastAPI.

All core work lives in `work/sinllama/`. A separate early-stage summarizer pipeline is in `summarizer/`.

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
python work/sinllama/scripts/train_headline.py
python work/sinllama/scripts/train_style.py
python work/sinllama/scripts/train_summarizer.py
```

### Evaluation / Testing
```bash
python work/sinllama/scripts/test_grammar.py
python work/sinllama/scripts/test_headline.py
python work/sinllama/scripts/test_style.py
python work/sinllama/scripts/test_summarizer.py
```

### Inference server
```bash
# Starts FastAPI on default port; loads SinLLaMA-merged-base plus all four
# task adapters at startup
uvicorn work.sinllama.serve_sinai:app --host 0.0.0.0 --port 8000

# API endpoints
POST /generate   # body: {"prompt": "...", "task": "grammar|headline|summarizer|style", "style": "formal|sports|youth|editorial|feature"}
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

Each NLP task has its own LoRA adapter stored under `models/adapters/`:
```
models/
├── SinLLaMA-merged-base/    ← shared base for all tasks
├── adapters/
│   ├── grammar_sinllama_v13/
│   ├── headline_sinllama_v17/
│   ├── style_sinllama_v07/
│   └── summarization_sinllama_v04/
```

`serve_sinai.py` loads `SinLLaMA-merged-base` plus all four task adapters as
named PEFT adapters (adapter name == task name), switching between them per
request via `model.set_adapter(task)` — see `ADAPTER_PATHS` at the top of the
file. Training/test scripts instead load the base and attach a single
adapter via `PeftModel.from_pretrained(model, ADAPTER_PATH)`.

### Prompt format (Alpaca-style)

All tasks use the same template. The `### Input:` block is task-dependent:
```
### Instruction:
{sinhala instruction text}

### Input:  (or Article: / Text:)
{content}

### Response:
```

`serve_sinai.py` auto-detects pre-formed prompts by checking for `"### Instruction:"` in the request body; raw text is wrapped by task-specific `prompt_*` builders.

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

## Summarizer (separate pipeline)

`summarizer/` is an independent earlier-stage project using mT5 fine-tuned for Sinhala summarization, predating the SinLlama approach. Dependencies are in `summarizer/requirements.txt`. It is not integrated with the `work/sinllama/` serving stack.
