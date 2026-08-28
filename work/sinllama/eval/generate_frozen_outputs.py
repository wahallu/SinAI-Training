#!/usr/bin/env python3
"""
Style Rewriter frozen-holdout evaluation -- Step 2 (generation).

Generates rewrites for all 100 frozen-holdout articles (see
style_frozen_holdout_v1.json, sampled from the full 665,887-article
corpus, zero overlap with anything used in style training/candidate
generation) x 5 styles, for two systems:

  - "baseline": plain SinLLaMA-merged-base, NO adapter attached
  - "adapter" : SinLLaMA-merged-base + style_sinllama_v13 LoRA adapter

Both systems use the IDENTICAL prompt -- work/tasks/style.py's
prompt_style(), i.e. the exact prompt shape used by production serving
(serve_sinai.py), confirmed byte-for-byte identical to the training
prompt in train_style.py (see prompt_consistency_check.json).

Checkpoints every generated record to a resumable JSONL file so a
crash/interruption doesn't lose completed work, then emits the two
required output files:
  style_frozen_adapter_outputs.json
  style_frozen_baseline_outputs.json
"""
import os
import re
import sys
import json
import time
import random
from pathlib import Path

import torch
from transformers import LlamaTokenizerFast, AutoModelForCausalLM
from peft import PeftModel

sys.path.insert(0, "/home/jovyan/work")
from tasks.style import prompt_style, STYLE_ID_MAP  # canonical serving prompt

EVAL_DIR = Path("/home/jovyan/work/sinllama/eval")
HOLDOUT_PATH = EVAL_DIR / "style_frozen_holdout_v1.json"
CHECKPOINT_PATH = EVAL_DIR / "style_frozen_generations_checkpoint.jsonl"
PROGRESS_PATH = EVAL_DIR / "generation_progress.txt"
ADAPTER_OUT_PATH = EVAL_DIR / "style_frozen_adapter_outputs.json"
BASELINE_OUT_PATH = EVAL_DIR / "style_frozen_baseline_outputs.json"

BASE_MODEL = "/home/jovyan/work/sinllama/models/SinLLaMA-merged-base"
ADAPTER_PATH = "/home/jovyan/work/sinllama/models/adapters/style_sinllama_v13"

MAX_SEQ_LENGTH = 4096
GEN_MAX_NEW_TOKENS = 450
TEMPERATURE = 0.10
TOP_P = 0.85
TOP_K = 50
REPETITION_PENALTY = 1.05
SEED = 42
BATCH_SIZE = 6  # conservative: shares the GPU with other work on this box

STYLES = list(STYLE_ID_MAP.keys())  # formal, editorial, sports, youth, feature


def set_seed(seed=SEED):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def log_progress(msg):
    print(msg, flush=True)
    with open(PROGRESS_PATH, "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")


def load_tokenizer():
    tokenizer = LlamaTokenizerFast.from_pretrained(BASE_MODEL, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_base_model(tokenizer):
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto",
        local_files_only=True, low_cpu_mem_usage=True,
    )
    input_embeddings = model.get_input_embeddings()
    cur = input_embeddings.weight.shape[0]
    tok_size = len(tokenizer)
    if cur != tok_size:
        new_emb = torch.nn.Embedding(
            tok_size, input_embeddings.weight.shape[1],
            dtype=input_embeddings.weight.dtype,
            device=input_embeddings.weight.device,
        )
        new_emb.weight.data[:cur] = input_embeddings.weight.data
        model.set_input_embeddings(new_emb)
    model.eval()
    return model


@torch.inference_mode()
def generate_batch(model, tokenizer, prompts):
    inputs = tokenizer(
        prompts, return_tensors="pt", truncation=True,
        max_length=MAX_SEQ_LENGTH - GEN_MAX_NEW_TOKENS, padding=True,
    )
    device = next(model.parameters()).device
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    output_ids = model.generate(
        input_ids=input_ids, attention_mask=attention_mask,
        max_new_tokens=GEN_MAX_NEW_TOKENS, do_sample=True,
        temperature=TEMPERATURE, top_p=TOP_P, top_k=TOP_K,
        repetition_penalty=REPETITION_PENALTY,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id, use_cache=True,
    )

    outputs = []
    prompt_len = input_ids.shape[1]
    for i in range(output_ids.shape[0]):
        text = tokenizer.decode(output_ids[i, prompt_len:], skip_special_tokens=True).strip()
        text = re.sub(r"^###\s*Response\s*:\s*", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"^###\s*Response\s*", "", text, flags=re.IGNORECASE).strip()
        outputs.append(text)
    return outputs


def load_checkpoint_done():
    done = set()
    if not CHECKPOINT_PATH.exists():
        return done
    with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            done.add((rec["system"], rec["style"], rec["article_id"]))
    return done


def run_system(system_name, model, tokenizer, articles, done, ckpt_file):
    for style in STYLES:
        pending = [a for a in articles if (system_name, style, a["article_id"]) not in done]
        if not pending:
            log_progress(f"[{system_name}/{style}] all {len(articles)} already done, skipping")
            continue

        pending_sorted = sorted(pending, key=lambda a: len(a["article_text"]))
        log_progress(f"[{system_name}/{style}] generating {len(pending_sorted)} (of {len(articles)}) articles")

        t_style_start = time.time()
        n_done = 0
        for i in range(0, len(pending_sorted), BATCH_SIZE):
            batch = pending_sorted[i:i + BATCH_SIZE]
            prompts = [prompt_style(a["article_text"], style=style) for a in batch]

            t0 = time.time()
            try:
                outputs = generate_batch(model, tokenizer, prompts)
            except Exception as exc:
                log_progress(f"[{system_name}/{style}] batch at {i} FAILED: {exc}")
                outputs = [""] * len(batch)
            dt = time.time() - t0

            for a, out in zip(batch, outputs):
                record = {
                    "system": system_name,
                    "style": style,
                    "article_id": a["article_id"],
                    "category": a.get("category"),
                    "source_publication": a.get("source_publication"),
                    "article_text": a["article_text"],
                    "output": out,
                }
                ckpt_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            ckpt_file.flush()

            n_done += len(batch)
            log_progress(
                f"[{system_name}/{style}] batch {i // BATCH_SIZE + 1}/"
                f"{(len(pending_sorted) + BATCH_SIZE - 1) // BATCH_SIZE} "
                f"({n_done}/{len(pending_sorted)}) took {dt:.1f}s"
            )

        log_progress(f"[{system_name}/{style}] style done in {time.time() - t_style_start:.1f}s")


def build_output_files():
    """Reshape the checkpoint jsonl into the two required output files,
    each keyed by article_id + style."""
    by_system = {"baseline": {}, "adapter": {}}
    with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            sysname = "baseline" if rec["system"] == "no_adapter" else "adapter"
            key = f"{rec['article_id']}::{rec['style']}"
            by_system[sysname][key] = {
                "article_id": rec["article_id"],
                "style": rec["style"],
                "category": rec.get("category"),
                "source_publication": rec.get("source_publication"),
                "output": rec["output"],
            }

    adapter_out = {
        "description": "style_sinllama_v13 adapter outputs on the frozen 100-article holdout, 5 styles each (500 total).",
        "adapter_path": ADAPTER_PATH,
        "base_model": BASE_MODEL,
        "prompt_source": "work/tasks/style.py::prompt_style() (production serving prompt)",
        "generation_config": {
            "max_new_tokens": GEN_MAX_NEW_TOKENS, "temperature": TEMPERATURE,
            "top_p": TOP_P, "top_k": TOP_K, "repetition_penalty": REPETITION_PENALTY,
            "seed": SEED, "do_sample": True,
        },
        "count": len(by_system["adapter"]),
        "generations": by_system["adapter"],
    }
    baseline_out = {
        "description": "Plain SinLLaMA-merged-base with NO style adapter attached, same holdout, same prompts (zero-shot baseline).",
        "base_model": BASE_MODEL,
        "prompt_source": "work/tasks/style.py::prompt_style() (production serving prompt)",
        "generation_config": adapter_out["generation_config"],
        "count": len(by_system["baseline"]),
        "generations": by_system["baseline"],
    }

    with open(ADAPTER_OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(adapter_out, f, ensure_ascii=False, indent=2)
    with open(BASELINE_OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(baseline_out, f, ensure_ascii=False, indent=2)

    log_progress(f"Wrote {ADAPTER_OUT_PATH} ({adapter_out['count']} records)")
    log_progress(f"Wrote {BASELINE_OUT_PATH} ({baseline_out['count']} records)")


def main():
    set_seed(SEED)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    log_progress("=== Frozen holdout GENERATION run starting ===")

    with open(HOLDOUT_PATH, "r", encoding="utf-8") as f:
        holdout = json.load(f)
    articles = holdout["articles"]
    log_progress(f"Loaded {len(articles)} frozen holdout articles (seed={holdout['sampling_seed']})")

    done = load_checkpoint_done()
    log_progress(f"Resuming: {len(done)} (system,style,article_id) records already checkpointed")

    tokenizer = load_tokenizer()
    log_progress("Tokenizer loaded")

    model = load_base_model(tokenizer)
    log_progress("Base model loaded (no adapter)")

    with open(CHECKPOINT_PATH, "a", encoding="utf-8") as ckpt_file:
        run_system("no_adapter", model, tokenizer, articles, done, ckpt_file)
        done = load_checkpoint_done()

        log_progress("Loading LoRA adapter (style_sinllama_v13)...")
        model = PeftModel.from_pretrained(model, ADAPTER_PATH, local_files_only=True, is_trainable=False)
        model.eval()
        log_progress("Adapter loaded")

        run_system("adapter", model, tokenizer, articles, done, ckpt_file)

    build_output_files()
    log_progress("=== Frozen holdout GENERATION run COMPLETE ===")


if __name__ == "__main__":
    main()
