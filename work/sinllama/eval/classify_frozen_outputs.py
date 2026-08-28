#!/usr/bin/env python3
"""
Style Rewriter frozen-holdout evaluation -- Step 4 (style classification).

No trained style classifier exists in this codebase, so this uses a
zero-shot LLM prompt classifier: plain SinLLaMA-merged-base (NO style
adapter attached -- style_sinllama_v13 is a REWRITING adapter, not a
classifier, and attaching it would bias the model toward whatever style
it was fine-tuned to produce rather than judging the text it's given)
is prompted to read a piece of Sinhala text and pick which of the 5
target styles it most resembles.

Classifies all 500 adapter outputs and all 500 baseline outputs from
Step 2, then compares predicted vs. actual target style to compute
accuracy/precision/recall/F1 per style and overall, for both systems.
"""
import re
import sys
import json
import time
from pathlib import Path
from collections import Counter, defaultdict

import torch
from transformers import LlamaTokenizerFast, AutoModelForCausalLM

EVAL_DIR = Path("/home/jovyan/work/sinllama/eval")
ADAPTER_OUT_PATH = EVAL_DIR / "style_frozen_adapter_outputs.json"
BASELINE_OUT_PATH = EVAL_DIR / "style_frozen_baseline_outputs.json"
CHECKPOINT_PATH = EVAL_DIR / "style_classification_checkpoint.jsonl"
PROGRESS_PATH = EVAL_DIR / "classification_progress.txt"
RESULT_PATH = EVAL_DIR / "style_classification_results.json"

BASE_MODEL = "/home/jovyan/work/sinllama/models/SinLLaMA-merged-base"
MAX_SEQ_LENGTH = 4096
GEN_MAX_NEW_TOKENS = 8
BATCH_SIZE = 12
SEED = 42

LABELS = ["FORMAL", "EDITORIAL", "SPORTS", "YOUTH", "FEATURE"]

LABEL_DESCRIPTIONS = {
    "FORMAL": "Formal, objective news-report style -- neutral, factual, no opinion.",
    "EDITORIAL": "Editorial / analytical style -- discusses significance, reflective tone.",
    "SPORTS": "Sports-journalism style -- energetic, focused on actions/events/outcomes.",
    "YOUTH": "Youth / conversational style -- simple, casual, accessible language.",
    "FEATURE": "Feature / narrative style -- descriptive, human-interest storytelling.",
}


def build_classification_prompt(text):
    options = "\n".join(f"- {label}: {desc}" for label, desc in LABEL_DESCRIPTIONS.items())
    return (
        "### Instruction:\n"
        "You are a Sinhala news-writing style classifier. Read the Sinhala "
        "text below and decide which ONE of the following 5 writing styles "
        "it was written in. Answer with EXACTLY ONE WORD from this list, "
        "nothing else:\n"
        f"{options}\n\n"
        "### Input:\n"
        f"{text.strip()}\n\n"
        "### Response:\n"
    )


def parse_label(raw_output):
    upper = raw_output.upper()
    for label in LABELS:
        if label in upper:
            return label
    return "UNKNOWN"


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
def classify_batch(model, tokenizer, texts):
    prompts = [build_classification_prompt(t) for t in texts]
    inputs = tokenizer(
        prompts, return_tensors="pt", truncation=True,
        max_length=MAX_SEQ_LENGTH - GEN_MAX_NEW_TOKENS, padding=True,
    )
    device = next(model.parameters()).device
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    output_ids = model.generate(
        input_ids=input_ids, attention_mask=attention_mask,
        max_new_tokens=GEN_MAX_NEW_TOKENS, do_sample=False,
        eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
        use_cache=True,
    )
    prompt_len = input_ids.shape[1]
    results = []
    for i in range(output_ids.shape[0]):
        raw = tokenizer.decode(output_ids[i, prompt_len:], skip_special_tokens=True).strip()
        results.append(parse_label(raw))
    return results


def load_checkpoint_done():
    done = set()
    if not CHECKPOINT_PATH.exists():
        return done
    with open(CHECKPOINT_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            done.add((rec["system"], rec["key"]))
    return done


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
            "precision": round(precision * 100, 2),
            "recall": round(recall * 100, 2),
            "f1": round(f1 * 100, 2),
            "support": sum(1 for t in y_true if t == label),
        }
    n = len(y_true)
    accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / n * 100 if n else 0.0
    macro_precision = sum(v["precision"] for v in per_label.values()) / len(labels)
    macro_recall = sum(v["recall"] for v in per_label.values()) / len(labels)
    macro_f1 = sum(v["f1"] for v in per_label.values()) / len(labels)
    return {
        "accuracy": round(accuracy, 2),
        "precision": round(macro_precision, 2),
        "recall": round(macro_recall, 2),
        "f1": round(macro_f1, 2),
        "per_style": per_label,
    }


def main():
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    log_progress("=== Style classification run starting ===")

    with open(ADAPTER_OUT_PATH, encoding="utf-8") as f:
        adapter_data = json.load(f)
    with open(BASELINE_OUT_PATH, encoding="utf-8") as f:
        baseline_data = json.load(f)

    STYLE_TO_LABEL = {
        "formal": "FORMAL", "editorial": "EDITORIAL", "sports": "SPORTS",
        "youth": "YOUTH", "feature": "FEATURE",
    }

    items = []  # (system, key, text, true_label)
    for key, rec in adapter_data["generations"].items():
        if rec["output"]:
            items.append(("adapter", key, rec["output"], STYLE_TO_LABEL[rec["style"]]))
    for key, rec in baseline_data["generations"].items():
        if rec["output"]:
            items.append(("baseline", key, rec["output"], STYLE_TO_LABEL[rec["style"]]))

    done = load_checkpoint_done()
    log_progress(f"{len(items)} total items to classify, {len(done)} already checkpointed")

    pending = [it for it in items if (it[0], it[1]) not in done]

    if pending:
        tokenizer = load_tokenizer()
        log_progress("Tokenizer loaded")
        model = load_base_model(tokenizer)
        log_progress("Base model loaded (zero-shot classifier, no adapter)")

        with open(CHECKPOINT_PATH, "a", encoding="utf-8") as ckpt:
            for i in range(0, len(pending), BATCH_SIZE):
                batch = pending[i:i + BATCH_SIZE]
                texts = [b[2] for b in batch]
                t0 = time.time()
                try:
                    preds = classify_batch(model, tokenizer, texts)
                except Exception as exc:
                    log_progress(f"batch at {i} FAILED: {exc}")
                    preds = ["UNKNOWN"] * len(batch)
                dt = time.time() - t0
                for (system, key, text, true_label), pred in zip(batch, preds):
                    ckpt.write(json.dumps({
                        "system": system, "key": key, "true_label": true_label, "pred_label": pred,
                    }, ensure_ascii=False) + "\n")
                ckpt.flush()
                log_progress(f"batch {i // BATCH_SIZE + 1}/{(len(pending) + BATCH_SIZE - 1) // BATCH_SIZE} "
                             f"({i + len(batch)}/{len(pending)}) took {dt:.1f}s")
    else:
        log_progress("Nothing pending, all already classified")

    # Load all results from checkpoint (covers both this run and any resumed prior run)
    by_system = defaultdict(list)
    with open(CHECKPOINT_PATH, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            by_system[rec["system"]].append(rec)

    result = {
        "method": (
            "Zero-shot LLM prompt classification: plain SinLLaMA-merged-base "
            "(NO style adapter attached) is prompted with the 5 style "
            "descriptions and the text to classify, and asked to answer with "
            "one label. No trained style classifier exists in this codebase; "
            "greedy decoding (do_sample=False), max_new_tokens=8."
        ),
        "labels": LABELS,
    }
    for system in ["adapter", "baseline"]:
        recs = by_system.get(system, [])
        y_true = [r["true_label"] for r in recs]
        y_pred = [r["pred_label"] for r in recs]
        result[system] = compute_prf(y_true, y_pred, LABELS)
        result[system]["n"] = len(recs)
        unknown_rate = sum(1 for p in y_pred if p == "UNKNOWN") / len(y_pred) * 100 if y_pred else 0.0
        result[system]["unknown_prediction_rate_pct"] = round(unknown_rate, 2)

    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    log_progress(f"Saved {RESULT_PATH}")
    print(json.dumps({k: v for k, v in result.items() if k in ("adapter", "baseline")}, ensure_ascii=False, indent=2))
    log_progress("=== Style classification run COMPLETE ===")


if __name__ == "__main__":
    main()
