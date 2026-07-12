"""
SinhalaJournal-LLM | Step 6: Test Summarization Adapter
--------------------------------------------------------
Loads model exactly like sinllama_run.py but with the
summarization adapter. Tests on articles from test_set.jsonl.

Usage:
    python 06_test_summarizer.py
    python 06_test_summarizer.py --article "ඔබේ ලිපිය..."
"""

import json
import torch
import argparse
from transformers import AutoTokenizer
from unsloth import FastLanguageModel
from peft import PeftModel


# ──────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────
BASE_MODEL_PATH  = "/home/jovyan/work/sinllama/models/llama-3-8b"
SINLLAMA_ADAPTER = "/home/jovyan/work/sinllama/models/SinLlama_v01"
SUMM_ADAPTER     = "/home/jovyan/work/sinllama/models/adapters/summarization_sinllama_v01"
TOKENIZER_PATH   = "/home/jovyan/work/sinllama/models/Extended-Sinhala-LLaMA"
TEST_DATA_PATH   = "/home/jovyan/summarizer/data/test_set.jsonl"

MAX_SEQ_LENGTH   = 2048
NUM_TEST_SAMPLES = 5


def build_prompt(article: str) -> str:
    return f"""### Instruction:
ඔබ සිංහල පුවත් ලිපි සාරාංශ කිරීමේ විශේෂඥයෙකි.
පහත සිංහල පුවත් ලිපිය කියවා, ලිපියේ ප්‍රධාන කරුණු ඇතුළත් සාරාංශයක් ලියන්න.
සාරාංශය ලිපියේ දිග මෙන් 10% ක් පමණ විය යුතුය.

Article:
{article}

### Response:
"""


def load_model():
    print("🔹 Loading Extended-Sinhala-LLaMA tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_PATH, local_files_only=True
    )
    tokenizer.pad_token = tokenizer.eos_token  # ← was missing

    print("🔹 Loading LLaMA-3-8B base model...")
    model, _ = FastLanguageModel.from_pretrained(
        model_name          = BASE_MODEL_PATH,
        max_seq_length      = MAX_SEQ_LENGTH,
        dtype               = torch.bfloat16,
        load_in_4bit        = True,
        local_files_only    = True,
        attn_implementation = "eager",
    )

    print("🔹 Resizing embeddings...")
    model = model.to("cpu")
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    model = model.to("cuda")

    # ✅ Must mirror training exactly: merge SinLlama first
    print("🔹 Loading and MERGING SinLlama_v01 into base weights...")
    model = PeftModel.from_pretrained(
        model, SINLLAMA_ADAPTER,
        local_files_only    = True,
        ensure_weight_tying = True,
    )
    model = model.merge_and_unload()  # ← bake into base, same as training
    print("   SinLlama_v01 merged ✅")

    # ✅ Now load summarization LoRA on top of merged model
    print("🔹 Loading summarization adapter...")
    model = PeftModel.from_pretrained(
        model, SUMM_ADAPTER,
        local_files_only = True,
    )

    FastLanguageModel.for_inference(model)
    model.eval()
    return model, tokenizer


def generate_summary(model, tokenizer, article: str) -> str:
    prompt = build_prompt(article)
    inputs = tokenizer(
        prompt, 
        return_tensors="pt",
        max_length     = 1800,      # leave room for generation
        truncation     = True,      # truncate article, not summary
    ).to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens     = 512,    # was 250 — Sinhala needs more tokens per word
            do_sample          = False,  # greedy is more stable for summarization
            num_beams          = 4,      # beam search for better coherence
            repetition_penalty = 1.05,   # was 1.2 — gentler for Sinhala morphology
            no_repeat_ngram_size = 3,    # prevents loops without cutting sentences
            early_stopping     = True,
            eos_token_id       = tokenizer.eos_token_id,
            pad_token_id       = tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main(custom_article: str = None):
    print("\n" + "="*55)
    print("  SinhalaJournal-LLM | Summarizer Test")
    print("="*55)

    model, tokenizer = load_model()
    print("\n✅ Model ready\n" + "─"*55)

    # ── Custom article test ──────────────────────────────────────
    if custom_article:
        print(f"\n📰 Article:\n   {custom_article[:200]}...\n")
        summary = generate_summary(model, tokenizer, custom_article)
        print(f"📝 Summary:\n   {summary}")
        words_in  = len(custom_article.split())
        words_out = len(summary.split())
        print(f"\n   {words_in} words → {words_out} words ({words_out/words_in*100:.1f}%)")
        return

    # ── Test on test_set.jsonl ───────────────────────────────────
    print(f"📂 Testing on: {TEST_DATA_PATH}\n")
    records = []
    with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line.strip())
            content = rec.get("content", rec.get("article", ""))
            if content:
                records.append(rec)
            if len(records) >= NUM_TEST_SAMPLES:
                break

    for i, rec in enumerate(records):
        article = rec.get("content", rec.get("article", ""))
        title   = rec.get("title", "no title")

        print(f"[{i+1}/{len(records)}]")
        print(f"  Title   : {title}")
        print(f"  Article : {article[:150]}...")

        summary = generate_summary(model, tokenizer, article)

        words_in  = len(article.split())
        words_out = len(summary.split())
        ratio     = words_out / words_in * 100 if words_in else 0

        print(f"  Summary : {summary}")
        print(f"  Length  : {words_in}w → {words_out}w ({ratio:.1f}%)")
        print("─"*55)

    print(f"\n✅ Done! Check summary quality above.")
    print(f"   Good quality → run ROUGE evaluation")
    print(f"   Poor quality → generate more Gemini summaries and retrain\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--article", type=str, default=None)
    args = parser.parse_args()
    main(args.article)
