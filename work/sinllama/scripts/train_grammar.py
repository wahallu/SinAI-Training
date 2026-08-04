import torch
from unsloth import FastLanguageModel
from transformers import AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
import datetime
import hashlib
import io
import json
import os
import re
import subprocess
import sys

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
# ✅ NEW: point directly to pre-merged SinLLaMA base
#    Same as test script — no more 4-step chain
SINLLAMA_BASE = "./models/SinLLaMA-merged-base"
# ⚠️ UPDATE per round. Previous: cleaned_v3 -> v16, cleaned_v4 -> v17,
#    cleaned_v5 -> v19, cleaned_v7_full -> v20/v21.
DATA_PATH     = "data/grammar_manual_dataset_stage12.jsonl"
OUTPUT_DIR    = "./models/adapters/grammar_sinllama_v23"

# ─────────────────────────────────────────────
# RUN CONTROL — the two switches worth having at the top
#
# SAVE_TRAINING_LOG
#   Writes the whole run to Training_logs/<adapter>_<timestamp>.md, plus a
#   per-epoch train/eval loss table and an overfitting verdict.
#
#   This exists because v23 could not be diagnosed after the fact. It changed
#   two things against v22 at once — the v6 hand-built share fell 40.2% ->
#   30.1%, and it took 33% more gradient updates (17,104 vs 12,856, i.e. 8.2
#   vs 6.2 v7-equivalent epochs) — and with no loss curve recorded anywhere
#   there was no way to tell dilution from over-training. The eval_loss curve
#   answers that directly: rising while train_loss falls is over-training,
#   both still falling is not.
#
# RUN_TEST_AFTER_TRAINING
#   Runs test_grammar.py on the adapter that was just trained, so a finished
#   run leaves both a training log and a scored transcript without anyone
#   having to be at the keyboard. Set False to train only.
# ─────────────────────────────────────────────
SAVE_TRAINING_LOG        = True
RUN_TEST_AFTER_TRAINING  = True
TEST_STAGES              = "all"   # passed to test_grammar.py --stage
TEST_NUM_CANDIDATES      = 1       # 1 = greedy, the comparable baseline


class _Tee:
    """Duplicate writes to several streams, so the run prints live AND is
    captured. Same approach test_grammar.py already uses for transcripts."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()


_LOG_BUFFER = None
if SAVE_TRAINING_LOG:
    _LOG_BUFFER = io.StringIO()
    sys.stdout = _Tee(sys.__stdout__, _LOG_BUFFER)
    print("🔹 Run started: " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

# ✅ v18 changes BOTH data and LoRA capacity at once (user-directed).
#    NOTE: this deliberately breaks the one-variable-at-a-time rule the
#    earlier rounds followed — if v18 improves, we won't know how much
#    came from the MLP/r=32 capacity bump vs. the cleaned_v5 data. That
#    tradeoff was accepted to save a ~2.5h train cycle. If v18 REGRESSES,
#    re-run once with the LoRA block reverted to r=16 / attention-only
#    (kept in the comment below) to find out which half caused it.
#
#    Previous LoRA config, for easy rollback:
#      r = 16, lora_alpha = 16,
#      target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

# ✅ INCREASED: was 256 — paragraphs need more space
#    Single sentence: ~50-80 tokens
#    Paragraph (3 sentences): ~150-200 tokens
#    Prompt overhead: ~60 tokens
#    Total needed: ~260 tokens → use 512 safely
MAX_SEQ_LENGTH = 512

# ─────────────────────────────────────────────
# DATASET BALANCE CHECK
# ─────────────────────────────────────────────
print("🔹 Checking dataset balance...")
with open(DATA_PATH, "r", encoding="utf-8") as f:
    samples = [json.loads(line) for line in f if line.strip()]

changed   = sum(1 for s in samples if s["input"].strip() != s["output"].strip())
unchanged = len(samples) - changed
print(f"   Total    : {len(samples)}")
print(f"   Changed  : {changed}  ({changed/len(samples)*100:.1f}%)")
print(f"   Unchanged: {unchanged} ({unchanged/len(samples)*100:.1f}%)")
if changed / len(samples) > 0.65:
    print("   ⚠️  WARNING: >65% changed — consider adding more no-change examples.")

# ─────────────────────────────────────────────
# DATASET FINGERPRINT
#
# v20 and v21 produced byte-identical scores on all four stages. The likeliest
# reason is simply that the v21 data delta was tiny (~0.5% of rows), but a
# stale upload or a reused HF cache would look EXACTLY the same from the
# outside — and we had no way to tell the two apart after the fact.
#
# So record what was actually trained on. The hash goes in the run log, and any
# two runs sharing a hash are the same data by definition, no inference needed.
# ─────────────────────────────────────────────
_h = hashlib.sha256()
for s in samples:
    _h.update((s["input"] + "\x00" + s["output"] + "\x01").encode("utf-8"))
DATA_FINGERPRINT = _h.hexdigest()[:16]

# Canaries: specific corrections whose presence/absence distinguishes dataset
# versions. Cheap to compute, and they fail loudly rather than silently.
_ZWJ = "‍"
_canaries = {
    "teaches තරග→තරඟ":  sum(1 for s in samples
                            if re.search(r"(?<![඀-෿])තරග", s["input"])
                            and "තරඟ" in s["output"]),
    "BAD gold තරග":     sum(1 for s in samples
                            if re.search(r"(?<![඀-෿])තරග", s["output"])),
    "conjunct ්‍ය/්‍ර": sum(1 for s in samples
                            if ("්" + _ZWJ + "ය" in s["output"]
                                and "්" + _ZWJ + "ය" not in s["input"])
                            or ("්" + _ZWJ + "ර" in s["output"]
                                and "්" + _ZWJ + "ර" not in s["input"])),
    "SOV word-order":   sum(1 for s in samples
                            if s["input"] != s["output"]
                            and sorted(s["input"].rstrip(".").split())
                             == sorted(s["output"].rstrip(".").split())),
}
print(f"   Fingerprint: {DATA_FINGERPRINT}   ({os.path.basename(DATA_PATH)})")
for k, v in _canaries.items():
    print(f"     {k:20s}: {v}")

# cleaned_v8_full.jsonl should report 27064 rows / 64.9% changed and canaries
# 315 / 10 / ~3932 / ~1059. A materially different reading means the file on
# this box is not the one that was built — stop and re-upload.
if _canaries["BAD gold තරග"] > 40:
    print("   ⚠️  WARNING: තරග appears as CORRECT gold in "
          f"{_canaries['BAD gold තරග']} rows. Expected <=10 — this looks like "
          "the PRE-FIX file. Re-upload before training.")

# ─────────────────────────────────────────────
# LOAD TOKENIZER
# ✅ NEW: tokenizer already saved in SinLLaMA-merged-base
#    No separate TOKENIZER_PATH needed
# ─────────────────────────────────────────────
print("🔹 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    SINLLAMA_BASE,
    local_files_only=True,
)
tokenizer.pad_token    = tokenizer.eos_token
tokenizer.padding_side = "right"

# ─────────────────────────────────────────────
# LOAD PRE-MERGED SINLLAMA BASE
# ✅ NEW: replaces the old 4-step chain:
#   1. FastLanguageModel.from_pretrained(llama-3-8b)   ← REMOVED
#   2. model.resize_token_embeddings(...)               ← REMOVED
#   3. PeftModel.from_pretrained(model, SinLlama_v01)  ← REMOVED
#   4. model.merge_and_unload()                         ← REMOVED
#
# The merged base already has the right embedding size baked in.
# ─────────────────────────────────────────────
print("🔹 Loading pre-merged SinLLaMA base...")
model, _ = FastLanguageModel.from_pretrained(
    model_name    = SINLLAMA_BASE,
    max_seq_length= MAX_SEQ_LENGTH,
    dtype         = None,
    load_in_4bit  = True,
)

# ─────────────────────────────────────────────
# ADD GRAMMAR LoRA
#
# IMPORTANT — why lora_dropout=0.05 and NOT 0.0:
#   Setting dropout=0.0 activates Unsloth's fast custom CUDA LoRA
#   kernel. After merge_and_unload() on a 4bit model, the bnb
#   quantized weights cause a dtype mismatch inside that kernel
#   (BFloat16 != float in fast_lora.py backward).
#   Using dropout=0.05 makes Unsloth fall back to standard PyTorch
#   autograd, which handles mixed dtypes correctly.
#
# NOTE: Now that we load SinLLaMA-merged-base directly (not merging
#   at runtime), this issue still applies — the base model weights
#   are still 4bit quantized when loaded via load_in_4bit=True.
#   Keep lora_dropout=0.05.
# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
# v18 CAPACITY BUMP — MLP layers + r 16->32
#
# WHY: several specific bugs survived 2+ rounds of targeted data
#   additions (register/word substitution ඇමතිවරයා->අමාත්‍යවරයා and
#   කිව්වා->පැවසුවේ; verb-stem selection පවත්වූහ vs පැවැත්වූහ; plural
#   literary endings on the same test sentences). These are lexical /
#   word-choice decisions, which in transformers live substantially in
#   the MLP (feed-forward) weights — and LoRA was only ever attached to
#   the attention projections, so no amount of data could reach them.
#   This was flagged as "Finding #2" back in the v14 investigation and
#   deferred until data alone showed diminishing returns. It has.
#
# COST: trainable params go from ~16M to ~84M (~5x). Expect training
#   time to rise from ~100 min to roughly 2.5h on the A40, and higher
#   VRAM use (should still fit in 44GB at batch 2 / seq 512 / 4bit).
#
# lora_alpha is kept equal to r (32/32), preserving the same
#   alpha/r = 1.0 scaling ratio the previous runs used — so this is a
#   capacity change, not an effective-learning-rate change.
# ─────────────────────────────────────────────
print("🔹 Adding grammar LoRA adapter...")
model = FastLanguageModel.get_peft_model(
    model,
    r              = 32,
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",     # attention (as before)
        "gate_proj", "up_proj", "down_proj",        # v18: MLP — lexical/word-choice
    ],
    lora_alpha     = 32,
    lora_dropout   = 0.05,  # must stay 0.05 — see note above
    bias           = "none",
)
model.print_trainable_parameters()

# ─────────────────────────────────────────────
# PROMPT FORMAT
#
# v15 CHANGE: emit separate "prompt"/"completion" columns instead of
#   one concatenated "text" field.
#
# WHY: v14 (and everything before it) built a single "text" field
#   (Instruction + Input + Response, all concatenated) and trained
#   with train_on_inputs=True — but that isn't even a real SFTConfig/
#   SFTTrainer parameter in this TRL version (0.24.0; confirmed via
#   `from trl import DataCollatorForCompletionOnlyLM` also failing —
#   that class was removed entirely, see huggingface/trl discussion
#   #3826 and issue #5324). So it was silently ignored, and per TRL's
#   current docs, a plain "text"-field dataset always computes loss
#   over the FULL sequence — instruction, input, AND response tokens.
#   A large, fixed share of every gradient step was spent reproducing
#   the identical instruction text and the given input, which diluted
#   the signal available to learn the actual correction. This is the
#   leading suspect for why v14 (2,851 rows, +535 new/expanded rows
#   across 10 new grammar categories vs v13's 2,316) scored bit-for-
#   bit identical to v13 on the 57-example stage2 test set — see
#   train_roadmap.md, Finding #1.
#
# FIX: TRL's current (0.24.0) supported way to do completion-only loss
#   is the "prompt-completion" dataset format: separate "prompt" and
#   "completion" text columns, with SFTConfig(completion_only_loss=True)
#   (this is also the DEFAULT for prompt-completion datasets — passed
#   explicitly below only for clarity). No custom collator needed.
# ─────────────────────────────────────────────
INSTRUCTION_TEXT = (
    "Correct the grammar of the Sinhala sentence. "
    "ONLY fix errors. "
    "If the sentence is already correct, return it EXACTLY unchanged — "
    "do not rephrase, reorder, or change tense."
)

def format_prompt(example):
    prompt = (
        f"### Instruction:\n{INSTRUCTION_TEXT}\n\n"
        f"### Input:\n{example['input']}\n\n"
        f"### Response:\n"
    )
    # NOTE: do NOT manually append tokenizer.eos_token here — SFTConfig's
    # `eos_token` (defaults to processing_class.eos_token) handles turn/
    # sequence termination for prompt-completion datasets. Appending it
    # ourselves too risks a doubled EOS token.
    return {"prompt": prompt, "completion": example["output"]}

# ─────────────────────────────────────────────
# LOAD + FORMAT DATASET
# ─────────────────────────────────────────────
print("🔹 Loading dataset...")
# download_mode="force_redownload" makes load_dataset re-read the file instead of
# reusing a cached Arrow build. The cache is keyed on the path, so editing a file
# in place and keeping its name is precisely the case it can get wrong — and it
# fails silently, which is the worst property for something feeding a 2.5h run.
dataset = load_dataset("json", data_files=DATA_PATH,
                       download_mode="force_redownload")
dataset = dataset.map(format_prompt, remove_columns=dataset["train"].column_names)
print(f"   Dataset size: {len(dataset['train'])} examples")

# Cross-check the loaded dataset against the raw file read at the top. If these
# disagree, load_dataset served something other than the file on disk.
if len(dataset["train"]) != len(samples):
    raise SystemExit(
        f"❌ Dataset mismatch: raw file has {len(samples)} rows but "
        f"load_dataset returned {len(dataset['train'])}. Cache is stale — "
        f"run: rm -rf ~/.cache/huggingface/datasets"
    )

# ─────────────────────────────────────────────
# v15 ADDITION — small held-out validation split
#
# WHY: v14 trained 5 fixed epochs with zero eval signal in between —
#   the only feedback was the expensive full test_grammar.py run at
#   the end (train_roadmap.md, Finding #4). Holding out 5% purely for
#   periodic eval_loss monitoring doesn't change what the model learns
#   from the training rows themselves, it just adds visibility into
#   whether 5 epochs is under/over-fit. cleaned_v2.jsonl is already
#   shuffled (see gen_data.py), so a contiguous slice is a fine random
#   sample — but train_test_split reshuffles with a fixed seed anyway
#   so this is robust either way.
# ─────────────────────────────────────────────
split       = dataset["train"].train_test_split(test_size=0.05, seed=42)
train_data  = split["train"]
eval_data   = split["test"]
print(f"   Train split  : {len(train_data)} examples")
print(f"   Eval split   : {len(eval_data)} examples (held out, monitoring only)")

# ─────────────────────────────────────────────
# COMPUTE HYPERPARAMETERS
# Formula (consistent with all previous versions):
#   effective_batch = per_device(2) × grad_accum(4) = 8
#   steps_per_epoch = ceil(samples / 8)
#   max_steps       = steps_per_epoch × 5 epochs
#   warmup_steps    = round(max_steps × 0.1)
#   save_steps      = round(max_steps / 2)
# NOTE: n_samples is now the TRAIN split (95% of the file) now that a
#   held-out eval slice exists, so max_steps is computed on the same
#   basis actually used for gradient updates.
# ─────────────────────────────────────────────
import math

# ── Training volume, in steps rather than epochs ──
#
# "Epochs" is the wrong unit to reason in here, because the dataset keeps
# growing. What the model actually sees is gradient updates, and at a fixed
# effective batch of 8 those scale with rows x epochs:
#
#   version  dataset      rows     epochs   max_steps   v7-equivalent epochs
#   v20/v21  v7_full      17,532     5        10,410           5.0
#   v22      v8_full      27,064     4        12,856           6.2   <- best
#   v23      v9_full      36,006     4        17,104           8.2   <- regressed
#
# v23 changed TWO things at once against v22: the v6 hand-built share fell
# 40.2% -> 30.1%, AND it took 33% more gradient updates. The comment this
# replaces set 4 epochs precisely to avoid a "~7.7-epoch-equivalent run, the
# first real overfitting risk in the series" — v23 landed at 8.2, past that
# line, without anyone re-deriving it for the bigger file. So the regression
# has two candidate causes and the results alone cannot separate them.
#
# TARGET_STEPS pins the volume directly so it stops drifting every time the
# dataset grows. Set it to None to go back to a fixed epoch count.
#
#   None  -> EPOCHS * steps_per_epoch, the historical behaviour
#   12856 -> exactly v22's update budget, whatever the dataset size
TARGET_STEPS = None
EPOCHS       = 4
n_samples    = len(train_data)
steps_epoch  = math.ceil(n_samples / 8)
max_steps    = TARGET_STEPS if TARGET_STEPS else steps_epoch * EPOCHS
effective_epochs = max_steps / steps_epoch
warmup_steps = round(max_steps * 0.1)
save_steps   = round(max_steps / 2)
# v18: eval once per epoch instead of twice per RUN. With ~5x the
#   trainable params, overfitting is a real risk and the old 2-evals-
#   total cadence was far too coarse to see it. This is monitoring only
#   — it does not change what the model learns.
#   WATCH FOR: if eval_loss stops falling (or rises) after epoch 3 while
#   train loss keeps dropping, that's overfitting — rerun with 3-4
#   epochs instead of 5.
eval_steps   = steps_epoch

print(f"\n📊 Hyperparameters:")
print(f"   data         = {DATA_PATH}")
print(f"   fingerprint  = {DATA_FINGERPRINT}")
print(f"   adapter      = {OUTPUT_DIR}")
print(f"   epochs       = {effective_epochs:.2f}"      f"{'  (pinned via TARGET_STEPS)' if TARGET_STEPS else ''}")
print(f"   samples      = {n_samples}")
print(f"   steps/epoch  = {steps_epoch}")
print(f"   max_steps    = {max_steps}")
print(f"   warmup_steps = {warmup_steps}")
print(f"   save_steps   = {save_steps}")
print(f"   eval_steps   = {eval_steps}")

# ─────────────────────────────────────────────
# TRAINER
#
# v15 CHANGE: args is now SFTConfig, not TrainingArguments.
#   TRL 0.24.0's SFTTrainer constructor doesn't accept
#   dataset_text_field / max_seq_length / packing / tokenizer= as
#   direct kwargs (confirmed against current trl docs) — those all
#   moved into SFTConfig (max_seq_length -> max_length), and the
#   tokenizer kwarg is now processing_class=. The v14 script passed
#   all of these the old way; they were most likely silently dropped
#   rather than erroring (Unsloth is known to shim old-style kwargs
#   for backward compatibility), which is further evidence
#   train_on_inputs=True never did anything either.
# ─────────────────────────────────────────────
trainer = SFTTrainer(
    model            = model,
    processing_class = tokenizer,
    train_dataset    = train_data,
    eval_dataset     = eval_data,

    args = SFTConfig(
        max_length                  = MAX_SEQ_LENGTH,
        packing                     = False,
        completion_only_loss        = True,  # default for prompt-completion data; explicit for clarity

        per_device_train_batch_size = 2,
        per_device_eval_batch_size  = 2,
        gradient_accumulation_steps = 4,
        max_steps                   = max_steps,
        learning_rate               = 5e-5,
        logging_steps               = 10,
        max_grad_norm               = 1.0,
        lr_scheduler_type           = "cosine",
        warmup_steps                = warmup_steps,
        output_dir                  = OUTPUT_DIR,
        save_steps                  = save_steps,
        save_total_limit            = 2,
        eval_strategy               = "steps",
        eval_steps                  = eval_steps,
        report_to                   = "none",
    ),
)

# ─────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────
print("\n🚀 Training started...")
trainer.train()

# ─────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────
print("💾 Saving model...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"✅ Training complete! Model saved to {OUTPUT_DIR}")

# ─────────────────────────────────────────────
# LOSS CURVE + OVERFITTING VERDICT
#
# The single number that was missing when v23 regressed. eval_loss rising
# while train_loss keeps falling is over-training; both still falling means
# the regression came from the data, not the schedule.
# ─────────────────────────────────────────────
ADAPTER_NAME = os.path.basename(os.path.normpath(OUTPUT_DIR))


def summarise_losses(history):
    """Pair train and eval loss by step. Returns (rows, verdict_lines)."""
    train_pts = [(h["step"], h["loss"]) for h in history if "loss" in h and "step" in h]
    eval_pts = [(h["step"], h["eval_loss"]) for h in history if "eval_loss" in h]

    rows = []
    for step, ev in eval_pts:
        # nearest logged train loss at or before this eval
        before = [l for s, l in train_pts if s <= step]
        tr = before[-1] if before else float("nan")
        rows.append((step, step / max(steps_epoch, 1), tr, ev))

    verdict = []
    if len(eval_pts) < 2:
        verdict.append("Not enough eval points to judge overfitting.")
        return rows, verdict

    losses = [e for _s, e in eval_pts]
    best_i = min(range(len(losses)), key=lambda i: losses[i])
    verdict.append("best eval_loss %.5f at eval #%d (step %d, epoch %.2f)"
                   % (losses[best_i], best_i + 1, eval_pts[best_i][0],
                      eval_pts[best_i][0] / max(steps_epoch, 1)))
    if best_i == len(losses) - 1:
        verdict.append("eval_loss was STILL FALLING at the end -> not over-trained; "
                       "more steps may still help.")
    else:
        rise = losses[-1] - losses[best_i]
        verdict.append("eval_loss ROSE %.5f after that point -> OVER-TRAINED past "
                       "eval #%d. Cut TARGET_STEPS to about %d."
                       % (rise, best_i + 1, eval_pts[best_i][0]))
    return rows, verdict


try:
    _history = trainer.state.log_history
except Exception:
    _history = []

_rows, _verdict = summarise_losses(_history)

print("\n📉 Loss curve")
print("   %-8s %-8s %-12s %-12s" % ("step", "epoch", "train_loss", "eval_loss"))
for step, ep, tr, ev in _rows:
    print("   %-8d %-8.2f %-12.5f %-12.5f" % (step, ep, tr, ev))
print("\n🔎 Overfitting check")
for line in _verdict:
    print("   " + line)

# ─────────────────────────────────────────────
# WRITE THE TRAINING LOG
# ─────────────────────────────────────────────
if SAVE_TRAINING_LOG and _LOG_BUFFER is not None:
    sys.stdout = sys.__stdout__
    logs_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                            "..", "Training_logs"))
    os.makedirs(logs_dir, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join(logs_dir, ADAPTER_NAME + "_" + stamp + ".md")

    header = [
        "# " + ADAPTER_NAME + " — training log",
        "",
        "- data       : " + os.path.basename(DATA_PATH),
        "- fingerprint: " + DATA_FINGERPRINT,
        "- rows       : " + str(len(samples)) + "  (" + str(changed) + " changed)",
        "- max_steps  : " + str(max_steps) + "   (" + ("%.2f" % effective_epochs) + " epochs"
        + (", pinned via TARGET_STEPS" if TARGET_STEPS else "") + ")",
        "- steps/epoch: " + str(steps_epoch),
        "",
        "## loss curve",
        "",
        "| step | epoch | train_loss | eval_loss |",
        "|---|---|---|---|",
    ]
    for step, ep, tr, ev in _rows:
        header.append("| %d | %.2f | %.5f | %.5f |" % (step, ep, tr, ev))
    header += ["", "## overfitting check", ""]
    header += ["- " + v for v in _verdict]
    header += ["", "## full run output", "", "```"]

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(header) + "\n")
        f.write(_LOG_BUFFER.getvalue())
        f.write("```\n")
    print("📝 Training log saved to " + log_path)

# ─────────────────────────────────────────────
# EVALUATE
#
# Free the training model first: test_grammar.py loads its own copy of the
# base in a subprocess, and holding both at once is an OOM on a 44GB A40.
# ─────────────────────────────────────────────
if RUN_TEST_AFTER_TRAINING:
    print("\n🧪 Running test_grammar.py on " + ADAPTER_NAME + " ...")
    try:
        del trainer
        del model
    except Exception:
        pass
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    test_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_grammar.py")
    cmd = [sys.executable, test_script,
           "--adapter", ADAPTER_NAME,
           "--stage", TEST_STAGES,
           "--num-candidates", str(TEST_NUM_CANDIDATES)]
    print("   " + " ".join(cmd))
    rc = subprocess.call(cmd)
    if rc == 0:
        print("✅ Evaluation finished — transcript is in Tested_results/")
    else:
        # Never let a failed eval look like a failed train: the adapter is
        # already saved and is fine.
        print("⚠️  test_grammar.py exited with code " + str(rc)
              + ". The adapter is saved; re-run the test manually.")
