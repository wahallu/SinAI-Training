import torch
from unsloth import FastLanguageModel
from transformers import AutoTokenizer, StoppingCriteria, StoppingCriteriaList
from peft import PeftModel
import os
import json
import unicodedata
from collections import Counter

# ── NLTK for sentence GLEU ──
try:
    from nltk.translate.gleu_score import sentence_gleu
    HAS_GLEU = True
except ImportError:
    HAS_GLEU = False
    print("⚠️  nltk not installed. Run: pip install nltk")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ✅ NEW: point directly to the pre-merged SinLLaMA base
#    Run prepare_sinllama_base.py once to generate this directory.
#    No more loading llama-3-8b + SinLlama_v01 + resize + merge here.
SINLLAMA_BASE   = os.path.join(BASE_DIR, "./models/SinLLaMA-merged-base")
GRAMMAR_ADAPTER = os.path.join(BASE_DIR, "./models/adapters/grammar_sinllama_v13")

MAX_SEQ_LENGTH = 512
MAX_NEW_TOKENS = 256

INSTRUCTION_TEXT = (
    "Correct the grammar of the Sinhala sentence. "
    "ONLY fix errors. "
    "If the sentence is already correct, return it EXACTLY unchanged — "
    "do not rephrase, reorder, or change tense."
)


# ─────────────────────────────────────────────
# STOPPING CRITERIA — stop generation at newline
# Prevents the model from appending garbage after the answer
# ─────────────────────────────────────────────
class NewlineStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, prompt_len):
        self.prompt_len = prompt_len
        self.stop_ids = set()
        for text in ["\n", "\n\n", "###"]:
            ids = tokenizer.encode(text, add_special_tokens=False)
            if ids:
                self.stop_ids.add(ids[0])

    def __call__(self, input_ids: torch.LongTensor, scores, **kwargs) -> bool:
        if input_ids.shape[1] <= self.prompt_len:
            return False
        last_token = input_ids[0, -1].item()
        return last_token in self.stop_ids


# ─────────────────────────────────────────────
# LOAD TOKENIZER
# ✅ NEW: tokenizer is already saved inside SinLLaMA-merged-base
#    (prepare_sinllama_base.py called tokenizer.save_pretrained there)
#    No need for a separate TOKENIZER_PATH.
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
#   1. FastLanguageModel.from_pretrained(llama-3-8b)   ← gone
#   2. model.resize_token_embeddings(...)               ← gone
#   3. PeftModel.from_pretrained(model, SinLlama_v01)  ← gone
#   4. model.merge_and_unload()                         ← gone
#
# The merged base already has the right embedding size baked in,
# so no resize step is needed here either.
# ─────────────────────────────────────────────
print("🔹 Loading pre-merged SinLLaMA base...")
model, _ = FastLanguageModel.from_pretrained(
    model_name    = SINLLAMA_BASE,
    max_seq_length= MAX_SEQ_LENGTH,
    dtype         = None,
    load_in_4bit  = True,
)


# ─────────────────────────────────────────────
# LOAD GRAMMAR LoRA ON TOP
# This part is unchanged — grammar adapter sits on top as before
# ─────────────────────────────────────────────
print("🔹 Loading grammar adapter...")
model = PeftModel.from_pretrained(model, GRAMMAR_ADAPTER)

FastLanguageModel.for_inference(model)
model.eval()
print("✅ Model ready.\n")


# ─────────────────────────────────────────────
# INFERENCE FUNCTION
# ─────────────────────────────────────────────
def correct_sentence(sentence: str) -> str:
    prompt = (
        f"### Instruction:\n{INSTRUCTION_TEXT}\n\n"
        f"### Input:\n{sentence}\n\n"
        f"### Response:\n"
    )

    inputs     = tokenizer(prompt, return_tensors="pt").to("cuda")
    prompt_len = inputs["input_ids"].shape[1]

    stopping_criteria = StoppingCriteriaList([
        NewlineStoppingCriteria(tokenizer, prompt_len)
    ])

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens     = MAX_NEW_TOKENS,
            do_sample          = False,
            temperature        = 1.0,
            repetition_penalty = 1.0,   # disabled — task requires reproducing input tokens
            eos_token_id       = tokenizer.eos_token_id,
            pad_token_id       = tokenizer.eos_token_id,
            stopping_criteria  = stopping_criteria,
            use_cache          = True,
        )

    # Decode ONLY the newly generated tokens, not the full prompt
    new_tokens = outputs[0][prompt_len:]
    result     = tokenizer.decode(new_tokens, skip_special_tokens=True)

    # Take first line only, strip whitespace
    result = result.strip().split("\n")[0].strip()

    # Safety: if output is empty or garbage, return input unchanged
    if not result or len(result) < 2:
        return sentence

    return result


# ─────────────────────────────────────────────
# LOAD TEST DATA
# ─────────────────────────────────────────────
test_data_path = os.path.join(BASE_DIR, "data/grammar_test_stage2.jsonl")
with open(test_data_path, "r", encoding="utf-8") as f:
    test_data = [
        (json.loads(line)["input"], json.loads(line)["output"])
        for line in f if line.strip()
    ]


# ─────────────────────────────────────────────────────────────────────────
# HELPER METRICS
# ─────────────────────────────────────────────────────────────────────────

def sinhala_tokenize(text: str) -> list:
    """Character-level tokenizer for Sinhala (Unicode grapheme clusters).
    Groups base + combining diacritics into single tokens so ක් is one unit."""
    tokens = []
    chars  = list(text)
    i = 0
    while i < len(chars):
        cluster = chars[i]
        # absorb following combining characters (diacritics, virama, etc.)
        i += 1
        while i < len(chars) and unicodedata.combining(chars[i]):
            cluster += chars[i]
            i += 1
        if cluster.strip():
            tokens.append(cluster)
    return tokens


def token_prf(pred: str, ref: str) -> tuple:
    """Token-level Precision / Recall / F1 using grapheme-cluster tokens."""
    pred_toks = sinhala_tokenize(pred)
    ref_toks  = sinhala_tokenize(ref)
    pred_cnt  = Counter(pred_toks)
    ref_cnt   = Counter(ref_toks)
    common    = sum((pred_cnt & ref_cnt).values())
    p = common / len(pred_toks) if pred_toks else 0.0
    r = common / len(ref_toks)  if ref_toks  else 0.0
    f = 2 * p * r / (p + r)    if (p + r)   else 0.0
    return p, r, f


def char_f1(pred: str, ref: str) -> float:
    """Character-level F1 (good for spelling/diacritic errors)."""
    pc = Counter(pred)
    rc = Counter(ref)
    common = sum((pc & rc).values())
    p = common / len(pred) if pred else 0.0
    r = common / len(ref)  if ref  else 0.0
    return 2 * p * r / (p + r) if (p + r) else 0.0


def gleu_score(pred: str, ref: str) -> float:
    """Sentence GLEU on grapheme tokens (handles Sinhala better than word BLEU)."""
    if not HAS_GLEU:
        return 0.0
    hyp  = sinhala_tokenize(pred)
    refs = [sinhala_tokenize(ref)]
    return sentence_gleu(refs, hyp)


def rouge_scores(pred: str, ref: str) -> dict:
    """ROUGE-1, ROUGE-2, ROUGE-L computed natively on Sinhala grapheme clusters.
    Uses LCS for ROUGE-L and n-gram overlap for ROUGE-1/2.
    Does NOT rely on rouge_score library which breaks on Sinhala Unicode."""
    def ngrams(tokens, n):
        return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1))

    def lcs_length(a, b):
        """Length of longest common subsequence."""
        m, n = len(a), len(b)
        # Use O(n) space DP
        prev = [0] * (n + 1)
        curr = [0] * (n + 1)
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if a[i-1] == b[j-1]:
                    curr[j] = prev[j-1] + 1
                else:
                    curr[j] = max(prev[j], curr[j-1])
            prev, curr = curr, [0] * (n + 1)
        return prev[n]

    pred_toks = sinhala_tokenize(pred)
    ref_toks  = sinhala_tokenize(ref)
    if not pred_toks or not ref_toks:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    # ROUGE-1
    p1 = ngrams(pred_toks, 1)
    r1 = ngrams(ref_toks,  1)
    c1 = sum((p1 & r1).values())
    prec1 = c1 / len(pred_toks)
    rec1  = c1 / len(ref_toks)
    f1_1  = 2*prec1*rec1/(prec1+rec1) if (prec1+rec1) else 0.0

    # ROUGE-2
    p2 = ngrams(pred_toks, 2)
    r2 = ngrams(ref_toks,  2)
    c2 = sum((p2 & r2).values())
    prec2 = c2 / max(len(pred_toks)-1, 1)
    rec2  = c2 / max(len(ref_toks)-1,  1)
    f1_2  = 2*prec2*rec2/(prec2+rec2) if (prec2+rec2) else 0.0

    # ROUGE-L (LCS)
    lcs = lcs_length(pred_toks, ref_toks)
    precL = lcs / len(pred_toks)
    recL  = lcs / len(ref_toks)
    f1_L  = 2*precL*recL/(precL+recL) if (precL+recL) else 0.0

    return {"rouge1": f1_1, "rouge2": f1_2, "rougeL": f1_L}


def over_correction_rate(pred: str, inp: str, ref: str) -> bool:
    """True if pred changes something that was already correct in the input
    (pred differs from ref in a place where inp == ref)."""
    return (inp.strip() == ref.strip()) and (pred.strip() != ref.strip())


# ─────────────────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("EVALUATION")
print("=" * 60)

correct_count    = 0
change_correct   = 0
nochange_correct = 0
change_total     = sum(1 for inp, exp in test_data if inp.strip() != exp.strip())
nochange_total   = len(test_data) - change_total

# Accumulators for aggregate metrics
agg_rouge1 = agg_rouge2 = agg_rougeL = 0.0
agg_gleu   = 0.0
agg_char_f1 = 0.0
agg_tok_p = agg_tok_r = agg_tok_f1 = 0.0
overcorrection_count = 0

predictions = []

for inp, expected in test_data:
    pred         = correct_sentence(inp)
    is_correct   = (pred.strip() == expected.strip())
    needs_change = (inp.strip() != expected.strip())

    # ── exact-match counters ──
    if is_correct:
        correct_count += 1
        if needs_change:
            change_correct += 1
        else:
            nochange_correct += 1

    # ── over-correction ──
    if over_correction_rate(pred, inp, expected):
        overcorrection_count += 1

    # ── per-sample metrics ──
    r = rouge_scores(pred, expected)
    g = gleu_score(pred, expected)
    cf = char_f1(pred, expected)
    tp, tr, tf = token_prf(pred, expected)

    agg_rouge1  += r["rouge1"]
    agg_rouge2  += r["rouge2"]
    agg_rougeL  += r["rougeL"]
    agg_gleu    += g
    agg_char_f1 += cf
    agg_tok_p   += tp
    agg_tok_r   += tr
    agg_tok_f1  += tf

    predictions.append((inp, pred, expected, is_correct, needs_change))

    status = "✅" if is_correct else "❌"
    print(f"\n{status} {'[CHANGE]' if needs_change else '[NO CHANGE]'}")
    print(f"   INPUT   : {inp}")
    print(f"   PREDICT : {pred}")
    print(f"   EXPECTED: {expected}")
    print(f"   ROUGE-L={r['rougeL']:.3f}  Char-F1={cf:.3f}  GLEU={g:.3f}")

# ── Averages ──
n = len(test_data)
overall_acc  = correct_count    / n             * 100
change_acc   = change_correct   / change_total  * 100 if change_total  else 0.0
nochange_acc = nochange_correct / nochange_total * 100 if nochange_total else 0.0
over_rate    = overcorrection_count / nochange_total * 100 if nochange_total else 0.0

print("\n" + "=" * 60)
print("EXACT-MATCH RESULTS")
print("=" * 60)
print(f"  Overall accuracy      : {correct_count}/{n}  →  {overall_acc:.1f}%")
print(f"  Change-needed accuracy: {change_correct}/{change_total}  →  {change_acc:.1f}%")
print(f"  No-change accuracy    : {nochange_correct}/{nochange_total}  →  {nochange_acc:.1f}%")
print(f"  Over-correction rate  : {overcorrection_count}/{nochange_total}  →  {over_rate:.1f}%  (changed correct sentences)")

print("\n" + "=" * 60)
print("CONTINUOUS METRICS  (avg over all samples)")
print("=" * 60)
print(f"  ROUGE-1   (grapheme): {agg_rouge1/n:.4f}")
print(f"  ROUGE-2   (grapheme): {agg_rouge2/n:.4f}")
print(f"  ROUGE-L   (grapheme): {agg_rougeL/n:.4f}")
print(f"  Sentence GLEU       : {agg_gleu/n:.4f}")
print(f"  Char-level F1       : {agg_char_f1/n:.4f}")
print(f"  Token Precision     : {agg_tok_p/n:.4f}")
print(f"  Token Recall        : {agg_tok_r/n:.4f}")
print(f"  Token F1            : {agg_tok_f1/n:.4f}")
print("=" * 60)
print()
print("  Metric guide for Sinhala grammar correction:")
print("  ┌──────────────────┬──────────┬──────────┬──────────┐")
print("  │ Metric           │  Poor    │   OK     │  Good    │")
print("  ├──────────────────┼──────────┼──────────┼──────────┤")
print("  │ ROUGE-L          │  < 0.80  │  0.80-93 │  > 0.93  │")
print("  │ Char-F1          │  < 0.85  │  0.85-95 │  > 0.95  │")
print("  │ GLEU             │  < 0.50  │  0.50-80 │  > 0.80  │")
print("  │ Token F1         │  < 0.80  │  0.80-93 │  > 0.93  │")
print("  │ Over-correction  │  > 30%   │  10-30%  │  < 10%   │")
print("  └──────────────────┴──────────┴──────────┴──────────┘")
print()

# ── Actionable warnings ──
if nochange_acc < 50:
    print("⚠️  No-change accuracy is low. Add more 'correct as-is' training examples.")
if change_acc < 50:
    print("⚠️  Change accuracy is low. Check dataset quality and max_new_tokens.")
if over_rate > 25:
    print(f"⚠️  Over-correction too high ({over_rate:.1f}%). Model is changing correct text.")
if (agg_rougeL / n) < 0.80:
    print("⚠️  ROUGE-L < 0.80 — model is making major token-level errors.")
if (agg_char_f1 / n) < 0.85:
    print("⚠️  Char-F1 < 0.85 — model has diacritic/spelling accuracy issues.")