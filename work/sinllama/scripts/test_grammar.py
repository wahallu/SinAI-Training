import argparse
import datetime
import glob
import io
import json
import os
import re
import sys
import unicodedata
from collections import Counter

import torch
from unsloth import FastLanguageModel
from transformers import AutoTokenizer, StoppingCriteria, StoppingCriteriaList
from peft import PeftModel

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
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SINLLAMA_BASE = os.path.join(BASE_DIR, "./models/SinLLaMA-merged-base")
ADAPTERS_DIR  = os.path.join(BASE_DIR, "models/adapters")

MAX_SEQ_LENGTH = 512
MAX_NEW_TOKENS = 256

# ── Prompt variant under test ──
# "english" — byte-for-byte identical to train_grammar.py's INSTRUCTION_TEXT
#             and to tasks/grammar.py's prompt_grammar() as of the fix that
#             aligned production to training. This is what the adapter was
#             actually trained on; scores here are the real ceiling.
# "sinhala" — the prompt tasks/grammar.py sent to production BEFORE that
#             fix: a different persona, different scope (spelling +
#             punctuation, never mentioned in training), and missing the
#             "return EXACTLY unchanged" restraint line entirely. Off-
#             distribution relative to training. Kept here so the accuracy
#             cost of that mismatch can be measured directly against the
#             same test sets, instead of argued about.
# Change this one line to switch what every run below tests.
PROMPT_VARIANT = "english"  # "english" | "sinhala"

# Training's fixed instruction — see train_grammar.py's INSTRUCTION_TEXT.
# Training never reads the per-row "instruction" field in the jsonl files
# (it builds this same fixed string for every row, sentence or paragraph
# alike) — using anything else here would test the model on a prompt
# distribution it never actually saw during training.
ENGLISH_INSTRUCTION_TEXT = (
    "Correct the grammar of the Sinhala sentence. "
    "ONLY fix errors. "
    "If the sentence is already correct, return it EXACTLY unchanged — "
    "do not rephrase, reorder, or change tense."
)

# The pre-fix production instruction — copied verbatim from tasks/grammar.py
# as it stood before the alignment fix (see git history on that file for the
# exact commit). Reproduced here, not imported, so this file keeps working
# even after that module changes further — this is a frozen snapshot of what
# production actually sent, on purpose.
SINHALA_INSTRUCTION_TEXT = (
    "ඔබ සිංහල භාෂා විශේෂඥයෙකි.\n"
    "පහත සිංහල පාඨයේ ඇති වාකරණ දෝෂ, අක්ෂර වින්‍යාස දෝෂ සහ විරාම ලකුණු දෝෂ නිවැරදි කරන්න.\n"
    "නිවැරදි කළ පාඨය පමණක් ලියන්න. වෙනත් කිසිදු පැහැදිලි කිරීමක් එකතු නොකරන්න."
)


def build_prompt(text: str) -> str:
    """
    Build the correction prompt for whichever PROMPT_VARIANT is selected.

    The two variants differ in more than instruction text: english uses
    training's "### Input:" label, sinhala used production's "Text:" label
    (see tasks/grammar.py's pre-fix prompt_grammar()) — reproduced exactly
    as each was, not normalized to share a label, since the whole point is
    to measure what each one actually did.
    """
    if PROMPT_VARIANT == "sinhala":
        return (
            f"### Instruction:\n{SINHALA_INSTRUCTION_TEXT}\n\n"
            f"Text:\n{text}\n\n"
            "### Response:\n"
        )
    if PROMPT_VARIANT == "english":
        return (
            f"### Instruction:\n{ENGLISH_INSTRUCTION_TEXT}\n\n"
            f"### Input:\n{text}\n\n"
            "### Response:\n"
        )
    raise ValueError(f"Unknown PROMPT_VARIANT {PROMPT_VARIANT!r}; expected 'english' or 'sinhala'")


# stage2 = single-sentence news examples, stage3 = 2-3 sentence paragraphs.
TEST_SETS = {
    "stage2": "grammar_test_stage2.jsonl",
    "stage3": "grammar_test_stage3.jsonl",
    "stage4": "grammar_test_stage4.jsonl",
}


def resolve_data_dir(override=None):
    """Locate the folder holding the test-set jsonl files.

    The remote training box keeps everything under 'data/'; this local
    checkout keeps test sets under 'test data/' (with a space). Try both
    so the script runs unmodified in either place instead of needing a
    manual path edit every time it moves.
    """
    if override:
        if os.path.isdir(override):
            return override
        raise FileNotFoundError(f"--data-dir does not exist: {override}")
    for name in ("data", "test data"):
        candidate = os.path.join(BASE_DIR, name)
        if os.path.isdir(candidate):
            return candidate
    raise FileNotFoundError(
        f"Could not find a test-data directory under {BASE_DIR} "
        f"(tried 'data/' and 'test data/'). Pass --data-dir explicitly."
    )


def latest_adapter(adapters_dir):
    """Auto-pick the highest-numbered grammar_sinllama_vN adapter directory
    so a new training run doesn't require editing this script's config."""
    candidates = []
    for path in glob.glob(os.path.join(adapters_dir, "grammar_sinllama_v*")):
        m = re.search(r"_v(\d+)$", os.path.basename(path))
        if m:
            candidates.append((int(m.group(1)), path))
    if not candidates:
        raise FileNotFoundError(f"No grammar_sinllama_vN adapters found in {adapters_dir}")
    return max(candidates)[1]


def resolve_adapter(name_or_path):
    """Accept a full path, a bare adapter dir name, or a 'v18'/'18' shorthand.
    Falls back to auto-detecting the latest version when nothing is given."""
    if name_or_path is None:
        return latest_adapter(ADAPTERS_DIR)
    if os.path.isdir(name_or_path):
        return name_or_path
    m = re.fullmatch(r"v?(\d+)", name_or_path)
    if m:
        candidate = os.path.join(ADAPTERS_DIR, f"grammar_sinllama_v{m.group(1)}")
        if os.path.isdir(candidate):
            return candidate
    candidate = os.path.join(ADAPTERS_DIR, name_or_path)
    if os.path.isdir(candidate):
        return candidate
    raise FileNotFoundError(
        f"Adapter not found: {name_or_path!r} "
        f"(tried as a path, as a vN shorthand, and under {ADAPTERS_DIR})"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the Sinhala grammar-correction adapter.")
    parser.add_argument(
        "--adapter", default=None,
        help="Adapter dir name (e.g. grammar_sinllama_v18), shorthand (e.g. v18 or 18), "
             "or full path. Default: auto-detect the highest vN under models/adapters/.",
    )
    parser.add_argument(
        "--stage", choices=["stage2", "stage3", "both"], default="both",
        help="Which test set(s) to run. Default: both.",
    )
    parser.add_argument("--data-dir", default=None, help="Override the auto-detected test-data folder.")
    parser.add_argument("--limit", type=int, default=None, help="Only evaluate the first N examples per stage.")
    parser.add_argument("--quiet", action="store_true", help="Only print stage summaries, not each example.")
    parser.add_argument("--no-save", action="store_true", help="Don't write a transcript to Tested_results/.")
    return parser.parse_args()


class Tee:
    """Duplicates writes to multiple streams so the run can be printed live
    AND captured into a transcript file at the same time."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()


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
# INFERENCE FUNCTION
# ─────────────────────────────────────────────
def correct_sentence(model, tokenizer, sentence: str) -> str:
    prompt = build_prompt(sentence)

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


def load_test_set(path):
    with open(path, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return [(row["input"], row["output"]) for row in rows]


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
def evaluate_stage(model, tokenizer, stage_name, test_data, limit=None, quiet=False):
    if limit is not None:
        test_data = test_data[:limit]
    n = len(test_data)

    print("\n" + "=" * 60)
    print(f"EVALUATION — {stage_name}  ({n} examples)")
    print("=" * 60)

    correct_count         = 0
    change_correct        = 0
    nochange_correct      = 0
    overcorrection_count  = 0
    change_total   = sum(1 for inp, exp in test_data if inp.strip() != exp.strip())
    nochange_total = n - change_total

    agg_rouge1 = agg_rouge2 = agg_rougeL = 0.0
    agg_gleu   = 0.0
    agg_char_f1 = 0.0
    agg_tok_p = agg_tok_r = agg_tok_f1 = 0.0

    for inp, expected in test_data:
        pred         = correct_sentence(model, tokenizer, inp)
        is_correct   = (pred.strip() == expected.strip())
        needs_change = (inp.strip() != expected.strip())

        if is_correct:
            correct_count += 1
            if needs_change:
                change_correct += 1
            else:
                nochange_correct += 1

        if over_correction_rate(pred, inp, expected):
            overcorrection_count += 1

        r  = rouge_scores(pred, expected)
        g  = gleu_score(pred, expected)
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

        if not quiet:
            status = "✅" if is_correct else "❌"
            print(f"\n{status} {'[CHANGE]' if needs_change else '[NO CHANGE]'}")
            print(f"   INPUT   : {inp}")
            print(f"   PREDICT : {pred}")
            print(f"   EXPECTED: {expected}")
            print(f"   ROUGE-L={r['rougeL']:.3f}  Char-F1={cf:.3f}  GLEU={g:.3f}")

    overall_acc  = correct_count    / n             * 100
    change_acc   = change_correct   / change_total  * 100 if change_total  else 0.0
    nochange_acc = nochange_correct / nochange_total * 100 if nochange_total else 0.0
    over_rate    = overcorrection_count / nochange_total * 100 if nochange_total else 0.0

    print("\n" + "-" * 60)
    print(f"EXACT-MATCH RESULTS — {stage_name}")
    print("-" * 60)
    print(f"  Overall accuracy      : {correct_count}/{n}  →  {overall_acc:.1f}%")
    print(f"  Change-needed accuracy: {change_correct}/{change_total}  →  {change_acc:.1f}%")
    print(f"  No-change accuracy    : {nochange_correct}/{nochange_total}  →  {nochange_acc:.1f}%")
    print(f"  Over-correction rate  : {overcorrection_count}/{nochange_total}  →  {over_rate:.1f}%  (changed correct sentences)")

    print(f"\nCONTINUOUS METRICS — {stage_name}  (avg over all samples)")
    print("-" * 60)
    print(f"  ROUGE-1   (grapheme): {agg_rouge1/n:.4f}")
    print(f"  ROUGE-2   (grapheme): {agg_rouge2/n:.4f}")
    print(f"  ROUGE-L   (grapheme): {agg_rougeL/n:.4f}")
    print(f"  Sentence GLEU       : {agg_gleu/n:.4f}")
    print(f"  Char-level F1       : {agg_char_f1/n:.4f}")
    print(f"  Token Precision     : {agg_tok_p/n:.4f}")
    print(f"  Token Recall        : {agg_tok_r/n:.4f}")
    print(f"  Token F1            : {agg_tok_f1/n:.4f}")

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

    return {
        "n": n,
        "overall_acc": overall_acc,
        "change_acc": change_acc,
        "nochange_acc": nochange_acc,
        "over_rate": over_rate,
        "rouge1": agg_rouge1 / n,
        "rouge2": agg_rouge2 / n,
        "rougeL": agg_rougeL / n,
        "gleu": agg_gleu / n,
        "char_f1": agg_char_f1 / n,
        "tok_p": agg_tok_p / n,
        "tok_r": agg_tok_r / n,
        "tok_f1": agg_tok_f1 / n,
    }


def print_comparison(results):
    order  = ["stage2", "stage3"]
    stages = [s for s in order if s in results]
    if len(stages) < 2:
        return

    rows = [
        ("Overall accuracy",     "overall_acc",  "%"),
        ("Change accuracy",      "change_acc",   "%"),
        ("No-change accuracy",   "nochange_acc", "%"),
        ("Over-correction rate", "over_rate",    "%"),
        ("ROUGE-L",              "rougeL",       ""),
        ("Char-F1",              "char_f1",      ""),
        ("GLEU",                 "gleu",         ""),
        ("Token F1",             "tok_f1",       ""),
    ]

    print("\n" + "=" * 60)
    print("STAGE COMPARISON")
    print("=" * 60)
    header = f"  {'Metric':<22}" + "".join(f"{s:>14}" for s in stages)
    print(header)
    for label, key, unit in rows:
        line = f"  {label:<22}"
        for s in stages:
            v = results[s][key]
            line += f"{v:>13.1f}%" if unit == "%" else f"{v:>14.4f}"
        print(line)
    print("=" * 60)


def print_metric_guide():
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


# ─────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────
def main():
    args   = parse_args()
    stages = ["stage2", "stage3"] if args.stage == "both" else [args.stage]

    data_dir     = resolve_data_dir(args.data_dir)
    adapter_path = resolve_adapter(args.adapter)

    buffer = None
    if not args.no_save:
        buffer = io.StringIO()
        sys.stdout = Tee(sys.__stdout__, buffer)

    try:
        print(f"🔹 Adapter : {adapter_path}")
        print(f"🔹 Data dir: {data_dir}")
        print(f"🔹 Stages  : {', '.join(stages)}")
        print(f"🔹 Prompt  : {PROMPT_VARIANT}")

        print("🔹 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(SINLLAMA_BASE, local_files_only=True)
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.padding_side = "right"

        print("🔹 Loading pre-merged SinLLaMA base...")
        model, _ = FastLanguageModel.from_pretrained(
            model_name     = SINLLAMA_BASE,
            max_seq_length = MAX_SEQ_LENGTH,
            dtype          = None,
            load_in_4bit   = True,
        )

        print("🔹 Loading grammar adapter...")
        model = PeftModel.from_pretrained(model, adapter_path)
        FastLanguageModel.for_inference(model)
        model.eval()
        print("✅ Model ready.\n")

        results = {}
        for stage in stages:
            data_path = os.path.join(data_dir, TEST_SETS[stage])
            test_data = load_test_set(data_path)
            results[stage] = evaluate_stage(
                model, tokenizer, stage, test_data,
                limit=args.limit, quiet=args.quiet,
            )

        print_comparison(results)
        print_metric_guide()
    finally:
        if buffer is not None:
            sys.stdout = sys.__stdout__
            results_dir = os.path.join(BASE_DIR, "Tested_results")
            os.makedirs(results_dir, exist_ok=True)
            adapter_name = os.path.basename(os.path.normpath(adapter_path))
            timestamp    = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            out_path = os.path.join(
                results_dir,
                f"{adapter_name}_{PROMPT_VARIANT}_{'-'.join(stages)}_{timestamp}.md",
            )
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(buffer.getvalue())
            print(f"📝 Transcript saved to {out_path}")


if __name__ == "__main__":
    main()
