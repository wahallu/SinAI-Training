# ================================================================
# SinhalaJournal-LLM
# STYLE ADAPTER -- EVAL PROMPT vs PRODUCTION PROMPT COMPARISON
#
# Purpose:
#   test_style.py measures v13 quality using the eval/training prompt
#   shape (train_style.py::format_prompt()). That is a development
#   evaluation, not proof that production scores the same way, because
#   the live SinAI system builds its prompt in a different file
#   (work/tasks/style.py::prompt_style()). If the two ever drift apart,
#   test_style.py's numbers stop describing what a real user gets.
#
#   This script closes that gap without retraining anything:
#     1. Import the REAL production prompt builder from
#        work/tasks/style.py (not a hand-copied duplicate -- so this
#        script can't itself go stale the way a copy could).
#     2. Sample the same 75 test cases test_style.py uses (15 articles
#        x 5 styles, same dataset, same SEED=42 shuffle).
#     3. Run the existing v13 adapter on every case TWICE: once with
#        the eval/training prompt, once with the production prompt.
#     4. Score both the same way test_style.py does (fact-preservation
#        vs source + ROUGE vs human reference), and diff the two
#        prompt strings byte-for-byte for every case so any future
#        drift between train_style.py and tasks/style.py is caught
#        immediately, not just inferred from a score gap.
#
# NOTE:
#   As of this writing, work/tasks/style.py's docstring states its
#   prompt shape is kept byte-for-byte in sync with train_style.py by
#   hand. This script verifies that claim empirically instead of
#   trusting the comment -- if `prompt_diff_count` in the report is 0,
#   eval and production are provably identical for these 75 cases; if
#   it's non-zero, the printed unified diff shows exactly where they
#   parted ways, and any score gap in the report is real, not noise.
#
# IMPORTANT:
#   If this script cannot be run (no GPU / model box available), do
#   NOT treat test_style.py's style_quality_report.json as a
#   production-prompt benchmark -- it is a development evaluation of
#   the eval/training prompt only. State that explicitly instead.
# ================================================================

import os
import re
import sys
import json
import math
import random
import difflib
import unicodedata
from pathlib import Path
from collections import Counter

import torch

# ---------------------------------------------------------------
# IMPORTANT:
# Do NOT use FastLanguageModel.from_pretrained() here -- same
# TokenizersBackend issue documented in test_style.py.
# ---------------------------------------------------------------

from transformers import (
    LlamaTokenizerFast,
    AutoModelForCausalLM,
)

from peft import PeftModel

# ---------------------------------------------------------------
# Import the REAL production prompt builder.
#
# work/serve_sinai.py is run with work/ as cwd and uses bare imports
# ("from tasks.summarizer import ..."). This script lives under
# work/sinllama/scripts/, so work/ is not on sys.path by default --
# add it explicitly so `tasks.style` resolves to the actual file
# serve_sinai.py loads in production, not a copy of it.
# ---------------------------------------------------------------

WORK_DIR = "/home/jovyan/work"

if WORK_DIR not in sys.path:
    sys.path.insert(0, WORK_DIR)

from tasks.style import (
    prompt_style as production_prompt_style,
    STYLE_ID_MAP,
)

# STYLE_ID_MAP is public_name -> style_id (e.g. "formal" ->
# "style_1_formal_news"). The dataset/STYLE_RULES below are keyed by
# style_id, so invert it once.
STYLE_ID_TO_PUBLIC_NAME = {
    style_id: public_name
    for public_name, style_id in STYLE_ID_MAP.items()
}


# ================================================================
# PATHS
# ================================================================

BASE_MODEL = (
    "/home/jovyan/work/sinllama/models/"
    "SinLLaMA-merged-base"
)

ADAPTER_PATH = (
    "/home/jovyan/work/sinllama/models/"
    "adapters/style_sinllama_v13"
)

# Same dataset test_style.py samples from -- see that file's
# DATASET_PATH comment for the QC/lineage note.
DATASET_PATH = (
    "/home/jovyan/style_rewriter/data/"
    "style_dataset_corrected.jsonl"
)

REPORT_PATH = Path(ADAPTER_PATH) / "style_prompt_comparison_report.json"

# ================================================================
# CONFIG
# ================================================================

MAX_SEQ_LENGTH = 4096

GEN_MAX_NEW_TOKENS = 1024

TEMPERATURE = 0.10
TOP_P = 0.85
TOP_K = 50

REPETITION_PENALTY = 1.05

SEED = 42

# Matches test_style.py: 15 articles/style x 5 styles = 75 cases.
SAMPLE_SIZE_PER_STYLE = 15

MIN_OUTPUT_CHARS = 50

PASS_THRESHOLD = 70.0


# ================================================================
# THE STYLE THE ADAPTER TARGETS (eval/training prompt side)
#
# Byte-for-byte match of STYLE_RULES in train_style.py -- this is
# what the adapter was actually fine-tuned on. Kept as a local literal
# (not imported) because train_style.py is a training script, not a
# module meant to be imported at inference time.
# ================================================================

STYLE_RULES = {

    "style_1_formal_news": """
Write the article as a professional Sinhala news report.

- Objective and factual
- Clear journalistic Sinhala
- Most important information first
- Formal vocabulary
- No personal opinion
- No unnecessary commentary
- Preserve all important facts
- Preserve names, numbers, dates and places
- Do not invent information
""",

    "style_2_editorial": """
Rewrite the article in a Sinhala editorial / analytical style.

- Analytical and reflective tone
- Discuss the significance of the reported facts
- Use natural editorial Sinhala
- May use phrases such as "සැලකිය යුතුය" or "සමස්තයක් ලෙස"
  only when they naturally fit
- Do not invent facts
- Do not exaggerate
- Preserve names, numbers, dates and places
- Preserve the factual basis of the original article
""",

    "style_3_sports": """
Rewrite the article in a Sinhala sports-news style.

- Energetic but professional
- Emphasize important actions, events and outcomes
- Use natural sports journalism vocabulary where appropriate
- Do not force sports terminology into non-sports stories
- Do not invent sporting events or actions
- Preserve all facts, names, numbers and dates
""",

    "style_4_youth": """
Rewrite the article in a modern, accessible Sinhala style
suitable for younger readers.

- Conversational but grammatically correct Sinhala
- Simple and clear sentences
- Engaging tone
- Natural modern vocabulary
- Do not force greetings into every article
- Do not use "යාලුවනේ" unless it naturally fits
- Do not add information
- Preserve names, numbers, dates and factual details
""",

    "style_5_feature": """
Rewrite the article in a Sinhala feature-writing style.

- Engaging narrative presentation
- More descriptive language where appropriate
- Smooth transitions
- Human-interest perspective where supported by the source
- Do not invent scenes, emotions or events
- Do not fabricate background information
- Preserve all factual information
"""
}

STYLE_NAMES = {
    "style_1_formal_news": "FORMAL NEWS",
    "style_2_editorial": "EDITORIAL",
    "style_3_sports": "SPORTS",
    "style_4_youth": "YOUTH",
    "style_5_feature": "FEATURE",
}

COMMON_RULES = (
    "1. Preserve the meaning of every factual statement.\n"
    "2. Do not add facts that are absent from the source.\n"
    "3. Do not remove important factual information.\n"
    "4. Preserve all person names exactly.\n"
    "5. Preserve all organization names exactly.\n"
    "6. Preserve all locations exactly.\n"
    "7. Preserve numbers and numerical values exactly.\n"
    "8. Preserve dates exactly.\n"
    "9. Preserve quoted text exactly.\n"
    "10. Preserve gender-marked honorifics such as "
    "මහතා / මහත්මිය.\n"
    "11. Do not invent durations, measurements or statistics.\n"
    "12. Do not translate proper names unnecessarily.\n"
    "13. Do not introduce unrelated information.\n"
    "14. Change the writing STYLE, not the underlying facts.\n"
    "15. Use natural Sinhala grammar and morphology.\n"
    "16. Do not deliberately replace correct Sinhala words with "
    "different words merely to make the sentence different.\n"
)


# ================================================================
# SEED
# ================================================================

def set_seed(seed=42):

    random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ================================================================
# LOAD TOKENIZER / MODEL
#
# Identical to test_style.py -- see that file's comments for why
# Transformers + PEFT is used instead of FastLanguageModel, and why
# the input embedding table is resized separately from lm_head.
# ================================================================

def load_tokenizer():

    print("\n🔹 Loading LlamaTokenizerFast explicitly...")

    tokenizer = LlamaTokenizerFast.from_pretrained(
        BASE_MODEL,
        local_files_only=True,
    )

    tokenizer.padding_side = "left"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"   Vocab size : {len(tokenizer):,}")
    print(f"   EOS token  : {tokenizer.eos_token!r}")
    print(f"   EOS ID     : {tokenizer.eos_token_id}")
    print(f"   PAD token  : {tokenizer.pad_token!r}")
    print(f"   PAD ID     : {tokenizer.pad_token_id}")

    return tokenizer


def load_model(tokenizer):

    print("\n🔹 Loading base model...")

    if not torch.cuda.is_available():

        print("   ⚠️ CUDA not available. Using CPU.")

        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float32,
            local_files_only=True,
            low_cpu_mem_usage=True,
        )

    else:

        print(f"   GPU: {torch.cuda.get_device_name(0)}")

        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            local_files_only=True,
            low_cpu_mem_usage=True,
        )

    print("   ✅ Base model loaded")

    input_embeddings = model.get_input_embeddings()
    current_vocab_size = input_embeddings.weight.shape[0]
    tokenizer_vocab_size = len(tokenizer)

    if current_vocab_size != tokenizer_vocab_size:

        print(
            f"   ⚠️ Resizing input embeddings only: "
            f"{current_vocab_size} -> {tokenizer_vocab_size} "
            f"(lm_head left untouched)"
        )

        new_embeddings = torch.nn.Embedding(
            tokenizer_vocab_size,
            input_embeddings.weight.shape[1],
            dtype=input_embeddings.weight.dtype,
            device=input_embeddings.weight.device,
        )

        new_embeddings.weight.data[:current_vocab_size] = (
            input_embeddings.weight.data
        )

        model.set_input_embeddings(new_embeddings)

    print("\n🔹 Loading LoRA adapter...")

    model = PeftModel.from_pretrained(
        model,
        ADAPTER_PATH,
        local_files_only=True,
        is_trainable=False,
    )

    print("   ✅ Adapter loaded")

    model.eval()

    return model


# ================================================================
# LOAD EVALUATION DATASET -- same 75 cases as test_style.py
# ================================================================

def load_style_samples():
    """Same grouping/QC/shuffle as test_style.py::load_style_samples()
    (same DATASET_PATH, same SEED=42, same SAMPLE_SIZE_PER_STYLE=15),
    so this script scores the identical 75 test cases -- the only
    variable being compared is the prompt shape, not the sample.
    """

    by_style = {style_id: [] for style_id in STYLE_RULES}
    seen = set()

    with open(DATASET_PATH, "r", encoding="utf-8") as f:

        for line_number, line in enumerate(f, 1):

            line = line.strip()

            if not line:
                continue

            try:
                rec = json.loads(line)

            except json.JSONDecodeError as e:

                print(f"⚠️ Invalid JSON at line {line_number}: {e}")
                continue

            style_id = rec.get("style")

            if style_id not in STYLE_RULES:
                continue

            content = str(rec.get("content") or "").strip()
            reference = str(rec.get("rewritten_text") or "").strip()

            if not content or not reference:
                continue

            if len(reference) < MIN_OUTPUT_CHARS:
                continue

            key = (style_id, content)

            if key in seen:
                continue

            seen.add(key)

            by_style[style_id].append({
                "content": content,
                "reference": reference,
                "url": rec.get("url"),
                "category": rec.get("category"),
            })

    rng = random.Random(SEED)

    samples = {}

    for style_id, rows in by_style.items():

        rng.shuffle(rows)

        samples[style_id] = rows[:SAMPLE_SIZE_PER_STYLE]

    return samples


# ================================================================
# BUILD BOTH PROMPT VARIANTS
# ================================================================

def build_eval_prompt(style_id, article):
    """Eval/training prompt -- byte-for-byte match of
    train_style.py::format_prompt(instruction, article, "").
    """

    instruction = STYLE_RULES[style_id].strip()

    return (
        "### Instruction:\n"
        f"{instruction}\n\n"

        "### FACT PRESERVATION RULES:\n"
        f"{COMMON_RULES}\n"

        "### Input:\n"
        f"{article.strip()}\n\n"

        "### Response:\n"
    )


def build_production_prompt(style_id, article):
    """The prompt production actually sends, via the real
    work/tasks/style.py::prompt_style() -- not a hand-copied
    duplicate.
    """

    public_name = STYLE_ID_TO_PUBLIC_NAME[style_id]

    return production_prompt_style(article, style=public_name)


def diff_prompts(eval_prompt, production_prompt):
    """Returns (are_equal, unified_diff_text). Empty diff text when
    the two prompts are identical.
    """

    if eval_prompt == production_prompt:
        return True, ""

    diff = "\n".join(
        difflib.unified_diff(
            eval_prompt.splitlines(),
            production_prompt.splitlines(),
            fromfile="eval_prompt",
            tofile="production_prompt",
            lineterm="",
        )
    )

    return False, diff


# ================================================================
# GENERATE (takes a prebuilt prompt string directly, so it can be
# reused for either prompt variant)
# ================================================================

@torch.inference_mode()
def generate_from_prompt(model, tokenizer, prompt):

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_SEQ_LENGTH - GEN_MAX_NEW_TOKENS,
    )

    device = next(model.parameters()).device

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    output_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=GEN_MAX_NEW_TOKENS,
        do_sample=True,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
        repetition_penalty=REPETITION_PENALTY,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=True,
    )

    generated_ids = output_ids[0, input_ids.shape[-1]:]

    output = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    ).strip()

    output = re.sub(
        r"^###\s*Response\s*:\s*",
        "",
        output,
        flags=re.IGNORECASE,
    ).strip()

    output = re.sub(
        r"^###\s*Response\s*",
        "",
        output,
        flags=re.IGNORECASE,
    ).strip()

    return output


# ================================================================
# TEXT NORMALIZATION / SCORING
#
# Identical to test_style.py so scores from this script are directly
# comparable to style_quality_report.json.
# ================================================================

def normalize_text(text):

    text = text.replace("​", "")
    text = text.replace("‌", "")
    text = text.replace("‍", "")
    text = text.replace("﻿", "")

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def extract_numbers(text):

    text = normalize_text(text)

    return re.findall(r"\d+(?:[.,]\d+)?", text)


def tokenize_sinhala(text):

    text = normalize_text(text)

    return re.findall(r"[඀-෿]+|[A-Za-z]+|\d+(?:[.,]\d+)?", text)


STOP_WORDS = {
    "සහ", "හා", "ද", "ය", "වෙත", "වන", "වූ", "වීම", "ඇත", "කර",
    "කළ", "කිරීම", "මෙම", "එම", "එක්", "විසින්", "සඳහා", "ලෙස",
    "සිට", "බව", "ඔහු", "ඇය", "ඔවුන්", "එය", "ඒ", "නම්",
    "පමණක්", "දැනට",
}


def important_tokens(text):

    tokens = tokenize_sinhala(text)

    result = []

    for token in tokens:

        clean = token.strip()

        if not clean:
            continue

        if clean in STOP_WORDS:
            continue

        if re.search(r"\d", clean):
            result.append(clean)
            continue

        if len(clean) <= 2:
            continue

        result.append(clean)

    return result


def score_numbers(original, rewritten):

    original_numbers = extract_numbers(original)
    rewritten_numbers = extract_numbers(rewritten)

    if not original_numbers:
        return 100.0, [], []

    original_counter = Counter(original_numbers)
    rewritten_counter = Counter(rewritten_numbers)

    matched = 0
    total = sum(original_counter.values())

    for number, count in original_counter.items():
        matched += min(count, rewritten_counter.get(number, 0))

    score = (matched / total) * 100

    missing = []

    for number, count in original_counter.items():

        if rewritten_counter.get(number, 0) < count:
            missing.extend(
                [number] * (count - rewritten_counter.get(number, 0))
            )

    extra = []

    for number, count in rewritten_counter.items():

        if number not in original_counter:
            extra.extend([number] * count)

    return score, missing, extra


def score_fact_tokens(original, rewritten):

    original_tokens = important_tokens(original)
    rewritten_tokens = important_tokens(rewritten)

    if not original_tokens:
        return 100.0, [], []

    rewritten_counter = Counter(rewritten_tokens)

    matched = 0
    missing = []

    for token in original_tokens:

        if rewritten_counter[token] > 0:
            matched += 1
            rewritten_counter[token] -= 1
        else:
            missing.append(token)

    score = (matched / len(original_tokens)) * 100

    return score, missing, []


def char_ngrams(text, n=3):

    text = normalize_text(text)

    if len(text) < n:
        return Counter()

    return Counter(text[i:i+n] for i in range(len(text) - n + 1))


def cosine_similarity_counter(a, b):

    if not a or not b:
        return 0.0

    common = set(a) & set(b)

    dot = sum(a[x] * b[x] for x in common)

    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)


def score_content_similarity(original, rewritten):

    original_ngrams = char_ngrams(original, 3)
    rewritten_ngrams = char_ngrams(rewritten, 3)

    similarity = cosine_similarity_counter(original_ngrams, rewritten_ngrams)

    return similarity * 100


def score_style_divergence(original, rewritten):

    norm_original = normalize_text(original)
    norm_rewritten = normalize_text(rewritten)

    is_verbatim_copy = (norm_original == norm_rewritten)

    similarity_ratio = difflib.SequenceMatcher(
        None, norm_original, norm_rewritten,
    ).ratio()

    divergence = (1 - similarity_ratio) * 100

    return divergence, is_verbatim_copy


def score_length(original, rewritten):

    original_length = len(normalize_text(original))
    rewritten_length = len(normalize_text(rewritten))

    if original_length == 0:
        return 0.0

    ratio = rewritten_length / original_length

    if 0.60 <= ratio <= 1.40:

        difference = abs(ratio - 1.0)
        score = max(0, 100 - difference * 200)

    else:

        score = max(0, 100 - abs(ratio - 1.0) * 100)

    return score


def score_extra_numbers(original, rewritten):

    original_numbers = Counter(extract_numbers(original))
    rewritten_numbers = Counter(extract_numbers(rewritten))

    extra = 0

    for number, count in rewritten_numbers.items():

        allowed = original_numbers.get(number, 0)

        if count > allowed:
            extra += count - allowed

    total_rewritten = sum(rewritten_numbers.values())

    if total_rewritten == 0:
        return 100.0, []

    if extra == 0:
        return 100.0, []

    penalty = (extra / total_rewritten) * 100

    score = max(0, 100 - penalty * 2)

    extras = []

    for number, count in rewritten_numbers.items():

        allowed = original_numbers.get(number, 0)

        if count > allowed:
            extras.extend([number] * (count - allowed))

    return score, extras


def score_quality(rewritten):

    text = normalize_text(rewritten)

    if not text:
        return 0.0

    if len(text) < 100:
        return 30.0

    words = tokenize_sinhala(text)

    if len(words) >= 8:

        repeated = 0

        for i in range(len(words) - 5):

            phrase1 = words[i:i+3]
            phrase2 = words[i+3:i+6]

            if phrase1 == phrase2:
                repeated += 1

        if repeated >= 3:
            return 30.0

    return 100.0


def grapheme_tokenize(text):

    tokens = []
    chars = list(text)
    i = 0

    while i < len(chars):

        cluster = chars[i]
        i += 1

        while i < len(chars) and unicodedata.combining(chars[i]):
            cluster += chars[i]
            i += 1

        if cluster.strip():
            tokens.append(cluster)

    return tokens


def reference_rouge_scores(hypothesis, reference):

    def ngrams(tokens, n):
        return Counter(
            tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)
        )

    def lcs_length(a, b):

        m, n = len(a), len(b)

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

    hyp_toks = grapheme_tokenize(hypothesis)
    ref_toks = grapheme_tokenize(reference)

    if not hyp_toks or not ref_toks:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    h1 = ngrams(hyp_toks, 1)
    r1 = ngrams(ref_toks, 1)
    c1 = sum((h1 & r1).values())
    prec1 = c1 / len(hyp_toks)
    rec1 = c1 / len(ref_toks)
    f1_1 = 2*prec1*rec1/(prec1+rec1) if (prec1+rec1) else 0.0

    h2 = ngrams(hyp_toks, 2)
    r2 = ngrams(ref_toks, 2)
    c2 = sum((h2 & r2).values())
    prec2 = c2 / max(len(hyp_toks)-1, 1)
    rec2 = c2 / max(len(ref_toks)-1, 1)
    f1_2 = 2*prec2*rec2/(prec2+rec2) if (prec2+rec2) else 0.0

    lcs = lcs_length(hyp_toks, ref_toks)
    precL = lcs / len(hyp_toks)
    recL = lcs / len(ref_toks)
    f1_L = 2*precL*recL/(precL+recL) if (precL+recL) else 0.0

    return {"rouge1": f1_1, "rouge2": f1_2, "rougeL": f1_L}


def calculate_correctness(original, rewritten):

    number_score, missing_numbers, extra_numbers = score_numbers(
        original, rewritten,
    )

    fact_score, missing_facts, _ = score_fact_tokens(original, rewritten)

    similarity_score = score_content_similarity(original, rewritten)

    length_score = score_length(original, rewritten)

    quality_score = score_quality(rewritten)

    extra_number_score, extra_numbers_2 = score_extra_numbers(
        original, rewritten,
    )

    style_divergence, is_verbatim_copy = score_style_divergence(
        original, rewritten,
    )

    number_final = number_score * 0.75 + extra_number_score * 0.25

    final_score = (
        number_final * 0.30
        + fact_score * 0.35
        + similarity_score * 0.20
        + length_score * 0.10
        + quality_score * 0.05
    )

    return {
        "overall": round(final_score, 2),
        "numbers": round(number_final, 2),
        "facts": round(fact_score, 2),
        "content_similarity": round(similarity_score, 2),
        "length": round(length_score, 2),
        "quality": round(quality_score, 2),
        "missing_numbers": list(dict.fromkeys(missing_numbers)),
        "extra_numbers": list(
            dict.fromkeys(extra_numbers + extra_numbers_2)
        ),
        "missing_facts": list(dict.fromkeys(missing_facts)),
        "style_divergence": round(style_divergence, 2),
        "is_verbatim_copy": is_verbatim_copy,
    }


ZERO_SCORE = {
    "overall": 0.0, "numbers": 0.0, "facts": 0.0,
    "content_similarity": 0.0, "length": 0.0, "quality": 0.0,
    "missing_numbers": [], "extra_numbers": [], "missing_facts": [],
    "style_divergence": 0.0, "is_verbatim_copy": False,
}

ZERO_ROUGE = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}


# ================================================================
# ONE VARIANT'S SCORE FOR ONE SAMPLE
# ================================================================

def score_variant(model, tokenizer, prompt, article, reference):

    try:
        rewritten = generate_from_prompt(model, tokenizer, prompt)

    except Exception as e:

        print(f"      ❌ Generation failed: {type(e).__name__}: {e}")
        rewritten = ""

    if not rewritten:
        return "", dict(ZERO_SCORE), dict(ZERO_ROUGE)

    score = calculate_correctness(article, rewritten)
    rouge = reference_rouge_scores(rewritten, reference)

    return rewritten, score, {k: round(v, 4) for k, v in rouge.items()}


# ================================================================
# MAIN
# ================================================================

def main():

    set_seed(SEED)

    print("\n" + "=" * 88)
    print("  SinhalaJournal-LLM | STYLE ADAPTER: EVAL vs PRODUCTION PROMPT")
    print("=" * 88)

    print(
        f"\nRunning v13 on the same {SAMPLE_SIZE_PER_STYLE * len(STYLE_RULES)} "
        f"test cases test_style.py uses, twice each:"
    )
    print("  (1) eval/training prompt  -- train_style.py::format_prompt()")
    print("  (2) production prompt     -- tasks.style.prompt_style() (imported live)")

    print("\n🔍 Checking paths...")

    for label, path in (
        ("Base model", BASE_MODEL),
        ("Adapter", ADAPTER_PATH),
        ("Dataset", DATASET_PATH),
    ):
        if not Path(path).exists():
            raise FileNotFoundError(f"{label} not found:\n{path}")

        print(f"   ✅ {label} found")

    tokenizer = load_tokenizer()
    model = load_model(tokenizer)

    print("\n📂 Sampling dataset:")
    print(f"   {DATASET_PATH}")

    dataset_samples = load_style_samples()

    total_cases = sum(len(rows) for rows in dataset_samples.values())

    print(f"\n   Total test cases: {total_cases} (x2 generations each = "
          f"{total_cases * 2} total)")

    # ------------------------------------------------------------
    # RUN BOTH VARIANTS ON EVERY CASE
    # ------------------------------------------------------------

    style_records = {style_id: [] for style_id in STYLE_RULES}
    all_records = []
    prompt_diff_count = 0
    first_diff_shown = False

    for style_id in STYLE_RULES:

        style_name = STYLE_NAMES[style_id]
        rows = dataset_samples[style_id]

        print("\n\n" + "=" * 88)
        print(f"  {style_name}  ({len(rows)} articles)")
        print("=" * 88)

        for idx, row in enumerate(rows, start=1):

            article = row["content"]
            reference = row["reference"]

            eval_prompt = build_eval_prompt(style_id, article)
            production_prompt = build_production_prompt(style_id, article)

            prompts_equal, prompt_diff = diff_prompts(
                eval_prompt, production_prompt,
            )

            if not prompts_equal:

                prompt_diff_count += 1

                if not first_diff_shown:

                    print("\n   🚨 PROMPT MISMATCH detected -- eval and "
                          "production prompts differ for this case:")
                    print(prompt_diff)

                    first_diff_shown = True

            eval_output, eval_score, eval_rouge = score_variant(
                model, tokenizer, eval_prompt, article, reference,
            )

            production_output, production_score, production_rouge = (
                score_variant(
                    model, tokenizer, production_prompt, article, reference,
                )
            )

            record = {
                "style": style_id,
                "url": row["url"],
                "category": row["category"],
                "prompts_equal": prompts_equal,
                "eval": {
                    "rewritten": eval_output,
                    "score": eval_score,
                    "rouge": eval_rouge,
                },
                "production": {
                    "rewritten": production_output,
                    "score": production_score,
                    "rouge": production_rouge,
                },
                "reference": reference,
            }

            style_records[style_id].append(record)
            all_records.append(record)

            gap = production_score["overall"] - eval_score["overall"]

            flag = "" if prompts_equal else "  🚨 PROMPT DIFF"

            print(
                f"   [{idx:>2}/{len(rows)}] "
                f"eval={eval_score['overall']:>6.1f}%  "
                f"prod={production_score['overall']:>6.1f}%  "
                f"gap={gap:>+6.1f}pp"
                f"{flag}"
            )

    # ============================================================
    # AGGREGATE
    # ============================================================

    def avg(records, variant, path_a, path_b=None):

        if not records:
            return 0.0

        if path_b is None:
            values = [r[variant][path_a] for r in records]
        else:
            values = [r[variant][path_a][path_b] for r in records]

        return sum(values) / len(values)

    style_summary = {}

    for style_id, records in style_records.items():

        n = len(records)

        if n == 0:
            style_summary[style_id] = None
            continue

        style_summary[style_id] = {
            "n": n,
            "eval_overall": round(avg(records, "eval", "score", "overall"), 2),
            "production_overall": round(
                avg(records, "production", "score", "overall"), 2
            ),
            "eval_rougeL": round(avg(records, "eval", "rouge", "rougeL"), 4),
            "production_rougeL": round(
                avg(records, "production", "rouge", "rougeL"), 4
            ),
            "prompt_diffs": sum(
                1 for r in records if not r["prompts_equal"]
            ),
        }

    n_total = len(all_records)

    overall_eval = avg(all_records, "eval", "score", "overall") if n_total else 0.0
    overall_production = (
        avg(all_records, "production", "score", "overall") if n_total else 0.0
    )
    overall_eval_rougeL = (
        avg(all_records, "eval", "rouge", "rougeL") if n_total else 0.0
    )
    overall_production_rougeL = (
        avg(all_records, "production", "rouge", "rougeL") if n_total else 0.0
    )

    # ============================================================
    # PRINT COMPARISON REPORT
    # ============================================================

    print("\n\n" + "=" * 96)
    print("  EVAL vs PRODUCTION PROMPT COMPARISON")
    print("=" * 96)

    print(
        f"\n  {'STYLE':<20}{'N':>4}{'EVAL%':>9}{'PROD%':>9}"
        f"{'GAP':>8}{'DIFFS':>8}"
    )
    print("  " + "-" * 60)

    for style_id in STYLE_RULES:

        s = style_summary[style_id]

        if s is None:
            continue

        gap = s["production_overall"] - s["eval_overall"]

        print(
            f"  {STYLE_NAMES[style_id]:<20}"
            f"{s['n']:>4}"
            f"{s['eval_overall']:>8.2f}%"
            f"{s['production_overall']:>8.2f}%"
            f"{gap:>+7.2f}p"
            f"{s['prompt_diffs']:>8}"
        )

    print("  " + "-" * 60)

    overall_gap = overall_production - overall_eval

    print(
        f"  {'OVERALL':<20}"
        f"{n_total:>4}"
        f"{overall_eval:>8.2f}%"
        f"{overall_production:>8.2f}%"
        f"{overall_gap:>+7.2f}p"
        f"{prompt_diff_count:>8}"
    )

    print("\n" + "=" * 96)

    if prompt_diff_count == 0:

        print(
            "\n  ✅ Eval and production prompts were BYTE-FOR-BYTE IDENTICAL "
            f"on all {n_total} test cases. Any score gap below reflects "
            "sampling stochasticity (do_sample=True), not a prompt "
            "mismatch -- test_style.py's report is representative of "
            "production for this adapter version."
        )

    else:

        print(
            f"\n  🚨 Eval and production prompts DIFFERED on "
            f"{prompt_diff_count}/{n_total} cases. test_style.py's "
            "style_quality_report.json is a development evaluation only "
            "and does NOT represent what production users see -- see the "
            "diff(s) printed above and tasks/style.py's docstring for "
            "what changed."
        )

    print(f"\n  Eval prompt        overall accuracy : {overall_eval:.2f}%")
    print(f"  Production prompt  overall accuracy : {overall_production:.2f}%")
    print(f"  Gap (prod - eval)                    : {overall_gap:+.2f}pp")
    print(f"  Eval prompt        ROUGE-L vs ref    : {overall_eval_rougeL:.4f}")
    print(f"  Production prompt  ROUGE-L vs ref    : {overall_production_rougeL:.4f}")

    # ============================================================
    # SAVE JSON REPORT
    # ============================================================

    report = {
        "base_model": BASE_MODEL,
        "adapter": ADAPTER_PATH,
        "dataset": DATASET_PATH,
        "sample_size_per_style": SAMPLE_SIZE_PER_STYLE,
        "seed": SEED,
        "generation": {
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "top_k": TOP_K,
            "repetition_penalty": REPETITION_PENALTY,
            "max_new_tokens": GEN_MAX_NEW_TOKENS,
        },
        "prompt_diff_count": prompt_diff_count,
        "style_summary": style_summary,
        "overall": {
            "n": n_total,
            "eval_accuracy": round(overall_eval, 2),
            "production_accuracy": round(overall_production, 2),
            "gap": round(overall_gap, 2),
            "eval_rougeL": round(overall_eval_rougeL, 4),
            "production_rougeL": round(overall_production_rougeL, 4),
        },
        "samples": all_records,
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n💾 Report saved:")
    print(f"   {REPORT_PATH}")

    print("\n✅ Eval vs production prompt comparison completed.")


if __name__ == "__main__":
    main()
