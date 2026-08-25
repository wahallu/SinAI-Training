# ================================================================
# SinhalaJournal-LLM
# STYLE ADAPTER QUALITY EVALUATION
#
# Purpose:
#   Sample real articles per style from the training dataset,
#   generate a rewrite for each with the trained adapter, and score:
#
#     1. Fact preservation vs the SOURCE article (numbers, names/
#        places/dates, content similarity, length, output quality,
#        style divergence -- unchanged from the original single-
#        article correctness test).
#     2. Style fidelity vs the human-written REFERENCE rewrite
#        (ROUGE-1/2/L), since fact preservation alone can't tell a
#        genuine style rewrite from a verbatim copy.
#
#   Produces a per-style + overall quality report with an overall
#   accuracy percentage, printed to stdout and saved as JSON.
#
# IMPORTANT:
#   "Correctness" here measures whether facts survived the rewrite,
#   not whether the rewrite is stylistically beautiful.
#
#   5 styles:
#       1. Formal News
#       2. Editorial
#       3. Sports
#       4. Youth
#       5. Feature
#
# Tokenizer:
#   Explicit LlamaTokenizerFast loading.
#   This avoids the Unsloth TokenizersBackend error.
# ================================================================

import os
import re
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
# Do NOT use FastLanguageModel.from_pretrained() here.
# It tries to reload AutoTokenizer and causes:
#
# ValueError:
# Tokenizer class TokenizersBackend does not exist
#
# ---------------------------------------------------------------

from transformers import (
    LlamaTokenizerFast,
    AutoModelForCausalLM,
)

from peft import PeftModel


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

# The dataset train_style.py's docstring/QC lineage traces back to --
# the same rows (by content/style/rewritten_text) that survive its
# QC filtering and dedup become style_dataset3_corrected_clean.jsonl,
# the 6,605-row set actually trained on. This file is used directly
# here (not the further-cleaned one) so the evaluation covers the
# full labeled set. rewritten_text is the field train_style.py's
# convert_record() treats as ground truth output, so it's also the
# reference used for ROUGE below (NOT corrected_text -- that field
# exists in this file but was never what the adapter was trained to
# reproduce).
DATASET_PATH = (
    "/home/jovyan/style_rewriter/data/"
    "style_dataset_corrected.jsonl"
)

REPORT_PATH = Path(ADAPTER_PATH) / "style_quality_report.json"

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

# Articles evaluated per style; 5 styles -> SAMPLE_SIZE_PER_STYLE * 5
# total generations. Runtime scales linearly with this -- lower it
# (e.g. 3) for a quick smoke test, raise it for a more statistically
# stable report.
SAMPLE_SIZE_PER_STYLE = 15

# Mirrors train_style.py's minimum output length QC.
MIN_OUTPUT_CHARS = 50

# Overall score at/above which a generation counts as a "pass".
PASS_THRESHOLD = 70.0


# ================================================================
# THE STYLE THE ADAPTER TARGETS
#
# NOTE:
# These must byte-for-byte match STYLE_RULES in train_style.py --
# the adapter was fine-tuned on this exact English instruction text
# via format_prompt(). An earlier version of this file used
# Sinhala-language rules with a different "### INPUT ARTICLE:" /
# "### RESPONSE:" prompt shape the adapter never saw during
# training, which caused every generation to degenerate into a
# verbatim copy of the input (0% style divergence, see
# score_style_divergence()) while still scoring "100% correctness"
# because a copy trivially preserves every fact.
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


# ================================================================
# COMMON INSTRUCTION
#
# Byte-for-byte match of the "### FACT PRESERVATION RULES:" block
# built inline by format_prompt() in train_style.py.
# ================================================================

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
# LOAD TOKENIZER
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


# ================================================================
# LOAD MODEL
#
# IMPORTANT:
# We deliberately use Transformers + PEFT instead of:
#
# FastLanguageModel.from_pretrained()
#
# because your SinLLaMA tokenizer config causes Unsloth's internal
# tokenizer loader to throw TokenizersBackend.
# ================================================================

def load_model(tokenizer):

    print("\n🔹 Loading base model...")

    if not torch.cuda.is_available():

        print("   ⚠️ CUDA not available. Using CPU.")

        dtype = torch.float32

        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=dtype,
            local_files_only=True,
            low_cpu_mem_usage=True,
        )

    else:

        print(
            f"   GPU: {torch.cuda.get_device_name(0)}"
        )

        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            local_files_only=True,
            low_cpu_mem_usage=True,
        )

    print("   ✅ Base model loaded")

    # ------------------------------------------------------------
    # Some adapter checkpoints (e.g. style_sinllama_v13) were saved
    # with embed_tokens resized to the tokenizer's vocab size
    # (139,337) while lm_head was left at the original size
    # (139,336) -- an asymmetric resize, not a tied-embeddings one.
    # model.resize_token_embeddings() resizes BOTH input and output
    # embeddings together, which would create a *new* mismatch on
    # lm_head, so we grow only the input embedding table here to
    # exactly match what the checkpoint expects.
    # ------------------------------------------------------------

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
# LOAD EVALUATION DATASET
# ================================================================

def load_style_samples():
    """Groups dataset rows by style, applies the same minimal QC as
    train_style.py's convert_record() (non-empty content/rewritten_text,
    output length >= 50 chars), dedupes identical (style, content) rows,
    then samples SAMPLE_SIZE_PER_STYLE rows per style for evaluation.

    NOTE: train_style.py's article-level train/val split is randomized
    at training time and never persisted, so there is no held-out set
    to evaluate against here. Scores below measure fact-preservation
    and style-fidelity against the human reference on labeled data,
    not generalization to unseen articles.
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
# FORMAT PROMPT
# ================================================================

def build_prompt(style_id, article):

    instruction = STYLE_RULES[style_id].strip()

    # Mirrors train_style.py's format_prompt(instruction, article, "")
    # exactly, including the trailing newline after "### Response:" --
    # that newline is where every training label began, so dropping
    # it (e.g. via a trailing .strip()) shifts the prompt out of
    # distribution.
    prompt = (
        "### Instruction:\n"
        f"{instruction}\n\n"

        "### FACT PRESERVATION RULES:\n"
        f"{COMMON_RULES}\n"

        "### Input:\n"
        f"{article.strip()}\n\n"

        "### Response:\n"
    )

    return prompt


# ================================================================
# GENERATE
# ================================================================

@torch.inference_mode()
def generate_rewrite(
    model,
    tokenizer,
    style_id,
    article,
):

    prompt = build_prompt(
        style_id,
        article,
    )

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

    generated_ids = output_ids[
        0,
        input_ids.shape[-1]:
    ]

    output = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    ).strip()

    # ------------------------------------------------------------
    # Remove accidental prompt markers
    # ------------------------------------------------------------

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
# TEXT NORMALIZATION
# ================================================================

def normalize_text(text):

    text = text.replace("​", "")
    text = text.replace("‌", "")
    text = text.replace("‍", "")
    text = text.replace("﻿", "")

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


# ================================================================
# EXTRACT FACTUAL TOKENS
#
# We specifically inspect:
#   - numbers
#   - percentages
#   - dates
#   - years
# ================================================================

def extract_numbers(text):

    text = normalize_text(text)

    values = []

    # Numbers such as:
    # 50
    # 2026
    # 16
    # 3
    for value in re.findall(
        r"\d+(?:[.,]\d+)?",
        text,
    ):
        values.append(value)

    return values


# ================================================================
# EXTRACT IMPORTANT FACTUAL PHRASES
#
# This article is Sinhala, therefore word-level exact matching
# alone is too strict. We use character n-grams + important tokens.
# ================================================================

def tokenize_sinhala(text):

    text = normalize_text(text)

    return re.findall(
        r"[඀-෿]+|[A-Za-z]+|\d+(?:[.,]\d+)?",
        text,
    )


# ================================================================
# IMPORTANT KEYWORD EXTRACTION
# ================================================================

STOP_WORDS = {
    "සහ",
    "හා",
    "ද",
    "ය",
    "වෙත",
    "වන",
    "වූ",
    "වීම",
    "ඇත",
    "කර",
    "කළ",
    "කිරීම",
    "මෙම",
    "එම",
    "එක්",
    "විසින්",
    "සඳහා",
    "ලෙස",
    "සිට",
    "බව",
    "ඔහු",
    "ඇය",
    "ඔවුන්",
    "එය",
    "ඒ",
    "නම්",
    "පමණක්",
    "දැනට",
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

        # Keep numbers
        if re.search(r"\d", clean):

            result.append(clean)

            continue

        # Ignore very short Sinhala particles
        if len(clean) <= 2:
            continue

        result.append(clean)

    return result


# ================================================================
# NUMBER CORRECTNESS
# ================================================================

def score_numbers(original, rewritten):

    original_numbers = extract_numbers(original)
    rewritten_numbers = extract_numbers(rewritten)

    if not original_numbers:
        return 100.0, [], []

    original_counter = Counter(
        original_numbers
    )

    rewritten_counter = Counter(
        rewritten_numbers
    )

    matched = 0
    total = sum(original_counter.values())

    for number, count in original_counter.items():

        matched += min(
            count,
            rewritten_counter.get(number, 0)
        )

    score = (
        matched / total
    ) * 100

    missing = []

    for number, count in original_counter.items():

        if rewritten_counter.get(number, 0) < count:

            missing.extend(
                [number] * (
                    count -
                    rewritten_counter.get(number, 0)
                )
            )

    extra = []

    for number, count in rewritten_counter.items():

        if number not in original_counter:

            extra.extend(
                [number] * count
            )

    return score, missing, extra


# ================================================================
# IMPORTANT WORD COVERAGE
# ================================================================

def score_fact_tokens(original, rewritten):

    original_tokens = important_tokens(
        original
    )

    rewritten_tokens = important_tokens(
        rewritten
    )

    if not original_tokens:

        return 100.0, [], []

    rewritten_counter = Counter(
        rewritten_tokens
    )

    matched = 0

    missing = []

    for token in original_tokens:

        if rewritten_counter[token] > 0:

            matched += 1

            rewritten_counter[token] -= 1

        else:

            missing.append(token)

    score = (
        matched /
        len(original_tokens)
    ) * 100

    return score, missing, []


# ================================================================
# CHARACTER N-GRAM SIMILARITY
#
# This helps detect whether the rewrite still contains the same
# factual content even when Sinhala words have been rearranged.
# ================================================================

def char_ngrams(text, n=3):

    text = normalize_text(text)

    if len(text) < n:
        return Counter()

    return Counter(
        text[i:i+n]
        for i in range(
            len(text) - n + 1
        )
    )


def cosine_similarity_counter(a, b):

    if not a or not b:
        return 0.0

    common = set(a) & set(b)

    dot = sum(
        a[x] * b[x]
        for x in common
    )

    norm_a = math.sqrt(
        sum(
            value * value
            for value in a.values()
        )
    )

    norm_b = math.sqrt(
        sum(
            value * value
            for value in b.values()
        )
    )

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (
        norm_a * norm_b
    )


def score_content_similarity(
    original,
    rewritten,
):

    original_ngrams = char_ngrams(
        original,
        3,
    )

    rewritten_ngrams = char_ngrams(
        rewritten,
        3,
    )

    similarity = cosine_similarity_counter(
        original_ngrams,
        rewritten_ngrams,
    )

    return similarity * 100


# ================================================================
# STYLE DIVERGENCE
#
# The correctness metrics above only measure whether facts were
# preserved -- a model that just copies the input verbatim trivially
# scores 100% on all of them, even though it performed no style
# rewrite at all. This checks whether the output actually differs
# from the input.
# ================================================================

def score_style_divergence(original, rewritten):

    norm_original = normalize_text(original)
    norm_rewritten = normalize_text(rewritten)

    is_verbatim_copy = (
        norm_original == norm_rewritten
    )

    similarity_ratio = difflib.SequenceMatcher(
        None,
        norm_original,
        norm_rewritten,
    ).ratio()

    divergence = (1 - similarity_ratio) * 100

    return divergence, is_verbatim_copy


# ================================================================
# LENGTH PRESERVATION
# ================================================================

def score_length(original, rewritten):

    original_length = len(
        normalize_text(original)
    )

    rewritten_length = len(
        normalize_text(rewritten)
    )

    if original_length == 0:
        return 0.0

    ratio = (
        rewritten_length /
        original_length
    )

    # Ideal around 1.0
    # Accept roughly 60%-140%
    if 0.60 <= ratio <= 1.40:

        # Maximum score at 1.0
        difference = abs(
            ratio - 1.0
        )

        score = max(
            0,
            100 -
            difference * 200
        )

    else:

        score = max(
            0,
            100 -
            abs(ratio - 1.0) * 100
        )

    return score


# ================================================================
# PENALIZE FABRICATED NUMBERS
# ================================================================

def score_extra_numbers(
    original,
    rewritten,
):

    original_numbers = Counter(
        extract_numbers(original)
    )

    rewritten_numbers = Counter(
        extract_numbers(rewritten)
    )

    extra = 0

    for number, count in rewritten_numbers.items():

        allowed = original_numbers.get(
            number,
            0,
        )

        if count > allowed:

            extra += count - allowed

    total_rewritten = sum(
        rewritten_numbers.values()
    )

    if total_rewritten == 0:

        return 100.0, []

    if extra == 0:

        return 100.0, []

    penalty = (
        extra /
        total_rewritten
    ) * 100

    score = max(
        0,
        100 - penalty * 2
    )

    extras = []

    for number, count in rewritten_numbers.items():

        allowed = original_numbers.get(
            number,
            0,
        )

        if count > allowed:

            extras.extend(
                [number] *
                (count - allowed)
            )

    return score, extras


# ================================================================
# REPETITION / GARBAGE CHECK
# ================================================================

def score_quality(rewritten):

    text = normalize_text(
        rewritten
    )

    if not text:

        return 0.0

    # Extremely short output
    if len(text) < 100:

        return 30.0

    # Detect repeated phrase stuttering
    words = tokenize_sinhala(
        text
    )

    if len(words) >= 8:

        repeated = 0

        for i in range(
            len(words) - 5
        ):

            phrase1 = words[i:i+3]
            phrase2 = words[i+3:i+6]

            if phrase1 == phrase2:

                repeated += 1

        if repeated >= 3:

            return 30.0

    return 100.0


# ================================================================
# REFERENCE ROUGE (grapheme-cluster, Sinhala-safe)
#
# IMPORTANT:
# The `rouge_score` pip library breaks on Sinhala Unicode (see
# test_grammar.py / CLAUDE.md). ROUGE here is computed natively on
# Unicode grapheme clusters (base char + combining diacritics grouped
# as one token) instead of relying on that library, mirroring
# test_grammar.py's rouge_scores() exactly.
#
# This measures style-fidelity against the human-written reference
# rewrite -- something the fact-preservation metrics above cannot,
# since a verbatim copy of the input trivially maxes those out.
# ================================================================

def grapheme_tokenize(text):

    tokens = []
    chars = list(text)
    i = 0

    while i < len(chars):

        cluster = chars[i]
        i += 1

        while (
            i < len(chars)
            and unicodedata.combining(chars[i])
        ):
            cluster += chars[i]
            i += 1

        if cluster.strip():
            tokens.append(cluster)

    return tokens


def reference_rouge_scores(hypothesis, reference):

    def ngrams(tokens, n):
        return Counter(
            tuple(tokens[i:i+n])
            for i in range(len(tokens) - n + 1)
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

    # ROUGE-1
    h1 = ngrams(hyp_toks, 1)
    r1 = ngrams(ref_toks, 1)
    c1 = sum((h1 & r1).values())
    prec1 = c1 / len(hyp_toks)
    rec1 = c1 / len(ref_toks)
    f1_1 = 2*prec1*rec1/(prec1+rec1) if (prec1+rec1) else 0.0

    # ROUGE-2
    h2 = ngrams(hyp_toks, 2)
    r2 = ngrams(ref_toks, 2)
    c2 = sum((h2 & r2).values())
    prec2 = c2 / max(len(hyp_toks)-1, 1)
    rec2 = c2 / max(len(ref_toks)-1, 1)
    f1_2 = 2*prec2*rec2/(prec2+rec2) if (prec2+rec2) else 0.0

    # ROUGE-L (LCS)
    lcs = lcs_length(hyp_toks, ref_toks)
    precL = lcs / len(hyp_toks)
    recL = lcs / len(ref_toks)
    f1_L = 2*precL*recL/(precL+recL) if (precL+recL) else 0.0

    return {"rouge1": f1_1, "rouge2": f1_2, "rougeL": f1_L}


# ================================================================
# OVERALL CORRECTNESS
#
# WEIGHTING:
#
#   Numbers          30%
#   Fact tokens      35%
#   Content similarity 20%
#   Length           10%
#   Output quality    5%
#
# This is deliberately NOT BLEU.
# BLEU would unfairly punish legitimate style rewriting.
# It measures FACT PRESERVATION vs the source article -- ROUGE
# against the human reference (computed separately, see
# reference_rouge_scores()) is what measures style fidelity.
# ================================================================

def calculate_correctness(
    original,
    rewritten,
):

    number_score, missing_numbers, extra_numbers = (
        score_numbers(
            original,
            rewritten,
        )
    )

    fact_score, missing_facts, _ = (
        score_fact_tokens(
            original,
            rewritten,
        )
    )

    similarity_score = (
        score_content_similarity(
            original,
            rewritten,
        )
    )

    length_score = score_length(
        original,
        rewritten,
    )

    quality_score = score_quality(
        rewritten,
    )

    extra_number_score, extra_numbers_2 = (
        score_extra_numbers(
            original,
            rewritten,
        )
    )

    style_divergence, is_verbatim_copy = (
        score_style_divergence(
            original,
            rewritten,
        )
    )

    # Average number preservation with hallucination penalty
    number_final = (
        number_score * 0.75
        +
        extra_number_score * 0.25
    )

    final_score = (

        number_final * 0.30

        +

        fact_score * 0.35

        +

        similarity_score * 0.20

        +

        length_score * 0.10

        +

        quality_score * 0.05
    )

    return {

        "overall": round(
            final_score,
            2,
        ),

        "numbers": round(
            number_final,
            2,
        ),

        "facts": round(
            fact_score,
            2,
        ),

        "content_similarity": round(
            similarity_score,
            2,
        ),

        "length": round(
            length_score,
            2,
        ),

        "quality": round(
            quality_score,
            2,
        ),

        "missing_numbers": list(
            dict.fromkeys(
                missing_numbers
            )
        ),

        "extra_numbers": list(
            dict.fromkeys(
                extra_numbers +
                extra_numbers_2
            )
        ),

        "missing_facts": list(
            dict.fromkeys(
                missing_facts
            )
        ),

        "style_divergence": round(
            style_divergence,
            2,
        ),

        "is_verbatim_copy": is_verbatim_copy,
    }


ZERO_SCORE = {
    "overall": 0.0,
    "numbers": 0.0,
    "facts": 0.0,
    "content_similarity": 0.0,
    "length": 0.0,
    "quality": 0.0,
    "missing_numbers": [],
    "extra_numbers": [],
    "missing_facts": [],
    "style_divergence": 0.0,
    "is_verbatim_copy": False,
}

ZERO_ROUGE = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}


# ================================================================
# STATUS BUCKET
# ================================================================

def status_for_score(score):

    if score >= 90:
        return "🟢 EXCELLENT"

    elif score >= 80:
        return "🟢 GOOD"

    elif score >= 70:
        return "🟡 ACCEPTABLE"

    elif score >= 60:
        return "🟠 NEEDS IMPROVEMENT"

    else:
        return "🔴 POOR"


# ================================================================
# MAIN
# ================================================================

def main():

    set_seed(SEED)

    print(
        "\n" +
        "=" * 80
    )

    print(
        "  SinhalaJournal-LLM | "
        "STYLE ADAPTER QUALITY EVALUATION"
    )

    print(
        "=" * 80
    )

    print(
        f"\nSampling up to {SAMPLE_SIZE_PER_STYLE} articles per style "
        f"from the labeled dataset and scoring each rewrite."
    )

    print(
        "\nFact-preservation metrics are measured against the SOURCE "
        "article; ROUGE is measured against the human-written "
        "reference rewrite."
    )

    print(
        "\nGeneration:"
    )

    print(f"  Temperature       : {TEMPERATURE}")
    print(f"  Top-P             : {TOP_P}")
    print(f"  Top-K             : {TOP_K}")
    print(f"  Repetition        : {REPETITION_PENALTY}")
    print(f"  Max new tokens    : {GEN_MAX_NEW_TOKENS}")

    # ------------------------------------------------------------
    # PATH CHECK
    # ------------------------------------------------------------

    print(
        "\n🔍 Checking paths..."
    )

    if not Path(BASE_MODEL).exists():

        raise FileNotFoundError(
            f"Base model not found:\n{BASE_MODEL}"
        )

    print("   ✅ Base model found")

    if not Path(ADAPTER_PATH).exists():

        raise FileNotFoundError(
            f"Adapter not found:\n{ADAPTER_PATH}"
        )

    print("   ✅ Adapter found")

    if not Path(DATASET_PATH).exists():

        raise FileNotFoundError(
            f"Dataset not found:\n{DATASET_PATH}"
        )

    print("   ✅ Dataset found")

    # ------------------------------------------------------------
    # LOAD
    # ------------------------------------------------------------

    tokenizer = load_tokenizer()

    model = load_model(tokenizer)

    # ------------------------------------------------------------
    # SAMPLE DATASET
    # ------------------------------------------------------------

    print(
        "\n📂 Sampling dataset:"
    )

    print(f"   {DATASET_PATH}")

    dataset_samples = load_style_samples()

    for style_id in STYLE_RULES:

        print(
            f"   {style_id:<28} "
            f"{len(dataset_samples[style_id]):>5,} sampled"
        )

    total_generations = sum(
        len(rows) for rows in dataset_samples.values()
    )

    print(f"\n   Total generations: {total_generations:,}")

    # ------------------------------------------------------------
    # EVALUATE EACH STYLE
    # ------------------------------------------------------------

    style_records = {style_id: [] for style_id in STYLE_RULES}
    all_records = []

    for style_id in STYLE_RULES:

        style_name = STYLE_NAMES[style_id]
        rows = dataset_samples[style_id]

        print(
            "\n\n" + "=" * 80
        )

        print(f"  {style_name}  ({len(rows)} articles)")
        print("=" * 80)

        for idx, row in enumerate(rows, start=1):

            article = row["content"]
            reference = row["reference"]

            try:

                rewritten = generate_rewrite(
                    model,
                    tokenizer,
                    style_id,
                    article,
                )

            except Exception as e:

                print(
                    f"\n❌ [{idx}/{len(rows)}] Generation failed: "
                    f"{type(e).__name__}: {e}"
                )

                rewritten = ""

            if not rewritten:

                score = dict(ZERO_SCORE)
                rouge = dict(ZERO_ROUGE)

            else:

                score = calculate_correctness(
                    article,
                    rewritten,
                )

                rouge = reference_rouge_scores(
                    rewritten,
                    reference,
                )

            record = {
                "style": style_id,
                "url": row["url"],
                "category": row["category"],
                "rewritten": rewritten,
                "reference": reference,
                "score": score,
                "rouge": {
                    k: round(v, 4) for k, v in rouge.items()
                },
            }

            style_records[style_id].append(record)
            all_records.append(record)

            flag = ""

            if not rewritten:
                flag = "  ❌ EMPTY"
            elif score["is_verbatim_copy"]:
                flag = "  🚨 VERBATIM COPY"

            print(
                f"   [{idx:>2}/{len(rows)}] "
                f"overall={score['overall']:>6.1f}%  "
                f"rougeL={rouge['rougeL']:.3f}  "
                f"divergence={score['style_divergence']:>5.1f}%"
                f"{flag}"
            )

            # Show the full first generation of each style so it can
            # be spot-checked by eye, not just by the numbers above.
            if idx == 1 and rewritten:

                print("\n   --- sample rewrite ---")
                print(f"   Source : {article[:200]}...")
                print(f"   Output : {rewritten[:400]}"
                      f"{'...' if len(rewritten) > 400 else ''}")
                print(f"   Ref    : {reference[:400]}"
                      f"{'...' if len(reference) > 400 else ''}")
                print("   ----------------------\n")

    # ============================================================
    # AGGREGATE PER STYLE
    # ============================================================

    def avg(records, path_a, path_b=None):

        if not records:
            return 0.0

        if path_b is None:
            values = [r[path_a] for r in records]
        else:
            values = [r[path_a][path_b] for r in records]

        return sum(values) / len(values)

    style_summary = {}

    for style_id, records in style_records.items():

        n = len(records)

        if n == 0:
            style_summary[style_id] = None
            continue

        pass_count = sum(
            1 for r in records
            if r["score"]["overall"] >= PASS_THRESHOLD
        )

        verbatim_count = sum(
            1 for r in records
            if r["score"]["is_verbatim_copy"]
        )

        style_summary[style_id] = {
            "n": n,
            "overall": round(avg(records, "score", "overall"), 2),
            "numbers": round(avg(records, "score", "numbers"), 2),
            "facts": round(avg(records, "score", "facts"), 2),
            "content_similarity": round(
                avg(records, "score", "content_similarity"), 2
            ),
            "length": round(avg(records, "score", "length"), 2),
            "quality": round(avg(records, "score", "quality"), 2),
            "style_divergence": round(
                avg(records, "score", "style_divergence"), 2
            ),
            "rouge1": round(avg(records, "rouge", "rouge1"), 4),
            "rouge2": round(avg(records, "rouge", "rouge2"), 4),
            "rougeL": round(avg(records, "rouge", "rougeL"), 4),
            "pass_rate": round(pass_count / n * 100, 2),
            "verbatim_rate": round(verbatim_count / n * 100, 2),
        }

    # ============================================================
    # AGGREGATE OVERALL
    # ============================================================

    n_total = len(all_records)

    overall_accuracy = (
        avg(all_records, "score", "overall") if n_total else 0.0
    )

    overall_pass_count = sum(
        1 for r in all_records
        if r["score"]["overall"] >= PASS_THRESHOLD
    )

    overall_verbatim_count = sum(
        1 for r in all_records
        if r["score"]["is_verbatim_copy"]
    )

    overall_rouge1 = avg(all_records, "rouge", "rouge1") if n_total else 0.0
    overall_rougeL = avg(all_records, "rouge", "rougeL") if n_total else 0.0

    # ============================================================
    # PRINT QUALITY REPORT
    # ============================================================

    print(
        "\n\n" + "=" * 96
    )

    print("  QUALITY REPORT")
    print("=" * 96)

    print(
        f"\n  {'STYLE':<20}{'N':>4}{'ACCURACY':>11}"
        f"{'ROUGE-L':>10}{'DIVERGENCE':>13}"
        f"{'PASS%':>9}{'VERBATIM%':>11}"
    )

    print("  " + "-" * 92)

    for style_id in STYLE_RULES:

        s = style_summary[style_id]

        if s is None:

            print(
                f"  {STYLE_NAMES[style_id]:<20}"
                f"{'--- no samples available ---':>60}"
            )

            continue

        print(
            f"  {STYLE_NAMES[style_id]:<20}"
            f"{s['n']:>4}"
            f"{s['overall']:>10.2f}%"
            f"{s['rougeL']:>10.3f}"
            f"{s['style_divergence']:>12.2f}%"
            f"{s['pass_rate']:>8.1f}%"
            f"{s['verbatim_rate']:>10.1f}%"
        )

    print("  " + "-" * 92)

    print(
        f"  {'OVERALL':<20}"
        f"{n_total:>4}"
        f"{overall_accuracy:>10.2f}%"
        f"{overall_rougeL:>10.3f}"
        f"{avg(all_records, 'score', 'style_divergence'):>12.2f}%"
        f"{(overall_pass_count / n_total * 100 if n_total else 0):>8.1f}%"
        f"{(overall_verbatim_count / n_total * 100 if n_total else 0):>10.1f}%"
    )

    print(
        "\n" + "=" * 96
    )

    print(f"\n  ⭐ OVERALL ACCURACY : {overall_accuracy:.2f}%")

    print(
        f"     Pass rate (>= {PASS_THRESHOLD:.0f}%) : "
        f"{overall_pass_count}/{n_total} "
        f"({(overall_pass_count / n_total * 100 if n_total else 0):.2f}%)"
    )

    print(
        f"     ROUGE-1 vs reference : {overall_rouge1:.4f}"
    )

    print(
        f"     ROUGE-L vs reference : {overall_rougeL:.4f}"
    )

    if overall_verbatim_count:

        print(
            f"\n  🚨 {overall_verbatim_count}/{n_total} generations were "
            f"VERBATIM COPIES of the input (0% style divergence). "
            f"The accuracy figure above is inflated by these -- a "
            f"copy trivially preserves every fact. Treat any style "
            f"with a non-zero verbatim rate as NOT reliably rewriting "
            f"style."
        )

    print("\n  Per-style status:")

    for style_id in STYLE_RULES:

        s = style_summary[style_id]

        if s is None:
            continue

        print(
            f"    {STYLE_NAMES[style_id]:<20}"
            f"{status_for_score(s['overall'])}"
        )

    # ============================================================
    # SAVE JSON REPORT
    # ============================================================

    report = {

        "base_model": BASE_MODEL,

        "adapter": ADAPTER_PATH,

        "dataset": DATASET_PATH,

        "sample_size_per_style": SAMPLE_SIZE_PER_STYLE,

        "pass_threshold": PASS_THRESHOLD,

        "generation": {

            "temperature": TEMPERATURE,

            "top_p": TOP_P,

            "top_k": TOP_K,

            "repetition_penalty": REPETITION_PENALTY,

            "max_new_tokens": GEN_MAX_NEW_TOKENS,
        },

        "style_summary": style_summary,

        "overall": {

            "n": n_total,

            "accuracy": round(overall_accuracy, 2),

            "pass_count": overall_pass_count,

            "pass_rate": round(
                overall_pass_count / n_total * 100 if n_total else 0.0, 2
            ),

            "verbatim_copies": overall_verbatim_count,

            "rouge1": round(overall_rouge1, 4),

            "rougeL": round(overall_rougeL, 4),
        },

        "samples": all_records,
    }

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\n💾 Report saved:")
    print(f"   {REPORT_PATH}")

    print("\n✅ Style quality evaluation completed.")


# ================================================================
# RUN
# ================================================================

if __name__ == "__main__":
    main()
