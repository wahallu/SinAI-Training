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
# PROMPT BUILDER
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


# ================================================================
# VIVA DEMONSTRATION MODE
#
# This version evaluates ONE fixed article supplied for the final viva.
#
# IMPORTANT:
#   This is a demonstration / case-study evaluation, NOT a statistical
#   benchmark. It is designed to let the panel see:
#
#     1. What changed stylistically?
#     2. Were the important facts preserved?
#     3. Did numbers/dates remain correct?
#     4. Did the output actually change rather than copy the source?
#     5. How similar is the output to a human reference, IF a reference
#        is supplied?
#
# Automatic metrics are supporting evidence. Human evaluation is the
# final evidence for style naturalness and overall usefulness.
#
# The fixed article below is the exact article supplied for the viva.
# ================================================================

VIVA_ARTICLE = """විදේශ විනිමය සංචිතවල පීඩනය සහ ආර්ථික ප්‍රතිසංස්කරණවල ප්‍රතිඵලයක් ලෙස, ශ්‍රී ලංකා රජය විසින් මේ වන විට වාහන ආනයනය සම්බන්ධයෙන් සැලකිය යුතු බදු ප්‍රතිපත්ති වෙනස්කම් ගණනාවක් හඳුන්වා දී ඇත. ඒ අතරින් ප්‍රධානතම වෙනස වන්නේ 2026 මැයි 16 වන දින සිට බලාත්මක කරන ලද සියයට 50ක තාවකාලික අතිරේක බද්දයි. ජනාධිපතිවරයා විසින් රේගු පනත යටතේ නිකුත් කරන ලද මෙම විශේෂ නියෝගය මගී රථ, ජීප් රථ, බස් රථ, භාණ්ඩ ප්‍රවාහන රථ, ගිලන් රථ මෙන්ම විදුලි හා දෙමුහුම් වාහන සඳහා ද අදාළ කර ඇති අතර, යතුරුපැදි, ත්‍රී රෝද රථ සහ වාණිජ කටයුතු සඳහා භාවිතා කරන ආනයනික වාහන පමණක් මෙම නීතියෙන් බැහැර කර තිබේ. මෙම තීරණය කෙටි කාලීන පියවරක් ලෙස ප්‍රකාශයට පත් කර ඇති අතර, ආර්ථිකය යම් තරමක ස්ථාවරත්වයකට පත්වන තෙක් මෙම අතිරේක බද්ද ක්‍රියාත්මක වනු ඇතැයි රජයේ නිලධාරිහු පෙන්වා දෙති."""

# Optional: if you have a human-written reference rewrite for THIS exact
# article/style, paste it here. Leave empty when unavailable.
HUMAN_REFERENCE = {
    "style_1_formal_news": "",
    "style_2_editorial": "",
    "style_3_sports": "",
    "style_4_youth": "",
    "style_5_feature": "",
}

# ---------------------------------------------------------------
# Ground-truth fact checklist for this exact viva article.
#
# These are NOT invented model facts. They are facts explicitly present
# in the user-supplied source article. Each output is checked against
# these facts. This gives the panel a transparent "fact matrix".
# ---------------------------------------------------------------
VIVA_FACTS = [
    ("Cause / context", "විදේශ විනිමය සංචිතවල පීඩනය", ["විදේශ විනිමය සංචිතවල පීඩනය"]),
    ("Cause / context", "ආර්ථික ප්‍රතිසංස්කරණ", ["ආර්ථික ප්‍රතිසංස්කරණ"]),
    ("Policy subject", "වාහන ආනයනය සම්බන්ධ බදු ප්‍රතිපත්ති වෙනස්කම්", ["වාහන ආනයනය"]),
    ("Tax rate", "සියයට 50ක තාවකාලික අතිරේක බද්ද", ["සියයට 50", "50%", "50"]),
    ("Effective date", "2026 මැයි 16", ["2026 මැයි 16", "2026", "මැයි 16"]),
    ("Legal basis", "රේගු පනත", ["රේගු පනත"]),
    ("Issuing authority", "ජනාධිපතිවරයා", ["ජනාධිපතිවරයා"]),
    ("Covered vehicle", "මගී රථ", ["මගී රථ"]),
    ("Covered vehicle", "ජීප් රථ", ["ජීප් රථ"]),
    ("Covered vehicle", "බස් රථ", ["බස් රථ"]),
    ("Covered vehicle", "භාණ්ඩ ප්‍රවාහන රථ", ["භාණ්ඩ ප්‍රවාහන රථ"]),
    ("Covered vehicle", "ගිලන් රථ", ["ගිලන් රථ"]),
    ("Covered vehicle", "විදුලි වාහන", ["විදුලි", "වාහන"]),
    ("Covered vehicle", "දෙමුහුම් වාහන", ["දෙමුහුම්", "වාහන"]),
    ("Excluded vehicle", "යතුරුපැදි", ["යතුරුපැදි"]),
    ("Excluded vehicle", "ත්‍රී රෝද රථ", ["ත්‍රී රෝද රථ"]),
    ("Excluded vehicle", "වාණිජ කටයුතු සඳහා භාවිතා කරන ආනයනික වාහන", ["වාණිජ කටයුතු", "ආනයනික වාහන"]),
    ("Duration", "කෙටි කාලීන පියවරක්", ["කෙටි කාලීන පියවරක්"]),
    ("End condition", "ආර්ථිකය ස්ථාවර වන තෙක්", ["ආර්ථිකය", "ස්ථාවරත්වයකට"]),
]

# A smaller, especially important set for the headline "critical facts".
CRITICAL_FACTS = [
    ("50% tax", ["සියයට 50", "50%"]),
    ("2026 May 16", ["2026 මැයි 16"]),
    ("Customs Act", ["රේගු පනත"]),
    ("President issued order", ["ජනාධිපතිවරයා"]),
    ("Motorcycles excluded", ["යතුරුපැදි"]),
    ("Three-wheelers excluded", ["ත්‍රී රෝද රථ"]),
    ("Commercial-use imported vehicles excluded", ["වාණිජ කටයුතු", "ආනයනික වාහන"]),
    ("Temporary measure", ["කෙටි කාලීන පියවරක්"]),
]

# ---------------------------------------------------------------
# A panel-friendly style score.
#
# This is NOT "accuracy". It is an automatic diagnostic based on
# observable style signals. Human raters must decide whether the
# output genuinely fits the target style.
# ---------------------------------------------------------------

STYLE_SIGNALS = {
    "style_1_formal_news": [
        "වෙත", "බව", "පෙන්වා දෙති", "බලාත්මක", "ප්‍රකාශයට",
        "අදාළ", "සම්බන්ධයෙන්",
    ],
    "style_2_editorial": [
        "සැලකිය යුතුය", "සමස්තයක් ලෙස", "කෙසේ වෙතත්",
        "වැදගත්", "ප්‍රතිපත්තිය", "පියවර",
    ],
    "style_3_sports": [
        "තරග", "ජය", "පරාජය", "ක්‍රීඩා", "ක්‍රීඩක",
        "කණ්ඩායම", "තරගයේ",
    ],
    "style_4_youth": [
        "සරලව", "ඉතාමත්", "අපි", "දැනට", "මේ", "ඔබට",
    ],
    "style_5_feature": [
        "පසුබිම", "කතාව", "සන්දර්භය", "විස්තර", "මානය",
        "අර්ථය", "කෙසේ වෙතත්",
    ],
}

STYLE_SIGNAL_LABELS = {
    "style_1_formal_news": "formal-news signals",
    "style_2_editorial": "editorial/analytical signals",
    "style_3_sports": "sports signals (expected LOW for this non-sports article)",
    "style_4_youth": "accessible/conversational signals",
    "style_5_feature": "feature/narrative signals",
}

# ---------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------

def contains_any(text, options):
    return any(option in text for option in options)

def viva_fact_matrix(source, output):
    rows = []
    for category, fact, variants in VIVA_FACTS:
        source_has = contains_any(source, variants)
        output_has = contains_any(output, variants)
        rows.append({
            "category": category,
            "fact": fact,
            "source": "YES" if source_has else "NO",
            "output": "YES" if output_has else "NO",
            "preserved": bool(source_has and output_has),
        })
    preserved = sum(r["preserved"] for r in rows)
    total = len(rows)
    return rows, (preserved / total * 100 if total else 0.0)

def critical_fact_matrix(source, output):
    rows = []
    for fact, variants in CRITICAL_FACTS:
        source_has = contains_any(source, variants)
        output_has = contains_any(output, variants)
        rows.append({
            "fact": fact,
            "source": source_has,
            "output": output_has,
            "preserved": source_has and output_has,
        })
    preserved = sum(r["preserved"] for r in rows)
    total = len(rows)
    return rows, (preserved / total * 100 if total else 0.0)

def score_style_signals(style_id, output):
    signals = STYLE_SIGNALS[style_id]
    hits = [s for s in signals if s in output]
    # Diagnostic only: presence of signals does not prove style.
    score = min(100.0, len(hits) / max(len(signals), 1) * 100.0)
    return round(score, 2), hits

def exact_number_check(source, output):
    source_numbers = Counter(extract_numbers(source))
    output_numbers = Counter(extract_numbers(output))
    matched = 0
    total = sum(source_numbers.values())
    for num, count in source_numbers.items():
        matched += min(count, output_numbers.get(num, 0))
    score = matched / total * 100 if total else 100.0
    extras = []
    for num, count in output_numbers.items():
        extra = max(0, count - source_numbers.get(num, 0))
        extras.extend([num] * extra)
    return round(score, 2), extras

def render_fact_matrix(rows):
    print("\n  FACT PRESERVATION MATRIX")
    print("  " + "-" * 86)
    print(f"  {'CATEGORY':<22}{'FACT':<42}{'RESULT':>10}")
    print("  " + "-" * 86)
    for r in rows:
        result = "✓ PRESERVED" if r["preserved"] else "✗ MISSING"
        print(f"  {r['category'][:21]:<22}{r['fact'][:41]:<42}{result:>10}")
    print("  " + "-" * 86)

def render_critical_matrix(rows):
    print("\n  CRITICAL FACT CHECK")
    print("  " + "-" * 72)
    print(f"  {'FACT':<48}{'RESULT':>18}")
    print("  " + "-" * 72)
    for r in rows:
        result = "✓" if r["preserved"] else "✗"
        print(f"  {r['fact'][:47]:<48}{result:>18}")
    print("  " + "-" * 72)

def human_evaluation_sheet(style_name, output):
    print("\n  HUMAN EVALUATION — ASK PANEL / RATER TO SCORE 1–5")
    print("  " + "-" * 72)
    print(f"  Target style: {style_name}")
    print("  1 = very poor, 5 = excellent")
    print("  Style adherence        : ____ / 5")
    print("  Meaning preservation   : ____ / 5")
    print("  Factual correctness    : ____ / 5")
    print("  Sinhala fluency        : ____ / 5")
    print("  Overall usefulness     : ____ / 5")
    print("  " + "-" * 72)

# ================================================================
# MAIN VIVA EVALUATION
# ================================================================

def main():
    set_seed(SEED)

    print("\n" + "=" * 96)
    print("  SinhalaJournal-LLM | FINAL VIVA STYLE REWRITER DEMONSTRATION")
    print("=" * 96)

    print("\n  IMPORTANT:")
    print("  This is a fixed single-article case study for the viva.")
    print("  It is NOT a statistical benchmark and must not be presented")
    print("  as the final general accuracy of the adapter.")
    print("\n  The panel can verify:")
    print("    1. Style transformation")
    print("    2. Fact preservation")
    print("    3. Number/date preservation")
    print("    4. Non-verbatim rewriting")
    print("    5. Human-rated fluency and usefulness")

    # ------------------------------------------------------------
    # PATH CHECK
    # ------------------------------------------------------------
    print("\n🔍 Checking paths...")
    if not Path(BASE_MODEL).exists():
        raise FileNotFoundError(f"Base model not found:\n{BASE_MODEL}")
    print("   ✅ Base model found")

    if not Path(ADAPTER_PATH).exists():
        raise FileNotFoundError(f"Adapter not found:\n{ADAPTER_PATH}")
    print("   ✅ Adapter found")

    tokenizer = load_tokenizer()
    model = load_model(tokenizer)

    print("\n" + "=" * 96)
    print("  SOURCE ARTICLE")
    print("=" * 96)
    print(VIVA_ARTICLE)

    results = {}

    # ------------------------------------------------------------
    # Generate all five target styles
    # ------------------------------------------------------------
    for style_id in STYLE_RULES:
        style_name = STYLE_NAMES[style_id]

        print("\n\n" + "=" * 96)
        print(f"  TARGET STYLE: {style_name}")
        print("=" * 96)

        rewritten = generate_rewrite(
            model,
            tokenizer,
            style_id,
            VIVA_ARTICLE,
        )

        if not rewritten:
            print("❌ Empty generation")
            results[style_id] = {"output": "", "error": "empty"}
            continue

        print("\n  REWRITTEN ARTICLE")
        print("  " + "-" * 88)
        print(rewritten)
        print("  " + "-" * 88)

        # Existing automatic diagnostics
        auto = calculate_correctness(VIVA_ARTICLE, rewritten)

        # Exact fact matrix for the fixed article
        fact_rows, fact_score = viva_fact_matrix(VIVA_ARTICLE, rewritten)
        critical_rows, critical_score = critical_fact_matrix(VIVA_ARTICLE, rewritten)
        number_score, extra_numbers = exact_number_check(VIVA_ARTICLE, rewritten)
        style_signal_score, style_hits = score_style_signals(style_id, rewritten)

        # Reference metrics ONLY if a human reference is supplied.
        reference = HUMAN_REFERENCE.get(style_id, "").strip()
        if reference:
            rouge = reference_rouge_scores(rewritten, reference)
        else:
            rouge = None

        # Copy check
        divergence, verbatim = score_style_divergence(
            VIVA_ARTICLE,
            rewritten,
        )

        print("\n  AUTOMATIC DIAGNOSTICS")
        print("  " + "-" * 72)
        print(f"  Critical fact preservation : {critical_score:6.2f}%")
        print(f"  Full fact checklist        : {fact_score:6.2f}%")
        print(f"  Number preservation        : {number_score:6.2f}%")
        print(f"  Content similarity         : {auto['content_similarity']:6.2f}%")
        print(f"  Length score               : {auto['length']:6.2f}%")
        print(f"  Style divergence           : {divergence:6.2f}%")
        print(f"  Verbatim copy              : {'YES' if verbatim else 'NO'}")
        print(f"  Diagnostic style signals   : {style_signal_score:6.2f}%")
        print(f"  Extra numbers              : {extra_numbers if extra_numbers else 'None'}")

        if rouge is not None:
            print(f"  ROUGE-1 vs human reference: {rouge['rouge1']:.4f}")
            print(f"  ROUGE-2 vs human reference: {rouge['rouge2']:.4f}")
            print(f"  ROUGE-L vs human reference: {rouge['rougeL']:.4f}")
        else:
            print("  ROUGE vs human reference  : N/A — no human reference supplied")

        print("\n  STYLE EVIDENCE")
        print(f"  Target: {style_name}")
        print(f"  Signals detected ({STYLE_SIGNAL_LABELS[style_id]}):")
        print(f"    {style_hits if style_hits else 'No automatic signal hits'}")

        render_critical_matrix(critical_rows)
        render_fact_matrix(fact_rows)

        human_evaluation_sheet(style_name, rewritten)

        results[style_id] = {
            "style": style_name,
            "source": VIVA_ARTICLE,
            "output": rewritten,
            "critical_fact_preservation": round(critical_score, 2),
            "full_fact_preservation": round(fact_score, 2),
            "number_preservation": number_score,
            "content_similarity": auto["content_similarity"],
            "length_score": auto["length"],
            "style_divergence": round(divergence, 2),
            "verbatim_copy": verbatim,
            "diagnostic_style_signal_score": style_signal_score,
            "style_signal_hits": style_hits,
            "extra_numbers": extra_numbers,
            "fact_matrix": fact_rows,
            "critical_fact_matrix": critical_rows,
            "human_reference_available": bool(reference),
            "rouge": (
                {k: round(v, 4) for k, v in rouge.items()}
                if rouge is not None else None
            ),
        }

    # ------------------------------------------------------------
    # FINAL VIVA SUMMARY
    # ------------------------------------------------------------
    print("\n\n" + "=" * 104)
    print("  FINAL VIVA SUMMARY — DO NOT CALL THIS 'MODEL ACCURACY'")
    print("=" * 104)

    print(
        f"\n  {'STYLE':<18}{'CRITICAL FACT':>16}"
        f"{'ALL FACTS':>14}{'NUMBERS':>12}"
        f"{'DIVERGENCE':>14}{'COPY?':>10}"
    )
    print("  " + "-" * 100)

    for style_id in STYLE_RULES:
        r = results.get(style_id)
        if not r or not r.get("output"):
            continue
        print(
            f"  {r['style']:<18}"
            f"{r['critical_fact_preservation']:>15.1f}%"
            f"{r['full_fact_preservation']:>13.1f}%"
            f"{r['number_preservation']:>11.1f}%"
            f"{r['style_divergence']:>13.1f}%"
            f"{('YES' if r['verbatim_copy'] else 'NO'):>10}"
        )

    print("  " + "-" * 100)

    print("\n  HOW TO EXPLAIN THIS TO THE PANEL")
    print("  1. 'I do not define rewriting accuracy as one number.'")
    print("  2. 'The rewrite must preserve facts while changing the writing style.'")
    print("  3. 'The fact matrix checks the source facts individually.'")
    print("  4. 'Numbers and dates receive explicit checks because they are")
    print("     high-risk factual elements in news text.'")
    print("  5. 'Divergence proves the model actually rewrote the article;")
    print("     divergence alone is NOT a correctness score.'")
    print("  6. 'Human raters finally judge style adherence, fluency and")
    print("     usefulness because automatic metrics cannot fully judge them.'")
    print("  7. 'ROUGE is shown only when a human-written reference rewrite")
    print("     exists for this exact article and target style.'")

    print("\n  IMPORTANT FOR THIS ARTICLE:")
    print("  The article is about vehicle-import taxation, not sports.")
    print("  Therefore a SPORTS rewrite should NOT invent matches, players,")
    print("  scores or sporting events. A good model should preserve the")
    print("  facts and adapt the style without fabricating a sports story.")

    # ------------------------------------------------------------
    # Save machine-readable report
    # ------------------------------------------------------------
    viva_report_path = Path(ADAPTER_PATH) / "viva_style_evaluation.json"
    with open(viva_report_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "evaluation_type": "single_article_viva_case_study",
                "warning": "Not a statistical benchmark or general model accuracy.",
                "source_article": VIVA_ARTICLE,
                "results": results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\n💾 Viva JSON report saved:")
    print(f"   {viva_report_path}")

    # Also save a clean text/markdown handout for the panel.
    handout_path = Path(ADAPTER_PATH) / "viva_style_evaluation_handout.md"
    with open(handout_path, "w", encoding="utf-8") as f:
        f.write("# SinhalaJournal-LLM — Style Rewriter Viva Evaluation\n\n")
        f.write("> Single-article case study. This is demonstration evidence, not a statistical benchmark.\n\n")
        f.write("## Evaluation logic\n\n")
        f.write("**Successful rewrite = style adherence + meaning/fact preservation + Sinhala fluency + non-verbatim transformation.**\n\n")
        f.write("## Source article\n\n")
        f.write(VIVA_ARTICLE + "\n\n")
        f.write("## Results\n\n")
        f.write("| Style | Critical facts | All facts | Numbers | Divergence | Verbatim copy |\n")
        f.write("|---|---:|---:|---:|---:|---|\n")
        for style_id in STYLE_RULES:
            r = results.get(style_id)
            if r and r.get("output"):
                f.write(
                    f"| {r['style']} | {r['critical_fact_preservation']:.1f}% | "
                    f"{r['full_fact_preservation']:.1f}% | "
                    f"{r['number_preservation']:.1f}% | "
                    f"{r['style_divergence']:.1f}% | "
                    f"{'YES' if r['verbatim_copy'] else 'NO'} |\n"
                )
        f.write("\n## Human evaluation\n\n")
        f.write("For each rewrite, rate 1–5:\n\n")
        f.write("- Style adherence\n- Meaning preservation\n- Factual correctness\n- Sinhala fluency\n- Overall usefulness\n\n")
        f.write("## Interpretation\n\n")
        f.write("- Fact preservation answers **Did the information survive?**\n")
        f.write("- Divergence answers **Did the model actually rewrite?**\n")
        f.write("- Human style rating answers **Does it sound like the requested style?**\n")
        f.write("- ROUGE should only be interpreted against a **human reference rewrite**, not the source.\n")

    print(f"\n📝 Viva handout saved:")
    print(f"   {handout_path}")

    print("\n✅ Viva demonstration completed.")


if __name__ == "__main__":
    main()