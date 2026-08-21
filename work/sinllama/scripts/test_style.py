# ================================================================
# SinhalaJournal-LLM
# STYLE ADAPTER CORRECTNESS TEST
#
# Purpose:
#   Same article -> 5 different styles
#   Evaluate factual correctness of each rewrite
#
# IMPORTANT:
#   This test is NOT measuring whether the rewrite is stylistically
#   beautiful. It measures whether the original facts are preserved.
#
#   5 outputs only:
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
import difflib
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

# ================================================================
# THE ARTICLE TO TEST
# ================================================================

ARTICLE = """
ශ්‍රී ලංකාවේ විදේශ විනිමය සංචිතවල පීඩනය සහ ආර්ථික ප්‍රතිසංස්කරණවල ප්‍රතිඵලයක් ලෙස, ශ්‍රී ලංකා රජය විසින් මේ වන විට වාහන ආනයනය සම්බන්ධයෙන් සැලකිය යුතු බදු ප්‍රතිපත්ති වෙනස්කම් කිහිපයක් හඳුන්වා දී ඇත. ඒ අතරින් ප්‍රධානතම වෙනස වන්නේ 2026 මැයි 16 වන දින සිට බලාත්මක කරන ලද සියයට 50ක තාවකාලික අතිරේක බද්දයි. ජනාධිපතිවරයා විසින් රේගු පනත යටතේ නිකුත් කරන ලද මෙම විශේෂ නියෝගය මගී රථ, ජීප් රථ, බස් රථ, භාණ්ඩ ප්‍රවාහන රථ, ගිලන් රථ මෙන්ම විදුලි හා දෙමුහුම් වාහන සඳහා ද අදාළ කර ඇති අතර, යතුරුපැදි, ත්‍රී රෝද රථ සහ වාණිජ කටයුතු සඳහා භාවිතා කරන ආනයනික වාහන පමණක් මෙම නීතියෙන් බැහැර කර තිබේ. මෙම තීරණය කෙටි කාලීන පියවරක් ලෙස ප්‍රකාශයට පත් කර ඇති අතර, ආර්ථිකය යම් තරමක ස්ථාවරත්වයකට පත්වන තෙක් මෙම අතිරේක බද්ද ක්‍රියාත්මක වනු ඇතැයි රජයේ නිලධාරිහු පෙන්වා දෙති.
""".strip()


# ================================================================
# STYLE RULES
#
# These must match the rules used by your trained adapter.
# ================================================================

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

    import random

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

    print(
        f"   Input tokens: {input_ids.shape[-1]}"
    )

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

    text = text.replace("\u200b", "")
    text = text.replace("\u200c", "")
    text = text.replace("\u200d", "")
    text = text.replace("\ufeff", "")

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
# EXTRACT DATES
# ================================================================

def extract_dates(text):

    text = normalize_text(text)

    dates = []

    # 2026 මැයි 16
    patterns = [

        r"\d{4}\s+[^\s]+\s+\d{1,2}",

        r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}",

        r"\d{4}[/-]\d{1,2}[/-]\d{1,2}",
    ]

    for pattern in patterns:

        dates.extend(
            re.findall(
                pattern,
                text,
            )
        )

    return dates


# ================================================================
# EXTRACT IMPORTANT FACTUAL PHRASES
#
# This article is Sinhala, therefore word-level exact matching
# alone is too strict. We use character n-grams + important tokens.
# ================================================================

def tokenize_sinhala(text):

    text = normalize_text(text)

    return re.findall(
        r"[\u0D80-\u0DFF]+|[A-Za-z]+|\d+(?:[.,]\d+)?",
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
# PRINT SCORE
# ================================================================

def print_score(
    style_id,
    result,
):

    print("\n" + "-" * 80)

    print(
        f"📊 CORRECTNESS: "
        f"{style_id}"
    )

    print(
        f"   ⭐ Overall correctness : "
        f"{result['overall']:.2f}%"
    )

    print(
        f"   Numbers preserved      : "
        f"{result['numbers']:.2f}%"
    )

    print(
        f"   Important facts        : "
        f"{result['facts']:.2f}%"
    )

    print(
        f"   Content similarity     : "
        f"{result['content_similarity']:.2f}%"
    )

    print(
        f"   Length preservation    : "
        f"{result['length']:.2f}%"
    )

    print(
        f"   Output quality         : "
        f"{result['quality']:.2f}%"
    )

    print(
        f"   Style divergence        : "
        f"{result['style_divergence']:.2f}%"
    )

    if result["is_verbatim_copy"]:

        print(
            "\n   🚨 VERBATIM COPY: output is character-identical "
            "to the input. No style rewrite occurred -- the "
            "correctness score above is meaningless for this style."
        )

    if result["missing_numbers"]:

        print(
            "\n   ⚠️ Missing numbers:"
        )

        print(
            "   ",
            ", ".join(
                result[
                    "missing_numbers"
                ]
            )
        )

    if result["extra_numbers"]:

        print(
            "\n   ⚠️ Extra numbers:"
        )

        print(
            "   ",
            ", ".join(
                result[
                    "extra_numbers"
                ]
            )
        )

    if result["missing_facts"]:

        print(
            "\n   ⚠️ Some important "
            "source words/facts not detected:"
        )

        print(
            "   ",
            ", ".join(
                result[
                    "missing_facts"
                ][:30]
            )
        )


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
        "5-STYLE CORRECTNESS TEST"
    )

    print(
        "=" * 80
    )

    print(
        "\nSame article will be rewritten "
        "into exactly 5 styles."
    )

    print(
        "\nCorrectness measures FACT "
        "preservation, not style similarity."
    )

    print(
        "\nGeneration:"
    )

    print(
        f"  Temperature       : "
        f"{TEMPERATURE}"
    )

    print(
        f"  Top-P             : "
        f"{TOP_P}"
    )

    print(
        f"  Top-K             : "
        f"{TOP_K}"
    )

    print(
        f"  Repetition        : "
        f"{REPETITION_PENALTY}"
    )

    print(
        f"  Max new tokens    : "
        f"{GEN_MAX_NEW_TOKENS}"
    )

    # ------------------------------------------------------------
    # PATH CHECK
    # ------------------------------------------------------------

    print(
        "\n🔍 Checking paths..."
    )

    if not Path(BASE_MODEL).exists():

        raise FileNotFoundError(
            f"Base model not found:\n"
            f"{BASE_MODEL}"
        )

    print(
        "   ✅ Base model found"
    )

    if not Path(ADAPTER_PATH).exists():

        raise FileNotFoundError(
            f"Adapter not found:\n"
            f"{ADAPTER_PATH}"
        )

    print(
        "   ✅ Adapter found"
    )

    # ------------------------------------------------------------
    # LOAD
    # ------------------------------------------------------------

    tokenizer = load_tokenizer()

    model = load_model(tokenizer)

    # ------------------------------------------------------------
    # PRINT ORIGINAL
    # ------------------------------------------------------------

    print(
        "\n" +
        "=" * 80
    )

    print(
        "📰 ORIGINAL ARTICLE"
    )

    print(
        "=" * 80
    )

    print(
        ARTICLE
    )

    # ------------------------------------------------------------
    # TEST ALL 5 STYLES
    # ------------------------------------------------------------

    results = {}

    style_names = {

        "style_1_formal_news":
            "FORMAL NEWS",

        "style_2_editorial":
            "EDITORIAL",

        "style_3_sports":
            "SPORTS",

        "style_4_youth":
            "YOUTH",

        "style_5_feature":
            "FEATURE",
    }

    for index, style_id in enumerate(
        STYLE_RULES.keys(),
        start=1,
    ):

        style_name = style_names[
            style_id
        ]

        print(
            "\n\n" +
            "=" * 80
        )

        print(
            f"  {index}/5  "
            f"{style_name}"
        )

        print(
            "=" * 80
        )

        print(
            "\n🔄 Generating rewrite..."
        )

        try:

            rewritten = generate_rewrite(
                model,
                tokenizer,
                style_id,
                ARTICLE,
            )

        except Exception as e:

            print(
                "\n❌ Generation failed:"
            )

            print(
                type(e).__name__,
                str(e),
            )

            rewritten = ""

        if not rewritten:

            print(
                "\n❌ EMPTY OUTPUT"
            )

            result = {
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

        else:

            # ----------------------------------------------------
            # OUTPUT
            # ----------------------------------------------------

            print(
                "\n📝 REWRITTEN ARTICLE:"
            )

            print(
                "-" * 80
            )

            print(
                rewritten
            )

            # ----------------------------------------------------
            # SCORE
            # ----------------------------------------------------

            result = calculate_correctness(
                ARTICLE,
                rewritten,
            )

        results[style_id] = {
            "style": style_name,
            "rewritten": rewritten,
            "score": result,
        }

        print_score(
            style_name,
            result,
        )

    # ============================================================
    # FINAL SUMMARY
    # ============================================================

    print(
        "\n\n" +
        "=" * 80
    )

    print(
        "  FINAL CORRECTNESS SUMMARY"
    )

    print(
        "=" * 80
    )

    print(
        "\n"
    )

    print(
        f"{'STYLE':<20}"
        f"{'CORRECTNESS':>15}"
        f"{'DIVERGENCE':>15}"
        f"{'COPY?':>10}"
    )

    print(
        "-" * 63
    )

    valid_scores = []
    verbatim_copies = 0

    for style_id, data in results.items():

        score = data["score"][
            "overall"
        ]

        divergence = data["score"][
            "style_divergence"
        ]

        is_copy = data["score"][
            "is_verbatim_copy"
        ]

        if is_copy:
            verbatim_copies += 1

        print(
            f"{data['style']:<20}"
            f"{score:>14.2f}%"
            f"{divergence:>14.2f}%"
            f"{'YES' if is_copy else 'no':>10}"
        )

        valid_scores.append(score)

    print(
        "-" * 63
    )

    average = (
        sum(valid_scores) /
        len(valid_scores)
        if valid_scores
        else 0
    )

    print(
        f"{'AVERAGE':<20}"
        f"{average:>14.2f}%"
    )

    if verbatim_copies:

        print(
            f"\n🚨 {verbatim_copies}/{len(results)} styles produced "
            f"a VERBATIM COPY of the input (0% style divergence). "
            f"The correctness average above is inflated by these -- "
            f"a copy trivially preserves every fact. Treat this "
            f"adapter as NOT rewriting style until divergence > 0 "
            f"is confirmed on real generations."
        )

    # ------------------------------------------------------------
    # PASS / WARNING
    # ------------------------------------------------------------

    print(
        "\n"
    )

    for style_id, data in results.items():

        score = data["score"][
            "overall"
        ]

        if score >= 90:

            status = "🟢 EXCELLENT"

        elif score >= 80:

            status = "🟢 GOOD"

        elif score >= 70:

            status = "🟡 ACCEPTABLE"

        elif score >= 60:

            status = "🟠 NEEDS IMPROVEMENT"

        else:

            status = "🔴 POOR"

        print(
            f"{data['style']:<20}"
            f"{status}"
        )

    # ============================================================
    # SAVE JSON REPORT
    # ============================================================

    report_path = (
        Path(ADAPTER_PATH)
        / "style_correctness_test.json"
    )

    report = {

        "base_model": BASE_MODEL,

        "adapter": ADAPTER_PATH,

        "article": ARTICLE,

        "generation": {

            "temperature":
                TEMPERATURE,

            "top_p":
                TOP_P,

            "top_k":
                TOP_K,

            "repetition_penalty":
                REPETITION_PENALTY,

            "max_new_tokens":
                GEN_MAX_NEW_TOKENS,
        },

        "results": results,

        "average_correctness":
            round(
                average,
                2,
            ),
    }

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(
        "\n💾 Report saved:"
    )

    print(
        f"   {report_path}"
    )

    print(
        "\n✅ 5-style test completed."
    )


# ================================================================
# RUN
# ================================================================

if __name__ == "__main__":
    main()