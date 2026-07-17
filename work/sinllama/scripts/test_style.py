#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# Unsloth MUST be first import
from unsloth import FastLanguageModel

import re
import json
import torch
import random
import numpy as np
from collections import Counter
from transformers import AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ──────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────
SINLLAMA_BASE  = "/home/jovyan/work/sinllama/models/SinLLaMA-merged-base"
# ✅ UPDATED: v06 → v07
STYLE_ADAPTER  = "/home/jovyan/work/sinllama/models/adapters/style_sinllama_v08"

# ✅ UPDATED: points at the same raw generate_style_dataset.py output the
# trainer used, instead of the old pre-converted stage2 file.
TEST_DATA_PATH = "/home/jovyan/style_rewriter/data/style_dataset2_dub.jsonl"

# ✅ NEW: must match TRAIN_SPLIT / SEED in train_style_rewriter.py exactly,
# or the "held-out" test set here won't actually match what the trainer
# held out - and you'd risk evaluating on articles it trained on.
TRAIN_SPLIT = 0.85
SEED        = 42


# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
# ✅ FIXED: was 1024, but train_style_rewriter.py trained with 4096.
# The mismatch caused Unsloth to silently truncate any article+instruction
# longer than 1024 tokens BEFORE generation even started - the model
# never saw the back half of longer articles, which is why
# length_preservation collapsed to 0.0 specifically on longer test cases
# while short ones scored fine. Must match the training value.
MAX_SEQ_LENGTH = 4096


# ──────────────────────────────────────────────
# LOAD TOKENIZER
# ──────────────────────────────────────────────
print("🔹 Loading tokenizer from pre-merged base...")
tokenizer = AutoTokenizer.from_pretrained(
    SINLLAMA_BASE,
    local_files_only=True,
)
tokenizer.pad_token    = tokenizer.eos_token
tokenizer.padding_side = "right"
print(f"   Vocab size: {len(tokenizer):,} tokens")


# ──────────────────────────────────────────────
# LOAD PRE-MERGED SINLLAMA BASE
# ──────────────────────────────────────────────
print("\n🔹 Loading pre-merged SinLLaMA base (4bit)...")
model, _ = FastLanguageModel.from_pretrained(
    model_name          = SINLLAMA_BASE,
    max_seq_length      = MAX_SEQ_LENGTH,
    dtype               = torch.bfloat16,
    load_in_4bit        = True,
    local_files_only    = True,
    attn_implementation = "eager",
)
print("   SinLLaMA base loaded ✅")


# ──────────────────────────────────────────────
# LOAD STYLE REWRITER ADAPTER
# ──────────────────────────────────────────────
print(f"\n🔹 Loading style rewriter adapter...")
print(f"   {STYLE_ADAPTER}")
model.load_adapter(STYLE_ADAPTER)

FastLanguageModel.for_inference(model)
model.eval()
print("   Adapter loaded ✅")


# ──────────────────────────────────────────────
# PROMPT FORMAT (must match training EXACTLY)
# ──────────────────────────────────────────────
# ✅ FIXED: this was missing the "### IMPORTANT RULES:" block that
# train_style_rewriter.py's format_prompt() bakes into every training
# example. The adapter was trained expecting that block between the
# instruction and the article - every eval run up to this point was
# feeding it a differently-shaped prompt than it actually learned on.
# Must stay byte-for-byte identical to train_style_rewriter.py's version.
def format_prompt(instruction: str, article: str) -> str:
    return (
        "### Instruction:\n"
        f"{instruction}\n\n"
        "### IMPORTANT RULES:\n"
        "1. Keep ALL the same facts in the EXACT same order\n"
        "2. Maintain approximately the SAME length as the original\n"
        "3. Do NOT add unrelated content (no greetings, no cricket references unless in original)\n"
        "4. Do NOT change the meaning of any fact\n"
        "5. Do NOT add new facts or events\n"
        "6. Apply the style while preserving ALL factual content\n"
        "7. Keep all names, numbers, and dates in EXACTLY their original "
        "wording - only change sentence structure and tone around them\n\n"
        "### Input:\n"
        f"{article}\n\n"
        "### Response:\n"
    )
    return (
        "### Instruction:\n"
        f"{instruction}\n\n"
        "### Input:\n"
        f"{article}\n\n"
        "### Response:\n"
    )


# ──────────────────────────────────────────────
# ✅ NEW: SCHEMA CONVERSION + QC FILTERING
# (mirrors convert_generated_record / DROP_QC_ISSUES in
#  train_style_rewriter.py so eval sees the same data shape and the
#  same exclusions the trainer applied)
# ──────────────────────────────────────────────
STYLE_IDS = {
    "style_1_formal_news",
    "style_2_editorial",
    "style_3_sports",
    "style_4_youth",
    "style_5_feature",
}

DROP_QC_ISSUES = {
    "missing_required_closing",
    "possible_stutter_duplication",
    "suspiciously_short",
}

# Must stay byte-for-byte identical to STYLE_RULES in
# train_style_rewriter.py - this is what the adapter was trained to
# expect as its instruction text.
STYLE_RULES = {
    "style_4_youth": """
- Start with casual greeting (දන්නවද? / ඇහුවද? / මේක අහන්න!)
- Use casual Sinhala: ගොඩක්, ටිකක්, හිතෙනවා
- Short, punchy sentences
- End with: ඒ නිසා යාලුවනේ, මේ ගැන අනිවාර්යයෙන්ම දැනගන්න!
""",
    "style_3_sports": """
- Lead with most dramatic fact
- Action verbs: පහර දුන්නා, ජයග්‍රහණය, ප්‍රහාරය
""",
    "style_2_editorial": """
- Start with: විශ්ලේෂණය කරන විට
- Use first person plural: අපි, අපට
- Add analytical language: සැලකිය යුතුය, සමස්තයක් ලෙස
- Express viewpoint on facts
""",
    "style_1_formal_news": """
- Objective, passive voice
- Inverted pyramid - most important fact first
- No opinion language
- Keep ALL facts in same order
""",
    "style_5_feature": """
- Narrative opening: set the scene
- Descriptive language
- Present tense where possible
- Human angle emphasized
"""
}


def convert_generated_record(rec: dict):
    """Same conversion as train_style_rewriter.py - maps the raw
    content/style/rewritten_text schema into instruction/input/output/
    metadata, dropping failed or QC-flagged rows. Returns None to skip."""
    if rec.get("status") == "failed" or rec.get("error"):
        return None

    style_id = rec.get("style")
    if style_id not in STYLE_IDS:
        return None

    article   = (rec.get("content") or "").strip()
    rewritten = (rec.get("rewritten_text") or "").strip()
    if not article or not rewritten:
        return None

    qc_issues = set(rec.get("qc_issues", []))
    if qc_issues & DROP_QC_ISSUES:
        return None

    if len(rewritten) < 50:
        return None

    return {
        "instruction": STYLE_RULES.get(style_id, "").strip(),
        "input":       article,
        "output":      rewritten,
        "metadata": {
            "style_id": style_id,
            "url":      rec.get("url"),
            "category": rec.get("category"),
        },
    }


# ──────────────────────────────────────────────
# EVALUATION METRICS
# ──────────────────────────────────────────────
def calculate_rouge_scores(reference: str, generated: str) -> dict:
    """
    Calculate ROUGE-like scores (precision, recall, F1)
    Based on n-gram overlap
    """
    def get_ngrams(text, n=1):
        words = text.split()
        return [tuple(words[i:i+n]) for i in range(len(words)-n+1)]
    
    metrics = {}
    
    for n in [1, 2]:  # Unigram and bigram
        ref_ngrams = Counter(get_ngrams(reference, n))
        gen_ngrams = Counter(get_ngrams(generated, n))
        
        # Calculate overlap
        overlap = sum((ref_ngrams & gen_ngrams).values())
        ref_total = sum(ref_ngrams.values())
        gen_total = sum(gen_ngrams.values())
        
        precision = overlap / gen_total if gen_total > 0 else 0
        recall = overlap / ref_total if ref_total > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics[f'ROUGE-{n}_precision'] = precision
        metrics[f'ROUGE-{n}_recall'] = recall
        metrics[f'ROUGE-{n}_f1'] = f1
    
    return metrics


def calculate_bleu_score(reference: str, generated: str, max_n=4) -> float:
    """
    Calculate simplified BLEU score, with smoothing.

    ✅ FIXED: the previous version returned 0.0 outright if ANY single
    n-gram order (1 through 4) had zero overlap - e.g. test case 1's
    FORMAL rewrite scored ROUGE-1=0.27 and cosine=0.82 (clearly
    substantial real overlap) but BLEU=0.0000, purely because 4-gram
    exact match happened to be zero. For genuine style rewriting, which
    restructures sentences, a 4-consecutive-word exact match against the
    reference is rare even for a GOOD rewrite - so this was collapsing
    BLEU to ~0 almost every time regardless of quality, wasting its full
    20% weight in the overall score. This adds standard additive
    smoothing (small epsilon on zero-count n-grams, same idea as NLTK's
    SmoothingFunction method1) instead of an all-or-nothing cutoff.
    """
    ref_words = reference.split()
    gen_words = generated.split()
    
    if len(gen_words) == 0:
        return 0.0
    
    precisions = []
    epsilon = 0.1  # smoothing constant for zero-count n-gram orders
    for n in range(1, max_n + 1):
        ref_ngrams = Counter()
        gen_ngrams = Counter()
        
        for i in range(len(ref_words) - n + 1):
            ref_ngrams[tuple(ref_words[i:i+n])] += 1
        
        for i in range(len(gen_words) - n + 1):
            gen_ngrams[tuple(gen_words[i:i+n])] += 1
        
        overlap = sum((ref_ngrams & gen_ngrams).values())
        total = sum(gen_ngrams.values())

        if total == 0:
            precisions.append(0.0)
        elif overlap == 0:
            # smoothed instead of a hard zero, so one missing n-gram
            # order doesn't zero out the whole score
            precisions.append(epsilon / total)
        else:
            precisions.append(overlap / total)

    if all(p == 0 for p in precisions):
        return 0.0

    geo_mean = np.exp(np.mean(np.log([max(p, 1e-9) for p in precisions])))
    brevity_penalty = min(1.0, len(gen_words) / len(ref_words)) if len(ref_words) > 0 else 0
    
    return geo_mean * brevity_penalty


def calculate_cosine_similarity(reference: str, generated: str) -> float:
    """
    Calculate cosine similarity using TF-IDF
    """
    try:
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([reference, generated])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return float(similarity)
    except:
        return 0.0


def calculate_length_preservation(original: str, generated: str) -> float:
    """
    Calculate how well the generated text preserves original length
    Score of 1.0 means perfect preservation
    """
    orig_len = len(original.split())
    gen_len = len(generated.split())
    
    if orig_len == 0:
        return 0.0
    
    ratio = gen_len / orig_len
    # Perfect score if ratio is between 0.8 and 1.2
    if 0.8 <= ratio <= 1.2:
        return 1.0
    elif ratio < 0.8:
        return ratio / 0.8
    else:
        return max(0, 2.0 - ratio / 1.2)


def calculate_vocabulary_diversity(text: str) -> float:
    """
    Calculate Type-Token Ratio (TTR) for vocabulary diversity
    """
    words = text.split()
    if len(words) == 0:
        return 0.0
    
    unique_words = len(set(words))
    return unique_words / len(words)


def calculate_style_distinctiveness(generated: str, style_name: str) -> dict:
    """
    Check if generated text has style-specific markers
    """
    metrics = {}
    
    # Check for style-specific patterns
    if style_name == 'youth':
        # Youth style should have conversational markers
        has_conversational = any([
            'ඇත්තටම' in generated,
            'මේක' in generated,
            'නම්' in generated,
            'ගොඩක්' in generated,
        ])
        metrics['conversational_markers'] = has_conversational
    
    elif style_name == 'sports':
        # Sports style should have action words
        has_action = any([
            'දැවැන්ත' in generated,
            'විශිෂ්ට' in generated,
            'ජය' in generated,
            '!' in generated,
        ])
        metrics['action_language'] = has_action
    
    elif style_name == 'editorial':
        # Editorial should have analytical markers
        has_analytical = any([
            'විශ්ලේෂණ' in generated,
            'තත්ත්වය' in generated,
            'අභියෝග' in generated,
            'මෙම' in generated,
        ])
        metrics['analytical_depth'] = has_analytical
    
    elif style_name == 'feature':
        # Feature should have narrative elements
        has_narrative = any([
            'කතාව' in generated,
            'ජීවිත' in generated,
            'සිහින' in generated,
            'බලාපොරොත්තු' in generated,
        ])
        metrics['narrative_elements'] = has_narrative
    
    elif style_name == 'formal':
        # Formal should be concise and objective
        is_concise = len(generated.split()) <= 50
        metrics['conciseness'] = is_concise
    
    return metrics


def evaluate_generation(original: str, generated: str, reference: str, style_name: str) -> dict:
    """
    Comprehensive evaluation of generated text
    
    Args:
        original: Original input article
        generated: Model's rewritten output
        reference: Expected output (from dataset)
        style_name: Target style name
    
    Returns:
        Dictionary of evaluation metrics
    """
    metrics = {}
    
    # Content preservation (ROUGE)
    rouge = calculate_rouge_scores(reference, generated)
    metrics.update(rouge)
    
    # BLEU score
    metrics['BLEU'] = calculate_bleu_score(reference, generated)
    
    # Semantic similarity
    metrics['cosine_similarity'] = calculate_cosine_similarity(reference, generated)
    
    # Length preservation
    metrics['length_preservation'] = calculate_length_preservation(original, generated)
    
    # Vocabulary diversity
    metrics['vocabulary_diversity'] = calculate_vocabulary_diversity(generated)
    
    # Style distinctiveness
    style_metrics = calculate_style_distinctiveness(generated, style_name)
    metrics.update(style_metrics)
    
    # Overall quality score (weighted average)
    weights = {
        'ROUGE-1_f1': 0.25,
        'ROUGE-2_f1': 0.20,
        'BLEU': 0.20,
        'cosine_similarity': 0.20,
        'length_preservation': 0.15,
    }
    
    quality_score = sum(metrics.get(k, 0) * v for k, v in weights.items())
    metrics['overall_quality_score'] = quality_score
    
    return metrics


# ──────────────────────────────────────────────
# STYLE REWRITING FUNCTION
# ──────────────────────────────────────────────
def rewrite_in_style(article: str, style_instruction: str) -> str:
    """
    Rewrite Sinhala news article in specified style.

    Args:
        article          : Original Sinhala article text
        style_instruction: Style transformation instruction

    Returns:
        Rewritten article in target style
    """
    prompt = format_prompt(style_instruction, article)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    # ✅ FIXED: max_new_tokens was a flat 256, which truncates almost any
    # real article mid-sentence (your style rules require ~same length as
    # the source). Scale it to the input length instead, same approach
    # used in generate_style_dataset.py's max_tokens fix.
    input_word_count = len(article.split())
    max_new_tokens = min(max(384, int(input_word_count * 3.5)), 2048)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens      = max_new_tokens,
            do_sample           = True,
            temperature         = 0.4,
            # ✅ FIXED: 1.5 repetition_penalty was aggressive enough to
            # break Sinhala subword/conjunct formation mid-word (visible
            # as corrupted fragments like "සිදුවේුවේ", stray Latin
            # characters). Dropped to a gentler value and paired with
            # no_repeat_ngram_size, which discourages repeated PHRASES
            # without penalizing the individual subword tokens Sinhala
            # needs to reuse just to spell a word correctly.
            repetition_penalty  = 1.15,
            no_repeat_ngram_size = 3,
            top_p                = 0.9,
            top_k                = 50,
            eos_token_id         = tokenizer.eos_token_id,
            pad_token_id         = tokenizer.eos_token_id,
            use_cache            = True,
        )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # STRICT CLEANING - Remove prompt and keep only response
    if "### Response:" in result:
        result = result.split("### Response:")[-1]

    # Remove instruction/input artifacts
    if "### Instruction:" in result:
        result = result.split("### Instruction:")[0]

    if "### Input:" in result:
        result = result.split("### Input:")[0]

    # ✅ FIXED: the old per-SENTENCE English-word filter would drop an
    # entire sentence (sometimes the entire output - see test cases
    # 22/37/38) if it contained a single corrupted Latin-looking
    # fragment. Now only strips stray Latin substrings out of an
    # otherwise-Sinhala sentence, instead of deleting the whole sentence -
    # much less likely to silently zero out the response.
    sentences = re.split(r'(?<=[।\.!?])\s+', result)
    cleaned_sentences = []
    for sentence in sentences:
        sinhala_chars = len(re.findall(r'[\u0D80-\u0DFF]', sentence))
        english_chars = len(re.findall(r'[A-Za-z]', sentence))

        if sinhala_chars == 0:
            continue  # nothing Sinhala in this sentence at all - skip it

        # If it's overwhelmingly English, this sentence probably isn't a
        # real part of the rewrite - skip it. Otherwise, just strip the
        # stray Latin fragments and keep the sentence.
        if english_chars > 0 and english_chars / (sinhala_chars + english_chars) > 0.4:
            continue

        sentence = re.sub(r'\b[A-Za-z]{1,4}\b', '', sentence)  # standalone stray Latin words
        # ✅ FIXED: \b doesn't separate Latin from Sinhala since Python's
        # Unicode-aware regex treats both as \w characters - so a fragment
        # like "way" glued directly onto a Sinhala word (e.g. "ලං wayයේ")
        # wasn't being caught by the \b version above. This second pass
        # catches Latin runs sitting immediately next to Sinhala characters
        # with no boundary between them.
        sentence = re.sub(r'(?<=[\u0D80-\u0DFF])[A-Za-z]{1,6}|[A-Za-z]{1,6}(?=[\u0D80-\u0DFF])', '', sentence)
        sentence = re.sub(r'\s{2,}', ' ', sentence).strip()

        if sentence and not re.match(r'^[#\*\-\>]', sentence):
            cleaned_sentences.append(sentence)

    result = '। '.join(cleaned_sentences)

    # ✅ REMOVED: the old "keep only first 3 lines" cap. That made sense
    # when max_new_tokens=256 produced short garbled output, but now that
    # generations are meant to run the full length of the article, this
    # was silently truncating otherwise-good long rewrites down to a
    # few lines - directly hurting length_preservation.
    result = re.sub(r'\n{2,}', '\n', result.strip())

    return result.strip()


# ──────────────────────────────────────────────
# STYLE INSTRUCTIONS (display labels only - actual instruction text
# passed to the model comes from STYLE_RULES/item["instruction"] above)
# ──────────────────────────────────────────────
STYLE_INSTRUCTIONS = {
    "formal":    "Rewrite the following Sinhala news article in formal news reporting style (ප්‍රධාන පුවත්). Use objective, concise language with inverted-pyramid structure.",
    "editorial": "Rewrite the following Sinhala news article in editorial/opinion style (කතුවැකිය/විශේෂාංග). Add analytical perspective with longer sentences.",
    "sports":    "Rewrite the following Sinhala news article in dynamic sports reporting style (ක්‍රීඩා). Use energetic, action-oriented language.",
    "youth":     "Rewrite the following Sinhala news article in youth-oriented conversational style (යෞවන-අනුකූල). Use friendly, relatable language.",
    "feature":   "Rewrite the following Sinhala news article in feature/narrative storytelling style (විශේෂාංග/Storytelling). Focus on human angle with emotional depth.",
}


# ──────────────────────────────────────────────
# ✅ UPDATED: TEST DATA LOADING
# Reads the raw generate_style_dataset.py output, converts it with the
# same logic + QC filtering as training, then reproduces the EXACT same
# article-level (by url) train/val split as train_style_rewriter.py so
# this evaluates only on articles the adapter never trained on.
# ──────────────────────────────────────────────
def load_test_data(filepath=TEST_DATA_PATH, split_ratio=TRAIN_SPLIT, seed=SEED,
                    n_articles=1, min_words=500):
    """
    ✅ UPDATED: selects N articles (default 1) with at least min_words
    words from the held-out val set that have all 5 styles present, and
    returns all 5 style rows for each - so a single long article gets
    tested across all 5 styles (5 total test cases) instead of a
    scattered sample.
    """
    raw_records = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                raw_records.append(json.loads(line))

    converted = []
    for rec in raw_records:
        c = convert_generated_record(rec)
        if c is not None:
            converted.append(c)

    print(f"   Raw records: {len(raw_records)} -> valid after conversion/QC: {len(converted)}")

    # Article-level split by url - must match train_style_rewriter.py's
    # logic exactly (same seed, same ratio) so this is a true held-out set.
    by_url = {}
    for rec in converted:
        url = rec["metadata"].get("url") or id(rec)
        by_url.setdefault(url, []).append(rec)

    urls = list(by_url.keys())
    random.seed(seed)
    random.shuffle(urls)
    n_train_urls = int(len(urls) * split_ratio)
    val_urls = set(urls[n_train_urls:])

    # ✅ NEW: only keep articles that (a) have all 5 styles present in the
    # held-out set and (b) are at least min_words words long, then rank
    # by length and take the N longest that qualify.
    qualifying_groups = []
    for url in val_urls:
        rows = by_url[url]
        styles_present = {r["metadata"]["style_id"] for r in rows}
        if styles_present != STYLE_IDS:
            continue
        word_count = len(rows[0]["input"].split())
        if word_count >= min_words:
            qualifying_groups.append((word_count, url, rows))

    qualifying_groups.sort(key=lambda g: g[0], reverse=True)
    selected = qualifying_groups[:n_articles]

    print(f"   Articles with all 5 styles AND >= {min_words} words: {len(qualifying_groups)}")
    print(f"   Selected {len(selected)} article(s) (word counts: "
          f"{[g[0] for g in selected]})")

    if not selected:
        print(f"   ⚠️  No article in the held-out set meets the {min_words}-word "
              f"minimum with all 5 styles present. Lower min_words or check "
              f"your dataset.")

    val_records = []
    for _, _, rows in selected:
        # Keep a stable style order per article for readability in output
        rows_sorted = sorted(rows, key=lambda r: r["metadata"]["style_id"])
        val_records.extend(rows_sorted)

    return val_records

print("\n📂 Loading test dataset...")
test_records = load_test_data(n_articles=1, min_words=500)
print(f"   Using {len(test_records)} samples for evaluation "
      f"({len(test_records) // len(STYLE_IDS)} article × 5 styles).")

# ──────────────────────────────────────────────
# RUN TESTS WITH METRICS
# ──────────────────────────────────────────────
print("\n" + "=" * 80)
print("  STYLE REWRITER TESTING & EVALUATION")
print("=" * 80)

all_metrics = []

for idx, item in enumerate(test_records, 1):
    article_num = (idx - 1) // len(STYLE_IDS) + 1
    style_num_in_article = (idx - 1) % len(STYLE_IDS) + 1
    print(f"\n{'=' * 80}")
    print(f"📄 ARTICLE {article_num} / {len(test_records) // len(STYLE_IDS)}"
          f"  —  STYLE {style_num_in_article} / {len(STYLE_IDS)}"
          f"  (test case {idx}/{len(test_records)})")
    print("=" * 80)
    print("\n📝 ORIGINAL ARTICLE:")
    print(item["input"][:250] + "...")
    
    test_case_metrics = {'test_case': idx}
    
    # Extract style name from metadata
    style_id = item.get("metadata", {}).get("style_id", "")
    if "formal" in style_id: style_name = "formal"
    elif "editorial" in style_id: style_name = "editorial"
    elif "sports" in style_id: style_name = "sports"
    elif "youth" in style_id: style_name = "youth"
    elif "feature" in style_id: style_name = "feature"
    else: style_name = "formal" # fallback
    
    print(f"\n{'─' * 80}")
    print(f"🎨 STYLE: {style_name.upper()}")
    print("─" * 80)
    
    instruction = item["instruction"]
    reference = item["output"]
    
    rewritten = rewrite_in_style(item["input"], instruction)
    
    print(f"\n✨ REWRITTEN OUTPUT:")
    print(rewritten[:250] + "...")
    
    metrics = evaluate_generation(
        original=item["input"],
        generated=rewritten,
        reference=reference,
        style_name=style_name
    )
    
    test_case_metrics[style_name] = metrics
    
    print(f"\n📊 EVALUATION METRICS:")
    print(f"  ROUGE-1 F1 Score:        {metrics.get('ROUGE-1_f1', 0):.4f}")
    print(f"  ROUGE-2 F1 Score:        {metrics.get('ROUGE-2_f1', 0):.4f}")
    print(f"  BLEU Score:              {metrics.get('BLEU', 0):.4f}")
    print(f"  Cosine Similarity:       {metrics.get('cosine_similarity', 0):.4f}")
    print(f"  Length Preservation:     {metrics.get('length_preservation', 0):.4f}")
    print(f"  Vocabulary Diversity:    {metrics.get('vocabulary_diversity', 0):.4f}")
    print(f"  Overall Quality Score:   {metrics.get('overall_quality_score', 0):.4f}")
    
    # Style-specific metrics
    if style_name == 'youth' and 'conversational_markers' in metrics:
        print(f"  Conversational Markers:  {'✅' if metrics['conversational_markers'] else '❌'}")
    elif style_name == 'sports' and 'action_language' in metrics:
        print(f"  Action Language:         {'✅' if metrics['action_language'] else '❌'}")
    elif style_name == 'editorial' and 'analytical_depth' in metrics:
        print(f"  Analytical Depth:        {'✅' if metrics['analytical_depth'] else '❌'}")
    elif style_name == 'feature' and 'narrative_elements' in metrics:
        print(f"  Narrative Elements:      {'✅' if metrics['narrative_elements'] else '❌'}")
    elif style_name == 'formal' and 'conciseness' in metrics:
        print(f"  Conciseness:             {'✅' if metrics['conciseness'] else '❌'}")
    
    all_metrics.append(test_case_metrics)

# ──────────────────────────────────────────────
# AGGREGATE METRICS SUMMARY
# ──────────────────────────────────────────────
print("\n" + "=" * 80)
print("  📈 AGGREGATE METRICS SUMMARY")
print("=" * 80)

# Collect all metrics across test cases
metric_keys = ['ROUGE-1_f1', 'ROUGE-2_f1', 'BLEU', 'cosine_similarity', 
               'length_preservation', 'vocabulary_diversity', 'overall_quality_score']

overall_quality_scores = []

for key in metric_keys:
    values = []
    for test_case in all_metrics:
        for style, metrics in test_case.items():
            if isinstance(metrics, dict) and key in metrics:
                values.append(metrics[key])
                if key == 'overall_quality_score':
                    overall_quality_scores.append(metrics[key])
    
    if values:
        avg = np.mean(values)
        std = np.std(values)
        min_val = np.min(values)
        max_val = np.max(values)
        
        print(f"\n{key}:")
        print(f"  Average: {avg:.4f} ± {std:.4f}")
        print(f"  Min: {min_val:.4f} | Max: {max_val:.4f}")

# Calculate Accuracy Percentage based on Overall Quality Score
accuracy_percentage = np.mean(overall_quality_scores) * 100 if overall_quality_scores else 0

# Per-style summary
print("\n" + "=" * 80)
print("  📊 PER-STYLE PERFORMANCE")
print("=" * 80)

for style_name in STYLE_INSTRUCTIONS.keys():
    style_scores = []
    for test_case in all_metrics:
        if style_name in test_case and isinstance(test_case[style_name], dict):
            style_scores.append(test_case[style_name].get('overall_quality_score', 0))
    
    if style_scores:
        avg_score = np.mean(style_scores)
        print(f"\n{style_name.upper()}: {avg_score:.4f} (avg quality score)")

print("\n" + "=" * 80)
print(f"  🎯 ACCURACY PERCENTAGE: {accuracy_percentage:.2f}%")
print("=" * 80)

# Save metrics to file
metrics_output_file = 'test_metrics_results.json'
with open(metrics_output_file, 'w', encoding='utf-8') as f:
    json.dump(all_metrics, f, ensure_ascii=False, indent=2)

print(f"\n✅ Testing complete!")
print(f"📄 Detailed metrics saved to: {metrics_output_file}\n")