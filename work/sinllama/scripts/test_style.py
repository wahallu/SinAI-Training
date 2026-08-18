# ============================================================
# SinhalaJournal-LLM | Style Rewriter Test v13
# ============================================================
# Changes from original v12:
#   - max_new_tokens now scales with article length (was flat 1024)
#   - clean_output strips .। danda artifact
#   - Added evaluation metrics (ROUGE-1/2, BLEU, cosine, length ratio)
#   - Added batch eval mode: run on N articles, report aggregate accuracy
#   - STYLE_RULES, GLOBAL_RULES, build_prompt unchanged
# ============================================================

from unsloth import FastLanguageModel

import os
import sys
import json
import shutil
import tempfile
import traceback
from pathlib import Path
from collections import Counter

import re
import unicodedata
import torch
import numpy as np

from transformers import LlamaTokenizerFast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# PATHS
# ============================================================

BASE_MODEL = "/home/jovyan/work/sinllama/models/SinLLaMA-merged-base"
ADAPTER_PATH = "/home/jovyan/work/sinllama/models/adapters/style_sinllama_v12"
DATASET_PATH = "/home/jovyan/style_rewriter/data/style_dataset2_fixed.jsonl"


# ============================================================
# CONFIG
# ============================================================

MAX_SEQ_LENGTH = 4096

GEN_TEMPERATURE = 0.10
GEN_TOP_P = 0.85
GEN_TOP_K = 50
GEN_REPETITION_PENALTY = 1.05

# ✅ CHANGED: was a flat 1024 regardless of article length.
# Long articles were getting truncated mid-sentence, same bug
# we fixed twice before in earlier test scripts. Now calculated
# per-article in generate_style() based on input length.
# This constant is now the FLOOR (minimum), not the ceiling.
GEN_MAX_NEW_TOKENS_FLOOR = 512
GEN_MAX_NEW_TOKENS_CEILING = 4096
GEN_TOKENS_PER_WORD_MULTIPLIER = 3.5  # Sinhala needs more tokens/word than English

SEED = 42

# Number of articles to evaluate in batch mode (--batch flag).
BATCH_EVAL_N = 10

# Diagnostic thresholds
FACT_ENTITY_PRESERVATION_TARGET = 0.90
LENGTH_RATIO_MIN = 0.80
LENGTH_RATIO_MAX = 1.20


# ============================================================
# STYLE IDs
# ============================================================

STYLE_IDS = {
    "style_1_formal_news",
    "style_2_editorial",
    "style_3_sports",
    "style_4_youth",
    "style_5_feature",
}


# ============================================================
# STYLE RULES  (unchanged from original v12)
# ============================================================

STYLE_RULES = {

    "style_1_formal_news": """
ඔබ සිංහල පුවත්පත් කලාවේ වෘත්තීය ප්‍රවෘත්ති ලේඛකයෙකි.

පහත නීති අනිවාර්යයෙන්ම අනුගමනය කරන්න:

1. ලිපියේ වඩාත්ම වැදගත් කරුණ මුලින්ම ඉදිරිපත් කරන්න.
2. වෘත්තීය, නිල සහ වස්තුමය සිංහල භාෂාව භාවිතා කරන්න.
3. පුවත් වාර්තාකරණයට සුදුසු පැහැදිලි හා සංක්ෂිප්ත වාක්‍ය භාවිතා කරන්න.
4. අනවශ්‍ය හැඟීම්බර හෝ අදහස් දැක්වීම් සහිත භාෂාව භාවිතා නොකරන්න.
5. මුල් ලිපියේ ඇති සියලුම කරුණු, නම්, ස්ථාන, සංඛ්‍යා සහ දිනයන් නිවැරදිව තබා ගන්න.
6. කිසිදු නව කරුණක් හෝ සිදුවීමක් එකතු නොකරන්න.
7. මුල් ලිපියේ ඇති තොරතුරු ඉවත් නොකරන්න.
8. මුල් ලිපියේ අනුපිළිවෙළ සහ කරුණු ආරක්ෂා කරන්න.
9. quotation marks තුළ ඇති වචන තිබේ නම් ඒවා වෙනස් නොකරන්න.
10. නිවැරදි සිංහල අක්ෂර වින්‍යාසය භාවිතා කරන්න.
11. ලිපියේ දිග මුල් ලිපියට ආසන්නව තබා ගන්න.
12. පුවත් වාර්තාවකට ගැළපෙන ස්වභාවික වෘත්තීය ශෛලියක් පවත්වා ගන්න.
""".strip(),

    "style_2_editorial": """
ඔබ සිංහල පුවත්පත් කතුවැකි ලේඛකයෙකි.

පහත නීති අනිවාර්යයෙන්ම අනුගමනය කරන්න:

1. කරුණ පිළිබඳ විශ්ලේෂණාත්මක ආරම්භයක් ලබා දෙන්න.
2. "අපි", "අපට", "සමස්තයක් ලෙස", "සැලකිය යුතුය" වැනි ස්වභාවික කතුවැකි භාෂාව භාවිතා කරන්න.
3. සිදුවීමේ පසුබිම, වැදගත්කම සහ බලපෑම විශ්ලේෂණය කරන ආකාරයේ ලිවීමක් භාවිතා කරන්න.
4. සරල පුවත් වාර්තාවක් ලෙස නොව කතුවැකිමය සහ විශ්ලේෂණාත්මක ස්වරයක් පවත්වා ගන්න.
5. කරුණු මත පදනම් වූ අදහස් සහ විශ්ලේෂණාත්මක භාෂාව භාවිතා කරන්න.
6. මුල් ලිපියේ ඇති සියලුම කරුණු, නම්, ස්ථාන, සංඛ්‍යා සහ දිනයන් ආරක්ෂා කරන්න.
7. නව කරුණු, සංඛ්‍යා හෝ සිදුවීම් නිර්මාණය නොකරන්න.
8. මුල් ලිපියේ අර්ථය වෙනස් නොකරන්න.
9. quotation marks තුළ ඇති මුල් වචන වෙනස් නොකරන්න.
10. පුද්ගලයන්ගේ නම් සහ ගෞරව නාම වෙනස් නොකරන්න.
11. නිවැරදි සිංහල අක්ෂර වින්‍යාසය භාවිතා කරන්න.
12. ලිපියේ දිග මුල් ලිපියට ආසන්නව තබා ගන්න.
13. අතිශයෝක්තියෙන් හෝ අනවශ්‍ය දේශපාලන ප්‍රකාශවලින් වළකින්න.
""".strip(),

    "style_3_sports": """
ඔබ සිංහල ක්‍රීඩා පුවත්පත් කලාවේ විශේෂඥයෙකි.

පහත නීති අනිවාර්යයෙන්ම අනුගමනය කරන්න:

1. වඩාත්ම නාටකීය සහ වැදගත් ක්‍රීඩා කරුණ මුලින්ම ගෙන එන්න. එය ලිපියේ පළමු වාක්‍යයෙන්ම පැහැදිලි විය යුතුය.

2. ඉහළ ශක්තියකින් යුත් නමුත් ස්වභාවික ක්‍රීඩා පුවත් භාෂාවක් භාවිතා කරන්න.

3. ක්‍රීඩා ක්‍රියාකාරකම් පෙන්වන ක්‍රියාකාරී ක්‍රියා පද භාවිතා කරන්න.
   උදාහරණ:
   "පහර දුන්නේය"
   "ජයග්‍රහණය කළේය"
   "පරාජය කළේය"
   "ප්‍රහාරය එල්ල කළේය"
   "ඉදිරියට පැමිණියේය"
   "ලකුණු රැස් කළේය"
   "විශිෂ්ට දක්ෂතාවක් දැක්වීය"
   "තීරණාත්මක ජයග්‍රහණයක් ලබා ගත්තේය"

4. වේගවත් රිද්මයක් සහ උද්වේගකර ක්‍රීඩා පුවත් ස්වරයක් පවත්වා ගන්න.

5. ක්‍රීඩාවේ තීරණාත්මක අවස්ථාව, ජයග්‍රහණය, පරාජය, වාර්තාව හෝ විශිෂ්ට දක්ෂතාවය පැහැදිලිව ඉස්මතු කරන්න.

6. මුල් ලිපියේ ඇති සියලුම කරුණු, ක්‍රීඩකයන්ගේ නම්, කණ්ඩායම් නම්, ස්ථාන, ලකුණු, සංඛ්‍යා, දිනයන් සහ තරග තොරතුරු නිවැරදිව තබා ගන්න.

7. කිසිදු නව ක්‍රීඩකයෙකු, කණ්ඩායමක්, ලකුණක්, තරගයක්, වාර්තාවක් හෝ සිදුවීමක් එකතු නොකරන්න.

8. මුල් ලිපියේ නොමැති ක්‍රීඩා තොරතුරු නිර්මාණය නොකරන්න.

9. මුල් ලිපියේ ඉංග්‍රීසි වචන නොතිබේ නම් අනවශ්‍ය ඉංග්‍රීසි වචන එකතු නොකරන්න.

10. quotation marks තුළ ඇති වචන හෝ ප්‍රකාශ වෙනස් නොකරන්න. ඒවා මුල් ලිපියේ ආකාරයටම තබා ගන්න.

11. පුද්ගලයන්ගේ නම්, ගෞරව නාම සහ ස්ත්‍රී/පුරුෂ සම්බන්ධ නාම වෙනස් නොකරන්න.

12. මුල් ලිපියේ ඇති සංඛ්‍යා, ලකුණු, දිනයන් සහ සංඛ්‍යාත්මක තොරතුරු වෙනස් නොකරන්න.

13. උද්වේගය පෙන්වීම සඳහා උපරිම වශයෙන් එක් විස්මයාර්ථක ලකුණක් (!) පමණක් භාවිතා කරන්න.

14. එකම වචනය හෝ වාක්‍ය ඛණ්ඩය අනවශ්‍ය ලෙස නැවත නැවත භාවිතා නොකරන්න.

15. ක්‍රීඩා පුවත්පත් කලාවට ගැළපෙන ස්වභාවික සිංහල භාෂාව භාවිතා කරන්න. අතිශයෝක්තියෙන් හෝ කෘතිම වාක්‍යවලින් වළකින්න.

16. සම්පූර්ණ වාක්‍ය පමණක් ලියන්න. වාක්‍ය අතරමගදී නතර නොකරන්න.

17. නැවත ලියන ලිපියේ දිග මුල් ලිපියේ දිගට ආසන්නව තබා ගන්න.

18. මුල් ලිපියේ ඇති සියලුම කරුණු ආරක්ෂා කරමින් ක්‍රීඩා පුවත්පත් ශෛලිය පමණක් වෙනස් කරන්න.
""".strip(),

    "style_4_youth": """
ඔබ තරුණ පිරිස සඳහා සිංහල ඩිජිටල් මාධ්‍ය ලේඛකයෙකි.

පහත නීති අනිවාර්යයෙන්ම අනුගමනය කරන්න:

1. තරුණ පාඨකයාගේ අවධානය ලබා ගන්නා ස්වභාවික ආරම්භයක් භාවිතා කරන්න.
2. සරල, පහසු සහ conversational සිංහල භාෂාව භාවිතා කරන්න.
3. "ගොඩක්", "ටිකක්", "හිතෙනවා" වැනි ස්වභාවික තරුණ භාෂා භාවිතා කළ හැක.
4. කෙටි සහ පැහැදිලි වාක්‍ය භාවිතා කරන්න.
5. ලිපිය කියවන විට තරුණ පාඨකයෙකුට ස්වභාවිකව දැනෙන රිද්මයක් පවත්වා ගන්න.
6. අතිශයෝක්තියෙන් හෝ කෘතිම slang භාවිතයෙන් වළකින්න.
7. මුල් ලිපියේ ඇති සියලුම කරුණු, නම්, ස්ථාන, සංඛ්‍යා සහ දිනයන් ආරක්ෂා කරන්න.
8. කිසිදු නව කරුණක් එකතු නොකරන්න.
9. මුල් ලිපියේ අර්ථය වෙනස් නොකරන්න.
10. quotation marks තුළ ඇති වචන වෙනස් නොකරන්න.
11. පුද්ගලයන්ගේ නම් සහ ගෞරව නාම වෙනස් නොකරන්න.
12. අනවශ්‍ය ඉංග්‍රීසි වචන එකතු නොකරන්න.
13. නිවැරදි සිංහල අක්ෂර වින්‍යාසය භාවිතා කරන්න.
14. ලිපියේ දිග මුල් ලිපියට ආසන්නව තබා ගන්න.
15. සම්පූර්ණ සහ පැහැදිලි වාක්‍ය භාවිතා කරන්න.
""".strip(),

    "style_5_feature": """
ඔබ සිංහල පුවත්පත් සහ සඟරා සඳහා විශේෂාංග ලිපි ලියන වෘත්තීය ලේඛකයෙකි.

පහත නීති අනිවාර්යයෙන්ම අනුගමනය කරන්න:

1. සිදුවීමේ පරිසරය හෝ මනුෂ්‍ය අත්දැකීම ඉස්මතු කරන ආකර්ෂණීය ආරම්භයක් භාවිතා කරන්න.
2. විස්තරාත්මක නමුත් ස්වභාවික සිංහල භාෂාව භාවිතා කරන්න.
3. පාඨකයාට සිදුවීම සිතින් දැකගත හැකි වන ආකාරයේ වාතාවරණයක් නිර්මාණය කරන්න.
4. මනුෂ්‍ය කේන්ද්‍රීය පැත්ත ඉස්මතු කරන්න.
5. අවශ්‍ය තැන්වල වර්තමාන කාල රටාව භාවිතා කළ හැක.
6. පුවත් වාර්තාවකට වඩා narrative සහ feature-style flow එකක් පවත්වා ගන්න.
7. නමුත් මුල් ලිපියේ නොමැති සිදුවීම්, හැඟීම් හෝ පසුබිම් තොරතුරු නිර්මාණය නොකරන්න.
8. මුල් ලිපියේ ඇති සියලුම කරුණු, නම්, ස්ථාන, සංඛ්‍යා සහ දිනයන් ආරක්ෂා කරන්න.
9. quotation marks තුළ ඇති වචන වෙනස් නොකරන්න.
10. පුද්ගලයන්ගේ නම් සහ ගෞරව නාම වෙනස් නොකරන්න.
11. මුල් ලිපියේ අර්ථය වෙනස් නොකරන්න.
12. අනවශ්‍ය ඉංග්‍රීසි වචන එකතු නොකරන්න.
13. නිවැරදි සිංහල අක්ෂර වින්‍යාසය භාවිතා කරන්න.
14. ලිපියේ දිග මුල් ලිපියට ආසන්නව තබා ගන්න.
15. සම්පූර්ණ වාක්‍ය භාවිතා කරන්න.
""".strip(),
}


# ============================================================
# GLOBAL FACT PRESERVATION RULES  (unchanged from original v12)
# ============================================================

GLOBAL_RULES = """
### IMPORTANT FACT-PRESERVATION RULES

1. Preserve ALL factual information from the source article.
2. Do NOT invent facts.
3. Do NOT remove important facts.
4. Do NOT change names.
5. Do NOT change numbers.
6. Do NOT change dates.
7. Do NOT change locations.
8. Do NOT change team names or player names.
9. Preserve gender-marked honorifics exactly.
10. "මහත්මිය" must remain "මහත්මිය".
11. "මහතා" must remain "මහතා".
12. Text inside quotation marks must remain VERBATIM.
13. Do NOT insert new words into quotations.
14. Do NOT translate quotations.
15. Do NOT introduce new durations.
16. Do NOT introduce new numbers.
17. Do NOT introduce new symbols.
18. Do NOT add events that are absent from the source.
19. Keep the rewritten article approximately the same length.
20. Write complete sentences.
21. Do not stop in the middle of a sentence.
22. Do not add an explanation before or after the rewritten article.
23. Return ONLY the rewritten article.
"""


# ============================================================
# EVALUATION METRICS
# ============================================================

def get_ngrams(words, n):
    return [tuple(words[i:i+n]) for i in range(len(words)-n+1)]


def rouge_n(reference, generated, n):
    ref_words = reference.split()
    gen_words = generated.split()
    ref_ngrams = Counter(get_ngrams(ref_words, n))
    gen_ngrams = Counter(get_ngrams(gen_words, n))
    overlap = sum((ref_ngrams & gen_ngrams).values())
    ref_total = sum(ref_ngrams.values())
    gen_total = sum(gen_ngrams.values())
    precision = overlap / gen_total if gen_total else 0
    recall = overlap / ref_total if ref_total else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    return {"precision": precision, "recall": recall, "f1": f1}


def smoothed_bleu(reference, generated, max_n=4):
    ref_words = reference.split()
    gen_words = generated.split()
    if not gen_words:
        return 0.0
    epsilon = 0.1
    precisions = []
    for n in range(1, max_n + 1):
        ref_ng = Counter(get_ngrams(ref_words, n))
        gen_ng = Counter(get_ngrams(gen_words, n))
        overlap = sum((ref_ng & gen_ng).values())
        total = sum(gen_ng.values())
        if total == 0:
            precisions.append(0.0)
        elif overlap == 0:
            precisions.append(epsilon / total)
        else:
            precisions.append(overlap / total)
    if all(p == 0 for p in precisions):
        return 0.0
    geo_mean = np.exp(np.mean(np.log([max(p, 1e-9) for p in precisions])))
    bp = min(1.0, len(gen_words) / len(ref_words)) if ref_words else 0
    return geo_mean * bp


def cosine_sim(reference, generated):
    try:
        vec = TfidfVectorizer()
        mat = vec.fit_transform([reference, generated])
        return float(cosine_similarity(mat[0:1], mat[1:2])[0][0])
    except Exception:
        return 0.0


def length_ratio(reference, generated):
    orig = len(reference.split())
    gen = len(generated.split())
    return gen / orig if orig else 0.0


def length_preservation(reference, generated):
    ratio = length_ratio(reference, generated)
    if ratio == 0:
        return 0.0
    if LENGTH_RATIO_MIN <= ratio <= LENGTH_RATIO_MAX:
        return 1.0
    elif ratio < LENGTH_RATIO_MIN:
        return ratio / LENGTH_RATIO_MIN
    return max(0.0, 2.0 - ratio / LENGTH_RATIO_MAX)


def normalize_token(token):
    """Normalize a Sinhala/text token for robust entity matching."""
    token = unicodedata.normalize("NFC", token)
    token = token.strip(" \t\n\r.,!?;:()[]{}\"'“”‘’«»/\\|—–-")
    return token


def extract_entities(source):
    """
    Lightweight factual/entity extraction.

    This intentionally avoids an external NER model. It focuses on:
      - numbers
      - dates/time-like numeric expressions
      - quoted text
      - capitalized/Latin tokens
      - multi-word Sinhala/Latin proper-name candidates

    The goal is a diagnostic, not a perfect Sinhala NER system.
    """
    entities = []

    # Numbers, including decimals and comma-separated numbers.
    entities += re.findall(r"\b\d+(?:[.,]\d+)*\b", source)

    # Quoted content must be preserved verbatim.
    entities += re.findall(r'[“"«](.*?)[”"»]', source, flags=re.DOTALL)

    # Latin/English tokens that may be names, organizations, roads, etc.
    entities += re.findall(r"\b[A-Za-z][A-Za-z0-9._/-]{1,}\b", source)

    # Sinhala multi-token candidates containing common proper-name markers.
    # Keep these conservative to avoid penalizing ordinary words.
    for phrase in re.findall(
        r"(?:ශ්‍රී\s+පාද|මස්කෙළිය|මාවුස්සාකැලේ|නාවලපිටිය|"
        r"රත්නපුර|නිවිතිගල|හැටන්|කොළඹ|නුවරඑලිය|නුවරඑළිය|"
        r"ගිනිගත්හේන|නෝටව්?බ්‍රිජ්|නෝටන්බ්‍රිජ්)",
        source
    ):
        entities.append(phrase)

    # Normalize and deduplicate while preserving order.
    out = []
    seen = set()
    for e in entities:
        e = normalize_token(e)
        if not e:
            continue
        key = e.casefold()
        if key not in seen:
            seen.add(key)
            out.append(e)
    return out


def entity_preservation(source, generated):
    """
    Percentage of extracted source entities/numbers that still occur
    in the generated article. Exact normalized substring matching is used.
    """
    entities = extract_entities(source)
    if not entities:
        return 1.0, [], []

    gen_norm = unicodedata.normalize("NFC", generated).casefold()
    preserved = []
    missing = []

    for entity in entities:
        if unicodedata.normalize("NFC", entity).casefold() in gen_norm:
            preserved.append(entity)
        else:
            missing.append(entity)

    return len(preserved) / len(entities), preserved, missing


def new_content_candidates(source, generated):
    """
    Find simple numeric/quoted/Latin facts that appear in generated text
    but were absent from the source. This is a conservative hallucination
    signal, not a semantic hallucination detector.
    """
    source_entities = set(e.casefold() for e in extract_entities(source))
    generated_entities = extract_entities(generated)

    new_items = []
    seen = set()

    for entity in generated_entities:
        key = entity.casefold()
        if key not in source_entities and key not in seen:
            seen.add(key)
            new_items.append(entity)

    return new_items


def factual_diagnostics(source, generated):
    preservation, preserved, missing = entity_preservation(source, generated)
    new_items = new_content_candidates(source, generated)

    return {
        "entity_preservation": preservation,
        "preserved_entities": preserved,
        "missing_entities": missing,
        "new_fact_candidates": new_items,
        "missing_count": len(missing),
        "new_fact_count": len(new_items),
    }


def evaluate(source, generated, reference=None):
    """
    Evaluate generated text.

    - Reference-based metrics measure similarity to the human/reference rewrite
      when one is available.
    - Source-based diagnostics always measure factual preservation against the
      original article.
    """
    comparison = reference if reference else source

    r1 = rouge_n(comparison, generated, 1)
    r2 = rouge_n(comparison, generated, 2)
    bleu = smoothed_bleu(comparison, generated)
    cos = cosine_sim(comparison, generated)
    lp = length_preservation(comparison, generated)

    factual = factual_diagnostics(source, generated)

    # Keep the original style/rewrite score, but explicitly penalize
    # factual loss. This prevents a short/high-overlap rewrite from looking
    # excellent while silently deleting facts.
    style_score = (
        r1["f1"] * 0.20 +
        r2["f1"] * 0.15 +
        bleu     * 0.15 +
        cos      * 0.15 +
        lp       * 0.10
    )

    factual_score = factual["entity_preservation"] * 0.25
    score = style_score + factual_score

    return {
        "rouge_1_f1": r1["f1"],
        "rouge_2_f1": r2["f1"],
        "bleu": bleu,
        "cosine": cos,
        "length_preservation": lp,
        "length_ratio": length_ratio(comparison, generated),
        "entity_preservation": factual["entity_preservation"],
        "missing_entities": factual["missing_entities"],
        "new_fact_candidates": factual["new_fact_candidates"],
        "missing_count": factual["missing_count"],
        "new_fact_count": factual["new_fact_count"],
        "overall": score,
        "comparison_type": "reference" if reference else "source",
    }

# ============================================================
# TOKENIZER PATCH  (unchanged from original v12)
# ============================================================

def create_patched_model_directory(base_path: str):
    base = Path(base_path)
    if not base.exists():
        raise FileNotFoundError(f"Base model does not exist: {base}")
    temp_dir = Path(tempfile.mkdtemp(prefix="sinllama_tokenizer_fixed_"))
    print()
    print("🔧 Creating temporary tokenizer-compatible model directory...")
    print(f"   Original : {base}")
    print(f"   Temporary: {temp_dir}")
    for item in base.iterdir():
        destination = temp_dir / item.name
        try:
            destination.symlink_to(item, target_is_directory=item.is_dir())
        except Exception:
            if item.is_file():
                shutil.copy2(item, destination)
    original_config_path = base / "tokenizer_config.json"
    if not original_config_path.exists():
        raise FileNotFoundError("tokenizer_config.json was not found in base model.")
    with open(original_config_path, "r", encoding="utf-8") as f:
        tokenizer_config = json.load(f)
    tokenizer_config["tokenizer_class"] = "LlamaTokenizerFast"
    if "backend_tokenizer_class" in tokenizer_config:
        tokenizer_config.pop("backend_tokenizer_class")
    patched_config_path = temp_dir / "tokenizer_config.json"
    with open(patched_config_path, "w", encoding="utf-8") as f:
        json.dump(tokenizer_config, f, ensure_ascii=False, indent=2)
    print("   ✅ Temporary tokenizer patch created.")
    return temp_dir


# ============================================================
# LOAD TOKENIZER  (unchanged)
# ============================================================

def load_tokenizer(model_path: str):
    print()
    print("🔹 Loading LLaMA tokenizer explicitly...")
    tokenizer = LlamaTokenizerFast.from_pretrained(model_path, local_files_only=True)
    tokenizer.padding_side = "right"
    if tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"   ✅ LlamaTokenizerFast loaded — vocab {len(tokenizer):,}")
    return tokenizer


# ============================================================
# LOAD MODEL + ADAPTER  (unchanged)
# ============================================================

def load_model():
    print()
    print("🔹 Preparing tokenizer-compatible base model...")
    patched_base = create_patched_model_directory(BASE_MODEL)
    tokenizer = load_tokenizer(str(patched_base))
    print()
    print("🔹 Loading SinLLaMA base through Unsloth...")
    try:
        model, _ = FastLanguageModel.from_pretrained(
            model_name=str(patched_base),
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=torch.bfloat16,
            load_in_4bit=True,
            local_files_only=True,
            attn_implementation="eager",
        )
    except Exception:
        print("❌ Failed to load base model.")
        traceback.print_exc()
        shutil.rmtree(patched_base, ignore_errors=True)
        raise
    print("   ✅ SinLLaMA base loaded.")
    print()
    print("🔹 Loading style rewriter adapter...")
    try:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, ADAPTER_PATH, is_trainable=False)
    except Exception:
        print("❌ Failed to load LoRA adapter.")
        traceback.print_exc()
        shutil.rmtree(patched_base, ignore_errors=True)
        raise
    print("   ✅ LoRA adapter loaded.")
    model.eval()
    try:
        model = FastLanguageModel.for_inference(model)
        print("   ✅ Unsloth inference optimization enabled.")
    except Exception as e:
        print(f"   ⚠️ Could not enable Unsloth inference mode: {e}")
    return model, tokenizer, patched_base


# ============================================================
# LOAD DATASET  (unchanged)
# ============================================================

def load_dataset():
    print()
    print(f"📂 Loading dataset: {DATASET_PATH}")
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")
    records = []
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    print(f"   ✅ Loaded {len(records):,} records.")
    return records


def get_article_text(record):
    return (record.get("content") or record.get("input") or record.get("article") or "").strip()


def get_style(record):
    return (
        record.get("style") or
        record.get("style_id") or
        record.get("metadata", {}).get("style_id") or ""
    )


def get_reference_text(record):
    return (
        record.get("rewritten_text") or
        record.get("output") or
        record.get("corrected_text") or ""
    ).strip()


def get_unique_articles(records):
    articles = {}
    for record in records:
        article = get_article_text(record)
        if not article:
            continue
        url = record.get("url") or record.get("metadata", {}).get("url")
        key = url if url else article[:200]
        if key not in articles:
            articles[key] = {"article": article, "record": record, "url": url}
    return list(articles.values())


# ============================================================
# BUILD PROMPT  (unchanged from original v12)
# ============================================================

def build_prompt(article: str, style_id: str):
    style_rules = STYLE_RULES.get(style_id, "")
    return (
        "### Instruction:\n"
        f"{style_rules}\n\n"
        "### GLOBAL REQUIREMENTS:\n"
        f"{GLOBAL_RULES}\n\n"
        "### INPUT ARTICLE:\n"
        f"{article}\n\n"
        "### RESPONSE:\n"
    )


# ============================================================
# GENERATE
# ============================================================

@torch.inference_mode()
def generate_style(model, tokenizer, article: str, style_id: str):

    prompt = build_prompt(article, style_id)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding=False,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # ✅ FIXED: scale max_new_tokens with article length instead of flat 1024.
    # A flat 1024 truncated longer articles mid-sentence — same bug fixed
    # twice in earlier test scripts. The floor/ceiling bounds keep it sane.
    word_count = len(article.split())
    max_new_tokens = int(
        max(GEN_MAX_NEW_TOKENS_FLOOR,
            min(GEN_MAX_NEW_TOKENS_CEILING,
                word_count * GEN_TOKENS_PER_WORD_MULTIPLIER))
    )

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=GEN_TEMPERATURE,
        top_p=GEN_TOP_P,
        top_k=GEN_TOP_K,
        repetition_penalty=GEN_REPETITION_PENALTY,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=True,
    )

    input_length = inputs["input_ids"].shape[1]
    generated_ids = output_ids[0, input_length:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return generated_text.strip()


# ============================================================
# CLEAN OUTPUT
# ============================================================

def clean_output(text):
    if not text:
        return ""

    # Remove accidental prompt headers
    for prefix in ["### Response:", "### Output:", "Response:", "Output:"]:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()

    # Stop if model starts generating another section
    for marker in ["### Instruction:", "### INPUT ARTICLE:", "### Input:", "### GLOBAL REQUIREMENTS:"]:
        if marker in text:
            text = text.split(marker, 1)[0].strip()

    # ✅ NEW: strip .। danda artifact (Devanagari full stop appearing after
    # a Sinhala sentence-ending period — model sometimes emits this,
    # possibly learned from training data that had it).
    text = re.sub(r'\.।\s*', '. ', text)
    text = re.sub(r'।\s*', ' ', text)

    # Remove stray markdown
    text = re.sub(r'^[#\*\-\>].*$', '', text, flags=re.MULTILINE)

    return text.strip()


# ============================================================
# DISPLAY RESULT
# ============================================================

def display_result(article, output, style_id, metrics=None):
    print()
    print("=" * 80)
    print(f"STYLE: {style_id}")
    print("=" * 80)
    print()
    print("SOURCE ARTICLE")
    print("-" * 80)
    print(article)
    print()
    print("REWRITTEN ARTICLE")
    print("-" * 80)
    print(output)
    print()
    print(f"Source characters : {len(article):,}")
    print(f"Output characters : {len(output):,}")
    if len(article) > 0:
        print(f"Length ratio      : {len(output)/len(article):.2f}")
    if metrics:
        print()
        print("EVALUATION METRICS")
        print("-" * 40)
        print(f"  ROUGE-1 F1        : {metrics['rouge_1_f1']:.4f}")
        print(f"  ROUGE-2 F1        : {metrics['rouge_2_f1']:.4f}")
        print(f"  BLEU (smoothed)   : {metrics['bleu']:.4f}")
        print(f"  Cosine similarity : {metrics['cosine']:.4f}")
        print(f"  Length ratio      : {metrics['length_ratio']:.3f}")
        print(f"  Length preserv.   : {metrics['length_preservation']:.4f}")
        print(f"  Entity preserv.   : {metrics['entity_preservation']:.4f}")
        print(f"  Missing entities  : {metrics['missing_count']}")
        print(f"  New fact signals  : {metrics['new_fact_count']}")
        print(f"  Overall score     : {metrics['overall']:.4f}  ({metrics['overall']*100:.2f}%)")
        if metrics.get("missing_entities"):
            print(f"  ⚠️ Missing: {', '.join(metrics['missing_entities'][:12])}")
        if metrics.get("new_fact_candidates"):
            print(f"  ⚠️ New fact signals: {', '.join(metrics['new_fact_candidates'][:12])}")
    print("=" * 80)


# ============================================================
# SAVE RESULT
# ============================================================

def save_result(article, outputs, output_path):
    result = {"source_article": article, "styles": outputs}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print()
    print(f"💾 Results saved to: {output_path}")


# ============================================================
# INTERACTIVE MODE  (unchanged flow from original v12)
# ============================================================

def select_article(articles):
    if not articles:
        raise RuntimeError("No valid articles found in dataset.")
    print()
    print("=" * 80)
    print("AVAILABLE ARTICLES")
    print("=" * 80)
    display_count = min(20, len(articles))
    for i in range(display_count):
        preview = articles[i]["article"].replace("\n", " ").strip()
        if len(preview) > 100:
            preview = preview[:100] + "..."
        print(f"[{i+1:02d}] {preview}")
    print()
    choice = input(f"Select article [1-{display_count}] (default 1): ").strip()
    try:
        index = int(choice) - 1 if choice else 0
    except ValueError:
        index = 0
    if index < 0 or index >= display_count:
        index = 0
    return articles[index]["article"]


def select_styles():
    print()
    print("=" * 80)
    print("STYLE SELECTION")
    print("=" * 80)
    print("1. Formal News")
    print("2. Editorial")
    print("3. Sports")
    print("4. Youth")
    print("5. Feature")
    print("6. ALL STYLES")
    print()
    choice = input("Select style [1-6] (default 6): ").strip() or "6"
    mapping = {
        "1": ["style_1_formal_news"],
        "2": ["style_2_editorial"],
        "3": ["style_3_sports"],
        "4": ["style_4_youth"],
        "5": ["style_5_feature"],
        "6": list(STYLE_IDS),
    }
    return mapping.get(choice, mapping["6"])


def run_interactive(model, tokenizer, records):
    articles = get_unique_articles(records)
    print(f"\n   Unique source articles: {len(articles):,}")
    article = select_article(articles)
    styles = select_styles()
    outputs = {}
    print()
    print("=" * 80)
    print("🚀 GENERATION STARTED")
    print("=" * 80)
    for style_id in styles:
        print(f"\n🔹 Generating: {style_id}")
        try:
            output = clean_output(generate_style(model, tokenizer, article, style_id))
            outputs[style_id] = output
            metrics = evaluate(article, output, reference=None)
            display_result(article, output, style_id, metrics)
        except Exception as e:
            print(f"❌ Generation failed for {style_id}: {type(e).__name__}: {e}")
            traceback.print_exc()
    if outputs:
        save_result(article, outputs, "/home/jovyan/work/sinllama/test_results_v13.json")


# ============================================================
# BATCH EVALUATION MODE
# ============================================================

def run_batch_eval(model, tokenizer, records, n=BATCH_EVAL_N):
    """
    Evaluate N articles × all 5 styles.

    IMPORTANT:
      - Reference-based metrics compare generated output with the dataset's
        human/reference rewrite for that style.
      - Factual preservation is ALWAYS checked against the original source.
      - Articles with all 5 style references are preferred.
    """
    print()
    print("=" * 80)
    print(f"🧪 BATCH EVALUATION — {n} articles × 5 styles = {n*5} test cases")
    print("=" * 80)

    # Group records by URL (fallback to source prefix).
    by_url = {}
    for rec in records:
        url = rec.get("url") or get_article_text(rec)[:200]
        by_url.setdefault(url, []).append(rec)

    qualifying = []
    for url, recs in by_url.items():
        article = get_article_text(recs[0])
        if not article:
            continue

        styles_present = {
            get_style(rec)
            for rec in recs
            if get_style(rec) in STYLE_IDS and get_reference_text(rec)
        }

        # Prefer complete 5-style articles, then longer articles.
        complete = len(styles_present) == len(STYLE_IDS)
        qualifying.append(
            (complete, len(styles_present), len(article.split()), url, recs)
        )

    qualifying.sort(reverse=True)

    selected = qualifying[:n]
    if not selected:
        print("⚠️ No articles found for batch evaluation.")
        return

    all_scores = []
    scores_by_style = {s: [] for s in STYLE_IDS}
    entity_scores_by_style = {s: [] for s in STYLE_IDS}
    missing_by_style = {s: [] for s in STYLE_IDS}
    hallucination_by_style = {s: [] for s in STYLE_IDS}

    for idx, (_, style_count, _, url, recs) in enumerate(selected, 1):
        article = get_article_text(recs[0])

        print(f"\n{'─'*80}")
        print(f"Article {idx}/{len(selected)} | reference styles: {style_count}/5")
        print(f"Words: {len(article.split())}")
        print(f"Preview: {article[:120]}...")

        ref_by_style = {}
        for rec in recs:
            sid = get_style(rec)
            ref = get_reference_text(rec)
            if sid in STYLE_IDS and ref:
                ref_by_style[sid] = ref

        for style_id in sorted(STYLE_IDS):
            try:
                output = clean_output(
                    generate_style(model, tokenizer, article, style_id)
                )

                reference = ref_by_style.get(style_id)

                m = evaluate(
                    source=article,
                    generated=output,
                    reference=reference
                )

                all_scores.append(m["overall"])
                scores_by_style[style_id].append(m["overall"])
                entity_scores_by_style[style_id].append(
                    m["entity_preservation"]
                )
                missing_by_style[style_id].append(m["missing_count"])
                hallucination_by_style[style_id].append(
                    m["new_fact_count"]
                )

                ref_label = "reference" if reference else "source"

                status = "PASS"
                if m["entity_preservation"] < FACT_ENTITY_PRESERVATION_TARGET:
                    status = "FACT-LOSS"
                if m["new_fact_count"] > 0:
                    status = "HALLUCINATION-SIGNAL"

                print(
                    f"  {style_id:<25} "
                    f"score={m['overall']*100:6.1f}% "
                    f"R1={m['rouge_1_f1']:.3f} "
                    f"R2={m['rouge_2_f1']:.3f} "
                    f"BLEU={m['bleu']:.3f} "
                    f"cos={m['cosine']:.3f} "
                    f"LP={m['length_preservation']:.3f} "
                    f"facts={m['entity_preservation']:.3f} "
                    f"missing={m['missing_count']} "
                    f"new={m['new_fact_count']} "
                    f"[{ref_label}] "
                    f"→ {status}"
                )

                if m["missing_entities"]:
                    print(
                        "      Missing: "
                        + ", ".join(m["missing_entities"][:10])
                    )

                if m["new_fact_candidates"]:
                    print(
                        "      New fact signals: "
                        + ", ".join(m["new_fact_candidates"][:10])
                    )

            except Exception as e:
                print(
                    f"  ❌ {style_id}: "
                    f"{type(e).__name__}: {e}"
                )

    print()
    print("=" * 80)
    print("📊 AGGREGATE RESULTS")
    print("=" * 80)
    print()
    print(
        f"{'Style':<25} "
        f"{'Score':>9} "
        f"{'Facts':>9} "
        f"{'Missing':>9} "
        f"{'New':>9} "
        f"{'N':>5}"
    )
    print("-" * 75)

    for style_id in sorted(STYLE_IDS):
        vals = scores_by_style[style_id]
        facts = entity_scores_by_style[style_id]
        miss = missing_by_style[style_id]
        new = hallucination_by_style[style_id]

        if vals:
            print(
                f"{style_id:<25} "
                f"{np.mean(vals)*100:8.2f}% "
                f"{np.mean(facts)*100:8.2f}% "
                f"{np.mean(miss):9.2f} "
                f"{np.mean(new):9.2f} "
                f"{len(vals):5d}"
            )
        else:
            print(
                f"{style_id:<25} "
                f"{'N/A':>9} {'N/A':>9} {'N/A':>9} {'N/A':>9} {'0':>5}"
            )

    print("-" * 75)

    if all_scores:
        overall = np.mean(all_scores) * 100
        print(
            f"{'OVERALL SCORE':<25} "
            f"{overall:8.2f}%"
        )

    print("=" * 80)

    report_path = "/home/jovyan/work/sinllama/batch_eval_v13.json"

    report = {
        "version": "v13",
        "n_articles": len(selected),
        "n_test_cases": len(all_scores),
        "overall_score_pct": (
            float(np.mean(all_scores) * 100)
            if all_scores else 0
        ),
        "targets": {
            "entity_preservation": FACT_ENTITY_PRESERVATION_TARGET,
            "length_ratio_min": LENGTH_RATIO_MIN,
            "length_ratio_max": LENGTH_RATIO_MAX,
        },
        "per_style": {}
    }

    for style_id in sorted(STYLE_IDS):
        vals = scores_by_style[style_id]
        facts = entity_scores_by_style[style_id]
        miss = missing_by_style[style_id]
        new = hallucination_by_style[style_id]

        report["per_style"][style_id] = {
            "avg_score_pct": float(np.mean(vals) * 100) if vals else None,
            "avg_entity_preservation_pct": (
                float(np.mean(facts) * 100) if facts else None
            ),
            "avg_missing_entities": float(np.mean(miss)) if miss else None,
            "avg_new_fact_signals": float(np.mean(new)) if new else None,
            "n": len(vals),
        }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n💾 Report saved to: {report_path}")

# ============================================================
# MAIN
# ============================================================

def main():
    # Determine mode: python test_style_rewriter_v13.py --batch
    batch_mode = "--batch" in sys.argv

    print()
    print("=" * 80)
    print("  SinhalaJournal-LLM | Style Rewriter Test v13")
    print(f"  Mode: {'BATCH EVALUATION' if batch_mode else 'INTERACTIVE'}")
    print("=" * 80)

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # Check paths
    for label, path in [("Base model", BASE_MODEL), ("Adapter", ADAPTER_PATH), ("Dataset", DATASET_PATH)]:
        if os.path.exists(path):
            print(f"   ✅ {label} found")
        else:
            print(f"   ❌ {label} NOT found: {path}")
            return

    records = load_dataset()
    patched_base = None

    try:
        model, tokenizer, patched_base = load_model()
    except Exception as e:
        print(f"\n❌ Model loading failed: {type(e).__name__}: {e}")
        traceback.print_exc()
        return

    try:
        if batch_mode:
            run_batch_eval(model, tokenizer, records, n=BATCH_EVAL_N)
        else:
            run_interactive(model, tokenizer, records)
    finally:
        if patched_base is not None:
            shutil.rmtree(patched_base, ignore_errors=True)
            print("\n🧹 Temporary tokenizer directory removed.")

    print()
    print("=" * 80)
    print("✅ COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()