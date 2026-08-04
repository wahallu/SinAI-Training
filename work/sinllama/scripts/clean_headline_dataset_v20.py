"""Strip scraper artifacts from both sides of the headline training pair.

v19's clean_headline_dataset.py only cleaned the reference headline (the
`output` field) and only stripped tags anchored to the end of the string.
That measurably worked -- artifact rate dropped from 11.2% to 1.1% overall
(see CLAUDE.md) -- but it wasn't zero, and live-server testing on fresh
articles after v19 still showed tags leaking through, e.g.:

    "ඇපල් AppStore හි යළිත් වැඩ පෙන්වන්නේ Telegram ය! (වීඩියෝ)"
    "... [video]", "... [photo]", "... - photo"

Two gaps caused this:
  1. The word list / separator class didn't cover square brackets
     ("[video]") or a bare word after a dash with no brackets ("- photo").
  2. Cleaning only the *label* (`output`) never touched the *input*
     (`article` text). Scraped articles often carry an inline tag right next
     to the sentence a headline gets built from -- "...Telegram ය. (Video)
     මේ පිළිබඳව..." -- and the model can copy that tag into the headline
     regardless of how clean the training labels are, because the tag is
     sitting in its input context, not just in something it memorized.

This script fixes both: a wider artifact-word/separator list, and it also
strips inline tags (not just trailing ones) from the `input` article field,
matching the equivalent fix on the serving side
(SinhalaJournalLLM/apps/backend-api/app/core/text_cleaning.py --
strip_article_media_tags / strip_headline_artifacts). Keep the word list and
separator class here in sync with that file; they don't need to be
byte-identical regexes (this one runs on dataset preprocessing, that one on
live request text) but a tag either script misses is a tag the other has to
catch instead.

Usage (on the GPU box, from work/sinllama/):
    python scripts/clean_headline_dataset_v20.py
"""

import json
import re
import unicodedata

DATA_DIR = "/home/jovyan/work/sinllama/data"
TRAIN_IN  = f"{DATA_DIR}/headline_dataset_48k_balanced_train.jsonl"
VAL_IN    = f"{DATA_DIR}/headline_dataset_48k_balanced_val.jsonl"
TRAIN_OUT = f"{DATA_DIR}/headline_dataset_48k_balanced_train_clean_v20.jsonl"
VAL_OUT   = f"{DATA_DIR}/headline_dataset_48k_balanced_val_clean_v20.jsonl"

MIN_HEADLINE_CHARS = 5
MIN_ARTICLE_CHARS  = 50

# Widened from v19: added GALLERY, and the separator class now includes
# square brackets and a couple more punctuation marks scrapers use to glue a
# tag onto the end of a sentence ("- photo" with no brackets at all).
ARTIFACT_WORD = (
    r"(?:වීඩියෝ|ඡායාරූප|VIDEO|PHOTOS?|Video|PICTURES?|Interview|GALLERY)"
)
SEP = r"[\s\-–—:()\[\]|•~!]*"

# Anchored to the end -- for the reference headline, same role as v19's.
ARTIFACT_TAIL = re.compile(
    rf"{SEP}{ARTIFACT_WORD}(?:{SEP}{ARTIFACT_WORD})*{SEP}$",
    re.IGNORECASE,
)

# NOT anchored -- for the article body, where a tag can sit mid-text right
# after any sentence, not just at the very end. Matches a bracketed tag
# ("(Video)", "[Photo]") or a dash-prefixed bare word ("- Video") following
# whitespace/sentence-end punctuation, so it won't eat a legitimate mid-word
# occurrence of one of these tokens.
ARTIFACT_INLINE = re.compile(
    rf"[(\[]\s*{ARTIFACT_WORD}\s*[)\]]|(?<=[\s.!?])[\-–—]\s*{ARTIFACT_WORD}\b",
    re.IGNORECASE,
)


def normalize_sinhala(text):
    return unicodedata.normalize("NFC", text.strip()) if text else ""


def has_sinhala(text):
    return any("඀" <= ch <= "෿" for ch in text)


def clean_headline(headline):
    """Returns (cleaned_text, was_changed). Trailing-only, same as v19."""
    original = normalize_sinhala(headline)
    cleaned = ARTIFACT_TAIL.sub("", original).strip()
    cleaned = re.sub(r"[\s\-–—:!]+$", "", cleaned).strip()
    return cleaned, cleaned != original


def clean_article(article):
    """Returns (cleaned_text, was_changed). Strips inline tags anywhere in
    the article body, then collapses the double-spacing left behind."""
    original = normalize_sinhala(article)
    cleaned = ARTIFACT_INLINE.sub("", original)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).strip()
    return cleaned, cleaned != original


def clean_file(path_in, path_out, label):
    total = hl_changed = art_changed = dropped = 0
    band_before = {"short": 0, "medium": 0, "long": 0, "other": 0}
    band_after  = {"short": 0, "medium": 0, "long": 0, "other": 0}

    def band_of(words):
        n = len(words.split())
        if 3 <= n <= 5: return "short"
        if 6 <= n <= 7: return "medium"
        if 8 <= n <= 10: return "long"
        return "other"

    with open(path_in, "r", encoding="utf-8") as fin, \
         open(path_out, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            rec = json.loads(line)

            headline_original = normalize_sinhala(rec.get("output", ""))
            band_before[band_of(headline_original)] += 1

            headline_clean, hl_was_changed = clean_headline(headline_original)
            if hl_was_changed:
                hl_changed += 1

            if not headline_clean or len(headline_clean) < MIN_HEADLINE_CHARS \
                    or not has_sinhala(headline_clean):
                dropped += 1
                continue

            # `input` carries "Category: X\nArticle: <body>" -- only the
            # article body gets cleaned, the category line is left alone.
            raw_input = rec.get("input", "")
            if "Article:" in raw_input:
                prefix, body = raw_input.split("Article:", 1)
                body_clean, art_was_changed = clean_article(body)
                if art_was_changed:
                    art_changed += 1
                if len(body_clean.strip()) < MIN_ARTICLE_CHARS:
                    dropped += 1
                    continue
                rec["input"] = f"{prefix}Article: {body_clean}"
            band_after[band_of(headline_clean)] += 1
            rec["output"] = headline_clean
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\n{label}:")
    print(f"  Total                 : {total}")
    print(f"  Headline tag stripped : {hl_changed} ({hl_changed/total*100:.1f}%)")
    print(f"  Article tag stripped  : {art_changed} ({art_changed/total*100:.1f}%)")
    print(f"  Dropped (empty/invalid after cleaning): {dropped}")
    print(f"  Kept                  : {total - dropped}")
    print(f"  Band distribution (before -> after):")
    for b in ("short", "medium", "long", "other"):
        print(f"    {b:8s}: {band_before[b]:6d} -> {band_after[b]:6d}")


if __name__ == "__main__":
    clean_file(TRAIN_IN, TRAIN_OUT, "TRAIN")
    clean_file(VAL_IN, VAL_OUT, "VAL")
    print(f"\nWrote:\n  {TRAIN_OUT}\n  {VAL_OUT}")
    print("\nNext: train_headline_v20.py trains on these cleaned files.")
