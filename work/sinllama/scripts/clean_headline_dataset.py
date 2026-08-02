"""Strip trailing scraper artifacts from headline training references.

Run on the GPU box before train_headline_v19.py. Motivation: the 300-article
per-band eval of v18 (scripts/test_headline_v18.py) measured a 22.3% artifact
rate on the long band and 11.0% on medium -- the model learned "long headlines
often end in a media tag" because a meaningful share of the 48K training
references do end in Hiru/ITN scraper tags like "(වීඩියෝ)", "- VIDEO",
"Interview-PHOTOS". Those tags are also extra words, so tagged headlines
cluster into exactly the medium/long buckets v18 was trained to fill.

This does NOT retrain anything -- it only rewrites the reference headline
field, dropping the artifact suffix (and, after cleaning, any example whose
headline no longer has a valid word band). Word-count bands are recomputed
downstream in train_headline_v19.py from the cleaned text, so a headline that
shrinks from "long" to "medium" after cleaning lands in the band it actually
belongs to.

Usage (on the GPU box, from work/sinllama/):
    python scripts/clean_headline_dataset.py
"""

import json
import re
import unicodedata

DATA_DIR = "/home/jovyan/work/sinllama/data"
TRAIN_IN  = f"{DATA_DIR}/headline_dataset_48k_balanced_train.jsonl"
VAL_IN    = f"{DATA_DIR}/headline_dataset_48k_balanced_val.jsonl"
TRAIN_OUT = f"{DATA_DIR}/headline_dataset_48k_balanced_train_clean.jsonl"
VAL_OUT   = f"{DATA_DIR}/headline_dataset_48k_balanced_val_clean.jsonl"

MIN_HEADLINE_CHARS = 5

# Same words test_headline_v18.py / compare_v17_v18.py flagged as scraper junk,
# plus the separators/brackets/dashes that glue a chain of them together
# ("Interview-VIDEO Interview-PHOTOS", "(වීඩියෝ)", "- VIDEO!").
ARTIFACT_WORD = r"(?:වීඩියෝ|ඡායාරූප|VIDEO|PHOTOS?|Video|PICTURES?|Interview)"
SEP           = r"[\s\-–—:()!]*"

# Matches a run of artifact words (with separators/punctuation between and
# around them) anchored to the END of the string, so only a trailing tag gets
# removed -- a headline that legitimately mentions "photos" mid-sentence is
# untouched.
ARTIFACT_TAIL = re.compile(
    rf"{SEP}{ARTIFACT_WORD}(?:{SEP}{ARTIFACT_WORD})*{SEP}$",
    re.IGNORECASE,
)


def normalize_sinhala(text):
    return unicodedata.normalize("NFC", text.strip()) if text else ""


def has_sinhala(text):
    return any("඀" <= ch <= "෿" for ch in text)


def clean_headline(headline):
    """Returns (cleaned_text, was_changed)."""
    original = normalize_sinhala(headline)
    cleaned = ARTIFACT_TAIL.sub("", original).strip()
    # Trailing punctuation left behind once the tag is gone (e.g. the "-" in
    # "...ජයග්‍රහණය -" once "(වීඩියෝ)" is stripped from further down the line).
    cleaned = re.sub(r"[\s\-–—:!]+$", "", cleaned).strip()
    return cleaned, cleaned != original


def clean_file(path_in, path_out, label):
    total = changed = dropped = 0
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
            original = normalize_sinhala(rec.get("output", ""))
            band_before[band_of(original)] += 1

            cleaned, was_changed = clean_headline(original)
            if was_changed:
                changed += 1

            if not cleaned or len(cleaned) < MIN_HEADLINE_CHARS or not has_sinhala(cleaned):
                dropped += 1
                continue

            band_after[band_of(cleaned)] += 1
            rec["output"] = cleaned
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\n{label}:")
    print(f"  Total          : {total}")
    print(f"  Tag stripped   : {changed} ({changed/total*100:.1f}%)")
    print(f"  Dropped (empty/invalid after cleaning): {dropped}")
    print(f"  Kept           : {total - dropped}")
    print(f"  Band distribution (before -> after):")
    for b in ("short", "medium", "long", "other"):
        print(f"    {b:8s}: {band_before[b]:6d} -> {band_after[b]:6d}")


if __name__ == "__main__":
    clean_file(TRAIN_IN, TRAIN_OUT, "TRAIN")
    clean_file(VAL_IN, VAL_OUT, "VAL")
    print(f"\nWrote:\n  {TRAIN_OUT}\n  {VAL_OUT}")
    print("\nNext: train_headline_v19.py trains on these cleaned files.")
