# ============================================================
# clean_style_dataset.py
#
# Fixes two root-cause data-quality problems found in
# style_dataset3_corrected.jsonl (the dataset staged for the
# v13 style-rewriter retrain):
#
# 1. FORCED OPEN/CLOSE TEMPLATES
#    Correct_style_dataset.py (the correction pass that produced
#    this file) hard-required an exact opening and/or closing
#    sentence per style (editorial, youth, feature) and had an
#    external LLM insert it into nearly every row - the opposite
#    of generate_style_dataset.py's own instructions, which
#    explicitly told the generator NOT to force these exact
#    sentences into every article. Verified: for every row where
#    the marker is present, it is the row's final sentence with
#    nothing meaningful after it (100% across a full scan), so it
#    can be removed as a whole trailing sentence with no risk of
#    cutting real content.
#
#    This is why the deployed model bolts on a generic opener/
#    closer almost regardless of the article ("Forced editorial
#    opener", "mechanical style markers", "generic ending" in
#    user QA notes).
#
# 2. CORRUPTED SINHALA GRAPHEMES
#    4-11% of rows (worst in feature) contain a virama
#    (U+0DCA) immediately followed by a dependent vowel sign,
#    with no consonant between them - an invalid Sinhala
#    grapheme sequence that cannot occur in real written Sinhala
#    (confirmed: 0% false-positive rate across the ~7,000 scraped
#    `content` fields in this same file). This is a text-
#    generation defect from the correction-pass LLM, not noise.
#    Rows containing it are dropped rather than "fixed" - there's
#    no reliable way to guess which character was corrupted.
#
# Usage:
#   python clean_style_dataset.py
#
# Reads  : style_rewriter/data/style_dataset3_corrected.jsonl
# Writes : style_rewriter/data/style_dataset3_corrected_clean.jsonl
# (input file is left untouched)
# ============================================================

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

INPUT_PATH = Path("/home/jovyan/style_rewriter/data/style_dataset3_corrected.jsonl")
OUTPUT_PATH = Path("/home/jovyan/style_rewriter/data/style_dataset3_corrected_clean.jsonl")

MIN_OUTPUT_LEN = 50  # matches train_style.py's own minimum-length filter

# ------------------------------------------------------------
# Mandatory templated phrases forced by Correct_style_dataset.py.
# These are stripped as whole sentences, not just the substring.
# ------------------------------------------------------------
FORCED_OPENINGS = {
    "style_2_editorial": ["විශ්ලේෂණය කරන විට"],
    "style_4_youth": ["දන්නවද?", "ඇහුවද?", "මේක අහන්න!"],
}

FORCED_CLOSINGS = {
    "style_2_editorial": ["ඉදිරි ක්‍රියාමාර්ග පිළිබඳව සාකච්ඡා කළ යුතු කාලය එළඹ ඇත"],
    "style_4_youth": ["අනිවාර්යයෙන්ම දැනගන්න"],
    "style_5_feature": ["අනාගත පරම්පරාවට ද ආදර්ශයක් වනු ඇත"],
}

SENTENCE_END_RE = re.compile(r"[.!?]\s+")

# Sinhala dependent vowel signs. A virama (්) immediately followed by one
# of these, with no consonant in between, is not a valid grapheme.
VOWEL_SIGNS = "ාැෑිීුූෘෙේෛොෝෞෟ"
GARBLE_RE = re.compile(f"[්][{VOWEL_SIGNS}]")


def strip_trailing_marker_sentence(text: str, markers: list) -> tuple:
    """Remove the final sentence if it contains one of `markers`.
    Returns (new_text, stripped: bool)."""
    for marker in markers:
        idx = text.find(marker)
        if idx == -1:
            continue
        # Confirm nothing meaningful follows the marker (it should be the
        # last sentence - verified 100% true on this dataset).
        trailing = text[idx + len(marker):].strip().strip("./!?…")
        if len(trailing) > 2:
            # Unexpected: marker isn't actually trailing. Leave it alone
            # rather than risk cutting real content.
            continue

        # Walk back to the previous sentence boundary.
        seg_start = 0
        for m in SENTENCE_END_RE.finditer(text[:idx]):
            seg_start = m.end()

        new_text = text[:seg_start].rstrip()
        return new_text, True

    return text, False


def strip_leading_marker_sentence(text: str, markers: list) -> tuple:
    """Remove the first sentence if it contains one of `markers` very
    close to the start of the text. Returns (new_text, stripped: bool)."""
    for marker in markers:
        idx = text.find(marker)
        if idx == -1 or idx > 5:
            # Only treat it as a forced opener if it's essentially at
            # position 0 - otherwise it may be organic mid-text usage.
            continue

        m = SENTENCE_END_RE.search(text)
        if not m:
            continue

        new_text = text[m.end():].strip()
        return new_text, True

    return text, False


def clean_record(rec: dict, stats: Counter, examples: defaultdict):
    style = rec.get("style")
    rewritten = (rec.get("rewritten_text") or "").strip()
    original_len = len(rewritten)

    if style in FORCED_OPENINGS:
        before = rewritten
        rewritten, stripped = strip_leading_marker_sentence(rewritten, FORCED_OPENINGS[style])
        if stripped:
            stats[f"{style}:stripped_opening"] += 1
            if len(examples[f"{style}:opening"]) < 2:
                examples[f"{style}:opening"].append((before[:120], rewritten[:120]))

    if style in FORCED_CLOSINGS:
        before = rewritten
        rewritten, stripped = strip_trailing_marker_sentence(rewritten, FORCED_CLOSINGS[style])
        if stripped:
            stats[f"{style}:stripped_closing"] += 1
            if len(examples[f"{style}:closing"]) < 2:
                examples[f"{style}:closing"].append((before[-160:], rewritten[-160:]))

    if GARBLE_RE.search(rewritten):
        stats[f"{style}:dropped_garbled"] += 1
        return None

    if len(rewritten) < MIN_OUTPUT_LEN:
        stats[f"{style}:dropped_short"] += 1
        return None

    if len(rewritten) != original_len:
        stats[f"{style}:modified"] += 1

    rec = dict(rec)
    rec["rewritten_text"] = rewritten
    return rec


def main():
    records = []
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print(f"Loaded {len(records):,} rows from {INPUT_PATH.name}")

    stats = Counter()
    examples = defaultdict(list)
    by_style_in = Counter(r.get("style") for r in records)

    cleaned = []
    for rec in records:
        out = clean_record(rec, stats, examples)
        if out is not None:
            cleaned.append(out)

    by_style_out = Counter(r.get("style") for r in cleaned)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for rec in cleaned:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(cleaned):,} rows to {OUTPUT_PATH.name}")

    print("\n" + "=" * 70)
    print("PER-STYLE SUMMARY")
    print("=" * 70)
    for style in sorted(by_style_in):
        n_in = by_style_in[style]
        n_out = by_style_out.get(style, 0)
        print(f"\n[{style}]  {n_in:,} -> {n_out:,} rows")
        for key in ("stripped_opening", "stripped_closing", "dropped_garbled", "dropped_short"):
            v = stats.get(f"{style}:{key}", 0)
            if v:
                print(f"   {key:<18}: {v:,} ({100*v/n_in:.1f}%)")

    print("\n" + "=" * 70)
    print("SAMPLE BEFORE / AFTER (closing-sentence strip)")
    print("=" * 70)
    for key, exs in examples.items():
        if not key.endswith(":closing"):
            continue
        print(f"\n[{key}]")
        for before, after in exs:
            print(f"  BEFORE: ...{before}")
            print(f"  AFTER : ...{after}")

    total_in = sum(by_style_in.values())
    total_out = sum(by_style_out.values())
    print("\n" + "=" * 70)
    print(f"TOTAL: {total_in:,} -> {total_out:,} rows "
          f"({100*(total_in-total_out)/total_in:.1f}% dropped)")
    print("=" * 70)


if __name__ == "__main__":
    main()
