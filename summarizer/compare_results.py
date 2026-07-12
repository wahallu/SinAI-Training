"""
Compare extractive (TextRank) vs abstractive (mT5) summarization.

Reads prediction files produced by:
  - summarizer/extractive/textrank.py         → data/extractive_preds.jsonl
  - summarizer/abstractive/evaluate_mt5.py    → data/mt5_preds.jsonl

Outputs:
  1. ROUGE comparison table  (primary: both vs article title)
  2. Qualitative side-by-side samples
  3. Secondary evaluation vs 500 LLM gold summaries (if URLs/titles match)

Usage:
    cd /home/jovyan
    python summarizer/compare_results.py
"""

import json
import unicodedata
from collections import Counter
from pathlib import Path

HERE           = Path(__file__).parent
EXT_PREDS      = HERE / "data" / "extractive_preds.jsonl"
MT5_PREDS      = HERE / "data" / "mt5_preds.jsonl"
GOLD_SUMMARIES = HERE / "summarization_dataset.jsonl"   # 500 LLM gold summaries

N_QUALITATIVE  = 8


# ─────────────────────────────────────────────
# ROUGE — grapheme-cluster, Sinhala-safe
# ─────────────────────────────────────────────

def grapheme_tokenize(text: str) -> list[str]:
    tokens, chars, i = [], list(text), 0
    while i < len(chars):
        cluster = chars[i]
        i += 1
        while i < len(chars) and unicodedata.combining(chars[i]):
            cluster += chars[i]
            i += 1
        if cluster.strip():
            tokens.append(cluster)
    return tokens


def _ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def _lcs(a: list, b: list) -> int:
    m, n = len(a), len(b)
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            curr[j] = (prev[j - 1] + 1
                       if a[i - 1] == b[j - 1]
                       else max(prev[j], curr[j - 1]))
        prev = curr
    return prev[n]


def rouge(pred: str, ref: str) -> dict[str, float]:
    p, r = grapheme_tokenize(pred), grapheme_tokenize(ref)
    if not p or not r:
        return {"r1": 0.0, "r2": 0.0, "rL": 0.0}

    def f1(a, b):
        return 2 * a * b / (a + b) if (a + b) else 0.0

    p1, r1 = _ngrams(p, 1), _ngrams(r, 1)
    c1 = sum((p1 & r1).values())

    p2, r2 = _ngrams(p, 2), _ngrams(r, 2)
    c2 = sum((p2 & r2).values())

    lcs = _lcs(p, r)

    return {
        "r1": f1(c1 / len(p), c1 / len(r)),
        "r2": f1(c2 / max(len(p) - 1, 1), c2 / max(len(r) - 1, 1)),
        "rL": f1(lcs / len(p), lcs / len(r)),
    }


def avg_rouge(preds: list[str], refs: list[str]) -> dict[str, float]:
    agg = {"r1": 0.0, "r2": 0.0, "rL": 0.0}
    for p, r in zip(preds, refs):
        s = rouge(p, r)
        for k in agg:
            agg[k] += s[k]
    n = len(preds)
    return {k: v / n for k, v in agg.items()}


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def print_row(label: str, scores: dict[str, float]) -> None:
    print(f"  {label:<30} R-1={scores['r1']:.4f}   R-2={scores['r2']:.4f}   R-L={scores['rL']:.4f}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main() -> None:
    for path in (EXT_PREDS, MT5_PREDS):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found.\n"
                "Run textrank.py and evaluate_mt5.py first."
            )

    print(f"Loading extractive predictions : {EXT_PREDS}")
    ext_rows = load_jsonl(EXT_PREDS)
    print(f"Loading mT5 predictions        : {MT5_PREDS}")
    mt5_rows = load_jsonl(MT5_PREDS)

    if len(ext_rows) != len(mt5_rows):
        print(f"WARNING: row counts differ  ext={len(ext_rows):,}  mt5={len(mt5_rows):,}")
        n = min(len(ext_rows), len(mt5_rows))
        ext_rows, mt5_rows = ext_rows[:n], mt5_rows[:n]

    titles        = [r["title"]               for r in ext_rows]
    ext_summaries = [r["extractive_summary"]   for r in ext_rows]
    mt5_summaries = [r["mt5_summary"]          for r in mt5_rows]

    # ── Primary: both vs article title ──────────────────────
    ext_scores = avg_rouge(ext_summaries, titles)
    mt5_scores = avg_rouge(mt5_summaries, titles)

    print("\n" + "=" * 65)
    print(f"  PRIMARY EVALUATION  (reference = article title, n={len(titles):,})")
    print("=" * 65)
    print_row("TextRank  (extractive)", ext_scores)
    print_row("mT5       (abstractive)", mt5_scores)
    print("=" * 65)

    winner = "mT5" if mt5_scores["rL"] > ext_scores["rL"] else "TextRank"
    diff   = abs(mt5_scores["rL"] - ext_scores["rL"])
    print(f"\n  ROUGE-L winner: {winner}  (margin +{diff:.4f})")

    # ── Secondary: both vs 500 LLM gold summaries ───────────
    if GOLD_SUMMARIES.exists():
        gold_rows = [r for r in load_jsonl(GOLD_SUMMARIES) if r.get("summary", "").strip()]
        print(f"\n  {len(gold_rows)} LLM gold summaries found → secondary evaluation")

        gold_by_url   = {r.get("url", ""): r for r in gold_rows if r.get("url")}
        gold_by_title = {r["title"]: r          for r in gold_rows}

        ext_g, mt5_g, gold_refs = [], [], []
        for ext_r, mt5_r in zip(ext_rows, mt5_rows):
            url   = ext_r.get("url", "")
            title = ext_r.get("title", "")
            gold  = (gold_by_url.get(url) or gold_by_title.get(title))
            if gold:
                ext_g.append(ext_r["extractive_summary"])
                mt5_g.append(mt5_r["mt5_summary"])
                gold_refs.append(gold["summary"])

        if gold_refs:
            ext_g_scores = avg_rouge(ext_g, gold_refs)
            mt5_g_scores = avg_rouge(mt5_g, gold_refs)
            print("\n" + "=" * 65)
            print(f"  SECONDARY EVALUATION  (reference = LLM summary, n={len(gold_refs)})")
            print("=" * 65)
            print_row("TextRank  (extractive)", ext_g_scores)
            print_row("mT5       (abstractive)", mt5_g_scores)
            print("=" * 65)
        else:
            print("  No URL/title overlap between test set and gold summaries.")
    else:
        print(f"\n  {GOLD_SUMMARIES.name} not found — skipping secondary evaluation.")

    # ── Qualitative side-by-side ─────────────────────────────
    print("\n" + "=" * 65)
    print(f"  QUALITATIVE SAMPLES  (first {N_QUALITATIVE})")
    print("=" * 65)

    for i, (ext_r, mt5_r) in enumerate(zip(ext_rows[:N_QUALITATIVE], mt5_rows[:N_QUALITATIVE])):
        title   = ext_r["title"]
        content = ext_r["content"][:250]
        ext_sum = ext_r["extractive_summary"]
        mt5_sum = mt5_r["mt5_summary"]

        ext_s = rouge(ext_sum, title)
        mt5_s = rouge(mt5_sum, title)

        print(f"\n  [{i+1}] Article snippet:")
        print(f"       {content}...")
        print(f"\n       Title    : {title}")
        print(f"       TextRank : {ext_sum[:180]}{'...' if len(ext_sum) > 180 else ''}")
        print(f"       mT5      : {mt5_sum}")
        print(f"\n       ROUGE-L  →  TextRank: {ext_s['rL']:.4f}   mT5: {mt5_s['rL']:.4f}")


if __name__ == "__main__":
    main()
