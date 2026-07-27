"""
Extractive summarization via TextRank.

Algorithm:
  1. Split article into sentences.
  2. Represent each sentence as a TF-IDF vector (grapheme-cluster tokens).
  3. Build cosine-similarity matrix between all sentence pairs.
  4. Run PageRank on the similarity graph.
  5. Return top-k sentences ordered by their original position.

Usage (from any directory):
    python extractive/textrank.py --n 3
    python extractive/textrank.py --test data/test.jsonl --out data/extractive_preds.jsonl --n 3

Prerequisites:
    python data/split_dataset.py    ← generates data/test.jsonl first

Output file: one JSON object per line —
    { "title": ..., "content": ..., "extractive_summary": ... }

ROUGE is computed inline using the grapheme-cluster implementation that
handles Sinhala Unicode correctly (rouge_score library breaks on Sinhala).
"""

import argparse
import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np


# ─────────────────────────────────────────────
# SINHALA TEXT UTILITIES
# ─────────────────────────────────────────────

def grapheme_tokenize(text: str) -> list[str]:
    """Character-level grapheme-cluster tokenizer — groups base + combining
    diacritics so e.g. 'ක්' is one token, not two."""
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


def split_sentences(text: str, min_chars: int = 15) -> list[str]:
    """Split on sentence-ending punctuation followed by whitespace."""
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in parts if len(s.strip()) >= min_chars]


# ─────────────────────────────────────────────
# TF-IDF + COSINE SIMILARITY
# ─────────────────────────────────────────────

def _tf(tokens: list[str]) -> dict[str, float]:
    n = len(tokens)
    if n == 0:
        return {}
    counts = Counter(tokens)
    return {t: c / n for t, c in counts.items()}


def _idf(token_lists: list[list[str]]) -> dict[str, float]:
    N = len(token_lists)
    df: Counter = Counter()
    for toks in token_lists:
        df.update(set(toks))
    return {t: math.log((N + 1) / (df[t] + 1)) + 1.0 for t in df}


def _tfidf_vec(tokens: list[str], idf: dict[str, float]) -> dict[str, float]:
    tf = _tf(tokens)
    return {t: tf[t] * idf.get(t, 1.0) for t in tf}


def _cosine(v1: dict[str, float], v2: dict[str, float]) -> float:
    common = set(v1) & set(v2)
    if not common:
        return 0.0
    dot  = sum(v1[t] * v2[t] for t in common)
    mag1 = math.sqrt(sum(x * x for x in v1.values()))
    mag2 = math.sqrt(sum(x * x for x in v2.values()))
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot / (mag1 * mag2)


# ─────────────────────────────────────────────
# PAGE RANK
# ─────────────────────────────────────────────

def _pagerank(
    sim: np.ndarray,
    damping: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> np.ndarray:
    n = sim.shape[0]
    row_sums = sim.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    P = (sim / row_sums).T                   # column-stochastic
    scores = np.ones(n) / n
    base   = (1 - damping) / n
    for _ in range(max_iter):
        new_scores = base + damping * (P @ scores)
        if np.abs(new_scores - scores).max() < tol:
            break
        scores = new_scores
    return scores


# ─────────────────────────────────────────────
# TEXTRANK CORE
# ─────────────────────────────────────────────

def textrank_summarize(text: str, n: int = 3) -> str:
    """Return the top-n most important sentences from `text`, preserving order."""
    sentences = split_sentences(text)
    if len(sentences) <= n:
        return " ".join(sentences)

    token_lists = [grapheme_tokenize(s) for s in sentences]
    idf         = _idf(token_lists)
    vecs        = [_tfidf_vec(toks, idf) for toks in token_lists]

    # Build similarity matrix
    m = len(sentences)
    sim = np.zeros((m, m))
    for i in range(m):
        for j in range(i + 1, m):
            s = _cosine(vecs[i], vecs[j])
            sim[i, j] = s
            sim[j, i] = s

    scores      = _pagerank(sim)
    top_indices = sorted(np.argsort(scores)[-n:])          # keep original order
    return " ".join(sentences[i] for i in top_indices)


def summarize(text: str, n: int = 3) -> str:
    return textrank_summarize(text, n=n)


# ─────────────────────────────────────────────
# ROUGE (grapheme-cluster, no external library)
# ─────────────────────────────────────────────

def _ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def _lcs_length(a: list, b: list) -> int:
    m, n = len(a), len(b)
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            curr[j] = prev[j - 1] + 1 if a[i - 1] == b[j - 1] else max(prev[j], curr[j - 1])
        prev = curr
    return prev[n]


def rouge_scores(pred: str, ref: str) -> dict[str, float]:
    p_tok = grapheme_tokenize(pred)
    r_tok = grapheme_tokenize(ref)
    if not p_tok or not r_tok:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    def _f1(prec, rec):
        return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    # ROUGE-1
    p1, r1 = _ngrams(p_tok, 1), _ngrams(r_tok, 1)
    c1     = sum((p1 & r1).values())
    f1     = _f1(c1 / len(p_tok), c1 / len(r_tok))

    # ROUGE-2
    p2, r2 = _ngrams(p_tok, 2), _ngrams(r_tok, 2)
    c2     = sum((p2 & r2).values())
    f2     = _f1(c2 / max(len(p_tok) - 1, 1), c2 / max(len(r_tok) - 1, 1))

    # ROUGE-L
    lcs    = _lcs_length(p_tok, r_tok)
    fL     = _f1(lcs / len(p_tok), lcs / len(r_tok))

    return {"rouge1": f1, "rouge2": f2, "rougeL": fL}


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main(test_path: Path, out_path: Path, n: int, limit: int = None) -> None:
    print(f"Loading test set: {test_path}")
    records = []
    with open(test_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"  Articles: {len(records):,}")

    if limit:
        print(f"  Limiting to first {limit} articles.")
        records = records[:limit]

    agg = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    results = []

    print(f"Running TextRank (n={n} sentences) ...")
    for i, rec in enumerate(records):
        content = rec.get("content", "")
        title   = rec.get("title", "")
        summary = textrank_summarize(content, n=n)
        scores  = rouge_scores(summary, title)

        for k in agg:
            agg[k] += scores[k]

        results.append({
            "title"              : title,
            "content"            : content,
            "extractive_summary" : summary,
            "rouge1"             : round(scores["rouge1"], 4),
            "rouge2"             : round(scores["rouge2"], 4),
            "rougeL"             : round(scores["rougeL"], 4),
        })

        if (i + 1) % 1000 == 0:
            print(f"  {i + 1:>6,} / {len(records):,}")

    num = len(records)
    print("\n" + "=" * 50)
    print(f"TextRank  (n={n})  —  avg over {num:,} articles")
    print("=" * 50)
    print(f"  ROUGE-1 : {agg['rouge1'] / num:.4f}")
    print(f"  ROUGE-2 : {agg['rouge2'] / num:.4f}")
    print(f"  ROUGE-L : {agg['rougeL'] / num:.4f}")
    print("=" * 50)

    with open(out_path, "w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\nPredictions saved → {out_path}")

    # ── Qualitative samples ──────────────────────
    print("\n── Sample predictions ──────────────────────")
    for rec in results[:5]:
        print(f"\n  Title   : {rec['title']}")
        print(f"  Extracted: {rec['extractive_summary'][:200]}...")
        print(f"  ROUGE-L : {rec['rougeL']:.4f}")


if __name__ == "__main__":
    _HERE = Path(__file__).parent.parent    # summarizer/
    _DATA = _HERE / "data"

    parser = argparse.ArgumentParser(description="TextRank extractive summarizer")
    parser.add_argument("--test", default=str(_DATA / "test.jsonl"),
                        help="Test JSONL (from split_dataset.py)")
    parser.add_argument("--out",  default=str(_DATA / "extractive_preds.jsonl"),
                        help="Output predictions JSONL")
    parser.add_argument("--n",    type=int, default=3,
                        help="Number of sentences to extract (default: 3)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit the number of articles to process")
    args = parser.parse_args()
    main(Path(args.test), Path(args.out), args.n, args.limit)
