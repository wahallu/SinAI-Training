"""
Extractive summarization via RAKE (Rapid Automatic Keyword Extraction).

Algorithm:
  1. Parse document into candidate keyword phrases by splitting on stop words and punctuation.
  2. Build word co-occurrence graph for candidate phrase words.
  3. Calculate word scores: score(w) = degree(w) / frequency(w).
  4. Score candidate phrases as the sum of member word scores.
  5. Score sentences by summing keyphrase scores contained within them.
  6. Return top-k sentences ordered by original position.

Usage:
    python extractive/rake.py --n 3
    python extractive/rake.py --test data/test.jsonl --out data/extractive_rake_preds.jsonl --n 3
"""

import argparse
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


# Common Sinhala Stopwords
SINHALA_STOPWORDS = {
    "සහ", "වන", "ලැබේ", "ඇත", "විසින්", "බව", "සඳහා", "මෙම", "අතර", "ගැන", "ගැනීම",
    "වූ", "සිට", "ද", "හා", "වෙත", "කර", "කරන", "කර ඇත", "ලෙස", "එම", "මගින්",
    "පිළිබඳ", "ගැනීමට", "ඉතිරි", "ලබා", "ලබා දීම", "නොමැත", "නොවේ", "ඇති", "නැත",
    "සිදු", "සිදු කර", "වී", "වීම", "පමණක්", "කිරීමට", "කිරීම", "වූහ", "වෙති", "නැවත",
    "අනුව", "වෙනුවෙන්", "ඔහු", "ඇය", "ඔවුන්", "එය", "මෙසේ", "අද", "ඊයේ", "හෙට"
}


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


def split_sentences(text: str, min_chars: int = 15) -> list[str]:
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in parts if len(s.strip()) >= min_chars]


def word_tokenize(text: str) -> list[str]:
    return [w for w in re.split(r'[^\w\u0D80-\u0DFF]+', text) if w.strip()]


def extract_candidate_keywords(text: str, stopwords: set[str]) -> list[list[str]]:
    """Split text into candidate phrases on punctuation or stop words."""
    words = word_tokenize(text.lower())
    phrases = []
    current_phrase = []

    for w in words:
        if w in stopwords or not re.search(r'[\u0D80-\u0DFF\w]', w):
            if current_phrase:
                phrases.append(current_phrase)
                current_phrase = []
        else:
            current_phrase.append(w)
    if current_phrase:
        phrases.append(current_phrase)

    return phrases


def calculate_word_scores(phrases: list[list[str]]) -> dict[str, float]:
    word_freq = Counter()
    word_degree = defaultdict(int)

    for phrase in phrases:
        degree = len(phrase) - 1
        for word in phrase:
            word_freq[word] += 1
            word_degree[word] += degree

    word_score = {}
    for word, freq in word_freq.items():
        # degree(w) includes self co-occurrence (freq + degree offset)
        deg = word_degree[word] + freq
        word_score[word] = deg / float(freq)

    return word_score


def rake_summarize(text: str, n: int = 3, stopwords: set[str] = SINHALA_STOPWORDS) -> str:
    """Return top-n sentences using RAKE keyphrase scoring."""
    sentences = split_sentences(text)
    if len(sentences) <= n:
        return " ".join(sentences)

    phrases = extract_candidate_keywords(text, stopwords)
    if not phrases:
        return " ".join(sentences[:n])

    word_scores = calculate_word_scores(phrases)

    sentence_scores = []
    for s in sentences:
        s_words = word_tokenize(s.lower())
        if not s_words:
            sentence_scores.append(0.0)
            continue
        
        score = sum(word_scores.get(w, 0.0) for w in s_words)
        # Length normalization
        sentence_scores.append(score / math.sqrt(len(s_words)))

    indexed = list(enumerate(sentence_scores))
    indexed.sort(key=lambda x: x[1], reverse=True)
    top_indices = sorted([idx for idx, _ in indexed[:n]])

    return " ".join(sentences[i] for i in top_indices)


def summarize(text: str, n: int = 3) -> str:
    return rake_summarize(text, n=n)


# ROUGE Evaluation helper
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

    p1, r1 = _ngrams(p_tok, 1), _ngrams(r_tok, 1)
    c1     = sum((p1 & r1).values())
    f1     = _f1(c1 / len(p_tok), c1 / len(r_tok))

    p2, r2 = _ngrams(p_tok, 2), _ngrams(r_tok, 2)
    c2     = sum((p2 & r2).values())
    f2     = _f1(c2 / max(len(p_tok) - 1, 1), c2 / max(len(r_tok) - 1, 1))

    lcs    = _lcs_length(p_tok, r_tok)
    fL     = _f1(lcs / len(p_tok), lcs / len(r_tok))

    return {"rouge1": f1, "rouge2": f2, "rougeL": fL}


def main(test_path: Path, out_path: Path, n: int, limit: int = None) -> None:
    print(f"Loading test set: {test_path}")
    if not test_path.exists():
        print(f"Test path {test_path} does not exist. Exiting main CLI.")
        return

    records = []
    with open(test_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    if limit:
        records = records[:limit]

    agg = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    results = []

    print(f"Running RAKE Summarizer (n={n}) ...")
    for rec in records:
        content = rec.get("content", "")
        title   = rec.get("title", "")
        summary = rake_summarize(content, n=n)
        scores  = rouge_scores(summary, title)

        for k in agg:
            agg[k] += scores[k]

        results.append({
            "title": title,
            "content": content,
            "extractive_summary": summary,
            "rouge1": round(scores["rouge1"], 4),
            "rouge2": round(scores["rouge2"], 4),
            "rougeL": round(scores["rougeL"], 4),
        })

    num = max(len(records), 1)
    print("\n" + "=" * 50)
    print(f"RAKE Summarizer (n={n}) — avg over {len(records):,} articles")
    print("=" * 50)
    print(f"  ROUGE-1 : {agg['rouge1'] / num:.4f}")
    print(f"  ROUGE-2 : {agg['rouge2'] / num:.4f}")
    print(f"  ROUGE-L : {agg['rougeL'] / num:.4f}")
    print("=" * 50)

    with open(out_path, "w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\nPredictions saved → {out_path}")


if __name__ == "__main__":
    _HERE = Path(__file__).parent.parent
    _DATA = _HERE / "data"

    parser = argparse.ArgumentParser(description="RAKE extractive summarizer")
    parser.add_argument("--test", default=str(_DATA / "test.jsonl"))
    parser.add_argument("--out",  default=str(_DATA / "extractive_rake_preds.jsonl"))
    parser.add_argument("--n",    type=int, default=3)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    main(Path(args.test), Path(args.out), args.n, args.limit)
