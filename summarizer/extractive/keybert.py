"""
Extractive summarization via KeyBERT-style Semantic Similarity & MMR.

Algorithm:
  1. Extract candidate sentences and full document representation.
  2. Compute semantic vector embeddings for sentences and document.
     - Uses `sentence-transformers` if available.
     - Falls back to high-dimensional Sinhala grapheme n-gram TF-IDF vector embeddings via NumPy.
  3. Rank sentences using Cosine Similarity & Maximal Marginal Relevance (MMR).
  4. Return top-k sentences ordered by original position.

Usage:
    python extractive/keybert.py --n 3
    python extractive/keybert.py --test data/test.jsonl --out data/extractive_keybert_preds.jsonl --n 3
"""

import argparse
import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path
import numpy as np

# Optional SentenceTransformers import
_HAS_SBERT = False
_SBERT_MODEL = None

try:
    from sentence_transformers import SentenceTransformer
    _HAS_SBERT = True
except ImportError:
    _HAS_SBERT = False


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


def _get_fallback_embeddings(sentences: list[str], full_text: str) -> tuple[np.ndarray, np.ndarray]:
    """Generates TF-IDF + Grapheme n-gram dense embeddings as a lightweight fallback."""
    corpus = sentences + [full_text]
    token_lists = [grapheme_tokenize(s) for s in corpus]
    
    # Vocabulary building
    vocab = {}
    for toks in token_lists:
        for t in toks:
            if t not in vocab:
                vocab[t] = len(vocab)
                
    num_vocab = max(len(vocab), 1)
    N = len(corpus)
    
    # IDF calculation
    df = Counter()
    for toks in token_lists:
        df.update(set(toks))
    
    idf = np.zeros(num_vocab)
    for term, idx in vocab.items():
        idf[idx] = math.log((N + 1.0) / (df[term] + 1.0)) + 1.0

    # TF-IDF matrix construction
    vectors = np.zeros((N, num_vocab))
    for i, toks in enumerate(token_lists):
        if not toks:
            continue
        counts = Counter(toks)
        n_toks = float(len(toks))
        for term, c in counts.items():
            idx = vocab[term]
            vectors[i, idx] = (c / n_toks) * idf[idx]
            
        # L2 norm
        norm = np.linalg.norm(vectors[i])
        if norm > 0:
            vectors[i] = vectors[i] / norm

    sentence_vecs = vectors[:-1]
    doc_vec = vectors[-1]
    return sentence_vecs, doc_vec


def keybert_summarize(text: str, n: int = 3, diversity: float = 0.5) -> str:
    """Return top-n sentences using KeyBERT-style semantic embeddings and MMR."""
    sentences = split_sentences(text)
    if len(sentences) <= n:
        return " ".join(sentences)

    global _SBERT_MODEL
    sentence_embeddings = None
    doc_embedding = None

    if _HAS_SBERT:
        try:
            if _SBERT_MODEL is None:
                _SBERT_MODEL = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
            sentence_embeddings = _SBERT_MODEL.encode(sentences, convert_to_numpy=True)
            doc_embedding = _SBERT_MODEL.encode([text], convert_to_numpy=True)[0]
        except Exception:
            sentence_embeddings, doc_embedding = None, None

    if sentence_embeddings is None or doc_embedding is None:
        sentence_embeddings, doc_embedding = _get_fallback_embeddings(sentences, text)

    # Cosine Similarity between sentences and document
    doc_norm = np.linalg.norm(doc_embedding)
    if doc_norm == 0:
        return " ".join(sentences[:n])
    
    sent_norms = np.linalg.norm(sentence_embeddings, axis=1)
    sent_norms[sent_norms == 0] = 1.0
    
    doc_sims = np.dot(sentence_embeddings, doc_embedding) / (sent_norms * doc_norm)

    # Maximal Marginal Relevance (MMR) Selection
    selected_indices = []
    unselected_indices = list(range(len(sentences)))

    # First sentence: highest doc similarity
    best_first = int(np.argmax(doc_sims))
    selected_indices.append(best_first)
    unselected_indices.remove(best_first)

    while len(selected_indices) < n and unselected_indices:
        mmr_scores = []
        for idx in unselected_indices:
            # Relevance to document
            rel = doc_sims[idx]
            
            # Max similarity to already selected sentences
            selected_vecs = sentence_embeddings[selected_indices]
            target_vec = sentence_embeddings[idx]
            
            sel_norms = np.linalg.norm(selected_vecs, axis=1)
            sel_norms[sel_norms == 0] = 1.0
            
            sims_to_selected = np.dot(selected_vecs, target_vec) / (sel_norms * sent_norms[idx])
            max_sim_selected = float(np.max(sims_to_selected)) if len(sims_to_selected) > 0 else 0.0
            
            # MMR formula
            mmr = (1.0 - diversity) * rel - diversity * max_sim_selected
            mmr_scores.append((mmr, idx))
            
        mmr_scores.sort(key=lambda x: x[0], reverse=True)
        best_idx = mmr_scores[0][1]
        selected_indices.append(best_idx)
        unselected_indices.remove(best_idx)

    top_indices = sorted(selected_indices)
    return " ".join(sentences[i] for i in top_indices)


def summarize(text: str, n: int = 3) -> str:
    return keybert_summarize(text, n=n)


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

    print(f"Running KeyBERT Summarizer (n={n}) ...")
    for rec in records:
        content = rec.get("content", "")
        title   = rec.get("title", "")
        summary = keybert_summarize(content, n=n)
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
    print(f"KeyBERT Summarizer (n={n}) — avg over {len(records):,} articles")
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

    parser = argparse.ArgumentParser(description="KeyBERT extractive summarizer")
    parser.add_argument("--test", default=str(_DATA / "test.jsonl"))
    parser.add_argument("--out",  default=str(_DATA / "extractive_keybert_preds.jsonl"))
    parser.add_argument("--n",    type=int, default=3)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    main(Path(args.test), Path(args.out), args.n, args.limit)
