"""
SinhalaJournal-LLM | BERTScore smoke test
-----------------------------------------
Verifies that multilingual BERTScore (xlm-roberta-large) loads and scores
Sinhala text cleanly on this box, before it is used by
8_evaluate_summarizer.py (offline eval) or work/serve_sinai.py (/compare).

Checks:
  * bert_score is importable and xlm-roberta-large is downloadable/cached.
  * bert_score.score() returns finite P/R/F1 tensors of the right shape.
  * The scorer discriminates: an identical pair must score higher than an
    unrelated pair (a scorer that returns ~the same number for both means
    the model isn't really reading Sinhala).
  * The BERTScorer object path (used by both call sites) works too.

Usage:
    python abstractive/8_bertscore_smoke_test.py
    python abstractive/8_bertscore_smoke_test.py --device cpu
"""

import argparse
import sys

import torch


# Same two knobs both call sites use — keep in sync with
# 8_evaluate_summarizer.BERTSCORE_MODEL and serve_sinai.BERTSCORE_MODEL.
BERTSCORE_MODEL = "xlm-roberta-large"
BERTSCORE_LANG = "si"

# Two Sinhala sentence pairs: one identical, one unrelated.
CANDIDATES = [
    "ශ්‍රී ලංකාවේ අධ්‍යාපන අමාත්‍යාංශය නව පාසල් විෂය මාලාවක් හඳුන්වා දුන්නේය.",
    "අද දින කොළඹ නගරයේ දැඩි වර්ෂාපතනයක් වාර්තා විය.",
]
REFERENCES = [
    "ශ්‍රී ලංකාවේ අධ්‍යාපන අමාත්‍යාංශය නව පාසල් විෂය මාලාවක් හඳුන්වා දුන්නේය.",
    "ක්‍රිකට් තරගාවලියේ අවසන් මහා තරගය ලබන සතියේ පැවැත්වේ.",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    import bert_score
    print(f"bert_score {bert_score.__version__} | torch {torch.__version__} | device={args.device}")

    # ── 1. Functional API (bert_score.score) ──
    print(f"\n🔹 Scoring {len(CANDIDATES)} pairs with {BERTSCORE_MODEL} (functional API)...")
    P, R, F1 = bert_score.score(
        CANDIDATES,
        REFERENCES,
        model_type=BERTSCORE_MODEL,
        lang=BERTSCORE_LANG,
        rescale_with_baseline=False,
        device=args.device,
        batch_size=8,
        verbose=True,
    )

    ok = True
    for t, name in ((P, "precision"), (R, "recall"), (F1, "f1")):
        shape_ok = tuple(t.shape) == (len(CANDIDATES),)
        finite_ok = bool(torch.isfinite(t).all())
        print(f"   {name:<9} {[round(float(v), 4) for v in t]} "
              f"shape={tuple(t.shape)} {'✓' if shape_ok and finite_ok else '✗'}")
        ok = ok and shape_ok and finite_ok

    # Identical pair (index 0) must beat the unrelated pair (index 1).
    identical, unrelated = float(F1[0]), float(F1[1])
    discriminates = identical > unrelated
    print(f"\n   identical-pair F1 = {identical:.4f}  vs  unrelated-pair F1 = {unrelated:.4f} "
          f"{'✓ discriminates' if discriminates else '✗ NOT discriminating'}")
    ok = ok and discriminates

    # ── 2. Object API (BERTScorer) — what both call sites actually use ──
    print("\n🔹 Re-scoring via BERTScorer(...) object...")
    scorer = bert_score.BERTScorer(
        model_type=BERTSCORE_MODEL,
        lang=BERTSCORE_LANG,
        rescale_with_baseline=False,
        device=args.device,
        batch_size=8,
    )
    P2, R2, F2 = scorer.score(CANDIDATES, REFERENCES)
    agree = torch.allclose(F1, F2, atol=1e-4)
    print(f"   f1        {[round(float(v), 4) for v in F2]} "
          f"{'✓ matches functional API' if agree else '✗ differs from functional API'}")
    ok = ok and agree

    print("\n" + ("BERTScore smoke test PASSED ✅" if ok else "BERTScore smoke test FAILED ❌"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
