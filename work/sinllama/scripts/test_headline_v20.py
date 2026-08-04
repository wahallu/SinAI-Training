"""Per-band evaluation for the input-and-output-cleaned headline adapter
(v20).

Identical methodology to test_headline_v18.py / test_headline_v19.py (each
val article generated once per band, in-band/artifact rate over all three
bands, ROUGE/BLEU on the own-band subset only) -- pointed at the v20 adapter
and evaluated against the v20 clean validation set
(headline_dataset_48k_balanced_val_clean_v20.jsonl).

Numbers to compare against, both measured at N=300 articles:
              in-band (short/medium/long/all)   artifact (short/medium/long/all)
  v18:        88.7 / 76.0 / 78.0 / 80.9         0.3 / 11.0 / 22.3 / 11.2
  v19:        89.7 / 74.3 / 75.0 / 79.7          0.0 /  0.3 /  3.0 /  1.1
v20's artifact numbers should drop further still (v19 wasn't zero, and the
whole point of v20's article-input cleaning is closing that remaining gap).
In-band rate should stay roughly where v19 left it -- if it drops noticeably,
the wider word list or the article-side cleaning likely stripped real
content, not just scraper tags.
"""

from unsloth import FastLanguageModel
import json, math, os, random, re, unicodedata, warnings
from collections import Counter
from transformers import AutoTokenizer

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*max_new_tokens.*")
warnings.filterwarnings("ignore", message=".*max_length.*")

SINLLAMA_BASE    = "/home/jovyan/work/sinllama/models/SinLLaMA-merged-base"
HEADLINE_ADAPTER = "/home/jovyan/work/sinllama/models/adapters/headline_sinllama_v20"
VAL_DATA_PATH    = "/home/jovyan/work/sinllama/data/headline_dataset_48k_balanced_val_clean_v20.jsonl"
OUTPUT_RESULTS   = "/home/jovyan/work/sinllama/results/headline_eval_results_v20.json"

MAX_SEQ_LENGTH     = 768
MAX_ARTICLE_CHARS  = 2000
MIN_HEADLINE_CHARS = 5
SAMPLE_SIZE        = 300  # articles; None = full val set (3x generations each)
SEED               = 42

# Must stay byte-identical to HEADLINE_LENGTHS in train_headline_v18.py /
# tasks/headline.py / backend-api's prompts.py.
HEADLINE_LENGTHS = {
    "short":  {"min_words": 3, "max_words": 5},
    "medium": {"min_words": 6, "max_words": 7},
    "long":   {"min_words": 8, "max_words": 10},
}

# Per-band generation budget, mirroring tasks/headline.py's
# TOKENS_PER_WORD_CEILING (4.0) / TOKENS_PER_WORD_FLOOR (1.7).
TOKENS_PER_WORD_CEILING = 4.0
TOKENS_PER_WORD_FLOOR   = 1.7
MIN_NEW_TOKENS_BASELINE = 5

# Trailing scraper artifacts (Hiru/ITN media tags) left in reference headlines.
ARTIFACT = re.compile(
    r"(වීඩියෝ|ජායාරූප|VIDEO|PHOTOS?|Video|PICTURES?|Interview)",
    re.IGNORECASE,
)


def normalize_sinhala(text):
    return unicodedata.normalize("NFC", text.strip()) if text else ""


def has_sinhala(text):
    return any("඀" <= ch <= "෿" for ch in text)


def band_for(headline):
    words = len(headline.split())
    for name, band in HEADLINE_LENGTHS.items():
        if band["min_words"] <= words <= band["max_words"]:
            return name
    return None


print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(SINLLAMA_BASE, local_files_only=True)
tokenizer.pad_token    = tokenizer.eos_token
tokenizer.padding_side = "right"

print("Loading pre-merged SinLLaMA base...")
model, _ = FastLanguageModel.from_pretrained(
    model_name=SINLLAMA_BASE, max_seq_length=MAX_SEQ_LENGTH,
    dtype="bfloat16", load_in_4bit=True,
    local_files_only=True, attn_implementation="sdpa",
)

print("Loading headline adapter v20...")
model.load_adapter(HEADLINE_ADAPTER)

import torch
FastLanguageModel.for_inference(model)
model.eval()
print("Model ready\n")


def build_prompt(article, length):
    """Byte-identical to train_headline_v18.py's build_prompt / tasks/headline.py."""
    article = normalize_sinhala(article)[:MAX_ARTICLE_CHARS]
    band = HEADLINE_LENGTHS[length]
    return (
        "### Instruction:\n"
        "Generate a concise Sinhala news headline for the article below.\n\n"
        "Rules:\n"
        "- Use formal Sinhala journalism style matching the article category\n"
        f"- Between {band['min_words']} and {band['max_words']} words"
        f" -- never fewer than {band['min_words']}\n"
        "- Capture the key person, event, number, or outcome\n"
        "- Output ONLY the headline, nothing else\n\n"
        f"### Input:\n{article}\n\n"
        "### Response:\n"
    )


def clean_output(result):
    result = result.split("\n")[0].strip()
    for marker in ["###", "Instruction:", "Input:", "Response:", "Category:", "Article:", "Rules:"]:
        if marker in result:
            result = result.split(marker)[0].strip()
    return normalize_sinhala(result.lstrip("-• ").strip())


def generate_headline(article_text, length):
    band = HEADLINE_LENGTHS[length]
    max_new_tokens = int(band["max_words"] * TOKENS_PER_WORD_CEILING) + 12
    min_new_tokens = max(MIN_NEW_TOKENS_BASELINE, int(band["min_words"] * TOKENS_PER_WORD_FLOOR))

    prompt = build_prompt(article_text, length)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens       = max_new_tokens,
            min_new_tokens       = min_new_tokens,
            do_sample            = True,
            temperature          = 0.3,
            top_p                = 0.9,
            repetition_penalty   = 1.1,
            no_repeat_ngram_size = 2,
            eos_token_id         = tokenizer.eos_token_id,
            pad_token_id         = tokenizer.eos_token_id,
        )
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    result = clean_output(tokenizer.decode(generated_ids, skip_special_tokens=True).strip())

    if not has_sinhala(result) or len(result) < MIN_HEADLINE_CHARS:
        return ""
    return result


# ── Metrics (same implementations as test_headline.py) ──────────────────────

def rouge_1(ref, hyp):
    r, h = set(ref.split()), set(hyp.split())
    if not r or not h: return 0.0
    ov = r & h
    p = len(ov)/len(h); rec = len(ov)/len(r)
    return (2*p*rec)/(p+rec) if (p+rec) > 0 else 0.0

def rouge_l(ref, hyp):
    rw, hw = ref.split(), hyp.split()
    if not rw or not hw: return 0.0
    m, n = len(rw), len(hw)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            dp[i][j] = dp[i-1][j-1]+1 if rw[i-1]==hw[j-1] else max(dp[i-1][j], dp[i][j-1])
    lcs = dp[m][n]
    p = lcs/n; rec = lcs/m
    return (2*p*rec)/(p+rec) if (p+rec) > 0 else 0.0

def bleu_score(ref, hyp, max_n=4):
    ref_t, hyp_t = ref.split(), hyp.split()
    if not hyp_t or not ref_t: return 0.0
    bp = 1.0 if len(hyp_t) >= len(ref_t) else math.exp(1 - len(ref_t)/len(hyp_t))
    log_avg = 0.0
    for n in range(1, max_n+1):
        def ngrams(t, n): return Counter(tuple(t[i:i+n]) for i in range(len(t)-n+1))
        rng = ngrams(ref_t, n); hng = ngrams(hyp_t, n)
        if not hng: return 0.0
        ov   = sum(min(rng[ng], hng.get(ng, 0)) for ng in hng)
        tot  = sum(hng.values())
        prec = ov/tot if tot > 0 else 0.0
        if prec == 0: return 0.0
        log_avg += math.log(prec) / max_n
    return bp * math.exp(log_avg)


# ── Load validation data ─────────────────────────────────────────────────────

print("Loading validation data...")
val_samples = []
with open(VAL_DATA_PATH, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line.strip())
        inp  = item["input"]; cat = "General"; art = ""
        if "Category:" in inp:
            parts = inp.split("\n", 1)
            cat   = parts[0].replace("Category:", "").strip()
            if len(parts) > 1 and "Article:" in parts[1]:
                art = parts[1].split("Article:", 1)[1].strip()
        headline = normalize_sinhala(item["output"])
        val_samples.append({
            "article":    normalize_sinhala(art),
            "category":   cat,
            "expected":   headline,
            "ref_band":   band_for(headline),
        })

random.seed(SEED)
random.shuffle(val_samples)
if SAMPLE_SIZE:
    val_samples = val_samples[:SAMPLE_SIZE]
print(f"   Evaluating {len(val_samples)} articles x 3 bands "
      f"({len(val_samples)*3} generations)\n")


# ── Evaluation loop ───────────────────────────────────────────────────────────

band_stats = {b: {"n": 0, "in_band": 0, "artifact": 0, "empty": 0} for b in HEADLINE_LENGTHS}
own_band_scores = []  # ROUGE/BLEU only where requested band == reference's own band

print("="*80)
print("  HEADLINE EVALUATION v20 -- per-band, artifact-cleaned")
print("="*80)

for i, item in enumerate(val_samples):
    for band_name in HEADLINE_LENGTHS:
        gen = generate_headline(item["article"], band_name)
        band = HEADLINE_LENGTHS[band_name]
        words = len(gen.split()) if gen else 0
        ok = bool(gen) and band["min_words"] <= words <= band["max_words"]
        junk = bool(gen) and bool(ARTIFACT.search(gen))

        s = band_stats[band_name]
        s["n"] += 1
        s["in_band"] += ok
        s["artifact"] += junk
        s["empty"] += (gen == "")

        if band_name == item["ref_band"] and gen:
            own_band_scores.append({
                "band":    band_name,
                "rouge1":  rouge_1(item["expected"], gen),
                "rougeL":  rouge_l(item["expected"], gen),
                "bleu":    bleu_score(item["expected"], gen),
            })

    if i < 15:
        print(f"\n--- Article {i+1} [{item['category']}] ref_band={item['ref_band']} ---")
        print(f"  Reference ({len(item['expected'].split())}w): {item['expected']}")

    if (i+1) % 50 == 0:
        print(f"  ... {i+1}/{len(val_samples)} articles done")
        for b in HEADLINE_LENGTHS:
            s = band_stats[b]
            print(f"      {b:6s}  in-band {s['in_band']}/{s['n']}  "
                  f"artifact {s['artifact']}/{s['n']}  empty {s['empty']}/{s['n']}")


# ── Summary ───────────────────────────────────────────────────────────────────

print("\n" + "="*80)
print("  RESULTS SUMMARY (v20 -- per-band, N={} articles)".format(len(val_samples)))
print("="*80)

print(f"\n  {'band':8s} {'target':10s} {'in-band':>10s} {'artifact':>10s} {'empty':>8s}")
for b in HEADLINE_LENGTHS:
    band = HEADLINE_LENGTHS[b]
    s = band_stats[b]
    print(f"  {b:8s} {band['min_words']}-{band['max_words']}w{'':5s} "
          f"{s['in_band']:4d}/{s['n']:<4d} {s['in_band']/s['n']*100:5.1f}%  "
          f"{s['artifact']:4d}/{s['n']:<4d}  {s['empty']:4d}/{s['n']:<4d}")

tot_n  = sum(s["n"] for s in band_stats.values())
tot_ok = sum(s["in_band"] for s in band_stats.values())
tot_junk = sum(s["artifact"] for s in band_stats.values())
print(f"\n  Overall in-band rate : {tot_ok}/{tot_n} ({tot_ok/tot_n*100:.1f}%)")
print(f"  Overall artifact rate: {tot_junk}/{tot_n} ({tot_junk/tot_n*100:.1f}%)")

if own_band_scores:
    n = len(own_band_scores)
    r1 = sum(s["rouge1"] for s in own_band_scores) / n
    rl = sum(s["rougeL"] for s in own_band_scores) / n
    bl = sum(s["bleu"]   for s in own_band_scores) / n
    print(f"\n  Own-band ROUGE/BLEU (requested band == reference's own band, N={n}):")
    print(f"    ROUGE-1 : {r1:.4f}")
    print(f"    ROUGE-L : {rl:.4f}")
    print(f"    BLEU    : {bl:.4f}")

print("="*80)

# ── Save ─────────────────────────────────────────────────────────────────────

os.makedirs(os.path.dirname(OUTPUT_RESULTS), exist_ok=True)
with open(OUTPUT_RESULTS, "w", encoding="utf-8") as f:
    json.dump({
        "config": {
            "adapter":     "headline_sinllama_v20",
            "sample_size": len(val_samples),
            "bands":       HEADLINE_LENGTHS,
        },
        "band_stats": band_stats,
        "own_band_rouge": {
            "rouge1": sum(s["rouge1"] for s in own_band_scores) / len(own_band_scores) if own_band_scores else None,
            "rougeL": sum(s["rougeL"] for s in own_band_scores) / len(own_band_scores) if own_band_scores else None,
            "bleu":   sum(s["bleu"]   for s in own_band_scores) / len(own_band_scores) if own_band_scores else None,
            "n":      len(own_band_scores),
        },
    }, f, ensure_ascii=False, indent=2)

print(f"\nResults saved to: {OUTPUT_RESULTS}")
