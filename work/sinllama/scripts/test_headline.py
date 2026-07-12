from unsloth import FastLanguageModel
import os, json, torch, unicodedata, warnings, math
from collections import Counter
from transformers import AutoTokenizer

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*max_new_tokens.*")
warnings.filterwarnings("ignore", message=".*max_length.*")

SINLLAMA_BASE    = "/home/jovyan/work/sinllama/models/SinLLaMA-merged-base"
HEADLINE_ADAPTER = "/home/jovyan/work/sinllama/models/adapters/headline_sinllama_v17"
VAL_DATA_PATH    = "/home/jovyan/work/sinllama/data/headline_dataset_48k_balanced_val.jsonl"
OUTPUT_RESULTS   = "/home/jovyan/work/sinllama/results/headline_eval_results_v16.json"

MAX_SEQ_LENGTH    = 768
MAX_ARTICLE_CHARS = 2000
MIN_HEADLINE_CHARS = 5


def normalize_sinhala(text):
    return unicodedata.normalize("NFC", text.strip()) if text else ""


def has_sinhala(text):
    return any("\u0d80" <= ch <= "\u0dff" for ch in text)


print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(SINLLAMA_BASE, local_files_only=True)
tokenizer.pad_token    = tokenizer.eos_token
tokenizer.padding_side = "right"

print("Loading pre-merged SinLLaMA base...")
model, _ = FastLanguageModel.from_pretrained(
    model_name=SINLLAMA_BASE, max_seq_length=MAX_SEQ_LENGTH,
    dtype=torch.bfloat16, load_in_4bit=True,
    local_files_only=True, attn_implementation="sdpa",
)

print("Loading headline adapter v16...")
model.load_adapter(HEADLINE_ADAPTER)
FastLanguageModel.for_inference(model)
model.eval()
print("Model ready\n")


def _run_generation(inputs, temperature, min_new_tokens):
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens       = 60,
            min_new_tokens       = min_new_tokens,
            do_sample            = True,
            temperature          = temperature,
            top_p                = 0.9,
            repetition_penalty   = 1.1,
            no_repeat_ngram_size = 2,
            length_penalty       = 1.0,
            eos_token_id         = tokenizer.eos_token_id,
            pad_token_id         = tokenizer.eos_token_id,
        )
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    result = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return result


def clean_output(result):
    result = result.split("\n")[0].strip()
    for marker in ["###","Instruction:","Input:","Response:","Category:","Article:","Rules:"]:
        if marker in result:
            result = result.split(marker)[0].strip()
    result = result.lstrip("-\u2022 ").strip()
    words = result.split()
    if len(words) > 10:
        result = " ".join(words[:10])
    return normalize_sinhala(result)


def generate_headline(article_text, category):
    article_text = article_text.strip()[:MAX_ARTICLE_CHARS]

    prompt = (
        "### Instruction:\n"
        "Generate a concise Sinhala news headline for the article below.\n\n"
        "Rules:\n"
        "- Use formal Sinhala journalism style matching the article category\n"
        "- Between 4 and 7 words -- never fewer than 4\n"
        "- Capture the key person, event, number, or outcome\n"
        "- Output ONLY the headline, nothing else\n\n"
        f"### Input:\nCategory: {category}\nArticle: {article_text}\n\n"
        "### Response:\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    # min_new_tokens=5 -- confirmed in v15 testing that 5 gives the best
    # balance vs 8 (fewer borderline-long 8-10 word outputs)
    result = clean_output(_run_generation(inputs, temperature=0.3, min_new_tokens=5))
    was_retried = False

    if len(result.split()) < 4:
        result = clean_output(_run_generation(inputs, temperature=0.6, min_new_tokens=8))
        was_retried = True

    if not has_sinhala(result) or len(result) < MIN_HEADLINE_CHARS:
        return "", was_retried

    return result, was_retried


# ── Metrics ───────────────────────────────────────────────────────────────────

def rouge_1(ref, hyp):
    r, h = set(ref.split()), set(hyp.split())
    if not r or not h: return 0.0
    ov = r & h
    p = len(ov)/len(h); rec = len(ov)/len(r)
    return (2*p*rec)/(p+rec) if (p+rec) > 0 else 0.0

def rouge_2(ref, hyp):
    def bg(w): return Counter(zip(w[:-1], w[1:])) if len(w) > 1 else Counter()
    rb, hb = bg(ref.split()), bg(hyp.split())
    if not rb or not hb: return 0.0
    ov  = sum(min(rb[ng], hb.get(ng, 0)) for ng in rb)
    p   = ov/sum(hb.values()); rec = ov/sum(rb.values())
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

def exact_match(ref, hyp):
    return 1.0 if normalize_sinhala(ref) == normalize_sinhala(hyp) else 0.0

def len_ratio(ref, hyp):
    r, h = len(ref.split()), len(hyp.split())
    return h/r if r > 0 else 0.0


# ── Load validation data ──────────────────────────────────────────────────────

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
        val_samples.append({
            "article":  normalize_sinhala(art),
            "category": cat,
            "expected": normalize_sinhala(item["output"]),
        })
print(f"   Loaded {len(val_samples)} samples\n")


# ── Evaluation loop ───────────────────────────────────────────────────────────

all_scores = []
cat_scores = {}

print("="*80)
print("  HEADLINE EVALUATION v16 -- 48K balanced dataset / 12 categories")
print("="*80)

for i, item in enumerate(val_samples):
    gen, retried = generate_headline(item["article"], item["category"])
    exp = item["expected"]

    s = {
        "index":          i + 1,
        "category":       item["category"],
        "expected":       exp,
        "generated":      gen,
        "rouge1":         rouge_1(exp, gen),
        "rouge2":         rouge_2(exp, gen),
        "rougeL":         rouge_l(exp, gen),
        "bleu":           bleu_score(exp, gen),
        "exact_match":    exact_match(exp, gen),
        "length_ratio":   len_ratio(exp, gen),
        "word_count_gen": len(gen.split()) if gen else 0,
        "word_count_ref": len(exp.split()),
        "is_empty":       gen == "",
        "has_sinhala":    has_sinhala(gen),
        "was_retried":    retried,
    }
    all_scores.append(s)

    cat = item["category"]
    if cat not in cat_scores: cat_scores[cat] = []
    cat_scores[cat].append(s)

    if i < 20:
        print(f"\n--- Test {i+1} [{item['category']}] ---")
        print(f"  Expected  ({s['word_count_ref']}w): {exp}")
        print(f"  Generated ({s['word_count_gen']}w): {gen if gen else '[EMPTY]'}")
        print(
            f"  R1:{s['rouge1']:.3f}  R2:{s['rouge2']:.3f}  "
            f"RL:{s['rougeL']:.3f}  BLEU:{s['bleu']:.3f}  "
            f"EM:{'v' if s['exact_match'] else 'x'}"
        )

    if (i+1) % 200 == 0:
        n   = len(all_scores)
        r1  = sum(s["rouge1"] for s in all_scores) / n
        rl  = sum(s["rougeL"] for s in all_scores) / n
        bl  = sum(s["bleu"]   for s in all_scores) / n
        wc  = sum(s["word_count_gen"] for s in all_scores) / n
        emp = sum(1 for s in all_scores if s["is_empty"])
        short = sum(1 for s in all_scores if 0 < s["word_count_gen"] < 4)
        print(f"  ... {i+1}/{len(val_samples)} | R1:{r1:.4f} | RL:{rl:.4f} | BLEU:{bl:.4f} | AvgW:{wc:.1f} | Short:{short} | Empty:{emp} ...")


# ── Summary ───────────────────────────────────────────────────────────────────

n = len(all_scores)
def avg(k): return sum(s[k] for s in all_scores) / n

total_em      = sum(s["exact_match"]  for s in all_scores)
total_empty   = sum(1 for s in all_scores if s["is_empty"])
no_sinhala    = sum(1 for s in all_scores if not s["has_sinhala"] and not s["is_empty"])
too_short     = sum(1 for s in all_scores if 0 < s["word_count_gen"] < 4)
ideal         = sum(1 for s in all_scores if 4 <= s["word_count_gen"] <= 7)
borderline    = sum(1 for s in all_scores if 8 <= s["word_count_gen"] <= 10)
too_long      = sum(1 for s in all_scores if s["word_count_gen"] > 10)
total_retried = sum(1 for s in all_scores if s["was_retried"])

print("\n" + "="*80)
print("  RESULTS SUMMARY (v16 -- 48K balanced dataset / 12 categories)")
print("="*80)
print(f"\n  Samples evaluated  : {n}")

print(f"\n  +------------------------------------------+")
print(f"  |           ROUGE SCORES                   |")
print(f"  +------------------------------------------+")
print(f"    ROUGE-1  : {avg('rouge1'):.4f}")
print(f"    ROUGE-2  : {avg('rouge2'):.4f}")
print(f"    ROUGE-L  : {avg('rougeL'):.4f}")
print(f"    BLEU     : {avg('bleu'):.4f}")

print(f"\n  Exact Match (NFC)  : {int(total_em)}/{n} ({total_em/n*100:.2f}%)")
print(f"  Avg gen words      : {avg('word_count_gen'):.2f}  (ref avg: {avg('word_count_ref'):.2f})")
print(f"  Avg length ratio   : {avg('length_ratio'):.3f}  (1.0 = perfect)")
print(f"  Retried generations: {total_retried}/{n} ({total_retried/n*100:.1f}%)")

print(f"\n  Output quality:")
print(f"  Empty outputs      : {total_empty}/{n} ({total_empty/n*100:.1f}%)")
print(f"  No Sinhala chars   : {no_sinhala}/{n} ({no_sinhala/n*100:.1f}%)")

print(f"\n  Length distribution:")
print(f"  <4 words  (too short)  : {too_short:4d}/{n} ({too_short/n*100:.1f}%)")
print(f"  4-7 words (ideal)      : {ideal:4d}/{n} ({ideal/n*100:.1f}%)")
print(f"  8-10 words              : {borderline:4d}/{n} ({borderline/n*100:.1f}%)")
print(f"  >10 words (too long)   : {too_long:4d}/{n} ({too_long/n*100:.1f}%)")

print(f"\n  Per-category breakdown:")
print(f"  {'Category':20s}  {'R1':>6s}  {'R2':>6s}  {'RL':>6s}  {'BLEU':>6s}  {'AvgW':>5s}  {'N':>5s}")
print(f"  {'-'*20}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*5}  {'-'*5}")
for cat in sorted(cat_scores.keys()):
    items = cat_scores[cat]
    cn    = len(items)
    def ca(k): return sum(s[k] for s in items) / cn
    print(
        f"  {cat:20s}  {ca('rouge1'):6.4f}  {ca('rouge2'):6.4f}  "
        f"{ca('rougeL'):6.4f}  {ca('bleu'):6.4f}  {ca('word_count_gen'):5.1f}  {cn:5d}"
    )

# v15 comparison baseline (from your last balanced run)
v15_scores = {
    "rouge1": 0.1389, "rouge2": 0.0247, "rougeL": 0.1353, "bleu": 0.0020,
}
print(f"\n  Comparison vs v15 (24K, imbalanced categories):")
print(f"  {'Metric':10s}  {'v15':>8s}  {'v16':>8s}  {'Change':>9s}")
print(f"  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*9}")
for label, key in [("ROUGE-1","rouge1"),("ROUGE-2","rouge2"),("ROUGE-L","rougeL"),("BLEU","bleu")]:
    old = v15_scores[key]
    new = avg(key)
    diff = new - old
    sign = "+" if diff >= 0 else ""
    print(f"  {label:10s}  {old:8.4f}  {new:8.4f}  {sign}{diff:.4f}")

print("="*80)

# ── Save ─────────────────────────────────────────────────────────────────────

os.makedirs(os.path.dirname(OUTPUT_RESULTS), exist_ok=True)
with open(OUTPUT_RESULTS, "w", encoding="utf-8") as f:
    json.dump({
        "config": {
            "adapter":             "headline_sinllama_v16",
            "dataset":             "48K train / balanced 12 categories",
            "lora_rank":           64,
            "loss_masking":        True,
            "do_sample":           True,
            "temperature":         0.3,
            "temperature_retry":   0.6,
            "top_p":               0.9,
            "length_penalty":      1.0,
            "min_new_tokens":      5,
            "max_new_tokens":      60,
            "no_repeat_ngram_size": 2,
            "repetition_penalty":  1.1,
            "nfc_normalization":   True,
            "retry_on_short":      True,
        },
        "summary": {
            "rouge1":           avg("rouge1"),
            "rouge2":           avg("rouge2"),
            "rougeL":           avg("rougeL"),
            "bleu":             avg("bleu"),
            "exact_matches":    int(total_em),
            "empty_outputs":    total_empty,
            "avg_words_gen":    avg("word_count_gen"),
            "avg_words_ref":    avg("word_count_ref"),
            "avg_length_ratio": avg("length_ratio"),
            "total_samples":    n,
            "retried":          total_retried,
        },
        "length_distribution": {
            "too_short_under4": too_short,
            "ideal_4_7":        ideal,
            "borderline_8_10":  borderline,
            "too_long_over10":  too_long,
        },
        "per_category": {
            cat: {
                "rouge1":        sum(s["rouge1"] for s in items) / len(items),
                "rouge2":        sum(s["rouge2"] for s in items) / len(items),
                "rougeL":        sum(s["rougeL"] for s in items) / len(items),
                "bleu":          sum(s["bleu"]   for s in items) / len(items),
                "avg_words_gen": sum(s["word_count_gen"] for s in items) / len(items),
                "count":         len(items),
            }
            for cat, items in cat_scores.items()
        },
        "results": all_scores,
    }, f, ensure_ascii=False, indent=2)

print(f"\nResults saved to: {OUTPUT_RESULTS}")