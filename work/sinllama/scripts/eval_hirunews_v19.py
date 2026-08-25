"""External evaluation of v19 on the same 100 Hiru News articles used for v17.

This replaces the v17 external-test numbers in the paper with v19 numbers,
using the same held-out article set so the comparison is apples-to-apples.

v19 differs from v17 in one important way: it is length-conditioned. A single
"generate a headline" call no longer exists -- every generation requests one
of three bands (short 3-5w / medium 6-7w / long 8-10w), and the prompt's own
rules line changes to match. See train_headline_v19.py for the band
definitions; this script imports nothing from it and re-declares the bands
locally so the eval has no import-time dependency on the training script.

Two passes over the same 100 articles:

  PRIMARY PASS  (paper drop-in): for each article, request the band the
  reference headline's own word count falls into. This is the fairest
  single number to place where the v17 external-test row currently sits,
  because the model is being asked to hit the length the real headline
  actually has -- same spirit as v17's unconditioned "just write a headline"
  request, but expressed through v19's new interface.

  BAND-SWEEP PASS (what v19 actually adds over v17): for each article,
  generate at all three bands regardless of the reference's own length. This
  is what "does the output match the requested length" is actually asking,
  and v17 has no equivalent to compare against -- it's new evidence, not a
  replacement number.

Four metrics are reported, defined precisely here because the paper needs
exact definitions alongside the numbers:

  ROUGE-1 / ROUGE-2 / ROUGE-L / BLEU
      Standard n-gram overlap between generated and reference headline,
      computed word-level (see functions below for exact formulas). Reported
      on the PRIMARY pass only, since BAND-SWEEP intentionally asks for
      lengths that don't match the reference and would depress ROUGE for
      reasons unrelated to quality.

  Length ratio
      mean(generated_word_count) / mean(reference_word_count) across the
      PRIMARY pass. 1.0 means the average generated headline is exactly as
      long as the average reference headline. Distinct from length
      accuracy below -- a model can average 1.0 while missing the target
      band on every individual article if its errors cancel out.

  Length accuracy (a.k.a. in-band rate)
      Fraction of BAND-SWEEP generations whose word count falls inside the
      min/max of the band that was explicitly requested for that
      generation. Computed per band and overall. This is the metric that
      answers "when you ask for short/medium/long, do you get it" -- v17
      has no equivalent because v17 never accepted a length request.

  Artifact rate
      Fraction of generations (pooled across both passes) containing any
      of: a leaked prompt-template marker (###, Instruction:, Response:,
      Category:, Article:, Rules:), a scraper media tag left over from the
      Hiru/ITN source (වීඩියෝ, PHOTOS, VIDEO, and bracketed variants of
      those), a truncation ellipsis, an empty string after cleaning, or any
      Latin-alphabet character (Sinhala headlines should contain none --
      their presence almost always means a leaked English tag or template
      fragment rather than a deliberate English word). This is the same
      failure mode the v19 training docstring describes finding in v18's
      long/medium bands via inherited scraper tags; this script checks
      whether cleaning the training data actually reduced it at inference
      time, not just in the training set.

Usage:
    python eval_hirunews_v19.py
"""

from unsloth import FastLanguageModel
import os, json, re, torch, unicodedata, warnings, math
from collections import Counter
from transformers import AutoTokenizer

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*max_new_tokens.*")
warnings.filterwarnings("ignore", message=".*max_length.*")

SINLLAMA_BASE    = "/home/jovyan/work/sinllama/models/SinLLaMA-merged-base"
HEADLINE_ADAPTER = "/home/jovyan/work/sinllama/models/adapters/headline_sinllama_v19"
VAL_DATA_PATH    = "/home/jovyan/work/sinllama/data/hirunews_general_val.jsonl"
OUTPUT_RESULTS   = "/home/jovyan/work/sinllama/results/hirunews_general_eval_results_v19.json"

MAX_SEQ_LENGTH     = 768
MAX_ARTICLE_CHARS  = 2000
MIN_HEADLINE_CHARS = 5
RESPONSE_MARKER    = "### Response:\n"

# Must stay byte-identical to HEADLINE_LENGTHS in train_headline_v19.py and
# to the serving path (prompts.py / tasks/headline.py) -- see that script's
# module docstring for why these three files have to move together.
HEADLINE_LENGTHS = {
    "short":  {"min_words": 3, "max_words": 5},
    "medium": {"min_words": 6, "max_words": 7},
    "long":   {"min_words": 8, "max_words": 10},
}


def normalize_sinhala(text):
    return unicodedata.normalize("NFC", text.strip()) if text else ""


def has_sinhala(text):
    return any("\u0d80" <= ch <= "\u0dff" for ch in text)


def band_for(headline):
    """Which band a headline's own word count falls into, or None if it
    falls outside every band (used to pick the PRIMARY pass's requested
    length -- see module docstring)."""
    words = len(headline.split())
    for name, band in HEADLINE_LENGTHS.items():
        if band["min_words"] <= words <= band["max_words"]:
            return name
    return None


def in_band(word_count, band_name):
    band = HEADLINE_LENGTHS[band_name]
    return band["min_words"] <= word_count <= band["max_words"]


# ── Artifact detection ──────────────────────────────────────────────────────
# See module docstring's "Artifact rate" section for what each check means.
PROMPT_LEAK_MARKERS = ["###", "Instruction:", "Response:", "Category:", "Article:", "Rules:"]
MEDIA_TAG_PATTERN = re.compile(
    r"(\(?\s*(වීඩියෝ|PHOTOS?|VIDEO)\s*\)?)", re.IGNORECASE
)
LATIN_CHAR_PATTERN = re.compile(r"[A-Za-z]")
TRUNCATION_PATTERN = re.compile(r"(\.\.\.|…)\s*$")


def artifact_reasons(raw_headline):
    """Returns a list of artifact reasons found in an UNCLEANED generation
    (i.e. before clean_output() strips anything). Empty list = clean."""
    reasons = []
    if not raw_headline or not raw_headline.strip():
        reasons.append("empty")
        return reasons
    if any(marker in raw_headline for marker in PROMPT_LEAK_MARKERS):
        reasons.append("prompt_leak")
    if MEDIA_TAG_PATTERN.search(raw_headline):
        reasons.append("media_tag")
    if LATIN_CHAR_PATTERN.search(raw_headline):
        reasons.append("latin_chars")
    if TRUNCATION_PATTERN.search(raw_headline.strip()):
        reasons.append("truncation_ellipsis")
    if not has_sinhala(raw_headline):
        reasons.append("no_sinhala")
    return reasons


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

print("Loading headline adapter v19...")
model.load_adapter(HEADLINE_ADAPTER)
FastLanguageModel.for_inference(model)
model.eval()
print("Model ready\n")


def build_prompt(article_text, category, length):
    """Byte-identical to build_prompt() in train_headline_v19.py."""
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
        f"### Input:\nCategory: {category}\nArticle: {article_text}\n\n"
        f"{RESPONSE_MARKER}"
    )


def _run_generation(inputs, temperature, min_new_tokens, max_new_tokens):
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens       = max_new_tokens,
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
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def clean_output(raw_result):
    result = raw_result.split("\n")[0].strip()
    for marker in PROMPT_LEAK_MARKERS:
        if marker in result:
            result = result.split(marker)[0].strip()
    result = MEDIA_TAG_PATTERN.sub("", result).strip()
    result = result.lstrip("-\u2022 ").strip()
    return normalize_sinhala(result)


def generate_headline(article_text, category, length):
    """Generate one headline at the requested band. Returns
    (raw_text, cleaned_text, word_count, was_retried, artifact_reasons)."""
    article_text = article_text.strip()[:MAX_ARTICLE_CHARS]
    band = HEADLINE_LENGTHS[length]
    prompt = build_prompt(article_text, category, length)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    raw = _run_generation(
        inputs, temperature=0.3,
        min_new_tokens=band["min_words"], max_new_tokens=60,
    )
    reasons = artifact_reasons(raw)
    cleaned = clean_output(raw)
    was_retried = False

    if len(cleaned.split()) < band["min_words"]:
        raw = _run_generation(
            inputs, temperature=0.6,
            min_new_tokens=max(band["min_words"], 6), max_new_tokens=60,
        )
        reasons = artifact_reasons(raw)
        cleaned = clean_output(raw)
        was_retried = True

    if not has_sinhala(cleaned) or len(cleaned) < MIN_HEADLINE_CHARS:
        cleaned = ""

    return raw, cleaned, len(cleaned.split()) if cleaned else 0, was_retried, reasons


# ── ROUGE / BLEU (word-level, same formulas as v17's eval script) ──────────

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


# ── Load the same 100 Hiru News articles used for v17 ───────────────────────

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


# ── PRIMARY PASS: band matched to each reference's own length ──────────────

print("="*80)
print("  PRIMARY PASS -- band requested = band the reference headline is in")
print("  (this is the number that replaces the v17 external-test row)")
print("="*80)

primary_results = []
all_artifact_hits = []  # pooled across both passes for the artifact-rate metric

for i, item in enumerate(val_samples):
    exp = item["expected"]
    req_band = band_for(exp) or "medium"  # references outside 3-10w default to medium

    raw, gen, wc, retried, reasons = generate_headline(item["article"], item["category"], req_band)
    all_artifact_hits.append(len(reasons) > 0)

    s = {
        "index":            i + 1,
        "category":         item["category"],
        "requested_band":   req_band,
        "expected":         exp,
        "generated":        gen,
        "rouge1":           rouge_1(exp, gen),
        "rouge2":           rouge_2(exp, gen),
        "rougeL":           rouge_l(exp, gen),
        "bleu":             bleu_score(exp, gen),
        "exact_match":      exact_match(exp, gen),
        "length_ratio":     len_ratio(exp, gen),
        "word_count_gen":   wc,
        "word_count_ref":   len(exp.split()),
        "in_requested_band": in_band(wc, req_band) if gen else False,
        "is_empty":         gen == "",
        "was_retried":      retried,
        "artifact_reasons": reasons,
    }
    primary_results.append(s)

    if i < 20:
        print(f"\n--- Test {i+1} [{item['category']}] band={req_band} ---")
        print(f"  Expected  ({s['word_count_ref']}w): {exp}")
        print(f"  Generated ({s['word_count_gen']}w): {gen if gen else '[EMPTY]'}")
        flag = f"  ARTIFACT: {reasons}" if reasons else ""
        print(f"  R1:{s['rouge1']:.3f}  RL:{s['rougeL']:.3f}  BLEU:{s['bleu']:.3f}  "
              f"in-band:{'y' if s['in_requested_band'] else 'n'}{flag}")

    if (i+1) % 25 == 0:
        n  = len(primary_results)
        r1 = sum(s["rouge1"] for s in primary_results) / n
        print(f"  ... {i+1}/{len(val_samples)} | R1:{r1:.4f} ...")


# ── BAND-SWEEP PASS: every article at all three bands ───────────────────────

print("\n" + "="*80)
print("  BAND-SWEEP PASS -- every article generated at short/medium/long")
print("  (this is what 'does it match the requested length' actually measures)")
print("="*80)

sweep_results = []

for i, item in enumerate(val_samples):
    for band_name in HEADLINE_LENGTHS:
        raw, gen, wc, retried, reasons = generate_headline(item["article"], item["category"], band_name)
        all_artifact_hits.append(len(reasons) > 0)

        sweep_results.append({
            "index":            i + 1,
            "category":         item["category"],
            "requested_band":   band_name,
            "generated":        gen,
            "word_count_gen":   wc,
            "in_requested_band": in_band(wc, band_name) if gen else False,
            "is_empty":         gen == "",
            "was_retried":      retried,
            "artifact_reasons": reasons,
        })

    if (i+1) % 25 == 0:
        print(f"  ... {i+1}/{len(val_samples)} articles swept (x3 bands) ...")


# ── Summary ──────────────────────────────────────────────────────────────────

n = len(primary_results)
def avg(k): return sum(s[k] for s in primary_results) / n

total_em      = sum(s["exact_match"] for s in primary_results)
total_empty_p = sum(1 for s in primary_results if s["is_empty"])
in_band_primary = sum(1 for s in primary_results if s["in_requested_band"])

artifact_rate_pooled = sum(1 for hit in all_artifact_hits if hit) / len(all_artifact_hits)
artifact_rate_primary = sum(1 for s in primary_results if s["artifact_reasons"]) / n
artifact_rate_sweep = sum(1 for s in sweep_results if s["artifact_reasons"]) / len(sweep_results)

# Length accuracy per band, from the sweep pass (this is the headline number)
band_accuracy = {}
for band_name in HEADLINE_LENGTHS:
    band_items = [s for s in sweep_results if s["requested_band"] == band_name]
    hits = sum(1 for s in band_items if s["in_requested_band"])
    band_accuracy[band_name] = {
        "n": len(band_items),
        "in_band": hits,
        "accuracy": hits / len(band_items) if band_items else 0.0,
    }
overall_sweep_accuracy = sum(1 for s in sweep_results if s["in_requested_band"]) / len(sweep_results)

print("\n" + "="*80)
print("  RESULTS SUMMARY (v19 -- Hiru News External Test, 100 articles)")
print("="*80)

print(f"\n  --- PRIMARY PASS (paper drop-in for v17's external-test row) ---")
print(f"  Samples evaluated  : {n}")
print(f"\n  ROUGE-1  : {avg('rouge1'):.4f}")
print(f"  ROUGE-2  : {avg('rouge2'):.4f}")
print(f"  ROUGE-L  : {avg('rougeL'):.4f}")
print(f"  BLEU     : {avg('bleu'):.4f}")
print(f"\n  Exact Match       : {int(total_em)}/{n} ({total_em/n*100:.2f}%)")
print(f"  Avg gen words     : {avg('word_count_gen'):.2f}  (ref avg: {avg('word_count_ref'):.2f})")
print(f"  Length ratio      : {avg('length_ratio'):.3f}  (1.0 = perfect)")
print(f"  In requested band : {in_band_primary}/{n} ({in_band_primary/n*100:.1f}%)")
print(f"  Empty outputs     : {total_empty_p}/{n} ({total_empty_p/n*100:.1f}%)")
print(f"  Artifact rate     : {artifact_rate_primary*100:.1f}%")

print(f"\n  --- BAND-SWEEP PASS (length-accuracy evidence, {len(sweep_results)} generations) ---")
print(f"  {'Band':10s}  {'Requested-range':>16s}  {'N':>5s}  {'In-band':>8s}  {'Accuracy':>9s}")
print(f"  {'-'*10}  {'-'*16}  {'-'*5}  {'-'*8}  {'-'*9}")
for band_name, b in HEADLINE_LENGTHS.items():
    stats = band_accuracy[band_name]
    print(f"  {band_name:10s}  {b['min_words']:>7d}-{b['max_words']:<7d}w  "
          f"{stats['n']:5d}  {stats['in_band']:8d}  {stats['accuracy']*100:8.1f}%")
print(f"  {'OVERALL':10s}  {'':16s}  {len(sweep_results):5d}  "
      f"{sum(1 for s in sweep_results if s['in_requested_band']):8d}  {overall_sweep_accuracy*100:8.1f}%")
print(f"\n  Artifact rate (sweep pass)  : {artifact_rate_sweep*100:.1f}%")
print(f"  Artifact rate (pooled, both passes, n={len(all_artifact_hits)}) : {artifact_rate_pooled*100:.1f}%")

print("\n" + "="*80)
print("  PAPER TABLE -- v19 replacing v17 in the external-test row")
print("="*80)
print(f"  {'Metric':22s}  {'v17 (prior)':>12s}  {'v19 (this run)':>15s}")
print(f"  {'-'*22}  {'-'*12}  {'-'*15}")
print(f"  {'ROUGE-1':22s}  {'0.1567':>12s}  {avg('rouge1'):15.4f}")
print(f"  {'ROUGE-2':22s}  {'0.0281':>12s}  {avg('rouge2'):15.4f}")
print(f"  {'ROUGE-L':22s}  {'0.1521':>12s}  {avg('rougeL'):15.4f}")
print(f"  {'BLEU':22s}  {'0.0000':>12s}  {avg('bleu'):15.4f}")
print(f"  {'Length ratio':22s}  {'0.814':>12s}  {avg('length_ratio'):15.3f}")
print(f"  {'Length accuracy':22s}  {'n/a':>12s}  {overall_sweep_accuracy*100:14.1f}%")
print(f"  {'Artifact rate':22s}  {'n/a':>12s}  {artifact_rate_pooled*100:14.1f}%")
print("="*80)

# ── Save ─────────────────────────────────────────────────────────────────────

os.makedirs(os.path.dirname(OUTPUT_RESULTS), exist_ok=True)
with open(OUTPUT_RESULTS, "w", encoding="utf-8") as f:
    json.dump({
        "config": {
            "adapter": "headline_sinllama_v19",
            "dataset": "Hiru News external test, 100 articles (same set as v17)",
            "bands":   HEADLINE_LENGTHS,
            "primary_pass_note": "requested band = band the reference headline's own word count falls into",
            "sweep_pass_note":   "every article generated at all three bands regardless of reference length",
            "do_sample": True, "temperature": 0.3, "temperature_retry": 0.6,
            "top_p": 0.9, "repetition_penalty": 1.1, "no_repeat_ngram_size": 2,
            "length_penalty": 1.0, "max_new_tokens": 60,
        },
        "metric_definitions": {
            "rouge": "Word-level ROUGE-1/2/L F1 between generated and reference headline, PRIMARY pass only.",
            "bleu": "4-gram BLEU with brevity penalty, PRIMARY pass only.",
            "length_ratio": "mean(generated words) / mean(reference words), PRIMARY pass.",
            "length_accuracy": "Fraction of BAND-SWEEP generations whose word count falls inside the requested band's min/max.",
            "artifact_rate": "Fraction of generations containing a leaked prompt marker, scraper media tag (වීඩියෝ/PHOTOS/VIDEO), truncation ellipsis, empty output, or any Latin character.",
        },
        "summary_primary": {
            "rouge1": avg("rouge1"), "rouge2": avg("rouge2"), "rougeL": avg("rougeL"),
            "bleu": avg("bleu"), "exact_matches": int(total_em),
            "avg_words_gen": avg("word_count_gen"), "avg_words_ref": avg("word_count_ref"),
            "length_ratio": avg("length_ratio"),
            "in_requested_band_rate": in_band_primary / n,
            "empty_outputs": total_empty_p,
            "artifact_rate": artifact_rate_primary,
            "total_samples": n,
        },
        "summary_band_sweep": {
            "per_band": band_accuracy,
            "overall_length_accuracy": overall_sweep_accuracy,
            "artifact_rate": artifact_rate_sweep,
            "total_generations": len(sweep_results),
        },
        "artifact_rate_pooled": artifact_rate_pooled,
        "v17_comparison": {
            "rouge1": 0.1567, "rouge2": 0.0281, "rougeL": 0.1521,
            "bleu": 0.0000, "length_ratio": 0.814,
        },
        "primary_results": primary_results,
        "band_sweep_results": sweep_results,
    }, f, ensure_ascii=False, indent=2)

print(f"\nResults saved to: {OUTPUT_RESULTS}")
