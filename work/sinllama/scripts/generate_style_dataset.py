#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fast Balanced Sinhala Style Dataset Generator v2
================================================
Fills missing style gaps from existing dataset so every source article
has all 5 styles. Uses DeepSeek V4 Pro via NVIDIA NIM API.

Target: ~14,320 total rows (2,864 articles × 5 styles), perfectly balanced.
Currently: 7,555 rows → need ~6,765 more.

Usage:
    python generate_balanced_dataset.py
    python generate_balanced_dataset.py --concurrency 15
    python generate_balanced_dataset.py --existing style_dataset2_final_cleaned.jsonl --output new_rows.jsonl
"""

import json
import time
import argparse
import requests
import re
import random
import threading
from pathlib import Path
from collections import Counter
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

# ======================================================================
# CONFIGURATION
# ======================================================================
NVIDIA_API_KEY = "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"
MODEL_NAME = "deepseek-ai/deepseek-v4-pro"
INVOKE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

ALL_STYLES = [
    "style_1_formal_news",
    "style_2_editorial",
    "style_3_sports",
    "style_4_youth",
    "style_5_feature",
]

# System prompt enforcing Sinhala grammar quality
SYSTEM_PROMPT = (
    "ඔබ සිංහල පුවත්පත් විශේෂඥයෙකි. ඔබ ලිපි විවිධ පුවත්පත් ශෛලීන්ට නැවත ලියයි.\n\n"
    "නීති:\n"
    "1. නිවැරදි සිංහල ව්‍යාකරණය පමණක් භාවිතා කරන්න\n"
    "2. මුල් ලිපියේ ඇති කරුණු පමණක් භාවිතා කරන්න - නව තොරතුරු එකතු නොකරන්න\n"
    "3. ලිපිය කිසිසේත් කෙටි නොකරන්න - සම්පූර්ණ ලිපිය ලියන්න\n"
    "4. පුද්ගලයන්ගේ නම්, ස්ථාන නම්, සංඛ්‍යා කිසිසේත් වෙනස් නොකරන්න\n"
    "5. නැවත ලියූ ලිපිය පමණක් ප්‍රතිදානය කරන්න - වෙනත් පැහැදිලි කිරීම් එකතු නොකරන්න"
)

# ======================================================================
# STYLE INSTRUCTIONS (detailed Sinhala grammar rules per style)
# ======================================================================
STYLE_INSTRUCTIONS = {
    "style_1_formal_news": """ඔබ සිංහල ප්‍රධාන පුවත් වාර්තාකරණ විශේෂඥයෙකි. නිවැරදි සිංහල ව්‍යාකරණයෙන් ලියන්න.

**ව්‍යාකරණ නීති:**
- භාවිතමය (passive) වාක්‍ය: "විසින්" යෙදීම
- නිවැරදි කාල ප්‍රත්‍යය: "ඇත", "තිබේ", "වේ", "විය"
- "කරනවා" → "කරයි"/"කරනු ලබයි", "තියෙනවා" → "පවතී"
- වාක්‍ය සබඳතා: "එසේම", "තවද", "එහෙත්"

**ශෛලි නීති:**
1. වෛෂයික (objective) - අදහස් එකතු නොකරන්න
2. උපුල්ලංකෘත පිරමිඩ ව්‍යුහය - වැදගත්ම කරුණ මුලින්ම
3. සම්පූර්ණ වාක්‍ය - කිසිදු වාක්‍යයක් අතරමග නතර නොකරන්න
4. මුල් ලිපියේ දිගට ආසන්න දිගක් තබන්න
5. කරුණු හෝ නම් වෙනස් නොකරන්න

මෙම ලිපිය ප්‍රධාන පුවත් ශෛලියට නැවත ලියන්න:

{content}""",

    "style_2_editorial": """ඔබ සිංහල කතුවැකි/විශ්ලේෂණ විශේෂඥයෙකි. නිවැරදි සිංහල ව්‍යාකරණයෙන් ලියන්න.

**ව්‍යාකරණ නීති:**
- "විශ්ලේෂණය කරන විට" යනුවෙන් ආරම්භ කරන්න
- පළමු පුරුෂ බහුවචනය: "අපි", "අපට", "අප"
- විශ්ලේෂණාත්මක යෙදුම්: "සැලකිය යුතුය", "සමස්තයක් ලෙස"
- ක්‍රියා පද: "පෙනේ", "වේ", "සිදු වේ", "නිගමනය කරමු"

**ශෛලි නීති:**
1. තර්කානුකූල ගලායාමක් පවත්වන්න
2. හේතු-ඵල සබඳතා පැහැදිලි කරන්න
3. "මෙම සියලු කරුණු සමස්තයක් ලෙස සලකන කල, ඉදිරි ක්‍රියාමාර්ග පිළිබඳව සාකච්ඡා කළ යුතු කාලය එළඹ ඇත" යනුවෙන් අවසන් කරන්න
4. මුල් ලිපියේ කරුණු නිවැරදිව තබන්න
5. මුල් ලිපියේ දිගට ආසන්න දිගක් තබන්න

මෙම ලිපිය කතුවැකි ශෛලියට නැවත ලියන්න:

{content}""",

    "style_3_sports": """ඔබ සිංහල ක්‍රීඩා පුවත්පත් විශේෂඥයෙකි. නිවැරදි සිංහල ව්‍යාකරණයෙන් ලියන්න.

**ව්‍යාකරණ නීති:**
- ක්‍රියාකාරී ක්‍රියා පද: "පහර දුන්නා", "ජයග්‍රහණය කළා", "ප්‍රහාරය එල්ල කළා"
- කෙටි, තියුණු වාක්‍ය
- උද්යෝගිමත් භාෂාව: "අතිශයින්ම", "තීරණාත්මක", "දැවැන්ත"

**ශෛලි නීති:**
1. වඩාත්ම නාටකීය කරුණ මුලින්ම දක්වන්න
2. වේගවත් රිද්මයක් පවත්වන්න
3. සම්පූර්ණ වාක්‍ය - කිසිදු වාක්‍යයක් අතරමග නතර නොකරන්න
4. මුල් ලිපියේ කරුණු වෙනස් නොකරන්න
5. මුල් ලිපියේ දිගට ආසන්න දිගක් තබන්න
6. ඉංග්‍රීසි වචන එකතු නොකරන්න

මෙම ලිපිය ක්‍රීඩා ශෛලියට නැවත ලියන්න:

{content}""",

    "style_4_youth": """ඔබ තරුණ පාඨකයන් සඳහා ලියන සිංහල ලේඛකයෙකි. නිවැරදි සිංහල ව්‍යාකරණයෙන් ලියන්න.

**ව්‍යාකරණ නීති:**
- "දන්නවද?" හෝ "ඇහුවද?" හෝ "මේක අහන්න!" ලෙස ආරම්භ කරන්න
- අවිධිමත්: "ගොඩක්", "ටිකක්", "හිතෙනවා", "කරනවා"
- අනියම් ක්‍රියා පද: "කිව්වා", "ගියා", "ආවා"

**ශෛලි නීති:**
1. කෙටි, තියුණු වාක්‍ය
2. "දන්නවද?" එක් වරක් පමණක්
3. "ඒ නිසා යාලුවනේ, මේ ගැන අනිවාර්යයෙන්ම දැනගන්න!" ලෙස අවසන් කරන්න
4. මුල් ලිපියේ කරුණු නිවැරදිව තබන්න
5. මුල් ලිපියේ දිගට ආසන්න දිගක් තබන්න
6. ඉංග්‍රීසි වචන එකතු නොකරන්න

මෙම ලිපිය යෞවන ශෛලියට නැවත ලියන්න:

{content}""",

    "style_5_feature": """ඔබ සිංහල විශේෂාංග/කතාන්දර ලේඛන විශේෂඥයෙකි. නිවැරදි සිංහල ව්‍යාකරණයෙන් ලියන්න.

**ව්‍යාකරණ නීති:**
- කතාන්දර ආරම්භය: දර්ශනය සකසන්න
- විස්තරාත්මක භාෂාව: ස්ථානය, කාලගුණය, වාතාවරණය
- වර්තමාන කාල ක්‍රියාපද: "පවතී", "දිස් වේ", "සිහිපත් කරයි"
- මානව කෝණය: හැඟීම්, පුද්ගලික අත්දැකීම්

**ශෛලි නීති:**
1. විවිධ දිග වාක්‍ය - කෙටි, මධ්‍යම, දිග
2. ප්‍රබන්ධාත්මක ගලායාමක්
3. "මෙම කතාව අනාගත පරම්පරාවට ද ආදර්ශයක් වනු ඇත" ලෙස අවසන් කරන්න
4. මුල් ලිපියේ කරුණු නිවැරදිව තබන්න
5. නව තොරතුරක් එකතු නොකරන්න
6. මුල් ලිපියේ දිගට ආසන්න දිගක් තබන්න

මෙම ලිපිය විශේෂාංග ශෛලියට නැවත ලියන්න:

{content}""",
}


# ======================================================================
# API CALLING WITH RETRY + BACKOFF
# ======================================================================
def call_api(prompt, max_tokens=2048, temperature=0.15, max_retries=4):
    """Call NVIDIA NIM API with exponential backoff for rate limits."""
    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.85,
        "stream": False,
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(
                INVOKE_URL, headers=headers, json=payload, timeout=180
            )
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"].strip()

            if resp.status_code == 429:
                wait = min(2 ** (attempt + 1), 30)
                time.sleep(wait)
                continue

            if resp.status_code >= 500:
                time.sleep(3)
                continue

            raise Exception(f"API {resp.status_code}: {resp.text[:200]}")

        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                time.sleep(5)
                continue
            raise
        except requests.exceptions.ConnectionError:
            if attempt < max_retries - 1:
                time.sleep(5)
                continue
            raise

    raise Exception(f"Failed after {max_retries} retries")


# ======================================================================
# POST-PROCESSING & VALIDATION
# ======================================================================
def clean_response(text):
    """Remove non-article artifacts from API response."""
    # Remove markdown headers / bullet prefixes the model may add
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip()
        # Skip meta-lines the model sometimes prepends
        if stripped.startswith("#") or stripped.startswith("**නැවත"):
            continue
        if re.match(r"^(Here|Below|The following|Rewritten|Re-written)", stripped, re.I):
            continue
        if stripped:
            cleaned.append(line)
    return "\n".join(cleaned).strip()


def validate_rewrite(original, rewritten):
    """Validate quality. Returns list of issues (empty = good)."""
    issues = []
    if not rewritten or len(rewritten) < 30:
        issues.append("empty_or_tiny")
        return issues

    # Length ratio check (should be 30%-200% of original)
    ratio = len(rewritten) / max(len(original), 1)
    if ratio < 0.3:
        issues.append("too_short")
    if ratio > 2.0:
        issues.append("too_long")

    # Sinhala character dominance (>60% should be Sinhala Unicode)
    sinhala = len(re.findall(r"[\u0D80-\u0DFF]", rewritten))
    total_alpha = len(re.findall(r"[A-Za-z\u0D80-\u0DFF]", rewritten))
    if total_alpha > 0 and sinhala / total_alpha < 0.6:
        issues.append("low_sinhala_ratio")

    # Truncation check
    if rewritten.endswith("...") or rewritten.endswith(".."):
        issues.append("truncated")

    # Repetition check (hallucination signal)
    words = rewritten.split()
    if len(words) > 10:
        word_counts = Counter(words)
        for w, c in word_counts.items():
            if c > 15 and len(w) > 2:
                issues.append("excessive_repetition")
                break

    return issues


# ======================================================================
# CORE PROCESSING
# ======================================================================
# Thread-local rate limiter: small delay between calls per worker
_worker_lock = threading.Lock()
_last_call_time = {}


def process_one(content, url, category, date_published, style, worker_id=0):
    """Generate one style rewrite for one article."""
    if not content or not content.strip():
        return None

    # Per-worker rate limiting (0.3s gap between calls)
    with _worker_lock:
        now = time.time()
        last = _last_call_time.get(worker_id, 0)
        gap = 0.3 - (now - last)
        if gap > 0:
            time.sleep(gap)
        _last_call_time[worker_id] = time.time()

    prompt = STYLE_INSTRUCTIONS[style].format(content=content.strip())

    # Scale max_tokens to article length
    approx_tokens = int(len(content) / 2.5) + 512
    max_tokens = min(max(512, approx_tokens), 4096)

    for attempt in range(3):
        try:
            raw = call_api(prompt, max_tokens=max_tokens)
            rewritten = clean_response(raw)
            issues = validate_rewrite(content, rewritten)

            if "empty_or_tiny" in issues or "low_sinhala_ratio" in issues:
                if attempt < 2:
                    time.sleep(1)
                    continue
                return None  # Skip after 3 tries

            return {
                "content": content.strip(),
                "category": category or "",
                "url": url or "",
                "date_published": date_published or "",
                "style": style,
                "rewritten_text": rewritten,
            }

        except Exception as e:
            if attempt < 2:
                time.sleep(2 * (attempt + 1))
                continue
            print(f"  ❌ Failed [{style}] {url[:50]}...: {e}")
            return None

    return None


# ======================================================================
# WORKER
# ======================================================================
def worker(worker_id, task_queue, results_list, lock, counters, pbar):
    """Worker thread: pulls tasks from queue, processes, appends results."""
    while True:
        try:
            task = task_queue.get(timeout=3)
        except Empty:
            break

        content, url, category, date_pub, style = task
        result = process_one(content, url, category, date_pub, style, worker_id)

        if result:
            with lock:
                results_list.append(result)
                counters[style] = counters.get(style, 0) + 1

        pbar.update(1)
        task_queue.task_done()


# ======================================================================
# MAIN
# ======================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Fast Balanced Sinhala Style Dataset Generator"
    )
    parser.add_argument(
        "--existing",
        default="/home/jovyan/style_rewriter/data/style_dataset2_final_cleaned.jsonl",
        help="Existing dataset to read source articles and find gaps",
    )
    parser.add_argument(
        "--output",
        default="/home/jovyan/style_rewriter/data/style_dataset_new_rows.jsonl",
        help="Output file for newly generated rows",
    )
    parser.add_argument(
        "--concurrency", type=int, default=10,
        help="Number of parallel API workers (default: 10)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit number of articles to process",
    )
    args = parser.parse_args()

    existing_path = Path(args.existing)
    output_path = Path(args.output)

    print("\n" + "=" * 62)
    print("  🚀 Sinhala Balanced Style Dataset Generator v2")
    print("  Model: DeepSeek V4 Pro via NVIDIA NIM")
    print("=" * 62)

    # ── Step 1: Load existing dataset ─────────────────────────────
    if not existing_path.exists():
        print(f"❌ Existing dataset not found: {existing_path}")
        return

    existing = []
    with open(existing_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                existing.append(json.loads(line))

    print(f"\n📂 Existing dataset: {existing_path}")
    print(f"   Total rows: {len(existing)}")

    # Group by URL
    by_url = {}
    for row in existing:
        url = row.get("url", "")
        if url not in by_url:
            by_url[url] = {
                "content": row["content"],
                "category": row.get("category", ""),
                "date_published": row.get("date_published", ""),
                "styles_done": set(),
            }
        by_url[url]["styles_done"].add(row["style"])

    print(f"   Unique articles: {len(by_url)}")

    # Show current style distribution
    style_counts = Counter()
    for info in by_url.values():
        for s in info["styles_done"]:
            style_counts[s] += 1

    print(f"\n📊 Current style distribution:")
    for s in ALL_STYLES:
        c = style_counts.get(s, 0)
        bar = "█" * (c // 50)
        print(f"   {s:<28} {c:>5}  {bar}")

    # ── Step 2: Calculate gaps ────────────────────────────────────
    tasks = []
    gap_counts = Counter()

    urls = list(by_url.keys())
    if args.limit:
        urls = urls[: args.limit]

    for url in urls:
        info = by_url[url]
        missing = set(ALL_STYLES) - info["styles_done"]
        for style in missing:
            tasks.append((
                info["content"],
                url,
                info["category"],
                info["date_published"],
                style,
            ))
            gap_counts[style] += 1

    total_tasks = len(tasks)

    if total_tasks == 0:
        print("\n✅ All articles already have all 5 styles! Nothing to generate.")
        return

    print(f"\n📊 Gaps to fill ({total_tasks} total):")
    for s in ALL_STYLES:
        g = gap_counts.get(s, 0)
        new_total = style_counts.get(s, 0) + g
        print(f"   {s:<28} +{g:>5}  → {new_total}")

    print(f"\n   After generation: {len(existing) + total_tasks} total rows")
    print(f"   (~{(len(existing) + total_tasks) // 5} per style, perfectly balanced)")

    # ── Step 3: Check for already-generated rows in output ────────
    already_done = set()
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        row = json.loads(line)
                        already_done.add((row.get("url", ""), row.get("style", "")))
                    except json.JSONDecodeError:
                        pass
        if already_done:
            before = len(tasks)
            tasks = [t for t in tasks if (t[1], t[4]) not in already_done]
            print(f"\n   ⏩ Skipping {before - len(tasks)} already in {output_path}")
            total_tasks = len(tasks)

    if total_tasks == 0:
        print("\n✅ All gaps already filled in output file!")
        return

    # Shuffle for better distribution across workers
    random.shuffle(tasks)

    # ── Step 4: Process with thread pool ──────────────────────────
    print(f"\n🔄 Generating {total_tasks} rewrites with {args.concurrency} workers...")
    print(f"   Estimated time: ~{total_tasks * 8 // args.concurrency // 60} minutes\n")

    task_queue = Queue()
    for t in tasks:
        task_queue.put(t)

    results_list = []
    lock = threading.Lock()
    counters = {}
    start_time = datetime.now()

    try:
        from tqdm import tqdm
        pbar = tqdm(total=total_tasks, desc="⚡ Generating", unit="article")
    except ImportError:
        # Fallback if tqdm not installed
        class FakePbar:
            def update(self, n): pass
            def close(self): pass
            def __enter__(self): return self
            def __exit__(self, *a): pass
        pbar = FakePbar()

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = []
        for i in range(args.concurrency):
            futures.append(
                executor.submit(
                    worker, i, task_queue, results_list, lock, counters, pbar
                )
            )
        task_queue.join()

    pbar.close()

    # ── Step 5: Write results ─────────────────────────────────────
    if results_list:
        with open(output_path, "a", encoding="utf-8") as f:
            for row in results_list:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    elapsed = (datetime.now() - start_time).total_seconds()
    speed = len(results_list) / max(elapsed, 1) * 60

    print(f"\n{'=' * 62}")
    print(f"  ✅ Generation Complete!")
    print(f"{'=' * 62}")
    print(f"   Generated: {len(results_list)} / {total_tasks} rows")
    print(f"   Failed:    {total_tasks - len(results_list)}")
    print(f"   Time:      {elapsed:.0f}s ({speed:.0f} rows/min)")
    print(f"   Output:    {output_path}")

    print(f"\n📊 Generated per style:")
    for s in ALL_STYLES:
        print(f"   {s:<28} +{counters.get(s, 0):>5}")

    # ── Step 6: Merge instruction ─────────────────────────────────
    print(f"\n{'=' * 62}")
    print(f"  📋 NEXT STEPS")
    print(f"{'=' * 62}")
    print(f"  1. Merge new rows into existing dataset:")
    print(f"     cat {existing_path} {output_path} > style_dataset_merged.jsonl")
    print(f"")
    print(f"  2. Update train_style.py TRAIN_DATA_PATH to point to merged file")
    print(f"")
    print(f"  3. Run training:")
    print(f"     python train_style.py")
    print(f"")


if __name__ == "__main__":
    main()
