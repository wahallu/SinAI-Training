"""
SinhalaJournal-LLM | Step 6: Multi-Length Silver Summary Generator (9-Router Gateway API)
-----------------------------------------------------------------------------------------
Generates THREE length-conditioned Sinhala summaries (short / medium / long)
per article in a single query via an OpenAI-compatible Gateway API (e.g. http://localhost:20128/v1).

Features:
- High-concurrency worker queue tuned for 9 parallel routers.
- Handles standard JSON and Server-Sent Event (SSE) data stream responses.
- Automatic quality validation (word caps, length ratios, hallucination & meta-commentary checks).
- Resume support (skips already processed article URLs).

Usage:
    python abstractive/6_multilength_summary_generator.py
    python abstractive/6_multilength_summary_generator.py \
        --input "D:\\SinhalaLLM\\cleaned_datasets\\all_articles_merged.json" \
        --output "D:\\SinhalaLLM\\cleaned_datasets\\6_multilength_summaries.jsonl" \
        --model "ollama/gpt-oss:120b" \
        --concurrency 9
"""

import os
import re
import json
import time
import argparse
import asyncio
import sys
import unicodedata
import urllib.request
import urllib.error
from pathlib import Path
from datetime import datetime
from threading import Lock

# Enable UTF-8 encoding for Windows console output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# ── Configuration Constants ───────────────────────────────────────────────────
DEFAULT_API_BASE     = "http://62.171.163.6:20128/v1"
DEFAULT_API_KEY      = "sk-6df33c3490e886a3-rqm4ns-31529652"
DEFAULT_MODEL        = "summarizer"
MAX_ARTICLE_RETRIES  = 5
RESPONSE_TIMEOUT_SEC = 120

# ── Length buckets ────────────────────────────────────────────────────────────
LENGTH_BUCKETS = {
    "short":  {"target_pct": 10, "min_ratio": 0.04, "max_ratio": 0.18, "max_words": 45},
    "medium": {"target_pct": 20, "min_ratio": 0.12, "max_ratio": 0.32, "max_words": 80},
    "long":   {"target_pct": 35, "min_ratio": 0.22, "max_ratio": 0.55, "max_words": 130},
}
MIN_SUMMARY_WORDS = 8
MIN_ARTICLE_WORDS = 40

SYSTEM_PROMPT = """You are an expert Sinhala news summarization assistant.
Respond only with valid JSON. All summary text must be in Sinhala."""

USER_INSTRUCTION = """ඔබ සිංහල පුවත් සාරාංශකරණ විශේෂඥයෙකි.

පහත ලිපිය සඳහා සාරාංශ තුනක් ලියන්න:
1. "short"  — ඉතා කෙටි සාරාංශයක්: ලිපියේ දිගෙන් 10%ක් පමණ (වචන {short_target}ක් පමණ).
2. "medium" — මධ්‍යම සාරාංශයක්: ලිපියේ දිගෙන් 20%ක් පමණ (වචන {medium_target}ක් පමණ).
3. "long"   — දීර්ඝ සාරාංශයක්: ලිපියේ දිගෙන් 35%ක් පමණ (වචන {long_target}ක් පමණ).

නීති (සියලු සාරාංශ සඳහා):
- ලිපියේ ඇති තොරතුරු පමණක් භාවිතා කරන්න. නව තොරතුරු එකතු නොකරන්න.
- පුද්ගල නාම, ස්ථාන, සංඛ්‍යා වෙනස් නොකරන්න.
- අදහස්, විශ්ලේෂණ හෝ අනුමාන එකතු නොකරන්න.
- සෑම සාරාංශයක්ම සම්පූර්ණ වාක්‍යයකින් අවසන් කරන්න.
- Markdown, ශීර්ෂ, ලැයිස්තු සලකුණු භාවිතා නොකරන්න — සරල ඡේද පමණි.

ප්‍රතිචාරය මෙම JSON ආකෘතියෙන් පමණක් ලබා දෙන්න:
{{"short": "...", "medium": "...", "long": "..."}}

Article:
{content}
"""

META_COMMENTARY_MARKERS = (
    "ඔබ ලබා දුන්",       # "according to what you provided..."
    "ඉහත ලිපිය",          # "the above article..."
    "පහත දැක්වේ",         # "shown below..."
    "සාරාංශය:",           # "Summary:" label leaking into the text
)

# ── Logging ───────────────────────────────────────────────────────────────────
_log_lock = Lock()

def log(msg: str, tag: str = ""):
    ts = datetime.now().strftime("%H:%M:%S")
    prefix = f"[{tag}] " if tag else ""
    with _log_lock:
        try:
            print(f"{ts}  {prefix}{msg}", flush=True)
        except UnicodeEncodeError:
            clean_msg = msg.encode("ascii", "replace").decode("ascii")
            print(f"{ts}  {prefix}{clean_msg}", flush=True)

# ── Quality Validation ────────────────────────────────────────────────────────
def digits_in(text: str) -> set:
    return set(re.findall(r"\d+", text))

def validate_summary(summary: str, article: str, bucket: str) -> str | None:
    """Returns a rejection reason, or None if the summary passes validation."""
    cfg = LENGTH_BUCKETS[bucket]

    if not summary or not summary.strip():
        return "empty"
    
    # Strip leading list markers / key prefixes if present
    summary = re.sub(r'^(?:[0-9]+[\.\)]|short:|medium:|long:)\s*', '', summary.strip(), flags=re.IGNORECASE).strip()

    if "\ufffd" in summary:
        return "replacement_char"
    if summary.startswith("#") or "\n#" in summary:
        return "markdown_header"
    if unicodedata.combining(summary[0]):
        return "leading_combining_mark"
    for marker in META_COMMENTARY_MARKERS:
        if marker in summary:
            return f"meta_commentary:{marker}"

    article_words = len(article.split())
    summary_words = len(summary.strip().split())

    if summary_words < MIN_SUMMARY_WORDS:
        return "too_short"

    # Scale max_words dynamically for longer articles
    max_words_allowed = max(cfg["max_words"], int(article_words * cfg["max_ratio"] + 15))
    if summary_words > max_words_allowed:
        return f"over_word_cap:{summary_words}>{max_words_allowed}"

    # Dynamic ratio thresholds: relax max_ratio for short articles (< 100 words)
    min_ratio = cfg["min_ratio"]
    max_ratio = cfg["max_ratio"]
    if article_words < 100:
        max_ratio = min(0.85, max_ratio + 0.25 * (1.0 - article_words / 100.0))

    ratio = summary_words / article_words if article_words else 0
    if ratio < min_ratio or ratio > max_ratio:
        return f"ratio_out_of_band:{ratio:.2f}"

    # Exclude prompt numbers & indices from hallucinated digits check
    extra_digits = (digits_in(summary) - digits_in(article)) - {"10", "20", "35", "1", "2", "3"}
    if extra_digits:
        return f"hallucinated_numbers:{sorted(extra_digits)[:3]}"

    return None

def parse_teacher_json(raw: str) -> dict | None:
    """Extract the {"short":..,"medium":..,"long":..} JSON object from the response."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    if not isinstance(data, dict):
        return None
    if not all(k in data and isinstance(data[k], str) for k in LENGTH_BUCKETS):
        return None
    return data

# ── Gateway SSE / JSON Response Parser ────────────────────────────────────────
def parse_router_response(raw_text: str) -> str:
    raw_text = raw_text.strip()
    if not raw_text:
        raise ValueError("Empty response body received from gateway API")

    # Standard OpenAI JSON object
    if raw_text.startswith('{'):
        data = json.loads(raw_text)
        if "error" in data:
            err_msg = data["error"].get("message") if isinstance(data["error"], dict) else data["error"]
            raise ValueError(f"API Error: {err_msg}")
        return data['choices'][0]['message']['content']

    # Server-Sent Events (SSE) data stream format: data: {"id":...}
    content_parts = []
    for line in raw_text.splitlines():
        line = line.strip()
        if line.startswith('data:'):
            json_str = line[5:].strip()
            if json_str == '[DONE]':
                continue
            try:
                chunk = json.loads(json_str)
                if "error" in chunk:
                    err_msg = chunk["error"].get("message") if isinstance(chunk["error"], dict) else chunk["error"]
                    raise ValueError(f"Stream Error: {err_msg}")
                choices = chunk.get('choices', [])
                if choices:
                    delta = choices[0].get('delta', {})
                    text = delta.get('content') or choices[0].get('message', {}).get('content') or ''
                    content_parts.append(text)
            except json.JSONDecodeError:
                pass

    if content_parts:
        return ''.join(content_parts)

    raise ValueError(f"Unparseable gateway response format: {raw_text[:200]}")

def call_gateway_api(prompt: str, api_base: str, model: str, api_key: str) -> str:
    url = f"{api_base.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=RESPONSE_TIMEOUT_SEC) as resp:
            raw_text = resp.read().decode("utf-8", errors="replace")
            return parse_router_response(raw_text)
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="ignore")[:250].replace("\n", " ")
        raise RuntimeError(f"HTTP {e.code} ({e.reason}): {err_body}")
    except Exception as e:
        raise RuntimeError(str(e))

def load_processed_urls(output_path: Path) -> set:
    processed = set()
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                except Exception:
                    continue
                if "url" not in data:
                    continue
                has_summary = any(data.get(f"summary_{b}") for b in LENGTH_BUCKETS)
                if has_summary:
                    processed.add(data["url"])
    return processed

# ── Async Worker ──────────────────────────────────────────────────────────────
async def worker(
    worker_id: int,
    queue: asyncio.Queue,
    output_path: Path,
    stats: dict,
    write_lock: asyncio.Lock,
    args: argparse.Namespace,
):
    tag = f"W{worker_id}"
    log(f"Worker {worker_id} started.", tag)
    loop = asyncio.get_running_loop()

    while True:
        try:
            idx, record = queue.get_nowait()
        except asyncio.QueueEmpty:
            break

        content = record.get("content", "").strip()
        article_words = len(content.split())
        label = f"[{idx}]"

        if not content or article_words < MIN_ARTICLE_WORDS:
            queue.task_done()
            continue

        prompt = USER_INSTRUCTION.format(
            content=content,
            short_target=max(10, int(article_words * 0.10)),
            medium_target=max(20, int(article_words * 0.20)),
            long_target=max(35, int(article_words * 0.35)),
        )

        title = record.get("title", "")
        log(f"{label} Summarizing: {title[:50]}… ({article_words} words)", tag)

        success = False
        for attempt in range(1, MAX_ARTICLE_RETRIES + 1):
            try:
                raw_response = await loop.run_in_executor(
                    None, call_gateway_api, prompt, args.api_base, args.model, args.api_key
                )
                parsed = parse_teacher_json(raw_response)

                if parsed is None:
                    log(f"  {label} ✗ Unparseable JSON in reply (attempt {attempt}/{MAX_ARTICLE_RETRIES})", tag)
                else:
                    result = record.copy()
                    result["teacher_model"] = args.model
                    kept_any = False
                    rejections = {}

                    for bucket in LENGTH_BUCKETS:
                        summary = parsed[bucket].strip()
                        reason = validate_summary(summary, content, bucket)
                        if reason is None:
                            result[f"summary_{bucket}"] = summary
                            kept_any = True
                            async with write_lock:
                                stats[f"kept_{bucket}"] += 1
                        else:
                            rejections[bucket] = reason
                            result[f"rejected_{bucket}"] = reason
                            async with write_lock:
                                stats[f"rejected_{bucket}"] += 1

                    if kept_any:
                        async with write_lock:
                            with open(output_path, "a", encoding="utf-8") as f:
                                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                        log(f"  {label} ✓ Kept summaries for {[b for b in LENGTH_BUCKETS if f'summary_{b}' in result]}", tag)
                        success = True
                        break
                    else:
                        log(f"  {label} ✗ All buckets rejected (attempt {attempt}/{MAX_ARTICLE_RETRIES}): {rejections}", tag)

            except Exception as e:
                log(f"  {label} ✗ Error on attempt {attempt}/{MAX_ARTICLE_RETRIES}: {e}", tag)

            if attempt < MAX_ARTICLE_RETRIES:
                await asyncio.sleep(1.5)

        if not success:
            log(f"  {label} ⚠️ Article failed after {MAX_ARTICLE_RETRIES} retries — skipping", tag)

        queue.task_done()

    log(f"Worker {worker_id} finished.", tag)

# ── Main Async Execution ──────────────────────────────────────────────────────
async def async_main(args):
    input_path  = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        log(f"Input file not found: {input_path}", "ERR")
        sys.exit(1)

    log(f"Loading dataset: {input_path}…")
    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
        if content.startswith("[") and content.endswith("]"):
            records = json.loads(content)
        else:
            for line in content.splitlines():
                if line.strip():
                    try:
                        records.append(json.loads(line))
                    except Exception:
                        pass

    processed_urls = load_processed_urls(output_path)
    to_process = [(i + 1, r) for i, r in enumerate(records) if r.get("url") not in processed_urls]

    print("=" * 70)
    print(f" SinhalaJournal-LLM | 9-Router Gateway Summary Generator")
    print("=" * 70)
    print(f" Gateway URL   : {args.api_base}")
    print(f" Model Name    : {args.model}")
    print(f" Total Records : {len(records):,}")
    print(f" Already Done  : {len(processed_urls):,}")
    print(f" Remaining     : {len(to_process):,}")
    print(f" Concurrent Wk : {args.workers}")
    print(f" Output File   : {output_path.resolve()}")
    print("=" * 70)

    if not to_process:
        print("Everything already processed.")
        return

    queue: asyncio.Queue = asyncio.Queue()
    for item in to_process:
        queue.put_nowait(item)

    write_lock = asyncio.Lock()
    stats = {f"{state}_{b}": 0 for b in LENGTH_BUCKETS for state in ("kept", "rejected")}

    n_workers = min(args.workers, len(to_process))
    tasks = [
        asyncio.create_task(worker(i + 1, queue, output_path, stats, write_lock, args))
        for i in range(n_workers)
    ]
    await asyncio.gather(*tasks)

    print("\nPer-bucket yield summary:")
    for bucket in LENGTH_BUCKETS:
        kept, rej = stats[f"kept_{bucket}"], stats[f"rejected_{bucket}"]
        total = kept + rej
        pct = kept / total * 100 if total else 0
        print(f"  {bucket:<7}: kept {kept:,} / {total:,} ({pct:.1f}%)")
    print(f"\nCompleted successfully. Output saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Multi-length Sinhala summary generator (9-Router Gateway API Edition)"
    )
    parser.add_argument("--input",       default="/home/jovyan/summarizer/all_articles_merged.json",
                        help="Path to input dataset file (JSON or JSONL)")
    parser.add_argument("--output",      default="/home/jovyan/summarizer/data/6_multilength_summaries.jsonl",
                        help="Path to output file")
    parser.add_argument("--workers", "--concurrency", type=int, default=9, dest="workers",
                        help="Number of concurrent workers/tasks across routers (default: 9)")
    parser.add_argument("--model",       default=DEFAULT_MODEL,
                        help="Teacher model name (default: ollama/gpt-oss:120b)")
    parser.add_argument("--api-base",    default=DEFAULT_API_BASE,
                        help="OpenAI-compatible Gateway API endpoint URL")
    parser.add_argument("--api-key",     default=DEFAULT_API_KEY,
                        help="API key for the gateway endpoint")
    args = parser.parse_args()
    asyncio.run(async_main(args))

if __name__ == "__main__":
    main()
