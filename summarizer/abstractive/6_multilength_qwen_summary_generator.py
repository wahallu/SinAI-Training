"""
SinhalaJournal-LLM | Step 6: Multi-length silver summary generator (Groq)
--------------------------------------------------------------------------
Generates THREE length-conditioned Sinhala summaries (short / medium / long)
per article in a single API call, returned as JSON. Teacher: qwen/qwen3.6-27b
via the Groq API.

Why multi-length (vs the single-length 2_/3_/4_/5_* generators):
  * An audit of 5_qwen_summaries.jsonl (151,438 records) showed the teacher
    naturally writes ~55% compression despite being instructed 10-30%, so
    73.6% of raw output was rejected by the training filter's
    MAX_COMP_RATIO=0.50 — roughly 4 API calls per usable sample. Asking for
    three explicit lengths per call yields up to 3 usable samples per call.
  * Length-labeled data lets the v06 adapter learn *native* length control
    (short/medium/long), which the web app's summary-length drawer needs.

Groq free-tier limits this script enforces client-side:
    30 requests/min  | 1,000 requests/day
    8,000 tokens/min |   200,000 tokens/day
The TOKEN caps are the binding constraint: a multi-length request costs
~1.5-3K tokens, so expect roughly 70-120 articles/day. The script exits
cleanly when the daily budget is spent — re-run it the next day; it resumes
by URL and skips everything already processed.

Usage:
    python abstractive/6_multilength_qwen_summary_generator.py
    python abstractive/6_multilength_qwen_summary_generator.py \
        --input data/train.jsonl \
        --output data/6_multilength_summaries.jsonl
"""

import os
import re
import json
import time
import random
import argparse
import threading
import unicodedata
from pathlib import Path
from collections import deque
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor

import requests
from tqdm import tqdm

# ── Groq API ────────────────────────────────────────────────────
# NOTE: keys are deliberately hardcoded (team decision, 2026-07-21). Do NOT
# publish this repo or share this file while live keys are present.
# One gsk_... key per line, from console.groq.com/keys (each key has its own
# free-tier budget, so throughput scales roughly linearly with key count).
GROQ_API_KEYS = [
    "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",
    "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",
    "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",
    "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",
]
INVOKE_URL    = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "qwen/qwen3.6-27b"

# Groq free-tier limits PER KEY (enforced client-side; server 429s are also
# handled defensively)
RPM_LIMIT           = 30
RPD_LIMIT           = 1_000
TPM_LIMIT           = 8_000
TPD_LIMIT           = 200_000
MAX_RETRIES         = 6
MAX_COMPLETION_TOKENS = 2_048

# Skip articles whose prompt alone would blow the per-minute token window.
MAX_PROMPT_TOKEN_ESTIMATE = 5_000
CHARS_PER_TOKEN_ESTIMATE  = 3.0   # conservative for Sinhala text

# ── Length buckets ──────────────────────────────────────────────
# ratio = summary_words / article_words. Bands are per-bucket (unlike the
# old single 0.05-0.50 band that discarded 73.6% of output). Word caps bound
# the serving-side token budget for each mode.
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


# ── Quality validation ──────────────────────────────────────────
# Patterns that showed up in v02-v05 outputs at serving time and traced back
# to unfiltered teacher output: markdown headers, meta-commentary about the
# prompt, replacement chars, and summaries starting with a bare combining
# mark (the "හිටපු" -> "ිටපු" corruption seen in v03/v04).
META_COMMENTARY_MARKERS = (
    "ඔබ ලබා දුන්",       # "according to what you provided..."
    "ඉහත ලිපිය",          # "the above article..."
    "පහත දැක්වේ",         # "shown below..."
    "සාරාංශය:",           # "Summary:" label leaking into the text
)


def digits_in(text: str) -> set:
    return set(re.findall(r"\d+", text))


def validate_summary(summary: str, article: str, bucket: str) -> str | None:
    """Returns a rejection reason, or None if the summary passes."""
    cfg = LENGTH_BUCKETS[bucket]

    if not summary or not summary.strip():
        return "empty"
    summary = summary.strip()

    if "�" in summary:
        return "replacement_char"
    if summary.startswith("#") or "\n#" in summary:
        return "markdown_header"
    if unicodedata.combining(summary[0]):
        return "leading_combining_mark"
    for marker in META_COMMENTARY_MARKERS:
        if marker in summary:
            return f"meta_commentary:{marker}"

    article_words = len(article.split())
    summary_words = len(summary.split())

    if summary_words < MIN_SUMMARY_WORDS:
        return "too_short"
    if summary_words > cfg["max_words"]:
        return "over_word_cap"

    ratio = summary_words / article_words if article_words else 0
    if ratio < cfg["min_ratio"] or ratio > cfg["max_ratio"]:
        return f"ratio_out_of_band:{ratio:.2f}"

    # Hallucinated-number guard: every digit sequence in the summary must
    # exist somewhere in the article. Catches invented scores/dates/counts —
    # the failure mode observed in the diffusiongemma-trained adapters.
    extra_digits = digits_in(summary) - digits_in(article)
    if extra_digits:
        return f"hallucinated_numbers:{sorted(extra_digits)[:3]}"

    return None


def parse_teacher_json(raw: str) -> dict | None:
    """Extract the {"short":..,"medium":..,"long":..} object from the reply."""
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


# ── Multi-key rate/budget limiter ───────────────────────────────
class _KeyState:
    """Rolling-window + daily counters for one Groq key."""

    __slots__ = ("key", "req_times", "token_events", "day_requests",
                 "day_tokens", "exhausted", "cooldown_until")

    def __init__(self, key: str):
        self.key = key
        self.req_times = deque()      # timestamps of recent requests
        self.token_events = deque()   # (timestamp, tokens) in last 60s
        self.day_requests = 0
        self.day_tokens = 0
        self.exhausted = False        # daily budget spent
        self.cooldown_until = 0.0     # server-side 429 backoff

    def prune(self, now: float):
        while self.req_times and now - self.req_times[0] >= 60:
            self.req_times.popleft()
        while self.token_events and now - self.token_events[0][0] >= 60:
            self.token_events.popleft()


class GroqKeyPool:
    """Client-side enforcement of Groq free-tier limits across N keys.

    Each key gets its own rolling 60s RPM/TPM windows and daily RPD/TPD
    budgets. acquire() RESERVES the estimated token cost up front — without
    the reservation, N workers all acquire before any response comes back,
    every key's window still looks empty, the combined burst blows the
    server-side TPM, and every key 429s at once (exactly what happened on
    the first live run). settle() replaces the estimate with the actual
    usage from the API response once known, or releases it if the request
    never consumed tokens (429/network error).

    acquire() blocks until SOME key has a free slot and returns that key;
    returns None once every key's daily budget is spent (stop for the day).
    """

    def __init__(self, keys: list):
        self.lock = threading.Lock()
        self.states = {k: _KeyState(k) for k in keys}

    def acquire(self, est_tokens: int) -> str | None:
        """Block until a request may be sent. Reserves est_tokens against
        the chosen key. Returns the key, or None when every key's daily
        budget is exhausted."""
        while True:
            with self.lock:
                now = time.time()
                next_wait = None

                for st in self.states.values():
                    if st.exhausted or now < st.cooldown_until:
                        continue
                    if (st.day_requests >= RPD_LIMIT
                            or st.day_tokens + est_tokens > TPD_LIMIT):
                        st.exhausted = True
                        continue

                    st.prune(now)
                    minute_tokens = sum(t for _, t in st.token_events)
                    if (len(st.req_times) < RPM_LIMIT
                            and minute_tokens + est_tokens <= TPM_LIMIT):
                        st.req_times.append(now)
                        st.day_requests += 1
                        # Reserve now; settle() adjusts to actual later.
                        st.token_events.append((now, est_tokens))
                        st.day_tokens += est_tokens
                        return st.key

                    waits = []
                    if st.req_times:
                        waits.append(st.req_times[0] + 60 - now)
                    if st.token_events:
                        waits.append(st.token_events[0][0] + 60 - now)
                    if waits:
                        w = min(waits)
                        next_wait = w if next_wait is None else min(next_wait, w)

                # Cooling-down keys also become available eventually.
                cooldowns = [st.cooldown_until - now for st in self.states.values()
                             if not st.exhausted and now < st.cooldown_until]
                if cooldowns:
                    w = min(cooldowns)
                    next_wait = w if next_wait is None else min(next_wait, w)

                if all(st.exhausted for st in self.states.values()):
                    return None
                wait = next_wait if next_wait is not None else 1.0
            time.sleep(max(0.1, min(wait, 10.0)))

    def settle(self, key: str, est_tokens: int, actual_tokens: int | None):
        """Correct the acquire()-time reservation once the outcome is known.

        actual_tokens=None means the request consumed nothing server-side
        (429, connection error) — release the whole reservation. Correction
        entries share the rolling window; a negative correction ages out
        together with its reservation, so the window sum stays honest.
        """
        delta = (actual_tokens if actual_tokens is not None else 0) - est_tokens
        if delta == 0:
            return
        with self.lock:
            st = self.states[key]
            st.token_events.append((time.time(), delta))
            st.day_tokens = max(0, st.day_tokens + delta)

    def cooldown(self, key: str, seconds: float):
        with self.lock:
            st = self.states[key]
            st.cooldown_until = max(st.cooldown_until, time.time() + seconds)

    @property
    def all_exhausted(self) -> bool:
        with self.lock:
            return all(st.exhausted for st in self.states.values())

    def snapshot(self) -> str:
        with self.lock:
            lines = []
            for st in self.states.values():
                tag = " (exhausted)" if st.exhausted else ""
                lines.append(f"  ...{st.key[-6:]}: requests {st.day_requests}/{RPD_LIMIT}, "
                             f"tokens {st.day_tokens:,}/{TPD_LIMIT:,}{tag}")
            return "\n".join(lines)


def load_processed_urls(output_path: Path) -> set:
    """URLs to skip on resume. Only records that produced at least one
    summary (or are permanently unusable, e.g. too-short articles) count as
    processed — transient failures (429 storms, timeouts, truncations) are
    deliberately NOT skipped, so a re-run retries them."""
    PERMANENT_ERRORS = {"article_too_short", "article_too_long_for_tpm"}
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
                permanent = data.get("error") in PERMANENT_ERRORS
                if has_summary or permanent:
                    processed.add(data["url"])
    return processed


# ── Worker ──────────────────────────────────────────────────────
def worker(pool, input_queue, output_file, lock, pbar, model_name, stats):
    while True:
        try:
            record = input_queue.get(timeout=5)
        except Empty:
            break

        try:
            if pool.all_exhausted:
                pbar.update(1)
                continue

            content = record.get("content", "").strip()
            article_words = len(content.split())

            if not content or article_words < MIN_ARTICLE_WORDS:
                with lock:
                    with open(output_file, "a", encoding="utf-8") as f:
                        err = record.copy()
                        err["error"] = "article_too_short"
                        f.write(json.dumps(err, ensure_ascii=False) + "\n")
                pbar.update(1)
                continue

            prompt = USER_INSTRUCTION.format(
                content=content,
                short_target=max(10, int(article_words * 0.10)),
                medium_target=max(20, int(article_words * 0.20)),
                long_target=max(35, int(article_words * 0.35)),
            )
            est_prompt_tokens = int(len(prompt) / CHARS_PER_TOKEN_ESTIMATE)
            if est_prompt_tokens > MAX_PROMPT_TOKEN_ESTIMATE:
                with lock:
                    with open(output_file, "a", encoding="utf-8") as f:
                        err = record.copy()
                        err["error"] = "article_too_long_for_tpm"
                        f.write(json.dumps(err, ensure_ascii=False) + "\n")
                pbar.update(1)
                continue

            payload = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "max_completion_tokens": MAX_COMPLETION_TOKENS,
                "temperature": 0.0,
                "top_p": 1.0,
                "stream": False,
            }

            success = False
            retries = 0
            last_error = ""

            while not success and retries < MAX_RETRIES:
                # Reservation must track the current max_completion_tokens (which grows
                # on truncated retries).
                est_total = est_prompt_tokens + payload["max_completion_tokens"]
                api_key = pool.acquire(est_total)
                if api_key is None:
                    last_error = "daily_budget_exhausted"
                    break
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                }
                try:
                    response = requests.post(INVOKE_URL, headers=headers, json=payload, timeout=(10, 180))

                    usage = {}
                    if response.headers.get("content-type", "").startswith("application/json"):
                        try:
                            usage = response.json().get("usage", {}) or {}
                        except Exception:
                            usage = {}
                    if response.status_code == 200:
                        pool.settle(api_key, est_total, int(usage.get("total_tokens", est_total)))
                    else:
                        # 429/errors consume no tokens server-side — release
                        # the reservation so it doesn't poison the window.
                        pool.settle(api_key, est_total, int(usage["total_tokens"]) if usage.get("total_tokens") else None)

                    if response.status_code == 200:
                        res_json = response.json()
                        choice = res_json["choices"][0]
                        raw = (choice["message"].get("content") or "").strip()

                        if choice.get("finish_reason") == "length":
                            # Same request at temperature=0 truncates
                            # identically — must raise the budget, not just
                            # retry (this exact loop caused the
                            # completion_truncated failures on the first run).
                            payload["max_completion_tokens"] = min(payload["max_completion_tokens"] + 1024, 4096)
                            retries += 1
                            last_error = "completion_truncated"
                            continue

                        parsed = parse_teacher_json(raw)
                        if parsed is None:
                            retries += 1
                            last_error = "unparseable_json"
                            continue

                        result = record.copy()
                        result["teacher_model"] = model_name
                        kept_any = False
                        for bucket in LENGTH_BUCKETS:
                            summary = parsed[bucket].strip()
                            reason = validate_summary(summary, content, bucket)
                            if reason is None:
                                result[f"summary_{bucket}"] = summary
                                kept_any = True
                                with lock:
                                    stats[f"kept_{bucket}"] += 1
                            else:
                                result[f"rejected_{bucket}"] = reason
                                with lock:
                                    stats[f"rejected_{bucket}"] += 1

                        if kept_any:
                            with lock:
                                with open(output_file, "a", encoding="utf-8") as f:
                                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                                    f.flush()
                                    os.fsync(f.fileno())
                            success = True
                        else:
                            retries += 1
                            last_error = "all_buckets_rejected"

                    elif response.status_code == 429:
                        # Rest THIS key and retry immediately — pool.acquire
                        # will hand out a different key if one has capacity.
                        retries += 1
                        retry_after = response.headers.get("retry-after")
                        wait = float(retry_after) if retry_after else 30.0
                        pool.cooldown(api_key, min(wait, 120))
                        last_error = "429 rate limit"

                    else:
                        retries += 1
                        last_error = f"{response.status_code}: {response.text[:200]}"
                        time.sleep(min(60, (2 ** retries) + 2))

                except requests.exceptions.Timeout as e:
                    retries += 1
                    last_error = f"timeout: {repr(e)}"
                    time.sleep(min(30, 5 * retries) + random.uniform(0, 2))
                except Exception as e:
                    retries += 1
                    last_error = repr(e)
                    time.sleep(min(60, 5 * retries))

            if not success and last_error != "daily_budget_exhausted":
                with lock:
                    with open(output_file, "a", encoding="utf-8") as f:
                        err = record.copy()
                        err["status"] = "failed"
                        err["error"] = f"failed after {retries} retries ({last_error})"
                        f.write(json.dumps(err, ensure_ascii=False) + "\n")
            # daily_budget_exhausted: record NOT written, so tomorrow's run
            # picks this article up again via the URL-resume mechanism.

        finally:
            input_queue.task_done()
            pbar.update(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",       default="/home/jovyan/summarizer/data/train.jsonl")
    parser.add_argument("--output",      default="/home/jovyan/summarizer/data/6_multilength_summaries.jsonl")
    parser.add_argument("--concurrency", type=int, default=0,
                        help="Worker threads. 0 = auto (~3 per key; 8K TPM "
                             "per key supports ~3 in-flight requests each).")
    parser.add_argument("--model",       default=DEFAULT_MODEL)
    args = parser.parse_args()

    keys = [k.strip() for k in GROQ_API_KEYS if k.strip() and "PASTE_YOUR" not in k]
    keys = list(dict.fromkeys(keys))  # dedupe, preserve order
    if not keys:
        print("Add at least one real key to GROQ_API_KEYS at the top of this "
              "script (console.groq.com/keys).")
        return

    input_path, output_path = Path(args.input), Path(args.output)
    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.touch(exist_ok=True)

    pool = GroqKeyPool(keys)
    processed_urls = load_processed_urls(output_path)

    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    to_process = [r for r in records if r.get("url") not in processed_urls]

    if args.concurrency > 0:
        concurrency = min(args.concurrency, 24)
    else:
        concurrency = min(len(keys) * 3, 24)
    concurrency = min(concurrency, max(1, len(to_process)))

    # Realistic daily throughput: per-key TPD is the binding cap; scales
    # linearly with key count.
    est_tokens_per_article = 2_500
    est_articles_today = min(RPD_LIMIT, TPD_LIMIT // est_tokens_per_article) * len(keys)

    print(f"Model             : {args.model}")
    print(f"Endpoint          : {INVOKE_URL}")
    print(f"API keys          : {len(keys)}")
    print(f"Concurrency       : {concurrency}")
    print(f"Total articles    : {len(records):,}")
    print(f"Already done      : {len(processed_urls):,}")
    print(f"Remaining         : {len(to_process):,}")
    print(f"Est. today (TPD)  : ~{est_articles_today} articles before daily token caps")
    print(f"Output            : {output_path.resolve()}")

    if not to_process:
        print("Everything already processed.")
        return

    input_queue = Queue()
    for r in to_process:
        input_queue.put(r)

    lock = threading.Lock()
    stats = {f"{state}_{b}": 0 for b in LENGTH_BUCKETS for state in ("kept", "rejected")}

    with tqdm(total=len(to_process), desc="Summarizing") as pbar:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            for _ in range(concurrency):
                executor.submit(worker, pool, input_queue, output_path, lock, pbar, args.model, stats)
            input_queue.join()

    print(f"\nPer-key budget used:\n{pool.snapshot()}")
    if pool.all_exhausted:
        print("All keys' daily Groq budgets exhausted — re-run tomorrow to continue (resumes by URL).")

    print("\nPer-bucket yield:")
    for bucket in LENGTH_BUCKETS:
        kept, rej = stats[f"kept_{bucket}"], stats[f"rejected_{bucket}"]
        total = kept + rej
        pct = kept / total * 100 if total else 0
        print(f"  {bucket:<7}: kept {kept:,} / {total:,} ({pct:.1f}%)")
    print(f"\nComplete. Results saved to: {output_path}")


if __name__ == "__main__":
    main()
