"""
SinhalaJournal-LLM | Step 5: Generate summaries using Gemma 4 31B (NVIDIA)
---------------------------------------------------------------------------
Generates silver-label Sinhala summaries using Qwen3-next-80b-asb-instruct model via
the NVIDIA NIM free endpoint. 

Usage:
    export NVIDIA_API_KEY=your_key
    python abstractive/5_qwen_summary_generator.py

    # Custom paths or concurrency:
    python abstractive/5_qwen_summary_generator.py \
        --input data/train.jsonl \
        --output data/5_gqwen_summaries.jsonl \
        --concurrency 10
"""

import os
import json
import time
import random
import argparse
import threading
from pathlib import Path
from collections import deque
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor

import requests
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

INVOKE_URL   = "https://integrate.api.nvidia.com/v1/chat/completions"
DEFAULT_MODEL = "qwen/qwen3-next-80b-a3b-instruct"

RPM_PER_KEY   = 40   # NVIDIA free-tier requests-per-minute per key
MAX_RETRIES   = 6
DEFAULT_KEYS_FILE = str(Path(__file__).with_name("nvidia-apikeys.txt"))

SYSTEM_PROMPT = """You are an expert Sinhala news summarization assistant.
Respond only in Sinhala. Be concise and factual."""

USER_INSTRUCTION = """ඔබ සිංහල පුවත් සාරාංශකරණ විශේෂඥයෙකි.

පහත නීති අනිවාර්යයෙන්ම අනුගමනය කරන්න:
1. ලිපියේ ඇති තොරතුරු පමණක් භාවිතා කරන්න.
2. නව තොරතුරු එකතු නොකරන්න.
3. පුද්ගලයන්, ස්ථාන, සංඛ්‍යා වෙනස් නොකරන්න.
4. අදහස්, විශ්ලේෂණ හෝ අනුමාන එකතු නොකරන්න.
5. සාරාංශය මුල් ලිපියේ දිගෙන් 10% ත් 30% ත් අතර විය යුතුය.
6. ප්‍රධාන කරුණු පමණක් ඇතුළත් කරන්න.
7. සාරාංශය පමණක් ලබා දෙන්න.

Article:
{content}
"""


def load_api_keys(keys_file: str) -> list:
    """Load API keys from a file (one per line), falling back to env var."""
    keys = []
    path = Path(keys_file)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                key = line.strip()
                if key and not key.startswith("#"):
                    keys.append(key)

    env_key = os.getenv("NVIDIA_API_KEY")
    if env_key and env_key not in keys:
        keys.append(env_key)

    if not keys:
        raise RuntimeError(
            f"No API keys found in {keys_file} or NVIDIA_API_KEY env var."
        )
    # De-duplicate while preserving order.
    seen = set()
    unique = []
    for k in keys:
        if k not in seen:
            seen.add(k)
            unique.append(k)
    return unique


class KeyRateLimiter:
    """Thread-safe pool that hands out API keys while respecting a per-key RPM.

    Each key may be used at most `rpm` times in any rolling 60-second window.
    acquire() blocks until some key has a free slot, then reserves it.
    A key can also be temporarily disabled (e.g. DEGRADED model) via cooldown.
    """

    def __init__(self, keys: list, rpm: int):
        self.rpm = rpm
        self.lock = threading.Lock()
        self.calls = {k: deque() for k in keys}   # timestamps of recent calls
        self.cooldown_until = {k: 0.0 for k in keys}
        self.keys = list(keys)

    def acquire(self) -> str:
        while True:
            now = time.time()
            best_key = None
            earliest_free = None

            with self.lock:
                for key in self.keys:
                    if now < self.cooldown_until[key]:
                        continue
                    dq = self.calls[key]
                    while dq and now - dq[0] >= 60:
                        dq.popleft()
                    if len(dq) < self.rpm:
                        dq.append(now)
                        return key
                    # Track when this key's oldest call ages out.
                    free_at = dq[0] + 60
                    if earliest_free is None or free_at < earliest_free:
                        earliest_free = free_at

                # All keys busy or cooling down; find soonest availability.
                soonest_cooldown = min(
                    (c for c in self.cooldown_until.values() if c > now),
                    default=None,
                )
                candidates = [t for t in (earliest_free, soonest_cooldown) if t]
                wait = (min(candidates) - now) if candidates else 1.0

            time.sleep(max(0.05, min(wait, 5.0)))

    def cooldown(self, key: str, seconds: float):
        with self.lock:
            self.cooldown_until[key] = max(
                self.cooldown_until[key], time.time() + seconds
            )


def load_processed_urls(output_path: Path) -> set:
    processed = set()
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if "url" in data:
                        processed.add(data["url"])
                except Exception:
                    continue
    return processed


def worker(limiter: KeyRateLimiter, input_queue: Queue, output_file: Path,
           lock: threading.Lock, pbar, model_name: str):
    while True:
        try:
            record = input_queue.get(timeout=5)
        except Empty:
            break

        try:
            content = record.get("content", "").strip()
            if not content:
                with lock:
                    with open(output_file, "a", encoding="utf-8") as f:
                        err = record.copy()
                        err["error"] = "empty content"
                        f.write(json.dumps(err, ensure_ascii=False) + "\n")
                        f.flush()
                pbar.update(1)
                continue

            payload = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": USER_INSTRUCTION.format(content=content)},
                ],
                "max_tokens": 1024,
                "temperature": 0.0,
                "top_p": 1.0,
                "stream": False,
            }

            success = False
            retries = 0
            last_error = ""

            while not success and retries < MAX_RETRIES:
                api_key = limiter.acquire()
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Accept": "application/json",
                }
                try:
                    response = requests.post(
                        INVOKE_URL,
                        headers=headers,
                        json=payload,
                        timeout=(10, 120),
                    )

                    if response.status_code == 200:
                        res_json = response.json()
                        try:
                            choice = res_json["choices"][0]
                            summary = choice["message"].get("content") or ""
                            summary = summary.strip()
                            truncated = choice.get("finish_reason") == "length"
                        except Exception:
                            summary = str(res_json)
                            truncated = False

                        if truncated:
                            payload["max_tokens"] = min(payload["max_tokens"] + 512, 2048)
                            # print(f"\n[Truncated] retrying with max_tokens={payload['max_tokens']}: {record.get('url')}")
                            retries += 1
                            continue

                        result = record.copy()
                        result["qwen_summary"] = summary

                        with lock:
                            with open(output_file, "a", encoding="utf-8") as f:
                                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                                f.flush()
                                os.fsync(f.fileno())

                        success = True

                    elif response.status_code == 429:
                        # This key is rate-limited; rest it for the RPM window.
                        retries += 1
                        limiter.cooldown(api_key, 60)
                        last_error = "429 rate limit"
                        # print(f"\n[RateLimit] key ...{api_key[-6:]} cooling down 60s (retry {retries}/{MAX_RETRIES})")

                    elif response.status_code == 400 and "DEGRADED" in response.text:
                        # Model function is degraded/unavailable; back off and retry.
                        retries += 1
                        wait = min(120, 20 * retries)
                        last_error = "DEGRADED function"
                        # print(f"\n[DEGRADED] model unavailable, waiting {wait}s (retry {retries}/{MAX_RETRIES})")
                        time.sleep(wait)

                    else:
                        retries += 1
                        last_error = f"{response.status_code}: {response.text[:200]}"
                        # print(f"\n[Error {response.status_code}] {response.text[:300]}")
                        time.sleep(min(60, (2 ** retries) + 2))

                except requests.exceptions.Timeout as e:
                    retries += 1
                    last_error = f"timeout: {repr(e)}"
                    # print(f"\n[Timeout] {repr(e)} (retry {retries}/{MAX_RETRIES})")
                    time.sleep(min(30, 5 * retries) + random.uniform(0, 2))

                except Exception as e:
                    retries += 1
                    last_error = repr(e)
                    # print(f"\n[Exception] {repr(e)} (retry {retries}/{MAX_RETRIES})")
                    time.sleep(min(60, 5 * retries))

            if not success:
                with lock:
                    with open(output_file, "a", encoding="utf-8") as f:
                        err = record.copy()
                        err["status"] = "failed"
                        err["error"] = f"failed after {retries} retries ({last_error})"
                        f.write(json.dumps(err, ensure_ascii=False) + "\n")
                        f.flush()
                        os.fsync(f.fileno())

        finally:
            input_queue.task_done()
            pbar.update(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",       default="/home/jovyan/summarizer/data/train.jsonl")
    parser.add_argument("--output",      default="/home/jovyan/summarizer/data/5_qwen_summaries.jsonl")
    parser.add_argument("--concurrency", type=int, default=0,
                        help="Worker threads. 0 = auto (RPM_PER_KEY/40 * num_keys, capped).")
    parser.add_argument("--keys-file",   default=DEFAULT_KEYS_FILE)
    parser.add_argument("--model",       default=DEFAULT_MODEL)
    args = parser.parse_args()

    input_path  = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.touch(exist_ok=True)

    api_keys       = load_api_keys(args.keys_file)
    limiter        = KeyRateLimiter(api_keys, RPM_PER_KEY)
    processed_urls = load_processed_urls(output_path)

    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    to_process = [r for r in records if r.get("url") not in processed_urls]

    # Auto concurrency: enough threads to keep all keys' RPM busy, but capped
    # so we don't oversubscribe (each request takes seconds, so threads ≈ keys*a few).
    if args.concurrency > 0:
        concurrency = args.concurrency
    else:
        concurrency = min(len(api_keys) * 8, 40)
    concurrency = min(concurrency, max(1, len(to_process)))

    total_rpm = len(api_keys) * RPM_PER_KEY

    # Rough completion estimate: RPM ceiling is the throughput bound, but real
    # throughput is also limited by request latency * concurrency. Take the
    # tighter of the two so the estimate isn't wildly optimistic.
    est_by_rpm      = len(to_process) / total_rpm * 60          # seconds
    avg_latency_s   = 8.0                                        # typical per-request wall time
    est_by_threads  = len(to_process) * avg_latency_s / concurrency
    est_seconds     = max(est_by_rpm, est_by_threads)
    eta_h, rem      = divmod(int(est_seconds), 3600)
    eta_m, eta_s    = divmod(rem, 60)

    print(f"Model          : {args.model}")
    print(f"API keys       : {len(api_keys)}")
    print(f"Aggregate RPM  : {total_rpm}")
    print(f"Concurrency    : {concurrency}")
    print(f"Total articles : {len(records)}")
    print(f"Already done   : {len(processed_urls)}")
    print(f"Remaining      : {len(to_process)}")
    print(f"Est. runtime   : ~{eta_h}h {eta_m}m {eta_s}s (live ETA shown below)")
    print(f"Output         : {output_path.resolve()}")

    if not to_process:
        print("Everything already processed.")
        return

    input_queue = Queue()
    for r in to_process:
        input_queue.put(r)

    lock = threading.Lock()

    with tqdm(total=len(to_process), desc="Summarizing") as pbar:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            for _ in range(concurrency):
                executor.submit(worker, limiter, input_queue, output_path, lock, pbar, args.model)
            input_queue.join()

    print(f"\nComplete. Results saved to: {output_path}")


if __name__ == "__main__":
    main()
