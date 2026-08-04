"""
Correct_style_dataset.py
========================
Post-processing / correction pass for the style-rewriter dataset.

Steps performed for every row:
  1. Skip rows that are "failed" (status == "failed" or error field present).
  2. For every remaining row, send the (original content + rewritten_text +
     style) to DeepSeek-V4-Pro and ask it to:
       - Fix Sinhala grammar / spelling errors in the rewritten text.
       - Remove stray \\n\\n paragraph breaks.
       - Ensure the style-specific rules (opening/closing phrases, register)
         are correctly followed.
       - Preserve all factual content from the original article.
       - Keep the same approximate length.
  3. Write the corrected row (with "corrected_text" field added) to the output
     JSONL, preserving all original fields.

Resumes safely from a partial output file.
"""

import os
import re
import json
import time
import argparse
import threading
import requests
from pathlib import Path
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────────────
# API CONFIG  (hard-coded as requested)
# ──────────────────────────────────────────────────────────────────────────────
NVIDIA_API_KEY = "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"
INVOKE_URL     = "https://integrate.api.nvidia.com/v1/chat/completions"
CORRECTION_MODEL = "deepseek-ai/deepseek-v4-pro"

# ──────────────────────────────────────────────────────────────────────────────
# STYLE DEFINITIONS  (mirrors train_style.py / the generation script)
# ──────────────────────────────────────────────────────────────────────────────
STYLE_DESCRIPTIONS = {
    "style_1_formal_news": (
        "ප්‍රධාන ප්‍රවෘත්ති වාර්තාකරණය (formal news): "
        "ව්‍යාජ ශ්‍රේෂ්ඨ, කර්තෘ-රහිත (passive) වාක්‍ය, ශීර්ෂ-ත්‍රිකෝණ ව්‍යුහය."
    ),
    "style_2_editorial": (
        "කතු-වැකි / විශ්ලේෂණ (editorial): "
        "'විශ්ලේෂණය කරන විට' ලෙස ආරම්භ, 'අපි' දෙවන-පුරුෂ බහු, "
        "'ඉදිරි ක්‍රියාමාර්ග පිළිබඳව සාකච්ඡා කළ යුතු කාලය එළඹ ඇත' ලෙස අවසන්."
    ),
    "style_3_sports": (
        "ක්‍රීඩා ප්‍රවෘත්ති (sports): "
        "ඉහළ-ශක්තිමත් භාෂාව, ක්‍රියාකාරී ක්‍රියා-පද, නාටකීය මූල-ඡේද, "
        "සීමාසහිත '!' භාවිතය (max 1)."
    ),
    "style_4_youth": (
        "තරුණ-ගොදුරු (youth): "
        "'දන්නවද?'/'ඇහුවද?'/'මේක අහන්න!' ලෙස ආරම්භ, "
        "අවිධිමත් සිංහල, "
        "'අනිවාර්යයෙන්ම දැනගන්න!' ලෙස අවසන්."
    ),
    "style_5_feature": (
        "විශේෂාංග / කතාන්දර (feature): "
        "දෘශ්‍ය-තෝරා-ගත් ආරම්භය, මානුෂික කෝණය, "
        "'මෙම කතාව අනාගත පරම්පරාවට ද ආදර්ශයක් වනු ඇත' ලෙස අවසන්."
    ),
}

SYSTEM_PROMPT = """You are an expert Sinhala linguistics editor. You receive a Sinhala news article in a
specific journalistic style and your job is to CORRECT it, not rewrite it from scratch.

Your ONLY tasks are:
1. Fix genuine Sinhala spelling and grammar errors.
2. Remove stray double-newlines (\\n\\n) - replace with a single space so the paragraph flows naturally.
3. Ensure the mandatory style-specific OPENING phrase is present (if the style requires one).
4. Ensure the mandatory style-specific CLOSING phrase is present (if the style requires one).
5. Make sure NO facts from the original article are missing or altered.
6. Make sure no new facts have been fabricated.
7. Keep the output approximately the same length as the input rewritten text.
8. Do NOT add titles, labels, markdown, or any preamble.

Respond with ONLY the corrected Sinhala article text."""


# ──────────────────────────────────────────────────────────────────────────────
# REQUIRED OPENING / CLOSING MARKERS PER STYLE
# ──────────────────────────────────────────────────────────────────────────────
STYLE_RULES = {
    "style_1_formal_news": {
        "required_opening": None,
        "required_closing": None,
        "register_note": "Use passive voice, objective tone, inverted-pyramid structure.",
    },
    "style_2_editorial": {
        "required_opening": "විශ්ලේෂණය කරන විට",
        "required_closing": "ඉදිරි ක්‍රියාමාර්ග පිළිබඳව සාකච්ඡා කළ යුතු කාලය එළඹ ඇත",
        "register_note": "First-person plural (අපි). Analytical tone.",
    },
    "style_3_sports": {
        "required_opening": None,
        "required_closing": None,
        "register_note": "High-energy, active verbs, dramatic lead. Max one '!' per article.",
    },
    "style_4_youth": {
        "required_opening": "දන්නවද?|ඇහුවද?|මේක අහන්න!",   # any one of these
        "required_closing": "අනිවාර්යයෙන්ම දැනගන්න",
        "register_note": "Informal Sinhala, short punchy sentences.",
    },
    "style_5_feature": {
        "required_opening": None,   # scene-setting open, varied wording - don't enforce exact phrase
        "required_closing": "අනාගත පරම්පරාවට ද ආදර්ශයක් වනු ඇත",
        "register_note": "Narrative/storytelling tone, sensory details, human angle.",
    },
}


def build_correction_prompt(style: str, original_content: str, rewritten_text: str) -> str:
    """Build the user-turn prompt for the correction call."""
    rules = STYLE_RULES.get(style, {})
    style_desc = STYLE_DESCRIPTIONS.get(style, style)

    opening_instruction = ""
    closing_instruction = ""

    req_open = rules.get("required_opening")
    if req_open:
        options = req_open.split("|")
        if len(options) > 1:
            opening_instruction = (
                f"- The article MUST begin with one of these phrases: "
                + ", ".join(f'"{o}"' for o in options)
            )
        else:
            opening_instruction = f'- The article MUST begin with: "{req_open}"'

    req_close = rules.get("required_closing")
    if req_close:
        closing_instruction = f'- The article MUST end with (or contain near the end): "{req_close}"'

    register_note = rules.get("register_note", "")

    constraint_block = "\n".join(filter(None, [opening_instruction, closing_instruction]))

    prompt = f"""=== STYLE ===
{style_desc}
Register: {register_note}

=== STYLE CONSTRAINTS ===
{constraint_block if constraint_block else "(No mandatory opening/closing phrase for this style.)"}

=== ORIGINAL ARTICLE (Sinhala - ground truth for facts) ===
{original_content}

=== REWRITTEN ARTICLE TO CORRECT ===
{rewritten_text}

=== YOUR TASK ===
Correct the rewritten article above by fixing grammar/spelling, removing stray \\n\\n breaks,
and ensuring the style constraints above are met - while preserving ALL facts from the original.
Output ONLY the corrected article text."""

    return prompt


# ──────────────────────────────────────────────────────────────────────────────
# FILTERING - which rows to skip entirely
# ──────────────────────────────────────────────────────────────────────────────
def is_failed_row(record: dict) -> bool:
    """Return True if this row is a generation failure and should be dropped."""
    if record.get("status") == "failed":
        return True
    error_val = record.get("error", "")
    if isinstance(error_val, str) and error_val:
        return True
    # Also drop rows with no rewritten text at all
    rewritten = record.get("rewritten_text", "")
    if not isinstance(rewritten, str) or not rewritten.strip():
        return True
    return False


def clean_newlines(text: str) -> str:
    """Replace stray \\n\\n sequences with a space for flow."""
    # Collapse 2+ consecutive newlines into one space
    text = re.sub(r"\n{2,}", " ", text)
    # Collapse leftover single newlines mid-sentence (optional, gentle)
    # We keep single \n if they look like intentional paragraph breaks after
    # short lines; otherwise collapse.
    text = re.sub(r"(?<![.!?])\n(?=[^\n])", " ", text)
    return text.strip()


# ──────────────────────────────────────────────────────────────────────────────
# RESUME SUPPORT
# ──────────────────────────────────────────────────────────────────────────────
def load_processed_keys(output_path: Path) -> set:
    """Return set of (url, style) already written to the output file."""
    done = set()
    if not output_path.exists():
        return done
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
                if "url" in d and "style" in d:
                    done.add((d["url"], d["style"]))
            except Exception:
                continue
    return done


# ──────────────────────────────────────────────────────────────────────────────
# ADAPTIVE RATE LIMITER  (same design as generation script)
# ──────────────────────────────────────────────────────────────────────────────
class RateLimiter:
    def __init__(self, rpm: int, max_rpm: int = None, min_rpm: int = 3,
                 ramp_every: int = 20, ramp_factor: float = 1.1,
                 backoff_factor: float = 0.6):
        self.rpm            = max(rpm, 1)
        self.max_rpm        = max_rpm or (rpm * 6)
        self.min_rpm        = min_rpm
        self.ramp_every     = ramp_every
        self.ramp_factor    = ramp_factor
        self.backoff_factor = backoff_factor
        self.success_streak = 0
        self.min_interval   = 60.0 / self.rpm
        self.lock           = threading.Lock()
        self.next_allowed   = 0.0
        self.pause_until    = 0.0

    def wait(self):
        while True:
            with self.lock:
                now      = time.monotonic()
                deadline = max(self.next_allowed, self.pause_until)
                gap      = deadline - now
                if gap <= 0:
                    self.next_allowed = now + self.min_interval
                    return
            time.sleep(gap)

    def report_success(self):
        with self.lock:
            self.success_streak += 1
            if self.success_streak >= self.ramp_every and self.rpm < self.max_rpm:
                old = self.rpm
                self.rpm = min(self.max_rpm, self.rpm * self.ramp_factor)
                self.min_interval = 60.0 / self.rpm
                self.success_streak = 0
                if int(self.rpm) != int(old):
                    print(f"\n[RateLimiter] Ramping up: {old:.1f} → {self.rpm:.1f} rpm")

    def report_429(self, retry_after: float):
        with self.lock:
            self.pause_until = max(self.pause_until, time.monotonic() + retry_after)
            old = self.rpm
            self.rpm = max(self.min_rpm, self.rpm * self.backoff_factor)
            self.min_interval = 60.0 / self.rpm
            self.success_streak = 0
            print(f"\n[RateLimiter] 429 → cutting rate: {old:.1f} → {self.rpm:.1f} rpm")


# ──────────────────────────────────────────────────────────────────────────────
# WORKER
# ──────────────────────────────────────────────────────────────────────────────
def worker(input_queue: Queue, output_path: Path, lock: threading.Lock,
           pbar, rate_limiter: RateLimiter, model_name: str):
    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Accept": "application/json",
    }

    while True:
        try:
            record = input_queue.get(timeout=5)
        except Empty:
            break

        try:
            style    = record.get("style", "")
            original = record.get("content", "")
            rewritten = record.get("rewritten_text", "")

            # Quick local clean before sending to the model
            rewritten_pre = clean_newlines(rewritten)

            prompt = build_correction_prompt(style, original, rewritten_pre)

            # Token budget: correction is shorter than generation but still
            # needs headroom for the full article + any additions for
            # mandatory opening/closing phrases.
            word_count   = len(rewritten_pre.split())
            max_tokens   = min(max(512, int(word_count * 2.5)), 4096)

            payload = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                "max_tokens":        max_tokens,
                "temperature":       0.15,   # low - we want conservative corrections
                "top_p":             0.9,
                "frequency_penalty": 0.2,
                "presence_penalty":  0.1,
                "stream":            False,
                # Required for DeepSeek reasoning models on NVIDIA NIM -
                # without this the request hangs indefinitely or 404s.
                "chat_template_kwargs": {
                    "thinking": False,   # disable chain-of-thought to save tokens
                },
            }

            success  = False
            retries  = 0
            corrected_text = rewritten_pre  # fallback: use the locally-cleaned version

            while not success and retries < 5:
                rate_limiter.wait()
                try:
                    response = requests.post(
                        INVOKE_URL,
                        headers=headers,
                        json=payload,
                        timeout=400,
                    )

                    if response.status_code == 200:
                        res_json = response.json()
                        try:
                            msg = res_json["choices"][0]["message"]
                            text = msg.get("content") or ""
                            corrected_text = clean_newlines(text.strip())
                        except Exception:
                            corrected_text = rewritten_pre

                        out_record = record.copy()
                        out_record["corrected_text"] = corrected_text

                        with lock:
                            with open(output_path, "a", encoding="utf-8") as f:
                                f.write(json.dumps(out_record, ensure_ascii=False) + "\n")
                                f.flush()
                                os.fsync(f.fileno())

                        success = True
                        rate_limiter.report_success()

                    elif response.status_code == 429:
                        header_val = response.headers.get("Retry-After")
                        retry_after = None
                        if header_val:
                            try:
                                retry_after = float(header_val)
                            except ValueError:
                                pass
                        if retry_after is None:
                            try:
                                body = response.json()
                                retry_after = float(body.get("retryDelay", 0)) or None
                            except Exception:
                                pass
                        retry_after = retry_after or (10 * (retries + 1))
                        print(f"\n[429] Pausing all threads {retry_after:.1f}s")
                        rate_limiter.report_429(retry_after)
                        retries += 1
                        time.sleep((2 ** retries) + 2)

                    else:
                        print(f"\n[Error {response.status_code}] {response.text[:300]}")
                        retries += 1
                        time.sleep((2 ** retries) + 2)

                except Exception as exc:
                    print(f"\n[Exception] {repr(exc)}")
                    retries += 1
                    time.sleep(3)

            # If all retries failed, still write the row with the locally-
            # cleaned rewritten text so progress isn't lost entirely.
            if not success:
                out_record = record.copy()
                out_record["corrected_text"]      = rewritten_pre
                out_record["correction_status"]   = f"api_failed_after_{retries}_retries"
                with lock:
                    with open(output_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(out_record, ensure_ascii=False) + "\n")
                        f.flush()

        finally:
            input_queue.task_done()
            pbar.update(1)


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Correct rewritten Sinhala articles using DeepSeek-V4-Pro."
    )
    parser.add_argument(
        "--input",
        default=r"/home/jovyan/style_rewriter/data/style_dataset2_dub_cleaned.jsonl",
        help="Input JSONL file produced by the generation script.",
    )
    parser.add_argument(
        "--output",
        default=r"/home/jovyan/style_rewriter/data/style_dataset_corrected.jsonl",
        help="Output JSONL file with corrected articles.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Number of parallel worker threads.",
    )
    parser.add_argument(
        "--rpm",
        type=int,
        default=10,
        help="Starting requests-per-minute (adaptive limiter will tune from here).",
    )
    parser.add_argument(
        "--max-rpm",
        type=int,
        default=None,
        help="Hard ceiling for the adaptive rate limiter.",
    )
    parser.add_argument(
        "--model",
        default=CORRECTION_MODEL,
        help="NVIDIA NIM model name to use for correction.",
    )
    parser.add_argument(
        "--styles",
        default=None,
        help="Comma-separated subset of styles to process, e.g. style_2_editorial,style_4_youth. "
             "Default: process all styles present in the input file.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N valid rows (useful for a quick test run).",
    )
    args = parser.parse_args()

    input_path  = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.touch(exist_ok=True)

    # ── Filter style whitelist ──────────────────────────────────────────────
    style_filter = None
    if args.styles:
        style_filter = {s.strip() for s in args.styles.split(",") if s.strip()}
        print(f"Style filter: {style_filter}")

    # ── Load input ─────────────────────────────────────────────────────────
    print(f"Loading input: {input_path}")
    all_records = []
    failed_count = 0
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            if is_failed_row(rec):
                failed_count += 1
                continue

            if style_filter and rec.get("style") not in style_filter:
                continue

            all_records.append(rec)

    print(f"  Total rows in file : {failed_count + len(all_records)}")
    print(f"  Dropped (failed)   : {failed_count}")
    print(f"  Valid rows          : {len(all_records)}")

    # ── Resume support ──────────────────────────────────────────────────────
    processed_keys = load_processed_keys(output_path)
    if processed_keys:
        print(f"  Already corrected  : {len(processed_keys)} (url, style) pairs — skipping.")

    to_process = [
        r for r in all_records
        if (r.get("url"), r.get("style")) not in processed_keys
    ]

    if args.limit is not None:
        to_process = to_process[: args.limit]

    print(f"  Remaining to fix   : {len(to_process)}")

    if not to_process:
        print("Nothing left to process. Exiting.")
        return

    # ── Enqueue ────────────────────────────────────────────────────────────
    q = Queue()
    for rec in to_process:
        q.put(rec)

    lock         = threading.Lock()
    rate_limiter = RateLimiter(args.rpm, max_rpm=args.max_rpm)

    print(f"\nModel       : {args.model}")
    print(f"Concurrency : {args.concurrency}")
    print(f"Starting RPM: {args.rpm}")
    eff_max = args.max_rpm or (args.rpm * 6)
    print(f"Max RPM     : {eff_max}")
    est_min_lo = len(to_process) / eff_max
    est_min_hi = len(to_process) / args.rpm
    print(f"Est. time   : {est_min_lo:.0f}–{est_min_hi:.0f} min "
          f"(~{est_min_lo/60:.1f}–{est_min_hi/60:.1f} h)\n")

    with tqdm(total=len(to_process), desc="Correcting") as pbar:
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            for _ in range(args.concurrency):
                executor.submit(worker, q, output_path, lock, pbar,
                                rate_limiter, args.model)
            q.join()

    print(f"\nDone! Corrected dataset saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
