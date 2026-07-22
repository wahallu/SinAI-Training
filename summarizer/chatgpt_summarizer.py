"""
SinhalaJournal-LLM | Step 6: Multi-length silver summary generator (ChatGPT / Playwright)
-----------------------------------------------------------------------------------------
Generates THREE length-conditioned Sinhala summaries (short / medium / long)
per article in a single ChatGPT query, returned as JSON.

Uses Playwright connected to Chrome via CDP (Chrome DevTools Protocol) to bypass
anti-bot detection on chat.openai.com without needing API keys.

Usage:
    python summarizer/chatgpt_summarizer.py
    python summarizer/chatgpt_summarizer.py \
        --input "D:\\SinhalaLLM\\cleaned_datasets\\all_articles_merged.json" \
        --output "D:\\SinhalaLLM\\cleaned_datasets\\6_multilength_summaries.jsonl" \
        --tabs 3
"""

import os
import re
import json
import time
import random
import argparse
import asyncio
import subprocess
import sys
import unicodedata
from pathlib import Path
from datetime import datetime
from threading import Lock

# Enable UTF-8 encoding for Windows console output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# ── Chrome CDP Settings ────────────────────────────────────────────────────────
CHROME_DEBUG_PORT = 9222
CHROME_USER_DATA  = r"C:\chrome_debug_profile"
CHATGPT_URL       = "https://chatgpt.com/"

SUBMIT_WAIT_SEC      = 4      # seconds after submit before polling starts
RESPONSE_TIMEOUT_SEC = 180    # max seconds to wait for a full response
POLL_INTERVAL_SEC    = 1.5    # seconds between DOM polls while streaming
BETWEEN_REQUESTS_SEC = 4      # seconds to rest after each completed request
STABLE_POLLS_NEEDED  = 3      # consecutive unchanged polls = done
MAX_ARTICLE_RETRIES  = 10     # retry up to 10 times before giving up on an article

# ── Length buckets ──────────────────────────────────────────────
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

# ── Quality validation ──────────────────────────────────────────
def digits_in(text: str) -> set:
    return set(re.findall(r"\d+", text))

def validate_summary(summary: str, article: str, bucket: str) -> str | None:
    """Returns a rejection reason, or None if the summary passes."""
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
    summary_words = len(summary.split())

    if summary_words < MIN_SUMMARY_WORDS:
        return "too_short"
    if summary_words > cfg["max_words"]:
        return "over_word_cap"

    ratio = summary_words / article_words if article_words else 0
    if ratio < cfg["min_ratio"] or ratio > cfg["max_ratio"]:
        return f"ratio_out_of_band:{ratio:.2f}"

    # Exclude prompt instruction numbers and list indices from hallucinated digits check
    extra_digits = (digits_in(summary) - digits_in(article)) - {"10", "20", "35", "1", "2", "3"}
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

# ── Chrome Launcher ───────────────────────────────────────────────────────────
def ensure_chrome_debug(custom_profile: str = "") -> bool:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        try:
            s.connect(("127.0.0.1", CHROME_DEBUG_PORT))
            log(f"Chrome already listening on port {CHROME_DEBUG_PORT}.", "INFO")
            return True
        except (ConnectionRefusedError, OSError):
            pass

    chrome_candidates = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
    ]
    chrome_path = next((p for p in chrome_candidates if os.path.exists(p)), None)

    if not chrome_path:
        log("Chrome not found. Launch it manually with:", "ERR")
        log(f'  chrome.exe --remote-debugging-port={CHROME_DEBUG_PORT} --user-data-dir="{CHROME_USER_DATA}"', "ERR")
        return False

    user_data = CHROME_USER_DATA
    cmd = [
        chrome_path,
        f"--remote-debugging-port={CHROME_DEBUG_PORT}",
        f"--user-data-dir={user_data}",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-extensions",
        "--disable-infobars",
        "about:blank",
    ]

    log(f"Launching Chrome: {chrome_path}", "INFO")
    os.makedirs(user_data, exist_ok=True)
    subprocess.Popen(cmd)

    for _ in range(20):
        time.sleep(1)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            try:
                s.connect(("127.0.0.1", CHROME_DEBUG_PORT))
                log("Chrome ready.", "INFO")
                return True
            except (ConnectionRefusedError, OSError):
                pass

    log("Chrome did not start in time.", "ERR")
    return False

def load_processed_urls(output_path: Path) -> set:
    processed = set()
    if output_path.exists():
        try:
            os.chmod(output_path, 0o666)
        except Exception:
            pass

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

# ── Playwright Page Helpers ───────────────────────────────────────────────────
async def dismiss_popups(page):
    selectors = [
        'button:has-text("Stay logged out")',
        'button:has-text("Continue without account")',
        'button:has-text("Continue without logging in")',
        'button:has-text("Continue")',
        '[data-testid="close-button"]',
        '[aria-label="Close"]',
    ]
    for sel in selectors:
        try:
            btn = page.locator(sel).first
            if await btn.is_visible(timeout=1000):
                await btn.click()
                await page.wait_for_timeout(500)
        except Exception:
            pass
    try:
        await page.keyboard.press("Escape")
    except Exception:
        pass

async def get_input_box(page):
    selectors = [
        '#mobile-composer-prompt',
        '#prompt-textarea',
        'textarea[placeholder*="Ask"]',
        'textarea',
        'div[contenteditable="true"]',
    ]
    for sel in selectors:
        try:
            el = page.locator(sel).first
            if await el.is_visible(timeout=2000):
                return el
        except Exception:
            pass
    return None

async def send_prompt(page, prompt: str):
    input_box = await get_input_box(page)
    if input_box is None:
        raise RuntimeError("Could not find the ChatGPT input box")

    await input_box.focus()
    await input_box.fill(prompt)
    await page.wait_for_timeout(500)

    send_btn = page.locator('button[data-testid="send-button"], button[aria-label*="Send"]').first
    if await send_btn.is_visible(timeout=1500):
        await send_btn.click()
    else:
        await page.keyboard.press("Control+Enter")

async def is_still_generating(page) -> bool:
    for sel in [
        'button[aria-label*="Stop"]',
        'button[data-testid="stop-button"]',
        'button:has-text("Stop")',
    ]:
        try:
            if await page.locator(sel).first.is_visible(timeout=400):
                return True
        except Exception:
            pass
    return False

async def get_last_assistant_text(page) -> str:
    """Extract assistant response text from ChatGPT DOM."""
    try:
        body_text = await page.evaluate("document.body.innerText")
        if "ChatGPT said:" in body_text:
            part = body_text.split("ChatGPT said:")[-1]
            for footer_marker in [
                "New chat", "Search chats", "Log in to get answers",
                "See plans and pricing", "Log in", "Sign up for free"
            ]:
                if footer_marker in part:
                    part = part.split(footer_marker)[0]
            txt = part.strip()
            if txt and len(txt) > 5:
                return txt
    except Exception:
        pass

    for sel in [
        '[data-message-author-role="assistant"]',
        '.markdown',
        'article[data-testid*="conversation-turn"]:last-child',
        'div[class*="prose"]:last-child',
    ]:
        try:
            els = page.locator(sel)
            count = await els.count()
            if count > 0:
                text = await els.nth(count - 1).inner_text(timeout=2000)
                if text and len(text.strip()) > 5:
                    return text.strip()
        except Exception:
            pass

    return ""

async def chatgpt_generate_raw(page, prompt: str, label: str) -> str:
    await page.goto(CHATGPT_URL, wait_until="domcontentloaded", timeout=60000)
    await page.wait_for_timeout(2500)
    await dismiss_popups(page)

    await send_prompt(page, prompt)

    log(f"  {label} — prompt submitted, waiting for response...", "INFO")
    await page.wait_for_timeout(SUBMIT_WAIT_SEC * 1000)

    deadline     = time.time() + RESPONSE_TIMEOUT_SEC
    last_text    = ""
    stable_count = 0

    while time.time() < deadline:
        still_gen    = await is_still_generating(page)
        current_text = await get_last_assistant_text(page)

        if current_text and current_text == last_text and not still_gen:
            stable_count += 1
            if stable_count >= STABLE_POLLS_NEEDED:
                break
        else:
            stable_count = 0

        last_text = current_text
        await page.wait_for_timeout(int(POLL_INTERVAL_SEC * 1000))

    final = await get_last_assistant_text(page)
    if not final:
        raise RuntimeError("Empty response — ChatGPT may have been blocked or UI changed")
    return final

# ── Worker ────────────────────────────────────────────────────────────────────
async def worker(
    worker_id: int,
    context,
    queue: asyncio.Queue,
    output_path: Path,
    stats: dict,
    write_lock: asyncio.Lock,
    model_name: str,
):
    tag  = f"W{worker_id}"
    page = await context.new_page()
    log(f"Worker {worker_id} ready.", tag)

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
                raw_response = await chatgpt_generate_raw(page, prompt, label)
                parsed = parse_teacher_json(raw_response)

                if parsed is None:
                    log(f"  {label} ✗ Unparseable JSON (attempt {attempt}/{MAX_ARTICLE_RETRIES})", tag)
                else:
                    result = record.copy()
                    result["teacher_model"] = model_name
                    kept_any = False

                    for bucket in LENGTH_BUCKETS:
                        summary = parsed[bucket].strip()
                        reason = validate_summary(summary, content, bucket)
                        if reason is None:
                            result[f"summary_{bucket}"] = summary
                            kept_any = True
                            async with write_lock:
                                stats[f"kept_{bucket}"] += 1
                        else:
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
                        log(f"  {label} ✗ All buckets rejected (attempt {attempt}/{MAX_ARTICLE_RETRIES})", tag)

            except Exception as e:
                log(f"  {label} ✗ Exception on attempt {attempt}/{MAX_ARTICLE_RETRIES}: {e}", tag)

            if attempt < MAX_ARTICLE_RETRIES:
                log(f"  {label} 🔄 Retrying article (attempt {attempt + 1}/{MAX_ARTICLE_RETRIES})…", tag)
                await asyncio.sleep(2)

        if not success:
            log(f"  {label} ⚠️ Article failed after {MAX_ARTICLE_RETRIES} retries — ignoring (not saved to output file)", tag)

        queue.task_done()
        await asyncio.sleep(BETWEEN_REQUESTS_SEC)

    await page.close()
    log(f"Worker {worker_id} done.", tag)

# ── Main Async Execution ──────────────────────────────────────────────────────
async def async_main(args):
    from playwright.async_api import async_playwright

    input_path  = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        log(f"Input file not found: {input_path}", "ERR")
        sys.exit(1)

    log(f"Loading {input_path}…")
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

    print(f"Model          : {args.model}")
    print(f"Total articles : {len(records):,}")
    print(f"Already done   : {len(processed_urls):,}")
    print(f"Remaining      : {len(to_process):,}")
    print(f"Tabs           : {args.tabs}")
    print(f"Output         : {output_path.resolve()}")

    if not to_process:
        print("Everything already processed.")
        return

    if not ensure_chrome_debug():
        sys.exit(1)

    queue: asyncio.Queue = asyncio.Queue()
    for item in to_process:
        queue.put_nowait(item)

    write_lock = asyncio.Lock()
    stats = {f"{state}_{b}": 0 for b in LENGTH_BUCKETS for state in ("kept", "rejected")}

    async with async_playwright() as pw:
        log(f"Connecting to Chrome on port {CHROME_DEBUG_PORT}…")
        try:
            browser = await pw.chromium.connect_over_cdp(f"http://localhost:{CHROME_DEBUG_PORT}")
        except Exception as e:
            log(f"Could not connect to Chrome: {e}", "ERR")
            sys.exit(1)

        context = browser.contexts[0] if browser.contexts else await browser.new_context()
        log("Connected to Chrome ✓")

        n_workers = min(args.tabs, len(to_process))
        tasks = [
            asyncio.create_task(worker(i + 1, context, queue, output_path, stats, write_lock, args.model))
            for i in range(n_workers)
        ]
        await asyncio.gather(*tasks)

    print("\nPer-bucket yield:")
    for bucket in LENGTH_BUCKETS:
        kept, rej = stats[f"kept_{bucket}"], stats[f"rejected_{bucket}"]
        total = kept + rej
        pct = kept / total * 100 if total else 0
        print(f"  {bucket:<7}: kept {kept:,} / {total:,} ({pct:.1f}%)")
    print(f"\nComplete. Results saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Multi-length Sinhala summary generator using chat.openai.com via Playwright"
    )
    parser.add_argument("--input",       default="D:\\SinhalaLLM\\cleaned_datasets\\all_articles_merged.json",
                        help="Path to input dataset file (JSON or JSONL)")
    parser.add_argument("--output",      default="D:\\SinhalaLLM\\cleaned_datasets\\6_multilength_summaries.jsonl",
                        help="Path to output file")
    parser.add_argument("--tabs",        type=int, default=3,
                        help="Number of concurrent browser tabs (default: 3)")
    parser.add_argument("--model",       default="chatgpt-guest",
                        help="Teacher model label (default: chatgpt-guest)")
    args = parser.parse_args()
    asyncio.run(async_main(args))

if __name__ == "__main__":
    main()
