import os
import json
import time
import random
import argparse
import requests
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
import threading
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor
os.environ["NVIDIA_API_KEY"] = "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"

load_dotenv()

INVOKE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
DEFAULT_MODEL = "google/diffusiongemma-26b-a4b-it"

SYSTEM_PROMPT = """You are an expert Sinhala journalism writer who rewrites news articles
into different stylistic registers while preserving every fact exactly.
This is a REWRITE task, not a summarization task: the output must cover every
fact and be approximately the same length as the source article — never
compress it into a shorter summary.
Respond only with the rewritten article text — no title, no preamble, no
explanation, no markdown, no labels like "Article:" or "Rewritten:"."""

# Junk patterns that sometimes leak into scraped article text (CMS embed
# shortcodes, leftover HTML, etc.) and should never reach the model.
import re

_JUNK_PATTERNS = [
    re.compile(r"\[ot-video[^\]]*\]?", re.IGNORECASE),
    re.compile(r"\[/?ot-video[^\]]*\]?", re.IGNORECASE),
    re.compile(r"<[^>]+>"),  # stray HTML tags
    re.compile(r"\u00ad"),  # soft hyphen
]


def clean_content(text: str) -> str:
    for pattern in _JUNK_PATTERNS:
        text = pattern.sub("", text)
    return text.strip()

# -----------------------------------------------------------------------
# Canonical style instructions. Keys MUST stay byte-for-byte identical to
# whatever is used at inference/training time (build_prompt / style tag).
# -----------------------------------------------------------------------
STYLE_INSTRUCTIONS = {
    "style_1_formal_news": """
ඔබ සිංහල ප්‍රධාන පුවත් වාර්තාකරණ විශේෂඥයෙකි.

පහත නීති අනිවාර්යයෙන්ම අනුගමනය කරන්න:
1. ලිපියේ ඇති කරුණු පමණක් භාවිතා කරන්න. නව තොරතුරු එකතු නොකරන්න.
2. පුද්ගලයන්, ස්ථාන, සංඛ්‍යා නිවැරදිව තබා ගන්න. කිසිවක් වෙනස් නොකරන්න.
3. සියලු වාක්‍ය වෛෂයික (objective) විය යුතුය - කිසිදු අදහසක් හෝ විශ්ලේෂණයක් එක් නොකරන්න.
4. හැකි සෑම තැනකම කර්තෘ විරහිත වාක්‍ය (passive voice) භාවිතා කරන්න.
5. උපුල්ලංකෘත පිරමිඩ ව්‍යුහය අනුගමනය කරන්න - වැදගත්ම කරුණ මුලින්ම දක්වන්න.
6. මුල් ලිපියේ ඇති සියලුම කරුණු එලෙසින්ම තබා ගන්න.
7. නැවත ලියූ ලිපියේ දිග මුල් ලිපියේ දිගට ආසන්න විය යුතුය (කෙටි කිරීම හෝ දිගු කිරීම නොකරන්න).
8. මුල් ලිපියේ ඉංග්‍රීසි වචන නොතිබුනේ නම්, ඉංග්‍රීසි වචන එකතු නොකරන්න.
9. අකුරු වැරදි නොකරන්න. සියලු වචන නිවැරදිව අක්ෂර වින්‍යාසයෙන් ලියන්න.
10. වාක්‍ය සම්පූර්ණ විය යුතුය - අතරමගදී කපා නොදමන්න.

Article:
{content}
""",

    "style_2_editorial": """
ඔබ සිංහල කතුවැකි/විශ්ලේෂණ විශේෂඥයෙකි.

පහත නීති අනිවාර්යයෙන්ම අනුගමනය කරන්න:
1. "විශ්ලේෂණය කරන විට" යන වදනින් ලිපිය ආරම්භ කරන්න.
2. පළමු පුරුෂ බහුවචනය භාවිතා කරන්න: "අපි", "අපට", "අප" යනාදිය.
3. විශ්ලේෂණාත්මක භාෂාවක් භාවිතා කරන්න: "සැලකිය යුතුය", "සමස්තයක් ලෙස", "විශේෂයෙන්ම", "එබැවින්" වැනි.
4. මුල් ලිපියේ කරුණු මත පදනම්ව දෘෂ්ටිකෝණයක් ප්‍රකාශ කරන්න.
5. "මෙම සියලු කරුණු සමස්තයක් ලෙස සලකන කල, ඉදිරි ක්‍රියාමාර්ග පිළිබඳව සාකච්ඡා කළ යුතු කාලය එළඹ ඇත" යන වදනින් ලිපිය අවසන් කරන්න.
6. මුල් ලිපියේ ඇති සියලුම කරුණු නිවැරදිව තබා ගන්න - කිසිදු නව තොරතුරක් එකතු නොකරන්න.
7. නැවත ලියූ ලිපියේ දිග මුල් ලිපියේ දිගට ආසන්න විය යුතුය.
8. අකුරු වැරදි නොකරන්න. සියලු වචන නිවැරදිව අක්ෂර වින්‍යාසයෙන් ලියන්න.
9. වාක්‍ය සම්පූර්ණ විය යුතුය - අතරමගදී කපා නොදමන්න.

Article:
{content}
""",

    "style_3_sports": """
ඔබ සිංහල ක්‍රීඩා පුවත්පත් කලාවේ විශේෂඥයෙකි.

පහත නීති අනිවාර්යයෙන්ම අනුගමනය කරන්න:
1. වඩාත්ම නාටකීය කරුණ මුලින්ම ගෙන එන්න - එය ලිපියේ පළමු වාක්‍යය විය යුතුය.
2. ඉහළ ශක්තියෙන් යුත් භාෂාව භාවිතා කරන්න. උදාහරණ: "ක්‍රීඩාංගණය උණුසුම් වාතාවරණයක් ගෙන ආවා!", "තීරණාත්මක ජයග්රහණයක්!", "විශිෂ්ට දක්ෂතාවයක්!"
3. ක්‍රියාකාරී ක්‍රියා පද භාවිතා කරන්න: "පහර දුන්නා", "ජයග්‍රහණය කළා", "ප්‍රහාරය එල්ල කළා", "තක්කඩි ලෙස පැන ගත්තා" වැනි.
4. වේගවත් රිද්මයක් සහ උද්වේගකර භාෂාවක් පවත්වා ගන්න.
5. මුල් ලිපියේ ඇති සියලුම කරුණු, නම්, ස්ථාන, සංඛ්‍යා නිවැරදිව තබා ගන්න. කිසිවක් වෙනස් නොකරන්න.
6. උද්ගාමී බව පෙන්වීමට උපරිම වශයෙන් එක් විස්මයන් සලකුණක් ( ! ) පමණක් භාවිතා කරන්න. එකට වඩා භාවිතා නොකරන්න.
7. මුල් ලිපියේ ඉංග්‍රීසි වචන නොතිබුනේ නම්, ඉංග්‍රීසි වචන එකතු නොකරන්න.
8. නැවත ලියූ ලිපියේ දිග මුල් ලිපියේ දිගට ආසන්න විය යුතුය.
9. අකුරු වැරදි නොකරන්න. සියලු වචන නිවැරදිව අක්ෂර වින්‍යාසයෙන් ලියන්න.
10. වාක්‍ය සම්පූර්ණ විය යුතුය - අතරමගදී කපා නොදමන්න.

Article:
{content}
""",

    "style_4_youth": """
ඔබ තරුණ පාඨකයන් සඳහා ලියන සිංහල ලේඛකයෙකි.

පහත නීති අනිවාර්යයෙන්ම අනුගමනය කරන්න:
1. සෘජු ආමන්ත්‍රණයකින් ලිපිය ආරම්භ කරන්න: "දන්නවද?" හෝ "ඇහුවද?" හෝ "මේක අහන්න!"
2. අවිධිමත් සිංහල භාෂාවක් භාවිතා කරන්න: "ගොඩක්", "ටිකක්", "හිතෙනවා", "නෙවෙයි", "ඉන්නවා" වැනි.
3. කෙටි, තියුණු වාක්‍ය භාවිතා කරන්න. එක් වාක්‍යයක අදහස් 2-3කට වඩා නොදමන්න.
4. "ඒ නිසා යාලුවනේ, මේ ගැන අනිවාර්යයෙන්ම දැනගන්න!" යන වදනින් ලිපිය අවසන් කරන්න.
5. මුල් ලිපියේ ඇති සියලුම කරුණු, නම්, ස්ථාන, සංඛ්‍යා නිවැරදිව තබා ගන්න.
6. නැවත ලියූ ලිපියේ දිග මුල් ලිපියේ දිගට ආසන්න විය යුතුය - දැඩි ලෙස කෙටි නොකරන්න.
7. මුල් ලිපියේ ඉංග්‍රීසි වචන නොතිබුනේ නම්, ඉංග්‍රීසි වචන එකතු නොකරන්න.
8. අකුරු වැරදි නොකරන්න. සියලු වචන නිවැරදිව අක්ෂර වින්‍යාසයෙන් ලියන්න.
9. වාක්‍ය සම්පූර්ණ විය යුතුය - අතරමගදී කපා නොදමන්න.

Article:
{content}
""",

    "style_5_feature": """
ඔබ සිංහල විශේෂාංග/කතාන්දර ලේඛන විශේෂඥයෙකි.

පහත නීති අනිවාර්යයෙන්ම අනුගමනය කරන්න:
1. කතාන්දර ආරම්භයකින් ලිපිය ආරම්භ කරන්න. උදා: "එක් දිනක, මෙම සිදුවීම සිදු විය" හෝ "අඳුරු උදෑසනක, ..." හෝ "සුළඟ හමා යන විට, ..."
2. දෘශ්‍ය විස්තර භාවිතා කරන්න - ස්ථානය, කාලගුණය, වාතාවරණය පිළිබඳ විස්තර එකතු කරන්න.
3. මානුෂීය දෘෂ්ටිකෝණයක් ගෙන එන්න - සිදුවීම මිනිසුන්ට බලපෑ ආකාරය විස්තර කරන්න.
4. හැකි සෑම තැනකම වර්තමාන කාල ක්‍රියාපද භාවිතා කරන්න.
5. "මෙම කතාව අනාගත පරම්පරාවට ද ආදර්ශයක් වනු ඇත" යන වදනින් ලිපිය අවසන් කරන්න.
6. මුල් ලිපියේ ඇති සියලුම කරුණු, නම්, ස්ථාන, සංඛ්‍යා නිවැරදිව තබා ගන්න.
7. කිසිදු නව තොරතුරක් එකතු නොකරන්න.
8. නැවත ලියූ ලිපියේ දිග මුල් ලිපියේ දිගට ආසන්න විය යුතුය.
9. අකුරු වැරදි නොකරන්න. සියලු වචන නිවැරදිව අක්ෂර වින්‍යාසයෙන් ලියන්න.
10. වාක්‍ය සම්පූර්ණ විය යුතුය - අතරමගදී කපා නොදමන්න.

Article:
{content}
""",
}
ALL_STYLES = list(STYLE_INSTRUCTIONS.keys())


def load_api_key():
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError("NVIDIA_API_KEY not found in environment.")
    return api_key


def load_processed_pairs(output_path: Path) -> set:
    """Returns a set of (url, style) tuples already written to output."""
    processed = set()
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if "url" in data and "style" in data:
                        processed.add((data["url"], data["style"]))
                except Exception:
                    continue
    return processed


def build_prompt(style: str, content: str) -> str:
    template = STYLE_INSTRUCTIONS[style]
    return template.format(content=content)


class RateLimiter:
    """Shared token-bucket limiter so ALL worker threads respect one
    global requests-per-minute budget, instead of each thread backing
    off independently (which still overshoots the API's real limit)."""

    def __init__(self, rpm: int):
        self.min_interval = 60.0 / max(rpm, 1)
        self.lock = threading.Lock()
        self.next_allowed = 0.0
        # if the API tells us to back off globally (e.g. after a 429),
        # every thread checks this before its next request.
        self.pause_until = 0.0

    def wait(self):
        while True:
            with self.lock:
                now = time.monotonic()
                wait_for = max(self.next_allowed, self.pause_until) - now
                if wait_for <= 0:
                    self.next_allowed = now + self.min_interval
                    return
            time.sleep(wait_for)

    def report_429(self, retry_after: float):
        with self.lock:
            self.pause_until = max(self.pause_until, time.monotonic() + retry_after)


def worker(api_key, input_queue, output_file, lock, pbar, model_name, rate_limiter):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }

    while True:
        try:
            record, style = input_queue.get(timeout=5)
        except Empty:
            break

        try:
            content = clean_content(record.get("content", ""))
            if not content:
                with lock:
                    with open(output_file, "a", encoding="utf-8") as out_f:
                        error_record = record.copy()
                        error_record.pop("title", None)
                        error_record["style"] = style
                        error_record["error"] = "empty content"
                        out_f.write(json.dumps(error_record, ensure_ascii=False) + "\n")
                        out_f.flush()
                pbar.update(1)
                continue

            prompt = build_prompt(style, content)

            # Give the model enough headroom to reproduce a rewrite that's
            # roughly the same length as the source article.
            approx_len_tokens = max(512, int(len(content.split()) * 2.2))
            max_tokens = min(approx_len_tokens, 4096)

            payload = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": max_tokens,
                "temperature": 0.4,
                "top_p": 0.9,
                "stream": False,
            }

            success = False
            retries = 0

            while not success and retries < 5:
                rate_limiter.wait()
                try:
                    response = requests.post(
                        INVOKE_URL,
                        headers=headers,
                        json=payload,
                        timeout=180,
                    )

                    if response.status_code == 200:
                        res_json = response.json()

                        rewritten = ""
                        try:
                            message = res_json["choices"][0]["message"]
                            rewritten = message.get("content")
                            if rewritten is None:
                                rewritten = ""
                            rewritten = rewritten.strip()
                        except Exception:
                            rewritten = str(res_json)

                        result = record.copy()
                        result.pop("title", None)
                        result["content"] = content  # cleaned version, not raw scrape
                        result["style"] = style
                        result["rewritten_text"] = rewritten

                        with lock:
                            with open(output_file, "a", encoding="utf-8") as out_f:
                                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                                out_f.flush()
                                os.fsync(out_f.fileno())

                        success = True

                    else:
                        retry_after = None
                        if response.status_code == 429:
                            # Prefer the server's own guidance if it gives one.
                            header_val = response.headers.get("Retry-After")
                            if header_val:
                                try:
                                    retry_after = float(header_val)
                                except ValueError:
                                    retry_after = None
                            if retry_after is None:
                                try:
                                    body = response.json()
                                    retry_after = float(body.get("retryDelay", 0)) or None
                                except Exception:
                                    retry_after = None
                            retry_after = retry_after or (10 * (retries + 1))
                            print(f"\n[429] Backing off ALL threads for {retry_after:.1f}s")
                            rate_limiter.report_429(retry_after)
                        else:
                            print(f"\n[Error {response.status_code}] {response.text[:500]}")

                        retries += 1
                        time.sleep((2 ** retries) + 2)

                except Exception as e:
                    print(f"\n[Exception] {repr(e)}")
                    retries += 1
                    time.sleep(3)

            if not success:
                with lock:
                    with open(output_file, "a", encoding="utf-8") as out_f:
                        error_record = record.copy()
                        error_record.pop("title", None)
                        error_record["style"] = style
                        error_record["status"] = "failed"
                        error_record["error"] = f"failed after {retries} retries"
                        out_f.write(json.dumps(error_record, ensure_ascii=False) + "\n")
                        out_f.flush()
                        os.fsync(out_f.fileno())

        finally:
            input_queue.task_done()
            pbar.update(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="/home/jovyan/style_rewriter/data/train1.jsonl")
    parser.add_argument("--output", default="/home/jovyan/style_rewriter/data/style_dataset2.jsonl")
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--styles",
        default=",".join(ALL_STYLES),
        help=f"Comma-separated list of styles to generate. Choices: {', '.join(ALL_STYLES)}",
    )
    parser.add_argument(
        "--rpm",
        type=int,
        default=15,
        help="Global requests-per-minute budget shared across ALL threads. "
             "Check your NVIDIA API tier's actual limit and set this below it. "
             "Lowered default to 15 since 30 was still triggering 429s.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N articles (file order, no shuffling). "
             "Use --sample instead if you want a randomized subset.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Randomly sample N articles (seeded, reproducible) instead of "
             "processing the whole corpus. Recommended over --limit for a "
             "representative style-rewriter training subset. Ignored if "
             "--limit is also set.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for --sample, so the same subset is picked every run.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.touch(exist_ok=True)
    print(f"Saving to: {output_path.resolve()}")

    if not input_path.exists():
        print(f"Input file {input_path} not found.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    requested_styles = [s.strip() for s in args.styles.split(",") if s.strip()]
    invalid = [s for s in requested_styles if s not in STYLE_INSTRUCTIONS]
    if invalid:
        print(f"Unknown style(s): {invalid}. Valid choices: {ALL_STYLES}")
        return

    api_key = load_api_key()
    processed_pairs = load_processed_pairs(output_path)

    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    if args.limit is not None:
        records = records[: args.limit]
    elif args.sample is not None:
        rng = random.Random(args.seed)
        if args.sample < len(records):
            records = rng.sample(records, args.sample)
        print(f"Sampled {len(records)} articles (seed={args.seed})")

    # Build the full (record, style) work list, skipping anything already done.
    to_process = []
    for r in records:
        url = r.get("url")
        for style in requested_styles:
            if (url, style) not in processed_pairs:
                to_process.append((r, style))

    print(f"Model: {args.model}")
    print(f"Styles: {requested_styles}")
    print(f"Total articles: {len(records)}")
    print(f"Total (article, style) pairs: {len(records) * len(requested_styles)}")
    print(f"Already processed: {len(processed_pairs)}")
    print(f"Remaining: {len(to_process)}")
    print(f"Concurrency: {args.concurrency}")
    print(f"Rate limit: {args.rpm} requests/minute (shared across all threads)")

    est_minutes = len(to_process) / args.rpm
    print(f"Estimated time at this rate: {est_minutes:.0f} min (~{est_minutes / 60:.1f} hours)")

    if not to_process:
        print("Everything already processed.")
        return

    input_queue = Queue()
    for item in to_process:
        input_queue.put(item)

    lock = threading.Lock()
    rate_limiter = RateLimiter(args.rpm)

    with tqdm(total=len(to_process), desc="Rewriting") as pbar:
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            for _ in range(args.concurrency):
                executor.submit(worker, api_key, input_queue, output_path, lock, pbar, args.model, rate_limiter)

            input_queue.join()

    print(f"\nComplete. Results: {output_path}")


if __name__ == "__main__":
    main()