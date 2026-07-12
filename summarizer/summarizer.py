import json, time, requests
from pathlib import Path
from datetime import datetime

# ── CONFIG ────────────────────────────────────────────────
OPENROUTER_API_KEY = "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"
MODEL              = "openrouter/free"   # Auto-routes to any available free model

DELAY_SEC       = 1    # seconds between requests
MAX_RETRIES     = 4    # retries on failure
RETRY_DELAY_SEC = 20   # wait between retries

# ── ARGS ──────────────────────────────────────────────────
class args:
    input  = "cleaned_dataset.jsonl"
    output = "summarization_dataset.jsonl"
    start  = 1882
    end    = 3000   # 0 = process all rows

# ── HELPERS ───────────────────────────────────────────────
def log(msg, tag=""):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"{ts}  {'['+tag+'] ' if tag else ''}{msg}", flush=True)

def build_prompt(title, content):
    target_words = max(15, round(len(content) / 40))
    return (
        "ඔබ දක්ෂ සිංහල ලිපිකරුවෙකි. පහත ලිපිය කියවා, නිවැරදි හා "
        "සම්පූර්ණ සිංහල සාරාංශයක් ලියන්න.\n\n"
        "නීතිරීති:\n"
        "- සාරාංශය සිංහල භාෂාවෙන් පමණක් ලියන්න\n"
        "- ලිපිය ඇතුළත ඇති ප්‍රධාන කරුණු ආවරණය කරන්න\n"
        f"- සාරාංශ දිග: ආසන්න වශයෙන් {target_words} වචන (ලිපි දිගෙහි 10% පමණ)\n"
        "- සාරාංශය පමණක් ලියන්න\n\n"
        f"ශීර්ෂය: {title}\n\n"
        f"ලිපිය:\n{content}"
    )

# ── API CALL ──────────────────────────────────────────────
def generate_summary(title, content):
    prompt  = build_prompt(title, content)
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type":  "application/json",
    }
    payload = {
        "model":       MODEL,
        "messages":    [{"role": "user", "content": prompt}],
        "temperature": 0.3,
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=90,
            )
            response.raise_for_status()
            data = response.json()

            # Grab which model was actually used (router tells us via response)
            used_model = data.get("model", MODEL)
            summary    = data["choices"][0]["message"]["content"].strip()

            if not summary:
                raise ValueError("Empty response from API")
            return summary, used_model

        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else None
            if status is None and "429" in str(e):
                status = 429

            log(f"HTTP {status or '?'} on attempt {attempt}/{MAX_RETRIES}: {e}", "ERR")

            if attempt < MAX_RETRIES:
                wait = RETRY_DELAY_SEC * attempt   # 20s, 40s, 60s
                log(f"Waiting {wait}s before retry...", "WARN")
                time.sleep(wait)
            else:
                raise

        except Exception as e:
            log(f"Attempt {attempt}/{MAX_RETRIES} failed: {e}", "ERR")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_SEC)
            else:
                raise

# ── MAIN ──────────────────────────────────────────────────
def main():
    input_path  = Path(args.input)
    output_path = Path(args.output)

    log(f"Loading {input_path}...")
    rows, bad = [], 0
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                bad += 1
    log(f"Loaded {len(rows):,} rows ({bad} bad lines skipped)")

    start_idx = max(0, args.start - 1)
    end_idx   = min(args.end if args.end > 0 else len(rows), len(rows))
    target    = rows[start_idx:end_idx]

    log(f"Processing rows {start_idx+1} to {end_idx}  ({len(target):,} entries)")
    log(f"Model: {MODEL}  (auto-routes to available free models)")

    done_count  = 0
    error_count = 0
    model_stats = {}   # track which underlying models get used

    with open(output_path, "a", encoding="utf-8") as out_fh:
        for idx, row in enumerate(target, start=start_idx + 1):
            url     = row.get("url", f"row_{idx}")
            title   = row.get("title", "")
            content = (row.get("content", "") or "").strip()

            log(f"[{idx}/{end_idx}] {title[:60]}...")

            try:
                summary, used_model = generate_summary(title, content)

                out = {
                    "title":    title,
                    "content":  content,
                    "summary":  summary,
                    "category": row.get("category", ""),
                    "url":      url,
                    "model":    used_model,
                }
                out_fh.write(json.dumps(out, ensure_ascii=False) + "\n")
                out_fh.flush()

                done_count += 1
                model_stats[used_model] = model_stats.get(used_model, 0) + 1
                ratio      = len(summary) / max(len(content), 1) * 100
                short_name = used_model.split("/")[-1][:25]
                log(f"  done [{short_name}] {ratio:.0f}% | {summary[:60].replace(chr(10), ' ')}...")

            except Exception as e:
                error_count += 1
                log(f"  error: {e}", "ERR")
                out_fh.write(json.dumps({
                    "title": title, "content": content, "summary": "",
                    "category": row.get("category", ""), "url": url, "model": "",
                }, ensure_ascii=False) + "\n")
                out_fh.flush()

            time.sleep(DELAY_SEC)

    log("=" * 60)
    log(f"Done!  {done_count:,} summaries written to {output_path}")
    log(f"Errors: {error_count}")
    if model_stats:
        log("Underlying models used by router:")
        for m, count in sorted(model_stats.items(), key=lambda x: -x[1]):
            log(f"  {m}: {count} rows")

if __name__ == "__main__":
    main()