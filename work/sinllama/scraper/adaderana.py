import asyncio
import json
import re
import os
import signal
import datetime
from bs4 import BeautifulSoup
from crawl4ai import AsyncWebCrawler
from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig

# =========================
# CONFIG
# =========================
SOURCE     = "sinhala_adaderana"
BASE_URL   = "https://sinhala.adaderana.lk/news"

START_ID   = 222116
END_ID     = 230167

OUTPUT_DIR    = "output"
PROGRESS_FILE = "sinhala_scrape_progress.json"

# ── VPS-SAFE crawl settings ──────────────────────────────────────────────────
PAGE_TIMEOUT       = 180_000   # 3 min
MAX_RETRY          = 4
RETRY_BASE_DELAY   = 5         # seconds; doubles each attempt (exponential back-off)
BETWEEN_ARTICLES   = 2         # seconds between article fetches
BROWSER_RESTART_N  = 50        # restart browser every N articles
# ─────────────────────────────────────────────────────────────────────────────

# =========================
# CATEGORY KEYWORDS
# =========================
POLITICS_STRONG = [
"අමාත්‍ය",
"අමාත්‍යවරයා",
"අමාත්‍යංශ",
"ජනාධිපති",
"අගමැති",
"පාර්ලිමේන්තුව",
"මන්ත්‍රී",
"මන්ත්‍රීවරයා",
"රජය",
"ආණ්ඩුව",
"විරෝධී පක්ෂය",
"පක්ෂය",
"ජනපති",
"පාලනය",
"රජයේ",
"මැතිවරණය",
"ඡන්දය",
"පාර්ලිමේන්තු",
"ආණ්ඩුක්‍රම ව්‍යවස්ථාව",
"රනිල්",
"ප්‍රා‍දේශීය සභා",
"පක්ෂයේ",
"මහලේකම්",
"දේශපාලනය",
"දේශපාලනයට",
"මැතිවරණ",
"ආණ්ඩු",
"මහින්ද රාජපක්ෂ",
"ජනාධිපති"
]

POLITICS_WEAK = [
"තීරණය",
"ප්‍රකාශ",
"චෝදනා",
"නියෝග",
"සම්මත",
"රැස්වීම",
"පත්",
"පාලන",
"විරෝධතා"
]

SPORTS_STRONG = [
"ක්‍රිකට්","T20","ODI","ටෙස්ට්",
"කඩුලු","කඩුල්ල","පන්දු",
"ශතකය","අර්ධ ශතකය","ඩබල් ශතකය",
"ඉනිම","විකට්",
"ලකුණු",
"කුසලානය","ලෝක කුසලානය",
"එක්දින","ක්‍රීඩක",
"ක්‍රීඩිකාවන්",
"ක්‍රීඩා",
"ක්‍රීඩාව",
"තරග",
"තරගාවලිය",
"කණ්ඩායම",
"ජාතික කණ්ඩායම",
"ලීග්",
"තරගාවලිය",
"නෙට්බෝල්",
"පාපන්දු",
"රග්බි",
"වොලිබෝල්",
"බැඩ්මින්ටන්",
"ටෙනිස්",
"හොකී",
"ඇතිලට්",
"මැරතන්",
]

SPORTS_WEAK = [
"තරගය","තරග","තරගාවලිය",
"ජය","ජයක්","පරාජය","ජයග්‍රහණය"
]

INTERNATIONAL_KEYWORDS = [
   "අමෙරිකා", "අමෙරිකාවේ", "අමෙරිකාව", "අමෙරිකානු",

"බ්‍රිතාන්‍ය", "බ්‍රිතාන්‍යයේ", "බ්‍රිතාන්‍යය",

"රුසියා", "රුසියාවේ", "රුසියානු",

"චීනය", "චීනයේ", "චීන",

"ඉන්දියාව", "ඉන්දියාවේ", "ඉන්දීය",

"ජපානය", "ජපානයේ", "ජපන්",

"කැනඩාව", "කැනඩාවේ", "කැනේඩියානු",

"ඕස්ට්‍රේලියාව", "ඕස්ට්‍රේලියාවේ", "ඕස්ට්‍රේලියානු",

"ජර්මනිය", "ජර්මනියේ", "ජර්මානු",

"ප්‍රංශය", "ප්‍රංශයේ", "ප්‍රංශ",

"ඉතාලිය", "ඉතාලියේ", "ඉතාලි",

"ස්පාඤ්ඤය", "ස්පාඤ්ඤයේ",

"නෙදර්ලන්තය", "නෙදර්ලන්තයේ",

"ස්විට්සර්ලන්තය", "ස්විට්සර්ලන්තයේ",

"නෝර්වේ", "නෝර්වේයේ",

"ස්වීඩනය", "ස්වීඩනයේ",

"ඩෙන්මාර්කය", "ඩෙන්මාර්කයේ",

"ෆින්ලන්තය", "ෆින්ලන්තයේ",

"තුර්කිය", "තුර්කියේ", "තුර්කි",

"ඉරානය", "ඉරානයේ", "ඉරාන",

"ඉරාකය", "ඉරාකයේ",

"සෞදි", "සෞදි අරාබිය", "සෞදි අරාබියේ",

"ඊශ්‍රායලය", "ඊශ්‍රායලයේ",

"පලස්තීනය", "පලස්තීනයේ",

"ඇෆ්ගනිස්ථානය", "ඇෆ්ගනිස්ථානයේ", "ඇෆ්ගන්",

"පාකිස්ථානය", "පාකිස්ථානයේ", "පාකිස්ථාන",

"බංග්ලාදේශය", "බංග්ලාදේශයේ",

"නේපාලය", "නේපාලයේ",

"භූතානය", "භූතානයේ",

"මියන්මාරය", "මියන්මාරයේ",

"තායිලන්තය", "තායිලන්තයේ",

"වියට්නාමය", "වියට්නාමයේ",

"දකුණු කොරියාව", "දකුණු කොරියාවේ",

"උතුරු කොරියාව", "උතුරු කොරියාවේ",

"සිංගප්පූරුව", "සිංගප්පූරුවේ",

"මැලේසියාව", "මැලේසියාවේ",

"ඉන්දුනීසියාව", "ඉන්දුනීසියාවේ",

"පිලිපීනය", "පිලිපීනයේ",

"මෙක්සිකෝව", "මෙක්සිකෝවේ",

"බ්‍රසීලය", "බ්‍රසීලයේ",

"ආර්ජෙන්ටිනාව", "ආර්ජෙන්ටිනාවේ",

"චිලී", "චිලියේ",

"දකුණු අප්‍රිකාව", "දකුණු අප්‍රිකාවේ",

"නයිජීරියාව", "නයිජීරියාවේ",

"ඊජිප්තුව", "ඊජිප්තුවේ",

"ලිබියාව", "ලිබියාවේ",

"යුක්රේනය", "යුක්රේනයේ",

"පෝලන්තය", "පෝලන්තයේ",

"හංගේරියාව", "හංගේරියාවේ",

"චෙක්", "චෙක් ජනරජය", "චෙක් ජනරජයේ",
]

# Output file names
OUTPUT_FILES = {
    "Sports":        "sinhala_sports.jsonl",
    "International": "sinhala_international.jsonl",
    "Politics":      "sinhala_politics.jsonl",
    "General":       "sinhala_general.jsonl",
}
# Graceful shutdown flag
_shutdown = False

def _handle_signal(sig, frame):
    global _shutdown
    print("\n⚠️  Interrupt received — will stop after the current article and save progress.")
    _shutdown = True

signal.signal(signal.SIGINT,  _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


def normalize(text: str) -> str:
    if not text:
        return ""

    # keep Sinhala + English + numbers only
    text = re.sub(r"[^\u0D80-\u0DFFa-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()


def count_matches(text, keywords):

    text = normalize(text)
    count = 0

    for kw in keywords:
        kw = normalize(kw)

        # Sinhala safe boundary match
        if f" {kw} " in f" {text} ":
            count += 1

    return count
# =========================
# CATEGORIZATION
# =========================
def categorize(headline: str, content: str) -> str:

    h = normalize(headline)
    c = normalize(content)

    # ---- SPORTS SCORE ----
    sports_score = (
        2 * count_matches(h, SPORTS_STRONG) +
        1 * count_matches(c, SPORTS_STRONG) +
        0.5 * count_matches(h, SPORTS_WEAK)
    )

    # ---- INTERNATIONAL SCORE ----
    intl_score = (
        2 * count_matches(h, INTERNATIONAL_KEYWORDS) +
        1 * count_matches(c, INTERNATIONAL_KEYWORDS)
    )

    # ---- POLITICS SCORE ----
    politics_score = (
        2 * count_matches(h, POLITICS_STRONG) +
        1 * count_matches(c, POLITICS_STRONG) +
        0.5 * count_matches(h, POLITICS_WEAK)
    )

    # ---- DECISION PRIORITY ----
    if sports_score >= 2:
        return "Sports"

    if intl_score >= 2:
        return "International"

    if politics_score >= 2:
        return "Politics"

    return "General"
# =========================
# PROGRESS TRACKING
# =========================
def load_progress() -> dict:
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "last_id":       START_ID - 1,
        "total_articles": 0,
        "counts": {"Sports": 0, "International": 0, "General": 0},
    }

def save_progress(progress: dict):
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)

def save_article(article_data: dict, category: str):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filepath = os.path.join(OUTPUT_DIR, OUTPUT_FILES[category])
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(article_data, ensure_ascii=False) + "\n")


# =========================
# HELPERS
# =========================
def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\xa0", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def extract_article(html: str):
    soup = BeautifulSoup(html, "lxml")

    # 🔹 HEADLINE
    # Try h1.news-heading first (standard)
    headline = ""
    h1 = soup.find("h1", class_="news-heading")
    if h1:
        headline = clean_text(h1.get_text())
    
    if not headline:
        h1 = soup.find("h1")
        headline = clean_text(h1.get_text()) if h1 else ""

    # 🔹 DATE
    date = ""
    date_elem = soup.find("p", class_="news-datestamp")
    if date_elem:
        date = clean_text(date_elem.get_text())

    # 🔹 ARTICLE BODY
    body = ""
    news_content = soup.find("div", class_="news-content")
    if news_content:
        # Get text from paragraphs to avoid scripts/ads
        paragraphs = news_content.find_all("p")
        if paragraphs:
            body = "\n".join([clean_text(p.get_text()) for p in paragraphs])
        else:
            body = clean_text(news_content.get_text(separator="\n"))
            
    if not body:
        # Fallback
        news_body = soup.find("div", class_="story-text") or soup.find("div", class_="news-body")
        if news_body:
            body = clean_text(news_body.get_text(separator="\n"))

    return headline, body, date

# =========================
# BROWSER FACTORY
# =========================
def make_browser_cfg() -> BrowserConfig:
    """VPS-hardened browser config."""
    return BrowserConfig(
        headless=True,
        text_mode=True,
        user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        extra_args=[
            "--no-sandbox",
            "--disable-setuid-sandbox",
            "--disable-dev-shm-usage",
            "--disable-gpu",
            "--disable-gpu-compositing",
            "--disable-software-rasterizer",
            "--disable-extensions",
            "--disable-background-networking",
            "--disable-default-apps",
            "--no-first-run",
        ],
    )

def make_run_cfg(timeout: int = PAGE_TIMEOUT) -> CrawlerRunConfig:
    return CrawlerRunConfig(
        exclude_external_links=True,
        page_timeout=timeout,
        wait_until="domcontentloaded",
    )


# =========================
# FETCH WITH RETRY
# =========================
_BROWSER_CRASH_PHRASES = (
    "browser has been closed",
    "browser context has been closed",
    "target page, context or browser has been closed",
    "browsercontext.new_page",
    "browser.new_context",
)

async def fetch_with_retry(
    crawler_holder: list,
    url: str,
    cfg: CrawlerRunConfig,
    new_crawler_fn,
    kill_crawler_fn,
    label: str = "",
):
    """
    Fetch `url` up to MAX_RETRY times with exponential back-off.
    On browser crash, restarts the browser before retrying.
    Returns the result on success, None on failure.
    """
    for attempt in range(1, MAX_RETRY + 1):
        if _shutdown:
            return None
        try:
            result = await crawler_holder[0].arun(url=url, config=cfg)
            return result
        except Exception as e:
            wait = RETRY_BASE_DELAY * (2 ** (attempt - 1))
            short_err = str(e).splitlines()[0][:120]
            is_crash = any(p in short_err.lower() for p in _BROWSER_CRASH_PHRASES)

            if attempt < MAX_RETRY:
                if is_crash:
                    print(f"   💥 [{label}] Browser crash — restarting …")
                    await kill_crawler_fn(crawler_holder[0])
                    await asyncio.sleep(3)
                    crawler_holder[0] = await new_crawler_fn()
                else:
                    print(f"   ⚠️  [{label}] Attempt {attempt}/{MAX_RETRY} failed — "
                          f"retrying in {wait}s. Error: {short_err}")
                    await asyncio.sleep(wait)
            else:
                print(f"   ❌  [{label}] All {MAX_RETRY} attempts failed — skipping. "
                      f"Error: {short_err}")
    return None


# =========================
# MAIN
# =========================
async def _new_crawler_async():
    crawler = AsyncWebCrawler(config=make_browser_cfg())
    await crawler.start()
    return crawler

async def kill_crawler(crawler):
    if crawler:
        await crawler.close()

async def main():
    global _shutdown

    # Load progress and initialize counters
    progress = load_progress()
    resume_id = progress.get("last_id", START_ID - 1) + 1
    total_articles = progress.get("total_articles", 0)
    counts = progress.get("counts", {"Sports": 0, "International": 0, "Politics": 0, "General": 0})
    articles_since_restart = 0

    # Crawler holder for restartable browser
    crawler_holder = [await _new_crawler_async()]
    article_cfg = make_run_cfg(timeout=PAGE_TIMEOUT)

    error_message = None  # To store any exception message
    try:
        for article_id in range(resume_id, END_ID + 1):
            if _shutdown:
                break

            url = f"{BASE_URL}/{article_id}"

            # Periodic browser restart to avoid memory leaks
            if articles_since_restart >= BROWSER_RESTART_N:
                print(f"🔄 Restarting browser (after {BROWSER_RESTART_N} articles) …")
                await kill_crawler(crawler_holder[0])
                await asyncio.sleep(3)
                crawler_holder[0] = await _new_crawler_async()
                articles_since_restart = 0
                article_cfg = make_run_cfg(timeout=PAGE_TIMEOUT)

            res = await fetch_with_retry(
                crawler_holder, url, article_cfg,
                new_crawler_fn=_new_crawler_async,
                kill_crawler_fn=kill_crawler,
                label=str(article_id),
            )

            # Update progress regardless (so we don't retry dead IDs)
            progress["last_id"] = article_id

            if not res or not res.html:
                await asyncio.sleep(BETWEEN_ARTICLES)
                continue

            headline, content, date = extract_article(res.html)

            # Skip 404s, redirects, or empty articles
            if not headline or not content or len(content.split()) < 10:
                await asyncio.sleep(BETWEEN_ARTICLES)
                continue

            category = categorize(headline, content)

            article_data = {
                "Source":       SOURCE,
                "Headline":     headline,
                "News Content": content,
                "URL":          url,
                "Category":     category,
                "Date":         date,
            }

            save_article(article_data, category)

            total_articles += 1
            articles_since_restart += 1
            counts[category] = counts.get(category, 0) + 1

            print(f"✅ [{article_id}] [{category}] {headline[:70]}")

            # Save progress after every article
            progress["total_articles"] = total_articles
            progress["counts"] = counts
            save_progress(progress)

            await asyncio.sleep(BETWEEN_ARTICLES)

    finally:
        await kill_crawler(crawler_holder[0])

        # Final progress save
        progress["total_articles"] = total_articles
        progress["counts"] = counts

        # Add stop reason, timestamp, and error if any
        if _shutdown:
            progress["stopped_reason"] = "Interrupted by user or signal"
        elif error_message:
            progress["stopped_reason"] = "Exception"
        else:
            progress["stopped_reason"] = "Completed range"
        progress["stopped_at"] = datetime.datetime.now().isoformat()
        if error_message:
            progress["error"] = error_message

        save_progress(progress)

        print()
        print(f"🎉 DONE")
        print(f"   Total articles   : {total_articles}")
        print(f"   Sports           : {counts.get('Sports', 0)}")
        print(f"   International    : {counts.get('International', 0)}")
        print(f"   Politics         : {counts.get('Politics', 0)}")
        print(f"   General          : {counts.get('General', 0)}")
        print(f"   Progress file    : {PROGRESS_FILE}")
        print(f"   Output dir       : {OUTPUT_DIR}/")
        print("💡 Run the script again to resume from where you left off!")


# =========================
if __name__ == "__main__":
    asyncio.run(main())
