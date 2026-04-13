"""
feed_listener.py
----------------
Lean Truth Social only listener.

Single source, single queue, minimal API cost.
Trump posts maybe 20 times/day — ~$0.10-0.30/day in Claude API costs.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    export ALPACA_API_KEY=PK...
    export ALPACA_SECRET_KEY=...
    python feed_listener.py

Requirements:
    pip install playwright anthropic yfinance aiohttp alpaca-py
    playwright install chromium
"""

import asyncio
import logging
import json
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from pathlib import Path
from playwright.async_api import async_playwright, Page, BrowserContext

from market_poller import MarketSnapshot, poll_market
from classifier import Classifier
from order_executor import OrderExecutor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

TARGET_URL      = "https://truthsocial.com/@realDonaldTrump"
POLL_INTERVAL   = 15
AUTH_STATE_FILE = "auth_state.json"
SEEN_IDS_FILE   = "seen_ids.json"
MAX_POSTS_CHECK = 5


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class Post:
    id: str
    text: str
    url: str
    scraped_utc: str
    source: str = "truth_social"
    headline: str = ""

    def to_dict(self):
        return asdict(self)

    def display_text(self):
        return self.text[:200]

    def __str__(self):
        return (
            f"\n{'='*64}\n"
            f"[🇺🇸 TRUTH SOCIAL] {self.scraped_utc}\n"
            f"URL: {self.url}\n"
            f"TEXT: {self.text[:400]}\n"
            f"{'='*64}"
        )


# ── Persistence ───────────────────────────────────────────────────────────────

def load_seen_ids() -> set:
    try:
        with open(SEEN_IDS_FILE, "r") as f:
            return set(json.load(f))
    except (FileNotFoundError, json.JSONDecodeError):
        return set()


def save_seen_ids(seen: set):
    with open(SEEN_IDS_FILE, "w") as f:
        json.dump(list(seen), f)


# ── Auth ──────────────────────────────────────────────────────────────────────

async def get_auth_context(playwright) -> BrowserContext:
    browser_type = playwright.chromium
    if Path(AUTH_STATE_FILE).exists():
        log.info("Auth state found — launching headless browser.")
        browser = await browser_type.launch(headless=True)
        context = await browser.new_context(storage_state=AUTH_STATE_FILE)
    else:
        log.info("No auth state — opening browser for manual login.")
        browser = await browser_type.launch(headless=False)
        context = await browser.new_context()
        page    = await context.new_page()
        await page.goto("https://truthsocial.com/login")
        await page.wait_for_url(
            lambda url: "/login" not in url and "/signup" not in url,
            timeout=120_000
        )
        log.info("Login detected — saving session.")
        await context.storage_state(path=AUTH_STATE_FILE)
        await page.close()
    return context


# ── Scraping ──────────────────────────────────────────────────────────────────

async def scrape_posts(page: Page) -> list[Post]:
    try:
        await page.goto(
            TARGET_URL, wait_until="domcontentloaded", timeout=45_000
        )
        await page.wait_for_selector('[data-testid="status"]', timeout=30_000)
        await page.evaluate("window.scrollTo(0, 500)")
        await page.wait_for_timeout(1000)

        elements = await page.query_selector_all('[data-testid="status"]')
        elements = elements[:MAX_POSTS_CHECK]
        now = datetime.now(timezone.utc).isoformat()
        posts = []

        for el in elements:
            post_id = await el.get_attribute("data-id")
            if not post_id:
                link = await el.query_selector("a[href*='/posts/']")
                if link:
                    href = await link.get_attribute("href") or ""
                    post_id = href.split("/posts/")[-1] if "/posts/" in href else None
            if not post_id:
                continue

            markup = await el.query_selector('[data-testid="markup"]')
            if not markup:
                continue
            text = (await markup.inner_text()).strip()
            if not text:
                continue

            link = await el.query_selector("a[href*='/posts/']")
            url = ""
            if link:
                href = await link.get_attribute("href") or ""
                url = f"https://truthsocial.com{href}" if href.startswith("/") else href

            posts.append(Post(id=post_id, text=text, url=url, scraped_utc=now))

        return posts

    except Exception as e:
        log.error(f"Scrape error: {e}")
        return []


# ── Feed polling ──────────────────────────────────────────────────────────────

async def poll_feed(queue: asyncio.Queue, context: BrowserContext, seen_ids: set):
    log.info(f"Truth Social listener started — polling every {POLL_INTERVAL}s")
    page = await context.new_page()
    await page.route(
        "**/*.{png,jpg,jpeg,gif,webp,svg,woff,woff2,ttf}",
        lambda route: route.abort()
    )
    while True:
        try:
            posts = await scrape_posts(page)
            new   = [p for p in posts if p.id not in seen_ids]
            if new:
                log.info(f"🚨 {len(new)} new post(s) from Truth Social")
                for p in new:
                    log.info(str(p))
                    await queue.put(p)
                    seen_ids.add(p.id)
                save_seen_ids(seen_ids)
            else:
                log.debug("No new posts.")
        except Exception as e:
            log.error(f"Poll error: {e}", exc_info=True)
        await asyncio.sleep(POLL_INTERVAL)


# ── Consumer ──────────────────────────────────────────────────────────────────

async def consume_posts(queue: asyncio.Queue, market: MarketSnapshot):
    classifier = Classifier()
    executor   = OrderExecutor()
    loop       = asyncio.get_event_loop()
    log.info("Consumer ready — waiting for Truth Social posts.")

    while True:
        post = await queue.get()

        log.info(f"[CONSUMER] Classifying: {post.display_text()[:80]}")

        if market.ready():
            log.info(f"[CONSUMER] Market context:\n{market.summary()}")
        else:
            log.warning("[CONSUMER] Market data not yet available.")

        # Fetch open positions for classifier context
        open_positions = await loop.run_in_executor(
            None, executor.get_open_positions
        )

        signal = await classifier.classify(post, market, open_positions)

        if signal:
            log.info(signal.log_summary())

            # Handle position management actions first
            if open_positions and signal.position_actions:
                log.info("[CONSUMER] 🔄 Processing position actions...")
                summaries = await executor.handle_position_actions(
                    signal, open_positions
                )
                for s in summaries:
                    log.info(f"[CONSUMER] {s}")

            # Fire new orders
            if signal.should_trade():
                log.info("[CONSUMER] 🟢 Firing orders...")
                orders = await executor.execute(signal)
                log.info(f"[CONSUMER] {len(orders)} order(s) submitted.")
            else:
                log.info("[CONSUMER] ⚪ No trade.")

        queue.task_done()


# ── Entry point ───────────────────────────────────────────────────────────────

async def main():
    queue    = asyncio.Queue()
    market   = MarketSnapshot()
    seen_ids = load_seen_ids()

    log.info(f"Sentimate Trader starting — {len(seen_ids)} seen IDs loaded.")
    log.info("Source: Truth Social only | Cost: ~$0.10-0.30/day")

    async with async_playwright() as playwright:
        context = await get_auth_context(playwright)
        try:
            await asyncio.gather(
                poll_market(market),
                poll_feed(queue, context, seen_ids),
                consume_posts(queue, market),
            )
        finally:
            await context.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("Stopped.")
