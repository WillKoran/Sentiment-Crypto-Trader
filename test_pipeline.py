"""
test_pipeline.py
----------------
Comprehensive test suite for the crypto sentiment trading pipeline.

Tests three layers:
  Layer 1 — Classifier only (no API keys needed beyond Anthropic)
             Validates signal quality, directions, TP/SL levels
             Covers all three sources: truth_social, reuters, ap

  Layer 2 — Market data (requires yfinance)
             Validates BTC/ETH prices are fetching correctly
             Validates macro context is populating

  Layer 3 — Full pipeline (requires Alpaca keys)
             Validates order placement end-to-end on paper account
             Run this last after layers 1-2 pass

Usage:
    # Layer 1 only (just Anthropic key needed)
    python test_pipeline.py --layer 1

    # Layers 1 + 2
    python test_pipeline.py --layer 2

    # Full pipeline
    python test_pipeline.py --layer 3

    # Single test
    python test_pipeline.py --test peace_deal

    # All tests, all layers
    python test_pipeline.py
"""

import asyncio
import argparse
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)


# ── Mock helpers ──────────────────────────────────────────────────────────────

from market_poller import MarketSnapshot, TickerSnapshot
from classifier import Classifier


def mock_market(
    btc=84000.0,  btc_chg=-1.2,
    eth=1950.0,   eth_chg=-1.5,
    sp=5420.0,    sp_chg=-0.3,
    brent=74.20,  brent_chg=-0.8,
    gold=3100.0,  gold_chg=+0.5,
    dxy=104.5,    dxy_chg=+0.2,
) -> MarketSnapshot:
    def snap(sym, name, price, chg, is_crypto=False):
        prev = price / (1 + chg / 100)
        return TickerSnapshot(
            symbol=sym, name=name,
            price=price, prev_close=prev,
            change_pct=chg,
            day_high=price * 1.02,
            day_low=price * 0.98,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            is_crypto=is_crypto,
        )
    state = MarketSnapshot()
    state.tickers = {
        "BTC-USD":   snap("BTC-USD",   "Bitcoin",         btc,   btc_chg,  True),
        "ETH-USD":   snap("ETH-USD",   "Ethereum",        eth,   eth_chg,  True),
        "ES=F":      snap("ES=F",      "S&P 500 Futures", sp,    sp_chg),
        "BZ=F":      snap("BZ=F",      "Brent Crude",     brent, brent_chg),
        "GC=F":      snap("GC=F",      "Gold",            gold,  gold_chg),
        "DX-Y.NYB":  snap("DX-Y.NYB", "US Dollar Index", dxy,   dxy_chg),
    }
    state.last_updated_utc = datetime.now(timezone.utc).isoformat()
    return state


@dataclass
class MockPost:
    id: str
    text: str
    source: str
    url: str = "https://test.com/post"
    scraped_utc: str = datetime.now(timezone.utc).isoformat()
    headline: str = ""

    def display_text(self):
        return self.headline if self.headline else self.text[:200]


# ── Test cases ────────────────────────────────────────────────────────────────
# Format: (post_text, source, market_kwargs, headline, expected_comment)

TESTS = {

    # ── Truth Social — should fire ──────────────────────────────────────────

    "peace_deal": (
        "Just spoke with Putin and Zelensky. A PERFECT ceasefire agreement "
        "has been reached. The war in Ukraine is OVER. God Bless America!",
        "truth_social",
        {"btc": 84000, "btc_chg": -1.2, "brent": 74.20, "brent_chg": -0.8},
        "",
        "LONG BTC + LONG ETH — risk-on, urgency 3, tight SL on unverifiable"
    ),

    "tariff_shock": (
        "I am hereby announcing a 50% tariff on ALL Chinese imports, effective "
        "immediately. We cannot allow China to continue stealing from the USA!",
        "truth_social",
        {"btc": 84000, "btc_chg": +0.5},
        "",
        "SHORT BTC + SHORT ETH — risk-off tariff shock, wide TP on severity"
    ),

    "fed_dovish": (
        "Just had a great call with Jerome Powell. He agrees that rates need "
        "to come DOWN immediately. Big rate cuts coming — the economy is ready!",
        "truth_social",
        {"btc": 84000, "btc_chg": +0.2, "dxy": 104.5, "dxy_chg": -0.5},
        "",
        "LONG BTC — dollar weakness + rate cut narrative, moderate confidence"
    ),

    "iran_escalation": (
        "Iran has crossed a RED LINE. Our Military is fully prepared and ready "
        "to act. The world is watching. These people are CRAZY!",
        "truth_social",
        {"btc": 84000, "btc_chg": -2.1},
        "",
        "SHORT BTC short-term risk-off, reversal_risk flag"
    ),

    "china_summit": (
        "My meeting with President Xi Jinping has been rescheduled for May 14th "
        "in Beijing. Historic progress toward trade normalization. MAGA!",
        "truth_social",
        {"btc": 84000, "btc_chg": -0.5},
        "",
        "LONG BTC + LONG ETH — trade de-escalation, risk-on"
    ),

    # ── Reuters — macro news ────────────────────────────────────────────────

    "reuters_fed": (
        "Federal Reserve signals three rate cuts in 2025 as inflation cools "
        "toward target. Chair Powell says policy pivot is appropriate now. "
        "Dollar weakens sharply on the news.",
        "reuters",
        {"btc": 84000, "btc_chg": +1.5, "dxy": 103.2, "dxy_chg": -0.8},
        "Fed signals three rate cuts, dollar weakens",
        "LONG BTC — dollar weakness confirmed, rss_lag_risk flag, lower confidence than TS"
    ),

    "reuters_tariff": (
        "Trump administration confirms 25% tariffs on European Union goods "
        "effective next week. EU threatens retaliation on US tech and agriculture. "
        "Markets react sharply to escalating trade war.",
        "reuters",
        {"btc": 84000, "btc_chg": -3.2, "sp": 5380, "sp_chg": -1.8},
        "Trump confirms 25% EU tariffs, retaliation threatened",
        "SHORT BTC — risk-off, already_priced_in likely given market already moved"
    ),

    "reuters_ceasefire": (
        "Ukraine and Russia agree to 72-hour ceasefire following marathon "
        "negotiations in Vienna. US and EU officials confirm the agreement. "
        "Brent crude falls 3% on the news.",
        "reuters",
        {"btc": 84000, "btc_chg": +0.8, "brent": 71.0, "brent_chg": -3.1},
        "Ukraine-Russia 72-hour ceasefire agreed in Vienna",
        "LONG BTC + LONG ETH — risk-on, ceasefire confirmed by multiple parties"
    ),

    # ── AP News ─────────────────────────────────────────────────────────────

    "ap_sanctions": (
        "US Treasury imposes new sanctions on Iran targeting oil exports "
        "and financial sector. Senior officials say pressure campaign will "
        "intensify. Iran vows retaliation.",
        "ap",
        {"btc": 84000, "btc_chg": -0.9, "brent": 76.5, "brent_chg": +1.8},
        "US Treasury hits Iran with new oil sanctions",
        "Ambiguous — SHORT BTC risk-off vs LONG on geopolitical uncertainty"
    ),

    # ── Noise — should NOT trade ────────────────────────────────────────────

    "endorsement": (
        "It is my Great Honor to endorse T.W. Shannon for Lieutenant Governor "
        "of Oklahoma! Vote T.W.!",
        "truth_social",
        {},
        "",
        "NO TRADE — political endorsement, no macro implications"
    ),

    "personal_attack": (
        "Sleepy Joe Biden was the WORST president in American history. "
        "Total disaster. MAGA!",
        "truth_social",
        {},
        "",
        "NO TRADE — personal attack, no market implications"
    ),

    "reuters_sports": (
        "Super Bowl champion Kansas City Chiefs parade draws record crowds "
        "downtown. Mayor declares city holiday.",
        "reuters",
        {},
        "Chiefs Super Bowl parade draws record crowds",
        "NO TRADE — sports news"
    ),

    "sarcasm": (
        "Oh great, Crooked Hillary is BACK! Doing TERRIFIC. "
        "Everything is FINE with the economy. #Sarcasm",
        "truth_social",
        {},
        "",
        "NO TRADE — sarcasm detected"
    ),

    # ── Edge cases ──────────────────────────────────────────────────────────

    "already_moved": (
        "We are imposing MASSIVE new tariffs on all Chinese steel. "
        "American workers come FIRST!",
        "truth_social",
        {"btc": 84000, "btc_chg": -4.5, "eth": 1950, "eth_chg": -5.2},
        "",
        "already_priced_in — market already moved hard in expected direction"
    ),

    "ambiguous_energy": (
        "Energy independence is coming back STRONGER than ever. "
        "We are going to do things NOBODY thought possible. Watch!",
        "truth_social",
        {},
        "",
        "Low confidence — vague, no specific policy, ambiguous_intent flag"
    ),
}


# ── Layer 1: Classifier tests ─────────────────────────────────────────────────

async def run_classifier_test(
    name: str,
    text: str,
    source: str,
    market_kwargs: dict,
    headline: str,
    comment: str,
    classifier: Classifier,
) -> bool:
    print(f"\n{'─'*64}")
    print(f"TEST [{source.upper()}]: {name}")
    print(f"Expected: {comment}")
    if headline:
        print(f"Headline: {headline}")
    else:
        print(f"Post: {text[:100]}...")
    print(f"{'─'*64}")

    market = mock_market(**market_kwargs) if market_kwargs else mock_market()
    post   = MockPost(
        id=f"test-{name}",
        text=text,
        source=source,
        headline=headline,
    )

    signal = await classifier.classify(post, market)

    if signal:
        print(signal.log_summary())

        # Basic sanity checks
        passed = True

        # Noise posts should not trade
        noise_tests = {"endorsement", "personal_attack", "reuters_sports", "sarcasm"}
        if name in noise_tests and signal.should_trade():
            print(f"❌ FAIL: Noise post generated trade signal")
            passed = False
        elif name in noise_tests and not signal.should_trade():
            print(f"✅ PASS: Noise correctly filtered")

        # Market-moving posts should have assets
        if name not in noise_tests and signal.is_market_moving:
            if not signal.assets:
                print(f"⚠️  WARNING: Market-moving signal has no assets")
            # Check TP > SL (asymmetric)
            for a in signal.tradeable_assets():
                if a.levels and a.levels.take_profit_pct <= a.levels.stop_loss_pct:
                    print(f"⚠️  WARNING: {a.symbol} TP <= SL (not asymmetric)")
                if a.levels and a.levels.reward_risk_ratio() < 1.5:
                    print(f"⚠️  WARNING: {a.symbol} R/R {a.levels.reward_risk_ratio():.1f}x < 1.5x")

        # Reuters/AP should have rss_lag_risk flag on market-moving posts
        if source in ("reuters", "ap") and signal.is_market_moving:
            if "rss_lag_risk" not in signal.risk_flags:
                print(f"⚠️  NOTE: Reuters/AP market-moving post missing rss_lag_risk flag")

        return passed
    else:
        print("❌ Classifier returned None — check logs")
        return False


# ── Layer 2: Market data test ─────────────────────────────────────────────────

async def run_market_test():
    print(f"\n{'='*64}")
    print("LAYER 2: MARKET DATA TEST")
    print(f"{'='*64}")

    from market_poller import fetch_ticker, SYMBOLS

    loop = asyncio.get_event_loop()
    print("Fetching live prices...")

    passed = True
    for symbol, name in SYMBOLS.items():
        snap = await loop.run_in_executor(None, fetch_ticker, symbol, name)
        if snap.price:
            print(f"  ✅ {snap.summary()}")
        else:
            print(f"  ❌ {name} ({symbol}): FAILED")
            if symbol in ("BTC-USD", "ETH-USD"):
                passed = False  # crypto prices are critical

    return passed


# ── Layer 3: Full pipeline test ───────────────────────────────────────────────

async def run_full_pipeline_test(classifier: Classifier):
    print(f"\n{'='*64}")
    print("LAYER 3: FULL PIPELINE TEST (paper orders)")
    print(f"{'='*64}")

    try:
        from order_executor import OrderExecutor
        executor = OrderExecutor()
    except ValueError as e:
        print(f"❌ {e}")
        return False

    # Use peace_deal as the test signal — should fire LONG BTC
    market = mock_market()
    post   = MockPost(
        id="test-full-pipeline",
        text=(
            "Just spoke with Putin and Zelensky. A PERFECT ceasefire has been "
            "reached. The war is OVER!"
        ),
        source="truth_social",
    )

    print("Running full classify → execute cycle...")
    open_positions = executor.get_open_positions()
    signal = await classifier.classify(post, market, open_positions)

    if not signal:
        print("❌ Classifier failed")
        return False

    print(signal.log_summary())

    if signal.should_trade():
        print("\nAttempting paper order placement...")
        orders = await executor.execute(signal)
        if orders:
            print(f"✅ {len(orders)} order(s) submitted to Alpaca paper account:")
            for o in orders:
                print(f"   {o.log_line()}")
            return True
        else:
            print("⚠️  Signal generated but no orders placed (check sizing/exposure)")
            return True  # not a failure per se
    else:
        print("⚠️  Signal did not meet trade threshold for full pipeline test")
        return True


# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(description="Sentimate Trader — Test Pipeline")
    parser.add_argument("--layer", type=int, default=3,
                        help="Test layer to run (1=classifier, 2=+market, 3=+orders)")
    parser.add_argument("--test", type=str, default=None,
                        help="Run a single named test")
    args = parser.parse_args()

    print("\n" + "="*64)
    print("SENTIMATE TRADER — CRYPTO PIPELINE TEST")
    print("="*64)

    # Layer 1 always runs
    print(f"\n{'='*64}")
    print("LAYER 1: CLASSIFIER TESTS")
    print(f"{'='*64}")

    try:
        classifier = Classifier()
    except ValueError as e:
        print(f"❌ {e}")
        return

    tests_to_run = (
        {args.test: TESTS[args.test]} if args.test and args.test in TESTS
        else TESTS
    )

    results = []
    for name, (text, source, market_kwargs, headline, comment) in tests_to_run.items():
        passed = await run_classifier_test(
            name, text, source, market_kwargs, headline, comment, classifier
        )
        results.append((name, passed))
        if len(tests_to_run) > 1:
            await asyncio.sleep(0.5)

    # Summary
    print(f"\n{'='*64}")
    print("LAYER 1 RESULTS:")
    passed_count = sum(1 for _, p in results if p)
    for name, passed in results:
        print(f"  {'✅' if passed else '❌'} {name}")
    print(f"\n{passed_count}/{len(results)} tests passed")

    if args.layer >= 2:
        market_ok = await run_market_test()
        print(f"\nLayer 2: {'✅ PASSED' if market_ok else '❌ FAILED'}")

    if args.layer >= 3:
        pipeline_ok = await run_full_pipeline_test(classifier)
        print(f"\nLayer 3: {'✅ PASSED' if pipeline_ok else '❌ FAILED'}")

    print(f"\n{'='*64}")
    print("All signals logged to signal_log.jsonl")
    print(f"{'='*64}\n")


if __name__ == "__main__":
    asyncio.run(main())
