"""
test_context.py
---------------
Quickly verifies all market context sources are populating before
you start the day's paper trading session.

Usage:
    python test_context.py

Checks:
  1. BTC/ETH prices fetching
  2. Fear & Greed Index
  3. BTC Funding Rate
  4. Polymarket (may return empty if API is down — non-critical)
  5. Full classifier context block as Claude will see it
  6. One sample classification against a real-looking post
"""

import asyncio
import logging
from datetime import datetime, timezone
from dataclasses import dataclass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

async def main():
    import aiohttp
    from market_poller import (
        MarketSnapshot, fetch_fear_greed, fetch_funding_rate,
        fetch_polymarket, fetch_ticker, SYMBOLS
    )
    from classifier import Classifier

    print("\n" + "="*64)
    print("SENTIMATE TRADER — CONTEXT VERIFICATION")
    print("="*64)

    loop = asyncio.get_event_loop()

    async with aiohttp.ClientSession() as session:

        # 1. Prices
        print("\n[1] LIVE PRICES")
        state = MarketSnapshot()
        tasks = [
            loop.run_in_executor(None, fetch_ticker, sym, name)
            for sym, name in SYMBOLS.items()
        ]
        results = await asyncio.gather(*tasks)
        state.tickers = {s.symbol: s for s in results}
        state.last_updated_utc = datetime.now(timezone.utc).isoformat()

        for sym in ["BTC-USD", "ETH-USD", "ES=F", "BZ=F", "GC=F", "DX-Y.NYB"]:
            snap = state.tickers.get(sym)
            if snap and snap.price:
                print(f"  ✅ {snap.summary()}")
            else:
                print(f"  ❌ {sym}: failed")

        # 2. Fear & Greed
        print("\n[2] FEAR & GREED INDEX")
        state.fear_greed = await fetch_fear_greed(session)
        if state.fear_greed.value:
            print(f"  ✅ {state.fear_greed.summary()}")
        else:
            print(f"  ❌ Fear & Greed: unavailable")

        # 3. Funding Rate
        print("\n[3] BTC FUNDING RATE")
        state.funding_rate = await fetch_funding_rate(session)
        if state.funding_rate.rate is not None:
            print(f"  ✅ {state.funding_rate.summary()}")
        else:
            print(f"  ❌ Funding rate: unavailable")

        # 4. Polymarket
        print("\n[4] POLYMARKET")
        state.polymarket = await fetch_polymarket(session)
        if state.polymarket.markets:
            for m in state.polymarket.markets:
                print(f"  ✅ {m['question'][:60]}: {m['probability']:.0%}")
        else:
            print(f"  ⚠️  Polymarket: no markets returned (non-critical)")

        # 5. Full context block
        print("\n[5] FULL CONTEXT BLOCK (as Claude sees it)")
        print("-"*64)
        print(state.to_classifier_context())
        print("-"*64)

        # 6. Sample classification
        print("\n[6] SAMPLE CLASSIFICATION")
        try:
            classifier = Classifier()

            @dataclass
            class MockPost:
                id: str = "test-context-verify"
                text: str = (
                    "Just announced a 25% tariff on ALL Chinese goods. "
                    "China has been ripping us off for years. No more!"
                )
                url: str = "https://truthsocial.com/test"
                scraped_utc: str = datetime.now(timezone.utc).isoformat()
                source: str = "truth_social"
                headline: str = ""

            signal = await classifier.classify(MockPost(), state)
            if signal:
                print(signal.log_summary())
                print(f"\n✅ Classifier working | Latency: {signal.latency_ms:.0f}ms")
            else:
                print("❌ Classifier returned None")
        except ValueError as e:
            print(f"⚠️  Classifier skipped: {e}")

    print("\n" + "="*64)
    print("Context verification complete.")
    print("If all checks passed, run: python feed_listener.py")
    print("="*64 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
