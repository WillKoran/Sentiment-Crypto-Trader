"""
market_poller.py
----------------
Market context poller. Updates every 15 seconds and provides Claude
with a rich market snapshot including:

    Prices:        BTC, ETH, ES, Brent, Gold, DXY (via yfinance)
    Fear & Greed:  Crypto Fear & Greed Index (alternative.me — free)
    Funding Rate:  BTC perpetual funding rate (Binance — free)
    Polymarket:    Key geopolitical event probabilities (free API)

The combined context lets Claude reason about:
    - Whether the market is already at extreme sentiment (fade vs follow)
    - Squeeze potential (negative funding + bullish news = outsized move)
    - Whether geopolitical risks are already priced into prediction markets

All free, zero per-call cost.

Requirements:
    pip install yfinance aiohttp
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import aiohttp
import yfinance as yf

log = logging.getLogger(__name__)

POLL_INTERVAL          = 15
SLOW_POLL_INTERVAL     = 300   # Fear/Greed and Polymarket — update every 5 min

SYMBOLS = {
    "BTC-USD":   "Bitcoin",
    "ETH-USD":   "Ethereum",
    "ES=F":      "S&P 500 Futures",
    "BZ=F":      "Brent Crude",
    "GC=F":      "Gold",
    "DX-Y.NYB":  "US Dollar Index",
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; SentimatTrader/1.0)"
}


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class TickerSnapshot:
    symbol: str
    name: str
    price: Optional[float]
    prev_close: Optional[float]
    change_pct: Optional[float]
    day_high: Optional[float]
    day_low: Optional[float]
    is_crypto: bool = False
    timestamp_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def summary(self) -> str:
        if self.price is None:
            return f"{self.name} ({self.symbol}): unavailable"
        direction = "▲" if (self.change_pct or 0) >= 0 else "▼"
        return (
            f"{self.name} ({self.symbol}): "
            f"${self.price:,.2f} "
            f"{direction}{abs(self.change_pct or 0):.2f}%"
        )


@dataclass
class FearGreedSnapshot:
    value: Optional[int]          # 0-100
    label: Optional[str]          # "Extreme Fear" → "Extreme Greed"
    prev_value: Optional[int]     # yesterday
    prev_label: Optional[str]
    timestamp_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def summary(self) -> str:
        if self.value is None:
            return "Fear & Greed: unavailable"
        trend = ""
        if self.prev_value:
            delta = self.value - self.prev_value
            trend = f" ({'↑' if delta > 0 else '↓'}{abs(delta)} from yesterday's {self.prev_label})"
        return f"Fear & Greed: {self.value}/100 — {self.label}{trend}"

    def classifier_context(self) -> str:
        if self.value is None:
            return ""
        interpretation = {
            range(0, 26):   "extreme fear — market oversold, bullish signals likely amplified",
            range(26, 47):  "fear — cautious sentiment, bullish signals carry more weight",
            range(47, 55):  "neutral",
            range(55, 76):  "greed — market extended, bearish signals carry more weight",
            range(76, 101): "extreme greed — market overbought, bullish signals likely to fade faster",
        }
        interp = next(
            (v for r, v in interpretation.items() if self.value in r), ""
        )
        return f"Crypto Fear & Greed Index: {self.value}/100 ({self.label}) — {interp}"


@dataclass
class FundingRateSnapshot:
    symbol: str                    # "BTCUSDT"
    rate: Optional[float]          # e.g. 0.0001 = 0.01% per 8h
    rate_annualized: Optional[float]
    next_funding_utc: Optional[str]
    timestamp_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def summary(self) -> str:
        if self.rate is None:
            return "BTC Funding Rate: unavailable"
        pct = self.rate * 100
        direction = "positive (longs pay)" if self.rate > 0 else "negative (shorts pay)"
        return f"BTC Funding Rate: {pct:+.4f}% per 8h ({direction})"

    def classifier_context(self) -> str:
        if self.rate is None:
            return ""
        pct = self.rate * 100
        if self.rate < -0.05:
            interp = "heavily negative — market is net short, bullish squeeze risk is HIGH"
        elif self.rate < -0.01:
            interp = "negative — slight short bias, moderate squeeze potential on bullish news"
        elif self.rate < 0.01:
            interp = "near-zero — balanced positioning"
        elif self.rate < 0.05:
            interp = "positive — slight long bias, bearish signals reinforced"
        else:
            interp = "heavily positive — market is net long, bearish squeeze risk is HIGH"
        return f"BTC Perpetual Funding Rate: {pct:+.4f}%/8h — {interp}"


@dataclass
class PolymarketSnapshot:
    markets: list[dict] = field(default_factory=list)
    timestamp_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def classifier_context(self) -> str:
        if not self.markets:
            return ""
        lines = ["Key prediction market probabilities:"]
        for m in self.markets:
            lines.append(f"  - {m['question']}: {m['probability']:.0%}")
        return "\n".join(lines)


@dataclass
class MarketSnapshot:
    tickers: dict = field(default_factory=dict)
    fear_greed: Optional[FearGreedSnapshot] = None
    funding_rate: Optional[FundingRateSnapshot] = None
    polymarket: Optional[PolymarketSnapshot] = None
    last_updated_utc: str = ""

    def ready(self) -> bool:
        return len(self.tickers) > 0

    def summary(self) -> str:
        if not self.ready():
            return "Market data not yet available."
        lines = [f"Market snapshot @ {self.last_updated_utc}"]
        for sym in ["BTC-USD", "ETH-USD"]:
            if sym in self.tickers:
                lines.append(f"  {self.tickers[sym].summary()}")
        for sym in ["ES=F", "BZ=F", "GC=F", "DX-Y.NYB"]:
            if sym in self.tickers:
                lines.append(f"  {self.tickers[sym].summary()}")
        if self.fear_greed:
            lines.append(f"  {self.fear_greed.summary()}")
        if self.funding_rate:
            lines.append(f"  {self.funding_rate.summary()}")
        return "\n".join(lines)

    def to_classifier_context(self) -> str:
        """Full market context block injected into Claude prompt."""
        if not self.ready():
            return "No market data available."

        sections = []

        # Prices
        price_lines = []
        for snap in self.tickers.values():
            if snap.price is not None:
                chg = "up" if (snap.change_pct or 0) >= 0 else "down"
                price_lines.append(
                    f"- {snap.name} ({snap.symbol}): "
                    f"${snap.price:,.2f}, {chg} {abs(snap.change_pct or 0):.2f}% today"
                )
        sections.append("PRICES:\n" + "\n".join(price_lines))

        # Sentiment
        if self.fear_greed and self.fear_greed.value:
            sections.append("SENTIMENT:\n" + self.fear_greed.classifier_context())

        # Positioning
        if self.funding_rate and self.funding_rate.rate is not None:
            sections.append("MARKET POSITIONING:\n" + self.funding_rate.classifier_context())

        # Prediction markets
        if self.polymarket and self.polymarket.markets:
            sections.append(self.polymarket.classifier_context())

        return "\n\n".join(sections)


# ── Fetchers ──────────────────────────────────────────────────────────────────

def fetch_ticker(symbol: str, name: str) -> TickerSnapshot:
    is_crypto = symbol in ("BTC-USD", "ETH-USD")
    try:
        fi = yf.Ticker(symbol).fast_info
        price = fi.last_price
        prev  = fi.previous_close
        chg   = ((price - prev) / prev * 100) if price and prev and prev != 0 else None
        return TickerSnapshot(
            symbol=symbol, name=name,
            price=price, prev_close=prev,
            change_pct=chg,
            day_high=fi.day_high, day_low=fi.day_low,
            is_crypto=is_crypto,
        )
    except Exception as e:
        log.warning(f"Price fetch failed {symbol}: {e}")
        return TickerSnapshot(
            symbol=symbol, name=name,
            price=None, prev_close=None,
            change_pct=None, day_high=None, day_low=None,
            is_crypto=is_crypto,
        )


async def fetch_fear_greed(session: aiohttp.ClientSession) -> FearGreedSnapshot:
    try:
        async with session.get(
            "https://api.alternative.me/fng/?limit=2",
            headers=HEADERS, timeout=aiohttp.ClientTimeout(total=10)
        ) as resp:
            if resp.status == 200:
                data = await resp.json()
                entries = data.get("data", [])
                today = entries[0] if entries else {}
                yesterday = entries[1] if len(entries) > 1 else {}
                return FearGreedSnapshot(
                    value=int(today.get("value", 0)),
                    label=today.get("value_classification", ""),
                    prev_value=int(yesterday.get("value", 0)) if yesterday else None,
                    prev_label=yesterday.get("value_classification") if yesterday else None,
                )
    except Exception as e:
        log.warning(f"Fear & Greed fetch failed: {e}")
    return FearGreedSnapshot(value=None, label=None, prev_value=None, prev_label=None)


async def fetch_funding_rate(session: aiohttp.ClientSession) -> FundingRateSnapshot:
    """
    Tries multiple sources for BTC perpetual funding rate.
    Priority: Binance → Bybit → Coinglass (free tier)
    """
    # Source 1: Binance futures
    try:
        async with session.get(
            "https://fapi.binance.com/fapi/v1/premiumIndex?symbol=BTCUSDT",
            headers=HEADERS, timeout=aiohttp.ClientTimeout(total=8)
        ) as resp:
            if resp.status == 200:
                data = await resp.json()
                rate = float(data.get("lastFundingRate", 0))
                next_time = data.get("nextFundingTime")
                next_utc = None
                if next_time:
                    next_utc = datetime.fromtimestamp(
                        int(next_time) / 1000, tz=timezone.utc
                    ).isoformat()
                log.debug("Funding rate: Binance")
                return FundingRateSnapshot(
                    symbol="BTCUSDT", rate=rate,
                    rate_annualized=rate * 3 * 365,
                    next_funding_utc=next_utc,
                )
    except Exception:
        pass

    # Source 2: Bybit
    try:
        async with session.get(
            "https://api.bybit.com/v5/market/tickers?category=linear&symbol=BTCUSDT",
            headers=HEADERS, timeout=aiohttp.ClientTimeout(total=8)
        ) as resp:
            if resp.status == 200:
                data = await resp.json()
                tickers = data.get("result", {}).get("list", [])
                if tickers:
                    rate = float(tickers[0].get("fundingRate", 0))
                    log.debug("Funding rate: Bybit")
                    return FundingRateSnapshot(
                        symbol="BTCUSDT", rate=rate,
                        rate_annualized=rate * 3 * 365,
                        next_funding_utc=None,
                    )
    except Exception:
        pass

    # Source 3: Coinglass (no auth needed for basic data)
    try:
        async with session.get(
            "https://open-api.coinglass.com/public/v2/funding?symbol=BTC",
            headers=HEADERS, timeout=aiohttp.ClientTimeout(total=8)
        ) as resp:
            if resp.status == 200:
                data = await resp.json()
                # Find Binance rate from list
                rates = data.get("data", [])
                binance = next(
                    (r for r in rates if r.get("exchangeName") == "Binance"), None
                )
                if binance:
                    rate = float(binance.get("rate", 0)) / 100  # convert % to decimal
                    log.debug("Funding rate: Coinglass")
                    return FundingRateSnapshot(
                        symbol="BTCUSDT", rate=rate,
                        rate_annualized=rate * 3 * 365,
                        next_funding_utc=None,
                    )
    except Exception:
        pass

    log.warning("All funding rate sources failed")
    return FundingRateSnapshot(
        symbol="BTCUSDT", rate=None,
        rate_annualized=None, next_funding_utc=None
    )


async def fetch_polymarket(session: aiohttp.ClientSession) -> PolymarketSnapshot:
    """
    Fetches geopolitically relevant prediction markets using Polymarket's
    Gamma API which has better keyword filtering than the CLOB API.

    Filters for macro-relevant markets only — skips sports, entertainment etc.
    """
    MACRO_KEYWORDS = [
        "tariff", "ceasefire", "fed rate", "ukraine", "iran",
        "russia", "china trade", "bitcoin etf", "crypto regulation",
        "interest rate", "recession", "nato", "oil price"
    ]

    markets = []
    try:
        async with session.get(
            "https://gamma-api.polymarket.com/markets",
            params={
                "active": "true",
                "closed": "false",
                "limit": 50,
                "order": "volume",
                "ascending": "false",
            },
            headers=HEADERS,
            timeout=aiohttp.ClientTimeout(total=10)
        ) as resp:
            if resp.status == 200:
                data = await resp.json()
                results = data if isinstance(data, list) else data.get("markets", [])

                for m in results:
                    question = m.get("question", "").lower()

                    # Only include macro-relevant markets
                    if not any(kw in question for kw in MACRO_KEYWORDS):
                        continue

                    # Get YES probability
                    outcomes_prices = m.get("outcomePrices", "[]")
                    outcomes = m.get("outcomes", "[]")
                    try:
                        if isinstance(outcomes_prices, str):
                            import json as _json
                            prices = _json.loads(outcomes_prices)
                            outcome_list = _json.loads(outcomes) if isinstance(outcomes, str) else outcomes
                        else:
                            prices = outcomes_prices
                            outcome_list = outcomes

                        yes_idx = next(
                            (i for i, o in enumerate(outcome_list)
                             if str(o).lower() == "yes"), 0
                        )
                        prob = float(prices[yes_idx]) if prices else 0.5

                        if 0.02 < prob < 0.98:  # skip trivially resolved markets
                            markets.append({
                                "question": m.get("question", "")[:80],
                                "probability": prob,
                            })
                    except Exception:
                        continue

                    if len(markets) >= 5:
                        break

    except Exception as e:
        log.warning(f"Polymarket fetch failed: {e}")

    return PolymarketSnapshot(markets=markets)


# ── Poll loops ────────────────────────────────────────────────────────────────

async def poll_market(state: MarketSnapshot):
    log.info(
        f"Market poller started — prices every {POLL_INTERVAL}s | "
        f"sentiment every {SLOW_POLL_INTERVAL}s"
    )
    loop = asyncio.get_event_loop()

    async with aiohttp.ClientSession() as session:
        # Initial slow-poll fetch immediately on startup
        state.fear_greed    = await fetch_fear_greed(session)
        state.funding_rate  = await fetch_funding_rate(session)
        state.polymarket    = await fetch_polymarket(session)

        slow_counter = 0

        while True:
            try:
                # Fast poll — prices every 15s
                tasks = [
                    loop.run_in_executor(None, fetch_ticker, sym, name)
                    for sym, name in SYMBOLS.items()
                ]
                results = await asyncio.gather(*tasks)
                state.tickers = {s.symbol: s for s in results}
                state.last_updated_utc = datetime.now(timezone.utc).isoformat()

                # Slow poll — sentiment/positioning every 5 min
                slow_counter += POLL_INTERVAL
                if slow_counter >= SLOW_POLL_INTERVAL:
                    slow_counter = 0
                    state.fear_greed   = await fetch_fear_greed(session)
                    state.funding_rate = await fetch_funding_rate(session)
                    state.polymarket   = await fetch_polymarket(session)
                    log.info(
                        f"[MARKET SENTIMENT]\n"
                        f"  {state.fear_greed.summary()}\n"
                        f"  {state.funding_rate.summary()}"
                    )

                log.info(f"[MARKET]\n{state.summary()}")

            except Exception as e:
                log.error(f"Market poll error: {e}", exc_info=True)

            await asyncio.sleep(POLL_INTERVAL)
