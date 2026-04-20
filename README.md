# Sentimate Trader

A fully automated crypto sentiment trading system that monitors Donald Trump's Truth Social posts in real time, classifies each post for macro market implications using an LLM, and executes trades on BTC/USD and ETH/USD via the Alpaca API.

The core thesis: Trump's direct posts move cryptocurrency markets faster and more reliably than traditional news sources. The correlation between Trump rhetoric and Bitcoin volatility increased from 0.31 in early 2024 to 0.48 in early 2025. The system is designed to capture sentiment-driven moves before they fully resolve.

---

## Architecture

```
Truth Social (@realDonaldTrump)
         │
         ▼
┌─────────────────────────┐
│  Playwright Scraper      │  Headless browser polls every 15s.
│  (feed_listener.py)      │  Deduplicates via seen_ids.json.
│                          │  Session cookie persists in auth_state.json.
└─────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────┐
│              Market Context Stack                    │
│                                                      │
│  yfinance    → BTC/ETH/ES/BZ/GC/DXY prices (15s)   │
│  alternative.me → Fear & Greed index (5min)          │
│  Binance/Bybit  → BTC perpetual funding rate         │
│  Polymarket     → Geopolitical event probabilities   │
└─────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Claude Sonnet           │  Classifies post + full market context.
│  Classifier              │  Returns: BTC/ETH direction, confidence,
│  (classifier.py)         │  urgency, TP%, SL%, is_market_moving flag.
└─────────────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Order Executor          │  Market entry + trailing stop + hard SL.
│  (order_executor.py)     │  Dynamic sizing: 30% base × confidence
│                          │  × urgency multiplier. 75% max exposure.
└─────────────────────────┘
         │
         ▼
    Alpaca Paper Trading API
    BTC/USD · ETH/USD
```

---

## Signal Classification

Each post is classified along two axes: **direction** (long/short/neutral per asset) and **urgency** (1–3). The classifier receives the full market context stack with every call, enabling context-adjusted signal strength.

**Signal categories:**

| Type | Examples | Effect |
|------|----------|--------|
| Risk-on | Peace deals, trade de-escalation, Fed rate cuts, dollar weakness | LONG BTC, LONG ETH |
| Risk-off | Tariffs, conflict escalation, dollar strength | SHORT BTC, SHORT ETH |
| Noise | Endorsements, rallies, personal attacks, domestic legal posts | `is_market_moving: false` — no trade |

**Context adjustments (Fear/Greed):**

| Level | Effect |
|-------|--------|
| Extreme Fear (0–25) | Bullish signals amplified — oversold squeeze potential |
| Greed (55–75) | Bullish signals dampened — market already extended |
| Extreme Greed (76+) | Bullish signals likely to fade — tighten TP/SL |

**Funding rate adjustments:**

| Funding Rate | Effect |
|--------------|--------|
| Heavily negative (< −0.05%) | Bullish news = squeeze potential — widen TP |
| Heavily positive (> +0.05%) | Bearish news = short squeeze potential |

---

## Position Sizing

```
base   = available_cash × 0.30
scaled = base × confidence × urgency_multiplier
sized  = min(scaled, MAX_SINGLE_TRADE, cash_cap, remaining_capacity)
```

| Parameter | Value |
|-----------|-------|
| Base allocation | 30% of cash per signal |
| Max single trade | 35% hard cap |
| Max total exposure | 75% |
| Urgency multiplier | 1: 0.5× · 2: 0.8× · 3: 1.0× |
| Minimum trade | $10 (fractional crypto) |

---

## Order Structure

Each signal fires three orders:

1. **Market entry** — fills immediately at best available price
2. **Trailing stop** — follows price upward; exits on reversal by `trail%` from peak
3. **Hard stop loss** — fixed price floor below trailing stop as safety net

The trailing stop and hard stop run simultaneously. Whichever fires first closes the position. A 1.5-second delay between entry and exit order submission allows the entry to fill before stops are placed.

---

## File Reference

| File | Role |
|------|------|
| `feed_listener.py` | Main entry point — manages auth, scraping loop, and post consumer |
| `classifier.py` | Claude Sonnet classifier — returns Signal with directions, confidence, TP/SL |
| `order_executor.py` | Alpaca executor — places market entry + trailing stop + hard SL |
| `market_poller.py` | Market context — prices, Fear/Greed, funding rate, Polymarket |
| `test_pipeline.py` | 14 test cases across signal types, run with `--layer 1/2/3` |
| `test_context.py` | Verifies all market data sources are live before starting a session |

**Runtime files (local only, not committed):**

| File | Purpose |
|------|---------|
| `seen_ids.json` | Post deduplication store |
| `auth_state.json` | Playwright Truth Social session cookie |
| `signal_log.jsonl` | Every classified signal — primary source for backtesting |
| `trade_log.jsonl` | Every submitted order with entry price, qty, trail%, SL, order IDs |

---

## Setup

### Dependencies
```bash
pip install playwright anthropic yfinance aiohttp alpaca-py feedparser
playwright install chromium
```

### Environment Variables
```bash
export ANTHROPIC_API_KEY=sk-ant-...     # Claude Sonnet classifier
export ALPACA_API_KEY=PK...             # Alpaca paper trading
export ALPACA_SECRET_KEY=...
```

### Startup
```bash
# Verify all data sources are live first
python test_context.py

# Start the system
python feed_listener.py
# First run opens a browser window — log into Truth Social
# Session is saved to auth_state.json; subsequent runs are headless
```

### Auto-restart (Linux)
```bash
while true; do python feed_listener.py; sleep 5; done
```

---

## Development Notes

A few things that came up during development worth documenting:

**RSS feeds removed.** Google News RSS returned 15–20 articles per poll, generating ~2,700 Claude API calls per hour. Cost was unsustainable. System now runs on Truth Social only.

**Groq classifier abandoned.** Groq returned unquoted string values in JSON responses, causing parse failures. Switched to Anthropic-only classifier.

**Duplicate position entries fixed.** Alpaca's lag between order submission and position registration caused duplicate entries on fast signals. Fixed with an in-memory `_pending_symbols` set.

**Buying power vs cash.** Alpaca returns 2× margin as `buying_power`. Sizing now uses `account.cash` directly to avoid overallocation.

---

## Tech Stack

| Component | Tool |
|-----------|------|
| Scraper | Playwright (headless Chromium) |
| Classifier | Claude Sonnet via Anthropic API |
| Market data | yfinance, alternative.me, Binance/Bybit, Polymarket Gamma API |
| Execution | Alpaca paper trading API |
| Language | Python 3.12 |

---

## Status

System is currently in paper trading validation. Paper trading integration targets BTC/USD and ETH/USD on Alpaca's paper environment. Live deployment pending signal log review and trail percentage calibration per asset.

---
