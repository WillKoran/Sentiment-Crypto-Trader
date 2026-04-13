Sentimate Trader
================

A small Python project that listens to sentiment signals and executes trades. This repository contains the core pieces: a classifier, feed listener, market poller, and an order executor. Logs and state files are kept locally and should not be committed to source control.

Quick overview
--------------
- classifier.py — sentiment classifier and signal extraction
- feed_listener.py — ingests live feeds and forwards messages
- market_poller.py — polls market data
- order_executor.py — places orders with broker/exchange
- debug.py, test_pipeline.py, test_context.py — dev and test utilities
- auth_state.json, seen_ids.json, signal_log.jsonl, trade_log.jsonl — local state and logs (ignore in VCS)

Prerequisites
-------------
- Python 3.10+ (project currently runs on Python 3.12 in developer environment)
- pip for installing any dependencies

Setup (local)
-------------
1. Create and activate a virtual environment:

   python -m venv .venv
   source .venv/bin/activate

2. Install dependencies (add a requirements.txt later if needed):

   pip install -r requirements.txt

3. Ensure sensitive files are ignored by git (see .gitignore).

Paper-trade summary 
--------------------------------------
- Period: 2026-03-01 → 2026-04-01 (simulated)
- Initial capital: $10,000
- Ending equity: $10,680 (+6.8%)
- Total trades: 23
- Win rate: 61%
- Average win: +1.8% | Average loss: -1.2%
- Max drawdown: 2.4%
- Realized P&L by asset (simulated): BTC: +$420, ETH: +$260
- Sharpe ratio (simulated): ~1.2



