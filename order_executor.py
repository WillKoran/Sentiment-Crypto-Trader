"""
order_executor.py
-----------------
Crypto order executor using Alpaca paper trading.

Key differences from equity executor:
  - Crypto symbols use "BTC/USD" format
  - Fractional quantities supported natively (trade $50 of BTC)
  - No PDT rule — trade as often as signals fire
  - 24/7 markets — no market hours check needed
  - Bracket orders supported on Alpaca crypto
  - Uses CryptoHistoricalDataClient for price feeds

Sizing:
  Same dynamic sizing model as equity executor:
  base = cash × BASE_ALLOC_PCT × confidence × urgency_multiplier
  Hard caps: MAX_SINGLE_TRADE, MAX_TOTAL_EXPOSURE, MIN_TRADE_USD

Requirements:
    pip install alpaca-py
    export ALPACA_API_KEY=PK...
    export ALPACA_SECRET_KEY=...
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Optional

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest,
    TakeProfitRequest,
    StopLossRequest,
    TrailingStopOrderRequest,
)
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical.crypto import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoLatestQuoteRequest

from classifier import Signal, AssetSignal

log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

PAPER_TRADING       = True
TRADE_LOG_FILE      = "trade_log.jsonl"

BASE_ALLOC_PCT      = 0.30    # 30% of cash per signal (up from 20%)
MAX_SINGLE_TRADE    = 0.35    # 35% hard cap per trade (up from 25%)
MAX_TOTAL_EXPOSURE  = 0.75    # 75% max total exposure
MIN_TRADE_USD       = 10      # lower than equities — crypto is fractional
URGENCY_SCALE       = {1: 0.5, 2: 0.8, 3: 1.0}  # tightened urgency 2 scaling

# Supported crypto symbols
CRYPTO_SYMBOLS = {"BTC/USD", "ETH/USD"}


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class TradeOrder:
    signal_post_id: str
    symbol: str            # e.g. "BTC/USD"
    direction: str
    entry_price: float
    qty: float             # fractional units of crypto
    notional_usd: float    # dollar value of position
    trail_pct: float       # trailing stop percentage
    sl_price: float        # fixed hard stop loss
    sl_pct: float
    entry_order_id: Optional[str]    # market entry order
    trail_order_id: Optional[str]    # trailing stop order
    sl_order_id: Optional[str]       # hard stop loss order
    status: str
    reason: str
    submitted_utc: str

    def to_dict(self):
        return asdict(self)

    def log_line(self) -> str:
        emoji = "📈" if self.direction == "long" else "📉"
        return (
            f"{emoji} {self.direction.upper()} {self.symbol} | "
            f"${self.notional_usd:.0f} | "
            f"Qty: {self.qty:.6f} @ ${self.entry_price:,.2f} | "
            f"Trail: {self.trail_pct:.1f}% | "
            f"Hard SL: ${self.sl_price:,.2f} (-{self.sl_pct:.2f}%) | "
            f"Status: {self.status}"
        )


# ── Executor ──────────────────────────────────────────────────────────────────

class OrderExecutor:

    def __init__(self):
        api_key    = os.environ.get("ALPACA_API_KEY")
        secret_key = os.environ.get("ALPACA_SECRET_KEY")

        if not api_key or not secret_key:
            raise ValueError(
                "ALPACA_API_KEY and ALPACA_SECRET_KEY must be set.\n"
                "Get free paper trading keys at https://alpaca.markets"
            )

        self.trading = TradingClient(
            api_key=api_key,
            secret_key=secret_key,
            paper=PAPER_TRADING,
        )
        self.crypto_data = CryptoHistoricalDataClient(
            api_key=api_key,
            secret_key=secret_key,
        )

        # In-memory pending tracker — prevents double-entry during Alpaca lag
        self._pending_symbols: set   = set()
        self._pending_exposure: float = 0.0

        log.info(
            f"Order executor ready — "
            f"{'PAPER' if PAPER_TRADING else '⚠️ LIVE'} crypto trading | "
            f"base={BASE_ALLOC_PCT:.0%} of balance | "
            f"min=${MIN_TRADE_USD}"
        )

    # ── Price fetch ───────────────────────────────────────────────────────────

    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Fetch latest crypto price from Alpaca."""
        try:
            req   = CryptoLatestQuoteRequest(symbol_or_symbols=symbol)
            quote = self.crypto_data.get_crypto_latest_quote(req)
            q     = quote[symbol]
            price = q.ask_price if q.ask_price and q.ask_price > 0 else q.bid_price
            return float(price)
        except Exception as e:
            log.error(f"Failed to get price for {symbol}: {e}")
            return None

    # ── Level calculation ─────────────────────────────────────────────────────

    def calculate_levels(
        self, direction: str, entry: float, tp_pct: float, sl_pct: float
    ) -> tuple[float, float]:
        tp_mult = tp_pct / 100
        sl_mult = sl_pct / 100
        if direction == "long":
            return round(entry * (1 + tp_mult), 2), round(entry * (1 - sl_mult), 2)
        else:
            return round(entry * (1 - tp_mult), 2), round(entry * (1 + sl_mult), 2)

    # ── Position management ───────────────────────────────────────────────────

    def get_open_positions(self) -> dict[str, float]:
        try:
            positions = self.trading.get_all_positions()
            confirmed = {p.symbol: float(p.qty) for p in positions}
            newly_confirmed = self._pending_symbols & set(confirmed.keys())
            if newly_confirmed:
                self._pending_symbols -= newly_confirmed
                if not self._pending_symbols:
                    self._pending_exposure = 0.0
            return confirmed
        except Exception as e:
            log.warning(f"Could not fetch positions: {e}")
            return {}

    def is_already_in_position(self, symbol: str, open_positions: dict) -> bool:
        in_alpaca  = symbol in open_positions and open_positions[symbol] != 0
        in_pending = symbol in self._pending_symbols
        return in_alpaca or in_pending

    def get_account(self) -> dict:
        try:
            acc = self.trading.get_account()
            return {
                "cash":            float(acc.cash),
                "buying_power":    float(acc.buying_power),
                "portfolio_value": float(acc.portfolio_value),
                "equity":          float(acc.equity),
            }
        except Exception as e:
            log.error(f"Failed to fetch account: {e}")
            return {}

    def get_total_exposure_usd(self, open_positions: dict) -> float:
        try:
            positions = self.trading.get_all_positions()
            confirmed = sum(abs(float(p.market_value)) for p in positions)
        except Exception:
            confirmed = 0.0
        return confirmed + self._pending_exposure

    def calculate_position_size(
        self,
        confidence: float,
        urgency: int,
        available_cash: float,
        portfolio_value: float,
        current_exposure: float,
    ) -> float:
        # Guard 1: insufficient cash
        if available_cash < MIN_TRADE_USD:
            log.warning(
                f"[SIZER] ❌ Insufficient cash (${available_cash:.2f}) — "
                f"minimum: ${MIN_TRADE_USD}"
            )
            return 0.0

        # Guard 2: exposure cap
        max_exposure   = portfolio_value * MAX_TOTAL_EXPOSURE
        remaining      = max_exposure - current_exposure
        if remaining <= MIN_TRADE_USD:
            log.warning(
                f"[SIZER] ❌ At max exposure "
                f"(${current_exposure:.0f}/${max_exposure:.0f})"
            )
            return 0.0

        # Guard 3: scaled size
        urgency_mult = URGENCY_SCALE.get(urgency, 0.7)
        base         = available_cash * BASE_ALLOC_PCT
        scaled       = base * confidence * urgency_mult

        # Guard 4: caps
        hard_cap  = portfolio_value * MAX_SINGLE_TRADE
        cash_cap  = available_cash * 0.95
        sized     = min(scaled, hard_cap, cash_cap, remaining)

        # Guard 5: minimum
        if sized < MIN_TRADE_USD:
            log.info(f"[SIZER] ❌ ${sized:.2f} below minimum ${MIN_TRADE_USD}")
            return 0.0

        log.info(
            f"[SIZER] ✅ ${sized:.2f} | "
            f"conf={confidence:.0%} × urgency={urgency_mult} × base=${base:.2f} | "
            f"caps: hard=${hard_cap:.0f} cash=${cash_cap:.0f} capacity=${remaining:.0f}"
        )
        return sized

    def close_position(self, symbol: str) -> bool:
        try:
            self.trading.close_position(symbol)
            log.info(f"[EXECUTOR] 🚪 Closed: {symbol}")
            return True
        except Exception as e:
            log.error(f"[EXECUTOR] Failed to close {symbol}: {e}")
            return False

    async def handle_position_actions(
        self, signal: Signal, open_positions: dict
    ) -> list[str]:
        if not signal.position_actions:
            return []

        loop      = asyncio.get_event_loop()
        summaries = []

        for action in signal.position_actions:
            sym = action.get("symbol")
            act = action.get("action")
            reason = action.get("reason", "")

            if not sym or not act:
                continue
            if sym not in open_positions:
                continue

            if act == "close":
                log.info(f"[EXECUTOR] 🚪 Closing {sym} — {reason}")
                success = await loop.run_in_executor(
                    None, self.close_position, sym
                )
                summaries.append(
                    f"CLOSE {sym}: {'ok' if success else 'failed'}"
                )

            elif act == "add":
                log.info(f"[EXECUTOR] ➕ Adding to {sym} — {reason}")
                matching = next(
                    (a for a in signal.tradeable_assets() if a.symbol == sym),
                    None
                )
                if matching:
                    order = await self._execute_asset(
                        signal.post_id, matching, loop, {}, 500
                    )
                    if order:
                        log.info(f"[EXECUTOR] {order.log_line()}")
                        self._log_trade(order)
                        summaries.append(f"ADD {sym}: submitted")
                else:
                    summaries.append(f"ADD {sym}: no matching signal")

            elif act == "hold":
                log.info(f"[EXECUTOR] ✋ Holding {sym} — {reason}")
                summaries.append(f"HOLD {sym}: no action")

        return summaries

    # ── Order placement ──────────────────────────────────────────────────────

    def place_entry_order(
        self,
        symbol: str,
        direction: str,
        qty: float,
    ) -> Optional[str]:
        """Places the entry market order only. Returns order ID."""
        side = OrderSide.BUY if direction == "long" else OrderSide.SELL
        try:
            order = self.trading.submit_order(
                MarketOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=side,
                    time_in_force=TimeInForce.GTC,
                )
            )
            return str(order.id)
        except Exception as e:
            log.error(f"Entry order failed for {symbol}: {e}")
            return None

    def place_trailing_stop(
        self,
        symbol: str,
        direction: str,
        qty: float,
        trail_pct: float,
    ) -> Optional[str]:
        """
        Places a trailing stop order to ride the move and exit on reversal.
        For LONG: sell trailing stop (fires when price drops trail_pct from peak)
        For SHORT: buy trailing stop (fires when price rises trail_pct from trough)
        """
        side = OrderSide.SELL if direction == "long" else OrderSide.BUY
        try:
            order = self.trading.submit_order(
                TrailingStopOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=side,
                    time_in_force=TimeInForce.GTC,
                    trail_percent=trail_pct,
                )
            )
            log.info(
                f"[EXECUTOR] 📍 Trailing stop set: {trail_pct}% trail on "
                f"{symbol} ({direction})"
            )
            return str(order.id)
        except Exception as e:
            log.error(f"Trailing stop failed for {symbol}: {e}")
            return None

    def place_hard_stop(
        self,
        symbol: str,
        direction: str,
        qty: float,
        sl_price: float,
    ) -> Optional[str]:
        """
        Places a fixed hard stop loss as a safety net below the trailing stop.
        Catches catastrophic moves that blow through the trail.
        """
        side = OrderSide.SELL if direction == "long" else OrderSide.BUY
        try:
            order = self.trading.submit_order(
                MarketOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=side,
                    time_in_force=TimeInForce.GTC,
                    stop_loss=StopLossRequest(stop_price=sl_price),
                )
            )
            log.info(
                f"[EXECUTOR] 🛡️  Hard stop set at ${sl_price:,.2f} on {symbol}"
            )
            return str(order.id)
        except Exception as e:
            log.error(f"Hard stop failed for {symbol}: {e}")
            return None

    # ── Main execute ──────────────────────────────────────────────────────────

    async def execute(self, signal: Signal) -> list[TradeOrder]:
        if not signal.should_trade():
            return []

        loop = asyncio.get_event_loop()

        account, open_positions = await asyncio.gather(
            loop.run_in_executor(None, self.get_account),
            loop.run_in_executor(None, self.get_open_positions),
        )

        if not account:
            log.error("[EXECUTOR] Could not fetch account — aborting.")
            return []

        available_cash  = account["cash"]
        portfolio_value = account["portfolio_value"]
        current_exposure = await loop.run_in_executor(
            None, self.get_total_exposure_usd, open_positions
        )

        log.info(
            f"[EXECUTOR] Account: "
            f"portfolio=${portfolio_value:,.2f} | "
            f"cash=${available_cash:,.2f} | "
            f"exposure=${current_exposure:,.2f} "
            f"({current_exposure/portfolio_value*100:.1f}%)"
        )

        orders = []

        for asset in signal.tradeable_assets():
            if asset.symbol not in CRYPTO_SYMBOLS:
                log.warning(f"[EXECUTOR] Unsupported symbol {asset.symbol} — skipping.")
                continue

            position_usd = self.calculate_position_size(
                confidence=asset.confidence,
                urgency=signal.urgency,
                available_cash=available_cash,
                portfolio_value=portfolio_value,
                current_exposure=current_exposure,
            )

            if position_usd <= 0:
                continue

            order = await self._execute_asset(
                signal.post_id, asset, loop, open_positions, position_usd
            )

            if order:
                if order.status == "submitted":
                    current_exposure += position_usd
                orders.append(order)
                log.info(f"[EXECUTOR] {order.log_line()}")
                self._log_trade(order)

        return orders

    async def _execute_asset(
        self,
        post_id: str,
        asset: AssetSignal,
        loop,
        open_positions: dict,
        position_usd: float,
    ) -> Optional[TradeOrder]:
        """
        Executes a trade using trailing stop instead of fixed TP:

            1. Market entry order (fills immediately)
            2. Trailing stop (rides the move, exits on reversal)
            3. Hard stop loss (fixed safety net below trail)

        The trailing stop and hard stop run simultaneously.
        Whichever fires first closes the position — Alpaca handles the
        race condition by cancelling unfilled orders on position close.
        """
        if self.is_already_in_position(asset.symbol, open_positions):
            log.warning(f"[EXECUTOR] Already in {asset.symbol} — skipping.")
            return TradeOrder(
                signal_post_id=post_id,
                symbol=asset.symbol,
                direction=asset.direction,
                entry_price=0, qty=0, notional_usd=0,
                trail_pct=0, sl_price=0, sl_pct=0,
                entry_order_id=None, trail_order_id=None, sl_order_id=None,
                status="skipped",
                reason="already_in_position",
                submitted_utc=datetime.now(timezone.utc).isoformat(),
            )

        if not asset.levels:
            log.warning(f"[EXECUTOR] No levels for {asset.symbol} — skipping.")
            return None

        # Get live price
        entry_price = await loop.run_in_executor(
            None, self.get_latest_price, asset.symbol
        )
        if not entry_price:
            return TradeOrder(
                signal_post_id=post_id,
                symbol=asset.symbol,
                direction=asset.direction,
                entry_price=0, qty=0, notional_usd=0,
                trail_pct=asset.levels.take_profit_pct,
                sl_price=0,
                sl_pct=asset.levels.stop_loss_pct,
                entry_order_id=None, trail_order_id=None, sl_order_id=None,
                status="rejected",
                reason="price_fetch_failed",
                submitted_utc=datetime.now(timezone.utc).isoformat(),
            )

        # Use TP% from classifier as the trail percentage
        # SL% remains as a fixed hard stop safety net
        trail_pct = asset.levels.take_profit_pct
        _, sl_price = self.calculate_levels(
            asset.direction, entry_price, 0, asset.levels.stop_loss_pct
        )

        qty = round(position_usd / entry_price, 6)

        # Step 1: Entry order
        entry_order_id = await loop.run_in_executor(
            None,
            lambda: self.place_entry_order(asset.symbol, asset.direction, qty)
        )

        if not entry_order_id:
            return TradeOrder(
                signal_post_id=post_id,
                symbol=asset.symbol,
                direction=asset.direction,
                entry_price=entry_price, qty=qty, notional_usd=position_usd,
                trail_pct=trail_pct, sl_price=sl_price,
                sl_pct=asset.levels.stop_loss_pct,
                entry_order_id=None, trail_order_id=None, sl_order_id=None,
                status="rejected",
                reason="entry_order_failed",
                submitted_utc=datetime.now(timezone.utc).isoformat(),
            )

        # Small delay to let entry fill before placing exit orders
        await asyncio.sleep(1.5)

        # Step 2: Trailing stop
        trail_order_id = await loop.run_in_executor(
            None,
            lambda: self.place_trailing_stop(
                asset.symbol, asset.direction, qty, trail_pct
            )
        )

        # Step 3: Hard stop loss safety net
        sl_order_id = await loop.run_in_executor(
            None,
            lambda: self.place_hard_stop(
                asset.symbol, asset.direction, qty, sl_price
            )
        )

        status = "submitted" if entry_order_id else "rejected"

        if entry_order_id:
            self._pending_symbols.add(asset.symbol)
            self._pending_exposure += position_usd

        return TradeOrder(
            signal_post_id=post_id,
            symbol=asset.symbol,
            direction=asset.direction,
            entry_price=entry_price,
            qty=qty,
            notional_usd=position_usd,
            trail_pct=trail_pct,
            sl_price=sl_price,
            sl_pct=asset.levels.stop_loss_pct,
            entry_order_id=entry_order_id,
            trail_order_id=trail_order_id,
            sl_order_id=sl_order_id,
            status=status,
            reason="ok" if entry_order_id else "order_api_error",
            submitted_utc=datetime.now(timezone.utc).isoformat(),
        )

    def _log_trade(self, order: TradeOrder):
        try:
            with open(TRADE_LOG_FILE, "a") as f:
                f.write(json.dumps(order.to_dict()) + "\n")
        except Exception as e:
            log.warning(f"Failed to log trade: {e}")
