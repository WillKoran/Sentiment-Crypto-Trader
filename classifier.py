"""
classifier.py
-------------
Claude Sonnet crypto classifier.

Receives: Trump Truth Social post + rich market context
Returns:  BTC/USD and ETH/USD directional signals with TP/SL

The richer market context (fear/greed, funding rates, Polymarket)
lets Claude reason about:
  - Whether sentiment is at an extreme that amplifies or fades the signal
  - Whether positioning creates squeeze potential
  - Whether the move is already priced into prediction markets

Requirements:
    pip install anthropic
    export ANTHROPIC_API_KEY=sk-ant-...
"""

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional

import anthropic

log = logging.getLogger(__name__)

MODEL           = "claude-sonnet-4-5"
MAX_TOKENS      = 900
MIN_CONFIDENCE  = 0.70
SIGNAL_LOG_FILE = "signal_log.jsonl"


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class TradeLevel:
    take_profit_pct: float
    stop_loss_pct: float

    def reward_risk_ratio(self) -> float:
        if self.stop_loss_pct == 0:
            return 0
        return round(self.take_profit_pct / self.stop_loss_pct, 2)


@dataclass
class AssetSignal:
    symbol: str
    name: str
    direction: str
    confidence: float
    reasoning: str
    levels: Optional[TradeLevel] = None


@dataclass
class Signal:
    post_id: str
    post_text: str
    post_url: str
    post_source: str
    classified_utc: str
    latency_ms: float

    is_market_moving: bool
    overall_confidence: float
    urgency: int
    macro_summary: str

    assets: list[AssetSignal] = field(default_factory=list)
    risk_flags: list[str] = field(default_factory=list)
    position_actions: list[dict] = field(default_factory=list)

    def should_trade(self) -> bool:
        return (
            self.is_market_moving
            and self.overall_confidence >= MIN_CONFIDENCE
            and len(self.tradeable_assets()) > 0
        )

    def tradeable_assets(self) -> list[AssetSignal]:
        return [
            a for a in self.assets
            if a.direction != "neutral" and a.confidence >= MIN_CONFIDENCE
        ]

    def log_summary(self) -> str:
        lines = [
            f"\n{'='*64}",
            f"[SIGNAL] 🇺🇸 TRUTH SOCIAL | {self.classified_utc}",
            f"Post: {self.post_text[:120]}",
            f"Market moving: {self.is_market_moving} | "
            f"Confidence: {self.overall_confidence:.0%} | "
            f"Urgency: {self.urgency}/3",
            f"Macro: {self.macro_summary}",
            f"Latency: {self.latency_ms:.0f}ms",
        ]
        if self.risk_flags:
            lines.append(f"⚠️  Flags: {', '.join(self.risk_flags)}")

        if self.should_trade():
            lines.append("🟢 CRYPTO SIGNALS:")
            for a in self.tradeable_assets():
                lines.append(
                    f"  {a.direction.upper():5} {a.name} ({a.symbol}) "
                    f"— {a.confidence:.0%} confidence"
                )
                lines.append(f"         {a.reasoning}")
                if a.levels:
                    lines.append(
                        f"         TP: +{a.levels.take_profit_pct:.2f}%  "
                        f"SL: -{a.levels.stop_loss_pct:.2f}%  "
                        f"R/R: {a.levels.reward_risk_ratio():.1f}x"
                    )
        else:
            lines.append("⚪ NO TRADE")

        if self.position_actions:
            lines.append("🔄 POSITION ACTIONS:")
            for a in self.position_actions:
                emoji = {"close": "🚪", "add": "➕", "hold": "✋"}.get(
                    a.get("action", ""), "•"
                )
                lines.append(
                    f"  {emoji} {a.get('action','').upper()} "
                    f"{a.get('symbol')} — {a.get('reason','')}"
                )

        lines.append("="*64)
        return "\n".join(lines)

    def to_dict(self):
        return asdict(self)


# ── Prompt ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a senior crypto trader at a quantitative fund specializing in
macro-driven cryptocurrency trading around political signals.

You receive Trump Truth Social posts alongside rich market context including:
  - Live prices for BTC, ETH, and macro assets
  - Crypto Fear & Greed Index (market sentiment regime)
  - BTC perpetual futures funding rate (market positioning)
  - Key prediction market probabilities (what's already priced in)

USE THE CONTEXT TO ADJUST YOUR SIGNALS:

Fear & Greed:
  Extreme Fear (0-25):  Bullish signals carry MORE weight — market oversold
  Fear (26-46):         Bullish signals slightly amplified
  Neutral (47-54):      No adjustment
  Greed (55-75):        Bullish signals carry LESS weight — already extended
  Extreme Greed (76+):  Bullish signals likely to fade — reduce TP, tighten SL

Funding Rate:
  Heavily negative (<-0.05%): Bullish news = SQUEEZE potential — widen TP
  Negative (-0.05% to -0.01%): Moderate squeeze potential on bullish news
  Near zero: Balanced — no adjustment
  Positive (>0.01%): Bearish news reinforced — market already long
  Heavily positive (>0.05%): Bearish news = SQUEEZE potential — widen TP

Prediction markets:
  If a geopolitical event is already >70% probability → already_priced_in flag
  If probability recently spiked → signal may be partially priced in

CRYPTO SIGNAL FRAMEWORK:
  Risk-on (peace, trade deal, rate cuts, dollar weakness) → LONG BTC, LONG ETH
  Risk-off (tariffs, conflict, dollar strength) → SHORT BTC, SHORT ETH
  ETH moves approximately 1.3-1.5x BTC in same direction

TP/SL calibration:
  Base BTC move: 2-5% on macro Trump posts
  Squeeze scenario (negative funding + bullish news): widen TP to 6-8%
  Already priced in: tighten TP by 30%
  Extreme greed: tighten TP by 20%
  Minimum R/R: 1.5:1

Return ONLY valid JSON. No preamble, no markdown.

JSON schema:
{
  "is_market_moving": boolean,
  "overall_confidence": float (0.0-1.0),
  "urgency": integer (1=low, 2=medium, 3=high),
  "macro_summary": string (one sentence),
  "assets": [
    {
      "symbol": string ("BTC/USD" or "ETH/USD" only),
      "name": string ("Bitcoin" or "Ethereum"),
      "direction": string ("long", "short", or "neutral"),
      "confidence": float (0.0-1.0),
      "reasoning": string (one sentence — must reference market context),
      "take_profit_pct": float,
      "stop_loss_pct": float
    }
  ],
  "risk_flags": [string],
  "position_actions": [
    {
      "symbol": string ("BTC/USD" or "ETH/USD"),
      "action": string ("close", "add", "hold"),
      "reason": string (one sentence)
    }
  ]
}

Risk flags: sarcasm_detected, old_news_reshare, unverifiable_claim,
            already_priced_in, reversal_risk, ambiguous_intent, squeeze_setup

Rules:
- Endorsements, rallies, personal attacks, domestic legal → is_market_moving: false
- Only market_moving: true for macro policy (tariffs, Fed, geopolitics, dollar, crypto policy)
- Add squeeze_setup flag when negative funding + bullish signal or positive funding + bearish
- If no open positions → empty position_actions []
- If open positions provided → evaluate each for close/add/hold
- When not market moving → empty assets []"""


def build_prompt(
    post_text: str,
    post_url: str,
    scraped_utc: str,
    market_context: str,
    positions_context: str = "",
) -> str:
    position_section = (
        f"\n\nOPEN CRYPTO POSITIONS:\n{positions_context}"
        if positions_context else ""
    )
    return f"""Analyze this Trump Truth Social post for crypto trading signals.

POST:
{post_text}

Posted: {scraped_utc}
URL: {post_url}

MARKET CONTEXT:
{market_context}{position_section}

Return JSON only."""


def parse_response(raw: str) -> dict:
    clean = raw.strip()
    if "```" in clean:
        parts = clean.split("```")
        clean = parts[1] if len(parts) > 1 else clean
        if clean.startswith("json"):
            clean = clean[4:]
        clean = clean.strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        s, e = clean.find("{"), clean.rfind("}")
        if s != -1 and e > s:
            return json.loads(clean[s:e+1])
        raise


# ── Classifier ────────────────────────────────────────────────────────────────

class Classifier:

    def __init__(self):
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("ANTHROPIC_API_KEY not set.")
        self.client = anthropic.Anthropic(api_key=key)

    async def classify(
        self,
        post,
        market_state,
        open_positions: dict = None,
    ) -> Optional[Signal]:
        start = time.perf_counter()
        loop  = asyncio.get_event_loop()

        positions_context = ""
        if open_positions:
            lines = []
            for sym, qty in open_positions.items():
                direction = "LONG" if float(qty) > 0 else "SHORT"
                lines.append(f"- {direction} {sym} ({abs(float(qty)):.6f} units)")
            positions_context = "\n".join(lines)

        prompt = build_prompt(
            post_text=post.text,
            post_url=post.url,
            scraped_utc=post.scraped_utc,
            market_context=market_state.to_classifier_context(),
            positions_context=positions_context,
        )

        try:
            response = await loop.run_in_executor(
                None,
                lambda: self.client.messages.create(
                    model=MODEL,
                    max_tokens=MAX_TOKENS,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": prompt}]
                )
            )

            latency_ms = (time.perf_counter() - start) * 1000
            data       = parse_response(response.content[0].text)

            assets = []
            for a in data.get("assets", []):
                tp = float(a.get("take_profit_pct", 0))
                sl = float(a.get("stop_loss_pct", 0))
                levels = TradeLevel(tp, sl) if tp > 0 or sl > 0 else None
                assets.append(AssetSignal(
                    symbol=a["symbol"],
                    name=a["name"],
                    direction=a["direction"],
                    confidence=float(a["confidence"]),
                    reasoning=a["reasoning"],
                    levels=levels,
                ))

            signal = Signal(
                post_id=post.id,
                post_text=post.text,
                post_url=post.url,
                post_source=getattr(post, "source", "truth_social"),
                classified_utc=datetime.now(timezone.utc).isoformat(),
                latency_ms=latency_ms,
                is_market_moving=data.get("is_market_moving", False),
                overall_confidence=float(data.get("overall_confidence", 0)),
                urgency=int(data.get("urgency", 1)),
                macro_summary=data.get("macro_summary", ""),
                assets=assets,
                risk_flags=data.get("risk_flags", []),
                position_actions=data.get("position_actions", []),
            )

            self._log(signal)
            return signal

        except Exception as e:
            log.error(f"[CLASSIFIER] Error: {e}", exc_info=True)
            return None

    def _log(self, signal: Signal):
        try:
            with open(SIGNAL_LOG_FILE, "a") as f:
                f.write(json.dumps(signal.to_dict()) + "\n")
        except Exception as e:
            log.warning(f"Failed to log signal: {e}")
