"""
Module C: TypeSafe AI Jev Gatekeeper (jev_guardrail.py)

Integrates TypeSafe AI's Jev model via the official `typesafe_sdk`:
- Leverages typed primitives: Choice, Noul, Score.
- Evaluates unified market state: spread, open positions, drawdown, quantitative forecast,
  macroeconomic and session context.
- Enforces strict typing and risk bounds:
  - Rejects execution if Jev's `risk_acceptable` probability < 0.85.
  - Rejects execution if `approved_action` contradicts the quantitative forecast.
  - Rejects execution if hard risk guardrails (spread, exposure, regime) are breached.
- Provides fallback local deterministic System 1 engine when running offline/dry-run.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
from pydantic import BaseModel, Field

from config import SystemSettings, settings, get_default_max_spread
from forecaster import ForecastSignal
from mt5_connector import AccountState, MarketState, PositionInfo

logger = logging.getLogger("JevGuardrail")

# TypeSafe SDK Import
try:
    from typesafe_sdk import Choice, Noul, Score, TypeSafeClient
    from typesafe_sdk._core.errors import TypeSafeError
    HAS_TYPESAFE_SDK = True
except ImportError:
    HAS_TYPESAFE_SDK = False
    logger.warning("typesafe-sdk not installed; running in local deterministic guardrail mode.")


# ============================================================================
# Gatekeeper Response Models
# ============================================================================

class GuardrailDecision(BaseModel):
    """Deterministic, strictly typed gatekeeper decision for order execution."""

    is_approved: bool = Field(
        ...,
        description="Master approval flag. True ONLY if all quantitative & Jev checks pass.",
    )
    approved_action: str = Field(
        ...,
        description="One of: 'EXECUTE_BUY', 'EXECUTE_SELL', 'HOLD_STAND_DOWN'.",
    )
    risk_acceptable_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Calibrated probability (0.0 to 1.0) from Jev Noul question.",
    )
    market_regime_score: float = Field(
        ...,
        description="Urgency/volatility score from Jev Score question.",
    )
    quant_direction: int = Field(..., description="Direction from quantitative forecaster (-1, 0, 1).")
    quant_confidence: float = Field(..., description="Confidence from quantitative forecaster.")
    spread_pips: float = Field(..., description="Observed market spread in pips.")
    execution_permitted: bool = Field(
        ...,
        description="Whether an actual trade dispatch should occur.",
    )
    rejection_reasons: List[str] = Field(
        default_factory=list,
        description="Explanatory audit list of all failed guardrails.",
    )
    jev_raw_response: Dict[str, Any] = Field(
        default_factory=dict,
        description="Raw telemetry from TypeSafe AI Jev model.",
    )


# ============================================================================
# Session / Macro Context Helper
# ============================================================================

def get_session_context() -> Dict[str, Any]:
    """Identifies the current active global Forex trading session."""
    now_utc = datetime.now(timezone.utc)
    hour = now_utc.hour

    # Forex session hours in UTC:
    # Tokyo/Asian: 00:00 - 09:00 UTC
    # London: 08:00 - 16:30 UTC
    # New York: 13:00 - 21:00 UTC
    # London/NY Overlap (Peak Liquidity): 13:00 - 16:30 UTC
    if 13 <= hour <= 16:
        session_name = "LONDON_NEW_YORK_OVERLAP"
        is_high_liquidity = True
    elif 8 <= hour < 13:
        session_name = "LONDON_SESSION"
        is_high_liquidity = True
    elif 17 <= hour <= 21:
        session_name = "NEW_YORK_AFTERNOON"
        is_high_liquidity = False
    elif 0 <= hour <= 7:
        session_name = "ASIAN_SESSION"
        is_high_liquidity = False
    else:
        session_name = "ROLLOVER_LOW_LIQUIDITY"
        is_high_liquidity = False

    return {
        "utc_time": now_utc.isoformat(),
        "utc_hour": hour,
        "session": session_name,
        "is_high_liquidity": is_high_liquidity,
    }


# ============================================================================
# TypeSafe AI Jev Gatekeeper
# ============================================================================

class JevGatekeeper:
    """
    Acts as a deterministic System 1 risk and execution supervisor.
    Evaluates market state, quantitative forecast, and risk limits via TypeSafe Jev.
    """

    def __init__(self, config: Optional[SystemSettings] = None, api_key: Optional[str] = None):
        self.config = config or settings
        self.api_key = api_key or self.config.TYPESAFE_API_KEY or os.environ.get("TYPESAFE_API_KEY")
        self.client: Optional[TypeSafeClient] = None
        self._init_client()

    def _init_client(self) -> None:
        """Initializes TypeSafeClient if API key is provided and SDK is present."""
        if not HAS_TYPESAFE_SDK:
            return

        if self.api_key:
            try:
                self.client = TypeSafeClient(api_key=self.api_key)
                logger.info("TypeSafeClient successfully initialized with provided API key.")
            except Exception as e:
                logger.error(f"Failed to initialize TypeSafeClient: {e}")
                self.client = None
        else:
            logger.info(
                "No TYPESAFE_API_KEY detected. Running in high-fidelity local deterministic System 1 mode."
            )
            self.client = None

    def construct_state(
        self,
        market: MarketState,
        signal: ForecastSignal,
        open_positions: List[PositionInfo],
        account: AccountState,
    ) -> Dict[str, Any]:
        """Synthesizes unified market and portfolio state for Jev evaluation."""
        session = get_session_context()

        # Calculate open risk and current instrument exposure
        symbol_positions = [p for p in open_positions if p.symbol == market.symbol]
        total_open_volume = sum(p.volume for p in symbol_positions)
        total_open_profit = sum(p.profit for p in symbol_positions)

        # Spread to ATR ratio
        spread_to_atr_ratio = round((market.spread_points * market.point) / max(1e-6, signal.atr), 4)

        state = {
            "symbol": market.symbol,
            "bid": market.bid,
            "ask": market.ask,
            "spread_pips": market.spread_pips,
            "spread_to_atr_ratio": spread_to_atr_ratio,
            "digits": market.digits,
            "open_positions_count": len(symbol_positions),
            "total_open_volume": round(total_open_volume, 2),
            "unrealized_profit": round(total_open_profit, 2),
            "account_equity": round(account.equity, 2),
            "account_free_margin": round(account.margin_free, 2),
            "quant_forecast": {
                "direction": "BULLISH" if signal.direction == 1 else ("BEARISH" if signal.direction == -1 else "NEUTRAL"),
                "predicted_return_pct": round(signal.predicted_return * 100, 4),
                "model_confidence": round(signal.model_confidence, 4),
                "chronos_p10_return_pct": round(signal.chronos_p10_return * 100, 4),
                "chronos_p90_return_pct": round(signal.chronos_p90_return * 100, 4),
                "models_agree": signal.agreement,
            },
            "macro_session": session,
            "risk_limits": {
                "max_spread_pips": get_default_max_spread(market.symbol, self.config.MAX_SPREAD_PIPS),
                "max_positions": self.config.MAX_CONCURRENT_POSITIONS,
                "min_risk_prob": self.config.MIN_RISK_ACCEPTABLE_PROB,
            },
        }
        return state

    def evaluate(
        self,
        market: MarketState,
        signal: ForecastSignal,
        open_positions: List[PositionInfo],
        account: AccountState,
    ) -> GuardrailDecision:
        """
        Submits market state to TypeSafe AI Jev System 1 supervisor,
        then validates returned primitives against strict risk thresholds.
        """
        state = self.construct_state(market, signal, open_positions, account)
        rejection_reasons: List[str] = []

        # ---------------------------------------------------------------------
        # 1. HARD PRE-CHECKS (Quantitative safety bounds)
        # ---------------------------------------------------------------------
        effective_max_spread = get_default_max_spread(market.symbol, self.config.MAX_SPREAD_PIPS)
        if market.spread_pips > effective_max_spread:
            rejection_reasons.append(
                f"Market spread {market.spread_pips:.2f} pips exceeds limit {effective_max_spread:.2f} pips for {market.symbol}."
            )

        symbol_positions = [p for p in open_positions if p.symbol == market.symbol]
        if len(symbol_positions) >= self.config.MAX_CONCURRENT_POSITIONS:
            rejection_reasons.append(
                f"Open positions ({len(symbol_positions)}) reached max limit ({self.config.MAX_CONCURRENT_POSITIONS})."
            )

        if signal.direction == 0:
            rejection_reasons.append("Quantitative forecaster indicates NEUTRAL / Stand Down.")

        if signal.model_confidence < self.config.MIN_MODEL_CONFIDENCE:
            rejection_reasons.append(
                f"Model confidence {signal.model_confidence:.2f} is below required {self.config.MIN_MODEL_CONFIDENCE:.2f}."
            )

        # ---------------------------------------------------------------------
        # 2. TYPESAFE AI JEV SYSTEM 1 PASS
        # ---------------------------------------------------------------------
        raw_jev: Dict[str, Any] = {}
        action_choice: str = "HOLD_STAND_DOWN"
        risk_acceptable_prob: float = 0.0
        market_regime_val: float = 2.0

        if self.client is not None and HAS_TYPESAFE_SDK:
            try:
                # Construct Typed Primitives
                questions = {
                    "approved_action": Choice(
                        instructions=(
                            "Based on the unified market state, spread conditions, open exposure, "
                            "and quantitative forecast, what is the approved execution action?"
                        ),
                        criteria={
                            "EXECUTE_BUY": "Spread is acceptable, forecast is bullish with high confidence, no risk breaches.",
                            "EXECUTE_SELL": "Spread is acceptable, forecast is bearish with high confidence, no risk breaches.",
                            "HOLD_STAND_DOWN": "Spread is wide, signal is weak or neutral, conflicting direction, or risk is unacceptable.",
                        },
                    ),
                    "risk_acceptable": Noul(
                        instructions=(
                            "Is the risk acceptable for new order execution without violating conservative "
                            "drawdown, spread, or position safety limits?"
                        )
                    ),
                    "market_regime": Score(
                        instructions="Evaluate current market regime urgency and volatility.",
                        criteria=["calm", "normal", "elevated", "high", "extreme"],
                    ),
                }

                response = self.client.system_one(
                    state=state,
                    questions=questions,
                    model=self.config.TYPESAFE_MODEL_VERSION,
                )

                ans_action = response.answers["approved_action"]
                ans_risk = response.answers["risk_acceptable"]
                ans_regime = response.answers["market_regime"]

                action_choice = str(ans_action.choice)
                risk_acceptable_prob = float(ans_risk.noul)
                market_regime_val = float(ans_regime.score)

                raw_jev = {
                    "model": response.model,
                    "action_choice": action_choice,
                    "action_confidence": getattr(ans_action, "confidence", 1.0),
                    "risk_acceptable_prob": risk_acceptable_prob,
                    "market_regime_score": market_regime_val,
                }
                logger.info(
                    f"TypeSafe Jev Response | Action: {action_choice} | Risk Acceptable Prob: {risk_acceptable_prob:.4f} | Regime: {market_regime_val:.2f}"
                )

            except Exception as e:
                logger.error(f"TypeSafe Jev API call failed: {e}. Falling back to deterministic supervisor.")
                action_choice, risk_acceptable_prob, market_regime_val = self._fallback_deterministic_supervisor(
                    state, signal, market, len(symbol_positions)
                )
                raw_jev = {"mode": "fallback_on_error", "error": str(e)}
        else:
            # Deterministic System 1 Local Engine
            action_choice, risk_acceptable_prob, market_regime_val = self._fallback_deterministic_supervisor(
                state, signal, market, len(symbol_positions)
            )
            raw_jev = {
                "mode": "deterministic_local_engine",
                "action_choice": action_choice,
                "risk_acceptable_prob": risk_acceptable_prob,
                "market_regime_score": market_regime_val,
            }

        # ---------------------------------------------------------------------
        # 3. STRICT TYPING & GATEKEEPING ENFORCEMENT
        # ---------------------------------------------------------------------
        # Rule A: Reject execution if Jev's risk_acceptable probability falls below 0.85
        if risk_acceptable_prob < self.config.MIN_RISK_ACCEPTABLE_PROB:
            rejection_reasons.append(
                f"Jev risk_acceptable probability ({risk_acceptable_prob:.3f}) falls below threshold ({self.config.MIN_RISK_ACCEPTABLE_PROB:.2f})."
            )

        # Rule B: Reject execution if approved action contradicts quantitative forecast
        if signal.direction == 1 and action_choice != "EXECUTE_BUY":
            rejection_reasons.append(
                f"Jev approved action '{action_choice}' contradicts Bullish (+1) quantitative forecast."
            )
        elif signal.direction == -1 and action_choice != "EXECUTE_SELL":
            rejection_reasons.append(
                f"Jev approved action '{action_choice}' contradicts Bearish (-1) quantitative forecast."
            )
        elif signal.direction == 0 and action_choice != "HOLD_STAND_DOWN":
            rejection_reasons.append(
                f"Jev approved action '{action_choice}' contradicts Neutral (0) quantitative forecast."
            )

        # Rule C: Extreme market regime rejection during off-hours
        session_info = state["macro_session"]
        if market_regime_val >= 4.0 and not session_info["is_high_liquidity"]:
            rejection_reasons.append(
                f"Market regime score ({market_regime_val:.2f}) indicates extreme volatility during low liquidity window."
            )

        # Determine overall approval
        is_approved = (len(rejection_reasons) == 0) and (action_choice in ("EXECUTE_BUY", "EXECUTE_SELL"))
        execution_permitted = is_approved

        decision = GuardrailDecision(
            is_approved=is_approved,
            approved_action=action_choice,
            risk_acceptable_probability=round(risk_acceptable_prob, 4),
            market_regime_score=round(market_regime_val, 2),
            quant_direction=signal.direction,
            quant_confidence=signal.model_confidence,
            spread_pips=market.spread_pips,
            execution_permitted=execution_permitted,
            rejection_reasons=rejection_reasons,
            jev_raw_response=raw_jev,
        )

        if is_approved:
            logger.info(f"*** JEV GATEKEEPER APPROVAL: {action_choice} APPROVED FOR EXECUTION ***")
        else:
            logger.info(
                f"Jev Gatekeeper Stand Down | Action: {action_choice} | Reasons: {'; '.join(rejection_reasons)}"
            )

        return decision

    def _fallback_deterministic_supervisor(
        self,
        state: Dict[str, Any],
        signal: ForecastSignal,
        market: MarketState,
        open_count: int,
    ) -> Tuple[str, float, float]:
        """
        High-fidelity local deterministic System 1 engine matching Jev's evaluation schema.
        Provides robust calibrated decision making when running offline or in initial testing.
        """
        spread = market.spread_pips
        max_spread = get_default_max_spread(market.symbol, self.config.MAX_SPREAD_PIPS)
        conf = signal.model_confidence
        direction = signal.direction
        is_liquid = state["macro_session"]["is_high_liquidity"]

        # Base risk probability computation
        spread_penalty = max(0.0, (spread / max_spread) - 0.7) * 0.40
        position_penalty = 0.35 if open_count >= self.config.MAX_CONCURRENT_POSITIONS else 0.0
        confidence_boost = (conf - 0.50) * 0.50 if conf > 0.50 else -0.30
        liquidity_penalty = 0.05 if not is_liquid else 0.0

        risk_prob = float(np.clip(0.88 + confidence_boost - spread_penalty - position_penalty - liquidity_penalty, 0.05, 0.96))

        # Regime score computation (0: calm, 4: extreme)
        spread_to_atr = state.get("spread_to_atr_ratio", 0.05)
        regime_score = float(np.clip(1.5 + (spread_to_atr * 15.0) + (0.8 if not is_liquid else 0.0), 0.5, 4.8))

        # Determine approved action
        if spread <= max_spread and open_count < self.config.MAX_CONCURRENT_POSITIONS and conf >= self.config.MIN_MODEL_CONFIDENCE:
            if direction == 1 and risk_prob >= self.config.MIN_RISK_ACCEPTABLE_PROB:
                action = "EXECUTE_BUY"
            elif direction == -1 and risk_prob >= self.config.MIN_RISK_ACCEPTABLE_PROB:
                action = "EXECUTE_SELL"
            else:
                action = "HOLD_STAND_DOWN"
        else:
            action = "HOLD_STAND_DOWN"

        return action, risk_prob, regime_score
