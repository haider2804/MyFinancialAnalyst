"""
FastAPI Live Backend & WebSocket Server (server.py)

Powers the interactive Quantitative Trading & Self-Evaluation Web Dashboard:
- Connects directly to MT5 terminal, Forecaster, and Jev Gatekeeper.
- Continuously streams live market ticks, model trajectory forecasts, and trade setups.
- Manages persistent SQLite self-evaluation trade records and win-rate statistics.
- Provides real-time WebSockets and REST APIs for instant symbol/timeframe switching.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import get_default_max_spread, settings
from forecaster import ForecastSignal, QuantitativeForecaster
from jev_guardrail import GuardrailDecision, JevGatekeeper
from mt5_connector import AccountState, MarketState, MT5Bridge
from tracker import PerformanceSummary, PerformanceTracker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("DashboardServer")

# ============================================================================
# Engine State Container
# ============================================================================

class AppState:
    """Holds runtime trading engine singletons and live telemetry."""

    def __init__(self):
        self.bridge = MT5Bridge(settings)
        self.forecaster: Optional[QuantitativeForecaster] = None
        self.gatekeeper: Optional[JevGatekeeper] = None
        self.tracker = PerformanceTracker()
        self.is_running = True
        self.is_scanning = True
        self.active_symbol = settings.SYMBOL
        self.active_timeframe = settings.TIMEFRAME
        settings.MAX_SPREAD_PIPS = get_default_max_spread(self.active_symbol)
        self.last_update = datetime.now(timezone.utc)
        self.iteration = 0
        self.latest_market: Optional[MarketState] = None
        self.latest_account: Optional[AccountState] = None
        self.latest_signal: Optional[ForecastSignal] = None
        self.latest_decision: Optional[GuardrailDecision] = None
        self.latest_setup: Dict[str, Any] = {}
        self.connected_clients: List[WebSocket] = []
        self.last_recorded_bar_time: Optional[Any] = None


state_container = AppState()


# ============================================================================
# Background Market Scanning Loop
# ============================================================================

async def market_worker():
    """Persistent background task that analyzes market prices and pushes telemetry."""
    logger.info("Initializing background quantitative market worker...")

    # Initialize MT5
    if not state_container.bridge.initialize():
        logger.error("Failed to connect MT5 bridge. Retrying in background...")

    # Load ML models
    try:
        state_container.forecaster = QuantitativeForecaster(settings)
        state_container.gatekeeper = JevGatekeeper(settings)
        logger.info("Forecaster and Jev Gatekeeper initialized successfully.")
    except Exception as e:
        logger.error(f"Error loading models: {e}", exc_info=True)

    # Note: Auto-seeding disabled to ensure strategy track record only contains authentic live signals
    first_cycle = True
    while state_container.is_running:
        try:
            if state_container.is_scanning and state_container.bridge.is_connected:
                await perform_analysis_cycle()
                if first_cycle:
                    first_cycle = False
                    try:
                        import webbrowser
                        webbrowser.open("http://localhost:8000")
                    except Exception:
                        pass
        except Exception as e:
            logger.error(f"Error in market worker cycle: {e}", exc_info=True)

        await asyncio.sleep(settings.POLL_INTERVAL_SECONDS)


def seed_historical_performance(target_symbol: Optional[str] = None):
    """Evaluates recent historical candles on startup so the Strategy Track Record tab is pre-populated."""
    try:
        sym = target_symbol or state_container.active_symbol
        tf = state_container.active_timeframe
        rates = state_container.bridge.get_rates(sym, tf, 80)
        market = state_container.bridge.get_market_state(sym)

        # Check existing records count
        summary = state_container.tracker.get_performance_summary(market.symbol)
        if summary.total_signals >= 5:
            return  # Already has sufficient historical data

        logger.info(f"Populating historical strategy track record for {market.symbol}...")
        point = market.point
        digits = market.digits
        pip_factor = 10.0 * point if digits in (3, 5) else point

        # Walk through historical bars to simulate past trade setups
        for i in range(30, len(rates) - 5, 4):
            sub_rates = rates.iloc[:i]
            entry_bar = rates.iloc[i]
            entry_price = float(entry_bar["close"])

            # Compute rough historical ATR
            tr = (sub_rates["high"] - sub_rates["low"]).tail(14).mean()
            sl_dist = 1.5 * tr
            tp_dist = 3.0 * tr

            # Detect momentum direction
            ret_5 = (entry_price - float(sub_rates["close"].iloc[-6])) / float(sub_rates["close"].iloc[-6])
            if abs(ret_5) < 0.0005:
                continue

            direction = 1 if ret_5 > 0 else -1
            sl_price = round(entry_price - sl_dist if direction == 1 else entry_price + sl_dist, digits)
            tp_price = round(entry_price + tp_dist if direction == 1 else entry_price - tp_dist, digits)

            # Record signal
            sig_id = state_container.tracker.record_signal(
                symbol=market.symbol,
                timeframe=tf,
                direction=direction,
                entry_price=entry_price,
                sl_price=sl_price,
                tp_price=tp_price,
                predicted_return=round(ret_5, 5),
                confidence=round(min(0.92, 0.65 + abs(ret_5) * 50), 3),
                horizon_bars=5,
                is_executed=True,
            )

            # Check outcome against subsequent candles
            forward_bars = rates.iloc[i + 1: i + 6]
            for _, fbar in forward_bars.iterrows():
                high = float(fbar["high"])
                low = float(fbar["low"])
                if direction == 1:
                    if high >= tp_price:
                        state_container.tracker.evaluate_pending_signals(market.symbol, high, low, tp_price, point, digits)
                        break
                    elif low <= sl_price:
                        state_container.tracker.evaluate_pending_signals(market.symbol, high, low, sl_price, point, digits)
                        break
                else:
                    if low <= tp_price:
                        state_container.tracker.evaluate_pending_signals(market.symbol, high, low, tp_price, point, digits)
                        break
                    elif high >= sl_price:
                        state_container.tracker.evaluate_pending_signals(market.symbol, high, low, sl_price, point, digits)
                        break

        logger.info("Historical performance evaluation pre-populated successfully.")
    except Exception as e:
        logger.warning(f"Notice during historical seed: {e}")


async def perform_analysis_cycle():
    """Runs a single live forecast iteration and broadcasts to all WebSocket clients."""
    sym = state_container.active_symbol
    tf = state_container.active_timeframe
    count = settings.HISTORY_BARS

    # 1. Fetch live market & account
    market = state_container.bridge.get_market_state(sym)
    account = state_container.bridge.get_account_state()
    open_positions = state_container.bridge.get_open_positions(sym)

    # 2. Fetch candles
    rates = state_container.bridge.get_rates(sym, tf, count)

    # 3. Evaluate pending signals
    latest_bar = rates.iloc[-1]
    state_container.tracker.evaluate_pending_signals(
        symbol=market.symbol,
        current_high=float(latest_bar["high"]),
        current_low=float(latest_bar["low"]),
        current_close=float(latest_bar["close"]),
        point=market.point,
        digits=market.digits,
    )

    # 4. Compute forecast
    signal_obj = state_container.forecaster.forecast(rates, market.symbol)

    # 5. Jev Gatekeeper
    decision = state_container.gatekeeper.evaluate(market, signal_obj, open_positions, account)

    # 6. Trade Setup Calculation
    sl_dist = settings.ATR_SL_MULTIPLIER * signal_obj.atr
    tp_dist = settings.ATR_TP_MULTIPLIER * signal_obj.atr
    pip_factor = 10.0 * market.point if market.digits in (3, 5) else market.point
    sl_pips = round(sl_dist / pip_factor, 1)
    tp_pips = round(tp_dist / pip_factor, 1)

    is_buy = (signal_obj.direction == 1)
    is_sell = (signal_obj.direction == -1)
    entry_price = market.ask if is_buy else market.bid
    sl_price = round(entry_price - sl_dist if is_buy else entry_price + sl_dist, market.digits)
    tp_price = round(entry_price + tp_dist if is_buy else entry_price - tp_dist, market.digits)

    setup = {
        "has_trade": decision.is_approved and (decision.approved_action in ("EXECUTE_BUY", "EXECUTE_SELL")),
        "action": "BUY" if is_buy else ("SELL" if is_sell else "HOLD"),
        "symbol": market.symbol,
        "entry_price": entry_price,
        "sl_price": sl_price,
        "tp_price": tp_price,
        "sl_pips": sl_pips,
        "tp_pips": tp_pips,
        "risk_reward": f"1 : {settings.ATR_TP_MULTIPLIER / settings.ATR_SL_MULTIPLIER:.1f}",
        "recommended_volume": settings.DEFAULT_VOLUME,
    }

    # Record signal in SQLite only once per candle bar (prevents duplicate spam on 15s polls)
    current_bar_time = rates.index[-1]
    if signal_obj.direction != 0 and current_bar_time != state_container.last_recorded_bar_time:
        state_container.last_recorded_bar_time = current_bar_time
        state_container.tracker.record_signal(
            symbol=market.symbol,
            timeframe=tf,
            direction=signal_obj.direction,
            entry_price=entry_price,
            sl_price=sl_price,
            tp_price=tp_price,
            predicted_return=signal_obj.predicted_return,
            confidence=signal_obj.model_confidence,
            horizon_bars=settings.FORECAST_HORIZON,
            is_executed=decision.is_approved,
        )

    # Store state
    state_container.iteration += 1
    state_container.last_update = datetime.now(timezone.utc)
    state_container.latest_market = market
    state_container.latest_account = account
    state_container.latest_signal = signal_obj
    state_container.latest_decision = decision
    state_container.latest_setup = setup

    # Broadcast payload to WebSockets
    payload = build_full_state_dict()
    await broadcast_ws(payload)

    # Clean, non-wrapping console telemetry
    perf = payload["performance"]
    dir_str = "BUY (+1)" if signal_obj.direction == 1 else ("SELL (-1)" if signal_obj.direction == -1 else "NEUTRAL (0)")
    logger.info(
        f"Iter #{state_container.iteration:03d} | {market.symbol} ({tf}) | Bid: {market.bid:.{market.digits}f} Ask: {market.ask:.{market.digits}f} "
        f"| Spread: {market.spread_pips:.1f} pips | Quant: {dir_str} ({signal_obj.model_confidence*100:.0f}%) "
        f"| Jev: {decision.approved_action} | Strategy: {perf['win_rate_pct']:.1f}% Win ({perf['directional_wins']}W-{perf['directional_losses']}L | Net: {perf['total_net_pips']:+.1f} pips)"
    )


def build_full_state_dict() -> Dict[str, Any]:
    """Serializes the complete runtime state into JSON-compatible dictionary."""
    m = state_container.latest_market
    a = state_container.latest_account
    s = state_container.latest_signal
    d = state_container.latest_decision
    setup = state_container.latest_setup
    perf = state_container.tracker.get_performance_summary(m.symbol if m else None)

    return {
        "iteration": state_container.iteration,
        "timestamp": state_container.last_update.isoformat(),
        "is_scanning": state_container.is_scanning,
        "active_symbol": state_container.active_symbol,
        "active_timeframe": state_container.active_timeframe,
        "market": m.model_dump(mode="json") if m else None,
        "account": a.model_dump(mode="json") if a else None,
        "forecast": s.model_dump(mode="json") if s else None,
        "decision": d.model_dump(mode="json") if d else None,
        "trade_setup": setup,
        "performance": perf.model_dump(mode="json"),
    }


async def broadcast_ws(data: Dict[str, Any]):
    """Sends state update to all active WebSocket clients."""
    if not state_container.connected_clients:
        return
    msg = json.dumps(data, default=str)
    for ws in list(state_container.connected_clients):
        try:
            await ws.send_text(msg)
        except Exception:
            state_container.connected_clients.remove(ws)


# ============================================================================
# FastAPI Lifespan & Application Setup
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: launch background worker task
    worker_task = asyncio.create_task(market_worker())
    yield
    # Shutdown: terminate worker and MT5
    state_container.is_running = False
    worker_task.cancel()
    state_container.bridge.shutdown()


app = FastAPI(title="TypeSafe Jev Quantitative Analyst", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/api/state")
async def get_state():
    """Returns the latest unified market and strategy state."""
    return JSONResponse(build_full_state_dict())


@app.get("/api/history")
async def get_history(count: int = 60):
    """Returns recent candle history for the interactive chart."""
    try:
        sym = state_container.active_symbol
        tf = state_container.active_timeframe
        rates = state_container.bridge.get_rates(sym, tf, count)

        candles = []
        for time_idx, row in rates.iterrows():
            candles.append({
                "time": int(time_idx.timestamp()),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": int(row["tick_volume"]),
            })
        return JSONResponse({"symbol": sym, "candles": candles})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/signals")
async def get_signals(limit: int = 50):
    """Returns past evaluated signals for the Strategy Track Record tab."""
    import sqlite3
    db_path = state_container.tracker.db_path
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, timestamp, symbol, timeframe, direction,
                   entry_price, sl_price, tp_price, predicted_return,
                   confidence, status, exit_price, realized_pips, realized_return, evaluated_at
            FROM signals
            WHERE status != 'NEUTRAL'
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = [dict(row) for row in cursor.fetchall()]
    return JSONResponse({"signals": rows})


class SettingsUpdateRequest(BaseModel):
    symbol: Optional[str] = None
    timeframe: Optional[str] = None
    is_scanning: Optional[bool] = None


@app.post("/api/settings")
async def update_settings(req: SettingsUpdateRequest):
    """Dynamically updates trading symbol, timeframe, or pauses/resumes scanning."""
    if req.symbol:
        state_container.active_symbol = req.symbol
        settings.SYMBOL = req.symbol
        settings.MAX_SPREAD_PIPS = get_default_max_spread(req.symbol)
        state_container.last_recorded_bar_time = None
        logger.info(f"Active symbol switched to: {req.symbol}")

    if req.timeframe:
        state_container.active_timeframe = req.timeframe
        settings.TIMEFRAME = req.timeframe
        state_container.last_recorded_bar_time = None
        logger.info(f"Active timeframe switched to: {req.timeframe}")

    if req.is_scanning is not None:
        state_container.is_scanning = req.is_scanning
        logger.info(f"Engine scanning state set to: {req.is_scanning}")

    # Trigger immediate analysis cycle on setting change
    asyncio.create_task(perform_analysis_cycle())
    return JSONResponse({"status": "updated", "symbol": state_container.active_symbol, "timeframe": state_container.active_timeframe, "is_scanning": state_container.is_scanning})


@app.post("/api/scan-now")
async def trigger_scan():
    """Triggers an immediate evaluation cycle."""
    asyncio.create_task(perform_analysis_cycle())
    return JSONResponse({"status": "scan_initiated"})


@app.post("/api/reset-signals")
async def reset_signals(symbol: Optional[str] = None):
    """Clears recorded strategy signals so user can track performance from clean slate."""
    deleted = state_container.tracker.clear_signals(symbol)
    state_container.last_recorded_bar_time = None
    payload = build_full_state_dict()
    await broadcast_ws(payload)
    return JSONResponse({"status": "cleared", "deleted_count": deleted})


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Real-time streaming channel for live prices, heartbeats, and forecast changes."""
    await websocket.accept()
    state_container.connected_clients.append(websocket)
    # Send immediate state upon connection
    await websocket.send_text(json.dumps(build_full_state_dict(), default=str))
    try:
        while True:
            # Keepalive receiver
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text(json.dumps({"type": "pong", "time": datetime.now(timezone.utc).isoformat()}))
    except WebSocketDisconnect:
        if websocket in state_container.connected_clients:
            state_container.connected_clients.remove(websocket)


# Mount static assets directory
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
async def serve_index():
    """Serves the primary Cockpit Dashboard HTML."""
    index_file = os.path.join(static_dir, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return JSONResponse({"message": "Dashboard UI initializing. Please refresh in a moment."})


if __name__ == "__main__":
    import uvicorn
    # Record PID for stop.bat
    pid_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.pid")
    try:
        with open(pid_file, "w") as f:
            f.write(str(os.getpid()))
    except Exception:
        pass

    try:
        uvicorn.run("server:app", host="127.0.0.1", port=8000, log_level="info", reload=False)
    finally:
        if os.path.exists(pid_file):
            try:
                os.remove(pid_file)
            except Exception:
                pass
