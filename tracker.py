"""
Performance & Self-Evaluation Tracking Engine (tracker.py)

Persistently logs forecasts, entry signals, and SL/TP levels into SQLite.
Continuously evaluates historical prediction accuracy, realized win-rates,
and risk-reward performance against subsequent market prices.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger("PerformanceTracker")

DB_FILE = "trading_performance.db"


class SignalRecord(BaseModel):
    """Data record for a generated trade signal."""

    id: Optional[int] = None
    timestamp: datetime
    symbol: str
    timeframe: str
    direction: int  # 1 = BUY, -1 = SELL, 0 = NEUTRAL
    entry_price: float
    sl_price: float
    tp_price: float
    predicted_return: float
    confidence: float
    horizon_bars: int
    is_executed: bool = False
    status: str = "PENDING"  # PENDING, WON_TP, LOST_SL, EXPIRED, NEUTRAL
    exit_price: Optional[float] = None
    realized_pips: Optional[float] = None
    realized_return: Optional[float] = None
    evaluated_at: Optional[datetime] = None


class PerformanceSummary(BaseModel):
    """Summary metrics of the system's self-evaluation progress."""

    total_signals: int = 0
    evaluated_signals: int = 0
    pending_signals: int = 0
    directional_wins: int = 0
    directional_losses: int = 0
    win_rate_pct: float = 0.0
    directional_accuracy_pct: float = 0.0
    total_net_pips: float = 0.0
    profit_factor: float = 0.0
    avg_win_pips: float = 0.0
    avg_loss_pips: float = 0.0


class PerformanceTracker:
    """Manages SQLite storage and self-evaluation tracking."""

    def __init__(self, db_path: str = DB_FILE):
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        """Creates the signals and evaluations tables if not present."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    direction INTEGER NOT NULL,
                    entry_price REAL NOT NULL,
                    sl_price REAL NOT NULL,
                    tp_price REAL NOT NULL,
                    predicted_return REAL NOT NULL,
                    confidence REAL NOT NULL,
                    horizon_bars INTEGER NOT NULL,
                    is_executed INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    exit_price REAL,
                    realized_pips REAL,
                    realized_return REAL,
                    evaluated_at TEXT
                )
                """
            )
            conn.commit()

    def record_signal(
        self,
        symbol: str,
        timeframe: str,
        direction: int,
        entry_price: float,
        sl_price: float,
        tp_price: float,
        predicted_return: float,
        confidence: float,
        horizon_bars: int,
        is_executed: bool = False,
    ) -> int:
        """Records a new forecasted trade signal into the tracker."""
        now_str = datetime.now(timezone.utc).isoformat()
        initial_status = "PENDING" if direction != 0 else "NEUTRAL"

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO signals (
                    timestamp, symbol, timeframe, direction, entry_price,
                    sl_price, tp_price, predicted_return, confidence,
                    horizon_bars, is_executed, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    now_str,
                    symbol,
                    timeframe,
                    direction,
                    entry_price,
                    sl_price,
                    tp_price,
                    predicted_return,
                    confidence,
                    horizon_bars,
                    1 if is_executed else 0,
                    initial_status,
                ),
            )
            conn.commit()
            return cursor.lastrowid

    def evaluate_pending_signals(
        self,
        symbol: str,
        current_high: float,
        current_low: float,
        current_close: float,
        point: float,
        digits: int,
    ) -> int:
        """
        Self-evaluation pass: Checks active/pending signals against current market prices
        to determine if Take Profit or Stop Loss was hit, or if the horizon completed.
        """
        pip_factor = 10.0 * point if digits in (3, 5) else point
        evaluated_count = 0

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, direction, entry_price, sl_price, tp_price
                FROM signals
                WHERE symbol = ? AND status = 'PENDING'
                """,
                (symbol,),
            )
            rows = cursor.fetchall()

            for signal_id, direction, entry, sl, tp in rows:
                status = None
                exit_price = None
                pips = 0.0

                if direction == 1:  # BUY
                    if tp <= entry or sl >= entry:
                        logger.warning(
                            f"Signal #{signal_id} has invalid BUY SL/TP levels (entry={entry}, sl={sl}, tp={tp}). Skipping evaluation."
                        )
                        continue
                    if current_high >= tp:
                        status = "WON_TP"
                        exit_price = tp
                        pips = (tp - entry) / pip_factor
                    elif current_low <= sl:
                        status = "LOST_SL"
                        exit_price = sl
                        pips = (sl - entry) / pip_factor
                elif direction == -1:  # SELL
                    if tp >= entry or sl <= entry:
                        logger.warning(
                            f"Signal #{signal_id} has invalid SELL SL/TP levels (entry={entry}, sl={sl}, tp={tp}). Skipping evaluation."
                        )
                        continue
                    if current_low <= tp:
                        status = "WON_TP"
                        exit_price = tp
                        pips = (entry - tp) / pip_factor
                    elif current_high >= sl:
                        status = "LOST_SL"
                        exit_price = sl
                        pips = (entry - sl) / pip_factor

                if status:
                    now_str = datetime.now(timezone.utc).isoformat()
                    ret = (exit_price - entry) / entry if direction == 1 else (entry - exit_price) / entry
                    cursor.execute(
                        """
                        UPDATE signals
                        SET status = ?, exit_price = ?, realized_pips = ?, realized_return = ?, evaluated_at = ?
                        WHERE id = ?
                        """,
                        (status, exit_price, round(pips, 1), round(ret, 5), now_str, signal_id),
                    )
                    evaluated_count += 1

            conn.commit()
        return evaluated_count

    def clear_signals(self, symbol: Optional[str] = None) -> int:
        """Deletes all signal records or records for a specific symbol."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            if symbol:
                cursor.execute("DELETE FROM signals WHERE symbol = ?", (symbol,))
            else:
                cursor.execute("DELETE FROM signals")
            deleted = cursor.rowcount
            conn.commit()
        logger.info(f"Cleared {deleted} signals from {self.db_path} (filter symbol: {symbol})")
        return deleted

    def get_performance_summary(self, symbol: Optional[str] = None) -> PerformanceSummary:
        """Calculates historical self-evaluation scorecards and win-rates."""
        query = "SELECT status, realized_pips FROM signals WHERE status != 'NEUTRAL'"
        params = []
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()

        if not rows:
            return PerformanceSummary()

        total = len(rows)
        pending = sum(1 for s, _ in rows if s == "PENDING")
        evaluated = total - pending

        wins = [pips for s, pips in rows if s == "WON_TP" and pips is not None]
        losses = [abs(pips) for s, pips in rows if s == "LOST_SL" and pips is not None]

        win_count = len(wins)
        loss_count = len(losses)
        completed = win_count + loss_count

        win_rate = (win_count / completed * 100.0) if completed > 0 else 0.0
        total_pips = sum(pips for _, pips in rows if pips is not None)
        gross_profit = sum(wins)
        gross_loss = sum(losses)
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (99.0 if gross_profit > 0 else 1.0)

        return PerformanceSummary(
            total_signals=total,
            evaluated_signals=completed,
            pending_signals=pending,
            directional_wins=win_count,
            directional_losses=loss_count,
            win_rate_pct=round(win_rate, 1),
            directional_accuracy_pct=round(win_rate, 1),
            total_net_pips=round(total_pips, 1),
            profit_factor=round(profit_factor, 2),
            avg_win_pips=round(gross_profit / win_count, 1) if win_count > 0 else 0.0,
            avg_loss_pips=round(gross_loss / loss_count, 1) if loss_count > 0 else 0.0,
        )
