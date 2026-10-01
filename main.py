"""
Module D: Orchestrator, Trade Advisor & Self-Evaluating CLI (main.py)

Master orchestration loop for the Quantitative Trading & Forecasting System:
- Connects to host MetaTrader 5 terminal.
- Continuously polls live ticks, account state, and historical candle bars.
- Generates combined Tabular ML + Chronos Foundation Model price movement forecasts.
- Passes unified market & risk state to TypeSafe AI Jev Gatekeeper.
- Calculates exact Entry, Stop Loss (SL), and Take Profit (TP) levels.
- Tracks and self-evaluates its own historical accuracy and win rate in SQLite.
- Supports ADVISORY (signals only), DRY-RUN (simulated execution), and LIVE modes.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from config import SystemSettings, settings
from forecaster import ForecastSignal, QuantitativeForecaster
from jev_guardrail import GuardrailDecision, JevGatekeeper
from mt5_connector import AccountState, MarketState, MT5Bridge, OrderResult
from tracker import PerformanceSummary, PerformanceTracker

# ============================================================================
# Console & Logging Setup
# ============================================================================

console = Console()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("MainOrchestrator")


# ============================================================================
# Visual Status Dashboard & Trade Card
# ============================================================================

def render_dashboard(
    market: MarketState,
    account: AccountState,
    signal: ForecastSignal,
    decision: GuardrailDecision,
    performance: PerformanceSummary,
    sl_price: float,
    tp_price: float,
    sl_pips: float,
    tp_pips: float,
    mode: str,
    iteration: int,
) -> None:
    """Renders a rich, real-time terminal dashboard with trade setup and self-evaluation progress."""
    table = Table(
        title=f"Iter #{iteration} | {market.symbol} ({settings.TIMEFRAME}) | {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        expand=True,
    )

    table.add_column("Category", style="cyan bold", width=22)
    table.add_column("Metrics & Values", style="white")

    # Account info
    table.add_row(
        "Account Health",
        f"Login: {account.login} | Equity: [green]${account.equity:,.2f}[/green] | Balance: ${account.balance:,.2f} | Free Margin: ${account.margin_free:,.2f}",
    )

    # Market tick
    spread_color = "green" if market.spread_pips <= 2.0 else ("yellow" if market.spread_pips <= 3.0 else "red")
    table.add_row(
        "Market State",
        f"Bid: {market.bid:.{market.digits}f} | Ask: {market.ask:.{market.digits}f} | Spread: [{spread_color}]{market.spread_pips:.1f} pips[/{spread_color}] ({market.spread_points:.0f} pts)",
    )

    # Quantitative forecaster
    dir_str = (
        "[bold green]BULLISH (+1)[/bold green]" if signal.direction == 1
        else ("[bold red]BEARISH (-1)[/bold red]" if signal.direction == -1
        else "[yellow]NEUTRAL (0)[/yellow]")
    )
    conf_color = "green" if signal.model_confidence >= 0.70 else "yellow"
    table.add_row(
        "Quant Forecast",
        f"Direction: {dir_str} | Exp Return: [cyan]{signal.predicted_return * 100:+.3f}%[/cyan] | "
        f"Confidence: [{conf_color}]{signal.model_confidence * 100:.1f}%[/{conf_color}] | "
        f"Tabular: {signal.tabular_direction} ({signal.tabular_confidence * 100:.0f}%) | "
        f"Chronos: {signal.chronos_direction} ({signal.chronos_confidence * 100:.0f}%)",
    )

    # Jev gatekeeper
    action_color = (
        "bold green" if decision.approved_action == "EXECUTE_BUY"
        else ("bold red" if decision.approved_action == "EXECUTE_SELL"
        else "yellow")
    )
    risk_color = "green" if decision.risk_acceptable_probability >= 0.85 else "red"
    approval_banner = (
        "[bold white on green]  TRADE APPROVED  [/bold white on green]"
        if decision.is_approved
        else "[bold white on red]  STAND DOWN  [/bold white on red]"
    )

    table.add_row(
        "Jev Supervisor",
        f"Action: [{action_color}]{decision.approved_action}[/{action_color}] | "
        f"Risk Acceptable: [{risk_color}]{decision.risk_acceptable_probability * 100:.1f}%[/{risk_color}] "
        f"(Req >= 85%) | Regime: {decision.market_regime_score:.2f}/5.0 | {approval_banner}",
    )

    # Trade Setup Card (Calculated Entry, SL, TP)
    if decision.approved_action in ("EXECUTE_BUY", "EXECUTE_SELL"):
        action_word = "BUY" if decision.approved_action == "EXECUTE_BUY" else "SELL"
        entry_price = market.ask if action_word == "BUY" else market.bid
        table.add_row(
            "[bold white on blue] TRADE SETUP [/bold white on blue]",
            f"[bold {action_color}]{action_word} {market.symbol}[/bold {action_color}] @ {entry_price:.{market.digits}f} | "
            f"SL: [bold red]{sl_price:.{market.digits}f}[/bold red] ({sl_pips:.1f} pips) | "
            f"TP: [bold green]{tp_price:.{market.digits}f}[/bold green] ({tp_pips:.1f} pips) | "
            f"R:R = 1:{settings.ATR_TP_MULTIPLIER/settings.ATR_SL_MULTIPLIER:.1f}",
        )

    # Self-Evaluation Progress Metrics
    win_color = "green" if performance.win_rate_pct >= 55.0 else "yellow"
    table.add_row(
        "Self-Evaluation",
        f"Evaluated Signals: [cyan]{performance.evaluated_signals}[/cyan] (Pending: {performance.pending_signals}) | "
        f"Win Rate: [{win_color}]{performance.win_rate_pct:.1f}%[/{win_color}] ({performance.directional_wins}W - {performance.directional_losses}L) | "
        f"Net Pips: [bold]{performance.total_net_pips:+.1f}[/bold] | Profit Factor: {performance.profit_factor:.2f}",
    )

    if decision.rejection_reasons:
        table.add_row(
            "Rejection Reasons",
            f"[dim red]{' | '.join(decision.rejection_reasons)}[/dim red]",
        )

    mode_label = (
        f"[bold cyan][ADVISORY MODE][/bold cyan]" if mode == "advisory"
        else (f"[bold yellow][DRY-RUN SIMULATION][/bold yellow]" if mode == "dry-run"
        else f"[bold red][LIVE ORDER DISPATCH][/bold red]")
    )
    console.print(Panel(table, subtitle=mode_label, border_style="blue"))


# ============================================================================
# Master Orchestrator
# ============================================================================

class TradingOrchestrator:
    """Manages the full polling, forecasting, supervision, and execution lifecycle."""

    def __init__(self, config: Optional[SystemSettings] = None, mode: str = "dry-run"):
        self.config = config or settings
        self.mode = mode  # 'advisory', 'dry-run', or 'live'
        self.bridge = MT5Bridge(self.config)
        self.forecaster = QuantitativeForecaster(self.config)
        self.gatekeeper = JevGatekeeper(self.config)
        self.tracker = PerformanceTracker()
        self.running: bool = False
        self.last_recorded_bar_time: Optional[Any] = None

    def setup_signal_handlers(self) -> None:
        """Configures clean termination traps."""
        def _handle_exit(sig, frame):
            logger.info("Shutdown signal received. Initiating graceful teardown...")
            self.running = False

        signal.signal(signal.SIGINT, _handle_exit)
        signal.signal(signal.SIGTERM, _handle_exit)

    def execute_iteration(self, iteration: int) -> Optional[OrderResult]:
        """Runs a single pass of the quantitative trading pipeline."""
        symbol = self.config.SYMBOL
        tf = self.config.TIMEFRAME
        count = self.config.HISTORY_BARS

        # 1. Fetch live market & account state
        market = self.bridge.get_market_state(symbol)
        account = self.bridge.get_account_state()
        open_positions = self.bridge.get_open_positions(symbol)

        # 2. Ingest candle history
        df_rates = self.bridge.get_rates(symbol, tf, count)

        # 3. Self-Evaluation: Check previously pending signals against latest price
        latest_bar = df_rates.iloc[-1]
        self.tracker.evaluate_pending_signals(
            symbol=market.symbol,
            current_high=float(latest_bar["high"]),
            current_low=float(latest_bar["low"]),
            current_close=float(latest_bar["close"]),
            point=market.point,
            digits=market.digits,
        )
        perf_summary = self.tracker.get_performance_summary(market.symbol)

        # 4. Compute quantitative forecast (ML + Chronos)
        signal_obj = self.forecaster.forecast(df_rates, market.symbol)

        # 5. TypeSafe AI Jev Gatekeeper Evaluation
        decision = self.gatekeeper.evaluate(market, signal_obj, open_positions, account)

        # 6. Calculate dynamic Stop Loss and Take Profit levels from ATR
        sl_dist = self.config.ATR_SL_MULTIPLIER * signal_obj.atr
        tp_dist = self.config.ATR_TP_MULTIPLIER * signal_obj.atr
        sl_points = round(sl_dist / market.point)
        tp_points = round(tp_dist / market.point)
        pip_factor = 10.0 * market.point if market.digits in (3, 5) else market.point
        sl_pips = sl_dist / pip_factor
        tp_pips = tp_dist / pip_factor

        is_buy = (signal_obj.direction == 1)
        is_sell = (signal_obj.direction == -1)
        entry_price = market.ask if is_buy else market.bid
        sl_price = round(entry_price - sl_dist if is_buy else entry_price + sl_dist, market.digits)
        tp_price = round(entry_price + tp_dist if is_buy else entry_price - tp_dist, market.digits)

        # 7. Render Visual Dashboard
        render_dashboard(
            market=market,
            account=account,
            signal=signal_obj,
            decision=decision,
            performance=perf_summary,
            sl_price=sl_price,
            tp_price=tp_price,
            sl_pips=sl_pips,
            tp_pips=tp_pips,
            mode=self.mode,
            iteration=iteration,
        )

        # 8. Record Signal in Self-Evaluation Tracker (1 signal per candle bar)
        current_bar_time = df_rates.index[-1]
        if signal_obj.direction != 0 and current_bar_time != self.last_recorded_bar_time:
            self.last_recorded_bar_time = current_bar_time
            self.tracker.record_signal(
                symbol=market.symbol,
                timeframe=tf,
                direction=signal_obj.direction,
                entry_price=entry_price,
                sl_price=sl_price,
                tp_price=tp_price,
                predicted_return=signal_obj.predicted_return,
                confidence=signal_obj.model_confidence,
                horizon_bars=self.config.FORECAST_HORIZON,
                is_executed=(decision.is_approved and self.mode in ("dry-run", "live")),
            )

        # 9. Order Dispatch (Dry-Run or Live)
        if decision.is_approved and decision.execution_permitted:
            order_type = "BUY" if decision.approved_action == "EXECUTE_BUY" else "SELL"

            if self.mode == "advisory":
                console.print(
                    Panel(
                        f"[bold green]🔔 TRADE ADVISORY SIGNAL GENERATED[/bold green]\n"
                        f"Action: [bold white]{order_type} {market.symbol}[/bold white]\n"
                        f"Entry: [white]{entry_price:.{market.digits}f}[/white]\n"
                        f"Stop Loss (SL): [red]{sl_price:.{market.digits}f}[/red] (-{sl_pips:.1f} pips)\n"
                        f"Take Profit (TP): [green]{tp_price:.{market.digits}f}[/green] (+{tp_pips:.1f} pips)\n"
                        f"Recommended Volume: [yellow]{self.config.DEFAULT_VOLUME} lots[/yellow]",
                        border_style="green",
                    )
                )
                return None

            is_dry = (self.mode == "dry-run")
            logger.info(
                f"Dispatching order: {order_type} {self.config.DEFAULT_VOLUME} {market.symbol} "
                f"| SL: {sl_price} ({sl_points} pts) | TP: {tp_price} ({tp_points} pts) | Dry Run: {is_dry}"
            )

            result = self.bridge.send_order(
                symbol=market.symbol,
                order_type=order_type,
                volume=self.config.DEFAULT_VOLUME,
                sl_points=sl_points,
                tp_points=tp_points,
                deviation=self.config.SLIPPAGE_DEVIATION,
                dry_run=is_dry,
                magic=self.config.MAGIC_NUMBER,
                comment=f"Jev-{order_type}-{signal_obj.model_confidence:.2f}",
            )

            if result.success:
                console.print(
                    f"[bold green]✔ ORDER PROCESSED[/bold green] | "
                    f"Deal: {result.deal_id} | Price: {result.price} | Volume: {result.volume}"
                )
            else:
                console.print(
                    f"[bold red]✘ ORDER FAILED[/bold red] | "
                    f"Retcode: {result.retcode} | {result.comment}"
                )
            return result

        return None

    def run_loop(self, single_run: bool = False) -> None:
        """Starts the persistent operational polling loop."""
        self.setup_signal_handlers()

        logger.info("Connecting to MT5 terminal...")
        if not self.bridge.initialize():
            logger.error("Failed to connect to MT5. Exiting.")
            return

        self.running = True
        iteration = 1
        poll_sec = self.config.POLL_INTERVAL_SECONDS

        # Record PID for stop.bat process management
        pid_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.pid")
        try:
            with open(pid_file, "w") as f:
                f.write(str(os.getpid()))
        except Exception:
            pass

        console.print(
            Panel(
                f"[bold cyan]TypeSafe Jev + MT5 Quantitative Trading & Self-Evaluation Engine[/bold cyan]\n"
                f"Symbol: [white bold]{self.config.SYMBOL}[/white bold] | Timeframe: [white bold]{self.config.TIMEFRAME}[/white bold] | "
                f"Mode: [yellow]{self.mode.upper()}[/yellow] | Poll Interval: [white]{poll_sec}s[/white]\n"
                f"Chronos Model: [white]{self.config.CHRONOS_MODEL_NAME}[/white] | Supervisor: [white]TypeSafe Jev (Deterministic)[/white]",
                border_style="green",
            )
        )

        try:
            while self.running:
                try:
                    self.execute_iteration(iteration)
                except Exception as e:
                    logger.error(f"Error during iteration #{iteration}: {e}", exc_info=True)

                if single_run:
                    logger.info("Single-run execution mode complete. Exiting loop.")
                    break

                for _ in range(poll_sec):
                    if not self.running:
                        break
                    time.sleep(1)

                iteration += 1

        finally:
            logger.info("Closing MT5 bridge...")
            self.bridge.shutdown()
            if os.path.exists(pid_file):
                try:
                    os.remove(pid_file)
                except Exception:
                    pass
            console.print("[bold yellow]System successfully shut down.[/bold yellow]")


# ============================================================================
# CLI Entrypoint
# ============================================================================

def parse_cli_args() -> argparse.Namespace:
    """Parses command-line arguments to override default settings."""
    parser = argparse.ArgumentParser(
        description="TypeSafe Jev + MetaTrader 5 Quantitative Trading and Self-Evaluating System."
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default=settings.SYMBOL,
        help="Target trading symbol (e.g. XAUUSD, GBPUSD, EURUSD, BTCUSD).",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default=settings.TIMEFRAME,
        help="Candle timeframe ('M1', 'M5', 'M15', 'H1', 'H4', 'D1').",
    )
    parser.add_argument(
        "--volume",
        type=float,
        default=settings.DEFAULT_VOLUME,
        help="Trade volume lot size.",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=settings.POLL_INTERVAL_SECONDS,
        help="Polling interval in seconds between analysis iterations.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["advisory", "dry-run", "live"],
        default="dry-run",
        help="Execution mode: 'advisory' (signals only), 'dry-run' (simulated), or 'live' (broker dispatch).",
    )
    parser.add_argument(
        "--single-run",
        action="store_true",
        default=False,
        help="Run a single analysis iteration and exit.",
    )
    parser.add_argument(
        "--max-spread",
        type=float,
        default=settings.MAX_SPREAD_PIPS,
        help="Maximum allowable spread in pips.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI application entrypoint."""
    args = parse_cli_args()

    settings.SYMBOL = args.symbol
    settings.TIMEFRAME = args.timeframe
    settings.DEFAULT_VOLUME = args.volume
    from config import get_default_max_spread
    settings.MAX_SPREAD_PIPS = get_default_max_spread(args.symbol, args.max_spread)

    orchestrator = TradingOrchestrator(settings, mode=args.mode)
    orchestrator.run_loop(single_run=args.single_run)


if __name__ == "__main__":
    main()
