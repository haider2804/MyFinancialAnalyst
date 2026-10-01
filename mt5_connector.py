"""
Module A: MetaTrader 5 Live Bridge (mt5_connector.py)

Interfaces directly with the local host MetaTrader 5 terminal:
- Connects, authenticates, and validates terminal & account health.
- Resolves broker-specific symbol variations (e.g. GBPUSD vs GBPUSDc).
- Streams live market state (Bid, Ask, Spread, Tick Volume).
- Ingests historical OHLCV bar rates into typed pandas DataFrames.
- Validates pre-trade risk and handles both dry-run simulation and live order dispatch.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Union

import MetaTrader5 as mt5
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from config import SystemSettings, settings

logger = logging.getLogger("MT5Connector")


# ============================================================================
# Data Models
# ============================================================================

class MarketState(BaseModel):
    """Snapshot of current live market tick and symbol specifications."""

    symbol: str = Field(..., description="Broker resolved symbol name.")
    bid: float = Field(..., description="Current highest buy price.")
    ask: float = Field(..., description="Current lowest sell price.")
    spread_points: float = Field(..., description="Spread in raw broker points.")
    spread_pips: float = Field(..., description="Spread normalized in standard pips.")
    tick_volume: int = Field(default=0, description="Latest tick volume.")
    last_tick_time: datetime = Field(..., description="Timestamp of the latest quote.")
    digits: int = Field(..., description="Decimal precision digits.")
    point: float = Field(..., description="Smallest point increment.")
    min_volume: float = Field(..., description="Minimum permitted trade volume.")
    max_volume: float = Field(..., description="Maximum permitted trade volume.")
    volume_step: float = Field(..., description="Volume step increment.")


class PositionInfo(BaseModel):
    """Details of an active open position."""

    ticket: int = Field(..., description="Unique position ticket ID.")
    symbol: str = Field(..., description="Symbol of the position.")
    type: str = Field(..., description="'BUY' or 'SELL'.")
    volume: float = Field(..., description="Position lot size.")
    price_open: float = Field(..., description="Execution opening price.")
    sl: float = Field(default=0.0, description="Stop Loss price.")
    tp: float = Field(default=0.0, description="Take Profit price.")
    profit: float = Field(..., description="Current floating unrealized profit in deposit currency.")
    swap: float = Field(default=0.0, description="Accumulated swap.")
    time_opened: datetime = Field(..., description="Timestamp when position was created.")


class AccountState(BaseModel):
    """Snapshot of account equity and margin state."""

    login: int
    balance: float
    equity: float
    profit: float
    margin: float
    margin_free: float
    margin_level: float
    leverage: int
    trade_allowed: bool


class OrderResult(BaseModel):
    """Result of an order execution or dry-run validation."""

    success: bool
    retcode: int
    deal_id: Optional[int] = None
    order_id: Optional[int] = None
    volume: float
    price: float
    comment: str
    is_dry_run: bool = False
    details: Dict[str, Union[str, float, int, bool]] = Field(default_factory=dict)


# ============================================================================
# MT5 Bridge Implementation
# ============================================================================

TIMEFRAME_MAP: Dict[str, int] = {
    "M1": mt5.TIMEFRAME_M1,
    "M2": mt5.TIMEFRAME_M2,
    "M3": mt5.TIMEFRAME_M3,
    "M4": mt5.TIMEFRAME_M4,
    "M5": mt5.TIMEFRAME_M5,
    "M6": mt5.TIMEFRAME_M6,
    "M10": mt5.TIMEFRAME_M10,
    "M12": mt5.TIMEFRAME_M12,
    "M15": mt5.TIMEFRAME_M15,
    "M20": mt5.TIMEFRAME_M20,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
    "H2": mt5.TIMEFRAME_H2,
    "H3": mt5.TIMEFRAME_H3,
    "H4": mt5.TIMEFRAME_H4,
    "H6": mt5.TIMEFRAME_H6,
    "H8": mt5.TIMEFRAME_H8,
    "H12": mt5.TIMEFRAME_H12,
    "D1": mt5.TIMEFRAME_D1,
    "W1": mt5.TIMEFRAME_W1,
    "MN1": mt5.TIMEFRAME_MN1,
}


class MT5Bridge:
    """Manages high-reliability interaction with the host MetaTrader 5 terminal."""

    def __init__(self, config: Optional[SystemSettings] = None):
        self.config = config or settings
        self.is_connected: bool = False
        self._resolved_symbols_cache: Dict[str, str] = {}

    def initialize(self) -> bool:
        """
        Initializes connection to the MT5 terminal and performs comprehensive health checks.
        Handles path resolution, credentials authentication, and verification of trading permissions.
        """
        logger.info("Initializing connection to MetaTrader 5 terminal...")

        init_kwargs = {
            "timeout": self.config.MT5_TIMEOUT_MS,
        }
        if self.config.MT5_PATH:
            init_kwargs["path"] = self.config.MT5_PATH
        if self.config.MT5_LOGIN:
            init_kwargs["login"] = self.config.MT5_LOGIN
        if self.config.MT5_PASSWORD:
            init_kwargs["password"] = self.config.MT5_PASSWORD
        if self.config.MT5_SERVER:
            init_kwargs["server"] = self.config.MT5_SERVER

        connected = mt5.initialize(**init_kwargs)
        if not connected:
            error_code, error_desc = mt5.last_error()
            logger.error(
                f"Failed to initialize MT5 terminal: error_code={error_code}, description='{error_desc}'"
            )
            self.is_connected = False
            return False

        # Verify terminal info
        term_info = mt5.terminal_info()
        if term_info is None:
            logger.error("Terminal initialization succeeded but terminal_info is None.")
            self.is_connected = False
            return False

        if not term_info.connected:
            logger.warning("MT5 terminal reports no active connection to broker server!")

        if not term_info.trade_allowed:
            logger.warning(
                "Algorithmic Trading is disabled in MT5 terminal settings! "
                "Enable 'Allow Algorithmic Trading' in Tools -> Options -> Expert Advisors."
            )

        # Verify account info
        acc_info = mt5.account_info()
        if acc_info is None:
            logger.error("Failed to retrieve MT5 account info.")
            self.is_connected = False
            return False

        logger.info(
            f"Connected to MT5 successfully | Account: {acc_info.login} "
            f"| Server: {acc_info.server} | Equity: ${acc_info.equity:,.2f} "
            f"| Balance: ${acc_info.balance:,.2f} | Leverage: 1:{acc_info.leverage} "
            f"| Build: {term_info.build}"
        )
        self.is_connected = True
        return True

    def shutdown(self) -> None:
        """Gracefully disconnects from MT5 terminal."""
        if self.is_connected:
            logger.info("Shutting down MT5 connection...")
            mt5.shutdown()
            self.is_connected = False
            logger.info("MT5 bridge successfully disconnected.")

    def __enter__(self) -> MT5Bridge:
        if not self.initialize():
            raise ConnectionError("Failed to initialize MT5 bridge.")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.shutdown()

    def resolve_symbol(self, requested_symbol: str) -> str:
        """
        Dynamically resolves broker-specific symbol names (e.g. 'GBPUSD' -> 'GBPUSDc', 'GBPUSD.m', 'GBPUSD_i').
        Handles aliases such as 'BTXUSDc' -> 'BTCUSDc', 'GOLD' -> 'XAUUSDc', and ensures the symbol is activated in MarketWatch.
        """
        if requested_symbol in self._resolved_symbols_cache:
            return self._resolved_symbols_cache[requested_symbol]

        all_symbols = mt5.symbols_get()
        if not all_symbols:
            logger.warning(f"Could not retrieve symbols list. Using raw '{requested_symbol}'.")
            return requested_symbol

        names = [s.name for s in all_symbols]

        # Alias dictionary
        alias_map = {
            "BTXUSD": "BTCUSD",
            "BTXUSDC": "BTCUSDC",
            "BTX": "BTCUSD",
            "BITCOIN": "BTCUSD",
            "GOLD": "XAUUSD",
            "XAU": "XAUUSD",
        }
        clean_target = requested_symbol.strip().upper()
        if clean_target in alias_map:
            clean_target = alias_map[clean_target].upper()

        # 1. Exact match
        resolved = None
        for name in names:
            if name.upper() == clean_target:
                resolved = name
                break

        # 2. Case-insensitive or common broker suffixes (e.g. BTCUSDc, XAUUSDc)
        if not resolved:
            # Strip trailing 'C' or '.M' if user specified base, or search prefixes
            base_target = clean_target.rstrip("C").rstrip(".M").rstrip("_I")
            for name in names:
                upper_name = name.upper()
                if upper_name == clean_target or upper_name == base_target:
                    resolved = name
                    break
                if upper_name.startswith(clean_target) and len(name) <= len(clean_target) + 4:
                    resolved = name
                    break
                if upper_name.startswith(base_target) and len(name) <= len(base_target) + 4:
                    resolved = name
                    break

        if not resolved:
            logger.warning(
                f"Target symbol '{requested_symbol}' not found in broker list. Fallback to raw symbol."
            )
            resolved = requested_symbol

        # Ensure symbol is visible in Market Watch
        selected = mt5.symbol_select(resolved, True)
        if not selected:
            logger.warning(f"Failed to select symbol '{resolved}' in MarketWatch.")

        self._resolved_symbols_cache[requested_symbol] = resolved
        logger.info(f"Resolved symbol request '{requested_symbol}' -> broker symbol '{resolved}'")
        return resolved

    def get_market_state(self, symbol: str) -> MarketState:
        """
        Fetches the current live tick and symbol parameters from MT5.
        Computes normalized pip spread and checks trading limits.
        """
        resolved = self.resolve_symbol(symbol)
        sym_info = mt5.symbol_info(resolved)
        if sym_info is None:
            raise ValueError(f"Could not query symbol_info for '{resolved}'. Error: {mt5.last_error()}")

        tick = mt5.symbol_info_tick(resolved)
        if tick is None:
            raise ValueError(f"Could not query live tick for '{resolved}'. Error: {mt5.last_error()}")

        point = sym_info.point
        digits = sym_info.digits

        # Calculate standard pip factor (10 points per pip for 3/5 digit quotes)
        pip_multiplier = 10.0 * point if (digits in (3, 5)) else point
        raw_spread_points = sym_info.spread if sym_info.spread > 0 else round((tick.ask - tick.bid) / point)
        spread_pips = raw_spread_points * (point / pip_multiplier)

        last_time = datetime.fromtimestamp(tick.time, tz=timezone.utc)

        return MarketState(
            symbol=resolved,
            bid=tick.bid,
            ask=tick.ask,
            spread_points=float(raw_spread_points),
            spread_pips=round(spread_pips, 2),
            tick_volume=int(tick.volume),
            last_tick_time=last_time,
            digits=digits,
            point=point,
            min_volume=sym_info.volume_min,
            max_volume=sym_info.volume_max,
            volume_step=sym_info.volume_step,
        )

    def get_rates(
        self,
        symbol: str,
        timeframe: Union[str, int] = "M15",
        count: int = 600,
    ) -> pd.DataFrame:
        """
        Fetches historical OHLCV bar rates for the specified symbol and timeframe.
        Returns a clean pandas DataFrame indexed by UTC datetime.
        """
        resolved = self.resolve_symbol(symbol)

        if isinstance(timeframe, str):
            tf_const = TIMEFRAME_MAP.get(timeframe.upper())
            if tf_const is None:
                raise ValueError(
                    f"Unsupported timeframe string '{timeframe}'. Valid options: {list(TIMEFRAME_MAP.keys())}"
                )
        else:
            tf_const = timeframe

        rates = mt5.copy_rates_from_pos(resolved, tf_const, 0, count)
        if rates is None or len(rates) == 0:
            error_code, error_desc = mt5.last_error()
            raise RuntimeError(
                f"Failed to fetch rates for '{resolved}' (timeframe={timeframe}, count={count}): "
                f"code={error_code}, desc='{error_desc}'"
            )

        df = pd.DataFrame(rates)
        # Convert timestamp to UTC datetime
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df.set_index("time", inplace=True)

        # Standardize column naming and types
        float_cols = ["open", "high", "low", "close"]
        int_cols = ["tick_volume", "spread", "real_volume"]

        for col in float_cols:
            if col in df.columns:
                df[col] = df[col].astype(np.float64)

        for col in int_cols:
            if col in df.columns:
                df[col] = df[col].astype(np.int64)

        # Sort chronologically and drop any corrupted duplicates
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="last")]

        if df.isna().any().any():
            df = df.ffill().bfill()

        return df

    def get_open_positions(self, symbol: Optional[str] = None) -> List[PositionInfo]:
        """Queries and returns currently active positions."""
        if symbol:
            resolved = self.resolve_symbol(symbol)
            raw_positions = mt5.positions_get(symbol=resolved)
        else:
            raw_positions = mt5.positions_get()

        if raw_positions is None:
            return []

        positions: List[PositionInfo] = []
        for pos in raw_positions:
            pos_type = "BUY" if pos.type == mt5.ORDER_TYPE_BUY else "SELL"
            positions.append(
                PositionInfo(
                    ticket=pos.ticket,
                    symbol=pos.symbol,
                    type=pos_type,
                    volume=pos.volume,
                    price_open=pos.price_open,
                    sl=pos.sl,
                    tp=pos.tp,
                    profit=pos.profit,
                    swap=pos.swap,
                    time_opened=datetime.fromtimestamp(pos.time, tz=timezone.utc),
                )
            )
        return positions

    def get_account_state(self) -> AccountState:
        """Retrieves real-time account financial and margin health."""
        acc = mt5.account_info()
        if acc is None:
            raise RuntimeError(f"Failed to query account_info: {mt5.last_error()}")

        return AccountState(
            login=acc.login,
            balance=acc.balance,
            equity=acc.equity,
            profit=acc.profit,
            margin=acc.margin,
            margin_free=acc.margin_free,
            margin_level=acc.margin_level,
            leverage=acc.leverage,
            trade_allowed=acc.trade_allowed,
        )

    def validate_order(
        self,
        symbol: str,
        order_type: str,
        volume: float,
        price: float,
    ) -> Tuple[bool, str]:
        """
        Performs thorough pre-trade verification:
        - Symbol tradeability
        - Lot size constraints (min, max, step)
        - Margin availability
        """
        resolved = self.resolve_symbol(symbol)
        sym_info = mt5.symbol_info(resolved)
        if sym_info is None:
            return False, f"Symbol '{resolved}' does not exist."

        if sym_info.trade_mode not in (
            mt5.SYMBOL_TRADE_MODE_FULL,
            mt5.SYMBOL_TRADE_MODE_LONGONLY,
            mt5.SYMBOL_TRADE_MODE_SHORTONLY,
        ):
            return False, f"Trading is disabled for symbol '{resolved}' (trade_mode={sym_info.trade_mode})."

        # Volume limits
        if volume < sym_info.volume_min:
            return False, f"Volume {volume} is below broker minimum {sym_info.volume_min}."
        if volume > sym_info.volume_max:
            return False, f"Volume {volume} exceeds broker maximum {sym_info.volume_max}."

        # Volume step alignment check
        step = sym_info.volume_step
        steps_count = round((volume - sym_info.volume_min) / step)
        expected_vol = round(sym_info.volume_min + (steps_count * step), 8)
        if abs(volume - expected_vol) > 1e-6:
            return False, f"Volume {volume} does not match required step {step}."

        # Margin requirement check
        mt5_order_type = mt5.ORDER_TYPE_BUY if order_type.upper() == "BUY" else mt5.ORDER_TYPE_SELL
        margin_req = mt5.order_calc_margin(mt5_order_type, resolved, volume, price)
        if margin_req is None:
            logger.warning("Could not calculate required margin via mt5.order_calc_margin.")
        else:
            acc = mt5.account_info()
            if acc and margin_req > acc.margin_free:
                return False, (
                    f"Insufficient free margin: required ${margin_req:.2f}, "
                    f"available ${acc.margin_free:.2f}."
                )

        return True, "Pre-trade validation passed."

    def _determine_filling_mode(self, sym_info) -> int:
        """Determines compatible order filling mode for the target symbol."""
        filling_mode = sym_info.filling_mode
        # Check flags: 1 = FOK, 2 = IOC, 3 = Return / Any
        if filling_mode & 2:  # IOC
            return mt5.ORDER_FILLING_IOC
        elif filling_mode & 1:  # FOK
            return mt5.ORDER_FILLING_FOK
        else:
            return mt5.ORDER_FILLING_RETURN

    def send_order(
        self,
        symbol: str,
        order_type: str,
        volume: float,
        sl_points: Optional[float] = None,
        tp_points: Optional[float] = None,
        deviation: int = 10,
        dry_run: bool = True,
        magic: int = 998877,
        comment: str = "TypeSafe-Quant",
    ) -> OrderResult:
        """
        Constructs, validates, and dispatches a market order.
        If dry_run=True, conducts complete payload formatting and sanity verification
        without transmitting the deal to the broker.
        """
        resolved = self.resolve_symbol(symbol)
        market = self.get_market_state(resolved)
        sym_info = mt5.symbol_info(resolved)

        order_type_upper = order_type.upper()
        if order_type_upper == "BUY":
            price = market.ask
            mt5_action_type = mt5.ORDER_TYPE_BUY
            sl_price = round(price - (sl_points * market.point), market.digits) if sl_points else 0.0
            tp_price = round(price + (tp_points * market.point), market.digits) if tp_points else 0.0
        elif order_type_upper == "SELL":
            price = market.bid
            mt5_action_type = mt5.ORDER_TYPE_SELL
            sl_price = round(price + (sl_points * market.point), market.digits) if sl_points else 0.0
            tp_price = round(price - (tp_points * market.point), market.digits) if tp_points else 0.0
        else:
            raise ValueError(f"Invalid order_type '{order_type}'. Expected 'BUY' or 'SELL'.")

        # Validate order
        is_valid, validation_msg = self.validate_order(resolved, order_type_upper, volume, price)
        if not is_valid:
            logger.error(f"Order validation rejected: {validation_msg}")
            return OrderResult(
                success=False,
                retcode=10013,  # Invalid request
                volume=volume,
                price=price,
                comment=f"Rejected: {validation_msg}",
                is_dry_run=dry_run,
            )

        filling_type = self._determine_filling_mode(sym_info)

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": resolved,
            "volume": float(volume),
            "type": mt5_action_type,
            "price": float(price),
            "sl": float(sl_price),
            "tp": float(tp_price),
            "deviation": int(deviation),
            "magic": int(magic),
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": filling_type,
        }

        if dry_run:
            logger.info(
                f"[DRY RUN SIMULATION] {order_type_upper} {volume} {resolved} @ {price:.{market.digits}f} "
                f"| SL: {sl_price} | TP: {tp_price} | Spread: {market.spread_pips} pips"
            )
            return OrderResult(
                success=True,
                retcode=mt5.TRADE_RETCODE_DONE,
                deal_id=99999999,
                order_id=88888888,
                volume=volume,
                price=price,
                comment="Dry run order simulated successfully.",
                is_dry_run=True,
                details=request,
            )

        # LIVE EXECUTION
        logger.warning(
            f"[LIVE DISPATCH] Executing {order_type_upper} {volume} {resolved} @ {price:.{market.digits}f}..."
        )
        result = mt5.order_send(request)

        if result is None:
            err_code, err_desc = mt5.last_error()
            logger.error(f"mt5.order_send returned None. Error {err_code}: {err_desc}")
            return OrderResult(
                success=False,
                retcode=err_code,
                volume=volume,
                price=price,
                comment=f"Order dispatch error: {err_desc}",
                is_dry_run=False,
            )

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(
                f"Order failed with retcode {result.retcode}: {result.comment}"
            )
            return OrderResult(
                success=False,
                retcode=result.retcode,
                volume=result.volume,
                price=result.price,
                comment=result.comment,
                is_dry_run=False,
            )

        logger.info(
            f"LIVE Order Executed! Ticket: {result.order}, Deal: {result.deal}, "
            f"Volume: {result.volume}, Price: {result.price}"
        )
        return OrderResult(
            success=True,
            retcode=result.retcode,
            deal_id=result.deal,
            order_id=result.order,
            volume=result.volume,
            price=result.price,
            comment=result.comment,
            is_dry_run=False,
        )
