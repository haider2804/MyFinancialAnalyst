"""
Centralized Configuration Module for Quantitative Trading and Forecasting System.
Defines typed settings, risk thresholds, MT5 bridge parameters, and API configurations.
"""

from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class SystemSettings(BaseSettings):
    """Global system configuration backed by environment variables or .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # MT5 Terminal Settings
    MT5_PATH: Optional[str] = Field(
        default=None,
        description="Path to terminal64.exe (e.g. 'C:/Program Files/MetaTrader 5/terminal64.exe'). If None, default registry path is used.",
    )
    MT5_LOGIN: Optional[int] = Field(
        default=None,
        description="MT5 trading account login ID. If None, active terminal session is used.",
    )
    MT5_PASSWORD: Optional[str] = Field(
        default=None,
        description="MT5 trading account password.",
    )
    MT5_SERVER: Optional[str] = Field(
        default=None,
        description="MT5 broker trade server name.",
    )
    MT5_TIMEOUT_MS: int = Field(
        default=60000,
        description="Connection timeout in milliseconds.",
    )

    # Trading Symbol & Timeframe
    SYMBOL: str = Field(
        default="XAUUSD",
        description="Target instrument symbol (e.g. 'XAUUSD', 'BTCUSD').",
    )
    TIMEFRAME: str = Field(
        default="M5",
        description="Candle timeframe ('M1', 'M5', 'M15', 'H1', 'H4', 'D1').",
    )
    HISTORY_BARS: int = Field(
        default=600,
        description="Historical bar count to ingest for feature calculation and ML training.",
    )

    # Order & Execution Parameters
    DEFAULT_VOLUME: float = Field(
        default=0.01,
        description="Default lot size for order execution.",
    )
    SLIPPAGE_DEVIATION: int = Field(
        default=10,
        description="Max acceptable slippage in points.",
    )
    DRY_RUN: bool = Field(
        default=True,
        description="When True, validates and simulates orders without broker dispatch.",
    )
    MAGIC_NUMBER: int = Field(
        default=998877,
        description="Unique EA / algorithm identification number.",
    )

    # Risk Controls
    MAX_SPREAD_PIPS: float = Field(
        default=3.0,
        description="Hard ceiling on spread in pips. Rejects trades if spread exceeds this.",
    )
    MAX_CONCURRENT_POSITIONS: int = Field(
        default=2,
        description="Maximum concurrent open positions allowed for the target symbol.",
    )
    ATR_SL_MULTIPLIER: float = Field(
        default=1.5,
        description="Stop loss distance as a multiple of Average True Range (ATR).",
    )
    ATR_TP_MULTIPLIER: float = Field(
        default=3.0,
        description="Take profit distance as a multiple of Average True Range (ATR).",
    )
    MIN_RISK_ACCEPTABLE_PROB: float = Field(
        default=0.85,
        description="TypeSafe Jev gatekeeper risk acceptance probability threshold.",
    )
    MIN_MODEL_CONFIDENCE: float = Field(
        default=0.60,
        description="Minimum combined quantitative model confidence required for signal approval.",
    )

    # TypeSafe AI / Jev Supervisor
    TYPESAFE_API_KEY: Optional[str] = Field(
        default=None,
        description="TypeSafe AI API key for Jev System 1 supervisor.",
    )
    TYPESAFE_MODEL_VERSION: str = Field(
        default="jev-latest",
        description="Jev model version to target (e.g. 'jev-latest', 'jev-1.13.0').",
    )
    USE_MOCK_JEV_IF_NO_KEY: bool = Field(
        default=True,
        description="If True and TYPESAFE_API_KEY is unset, use high-fidelity local deterministic System 1 engine.",
    )

    # Chronos Zero-Shot Model
    CHRONOS_MODEL_NAME: str = Field(
        default="amazon/chronos-bolt-tiny",
        description="HuggingFace model ID for zero-shot time series prior.",
    )
    CHRONOS_CONTEXT_LENGTH: int = Field(
        default=128,
        description="Number of trailing bars supplied as context to Chronos.",
    )
    FORECAST_HORIZON: int = Field(
        default=5,
        description="Forward bar horizon for return forecasting.",
    )

    # Polling & Loop Settings
    POLL_INTERVAL_SECONDS: int = Field(
        default=15,
        description="Interval in seconds between system analysis iterations.",
    )


# Global settings singleton
settings = SystemSettings()


def get_default_max_spread(symbol: str, override: Optional[float] = None) -> float:
    """Smart spread ceiling resolution based on instrument class."""
    if override is not None and override != 3.0:
        return override

    sym_upper = symbol.upper()
    if "XAU" in sym_upper or "GOLD" in sym_upper:
        return 35.0  # Gold spread typical: 15-30 pips
    elif "BTC" in sym_upper or "BTX" in sym_upper:
        return 2000.0  # Bitcoin spread typical: 800-1500 points/pips ($8-$15 spread on $83k BTC)
    elif "ETH" in sym_upper:
        return 100.0   # Ethereum spread typical: 10-50 pips
    elif "XAG" in sym_upper or "SILVER" in sym_upper:
        return 25.0
    return 3.0  # Standard Forex pairs (GBPUSD, EURUSD, etc.)

