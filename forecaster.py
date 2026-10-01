"""
Module B: Quantitative Forecasting Pipeline (forecaster.py)

Combines two complementary forecasting tiers:
1. Tabular / Feature-Engineered ML:
   - High-dimensional technical indicator features (RSI, ATR, MACD, multi-horizon lags,
     rolling volatility, cyclical diurnal/day encodings, volume z-scores).
   - Fast LightGBM / XGBoost classifier for directional probability distributions.
2. Zero-Shot Foundation Model Prior:
   - Amazon Chronos (ChronosBoltPipeline) probabilistic trajectory distribution.
   - Extracts quantiles (p10, p50, p90), zero-shot drift, and trajectory dispersion.
3. Standardized Ensemble Synthesizer:
   - Merges tabular ML and foundation model prior into a unified signal schema:
     predicted_return, direction (-1, 0, 1), model_confidence [0.0, 1.0].
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from pydantic import BaseModel, Field

from config import SystemSettings, settings

logger = logging.getLogger("Forecaster")

# Attempt importing ML backends
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from chronos import ChronosBoltPipeline, ChronosPipeline
    HAS_CHRONOS = True
except ImportError:
    HAS_CHRONOS = False


# ============================================================================
# Signal Schema
# ============================================================================

class ForecastSignal(BaseModel):
    """Unified quantitative forecast output adhering to the standardized schema."""

    symbol: str
    current_price: float
    predicted_return: float = Field(
        ...,
        description="Expected forward return over the prediction horizon (e.g. +0.0025 = +0.25%).",
    )
    direction: int = Field(
        ...,
        description="Forecasted market direction: 1 = Bullish / Long, -1 = Bearish / Short, 0 = Neutral / Stand Down.",
    )
    model_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Ensemble confidence score bounded between 0.0 and 1.0.",
    )
    horizon_bars: int = Field(default=5, description="Number of bars ahead forecasted.")

    # Breakdown components
    tabular_direction: int
    tabular_confidence: float
    chronos_direction: int
    chronos_confidence: float
    chronos_return: float
    chronos_p10_return: float
    chronos_p90_return: float
    atr: float
    agreement: bool = Field(..., description="True if tabular ML and foundation prior agree on direction.")

    diagnostics: Dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# Technical Feature Engineering
# ============================================================================

class FeatureExtractor:
    """Computes technical, statistical, and temporal features from OHLCV bar series."""

    @staticmethod
    def extract_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
        """
        Extracts temporal and statistical features from historical OHLCV data.
        Returns the feature DataFrame along with the current ATR value.
        """
        if len(df) < 50:
            raise ValueError(f"Insufficient history length ({len(df)} bars) for robust feature extraction.")

        feat = pd.DataFrame(index=df.index)
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["tick_volume"] if "tick_volume" in df.columns else df.get("volume", pd.Series(1, index=df.index))

        # 1. Price Momentum & Multi-Horizon Log Returns
        log_close = np.log(close)
        for lag in [1, 2, 3, 5, 8, 13, 21]:
            feat[f"ret_lag_{lag}"] = log_close - log_close.shift(lag)

        # 2. Relative Strength Index (RSI - 14)
        delta = close.diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)
        roll_gain = gain.ewm(alpha=1.0 / 14.0, adjust=False).mean()
        roll_loss = loss.ewm(alpha=1.0 / 14.0, adjust=False).mean()
        rs = roll_gain / (roll_loss + 1e-10)
        feat["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))

        # 3. Average True Range (ATR - 14)
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr_14 = true_range.ewm(alpha=1.0 / 14.0, adjust=False).mean()
        feat["atr_14"] = atr_14
        feat["atr_pct"] = atr_14 / close

        # 4. Moving Average Convergence Divergence (MACD 12, 26, 9)
        ema_fast = close.ewm(span=12, adjust=False).mean()
        ema_slow = close.ewm(span=26, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        macd_signal = macd_line.ewm(span=9, adjust=False).mean()
        feat["macd_line"] = macd_line / close
        feat["macd_signal"] = macd_signal / close
        feat["macd_hist"] = (macd_line - macd_signal) / close

        # 5. Multi-Window Rolling Volatilities (annualized / normalized)
        ret_1 = feat["ret_lag_1"]
        feat["vol_5"] = ret_1.rolling(5).std()
        feat["vol_15"] = ret_1.rolling(15).std()
        feat["vol_30"] = ret_1.rolling(30).std()

        # 6. Price Location & Candle Microstructure
        candle_range = (high - low).replace(0, 1e-6)
        feat["bar_body"] = (close - df["open"]) / candle_range
        feat["bar_close_pos"] = (close - low) / candle_range

        # 7. Volume Dynamics & Tick Intensity
        vol_roll_mean = volume.rolling(20).mean()
        vol_roll_std = volume.rolling(20).std().replace(0, 1.0)
        feat["vol_zscore"] = (volume - vol_roll_mean) / vol_roll_std

        # 8. Cyclical Time Encodings (Hour of Day & Day of Week)
        hours = df.index.hour
        dows = df.index.dayofweek
        feat["hour_sin"] = np.sin(2.0 * np.pi * hours / 24.0)
        feat["hour_cos"] = np.cos(2.0 * np.pi * hours / 24.0)
        feat["dow_sin"] = np.sin(2.0 * np.pi * dows / 7.0)
        feat["dow_cos"] = np.cos(2.0 * np.pi * dows / 7.0)

        # Current ATR for risk / target normalization
        current_atr = float(atr_14.iloc[-1])

        # Forward fill and backfill warmup bars
        feat = feat.ffill().bfill().fillna(0.0)

        return feat, current_atr


# ============================================================================
# Tier 1: Tabular ML Engine (LightGBM / XGBoost)
# ============================================================================

class TabularMLEngine:
    """Trains and infers directional return probabilities via gradient boosted trees."""

    def __init__(self, horizon_bars: int = 5):
        self.horizon_bars = horizon_bars
        self.feature_columns: List[str] = []

    def fit_and_predict(
        self,
        features: pd.DataFrame,
        close_series: pd.Series,
        current_atr: float,
    ) -> Tuple[int, float, float]:
        """
        Fits a calibrated tree model on historical bars and outputs:
        - direction: -1 (Down), 0 (Flat), 1 (Up)
        - confidence: calibrated probability [0.0, 1.0]
        - predicted_return: estimated forward return
        """
        # Forward return target
        forward_close = close_series.shift(-self.horizon_bars)
        forward_returns = (forward_close - close_series) / close_series

        # Target classification threshold: 0.25 * ATR / Close
        current_price = float(close_series.iloc[-1])
        vol_threshold = max(0.0003, (0.25 * current_atr) / current_price)

        # Classes: 0 -> Bearish (-1), 1 -> Neutral (0), 2 -> Bullish (+1)
        target = pd.Series(1, index=forward_returns.index)
        target[forward_returns > vol_threshold] = 2
        target[forward_returns < -vol_threshold] = 0

        # Exclude the most recent horizon_bars for training as their target is unobserved
        valid_mask = ~forward_returns.isna()
        X_train = features.loc[valid_mask]
        y_train = target.loc[valid_mask]

        if len(X_train) < 60:
            logger.warning("Historical training samples < 60; returning neutral tabular prior.")
            return 0, 0.50, 0.0

        # Features to use (drop non-predictive or leaky columns)
        self.feature_columns = list(features.columns)
        X_matrix = X_train[self.feature_columns].values
        y_vector = y_train.values

        # Live inference feature vector (latest candle)
        X_live = features[self.feature_columns].iloc[[-1]].values

        probs = None
        if HAS_LIGHTGBM:
            clf = lgb.LGBMClassifier(
                n_estimators=45,
                max_depth=3,
                num_leaves=7,
                learning_rate=0.08,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=42,
                verbose=-1,
                n_jobs=1,
            )
            clf.fit(X_matrix, y_vector)
            probs = clf.predict_proba(X_live)[0]
        elif HAS_XGBOOST:
            clf = xgb.XGBClassifier(
                n_estimators=45,
                max_depth=3,
                learning_rate=0.08,
                random_state=42,
                verbosity=0,
                n_jobs=1,
            )
            clf.fit(X_matrix, y_vector)
            probs = clf.predict_proba(X_live)[0]
        else:
            # Fallback simple momentum heuristic if neither tree library is present
            rsi_val = float(features["rsi_14"].iloc[-1])
            macd_hist = float(features["macd_hist"].iloc[-1])
            if rsi_val > 55 and macd_hist > 0:
                return 1, 0.65, vol_threshold
            elif rsi_val < 45 and macd_hist < 0:
                return -1, 0.65, -vol_threshold
            return 0, 0.50, 0.0

        # Interpret probabilities: probs[0]=Bearish, probs[1]=Neutral, probs[2]=Bullish
        p_bear = float(probs[0])
        p_neutral = float(probs[1]) if len(probs) > 1 else 0.0
        p_bull = float(probs[2]) if len(probs) > 2 else 0.0

        # Directional assignment
        if p_bull > p_bear and p_bull >= 0.42 and p_bull > p_neutral:
            direction = 1
            confidence = p_bull
            expected_ret = (p_bull - p_bear) * vol_threshold * 1.5
        elif p_bear > p_bull and p_bear >= 0.42 and p_bear > p_neutral:
            direction = -1
            confidence = p_bear
            expected_ret = -(p_bear - p_bull) * vol_threshold * 1.5
        else:
            direction = 0
            confidence = max(p_neutral, 0.50)
            expected_ret = 0.0

        return direction, confidence, expected_ret


# ============================================================================
# Tier 2: Zero-Shot Time-Series Foundation Model Prior (Chronos)
# ============================================================================

class ChronosPriorEngine:
    """Manages zero-shot probabilistic forecasting via Amazon Chronos Foundation Model."""

    def __init__(self, model_name: str = "amazon/chronos-bolt-tiny", context_length: int = 128):
        self.model_name = model_name
        self.context_length = context_length
        self.pipeline = None
        self._is_bolt: bool = False
        self._init_pipeline()

    def _init_pipeline(self) -> None:
        """Loads Chronos pretrained weights onto available hardware."""
        if not HAS_CHRONOS:
            logger.warning("chronos-forecasting library not available. Using analytical Bayesian prior.")
            return

        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Loading Chronos Foundation Model '{self.model_name}' on {device}...")

            if "bolt" in self.model_name.lower():
                self.pipeline = ChronosBoltPipeline.from_pretrained(
                    self.model_name,
                    device_map=device,
                    torch_dtype=torch.float32,
                )
                self._is_bolt = True
            else:
                self.pipeline = ChronosPipeline.from_pretrained(
                    self.model_name,
                    device_map=device,
                    torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
                )
                self._is_bolt = False

            logger.info(f"Chronos Foundation Model successfully loaded on {device}.")
        except Exception as e:
            logger.error(f"Failed to load Chronos model: {e}. Falling back to analytical prior.")
            self.pipeline = None

    def predict_trajectory(
        self,
        close_series: pd.Series,
        horizon_bars: int = 5,
    ) -> Tuple[float, int, float, float, float]:
        """
        Runs probabilistic zero-shot trajectory forecasting.
        Returns:
        - expected_return: median forecast forward return
        - direction: -1, 0, or 1
        - confidence: trajectory consensus score [0.0, 1.0]
        - p10_return: 10th percentile return
        - p90_return: 90th percentile return
        """
        current_price = float(close_series.iloc[-1])
        context = close_series.iloc[-self.context_length:].values

        if self.pipeline is None:
            # Analytical Bayesian Random Walk with Drift fallback
            returns = np.diff(np.log(context))
            mu = float(np.mean(returns))
            sigma = float(np.std(returns)) + 1e-8
            horizon_drift = mu * horizon_bars
            horizon_std = sigma * np.sqrt(horizon_bars)

            p10_ret = horizon_drift - (1.28 * horizon_std)
            p90_ret = horizon_drift + (1.28 * horizon_std)
            exp_ret = horizon_drift

            direction = 1 if exp_ret > (0.5 * horizon_std) else (-1 if exp_ret < (-0.5 * horizon_std) else 0)
            confidence = float(np.clip(abs(exp_ret) / (horizon_std + 1e-6), 0.45, 0.85))
            return exp_ret, direction, confidence, p10_ret, p90_ret

        try:
            context_tensor = torch.tensor(context, dtype=torch.float32)

            if self._is_bolt:
                # ChronosBolt outputs [batch, 9, horizon] where quantiles are 0.1, 0.2 ... 0.9
                forecast = self.pipeline.predict(context_tensor, prediction_length=horizon_bars)
                # forecast shape: [1, 9, horizon]
                p10_price = float(forecast[0, 0, -1].item())
                p50_price = float(forecast[0, 4, -1].item())
                p90_price = float(forecast[0, 8, -1].item())
            else:
                forecast = self.pipeline.predict(
                    context_tensor,
                    prediction_length=horizon_bars,
                    num_samples=25,
                )
                # forecast shape: [1, num_samples, horizon]
                terminal_samples = forecast[0, :, -1].numpy()
                p10_price = float(np.percentile(terminal_samples, 10))
                p50_price = float(np.percentile(terminal_samples, 50))
                p90_price = float(np.percentile(terminal_samples, 90))

            p10_ret = (p10_price - current_price) / current_price
            p50_ret = (p50_price - current_price) / current_price
            p90_ret = (p90_price - current_price) / current_price

            # Trajectory dispersion (uncertainty)
            dispersion = max(1e-6, p90_ret - p10_ret)

            # Directional determination based on median drift & distribution skew
            if p50_ret > 0.0003 and p10_ret > -0.0008:
                direction = 1
                confidence = float(np.clip(0.50 + (p50_ret / dispersion) * 0.5, 0.50, 0.95))
            elif p50_ret < -0.0003 and p90_ret < 0.0008:
                direction = -1
                confidence = float(np.clip(0.50 + (-p50_ret / dispersion) * 0.5, 0.50, 0.95))
            else:
                direction = 0
                confidence = 0.50

            return p50_ret, direction, confidence, p10_ret, p90_ret

        except Exception as e:
            logger.error(f"Chronos inference exception: {e}. Falling back to neutral.")
            return 0.0, 0, 0.50, -0.001, 0.001


# ============================================================================
# Master Quantitative Forecaster
# ============================================================================

class QuantitativeForecaster:
    """
    Integrates Feature Engineering, Tabular ML, and Foundation Model Prior
    to produce a standardized, highly calibrated ForecastSignal.
    """

    def __init__(self, config: Optional[SystemSettings] = None):
        self.config = config or settings
        self.feature_extractor = FeatureExtractor()
        self.tabular_engine = TabularMLEngine(horizon_bars=self.config.FORECAST_HORIZON)
        self.chronos_prior = ChronosPriorEngine(
            model_name=self.config.CHRONOS_MODEL_NAME,
            context_length=self.config.CHRONOS_CONTEXT_LENGTH,
        )

    def forecast(self, df_rates: pd.DataFrame, symbol: str) -> ForecastSignal:
        """
        Executes end-to-end forecasting pipeline on historical candle rates.
        Returns a standardized ForecastSignal object.
        """
        if len(df_rates) < 60:
            raise ValueError(f"Need at least 60 historical bars to forecast, got {len(df_rates)}.")

        # 1. Feature Engineering
        features, current_atr = self.feature_extractor.extract_features(df_rates)
        close_series = df_rates["close"]
        current_price = float(close_series.iloc[-1])

        # 2. Tier 1: Tabular ML Engine
        tab_dir, tab_conf, tab_ret = self.tabular_engine.fit_and_predict(
            features, close_series, current_atr
        )

        # 3. Tier 2: Zero-Shot Chronos Prior
        chr_ret, chr_dir, chr_conf, chr_p10, chr_p90 = self.chronos_prior.predict_trajectory(
            close_series, horizon_bars=self.config.FORECAST_HORIZON
        )

        # 4. Ensemble & Synthesis Logic
        models_agree = (tab_dir != 0) and (tab_dir == chr_dir)

        if models_agree:
            # Both models independently confirm the directional move
            direction = tab_dir
            blended_return = 0.55 * tab_ret + 0.45 * chr_ret
            # Bolster confidence when consensus is achieved
            confidence = min(0.98, float(np.mean([tab_conf, chr_conf]) * 1.20))
        elif tab_dir != 0 and chr_dir == 0:
            # Tabular ML has a signal while Chronos is neutral
            if tab_conf >= 0.65:
                direction = tab_dir
                blended_return = tab_ret * 0.75
                confidence = tab_conf * 0.85
            else:
                direction = 0
                blended_return = 0.0
                confidence = 0.50
        elif chr_dir != 0 and tab_dir == 0:
            # Chronos has a signal while Tabular ML is neutral
            if chr_conf >= 0.70:
                direction = chr_dir
                blended_return = chr_ret * 0.75
                confidence = chr_conf * 0.85
            else:
                direction = 0
                blended_return = 0.0
                confidence = 0.50
        else:
            # Models directly contradict each other (+1 vs -1) -> STAND DOWN
            direction = 0
            blended_return = 0.0
            confidence = min(tab_conf, chr_conf) * 0.50

        # Snapshot of key features for explainability
        recent_features = {
            "rsi_14": round(float(features["rsi_14"].iloc[-1]), 2),
            "macd_hist": round(float(features["macd_hist"].iloc[-1]) * 10000, 4),
            "atr_pips": round(current_atr / (10 * df_rates["close"].iloc[-1] * 1e-4), 1) if "point" in df_rates else round(current_atr, 5),
            "vol_zscore": round(float(features["vol_zscore"].iloc[-1]), 2),
            "ret_lag_5": round(float(features["ret_lag_5"].iloc[-1]) * 100, 3),
        }

        return ForecastSignal(
            symbol=symbol,
            current_price=current_price,
            predicted_return=round(float(blended_return), 6),
            direction=direction,
            model_confidence=round(float(confidence), 4),
            horizon_bars=self.config.FORECAST_HORIZON,
            tabular_direction=tab_dir,
            tabular_confidence=round(tab_conf, 4),
            chronos_direction=chr_dir,
            chronos_confidence=round(chr_conf, 4),
            chronos_return=round(chr_ret, 6),
            chronos_p10_return=round(chr_p10, 6),
            chronos_p90_return=round(chr_p90, 6),
            atr=round(current_atr, 6),
            agreement=models_agree,
            diagnostics={
                "features": recent_features,
                "tab_raw_ret": tab_ret,
                "chr_dispersion": round(chr_p90 - chr_p10, 6),
            },
        )
