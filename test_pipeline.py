"""
Automated End-to-End Test Suite for Quantitative Trading & Forecasting System.
Validates MT5 Live Bridge, Quantitative Forecaster, and TypeSafe AI Jev Gatekeeper.
"""

import sys
import unittest
from datetime import datetime, timezone
import pandas as pd
import numpy as np

from config import settings
from mt5_connector import MT5Bridge, MarketState, AccountState, PositionInfo, OrderResult
from forecaster import QuantitativeForecaster, ForecastSignal, FeatureExtractor
from jev_guardrail import JevGatekeeper, GuardrailDecision


class TestQuantitativeSystem(unittest.TestCase):
    """Full regression suite testing all modules and risk guardrails."""

    @classmethod
    def setUpClass(cls):
        print("\n=======================================================")
        print("Starting Quantitative Trading & Forecasting Test Suite")
        print("=======================================================")
        cls.bridge = MT5Bridge(settings)
        cls.mt5_online = cls.bridge.initialize()
        if cls.mt5_online:
            print("[OK] Host MT5 Terminal connection verified.")
        else:
            print("[WARN] MT5 Terminal not active on host. Some live tests will be skipped.")

    @classmethod
    def tearDownClass(cls):
        if cls.mt5_online:
            cls.bridge.shutdown()
            print("[OK] MT5 Bridge safely disconnected.")

    # -------------------------------------------------------------------------
    # Module A: MT5 Live Bridge Tests
    # -------------------------------------------------------------------------
    def test_01_mt5_symbol_resolution_and_rates(self):
        """Test symbol resolution (e.g. GBPUSD -> broker symbol) and historical rate ingestion."""
        if not self.mt5_online:
            self.skipTest("MT5 terminal offline")

        resolved = self.bridge.resolve_symbol("GBPUSD")
        self.assertTrue(len(resolved) >= 6)
        print(f"  [MT5] Symbol 'GBPUSD' resolved to: '{resolved}'")

        market = self.bridge.get_market_state("GBPUSD")
        self.assertIsInstance(market, MarketState)
        self.assertGreater(market.bid, 0.0)
        self.assertGreater(market.ask, market.bid)
        self.assertGreater(market.spread_points, 0.0)
        print(f"  [MT5] Live Market State: Bid={market.bid}, Ask={market.ask}, Spread={market.spread_pips} pips")

        rates = self.bridge.get_rates("GBPUSD", timeframe="M15", count=100)
        self.assertIsInstance(rates, pd.DataFrame)
        self.assertEqual(len(rates), 100)
        self.assertIn("open", rates.columns)
        self.assertIn("close", rates.columns)
        self.assertIn("high", rates.columns)
        self.assertIn("low", rates.columns)
        self.assertIn("tick_volume", rates.columns)
        print(f"  [MT5] Rates Ingested: {rates.shape[0]} candles x {rates.shape[1]} columns")

    def test_02_mt5_dry_run_execution(self):
        """Test dry-run order execution scaffolding without sending real orders."""
        if not self.mt5_online:
            self.skipTest("MT5 terminal offline")

        order_res = self.bridge.send_order(
            symbol="GBPUSD",
            order_type="BUY",
            volume=0.01,
            sl_points=200,
            tp_points=400,
            dry_run=True,
        )
        self.assertIsInstance(order_res, OrderResult)
        self.assertTrue(order_res.success)
        self.assertTrue(order_res.is_dry_run)
        self.assertIn("action", order_res.details)
        print("  [MT5] Dry-run buy order validated and formatted successfully.")

    # -------------------------------------------------------------------------
    # Module B: Quantitative Forecasting Pipeline Tests
    # -------------------------------------------------------------------------
    def test_03_feature_engineering(self):
        """Verify technical indicator calculations (RSI, ATR, MACD, volatilities)."""
        # Create synthetic OHLCV data
        dates = pd.date_range("2026-01-01", periods=100, freq="15min", tz="UTC")
        prices = 1.3000 + np.cumsum(np.random.normal(0, 0.0005, size=100))
        df = pd.DataFrame(
            {
                "open": prices,
                "high": prices + 0.0004,
                "low": prices - 0.0004,
                "close": prices + 0.0001,
                "tick_volume": np.random.randint(10, 100, size=100),
            },
            index=dates,
        )
        features, atr = FeatureExtractor.extract_features(df)
        self.assertIn("rsi_14", features.columns)
        self.assertIn("atr_14", features.columns)
        self.assertIn("macd_hist", features.columns)
        self.assertIn("vol_5", features.columns)
        self.assertIn("hour_sin", features.columns)
        self.assertGreater(atr, 0.0)
        self.assertFalse(features.isna().any().any())
        print(f"  [Forecaster] Feature matrix generated: {features.shape[1]} features, ATR={atr:.5f}")

    def test_04_forecasting_pipeline_live(self):
        """Verify end-to-end forecasting pipeline output conforms to ForecastSignal schema."""
        if not self.mt5_online:
            self.skipTest("MT5 terminal offline")

        rates = self.bridge.get_rates("GBPUSD", timeframe="M15", count=150)
        forecaster = QuantitativeForecaster(settings)
        signal = forecaster.forecast(rates, "GBPUSD")

        self.assertIsInstance(signal, ForecastSignal)
        self.assertIn(signal.direction, (-1, 0, 1))
        self.assertGreaterEqual(signal.model_confidence, 0.0)
        self.assertLessEqual(signal.model_confidence, 1.0)
        self.assertGreater(signal.current_price, 0.0)
        print(
            f"  [Forecaster] Forecast Signal: Dir={signal.direction} | "
            f"ExpReturn={signal.predicted_return * 100:+.3f}% | Confidence={signal.model_confidence * 100:.1f}%"
        )

    # -------------------------------------------------------------------------
    # Module C: TypeSafe AI Jev Gatekeeper Tests
    # -------------------------------------------------------------------------
    def test_05_jev_guardrail_bullish_approval(self):
        """Test that a compliant bullish signal with low spread produces an approved BUY."""
        market = MarketState(
            symbol="GBPUSDc",
            bid=1.32500,
            ask=1.32510,
            spread_points=10.0,
            spread_pips=1.0,
            tick_volume=20,
            last_tick_time=datetime.now(timezone.utc),
            digits=5,
            point=0.00001,
            min_volume=0.01,
            max_volume=100.0,
            volume_step=0.01,
        )
        account = AccountState(
            login=999, balance=1000.0, equity=1000.0, profit=0.0,
            margin=0.0, margin_free=1000.0, margin_level=100.0, leverage=500, trade_allowed=True,
        )
        signal = ForecastSignal(
            symbol="GBPUSDc", current_price=1.32505, predicted_return=0.0020,
            direction=1, model_confidence=0.85, tabular_direction=1, tabular_confidence=0.82,
            chronos_direction=1, chronos_confidence=0.88, chronos_return=0.0020,
            chronos_p10_return=0.0003, chronos_p90_return=0.0038, atr=0.0010, agreement=True,
        )
        gk = JevGatekeeper(settings)
        decision = gk.evaluate(market, signal, [], account)

        self.assertIsInstance(decision, GuardrailDecision)
        self.assertTrue(decision.is_approved)
        self.assertEqual(decision.approved_action, "EXECUTE_BUY")
        self.assertGreaterEqual(decision.risk_acceptable_probability, 0.85)
        print(f"  [Jev] Bullish Signal Approved: Action={decision.approved_action}, RiskProb={decision.risk_acceptable_probability:.3f}")

    def test_06_jev_guardrail_bearish_approval(self):
        """Test that a compliant bearish signal produces an approved SELL."""
        market = MarketState(
            symbol="GBPUSDc",
            bid=1.32500,
            ask=1.32510,
            spread_points=10.0,
            spread_pips=1.0,
            tick_volume=20,
            last_tick_time=datetime.now(timezone.utc),
            digits=5,
            point=0.00001,
            min_volume=0.01,
            max_volume=100.0,
            volume_step=0.01,
        )
        account = AccountState(
            login=999, balance=1000.0, equity=1000.0, profit=0.0,
            margin=0.0, margin_free=1000.0, margin_level=100.0, leverage=500, trade_allowed=True,
        )
        signal = ForecastSignal(
            symbol="GBPUSDc", current_price=1.32505, predicted_return=-0.0020,
            direction=-1, model_confidence=0.85, tabular_direction=-1, tabular_confidence=0.82,
            chronos_direction=-1, chronos_confidence=0.88, chronos_return=-0.0020,
            chronos_p10_return=-0.0038, chronos_p90_return=-0.0003, atr=0.0010, agreement=True,
        )
        gk = JevGatekeeper(settings)
        decision = gk.evaluate(market, signal, [], account)

        self.assertTrue(decision.is_approved)
        self.assertEqual(decision.approved_action, "EXECUTE_SELL")
        self.assertGreaterEqual(decision.risk_acceptable_probability, 0.85)
        print(f"  [Jev] Bearish Signal Approved: Action={decision.approved_action}, RiskProb={decision.risk_acceptable_probability:.3f}")

    def test_07_jev_guardrail_rejections(self):
        """Test strict guardrail rejections on wide spread, low confidence, and conflicting directions."""
        account = AccountState(
            login=999, balance=1000.0, equity=1000.0, profit=0.0,
            margin=0.0, margin_free=1000.0, margin_level=100.0, leverage=500, trade_allowed=True,
        )
        gk = JevGatekeeper(settings)

        # 1. Wide Spread Rejection
        market_wide = MarketState(
            symbol="GBPUSDc", bid=1.32500, ask=1.32550, spread_points=50.0, spread_pips=5.0,
            tick_volume=5, last_tick_time=datetime.now(timezone.utc), digits=5, point=0.00001,
            min_volume=0.01, max_volume=100.0, volume_step=0.01,
        )
        signal_ok = ForecastSignal(
            symbol="GBPUSDc", current_price=1.32505, predicted_return=0.0020,
            direction=1, model_confidence=0.85, tabular_direction=1, tabular_confidence=0.82,
            chronos_direction=1, chronos_confidence=0.88, chronos_return=0.0020,
            chronos_p10_return=0.0003, chronos_p90_return=0.0038, atr=0.0010, agreement=True,
        )
        dec_wide = gk.evaluate(market_wide, signal_ok, [], account)
        self.assertFalse(dec_wide.is_approved)
        self.assertTrue(any("spread" in r.lower() for r in dec_wide.rejection_reasons))
        print("  [Jev] Correctly rejected wide spread trade.")

        # 2. Neutral Direction Stand Down
        market_ok = MarketState(
            symbol="GBPUSDc", bid=1.32500, ask=1.32510, spread_points=10.0, spread_pips=1.0,
            tick_volume=20, last_tick_time=datetime.now(timezone.utc), digits=5, point=0.00001,
            min_volume=0.01, max_volume=100.0, volume_step=0.01,
        )
        signal_neutral = ForecastSignal(
            symbol="GBPUSDc", current_price=1.32505, predicted_return=0.0,
            direction=0, model_confidence=0.50, tabular_direction=0, tabular_confidence=0.50,
            chronos_direction=0, chronos_confidence=0.50, chronos_return=0.0,
            chronos_p10_return=-0.001, chronos_p90_return=0.001, atr=0.0010, agreement=False,
        )
        dec_neutral = gk.evaluate(market_ok, signal_neutral, [], account)
        self.assertFalse(dec_neutral.is_approved)
        self.assertEqual(dec_neutral.approved_action, "HOLD_STAND_DOWN")
        print("  [Jev] Correctly stood down on neutral forecast.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
