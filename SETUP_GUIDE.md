# TypeSafe Jev + MetaTrader 5 Quantitative Trading System
### Production Setup, Architecture & Operations Guide

---

## 1. System Overview

This system is an enterprise-grade quantitative trading and forecasting architecture that merges:
1. **Live MetaTrader 5 Bridge (`mt5_connector.py`)**: Interacts directly with the local host MT5 terminal for tick streaming, OHLCV ingestion, and dry-run/live order dispatch.
2. **Dual-Tier Quantitative Forecasting Engine (`forecaster.py`)**:
   - **Tabular ML**: LightGBM/XGBoost on engineered multi-horizon features (RSI-14, ATR-14, MACD, rolling volatilities, cyclical diurnal encodings).
   - **Zero-Shot Foundation Model Prior**: Amazon Chronos (`amazon/chronos-bolt-tiny`) probabilistic trajectory forecasting across quantiles ($p_{10}, p_{50}, p_{90}$).
3. **TypeSafe AI Jev Gatekeeper (`jev_guardrail.py`)**: A deterministic System 1 supervisor using `typesafe-sdk` (`Choice`, `Noul`, `Score`) to strictly enforce spread limits, volatility regimes, and require $\ge 0.85$ risk acceptance before any trade can execute.
4. **Master Orchestrator & CLI (`main.py`)**: Real-time terminal dashboard, configurable polling loop, and graceful SIGINT/SIGTERM shutdown traps.

---

## 2. MetaTrader 5 Terminal Host Configuration

To allow Python to stream market data and dispatch orders through your local MT5 terminal, you must enable algorithmic trading and network permissions within MT5.

### Step 2.1: Enable "Allow Algorithmic Trading"
1. Open your **MetaTrader 5** terminal desktop application.
2. In the top menu bar, navigate to:
   `Tools` -> `Options` (or press `Ctrl + O`).
3. Select the **Expert Advisors** tab.
4. Check the following boxes:
   - [x] **Allow Algorithmic Trading**
   - [ ] *(Optional)* Disable "Disable algorithmic trading when the account has been changed" if using multi-account scripts.
   - [ ] *(Optional)* Disable "Disable algorithmic trading when the profile has been changed".
5. Ensure the **Algo Trading** button on the MT5 main toolbar is highlighted green.

### Step 2.2: Enable "Allow WebRequest"
If you use MQL5 indicators or webhook bridges that call TypeSafe AI directly from within MetaTrader:
1. In the same **Expert Advisors** tab (`Ctrl + O`), check:
   - [x] **Allow WebRequest for listed URL:**
2. Click the green `+` or double click to add the following endpoints:
   - `https://api.typesafe.ai`
   - `https://huggingface.co` (for Chronos model weight downloads)
3. Click **OK** to save changes.

### Step 2.3: Broker Symbol Suffixes & MarketWatch
Brokers frequently append suffixes to symbol names depending on your account type:
- Standard / Cent accounts: `GBPUSDc`, `EURUSDc`
- Zero / Raw spread accounts: `GBPUSD.m`, `GBPUSD_i`, `GBPUSDpro`

> **Note:** The `MT5Bridge` class in `mt5_connector.py` automatically resolves symbol aliases. Requesting `GBPUSD` will automatically discover and bind to `GBPUSDc` (or equivalent) in your broker's active instrument directory and activate it in MarketWatch.

---

## 3. Environment & Dependency Setup

### Prerequisites
- Python 3.10+ (tested on Python 3.13)
- Active MetaTrader 5 terminal logged into your broker (Demo or Live)
- NVIDIA GPU with CUDA (optional; runs seamlessly on CPU)

### Installation
```bash
# 1. Clone or navigate to the project directory
cd f:/MyPersonalProjects/MyFinancialAnalyst

# 2. Install all required dependencies
pip install -r requirements.txt
```

### Configuration (.env)
Copy the template configuration:
```bash
cp .env.example .env
```

Edit `.env` to supply your credentials:
```env
# TypeSafe AI API Key
TYPESAFE_API_KEY=your_typesafe_api_key_here
TYPESAFE_MODEL_VERSION=jev-latest

# Operational Mode
DRY_RUN=True
SYMBOL=GBPUSD
TIMEFRAME=M15
POLL_INTERVAL_SECONDS=15
```

---

## 4. System Architecture & Component Verification

```
+-----------------------------------------------------------------------------------+
|                           MetaTrader 5 Host Terminal                             |
|          Live Ticks | Historical Candles | Account Margin | Order Dispatch        |
+-----------------------------------------------------------------------------------+
                                         ▲
                                         │ IPC / Direct Python API
                                         ▼
+-----------------------------------------------------------------------------------+
|                         Module A: MT5 Live Bridge                                 |
|                         (mt5_connector.py)                                        |
|  - Auto-symbol resolution (GBPUSD -> GBPUSDc)                                     |
|  - Ingests M15/H1 rates -> typed pandas DataFrame                                 |
|  - Pre-trade validation & dry-run order scaffolding                               |
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼
+-----------------------------------------------------------------------------------+
|                    Module B: Quantitative Forecaster                              |
|                    (forecaster.py)                                                |
|  ┌───────────────────────────────────┐    ┌────────────────────────────────────┐  |
|  │ Tier 1: Feature-Engineered ML     │    │ Tier 2: Zero-Shot Foundation Prior │  |
|  │ - RSI, ATR, MACD, Volatility Lags │    │ - Amazon Chronos Bolt (Transformer)│  |
|  │ - LightGBM Direction Classifier   │    │ - Trajectory Quantiles p10/p50/p90 │  |
|  └───────────────────────────────────┘    └────────────────────────────────────┘  |
|                                         │                                         |
|                                         ▼                                         |
|                           Synthesized ForecastSignal                              |
|               predicted_return | direction (-1, 0, 1) | model_confidence           |
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼
+-----------------------------------------------------------------------------------+
|                    Module C: TypeSafe AI Jev Gatekeeper                           |
|                    (jev_guardrail.py)                                             |
|  Unified State: Spread Pips | ATR Ratio | Open Positions | Quant Signal | Session  |
|                                                                                   |
|  Typed Jev Questions:                                                             |
|  • approved_action: Choice ["EXECUTE_BUY", "EXECUTE_SELL", "HOLD_STAND_DOWN"]      |
|  • risk_acceptable: Noul (Yes/No calibrated probability)                          |
|  • market_regime:   Score (Calm -> Extreme scale)                                 |
|                                                                                   |
|  Enforcement Rules:                                                               |
|  ✔ Reject if risk_acceptable probability < 0.85                                  |
|  ✔ Reject if approved_action contradicts quantitative forecast                    |
|  ✔ Reject if spread > 3.0 pips or max positions reached                           |
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼
+-----------------------------------------------------------------------------------+
|                    Module D: Master Orchestrator CLI                              |
|                    (main.py)                                                      |
|  - Live terminal dashboard (Rich tables & telemetry)                              |
|  - Dispatches validated order (with ATR-scaled Stop Loss & Take Profit)           |
|  - Graceful SIGINT/SIGTERM shutdown calling mt5.shutdown()                        |
+-----------------------------------------------------------------------------------+
```

---

## 5. Running the System

### Running Automated Test Suite
To verify end-to-end functionality across all 3 modules and the terminal bridge:
```bash
python test_pipeline.py
```
Expected output:
```
Ran 7 tests in ~2.3s
OK
```

### Running Single Diagnostic Pass
Runs a single analysis iteration and prints the visual dashboard:
```bash
python main.py --single-run --symbol GBPUSD --timeframe M15
```

### Running Persistent Polling Loop (Dry-Run Mode)
Monitors the market continuously every 15 seconds in simulated dry-run mode:
```bash
python main.py --symbol GBPUSD --timeframe M15 --poll-interval 15
```

### Live Execution Mode
> **CAUTION:** Only activate live trading after thoroughly testing your risk thresholds in a Demo account!
```bash
python main.py --live --symbol GBPUSD --volume 0.01 --max-spread 2.5
```

---

## 6. CLI Command-Line Reference

| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--symbol` | `str` | `GBPUSD` | Target currency pair or CFD. |
| `--timeframe` | `str` | `M15` | Candle timeframe (`M1`, `M5`, `M15`, `H1`, `D1`). |
| `--volume` | `float` | `0.01` | Lot size per order execution. |
| `--poll-interval` | `int` | `15` | Polling cadence in seconds. |
| `--max-spread` | `float` | `3.0` | Maximum acceptable spread in pips before trade rejection. |
| `--single-run` | `flag` | `False` | Run one analysis iteration and exit (for cron/scripts). |
| `--live` | `flag` | `False` | Dispatches orders live to broker (default is dry-run). |
