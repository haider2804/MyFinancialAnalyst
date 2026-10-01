# TypeSafe Jev + MT5 Quantitative Trading & Self-Evaluating Analyst

An enterprise-grade quantitative forecasting, risk-governed trade execution, and self-evaluating analytics platform built on MetaTrader 5 (MT5), machine learning, Amazon Chronos foundation time-series priors, and deterministic AI risk guardrails.

---

## Key Highlights & Architecture

1. **MetaTrader 5 Live Bridge (`mt5_connector.py`)**
   - High-reliability IPC bridge to active MetaTrader 5 terminal sessions.
   - Real-time tick streaming, multi-timeframe OHLCV bar ingestion, and auto-suffix resolution (e.g. `XAUUSD` $\to$ `XAUUSDc`).
   - Supports Advisory, Dry-Run simulated execution, and live broker order routing.

2. **Dual-Tier Quantitative Forecasting Engine (`forecaster.py`)**
   - **Tabular ML:** LightGBM on engineered features (RSI-14, ATR-14, MACD histogram, rolling volume z-scores, return lags).
   - **Zero-Shot Foundation Prior:** Amazon Chronos (`amazon/chronos-bolt-tiny`) probabilistic quantile forecasting ($p_{10}, p_{50}, p_{90}$).
   - Dynamic Average True Range (ATR) based Stop Loss (1.5x) and Take Profit (3.0x) levels.

3. **TypeSafe AI Jev Gatekeeper (`jev_guardrail.py`)**
   - Deterministic risk supervisor that inspects market spread, volatility regime, margin health, and concurrent exposure.
   - Enforces a strict $\ge 85\%$ risk acceptance threshold before any trade is permitted.

4. **Self-Evaluating Strategy Tracker (`tracker.py`)**
   - Persistently logs all trade setups and direction into an SQLite database (`trading_performance.db`).
   - Self-evaluates prediction accuracy, win rate, net pips, and profit factor against subsequent live market prices.
   - Throttled to 1 signal per candle bar to eliminate duplicate polling noise.

5. **Interactive Real-Time Web Cockpit (`server.py` + `static/`)**
   - FastAPI and WebSockets streaming live charts, Chronos probabilistic fan trajectories, and strategy track record.
   - Real-time symbol and timeframe switching (Gold `XAUUSD`, Bitcoin `BTCUSD`, Forex pairs).
   - Interactive Historical Signals & Realized Outcomes table with dynamic filtering and Reset History controls.

---

## Getting Started

### 1. Requirements
* Python 3.10+ (tested on Python 3.13)
* MetaTrader 5 Desktop Terminal installed and logged in (Demo or Live)

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/haider2804/MyFinancialAnalyst.git
cd MyFinancialAnalyst

# Install dependencies
pip install -r requirements.txt
```

### 3. MT5 Configuration
1. Open your **MetaTrader 5** terminal.
2. In the top menu, go to **Tools** $\to$ **Options** (or press `Ctrl + O`).
3. Select the **Expert Advisors** tab and check:
   - [x] **Allow Algorithmic Trading**
4. Ensure the **Algo Trading** button on the MT5 toolbar is green.

### 4. Running the System

#### Option A: Web Cockpit Server (Recommended)
Double-click `start.bat` or run:
```bash
python server.py
```
Open [http://localhost:8000](http://localhost:8000) in your browser.

To stop the server cleanly, double-click `stop.bat` or run:
```bash
python stop_analyst.py
```

#### Option B: Terminal CLI Orchestrator
```bash
# Advisory Mode (signals only)
python main.py --mode advisory --symbol XAUUSD --timeframe M5

# Dry-Run Mode (simulated order validation)
python main.py --mode dry-run --symbol XAUUSD --timeframe M5
```

---

## Project Structure

```
MyFinancialAnalyst/
├── config.py              # Centralized Pydantic settings & spread ceilings
├── forecaster.py          # Tabular ML + Chronos foundation model forecaster
├── jev_guardrail.py       # TypeSafe Jev deterministic risk gatekeeper
├── main.py                # Terminal CLI orchestrator
├── mt5_connector.py       # High-reliability MetaTrader 5 bridge
├── requirements.txt       # Python dependencies
├── server.py              # FastAPI live server + WebSocket streaming
├── SETUP_GUIDE.md         # Detailed operations and MT5 setup documentation
├── start.bat              # One-click startup script
├── stop.bat               # One-click termination script
├── stop_analyst.py        # Process cleanup utility
├── test_pipeline.py       # End-to-end regression test suite
├── tracker.py             # SQLite strategy self-evaluation engine
└── static/                # Interactive cockpit frontend (HTML/CSS/JS)
```

---

## License
MIT License
