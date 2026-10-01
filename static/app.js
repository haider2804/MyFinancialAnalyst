/**
 * TypeSafe Jev + MT5 Interactive Cockpit Application (app.js)
 * Real-time WebSocket streaming, interactive candlestick rendering,
 * strategy track-record analytics, and instant instrument switching.
 */

class TradingCockpitApp {
  constructor() {
    this.ws = null;
    this.state = null;
    this.signals = [];
    this.candles = [];
    this.currentFilter = 'ALL';
    this.activeTab = 'tab-cockpit';

    this.initElements();
    this.initTabs();
    this.initFilters();
    this.initControls();
    this.initWebSocket();
    this.fetchInitialData();
  }

  initElements() {
    // Top Bar
    this.heartbeatDot = document.getElementById('heartbeat-dot');
    this.topSymbol = document.getElementById('top-symbol');
    this.topTimeframe = document.getElementById('top-timeframe');
    this.engineStatusBadge = document.getElementById('engine-status-badge');
    this.lastSyncTime = document.getElementById('last-sync-time');
    this.accServer = document.getElementById('acc-server');
    this.accLogin = document.getElementById('acc-login');
    this.accEquity = document.getElementById('acc-equity');
    this.accMargin = document.getElementById('acc-margin');
    this.btnScanNow = document.getElementById('btn-scan-now');
    this.badgeTotalEval = document.getElementById('badge-total-eval');

    // Ticker & Quant
    this.tickerBid = document.getElementById('ticker-bid');
    this.tickerAsk = document.getElementById('ticker-ask');
    this.tickerSpreadPips = document.getElementById('ticker-spread-pips');
    this.tickerSpreadPts = document.getElementById('ticker-spread-pts');
    this.forecastDirText = document.getElementById('forecast-dir-text');
    this.forecastReturn = document.getElementById('forecast-return');
    this.forecastConfidence = document.getElementById('forecast-confidence');
    this.confidenceBar = document.getElementById('confidence-bar');
    this.metaTabDir = document.getElementById('meta-tab-dir');
    this.metaChrDir = document.getElementById('meta-chr-dir');

    // Jev
    this.jevStatusBadge = document.getElementById('jev-status-badge');
    this.jevRiskProb = document.getElementById('jev-risk-prob');
    this.jevRegimeScore = document.getElementById('jev-regime-score');
    this.jevApprovalText = document.getElementById('jev-approval-text');

    // Trade Card
    this.tradeCard = document.getElementById('trade-card');
    this.tradeCardTag = document.getElementById('trade-card-tag');
    this.tradeCardSym = document.getElementById('trade-card-sym');
    this.tradeActionBadge = document.getElementById('trade-action-badge');
    this.tradeCardDesc = document.getElementById('trade-card-desc');
    this.tradeLevelsGrid = document.getElementById('trade-levels-grid');
    this.levelEntry = document.getElementById('level-entry');
    this.levelSl = document.getElementById('level-sl');
    this.levelSlPips = document.getElementById('level-sl-pips');
    this.levelTp = document.getElementById('level-tp');
    this.levelTpPips = document.getElementById('level-tp-pips');
    this.rejectionReasonsBox = document.getElementById('rejection-reasons-box');
    this.rejectionList = document.getElementById('rejection-list');

    // KPI & Performance
    this.kpiWinrate = document.getElementById('kpi-winrate');
    this.kpiWinrateBadge = document.getElementById('kpi-winrate-badge');
    this.kpiWins = document.getElementById('kpi-wins');
    this.kpiLosses = document.getElementById('kpi-losses');
    this.kpiNetPips = document.getElementById('kpi-net-pips');
    this.kpiPf = document.getElementById('kpi-pf');

    // Track Record Tab Elements
    this.trWinrate = document.getElementById('tr-winrate');
    this.trWinLossCount = document.getElementById('tr-win-loss-count');
    this.trNetPips = document.getElementById('tr-net-pips');
    this.trProfitFactor = document.getElementById('tr-profit-factor');
    this.trTotalTrades = document.getElementById('tr-total-trades');
    this.trPendingCount = document.getElementById('tr-pending-count');
    this.signalsTableBody = document.getElementById('signals-table-body');
    this.tableSummaryPill = document.getElementById('table-summary-pill');

    // Canvases
    this.candleCanvas = document.getElementById('candle-chart');
    this.chronosCanvas = document.getElementById('chronos-chart');
    this.equityCanvas = document.getElementById('equity-chart');
  }

  initTabs() {
    const tabButtons = document.querySelectorAll('.tab-btn');
    tabButtons.forEach(btn => {
      btn.addEventListener('click', () => {
        tabButtons.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        const targetId = btn.getAttribute('data-tab');
        this.activeTab = targetId;
        document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
        const pane = document.getElementById(targetId);
        if (pane) pane.classList.add('active');

        // Re-render charts when their tab becomes visible
        if (targetId === 'tab-cockpit') {
          this.renderCandleChart();
          this.renderChronosFan();
        } else if (targetId === 'tab-track-record') {
          this.fetchSignals();
          this.renderEquityChart();
        }
      });
    });
  }

  initFilters() {
    const filterButtons = document.querySelectorAll('.btn-filter');
    filterButtons.forEach(btn => {
      btn.addEventListener('click', () => {
        filterButtons.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        this.currentFilter = btn.getAttribute('data-filter');
        this.renderSignalsTable();
      });
    });
  }

  initControls() {
    // Ribbon Switcher Buttons (Gold vs Bitcoin)
    document.querySelectorAll('.btn-asset-switch').forEach(btn => {
      btn.addEventListener('click', async () => {
        const sym = btn.getAttribute('data-sym');
        document.querySelectorAll('.btn-asset-switch').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        document.querySelectorAll('.btn-symbol').forEach(b => {
          b.classList.toggle('active', b.getAttribute('data-sym') === sym);
        });
        await this.updateSettings({ symbol: sym });
        await this.fetchCandles();
        await this.fetchSignals();
      });
    });

    // Quick Symbol Buttons in Settings Tab
    document.querySelectorAll('.btn-symbol').forEach(btn => {
      btn.addEventListener('click', async () => {
        const sym = btn.getAttribute('data-sym');
        document.querySelectorAll('.btn-symbol').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        const isGold = sym.includes('XAU');
        const isBtc = sym.includes('BTC');
        const btnXau = document.getElementById('btn-asset-xau');
        const btnBtc = document.getElementById('btn-asset-btc');
        if (btnXau) btnXau.classList.toggle('active', isGold);
        if (btnBtc) btnBtc.classList.toggle('active', isBtc);
        await this.updateSettings({ symbol: sym });
        await this.fetchCandles();
        await this.fetchSignals();
      });
    });

    // Timeframe Buttons
    document.querySelectorAll('.btn-tf').forEach(btn => {
      btn.addEventListener('click', async () => {
        const tf = btn.getAttribute('data-tf');
        document.querySelectorAll('.btn-tf').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        await this.updateSettings({ timeframe: tf });
        await this.fetchCandles();
        await this.fetchSignals();
      });
    });

    // Chart Timeframe Quick Toggles
    const btnChart5m = document.getElementById('btn-chart-5m');
    const btnChart15m = document.getElementById('btn-chart-15m');
    if (btnChart5m && btnChart15m) {
      btnChart5m.addEventListener('click', async () => {
        btnChart5m.classList.add('active');
        btnChart15m.classList.remove('active');
        await this.updateSettings({ timeframe: 'M5' });
        await this.fetchCandles();
      });
      btnChart15m.addEventListener('click', async () => {
        btnChart15m.classList.add('active');
        btnChart5m.classList.remove('active');
        await this.updateSettings({ timeframe: 'M15' });
        await this.fetchCandles();
      });
    }

    // Scan Now Button
    if (this.btnScanNow) {
      this.btnScanNow.addEventListener('click', async () => {
        this.btnScanNow.classList.add('loading');
        try {
          await fetch('/api/scan-now', { method: 'POST' });
        } catch (e) {
          console.error('Scan now error:', e);
        }
        setTimeout(() => this.btnScanNow.classList.remove('loading'), 800);
      });
    }

    // Reset History Button
    const btnResetHistory = document.getElementById('btn-reset-history');
    if (btnResetHistory) {
      btnResetHistory.addEventListener('click', async () => {
        if (!confirm('Are you sure you want to clear historical signal records and start tracking fresh?')) return;
        try {
          await fetch('/api/reset-signals', { method: 'POST' });
          await this.fetchSignals();
          this.renderEquityChart();
        } catch (e) {
          console.error('Reset signals error:', e);
        }
      });
    }
  }

  async updateSettings(payload) {
    try {
      const res = await fetch('/api/settings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      console.log('Settings updated:', data);
    } catch (e) {
      console.error('Failed to update settings:', e);
    }
  }

  initWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;

    console.log('Connecting to WebSocket:', wsUrl);
    this.ws = new WebSocket(wsUrl);

    this.ws.onopen = () => {
      console.log('WebSocket connected.');
      this.heartbeatDot.className = 'pulse-indicator online';
      this.engineStatusBadge.textContent = 'ONLINE & EVALUATING';
      this.engineStatusBadge.className = 'status-badge active';
    };

    this.ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        this.handleStateUpdate(data);
      } catch (e) {
        console.error('WS parse error:', e);
      }
    };

    this.ws.onclose = () => {
      console.warn('WebSocket closed. Reconnecting in 3s...');
      this.heartbeatDot.className = 'pulse-indicator';
      this.engineStatusBadge.textContent = 'RECONNECTING...';
      this.engineStatusBadge.className = 'status-badge';
      setTimeout(() => this.initWebSocket(), 3000);
    };

    this.ws.onerror = (err) => {
      console.error('WS error:', err);
    };
  }

  async fetchInitialData() {
    try {
      const [stateRes, historyRes, signalsRes] = await Promise.all([
        fetch('/api/state').then(r => r.json()),
        fetch('/api/history').then(r => r.json()),
        fetch('/api/signals').then(r => r.json()),
      ]);

      if (stateRes) this.handleStateUpdate(stateRes);
      if (historyRes && historyRes.candles) {
        this.candles = historyRes.candles;
        this.renderCandleChart();
      }
      if (signalsRes && signalsRes.signals) {
        this.signals = signalsRes.signals;
        this.renderSignalsTable();
        this.renderEquityChart();
      }
    } catch (e) {
      console.error('Error fetching initial data:', e);
    }
  }

  async fetchCandles() {
    try {
      const res = await fetch('/api/history');
      const data = await res.json();
      if (data && data.candles) {
        this.candles = data.candles;
        this.renderCandleChart();
      }
    } catch (e) {
      console.error('Error fetching candles:', e);
    }
  }

  async fetchSignals() {
    try {
      const res = await fetch('/api/signals');
      const data = await res.json();
      if (data && data.signals) {
        this.signals = data.signals;
        this.renderSignalsTable();
        this.renderEquityChart();
      }
    } catch (e) {
      console.error('Error fetching signals:', e);
    }
  }

  handleStateUpdate(data) {
    this.state = data;
    const now = new Date();
    this.lastSyncTime.textContent = now.toLocaleTimeString();

    // Flash heartbeat
    this.heartbeatDot.style.transform = 'scale(1.4)';
    setTimeout(() => this.heartbeatDot.style.transform = 'scale(1.0)', 250);

    // 1. Top Bar & Asset Switcher Sync
    if (data.active_symbol) {
      this.topSymbol.textContent = data.active_symbol;
      const activeSym = data.active_symbol.toUpperCase();
      const isGold = activeSym.includes('XAU') || activeSym.includes('GOLD');
      const isBtc = activeSym.includes('BTC') || activeSym.includes('BTX');

      const btnXau = document.getElementById('btn-asset-xau');
      const btnBtc = document.getElementById('btn-asset-btc');
      if (btnXau) btnXau.classList.toggle('active', isGold);
      if (btnBtc) btnBtc.classList.toggle('active', isBtc);

      document.querySelectorAll('.btn-symbol').forEach(b => {
        const sym = (b.getAttribute('data-sym') || '').toUpperCase();
        if (isGold && sym.includes('XAU')) b.classList.add('active');
        else if (isBtc && (sym.includes('BTC') || sym.includes('BTX'))) b.classList.add('active');
        else if (sym === activeSym) b.classList.add('active');
        else b.classList.remove('active');
      });

      // Spread status text
      const spreadIndicator = document.getElementById('spread-indicator');
      if (spreadIndicator && data.market) {
        if (isGold) {
          spreadIndicator.textContent = data.market.spread_pips <= 35 ? 'Optimal Gold Spread' : 'High Gold Spread';
          spreadIndicator.className = data.market.spread_pips <= 35 ? 'spread-indicator optimal' : 'spread-indicator warning';
        } else if (isBtc) {
          spreadIndicator.textContent = data.market.spread_points <= 2000 ? 'Optimal BTC Spread' : 'High BTC Spread';
          spreadIndicator.className = data.market.spread_points <= 2000 ? 'spread-indicator optimal' : 'spread-indicator warning';
        } else {
          spreadIndicator.textContent = data.market.spread_pips <= 3.0 ? 'Optimal FX Spread' : 'High FX Spread';
          spreadIndicator.className = data.market.spread_pips <= 3.0 ? 'spread-indicator optimal' : 'spread-indicator warning';
        }
      }
    }
    if (data.active_timeframe) this.topTimeframe.textContent = data.active_timeframe;

    if (data.account) {
      this.accServer.textContent = 'Exness-MT5Real34';
      this.accLogin.textContent = `#${data.account.login}`;
      this.accEquity.textContent = `$${data.account.equity.toLocaleString('en-US', { minimumFractionDigits: 2 })}`;
      this.accMargin.textContent = `$${data.account.margin_free.toLocaleString('en-US', { minimumFractionDigits: 2 })}`;
    }

    // 2. Ticker
    if (data.market) {
      const digits = data.market.digits !== undefined ? data.market.digits : 2;
      this.tickerBid.textContent = data.market.bid.toFixed(digits);
      this.tickerAsk.textContent = data.market.ask.toFixed(digits);
      this.tickerSpreadPips.textContent = data.market.spread_pips.toFixed(1);
      this.tickerSpreadPts.textContent = data.market.spread_points.toFixed(0);
      const digitsBadge = document.getElementById('market-digits-badge');
      if (digitsBadge) digitsBadge.textContent = `${digits}-Digit`;
    }

    // 3. Quant Forecast
    if (data.forecast) {
      const dir = data.forecast.direction;
      const conf = (data.forecast.model_confidence * 100).toFixed(1);
      const ret = (data.forecast.predicted_return * 100).toFixed(3);

      this.forecastConfidence.textContent = `${conf}%`;
      this.confidenceBar.style.width = `${Math.min(100, conf)}%`;
      this.forecastReturn.textContent = `${data.forecast.predicted_return >= 0 ? '+' : ''}${ret}%`;

      if (dir === 1) {
        this.forecastDirText.textContent = 'BULLISH (+1)';
        this.forecastDirText.className = 'forecast-direction-text text-green';
      } else if (dir === -1) {
        this.forecastDirText.textContent = 'BEARISH (-1)';
        this.forecastDirText.className = 'forecast-direction-text text-red';
      } else {
        this.forecastDirText.textContent = 'NEUTRAL (0)';
        this.forecastDirText.className = 'forecast-direction-text text-amber';
      }

      this.metaTabDir.textContent = data.forecast.tabular_direction;
      this.metaChrDir.textContent = data.forecast.chronos_direction;

      // Update Diagnostics Tab indicators
      if (data.forecast.diagnostics && data.forecast.diagnostics.features) {
        const feat = data.forecast.diagnostics.features;
        const rsiEl = document.getElementById('feat-rsi');
        const atrEl = document.getElementById('feat-atr');
        const macdEl = document.getElementById('feat-macd');
        const volzEl = document.getElementById('feat-volz');
        const ret5El = document.getElementById('feat-ret5');

        if (rsiEl) rsiEl.textContent = feat.rsi_14 || '--';
        if (atrEl) atrEl.textContent = feat.atr_pips || '--';
        if (macdEl) macdEl.textContent = feat.macd_hist || '--';
        if (volzEl) volzEl.textContent = feat.vol_zscore || '--';
        if (ret5El) ret5El.textContent = `${feat.ret_lag_5 || 0}%`;
      }
    }

    // 4. Jev Supervisor
    if (data.decision) {
      const prob = (data.decision.risk_acceptable_probability * 100).toFixed(1);
      this.jevRiskProb.textContent = `${prob}%`;
      this.jevRiskProb.className = data.decision.risk_acceptable_probability >= 0.85 ? 'jev-val text-green' : 'jev-val text-red';
      this.jevRegimeScore.textContent = `${data.decision.market_regime_score.toFixed(2)} / 5.0`;

      if (data.decision.is_approved) {
        this.jevApprovalText.textContent = 'TRADE APPROVED';
        this.jevApprovalText.className = 'text-green';
        this.jevStatusBadge.textContent = data.decision.approved_action;
        this.jevStatusBadge.className = 'badge badge-green';
      } else {
        this.jevApprovalText.textContent = 'STAND DOWN';
        this.jevApprovalText.className = 'text-amber';
        this.jevStatusBadge.textContent = data.decision.approved_action;
        this.jevStatusBadge.className = 'badge badge-amber';
      }
    }

    // 5. Master Trade Advisory Card
    this.updateTradeCard(data);

    // 6. Strategy Performance KPIs
    if (data.performance) {
      const p = data.performance;
      this.kpiWinrate.textContent = `${p.win_rate_pct.toFixed(1)}%`;
      this.kpiWinrateBadge.textContent = `${p.win_rate_pct.toFixed(1)}% Win`;
      this.kpiWins.textContent = `${p.directional_wins} Wins`;
      this.kpiLosses.textContent = `${p.directional_losses} Losses`;
      this.kpiNetPips.textContent = `${p.total_net_pips >= 0 ? '+' : ''}${p.total_net_pips.toFixed(1)}`;
      this.kpiPf.textContent = p.profit_factor.toFixed(2);

      // Track Record tab mirrors
      this.trWinrate.textContent = `${p.win_rate_pct.toFixed(1)}%`;
      this.trWinLossCount.textContent = `${p.directional_wins} Wins / ${p.directional_losses} Losses`;
      this.trNetPips.textContent = `${p.total_net_pips >= 0 ? '+' : ''}${p.total_net_pips.toFixed(1)} pips`;
      this.trProfitFactor.textContent = p.profit_factor.toFixed(2);
      this.trTotalTrades.textContent = p.evaluated_signals;
      this.trPendingCount.textContent = p.pending_signals;
      this.badgeTotalEval.textContent = p.evaluated_signals;
    }

    // Re-render Chronos Trajectory fan if forecast present
    if (data.forecast) {
      this.renderChronosFan();
    }
  }

  updateTradeCard(data) {
    const setup = data.trade_setup;
    const decision = data.decision;
    const digits = data.market ? data.market.digits : 2;

    if (!setup || !decision) return;

    this.tradeCardSym.textContent = `${data.active_symbol} (${data.active_timeframe})`;

    if (decision.is_approved && setup.has_trade) {
      const isBuy = setup.action === 'BUY';
      this.tradeCard.className = isBuy ? 'trade-card active-buy' : 'trade-card active-sell';
      this.tradeCardTag.textContent = `⚡ LIVE ${setup.action} OPPORTUNITY CONFIRMED`;
      this.tradeActionBadge.textContent = `${setup.action} ${data.active_symbol}`;
      this.tradeCardDesc.textContent = `All quantitative ML, Chronos foundation priors, and TypeSafe Jev safety checks approved this execution setup.`;

      // Fill Levels
      this.levelEntry.textContent = setup.entry_price.toFixed(digits);
      this.levelSl.textContent = setup.sl_price.toFixed(digits);
      this.levelSlPips.textContent = `-${setup.sl_pips} pips (1.5x ATR)`;
      this.levelTp.textContent = setup.tp_price.toFixed(digits);
      this.levelTpPips.textContent = `+${setup.tp_pips} pips (3.0x ATR)`;

      this.tradeLevelsGrid.style.display = 'grid';
      this.rejectionReasonsBox.style.display = 'none';
    } else {
      this.tradeCard.className = 'trade-card stand-down';
      this.tradeCardTag.textContent = 'STAND DOWN - AWAITING SETUP';
      this.tradeActionBadge.textContent = 'HOLD POSITION';
      this.tradeCardDesc.textContent = 'Quantitative criteria or Jev risk bounds are currently active. Engine is monitoring every tick.';
      this.tradeLevelsGrid.style.display = 'none';
      this.rejectionReasonsBox.style.display = 'block';

      // Populate rejection reasons
      const reasons = decision.rejection_reasons || ['Awaiting directional breakout confirmation.'];
      this.rejectionList.innerHTML = reasons.map(r => `<li>${r}</li>`).join('');
    }
  }

  renderSignalsTable() {
    if (!this.signalsTableBody) return;

    let filtered = this.signals;
    if (this.currentFilter !== 'ALL') {
      filtered = this.signals.filter(s => s.status === this.currentFilter);
    }

    this.tableSummaryPill.textContent = `Showing ${filtered.length} of ${this.signals.length} records`;

    if (filtered.length === 0) {
      this.signalsTableBody.innerHTML = `
        <tr>
          <td colspan="10" class="text-center py-4 text-muted">
            No trade signals found under filter "${this.currentFilter}".
          </td>
        </tr>
      `;
      return;
    }

    const html = filtered.map(sig => {
      const isWon = sig.status === 'WON_TP';
      const isLost = sig.status === 'LOST_SL';
      const isPending = sig.status === 'PENDING';

      let statusBadge = '<span class="outcome-pending">PENDING</span>';
      if (isWon) statusBadge = '<span class="outcome-won">WON (TP)</span>';
      else if (isLost) statusBadge = '<span class="outcome-lost">LOST (SL)</span>';

      const dirBadge = sig.direction === 1
        ? '<span class="badge badge-green">BUY</span>'
        : '<span class="badge badge-red">SELL</span>';

      const pips = sig.realized_pips != null
        ? `${sig.realized_pips >= 0 ? '+' : ''}${sig.realized_pips.toFixed(1)}`
        : '--';
      const pipsClass = sig.realized_pips >= 0 ? 'text-green' : (sig.realized_pips < 0 ? 'text-red' : '');

      const ret = sig.realized_return != null
        ? `${(sig.realized_return * 100).toFixed(2)}%`
        : '--';

      const timeStr = sig.timestamp ? new Date(sig.timestamp).toLocaleTimeString() : '--';

      return `
        <tr>
          <td><strong>#${sig.id}</strong></td>
          <td>${timeStr}</td>
          <td><strong>${sig.symbol}</strong> <span class="text-dim text-xs">${sig.timeframe}</span></td>
          <td>${dirBadge}</td>
          <td>${sig.entry_price.toFixed(2)}</td>
          <td class="text-red">${sig.sl_price.toFixed(2)}</td>
          <td class="text-green">${sig.tp_price.toFixed(2)}</td>
          <td>${statusBadge}</td>
          <td class="${pipsClass}"><strong>${pips}</strong></td>
          <td>${ret}</td>
        </tr>
      `;
    }).join('');

    this.signalsTableBody.innerHTML = html;
  }

  // ========================================================================
  // Canvas Charts (Candles, Chronos Fan, Equity Curve)
  // ========================================================================

  renderCandleChart() {
    const canvas = this.candleCanvas;
    if (!canvas || this.candles.length === 0) return;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    ctx.clearRect(0, 0, width, height);

    // Grid lines
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.04)';
    ctx.lineWidth = 1;
    for (let y = 30; y < height; y += 45) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }

    const n = Math.min(45, this.candles.length);
    const subset = this.candles.slice(-n);

    let minPrice = Math.min(...subset.map(c => c.low));
    let maxPrice = Math.max(...subset.map(c => c.high));
    const range = (maxPrice - minPrice) || 1;
    minPrice -= range * 0.05;
    maxPrice += range * 0.05;

    const toY = (val) => height - 20 - ((val - minPrice) / (maxPrice - minPrice)) * (height - 40);

    const candleWidth = (width - 60) / n;

    subset.forEach((c, idx) => {
      const x = 20 + idx * candleWidth;
      const isUp = c.close >= c.open;
      const bodyColor = isUp ? '#10b981' : '#ef4444';

      // Wick
      ctx.strokeStyle = bodyColor;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(x + candleWidth / 2, toY(c.high));
      ctx.lineTo(x + candleWidth / 2, toY(c.low));
      ctx.stroke();

      // Body
      const topY = toY(Math.max(c.open, c.close));
      const bottomY = toY(Math.min(c.open, c.close));
      const bodyH = Math.max(2, bottomY - topY);

      ctx.fillStyle = bodyColor;
      ctx.fillRect(x + 2, topY, Math.max(2, candleWidth - 4), bodyH);
    });

    // Draw Live Price Line
    if (this.state && this.state.market) {
      const ask = this.state.market.ask;
      const askY = toY(ask);
      ctx.strokeStyle = '#06b6d4';
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(0, askY);
      ctx.lineTo(width - 5, askY);
      ctx.stroke();
      ctx.setLineDash([]);

      ctx.fillStyle = '#06b6d4';
      ctx.font = '11px JetBrains Mono';
      ctx.fillText(`${ask.toFixed(2)}`, width - 55, askY - 4);
    }
  }

  renderChronosFan() {
    const canvas = this.chronosCanvas;
    if (!canvas || !this.state || !this.state.forecast) return;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    ctx.clearRect(0, 0, width, height);

    const f = this.state.forecast;
    const curr = f.current_price || 4150.0;
    const p10 = curr * (1 + (f.chronos_p10_return || -0.001));
    const p50 = curr * (1 + (f.chronos_return || 0.0003));
    const p90 = curr * (1 + (f.chronos_p90_return || 0.0015));

    const minVal = Math.min(curr, p10) * 0.9995;
    const maxVal = Math.max(curr, p90) * 1.0005;
    const toY = (val) => height - 30 - ((val - minVal) / (maxVal - minVal)) * (height - 60);

    const startX = 40;
    const endX = width - 40;

    // Fan Fill
    const gradient = ctx.createLinearGradient(0, toY(p90), 0, toY(p10));
    gradient.addColorStop(0, 'rgba(139, 92, 246, 0.35)');
    gradient.addColorStop(1, 'rgba(6, 182, 212, 0.05)');

    ctx.fillStyle = gradient;
    ctx.beginPath();
    ctx.moveTo(startX, toY(curr));
    ctx.lineTo(endX, toY(p90));
    ctx.lineTo(endX, toY(p10));
    ctx.closePath();
    ctx.fill();

    // p90 Line
    ctx.strokeStyle = '#8b5cf6';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(startX, toY(curr));
    ctx.lineTo(endX, toY(p90));
    ctx.stroke();

    // p50 Median Line
    ctx.strokeStyle = '#06b6d4';
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.moveTo(startX, toY(curr));
    ctx.lineTo(endX, toY(p50));
    ctx.stroke();

    // p10 Line
    ctx.strokeStyle = '#ef4444';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(startX, toY(curr));
    ctx.lineTo(endX, toY(p10));
    ctx.stroke();

    // Labels
    ctx.font = '11px JetBrains Mono';
    ctx.fillStyle = '#8b5cf6';
    ctx.fillText(`p90: ${p90.toFixed(2)}`, endX - 60, toY(p90) - 6);
    ctx.fillStyle = '#06b6d4';
    ctx.fillText(`p50: ${p50.toFixed(2)}`, endX - 60, toY(p50) - 6);
    ctx.fillStyle = '#ef4444';
    ctx.fillText(`p10: ${p10.toFixed(2)}`, endX - 60, toY(p10) + 14);
  }

  renderEquityChart() {
    const canvas = this.equityCanvas;
    if (!canvas || this.signals.length === 0) return;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    ctx.clearRect(0, 0, width, height);

    // Calculate cumulative pips curve
    const evaluated = this.signals.filter(s => s.status !== 'PENDING').reverse();
    if (evaluated.length === 0) {
      ctx.fillStyle = '#64748b';
      ctx.font = '13px Inter';
      ctx.fillText('Awaiting evaluated trade outcomes to build growth curve...', width / 2 - 150, height / 2);
      return;
    }

    let runningPips = 0;
    const points = [0];
    evaluated.forEach(s => {
      runningPips += (s.realized_pips || 0);
      points.push(runningPips);
    });

    const minP = Math.min(0, ...points);
    const maxP = Math.max(20, ...points);
    const toY = (val) => height - 25 - ((val - minP) / (maxP - minP || 1)) * (height - 50);

    const stepX = (width - 60) / (points.length - 1 || 1);

    // Gradient fill under curve
    const grad = ctx.createLinearGradient(0, 0, 0, height);
    grad.addColorStop(0, 'rgba(16, 185, 129, 0.3)');
    grad.addColorStop(1, 'rgba(16, 185, 129, 0.0)');

    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.moveTo(30, toY(points[0]));
    points.forEach((p, idx) => {
      ctx.lineTo(30 + idx * stepX, toY(p));
    });
    ctx.lineTo(30 + (points.length - 1) * stepX, height);
    ctx.lineTo(30, height);
    ctx.closePath();
    ctx.fill();

    // Line
    ctx.strokeStyle = '#10b981';
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    points.forEach((p, idx) => {
      const x = 30 + idx * stepX;
      const y = toY(p);
      if (idx === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Zero baseline
    const zeroY = toY(0);
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(30, zeroY);
    ctx.lineTo(width - 30, zeroY);
    ctx.stroke();
    ctx.setLineDash([]);
  }
}

// Instantiate on DOM load
window.addEventListener('DOMContentLoaded', () => {
  window.app = new TradingCockpitApp();
});
