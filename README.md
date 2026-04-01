![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python: 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)
![Framework: Streamlit](https://img.shields.io/badge/Framework-Streamlit-FF4B4B.svg)
![Data: yfinance](https://img.shields.io/badge/Data-yfinance-green.svg)

# Multi-Asset Trading Algorithm & Dashboard
A comprehensive system for automated backtesting, real-time analysis, and market trend forecasting (Monte Carlo & Meta Prophet) for Commodities, Stocks, and Forex.

<img width="1850" height="957" alt="obrazek" src="https://github.com/user-attachments/assets/80b738e4-fbb8-45c4-b7b5-92a173dbc420" />

## Quick Start
1. **Install Dependencies:** `pip install -r requirements.txt`
2. **Launch Dashboard:** `streamlit run dashboard.py`
3. **Terminal Analysis:** `python trading_backtest.py --analyze Gold --interval 1h`

## 1. System Overview
The script performs a historical **backtest of a combined technical strategy** on 15+ assets. The system automatically fetches OHLCV data, calculates indicators, and generates signals based on a voting consensus (at least 3 out of 5 indicators must agree).

## 2. Configuration & Asset Profiles
Different markets require different approaches. The algorithm utilizes **Dynamic Profiles**:
* **COMMODITY:** Wider bands, slower averages (Gold, Oil).
* **TECH:** Aggressive settings for high-beta stocks (NVDA, AMD).
* **DEFENSIVE:** Conservative stop-losses for dividend stocks (KO, Moneta).
* **FOREX_IDX:** Tight bands for low-volatility currency moves (USD Index).

## 3. Technical Indicators
The strategy is built on 5 pillars:
* **EMA Crossover (20/50):** Trend detection.
* **RSI (14):** Momentum strength.
* **Bollinger Bands:** Statistical price volatility.
* **MACD:** Trend convergence/divergence.
* **ATR:** Dynamic risk management (Stop-Loss).

## 4. Combined Trading Logic
**No indicator acts alone.**
* **BUY Signal:** Requires a score of ≥ 3/5 (e.g., Rising EMA + RSI < 50 + Bullish MACD).
* **Risk Management:** Automatic Stop-Loss calculated as `2.0 * ATR`.

## 5. Backtesting & Performance Metrics
The system provides professional metrics for objective evaluation:
* **Alpha:** Excess return over the "Buy & Hold" benchmark.
* **Sharpe Ratio:** Risk-adjusted return.
* **Max Drawdown:** Peak-to-trough capital decline.
* **Profit Factor:** Ratio of gross profits to gross losses.

<img width="2384" height="1477" alt="summary_comparison" src="https://github.com/user-attachments/assets/2b867069-9616-4cdb-a66e-6f68d010fa3e" />

## 6. Speed & Volume Analysis
This section supplements core signals with market conviction data:
* **ROC (Rate of Change):** Price velocity.
* **OBV Divergence:** Key institutional accumulation/distribution indicator.
* **Candle Body Ratio:** Strength of current price action.

<img width="548" height="406" alt="Snímek obrazovky z 2026-04-01 19-24-11" src="https://github.com/user-attachments/assets/ca8174c4-fbfc-4610-8c42-c67b9ea89148" />

## 7. Forecasting Models
Two advanced forecasting methods are integrated:
1.  **Monte Carlo (1000+ Simulations):** Probability fans (Random Walk, GARCH for Crypto, Mean Reversion for Commodities).
2.  **Meta Prophet:** A robust statistical model capturing seasonality and trend shifts.

<img width="642" height="468" alt="Snímek obrazovky z 2026-04-01 19-02-22" src="https://github.com/user-attachments/assets/209e1727-6ac8-4246-ae97-75f643eee5cb" />

## 8. Usage & Modes
* `--analyze [Asset]`: In-depth technical analysis for a single ticker.
* `--signals-hourly`: Market-wide signal overview (1h/4h intervals).
* **Default Mode:** Full historical portfolio backtest.

<img width="969" height="802" alt="Snímek obrazovky z 2026-04-01 19-01-01" src="https://github.com/user-attachments/assets/4b72f51e-2a77-4d20-b01b-465c4082e275" />

## 9. Streamlit Dashboard
The interactive web UI features:
* **Signal Overview:** Real-time buy zones and target levels.
<img width="1850" height="957" alt="Snímek obrazovky z 2026-04-01 19-08-54" src="https://github.com/user-attachments/assets/3ef8964f-a8b4-454e-93c0-985a54c27557" />
<p></p>

* **Asset Detail:** Interactive Plotly charts & Volume profiles.
<img width="1850" height="957" alt="Snímek obrazovky z 2026-04-01 19-06-14" src="https://github.com/user-attachments/assets/50ac244a-0793-4e81-b0c1-9d849bacfeb9" />
<p></p>

* **Backtest Summary:** Equity curve visualizations and multi-asset comparisons.


## 10. Disclaimer
This software is for educational purposes only. Trading financial markets involves significant risk. Past performance does not guarantee future results.
