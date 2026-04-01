![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python: 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)
![Framework: Streamlit](https://img.shields.io/badge/Framework-Streamlit-FF4B4B.svg)
![Data: yfinance](https://img.shields.io/badge/Data-yfinance-green.svg)

# Multi-Asset Trading Algorithm & Dashboard
A comprehensive system for automated backtesting, real-time analysis, and market trend forecasting (Monte Carlo & Meta Prophet) for Commodities, Stocks, and Forex.

## Quick Start
1. **Install Dependencies:** `pip install -r requirements.txt`
2. **Launch Dashboard:** `streamlit run dashboard.py`
3. **Terminal Analysis:** `python trading_backtest.py --analyze Gold --interval 1h`

## Table of Contents
1. [System Overview](#1-system-overview)
2. [Configuration & Asset Profiles](#2-configuration--asset-profiles)
3. [Technical Indicators](#3-technical-indicators)
4. [Combined Trading Logic](#4-combined-trading-logic)
5. [Backtesting & Performance Metrics](#5-backtesting--performance-metrics)
6. [Speed & Volume Analysis](#6-speed--volume-analysis)
7. [Forecasting Models](#7-forecasting-models)
8. [Usage & Modes](#8-usage--modes)
9. [Streamlit Dashboard](#9-streamlit-dashboard)
10. [Disclaimer](#10-disclaimer)

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

## 6. Speed & Volume Analysis
This section supplements core signals with market conviction data:
* **ROC (Rate of Change):** Price velocity.
* **OBV Divergence:** Key institutional accumulation/distribution indicator.
* **Candle Body Ratio:** Strength of current price action.

## 7. Forecasting Models
Two advanced forecasting methods are integrated:
1.  **Monte Carlo (1000+ Simulations):** Probability fans (Random Walk, GARCH for Crypto, Mean Reversion for Commodities).
2.  **Meta Prophet:** A robust statistical model capturing seasonality and trend shifts.

## 8. Usage & Modes
* `--analyze [Asset]`: In-depth technical analysis for a single ticker.
* `--signals-hourly`: Market-wide signal overview (1h/4h intervals).
* **Default Mode:** Full historical portfolio backtest.

## 9. Streamlit Dashboard
The interactive web UI features:
* **Signal Overview:** Real-time buy zones and target levels.
* **Asset Detail:** Interactive Plotly charts & Volume profiles.
* **Backtest Summary:** Equity curve visualizations and multi-asset comparisons.

## 10. Disclaimer
This software is for educational purposes only. Trading financial markets involves significant risk. Past performance does not guarantee future results.