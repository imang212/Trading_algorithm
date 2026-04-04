"""
MULTI-ASSET TRADING ALGORITHM BACKTEST
Indicators: MA Crossover, RSI, Bollinger Bands, MACD, ATR
Assets: Gold, Silver, MSFT, GOOGL, MONET.PR, ORCL, NVDA, AMD, SPOT, ...
Installation of dependencies:
    pip install yfinance pandas numpy matplotlib seaborn tabulate prophet
Usage:
    python trading_backtest_oop.py                              # full backtest
    python trading_backtest_oop.py --analyze Gold               # quick analysis
    python trading_backtest_oop.py --analyze NVDA --interval 1h # hourly analysis
    python trading_backtest_oop.py --signals-hourly             # all assets, hourly
    python trading_backtest_oop.py --signals-hourly --interval 4h
"""
import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use('Agg') 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from tabulate import tabulate
import yfinance as yf
from datetime import datetime
import os

# Configuration
ASSETS = {
    # Commodities & futures
    "Gold": "GC=F", "Silver": "SI=F", "Oil": "CL=F", "Brent_Oil": "BZ=F", "USD": "DX-Y.NYB", 
    # Crypto
    "Bitcoin": "BTC-USD",
    # ETF
    "SP500": "SXR8.DE", "MSCIWorld": "EUNL.DE", "Nasdaq100": "CNDX.L",
    # Tech stocks
    "MSFT": "MSFT", "Nokia": "NOKIA.HE", "Ericsson": "ERIC", "GOOGL": "GOOGL", "Apple": "AAPL", "Tesla": "TSLA", "Netflix": "NFLX", "Netflix_DE": "NFC.DE", "Colt": "CZG.PR", "CEZ": "CEZ.PR", "ORCL": "ORCL", "NVDA": "NVDA", "AMD": "AMD", "Adobe": "ADBE", "Intel": "INTC", "Spotify": "SPOT", "Coinbase": "COIN",
    # Defensive stocks
    "Coca-Cola": "KO", "CocaColaCCH": "CCH.L", "Altria": "MO", "Nestle": "NESN.SW", "AgnicoEagle": "AEM", "NewmontMining": "NEM", "NovoNordisk": "NOVO-B.CO", "Moneta": "MONET.PR", "KomBanka": "KOMB.PR", "UBS": "UBSG.SW", "Zurrich_Insurance": "ZURN.SW", "Nordea_Bank": "NDA-FI.HE", 
    "British_American_Tobacco": "BTI", "Equinor": "EQNR", "Equinor_NO": "EQNR.NO", "Allianz": "ALV.DE", "Procter&Gamble": "PG",
}
# Yearly data range for backtest
START_DATE = "2018-01-01"; END_DATE = datetime.today().strftime("%Y-%m-%d")
OUTPUT_DIR = "backtest_results"
INITIAL_CAP = 10_000  # USD per asset
COMMISSION = 0.001 # 0.1% per trade
SLIPPAGE = 0.0005 # 0.05%

_FX_PAIRS = {
    "EUR": "EURUSD=X", "GBP": "GBPUSD=X", "CZK": "CZKUSD=X",
    "DKK": "DKKUSD=X", "SEK": "SEKUSD=X", "NOK": "NOKUSD=X",
    "CHF": "CHFUSD=X", "JPY": "JPYUSD=X", "CAD": "CADUSD=X",
    "AUD": "AUDUSD=X", "HKD": "HKDUSD=X", "SGD": "SGDUSD=X",
    "KRW": "KRWUSD=X", "CNY": "CNYUSD=X", "INR": "INRUSD=X",
    "BRL": "BRLUSD=X", "MXN": "MXNUSD=X",
}

# Runtime cache so we don't re-fetch the same currency twice per session
_currency_cache: dict = {}
 
def detect_currency(ticker: str) -> str:
    """
    Auto-detect the trading currency of *ticker* using yf.Ticker.fast_info.
    Returns the ISO 4217 currency code (e.g. "USD", "EUR", "CZK").
    Falls back to "USD" if detection fails.
    """
    if ticker in _currency_cache:
        return _currency_cache[ticker]
    try:
        info = yf.Ticker(ticker).fast_info
        currency = str(info.get("currency", "USD") or "USD").upper()
        # Yahoo Finance sometimes returns "GBp" (pence) for London stocks
        if currency == "GBP" or currency == "GBP":
            currency = "GBP"
        if currency == "GBP" and ticker.endswith(".L"):
            # LSE prices are in pence (GBp), convert to pounds first
            currency = "GBp"   # special marker handled in convert_to_usd
    except Exception:
        currency = "USD"
    _currency_cache[ticker] = currency
    return currency
 
def get_fx_rate(currency: str, start: str = None, end: str = None):
    """
    Return a USD conversion rate for *currency*.
    Parameters
    currency : ISO 4217 code detected by detect_currency()
    start, end : optional date range for historical series
    Returns
    pd.Series (daily rates) if start/end given, else float scalar.
    1.0 for USD (no conversion needed).
    """
    # GBp (pence) = GBP / 100
    gbp_pence = currency == "GBp"
    base_currency = "GBP" if gbp_pence else currency
    if base_currency == "USD":
        if start and end: return pd.Series(dtype=float)
        return 1.0
    pair = _FX_PAIRS.get(base_currency)
    if pair is None:
        return 1.0
    try:
        if start and end:
            raw = yf.download(pair, start=start, end=end, progress=False, auto_adjust=True)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            rate = raw["Close"].astype(float) if not raw.empty else pd.Series(dtype=float)
        else:
            raw = yf.download(pair, period="5d", progress=False, auto_adjust=True)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            rate = float(raw["Close"].dropna().iloc[-1]) if not raw.empty else 1.0
        # Apply pence → pounds → USD (divide by 100)
        if gbp_pence:
            if isinstance(rate, pd.Series):
                rate = rate / 100
            else:
                rate = rate / 100
        return rate
    except Exception:
        return 1.0
 
def convert_to_usd(df: pd.DataFrame, currency: str, start: str = None, end: str = None) -> pd.DataFrame:
    """
    Convert OHLCV DataFrame from *currency* to USD.
    Fetches daily FX rates and multiplies Open/High/Low/Close columns.
    Volume is unchanged (contracts/shares keep their count).
    Handles GBp (pence) automatically by dividing by 100 before conversion.
    """
    if currency in ("USD",):
        return df
    fx = get_fx_rate(currency, start=start, end=end)
    df = df.copy()
    price_cols = [c for c in ["Open", "High", "Low", "Close"] if c in df.columns]
    if isinstance(fx, float):
        for col in price_cols:
            df[col] = df[col].astype(float) * fx
    elif isinstance(fx, pd.Series) and not fx.empty:
        fx_aligned = fx.reindex(df.index, method="ffill").bfill()
        for col in price_cols:
            df[col] = df[col].astype(float) * fx_aligned.values
    return df

# PARAMETER PROFILES BY ASSET TYPE
# COMMODITY – Gold, Silver, Oil
#   Strong trends, high volatility → slower MA,
#   wider BB bands, larger ATR stop-loss multiplier.
# FOREX_IDX – USD Index
#   Very low volatility, slow moves → very
#   slow MA, narrow BB bands, small ATR multiplier.
# TECH – NVDA, AMD, MSFT, GOOGL, Netflix, Spotify, ORCL
#   High beta, quick trends → standard/aggressive
#   settings, medium ATR multiplier.
# DEFENSIVE – Coca-Cola, Novo Nordisk, Moneta, Agnico Eagle
#   Low beta, slow moves, dividend stocks →
#   slower indicators, conservative stop-loss.

PROFILES = {
    "COMMODITY": dict(MA_SHORT=30, MA_LONG=75,  RSI_PERIOD=14, RSI_OB=70, RSI_OS=30, BB_PERIOD=25, BB_STD=2.5, MACD_FAST=12, MACD_SLOW=30, MACD_SIGNAL=9, ATR_PERIOD=14, ATR_SL_MULT=2.5),
    "CRYPTO":    dict(MA_SHORT=30, MA_LONG=75,  RSI_PERIOD=14, RSI_OB=70, RSI_OS=30, BB_PERIOD=25, BB_STD=3.5, MACD_FAST=12, MACD_SLOW=30, MACD_SIGNAL=9, ATR_PERIOD=14, ATR_SL_MULT=2.5),
    "FOREX_IDX": dict(MA_SHORT=40, MA_LONG=100, RSI_PERIOD=21, RSI_OB=65, RSI_OS=35, BB_PERIOD=30, BB_STD=1.8, MACD_FAST=14, MACD_SLOW=35, MACD_SIGNAL=9, ATR_PERIOD=21, ATR_SL_MULT=1.5),
    "TECH":      dict(MA_SHORT=20, MA_LONG=50,  RSI_PERIOD=14, RSI_OB=70, RSI_OS=30, BB_PERIOD=20, BB_STD=2.0, MACD_FAST=12, MACD_SLOW=26, MACD_SIGNAL=9, ATR_PERIOD=14, ATR_SL_MULT=2.0),
    "DEFENSIVE": dict(MA_SHORT=25, MA_LONG=60,  RSI_PERIOD=14, RSI_OB=65, RSI_OS=35, BB_PERIOD=20, BB_STD=1.8, MACD_FAST=12, MACD_SLOW=26, MACD_SIGNAL=9, ATR_PERIOD=14, ATR_SL_MULT=1.8),
}
# Profile assignment for each asset
ASSET_PROFILES = {
    "Gold": "COMMODITY", "Silver": "COMMODITY", "Oil": "COMMODITY", "Brent_Oil": "COMMODITY", "USD": "FOREX_IDX", "Bitcoin": "CRYPTO", "SP500": "DEFENSIVE", "MSCIWorld": "DEFENSIVE", "Nasdaq100": "DEFENSIVE", 
    "MSFT": "TECH", "Nokia": "TECH", "Ericsson": "TECH", "GOOGL": "TECH", "Apple": "TECH", "Tesla": "TECH", "Netflix": "TECH", "Netflix_DE": "TECH", "Colt": "TECH", "Spotify": "TECH", "ORCL": "TECH", "NVDA": "TECH", "AMD": "TECH", "Adobe": "TECH", "Intel": "TECH", "Coinbase": "TECH",
    "Coca-Cola": "DEFENSIVE", "CocaColaCCH": "DEFENSIVE", "Altria": "DEFENSIVE", "Nestle": "DEFENSIVE", "AgnicoEagle": "DEFENSIVE", "NewmontMining": "DEFENSIVE", "NovoNordisk": "DEFENSIVE", "Moneta": "DEFENSIVE", "KomBanka": "DEFENSIVE", "UBS":"DEFENSIVE", "Zurrich_Insurrance": "DEFENSIVE", "Nordea_Bank": "DEFENSIVE",
    "British_American_Tobacco": "DEFENSIVE", "Equinor": "DEFENSIVE", "Equinor_NO": "DEFENSIVE", "Allianz": "DEFENSIVE", "Procter&Gamble": "DEFENSIVE",
}

INTERVAL_SETTINGS = {
    "1m":  {"period": "5d",   "lookback": 200,  "label": "1 minute",    "mc_days": 60},
    "5m":  {"period": "30d",  "lookback": 200,  "label": "5 minutes",   "mc_days": 120},
    "15m": {"period": "30d",  "lookback": 200,  "label": "15 minutes",  "mc_days": 96},
    "30m": {"period": "30d",  "lookback": 200,  "label": "30 minutes",  "mc_days": 48},
    "1h":  {"period": "180d", "lookback": 200,  "label": "1 hour",      "mc_days": 48},
    "4h":  {"period": "180d", "lookback": 200,  "label": "4 hours",     "mc_days": 30},
    "1d":  {"period": "6mo",  "lookback": 90,   "label": "1 day",       "mc_days": 30},
}

# INDICATOR CALCULATION - MA Crossover, RSI, Bollinger Bands, MACD, ATR
def compute_indicators(df: pd.DataFrame, p: dict) -> pd.DataFrame:
    """
    Add all five indicator column groups to *df* using profile params *p*.
    Columns added:
        SMA_short, SMA_long, EMA_short, EMA_long, RSI, BB_mid, BB_upper, BB_lower, BB_pct, MACD, MACD_sig, MACD_hist, ATR
    """
    c = df["Close"].astype(float)
    # Moving Averages
    df["SMA_short"] = c.rolling(p["MA_SHORT"]).mean(); df["SMA_long"] = c.rolling(p["MA_LONG"]).mean()
    df["EMA_short"] = c.ewm(span=p["MA_SHORT"], adjust=False).mean(); df["EMA_long"]  = c.ewm(span=p["MA_LONG"], adjust=False).mean()
    # RSI
    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(p["RSI_PERIOD"]).mean()
    loss  = (-delta.clip(upper=0)).rolling(p["RSI_PERIOD"]).mean()
    rs    = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))
    # Bollinger Bands
    df["BB_mid"] = c.rolling(p["BB_PERIOD"]).mean()
    bb_std = c.rolling(p["BB_PERIOD"]).std()
    df["BB_upper"] = df["BB_mid"] + p["BB_STD"] * bb_std
    df["BB_lower"] = df["BB_mid"] - p["BB_STD"] * bb_std
    df["BB_pct"] = (c - df["BB_lower"]) / (df["BB_upper"] - df["BB_lower"])
    # MACD
    ema_fast = c.ewm(span=p["MACD_FAST"], adjust=False).mean(); ema_slow = c.ewm(span=p["MACD_SLOW"], adjust=False).mean()
    df["MACD"] = ema_fast - ema_slow
    df["MACD_sig"] = df["MACD"].ewm(span=p["MACD_SIGNAL"], adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_sig"]
    # ATR
    high = df["High"].squeeze(); low  = df["Low"].squeeze()
    tr = pd.concat([high - low, (high - c.shift()).abs(), (low - c.shift()).abs()], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(p["ATR_PERIOD"]).mean()
    return df

# COMBINED STRATEGY (signals, BUY, SELL score)
def generate_signals(df: pd.DataFrame, p: dict) -> pd.DataFrame:
    """BUY if at least 3 out of 5 conditions are met:
        1) MA Crossover: EMA_short > EMA_long
        2) RSI:          RSI < 50 (room to grow, but not oversold)
        3) Bollinger:    price near lower band (BB_pct < 0.4)
        4) MACD:         MACD > Signal line
        5) ATR trend:    price > SMA_short (momentum move)
       SELL if at least 3 out of 5 conditions are met (opposite of above):
       RSI_mid = mid between RSI_OS and RSI_OB (typically 50).
    """
    c = df["Close"].astype(float)
    rsi_mid = (p["RSI_OB"] + p["RSI_OS"]) / 2 # middle of band (50 for 70/30, 50 for 65/35)
    cond_buy = pd.DataFrame({
        "ma":   (df["EMA_short"] > df["EMA_long"]).astype(int),
        "rsi":  (df["RSI"] < rsi_mid).astype(int),
        "bb":   (df["BB_pct"] < 0.4).astype(int),
        "macd": (df["MACD"] > df["MACD_sig"]).astype(int),
        "atr":  (c > df["SMA_short"]).astype(int),
    })
    cond_sell = pd.DataFrame({
        "ma":   (df["EMA_short"] < df["EMA_long"]).astype(int),
        "rsi":  (df["RSI"] > rsi_mid).astype(int),
        "bb":   (df["BB_pct"] > 0.6).astype(int),
        "macd": (df["MACD"] < df["MACD_sig"]).astype(int),
        "atr":  (c < df["SMA_short"]).astype(int),
    })
    df["buy_score"] = cond_buy.sum(axis=1); df["sell_score"] = cond_sell.sum(axis=1)
    df["signal"] = 0
    df.loc[df["buy_score"] >= 3, "signal"] = 1
    df.loc[df["sell_score"] >= 3, "signal"] = -1
    return df

# BACKTEST ENGINE (final value, profit, Buy&Hold, Alpha, Win rate, Sharpe, Max Drawdown, Profit Factor, etc.)
def run_backtest(df: pd.DataFrame, asset_name: str, p: dict) -> dict:
    """
        Simulate bar-by-bar trade execution and compute performance statistics.
        Entry logic:    BUY signal, no open position → buy 95 % of capital.
        Exit logic:     SELL signal or price < dynamic ATR stop-loss.
        Costs:          commission + slippage applied on every fill.
        Returns a result dict with keys:
            asset, p, profile, final_value, total_return, bh_return, num_trades, win_rate, sharpe, max_drawdown, profit_factor, avg_buy_score, avg_sell_score, equity_df, trades_df, price_df.
    """
    df = df.copy()
    close = df["Close"].astype(float)
    capital = INITIAL_CAP
    position = 0.0  # number of contracts/units
    entry_px = 0.0; stop_loss = 0.0
    trades = []; equity = []; prev_sig = 0
    _last_valid_price = float(close.dropna().iloc[0]) if len(close.dropna()) > 0 else 0.0
    for i in range(len(df)):
        row = df.iloc[i]
        price = float(close.iloc[i])
        # skip NaN rollover gaps in futures or missing data – keep equity flat, no trades
        if pd.isna(price) or price <= 0:
            equity.append({"date": df.index[i], "equity": capital + position * _last_valid_price})
            continue
        _last_valid_price = price
        sig = int(row["signal"]) if not pd.isna(row["signal"]) else 0
        atr = float(row["ATR"]) if not np.isnan(row["ATR"]) else 0
         # Stop-loss check
        if position > 0 and price < stop_loss:
            proceeds = position * price * (1 - COMMISSION - SLIPPAGE)
            pnl = proceeds - position * entry_px
            capital += proceeds
            trades.append({"date": df.index[i], "type": "STOP-LOSS", "price": price, "pnl": pnl})
            position = 0
        # BUY
        if sig == 1 and prev_sig != 1 and position == 0 and capital > 0:
            buy_px = price * (1 + SLIPPAGE)
            qty = (capital * 0.95) / buy_px   # invest 95% of capital
            cost = qty * buy_px * (1 + COMMISSION)
            if cost <= capital:
                capital -= cost
                position = qty
                entry_px = buy_px
                stop_loss = buy_px - p["ATR_SL_MULT"] * atr
                trades.append({"date": df.index[i], "type": "BUY", "price": buy_px, "pnl": 0})
        # SELL
        elif sig == -1 and prev_sig != -1 and position > 0:
            sell_px = price * (1 - SLIPPAGE)
            proceeds = position * sell_px * (1 - COMMISSION)
            pnl = proceeds - position * entry_px
            capital += proceeds
            trades.append({"date": df.index[i], "type": "SELL", "price": sell_px, "pnl": pnl})
            position = 0
        total_val = capital + position * price
        equity.append({"date": df.index[i], "equity": total_val})
        prev_sig = sig
    # Close open position at the end 
    if position > 0:
        last_px = float(close.dropna().iloc[-1]) if len(close.dropna()) > 0 else _last_valid_price
        proceeds = position * last_px * (1 - COMMISSION)
        pnl = proceeds - position * entry_px
        capital += proceeds
        trades.append({"date": df.index[-1], "type": "CLOSE", "price": last_px, "pnl": pnl})
        position = 0
    result_p = p
    equity_df = pd.DataFrame(equity).set_index("date")
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame(columns=["date", "type", "price", "pnl"])
    # Statistics
    final_val = capital
    total_return = (final_val - INITIAL_CAP) / INITIAL_CAP * 100
    # Buy & Hold benchmark
    close_valid = close.dropna()
    first_close = float(close_valid.iloc[0])  if len(close_valid) > 0 else float("nan")
    last_close  = float(close_valid.iloc[-1]) if len(close_valid) > 0 else float("nan")
    bh_return   = ((last_close - first_close) / first_close * 100 if first_close and first_close > 0 else float("nan"))
    sell_trades = trades_df[trades_df["type"].isin(["SELL", "STOP-LOSS", "CLOSE"])]
    wins = sell_trades[sell_trades["pnl"] > 0]
    losses = sell_trades[sell_trades["pnl"] <= 0]
    win_rate = len(wins) / len(sell_trades) * 100 if len(sell_trades) > 0 else 0
    avg_win = wins["pnl"].mean() if len(wins) > 0 else 0
    avg_loss = losses["pnl"].mean() if len(losses) > 0 else 0
    profit_factor= abs(wins["pnl"].sum() / losses["pnl"].sum()) if losses["pnl"].sum() != 0 else np.inf
    # Sharpe Ratio (daily returns)
    eq_series = equity_df["equity"]
    daily_ret = eq_series.pct_change().dropna()
    sharpe = (daily_ret.mean() / daily_ret.std() * np.sqrt(252) if daily_ret.std() > 0 else 0)
    # Max Drawdown
    roll_max = eq_series.cummax()
    drawdown = (eq_series - roll_max) / roll_max
    max_dd = drawdown.min() * 100
    return {"asset": asset_name, "p": result_p, "final_value": final_val,  "total_return": total_return,  "bh_return": bh_return,  "num_trades": len(sell_trades), "win_rate": win_rate,  "avg_win": avg_win,  "avg_loss": avg_loss,  "profit_factor": profit_factor,  "sharpe": sharpe, "max_drawdown": max_dd,  "equity_df": equity_df,  "trades_df": trades_df,  "price_df": df, "avg_buy_score":  round(df["buy_score"].mean(), 2), "avg_sell_score": round(df["sell_score"].mean(), 2),}

# VISUALIZATION AND PREDICTION
MC_DAYS        = 30     # forecast horizon (trading days)
MC_SIMULATIONS = 1000   # number of simulations
MC_PROFILE_META = {
    "DEFENSIVE": {"color": "#1565c0", "label": "Random Walk", "short": "RW"},
    "TECH":      {"color": "#6a1b9a", "label": "RW + Earnings jumps", "short": "RW+E"},
    "COMMODITY": {"color": "#1b00e6", "label": "GBM + Mean Reversion", "short": "GBM-MR"},
    "CRYPTO":    {"color": "#b71c1c", "label": "GARCH volatility", "short": "GARCH"},
    "FOREX_IDX": {"color": "#1b5e20", "label": "Ornstein-Uhlenbeck", "short": "O-U"},
}

def _mc_random_walk(returns, last_price, n_sim, n_days, rng):
    """DEFENSIVE, base – simple random walk."""
    mu, sigma = returns.mean(), returns.std()
    shocks = rng.normal(mu, sigma, size=(n_sim, n_days))
    return last_price * np.cumprod(1 + shocks, axis=1)

def _mc_random_walk_earnings(returns, last_price, n_sim, n_days, rng):
    """TECH – random walk with random earnings jumps (~every 63 days, ±5-15%)."""
    mu, sigma = returns.mean(), returns.std()
    shocks = rng.normal(mu, sigma, size=(n_sim, n_days))
    # Earnings arrive roughly every quarterly cycle
    # In a 30-day window there's ~48% chance of earnings – add random jump
    for sim in range(n_sim):
        if rng.random() < 0.48:
            day = rng.integers(0, n_days)
            jump = rng.normal(0, 0.08)   # average earnings jump ±8%
            shocks[sim, day] += jump
    return last_price * np.cumprod(1 + shocks, axis=1)

def _mc_gbm_mean_reversion(close, last_price, n_sim, n_days, rng, lookback=252):
    """COMMODITY – Geometric Brownian Motion with mean reversion (attraction to long-term average)."""
    returns = close.pct_change().dropna()
    sigma = returns.std()
    # Long-term average = SMA of last year
    long_mean = float(close.iloc[-min(lookback, len(close)):].mean())
    theta = 0.05 # speed of return to mean (higher = faster)
    paths = np.zeros((n_sim, n_days))
    for t in range(n_days):
        prev = last_price if t == 0 else paths[:, t-1]
        # Drift: theta × (long_mean - current price) + random shock
        drift  = theta * (long_mean - prev) / last_price
        shocks = rng.normal(drift, sigma, size=n_sim)
        paths[:, t] = prev * (1 + shocks)
    return paths

def _mc_garch(returns, last_price, n_sim, n_days, rng):
    """
        CRYPTO – GARCH(1,1): volatility depends on prior volatility and shocks.
        Captures volatility clustering typical for crypto.
    """
    mu = returns.mean()
    # GARCH(1,1) parameters estimated from data (method of moments)
    var_long = returns.var()
    omega = var_long * 0.05    # long-term weight
    alpha = 0.15               # weight of last shock (shock reaction)
    beta = 0.80               # volatility persistence
    paths = np.zeros((n_sim, n_days))
    for sim in range(n_sim):
        h  = var_long   # initial variance
        ep = returns.iloc[-1] - mu   # last residual
        price = last_price
        for t in range(n_days):
            h = omega + alpha * ep**2 + beta * h
            h = max(h, 1e-8)
            shock = rng.normal(mu, np.sqrt(h))
            ep    = shock - mu
            price = price * (1 + shock)
            paths[sim, t] = max(price, 1e-3)
    return paths

def _mc_ornstein_uhlenbeck(close, last_price, n_sim, n_days, rng, lookback=252):
    """
        FOREX_IDX – Ornstein-Uhlenbeck (mean reversion).
        Currency pairs gravitate to long-term equilibrium – stronger attraction than GBM.
    """
    returns = close.pct_change().dropna()
    sigma = returns.std()
    mu_price = float(close.iloc[-min(lookback, len(close)):].mean()) # equilibrium price
    theta = 0.12 # speed of return (significantly stronger than commodities)
    dt = 1.0
    paths = np.zeros((n_sim, n_days))
    for sim in range(n_sim):
        price = last_price
        for t in range(n_days):
            drift = theta * (mu_price - price) * dt
            shock = rng.normal(0, sigma * price * np.sqrt(dt))
            price = price + drift + shock
            paths[sim, t] = max(price, 1e-3)
    return paths

def monte_carlo_forecast(close: pd.Series, profile: str = "TECH", n_days: int = MC_DAYS, n_sim: int = MC_SIMULATIONS, lookback: int = 90):
    """
    Per-profile Monte Carlo simulation:
      DEFENSIVE → Random Walk (GBM)
      TECH      → Random Walk + random earnings jumps
      COMMODITY → GBM with mean reversion for long-term average 
      CRYPTO    → GARCH(1,1) with variable volatility
      FOREX_IDX → Ornstein-Uhlenbeck strong mean reversion
    Returns:
        dict with keys: dates, p10, p25, p50, p75, p90, last, profile
    """
    returns = close.iloc[-lookback:].pct_change().dropna(); last_price = float(close.iloc[-1])
    rng = np.random.default_rng(42)
    if profile == "DEFENSIVE":
        paths = _mc_random_walk(returns, last_price, n_sim, n_days, rng)
    elif profile == "TECH":
        paths = _mc_random_walk_earnings(returns, last_price, n_sim, n_days, rng)
    elif profile == "COMMODITY":
        paths = _mc_gbm_mean_reversion(close.iloc[-lookback:], last_price, n_sim, n_days, rng)
    elif profile == "CRYPTO":
        paths = _mc_garch(returns, last_price, n_sim, n_days, rng)
    elif profile == "FOREX_IDX":
        paths = _mc_ornstein_uhlenbeck(close.iloc[-lookback:], last_price, n_sim, n_days, rng)
    else:
        paths = _mc_random_walk(returns, last_price, n_sim, n_days, rng)
    last_date = close.index[-1]
    # Detect bar frequency from index so future dates match the interval
    # (e.g. 1h data → future timestamps are hourly, not daily business days)
    if len(close.index) >= 2: median_delta = pd.Series(close.index).diff().dropna().median()
    else: median_delta = pd.Timedelta(days=1)
    # Round median_delta to the nearest clean interval
    total_seconds = median_delta.total_seconds()
    if total_seconds <= 90: bar_delta = pd.Timedelta(minutes=1)
    elif total_seconds <= 360: bar_delta = pd.Timedelta(minutes=5)
    elif total_seconds <= 1080: bar_delta = pd.Timedelta(minutes=15)
    elif total_seconds <= 2160: bar_delta = pd.Timedelta(minutes=30)
    elif total_seconds <= 7200: bar_delta = pd.Timedelta(hours=1)
    elif total_seconds <= 21600: bar_delta = pd.Timedelta(hours=4)
    else: bar_delta = pd.Timedelta(days=1)
    if bar_delta >= pd.Timedelta(days=1):
        # Daily or longer – use business days (skip weekends)
        future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=n_days)
    else:
        # Intraday – generate at exact bar frequency
        # Skip known market-closed hours (simple heuristic: 00:00–06:00 UTC)
        future_dates = []
        ts = last_date + bar_delta
        while len(future_dates) < n_days:
            # Skip weekends
            if ts.weekday() < 5: future_dates.append(ts)
            ts += bar_delta
        future_dates = pd.DatetimeIndex(future_dates)
    return {"dates": future_dates, "p10": np.percentile(paths, 10, axis=0), "p25": np.percentile(paths, 25, axis=0),
        "p50": np.percentile(paths, 50, axis=0), "p75": np.percentile(paths, 75, axis=0), "p90": np.percentile(paths, 90, axis=0),
        "last": last_price, "profile": profile, "bar_delta": bar_delta,}

def draw_monte_carlo(ax, close: pd.Series, profile: str = "TECH"):
    """
        Draw a Monte Carlo fan chart (percentile bands + median line) onto *ax*.
        Uses PROFILE_META for colour and label. Called by Plotter._draw_price_panel.
    """
    mc = monte_carlo_forecast(close, profile=profile); d = mc["dates"] 
    meta = MC_PROFILE_META.get(profile, MC_PROFILE_META["TECH"]); color = meta["color"]; short = meta["short"]
    ax.fill_between(d, mc["p10"], mc["p90"], alpha=0.10, color=color, label=f"MC 10–90 %")
    ax.fill_between(d, mc["p25"], mc["p75"], alpha=0.22, color=color, label=f"MC 25–75 %")
    ax.plot(d, mc["p50"], color=color, lw=1.8, ls="--", label=f"MC median ({short}, {MC_DAYS}d)", zorder=4)
    ax.axvline(close.index[-1], color=color, lw=1.0, ls=":", alpha=0.7)
    ax.annotate(f"${mc['p50'][-1]:,.0f}", xy=(d[-1], mc["p50"][-1]), fontsize=7.5, color=color, va="center")

def prophet_forecast(close: pd.Series, n_days: int = 30) -> dict | None:
    """
    Fit a Facebook Prophet model on *close* and return a 30-day forecast. 
    Prophet decomposes the series into:
      trend      – detected changepoints in the long-term direction
      seasonality – weekly + yearly patterns fitted by Fourier series
      residual   – unexplained noise
    Returns dict with keys: dates, yhat, yhat_lower, yhat_upper, trend,
    components (weekly/yearly if available), last, model_info.
    Returns None if Prophet is not installed or fitting fails.
    """
    try:
        from prophet import Prophet
        import logging
        logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
    except ImportError:
        return None
    try:
        # Prophet requires a DataFrame with columns ds (date) and y (value)
        df_p = pd.DataFrame({"ds": close.index.tz_localize(None) if close.index.tzinfo else close.index, "y":  close.values.astype(float),}).dropna()
        if len(df_p) < 30: return None 
        # Detect interval – intraday data gets no yearly seasonality
        freq_delta = df_p["ds"].diff().median()
        is_intraday = freq_delta < pd.Timedelta("1D")
        model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=not is_intraday,
            yearly_seasonality=not is_intraday,
            changepoint_prior_scale=0.05,   # flexibility of trend changes
            seasonality_prior_scale=10.0,
            interval_width=0.80,            # 80% confidence interval
            uncertainty_samples=500,
        )
        # Add custom intraday seasonality for hourly data
        if is_intraday and freq_delta <= pd.Timedelta("1H"):
            model.add_seasonality(name="daily", period=1, fourier_order=8)
        model.fit(df_p)
        # Build future dates – business days for daily, freq-based for intraday
        if is_intraday:
            last_ts = df_p["ds"].iloc[-1]
            future_dates = [last_ts + freq_delta * i for i in range(1, n_days + 1)]
            future = pd.DataFrame({"ds": future_dates})
        else:
            future = model.make_future_dataframe(periods=n_days, freq="B")
        forecast = model.predict(future)
        fcast_future = forecast[forecast["ds"] > df_p["ds"].iloc[-1]].copy()
        # last_close, preserving the shape and direction of the curve.
        last_close  = float(close.iloc[-1])
        prophet_t0  = float(fcast_future["yhat"].iloc[0])
        anchor_shift = last_close - prophet_t0   # additive offset
        fcast_future = fcast_future.copy()
        fcast_future["yhat"] += anchor_shift
        fcast_future["yhat_lower"] += anchor_shift
        fcast_future["yhat_upper"] += anchor_shift
        fcast_future["trend"] += anchor_shift
        # Extract trend direction from anchored values
        trend_start = float(fcast_future["yhat"].iloc[0])
        trend_end   = float(fcast_future["yhat"].iloc[-1])
        trend_pct   = (trend_end - trend_start) / trend_start * 100 if trend_start > 0 else 0
        if trend_pct >  3: trend_label = f"▲ UPTREND  +{trend_pct:.1f}%"
        elif trend_pct < -3: trend_label = f"▼ DOWNTREND  {trend_pct:.1f}%"
        else: trend_label = f"→ SIDEWAYS  {trend_pct:+.1f}%"
        return {
            "dates": fcast_future["ds"].values,
            "yhat": fcast_future["yhat"].values,
            "yhat_lower": fcast_future["yhat_lower"].values,
            "yhat_upper": fcast_future["yhat_upper"].values,
            "trend": fcast_future["trend"].values,
            "trend_label": trend_label,
            "trend_pct": trend_pct,
            "last": last_close,
            "n_days": n_days,
            "model_info": f"Prophet  |  changepoints={model.n_changepoints}  |  {'intraday' if is_intraday else 'daily'}",
        }
    except Exception as e:
        return None

def draw_prophet(ax, close: pd.Series, n_days: int = 30, color: str = "#e67e22"):
    """
    Fit Prophet and draw its forecast onto *ax* alongside Monte Carlo.
    Draws:
      - Orange shaded 80% confidence interval
      - Solid median (yhat) line
      - Dashed trend component line
      - Annotation with trend label and final price
    If Prophet is not installed or fitting fails, draws nothing silently.
    """
    pf = prophet_forecast(close, n_days=n_days)
    if pf is None: return
    d = pf["dates"]
    ax.fill_between(d, pf["yhat_lower"], pf["yhat_upper"], alpha=0.15, color=color, label="Prophet 80% CI")
    ax.plot(d, pf["yhat"], color=color, lw=2.0, ls="-", label=f"Prophet forecast ({pf['trend_label']})", zorder=5)
    ax.plot(d, pf["trend"], color=color, lw=1.2, ls="--", alpha=0.6, label="Prophet trend", zorder=4)
    ax.axvline(close.index[-1], color=color, lw=1.0, ls=":", alpha=0.5)
    ax.annotate(f"${pf['yhat'][-1]:,.0f}", xy=(d[-1], pf["yhat"][-1]), fontsize=7.5, color=color, va="center",)

def volume_profile(df: pd.DataFrame, bins: int = 40) -> dict:
    """
    Compute Volume Profile – distribution of traded volume across price levels.d
    Divides the price range into *bins* equal-width buckets and accumulates
    the volume of every bar whose Close falls in that bucket.
    Returns
    -------
    dict with keys:
        bin_prices  – centre price of each bucket
        bin_vols    – total volume per bucket
        poc_price   – Point of Control (price with highest volume)
        poc_bin     – index of POC bucket
        va_low      – Value Area lower bound (70 % of volume)
        va_high     – Value Area upper bound
        hvn_prices  – High Volume Node prices (top 20 % buckets)
        lvn_prices  – Low Volume Node prices (bottom 20 % buckets)
        total_vol   – total volume in the period
    """
    close = df["Close"].astype(float)
    vol = df["Volume"].astype(float) if "Volume" in df.columns else pd.Series(np.ones(len(df)))
    price_min, price_max = float(close.min()), float(close.max())
    if price_max <= price_min:
        return None
    bin_size = (price_max - price_min) / bins
    bin_vols = np.zeros(bins)
    bin_prices = np.array([price_min + (i + 0.5) * bin_size for i in range(bins)])
    for pr, vl in zip(close.values, vol.values):
        if pd.isna(pr) or pd.isna(vl) or vl <= 0:
            continue
        idx = min(int((pr - price_min) / bin_size), bins - 1)
        bin_vols[idx] += vl
    poc_bin = int(np.argmax(bin_vols))
    poc_price = float(bin_prices[poc_bin])
    # Value Area – bins containing 70 % of total volume (sorted by volume desc)
    total_vol = float(bin_vols.sum())
    sorted_idx = np.argsort(bin_vols)[::-1]
    cum = 0.0
    va_bins = set()
    for idx in sorted_idx:
        cum += bin_vols[idx]
        va_bins.add(idx)
        if cum / total_vol >= 0.70:
            break
    va_low  = float(bin_prices[min(va_bins)])
    va_high = float(bin_prices[max(va_bins)])
    # HVN / LVN – top/bottom 20 % of bins by volume
    vol_thresh_high = np.percentile(bin_vols[bin_vols > 0], 80)
    vol_thresh_low  = np.percentile(bin_vols[bin_vols > 0], 20)
    hvn_prices = [float(bin_prices[i]) for i in range(bins) if bin_vols[i] >= vol_thresh_high]
    lvn_prices = [float(bin_prices[i]) for i in range(bins) if 0 < bin_vols[i] <= vol_thresh_low]
    return {
        "bin_prices": bin_prices.tolist(),
        "bin_vols": bin_vols.tolist(),
        "poc_price": poc_price,
        "poc_bin": poc_bin,
        "va_low": va_low,
        "va_high": va_high,
        "hvn_prices": hvn_prices,
        "lvn_prices": lvn_prices,
        "total_vol": total_vol,
        "bin_size": bin_size,
    }
 
def analyze_volume_momentum(df: pd.DataFrame, forward: int = 1) -> dict:
    """
    Analyse the relationship between current candle volume and the next
    *forward* candle(s) return.
    Splits volume into quartiles and computes per-quartile:
      - average next-candle return
      - hit rate  (% of candles where next close > current close)
      - median next-candle return
    Also analyses breakout quality: for every bar where price crossed above
    BB_upper or MA crossover fired, compare volume of successful vs failed moves.
    Returns
    dict with keys:
        quartile_labels   – ["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"]
        quartile_avg_ret  – avg next-candle return per quartile
        quartile_hit_rate – hit rate per quartile
        quartile_med_ret  – median next-candle return per quartile
        quartile_bounds   – (q25, q50, q75) volume thresholds
        vol_avg           – 20-bar rolling average volume (last bar)
        breakout_success_vol_ratio – avg vol/vol_avg for successful BUY signals
        breakout_fail_vol_ratio    – avg vol/vol_avg for failed BUY signals
        n_success, n_fail          – counts
        correlation       – Pearson correlation between volume and next return
        signal_vol_summary – text summary of key finding
    """
    close = df["Close"].astype(float)
    vol = df["Volume"].astype(float) if "Volume" in df.columns else None
    if vol is None or vol.sum() == 0 or len(df) < 30:
        return None
    vol_avg20 = vol.rolling(20).mean()
    returns = close.pct_change(forward).shift(-forward) * 100  # forward return
    # Remove NaN rows
    mask = vol.notna() & returns.notna() & (vol > 0)
    v = vol[mask].values
    r = returns[mask].values
    va20 = vol_avg20[mask].values
    if len(v) < 20:
        return None
    # Quartile boundaries
    q25, q50, q75 = np.percentile(v, [25, 50, 75])
    quartile_labels  = ["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"]
    quartile_avg_ret, quartile_hit_rate, quartile_med_ret = [], [], []
    for lo, hi in [(0, q25), (q25, q50), (q50, q75), (q75, np.inf)]:
        mask_q = (v >= lo) & (v < hi) if hi < np.inf else (v >= lo)
        r_q = r[mask_q]
        if len(r_q) == 0:
            quartile_avg_ret.append(0.0)
            quartile_hit_rate.append(50.0)
            quartile_med_ret.append(0.0)
        else:
            quartile_avg_ret.append(float(np.mean(r_q)))
            quartile_hit_rate.append(float(np.mean(r_q > 0) * 100))
            quartile_med_ret.append(float(np.median(r_q)))
    # Pearson correlation vol → next return
    corr = float(np.corrcoef(v, r)[0, 1]) if len(v) > 2 else 0.0
    # Breakout analysis – need signal column
    success_ratios, fail_ratios = [], []
    if "signal" in df.columns and "EMA_short" in df.columns:
        sig = df["signal"].fillna(0)
        ema_s = df["EMA_short"].astype(float)
        ema_l = df["EMA_long"].astype(float)
        sig_arr = sig.values
        v_full = vol.values
        va_full = vol_avg20.values
        cl_full = close.values
        for i in range(1, len(df) - 1):
            if sig_arr[i] == 1 and sig_arr[i-1] != 1:  # BUY signal fired
                if pd.isna(va_full[i]) or va_full[i] == 0:
                    continue
                ratio = v_full[i] / va_full[i]
                # Success = next candle closed higher
                if cl_full[i+1] > cl_full[i]:
                    success_ratios.append(ratio)
                else:
                    fail_ratios.append(ratio)
    bo_success = float(np.mean(success_ratios)) if success_ratios else None
    bo_fail = float(np.mean(fail_ratios)) if fail_ratios else None
    # Text summary
    vol_last = float(v[-1]) if len(v) else 0
    va20_last = float(va20[-1]) if len(va20) and not np.isnan(va20[-1]) else 1
    vol_ratio = vol_last / va20_last if va20_last > 0 else 1.0
    q4_ret  = quartile_avg_ret[3]
    q1_ret  = quartile_avg_ret[0]
    vol_effect = q4_ret - q1_ret
    if vol_effect > 0.3: summary = f"HIGH VOLUME predicts UPSIDE: Q4 avg +{q4_ret:.2f}% vs Q1 +{q1_ret:.2f}% (diff {vol_effect:+.2f}%)"
    elif vol_effect < -0.3: summary = f"HIGH VOLUME predicts DOWNSIDE: Q4 avg {q4_ret:.2f}% vs Q1 {q1_ret:.2f}%"
    else: summary = f"Volume has LOW predictive power for next-candle direction (diff {vol_effect:+.2f}%)"
 
    if bo_success and bo_fail and bo_success > bo_fail * 1.2:
        summary += f" | Breakouts confirmed by volume: success avg {bo_success:.1f}x vs fail {bo_fail:.1f}x avg"
    return {
        "quartile_labels": quartile_labels,
        "quartile_avg_ret": quartile_avg_ret,
        "quartile_hit_rate": quartile_hit_rate,
        "quartile_med_ret": quartile_med_ret,
        "quartile_bounds": (float(q25), float(q50), float(q75)),
        "vol_avg": float(va20_last),
        "vol_ratio_now": vol_ratio,
        "breakout_success_vol_ratio": bo_success,
        "breakout_fail_vol_ratio": bo_fail,
        "n_success": len(success_ratios),
        "n_fail": len(fail_ratios),
        "correlation": corr,
        "signal_vol_summary": summary,
    }
 
def draw_volume_profile(ax_price, ax_vp, df: pd.DataFrame, bins: int = 35):
    """
    Draw Volume Profile as a horizontal bar chart on *ax_vp*, and annotate
    POC / Value Area lines on the price panel *ax_price*.
    Parameters
    ax_price: matplotlib Axes – the main price panel
    ax_vp: matplotlib Axes – dedicated VP panel (placed to the right)
    df: price DataFrame with Close + Volume columns
    bins: number of price buckets
    """
    vp = volume_profile(df, bins=bins)
    if vp is None:
        return
    bin_prices, bin_vols = np.array(vp["bin_prices"]), np.array(vp["bin_vols"])
    poc_price, va_low, va_high = float(vp["poc_price"]), float(vp["va_low"]), float(vp["va_high"]) 
    # Colour coding
    vol_max = bin_vols.max()
    colors = []
    for i, (bp, bv) in enumerate(zip(bin_prices, bin_vols)):
        if i == vp["poc_bin"]: 
            colors.append("#e53935") # POC – red
        elif va_low <= bp <= va_high:
            colors.append("#1565c0") # Value Area – blue
        elif bv >= np.percentile(bin_vols[bin_vols>0], 75):
            colors.append("#4fc3f7") # HVN – light blue
        else:
            colors.append("#b0bec5") # Normal – grey
    ax_vp.barh(bin_prices, bin_vols, height=vp["bin_size"] * 0.85, color=colors, alpha=0.85)
    ax_vp.axhline(poc_price, color="#e53935", lw=1.2, ls="--", alpha=0.9)
    ax_vp.axhline(va_low, color="#1565c0", lw=0.8, ls=":", alpha=0.7)
    ax_vp.axhline(va_high, color="#1565c0", lw=0.8, ls=":", alpha=0.7)
    ax_vp.set_xlabel("Volume", fontsize=7)
    ax_vp.tick_params(axis="both", labelsize=7)
    ax_vp.set_title("Volume Profile", fontsize=8)
    ax_vp.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}k"))
    # Annotate price panel with POC + VA
    if ax_price is not None:
        ax_price.axhline(poc_price, color="#e53935", lw=1.0, ls="--", alpha=0.6, label=f"POC ${poc_price:,.0f}")
        ax_price.axhspan(va_low, va_high, alpha=0.05, color="#1565c0", label=f"Value Area ${va_low:,.0f}–${va_high:,.0f}")

def _draw_price_panel(ax, df, close, trades, p, zoom=False):
    """Draw price panel with MA, BB and trades. zoom=True = last 6 months."""
    ax.plot(close.index, close.values, color="#1a1a2e", lw=1.2 if not zoom else 1.5, label="Price", zorder=3)
    ax.plot(df.index, df["SMA_short"], color="#e94560", lw=1, ls="--", label=f'SMA{p["MA_SHORT"]}', alpha=0.8)
    ax.plot(df.index, df["SMA_long"],  color="#0f3460", lw=1, ls="--", label=f'SMA{p["MA_LONG"]}', alpha=0.8)
    ax.fill_between(df.index, df["BB_upper"], df["BB_lower"], alpha=0.10, color="#4fc3f7", label="Bollinger Bands")
    ax.plot(df.index, df["BB_upper"], color="#4fc3f7", lw=0.8, ls=":")
    ax.plot(df.index, df["BB_lower"], color="#4fc3f7", lw=0.8, ls=":")
    buys  = trades[trades["type"] == "BUY"]; sells = trades[trades["type"].isin(["SELL", "CLOSE"])]; stops = trades[trades["type"] == "STOP-LOSS"]
    # If zoom – draw only trades in zoomed window
    if zoom:
        zoom_start = df.index[-1] - pd.DateOffset(months=6)
        buys  = buys[buys["date"]  >= zoom_start]; sells = sells[sells["date"] >= zoom_start]; stops = stops[stops["date"] >= zoom_start]
    sz = 11 if zoom else 9
    for _, t in buys.iterrows():
        if t["date"] in close.index:
            ax.annotate("▲", xy=(t["date"], t["price"]), color="#00c853", fontsize=sz, ha="center", va="top", fontweight="bold")
    for _, t in sells.iterrows():
        if t["date"] in close.index:
            ax.annotate("▼", xy=(t["date"], t["price"]), color="#d50000", fontsize=sz, ha="center", va="bottom", fontweight="bold")
    for _, t in stops.iterrows():
        if t["date"] in close.index:
            ax.annotate("✕", xy=(t["date"], t["price"]), color="#ff6d00", fontsize=sz, ha="center")
    legend_elems = [Patch(color="#00c853", label="BUY"), Patch(color="#d50000", label="SELL"), Patch(color="#ff6d00", label="STOP-LOSS"),]
    ax.legend(loc="upper left", fontsize=7, ncol=4, framealpha=0.7, handles=ax.get_legend_handles_labels()[0] + legend_elems)
    ax.set_ylabel("Price (USD)", fontsize=9)
    ax.grid(alpha=0.2)

def plot_asset(result: dict, save_path: str = None):
    """
    Produce the full 5-row × 2-column per-asset chart and save as PNG.
    Layout:
        Left column  : Full history + Monte Carlo fan (profile colour).
        Right column : Last 6-month zoom + MC fan (blue border highlight).
        Rows         : Price panel, Equity curve, RSI, MACD, ATR.
    """
    df = result["price_df"]; eq = result["equity_df"]; trades = result["trades_df"]
    close = df["Close"].astype(float)
    name = result["asset"]; p = result["p"]
    # Zoom window – last 6 months
    zoom_start = df.index[-1] - pd.DateOffset(months=6)
    df_z = df[df.index >= zoom_start]; close_z = close[close.index >= zoom_start]; eq_z = eq[eq.index >= zoom_start]
    ts = datetime.now().strftime("%d.%m.%Y  %H:%M:%S")
    fig = plt.figure(figsize=(26, 18))
    fig.suptitle(f"{name}  {result.get('profile','')} – Backtest Results\n Generated: {ts}", fontsize=16, fontweight="bold", y=0.995)
    gs = gridspec.GridSpec(5, 2, hspace=0.55, wspace=0.08, height_ratios=[3, 1, 1, 1, 1], width_ratios=[2, 1])
    # Price panels
    ax1 = fig.add_subplot(gs[0, 0])
    _draw_price_panel(ax1, df, close, trades, p, zoom=False)
    mc_profile = result.get("profile", "TECH")
    mc_meta = MC_PROFILE_META.get(mc_profile, MC_PROFILE_META["TECH"])
    draw_monte_carlo(ax1, close, profile=mc_profile)
    draw_prophet(ax1, close, n_days=MC_DAYS)
    ax1.set_title(f"Price + MA + Bollinger Bands + Trades (full period) | MC {MC_SIMULATIONS}× / {MC_DAYS}d [{mc_meta['label']}] + Prophet", fontsize=10)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    # Price zoom     
    ax1z = fig.add_subplot(gs[0, 1])
    _draw_price_panel(ax1z, df_z, close_z, trades, p, zoom=True)
    draw_monte_carlo(ax1z, close_z, profile=mc_profile)
    draw_prophet(ax1z, close, n_days=MC_DAYS)
    ax1z.xaxis.set_major_formatter(mdates.DateFormatter("%m/%y"))
    ax1z.set_title("Zoom – last 6 months + MC & Prophet forecast", fontsize=10, color="#1565c0")
    for spine in ax1z.spines.values():
        spine.set_edgecolor("#1565c0")
        spine.set_linewidth(1.5)
    ax1z.set_ylabel("")
    # Equity curve
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax2.plot(eq.index, eq["equity"], color="#7b1fa2", lw=1.5, label="Equity")
    ax2.axhline(INITIAL_CAP, color="gray", ls="--", lw=0.8, label="Initial capital")
    ax2.fill_between(eq.index, INITIAL_CAP, eq["equity"], where=eq["equity"] >= INITIAL_CAP, alpha=0.2, color="#00c853")
    ax2.fill_between(eq.index, INITIAL_CAP, eq["equity"], where=eq["equity"] <  INITIAL_CAP, alpha=0.2, color="#d50000")
    ax2.set_ylabel("Capital (USD)", fontsize=9)
    ax2.set_title("Equity curve", fontsize=10)
    ax2.legend(fontsize=8, loc="upper left")
    ax2.grid(alpha=0.2)
    # zoom equity
    ax2z = fig.add_subplot(gs[1, 1])
    ax2z.plot(eq_z.index, eq_z["equity"], color="#7b1fa2", lw=1.8)
    ax2z.axhline(INITIAL_CAP, color="gray", ls="--", lw=0.8)
    ax2z.fill_between(eq_z.index, INITIAL_CAP, eq_z["equity"], where=eq_z["equity"] >= INITIAL_CAP, alpha=0.2, color="#00c853")
    ax2z.fill_between(eq_z.index, INITIAL_CAP, eq_z["equity"], where=eq_z["equity"] <  INITIAL_CAP, alpha=0.2, color="#d50000")
    ax2z.xaxis.set_major_formatter(mdates.DateFormatter("%m/%y"))
    ax2z.grid(alpha=0.2)
    for spine in ax2z.spines.values():
        spine.set_edgecolor("#1565c0"); spine.set_linewidth(1.5)
    ax2z.set_ylabel("")
    # RSI
    rsi_ob = p["RSI_OB"]; rsi_os = p["RSI_OS"]
    ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)
    ax3.plot(df.index, df["RSI"], color="#f57c00", lw=1.2, label="RSI")
    ax3.axhline(rsi_ob, color="#d50000", ls="--", lw=0.8, alpha=0.7, label=f"OB {rsi_ob}")
    ax3.axhline(rsi_os, color="#00c853", ls="--", lw=0.8, alpha=0.7, label=f"OS {rsi_os}")
    ax3.axhline(50, color="gray", ls=":", lw=0.6)
    ax3.fill_between(df.index, rsi_ob, df["RSI"], where=df["RSI"] >= rsi_ob, alpha=0.2, color="#d50000")
    ax3.fill_between(df.index, rsi_os, df["RSI"], where=df["RSI"] <= rsi_os, alpha=0.2, color="#00c853")
    ax3.set_ylim(0, 100); ax3.set_ylabel("RSI", fontsize=9)
    ax3.set_title(f'RSI ({p["RSI_PERIOD"]})', fontsize=10)
    ax3.legend(fontsize=8, loc="upper left", ncol=3); ax3.grid(alpha=0.2)
    # RSI zoom
    ax3z = fig.add_subplot(gs[2, 1])
    ax3z.plot(df_z.index, df_z["RSI"], color="#f57c00", lw=1.5)
    ax3z.axhline(rsi_ob, color="#d50000", ls="--", lw=0.8, alpha=0.7)
    ax3z.axhline(rsi_os, color="#00c853", ls="--", lw=0.8, alpha=0.7)
    ax3z.axhline(50, color="gray", ls=":", lw=0.6)
    ax3z.fill_between(df_z.index, rsi_ob, df_z["RSI"], where=df_z["RSI"] >= rsi_ob, alpha=0.2, color="#d50000")
    ax3z.fill_between(df_z.index, rsi_os, df_z["RSI"], where=df_z["RSI"] <= rsi_os, alpha=0.2, color="#00c853")
    ax3z.set_ylim(0, 100)
    ax3z.xaxis.set_major_formatter(mdates.DateFormatter("%m/%y"))
    ax3z.grid(alpha=0.2)
    for spine in ax3z.spines.values():
        spine.set_edgecolor("#1565c0"); spine.set_linewidth(1.5)
    ax3z.set_ylabel("")
    # MACD
    ax4 = fig.add_subplot(gs[3, 0], sharex=ax1)
    ax4.plot(df.index, df["MACD"],     color="#1565c0", lw=1.2, label="MACD")
    ax4.plot(df.index, df["MACD_sig"], color="#e53935", lw=1.2, label="Signal")
    hist_colors = ["#00c853" if v >= 0 else "#d50000" for v in df["MACD_hist"]]
    ax4.bar(df.index, df["MACD_hist"], color=hist_colors, alpha=0.5, width=1, label="Histogram")
    ax4.axhline(0, color="gray", ls="-", lw=0.5)
    ax4.set_ylabel("MACD", fontsize=9)
    ax4.set_title(f'MACD ({p["MACD_FAST"]},{p["MACD_SLOW"]},{p["MACD_SIGNAL"]})', fontsize=10)
    ax4.legend(fontsize=8, loc="upper left", ncol=3); ax4.grid(alpha=0.2)
    # MACD zoom
    ax4z = fig.add_subplot(gs[3, 1])
    ax4z.plot(df_z.index, df_z["MACD"],     color="#1565c0", lw=1.5, label="MACD")
    ax4z.plot(df_z.index, df_z["MACD_sig"], color="#e53935", lw=1.5, label="Signal")
    hist_colors_z = ["#00c853" if v >= 0 else "#d50000" for v in df_z["MACD_hist"]]
    ax4z.bar(df_z.index, df_z["MACD_hist"], color=hist_colors_z, alpha=0.5, width=1)
    ax4z.axhline(0, color="gray", ls="-", lw=0.5)
    ax4z.xaxis.set_major_formatter(mdates.DateFormatter("%m/%y"))
    ax4z.grid(alpha=0.2)
    for spine in ax4z.spines.values():
        spine.set_edgecolor("#1565c0"); spine.set_linewidth(1.5)
    ax4z.set_ylabel("")
    # ATR
    ax5 = fig.add_subplot(gs[4, 0], sharex=ax1)
    ax5.plot(df.index, df["ATR"], color="#00838f", lw=1.2, label="ATR")
    ax5.fill_between(df.index, 0, df["ATR"], alpha=0.15, color="#00838f")
    ax5.set_ylabel("ATR", fontsize=9)
    ax5.set_title(f'Average True Range ({p["ATR_PERIOD"]}) – volatility', fontsize=10)
    ax5.legend(fontsize=8); ax5.grid(alpha=0.2)
    # ATR zoom
    ax5z = fig.add_subplot(gs[4, 1])
    ax5z.plot(df_z.index, df_z["ATR"], color="#00838f", lw=1.5)
    ax5z.fill_between(df_z.index, 0, df_z["ATR"], alpha=0.15, color="#00838f")
    ax5z.xaxis.set_major_formatter(mdates.DateFormatter("%m/%y"))
    ax5z.grid(alpha=0.2)
    for spine in ax5z.spines.values():
        spine.set_edgecolor("#1565c0"); spine.set_linewidth(1.5)
    ax5z.set_ylabel("")
    # Bottom axis for left column – annual dates
    for i in range(1, 6):
        eval(f"ax{i}.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))")
        eval(f"ax{i}.xaxis.set_major_locator(mdates.MonthLocator(interval=3))")
        eval(f"ax{i}.tick_params(axis='x', labelrotation=45, labelsize=8)")
        for lbl in eval(f"ax{i}.get_xticklabels()"):
            lbl.set_ha("right")
    # Bottom axis for right column (zoom) – monthly dates
    for i in range(1, 6):
        eval(f"ax{i}z.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))")
        eval(f"ax{i}z.xaxis.set_major_locator(mdates.MonthLocator(interval=1))")
        eval(f"ax{i}z.tick_params(axis='x', labelrotation=45, labelsize=8)")
        for lbl in eval(f"ax{i}z.get_xticklabels()"):   
            lbl.set_ha("right")
    # Column headers
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  → Chart saved: {save_path}")
    else:
        plt.show()
    plt.close()

def plot_summary(results: list):
    """Comparative chart of all assets."""
    ts_label = datetime.now().strftime("%d.%m.%Y  %H:%M:%S")
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f"Summary Comparison of All Assets: {ts_label}", fontsize=15, fontweight="bold")
    names = [r["asset"] for r in results]
    returns = [r["total_return"] for r in results]
    bh = [r["bh_return"] for r in results]
    sharpes = [r["sharpe"] for r in results]
    drawdowns= [r["max_drawdown"] for r in results]
    winrates= [r["win_rate"] for r in results]
    x = np.arange(len(names))
    w = 0.38
    # Returns vs Buy & Hold
    ax = axes[0, 0]
    bars1 = ax.bar(x - w/2, returns, w, label="Strategy", color="#1565c0", alpha=0.85)
    bars2 = ax.bar(x + w/2, bh, w, label="Buy & Hold", color="#78909c", alpha=0.85)
    ax.axhline(0, color="black", lw=0.7)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Return (%)"); ax.set_title("Total Return vs Buy & Hold")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + (1 if h >= 0 else -3),
                f"{h:+.0f}%", ha="center", va="bottom", fontsize=6.5, color="#1565c0", fontweight="bold")
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + (1 if h >= 0 else -3),
                f"{h:+.0f}%", ha="center", va="bottom", fontsize=6.5, color="#546e7a")
    # Sharpe Ratio
    ax = axes[0, 1]
    colors = ["#00c853" if s > 1 else "#ff6d00" if s > 0 else "#d50000" for s in sharpes]
    ax.bar(names, sharpes, color=colors, alpha=0.85)
    ax.axhline(1, color="gray", ls="--", lw=0.8, label="Sharpe = 1")
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Sharpe Ratio"); ax.set_title("Sharpe Ratio (>1 = good)")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    # Max Drawdown
    ax = axes[1, 0]
    ax.bar(names, drawdowns, color="#e53935", alpha=0.75)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Max Drawdown (%)"); ax.set_title("Maximum Capital Loss")
    ax.grid(axis="y", alpha=0.3)
    # Win Rate
    ax = axes[1, 1]
    colors = ["#00c853" if w > 55 else "#ff6d00" if w > 45 else "#d50000" for w in winrates]
    ax.bar(names, winrates, color=colors, alpha=0.85)
    ax.axhline(50, color="gray", ls="--", lw=0.8, label="50 %")
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Win Rate (%)"); ax.set_title("Trade Success Rate")
    ax.set_ylim(0, 100); ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fname = os.path.join(OUTPUT_DIR, "summary_comparison.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"\n  → Summary chart saved: {fname}")
    plt.close()

def export_table_png(table: list, headers: list, results: list):
    """Export summary results table to PNG file."""
    ts_label = datetime.now().strftime("%d.%m.%Y  %H:%M:%S")
    fname = os.path.join(OUTPUT_DIR, "summary_table.png") 
    n_rows = len(table); n_cols = len(headers)
    fig_h = 1.2 + n_rows * 0.42
    fig, ax = plt.subplots(figsize=(max(18, n_cols * 1.5), fig_h))
    ax.axis("off")
    # Cell colors
    col_colors = ["#1E50A0"] * n_cols
    cell_colors = []
    for i, row in enumerate(table):
        row_c = []
        bg = "#EEF2F7" if i % 2 == 0 else "#FFFFFF"
        for j, val in enumerate(row):
            if j in (3, 4, 5): # Return, B&H, Alpha – green/red
                try:
                    v = float(str(val).replace(" %","").replace("+",""))
                    if j == 5:   # Alpha
                        c = "#d4edda" if v >= 0 else "#f8d7da"
                    else:
                        c = "#d4edda" if v >= 0 else "#f8d7da"
                except:
                    c = bg
                row_c.append(c)
            else:
                row_c.append(bg)
        cell_colors.append(row_c)
    tbl = ax.table( cellText=table, colLabels=headers, cellColours=cell_colors, colColours=col_colors, cellLoc="center", loc="center",)
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.5)
    # Header style
    for j in range(n_cols):
        cell = tbl[0, j]
        cell.set_text_props(color="white", fontweight="bold")
        cell.set_facecolor("#1E50A0")
    # Highlight best return
    best_idx = max(range(n_rows), key=lambda i: float(str(table[i][3]).replace(" %","").replace("+","")))
    for j in range(n_cols):
        tbl[best_idx + 1, j].set_facecolor("#fff3cd")
    fig.suptitle(f"MULTI-ASSET BACKTEST  –  Summary Results\nGenerated: {ts_label}", fontsize=12, fontweight="bold", y=0.98, color="#1E50A0")
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"  → Table exported: {fname}")
    plt.close()

def run_hourly_signals(interval: str = "1h"):
    """
    Downloads hourly (or 4h) data for all assets,
    calculates indicators and exports PNG signal table.
    Usage:
        python trading_backtest.py --signals-hourly
        python trading_backtest.py --signals-hourly --interval 4h
    """
    iv = INTERVAL_SETTINGS.get(interval, INTERVAL_SETTINGS["1h"])
    ts_label = datetime.now().strftime("%d.%m.%Y  %H:%M:%S")
    fname = os.path.join(OUTPUT_DIR, f"signals_{interval}.png")
    print(f"HOURLY ANALYSIS OF ALL ASSETS  [{iv['label']}]")
    print(f"{ts_label}")
    rows = []
    for name, ticker in ASSETS.items():
        profile_name = ASSET_PROFILES.get(name, "TECH"); p = PROFILES[profile_name]
        print(f"{name:<18} ({ticker})...", end="", flush=True)
        try:
            if interval == "4h":
                raw = yf.download(ticker, period=iv["period"], interval="1h", progress=False, auto_adjust=True)
                if isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = raw.columns.get_level_values(0)
                raw = raw.resample("4h").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna()
            else:
                raw = yf.download(ticker, period=iv["period"], interval=interval, progress=False, auto_adjust=True)
                if isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = raw.columns.get_level_values(0)
            if raw.empty or len(raw) < max(p["MA_LONG"] + 5, 50):
                print(f"insufficient data ({len(raw) if not raw.empty else 0} candles)")
                rows.append([name, profile_name, ticker, "N/A", "–", "–", "–", "–", "–", "–", "–", "–"])
                continue
            df = compute_indicators(raw.copy(), p); last = df.iloc[-1]; c = df["Close"].astype(float)
            price = float(c.iloc[-1]); prev = float(c.iloc[-2]); change = (price - prev) / prev * 100
            atr = float(last["ATR"]) if not pd.isna(last["ATR"]) else 0
            rsi = float(last["RSI"]) if not pd.isna(last["RSI"]) else 50
            bb_pct = float(last["BB_pct"]) if not pd.isna(last["BB_pct"]) else 0.5
            bb_upper = float(last["BB_upper"]) if not pd.isna(last["BB_upper"]) else price
            bb_lower = float(last["BB_lower"]) if not pd.isna(last["BB_lower"]) else price
            macd = float(last["MACD"]) if not pd.isna(last["MACD"]) else 0
            macd_sig = float(last["MACD_sig"]) if not pd.isna(last["MACD_sig"]) else 0
            macd_hist = float(last["MACD_hist"]) if not pd.isna(last["MACD_hist"]) else 0
            ema_short = float(last["EMA_short"]) if not pd.isna(last["EMA_short"]) else price
            ema_long = float(last["EMA_long"]) if not pd.isna(last["EMA_long"])  else price
            sma_short = float(last["SMA_short"]) if not pd.isna(last["SMA_short"]) else price
            rsi_mid = (p["RSI_OB"] + p["RSI_OS"]) / 2
            conds_buy = {"MA": ema_short > ema_long,"RSI": rsi < rsi_mid,"BB": bb_pct < 0.4,"MACD": macd > macd_sig,"ATR": price > sma_short,}
            buy_score = sum(conds_buy.values()); sell_score = sum(not v for v in conds_buy.values())
            if buy_score >= 3: signal = "BUY"
            elif sell_score >= 3: signal = "SELL"
            else: signal = "NEU"
            def _ic(v): return "✔" if v else "x"
            ind_icons = (f'{_ic(conds_buy["MA"])}  {_ic(conds_buy["RSI"])}  {_ic(conds_buy["BB"])}  {_ic(conds_buy["MACD"])}  {_ic(conds_buy["ATR"])}')
            # Price levels
            buffer = 0.005
            buy_limit = bb_lower * (1 + buffer)
            stop_loss = buy_limit - p["ATR_SL_MULT"] * atr
            risk_per = buy_limit - stop_loss
            tp1 = buy_limit + risk_per
            sl_pct = (price - stop_loss) / price * 100 if price > 0 else 0
            tp1_pct = (tp1 - price) / price * 100 if price > 0 else 0
            bb_up_pct = (bb_upper - price) / price * 100 if price > 0 else 0
            chg_str = f"{change:+.2f}%"
            arrow = "▲" if change >= 0 else "▼"
            print(f"{signal:<4}  BUY:{buy_score}/5")
            rows.append([name,profile_name,f"${price:,.2f} {arrow}{chg_str}", signal, f"{buy_score}/5", f"{sell_score}/5", ind_icons,f"${buy_limit:,.2f}", f"${stop_loss:,.2f} (-{sl_pct:.1f}%)", f"${tp1:,.2f} (+{tp1_pct:.1f}%)", f"${bb_upper:,.2f} (+{bb_up_pct:.1f}%)", f"RSI:{rsi:.0f}  MACD:{'▲' if macd_hist>=0 else '▼'}",])
        except Exception as e:
            print(f" {e}")
            rows.append([name, profile_name, ticker, "ERR", "–", "–", "–", "–", "–", "–", "–", str(e)[:30]])
 
    headers = ["Asset", "Profile", f"Price ({interval})", "Signal","BUY sc.", "SELL sc.", "MA / RSI / BB / MACD / ATR","Buy Limit", "Stop-Loss", "Take Profit 1", "SELL target (BB upper)","RSI / MACD trend"]
    n_rows, n_cols = len(rows), len(headers)
    signal_bg = {"BUY": "#d4edda", "SELL": "#f8d7da", "NEU": "#fff3cd", "N/A": "#eeeeee", "ERR": "#eeeeee"}
    cell_colors = []
    for i, row in enumerate(rows):
        bg = "#EEF2F7" if i % 2 == 0 else "#FFFFFF"
        row_c = []
        for j in range(n_cols):
            if j == 3: row_c.append(signal_bg.get(row[j], bg))
            elif j == 7: row_c.append("#e8f5e9")
            elif j == 8: row_c.append("#fdecea")
            elif j in (9, 10): row_c.append("#e3f2fd")
            else: row_c.append(bg)
        cell_colors.append(row_c)
    fig, ax = plt.subplots(figsize=(28, 1.8 + n_rows * 0.60))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=headers, cellColours=cell_colors, colColours=["#1E50A0"] * n_cols, cellLoc="center", loc="center",)
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 2.0)
    for j in range(n_cols):
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    for i in range(1, n_rows + 1):
        if rows[i-1][3] in ("BUY", "SELL"):
            tbl[i, 3].set_text_props(fontweight="bold")
    fig.suptitle(
        f"HOURLY ANALYSIS OF ALL ASSETS  |  Interval: {interval} ({iv['label']})"
        f"  |  Generated: {ts_label}", fontsize=11, fontweight="bold", y=0.99, color="#1E50A0"
    )
    plt.tight_layout(); plt.savefig(fname, dpi=150, bbox_inches="tight"); print(f"\n  -> PNG table saved: {fname}"); plt.close()

def export_signals_png(results: list):
    """Export current signals and price levels to PNG table."""
    ts_label = datetime.now().strftime("%d.%m.%Y  %H:%M:%S")
    fname = os.path.join(OUTPUT_DIR, "signals.png") 

    rows = []
    for r in results:
        df = r["price_df"].copy()
        df = df[df["Close"].notna() & (df["Close"] > 0)]
        if df.empty:
            continue
        p = r["p"]; name = r["asset"]
        last = df.iloc[-1]
        price = float(df["Close"].astype(float).iloc[-1])
        atr = float(last["ATR"]) if not pd.isna(last["ATR"]) else 0
        rsi = float(last["RSI"]) if not pd.isna(last["RSI"]) else 50
        bb_pct = float(last["BB_pct"]) if not pd.isna(last["BB_pct"]) else 0.5
        bb_upper = float(last["BB_upper"]) if not pd.isna(last["BB_upper"]) else price
        bb_lower = float(last["BB_lower"]) if not pd.isna(last["BB_lower"]) else price
        macd = float(last["MACD"]) if not pd.isna(last["MACD"]) else 0
        macd_sig = float(last["MACD_sig"]) if not pd.isna(last["MACD_sig"]) else 0
        ema_short = float(last["EMA_short"]) if not pd.isna(last["EMA_short"]) else price
        ema_long = float(last["EMA_long"]) if not pd.isna(last["EMA_long"]) else price
        sma_short = float(last["SMA_short"]) if not pd.isna(last["SMA_short"]) else price
        rsi_mid = (p["RSI_OB"] + p["RSI_OS"]) / 2
        conds_buy = {"MA": ema_short > ema_long,"RSI": rsi < rsi_mid,"BB": bb_pct < 0.4,"MACD": macd > macd_sig,"ATR": price > sma_short,}
        buy_score  = sum(conds_buy.values()); sell_score = sum(not v for v in conds_buy.values())
        if buy_score >= 3: signal = "BUY"
        elif sell_score >= 3: signal = "SELL"
        else: signal = "NEU"
        stop_loss = price - p["ATR_SL_MULT"] * atr
        take_profit = price + 2 * p["ATR_SL_MULT"] * atr
        sell_target = bb_upper
        buy_zone = bb_lower
        sl_pct = (price - stop_loss) / price * 100
        tp_pct = (take_profit - price) / price * 100
        st_pct = (sell_target - price) / price * 100
        def _ic(v): return "✔" if v else "×"
        ind_icons = f'{_ic(conds_buy["MA"])}    {_ic(conds_buy["RSI"])}    {_ic(conds_buy["BB"])}    {_ic(conds_buy["MACD"])}    {_ic(conds_buy["ATR"])}'
        rows.append([name, r.get("profile", "-"), f"${price:,.2f}", signal, f"{buy_score}/5", f"{sell_score}/5", ind_icons, f"${buy_zone:,.2f}", f"${stop_loss:,.2f}  ({sl_pct:.1f}%)", f"${take_profit:,.2f}  (+{tp_pct:.1f}%)", f"${sell_target:,.2f}  (+{st_pct:.1f}%)",])
    headers = ["Asset", "Profile", "Price", "Signal", "BUY sc.", "SELL sc.", "MA RSI BB MACD ATR", "BUY zone", "Stop-Loss", "Take Profit", "SELL target"]
    n_rows, n_cols = len(rows), len(headers)
    fig, ax = plt.subplots(figsize=(24, 1.4 + n_rows * 0.52))
    ax.axis("off")
    # Cell colors
    signal_colors = {"BUY": "#d4edda", "SELL": "#f8d7da", "NEU": "#fff3cd"}
    cell_colors = []
    for i, row in enumerate(rows):
        bg   = "#EEF2F7" if i % 2 == 0 else "#FFFFFF"
        row_c = []
        for j in range(n_cols):
            if j == 3:   # Signal column
                row_c.append(signal_colors.get(row[j], bg))
            else:
                row_c.append(bg)
        cell_colors.append(row_c)
    col_colors = ["#1E50A0"] * n_cols
    tbl = ax.table(cellText=rows, colLabels=headers, cellColours=cell_colors, colColours=col_colors, cellLoc="center", loc="center",)
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 1.8)
    # Header – white bold text
    for j in range(n_cols):
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    # Bold text for signal column
    for i in range(1, n_rows + 1):
        tbl[i, 3].set_text_props(fontweight="bold")
    fig.suptitle(f"CURRENT SIGNALS AND PRICE LEVELS | Generated: {ts_label}", fontsize=12, fontweight="bold", y=0.98, color="#1E50A0")
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"  → Signal table exported: {fname}")
    plt.close()

def export_order_levels_png(results: list):
    """
    PNG table with recommended price orders for each asset:
      - Buy Limit  = BB lower + 0.5% buffer (wait for slight bounce)
      - Stop-Loss  = Buy Limit − ATR_SL_MULT × ATR
      - Take Profit 1 = Buy Limit + 1× risk (R:R 1:1)
      - Take Profit 2 = BB upper              (R:R natural resistance)
      - Risk USD   = (Buy Limit − Stop-Loss) × qty at $10,000 capital
    """    
    ts_label = datetime.now().strftime("%d.%m.%Y  %H:%M:%S")
    fname = os.path.join(OUTPUT_DIR, "order_levels.png") 
    rows = []
    for r in results:
        df = r["price_df"].copy()
        # Drop NaN close rows so .iloc[-1] always gives a valid bar
        df = df[df["Close"].notna() & (df["Close"] > 0)]
        if df.empty:
            continue
        p = r["p"]; name = r["asset"]; last = df.iloc[-1]
        price = float(df["Close"].astype(float).iloc[-1])
        atr = float(last["ATR"]) if not pd.isna(last["ATR"]) else 0
        bb_upper = float(last["BB_upper"]) if not pd.isna(last["BB_upper"]) else price * 1.05
        bb_lower = float(last["BB_lower"]) if not pd.isna(last["BB_lower"]) else price * 0.95
        bb_pct = float(last["BB_pct"]) if not pd.isna(last["BB_pct"]) else 0.5
        ema_short = float(last["EMA_short"]) if not pd.isna(last["EMA_short"]) else price
        ema_long = float(last["EMA_long"]) if not pd.isna(last["EMA_long"]) else price
        macd = float(last["MACD"]) if not pd.isna(last["MACD"]) else 0
        macd_sig = float(last["MACD_sig"])  if not pd.isna(last["MACD_sig"])  else 0
        sma_short = float(last["SMA_short"]) if not pd.isna(last["SMA_short"]) else price
        rsi = float(last["RSI"]) if not pd.isna(last["RSI"]) else 50
        rsi_mid = (p["RSI_OB"] + p["RSI_OS"]) / 2
        # Overall signal
        conds_buy = [ema_short > ema_long, rsi < rsi_mid, bb_pct < 0.4, macd > macd_sig, price > sma_short]
        buy_score = sum(conds_buy)
        if buy_score >= 3: signal = "BUY"
        elif sum(not c for c in conds_buy) >= 3: signal = "SELL"
        else: signal = "NEU"
        # Price levels
        buffer = 0.005                                 # 0.5% buffer above BB lower
        buy_limit = bb_lower * (1 + buffer)            # Buy Limit = BB lower + buffer
        stop_loss = buy_limit - p["ATR_SL_MULT"] * atr # Stop loss under buy limit
        risk_per = buy_limit - stop_loss               # Risk per 1 item
        tp1 = buy_limit + risk_per                     # TP1 = R:R 1:1
        tp2 = bb_upper                                 # TP2 = BB upper
        # Risk in USD with capital INITIAL_CAP
        qty_est = (INITIAL_CAP * 0.95) / buy_limit if buy_limit > 0 else 0
        risk_usd = risk_per * qty_est
        # Percentage distance from current price
        bl_pct = (buy_limit - price) / price * 100
        sl_pct = (stop_loss - price) / price * 100
        tp1_pct = (tp1 - price) / price * 100
        tp2_pct = (tp2 - price) / price * 100
        rr1 = abs((tp1 - buy_limit) / risk_per) if risk_per > 0 else 0
        rr2 = abs((tp2 - buy_limit) / risk_per) if risk_per > 0 else 0
        rows.append([name,r.get("profile", "-"), f"${price:,.2f}", signal, f"${buy_limit:,.2f} ({bl_pct:+.1f}%)", f"${stop_loss:,.2f} ({sl_pct:+.1f}%)", f"${tp1:,.2f} ({tp1_pct:+.1f}%) 1:{rr1:.1f}", f"${tp2:,.2f} ({tp2_pct:+.1f}%) 1:{rr2:.1f}", f"${risk_usd:,.0f}",])
    headers = ["Asset", "Profile", "Price", "Signal", "Buy Limit (BB low+0.5%)", "Stop-Loss (ATR x mult.)", "Take Profit 1 (R:R 1:1)", "Take Profit 2 (BB upper)", "Risk/trade USD"]
    n_rows = len(rows); n_cols = len(headers)
    fig, ax = plt.subplots(figsize=(26, 1.6 + n_rows * 0.72))
    ax.axis("off")
    signal_bg = {"BUY": "#d4edda", "SELL": "#f8d7da", "NEU": "#fff3cd"}
    cell_colors = []
    for i, row in enumerate(rows):
        bg = "#EEF2F7" if i % 2 == 0 else "#FFFFFF"
        row_c = []
        for j in range(n_cols):
            if j == 3:
                row_c.append(signal_bg.get(row[j], bg))
            elif j == 4:   # Buy Limit – greenish
                row_c.append("#e8f5e9")
            elif j == 5:   # Stop-Loss – reddish
                row_c.append("#fdecea")
            elif j in (6, 7):   # Take Profit – blue
                row_c.append("#e3f2fd")
            elif j == 8:   # Risk – yellow
                row_c.append("#fffde7")
            else:
                row_c.append(bg)
        cell_colors.append(row_c)
    col_colors = ["#1E50A0"] * n_cols
    tbl = ax.table(cellText=rows, colLabels=headers, cellColours=cell_colors, colColours=col_colors, cellLoc="center", loc="center",)
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 2.8)
    for j in range(n_cols):
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    for i in range(1, n_rows + 1):
        tbl[i, 3].set_text_props(fontweight="bold")
    fig.suptitle(fig.suptitle(f"RECOMMENDED PRICE ORDERS  |  Generated: {ts_label}  | Buy Limit = BB low+0.5% | SL = BuyLim - ATR x mult. | TP1 = R:R 1:1 | TP2 = BB upper", fontsize=10, fontweight="bold", y=0.99, color="#1E50A0"))
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"  -> Order levels table exported: {fname}")
    plt.close()

def print_current_signals(results: list):
    """
    For each asset, display current status of all 5 indicators
    and calculate specific price levels:
      - BUY zone       (current price if BUY signal active, otherwise price where it would activate)
      - Stop-Loss      (entry price − ATR_SL_MULT × ATR)
      - SELL target    (Bollinger Band upper = natural profit target)
      - Take Profit    (entry price + 2 × ATR_SL_MULT × ATR – symmetric R:R 1:2)
    """
    ts = datetime.now().strftime("%d.%m.%Y  %H:%M:%S")
    print("\n" + "=" * 72)
    print(f"CURRENT SIGNALS AND PRICE LEVELS  –  {ts}")
    for r in results:
        df = r["price_df"].copy()
        # Drop NaN close rows so .iloc[-1] always gives a valid bar
        df = df[df["Close"].notna() & (df["Close"] > 0)]
        if df.empty:
            continue
        p = r["p"]; name = r["asset"]; last = df.iloc[-1]
        price = float(df["Close"].astype(float).iloc[-1])
        atr = float(last["ATR"]) if not pd.isna(last["ATR"]) else 0
        rsi = float(last["RSI"]) if not pd.isna(last["RSI"]) else 50
        bb_pct = float(last["BB_pct"]) if not pd.isna(last["BB_pct"]) else 0.5
        bb_upper = float(last["BB_upper"]) if not pd.isna(last["BB_upper"]) else price
        bb_lower = float(last["BB_lower"]) if not pd.isna(last["BB_lower"]) else price
        macd = float(last["MACD"]) if not pd.isna(last["MACD"]) else 0
        macd_sig = float(last["MACD_sig"]) if not pd.isna(last["MACD_sig"]) else 0
        ema_short = float(last["EMA_short"]) if not pd.isna(last["EMA_short"]) else price
        ema_long = float(last["EMA_long"])  if not pd.isna(last["EMA_long"]) else price
        sma_short = float(last["SMA_short"]) if not pd.isna(last["SMA_short"]) else price
        rsi_mid = (p["RSI_OB"] + p["RSI_OS"]) / 2
        # State of each indicator
        conds_buy = {"MA Crossover": ema_short > ema_long, "RSI": rsi < rsi_mid, "Bollinger": bb_pct < 0.4, "MACD": macd > macd_sig, "ATR trend": price > sma_short,}
        buy_score  = sum(conds_buy.values()); sell_score = sum(not v for v in conds_buy.values())
        if buy_score >= 3:
            signal_str = "✔ ACTIVE BUY SIGNAL"; signal_col = "BUY"
        elif sell_score >= 3:
            signal_str = "x ACTIVE SELL SIGNAL"; signal_col = "SELL"
        else:
            signal_str = "- NEUTRAL"; signal_col = "NEU"
        # Price levels
        stop_loss = price - p["ATR_SL_MULT"] * atr
        take_profit = price + 2 * p["ATR_SL_MULT"] * atr   # R:R 1:2
        sell_target = bb_upper # BB upper zone
        buy_zone_lo = bb_lower # BB lower zone
        buy_zone_hi = price
        print(f"\n{'─'*68}")
        print(f"{name:<14} [{r.get('profile',''):10s}]   Price: ${price:>10.2f}   {signal_str}")
        print(f"{'─'*68}")
        print(f"{'Indicator':<16} {'Value':>12}   {'BUY?':^5}   {'Details'}")
        print(f"{'':-<16} {'':-<12}   {'':-<5}   {'':-<30}")
        details = {
            "MA Crossover": (f"EMA{p['MA_SHORT']}={'>' if ema_short>ema_long else '<'}EMA{p['MA_LONG']}", f"EMA{p['MA_SHORT']}={ema_short:.2f}  EMA{p['MA_LONG']}={ema_long:.2f}"),
            "RSI": (f"{rsi:.1f}", f"< {rsi_mid:.0f} for BUY  |  > {rsi_mid:.0f} for SELL"),
            "Bollinger": (f"BB%={bb_pct:.2f}", f"BB lower={bb_lower:.2f}  BB upper={bb_upper:.2f}"),
            "MACD": (f"{'MACD>Sig' if macd>macd_sig else 'MACD<Sig'}", f"MACD={macd:.3f}  Signal={macd_sig:.3f}"),
            "ATR trend": (f"{'price>SMA' if price>sma_short else 'price<SMA'}", f"Price={price:.2f}  SMA{p['MA_SHORT']}={sma_short:.2f}"),
        }
        for ind, is_buy in conds_buy.items():
            val, det = details[ind]
            icon = "✔" if is_buy else "x"
            print(f"{ind:<16} {val:>12}   {icon}     {det}")
        print(f"{'─'*68}")
        print(f"BUY score: {buy_score}/5   SELL score: {sell_score}/5")
        print(f"BUY zone:      ${buy_zone_lo:>10.2f}  –  ${buy_zone_hi:.2f}  (BB lower – current price)")
        print(f"Stop-Loss:     ${stop_loss:>10.2f}           ({p['ATR_SL_MULT']}× ATR={atr:.2f} below price)")
        sl_pct = (price - stop_loss) / price * 100
        tp_pct = (take_profit - price) / price * 100
        st_pct = (sell_target - price) / price * 100
        print(f"                            ({sl_pct:.1f} % below current price)")
        print(f"Take Profit:   ${take_profit:>10.2f}           (+{tp_pct:.1f} %, R:R 1:2)")
        print(f"SELL target:   ${sell_target:>10.2f}           (+{st_pct:.1f} %, BB upper)")
    print()

# Main program
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output folder: {OUTPUT_DIR}/")
    print("MULTI-ASSET TRADING ALGORITHM BACKTEST")
    print(f"Period: {START_DATE}  →  {END_DATE}")
    print(f"Initial capital: ${INITIAL_CAP:,.0f} / asset")
    print(f"Indicators: MA Crossover, RSI, Bollinger Bands, MACD, ATR")
    print(f"Profiles: COMMODITY / FOREX_IDX / TECH / DEFENSIVE (per-asset)")
    results = []
    for name, ticker in ASSETS.items():
        print(f"\nDownloading data: {name} ({ticker}) ...")
        try:
            raw = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False, auto_adjust=True)
            if raw.empty:
                print(f"Insufficient data for {name}, skipping.")
                continue
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            profile_name = ASSET_PROFILES.get(name, "TECH")
            p = PROFILES[profile_name]
            min_bars = p["MA_LONG"] + 10
            if len(raw) < min_bars:
                print(f"Insufficient data for {name}, skipping.")
                continue
            raw = raw[raw["Close"].notna() & (raw["Close"] > 0)]  # drop NaN/zero closes upfront
            # Auto-detect currency and convert to USD if needed
            currency = detect_currency(ticker)
            if currency != "USD":
                raw = convert_to_usd(raw, currency, start=START_DATE, end=END_DATE)
                print(f"  [FX] {name} ({currency} → USD)")
            df = compute_indicators(raw.copy(), p)
            df = generate_signals(df, p)
            res = run_backtest(df, name, p)
            res["profile"] = profile_name
            results.append(res)
            print(f"  ✔  [{profile_name}]  return: {res['total_return']:+.1f}% "f"(B&H: {res['bh_return']:+.1f} %) "f"| Sharpe: {res['sharpe']:.2f} "f"| Win rate: {res['win_rate']:.0f} %")
            # Individual graph
            save_path = os.path.join(OUTPUT_DIR, f"chart_{name.lower()}.png")
            #print(f"  → Saving chart: {save_path}")
            plot_asset(res, save_path=save_path)
        except Exception as e:
            print(f"Error for {name}: {e}")
    if not results:
        print("\nFailed to download any data.")
        return
    # sort results by total return
    results = sorted(results, key=lambda r: r["total_return"]  if r["total_return"] == r["total_return"] else float("-inf"), reverse=True)    
    # Summary table and graphs
    print("\n"+"=" * 62)
    print("SUMMARY RESULTS")
    table = []
    for r in results:
        tr = r["trades_df"]; n_buy  = len(tr[tr["type"] == "BUY"]); n_sell = len(tr[tr["type"] == "SELL"]); n_stop = len(tr[tr["type"] == "STOP-LOSS"])
        table.append([r["asset"], r.get("profile", "-"), f"${r['final_value']:,.0f}", f"{r['total_return']:+.1f} %", f"{r['bh_return']:+.1f} %", f"{r['total_return'] - r['bh_return']:+.1f} %", f"{r['win_rate']:.0f} %", f"{r['sharpe']:.2f}", f"{r['max_drawdown']:.1f} %", f"{r['profit_factor']:.2f}",])
    headers = ["Asset", "Profile", "Final value", "Return", "B&H", "Alpha", "Win rate", "Sharpe", "Max Drawdown", "Profit Factor ",]
    print(tabulate(table, headers=headers, tablefmt="rounded_outline", stralign="right", numalign="right"))
    # Best asset
    best = max(results, key=lambda r: r["total_return"])
    print(f"\nBest asset: {best['asset']}" f"(return {best['total_return']:+.1f} %)")
    export_signals_png(results)
    export_table_png(table, headers, results)
    export_order_levels_png(results)    
    # Summary comparison graph
    plot_summary(results)
    print("\nDone! Charts are saved as PNG files.")

def analyze_asset(name: str, interval: str = "1d"):
    """
    Quick analysis of a single asset – downloads current data
    and shows status of all indicators + recommendation to terminal.
    Usage:
        python trading_backtest.py --analyze Gold
        python trading_backtest.py --analyze Gold --interval 1h
        python trading_backtest.py --analyze Gold --interval 4h
        python trading_backtest.py --analyze Bitcoin --interval 15m
    """
    # Interval Validation
    if interval not in INTERVAL_SETTINGS:
        print(f"\n Unknown interval '{interval}'.")
        print(f"   Available intervals: {', '.join(INTERVAL_SETTINGS.keys())}")
        return
    iv = INTERVAL_SETTINGS[interval]
    # Find the ticker
    ticker = ASSETS.get(name)
    if not ticker:
        for k, v in ASSETS.items():
            if k.lower() == name.lower():
                name, ticker = k, v
                break
    if not ticker:
        print(f"\n Asset '{name}' not found.")
        print(f"   Available assets: {', '.join(ASSETS.keys())}")
        return
    profile_name = ASSET_PROFILES.get(name, "TECH"); p = PROFILES[profile_name]
    ts = datetime.now().strftime("%d.%m.%Y  %H:%M:%S")
    print(f"QUICK ANALYSIS: {name} ({ticker})  [{iv['label']}]")
    print(f"{ts}")
    print(f"Downloading data  (interval={interval}, period={iv['period']})...")
    try:
        # 4h = resample from 1h (Yahoo Finance doesn't support 4h)
        if interval == "4h":
            raw = yf.download(ticker, period=iv["period"], interval="1h", progress=False, auto_adjust=True)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            raw = raw.resample("4h").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna()
        else:
            raw = yf.download(ticker, period=iv["period"], interval=interval, progress=False, auto_adjust=True)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
        if raw.empty:
            print(f"Failed to download data for {name}.")
            return
        # Auto-detect currency and convert to USD if needed
        currency = detect_currency(ticker)
        if currency != "USD":
            rate = get_fx_rate(currency)
            rate_val = float(rate) if isinstance(rate, float) else float(rate.dropna().iloc[-1]) if hasattr(rate, "dropna") else 1.0
            raw = convert_to_usd(raw, currency)
            print(f"  [FX] Converted from {currency} to USD  (rate ~{rate_val:.5f})")
        min_bars = max(p["MA_LONG"] + 5, 50)
        if len(raw) < min_bars:
            print(f"Insufficient data ({len(raw)} candles, need {min_bars}).")
            print(f"Try longer period or different interval.")
            return
    except Exception as e:
        print(f"Download error: {e}")
        return
    df = compute_indicators(raw.copy(), p); last = df.iloc[-1]; c = df["Close"].astype(float)
    price = float(c.iloc[-1])
    prev_close = float(c.iloc[-2])
    change = (price - prev_close) / prev_close * 100
    atr = float(last["ATR"]) if not pd.isna(last["ATR"]) else 0
    rsi = float(last["RSI"]) if not pd.isna(last["RSI"]) else 50
    bb_pct = float(last["BB_pct"]) if not pd.isna(last["BB_pct"]) else 0.5
    bb_upper = float(last["BB_upper"]) if not pd.isna(last["BB_upper"]) else price
    bb_lower = float(last["BB_lower"]) if not pd.isna(last["BB_lower"]) else price
    macd = float(last["MACD"]) if not pd.isna(last["MACD"]) else 0
    macd_sig = float(last["MACD_sig"]) if not pd.isna(last["MACD_sig"]) else 0
    macd_hist = float(last["MACD_hist"]) if not pd.isna(last["MACD_hist"])else 0
    ema_short = float(last["EMA_short"]) if not pd.isna(last["EMA_short"])else price
    ema_long = float(last["EMA_long"]) if not pd.isna(last["EMA_long"]) else price
    sma_short = float(last["SMA_short"]) if not pd.isna(last["SMA_short"])else price
    rsi_mid = (p["RSI_OB"] + p["RSI_OS"]) / 2
    # Indicator status
    conds = {"MA Crossover": ema_short > ema_long,"RSI": rsi < rsi_mid,"Bollinger": bb_pct < 0.4,"MACD": macd > macd_sig,"ATR trend": price > sma_short,}
    buy_score = sum(conds.values()); sell_score = sum(not v for v in conds.values())
    # Overall recommendation
    if buy_score >= 4: rec = "✔ STRONG BUY SIGNAL"
    elif buy_score == 3: rec = "✔ BUY SIGNAL"
    elif sell_score >= 4: rec = "x STRONG SELL SIGNAL"
    elif sell_score == 3: rec = "x SELL SIGNAL"
    else: rec = "o NEUTRAL – WAIT"
    # MACD histogram trend strength
    hist_trend = "strengthening ▲" if macd_hist > 0 else "weakening ▼"
    arrow = "▲" if change >= 0 else "▼"
    candle_label = iv["label"]
    print(f"Current price:  ${price:>12,.2f}  {arrow} {change:+.2f}% (previous candle)")
    print(f"Interval:       {candle_label}  (period: {iv['period']}, candles: {len(df)})")
    print(f"Profile:        {profile_name}")
    print(f"ATR (volatility): {atr:.2f}  ({atr/price*100:.1f}% per candle)")
    print()
    # Indicator table
    details = {
        "MA Crossover": (f"EMA{p['MA_SHORT']}={'>' if ema_short>ema_long else '<'}EMA{p['MA_LONG']}", f"EMA{p['MA_SHORT']}={ema_short:.2f}  EMA{p['MA_LONG']}={ema_long:.2f}"),
        "RSI": (f"{rsi:.1f}", f"{'Below' if rsi < rsi_mid else 'Above'} midpoint {rsi_mid:.0f}  |  OB={p['RSI_OB']} OS={p['RSI_OS']}"),
        "Bollinger": (f"BB%={bb_pct:.2f}", f"Low={bb_lower:.2f}  Mid={float(last['BB_mid']):.2f}  Up={bb_upper:.2f}"),
        "MACD": (f"{'▲' if macd>macd_sig else '▼'} hist={macd_hist:.3f}", f"MACD={macd:.3f}  Signal={macd_sig:.3f}  ({hist_trend})"),
        "ATR trend": (f"{'price>SMA' if price>sma_short else 'price<SMA'}", f"Price={price:.2f}  SMA{p['MA_SHORT']}={sma_short:.2f}"),
    }
    print(f"{'Indicator':<16} {'Value':>14}   {'BUY?':^5}   Detail")
    print(f"{'─'*16} {'─'*14}   {'─'*5}   {'─'*36}")
    for ind, is_buy in conds.items():
        val, det = details[ind]
        icon = "✔" if is_buy else "x"
        print(f"{ind:<16} {val:>14}   {icon}     {det}")
    print(f"BUY score: {buy_score}/5   SELL score: {sell_score}/5")
    print(f"  👉  {rec}")
    # Price levels
    buffer = 0.005
    buy_limit = bb_lower * (1 + buffer)
    stop_loss = buy_limit - p["ATR_SL_MULT"] * atr
    risk_per = buy_limit - stop_loss
    tp1 = buy_limit + risk_per
    tp2 = bb_upper
    risk_usd = risk_per * (INITIAL_CAP * 0.95 / buy_limit) if buy_limit > 0 else 0
    sl_pct = (price - stop_loss) / price * 100
    tp1_pct = (tp1 - price) / price * 100
    tp2_pct = (tp2 - price) / price * 100
    bl_pct_diff = (buy_limit - price)  / price * 100
    print(f"\nPRICE LEVELS:")
    print(f"Buy Limit:   ${buy_limit:>10,.2f}  ({bl_pct_diff:+.1f}% from current price)")
    print(f"Stop-Loss:   ${stop_loss:>10,.2f}  (-{sl_pct:.1f}%,  {p['ATR_SL_MULT']}× ATR)")
    print(f"Take Profit: ${tp1:>10,.2f}  (+{tp1_pct:.1f}%,  R:R 1:1)")
    print(f"SELL target: ${tp2:>10,.2f}  (+{tp2_pct:.1f}%,  BB upper)")
    print(f"Risk/trade:  ${risk_usd:>9,.0f}  (with capital ${INITIAL_CAP:,})")
    # Context – where we are in BB band
    print(f"\nCONTEXT:")
    if bb_pct < 0.2: bb_comment = "Very close to lower band – historically good buy zone"
    elif bb_pct < 0.4: bb_comment = "Close to lower band – slightly undervalued"
    elif bb_pct < 0.6: bb_comment = "Middle of band – neutral position"
    elif bb_pct < 0.8: bb_comment = "Close to upper band – slightly overvalued"
    else: bb_comment = "Very close to upper band – historically sell zone"
    
    if rsi < p["RSI_OS"]: rsi_comment = f"RSI oversold ({rsi:.0f}) – strong bounce possible"
    elif rsi > p["RSI_OB"]: rsi_comment = f"RSI overbought ({rsi:.0f}) – reversal possible"
    elif rsi < rsi_mid: rsi_comment = f"RSI ({rsi:.0f}) below midpoint – room to grow"
    else: rsi_comment = f"RSI ({rsi:.0f}) above midpoint – momentum weakening"
    print(f"Bollinger: {bb_comment}")
    print(f"RSI:       {rsi_comment}")
    print(f"MACD:      Histogram {hist_trend} – trend {'strengthening, hold position' if macd_hist > 0 else 'weakening, be cautious'}")
    print(f"\nSPEED & VOLUME (last 10 intervals):")
    close_s, high_s, low_s = df["Close"].astype(float), df["High"].astype(float), df["Low"].astype(float)
    vol_s = df["Volume"].astype(float) if "Volume" in df.columns else None
    # Rate of Change – how much % price moved in last 10 candles
    roc_period = min(10, len(df) - 1)
    roc = (close_s.iloc[-1] - close_s.iloc[-roc_period - 1]) / close_s.iloc[-roc_period - 1] * 100
    roc_arrow = "▲" if roc >= 0 else "▼"
    print(f"Rate of Change (ROC-{roc_period}):  {roc_arrow} {roc:+.2f}%  over last {roc_period} candles")
    # ATR trend – expanding or contracting vs 10 bars ago
    atr_now = float(df["ATR"].iloc[-1]) if not pd.isna(df["ATR"].iloc[-1])  else 0
    atr_prev = float(df["ATR"].iloc[-min(11, len(df))]) if not pd.isna(df["ATR"].iloc[-min(11, len(df))]) else atr_now
    atr_change = (atr_now - atr_prev) / atr_prev * 100 if atr_prev > 0 else 0
    if atr_change > 10: atr_trend_str = f"▲ EXPANDING  +{atr_change:.1f}%  – volatility increasing, momentum building"
    elif atr_change < -10: atr_trend_str = f"▼ CONTRACTING {atr_change:.1f}%  – volatility decreasing, move losing steam"
    else: atr_trend_str = f"→ STABLE  {atr_change:+.1f}%  – normal volatility"
    print(f"ATR trend (vs 10 bars ago):  {atr_trend_str}")
    # Candle body size – current vs 20-bar average
    body_now = abs(float(df["Close"].iloc[-1]) - float(df["Open"].iloc[-1])) if "Open" in df.columns else 0
    bodies = (df["Close"].astype(float) - df["Open"].astype(float)).abs() if "Open" in df.columns else pd.Series([0])
    body_avg = float(bodies.iloc[-20:].mean()) if len(bodies) >= 20 else float(bodies.mean())
    body_ratio = body_now / body_avg if body_avg > 0 else 1.0
    if body_ratio > 1.5: body_str = f"${body_now:.2f} ({body_ratio:.1f}x avg) – LARGE candle, strong conviction"
    elif body_ratio < 0.5: body_str = f"${body_now:.2f}  ({body_ratio:.1f}x avg) – small candle, low conviction / indecision"
    else: body_str = f"${body_now:.2f} ({body_ratio:.1f}x avg) – normal candle size"
    print(f"Candle body size:            {body_str}")
    # Volume analysis (skip for assets without volume data e.g. forex futures)
    if vol_s is not None and vol_s.iloc[-1] > 0 and vol_s.sum() > 0:
        vol_now = float(vol_s.iloc[-1])
        vol_avg = float(vol_s.iloc[-20:].mean()) if len(vol_s) >= 20 else float(vol_s.mean())
        vol_ratio = vol_now / vol_avg if vol_avg > 0 else 1.0
        vol_pct = (vol_ratio - 1) * 100
        if vol_ratio > 2.0:
            vol_str = f"SPIKE  {vol_pct:+.0f}% above avg – strong institutional interest"
        elif vol_ratio > 1.3:
            vol_str = f"ABOVE AVERAGE  {vol_pct:+.0f}% – confirms price move"
        elif vol_ratio < 0.7:
            vol_str = f"BELOW AVERAGE  {vol_pct:+.0f}% – weak conviction, be cautious"
        else:
            vol_str = f"AVERAGE  {vol_pct:+.0f}% – normal activity"
        print(f"Volume (current candle):     {vol_str}, Volume value {vol_now:,.0f}  (avg {vol_avg:,.0f})")
        # 5. Volume trend – last 5 candles increasing or decreasing?
        if len(vol_s) >= 6:
            vol_5 = vol_s.iloc[-5:].values
            vol_trend_slope = float(np.polyfit(range(len(vol_5)), vol_5, 1)[0])
            if vol_trend_slope > vol_avg * 0.02:
                vol_trend_str = "▲ increasing last 5 candles – momentum building"
            elif vol_trend_slope < -vol_avg * 0.02:
                vol_trend_str = "▼ decreasing last 5 candles – momentum fading"
            else:
                vol_trend_str = "→ flat last 5 candles"
            print(f"Volume trend (5 candles):    {vol_trend_str}")
        # 6. OBV direction – does volume confirm price direction?
        if len(vol_s) >= 10:
            obv = (np.sign(close_s.diff()) * vol_s).fillna(0).cumsum()
            obv_now = float(obv.iloc[-1])
            obv_prev = float(obv.iloc[-6])
            obv_dir = obv_now - obv_prev
            if obv_dir > 0 and change >= 0:
                obv_str = "▲ BULLISH – volume confirms price rise"
            elif obv_dir < 0 and change < 0:
                obv_str = "▼ BEARISH – volume confirms price drop"
            elif obv_dir > 0 and change < 0:
                obv_str = "⚠ DIVERGENCE – price down but OBV up (potential reversal up)"
            elif obv_dir < 0 and change >= 0:
                obv_str = "⚠ DIVERGENCE – price up but OBV down (potential reversal down)"
            else:
                obv_str = "→ neutral"
            print(f"OBV direction (5 candles):   {obv_str}")
    else:
        print(f"Volume:                      N/A (not available for this asset/interval)")
    # VOLUME PROFILE & MOMENTUM
    print(f"\nVOLUME PROFILE & MOMENTUM:")
    df_sig = generate_signals(df.copy(), p)
    vm = analyze_volume_momentum(df_sig)
    vp = volume_profile(df_sig, bins=35)
    if vm is None:
        print(f"Volume data not available for this asset/interval.")
    else:
        # Volume Profile key levels
        if vp:
            print(f"Point of Control (POC): ${vp['poc_price']:,.2f} (most traded price)")
            print(f"Value Area: ${vp['va_low']:,.2f}  –  ${vp['va_high']:,.2f}(70% of volume)")
            hvn_str = "  ".join([f"${h:,.2f}" for h in vp["hvn_prices"][:4]])
            lvn_str = "  ".join([f"${l:,.2f}" for l in vp["lvn_prices"][:4]])
            print(f"HVN (support/resistance): {hvn_str}")
            print(f"LVN (fast-pass zones): {lvn_str}")
            # Where is current price relative to POC
            poc_diff = (price - vp["poc_price"]) / vp["poc_price"] * 100
            if abs(poc_diff) < 1.0: poc_comment = "At POC – strong support/resistance zone"
            elif poc_diff > 0: poc_comment = f"Above POC by {poc_diff:+.1f}% – price above fair value"
            else:  poc_comment = f"Below POC by {poc_diff:+.1f}% – price below fair value"
            print(f"Current vs POC: {poc_comment}")
        # Volume quartile analysis
        print(f"\nNext-candle return by volume quartile:")
        print(f"{'Quartile':<16} {'Avg return':>10} {'Hit rate':>10} {'Median ret':>10}")
        print(f"{'─'*16} {'─'*10} {'─'*10} {'─'*10}")
        for i, (lbl, avg, hr, med) in enumerate(zip(vm["quartile_labels"], vm["quartile_avg_ret"], vm["quartile_hit_rate"], vm["quartile_med_ret"])):
            arrow = "▲" if avg > 0 else "▼"
            mark  = " ←" if i == 3 else ""
            print(f"{lbl:<16} {arrow} {avg:>+7.3f}%  {hr:>8.1f}%  {med:>+9.3f}%{mark}")
        print(f"Pearson corr (vol → return): {vm['correlation']:+.3f}  ", end="")
        if abs(vm["correlation"]) > 0.15:
            print(f"({'significant' if abs(vm['correlation']) > 0.25 else 'moderate'} relationship)")
        else:
            print("(weak relationship)")

if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    def _get_interval(args, default="1h"):
        if "--interval" in args:
            idx = args.index("--interval")
            if idx + 1 < len(args):
                return args[idx + 1]
        return default
    if args and args[0] == "--analyze":
        if len(args) < 2:
            print("\nUsage: python trading_backtest.py --analyze <Asset> [--interval <interval>]")
            print("Example:  python trading_backtest.py --analyze Gold --interval 1h")
            print(f"Intervals: {', '.join(INTERVAL_SETTINGS.keys())}")
        else:
            analyze_asset(args[1], interval=_get_interval(args, default="1d"))
    elif args and args[0] == "--signals-hourly":
        interval = _get_interval(args, default="1h")
        run_hourly_signals(interval=interval)
    else:
        main()
