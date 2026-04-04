"""
Trading Algorithm Dashboard – Streamlit
Installation:
    pip install streamlit plotly
Run:
    streamlit run dashboard.py
"""
import warnings
warnings.filterwarnings("ignore")
import sys
import os
import io
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import yfinance as yf
# Import backtest engine from the main script
sys.path.insert(0, os.path.dirname(__file__))
try:
    from trading_backtest_script import (
        ASSETS, PROFILES, ASSET_PROFILES, INTERVAL_SETTINGS, INITIAL_CAP, START_DATE, MC_PROFILE_META, 
        detect_currency, convert_to_usd, compute_indicators, generate_signals, run_backtest, monte_carlo_forecast, prophet_forecast, volume_profile, analyze_volume_momentum
    )
    SCRIPT_LOADED = True
except Exception as e:
    SCRIPT_LOADED = False
    LOAD_ERROR = str(e)

# Page config
st.set_page_config( page_title="Trading Dashboard", page_icon="📊", layout="wide", initial_sidebar_state="expanded",)
# Custom CSS
st.markdown("""
<style>
/*[data-testid="stSidebar"] { background: #0f1117; }
[data-testid="stSidebar"] * { color: #e0e0e0 !important; }*/
.metric-card {
    background: #1e2130;
    border-radius: 10px;
    padding: 14px 18px;
    border-left: 3px solid #378ADD;
    margin-bottom: 8px;
}
.metric-card .label { font-size: 11px; color: #888; text-transform: uppercase; letter-spacing: .05em; }
.metric-card .value { font-size: 22px; font-weight: 600; color: #fff; }
.metric-card .sub   { font-size: 11px; color: #888; margin-top: 2px; }
.signal-buy  { color: #2ecc71; font-weight: 700; }
.signal-sell { color: #e74c3c; font-weight: 700; }
.signal-neu  { color: #f39c12; font-weight: 700; }
.stDataFrame { border-radius: 8px; overflow: hidden; }
div[data-testid="stExpander"] { border: 1px solid #2a2d3e; border-radius: 8px; }
.section-title {
    font-size: 13px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: .08em;
    color: #888;
    margin: 1.2rem 0 .6rem;
    border-bottom: 1px solid #2a2d3e;
    padding-bottom: 4px;
}
</style>
""", unsafe_allow_html=True)

if not SCRIPT_LOADED:
    st.error(f"x Could not import trading_backtest_script.py: {LOAD_ERROR}")
    st.info("Make sure dashboard.py is in the same folder as trading_backtest_script.py")
    st.stop()

#  CACHED DATA FUNCTIONS
@st.cache_data(ttl=1800, show_spinner=False)
def load_signals(interval: str, capital: int = 10_000, convert_currencies: bool = True) -> list:
    """Download intraday data for all assets and compute signals."""
    iv = INTERVAL_SETTINGS.get(interval, INTERVAL_SETTINGS["1h"])
    results = []
    for name, ticker in ASSETS.items():
        profile_name = ASSET_PROFILES.get(name, "TECH"); p = PROFILES[profile_name]
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
            raw = raw[raw["Close"].notna() & (raw["Close"] > 0)]
            if convert_currencies:
                currency = detect_currency(ticker)
                if currency != "USD":
                    raw = convert_to_usd(raw, currency)
            min_bars = max(p["MA_LONG"] + 5, 50)
            if raw.empty or len(raw) < min_bars:
                continue
            df = compute_indicators(raw.copy(), p)
            df = generate_signals(df, p)
            last = df.iloc[-1]
            c = df["Close"].astype(float)
            price = float(c.iloc[-1])
            prev = float(c.iloc[-2]) if len(c) > 1 else price
            change = (price - prev) / prev * 100
            def safe(v, default=0.0):
                return float(v) if not pd.isna(v) else default
            atr = safe(last["ATR"])
            rsi = safe(last["RSI"], 50)
            bb_pct = safe(last["BB_pct"], 0.5)
            bb_upper = safe(last["BB_upper"], price)
            bb_lower = safe(last["BB_lower"], price)
            bb_mid = safe(last["BB_mid"], price)
            macd = safe(last["MACD"])
            macd_sig = safe(last["MACD_sig"])
            macd_hist = safe(last["MACD_hist"])
            ema_short = safe(last["EMA_short"], price)
            ema_long = safe(last["EMA_long"], price)
            sma_short = safe(last["SMA_short"], price)
            rsi_mid = (p["RSI_OB"] + p["RSI_OS"]) / 2

            conds_buy = {"MA": ema_short > ema_long, "RSI": rsi < rsi_mid, "BB": bb_pct < 0.4, "MACD": macd > macd_sig, "ATR": price > sma_short,}
            buy_score = sum(conds_buy.values())
            sell_score = sum(not v for v in conds_buy.values())
            if buy_score >= 3: signal = "BUY"
            elif sell_score >= 3: signal = "SELL"
            else: signal = "NEU"
            # Order levels
            buf = 0.005
            buy_limit = bb_lower * (1 + buf)
            stop_loss = buy_limit - p["ATR_SL_MULT"] * atr
            risk_per = buy_limit - stop_loss
            tp1 = buy_limit + risk_per
            qty_est = (capital * 0.95) / buy_limit if buy_limit > 0 else 0
            risk_usd = risk_per * qty_est
            # Speed & Volume
            close_s, high_s, low_s, vol_s = df["Close"].astype(float), df["High"].astype(float), df["Low"].astype(float), df["Volume"].astype(float) if "Volume" in df.columns else None
            roc_p = min(10, len(df) - 1)
            roc = (close_s.iloc[-1] - close_s.iloc[-roc_p-1]) / close_s.iloc[-roc_p-1] * 100

            atr_now = safe(df["ATR"].iloc[-1])
            atr_prev = safe(df["ATR"].iloc[-min(11, len(df))])
            atr_chg = (atr_now - atr_prev) / atr_prev * 100 if atr_prev > 0 else 0

            body_now = abs(float(df["Close"].iloc[-1]) - float(df["Open"].iloc[-1])) if "Open" in df.columns else 0
            bodies = (df["Close"].astype(float) - df["Open"].astype(float)).abs() if "Open" in df.columns else pd.Series([0])
            body_avg = float(bodies.iloc[-20:].mean()) if len(bodies) >= 20 else float(bodies.mean())
            body_ratio = body_now / body_avg if body_avg > 0 else 1.0

            vol_ratio = None
            vol_now = vol_avg = 0
            obv_signal = "N/A"
            if vol_s is not None and vol_s.sum() > 0:
                vol_now  = float(vol_s.iloc[-1])
                vol_avg  = float(vol_s.iloc[-20:].mean()) if len(vol_s) >= 20 else float(vol_s.mean())
                vol_ratio = vol_now / vol_avg if vol_avg > 0 else 1.0
                if len(vol_s) >= 10:
                    obv = (np.sign(close_s.diff()) * vol_s).fillna(0).cumsum()
                    obv_dir = float(obv.iloc[-1]) - float(obv.iloc[-6])
                    if obv_dir > 0 and change >= 0:
                        obv_signal = "BULLISH"
                    elif obv_dir < 0 and change < 0:
                        obv_signal = "BEARISH"
                    elif obv_dir > 0 and change < 0:
                        obv_signal = "DIVERGENCE ▲"
                    elif obv_dir < 0 and change >= 0:
                        obv_signal = "DIVERGENCE ▼"
            results.append({
                "asset": name, "ticker": ticker, "profile": profile_name,
                "price": price, "change": change,
                "signal": signal, "buy_score": buy_score, "sell_score": sell_score,
                "conds": conds_buy,
                "rsi": rsi, "rsi_mid": rsi_mid,
                "bb_pct": bb_pct, "bb_upper": bb_upper, "bb_lower": bb_lower, "bb_mid": bb_mid,
                "macd": macd, "macd_sig": macd_sig, "macd_hist": macd_hist,
                "ema_short": ema_short, "ema_long": ema_long, "sma_short": sma_short,
                "atr": atr, "p": p,
                "buy_limit": buy_limit, "stop_loss": stop_loss,
                "tp1": tp1, "bb_upper_target": bb_upper,
                "risk_usd": risk_usd,
                "roc": roc, "atr_chg": atr_chg, "body_ratio": body_ratio,
                "vol_ratio": vol_ratio, "vol_now": vol_now, "vol_avg": vol_avg,
                "obv_signal": obv_signal,
                "df": df,
            })
        except Exception:
            continue
    return results

@st.cache_data(ttl=7200, show_spinner=False)
def run_full_backtest(start_date: str = "2018-01-01", capital: int = 10_000, convert_currencies: bool = True) -> list:
    """Run full historical backtest for all assets."""
    results = []
    for name, ticker in ASSETS.items():
        profile_name = ASSET_PROFILES.get(name, "TECH"); p = PROFILES[profile_name]
        try:
            raw = yf.download(ticker, start=start_date, end=pd.Timestamp.today().strftime("%Y-%m-%d"), progress=False, auto_adjust=True)
            if raw.empty:
                continue
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            raw = raw[raw["Close"].notna() & (raw["Close"] > 0)]
            if len(raw) < p["MA_LONG"] + 10:
                continue
            if convert_currencies:
                currency = detect_currency(ticker)
                if currency != "USD":
                    raw = convert_to_usd(raw, currency, start=start_date, end=pd.Timestamp.today().strftime("%Y-%m-%d"))
            df = compute_indicators(raw.copy(), p)
            df = generate_signals(df, p); 
            # Pass capital into run_backtest via monkey-patch of global
            import trading_backtest_script as _tbs
            _orig_cap = _tbs.INITIAL_CAP
            _tbs.INITIAL_CAP = capital
            res = run_backtest(df, name, p)
            _tbs.INITIAL_CAP = _orig_cap
            res["profile"] = profile_name
            results.append(res)
        except Exception:
            continue
    return sorted(results, key=lambda r: r["total_return"] if r["total_return"] == r["total_return"] else float("-inf"), reverse=True)

# SIDEBAR
with st.sidebar:
    st.markdown("## 📊 Trading Dashboard")
    page = st.radio("Navigation", ["Signal Overview", "Asset Detail", "Order Levels", "Backtest Summary", "Comparison Charts",])
    st.markdown("**Signal settings**")
    interval = st.selectbox("Interval", ["1d", "4h", "1h", "30m", "15m", "5m"], index=0)
    st.markdown("**Backtest settings**")
    user_capital = st.number_input("Initial capital (USD)", min_value=100, max_value=10_000_000, value=10_000, step=1_000, help="Capital allocated per asset in backtest and risk calculations")
    col_d1 = st.columns(1)[0]
    with col_d1:
        user_start = st.date_input("Start date", value=pd.Timestamp("2018-01-01"), min_value=pd.Timestamp("2007-01-01"), max_value=pd.Timestamp.today() - pd.Timedelta(days=90),)
    user_start_str = user_start.strftime("%Y-%m-%d")   
    convert_fx = st.toggle("Convert to USD", value=True, help="Convert non-USD assets (EUR, CZK, GBP, DKK) to USD automatically")
    st.markdown("**Filter**")
    sig_filter = st.multiselect("Signal", ["BUY", "SELL", "NEU"], default=["BUY", "SELL", "NEU"])
    prof_filter = st.multiselect("Profile", ["TECH", "COMMODITY", "DEFENSIVE", "FOREX_IDX", "CRYPTO"], default=["TECH", "COMMODITY", "DEFENSIVE", "FOREX_IDX", "CRYPTO"])
    if st.button("Refresh signals", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.markdown("---")
    st.caption(f"Updated: {datetime.now().strftime('%d.%m.%Y %H:%M')}")
# LOAD DATA
with st.spinner(f"Loading signals ({interval})…"):
        all_signals = load_signals(interval,capital=user_capital, convert_currencies=convert_fx)
# Apply filters
signals = [s for s in all_signals if s["signal"] in sig_filter and s["profile"] in prof_filter]
# PAGE: SIGNAL OVERVIEW
if "Signal Overview" in page:
    st.title("Signal Overview")
    st.caption(f"Interval: **{interval}** · {len(all_signals)} assets · {datetime.now().strftime('%d.%m.%Y %H:%M')}")
    # Portfolio metrics
    buys, sells, neus = sum(1 for s in all_signals if s["signal"] == "BUY"), sum(1 for s in all_signals if s["signal"] == "SELL"), sum(1 for s in all_signals if s["signal"] == "NEU")
    avg_score = np.mean([s["buy_score"] for s in all_signals]) if all_signals else 0
    c1, c2, c3 = st.columns(3)
    c1.metric("Total assets", len(all_signals))
    c2.metric("BUY signals", buys, f"{buys/len(all_signals)*100:.0f}%" if all_signals else "")
    c3.metric("SELL signals", sells, f"{sells/len(all_signals)*100:.0f}%" if all_signals else "")
    # Signal distribution chart
    col_chart, col_prof = st.columns(2)
    with col_chart:
        st.markdown('<div class="section-title">Signal distribution</div>', unsafe_allow_html=True)
        fig = go.Figure(go.Bar(x=["BUY", "SELL", "NEU"], y=[buys, sells, neus], marker_color=["#2ecc71", "#e74c3c", "#f39c12"], text=[buys, sells, neus], textposition="outside",))
        fig.update_layout(height=220, margin=dict(t=10, b=10, l=10, r=10), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#ccc", showlegend=False, yaxis=dict(showgrid=False, visible=False), xaxis=dict(showgrid=False))
        st.plotly_chart(fig, use_container_width=True)
    with col_prof:
        st.markdown('<div class="section-title">By profile</div>', unsafe_allow_html=True)
        profiles = ["TECH", "COMMODITY", "DEFENSIVE", "FOREX_IDX", "CRYPTO"]
        fig2 = go.Figure()
        for sig, color in [("BUY","#2ecc71"), ("SELL","#e74c3c"), ("NEU","#f39c12")]:
            fig2.add_trace(go.Bar(name=sig, x=profiles, y=[sum(1 for s in all_signals if s["profile"]==p and s["signal"]==sig) for p in profiles], marker_color=color,))
        fig2.update_layout(barmode="stack", height=220, margin=dict(t=10, b=10, l=10, r=10), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#ccc", legend=dict(orientation="h", y=1.1), xaxis=dict(showgrid=False), yaxis=dict(showgrid=False))
        st.plotly_chart(fig2, use_container_width=True)
    # Signal table
    st.markdown('<div class="section-title">All signals</div>', unsafe_allow_html=True)
    if not signals:
        st.info("No assets match current filters.")
    else:
        rows = []
        for s in signals:
            def ic(v): return "✔" if v else "✕"
            price, bl, sl, tp1, tp2 = s["price"], s["buy_limit"], s["stop_loss"], s["tp1"], s["bb_upper_target"]
            rows.append({"Asset": s["asset"], "Profile": s["profile"], "Price": f"${s['price']:,.2f}", "Change": f"{s['change']:+.2f}%", "Signal": s["signal"], "BUY sc.": f"{s['buy_score']}/5", "MA": ic(s["conds"]["MA"]), "RSI": ic(s["conds"]["RSI"]), "BB": ic(s["conds"]["BB"]), "MACD": ic(s["conds"]["MACD"]), "ATR": ic(s["conds"]["ATR"]), "RSI val": f"{s['rsi']:.1f}", "BB%": f"{s['bb_pct']:.2f}",
                         "Buy zone": f"${s['bb_lower']:,.2f}", "Buy Limit": f"${bl:,.2f} ({(bl-price)/price*100:+.1f}%)", "Stop-Loss": f"${sl:,.2f} ({(sl-price)/price*100:+.1f}%)", "Take Profit": f"${tp1:,.2f} ({(tp1-price)/price*100:+.1f}%)", "SELL target": f"${tp2:,.2f} ({(tp2-price)/price*100:+.1f}%)", "Risk USD": f"${s['risk_usd']:,.0f}",})
        df_tbl = pd.DataFrame(rows)
        def color_signal(val):
            if val == "BUY": return "color: #2ecc71; font-weight: bold"
            if val == "SELL": return "color: #e74c3c; font-weight: bold"
            if val == "NEU": return "color: #f39c12; font-weight: bold"
            if val == "✔": return "color: #2ecc71"
            if val == "✕": return "color: #e74c3c"
            if val and val.startswith("+"):
                try: return "color: #2ecc71" if float(val.replace("%","")) >= 0 else "color: #e74c3c"
                except: pass
            return ""
        styled = df_tbl.style.applymap(color_signal, subset=["Signal","MA","RSI","BB","MACD","ATR","Change"])
        st.dataframe(styled, use_container_width=True, height="content", width="content", hide_index=True)
# PAGE: ASSET DETAIL
elif "Asset Detail" in page:
    st.title("Asset Detail")
    asset_names = [s["asset"] for s in all_signals]
    if not asset_names:
        st.warning("No data loaded. Try refreshing.")
        st.stop()
    selected = st.selectbox("Select asset", asset_names)
    s = next((x for x in all_signals if x["asset"] == selected), None)
    if not s:
        st.warning("Asset not found in current data.")
        st.stop()
    df, p = s["df"], s["p"]
    # Header 
    col_h1, col_h2, col_h3, col_h4, col_h5 = st.columns(5)
    col_h1.metric("Price", f"${s['price']:,.2f}", f"{s['change']:+.2f}%")
    col_h2.metric("Profile", s["profile"])
    col_h3.metric("Signal", s["signal"])
    col_h4.metric("BUY score", f"{s['buy_score']}/5")
    col_h5.metric("ATR", f"{s['atr']:.2f}", f"{s['atr']/s['price']*100:.1f}% per candle")
    bb_pct = s["bb_pct"]; rsi = s["rsi"]; p = s["p"]
    if bb_pct < 0.2: bb_ctx = "🟢 Near lower band – good buy zone"
    elif bb_pct < 0.4: bb_ctx = "🟡 Below mid – slightly undervalued"
    elif bb_pct < 0.6: bb_ctx = "⚪ Mid band – neutral"
    elif bb_pct < 0.8: bb_ctx = "🟠 Above mid – slightly overvalued"
    else: bb_ctx = "🔴 Near upper band – sell zone"
    if rsi < p["RSI_OS"]: rsi_ctx = f"🟢 Oversold ({rsi:.0f}) – bounce possible"
    elif rsi > p["RSI_OB"]: rsi_ctx = f"🔴 Overbought ({rsi:.0f}) – reversal possible"
    elif rsi < s["rsi_mid"]: rsi_ctx = f"🟡 ({rsi:.0f}) below mid – room to grow"
    else: rsi_ctx = f"🟠 ({rsi:.0f}) above mid – weakening"
    macd_hist = s["macd_hist"]
    macd_ctx  = f"{'🟢 Strengthening ▲' if macd_hist >= 0 else '🔴 Weakening ▼'}  hist={macd_hist:.3f}"

    c_bb, c_rsi, c_macd = st.columns(3)
    c_bb.info(f"**Bollinger:** {bb_ctx}")
    c_rsi.info(f"**RSI:** {rsi_ctx}")
    c_macd.info(f"**MACD:** {macd_ctx}")
    # Price levels cards 
    price, bl, sl_val, tp1, tp2, risk = s["price"], s["buy_limit"], s["stop_loss"], s["tp1"], s["bb_upper_target"], s["risk_usd"]
    risk_per = bl - sl_val
    rr2 = abs((tp2 - bl) / risk_per) if risk_per > 0 else 0

    st.markdown('<div class="section-title">Price levels</div>', unsafe_allow_html=True)
    lc1, lc2, lc3, lc4, lc5 = st.columns(5)
    lc1.metric("Buy Limit",   f"${bl:,.2f}",    f"{(bl-price)/price*100:+.1f}%")
    lc2.metric("Stop-Loss",   f"${sl_val:,.2f}", f"{(sl_val-price)/price*100:+.1f}%")
    lc3.metric("Take Profit", f"${tp1:,.2f}",   f"{(tp1-price)/price*100:+.1f}%  R:R 1:1")
    lc4.metric("SELL target", f"${tp2:,.2f}",   f"{(tp2-price)/price*100:+.1f}%  1:{rr2:.1f}")
    lc5.metric("Risk/trade",  f"${risk:,.0f}",  f"at ${INITIAL_CAP:,} capital")
    # Indicators + Speed & Volume 
    col_ind, col_sv = st.columns(2)
    with col_ind:
        st.markdown('<div class="section-title">Indicator status</div>', unsafe_allow_html=True)
        ind_data = [
            ("MA Crossover", f"EMA{p['MA_SHORT']} {'>' if s['ema_short']>s['ema_long'] else '<'} EMA{p['MA_LONG']}", s["conds"]["MA"], f"EMA{p['MA_SHORT']}={s['ema_short']:.2f}  EMA{p['MA_LONG']}={s['ema_long']:.2f}"),
            ("RSI", f"{s['rsi']:.1f}", s["conds"]["RSI"], f"{'Below' if s['rsi'] < s['rsi_mid'] else 'Above'} mid {s['rsi_mid']:.0f}  |  OB={p['RSI_OB']} OS={p['RSI_OS']}"),
            ("Bollinger", f"BB% = {s['bb_pct']:.2f}", s["conds"]["BB"], f"Low={s['bb_lower']:.2f}  Mid={s['bb_mid']:.2f}  Up={s['bb_upper']:.2f}"),
            ("MACD", f"hist = {s['macd_hist']:.3f}", s["conds"]["MACD"],f"MACD={s['macd']:.3f}  Signal={s['macd_sig']:.3f}  ({'strengthening ▲' if s['macd_hist']>=0 else 'weakening ▼'})"),
            ("ATR trend", f"{'price > SMA' if s['price'] > s['sma_short'] else 'price < SMA'}", s["conds"]["ATR"], f"Price={s['price']:.2f}  SMA{p['MA_SHORT']}={s['sma_short']:.2f}  ATR={s['atr']:.2f}"),
        ]
        def _buy_icon(ok):
            color = "#27ae60" if ok else "#e74c3c"
            icon  = "✔" if ok else "✕"
            return f'<span style="color:{color};font-weight:600">{icon} {"YES" if ok else "NO"}</span>'
        ind_html  = "<table style='width:100%;font-size:13px;border-collapse:collapse'>"
        ind_html += ("<tr>"
                     "<th style='text-align:left;color:#666;padding:5px 8px;border-bottom:1px solid #ddd'>Indicator</th>"
                     "<th style='text-align:left;color:#666;padding:5px 8px;border-bottom:1px solid #ddd'>Value</th>"
                     "<th style='text-align:center;color:#666;padding:5px 8px;border-bottom:1px solid #ddd'>BUY?</th>"
                     "<th style='text-align:left;color:#666;padding:5px 8px;border-bottom:1px solid #ddd'>Detail</th>"
                     "</tr>")
        for ind_name, val, ok, detail in ind_data:
            ind_html += (f"<tr>"
                         f"<td style='padding:6px 8px;border-bottom:1px solid #eee;font-weight:500;color:#333'>{ind_name}</td>"
                         f"<td style='padding:6px 8px;border-bottom:1px solid #eee;color:#333'>{val}</td>"
                         f"<td style='padding:6px 8px;border-bottom:1px solid #eee;text-align:center'>{_buy_icon(ok)}</td>"
                         f"<td style='padding:6px 8px;border-bottom:1px solid #eee;color:#666'>{detail}</td>"
                         f"</tr>")
        ind_html += "</table>"
        st.markdown(ind_html, unsafe_allow_html=True)
        # Score bar
        score_segs = "".join([f'<span style="display:inline-block;width:28px;height:8px;border-radius:4px;margin:0 2px;background:{"#27ae60" if i < s["buy_score"] else "#ddd"}"></span>' for i in range(5)])
        buy_col  = "#27ae60" if s["buy_score"] >= 3 else "#e67e22" if s["buy_score"] == 2 else "#e74c3c"
        sell_col = "#e74c3c" if s["sell_score"] >= 3 else "#e67e22" if s["sell_score"] == 2 else "#27ae60"
        st.markdown(
            f'<div style="margin-top:10px;font-size:13px">' 
            f'<span style="font-weight:600;color:{buy_col}">BUY {s["buy_score"]}/5</span>' 
            f' &nbsp; {score_segs} &nbsp; ' 
            f'<span style="font-weight:600;color:{sell_col}">SELL {s["sell_score"]}/5</span>' 
            f'</div>',
            unsafe_allow_html=True
        )
    with col_sv:
        st.markdown('<div class="section-title">Speed & Volume</div>', unsafe_allow_html=True) 
        def sv_color(val, good_condition):
            color = "#27ae60" if good_condition else "#e74c3c"
            return f'<span style="color:{color};font-weight:600">{val}</span>'
        roc_good = -3 < s["roc"] < 5
        atr_good = s["atr_chg"] > 5
        body_good = s["body_ratio"] > 1.0
        vol_good = (s["vol_ratio"] or 0) > 1.0
        obv_good = "BULLISH" in s["obv_signal"] or "DIVERGENCE ▲" in s["obv_signal"]
        roc_str = f"{'+' if s['roc']>=0 else ''}{s['roc']:.2f}%"
        atr_str = f"{'EXPANDING' if s['atr_chg']>10 else 'CONTRACTING' if s['atr_chg']<-10 else 'STABLE'} {s['atr_chg']:+.1f}%"
        body_str = f"{s['body_ratio']:.1f}x avg"
        vol_str = f"{((s['vol_ratio'] or 1)-1)*100:+.0f}% vs avg" if s["vol_ratio"] else "N/A"
        sv_data = [
            ("ROC-10", sv_color(roc_str,  roc_good), "% move last 10 candles"),
            ("ATR trend", sv_color(atr_str,  atr_good), "vs 10 bars ago"),
            ("Candle body", sv_color(body_str, body_good), "vs 20-bar average"),
            ("Volume", sv_color(vol_str,  vol_good), "vs 20-bar average"),
            ("OBV", sv_color(s["obv_signal"], obv_good), "5-candle direction"),
        ]
        sv_html = "<table style='width:100%;font-size:13px;border-collapse:collapse'>"
        sv_html += ("<tr>"
                     "<th style='text-align:left;color:#666;padding:5px 8px;border-bottom:1px solid #ddd'>Metric</th>"
                     "<th style='text-align:left;color:#666;padding:5px 8px;border-bottom:1px solid #ddd'>Value</th>"
                     "<th style='text-align:left;color:#666;padding:5px 8px;border-bottom:1px solid #ddd'>Description</th>"
                     "</tr>")
        for metric, val_html, desc in sv_data:
            sv_html += (f"<tr>"
                         f"<td style='padding:6px 8px;border-bottom:1px solid #eee;font-weight:500;color:#333'>{metric}</td>"
                         f"<td style='padding:6px 8px;border-bottom:1px solid #eee'>{val_html}</td>"
                         f"<td style='padding:6px 8px;border-bottom:1px solid #eee;color:#666'>{desc}</td>"
                         f"</tr>")
        sv_html += "</table>"
        st.markdown(sv_html, unsafe_allow_html=True)
    # Price chart with indicators
    st.markdown('<div class="section-title">Price chart</div>', unsafe_allow_html=True)
    if interval == "1m":
        _xfmt, _nticks, _dtick, angle = "%H:%M", 25, 5 * 60 * 1000, 25
    elif interval in ("5m", "15m"):
        _xfmt, _nticks, _dtick, angle = "%d %b %H:%M", 25, None, 25
    elif interval == "30m":
        _xfmt, _nticks, _dtick, angle = "%d %b %H:%M", 25, None, 25
    elif interval in ("1h", "4h"):
        _xfmt, _nticks, _dtick, angle = "%d %b", 25, None, 25
    else:                                
        _xfmt, _nticks, _dtick, angle = "%d %b %Y", 15, None, 0
    def _xaxis_cfg(show_labels=True):
        cfg = dict(showticklabels=show_labels, tickangle=angle, tickfont=dict(size=10), nticks=_nticks,)
        if _xfmt: cfg["tickformat"] = _xfmt
        if _dtick: cfg["dtick"] = _dtick
        return cfg
    close = df["Close"].astype(float)
    vol = df["Volume"].astype(float) if "Volume" in df.columns else None
    vol_avg20 = vol.rolling(20).mean() if vol is not None else None
    fig_pv = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.02)
    # Price + MA + BB
    fig_pv.add_trace(go.Scatter(x=df.index, y=close, name="Price", line=dict(color="#1a1a2e", width=1.6)), row=1, col=1)
    fig_pv.add_trace(go.Scatter(x=df.index, y=df["EMA_short"], name=f"EMA{p['MA_SHORT']}", line=dict(color="#f39c12", width=1, dash="dot")), row=1, col=1)
    fig_pv.add_trace(go.Scatter(x=df.index, y=df["EMA_long"], name=f"EMA{p['MA_LONG']}", line=dict(color="#3498db", width=1, dash="dot")), row=1, col=1)
    fig_pv.add_trace(go.Scatter(x=df.index, y=df["BB_upper"], name="BB Upper", line=dict(color="#4fc3f7", width=0.8, dash="dash")), row=1, col=1)
    fig_pv.add_trace(go.Scatter(x=df.index, y=df["BB_lower"], name="BB Lower", line=dict(color="#4fc3f7", width=0.8, dash="dash"), fill="tonexty", fillcolor="rgba(79,195,247,0.05)"), row=1, col=1)
    # BUY/SELL markers
    if not s["df"]["signal"].isna().all():
        buys_idx  = df[df["signal"] ==  1].index; sells_idx = df[df["signal"] == -1].index
        if len(buys_idx):
            fig_pv.add_trace(go.Scatter(x=buys_idx, y=close[buys_idx], mode="markers", name="BUY", marker=dict(symbol="triangle-up", size=7, color="#2ecc71")), row=1, col=1)
        if len(sells_idx):
            fig_pv.add_trace(go.Scatter(x=sells_idx, y=close[sells_idx], mode="markers", name="SELL", marker=dict(symbol="triangle-down", size=7, color="#e74c3c")), row=1, col=1) 
    # Price levels – horizontal lines
    for level, color, label in [(s["buy_limit"], "#2ecc71", "Buy Limit"), (s["stop_loss"], "#e74c3c", "Stop-Loss"), (s["tp1"], "#3498db", "TP1"),]:
        fig_pv.add_hline(y=level, line_dash="dot", line_color=color, annotation_text=f" {label} ${level:,.2f}", annotation_position="right", row=1, col=1)
    # Volume bars – colour by high/low vs 20-bar avg
    if vol is not None:
        vol_avg_last = float(vol_avg20.dropna().iloc[-1]) if vol_avg20 is not None else 0
        vol_colors = []
        for i, vv in enumerate(vol.values):
            avg = float(vol_avg20.iloc[i]) if vol_avg20 is not None and not pd.isna(vol_avg20.iloc[i]) else vol_avg_last
            if vv > avg * 1.5: vol_colors.append("#e74c3c")   # spike – red
            elif vv > avg * 1.0: vol_colors.append("#27ae60")   # above avg – green
            else: vol_colors.append("#b0bec5")   # below avg – grey
        fig_pv.add_trace(go.Bar(x=df.index, y=vol, name="Volume", marker_color=vol_colors, opacity=1), row=2, col=1)
        if vol_avg20 is not None:
            fig_pv.add_trace(go.Scatter(x=df.index, y=vol_avg20, name="Vol avg 20", line=dict(color="#b0bec5", width=1, dash="dot"), showlegend=True), row=2, col=1)
    fig_pv.update_layout(height=520, template="plotly_white", plot_bgcolor="#f8f9fa", paper_bgcolor="#ffffff", margin=dict(t=10, b=60, l=70, r=140), legend=dict(orientation="h", y=1.02, font_size=10), xaxis=_xaxis_cfg(show_labels=False), xaxis2=_xaxis_cfg(show_labels=True), barmode="overlay",)
    fig_pv.update_yaxes(row=1, col=1, title_text="Price (USD)")
    fig_pv.update_yaxes(range=[0, max(vol) * 1.1], row=2, col=1, title_text="Volume", tickformat=".2s")
    st.plotly_chart(fig_pv, use_container_width=True)
    # Technical indicators chart (RSI / MACD / ATR)
    st.markdown('<div class="section-title">Technical indicators</div>', unsafe_allow_html=True)
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.4, 0.35, 0.25], vertical_spacing=0.03)
    # RSI
    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI", line=dict(color="#e67e22", width=1.4)), row=1, col=1)
    fig.add_hline(y=p["RSI_OB"], line_color="#c0392b", line_dash="dash", row=1, col=1)
    fig.add_hline(y=p["RSI_OS"], line_color="#27ae60", line_dash="dash", row=1, col=1)
    fig.add_hline(y=50, line_color="#95a5a6", line_dash="dot", row=1, col=1)
    # MACD
    macd_colors = ["#27ae60" if v >= 0 else "#e74c3c" for v in df["MACD_hist"].fillna(0)]
    fig.add_trace(go.Bar(x=df.index, y=df["MACD_hist"], name="Histogram", marker_color=macd_colors, opacity=0.7), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD", line=dict(color="#378ADD", width=1.2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD_sig"], name="Signal", line=dict(color="#e74c3c", width=1.2)), row=2, col=1)
    # ATR
    fig.add_trace(go.Scatter(x=df.index, y=df["ATR"], name="ATR", line=dict(color="#9b59b6", width=1), fill="tozeroy", fillcolor="rgba(155,89,182,0.08)"), row=3, col=1)
    fig.update_layout(height=440, template="plotly_white", plot_bgcolor="#f8f9fa", paper_bgcolor="#ffffff", margin=dict(t=10, b=20, l=70, r=20), legend=dict(orientation="h", y=1.02, font_size=10), xaxis3=dict(showticklabels=True),)
    fig.update_yaxes(row=1, col=1, title_text="RSI", range=[0, 100])
    fig.update_yaxes(row=2, col=1, title_text="MACD")
    fig.update_yaxes(row=3, col=1, title_text="ATR")
    st.plotly_chart(fig, use_container_width=True)
    # Volume Profile & Momentum 
    st.markdown('<div class="section-title">Volume profile & momentum</div>', unsafe_allow_html=True)
    try:
        df_sig = generate_signals(df.copy(), p)
        vp = volume_profile(df_sig, bins=35)
        vm = analyze_volume_momentum(df_sig)
        col_vp, col_vm, col_bo = st.columns([1.2, 1, 1])
        with col_vp:
            st.caption("Volume profile – price vs volume")
            if vp:
                import plotly.graph_objects as go_vp
                bin_prices, bin_vols, poc_price, va_low, va_high, poc_bin, total_vol = vp["bin_prices"], vp["bin_vols"], vp["poc_price"], vp["va_low"], vp["va_high"], vp["poc_bin"], vp["total_vol"]
                # Color each bar
                bar_colors = []
                import numpy as _np
                thresh_hvn = _np.percentile([b for b in bin_vols if b > 0], 75)
                for i, (bp, bv) in enumerate(zip(bin_prices, bin_vols)):
                    if i == poc_bin: bar_colors.append("#E24B4A")
                    elif va_low <= bp <= va_high: bar_colors.append("#378ADD")
                    elif bv >= thresh_hvn: bar_colors.append("#85B7EB")
                    else: bar_colors.append("#D3D1C7")
                fig_vp = go.Figure(go.Bar(x=[round(v/1e6, 2) if v >= 1e6 else round(v/1e3, 1) for v in bin_vols], y=[round(p, 2) for p in bin_prices], orientation="h", marker_color=bar_colors, hovertemplate="%{y:.2f} | %{x:.2f}M vol<extra></extra>",))
                fig_vp.add_hline(y=poc_price, line_color="#E24B4A", line_dash="dash", annotation_text=f" POC ${poc_price:,.2f}", annotation_font_color="#E24B4A", annotation_font_size=10)
                fig_vp.add_hrect(y0=va_low, y1=va_high, fillcolor="#378ADD", opacity=0.06, line_width=0)
                fig_vp.update_layout(height=320, template="plotly_white", plot_bgcolor="#f8f9fa", paper_bgcolor="#ffffff", margin=dict(t=10, b=30, l=70, r=20), showlegend=False, xaxis=dict(title="Volume", ticksuffix="M", tickfont=dict(size=10)), yaxis=dict(title="Price", tickformat="$.2f", tickfont=dict(size=10)),)
                st.plotly_chart(fig_vp, use_container_width=True) 
                # Key levels table
                poc_diff = (s["price"] - poc_price) / poc_price * 100
                st.markdown(f"""
                    <table style='width:100%;font-size:12px;border-collapse:collapse'>
                    <tr><td style='color:#666;padding:4px 6px;border-bottom:1px solid #eee'>POC</td>
                        <td style='font-weight:500;padding:4px 6px;border-bottom:1px solid #eee;color:#E24B4A'>${poc_price:,.2f}</td>
                        <td style='color:#666;padding:4px 6px;border-bottom:1px solid #eee'>{poc_diff:+.1f}% from price</td></tr>
                    <tr><td style='color:#666;padding:4px 6px;border-bottom:1px solid #eee'>Value Area</td>
                        <td style='font-weight:500;padding:4px 6px;border-bottom:1px solid #eee;color:#378ADD'>${va_low:,.2f}–${va_high:,.2f}</td>
                        <td style='color:#666;padding:4px 6px;border-bottom:1px solid #eee'>70% of volume</td></tr>
                    </table>
                """, unsafe_allow_html=True)
        with col_vm:
            st.caption("Next-candle return by volume quartile")
            if vm:
                fig_q = go.Figure()
                colors_q = ["#27ae60" if v >= 0 else "#E24B4A" for v in vm["quartile_avg_ret"]]
                fig_q.add_trace(go.Bar(name="Avg return", x=vm["quartile_labels"], y=[round(v, 3) for v in vm["quartile_avg_ret"]], marker_color=colors_q, yaxis="y", text=[f"{v:+.3f}%" for v in vm["quartile_avg_ret"]], textposition="outside", textfont=dict(size=10),))
                fig_q.add_trace(go.Scatter(name="Hit rate", x=vm["quartile_labels"], y=[round(v, 1) for v in vm["quartile_hit_rate"]], mode="lines+markers", line=dict(color="#378ADD", width=2), marker=dict(size=7), yaxis="y2",))
                fig_q.update_layout(height=280, template="plotly_white", plot_bgcolor="#f8f9fa", paper_bgcolor="#ffffff", margin=dict(t=10, b=40, l=50, r=50), legend=dict(orientation="h", y=1.08, font_size=10), yaxis=dict(title="Avg return %", ticksuffix="%", tickfont=dict(size=10), zeroline=True, zerolinecolor="#ccc"), yaxis2=dict(title="Hit rate %", overlaying="y", side="right", range=[40, 75], ticksuffix="%", tickfont=dict(size=10)), xaxis=dict(tickfont=dict(size=10)), barmode="group", showlegend=True,)
                st.plotly_chart(fig_q, use_container_width=True)
                # Correlation badge
                corr = vm["correlation"]
                if abs(corr) > 0.25: st.success(f"Strong vol-return correlation: {corr:+.3f}")
                elif abs(corr) > 0.10: st.info(f"Moderate vol-return correlation: {corr:+.3f}")
                else: st.caption(f"Weak vol-return correlation: {corr:+.3f}")
        with col_bo:
            st.caption("Breakout quality – volume vs avg")
            if vm and vm["breakout_success_vol_ratio"] and vm["breakout_fail_vol_ratio"]:
                fig_bo = go.Figure(go.Bar(x=["Successful BUY", "Failed BUY"], y=[round(vm["breakout_success_vol_ratio"], 2), round(vm["breakout_fail_vol_ratio"], 2)], marker_color=["#27ae60", "#E24B4A"], text=[f"{vm['breakout_success_vol_ratio']:.2f}×",f"{vm['breakout_fail_vol_ratio']:.2f}×"], textposition="outside", textfont=dict(size=11),))
                fig_bo.add_hline(y=1.0, line_color="#aaa", line_dash="dot", annotation_text=" avg volume", annotation_font_color="#666")
                fig_bo.update_layout(height=280, template="plotly_white", plot_bgcolor="#f8f9fa", paper_bgcolor="#ffffff", margin=dict(t=10, b=40, l=50, r=20), showlegend=False, yaxis=dict(title="Volume / 20-bar avg", ticksuffix="×", tickfont=dict(size=10)), xaxis=dict(tickfont=dict(size=11)),)
                st.plotly_chart(fig_bo, use_container_width=True)
                ratio = vm["breakout_success_vol_ratio"] / max(vm["breakout_fail_vol_ratio"], 0.01)
                if ratio > 1.25: st.success(f"Volume confirms breakouts: success {ratio:.1f}× higher than fails")
                else: st.caption("Volume does not clearly distinguish success from failure")
                st.caption(f"n={vm['n_success']} success  /  {vm['n_fail']} fail")
            else:
                st.caption("Not enough BUY signals to analyse breakout volume.")
        # Summary line
        if vm:
            st.info(f"**Volume summary:** {vm['signal_vol_summary']}")
    except Exception as e:
        st.caption(f"Volume analysis unavailable: {e}")
    # Last 90 bars + interval-aware MC & Prophet forecast
    st.markdown('<div class="section-title">Last 90 bars + forecast (30 bars ahead)</div>', unsafe_allow_html=True)
    _INTERVAL_DELTA = {"1m": pd.Timedelta(minutes=1), "5m": pd.Timedelta(minutes=5), "15m": pd.Timedelta(minutes=15), "30m": pd.Timedelta(minutes=30), "1h": pd.Timedelta(hours=1), "4h": pd.Timedelta(hours=4), "1d": pd.Timedelta(days=1),}
    _bar_delta = _INTERVAL_DELTA.get(interval, pd.Timedelta(days=1))
    _n_forecast = 30
    _is_daily = _bar_delta >= pd.Timedelta(days=1)
    _df90 = df.iloc[-90:] if len(df) >= 90 else df
    _c90 = _df90["Close"].astype(float)
    _last_ts = _df90.index[-1]
    # Generate future timestamps at correct frequency
    def _future_ts(n):
        dates, ts = [], _last_ts + _bar_delta
        while len(dates) < n:
            if ts.weekday() < 5:   # skip weekends
                dates.append(ts)
            ts += _bar_delta
        return pd.DatetimeIndex(dates)
    if _is_daily: _fut_idx = pd.bdate_range(start=_last_ts + pd.Timedelta(days=1), periods=_n_forecast)
    else: _fut_idx = _future_ts(_n_forecast)
    # Build subplots: price on row1, volume on row2
    _has_vol = "Volume" in _df90.columns and _df90["Volume"].sum() > 0
    _row_h = [0.72, 0.28] if _has_vol else [1.0]
    _n_rows = 2 if _has_vol else 1
    fig_z6 = make_subplots(rows=_n_rows, cols=1, shared_xaxes=True, row_heights=_row_h, vertical_spacing=0.02,)
    # Price line
    fig_z6.add_trace(go.Scatter(x=_df90.index, y=_c90, name="Price", line=dict(color="#185FA5", width=2)), row=1, col=1)
    fig_z6.add_trace(go.Scatter(x=_df90.index, y=_df90["EMA_short"], name=f"EMA{p['MA_SHORT']}", line=dict(color="#e67e22", width=1, dash="dot")), row=1, col=1)
    fig_z6.add_trace(go.Scatter(x=_df90.index, y=_df90["EMA_long"], name=f"EMA{p['MA_LONG']}", line=dict(color="#8e44ad", width=1, dash="dot")), row=1, col=1)
    fig_z6.add_trace(go.Scatter(x=_df90.index, y=_df90["BB_upper"], line=dict(color="#4fc3f7", width=0.7, dash="dash"), showlegend=False), row=1, col=1)
    fig_z6.add_trace(go.Scatter(x=_df90.index, y=_df90["BB_lower"], name="BB Band", line=dict(color="#4fc3f7", width=0.7, dash="dash"), fill="tonexty", fillcolor="rgba(79,195,247,0.06)"), row=1, col=1) 
    # BUY/SELL markers
    if "signal" in _df90.columns:
        _b = _df90[_df90["signal"] ==  1]
        _se = _df90[_df90["signal"] == -1]
        if len(_b):
            fig_z6.add_trace(go.Scatter(x=_b.index, y=_c90[_b.index], mode="markers", name="BUY", marker=dict(symbol="triangle-up", size=9, color="#27ae60")), row=1, col=1)
        if len(_se):
            fig_z6.add_trace(go.Scatter(x=_se.index, y=_c90[_se.index], mode="markers", name="SELL", marker=dict(symbol="triangle-down", size=9, color="#e74c3c")), row=1, col=1)
    # Monte Carlo – interval-aware future dates
    try:
        mc_z = monte_carlo_forecast(_c90, profile=s["profile"], n_days=_n_forecast, n_sim=500)
        # Override dates with our interval-correct future index
        mc_dates = list(_fut_idx)
        _n_use = min(len(mc_dates), len(mc_z["p50"]))
        mc_dates = mc_dates[:_n_use]
        fig_z6.add_trace(go.Scatter(x=mc_dates + mc_dates[::-1], y=list(mc_z["p10"][:_n_use]) + list(mc_z["p90"][:_n_use])[::-1], fill="toself", fillcolor="rgba(55,138,221,0.08)", line=dict(color="rgba(0,0,0,0)"), name="MC 10–90 %"), row=1, col=1)
        fig_z6.add_trace(go.Scatter(x=mc_dates + mc_dates[::-1], y=list(mc_z["p25"][:_n_use]) + list(mc_z["p75"][:_n_use])[::-1], fill="toself", fillcolor="rgba(55,138,221,0.20)", line=dict(color="rgba(0,0,0,0)"), name="MC 25–75 %"), row=1, col=1)
        fig_z6.add_trace(go.Scatter(x=mc_dates, y=mc_z["p50"][:_n_use], name="MC median", line=dict(color="#378ADD", width=2, dash="dash")), row=1, col=1)
        # Annotation at end of MC median
        fig_z6.add_annotation(x=mc_dates[-1], y=float(mc_z["p50"][_n_use-1]), text=f" ${float(mc_z["p50"][_n_use-1]):,.0f}", showarrow=False, font=dict(color="#378ADD", size=10), xanchor="left")
    except Exception:
        pass
    # Prophet – interval-aware future dates
    try:
        pf_z = prophet_forecast(close, n_days=_n_forecast)
        if pf_z:
            fig_z6.add_trace(go.Scatter(x=list(pf_z["dates"]) + list(pf_z["dates"])[::-1], y=list(pf_z["yhat_lower"]) + list(pf_z["yhat_upper"])[::-1], fill="toself", fillcolor="rgba(230,126,34,0.15)", line=dict(color="rgba(0,0,0,0)"), name="80% CI"))
            fig_z6.add_trace(go.Scatter(x=pf_z["dates"], y=pf_z["yhat"], name="Forecast (yhat)", line=dict(color="#e67e22", width=2)))
            fig_z6.add_trace(go.Scatter(x=pf_z["dates"], y=pf_z["trend"], name="Trend", line=dict(color="#e67e22", width=1.2, dash="dash"), opacity=0.6))
            #fig_z6.add_hline(y=pf_z["last"], line_color="#aaa", line_dash="dot", annotation_text=f" Current ${pf_z['last']:,.2f}", annotation_font_color="#666")    
            fig_z6.add_annotation(x=pf_z["ds"][-1], y=float(pf_z["yhat"]), text=f" ${float(pf_z["yhat"]):,.0f}", showarrow=False, font=dict(color="#e67e22", size=10), xanchor="left")
    except Exception:
        pass
    # Vertical separator: history | forecast
    fig_z6.add_vline(x=_last_ts, line_color="#888", line_dash="dot", line_width=1.2, row=1, col=1)
    # Volume bars
    if _has_vol:
        _vol = _df90["Volume"].astype(float)
        _va20 = _vol.rolling(20).mean()
        _va_last = float(_va20.dropna().iloc[-1]) if len(_va20.dropna()) else 1
        _vcol = []
        for i, vv in enumerate(_vol.values):
            avg = float(_va20.iloc[i]) if not pd.isna(_va20.iloc[i]) else _va_last
            _vcol.append("#e74c3c" if vv > avg * 1.5 else "#27ae60" if vv > avg else "#b0bec5")
        fig_z6.add_trace(go.Bar(x=_df90.index, y=_vol, name="Volume", marker_color=_vcol, opacity=0.85), row=2, col=1)
        fig_z6.add_trace(go.Scatter(x=_df90.index, y=_va20, name="Vol avg 20", line=dict(color="#888", width=1, dash="dot")), row=2, col=1)
    # Layout
    _interval_lbl = {"1m":"1-min","5m":"5-min","15m":"15-min","30m":"30-min", "1h":"1-hour","4h":"4-hour","1d":"daily"}.get(interval,"bar")
    fig_z6.update_layout(height=480 if _has_vol else 380, template="plotly_white", plot_bgcolor="#f8f9fa", paper_bgcolor="#ffffff", margin=dict(t=10, b=60, l=70, r=80), legend=dict(orientation="h", y=1.02, font_size=10), title=dict(text=f"Last 90 {_interval_lbl} bars + 30 {_interval_lbl} forecast", font_size=12, x=0), xaxis=_xaxis_cfg(show_labels=not _has_vol),)
    if _has_vol:
        fig_z6.update_layout(xaxis2=_xaxis_cfg(show_labels=True))
        fig_z6.update_yaxes(row=2, col=1, title_text="Volume", tickformat=".2s")
    fig_z6.update_yaxes(row=1, col=1, title_text="Price (USD)")
    fig_z6.update_xaxes(range=[_df90.index.min(), max(pf_z["dates"])])
    st.plotly_chart(fig_z6, use_container_width=True)
    # Separate MC / Prophet detail panels
    st.markdown('<div class="section-title">Forecast detail – Monte Carlo vs Prophet</div>', unsafe_allow_html=True)
    close_s = df["Close"].astype(float)
    col_mc, col_pr = st.columns(2)
    with col_mc:
        try:
            mc = monte_carlo_forecast(close_s, profile=s["profile"], n_days=30, n_sim=500)
            meta = MC_PROFILE_META.get(s["profile"], {"color": "#378ADD", "label": "Random Walk", "short": "RW"})
            mc_med_last = float(mc["p50"][-1])
            mc_chg = (mc_med_last - mc["last"]) / mc["last"] * 100
            fig_mc = go.Figure()
            fig_mc.add_trace(go.Scatter(x=list(mc["dates"]) + list(mc["dates"])[::-1], y=list(mc["p10"]) + list(mc["p90"])[::-1], fill="toself", fillcolor="rgba(55,138,221,0.10)", line=dict(color="rgba(0,0,0,0)"), name="10–90 %"))
            fig_mc.add_trace(go.Scatter(x=list(mc["dates"]) + list(mc["dates"])[::-1], y=list(mc["p25"]) + list(mc["p75"])[::-1], fill="toself", fillcolor="rgba(55,138,221,0.22)", line=dict(color="rgba(0,0,0,0)"), name="25–75 %"))
            fig_mc.add_trace(go.Scatter(x=mc["dates"], y=mc["p50"], name=f"Median ({meta['short']})", line=dict(color="#378ADD", width=2, dash="dash")))
            fig_mc.add_hline(y=mc["last"], line_color="#aaa", line_dash="dot", annotation_text=f" Current ${mc['last']:,.2f}", annotation_font_color="#666")
            fig_mc.update_layout(height=300, template="plotly_white", plot_bgcolor="#f8f9fa", paper_bgcolor="#ffffff", margin=dict(t=30, b=30, l=60, r=80), legend=dict(orientation="h", y=1.08, font_size=10), title=dict(text=f"Monte Carlo – {meta['label']}  |  30d: ${mc_med_last:,.2f} ({mc_chg:+.1f}%)", font_size=11, x=0),)
            st.plotly_chart(fig_mc, use_container_width=True)
        except Exception as e:
            st.caption(f"Monte Carlo unavailable: {e}")
    with col_pr:
        try:
            pf = prophet_forecast(close_s, n_days=30)
            if pf is None: st.info("Prophet not installed.\n\n```\npip install prophet\n```")
            else:
                pr_med = float(pf["yhat"][-1])
                pr_chg = (pr_med - pf["last"]) / pf["last"] * 100
                fig_pr = go.Figure()
                fig_pr.add_trace(go.Scatter(x=list(pf["dates"]) + list(pf["dates"])[::-1], y=list(pf["yhat_lower"]) + list(pf["yhat_upper"])[::-1], fill="toself", fillcolor="rgba(230,126,34,0.15)", line=dict(color="rgba(0,0,0,0)"), name="80% CI"))
                fig_pr.add_trace(go.Scatter(x=pf["dates"], y=pf["yhat"], name="Forecast (yhat)", line=dict(color="#e67e22", width=2)))
                fig_pr.add_trace(go.Scatter(x=pf["dates"], y=pf["trend"], name="Trend", line=dict(color="#e67e22", width=1.2, dash="dash"), opacity=0.6))
                fig_pr.add_hline(y=pf["last"], line_color="#aaa", line_dash="dot", annotation_text=f" Current ${pf['last']:,.2f}", annotation_font_color="#666")
                fig_pr.update_layout(height=300, template="plotly_white", plot_bgcolor="#f8f9fa", paper_bgcolor="#ffffff", margin=dict(t=30, b=30, l=60, r=80), legend=dict(orientation="h", y=1.08, font_size=10), title=dict(text=f"Prophet  |  {pf['trend_label']}  |  30d: ${pr_med:,.2f} ({pr_chg:+.1f}%)", font_size=11, x=0),)
                st.plotly_chart(fig_pr, use_container_width=True)
                # Agreement summary
                try:
                    mc2 = monte_carlo_forecast(close_s, profile=s["profile"], n_days=30, n_sim=200)
                    mc_m = float(mc2["p50"][-1])
                    diff = (pr_med - mc_m) / pf["last"] * 100
                    if abs(diff) < 1.0: st.success("Both models agree – forecast aligned")
                    elif diff > 0: st.info(f"Prophet more bullish than MC by {diff:+.1f}%")
                    else: st.warning(f"Prophet more bearish than MC by {diff:+.1f}%")
                except Exception:
                    pass
        except Exception as e:
            st.caption(f"Prophet error: {e}")
 
# PAGE: ORDER LEVELS
elif "Order Levels" in page:
    st.title("Order Levels")
    st.caption("Buy Limit = BB lower + 0.5%  ·  Stop-Loss = BuyLimit − ATR×mult  ·  TP1 = R:R 1:1  ·  TP2 = BB upper")
    if not signals:
        st.info("No assets match current filters.")
        st.stop()
    rows = []
    for s in signals:
        bl, sl, tp1, tp2, price = s["buy_limit"], s["stop_loss"], s["tp1"], s["bb_upper_target"], s["price"]
        risk_per = bl - sl
        def pct(t): return (t - price) / price * 100
        def rr(t):  return abs((t - bl) / risk_per) if risk_per > 0 else 0
        rows.append({"Asset": s["asset"], "Profile": s["profile"], "Price": f"${price:,.2f}", "Signal": s["signal"], "Buy Limit": f"${bl:,.2f}  ({pct(bl):+.1f}%)", "Stop-Loss": f"${sl:,.2f}  ({pct(sl):+.1f}%)", "TP1 (R:R 1:1)": f"${tp1:,.2f}  ({pct(tp1):+.1f}%)  1:{rr(tp1):.1f}", "TP2 (BB up)": f"${tp2:,.2f}  ({pct(tp2):+.1f}%)  1:{rr(tp2):.1f}", "Risk USD": f"${s['risk_usd']:,.0f}",})
    df_ord = pd.DataFrame(rows)
    def color_order(val):
        col = str(val)
        if "BUY"  in col: return "color: #2ecc71; font-weight: bold"
        if "SELL" in col: return "color: #e74c3c; font-weight: bold"
        if "NEU"  in col: return "color: #f39c12; font-weight: bold"
        return ""
    styled_ord = df_ord.style.applymap(color_order, subset=["Signal"])
    st.dataframe(styled_ord, use_container_width=True, height="content", hide_index=True)
# PAGE: BACKTEST SUMMARY
elif "Backtest Summary" in page:
    st.title("Backtest Summary")
    st.caption(f"Historical backtest from {user_start_str if user_start_str else START_DATE} · Initial capital ${user_capital if user_capital else INITIAL_CAP:,}/asset")
    if st.button("▶  Run full backtest  (may take 2–4 minutes)", type="primary"):
        st.cache_data.clear()
    with st.spinner("Running backtest for all assets…"):
        bt_results = run_full_backtest(start_date=user_start_str, capital=user_capital, convert_currencies=convert_fx)
    if not bt_results:
        st.warning("No backtest results. Check internet connection.")
        st.stop()
    # Summary table
    st.markdown('<div class="section-title">Results (sorted best → worst return)</div>', unsafe_allow_html=True)
    bt_rows = []
    for rank, r in enumerate(bt_results, 1):
        tr = r["trades_df"]; alpha = r["total_return"] - r["bh_return"]
        bt_rows.append({"#": rank, "Asset": r["asset"], "Profile": r.get("profile", "-"), "Final ($)": f"${r['final_value']:,.0f}", "Return": r["total_return"], "B&H": r["bh_return"], "Alpha": alpha, "Win rate": r["win_rate"], "Sharpe": r["sharpe"], "Max DD": r["max_drawdown"], "Profit F.": r["profit_factor"] if r["profit_factor"] != np.inf else 999,})
    df_bt = pd.DataFrame(bt_rows)
    def color_bt(val):
        try:
            v = float(val)
            return "color: #2ecc71" if v > 0 else "color: #e74c3c" if v < 0 else ""
        except: pass
        return ""
    styled_bt = df_bt.style\
        .applymap(color_bt, subset=["Return","B&H","Alpha"])\
        .format({"Return": "{:+.1f}%", "B&H": "{:+.1f}%", "Alpha": "{:+.1f}%", "Win rate": "{:.0f}%", "Sharpe": "{:.2f}", "Max DD": "{:.1f}%", "Profit F.": "{:.2f}",})
    st.dataframe(styled_bt, use_container_width=True, height="content", hide_index=True)
    # Top / Bottom 
    col_top, col_bot = st.columns(2)
    best, worst = bt_results[0], bt_results[-1]
    col_top.success(f"Best: **{best['asset']}**  {best['total_return']:+.1f}%  (Sharpe {best['sharpe']:.2f})")
    col_bot.error(  f"Worst: **{worst['asset']}**  {worst['total_return']:+.1f}%  (Sharpe {worst['sharpe']:.2f})")
#  PAGE: COMPARISON CHARTS
elif "Comparison Charts" in page:
    st.title("Comparison Charts")
    with st.spinner("Loading backtest data…"):
        bt_results = run_full_backtest(start_date=user_start_str, capital=user_capital, convert_currencies=convert_fx)
    if not bt_results:
        st.warning("No backtest results available.")
        st.stop()
    # Asset selector
    all_names = [r["asset"] for r in bt_results]
    selected_assets = st.multiselect("Select assets to compare", all_names, default=all_names[:8],)
    filtered_bt = [r for r in bt_results if r["asset"] in selected_assets]
    if not filtered_bt:
        st.info("Select at least one asset.")
        st.stop()
    # Equity curves
    st.markdown('<div class="section-title">Equity curves</div>', unsafe_allow_html=True)
    fig_eq = go.Figure()
    colors = px.colors.qualitative.Plotly
    for i, r in enumerate(filtered_bt):
        eq = r["equity_df"]["equity"]
        fig_eq.add_trace(go.Scatter(x=eq.index, y=eq.values, name=r["asset"], line=dict(color=colors[i % len(colors)], width=2),))
    fig_eq.add_hline(y=INITIAL_CAP, line_color="#aaa", line_dash="dot", annotation_text=f" Initial ${INITIAL_CAP:,}")
    fig_eq.update_layout(height=420, template="plotly_white", plot_bgcolor="#f8f9fa", paper_bgcolor="#ffffff", margin=dict(t=10, b=30, l=70, r=20), yaxis=dict(title="Capital (USD)"), legend=dict(orientation="h", y=1.05, font_size=10),)
    st.plotly_chart(fig_eq, use_container_width=True)
    # 4-panel comparison
    st.markdown('<div class="section-title">Performance comparison</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    names_f, returns_f, bh_f, sharpes_f, dd_f, wr_f = [r["asset"] for r in filtered_bt], [r["total_return"] for r in filtered_bt], [r["bh_return"] for r in filtered_bt], [r["sharpe"] for r in filtered_bt], [r["max_drawdown"] for r in filtered_bt], [r["win_rate"] for r in filtered_bt]
    with col1:
        # Return vs B&H
        fig_c1 = go.Figure()
        fig_c1.add_trace(go.Bar(name="Strategy", x=names_f, y=returns_f, marker_color=["#2ecc71" if v>=0 else "#e74c3c" for v in returns_f]))
        fig_c1.add_trace(go.Bar(name="B&H", x=names_f, y=bh_f, marker_color="#94a3b8", opacity=0.7))
        fig_c1.update_layout(title="Return vs Buy & Hold", barmode="group", height=300, template="plotly_white", plot_bgcolor="#f8f9fa", paper_bgcolor="#ffffff", margin=dict(t=40,b=60,l=50,r=10), xaxis=dict(tickangle=-45), yaxis=dict(ticksuffix="%"), legend=dict(orientation="h",y=1.05,font_size=10))
        st.plotly_chart(fig_c1, use_container_width=True)
        # Max drawdown
        fig_c3 = go.Figure(go.Bar(x=names_f, y=dd_f, marker_color="#e74c3c", opacity=0.8, text=[f"{d:.1f}%" for d in dd_f], textposition="outside"))
        fig_c3.update_layout(title="Max Drawdown", height=300, template="plotly_white", plot_bgcolor="#f8f9fa", paper_bgcolor="#ffffff", margin=dict(t=40,b=60,l=50,r=10), xaxis=dict(tickangle=-45), yaxis=dict(ticksuffix="%"), showlegend=False)
        st.plotly_chart(fig_c3, use_container_width=True)
    with col2:
        # Sharpe
        fig_c2 = go.Figure(go.Bar(x=names_f, y=sharpes_f, marker_color=["#2ecc71" if s>1 else "#f39c12" if s>0 else "#e74c3c" for s in sharpes_f], text=[f"{s:.2f}" for s in sharpes_f], textposition="outside",))
        fig_c2.add_hline(y=1, line_color="#aaa", line_dash="dot")
        fig_c2.update_layout(title="Sharpe Ratio", height=300, template="plotly_white", plot_bgcolor="#f8f9fa", paper_bgcolor="#ffffff", margin=dict(t=40,b=60,l=50,r=10), xaxis=dict(tickangle=-45), showlegend=False)
        st.plotly_chart(fig_c2, use_container_width=True)
        # Win rate
        fig_c4 = go.Figure(go.Bar(x=names_f, y=wr_f, marker_color=["#2ecc71" if w>55 else "#f39c12" if w>45 else "#e74c3c" for w in wr_f], text=[f"{w:.0f}%" for w in wr_f], textposition="outside",))
        fig_c4.add_hline(y=50, line_color="#aaa", line_dash="dot")
        fig_c4.update_layout(title="Win Rate", height=300, template="plotly_white", plot_bgcolor="#f8f9fa", paper_bgcolor="#ffffff", margin=dict(t=40,b=60,l=50,r=10), xaxis=dict(tickangle=-45), yaxis=dict(range=[0,100], ticksuffix="%"), showlegend=False)
        st.plotly_chart(fig_c4, use_container_width=True)
    # Return vs Sharpe 
    st.markdown('<div class="section-title">Return vs Sharpe – risk/reward scatter</div>', unsafe_allow_html=True)
    fig_sc = go.Figure(go.Scatter(x=sharpes_f, y=returns_f, mode="markers+text", text=names_f, textposition="top center", textfont=dict(size=10), marker=dict(size=12, color=returns_f, colorscale="RdYlGn", showscale=True, colorbar=dict(title="Return %"),),))
    fig_sc.add_vline(x=1, line_color="#aaa", line_dash="dot", annotation_text=" Sharpe = 1")
    fig_sc.add_hline(y=0, line_color="#aaa", line_dash="dot")
    fig_sc.update_layout(
    height=420, template="plotly_white", plot_bgcolor="#f8f9fa", paper_bgcolor="#ffffff", margin=dict(t=20, b=40, l=70, r=20), xaxis=dict(title="Sharpe Ratio"), yaxis=dict(title="Total Return (%)", ticksuffix="%"),)
    st.plotly_chart(fig_sc, use_container_width=True)
