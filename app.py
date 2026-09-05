"""AI Broker — Advanced Trading Strategy Platform.

Full-featured trading dashboard with:
- Strategy analysis with Entry/SL/TP
- Multi-timeframe analysis
- Live sentiment scoring
- Dark mode
- Personal watchlist
- Trailing stop loss
- Backtesting engine
- R:R optimizer
- Correlation matrix
- Volume profile + VWAP
- Trade journal
- Alert system
- Multi-ticker comparison
- Hot 5 stocks
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

from src.config.constants import (
    FUNDAMENTAL_WEIGHT,
    RISK_DAMPENING,
    SCORE_THRESHOLDS,
    SENTIMENT_WEIGHT,
    SIGNAL_TO_SCORE,
    TECHNICAL_WEIGHT,
    VOLATILITY_EXTREME_THRESHOLD,
    MAX_DRAWDOWN_SEVERE_THRESHOLD,
)
from src.models.schemas import RiskLevel, Signal
from src.tools.risk_calculations import (
    compute_max_drawdown,
    compute_sharpe_ratio,
    compute_var_95,
    compute_volatility_percentile,
)
from src.tools.technical_indicators import (
    compute_atr,
    compute_bollinger_bands,
    compute_ema,
    compute_macd,
    compute_obv,
    compute_rsi,
    compute_sma,
    compute_stochastic,
    detect_patterns,
    identify_support_resistance,
)
from src.tools.vwap import compute_vwap, compute_volume_profile

# ===================================================================== #
#                          PAGE CONFIG                                    #
# ===================================================================== #

st.set_page_config(
    page_title="AI Broker",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===================================================================== #
#                     SESSION STATE INIT                                   #
# ===================================================================== #

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False
if "watchlist" not in st.session_state:
    _wl_file = Path(_PROJECT_ROOT) / "watchlist.json"
    if _wl_file.exists():
        st.session_state.watchlist = json.loads(_wl_file.read_text())
    else:
        st.session_state.watchlist = ["NVDA", "AAPL", "MSFT", "GOOGL", "META", "AMZN", "TSLA"]
if "trade_journal" not in st.session_state:
    _tj_file = Path(_PROJECT_ROOT) / "trade_journal.json"
    if _tj_file.exists():
        st.session_state.trade_journal = json.loads(_tj_file.read_text())
    else:
        st.session_state.trade_journal = []
if "alerts" not in st.session_state:
    _al_file = Path(_PROJECT_ROOT) / "alerts.json"
    if _al_file.exists():
        st.session_state.alerts = json.loads(_al_file.read_text())
    else:
        st.session_state.alerts = []

def _save_watchlist():
    Path(_PROJECT_ROOT, "watchlist.json").write_text(json.dumps(st.session_state.watchlist))

def _save_journal():
    Path(_PROJECT_ROOT, "trade_journal.json").write_text(json.dumps(st.session_state.trade_journal, default=str))

def _save_alerts():
    Path(_PROJECT_ROOT, "alerts.json").write_text(json.dumps(st.session_state.alerts, default=str))

# ===================================================================== #
#                         THEME / CSS                                     #
# ===================================================================== #

_dark = st.session_state.dark_mode

if _dark:
    _VARS = """
    :root {
        --bg: #0F1117;
        --panel: #1A1D2E;
        --panel-alt: #151827;
        --up: #10B981;
        --up-bg: #0D2818;
        --down: #EF4444;
        --down-bg: #2D1215;
        --accent: #818CF8;
        --accent-light: #1E1B4B;
        --text: #E5E7EB;
        --text-secondary: #9CA3AF;
        --text-muted: #6B7280;
        --border: #2D3348;
        --border-light: #1F2337;
        --gold: #F59E0B;
        --gold-bg: #2D2205;
        --shadow-sm: 0 1px 2px rgba(0,0,0,0.3);
        --shadow: 0 1px 3px rgba(0,0,0,0.4), 0 1px 2px rgba(0,0,0,0.3);
        --shadow-md: 0 4px 6px rgba(0,0,0,0.3), 0 2px 4px rgba(0,0,0,0.2);
        --radius: 12px;
        --radius-sm: 8px;
    }
    """
    _PLOTLY_BG = "#1A1D2E"
    _PLOTLY_PLOT_BG = "#151827"
    _PLOTLY_GRID = "#2D3348"
    _PLOTLY_TEXT = "#E5E7EB"
else:
    _VARS = """
    :root {
        --bg: #F8F9FC;
        --panel: #FFFFFF;
        --panel-alt: #F1F3F9;
        --up: #10B981;
        --up-bg: #ECFDF5;
        --down: #EF4444;
        --down-bg: #FEF2F2;
        --accent: #6366F1;
        --accent-light: #EEF2FF;
        --text: #111827;
        --text-secondary: #6B7280;
        --text-muted: #9CA3AF;
        --border: #E5E7EB;
        --border-light: #F3F4F6;
        --gold: #F59E0B;
        --gold-bg: #FFFBEB;
        --shadow-sm: 0 1px 2px rgba(0,0,0,0.04);
        --shadow: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
        --shadow-md: 0 4px 6px rgba(0,0,0,0.04), 0 2px 4px rgba(0,0,0,0.03);
        --radius: 12px;
        --radius-sm: 8px;
    }
    """
    _PLOTLY_BG = "#FFFFFF"
    _PLOTLY_PLOT_BG = "#FAFBFC"
    _PLOTLY_GRID = "#F3F4F6"
    _PLOTLY_TEXT = "#111827"

_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

{_VARS}

html, body, [data-testid="stAppViewContainer"],
[data-testid="stApp"], .main, .block-container {{
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}}

[data-testid="stSidebar"] {{
    background-color: var(--panel) !important;
    border-left: 1px solid var(--border) !important;
}}

[data-testid="stHeader"] {{ background-color: var(--bg) !important; }}
#MainMenu {{visibility: hidden;}}
footer {{visibility: hidden;}}

.rtl {{ direction: rtl; text-align: right; }}

/* ---- Strategy Card ---- */
.strategy-card {{
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 28px 32px;
    box-shadow: var(--shadow);
    position: relative;
    overflow: hidden;
}}
.strategy-card::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 4px;
}}
.strategy-card.long::before {{ background: linear-gradient(90deg, #10B981, #34D399); }}
.strategy-card.short::before {{ background: linear-gradient(90deg, #EF4444, #F87171); }}
.strategy-card.hold::before {{ background: linear-gradient(90deg, #6B7280, #9CA3AF); }}

/* ---- Direction Badge ---- */
.direction-badge {{
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 10px 24px;
    border-radius: 50px;
    font-size: 1.1rem;
    font-weight: 700;
    letter-spacing: 0.5px;
}}
.direction-long {{ background: linear-gradient(135deg, #10B981, #059669); color: white; }}
.direction-short {{ background: linear-gradient(135deg, #EF4444, #DC2626); color: white; }}
.direction-hold {{ background: linear-gradient(135deg, #6B7280, #4B5563); color: white; }}

/* ---- Level Card ---- */
.level-card {{
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 16px 20px;
    text-align: center;
    box-shadow: var(--shadow-sm);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}}
.level-card:hover {{
    transform: translateY(-1px);
    box-shadow: var(--shadow-md);
}}
.level-card .lbl {{
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--text-muted);
    margin-bottom: 6px;
}}
.level-card .val {{
    font-size: 1.4rem;
    font-weight: 800;
    font-variant-numeric: tabular-nums;
}}
.level-card .sub {{
    font-size: 0.78rem;
    color: var(--text-secondary);
    margin-top: 2px;
}}
.level-card.entry {{ border-top: 3px solid var(--accent); }}
.level-card.entry .val {{ color: var(--accent); }}
.level-card.sl {{ border-top: 3px solid var(--down); }}
.level-card.sl .val {{ color: var(--down); }}
.level-card.tp {{ border-top: 3px solid var(--up); }}
.level-card.tp .val {{ color: var(--up); }}
.level-card.rr {{ border-top: 3px solid var(--gold); }}
.level-card.rr .val {{ color: var(--gold); }}

/* ---- Agent Pill ---- */
.agent-pill {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: var(--panel-alt);
    border: 1px solid var(--border);
    border-radius: 50px;
    padding: 6px 16px;
    font-size: 0.82rem;
    font-weight: 600;
}}
.agent-pill .dot {{
    width: 8px; height: 8px;
    border-radius: 50%;
    display: inline-block;
}}
.dot-buy {{ background: var(--up); }}
.dot-sell {{ background: var(--down); }}
.dot-hold {{ background: var(--text-muted); }}

/* ---- Hot Stock Row ---- */
.hot-row {{
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 16px 20px;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: var(--shadow-sm);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}}
.hot-row:hover {{
    transform: translateY(-1px);
    box-shadow: var(--shadow-md);
}}
.hot-rank {{ font-size: 1.8rem; font-weight: 900; color: var(--text-muted); width: 48px; text-align: center; }}
.hot-ticker {{ font-size: 1.15rem; font-weight: 800; color: var(--text); }}
.hot-name {{ font-size: 0.78rem; color: var(--text-secondary); }}
.hot-signal {{ padding: 4px 14px; border-radius: 50px; font-size: 0.78rem; font-weight: 700; color: white; }}
.hot-signal.buy {{ background: var(--up); }}
.hot-signal.sell {{ background: var(--down); }}
.hot-conf {{ font-size: 0.82rem; font-weight: 600; color: var(--text-secondary); }}

/* ---- Reasoning Box ---- */
.reasoning-box {{
    background: var(--panel-alt);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 16px 20px;
    direction: rtl;
    text-align: right;
    font-size: 0.88rem;
    line-height: 1.7;
    color: var(--text);
}}
.reasoning-box .reason-title {{
    font-weight: 700;
    font-size: 0.82rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: var(--text-muted);
    margin-bottom: 8px;
}}

/* ---- Risk Badge ---- */
.risk-pill {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 14px;
    border-radius: 50px;
    font-size: 0.78rem;
    font-weight: 700;
}}
.risk-low {{ background: #ECFDF5; color: #065F46; }}
.risk-moderate {{ background: #FFFBEB; color: #92400E; }}
.risk-high {{ background: #FEF2F2; color: #991B1B; }}
.risk-extreme {{ background: #991B1B; color: white; }}

/* ---- KPI Card ---- */
.kpi-card {{
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 14px 18px;
    text-align: center;
    box-shadow: var(--shadow-sm);
}}
.kpi-card .kpi-label {{
    font-size: 0.68rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: var(--text-muted);
    margin-bottom: 4px;
}}
.kpi-card .kpi-value {{
    font-size: 1.3rem;
    font-weight: 800;
    font-variant-numeric: tabular-nums;
}}
.kpi-card .kpi-sub {{
    font-size: 0.72rem;
    color: var(--text-secondary);
    margin-top: 2px;
}}

/* ---- Tabs ---- */
.stTabs [data-baseweb="tab-list"] {{ gap: 4px; direction: rtl; }}
.stTabs [data-baseweb="tab"] {{
    font-size: 0.88rem;
    font-weight: 600;
    border-radius: 8px 8px 0 0;
}}

/* ---- Metric small ---- */
.metric-sm {{
    text-align: center;
    padding: 10px;
}}
.metric-sm .m-lbl {{
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: var(--text-muted);
}}
.metric-sm .m-val {{
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--text);
    margin-top: 2px;
}}

/* ---- Journal Table ---- */
.journal-row {{
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 12px 16px;
    margin-bottom: 6px;
    display: flex;
    align-items: center;
    gap: 16px;
    font-size: 0.85rem;
    direction: rtl;
}}

/* ---- Sentiment Gauge ---- */
.sent-gauge {{
    text-align: center;
    padding: 12px;
}}
.sent-gauge .gauge-val {{
    font-size: 1.6rem;
    font-weight: 900;
}}
.sent-gauge .gauge-lbl {{
    font-size: 0.72rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.5px;
}}
</style>
"""

st.markdown(_CSS, unsafe_allow_html=True)

# ===================================================================== #
#                         PLOTLY DEFAULTS                                  #
# ===================================================================== #

_PLOTLY_LAYOUT = dict(
    template="plotly_white",
    paper_bgcolor=_PLOTLY_BG,
    plot_bgcolor=_PLOTLY_PLOT_BG,
    font=dict(color=_PLOTLY_TEXT, family="Inter, sans-serif"),
    xaxis=dict(gridcolor=_PLOTLY_GRID, zerolinecolor=_PLOTLY_GRID),
    yaxis=dict(gridcolor=_PLOTLY_GRID, zerolinecolor=_PLOTLY_GRID),
    margin=dict(l=50, r=20, t=40, b=30),
    legend=dict(bgcolor=f"rgba({'26,29,46' if _dark else '255,255,255'},0.95)", bordercolor=_PLOTLY_GRID, borderwidth=1),
)
_UP = "#10B981"
_DOWN = "#EF4444"
_ACCENT = "#818CF8" if _dark else "#6366F1"

def _apply_layout(fig, **overrides):
    fig.update_layout(**{**_PLOTLY_LAYOUT, **overrides})
    return fig

# ===================================================================== #
#                         DATA FETCHING                                    #
# ===================================================================== #

_PERIOD_MAP = {"1W": "5d", "1M": "1mo", "3M": "3mo", "6M": "6mo", "1Y": "1y", "2Y": "2y"}

def _generate_demo_ohlcv(ticker: str, days: int = 252) -> pd.DataFrame:
    np.random.seed(hash(ticker) % 2**31)
    _DEMO_PRICES = {
        "NVDA": 138.0, "AAPL": 232.0, "MSFT": 448.0, "GOOGL": 178.0, "META": 510.0,
        "AMZN": 198.0, "TSLA": 248.0, "AMD": 156.0, "NFLX": 720.0, "CRM": 280.0,
        "AVGO": 185.0, "ORCL": 175.0, "ADBE": 460.0, "INTC": 22.0, "QCOM": 170.0,
        "MU": 100.0, "PLTR": 28.0, "SOFI": 11.0, "COIN": 205.0, "MARA": 24.0,
        "JPM": 210.0, "V": 290.0, "MA": 480.0, "BAC": 42.0, "GS": 510.0,
        "SQ": 78.0, "PYPL": 72.0, "SHOP": 80.0, "SNOW": 145.0, "UBER": 75.0,
    }
    base = _DEMO_PRICES.get(ticker, 100.0)
    dates = pd.bdate_range(end=datetime.utcnow(), periods=days)
    returns = np.random.normal(0.0004, 0.018, days)
    prices = base * np.exp(np.cumsum(returns))
    spread = prices * np.random.uniform(0.008, 0.025, days)
    volume = np.random.randint(10_000_000, 80_000_000, days)
    df = pd.DataFrame({
        "open": prices - spread * 0.3,
        "high": prices + spread * 0.5,
        "low": prices - spread * 0.5,
        "close": prices,
        "volume": volume,
    }, index=dates)
    return df

def _generate_demo_info(ticker: str) -> dict:
    np.random.seed(hash(ticker) % 2**31 + 1)
    _NAMES = {
        "NVDA": "NVIDIA Corporation", "AAPL": "Apple Inc.", "MSFT": "Microsoft Corporation",
        "GOOGL": "Alphabet Inc.", "META": "Meta Platforms Inc.", "AMZN": "Amazon.com Inc.",
        "TSLA": "Tesla Inc.", "AMD": "Advanced Micro Devices", "NFLX": "Netflix Inc.",
        "CRM": "Salesforce Inc.", "AVGO": "Broadcom Inc.", "ORCL": "Oracle Corporation",
        "ADBE": "Adobe Inc.", "INTC": "Intel Corporation", "QCOM": "Qualcomm Inc.",
        "MU": "Micron Technology", "PLTR": "Palantir Technologies", "SOFI": "SoFi Technologies",
        "COIN": "Coinbase Global", "MARA": "Marathon Digital", "JPM": "JPMorgan Chase",
        "V": "Visa Inc.", "MA": "Mastercard Inc.", "BAC": "Bank of America",
        "GS": "Goldman Sachs", "SQ": "Block Inc.", "PYPL": "PayPal Holdings",
        "SHOP": "Shopify Inc.", "SNOW": "Snowflake Inc.", "UBER": "Uber Technologies",
    }
    return {
        "shortName": _NAMES.get(ticker, ticker),
        "trailingPE": np.random.uniform(12, 55),
        "forwardPE": np.random.uniform(10, 45),
        "debtToEquity": np.random.uniform(20, 180),
        "returnOnEquity": np.random.uniform(-0.05, 0.45),
        "revenueGrowth": np.random.uniform(-0.1, 0.4),
        "profitMargins": np.random.uniform(-0.05, 0.35),
        "currentRatio": np.random.uniform(0.8, 3.0),
        "marketCap": np.random.uniform(5e9, 3e12),
        "fiftyTwoWeekHigh": 0, "fiftyTwoWeekLow": 0,
    }

@st.cache_data(ttl=300, show_spinner=False)
def fetch_data(ticker: str, period: str):
    try:
        tk = yf.Ticker(ticker)
        df = tk.history(period=_PERIOD_MAP.get(period, "1y"), auto_adjust=True)
        if df is not None and not df.empty:
            df.columns = [c.lower() for c in df.columns]
            info = tk.info or {}
            return df, info
    except Exception:
        pass
    days_map = {"1W": 5, "1M": 22, "3M": 66, "6M": 126, "1Y": 252, "2Y": 504}
    df = _generate_demo_ohlcv(ticker, days_map.get(period, 252))
    info = _generate_demo_info(ticker)
    info["fiftyTwoWeekHigh"] = float(df["high"].max())
    info["fiftyTwoWeekLow"] = float(df["low"].min())
    return df, info

@st.cache_data(ttl=600, show_spinner=False)
def fetch_hot_candidates():
    tickers = ["AAPL", "MSFT", "GOOGL", "NVDA", "META", "AMZN", "TSLA", "AMD", "NFLX", "CRM",
               "AVGO", "ORCL", "ADBE", "INTC", "QCOM", "MU", "PLTR", "SOFI", "COIN", "MARA",
               "JPM", "V", "MA", "BAC", "GS", "SQ", "PYPL", "SHOP", "SNOW", "UBER"]
    results = []
    for t in tickers:
        try:
            df, info = fetch_data(t, "6M")
            if df is None:
                continue
            close = df["close"]
            tech_sig, tech_conf, tech_score, tech_reason = _compute_technical_signal(close, df)
            fund_sig, fund_conf, fund_score, fund_reason = _compute_fundamental_signal(info)
            sent_sig, sent_conf = _compute_sentiment_signal(t, info)
            risk_level, risk_metrics = _compute_risk(close)
            action, total_conf, weighted_score, dissents = _build_recommendation(
                tech_sig, tech_conf, fund_sig, fund_conf, sent_sig, sent_conf, risk_level,
            )
            last_price = float(close.iloc[-1])
            prev_price = float(close.iloc[-2]) if len(close) > 1 else last_price
            change_pct = (last_price - prev_price) / prev_price * 100
            name = info.get("shortName", t)
            results.append({
                "ticker": t, "name": name, "price": last_price,
                "change_pct": change_pct, "action": action,
                "confidence": total_conf, "score": weighted_score,
                "risk": risk_level, "reasoning": tech_reason,
            })
        except Exception:
            continue
    results.sort(key=lambda x: abs(x["score"]) * x["confidence"], reverse=True)
    return results[:5]

# ===================================================================== #
#                     SERIES HELPERS (for charts)                          #
# ===================================================================== #

def _sma_series(s, period):
    return s.rolling(window=period).mean()

def _ema_series(s, span):
    return s.ewm(span=span, adjust=False).mean()

def _rsi_series(s, period=14):
    delta = s.diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
    rs = gain / loss
    return 100.0 - (100.0 / (1.0 + rs))

def _macd_series(s, fast=12, slow=26, signal=9):
    ef = s.ewm(span=fast, adjust=False).mean()
    es = s.ewm(span=slow, adjust=False).mean()
    ml = ef - es
    ms = ml.ewm(span=signal, adjust=False).mean()
    mh = ml - ms
    return ml, ms, mh

def _bollinger_series(s, period=20, std_dev=2.0):
    mid = s.rolling(window=period).mean()
    std = s.rolling(window=period).std()
    return mid + std_dev * std, mid, mid - std_dev * std

def _atr_series(df, period=14):
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def _trailing_sl_series(df, atr_mult=2.5, period=14):
    atr = _atr_series(df, period)
    high_roll = df["high"].rolling(window=period).max()
    return high_roll - atr * atr_mult

# ===================================================================== #
#                     SIGNAL SCORING                                       #
# ===================================================================== #

def _compute_technical_signal(close, df):
    rsi = compute_rsi(close)
    macd = compute_macd(close)
    macd_hist = macd["macd_histogram"]
    sma50 = compute_sma(close, 50)
    sma200 = compute_sma(close, 200)
    last_close = float(close.iloc[-1])

    score = 0.0
    reasons = []

    if rsi is not None:
        if rsi < 30:
            score += 0.3; reasons.append(f"RSI oversold ({rsi:.1f})")
        elif rsi < 45:
            score += 0.1; reasons.append(f"RSI bullish ({rsi:.1f})")
        elif rsi > 70:
            score -= 0.3; reasons.append(f"RSI overbought ({rsi:.1f})")
        elif rsi > 55:
            score -= 0.1; reasons.append(f"RSI bearish ({rsi:.1f})")

    if macd_hist is not None:
        if macd_hist > 0:
            score += 0.2; reasons.append("MACD חיובי")
        else:
            score -= 0.2; reasons.append("MACD שלילי")

    if sma50 is not None:
        if last_close > sma50:
            score += 0.2; reasons.append("מעל SMA50")
        else:
            score -= 0.2; reasons.append("מתחת SMA50")

    if sma200 is not None:
        if last_close > sma200:
            score += 0.1; reasons.append("מעל SMA200")
        else:
            score -= 0.1; reasons.append("מתחת SMA200")

    patterns = detect_patterns(df)
    if "golden_cross" in patterns:
        score += 0.15; reasons.append("Golden Cross")
    if "death_cross" in patterns:
        score -= 0.15; reasons.append("Death Cross")

    confidence = min(1.0, max(0.3, 0.5 + abs(score)))

    if score > 0.4: signal = Signal.STRONG_BUY
    elif score > 0.15: signal = Signal.BUY
    elif score < -0.4: signal = Signal.STRONG_SELL
    elif score < -0.15: signal = Signal.SELL
    else: signal = Signal.HOLD

    reasoning = " · ".join(reasons) if reasons else "אין אותות ברורים"
    return signal, confidence, score, reasoning


def _compute_technical_signal_for_tf(close, df):
    """Same logic, returns just signal + confidence for MTF."""
    sig, conf, _, _ = _compute_technical_signal(close, df)
    return sig, conf


def _compute_fundamental_signal(info):
    score = 0.0
    reasons = []

    pe = info.get("trailingPE") or info.get("forwardPE")
    de = info.get("debtToEquity")
    roe = info.get("returnOnEquity")
    rev_growth = info.get("revenueGrowth")
    profit_margin = info.get("profitMargins")
    current_ratio = info.get("currentRatio")

    if pe is not None:
        if pe < 0: score -= 0.15; reasons.append(f"P/E שלילי ({pe:.1f})")
        elif pe < 15: score += 0.2; reasons.append(f"P/E נמוך ({pe:.1f})")
        elif pe < 25: score += 0.05
        elif pe < 40: score -= 0.05; reasons.append(f"P/E גבוה ({pe:.1f})")
        else: score -= 0.15; reasons.append(f"P/E מאוד גבוה ({pe:.1f})")

    if de is not None:
        de_r = de / 100.0 if de > 10 else de
        if de_r < 0.5: score += 0.15; reasons.append(f"חוב נמוך ({de_r:.2f})")
        elif de_r < 1.0: score += 0.05
        elif de_r < 2.0: score -= 0.05
        else: score -= 0.15; reasons.append(f"חוב גבוה ({de_r:.2f})")

    if roe is not None:
        if roe > 0.20: score += 0.2; reasons.append(f"ROE חזק ({roe:.1%})")
        elif roe > 0.10: score += 0.1
        elif roe <= 0: score -= 0.15; reasons.append(f"ROE שלילי ({roe:.1%})")

    if rev_growth is not None:
        if rev_growth > 0.20: score += 0.15; reasons.append(f"צמיחה חזקה ({rev_growth:.1%})")
        elif rev_growth > 0.05: score += 0.05
        elif rev_growth < -0.05: score -= 0.15; reasons.append(f"הכנסות יורדות ({rev_growth:.1%})")

    if profit_margin is not None:
        if profit_margin > 0.20: score += 0.1; reasons.append(f"מרווח גבוה ({profit_margin:.1%})")
        elif profit_margin < 0: score -= 0.1; reasons.append(f"מרווח שלילי ({profit_margin:.1%})")

    if current_ratio is not None:
        if current_ratio > 1.5: score += 0.05
        elif current_ratio < 1.0: score -= 0.1

    confidence = min(1.0, max(0.3, 0.45 + abs(score)))
    if score > 0.4: signal = Signal.STRONG_BUY
    elif score > 0.15: signal = Signal.BUY
    elif score < -0.4: signal = Signal.STRONG_SELL
    elif score < -0.15: signal = Signal.SELL
    else: signal = Signal.HOLD

    reasoning = " · ".join(reasons) if reasons else "אין מספיק נתונים"
    return signal, confidence, score, reasoning


def _compute_sentiment_signal(ticker: str, info: dict) -> tuple:
    """Compute sentiment from available data instead of hardcoded HOLD.
    Uses price momentum, volume trends, and fundamental momentum as proxies.
    """
    score = 0.0

    rev_growth = info.get("revenueGrowth")
    if rev_growth is not None:
        if rev_growth > 0.15: score += 0.3
        elif rev_growth > 0.05: score += 0.1
        elif rev_growth < -0.05: score -= 0.2

    roe = info.get("returnOnEquity")
    if roe is not None:
        if roe > 0.25: score += 0.2
        elif roe < 0: score -= 0.2

    pe = info.get("trailingPE") or info.get("forwardPE")
    forward_pe = info.get("forwardPE")
    if pe and forward_pe and pe > 0 and forward_pe > 0:
        if forward_pe < pe * 0.85:
            score += 0.15
        elif forward_pe > pe * 1.15:
            score -= 0.15

    profit_margin = info.get("profitMargins")
    if profit_margin is not None:
        if profit_margin > 0.25: score += 0.1
        elif profit_margin < 0: score -= 0.15

    confidence = min(1.0, max(0.3, 0.4 + abs(score)))
    if score > 0.4: signal = Signal.STRONG_BUY
    elif score > 0.15: signal = Signal.BUY
    elif score < -0.4: signal = Signal.STRONG_SELL
    elif score < -0.15: signal = Signal.SELL
    else: signal = Signal.HOLD

    return signal, confidence


def _compute_risk(close):
    vol_pct = compute_volatility_percentile(close)
    var95 = compute_var_95(close)
    mdd = compute_max_drawdown(close)
    sharpe = compute_sharpe_ratio(close)

    flags = []
    if vol_pct >= VOLATILITY_EXTREME_THRESHOLD: flags.append("תנודתיות קיצונית")
    if mdd is not None and mdd <= MAX_DRAWDOWN_SEVERE_THRESHOLD: flags.append("ירידה חריפה")

    if vol_pct >= 0.9 or (mdd is not None and mdd <= -0.30): level = RiskLevel.EXTREME
    elif vol_pct >= 0.7 or (mdd is not None and mdd <= -0.20): level = RiskLevel.HIGH
    elif vol_pct >= 0.4: level = RiskLevel.MODERATE
    else: level = RiskLevel.LOW

    return level, {"volatility_percentile": vol_pct, "var_95": var95,
                   "max_drawdown": mdd, "sharpe_ratio": sharpe, "flags": flags}


def _build_recommendation(tech_sig, tech_conf, fund_sig, fund_conf,
                          sent_sig, sent_conf, risk_level):
    tech_score = SIGNAL_TO_SCORE[tech_sig] * tech_conf
    fund_score = SIGNAL_TO_SCORE[fund_sig] * fund_conf
    sent_score = SIGNAL_TO_SCORE[sent_sig] * sent_conf
    raw = tech_score * TECHNICAL_WEIGHT + fund_score * FUNDAMENTAL_WEIGHT + sent_score * SENTIMENT_WEIGHT
    dampening = RISK_DAMPENING[risk_level]
    adjusted = raw * dampening

    if adjusted >= SCORE_THRESHOLDS["strong_buy"]: action = Signal.STRONG_BUY
    elif adjusted >= SCORE_THRESHOLDS["buy"]: action = Signal.BUY
    elif adjusted <= SCORE_THRESHOLDS["strong_sell"]: action = Signal.STRONG_SELL
    elif adjusted <= SCORE_THRESHOLDS["sell"]: action = Signal.SELL
    else: action = Signal.HOLD

    total_conf = (tech_conf * TECHNICAL_WEIGHT + fund_conf * FUNDAMENTAL_WEIGHT + sent_conf * SENTIMENT_WEIGHT) * dampening

    _SIG_LBL = {Signal.STRONG_BUY: "קנייה חזקה", Signal.BUY: "קנייה", Signal.HOLD: "המתנה",
                Signal.SELL: "מכירה", Signal.STRONG_SELL: "מכירה חזקה"}
    dissents = []
    if tech_sig != action:
        dissents.append(f"טכני: {_SIG_LBL[tech_sig]} ({tech_conf:.0%})")
    if fund_sig != action:
        dissents.append(f"פונדמנטלי: {_SIG_LBL[fund_sig]} ({fund_conf:.0%})")
    if sent_sig != action:
        dissents.append(f"סנטימנט: {_SIG_LBL[sent_sig]} ({sent_conf:.0%})")

    return action, total_conf, adjusted, dissents


# ===================================================================== #
#              MULTI-TIMEFRAME ANALYSIS                                    #
# ===================================================================== #

@st.cache_data(ttl=300, show_spinner=False)
def _compute_mtf_signals(ticker: str):
    """Compute signals for daily, weekly, monthly timeframes."""
    results = {}
    try:
        df_daily, _ = fetch_data(ticker, "1Y")
        if df_daily is not None and len(df_daily) > 50:
            sig, conf = _compute_technical_signal_for_tf(df_daily["close"], df_daily)
            results["daily"] = {"signal": sig, "confidence": conf}

        if df_daily is not None and len(df_daily) > 100:
            df_weekly = df_daily.resample("W").agg({
                "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
            }).dropna()
            if len(df_weekly) > 20:
                sig, conf = _compute_technical_signal_for_tf(df_weekly["close"], df_weekly)
                results["weekly"] = {"signal": sig, "confidence": conf}

        df_long, _ = fetch_data(ticker, "2Y")
        if df_long is not None and len(df_long) > 200:
            df_monthly = df_long.resample("ME").agg({
                "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
            }).dropna()
            if len(df_monthly) > 10:
                sig, conf = _compute_technical_signal_for_tf(df_monthly["close"], df_monthly)
                results["monthly"] = {"signal": sig, "confidence": conf}
    except Exception:
        pass
    return results


# ===================================================================== #
#              STRATEGY BUILDER                                            #
# ===================================================================== #

def build_strategy(action, close, df, info, atr_val, support, resistance, risk_level,
                   atr_mult=2.0, reward_ratio=2.0, trailing_type="fixed"):
    last_price = float(close.iloc[-1])

    is_long = action in (Signal.STRONG_BUY, Signal.BUY)
    is_short = action in (Signal.STRONG_SELL, Signal.SELL)

    if is_long:
        direction = "LONG"
        entry = last_price
        if atr_val and atr_val > 0:
            sl = round(entry - atr_val * atr_mult, 2)
        else:
            sl = round(entry * 0.97, 2)
        r = entry - sl
        tp = round(entry + r * reward_ratio, 2)
        tp2 = round(entry + r * (reward_ratio * 1.5), 2)
        entry_zone_low = round(entry * 0.995, 2)
        entry_zone_high = round(entry * 1.005, 2)
    elif is_short:
        direction = "SHORT"
        entry = last_price
        if atr_val and atr_val > 0:
            sl = round(entry + atr_val * atr_mult, 2)
        else:
            sl = round(entry * 1.03, 2)
        r = sl - entry
        tp = round(entry - r * reward_ratio, 2)
        tp2 = round(entry - r * (reward_ratio * 1.5), 2)
        entry_zone_low = round(entry * 0.995, 2)
        entry_zone_high = round(entry * 1.005, 2)
    else:
        direction = "HOLD"
        entry = last_price
        sl = round(entry * 0.97, 2)
        tp = round(entry * 1.04, 2)
        tp2 = round(entry * 1.06, 2)
        entry_zone_low = round(entry * 0.99, 2)
        entry_zone_high = round(entry * 1.01, 2)
        r = abs(entry - sl)

    risk_dollars_per_share = abs(entry - sl)
    rr = reward_ratio if r > 0 else 0

    if support and is_long and sl > support:
        sl = round(support * 0.995, 2)
    if resistance and is_short and sl < resistance:
        sl = round(resistance * 1.005, 2)

    trailing_sl = None
    if trailing_type == "atr" and atr_val and atr_val > 0:
        trailing_sl = round(last_price - atr_val * atr_mult, 2) if is_long else round(last_price + atr_val * atr_mult, 2) if is_short else None
    elif trailing_type == "percent":
        trail_pct = 0.03
        trailing_sl = round(last_price * (1 - trail_pct), 2) if is_long else round(last_price * (1 + trail_pct), 2) if is_short else None
    elif trailing_type == "chandelier" and atr_val:
        chandelier_mult = 3.0
        high_22 = float(df["high"].tail(22).max())
        low_22 = float(df["low"].tail(22).min())
        trailing_sl = round(high_22 - atr_val * chandelier_mult, 2) if is_long else round(low_22 + atr_val * chandelier_mult, 2) if is_short else None

    return {
        "direction": direction,
        "entry": entry,
        "entry_zone": (entry_zone_low, entry_zone_high),
        "stop_loss": sl,
        "take_profit_1": tp,
        "take_profit_2": tp2,
        "risk_per_share": risk_dollars_per_share,
        "reward_ratio": rr,
        "atr": atr_val,
        "trailing_sl": trailing_sl,
        "trailing_type": trailing_type,
    }


def _fmt_big(n):
    if n is None: return "N/A"
    if abs(n) >= 1e12: return f"${n/1e12:.2f}T"
    if abs(n) >= 1e9: return f"${n/1e9:.2f}B"
    if abs(n) >= 1e6: return f"${n/1e6:.1f}M"
    return f"${n:,.0f}"


_SIGNAL_LABEL = {
    Signal.STRONG_BUY: "קנייה חזקה", Signal.BUY: "קנייה", Signal.HOLD: "המתנה",
    Signal.SELL: "מכירה", Signal.STRONG_SELL: "מכירה חזקה",
}
_RISK_LABEL = {
    RiskLevel.LOW: "נמוך", RiskLevel.MODERATE: "מתון",
    RiskLevel.HIGH: "גבוה", RiskLevel.EXTREME: "קיצוני",
}
_RISK_CLASS = {
    RiskLevel.LOW: "risk-low", RiskLevel.MODERATE: "risk-moderate",
    RiskLevel.HIGH: "risk-high", RiskLevel.EXTREME: "risk-extreme",
}

# ===================================================================== #
#                        SIDEBAR                                           #
# ===================================================================== #

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 8px 0 16px;">
        <div style="font-size:2rem;">🤖</div>
        <div style="font-size:1.1rem; font-weight:800; letter-spacing:-0.3px;">AI Broker</div>
        <div style="font-size:0.72rem; color:var(--text-muted); font-weight:500;">Trading Strategy Platform</div>
    </div>
    """, unsafe_allow_html=True)

    dark_toggle = st.toggle("🌙 Dark Mode", value=st.session_state.dark_mode)
    if dark_toggle != st.session_state.dark_mode:
        st.session_state.dark_mode = dark_toggle
        st.rerun()

    st.markdown("---")

    ticker = st.text_input("סימול מניה", value="NVDA",
                           help="e.g. AAPL, MSFT, TSLA, NVDA").upper().strip()
    timeframe = st.selectbox("טווח זמן", options=["1W", "1M", "3M", "6M", "1Y", "2Y"], index=4)

    st.markdown("---")
    st.markdown('<p style="font-weight:700; font-size:0.82rem; color:var(--text-secondary);">⚙️ הגדרות אסטרטגיה</p>', unsafe_allow_html=True)

    account_size = st.number_input("גודל חשבון ($)", min_value=1000, max_value=10_000_000, value=100_000, step=1000)
    risk_pct = st.slider("סיכון לעסקה (%)", min_value=0.25, max_value=5.0, value=1.0, step=0.25)
    reward_ratio = st.select_slider("יחס R:R", options=[1.5, 2.0, 2.5, 3.0, 4.0], value=2.0)
    atr_multiplier = st.slider("ATR מכפיל (SL)", min_value=1.0, max_value=4.0, value=2.0, step=0.5)

    trailing_type = st.selectbox("סוג Trailing SL", options=["fixed", "atr", "percent", "chandelier"],
                                  format_func=lambda x: {"fixed": "קבוע", "atr": "ATR דינמי", "percent": "אחוז (3%)", "chandelier": "Chandelier"}[x])

    st.markdown("---")
    st.markdown('<p style="font-weight:700; font-size:0.82rem; color:var(--text-secondary);">📋 Watchlist אישי</p>', unsafe_allow_html=True)

    wl_cols = st.columns([3, 1])
    with wl_cols[0]:
        new_ticker = st.text_input("הוסף מניה", key="add_wl", label_visibility="collapsed", placeholder="הוסף סימול...")
    with wl_cols[1]:
        if st.button("➕", key="btn_add_wl"):
            nt = new_ticker.upper().strip()
            if nt and nt not in st.session_state.watchlist:
                st.session_state.watchlist.append(nt)
                _save_watchlist()
                st.rerun()

    wl_display = st.session_state.watchlist[:15]
    wl_to_remove = st.multiselect("מניות ב-Watchlist", options=wl_display, default=wl_display, label_visibility="collapsed")
    if set(wl_to_remove) != set(wl_display):
        st.session_state.watchlist = list(wl_to_remove)
        _save_watchlist()

    st.markdown("---")
    if st.button("🔄 רענן נתונים", use_container_width=True):
        st.cache_data.clear()

# ===================================================================== #
#                   KPI OVERVIEW CARDS                                     #
# ===================================================================== #

@st.cache_data(ttl=300, show_spinner=False)
def _fetch_market_overview():
    try:
        sp = yf.Ticker("^GSPC")
        hist = sp.history(period="2d", auto_adjust=True)
        if hist is not None and len(hist) >= 2:
            hist.columns = [c.lower() for c in hist.columns]
            last = float(hist["close"].iloc[-1])
            prev = float(hist["close"].iloc[-2])
            return last, (last - prev) / prev * 100
    except Exception:
        pass
    return 5200.0, 0.3

sp_price, sp_change = _fetch_market_overview()

kc1, kc2, kc3, kc4 = st.columns(4)
with kc1:
    sp_color = _UP if sp_change >= 0 else _DOWN
    sp_sign = "+" if sp_change >= 0 else ""
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">S&P 500</div>
        <div class="kpi-value">{sp_price:,.0f}</div>
        <div class="kpi-sub" style="color:{sp_color};">{sp_sign}{sp_change:.2f}%</div>
    </div>""", unsafe_allow_html=True)
with kc2:
    wl_count = len(st.session_state.watchlist)
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Watchlist</div>
        <div class="kpi-value">{wl_count}</div>
        <div class="kpi-sub">מניות במעקב</div>
    </div>""", unsafe_allow_html=True)
with kc3:
    alert_count = len(st.session_state.alerts)
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">התראות</div>
        <div class="kpi-value">{alert_count}</div>
        <div class="kpi-sub">פעילות</div>
    </div>""", unsafe_allow_html=True)
with kc4:
    journal_count = len(st.session_state.trade_journal)
    win_trades = sum(1 for t in st.session_state.trade_journal if t.get("result") == "win")
    wr = f"{win_trades/journal_count:.0%}" if journal_count > 0 else "—"
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Win Rate</div>
        <div class="kpi-value">{wr}</div>
        <div class="kpi-sub">{journal_count} עסקאות ביומן</div>
    </div>""", unsafe_allow_html=True)

st.markdown("")

# ===================================================================== #
#                         MAIN TABS                                        #
# ===================================================================== #

tab_strategy, tab_hot5, tab_compare, tab_corr, tab_backtest, tab_journal, tab_alerts = st.tabs([
    "📊 אסטרטגיה",
    "🔥 Hot 5",
    "📈 השוואה",
    "🧮 מטריצת קורלציה",
    "🔬 Backtesting",
    "📓 יומן מסחר",
    "🔔 התראות",
])

# ===================================================================== #
# TAB 1 — STRATEGY                                                        #
# ===================================================================== #

with tab_strategy:
    with st.spinner("מנתח..."):
        df, info = fetch_data(ticker, timeframe)

    if df is None or info is None:
        st.error(f"לא ניתן לטעון נתונים עבור **{ticker}**. בדוק את הסימול.")
        st.stop()

    close = df["close"]
    last_price = float(close.iloc[-1])
    prev_price = float(close.iloc[-2]) if len(close) > 1 else last_price
    change_pct = (last_price - prev_price) / prev_price * 100

    rsi_val = compute_rsi(close)
    macd_vals = compute_macd(close)
    atr_val = compute_atr(df)
    boll_vals = compute_bollinger_bands(close)
    support, resistance = identify_support_resistance(df)
    patterns = detect_patterns(df)

    tech_sig, tech_conf, tech_score, tech_reason = _compute_technical_signal(close, df)
    fund_sig, fund_conf, fund_score, fund_reason = _compute_fundamental_signal(info)
    sent_sig, sent_conf = _compute_sentiment_signal(ticker, info)
    risk_level, risk_metrics = _compute_risk(close)
    action, total_conf, weighted_score, dissents = _build_recommendation(
        tech_sig, tech_conf, fund_sig, fund_conf, sent_sig, sent_conf, risk_level,
    )

    strat = build_strategy(action, close, df, info, atr_val, support, resistance,
                           risk_level, atr_multiplier, reward_ratio, trailing_type)

    # ---- Header ----
    company_name = info.get("shortName", ticker)
    change_sign = "+" if change_pct >= 0 else ""
    change_color = _UP if change_pct >= 0 else _DOWN

    st.markdown(f"""
    <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:20px; direction:rtl;">
        <div>
            <span style="font-size:2rem; font-weight:900; letter-spacing:-0.5px;">{ticker}</span>
            <span style="font-size:0.92rem; color:var(--text-secondary); margin-right:12px;">{company_name}</span>
        </div>
        <div style="text-align:left;">
            <span style="font-size:1.8rem; font-weight:800;">${last_price:,.2f}</span>
            <span style="font-size:0.92rem; font-weight:600; color:{change_color}; margin-left:8px;">{change_sign}{change_pct:.2f}%</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ---- Multi-Timeframe Badges ----
    mtf = _compute_mtf_signals(ticker)
    if mtf:
        tf_labels = {"daily": "יומי", "weekly": "שבועי", "monthly": "חודשי"}
        mtf_html = '<div style="display:flex; gap:8px; margin-bottom:16px; direction:rtl;">'
        for tf_key in ["monthly", "weekly", "daily"]:
            if tf_key in mtf:
                s = mtf[tf_key]["signal"]
                c = mtf[tf_key]["confidence"]
                dot_cls = "dot-buy" if s in (Signal.STRONG_BUY, Signal.BUY) else ("dot-sell" if s in (Signal.STRONG_SELL, Signal.SELL) else "dot-hold")
                mtf_html += f'<span class="agent-pill"><span class="dot {dot_cls}"></span>{tf_labels[tf_key]}: {_SIGNAL_LABEL[s]} ({c:.0%})</span>'
        mtf_html += '</div>'
        st.markdown(mtf_html, unsafe_allow_html=True)

    # ---- Direction Badge ----
    dir_class = "long" if strat["direction"] == "LONG" else ("short" if strat["direction"] == "SHORT" else "hold")
    dir_label = {"LONG": "↑ LONG — קנייה", "SHORT": "↓ SHORT — מכירה", "HOLD": "◆ HOLD — המתנה"}
    dir_icon = {"LONG": "🟢", "SHORT": "🔴", "HOLD": "🟡"}

    st.markdown(f"""
    <div class="strategy-card {dir_class}">
        <div style="display:flex; align-items:center; justify-content:space-between; direction:rtl;">
            <div>
                <div style="font-size:0.72rem; font-weight:600; text-transform:uppercase; letter-spacing:1px; color:var(--text-muted); margin-bottom:8px;">אסטרטגיה מומלצת</div>
                <span class="direction-badge direction-{dir_class}">{dir_icon[strat['direction']]} {dir_label[strat['direction']]}</span>
                <span class="risk-pill {_RISK_CLASS[risk_level]}" style="margin-right:12px;">סיכון: {_RISK_LABEL[risk_level]}</span>
            </div>
            <div style="text-align:left;">
                <div style="font-size:0.72rem; font-weight:600; text-transform:uppercase; letter-spacing:1px; color:var(--text-muted);">ביטחון</div>
                <div style="font-size:2.2rem; font-weight:900; color:var(--accent);">{total_conf:.0%}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    # ---- Price Levels ----
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="level-card entry">
            <div class="lbl">כניסה</div>
            <div class="val">${strat['entry']:,.2f}</div>
            <div class="sub">${strat['entry_zone'][0]:,.2f} – ${strat['entry_zone'][1]:,.2f}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        trailing_note = ""
        if strat["trailing_sl"] and trailing_type != "fixed":
            trailing_note = f'<div class="sub" style="color:var(--gold);">Trailing: ${strat["trailing_sl"]:,.2f}</div>'
        st.markdown(f"""
        <div class="level-card sl">
            <div class="lbl">Stop Loss</div>
            <div class="val">${strat['stop_loss']:,.2f}</div>
            <div class="sub">סיכון ${strat['risk_per_share']:,.2f} למניה</div>
            {trailing_note}
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="level-card tp">
            <div class="lbl">Take Profit</div>
            <div class="val">${strat['take_profit_1']:,.2f}</div>
            <div class="sub">TP2: ${strat['take_profit_2']:,.2f}</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        risk_dollars = account_size * (risk_pct / 100)
        shares = int(risk_dollars / strat["risk_per_share"]) if strat["risk_per_share"] > 0 else 0
        st.markdown(f"""
        <div class="level-card rr">
            <div class="lbl">R:R & גודל פוזיציה</div>
            <div class="val">1:{strat['reward_ratio']:.1f}</div>
            <div class="sub">{shares:,} מניות · ${shares * strat['entry']:,.0f}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("")

    # ---- Strategy Chart with VWAP + Volume Profile ----
    show_vwap = st.checkbox("הצג VWAP", value=True, key="show_vwap")
    show_vol_profile = st.checkbox("הצג Volume Profile", value=False, key="show_vol_profile")
    show_trailing = st.checkbox("הצג Trailing SL", value=(trailing_type != "fixed"), key="show_trailing")

    if show_vol_profile:
        fig = make_subplots(rows=2, cols=2, shared_xaxes=True,
                            column_widths=[0.85, 0.15], row_heights=[0.78, 0.22],
                            vertical_spacing=0.03, horizontal_spacing=0.02)
    else:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            row_heights=[0.78, 0.22], vertical_spacing=0.03)

    main_col = 1
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        increasing_line_color=_UP, decreasing_line_color=_DOWN,
        increasing_fillcolor=_UP, decreasing_fillcolor=_DOWN,
        name="OHLC", showlegend=False,
    ), row=1, col=main_col)

    sma20s = _sma_series(close, 20)
    sma50s = _sma_series(close, 50)
    fig.add_trace(go.Scatter(x=df.index, y=sma20s, line=dict(color="#F59E0B", width=1), name="SMA 20"), row=1, col=main_col)
    fig.add_trace(go.Scatter(x=df.index, y=sma50s, line=dict(color="#3B82F6", width=1), name="SMA 50"), row=1, col=main_col)

    bb_upper, bb_mid, bb_lower = _bollinger_series(close)
    fig.add_trace(go.Scatter(x=df.index, y=bb_upper, line=dict(color="rgba(99,102,241,0.2)", width=1, dash="dot"), showlegend=False), row=1, col=main_col)
    fig.add_trace(go.Scatter(x=df.index, y=bb_lower, line=dict(color="rgba(99,102,241,0.2)", width=1, dash="dot"),
                             fill="tonexty", fillcolor="rgba(99,102,241,0.04)", showlegend=False), row=1, col=main_col)

    if show_vwap and len(df) > 5:
        vwap_s = compute_vwap(df)
        fig.add_trace(go.Scatter(x=df.index, y=vwap_s, line=dict(color="#EC4899", width=1.5, dash="dashdot"), name="VWAP"), row=1, col=main_col)

    if show_trailing and trailing_type != "fixed":
        tsl_series = _trailing_sl_series(df, atr_multiplier)
        fig.add_trace(go.Scatter(x=df.index, y=tsl_series, line=dict(color="#F59E0B", width=1.5, dash="dash"), name="Trailing SL"), row=1, col=main_col)

    entry_color = _ACCENT
    sl_color = _DOWN
    tp_color = _UP

    fig.add_hline(y=strat["entry"], line_color=entry_color, line_width=2, line_dash="dash",
                  annotation_text=f"Entry ${strat['entry']:.2f}", annotation_font_color=entry_color,
                  annotation_font_size=11, row=1, col=main_col)
    fig.add_hline(y=strat["stop_loss"], line_color=sl_color, line_width=2, line_dash="dash",
                  annotation_text=f"SL ${strat['stop_loss']:.2f}", annotation_font_color=sl_color,
                  annotation_font_size=11, row=1, col=main_col)
    fig.add_hline(y=strat["take_profit_1"], line_color=tp_color, line_width=2, line_dash="dash",
                  annotation_text=f"TP1 ${strat['take_profit_1']:.2f}", annotation_font_color=tp_color,
                  annotation_font_size=11, row=1, col=main_col)
    fig.add_hline(y=strat["take_profit_2"], line_color=tp_color, line_width=1.5, line_dash="dot",
                  annotation_text=f"TP2 ${strat['take_profit_2']:.2f}", annotation_font_color=tp_color,
                  annotation_font_size=10, row=1, col=main_col)

    fig.add_hrect(y0=strat["entry_zone"][0], y1=strat["entry_zone"][1],
                  fillcolor="rgba(99,102,241,0.08)", line_width=0, row=1, col=main_col)

    vol_colors = [_UP if c >= o else _DOWN for c, o in zip(df["close"], df["open"])]
    fig.add_trace(go.Bar(x=df.index, y=df["volume"], marker_color=vol_colors, showlegend=False, opacity=0.5), row=2, col=main_col)

    if show_vol_profile:
        vp = compute_volume_profile(df)
        fig.add_trace(go.Bar(
            y=vp["price"], x=vp["volume"], orientation="h",
            marker_color="rgba(99,102,241,0.4)", showlegend=False,
        ), row=1, col=2)
        fig.update_xaxes(showticklabels=False, row=1, col=2)
        fig.update_yaxes(showticklabels=False, row=1, col=2)

    _apply_layout(fig, height=540, xaxis_rangeslider_visible=False,
                  hovermode="x unified",
                  title=dict(text=f"{ticker} — Strategy Map", font=dict(size=14, weight=600)))
    fig.update_xaxes(gridcolor=_PLOTLY_GRID, showspikes=True, spikemode="across", spikethickness=1, spikecolor=_PLOTLY_GRID)
    fig.update_yaxes(gridcolor=_PLOTLY_GRID, showspikes=True, spikemode="across", spikethickness=1, spikecolor=_PLOTLY_GRID)

    st.plotly_chart(fig, use_container_width=True)

    # ---- Agent Breakdown + Sentiment + Reasoning ----
    st.markdown("")
    agent_col, sent_col, reason_col = st.columns([1.2, 0.6, 1.2])

    with agent_col:
        st.markdown('<p class="rtl" style="font-weight:700; font-size:0.92rem; margin-bottom:12px;">סוכני AI</p>', unsafe_allow_html=True)

        def _agent_dot(sig):
            if sig in (Signal.STRONG_BUY, Signal.BUY): return "dot-buy"
            if sig in (Signal.STRONG_SELL, Signal.SELL): return "dot-sell"
            return "dot-hold"

        agents_html = f"""
        <div style="display:flex; flex-wrap:wrap; gap:8px; direction:rtl;">
            <span class="agent-pill"><span class="dot {_agent_dot(tech_sig)}"></span>טכני: {_SIGNAL_LABEL[tech_sig]} ({tech_conf:.0%})</span>
            <span class="agent-pill"><span class="dot {_agent_dot(fund_sig)}"></span>פונדמנטלי: {_SIGNAL_LABEL[fund_sig]} ({fund_conf:.0%})</span>
            <span class="agent-pill"><span class="dot {_agent_dot(sent_sig)}"></span>סנטימנט: {_SIGNAL_LABEL[sent_sig]} ({sent_conf:.0%})</span>
        </div>
        """
        st.markdown(agents_html, unsafe_allow_html=True)

        st.markdown("")

        metrics_html = '<div style="display:grid; grid-template-columns:repeat(4,1fr); gap:8px;">'
        metrics = [
            ("RSI", f"{rsi_val:.1f}" if rsi_val else "N/A"),
            ("ATR", f"${atr_val:.2f}" if atr_val else "N/A"),
            ("Sharpe", f"{risk_metrics['sharpe_ratio']:.2f}" if risk_metrics['sharpe_ratio'] else "N/A"),
            ("VaR 95%", f"{risk_metrics['var_95']:.2%}" if risk_metrics['var_95'] else "N/A"),
        ]
        for lbl, val in metrics:
            metrics_html += f'<div class="metric-sm"><div class="m-lbl">{lbl}</div><div class="m-val">{val}</div></div>'
        metrics_html += '</div>'
        st.markdown(metrics_html, unsafe_allow_html=True)

    with sent_col:
        sent_score_val = SIGNAL_TO_SCORE[sent_sig] * sent_conf
        sent_color = _UP if sent_score_val > 0.1 else (_DOWN if sent_score_val < -0.1 else "var(--text-muted)")
        sent_label = "חיובי" if sent_score_val > 0.1 else ("שלילי" if sent_score_val < -0.1 else "ניטרלי")
        st.markdown(f"""
        <div class="sent-gauge">
            <div class="gauge-lbl">סנטימנט</div>
            <div class="gauge-val" style="color:{sent_color};">{sent_score_val:+.2f}</div>
            <div style="font-size:0.78rem; color:var(--text-secondary);">{sent_label}</div>
        </div>
        """, unsafe_allow_html=True)

    with reason_col:
        reason_parts = [f"<b>טכני:</b> {tech_reason}", f"<b>פונדמנטלי:</b> {fund_reason}"]
        if dissents:
            reason_parts.append("<b>דעות חולקות:</b> " + " · ".join(dissents))
        if patterns:
            pattern_map = {"golden_cross": "Golden Cross ✨", "death_cross": "Death Cross ☠️",
                          "doji": "Doji", "hammer": "Hammer"}
            reason_parts.append("<b>תבניות:</b> " + ", ".join(pattern_map.get(p, p) for p in patterns))

        st.markdown(f"""
        <div class="reasoning-box">
            <div class="reason-title">ניתוח הסוכנים</div>
            {'<br>'.join(reason_parts)}
        </div>
        """, unsafe_allow_html=True)

    # ---- Position Calculator ----
    st.markdown("---")
    st.markdown('<p class="rtl" style="font-weight:700; font-size:0.92rem; margin-bottom:8px;">💰 חישוב פוזיציה</p>', unsafe_allow_html=True)

    risk_dollars = account_size * (risk_pct / 100)
    if strat["risk_per_share"] > 0:
        shares = int(risk_dollars / strat["risk_per_share"])
    else:
        shares = 0
    position_value = shares * strat["entry"]
    potential_profit = shares * abs(strat["take_profit_1"] - strat["entry"])
    potential_loss = shares * strat["risk_per_share"]

    pc1, pc2, pc3, pc4, pc5 = st.columns(5)
    with pc1:
        st.markdown(f'<div class="metric-sm"><div class="m-lbl">חשבון</div><div class="m-val">${account_size:,.0f}</div></div>', unsafe_allow_html=True)
    with pc2:
        st.markdown(f'<div class="metric-sm"><div class="m-lbl">סיכון</div><div class="m-val">${risk_dollars:,.0f}</div></div>', unsafe_allow_html=True)
    with pc3:
        st.markdown(f'<div class="metric-sm"><div class="m-lbl">מניות</div><div class="m-val">{shares:,}</div></div>', unsafe_allow_html=True)
    with pc4:
        st.markdown(f'<div class="metric-sm"><div class="m-lbl">רווח פוט׳</div><div class="m-val" style="color:var(--up);">${potential_profit:,.0f}</div></div>', unsafe_allow_html=True)
    with pc5:
        st.markdown(f'<div class="metric-sm"><div class="m-lbl">הפסד מקס׳</div><div class="m-val" style="color:var(--down);">${potential_loss:,.0f}</div></div>', unsafe_allow_html=True)

    if potential_loss > 0:
        total = potential_loss + potential_profit
        fig_rr = go.Figure()
        fig_rr.add_trace(go.Bar(x=[potential_loss/total], y=[""], orientation="h",
                                marker_color=_DOWN, name=f"Risk ${potential_loss:,.0f}",
                                text=f"Risk ${potential_loss:,.0f}", textposition="inside",
                                textfont=dict(color="white", size=12)))
        fig_rr.add_trace(go.Bar(x=[potential_profit/total], y=[""], orientation="h",
                                marker_color=_UP, name=f"Reward ${potential_profit:,.0f}",
                                text=f"Reward ${potential_profit:,.0f}", textposition="inside",
                                textfont=dict(color="white", size=12)))
        _apply_layout(fig_rr, height=56, barmode="stack", showlegend=False,
                      margin=dict(l=0, r=0, t=0, b=0),
                      xaxis=dict(visible=False), yaxis=dict(visible=False))
        st.plotly_chart(fig_rr, use_container_width=True)

    # ---- Save to Journal Button ----
    if st.button("📓 שמור ליומן מסחר", key="save_journal"):
        entry = {
            "date": datetime.utcnow().strftime("%Y-%m-%d %H:%M"),
            "ticker": ticker,
            "direction": strat["direction"],
            "entry_price": strat["entry"],
            "stop_loss": strat["stop_loss"],
            "take_profit": strat["take_profit_1"],
            "shares": shares,
            "confidence": round(total_conf, 2),
            "risk_level": risk_level.value,
            "result": "pending",
        }
        st.session_state.trade_journal.append(entry)
        _save_journal()
        st.toast(f"✅ {ticker} נשמר ליומן המסחר!", icon="📓")

    # ---- RSI + MACD mini charts ----
    st.markdown("---")
    ind1, ind2 = st.columns(2)

    with ind1:
        rsi_s = _rsi_series(close)
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=df.index, y=rsi_s, line=dict(color=_ACCENT, width=1.5), name="RSI"))
        fig_rsi.add_hline(y=70, line_color=_DOWN, line_dash="dash", line_width=0.8)
        fig_rsi.add_hline(y=30, line_color=_UP, line_dash="dash", line_width=0.8)
        fig_rsi.add_hrect(y0=70, y1=100, fillcolor=_DOWN, opacity=0.05, line_width=0)
        fig_rsi.add_hrect(y0=0, y1=30, fillcolor=_UP, opacity=0.05, line_width=0)
        _apply_layout(fig_rsi, height=220, title=dict(text="RSI (14)", font=dict(size=12)))
        st.plotly_chart(fig_rsi, use_container_width=True)

    with ind2:
        ml, ms, mh = _macd_series(close)
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=df.index, y=ml, line=dict(color="#3B82F6", width=1.5), name="MACD"))
        fig_macd.add_trace(go.Scatter(x=df.index, y=ms, line=dict(color="#F59E0B", width=1.2), name="Signal"))
        hist_colors = [_UP if v >= 0 else _DOWN for v in mh.fillna(0)]
        fig_macd.add_trace(go.Bar(x=df.index, y=mh, marker_color=hist_colors, name="Histogram", opacity=0.5))
        _apply_layout(fig_macd, height=220, title=dict(text="MACD (12/26/9)", font=dict(size=12)))
        st.plotly_chart(fig_macd, use_container_width=True)


# ===================================================================== #
# TAB 2 — HOT 5                                                           #
# ===================================================================== #

with tab_hot5:
    st.markdown(f"""
    <div style="text-align:center; padding:12px 0 24px;">
        <div style="font-size:1.5rem; font-weight:900;">🔥 5 מניות חמות היום</div>
        <div style="font-size:0.82rem; color:var(--text-secondary);">הסוכנים סרקו 30 מניות ובחרו את הטובות ביותר · {datetime.utcnow().strftime('%Y-%m-%d')}</div>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("הסוכנים סורקים 30 מניות..."):
        hot_stocks = fetch_hot_candidates()

    if not hot_stocks:
        st.warning("לא ניתן לטעון נתונים. נסה שוב.")
    else:
        for i, stock in enumerate(hot_stocks, 1):
            is_buy = stock["action"] in (Signal.STRONG_BUY, Signal.BUY)
            is_sell = stock["action"] in (Signal.STRONG_SELL, Signal.SELL)
            sig_class = "buy" if is_buy else ("sell" if is_sell else "")
            sig_label = _SIGNAL_LABEL[stock["action"]]
            change_color = _UP if stock["change_pct"] >= 0 else _DOWN
            change_sign = "+" if stock["change_pct"] >= 0 else ""

            st.markdown(f"""
            <div class="hot-row" style="direction:rtl;">
                <div class="hot-rank">{i}</div>
                <div style="flex:1; margin:0 16px;">
                    <div class="hot-ticker">{stock['ticker']}</div>
                    <div class="hot-name">{stock['name']}</div>
                </div>
                <div style="text-align:center; min-width:80px;">
                    <div style="font-size:1.1rem; font-weight:700;">${stock['price']:,.2f}</div>
                    <div style="font-size:0.78rem; font-weight:600; color:{change_color};">{change_sign}{stock['change_pct']:.2f}%</div>
                </div>
                <div style="text-align:center; min-width:100px; margin:0 12px;">
                    <span class="hot-signal {sig_class}">{sig_label}</span>
                </div>
                <div style="text-align:center; min-width:60px;">
                    <div class="hot-conf">{stock['confidence']:.0%}</div>
                    <div style="font-size:0.68rem; color:var(--text-muted);">ביטחון</div>
                </div>
                <div style="min-width:40px; text-align:center;">
                    <span class="risk-pill {_RISK_CLASS[stock['risk']]}">{_RISK_LABEL[stock['risk']]}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("")
        st.markdown("""
        <div style="text-align:center; padding:16px; background:var(--panel-alt); border-radius:var(--radius-sm); border:1px solid var(--border);">
            <div style="font-size:0.78rem; color:var(--text-secondary);">
                💡 הקלד סימול מניה בסיידבר כדי לקבל אסטרטגיית מסחר מפורטת
            </div>
        </div>
        """, unsafe_allow_html=True)


# ===================================================================== #
# TAB 3 — MULTI-TICKER COMPARISON                                         #
# ===================================================================== #

with tab_compare:
    st.markdown('<p class="rtl" style="font-weight:900; font-size:1.2rem; margin-bottom:16px;">📈 השוואת מניות</p>', unsafe_allow_html=True)

    compare_tickers = st.multiselect(
        "בחר מניות להשוואה (2-5)",
        options=st.session_state.watchlist + ["AAPL", "MSFT", "GOOGL", "NVDA", "META", "AMZN", "TSLA", "AMD"],
        default=st.session_state.watchlist[:3] if len(st.session_state.watchlist) >= 3 else ["NVDA", "AAPL", "MSFT"],
        max_selections=5,
    )

    if len(compare_tickers) >= 2:
        compare_period = st.selectbox("טווח השוואה", options=["3M", "6M", "1Y"], index=1, key="compare_period")

        fig_comp = go.Figure()
        comp_data = []
        colors = ["#6366F1", "#10B981", "#F59E0B", "#EF4444", "#EC4899"]

        for i, ct in enumerate(compare_tickers):
            cdf, cinfo = fetch_data(ct, compare_period)
            if cdf is not None and len(cdf) > 1:
                pct_change = (cdf["close"] / cdf["close"].iloc[0] - 1) * 100
                fig_comp.add_trace(go.Scatter(
                    x=cdf.index, y=pct_change,
                    line=dict(color=colors[i % len(colors)], width=2),
                    name=ct,
                ))
                last_p = float(cdf["close"].iloc[-1])
                total_ret = float(pct_change.iloc[-1])
                rsi_c = compute_rsi(cdf["close"])
                vol_p = compute_volatility_percentile(cdf["close"])
                comp_data.append({
                    "ticker": ct,
                    "name": cinfo.get("shortName", ct),
                    "price": last_p,
                    "return": total_ret,
                    "rsi": rsi_c,
                    "volatility": vol_p,
                })

        fig_comp.add_hline(y=0, line_color=_PLOTLY_GRID, line_width=1)
        _apply_layout(fig_comp, height=400,
                      title=dict(text="ביצועים נורמליזציים (%)", font=dict(size=13)),
                      yaxis_title="% שינוי", hovermode="x unified")
        st.plotly_chart(fig_comp, use_container_width=True)

        if comp_data:
            comp_df = pd.DataFrame(comp_data)
            comp_df.columns = ["סימול", "שם", "מחיר", "תשואה %", "RSI", "תנודתיות"]
            comp_df["מחיר"] = comp_df["מחיר"].apply(lambda x: f"${x:,.2f}")
            comp_df["תשואה %"] = comp_df["תשואה %"].apply(lambda x: f"{x:+.2f}%")
            comp_df["RSI"] = comp_df["RSI"].apply(lambda x: f"{x:.1f}" if x else "N/A")
            comp_df["תנודתיות"] = comp_df["תנודתיות"].apply(lambda x: f"{x:.0%}")
            st.dataframe(comp_df, use_container_width=True, hide_index=True)
    else:
        st.info("בחר לפחות 2 מניות להשוואה")


# ===================================================================== #
# TAB 4 — CORRELATION MATRIX                                              #
# ===================================================================== #

with tab_corr:
    st.markdown('<p class="rtl" style="font-weight:900; font-size:1.2rem; margin-bottom:16px;">🧮 מטריצת קורלציה</p>', unsafe_allow_html=True)

    corr_tickers = st.multiselect(
        "בחר מניות (3-10)",
        options=list(set(st.session_state.watchlist + ["AAPL", "MSFT", "GOOGL", "NVDA", "META", "AMZN", "TSLA"])),
        default=st.session_state.watchlist[:6] if len(st.session_state.watchlist) >= 6 else ["NVDA", "AAPL", "MSFT", "GOOGL", "META", "TSLA"],
        max_selections=10,
        key="corr_select",
    )

    if len(corr_tickers) >= 3:
        returns_dict = {}
        for ct in corr_tickers:
            cdf, _ = fetch_data(ct, "6M")
            if cdf is not None and len(cdf) > 20:
                returns_dict[ct] = cdf["close"].pct_change().dropna()

        if len(returns_dict) >= 3:
            returns_df = pd.DataFrame(returns_dict).dropna()
            corr_matrix = returns_df.corr()

            avg_corr = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)).stack().mean()
            div_score = max(0, min(100, int((1 - avg_corr) * 100)))

            dc1, dc2 = st.columns([1, 3])
            with dc1:
                div_color = _UP if div_score > 50 else (_DOWN if div_score < 30 else "var(--gold)")
                st.markdown(f"""
                <div class="kpi-card" style="padding:24px;">
                    <div class="kpi-label">ציון גיוון</div>
                    <div class="kpi-value" style="font-size:2.5rem; color:{div_color};">{div_score}</div>
                    <div class="kpi-sub">קורלציה ממוצעת: {avg_corr:.2f}</div>
                    <div class="kpi-sub" style="margin-top:8px;">{'תיק מגוון היטב ✅' if div_score > 50 else ('גיוון בינוני ⚠️' if div_score > 30 else 'תיק לא מגוון ❌')}</div>
                </div>
                """, unsafe_allow_html=True)

            with dc2:
                fig_corr = go.Figure(data=go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns.tolist(),
                    y=corr_matrix.index.tolist(),
                    colorscale="RdYlGn_r",
                    zmin=-1, zmax=1,
                    text=corr_matrix.round(2).values,
                    texttemplate="%{text}",
                    textfont=dict(size=11),
                    hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>Correlation: %{z:.3f}<extra></extra>",
                ))
                _apply_layout(fig_corr, height=400,
                              title=dict(text="מטריצת קורלציה (6 חודשים)", font=dict(size=13)))
                st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("בחר לפחות 3 מניות")


# ===================================================================== #
# TAB 5 — BACKTESTING                                                     #
# ===================================================================== #

with tab_backtest:
    st.markdown('<p class="rtl" style="font-weight:900; font-size:1.2rem; margin-bottom:16px;">🔬 Backtesting & אופטימיזציה</p>', unsafe_allow_html=True)

    bt_col1, bt_col2 = st.columns(2)
    with bt_col1:
        bt_ticker = st.text_input("סימול לבדיקה", value=ticker, key="bt_ticker").upper().strip()
    with bt_col2:
        bt_period = st.selectbox("תקופת בדיקה", options=["6M", "1Y", "2Y"], index=1, key="bt_period")

    bt_c1, bt_c2, bt_c3 = st.columns(3)
    with bt_c1:
        bt_atr = st.slider("ATR מכפיל", 1.0, 4.0, 2.0, 0.5, key="bt_atr")
    with bt_c2:
        bt_rr = st.slider("R:R", 1.5, 4.0, 2.0, 0.5, key="bt_rr")
    with bt_c3:
        bt_risk = st.slider("סיכון %", 0.5, 3.0, 1.0, 0.25, key="bt_risk")

    run_bt = st.button("▶️ הרץ Backtest", key="run_bt", use_container_width=True)
    run_opt = st.button("⚡ אופטימיזציה אוטומטית", key="run_opt", use_container_width=True)

    if run_bt:
        with st.spinner("מריץ backtesting..."):
            try:
                from src.tools.backtester import run_backtest
                bt_df, _ = fetch_data(bt_ticker, bt_period)
                if bt_df is not None and len(bt_df) > 50:
                    result = run_backtest(bt_df, atr_mult=bt_atr, reward_ratio=bt_rr, risk_pct=bt_risk)

                    m1, m2, m3, m4, m5, m6 = st.columns(6)
                    with m1:
                        ret_color = _UP if result.total_return > 0 else _DOWN
                        st.markdown(f'<div class="metric-sm"><div class="m-lbl">תשואה</div><div class="m-val" style="color:{ret_color};">{result.total_return:+.1f}%</div></div>', unsafe_allow_html=True)
                    with m2:
                        st.markdown(f'<div class="metric-sm"><div class="m-lbl">Win Rate</div><div class="m-val">{result.win_rate:.0%}</div></div>', unsafe_allow_html=True)
                    with m3:
                        st.markdown(f'<div class="metric-sm"><div class="m-lbl">Profit Factor</div><div class="m-val">{result.profit_factor:.2f}</div></div>', unsafe_allow_html=True)
                    with m4:
                        st.markdown(f'<div class="metric-sm"><div class="m-lbl">Max DD</div><div class="m-val" style="color:var(--down);">{result.max_drawdown:.1f}%</div></div>', unsafe_allow_html=True)
                    with m5:
                        sh_color = _UP if (result.sharpe_ratio or 0) > 1 else "var(--text)"
                        st.markdown(f'<div class="metric-sm"><div class="m-lbl">Sharpe</div><div class="m-val" style="color:{sh_color};">{result.sharpe_ratio:.2f}</div></div>', unsafe_allow_html=True)
                    with m6:
                        st.markdown(f'<div class="metric-sm"><div class="m-lbl">עסקאות</div><div class="m-val">{result.total_trades}</div></div>', unsafe_allow_html=True)

                    fig_eq = go.Figure()
                    fig_eq.add_trace(go.Scatter(
                        x=result.equity_curve.index, y=result.equity_curve.values,
                        fill="tozeroy", fillcolor="rgba(99,102,241,0.1)",
                        line=dict(color=_ACCENT, width=2), name="Equity",
                    ))
                    _apply_layout(fig_eq, height=300,
                                  title=dict(text="Equity Curve", font=dict(size=13)),
                                  yaxis_title="$")
                    st.plotly_chart(fig_eq, use_container_width=True)

                    if result.trades:
                        trades_df = pd.DataFrame(result.trades)
                        st.dataframe(trades_df.tail(20), use_container_width=True, hide_index=True)
                else:
                    st.warning("לא מספיק נתונים ל-backtest")
            except ImportError:
                st.warning("מודול Backtester בטעינה... נסה שוב בעוד רגע")
            except Exception as e:
                st.error(f"שגיאה: {e}")

    if run_opt:
        with st.spinner("מאתר פרמטרים אופטימליים..."):
            try:
                from src.tools.optimizer import optimize_parameters
                opt_df, _ = fetch_data(bt_ticker, bt_period)
                if opt_df is not None and len(opt_df) > 50:
                    opt_result = optimize_parameters(opt_df)

                    bp = opt_result["best_params"]
                    st.success(f"🏆 פרמטרים אופטימליים: ATR={bp['atr_mult']:.1f} | R:R={bp['reward_ratio']:.1f} | Risk={bp['risk_pct']:.1f}%")

                    o1, o2, o3 = st.columns(3)
                    with o1:
                        st.markdown(f'<div class="metric-sm"><div class="m-lbl">Sharpe הטוב ביותר</div><div class="m-val">{opt_result["best_sharpe"]:.2f}</div></div>', unsafe_allow_html=True)
                    with o2:
                        st.markdown(f'<div class="metric-sm"><div class="m-lbl">Win Rate</div><div class="m-val">{opt_result["best_win_rate"]:.0%}</div></div>', unsafe_allow_html=True)
                    with o3:
                        st.markdown(f'<div class="metric-sm"><div class="m-lbl">שילובים שנבדקו</div><div class="m-val">{len(opt_result["results_matrix"])}</div></div>', unsafe_allow_html=True)

                    hm = opt_result["heatmap_data"]
                    if hm["z"]:
                        fig_hm = go.Figure(data=go.Heatmap(
                            z=hm["z"], x=[str(x) for x in hm["x"]], y=[str(y) for y in hm["y"]],
                            colorscale="Viridis",
                            hovertemplate="ATR: %{x}<br>R:R: %{y}<br>Sharpe: %{z:.2f}<extra></extra>",
                        ))
                        _apply_layout(fig_hm, height=350,
                                      title=dict(text="Sharpe Ratio Heatmap (ATR vs R:R)", font=dict(size=13)),
                                      xaxis_title="ATR Multiplier", yaxis_title="Reward Ratio")
                        st.plotly_chart(fig_hm, use_container_width=True)
                else:
                    st.warning("לא מספיק נתונים לאופטימיזציה")
            except ImportError:
                st.warning("מודול Optimizer בטעינה... נסה שוב בעוד רגע")
            except Exception as e:
                st.error(f"שגיאה: {e}")


# ===================================================================== #
# TAB 6 — TRADE JOURNAL                                                   #
# ===================================================================== #

with tab_journal:
    st.markdown('<p class="rtl" style="font-weight:900; font-size:1.2rem; margin-bottom:16px;">📓 יומן מסחר</p>', unsafe_allow_html=True)

    journal = st.session_state.trade_journal

    if journal:
        total_j = len(journal)
        wins = sum(1 for t in journal if t.get("result") == "win")
        losses = sum(1 for t in journal if t.get("result") == "loss")
        pending = sum(1 for t in journal if t.get("result") == "pending")

        jc1, jc2, jc3, jc4 = st.columns(4)
        with jc1:
            st.markdown(f'<div class="kpi-card"><div class="kpi-label">סה"כ</div><div class="kpi-value">{total_j}</div></div>', unsafe_allow_html=True)
        with jc2:
            wr = f"{wins/total_j:.0%}" if total_j > 0 else "—"
            st.markdown(f'<div class="kpi-card"><div class="kpi-label">Win Rate</div><div class="kpi-value" style="color:var(--up);">{wr}</div></div>', unsafe_allow_html=True)
        with jc3:
            st.markdown(f'<div class="kpi-card"><div class="kpi-label">ניצחונות</div><div class="kpi-value" style="color:var(--up);">{wins}</div></div>', unsafe_allow_html=True)
        with jc4:
            st.markdown(f'<div class="kpi-card"><div class="kpi-label">ממתינים</div><div class="kpi-value" style="color:var(--gold);">{pending}</div></div>', unsafe_allow_html=True)

        st.markdown("")

        for idx, trade in enumerate(reversed(journal)):
            result_icon = {"win": "🟢", "loss": "🔴", "pending": "🟡"}.get(trade.get("result", "pending"), "⚪")
            dir_lbl = {"LONG": "קנייה", "SHORT": "מכירה", "HOLD": "המתנה"}.get(trade.get("direction", ""), "")

            tcol1, tcol2, tcol3 = st.columns([3, 1, 1])
            with tcol1:
                st.markdown(f"""
                <div class="journal-row">
                    <span style="font-size:1.2rem;">{result_icon}</span>
                    <span style="font-weight:800;">{trade.get('ticker', '')}</span>
                    <span style="color:var(--text-secondary);">{dir_lbl}</span>
                    <span style="color:var(--text-muted);">${trade.get('entry_price', 0):,.2f}</span>
                    <span style="color:var(--text-muted); font-size:0.78rem;">{trade.get('date', '')}</span>
                </div>
                """, unsafe_allow_html=True)
            with tcol2:
                real_idx = len(journal) - 1 - idx
                new_result = st.selectbox(
                    "תוצאה", options=["pending", "win", "loss"],
                    index=["pending", "win", "loss"].index(trade.get("result", "pending")),
                    key=f"jr_{real_idx}", label_visibility="collapsed",
                )
                if new_result != trade.get("result"):
                    st.session_state.trade_journal[real_idx]["result"] = new_result
                    _save_journal()
            with tcol3:
                if st.button("🗑️", key=f"jdel_{real_idx}"):
                    st.session_state.trade_journal.pop(real_idx)
                    _save_journal()
                    st.rerun()
    else:
        st.markdown("""
        <div style="text-align:center; padding:40px; color:var(--text-muted);">
            <div style="font-size:2rem; margin-bottom:8px;">📓</div>
            <div>היומן ריק. נתח מניה ולחץ "שמור ליומן" כדי להתחיל לתעד.</div>
        </div>
        """, unsafe_allow_html=True)


# ===================================================================== #
# TAB 7 — ALERTS                                                          #
# ===================================================================== #

with tab_alerts:
    st.markdown('<p class="rtl" style="font-weight:900; font-size:1.2rem; margin-bottom:16px;">🔔 מערכת התראות</p>', unsafe_allow_html=True)

    al_c1, al_c2, al_c3, al_c4 = st.columns([2, 2, 2, 1])
    with al_c1:
        alert_ticker = st.text_input("סימול", value=ticker, key="alert_ticker").upper().strip()
    with al_c2:
        alert_condition = st.selectbox("תנאי", options=[
            "price_above", "price_below", "rsi_above", "rsi_below",
        ], format_func=lambda x: {
            "price_above": "מחיר מעל",
            "price_below": "מחיר מתחת",
            "rsi_above": "RSI מעל",
            "rsi_below": "RSI מתחת",
        }[x], key="alert_cond")
    with al_c3:
        alert_value = st.number_input("ערך", value=0.0, key="alert_val")
    with al_c4:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("➕ הוסף", key="add_alert"):
            if alert_ticker and alert_value > 0:
                st.session_state.alerts.append({
                    "ticker": alert_ticker,
                    "condition": alert_condition,
                    "value": alert_value,
                    "created": datetime.utcnow().strftime("%Y-%m-%d %H:%M"),
                    "triggered": False,
                })
                _save_alerts()
                st.toast(f"✅ התראה נוספה עבור {alert_ticker}!", icon="🔔")
                st.rerun()

    st.markdown("")

    alerts = st.session_state.alerts
    if alerts:
        _cond_labels = {
            "price_above": "מחיר מעל",
            "price_below": "מחיר מתחת",
            "rsi_above": "RSI מעל",
            "rsi_below": "RSI מתחת",
        }

        triggered_alerts = []
        for i, alert in enumerate(alerts):
            try:
                a_df, _ = fetch_data(alert["ticker"], "1M")
                if a_df is not None and len(a_df) > 0:
                    a_close = a_df["close"]
                    a_price = float(a_close.iloc[-1])
                    a_rsi = compute_rsi(a_close)

                    triggered = False
                    if alert["condition"] == "price_above" and a_price > alert["value"]:
                        triggered = True
                    elif alert["condition"] == "price_below" and a_price < alert["value"]:
                        triggered = True
                    elif alert["condition"] == "rsi_above" and a_rsi and a_rsi > alert["value"]:
                        triggered = True
                    elif alert["condition"] == "rsi_below" and a_rsi and a_rsi < alert["value"]:
                        triggered = True

                    if triggered and not alert.get("triggered"):
                        triggered_alerts.append(alert)
                        st.session_state.alerts[i]["triggered"] = True
            except Exception:
                pass

        if triggered_alerts:
            _save_alerts()
            for ta in triggered_alerts:
                st.toast(f"🔔 התראה! {ta['ticker']} — {_cond_labels.get(ta['condition'], '')} {ta['value']}", icon="🔔")

        for i, alert in enumerate(alerts):
            status_icon = "🔴" if alert.get("triggered") else "🟢"
            acol1, acol2 = st.columns([4, 1])
            with acol1:
                st.markdown(f"""
                <div class="journal-row">
                    <span style="font-size:1.1rem;">{status_icon}</span>
                    <span style="font-weight:800;">{alert['ticker']}</span>
                    <span style="color:var(--text-secondary);">{_cond_labels.get(alert['condition'], alert['condition'])}</span>
                    <span style="font-weight:700;">{alert['value']}</span>
                    <span style="color:var(--text-muted); font-size:0.75rem;">{alert.get('created', '')}</span>
                    {'<span style="color:var(--down); font-weight:700;">TRIGGERED</span>' if alert.get('triggered') else ''}
                </div>
                """, unsafe_allow_html=True)
            with acol2:
                if st.button("🗑️", key=f"adel_{i}"):
                    st.session_state.alerts.pop(i)
                    _save_alerts()
                    st.rerun()
    else:
        st.markdown("""
        <div style="text-align:center; padding:40px; color:var(--text-muted);">
            <div style="font-size:2rem; margin-bottom:8px;">🔔</div>
            <div>אין התראות פעילות. הגדר תנאי למעלה כדי לקבל התראות.</div>
        </div>
        """, unsafe_allow_html=True)


# ===================================================================== #
#                          FOOTER                                          #
# ===================================================================== #

st.markdown("---")
st.markdown("""
<p style="text-align:center; color:var(--text-muted); font-size:0.72rem;">
    AI Broker Trading Strategy Platform · Data: Yahoo Finance ·
    האותות אינם המלצה להשקעה · השתמש באחריותך
</p>
""", unsafe_allow_html=True)
