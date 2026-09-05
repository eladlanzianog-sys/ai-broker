"""AI Broker — Trading Strategy Platform.

Enter a ticker, get a full trading strategy: Long/Short direction,
entry zone, stop loss, take profit, position sizing — all with charts.
Plus a daily "Hot 5" stocks list from the AI agents.
"""
from __future__ import annotations

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
#                         CUSTOM CSS                                      #
# ===================================================================== #

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

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

html, body, [data-testid="stAppViewContainer"],
[data-testid="stApp"], .main, .block-container {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

[data-testid="stSidebar"] {
    background-color: var(--panel) !important;
    border-left: 1px solid var(--border) !important;
}

[data-testid="stHeader"] { background-color: var(--bg) !important; }
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

.rtl { direction: rtl; text-align: right; }

/* ---- Strategy Card ---- */
.strategy-card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 28px 32px;
    box-shadow: var(--shadow);
    position: relative;
    overflow: hidden;
}
.strategy-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 4px;
}
.strategy-card.long::before { background: linear-gradient(90deg, #10B981, #34D399); }
.strategy-card.short::before { background: linear-gradient(90deg, #EF4444, #F87171); }
.strategy-card.hold::before { background: linear-gradient(90deg, #6B7280, #9CA3AF); }

/* ---- Direction Badge ---- */
.direction-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 10px 24px;
    border-radius: 50px;
    font-size: 1.1rem;
    font-weight: 700;
    letter-spacing: 0.5px;
}
.direction-long {
    background: linear-gradient(135deg, #10B981, #059669);
    color: white;
}
.direction-short {
    background: linear-gradient(135deg, #EF4444, #DC2626);
    color: white;
}
.direction-hold {
    background: linear-gradient(135deg, #6B7280, #4B5563);
    color: white;
}

/* ---- Level Card ---- */
.level-card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 16px 20px;
    text-align: center;
    box-shadow: var(--shadow-sm);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.level-card:hover {
    transform: translateY(-1px);
    box-shadow: var(--shadow-md);
}
.level-card .lbl {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--text-muted);
    margin-bottom: 6px;
}
.level-card .val {
    font-size: 1.4rem;
    font-weight: 800;
    font-variant-numeric: tabular-nums;
}
.level-card .sub {
    font-size: 0.78rem;
    color: var(--text-secondary);
    margin-top: 2px;
}
.level-card.entry { border-top: 3px solid var(--accent); }
.level-card.entry .val { color: var(--accent); }
.level-card.sl { border-top: 3px solid var(--down); }
.level-card.sl .val { color: var(--down); }
.level-card.tp { border-top: 3px solid var(--up); }
.level-card.tp .val { color: var(--up); }
.level-card.rr { border-top: 3px solid var(--gold); }
.level-card.rr .val { color: var(--gold); }

/* ---- Agent Pill ---- */
.agent-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: var(--panel-alt);
    border: 1px solid var(--border);
    border-radius: 50px;
    padding: 6px 16px;
    font-size: 0.82rem;
    font-weight: 600;
}
.agent-pill .dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    display: inline-block;
}
.dot-buy { background: var(--up); }
.dot-sell { background: var(--down); }
.dot-hold { background: var(--text-muted); }

/* ---- Hot Stock Row ---- */
.hot-row {
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
}
.hot-row:hover {
    transform: translateY(-1px);
    box-shadow: var(--shadow-md);
}
.hot-rank {
    font-size: 1.8rem;
    font-weight: 900;
    color: var(--text-muted);
    width: 48px;
    text-align: center;
}
.hot-ticker {
    font-size: 1.15rem;
    font-weight: 800;
    color: var(--text);
}
.hot-name {
    font-size: 0.78rem;
    color: var(--text-secondary);
}
.hot-signal {
    padding: 4px 14px;
    border-radius: 50px;
    font-size: 0.78rem;
    font-weight: 700;
    color: white;
}
.hot-signal.buy { background: var(--up); }
.hot-signal.sell { background: var(--down); }
.hot-conf {
    font-size: 0.82rem;
    font-weight: 600;
    color: var(--text-secondary);
}

/* ---- Reasoning Box ---- */
.reasoning-box {
    background: var(--panel-alt);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 16px 20px;
    direction: rtl;
    text-align: right;
    font-size: 0.88rem;
    line-height: 1.7;
    color: var(--text);
}
.reasoning-box .reason-title {
    font-weight: 700;
    font-size: 0.82rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: var(--text-muted);
    margin-bottom: 8px;
}

/* ---- Risk Badge ---- */
.risk-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 14px;
    border-radius: 50px;
    font-size: 0.78rem;
    font-weight: 700;
}
.risk-low { background: #ECFDF5; color: #065F46; }
.risk-moderate { background: #FFFBEB; color: #92400E; }
.risk-high { background: #FEF2F2; color: #991B1B; }
.risk-extreme { background: #991B1B; color: white; }

/* ---- Tabs ---- */
.stTabs [data-baseweb="tab-list"] { gap: 4px; direction: rtl; }
.stTabs [data-baseweb="tab"] {
    font-size: 0.92rem;
    font-weight: 600;
    border-radius: 8px 8px 0 0;
}

/* ---- Metric small ---- */
.metric-sm {
    text-align: center;
    padding: 10px;
}
.metric-sm .m-lbl {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: var(--text-muted);
}
.metric-sm .m-val {
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--text);
    margin-top: 2px;
}

/* Header */
.app-header {
    text-align: center;
    padding: 12px 0 24px;
}
.app-header h1 {
    font-size: 1.8rem;
    font-weight: 900;
    letter-spacing: -0.5px;
    color: var(--text);
    margin: 0;
}
.app-header p {
    font-size: 0.88rem;
    color: var(--text-secondary);
    margin: 4px 0 0;
}
</style>
"""

st.markdown(_CSS, unsafe_allow_html=True)

# ===================================================================== #
#                         PLOTLY DEFAULTS                                  #
# ===================================================================== #

_PLOTLY_LAYOUT = dict(
    template="plotly_white",
    paper_bgcolor="#FFFFFF",
    plot_bgcolor="#FAFBFC",
    font=dict(color="#111827", family="Inter, sans-serif"),
    xaxis=dict(gridcolor="#F3F4F6", zerolinecolor="#E5E7EB"),
    yaxis=dict(gridcolor="#F3F4F6", zerolinecolor="#E5E7EB"),
    margin=dict(l=50, r=20, t=40, b=30),
    legend=dict(bgcolor="rgba(255,255,255,0.95)", bordercolor="#E5E7EB", borderwidth=1),
)
_UP = "#10B981"
_DOWN = "#EF4444"
_ACCENT = "#6366F1"

def _apply_layout(fig, **overrides):
    fig.update_layout(**{**_PLOTLY_LAYOUT, **overrides})
    return fig

# ===================================================================== #
#                         DATA FETCHING                                    #
# ===================================================================== #

_PERIOD_MAP = {"1M": "1mo", "3M": "3mo", "6M": "6mo", "1Y": "1y", "2Y": "2y"}

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
    days_map = {"1M": 22, "3M": 66, "6M": 126, "1Y": 252, "2Y": 504}
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
            risk_level, risk_metrics = _compute_risk(close)
            action, total_conf, weighted_score, dissents = _build_recommendation(
                tech_sig, tech_conf, fund_sig, fund_conf,
                Signal.HOLD, 0.35, risk_level,
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

    return action, total_conf, adjusted, dissents


# ===================================================================== #
#              STRATEGY BUILDER                                            #
# ===================================================================== #

def build_strategy(action, close, df, info, atr_val, support, resistance, risk_level, atr_mult=2.0, reward_ratio=2.0):
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
    st.markdown("---")

    ticker = st.text_input("סימול מניה", value="NVDA",
                           help="e.g. AAPL, MSFT, TSLA, NVDA").upper().strip()
    timeframe = st.selectbox("טווח זמן", options=["3M", "6M", "1Y", "2Y"], index=2)

    st.markdown("---")
    st.markdown('<p style="font-weight:700; font-size:0.82rem; color:var(--text-secondary);">⚙️ הגדרות אסטרטגיה</p>', unsafe_allow_html=True)

    account_size = st.number_input("גודל חשבון ($)", min_value=1000, max_value=10_000_000, value=100_000, step=1000)
    risk_pct = st.slider("סיכון לעסקה (%)", min_value=0.25, max_value=5.0, value=1.0, step=0.25)
    reward_ratio = st.select_slider("יחס R:R", options=[1.5, 2.0, 2.5, 3.0, 4.0], value=2.0)
    atr_multiplier = st.slider("ATR מכפיל (SL)", min_value=1.0, max_value=4.0, value=2.0, step=0.5)

    st.markdown("---")
    if st.button("🔄 רענן", use_container_width=True):
        st.cache_data.clear()

# ===================================================================== #
#                         MAIN PAGE                                        #
# ===================================================================== #

tab_strategy, tab_hot5 = st.tabs(["📊 אסטרטגיית מסחר", "🔥 5 מניות חמות"])

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
    risk_level, risk_metrics = _compute_risk(close)
    action, total_conf, weighted_score, dissents = _build_recommendation(
        tech_sig, tech_conf, fund_sig, fund_conf, Signal.HOLD, 0.35, risk_level,
    )

    strat = build_strategy(action, close, df, info, atr_val, support, resistance,
                           risk_level, atr_multiplier, reward_ratio)

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
        st.markdown(f"""
        <div class="level-card sl">
            <div class="lbl">Stop Loss</div>
            <div class="val">${strat['stop_loss']:,.2f}</div>
            <div class="sub">סיכון ${strat['risk_per_share']:,.2f} למניה</div>
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

    # ---- Strategy Chart ----
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.78, 0.22], vertical_spacing=0.03)

    fig.add_trace(go.Candlestick(
        x=df.index, open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        increasing_line_color=_UP, decreasing_line_color=_DOWN,
        increasing_fillcolor=_UP, decreasing_fillcolor=_DOWN,
        name="OHLC", showlegend=False,
    ), row=1, col=1)

    sma20s = _sma_series(close, 20)
    sma50s = _sma_series(close, 50)
    fig.add_trace(go.Scatter(x=df.index, y=sma20s, line=dict(color="#F59E0B", width=1), name="SMA 20"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=sma50s, line=dict(color="#3B82F6", width=1), name="SMA 50"), row=1, col=1)

    bb_upper, bb_mid, bb_lower = _bollinger_series(close)
    fig.add_trace(go.Scatter(x=df.index, y=bb_upper, line=dict(color="rgba(99,102,241,0.2)", width=1, dash="dot"), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=bb_lower, line=dict(color="rgba(99,102,241,0.2)", width=1, dash="dot"),
                             fill="tonexty", fillcolor="rgba(99,102,241,0.04)", showlegend=False), row=1, col=1)

    entry_color = _ACCENT
    sl_color = _DOWN
    tp_color = _UP

    fig.add_hline(y=strat["entry"], line_color=entry_color, line_width=2, line_dash="dash",
                  annotation_text=f"Entry ${strat['entry']:.2f}", annotation_font_color=entry_color,
                  annotation_font_size=11, row=1, col=1)
    fig.add_hline(y=strat["stop_loss"], line_color=sl_color, line_width=2, line_dash="dash",
                  annotation_text=f"SL ${strat['stop_loss']:.2f}", annotation_font_color=sl_color,
                  annotation_font_size=11, row=1, col=1)
    fig.add_hline(y=strat["take_profit_1"], line_color=tp_color, line_width=2, line_dash="dash",
                  annotation_text=f"TP1 ${strat['take_profit_1']:.2f}", annotation_font_color=tp_color,
                  annotation_font_size=11, row=1, col=1)
    fig.add_hline(y=strat["take_profit_2"], line_color=tp_color, line_width=1.5, line_dash="dot",
                  annotation_text=f"TP2 ${strat['take_profit_2']:.2f}", annotation_font_color=tp_color,
                  annotation_font_size=10, row=1, col=1)

    fig.add_hrect(y0=strat["entry_zone"][0], y1=strat["entry_zone"][1],
                  fillcolor="rgba(99,102,241,0.08)", line_width=0, row=1, col=1)

    vol_colors = [_UP if c >= o else _DOWN for c, o in zip(df["close"], df["open"])]
    fig.add_trace(go.Bar(x=df.index, y=df["volume"], marker_color=vol_colors, showlegend=False, opacity=0.5), row=2, col=1)

    _apply_layout(fig, height=520, xaxis_rangeslider_visible=False,
                  title=dict(text=f"{ticker} — Strategy Map", font=dict(size=14, weight=600)))
    fig.update_xaxes(gridcolor="#F3F4F6")
    fig.update_yaxes(gridcolor="#F3F4F6")

    st.plotly_chart(fig, use_container_width=True)

    # ---- Agent Breakdown + Reasoning ----
    st.markdown("")
    agent_col, reason_col = st.columns([1, 1])

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
    position_pct = (position_value / account_size) * 100 if account_size > 0 else 0
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

    # R:R bar
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
                💡 לחץ על שם מניה בסיידבר כדי לקבל אסטרטגיית מסחר מפורטת
            </div>
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
