"""AI Broker Terminal -- Streamlit stock-trading dashboard.

Pulls live market data from yfinance, computes every technical indicator
using the existing ``src/tools/`` modules, runs a deterministic weighted
voting system identical to the LangGraph pipeline (minus the LLM call),
and presents the result in a dark, RTL Hebrew interface.
"""

from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Make sure the project root is on sys.path so ``from src.…`` works when
# Streamlit is launched from any directory.
# ---------------------------------------------------------------------------
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
    page_title="AI Broker Terminal",
    page_icon="\U0001F4C8",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===================================================================== #
#                         CUSTOM CSS                                      #
# ===================================================================== #

_CSS = """
<style>
:root {
    --bg: #131722;
    --panel: #1E222D;
    --up: #26A69A;
    --down: #EF5350;
    --accent: #2962FF;
    --text: #D1D4DC;
    --text-dim: #787B86;
}

/* Force dark background everywhere */
html, body, [data-testid="stAppViewContainer"],
[data-testid="stApp"], .main, .block-container {
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

[data-testid="stSidebar"] {
    background-color: var(--panel) !important;
}

[data-testid="stHeader"] {
    background-color: var(--bg) !important;
}

/* RTL for Hebrew content */
.rtl {
    direction: rtl;
    text-align: right;
}

/* Metric cards */
.metric-card {
    background: var(--panel);
    border-radius: 8px;
    padding: 16px 20px;
    text-align: center;
}
.metric-card .label {
    font-size: 0.78rem;
    color: var(--text-dim);
    margin-bottom: 4px;
}
.metric-card .value {
    font-size: 1.45rem;
    font-weight: 700;
    color: var(--text);
}
.metric-card .delta-up   { color: var(--up); font-size: 0.85rem; }
.metric-card .delta-down { color: var(--down); font-size: 0.85rem; }

/* Signal badge */
.signal-badge {
    display: inline-block;
    padding: 8px 28px;
    border-radius: 6px;
    font-size: 1.3rem;
    font-weight: 700;
    letter-spacing: 1px;
}
.badge-strong-buy  { background: #00695C; color: #E0F2F1; }
.badge-buy         { background: #26A69A; color: #E0F2F1; }
.badge-hold        { background: #546E7A; color: #ECEFF1; }
.badge-sell        { background: #EF5350; color: #FFEBEE; }
.badge-strong-sell { background: #B71C1C; color: #FFCDD2; }

/* Risk badge */
.risk-low      { background: #26A69A; color: #E0F2F1; padding: 4px 14px; border-radius: 4px; font-weight: 600; }
.risk-moderate { background: #FFA726; color: #1E222D; padding: 4px 14px; border-radius: 4px; font-weight: 600; }
.risk-high     { background: #EF5350; color: #FFEBEE; padding: 4px 14px; border-radius: 4px; font-weight: 600; }
.risk-extreme  { background: #B71C1C; color: #FFCDD2; padding: 4px 14px; border-radius: 4px; font-weight: 600; }

/* Agent card */
.agent-card {
    background: var(--panel);
    border-radius: 8px;
    padding: 18px 22px;
    margin-bottom: 10px;
    border-left: 4px solid var(--accent);
}

/* Calculator result */
.calc-result {
    background: var(--panel);
    border-radius: 8px;
    padding: 20px 24px;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    direction: rtl;
}
.stTabs [data-baseweb="tab"] {
    font-size: 1rem;
}

/* Hide Streamlit default elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""

st.markdown(_CSS, unsafe_allow_html=True)

# ===================================================================== #
#                     HELPER: plotly layout defaults                      #
# ===================================================================== #

_PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="#131722",
    plot_bgcolor="#131722",
    font=dict(color="#D1D4DC"),
    xaxis=dict(gridcolor="#2A2E39", zerolinecolor="#2A2E39"),
    yaxis=dict(gridcolor="#2A2E39", zerolinecolor="#2A2E39"),
    margin=dict(l=50, r=20, t=40, b=30),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
)

_UP = "#26A69A"
_DOWN = "#EF5350"
_ACCENT = "#2962FF"


def _apply_layout(fig: go.Figure, **overrides) -> go.Figure:
    merged = {**_PLOTLY_LAYOUT, **overrides}
    fig.update_layout(**merged)
    return fig


# ===================================================================== #
#                        DATA FETCHING                                    #
# ===================================================================== #

_PERIOD_MAP = {"1M": "1mo", "3M": "3mo", "6M": "6mo", "1Y": "1y", "2Y": "2y"}


@st.cache_data(ttl=300, show_spinner=False)
def fetch_data(ticker: str, period: str) -> tuple[pd.DataFrame | None, dict | None]:
    """Return (ohlcv_df, info_dict).  Returns (None, None) on failure."""
    try:
        tk = yf.Ticker(ticker)
        df = tk.history(period=_PERIOD_MAP.get(period, "1y"), auto_adjust=True)
        if df is None or df.empty:
            return None, None
        # Normalise column names to lowercase
        df.columns = [c.lower() for c in df.columns]
        info = tk.info or {}
        return df, info
    except Exception:
        return None, None


# ===================================================================== #
#                  INDICATOR SERIES (for charts)                          #
# ===================================================================== #

def _sma_series(s: pd.Series, period: int) -> pd.Series:
    return s.rolling(window=period).mean()


def _ema_series(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def _rsi_series(s: pd.Series, period: int = 14) -> pd.Series:
    delta = s.diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
    rs = gain / loss
    return 100.0 - (100.0 / (1.0 + rs))


def _macd_series(s: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ef = s.ewm(span=fast, adjust=False).mean()
    es = s.ewm(span=slow, adjust=False).mean()
    ml = ef - es
    ms = ml.ewm(span=signal, adjust=False).mean()
    mh = ml - ms
    return ml, ms, mh


def _stochastic_series(df: pd.DataFrame, k_period: int = 14, d_period: int = 3):
    low_min = df["low"].rolling(window=k_period).min()
    high_max = df["high"].rolling(window=k_period).max()
    denom = high_max - low_min
    denom = denom.replace(0, np.nan)
    k = 100.0 * (df["close"] - low_min) / denom
    d = k.rolling(window=d_period).mean()
    return k, d


def _atr_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def _bollinger_series(s: pd.Series, period: int = 20, std_dev: float = 2.0):
    mid = s.rolling(window=period).mean()
    std = s.rolling(window=period).std()
    return mid + std_dev * std, mid, mid - std_dev * std


# ===================================================================== #
#                TECHNICAL SIGNAL SCORING (no LLM)                        #
# ===================================================================== #

def compute_technical_signal(
    close: pd.Series, df: pd.DataFrame
) -> tuple[Signal, float, float, str]:
    """Return (signal, confidence, raw_score, reasoning)."""
    rsi = compute_rsi(close)
    macd = compute_macd(close)
    macd_hist = macd["macd_histogram"]
    sma50 = compute_sma(close, 50)
    sma200 = compute_sma(close, 200)
    last_close = float(close.iloc[-1])

    score = 0.0
    reasons: list[str] = []

    # RSI
    if rsi is not None:
        if rsi < 30:
            score += 0.3
            reasons.append(f"RSI oversold ({rsi:.1f})")
        elif rsi < 45:
            score += 0.1
            reasons.append(f"RSI leaning bullish ({rsi:.1f})")
        elif rsi > 70:
            score -= 0.3
            reasons.append(f"RSI overbought ({rsi:.1f})")
        elif rsi > 55:
            score -= 0.1
            reasons.append(f"RSI leaning bearish ({rsi:.1f})")

    # MACD histogram
    if macd_hist is not None:
        if macd_hist > 0:
            score += 0.2
            reasons.append("MACD histogram positive")
        else:
            score -= 0.2
            reasons.append("MACD histogram negative")

    # Price vs SMA 50
    if sma50 is not None:
        if last_close > sma50:
            score += 0.2
            reasons.append("Price above SMA50")
        else:
            score -= 0.2
            reasons.append("Price below SMA50")

    # Price vs SMA 200
    if sma200 is not None:
        if last_close > sma200:
            score += 0.1
            reasons.append("Price above SMA200")
        else:
            score -= 0.1
            reasons.append("Price below SMA200")

    # Patterns
    patterns = detect_patterns(df)
    if "golden_cross" in patterns:
        score += 0.15
        reasons.append("Golden cross detected")
    if "death_cross" in patterns:
        score -= 0.15
        reasons.append("Death cross detected")

    confidence = min(1.0, max(0.3, 0.5 + abs(score)))

    if score > 0.4:
        signal = Signal.STRONG_BUY
    elif score > 0.15:
        signal = Signal.BUY
    elif score < -0.4:
        signal = Signal.STRONG_SELL
    elif score < -0.15:
        signal = Signal.SELL
    else:
        signal = Signal.HOLD

    reasoning = "; ".join(reasons) if reasons else "No clear signals"
    return signal, confidence, score, reasoning


# ===================================================================== #
#               FUNDAMENTAL SIGNAL SCORING (no LLM)                       #
# ===================================================================== #

def compute_fundamental_signal(
    info: dict,
) -> tuple[Signal, float, float, str]:
    """Compute a simple fundamental score from yfinance .info dict."""
    score = 0.0
    reasons: list[str] = []

    pe = info.get("trailingPE") or info.get("forwardPE")
    de = info.get("debtToEquity")
    roe = info.get("returnOnEquity")
    rev_growth = info.get("revenueGrowth")
    profit_margin = info.get("profitMargins")
    current_ratio = info.get("currentRatio")

    # PE ratio
    if pe is not None:
        if pe < 0:
            score -= 0.15
            reasons.append(f"Negative P/E ({pe:.1f})")
        elif pe < 15:
            score += 0.2
            reasons.append(f"Low P/E ({pe:.1f})")
        elif pe < 25:
            score += 0.05
            reasons.append(f"Moderate P/E ({pe:.1f})")
        elif pe < 40:
            score -= 0.05
            reasons.append(f"High P/E ({pe:.1f})")
        else:
            score -= 0.15
            reasons.append(f"Very high P/E ({pe:.1f})")
    else:
        reasons.append("P/E unavailable")

    # Debt-to-equity
    if de is not None:
        de_ratio = de / 100.0 if de > 10 else de  # yfinance sometimes returns %
        if de_ratio < 0.5:
            score += 0.15
            reasons.append(f"Low debt/equity ({de_ratio:.2f})")
        elif de_ratio < 1.0:
            score += 0.05
            reasons.append(f"Moderate debt/equity ({de_ratio:.2f})")
        elif de_ratio < 2.0:
            score -= 0.05
            reasons.append(f"High debt/equity ({de_ratio:.2f})")
        else:
            score -= 0.15
            reasons.append(f"Very high debt/equity ({de_ratio:.2f})")

    # ROE
    if roe is not None:
        if roe > 0.20:
            score += 0.2
            reasons.append(f"Strong ROE ({roe:.1%})")
        elif roe > 0.10:
            score += 0.1
            reasons.append(f"Good ROE ({roe:.1%})")
        elif roe > 0:
            score += 0.0
            reasons.append(f"Weak ROE ({roe:.1%})")
        else:
            score -= 0.15
            reasons.append(f"Negative ROE ({roe:.1%})")

    # Revenue growth
    if rev_growth is not None:
        if rev_growth > 0.20:
            score += 0.15
            reasons.append(f"Strong revenue growth ({rev_growth:.1%})")
        elif rev_growth > 0.05:
            score += 0.05
            reasons.append(f"Moderate revenue growth ({rev_growth:.1%})")
        elif rev_growth > -0.05:
            score += 0.0
            reasons.append(f"Flat revenue ({rev_growth:.1%})")
        else:
            score -= 0.15
            reasons.append(f"Revenue declining ({rev_growth:.1%})")

    # Profit margin
    if profit_margin is not None:
        if profit_margin > 0.20:
            score += 0.1
            reasons.append(f"High profit margin ({profit_margin:.1%})")
        elif profit_margin > 0.05:
            score += 0.0
        elif profit_margin > 0:
            score -= 0.05
            reasons.append(f"Thin margin ({profit_margin:.1%})")
        else:
            score -= 0.1
            reasons.append(f"Negative margin ({profit_margin:.1%})")

    # Current ratio
    if current_ratio is not None:
        if current_ratio > 1.5:
            score += 0.05
        elif current_ratio < 1.0:
            score -= 0.1
            reasons.append(f"Low current ratio ({current_ratio:.2f})")

    confidence = min(1.0, max(0.3, 0.45 + abs(score)))

    if score > 0.4:
        signal = Signal.STRONG_BUY
    elif score > 0.15:
        signal = Signal.BUY
    elif score < -0.4:
        signal = Signal.STRONG_SELL
    elif score < -0.15:
        signal = Signal.SELL
    else:
        signal = Signal.HOLD

    reasoning = "; ".join(reasons) if reasons else "Insufficient fundamental data"
    return signal, confidence, score, reasoning


# ===================================================================== #
#                SENTIMENT PLACEHOLDER                                    #
# ===================================================================== #

def compute_sentiment_signal() -> tuple[Signal, float, float, str]:
    """Placeholder -- no free news API available."""
    return Signal.HOLD, 0.35, 0.0, "לא זמין – אין API חדשות"


# ===================================================================== #
#                   RISK ASSESSMENT                                       #
# ===================================================================== #

def compute_risk(close: pd.Series) -> tuple[RiskLevel, dict]:
    vol_pct = compute_volatility_percentile(close)
    var95 = compute_var_95(close)
    mdd = compute_max_drawdown(close)
    sharpe = compute_sharpe_ratio(close)

    flags = []
    if vol_pct >= VOLATILITY_EXTREME_THRESHOLD:
        flags.append("Extreme volatility")
    if mdd is not None and mdd <= MAX_DRAWDOWN_SEVERE_THRESHOLD:
        flags.append("Severe drawdown")

    if vol_pct >= 0.9 or (mdd is not None and mdd <= -0.30):
        level = RiskLevel.EXTREME
    elif vol_pct >= 0.7 or (mdd is not None and mdd <= -0.20):
        level = RiskLevel.HIGH
    elif vol_pct >= 0.4:
        level = RiskLevel.MODERATE
    else:
        level = RiskLevel.LOW

    metrics = {
        "volatility_percentile": vol_pct,
        "var_95": var95,
        "max_drawdown": mdd,
        "sharpe_ratio": sharpe,
        "flags": flags,
    }
    return level, metrics


# ===================================================================== #
#                FINAL RECOMMENDATION (replicate strategist)              #
# ===================================================================== #

_SIGNAL_LABEL = {
    Signal.STRONG_BUY: "קנייה חזקה",
    Signal.BUY: "קנייה",
    Signal.HOLD: "המתנה",
    Signal.SELL: "מכירה",
    Signal.STRONG_SELL: "מכירה חזקה",
}

_SIGNAL_BADGE_CLASS = {
    Signal.STRONG_BUY: "badge-strong-buy",
    Signal.BUY: "badge-buy",
    Signal.HOLD: "badge-hold",
    Signal.SELL: "badge-sell",
    Signal.STRONG_SELL: "badge-strong-sell",
}

_RISK_LABEL = {
    RiskLevel.LOW: "נמוך",
    RiskLevel.MODERATE: "מתון",
    RiskLevel.HIGH: "גבוה",
    RiskLevel.EXTREME: "קיצוני",
}

_RISK_BADGE_CLASS = {
    RiskLevel.LOW: "risk-low",
    RiskLevel.MODERATE: "risk-moderate",
    RiskLevel.HIGH: "risk-high",
    RiskLevel.EXTREME: "risk-extreme",
}


def build_recommendation(
    tech_signal: Signal,
    tech_conf: float,
    fund_signal: Signal,
    fund_conf: float,
    sent_signal: Signal,
    sent_conf: float,
    risk_level: RiskLevel,
) -> tuple[Signal, float, float, list[str]]:
    tech_score = SIGNAL_TO_SCORE[tech_signal] * tech_conf
    fund_score = SIGNAL_TO_SCORE[fund_signal] * fund_conf
    sent_score = SIGNAL_TO_SCORE[sent_signal] * sent_conf

    raw_score = (
        tech_score * TECHNICAL_WEIGHT
        + fund_score * FUNDAMENTAL_WEIGHT
        + sent_score * SENTIMENT_WEIGHT
    )

    dampening = RISK_DAMPENING[risk_level]
    adjusted = raw_score * dampening

    if adjusted >= SCORE_THRESHOLDS["strong_buy"]:
        action = Signal.STRONG_BUY
    elif adjusted >= SCORE_THRESHOLDS["buy"]:
        action = Signal.BUY
    elif adjusted <= SCORE_THRESHOLDS["strong_sell"]:
        action = Signal.STRONG_SELL
    elif adjusted <= SCORE_THRESHOLDS["sell"]:
        action = Signal.SELL
    else:
        action = Signal.HOLD

    total_conf = (
        tech_conf * TECHNICAL_WEIGHT
        + fund_conf * FUNDAMENTAL_WEIGHT
        + sent_conf * SENTIMENT_WEIGHT
    ) * dampening

    dissents: list[str] = []
    if tech_signal != action:
        dissents.append(
            f"טכני: {_SIGNAL_LABEL[tech_signal]} "
            f"(ביטחון {tech_conf:.0%})"
        )
    if fund_signal != action:
        dissents.append(
            f"פונדמנטלי: {_SIGNAL_LABEL[fund_signal]} "
            f"(ביטחון {fund_conf:.0%})"
        )
    if sent_signal != action:
        dissents.append(
            f"סנטימנט: {_SIGNAL_LABEL[sent_signal]} "
            f"(בטחון {sent_conf:.0%})"
        )

    return action, total_conf, adjusted, dissents


# ===================================================================== #
#                        METRIC CARD helper                               #
# ===================================================================== #

def _metric_html(label: str, value: str, delta: str = "", delta_up: bool = True) -> str:
    delta_cls = "delta-up" if delta_up else "delta-down"
    delta_block = f'<div class="{delta_cls}">{delta}</div>' if delta else ""
    return (
        f'<div class="metric-card">'
        f'<div class="label">{label}</div>'
        f'<div class="value">{value}</div>'
        f'{delta_block}'
        f'</div>'
    )


def _fmt_big_number(n: float | None) -> str:
    if n is None:
        return "N/A"
    if abs(n) >= 1e12:
        return f"${n / 1e12:.2f}T"
    if abs(n) >= 1e9:
        return f"${n / 1e9:.2f}B"
    if abs(n) >= 1e6:
        return f"${n / 1e6:.1f}M"
    return f"${n:,.0f}"


# ===================================================================== #
#                           SIDEBAR                                       #
# ===================================================================== #

with st.sidebar:
    st.markdown(
        '<h2 class="rtl">\U0001F4C8 AI Broker Terminal</h2>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    ticker = st.text_input(
        "סימול מניה",
        value="AAPL",
        help="e.g. AAPL, MSFT, TSLA, AMZN",
    ).upper().strip()

    timeframe = st.selectbox(
        "טווח זמן",
        options=["1M", "3M", "6M", "1Y", "2Y"],
        index=3,
    )

    refresh = st.button("\U0001F504 רענן נתונים", use_container_width=True)
    if refresh:
        st.cache_data.clear()

    st.markdown("---")
    st.markdown('<p class="rtl" style="font-weight:600;">⚙️ הגדרות</p>', unsafe_allow_html=True)

    account_size = st.number_input(
        "גודל חשבון ($)",
        min_value=1000,
        max_value=100_000_000,
        value=100_000,
        step=1000,
    )
    risk_pct = st.slider(
        "סיכון לעסקה (%)",
        min_value=0.25,
        max_value=5.0,
        value=1.0,
        step=0.25,
    )
    reward_ratio = st.selectbox(
        "יחס סיכון/תגמול",
        options=["1:1.5", "1:2", "1:3", "1:4"],
        index=1,
    )
    atr_multiplier = st.slider(
        "ATR Multiplier (Stop Loss)",
        min_value=0.5,
        max_value=5.0,
        value=2.0,
        step=0.5,
    )

# ===================================================================== #
#                         MAIN AREA                                       #
# ===================================================================== #

st.markdown(
    f'<h1 class="rtl" style="margin-bottom:0;">{ticker} – טרמינל AI</h1>',
    unsafe_allow_html=True,
)

with st.spinner("טוען נתונים..."):
    df, info = fetch_data(ticker, timeframe)

if df is None or info is None:
    st.error(
        f"לא ניתן לטעון נתונים עבור **{ticker}**. "
        "בדוק את הסימול ונסה שוב."
    )
    st.stop()

close = df["close"]
last_close = float(close.iloc[-1])

# ===================================================================== #
#                      COMPUTE ALL INDICATORS                             #
# ===================================================================== #

# Scalar indicators (using existing src/tools functions)
rsi_val = compute_rsi(close)
macd_vals = compute_macd(close)
boll_vals = compute_bollinger_bands(close)
atr_val = compute_atr(df)
obv_val = compute_obv(df)
stoch_vals = compute_stochastic(df)
sma20_val = compute_sma(close, 20)
sma50_val = compute_sma(close, 50)
sma200_val = compute_sma(close, 200)
ema12_val = compute_ema(close, 12)
ema26_val = compute_ema(close, 26)
support, resistance = identify_support_resistance(df)
patterns = detect_patterns(df)

# Signals
tech_signal, tech_conf, tech_raw_score, tech_reasoning = compute_technical_signal(close, df)
fund_signal, fund_conf, fund_raw_score, fund_reasoning = compute_fundamental_signal(info)
sent_signal, sent_conf, sent_raw_score, sent_reasoning = compute_sentiment_signal()

# Risk
risk_level, risk_metrics = compute_risk(close)

# Final
action, total_conf, weighted_score, dissents = build_recommendation(
    tech_signal, tech_conf,
    fund_signal, fund_conf,
    sent_signal, sent_conf,
    risk_level,
)

# ===================================================================== #
#                              TABS                                       #
# ===================================================================== #

tab_overview, tab_indicators, tab_ai, tab_calc = st.tabs([
    "\U0001F4CA סקירה",
    "\U0001F4C9 אינדיקטורים",
    "\U0001F916 אות AI",
    "\U0001F5A9 מחשבון",
])

# ===================================================================== #
# TAB 1 -- Overview                                                       #
# ===================================================================== #

with tab_overview:
    # --- Metric cards ---
    prev_close = float(close.iloc[-2]) if len(close) > 1 else last_close
    change_pct = (last_close - prev_close) / prev_close * 100 if prev_close else 0
    change_abs = last_close - prev_close
    is_up = change_pct >= 0
    last_vol = int(df["volume"].iloc[-1]) if "volume" in df.columns else 0
    w52_high = info.get("fiftyTwoWeekHigh", df["high"].max())
    w52_low = info.get("fiftyTwoWeekLow", df["low"].min())
    market_cap = info.get("marketCap")

    cols = st.columns(5)
    with cols[0]:
        st.markdown(
            _metric_html(
                "מחיר נוכחי",
                f"${last_close:,.2f}",
                f"{'+'if is_up else ''}{change_abs:,.2f} ({change_pct:+.2f}%)",
                is_up,
            ),
            unsafe_allow_html=True,
        )
    with cols[1]:
        st.markdown(
            _metric_html(
                "שינוי %",
                f"{change_pct:+.2f}%",
                "",
                is_up,
            ),
            unsafe_allow_html=True,
        )
    with cols[2]:
        st.markdown(
            _metric_html("מחזור", f"{last_vol:,}"),
            unsafe_allow_html=True,
        )
    with cols[3]:
        st.markdown(
            _metric_html(
                "52W Range",
                f"${w52_low:,.2f} – ${w52_high:,.2f}",
            ),
            unsafe_allow_html=True,
        )
    with cols[4]:
        st.markdown(
            _metric_html("שווי שוק", _fmt_big_number(market_cap)),
            unsafe_allow_html=True,
        )

    st.markdown("")

    # --- Candlestick chart ---
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.75, 0.25],
        vertical_spacing=0.03,
    )

    # Candlesticks
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            increasing_line_color=_UP,
            decreasing_line_color=_DOWN,
            increasing_fillcolor=_UP,
            decreasing_fillcolor=_DOWN,
            name="OHLC",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # SMA lines
    sma20s = _sma_series(close, 20)
    sma50s = _sma_series(close, 50)
    sma200s = _sma_series(close, 200)

    fig.add_trace(
        go.Scatter(x=df.index, y=sma20s, line=dict(color="#FFCA28", width=1), name="SMA 20"),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=sma50s, line=dict(color="#42A5F5", width=1), name="SMA 50"),
        row=1, col=1,
    )
    if len(df) >= 200:
        fig.add_trace(
            go.Scatter(x=df.index, y=sma200s, line=dict(color="#AB47BC", width=1), name="SMA 200"),
            row=1, col=1,
        )

    # Bollinger Bands
    bb_upper, bb_mid, bb_lower = _bollinger_series(close)
    fig.add_trace(
        go.Scatter(
            x=df.index, y=bb_upper,
            line=dict(color="rgba(150,150,150,0.3)", width=1, dash="dot"),
            name="BB Upper",
            showlegend=False,
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df.index, y=bb_lower,
            line=dict(color="rgba(150,150,150,0.3)", width=1, dash="dot"),
            fill="tonexty",
            fillcolor="rgba(150,150,150,0.06)",
            name="BB Lower",
            showlegend=False,
        ),
        row=1, col=1,
    )

    # Support / Resistance lines
    if support is not None:
        fig.add_hline(y=support, line_color="#26A69A", line_dash="dash", line_width=1,
                      annotation_text=f"Support ${support:.2f}", annotation_font_color="#26A69A",
                      row=1, col=1)
    if resistance is not None:
        fig.add_hline(y=resistance, line_color="#EF5350", line_dash="dash", line_width=1,
                      annotation_text=f"Resistance ${resistance:.2f}", annotation_font_color="#EF5350",
                      row=1, col=1)

    # Volume bars
    vol_colors = [
        _UP if c >= o else _DOWN
        for c, o in zip(df["close"], df["open"])
    ]
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["volume"],
            marker_color=vol_colors,
            name="Volume",
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    _apply_layout(
        fig,
        height=600,
        title=dict(text=f"{ticker} – Candlestick", font=dict(size=15)),
        xaxis_rangeslider_visible=False,
        yaxis2=dict(gridcolor="#2A2E39", zerolinecolor="#2A2E39"),
    )
    fig.update_xaxes(gridcolor="#2A2E39", zerolinecolor="#2A2E39")
    fig.update_yaxes(gridcolor="#2A2E39", zerolinecolor="#2A2E39")

    st.plotly_chart(fig, use_container_width=True)

    # Patterns detected
    if patterns:
        pattern_labels = {
            "golden_cross": "✨ Golden Cross",
            "death_cross": "☠️ Death Cross",
            "doji": "➕ Doji",
            "hammer": "\U0001F528 Hammer",
        }
        detected = ", ".join(pattern_labels.get(p, p) for p in patterns)
        st.info(f"תבניות שזוהו: {detected}")

# ===================================================================== #
# TAB 2 -- Indicators                                                     #
# ===================================================================== #

with tab_indicators:
    ind_col1, ind_col2 = st.columns(2)

    # --- RSI ---
    with ind_col1:
        rsi_s = _rsi_series(close)
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=df.index, y=rsi_s, line=dict(color="#AB47BC", width=1.5), name="RSI 14"))
        fig_rsi.add_hline(y=70, line_color=_DOWN, line_dash="dash", line_width=0.8,
                          annotation_text="70", annotation_font_color=_DOWN)
        fig_rsi.add_hline(y=30, line_color=_UP, line_dash="dash", line_width=0.8,
                          annotation_text="30", annotation_font_color=_UP)
        fig_rsi.add_hrect(y0=70, y1=100, fillcolor=_DOWN, opacity=0.07, line_width=0)
        fig_rsi.add_hrect(y0=0, y1=30, fillcolor=_UP, opacity=0.07, line_width=0)
        _apply_layout(fig_rsi, height=300, title=dict(text="RSI (14)", font=dict(size=13)))
        st.plotly_chart(fig_rsi, use_container_width=True)

    # --- MACD ---
    with ind_col2:
        ml, ms, mh = _macd_series(close)
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=df.index, y=ml, line=dict(color="#42A5F5", width=1.5), name="MACD"))
        fig_macd.add_trace(go.Scatter(x=df.index, y=ms, line=dict(color="#FFCA28", width=1.2), name="Signal"))
        hist_colors = [_UP if v >= 0 else _DOWN for v in mh.fillna(0)]
        fig_macd.add_trace(go.Bar(x=df.index, y=mh, marker_color=hist_colors, name="Histogram", opacity=0.6))
        _apply_layout(fig_macd, height=300, title=dict(text="MACD (12/26/9)", font=dict(size=13)))
        st.plotly_chart(fig_macd, use_container_width=True)

    ind_col3, ind_col4 = st.columns(2)

    # --- Stochastic ---
    with ind_col3:
        stoch_k, stoch_d = _stochastic_series(df)
        fig_stoch = go.Figure()
        fig_stoch.add_trace(go.Scatter(x=df.index, y=stoch_k, line=dict(color="#42A5F5", width=1.5), name="%K"))
        fig_stoch.add_trace(go.Scatter(x=df.index, y=stoch_d, line=dict(color="#FFCA28", width=1.2), name="%D"))
        fig_stoch.add_hline(y=80, line_color=_DOWN, line_dash="dash", line_width=0.8)
        fig_stoch.add_hline(y=20, line_color=_UP, line_dash="dash", line_width=0.8)
        fig_stoch.add_hrect(y0=80, y1=100, fillcolor=_DOWN, opacity=0.07, line_width=0)
        fig_stoch.add_hrect(y0=0, y1=20, fillcolor=_UP, opacity=0.07, line_width=0)
        _apply_layout(fig_stoch, height=300, title=dict(text="Stochastic (14/3)", font=dict(size=13)))
        st.plotly_chart(fig_stoch, use_container_width=True)

    # --- ATR ---
    with ind_col4:
        atr_s = _atr_series(df)
        fig_atr = go.Figure()
        fig_atr.add_trace(go.Scatter(x=df.index, y=atr_s, line=dict(color="#FF7043", width=1.5), name="ATR 14", fill="tozeroy", fillcolor="rgba(255,112,67,0.10)"))
        _apply_layout(fig_atr, height=300, title=dict(text="ATR (14)", font=dict(size=13)))
        st.plotly_chart(fig_atr, use_container_width=True)

    # Indicator summary table
    st.markdown('<h4 class="rtl">סיכום אינדיקטורים</h4>', unsafe_allow_html=True)
    summary_data = {
        "אינדיקטור": [
            "RSI (14)", "MACD Line", "MACD Signal", "MACD Histogram",
            "SMA 20", "SMA 50", "SMA 200",
            "EMA 12", "EMA 26",
            "Bollinger Upper", "Bollinger Mid", "Bollinger Lower",
            "ATR (14)", "OBV",
            "Stochastic %K", "Stochastic %D",
            "Support", "Resistance",
        ],
        "ערך": [
            f"{rsi_val:.2f}" if rsi_val else "N/A",
            f"{macd_vals['macd_line']:.4f}" if macd_vals["macd_line"] is not None else "N/A",
            f"{macd_vals['macd_signal']:.4f}" if macd_vals["macd_signal"] is not None else "N/A",
            f"{macd_vals['macd_histogram']:.4f}" if macd_vals["macd_histogram"] is not None else "N/A",
            f"${sma20_val:.2f}" if sma20_val else "N/A",
            f"${sma50_val:.2f}" if sma50_val else "N/A",
            f"${sma200_val:.2f}" if sma200_val else "N/A",
            f"${ema12_val:.2f}" if ema12_val else "N/A",
            f"${ema26_val:.2f}" if ema26_val else "N/A",
            f"${boll_vals['bollinger_upper']:.2f}" if boll_vals["bollinger_upper"] is not None else "N/A",
            f"${boll_vals['bollinger_middle']:.2f}" if boll_vals["bollinger_middle"] is not None else "N/A",
            f"${boll_vals['bollinger_lower']:.2f}" if boll_vals["bollinger_lower"] is not None else "N/A",
            f"${atr_val:.2f}" if atr_val else "N/A",
            f"{obv_val:,.0f}" if obv_val else "N/A",
            f"{stoch_vals['stochastic_k']:.2f}" if stoch_vals["stochastic_k"] is not None else "N/A",
            f"{stoch_vals['stochastic_d']:.2f}" if stoch_vals["stochastic_d"] is not None else "N/A",
            f"${support:.2f}" if support else "N/A",
            f"${resistance:.2f}" if resistance else "N/A",
        ],
    }
    st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

# ===================================================================== #
# TAB 3 -- AI Signal                                                      #
# ===================================================================== #

with tab_ai:
    # --- Signal badge ---
    st.markdown(
        f'<div style="text-align:center; margin:20px 0;">'
        f'<span class="signal-badge {_SIGNAL_BADGE_CLASS[action]}">'
        f'{_SIGNAL_LABEL[action]}'
        f'</span></div>',
        unsafe_allow_html=True,
    )

    # --- Confidence & weighted score ---
    conf_col, score_col = st.columns(2)
    with conf_col:
        st.markdown('<p class="rtl" style="font-weight:600;">ביטחון כללי</p>', unsafe_allow_html=True)
        st.progress(min(1.0, max(0.0, total_conf)))
        st.markdown(f"<p style='text-align:center; font-size:1.3rem; font-weight:700;'>{total_conf:.1%}</p>", unsafe_allow_html=True)
    with score_col:
        st.markdown('<p class="rtl" style="font-weight:600;">ציון משוקלל</p>', unsafe_allow_html=True)
        # Normalise score to 0-1 for progress bar display (score range -1..+1)
        normalised = (weighted_score + 1) / 2
        st.progress(min(1.0, max(0.0, normalised)))
        st.markdown(f"<p style='text-align:center; font-size:1.3rem; font-weight:700;'>{weighted_score:+.4f}</p>", unsafe_allow_html=True)

    st.markdown("---")

    # --- Agent breakdown ---
    st.markdown('<h3 class="rtl">פירוט סוכנים</h3>', unsafe_allow_html=True)

    agent_cols = st.columns(3)

    # Technical agent
    with agent_cols[0]:
        st.markdown(
            f'<div class="agent-card">'
            f'<div style="font-weight:700; font-size:1.05rem;">\U0001F4C9 טכני ({TECHNICAL_WEIGHT:.0%})</div>'
            f'<div style="margin:8px 0;">'
            f'<span class="signal-badge {_SIGNAL_BADGE_CLASS[tech_signal]}" style="font-size:0.9rem; padding:4px 14px;">'
            f'{_SIGNAL_LABEL[tech_signal]}</span></div>'
            f'<div style="color:var(--text-dim); font-size:0.82rem;">{tech_reasoning}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.caption(f"ביטחון: {tech_conf:.0%}")
        st.progress(min(1.0, max(0.0, tech_conf)))

    # Fundamental agent
    with agent_cols[1]:
        st.markdown(
            f'<div class="agent-card">'
            f'<div style="font-weight:700; font-size:1.05rem;">\U0001F4CA פונדמנטלי ({FUNDAMENTAL_WEIGHT:.0%})</div>'
            f'<div style="margin:8px 0;">'
            f'<span class="signal-badge {_SIGNAL_BADGE_CLASS[fund_signal]}" style="font-size:0.9rem; padding:4px 14px;">'
            f'{_SIGNAL_LABEL[fund_signal]}</span></div>'
            f'<div style="color:var(--text-dim); font-size:0.82rem;">{fund_reasoning}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.caption(f"ביטחון: {fund_conf:.0%}")
        st.progress(min(1.0, max(0.0, fund_conf)))

    # Sentiment agent
    with agent_cols[2]:
        st.markdown(
            f'<div class="agent-card">'
            f'<div style="font-weight:700; font-size:1.05rem;">\U0001F4F0 סנטימנט ({SENTIMENT_WEIGHT:.0%})</div>'
            f'<div style="margin:8px 0;">'
            f'<span class="signal-badge {_SIGNAL_BADGE_CLASS[sent_signal]}" style="font-size:0.9rem; padding:4px 14px;">'
            f'{_SIGNAL_LABEL[sent_signal]}</span></div>'
            f'<div style="color:var(--text-dim); font-size:0.82rem;">{sent_reasoning}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.caption(f"ביטחון: {sent_conf:.0%}")
        st.progress(min(1.0, max(0.0, sent_conf)))

    st.markdown("---")

    # --- Risk assessment ---
    st.markdown('<h3 class="rtl">הערכת סיכון</h3>', unsafe_allow_html=True)

    risk_cols = st.columns(5)
    with risk_cols[0]:
        st.markdown(
            f'<div class="metric-card"><div class="label">רמת סיכון</div>'
            f'<div><span class="{_RISK_BADGE_CLASS[risk_level]}">{_RISK_LABEL[risk_level]}</span></div></div>',
            unsafe_allow_html=True,
        )
    with risk_cols[1]:
        st.markdown(
            _metric_html(
                "אחוז תנודתיות",
                f"{risk_metrics['volatility_percentile']:.0%}",
            ),
            unsafe_allow_html=True,
        )
    with risk_cols[2]:
        var_display = f"{risk_metrics['var_95']:.2%}" if risk_metrics["var_95"] is not None else "N/A"
        st.markdown(
            _metric_html("VaR 95%", var_display),
            unsafe_allow_html=True,
        )
    with risk_cols[3]:
        sharpe_display = f"{risk_metrics['sharpe_ratio']:.2f}" if risk_metrics["sharpe_ratio"] is not None else "N/A"
        st.markdown(
            _metric_html("Sharpe", sharpe_display),
            unsafe_allow_html=True,
        )
    with risk_cols[4]:
        mdd_display = f"{risk_metrics['max_drawdown']:.2%}" if risk_metrics["max_drawdown"] is not None else "N/A"
        st.markdown(
            _metric_html("Max Drawdown", mdd_display),
            unsafe_allow_html=True,
        )

    # Risk flags
    if risk_metrics["flags"]:
        for flag in risk_metrics["flags"]:
            st.warning(flag)

    # --- Dissenting opinions ---
    if dissents:
        st.markdown("---")
        st.markdown('<h4 class="rtl">דעות חולקות</h4>', unsafe_allow_html=True)
        for d in dissents:
            st.markdown(f'<div class="rtl" style="background:var(--panel); padding:10px 16px; border-radius:6px; margin-bottom:6px; border-left:3px solid #FFA726;">{d}</div>', unsafe_allow_html=True)

# ===================================================================== #
# TAB 4 -- Calculator                                                     #
# ===================================================================== #

with tab_calc:
    st.markdown('<h3 class="rtl">מחשבון גודל פוזיציה</h3>', unsafe_allow_html=True)

    calc_col1, calc_col2 = st.columns(2)

    with calc_col1:
        st.markdown('<p class="rtl" style="font-weight:600;">קלטים</p>', unsafe_allow_html=True)

        entry_price = st.number_input(
            "מחיר כניסה ($)",
            min_value=0.01,
            value=round(last_close, 2),
            step=0.01,
            format="%.2f",
        )

        stop_mode = st.radio(
            "שיטת Stop Loss",
            options=["ידני", "ATR"],
            horizontal=True,
        )

        if stop_mode == "ATR" and atr_val is not None:
            sl_price = round(entry_price - atr_val * atr_multiplier, 2)
            st.markdown(
                f'<p style="color:var(--text-dim); font-size:0.85rem;">'
                f'ATR={atr_val:.2f} x {atr_multiplier} = ${atr_val * atr_multiplier:.2f} below entry</p>',
                unsafe_allow_html=True,
            )
        else:
            sl_price = round(entry_price * 0.95, 2)

        stop_loss_price = st.number_input(
            "Stop Loss ($)",
            min_value=0.01,
            value=sl_price,
            step=0.01,
            format="%.2f",
        )

        calc_account = st.number_input(
            "גודל חשבון ($)",
            min_value=1000,
            value=account_size,
            step=1000,
            key="calc_account",
        )

        calc_risk_pct = st.slider(
            "סיכון (%)",
            min_value=0.25,
            max_value=5.0,
            value=risk_pct,
            step=0.25,
            key="calc_risk_pct",
        )

        calc_reward = st.selectbox(
            "יחס סיכון:תגמול",
            options=["1:1.5", "1:2", "1:3", "1:4"],
            index=["1:1.5", "1:2", "1:3", "1:4"].index(reward_ratio),
            key="calc_reward",
        )

    # --- Calculations ---
    r_value = entry_price - stop_loss_price
    if r_value <= 0:
        with calc_col2:
            st.error("מחיר ה-Stop Loss חייב להיות נמוך ממחיר הכניסה.")
    else:
        risk_dollars = calc_account * (calc_risk_pct / 100)
        shares = int(risk_dollars / r_value)
        reward_mult = float(calc_reward.split(":")[1])
        tp_price = entry_price + r_value * reward_mult
        potential_profit = shares * r_value * reward_mult
        position_value = shares * entry_price
        position_pct = (position_value / calc_account) * 100 if calc_account > 0 else 0

        with calc_col2:
            st.markdown('<p class="rtl" style="font-weight:600;">תוצאות</p>', unsafe_allow_html=True)

            res_cols = st.columns(2)
            with res_cols[0]:
                st.markdown(
                    _metric_html("R Value", f"${r_value:.2f}"),
                    unsafe_allow_html=True,
                )
            with res_cols[1]:
                st.markdown(
                    _metric_html("כמות מניות", f"{shares:,}"),
                    unsafe_allow_html=True,
                )

            res_cols2 = st.columns(2)
            with res_cols2[0]:
                st.markdown(
                    _metric_html("סיכון ($)", f"${risk_dollars:,.2f}", f"{calc_risk_pct:.2f}% מהחשבון", False),
                    unsafe_allow_html=True,
                )
            with res_cols2[1]:
                st.markdown(
                    _metric_html("Take Profit", f"${tp_price:.2f}"),
                    unsafe_allow_html=True,
                )

            res_cols3 = st.columns(2)
            with res_cols3[0]:
                st.markdown(
                    _metric_html(
                        "רווח פוטנציאלי",
                        f"${potential_profit:,.2f}",
                        f"{calc_reward}",
                        True,
                    ),
                    unsafe_allow_html=True,
                )
            with res_cols3[1]:
                st.markdown(
                    _metric_html(
                        "שווי פוזיציה",
                        f"${position_value:,.2f}",
                        f"{position_pct:.1f}% מהחשבון",
                        position_pct <= 10,
                    ),
                    unsafe_allow_html=True,
                )

            # --- Risk/Reward visual bar ---
            st.markdown("")
            st.markdown('<p style="font-weight:600; text-align:center;">Risk / Reward</p>', unsafe_allow_html=True)

            total_range = r_value + r_value * reward_mult
            risk_frac = r_value / total_range if total_range > 0 else 0.5
            reward_frac = 1 - risk_frac

            fig_rr = go.Figure()
            fig_rr.add_trace(go.Bar(
                x=[risk_frac],
                y=["R/R"],
                orientation="h",
                marker_color=_DOWN,
                name=f"Risk ${risk_dollars:,.0f}",
                text=f"Risk ${risk_dollars:,.0f}",
                textposition="inside",
                textfont=dict(color="white", size=13),
                hoverinfo="name",
            ))
            fig_rr.add_trace(go.Bar(
                x=[reward_frac],
                y=["R/R"],
                orientation="h",
                marker_color=_UP,
                name=f"Reward ${potential_profit:,.0f}",
                text=f"Reward ${potential_profit:,.0f}",
                textposition="inside",
                textfont=dict(color="white", size=13),
                hoverinfo="name",
            ))
            _apply_layout(
                fig_rr,
                height=80,
                barmode="stack",
                showlegend=False,
                margin=dict(l=0, r=0, t=0, b=0),
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
            )
            st.plotly_chart(fig_rr, use_container_width=True)

            # Price levels on a mini chart
            fig_levels = go.Figure()
            fig_levels.add_trace(go.Scatter(
                x=["Stop Loss", "Entry", "Take Profit"],
                y=[stop_loss_price, entry_price, tp_price],
                mode="lines+markers+text",
                text=[f"${stop_loss_price:.2f}", f"${entry_price:.2f}", f"${tp_price:.2f}"],
                textposition="top center",
                textfont=dict(size=13),
                marker=dict(size=14, color=[_DOWN, _ACCENT, _UP]),
                line=dict(color="#546E7A", width=2, dash="dot"),
                showlegend=False,
            ))
            _apply_layout(
                fig_levels,
                height=220,
                title=dict(text="רמות מחיר", font=dict(size=13)),
            )
            st.plotly_chart(fig_levels, use_container_width=True)

# ===================================================================== #
#                          FOOTER                                         #
# ===================================================================== #

st.markdown("---")
st.markdown(
    '<p style="text-align:center; color:var(--text-dim); font-size:0.75rem;">'
    "AI Broker Terminal | Data: Yahoo Finance | "
    "האותות אינם המלצה להשקעה. "
    "השתמש באחריותך."
    "</p>",
    unsafe_allow_html=True,
)
