"""Technical indicator calculations using pandas and numpy.

Every function here is pure — no side effects, no LLM calls, no network.
"""
import numpy as np
import pandas as pd


def compute_sma(series: pd.Series, period: int) -> float | None:
    if len(series) < period:
        return None
    return float(series.rolling(window=period).mean().iloc[-1])


def compute_ema(series: pd.Series, span: int) -> float | None:
    if len(series) < span:
        return None
    return float(series.ewm(span=span, adjust=False).mean().iloc[-1])


def compute_rsi(series: pd.Series, period: int = 14) -> float | None:
    if len(series) < period + 1:
        return None
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return float(rsi.iloc[-1])


def compute_macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> dict:
    if len(series) < slow + signal:
        return {"macd_line": None, "macd_signal": None, "macd_histogram": None}
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=signal, adjust=False).mean()
    macd_hist = macd_line - macd_signal
    return {
        "macd_line": float(macd_line.iloc[-1]),
        "macd_signal": float(macd_signal.iloc[-1]),
        "macd_histogram": float(macd_hist.iloc[-1]),
    }


def compute_bollinger_bands(
    series: pd.Series,
    period: int = 20,
    std_dev: float = 2.0,
) -> dict:
    if len(series) < period:
        return {"bollinger_upper": None, "bollinger_middle": None, "bollinger_lower": None}
    mid = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    return {
        "bollinger_upper": float((mid + std_dev * std).iloc[-1]),
        "bollinger_middle": float(mid.iloc[-1]),
        "bollinger_lower": float((mid - std_dev * std).iloc[-1]),
    }


def compute_atr(df: pd.DataFrame, period: int = 14) -> float | None:
    if len(df) < period + 1:
        return None
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    return float(atr.iloc[-1])


def compute_obv(df: pd.DataFrame) -> float | None:
    if len(df) < 2:
        return None
    direction = np.sign(df["close"].diff())
    obv = (direction * df["volume"]).cumsum()
    return float(obv.iloc[-1])


def compute_stochastic(
    df: pd.DataFrame,
    k_period: int = 14,
    d_period: int = 3,
) -> dict:
    if len(df) < k_period + d_period:
        return {"stochastic_k": None, "stochastic_d": None}
    low_min = df["low"].rolling(window=k_period).min()
    high_max = df["high"].rolling(window=k_period).max()
    denom = high_max - low_min
    denom = denom.replace(0, np.nan)
    k = 100.0 * (df["close"] - low_min) / denom
    d = k.rolling(window=d_period).mean()
    return {"stochastic_k": float(k.iloc[-1]), "stochastic_d": float(d.iloc[-1])}


def detect_patterns(df: pd.DataFrame) -> list[str]:
    patterns = []
    if len(df) < 5:
        return patterns

    if len(df) >= 200:
        sma50 = df["close"].rolling(50).mean()
        sma200 = df["close"].rolling(200).mean()
        if sma50.iloc[-1] > sma200.iloc[-1] and sma50.iloc[-2] <= sma200.iloc[-2]:
            patterns.append("golden_cross")
        if sma50.iloc[-1] < sma200.iloc[-1] and sma50.iloc[-2] >= sma200.iloc[-2]:
            patterns.append("death_cross")

    last = df.iloc[-1]
    body = abs(last["close"] - last["open"])
    wick = last["high"] - last["low"]
    if wick > 0 and body / wick < 0.1:
        patterns.append("doji")

    if wick > 0:
        lower_wick = min(last["open"], last["close"]) - last["low"]
        upper_wick = last["high"] - max(last["open"], last["close"])
        if lower_wick > 2 * body and upper_wick < body:
            patterns.append("hammer")

    return patterns


def identify_support_resistance(
    df: pd.DataFrame,
    window: int = 20,
) -> tuple[float | None, float | None]:
    if len(df) < window:
        return None, None
    recent = df.tail(window)
    support = float(recent["low"].min())
    resistance = float(recent["high"].max())
    return support, resistance
