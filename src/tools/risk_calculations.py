"""Risk metric calculations.

Pure deterministic functions for portfolio risk assessment.
"""
import numpy as np
import pandas as pd


def compute_volatility_percentile(
    prices: pd.Series,
    lookback: int = 252,
    window: int = 20,
) -> float:
    if len(prices) < window + 1:
        return 0.5
    actual_lookback = min(lookback, len(prices))
    prices = prices.tail(actual_lookback)
    returns = prices.pct_change().dropna()
    rolling_vol = returns.rolling(window=window).std() * np.sqrt(252)
    rolling_vol = rolling_vol.dropna()
    if len(rolling_vol) == 0:
        return 0.5
    current_vol = rolling_vol.iloc[-1]
    percentile = float((rolling_vol < current_vol).sum() / len(rolling_vol))
    return percentile


def compute_var_95(prices: pd.Series, horizon_days: int = 1) -> float | None:
    returns = prices.pct_change().dropna()
    if len(returns) < 30:
        return None
    mean = returns.mean() * horizon_days
    std = returns.std() * np.sqrt(horizon_days)
    var_95 = mean - 1.645 * std
    return float(var_95)


def compute_max_drawdown(prices: pd.Series) -> float | None:
    if len(prices) < 2:
        return None
    cumulative = prices / prices.iloc[0]
    running_max = cumulative.cummax()
    drawdowns = (cumulative - running_max) / running_max
    return float(drawdowns.min())


def compute_sharpe_ratio(
    prices: pd.Series,
    risk_free_rate: float = 0.05,
) -> float | None:
    returns = prices.pct_change().dropna()
    if len(returns) < 30:
        return None
    excess = returns.mean() * 252 - risk_free_rate
    vol = returns.std() * np.sqrt(252)
    if vol == 0:
        return None
    return float(excess / vol)
