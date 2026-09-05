"""
Risk-Reward Parameter Optimizer for AI Broker Trading Platform.

Performs grid search over ATR multiplier, reward ratio, and risk percentage
parameters to find optimal trading configurations based on Sharpe ratio.
Works with the backtester module to evaluate each parameter combination.
"""

import itertools
from typing import Any

import numpy as np
import pandas as pd

from src.tools.backtester import run_backtest


def optimize_parameters(
    df: pd.DataFrame,
    atr_range: tuple[float, float, float] = (1.0, 4.0, 0.5),
    rr_range: tuple[float, float, float] = (1.5, 4.0, 0.5),
    risk_range: tuple[float, float, float] = (0.5, 3.0, 0.5),
) -> dict[str, Any]:
    """
    Grid search over ATR multiplier, reward ratio, and risk percentage
    to find the parameter combination that maximizes Sharpe ratio.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with signal columns expected by the backtester.
    atr_range : tuple of (start, stop, step)
        Range for the ATR multiplier parameter (inclusive of stop).
    rr_range : tuple of (start, stop, step)
        Range for the reward ratio parameter (inclusive of stop).
    risk_range : tuple of (start, stop, step)
        Range for the risk percentage parameter (inclusive of stop).

    Returns
    -------
    dict with keys:
        best_params : dict with best atr_mult, reward_ratio, risk_pct
        best_sharpe : float
        best_win_rate : float
        results_matrix : list of dicts (all parameter combos with results)
        heatmap_data : dict with x (atr values), y (rr values),
                       z (sharpe values as 2D list) — ready for Plotly imshow
    """
    atr_values = np.arange(atr_range[0], atr_range[1] + atr_range[2] / 2, atr_range[2])
    rr_values = np.arange(rr_range[0], rr_range[1] + rr_range[2] / 2, rr_range[2])
    risk_values = np.arange(risk_range[0], risk_range[1] + risk_range[2] / 2, risk_range[2])

    # Round to avoid floating point drift
    atr_values = np.round(atr_values, 4)
    rr_values = np.round(rr_values, 4)
    risk_values = np.round(risk_values, 4)

    results_matrix: list[dict[str, Any]] = []
    best_sharpe = -np.inf
    best_params: dict[str, float] = {}
    best_win_rate = 0.0

    combos = list(itertools.product(atr_values, rr_values, risk_values))

    for atr_mult, reward_ratio, risk_pct in combos:
        atr_mult = float(atr_mult)
        reward_ratio = float(reward_ratio)
        risk_pct = float(risk_pct)

        result = run_backtest(
            df,
            atr_mult=atr_mult,
            reward_ratio=reward_ratio,
            risk_pct=risk_pct,
        )

        entry = {
            "atr_mult": atr_mult,
            "reward_ratio": reward_ratio,
            "risk_pct": risk_pct,
            "sharpe_ratio": result.sharpe_ratio,
            "total_return": result.total_return,
            "win_rate": result.win_rate,
            "profit_factor": result.profit_factor,
            "max_drawdown": result.max_drawdown,
            "total_trades": result.total_trades,
        }
        results_matrix.append(entry)

        if result.sharpe_ratio > best_sharpe:
            best_sharpe = result.sharpe_ratio
            best_win_rate = result.win_rate
            best_params = {
                "atr_mult": atr_mult,
                "reward_ratio": reward_ratio,
                "risk_pct": risk_pct,
            }

    # Build heatmap data: average Sharpe across risk_pct values
    # for each (atr_mult, reward_ratio) pair
    heatmap_z: list[list[float]] = []
    for rr in rr_values:
        row: list[float] = []
        for atr in atr_values:
            sharpes = [
                r["sharpe_ratio"]
                for r in results_matrix
                if r["atr_mult"] == float(atr) and r["reward_ratio"] == float(rr)
            ]
            avg_sharpe = float(np.mean(sharpes)) if sharpes else 0.0
            row.append(round(avg_sharpe, 4))
        heatmap_z.append(row)

    heatmap_data = {
        "x": [float(v) for v in atr_values],
        "y": [float(v) for v in rr_values],
        "z": heatmap_z,
    }

    return {
        "best_params": best_params,
        "best_sharpe": round(best_sharpe, 4),
        "best_win_rate": round(best_win_rate, 4),
        "results_matrix": results_matrix,
        "heatmap_data": heatmap_data,
    }
