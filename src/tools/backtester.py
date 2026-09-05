"""Backtesting engine for the AI Broker trading platform.

Simulates trades on historical OHLCV data using the same technical
indicator logic the live platform relies on (RSI, MACD, SMA crossovers).
Positions are sized by risk percentage, with ATR-based stop losses and
reward-ratio-based take profits.

Pure Python + pandas/numpy — no extra dependencies.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from src.models.schemas import Signal
from src.tools.technical_indicators import (
    compute_atr,
    compute_ema,
    compute_macd,
    compute_rsi,
    compute_sma,
    detect_patterns,
)

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class Trade:
    """A single completed (or still-open) trade."""

    entry_date: pd.Timestamp
    exit_date: pd.Timestamp | None
    direction: Literal["long", "short"]
    entry_price: float
    exit_price: float | None
    stop_loss: float
    take_profit: float
    shares: float
    pnl: float = 0.0
    pnl_pct: float = 0.0
    exit_reason: str = ""  # "stop_loss", "take_profit", "signal_exit", "end_of_data"


@dataclass
class BacktestResult:
    """Comprehensive output of a backtest run."""

    trades: list[Trade] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    total_return_pct: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    initial_capital: float = 0.0
    final_capital: float = 0.0


# ---------------------------------------------------------------------------
# Signal generation (mirrors the live platform logic)
# ---------------------------------------------------------------------------

_MIN_HISTORY = 50  # bars needed before we start generating signals


def _generate_signal(df_slice: pd.DataFrame) -> Signal:
    """Derive a composite signal from RSI, MACD and SMA crossovers.

    ``df_slice`` must contain at least ``_MIN_HISTORY`` rows so that every
    indicator has enough look-back data.
    """
    close = df_slice["close"]
    score = 0  # accumulator: positive = bullish, negative = bearish

    # --- RSI ----------------------------------------------------------
    rsi = compute_rsi(close, period=14)
    if rsi is not None:
        if rsi < 30:
            score += 2
        elif rsi < 40:
            score += 1
        elif rsi > 70:
            score -= 2
        elif rsi > 60:
            score -= 1

    # --- MACD ---------------------------------------------------------
    macd = compute_macd(close, fast=12, slow=26, signal=9)
    if macd["macd_histogram"] is not None:
        if macd["macd_histogram"] > 0 and macd["macd_line"] is not None and macd["macd_line"] > 0:
            score += 2
        elif macd["macd_histogram"] > 0:
            score += 1
        elif macd["macd_histogram"] < 0 and macd["macd_line"] is not None and macd["macd_line"] < 0:
            score -= 2
        elif macd["macd_histogram"] < 0:
            score -= 1

    # --- SMA crossover (20 / 50) -------------------------------------
    sma_20 = compute_sma(close, 20)
    sma_50 = compute_sma(close, 50)
    if sma_20 is not None and sma_50 is not None:
        if sma_20 > sma_50:
            score += 1
        else:
            score -= 1

    # --- EMA trend (12 vs 26) ----------------------------------------
    ema_12 = compute_ema(close, 12)
    ema_26 = compute_ema(close, 26)
    if ema_12 is not None and ema_26 is not None:
        if ema_12 > ema_26:
            score += 1
        else:
            score -= 1

    # --- Pattern bonus ------------------------------------------------
    patterns = detect_patterns(df_slice)
    if "golden_cross" in patterns:
        score += 2
    if "death_cross" in patterns:
        score -= 2
    if "hammer" in patterns:
        score += 1

    # --- Map score to Signal ------------------------------------------
    if score >= 5:
        return Signal.STRONG_BUY
    if score >= 2:
        return Signal.BUY
    if score <= -5:
        return Signal.STRONG_SELL
    if score <= -2:
        return Signal.SELL
    return Signal.HOLD


# ---------------------------------------------------------------------------
# Core back-test loop
# ---------------------------------------------------------------------------


def run_backtest(
    df: pd.DataFrame,
    *,
    atr_mult: float = 2.0,
    reward_ratio: float = 2.0,
    risk_pct: float = 1.0,
    initial_capital: float = 100_000,
) -> BacktestResult:
    """Run a backtest over *df* and return a :class:`BacktestResult`.

    Parameters
    ----------
    df:
        OHLCV DataFrame with columns ``open, high, low, close, volume`` and
        a ``DatetimeIndex``.
    atr_mult:
        Multiplier applied to the ATR to set the stop-loss distance.
    reward_ratio:
        Take-profit distance as a multiple of the stop-loss distance.
    risk_pct:
        Percentage of current equity risked per trade (position sizing).
    initial_capital:
        Starting cash balance.
    """
    # ---- validation --------------------------------------------------
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing columns: {missing}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex")
    if len(df) < _MIN_HISTORY:
        raise ValueError(f"Need at least {_MIN_HISTORY} bars, got {len(df)}")

    df = df.sort_index().copy()

    # ---- state -------------------------------------------------------
    capital = float(initial_capital)
    equity_values: list[float] = []
    equity_dates: list[pd.Timestamp] = []
    trades: list[Trade] = []
    position: Trade | None = None
    prev_signal: Signal = Signal.HOLD

    # ---- iterate bar-by-bar -----------------------------------------
    for i in range(len(df)):
        bar = df.iloc[i]
        bar_date = df.index[i]

        # --- check open-position exits first --------------------------
        if position is not None:
            closed = False
            if position.direction == "long":
                if bar["low"] <= position.stop_loss:
                    _close_trade(position, position.stop_loss, bar_date, "stop_loss")
                    closed = True
                elif bar["high"] >= position.take_profit:
                    _close_trade(position, position.take_profit, bar_date, "take_profit")
                    closed = True
            else:  # short
                if bar["high"] >= position.stop_loss:
                    _close_trade(position, position.stop_loss, bar_date, "stop_loss")
                    closed = True
                elif bar["low"] <= position.take_profit:
                    _close_trade(position, position.take_profit, bar_date, "take_profit")
                    closed = True

            if closed:
                capital += position.pnl + position.entry_price * position.shares
                trades.append(position)
                position = None

        # --- generate signal if we have enough history ----------------
        if i < _MIN_HISTORY:
            equity_values.append(capital if position is None else capital + _unrealized(position, bar["close"]))
            equity_dates.append(bar_date)
            continue

        window = df.iloc[max(0, i - 250) : i + 1]  # up to ~250-bar look-back
        signal = _generate_signal(window)

        atr_val = compute_atr(window, period=14)

        # --- entry logic ----------------------------------------------
        if position is None and atr_val is not None and atr_val > 0:
            entry_price = bar["close"]
            stop_dist = atr_mult * atr_val

            if signal in (Signal.STRONG_BUY, Signal.BUY):
                sl = entry_price - stop_dist
                tp = entry_price + stop_dist * reward_ratio
                shares = _position_size(capital, risk_pct, stop_dist)
                if shares > 0:
                    position = Trade(
                        entry_date=bar_date,
                        exit_date=None,
                        direction="long",
                        entry_price=entry_price,
                        exit_price=None,
                        stop_loss=sl,
                        take_profit=tp,
                        shares=shares,
                    )
                    capital -= entry_price * shares

            elif signal in (Signal.STRONG_SELL, Signal.SELL):
                sl = entry_price + stop_dist
                tp = entry_price - stop_dist * reward_ratio
                shares = _position_size(capital, risk_pct, stop_dist)
                if shares > 0:
                    position = Trade(
                        entry_date=bar_date,
                        exit_date=None,
                        direction="short",
                        entry_price=entry_price,
                        exit_price=None,
                        stop_loss=sl,
                        take_profit=tp,
                        shares=shares,
                    )
                    capital -= entry_price * shares  # margin collateral

        # --- signal-based exit (reversal) -----------------------------
        elif position is not None:
            if position.direction == "long" and signal in (Signal.SELL, Signal.STRONG_SELL):
                _close_trade(position, bar["close"], bar_date, "signal_exit")
                capital += position.pnl + position.entry_price * position.shares
                trades.append(position)
                position = None
            elif position.direction == "short" and signal in (Signal.BUY, Signal.STRONG_BUY):
                _close_trade(position, bar["close"], bar_date, "signal_exit")
                capital += position.pnl + position.entry_price * position.shares
                trades.append(position)
                position = None

        prev_signal = signal

        # --- track equity ---------------------------------------------
        equity = capital if position is None else capital + _unrealized(position, bar["close"])
        equity_values.append(equity)
        equity_dates.append(bar_date)

    # --- close any open position at end of data -----------------------
    if position is not None:
        last_close = float(df.iloc[-1]["close"])
        last_date = df.index[-1]
        _close_trade(position, last_close, last_date, "end_of_data")
        capital += position.pnl + position.entry_price * position.shares
        trades.append(position)
        # update final equity
        if equity_values:
            equity_values[-1] = capital

    # --- build result -------------------------------------------------
    equity_curve = pd.Series(equity_values, index=equity_dates, name="equity")

    return _build_result(
        trades=trades,
        equity_curve=equity_curve,
        initial_capital=initial_capital,
        final_capital=capital,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _position_size(capital: float, risk_pct: float, stop_dist: float) -> float:
    """Calculate the number of shares to buy given risk parameters.

    Returns fractional shares (the caller can floor if needed).
    """
    if stop_dist <= 0 or capital <= 0:
        return 0.0
    risk_amount = capital * (risk_pct / 100.0)
    shares = risk_amount / stop_dist
    # Ensure position cost does not exceed available capital
    max_shares = capital / (stop_dist / (risk_pct / 100.0)) if stop_dist > 0 else 0.0
    return min(shares, max_shares)


def _unrealized(position: Trade, current_price: float) -> float:
    """Mark-to-market unrealised PnL for an open position."""
    if position.direction == "long":
        return (current_price - position.entry_price) * position.shares
    return (position.entry_price - current_price) * position.shares


def _close_trade(
    trade: Trade,
    exit_price: float,
    exit_date: pd.Timestamp,
    reason: str,
) -> None:
    """Mutate *trade* in place to record the exit."""
    trade.exit_price = exit_price
    trade.exit_date = exit_date
    trade.exit_reason = reason
    if trade.direction == "long":
        trade.pnl = (exit_price - trade.entry_price) * trade.shares
    else:
        trade.pnl = (trade.entry_price - exit_price) * trade.shares
    if trade.entry_price != 0:
        trade.pnl_pct = trade.pnl / (trade.entry_price * trade.shares) * 100.0


def _build_result(
    *,
    trades: list[Trade],
    equity_curve: pd.Series,
    initial_capital: float,
    final_capital: float,
) -> BacktestResult:
    result = BacktestResult(
        trades=trades,
        equity_curve=equity_curve,
        initial_capital=initial_capital,
        final_capital=final_capital,
        total_trades=len(trades),
    )

    if not trades:
        return result

    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]
    result.winning_trades = len(wins)
    result.losing_trades = len(losses)
    result.win_rate = len(wins) / len(trades) * 100.0

    gross_profit = sum(t.pnl for t in wins) if wins else 0.0
    gross_loss = abs(sum(t.pnl for t in losses)) if losses else 0.0
    result.profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf") if gross_profit > 0 else 0.0

    result.avg_win_pct = (sum(t.pnl_pct for t in wins) / len(wins)) if wins else 0.0
    result.avg_loss_pct = (sum(t.pnl_pct for t in losses) / len(losses)) if losses else 0.0

    result.total_return_pct = (final_capital - initial_capital) / initial_capital * 100.0

    # --- max drawdown -------------------------------------------------
    if len(equity_curve) > 0:
        running_max = equity_curve.cummax()
        drawdowns = (equity_curve - running_max) / running_max
        result.max_drawdown_pct = float(drawdowns.min()) * 100.0  # negative

    # --- Sharpe ratio (annualised, assuming daily bars) ----------------
    if len(equity_curve) > 1:
        returns = equity_curve.pct_change().dropna()
        if returns.std() > 0:
            result.sharpe_ratio = float(returns.mean() / returns.std() * np.sqrt(252))

    return result
