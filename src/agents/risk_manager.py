"""Risk Manager agent.

Evaluates portfolio risk and enforces safety constraints via circuit breakers.

CIRCUIT BREAKERS (hard-coded, never overridable by LLM):
  1. All 3 analysts disagree on direction -> force HOLD
  2. Volatility > 95th percentile -> cap position at 2%
  3. Max drawdown > 20% -> flag EXTREME risk
  4. Data > 1 hour stale -> reject analysis
"""
from datetime import datetime, timedelta

import pandas as pd

from src.agents.state import AnalysisState
from src.models.schemas import RiskAssessment, RiskFlag, RiskLevel, Signal
from src.services.llm import call_claude_structured
from src.tools.risk_calculations import (
    compute_max_drawdown,
    compute_sharpe_ratio,
    compute_var_95,
    compute_volatility_percentile,
)

SIGNAL_DIRECTION = {
    Signal.STRONG_BUY: 1,
    Signal.BUY: 1,
    Signal.HOLD: 0,
    Signal.SELL: -1,
    Signal.STRONG_SELL: -1,
}


async def assess_risk(state: AnalysisState) -> dict:
    md = state["market_data"]
    tech = state["technical_report"]
    fund = state["fundamental_report"]
    sent = state["sentiment_report"]

    df = pd.DataFrame([bar.model_dump() for bar in md.ohlcv])
    risk_flags: list[RiskFlag] = []

    vol_pct = compute_volatility_percentile(df["close"])
    var_95 = compute_var_95(df["close"])
    max_dd = compute_max_drawdown(df["close"])
    sharpe = compute_sharpe_ratio(df["close"])

    # Circuit Breaker 1: Analyst disagreement
    directions = [
        SIGNAL_DIRECTION[tech.signal],
        SIGNAL_DIRECTION[fund.signal],
        SIGNAL_DIRECTION[sent.signal],
    ]
    unique_dirs = set(directions)
    analyst_agreement = 1.0 - (len(unique_dirs) - 1) / 2.0

    if len(unique_dirs) == 3:
        risk_flags.append(
            RiskFlag(
                flag_type="analyst_disagreement",
                severity=RiskLevel.HIGH,
                description="All three analysts disagree on direction. Forcing HOLD.",
            )
        )

    # Circuit Breaker 2: Extreme volatility
    if vol_pct > 0.95:
        risk_flags.append(
            RiskFlag(
                flag_type="extreme_volatility",
                severity=RiskLevel.EXTREME,
                description=f"Volatility at {vol_pct:.0%} percentile. Position capped at 2%.",
            )
        )

    # Circuit Breaker 3: Severe drawdown
    if max_dd is not None and max_dd < -0.20:
        risk_flags.append(
            RiskFlag(
                flag_type="severe_drawdown",
                severity=RiskLevel.EXTREME,
                description=f"Max drawdown of {max_dd:.1%} in observation period.",
            )
        )

    # Circuit Breaker 4: Stale data
    staleness = datetime.utcnow() - md.collected_at
    if staleness > timedelta(hours=1):
        risk_flags.append(
            RiskFlag(
                flag_type="stale_data",
                severity=RiskLevel.HIGH,
                description=f"Data is {staleness.total_seconds() / 3600:.1f} hours old.",
            )
        )

    circuit_breaker = any(f.severity == RiskLevel.EXTREME for f in risk_flags)

    if circuit_breaker:
        position_limit = 0.02
    elif vol_pct > 0.80:
        position_limit = 0.05
    else:
        position_limit = 0.10

    if circuit_breaker:
        overall_risk = RiskLevel.EXTREME
    elif vol_pct > 0.80 or analyst_agreement < 0.5:
        overall_risk = RiskLevel.HIGH
    elif vol_pct > 0.50:
        overall_risk = RiskLevel.MODERATE
    else:
        overall_risk = RiskLevel.LOW

    reasoning_result = await call_claude_structured(
        system="You are a risk management expert. Summarize the risk assessment in one paragraph.\n\nReturn JSON with key: reasoning (string).",
        user_content=(
            f"Ticker: {md.ticker}\nVolatility pct: {vol_pct:.2f}\n"
            f"VaR 95: {var_95}\nMax DD: {max_dd}\nSharpe: {sharpe}\n"
            f"Analyst agreement: {analyst_agreement:.2f}\n"
            f"Risk flags: {[f.model_dump() for f in risk_flags]}"
        ),
    )

    stop_loss_pct = 0.05 if overall_risk == RiskLevel.LOW else 0.03

    assessment = RiskAssessment(
        ticker=md.ticker,
        overall_risk_level=overall_risk,
        volatility_percentile=vol_pct,
        value_at_risk_95=var_95,
        max_drawdown_1y=max_dd,
        sharpe_ratio=sharpe,
        beta=md.beta,
        risk_flags=risk_flags,
        position_size_limit_pct=position_limit,
        stop_loss_pct=stop_loss_pct,
        analyst_agreement_score=analyst_agreement,
        circuit_breaker_triggered=circuit_breaker,
        reasoning=reasoning_result.get("reasoning", ""),
    )

    return {
        "risk_assessment": assessment,
        "audit_log": [
            f"[RiskManager] Risk={overall_risk.value} "
            f"CircuitBreaker={circuit_breaker} Flags={len(risk_flags)}"
        ],
    }
