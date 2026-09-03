"""Portfolio Strategist agent.

Synthesizes all analyst reports and risk assessment into a single recommendation
using weighted confidence-adjusted voting.
"""
from src.agents.state import AnalysisState
from src.config.constants import (
    FUNDAMENTAL_WEIGHT,
    RISK_DAMPENING,
    SCORE_THRESHOLDS,
    SENTIMENT_WEIGHT,
    SIGNAL_TO_SCORE,
    TECHNICAL_WEIGHT,
)
from src.models.schemas import FinalRecommendation, Signal
from src.services.llm import call_claude_structured


async def synthesize_recommendation(state: AnalysisState) -> dict:
    tech = state["technical_report"]
    fund = state["fundamental_report"]
    sent = state["sentiment_report"]
    risk = state["risk_assessment"]
    md = state["market_data"]

    # Convert signals to confidence-weighted numeric scores
    tech_score = SIGNAL_TO_SCORE[tech.signal] * tech.confidence
    fund_score = SIGNAL_TO_SCORE[fund.signal] * fund.confidence
    sent_score = SIGNAL_TO_SCORE[sent.signal] * sent.confidence

    raw_score = (
        tech_score * TECHNICAL_WEIGHT
        + fund_score * FUNDAMENTAL_WEIGHT
        + sent_score * SENTIMENT_WEIGHT
    )

    dampening = RISK_DAMPENING[risk.overall_risk_level]
    adjusted_score = raw_score * dampening

    if risk.circuit_breaker_triggered:
        adjusted_score = 0.0

    if adjusted_score >= SCORE_THRESHOLDS["strong_buy"]:
        action = Signal.STRONG_BUY
    elif adjusted_score >= SCORE_THRESHOLDS["buy"]:
        action = Signal.BUY
    elif adjusted_score <= SCORE_THRESHOLDS["strong_sell"]:
        action = Signal.STRONG_SELL
    elif adjusted_score <= SCORE_THRESHOLDS["sell"]:
        action = Signal.SELL
    else:
        action = Signal.HOLD

    total_confidence = (
        tech.confidence * TECHNICAL_WEIGHT
        + fund.confidence * FUNDAMENTAL_WEIGHT
        + sent.confidence * SENTIMENT_WEIGHT
    ) * dampening

    dissents = []
    if tech.signal != action:
        dissents.append(
            f"Technical analysis suggests {tech.signal.value} "
            f"(confidence: {tech.confidence:.0%})"
        )
    if fund.signal != action:
        dissents.append(
            f"Fundamental analysis suggests {fund.signal.value} "
            f"(confidence: {fund.confidence:.0%})"
        )
    if sent.signal != action:
        dissents.append(
            f"Sentiment analysis suggests {sent.signal.value} "
            f"(confidence: {sent.confidence:.0%})"
        )

    reasoning_result = await call_claude_structured(
        system=(
            "You are a senior portfolio strategist. Write a concise "
            "recommendation justification.\n\n"
            "Return JSON with keys: reasoning (string), "
            "time_horizon (short_term/medium_term/long_term)."
        ),
        user_content=(
            f"Ticker: {md.ticker} | Price: ${md.current_price:.2f}\n"
            f"ACTION: {action.value} (score: {adjusted_score:.3f})\n\n"
            f"Technical: {tech.signal.value} ({tech.confidence:.0%}) - {tech.reasoning}\n"
            f"Fundamental: {fund.signal.value} ({fund.confidence:.0%}) - {fund.reasoning}\n"
            f"Sentiment: {sent.signal.value} ({sent.confidence:.0%}) - {sent.reasoning}\n"
            f"Risk: {risk.overall_risk_level.value} - {risk.reasoning}\n"
            f"Dissents: {dissents}"
        ),
    )

    stop_loss_price = (
        md.current_price * (1 - risk.stop_loss_pct)
        if risk.stop_loss_pct
        else None
    )

    recommendation = FinalRecommendation(
        ticker=md.ticker,
        action=action,
        confidence=round(total_confidence, 3),
        weighted_score=round(adjusted_score, 4),
        technical_weight=TECHNICAL_WEIGHT,
        fundamental_weight=FUNDAMENTAL_WEIGHT,
        sentiment_weight=SENTIMENT_WEIGHT,
        risk_adjusted=True,
        position_size_suggestion_pct=risk.position_size_limit_pct,
        entry_price=md.current_price,
        stop_loss_price=stop_loss_price,
        take_profit_price=tech.resistance_level,
        time_horizon=reasoning_result.get("time_horizon", "medium_term"),
        reasoning=reasoning_result.get("reasoning", ""),
        dissenting_opinions=dissents,
    )

    return {
        "recommendation": recommendation,
        "audit_log": [
            f"[PortfolioStrategist] FINAL: {action.value} "
            f"confidence={total_confidence:.2f} score={adjusted_score:.4f}"
        ],
    }
