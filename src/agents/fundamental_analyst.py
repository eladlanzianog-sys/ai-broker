"""Fundamental Analyst agent.

Computes financial ratios (deterministic), then uses Claude
for qualitative assessment and reasoning.
"""
from src.agents.state import AnalysisState
from src.models.schemas import FundamentalReport, Signal
from src.services.llm import call_claude_structured
from src.tools.fundamental_metrics import (
    compute_margin_of_safety,
    estimate_intrinsic_value,
    score_financial_health,
    score_growth,
    score_profitability,
)


async def analyze_fundamentals(state: AnalysisState) -> dict:
    md = state["market_data"]
    fin = md.financials

    health_score = score_financial_health(fin)
    growth_score = score_growth(fin)
    profit_score = score_profitability(fin)
    intrinsic_value = estimate_intrinsic_value(fin)
    margin_of_safety = (
        compute_margin_of_safety(intrinsic_value, md.current_price)
        if intrinsic_value
        else None
    )

    assessment = await call_claude_structured(
        system=(
            "You are a fundamental analysis expert. Assess the company's "
            "valuation and financial health.\n\n"
            "Return JSON with keys: valuation_assessment "
            "(undervalued/fairly_valued/overvalued), "
            "key_strengths (list of strings), key_risks (list of strings), "
            "signal (one of: strong_buy, buy, hold, sell, strong_sell), "
            "confidence (float 0-1), reasoning (string)."
        ),
        user_content=(
            f"Ticker: {md.ticker}, Price: {md.current_price}\n"
            f"Health: {health_score:.2f}, Growth: {growth_score:.2f}, "
            f"Profitability: {profit_score:.2f}\n"
            f"Intrinsic Value Est: {intrinsic_value}\n"
            f"Margin of Safety: {margin_of_safety}\n"
            f"Financials: {fin.model_dump_json()}"
        ),
    )

    report = FundamentalReport(
        ticker=md.ticker,
        valuation_assessment=assessment.get("valuation_assessment", "fairly_valued"),
        financial_health_score=health_score,
        growth_score=growth_score,
        profitability_score=profit_score,
        intrinsic_value_estimate=intrinsic_value,
        margin_of_safety=margin_of_safety,
        key_strengths=assessment.get("key_strengths", []),
        key_risks=assessment.get("key_risks", []),
        signal=Signal(assessment["signal"]),
        confidence=assessment["confidence"],
        reasoning=assessment["reasoning"],
    )

    return {
        "fundamental_report": report,
        "audit_log": [
            f"[FundamentalAnalyst] Signal={report.signal.value} "
            f"Confidence={report.confidence:.2f}"
        ],
    }
