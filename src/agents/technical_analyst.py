"""Technical Analyst agent.

Computes technical indicators (deterministic), then uses Claude
for pattern interpretation and reasoning.
"""
import pandas as pd

from src.agents.state import AnalysisState
from src.models.schemas import Signal, TechnicalIndicators, TechnicalReport
from src.services.llm import call_claude_structured
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


async def analyze_technicals(state: AnalysisState) -> dict:
    md = state["market_data"]
    df = pd.DataFrame([bar.model_dump() for bar in md.ohlcv])

    indicators = TechnicalIndicators(
        rsi_14=compute_rsi(df["close"], period=14),
        sma_20=compute_sma(df["close"], 20),
        sma_50=compute_sma(df["close"], 50),
        sma_200=compute_sma(df["close"], 200),
        ema_12=compute_ema(df["close"], 12),
        ema_26=compute_ema(df["close"], 26),
        atr_14=compute_atr(df, period=14),
        obv=compute_obv(df),
        **compute_macd(df["close"]),
        **compute_bollinger_bands(df["close"]),
        **compute_stochastic(df),
    )

    patterns = detect_patterns(df)
    support, resistance = identify_support_resistance(df)

    interpretation = await call_claude_structured(
        system=(
            "You are a technical analysis expert. Given the indicators and "
            "patterns, determine the signal and confidence.\n\n"
            "Return JSON with keys: trend_direction (bullish/bearish/sideways), "
            "signal (one of: strong_buy, buy, hold, sell, strong_sell), "
            "confidence (float 0-1), reasoning (string)."
        ),
        user_content=(
            f"Ticker: {md.ticker}\nPrice: {md.current_price}\n"
            f"Indicators: {indicators.model_dump_json()}\n"
            f"Patterns: {patterns}\n"
            f"Support: {support}, Resistance: {resistance}"
        ),
    )

    report = TechnicalReport(
        ticker=md.ticker,
        indicators=indicators,
        detected_patterns=patterns,
        trend_direction=interpretation.get("trend_direction", "sideways"),
        support_level=support,
        resistance_level=resistance,
        signal=Signal(interpretation["signal"]),
        confidence=interpretation["confidence"],
        reasoning=interpretation["reasoning"],
    )

    return {
        "technical_report": report,
        "audit_log": [
            f"[TechnicalAnalyst] Signal={report.signal.value} "
            f"Confidence={report.confidence:.2f}"
        ],
    }
