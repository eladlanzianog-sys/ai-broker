"""Shared test fixtures."""
import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from src.models.schemas import (
    AnalysisRequest,
    FinancialStatements,
    FinalRecommendation,
    FundamentalReport,
    MarketData,
    NewsItem,
    OHLCVBar,
    RiskAssessment,
    RiskFlag,
    RiskLevel,
    SentimentReport,
    SentimentScore,
    Signal,
    TechnicalIndicators,
    TechnicalReport,
)


@pytest.fixture
def sample_request():
    return AnalysisRequest(
        ticker="AAPL",
        date_range_days=365,
        request_id="test-req-001",
    )


@pytest.fixture
def sample_ohlcv():
    import numpy as np

    np.random.seed(42)
    base_price = 150.0
    bars = []
    for i in range(252):
        date = datetime(2025, 1, 1) + timedelta(days=i)
        change = np.random.normal(0.0005, 0.015)
        base_price *= 1 + change
        high = base_price * (1 + abs(np.random.normal(0, 0.005)))
        low = base_price * (1 - abs(np.random.normal(0, 0.005)))
        bars.append(
            OHLCVBar(
                date=date,
                open=round(base_price * (1 + np.random.normal(0, 0.002)), 2),
                high=round(high, 2),
                low=round(low, 2),
                close=round(base_price, 2),
                volume=int(np.random.uniform(50_000_000, 120_000_000)),
            )
        )
    return bars


@pytest.fixture
def sample_financials():
    return FinancialStatements(
        revenue_ttm=394_000_000_000,
        net_income_ttm=97_000_000_000,
        total_debt=111_000_000_000,
        total_cash=62_000_000_000,
        free_cash_flow_ttm=112_000_000_000,
        pe_ratio=28.5,
        pb_ratio=45.0,
        ps_ratio=7.5,
        debt_to_equity=1.73,
        current_ratio=1.07,
        roe=1.47,
        revenue_growth_yoy=0.08,
        earnings_growth_yoy=0.11,
        market_cap=2_800_000_000_000,
        sector="Technology",
        industry="Consumer Electronics",
    )


@pytest.fixture
def sample_market_data(sample_ohlcv, sample_financials):
    return MarketData(
        ticker="AAPL",
        collected_at=datetime.utcnow(),
        current_price=sample_ohlcv[-1].close,
        ohlcv=sample_ohlcv,
        financials=sample_financials,
        news=[
            NewsItem(
                title="Apple Reports Strong Q4 Earnings",
                source="Reuters",
                published_at=datetime.utcnow() - timedelta(hours=6),
                url="https://example.com/1",
                summary="Apple beat revenue expectations...",
            ),
            NewsItem(
                title="New iPhone Launch Exceeds Expectations",
                source="Bloomberg",
                published_at=datetime.utcnow() - timedelta(hours=12),
                url="https://example.com/2",
                summary="Initial sales figures show strong demand...",
            ),
        ],
        fifty_two_week_high=199.62,
        fifty_two_week_low=164.08,
        average_volume_10d=65_000_000,
        beta=1.24,
    )


@pytest.fixture
def sample_technical_report():
    return TechnicalReport(
        ticker="AAPL",
        indicators=TechnicalIndicators(
            rsi_14=58.0,
            macd_line=1.5,
            macd_signal=1.2,
            macd_histogram=0.3,
            sma_20=188.0,
            sma_50=185.0,
            sma_200=178.0,
        ),
        detected_patterns=[],
        trend_direction="bullish",
        support_level=182.0,
        resistance_level=199.0,
        signal=Signal.BUY,
        confidence=0.75,
        reasoning="Price above all major MAs, RSI at neutral-bullish 58.",
    )


@pytest.fixture
def sample_fundamental_report():
    return FundamentalReport(
        ticker="AAPL",
        valuation_assessment="fairly_valued",
        financial_health_score=0.65,
        growth_score=0.60,
        profitability_score=0.85,
        intrinsic_value_estimate=200.0,
        margin_of_safety=0.05,
        key_strengths=["Strong cash flow", "High margins"],
        key_risks=["China exposure", "Regulatory risk"],
        signal=Signal.BUY,
        confidence=0.80,
        reasoning="Strong fundamentals with moderate growth.",
    )


@pytest.fixture
def sample_sentiment_report():
    return SentimentReport(
        ticker="AAPL",
        overall_sentiment=0.4,
        sentiment_by_source=[
            SentimentScore(source="Reuters", score=0.5, article_count=3),
            SentimentScore(source="Bloomberg", score=0.3, article_count=2),
        ],
        key_themes=["earnings beat", "product launch"],
        notable_events=["Q4 earnings report"],
        signal=Signal.BUY,
        confidence=0.65,
        reasoning="Generally positive coverage around earnings.",
    )


@pytest.fixture
def sample_risk_assessment():
    return RiskAssessment(
        ticker="AAPL",
        overall_risk_level=RiskLevel.MODERATE,
        volatility_percentile=0.45,
        value_at_risk_95=-0.025,
        max_drawdown_1y=-0.12,
        sharpe_ratio=1.2,
        beta=1.24,
        risk_flags=[],
        position_size_limit_pct=0.10,
        stop_loss_pct=0.05,
        analyst_agreement_score=1.0,
        circuit_breaker_triggered=False,
        reasoning="Moderate risk with acceptable volatility.",
    )
