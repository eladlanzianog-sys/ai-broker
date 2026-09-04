"""Inter-agent communication schemas.

All data flowing between agents is defined here as Pydantic models.
These serve as both validation and documentation of the agent protocol.
"""
from __future__ import annotations

import enum
from datetime import datetime

from pydantic import BaseModel, Field


class Signal(str, enum.Enum):
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


class RiskLevel(str, enum.Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    EXTREME = "extreme"


class AnalysisRequest(BaseModel):
    ticker: str = Field(..., pattern=r"^[A-Z0-9.\-]{1,10}$", description="Stock ticker symbol, e.g. 'AAPL'")
    date_range_days: int = Field(default=365, ge=30, le=1825)
    request_id: str = Field(..., description="Unique idempotency key")
    requested_at: datetime = Field(default_factory=datetime.utcnow)


class OHLCVBar(BaseModel):
    date: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int


class FinancialStatements(BaseModel):
    revenue_ttm: float | None = None
    net_income_ttm: float | None = None
    total_debt: float | None = None
    total_cash: float | None = None
    free_cash_flow_ttm: float | None = None
    pe_ratio: float | None = None
    pb_ratio: float | None = None
    ps_ratio: float | None = None
    debt_to_equity: float | None = None
    current_ratio: float | None = None
    roe: float | None = None
    revenue_growth_yoy: float | None = None
    earnings_growth_yoy: float | None = None
    market_cap: float | None = None
    sector: str | None = None
    industry: str | None = None


class NewsItem(BaseModel):
    title: str
    source: str
    published_at: datetime
    url: str
    summary: str | None = None


class MarketData(BaseModel):
    ticker: str
    collected_at: datetime
    current_price: float
    ohlcv: list[OHLCVBar]
    financials: FinancialStatements
    news: list[NewsItem]
    fifty_two_week_high: float
    fifty_two_week_low: float
    average_volume_10d: int
    beta: float | None = None


class TechnicalIndicators(BaseModel):
    rsi_14: float | None = None
    macd_line: float | None = None
    macd_signal: float | None = None
    macd_histogram: float | None = None
    sma_20: float | None = None
    sma_50: float | None = None
    sma_200: float | None = None
    ema_12: float | None = None
    ema_26: float | None = None
    bollinger_upper: float | None = None
    bollinger_middle: float | None = None
    bollinger_lower: float | None = None
    atr_14: float | None = None
    obv: float | None = None
    stochastic_k: float | None = None
    stochastic_d: float | None = None


class TechnicalReport(BaseModel):
    ticker: str
    indicators: TechnicalIndicators
    detected_patterns: list[str]
    trend_direction: str
    support_level: float | None = None
    resistance_level: float | None = None
    signal: Signal
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str


class FundamentalReport(BaseModel):
    ticker: str
    valuation_assessment: str
    financial_health_score: float = Field(..., ge=0.0, le=1.0)
    growth_score: float = Field(..., ge=0.0, le=1.0)
    profitability_score: float = Field(..., ge=0.0, le=1.0)
    intrinsic_value_estimate: float | None = None
    margin_of_safety: float | None = None
    key_strengths: list[str]
    key_risks: list[str]
    signal: Signal
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str


class SentimentScore(BaseModel):
    source: str
    score: float = Field(..., ge=-1.0, le=1.0)
    article_count: int


class SentimentReport(BaseModel):
    ticker: str
    overall_sentiment: float = Field(..., ge=-1.0, le=1.0)
    sentiment_by_source: list[SentimentScore]
    key_themes: list[str]
    notable_events: list[str]
    signal: Signal
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str


class RiskFlag(BaseModel):
    flag_type: str
    severity: RiskLevel
    description: str


class RiskAssessment(BaseModel):
    ticker: str
    overall_risk_level: RiskLevel
    volatility_percentile: float = Field(..., ge=0.0, le=1.0)
    value_at_risk_95: float | None = None
    max_drawdown_1y: float | None = None
    sharpe_ratio: float | None = None
    beta: float | None = None
    risk_flags: list[RiskFlag]
    position_size_limit_pct: float = Field(..., ge=0.0, le=1.0)
    stop_loss_pct: float | None = None
    analyst_agreement_score: float = Field(..., ge=0.0, le=1.0)
    circuit_breaker_triggered: bool = False
    reasoning: str


class FinalRecommendation(BaseModel):
    ticker: str
    action: Signal
    confidence: float = Field(..., ge=0.0, le=1.0)
    weighted_score: float = Field(..., ge=-1.0, le=1.0)
    technical_weight: float
    fundamental_weight: float
    sentiment_weight: float
    risk_adjusted: bool
    position_size_suggestion_pct: float = Field(..., ge=0.0, le=1.0)
    entry_price: float | None = None
    stop_loss_price: float | None = None
    take_profit_price: float | None = None
    time_horizon: str
    reasoning: str
    dissenting_opinions: list[str]
    generated_at: datetime = Field(default_factory=datetime.utcnow)
