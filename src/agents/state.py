"""LangGraph shared state definition.

This TypedDict is the single mutable object that flows through the graph.
Each agent node reads what it needs and writes its output field.
"""
from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from src.models.schemas import (
    AnalysisRequest,
    FinalRecommendation,
    FundamentalReport,
    MarketData,
    RiskAssessment,
    SentimentReport,
    TechnicalReport,
)


class AnalysisState(TypedDict, total=False):
    request: AnalysisRequest
    market_data: MarketData
    technical_report: TechnicalReport
    fundamental_report: FundamentalReport
    sentiment_report: SentimentReport
    risk_assessment: RiskAssessment
    recommendation: FinalRecommendation
    errors: Annotated[list[str], operator.add]
    audit_log: Annotated[list[str], operator.add]
