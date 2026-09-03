"""SQLAlchemy ORM models for the analysis audit trail."""
from datetime import datetime

from sqlalchemy import Column, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class AnalysisRun(Base):
    __tablename__ = "analysis_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    request_id = Column(String(100), unique=True, nullable=False, index=True)
    ticker = Column(String(10), nullable=False, index=True)
    market_data_json = Column(Text, nullable=True)
    technical_report_json = Column(Text, nullable=True)
    fundamental_report_json = Column(Text, nullable=True)
    sentiment_report_json = Column(Text, nullable=True)
    risk_assessment_json = Column(Text, nullable=True)
    recommendation_json = Column(Text, nullable=False)
    action = Column(String(20), nullable=False)
    confidence = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
