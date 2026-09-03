"""PostgreSQL persistence for analysis audit trail."""
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from src.config.settings import Settings
from src.models.database import AnalysisRun

_engine = None


async def get_session() -> AsyncSession:
    global _engine
    if _engine is None:
        settings = Settings()
        _engine = create_async_engine(settings.database_url)
    async_session = sessionmaker(_engine, class_=AsyncSession, expire_on_commit=False)
    return async_session()


async def save_analysis_run(**kwargs) -> None:
    session = await get_session()
    async with session.begin():
        run = AnalysisRun(
            request_id=kwargs["request"].request_id,
            ticker=kwargs["request"].ticker,
            market_data_json=(
                kwargs["market_data"].model_dump_json()
                if kwargs.get("market_data")
                else None
            ),
            technical_report_json=(
                kwargs["technical_report"].model_dump_json()
                if kwargs.get("technical_report")
                else None
            ),
            fundamental_report_json=(
                kwargs["fundamental_report"].model_dump_json()
                if kwargs.get("fundamental_report")
                else None
            ),
            sentiment_report_json=(
                kwargs["sentiment_report"].model_dump_json()
                if kwargs.get("sentiment_report")
                else None
            ),
            risk_assessment_json=(
                kwargs["risk_assessment"].model_dump_json()
                if kwargs.get("risk_assessment")
                else None
            ),
            recommendation_json=kwargs["recommendation"].model_dump_json(),
            action=kwargs["recommendation"].action.value,
            confidence=kwargs["recommendation"].confidence,
        )
        session.add(run)
