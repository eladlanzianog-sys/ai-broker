"""Market Data Collector agent.

Deterministic agent — no LLM calls. Wraps yfinance and news APIs.
"""
from datetime import datetime, timedelta

from src.agents.state import AnalysisState
from src.models.schemas import MarketData
from src.tools.market_data import fetch_company_info, fetch_financials, fetch_ohlcv
from src.tools.news_fetcher import fetch_news


async def collect_market_data(state: AnalysisState) -> dict:
    request = state["request"]
    ticker = request.ticker
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=request.date_range_days)

    ohlcv = await fetch_ohlcv(ticker, start_date, end_date)
    financials = await fetch_financials(ticker)
    info = await fetch_company_info(ticker)
    news = await fetch_news(ticker, days_back=7)

    market_data = MarketData(
        ticker=ticker,
        collected_at=datetime.utcnow(),
        current_price=ohlcv[-1].close if ohlcv else 0.0,
        ohlcv=ohlcv,
        financials=financials,
        news=news,
        fifty_two_week_high=info.get("fifty_two_week_high", 0.0),
        fifty_two_week_low=info.get("fifty_two_week_low", 0.0),
        average_volume_10d=info.get("average_volume_10d", 0),
        beta=info.get("beta"),
    )

    return {
        "market_data": market_data,
        "audit_log": [
            f"[MarketDataCollector] Collected {len(ohlcv)} bars, "
            f"{len(news)} news items for {ticker}"
        ],
    }
