"""News article fetcher for sentiment analysis."""
from datetime import datetime, timedelta

import httpx

from src.config.settings import Settings
from src.models.schemas import NewsItem


async def fetch_news(ticker: str, days_back: int = 7) -> list[NewsItem]:
    settings = Settings()
    if not settings.finnhub_api_key:
        return _fetch_from_yfinance(ticker)

    from_date = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    to_date = datetime.utcnow().strftime("%Y-%m-%d")

    async with httpx.AsyncClient() as client:
        resp = await client.get(
            "https://finnhub.io/api/v1/company-news",
            params={
                "symbol": ticker,
                "from": from_date,
                "to": to_date,
                "token": settings.finnhub_api_key,
            },
            timeout=10.0,
        )
        if resp.status_code != 200:
            return _fetch_from_yfinance(ticker)

        articles = resp.json()
        return [
            NewsItem(
                title=a.get("headline", ""),
                source=a.get("source", ""),
                published_at=datetime.fromtimestamp(a.get("datetime", 0)),
                url=a.get("url", ""),
                summary=a.get("summary"),
            )
            for a in articles[:50]
        ]


def _fetch_from_yfinance(ticker: str) -> list[NewsItem]:
    import yfinance as yf

    stock = yf.Ticker(ticker)
    news = stock.news or []
    items = []
    for article in news[:20]:
        content = article.get("content", {})
        items.append(NewsItem(
            title=content.get("title", article.get("title", "")),
            source=content.get("provider", {}).get("displayName", "Yahoo Finance"),
            published_at=datetime.utcnow(),
            url=content.get("canonicalUrl", {}).get("url", ""),
            summary=content.get("summary"),
        ))
    return items
