"""Claude-based sentiment classification helper."""
from src.services.llm import call_claude_structured


async def score_news_sentiment(ticker: str, news_text: str) -> dict:
    return await call_claude_structured(
        system=(
            "You are a financial sentiment analyst. Score each news source's "
            "sentiment from -1.0 (very bearish) to +1.0 (very bullish). "
            "Identify key themes and notable events. Produce an overall signal.\n\n"
            "Return JSON with keys: overall_sentiment (float), "
            "sentiment_by_source (list of {source, score, article_count}), "
            "key_themes (list of strings), notable_events (list of strings), "
            "signal (one of: strong_buy, buy, hold, sell, strong_sell), "
            "confidence (float 0-1), reasoning (string)."
        ),
        user_content=f"Ticker: {ticker}\n\nNEWS ARTICLES:\n{news_text}",
    )
