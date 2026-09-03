"""Sentiment Analyst agent.

LLM-heavy agent — uses Claude for news sentiment classification.
"""
from src.agents.state import AnalysisState
from src.models.schemas import SentimentReport, SentimentScore, Signal
from src.tools.sentiment_scorer import score_news_sentiment


async def analyze_sentiment(state: AnalysisState) -> dict:
    md = state["market_data"]

    if not md.news:
        return {
            "sentiment_report": SentimentReport(
                ticker=md.ticker,
                overall_sentiment=0.0,
                sentiment_by_source=[],
                key_themes=[],
                notable_events=[],
                signal=Signal.HOLD,
                confidence=0.2,
                reasoning="No recent news articles found for analysis.",
            ),
            "audit_log": ["[SentimentAnalyst] No news data available"],
        }

    news_text = "\n\n".join(
        f"[{item.source} | {item.published_at.isoformat()}] {item.title}"
        + (f"\n{item.summary}" if item.summary else "")
        for item in md.news[:50]
    )

    result = await score_news_sentiment(md.ticker, news_text)

    report = SentimentReport(
        ticker=md.ticker,
        overall_sentiment=result["overall_sentiment"],
        sentiment_by_source=[
            SentimentScore(**s) for s in result.get("sentiment_by_source", [])
        ],
        key_themes=result.get("key_themes", []),
        notable_events=result.get("notable_events", []),
        signal=Signal(result["signal"]),
        confidence=result["confidence"],
        reasoning=result["reasoning"],
    )

    return {
        "sentiment_report": report,
        "audit_log": [
            f"[SentimentAnalyst] Signal={report.signal.value} "
            f"Confidence={report.confidence:.2f} "
            f"Overall={report.overall_sentiment:.2f}"
        ],
    }
