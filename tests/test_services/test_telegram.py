"""Unit tests for Telegram alert formatting and dispatch."""
from datetime import datetime

import pytest

from src.models.schemas import FinalRecommendation, Signal
from src.services.telegram import format_recommendation, send_telegram_alert


@pytest.fixture
def buy_recommendation():
    return FinalRecommendation(
        ticker="AAPL",
        action=Signal.BUY,
        confidence=0.82,
        weighted_score=0.45,
        technical_weight=0.30,
        fundamental_weight=0.40,
        sentiment_weight=0.30,
        risk_adjusted=True,
        position_size_suggestion_pct=0.05,
        entry_price=185.50,
        stop_loss_price=178.20,
        take_profit_price=198.00,
        time_horizon="medium_term",
        reasoning="Strong fundamentals with positive momentum",
        dissenting_opinions=["Sentiment analyst suggests caution"],
        generated_at=datetime(2024, 6, 15, 14, 30),
    )


@pytest.fixture
def hold_recommendation():
    return FinalRecommendation(
        ticker="TSLA",
        action=Signal.HOLD,
        confidence=0.45,
        weighted_score=0.05,
        technical_weight=0.30,
        fundamental_weight=0.40,
        sentiment_weight=0.30,
        risk_adjusted=True,
        position_size_suggestion_pct=0.0,
        time_horizon="short_term",
        reasoning="Mixed signals across all analysts",
        dissenting_opinions=[],
        generated_at=datetime(2024, 6, 15, 14, 30),
    )


class TestFormatRecommendation:
    def test_contains_ticker(self, buy_recommendation):
        msg = format_recommendation(buy_recommendation)
        assert "AAPL" in msg

    def test_contains_signal_hebrew(self, buy_recommendation):
        msg = format_recommendation(buy_recommendation)
        assert "קניה" in msg

    def test_contains_prices(self, buy_recommendation):
        msg = format_recommendation(buy_recommendation)
        assert "$185.50" in msg
        assert "$178.20" in msg
        assert "$198.00" in msg

    def test_contains_confidence(self, buy_recommendation):
        msg = format_recommendation(buy_recommendation)
        assert "82%" in msg

    def test_contains_dissenting(self, buy_recommendation):
        msg = format_recommendation(buy_recommendation)
        assert "caution" in msg

    def test_hold_no_prices(self, hold_recommendation):
        msg = format_recommendation(hold_recommendation)
        assert "המתנה" in msg
        assert "$" not in msg.split("גודל פוזיציה")[0].split("ציון משוקלל")[1]

    def test_strong_sell_emoji(self):
        rec = FinalRecommendation(
            ticker="GME",
            action=Signal.STRONG_SELL,
            confidence=0.91,
            weighted_score=-0.78,
            technical_weight=0.30,
            fundamental_weight=0.40,
            sentiment_weight=0.30,
            risk_adjusted=True,
            position_size_suggestion_pct=0.0,
            time_horizon="short_term",
            reasoning="All analysts bearish",
            dissenting_opinions=[],
            generated_at=datetime(2024, 6, 15, 14, 30),
        )
        msg = format_recommendation(rec)
        assert "מכירה חזקה" in msg


class TestSendTelegramAlert:
    async def test_disabled_returns_false(self, buy_recommendation):
        from src.config.settings import Settings

        settings = Settings(telegram_enabled=False)
        result = await send_telegram_alert(buy_recommendation, settings)
        assert result is False

    async def test_no_token_returns_false(self, buy_recommendation):
        from src.config.settings import Settings

        settings = Settings(telegram_enabled=True, telegram_bot_token="", telegram_chat_id="123")
        result = await send_telegram_alert(buy_recommendation, settings)
        assert result is False

    async def test_low_confidence_skipped(self, hold_recommendation):
        from src.config.settings import Settings

        settings = Settings(
            telegram_enabled=True,
            telegram_bot_token="fake",
            telegram_chat_id="123",
            alert_min_confidence=0.8,
        )
        result = await send_telegram_alert(hold_recommendation, settings)
        assert result is False
