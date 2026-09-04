"""Telegram bot integration for trading signal alerts."""
import httpx

from src.config.settings import Settings
from src.models.schemas import FinalRecommendation, Signal

SIGNAL_EMOJI = {
    Signal.STRONG_BUY: "\U0001f7e2\U0001f7e2",
    Signal.BUY: "\U0001f7e2",
    Signal.HOLD: "\U0001f7e1",
    Signal.SELL: "\U0001f534",
    Signal.STRONG_SELL: "\U0001f534\U0001f534",
}

SIGNAL_HEBREW = {
    Signal.STRONG_BUY: "קניה חזקה",
    Signal.BUY: "קניה",
    Signal.HOLD: "המתנה",
    Signal.SELL: "מכירה",
    Signal.STRONG_SELL: "מכירה חזקה",
}


def format_recommendation(rec: FinalRecommendation) -> str:
    emoji = SIGNAL_EMOJI.get(rec.action, "")
    signal_name = SIGNAL_HEBREW.get(rec.action, rec.action.value)

    lines = [
        f"{emoji} *{rec.ticker}* — {signal_name}",
        "",
        f"\U0001f3af ביטחון: {rec.confidence:.0%}",
        f"\U0001f4ca ציון משוקלל: {rec.weighted_score:+.3f}",
    ]

    if rec.entry_price is not None:
        lines.append(f"\U0001f4b5 מחיר כניסה: ${rec.entry_price:.2f}")
    if rec.stop_loss_price is not None:
        lines.append(f"\U0001f6d1 סטופ לוס: ${rec.stop_loss_price:.2f}")
    if rec.take_profit_price is not None:
        lines.append(f"✅ טייק פרופיט: ${rec.take_profit_price:.2f}")

    lines.extend([
        "",
        f"\U0001f4e6 גודל פוזיציה: {rec.position_size_suggestion_pct:.1%}",
        f"⏰ אופק השקעה: {rec.time_horizon}",
        f"⚖️ ריסון סיכון: {'כן' if rec.risk_adjusted else 'לא'}",
    ])

    lines.extend([
        "",
        "\U0001f50d *משקלות סוכנים:*",
        f"  טכני ({rec.technical_weight:.0%}) | "
        f"פונדמנטלי ({rec.fundamental_weight:.0%}) | "
        f"סנטימנט ({rec.sentiment_weight:.0%})",
    ])

    if rec.dissenting_opinions:
        lines.append("")
        lines.append("⚠️ *דעות חולקות:*")
        for opinion in rec.dissenting_opinions:
            lines.append(f"  • {opinion}")

    lines.extend([
        "",
        f"\U0001f4ac _{rec.reasoning}_",
        "",
        f"\U0001f552 {rec.generated_at.strftime('%Y-%m-%d %H:%M UTC')}",
    ])

    return "\n".join(lines)


async def send_telegram_alert(rec: FinalRecommendation, settings: Settings | None = None) -> bool:
    if settings is None:
        settings = Settings()

    if not settings.telegram_enabled:
        return False
    if not settings.telegram_bot_token or not settings.telegram_chat_id:
        return False
    if rec.confidence < settings.alert_min_confidence:
        return False

    message = format_recommendation(rec)
    url = f"https://api.telegram.org/bot{settings.telegram_bot_token}/sendMessage"

    async with httpx.AsyncClient() as client:
        response = await client.post(url, json={
            "chat_id": settings.telegram_chat_id,
            "text": message,
            "parse_mode": "Markdown",
            "disable_web_page_preview": True,
        })
        return response.status_code == 200
