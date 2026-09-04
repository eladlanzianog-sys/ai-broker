"""Notification dispatcher — routes alerts to configured channels."""
from src.config.settings import Settings
from src.models.schemas import FinalRecommendation
from src.services.telegram import send_telegram_alert


async def dispatch_alert(rec: FinalRecommendation, settings: Settings | None = None) -> dict[str, bool]:
    if settings is None:
        settings = Settings()

    results: dict[str, bool] = {}

    if settings.telegram_enabled:
        try:
            results["telegram"] = await send_telegram_alert(rec, settings)
        except Exception:
            results["telegram"] = False

    return results
