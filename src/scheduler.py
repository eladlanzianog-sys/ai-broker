"""Daily scan scheduler — runs the watchlist scan at a configured time."""
import asyncio
import logging
from datetime import datetime, timedelta

from src.config.settings import Settings
from src.scanner import run_scan

logger = logging.getLogger(__name__)


def _seconds_until(hour: int, minute: int) -> float:
    now = datetime.utcnow()
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return (target - now).total_seconds()


async def run_scheduler() -> None:
    settings = Settings()
    hour = settings.scan_hour
    minute = settings.scan_minute
    tickers = settings.get_watchlist()

    print(f"\n{'='*50}")
    print(f"  סורק אוטומטי — מצב תזמון")
    print(f"{'='*50}")
    print(f"  שעת סריקה: {hour:02d}:{minute:02d} UTC")
    print(f"  מניות: {', '.join(tickers)}")
    print(f"  טלגרם: {'מופעל' if settings.telegram_enabled else 'כבוי'}")
    print(f"{'='*50}\n")

    while True:
        wait = _seconds_until(hour, minute)
        next_run = datetime.utcnow() + timedelta(seconds=wait)
        print(f"סריקה הבאה: {next_run.strftime('%Y-%m-%d %H:%M UTC')} (בעוד {wait/3600:.1f} שעות)")

        await asyncio.sleep(wait)

        print(f"\nמתחיל סריקה יומית — {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
        try:
            await run_scan(settings)
        except Exception as e:
            logger.error(f"Scan failed: {e}")
            print(f"שגיאה בסריקה: {e}")

        await asyncio.sleep(60)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    asyncio.run(run_scheduler())


if __name__ == "__main__":
    main()
