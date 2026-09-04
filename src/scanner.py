"""Automated stock scanner — scans a watchlist and sends Telegram alerts."""
import asyncio
import logging
from datetime import datetime

from src.agents.graph import compile_graph
from src.config.settings import Settings
from src.models.schemas import AnalysisRequest, FinalRecommendation, Signal
from src.services.telegram import format_recommendation, send_telegram_alert, _esc

logger = logging.getLogger(__name__)

ACTIONABLE_SIGNALS = {Signal.STRONG_BUY, Signal.BUY, Signal.SELL, Signal.STRONG_SELL}


async def scan_single(ticker: str, graph) -> dict:
    try:
        request = AnalysisRequest(ticker=ticker)
        result = await graph.ainvoke({"request": request})
        rec = result.get("recommendation")
        errors = result.get("errors", [])
        return {"ticker": ticker, "recommendation": rec, "errors": errors}
    except Exception as e:
        logger.error(f"Failed to scan {ticker}: {e}")
        return {"ticker": ticker, "recommendation": None, "errors": [str(e)]}


async def run_scan(settings: Settings | None = None) -> list[dict]:
    if settings is None:
        settings = Settings()

    tickers = settings.get_watchlist()
    if not tickers:
        logger.warning("Watchlist is empty")
        return []

    graph = compile_graph()
    results = []

    print(f"\n{'='*50}")
    print(f"  סריקה אוטומטית — {len(tickers)} מניות")
    print(f"  {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{'='*50}\n")

    for i, ticker in enumerate(tickers, 1):
        print(f"[{i}/{len(tickers)}] סורק {ticker}...")
        result = await scan_single(ticker, graph)
        results.append(result)

        rec = result["recommendation"]
        if rec:
            print(f"  → {rec.action.value} (ביטחון: {rec.confidence:.0%})")
        elif result["errors"]:
            print(f"  → שגיאה: {result['errors'][0][:80]}")
        else:
            print(f"  → אין המלצה")

        if i < len(tickers):
            await asyncio.sleep(2)

    actionable = [
        r for r in results
        if r["recommendation"] and r["recommendation"].action in ACTIONABLE_SIGNALS
    ]

    if settings.scan_only_actionable:
        alerts = actionable
    else:
        alerts = [r for r in results if r["recommendation"]]

    if settings.telegram_enabled and alerts:
        await _send_scan_summary(results, actionable, settings)
        for r in alerts:
            try:
                await send_telegram_alert(r["recommendation"], settings)
                await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Telegram alert failed for {r['ticker']}: {e}")

    _print_summary(results, actionable)
    return results


async def _send_scan_summary(
    all_results: list[dict],
    actionable: list[dict],
    settings: Settings,
) -> None:
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    total = len(all_results)
    ok = sum(1 for r in all_results if r["recommendation"])
    failed = total - ok

    lines = [
        f"<b>סיכום סריקה יומית</b>",
        f"{now}",
        "",
        f"נסרקו: {total} מניות",
        f"הצליחו: {ok} | נכשלו: {failed}",
        f"אותות פעולה: {len(actionable)}",
    ]

    if actionable:
        lines.append("")
        lines.append("<b>מניות עם אותות:</b>")
        for r in actionable:
            rec: FinalRecommendation = r["recommendation"]
            emoji = "\U0001f7e2" if rec.action in (Signal.STRONG_BUY, Signal.BUY) else "\U0001f534"
            hebrew = {
                Signal.STRONG_BUY: "קניה חזקה",
                Signal.BUY: "קניה",
                Signal.SELL: "מכירה",
                Signal.STRONG_SELL: "מכירה חזקה",
            }
            sig_name = hebrew.get(rec.action, rec.action.value)
            lines.append(
                f"{emoji} <b>{_esc(rec.ticker)}</b> — {sig_name} "
                f"({rec.confidence:.0%})"
            )
    else:
        lines.append("")
        lines.append("אין אותות פעולה כרגע — הכל במצב המתנה.")

    lines.append("")
    lines.append("הפירוט המלא נשלח בהודעות נפרדות.")

    import httpx
    url = f"https://api.telegram.org/bot{settings.telegram_bot_token}/sendMessage"
    async with httpx.AsyncClient() as client:
        await client.post(url, json={
            "chat_id": settings.telegram_chat_id,
            "text": "\n".join(lines),
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        })


def _print_summary(all_results: list[dict], actionable: list[dict]) -> None:
    total = len(all_results)
    ok = sum(1 for r in all_results if r["recommendation"])
    failed = total - ok

    print(f"\n{'='*50}")
    print(f"  סיכום סריקה")
    print(f"{'='*50}")
    print(f"  נסרקו: {total} מניות")
    print(f"  הצליחו: {ok} | נכשלו: {failed}")
    print(f"  אותות פעולה: {len(actionable)}")

    if actionable:
        print(f"\n  מניות עם אותות:")
        for r in actionable:
            rec = r["recommendation"]
            print(f"    {rec.ticker}: {rec.action.value} ({rec.confidence:.0%})")

    print(f"{'='*50}\n")


def main():
    import sys
    logging.basicConfig(level=logging.INFO)
    settings = Settings()

    if "--tickers" in sys.argv:
        idx = sys.argv.index("--tickers")
        if idx + 1 < len(sys.argv):
            settings.watchlist = sys.argv[idx + 1]

    asyncio.run(run_scan(settings))


if __name__ == "__main__":
    main()
