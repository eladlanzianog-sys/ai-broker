"""Autonomous Trading Automation Engine.

Runs independently 24/7:
  - Pre-market scan & analysis (configurable schedule)
  - Intraday monitoring of open positions
  - Auto-execution via IBKR with full risk management
  - End-of-day portfolio summary
  - Health monitoring & self-recovery
  - Telegram reporting at every stage
"""
from __future__ import annotations

import asyncio
import logging
import signal
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path

import httpx

from src.agents.graph import compile_graph
from src.config.settings import Settings
from src.execution import ExecutionEngine, ExecutionResult
from src.models.schemas import AnalysisRequest, FinalRecommendation, Signal
from src.scanner import run_scan, scan_single
from src.services.telegram import format_recommendation, _esc

logger = logging.getLogger(__name__)


class AutomationPhase(str, Enum):
    IDLE = "idle"
    PRE_MARKET_SCAN = "pre_market_scan"
    MARKET_OPEN_EXECUTION = "market_open_execution"
    INTRADAY_MONITOR = "intraday_monitor"
    POSITION_CHECK = "position_check"
    END_OF_DAY_REPORT = "end_of_day_report"
    WEEKEND_SLEEP = "weekend_sleep"


@dataclass
class AutomationState:
    phase: AutomationPhase = AutomationPhase.IDLE
    started_at: datetime = field(default_factory=datetime.utcnow)
    last_scan_at: datetime | None = None
    last_execution_at: datetime | None = None
    last_monitor_at: datetime | None = None
    last_health_check_at: datetime | None = None
    scans_today: int = 0
    executions_today: int = 0
    errors_today: int = 0
    total_scans: int = 0
    total_executions: int = 0
    is_running: bool = False
    pending_signals: list[FinalRecommendation] = field(default_factory=list)
    execution_results: list[ExecutionResult] = field(default_factory=list)
    daily_pnl: float = 0.0


class AutonomousEngine:
    """Self-running trading automation that manages the full trading cycle."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self.state = AutomationState()
        self.graph = None
        self.execution_engine: ExecutionEngine | None = None
        self._shutdown = asyncio.Event()
        self._tasks: list[asyncio.Task] = []

    async def start(self) -> None:
        self.state.is_running = True
        self.state.started_at = datetime.utcnow()
        self.graph = compile_graph()

        await self._send_telegram(
            "\U0001F680 <b>AI Broker — אוטומציה הופעלה</b>\n\n"
            f"\U0001F4CB מניות: {', '.join(self.settings.get_watchlist())}\n"
            f"\U0001F551 סריקה: {self.settings.scan_hour:02d}:{self.settings.scan_minute:02d} UTC\n"
            f"\U0001F4B0 IBKR: {'מופעל' if self.settings.ibkr_trading_enabled else 'DRY RUN'}\n"
            f"\U0001F4E2 טלגרם: {'מופעל' if self.settings.telegram_enabled else 'כבוי'}\n"
            f"\n⏰ {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}"
        )

        self._print_banner()

        loop = asyncio.get_event_loop()
        for sig_name in ("SIGINT", "SIGTERM"):
            try:
                loop.add_signal_handler(
                    getattr(signal, sig_name),
                    lambda: asyncio.create_task(self._graceful_shutdown()),
                )
            except (NotImplementedError, AttributeError):
                pass

        self._tasks = [
            asyncio.create_task(self._scheduler_loop(), name="scheduler"),
            asyncio.create_task(self._monitor_loop(), name="monitor"),
            asyncio.create_task(self._health_loop(), name="health"),
        ]

        try:
            await self._shutdown.wait()
        finally:
            for t in self._tasks:
                t.cancel()
            await asyncio.gather(*self._tasks, return_exceptions=True)

        await self._send_telegram(
            "\U0001F6D1 <b>AI Broker — אוטומציה נעצרה</b>\n\n"
            f"סריקות היום: {self.state.scans_today}\n"
            f"ביצועים היום: {self.state.executions_today}\n"
            f"שגיאות: {self.state.errors_today}\n"
            f"\n\U0001F552 {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}"
        )

        self.state.is_running = False
        logger.info("Automation engine stopped")

    async def _graceful_shutdown(self) -> None:
        logger.info("Shutdown signal received")
        print("\n\U0001F6D1 מכבה את המערכת...")
        self._shutdown.set()

    def _print_banner(self) -> None:
        tickers = self.settings.get_watchlist()
        mode = "LIVE" if self.settings.ibkr_trading_enabled else "DRY RUN"
        print(f"""
{'='*60}
   AI Broker — Autonomous Trading Engine
{'='*60}
   Mode:       {mode}
   Watchlist:  {', '.join(tickers[:5])}{'...' if len(tickers) > 5 else ''}
   Scan Time:  {self.settings.scan_hour:02d}:{self.settings.scan_minute:02d} UTC
   IBKR:       {'Connected' if self.settings.ibkr_trading_enabled else 'Disabled'}
   Telegram:   {'ON' if self.settings.telegram_enabled else 'OFF'}
   Started:    {self.state.started_at.strftime('%Y-%m-%d %H:%M UTC')}
{'='*60}
""")

    # ------------------------------------------------------------------ #
    #                        SCHEDULER LOOP                                #
    # ------------------------------------------------------------------ #

    async def _scheduler_loop(self) -> None:
        while not self._shutdown.is_set():
            try:
                now = datetime.utcnow()

                if self._is_weekend(now):
                    self.state.phase = AutomationPhase.WEEKEND_SLEEP
                    self._log_phase("שבת — המערכת במצב שינה")
                    await self._sleep_until_next_weekday()
                    continue

                if self._should_reset_daily(now):
                    await self._daily_reset()

                if self._is_scan_time(now):
                    await self._run_pre_market_scan()

                    if self.state.pending_signals:
                        await self._run_execution()

                if self._is_end_of_day(now):
                    await self._send_daily_report()

                await self._interruptible_sleep(60)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.state.errors_today += 1
                logger.error(f"Scheduler error: {e}", exc_info=True)
                await self._send_telegram(
                    f"\U0001F6A8 <b>שגיאת מתזמן</b>\n\n{_esc(str(e)[:200])}"
                )
                await self._interruptible_sleep(300)

    # ------------------------------------------------------------------ #
    #                     PRE-MARKET SCAN                                   #
    # ------------------------------------------------------------------ #

    async def _run_pre_market_scan(self) -> None:
        self.state.phase = AutomationPhase.PRE_MARKET_SCAN
        self._log_phase("סריקת טרום-שוק")

        tickers = self.settings.get_watchlist()
        results = []

        await self._send_telegram(
            f"\U0001F50D <b>סריקת טרום-שוק — {len(tickers)} מניות</b>\n"
            f"\U0001F552 {datetime.utcnow().strftime('%H:%M UTC')}"
        )

        for i, ticker in enumerate(tickers, 1):
            print(f"  [{i}/{len(tickers)}] סורק {ticker}...", end=" ", flush=True)
            try:
                result = await scan_single(ticker, self.graph)
                results.append(result)
                rec = result.get("recommendation")
                if rec:
                    print(f"→ {rec.action.value} ({rec.confidence:.0%})")
                else:
                    print("→ אין המלצה")
            except Exception as e:
                print(f"→ שגיאה: {e}")
                self.state.errors_today += 1

            if i < len(tickers):
                await asyncio.sleep(2)

        actionable_signals = {Signal.STRONG_BUY, Signal.BUY, Signal.SELL, Signal.STRONG_SELL}
        self.state.pending_signals = [
            r["recommendation"] for r in results
            if r.get("recommendation") and r["recommendation"].action in actionable_signals
        ]

        self.state.last_scan_at = datetime.utcnow()
        self.state.scans_today += 1
        self.state.total_scans += 1

        await self._send_scan_report(results)

    async def _send_scan_report(self, results: list[dict]) -> None:
        total = len(results)
        ok = sum(1 for r in results if r.get("recommendation"))
        failed = total - ok
        actionable = len(self.state.pending_signals)

        lines = [
            "\U0001F4CA <b>תוצאות סריקה</b>",
            "",
            f"נסרקו: {total} | הצליחו: {ok} | נכשלו: {failed}",
            f"אותות פעולה: {actionable}",
        ]

        if self.state.pending_signals:
            lines.append("")
            lines.append("<b>אותות:</b>")
            for rec in self.state.pending_signals:
                emoji = "\U0001f7e2" if rec.action in (Signal.STRONG_BUY, Signal.BUY) else "\U0001f534"
                hebrew = {
                    Signal.STRONG_BUY: "קניה חזקה",
                    Signal.BUY: "קניה",
                    Signal.SELL: "מכירה",
                    Signal.STRONG_SELL: "מכירה חזקה",
                }
                sig_name = hebrew.get(rec.action, rec.action.value)
                lines.append(
                    f"  {emoji} <b>{_esc(rec.ticker)}</b> — {sig_name} "
                    f"({rec.confidence:.0%})"
                )

                if rec.entry_price:
                    lines.append(
                        f"     \U0001f4b5 כניסה: ${rec.entry_price:.2f} | "
                        f"SL: ${rec.stop_loss_price or 0:.2f} | "
                        f"TP: ${rec.take_profit_price or 0:.2f}"
                    )
        else:
            lines.append("")
            lines.append("אין אותות פעולה — הכל במצב המתנה \U0001f7e1")

        lines.append(f"\n\U0001F552 {datetime.utcnow().strftime('%H:%M UTC')}")
        await self._send_telegram("\n".join(lines))

    # ------------------------------------------------------------------ #
    #                     EXECUTION                                         #
    # ------------------------------------------------------------------ #

    async def _run_execution(self) -> None:
        self.state.phase = AutomationPhase.MARKET_OPEN_EXECUTION
        self._log_phase(f"ביצוע — {len(self.state.pending_signals)} אותות")

        if not self.settings.ibkr_trading_enabled:
            await self._send_telegram(
                "\U0001F4E1 <b>DRY RUN — ביצוע מדומה</b>\n\n"
                "IBKR לא מופעל. האותות נשמרים ללא ביצוע.\n"
                f"אותות: {len(self.state.pending_signals)}"
            )

            for rec in self.state.pending_signals:
                self.state.execution_results.append(
                    ExecutionResult(
                        ticker=rec.ticker,
                        action=rec.action.value,
                        success=True,
                        reason="DRY RUN",
                        recommendation=rec,
                    )
                )
            self.state.executions_today += len(self.state.pending_signals)
            self.state.pending_signals.clear()
            return

        engine = ExecutionEngine(self.settings)
        if not await engine.start():
            await self._send_telegram(
                "\U0001F6A8 <b>שגיאת IBKR</b>\n\n"
                "לא ניתן להתחבר ל-TWS/Gateway.\n"
                "הסריקה בוצעה אבל לא יבוצעו עסקאות."
            )
            self.state.errors_today += 1
            return

        try:
            for rec in self.state.pending_signals:
                try:
                    result = await engine.analyze_and_execute(rec.ticker)
                    self.state.execution_results.append(result)
                    self.state.executions_today += 1
                    self.state.total_executions += 1

                    status = "OK" if result.success else "FAIL"
                    print(f"  → {rec.ticker}: {status} — {result.reason or 'executed'}")

                except Exception as e:
                    logger.error(f"Execution failed for {rec.ticker}: {e}")
                    self.state.errors_today += 1

                await asyncio.sleep(3)

        finally:
            engine.stop()

        self.state.last_execution_at = datetime.utcnow()
        self.state.pending_signals.clear()
        await self._send_execution_report()

    async def _send_execution_report(self) -> None:
        results = self.state.execution_results
        if not results:
            return

        executed = [r for r in results if r.bracket]
        skipped = [r for r in results if r.success and not r.bracket]
        failed = [r for r in results if not r.success]

        lines = [
            "\U0001F4E1 <b>דוח ביצוע</b>",
            "",
            f"בוצעו: {len(executed)} | דולגו: {len(skipped)} | נכשלו: {len(failed)}",
        ]

        if executed:
            lines.append("")
            lines.append("<b>עסקאות שבוצעו:</b>")
            for r in executed:
                b = r.bracket
                side = "קניה" if b.side == "BUY" else "מכירה"
                lines.append(
                    f"  \U0001f4cc {_esc(b.ticker)} — {side} {b.quantity} מניות\n"
                    f"     SL: ${b.stop_loss_price:.2f} | TP: ${b.take_profit_price:.2f}"
                )

        if failed:
            lines.append("")
            lines.append("<b>נכשלו:</b>")
            for r in failed:
                lines.append(f"  \U0001f534 {_esc(r.ticker)} — {_esc(r.reason[:100])}")

        lines.append(f"\n\U0001F552 {datetime.utcnow().strftime('%H:%M UTC')}")
        await self._send_telegram("\n".join(lines))

    # ------------------------------------------------------------------ #
    #                     POSITION MONITOR                                  #
    # ------------------------------------------------------------------ #

    async def _monitor_loop(self) -> None:
        while not self._shutdown.is_set():
            try:
                now = datetime.utcnow()

                if self._is_market_hours(now) and self.settings.ibkr_trading_enabled:
                    await self._check_positions()

                await self._interruptible_sleep(900)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.state.errors_today += 1
                logger.error(f"Monitor error: {e}", exc_info=True)
                await self._interruptible_sleep(300)

    async def _check_positions(self) -> None:
        self.state.phase = AutomationPhase.POSITION_CHECK

        from src.services.ibkr import IBKRService

        ibkr = IBKRService(
            host=self.settings.ibkr_host,
            port=self.settings.ibkr_port,
            client_id=self.settings.ibkr_client_id + 20,
            readonly=True,
        )

        if not await ibkr.connect():
            return

        try:
            positions = await ibkr.get_positions()
            account = await ibkr.get_account_summary()

            if not positions:
                self.state.last_monitor_at = datetime.utcnow()
                return

            total_pnl = sum(p.unrealized_pnl or 0 for p in positions)
            self.state.daily_pnl = total_pnl

            critical_positions = []
            for pos in positions:
                if pos.unrealized_pnl is not None:
                    pnl_pct = (pos.unrealized_pnl / (pos.avg_cost * pos.quantity)) * 100 if pos.avg_cost > 0 and pos.quantity > 0 else 0
                    if pnl_pct <= -5:
                        critical_positions.append((pos, pnl_pct))

            if critical_positions:
                lines = ["\U0001F6A8 <b>התראת פוזיציות</b>", ""]
                for pos, pnl_pct in critical_positions:
                    lines.append(
                        f"\U0001f534 <b>{_esc(pos.ticker)}</b> — "
                        f"P&L: {pnl_pct:+.1f}% "
                        f"(${pos.unrealized_pnl:+,.2f})"
                    )
                lines.append(f"\n\U0001F4B0 סה\"כ P&L: ${total_pnl:+,.2f}")
                await self._send_telegram("\n".join(lines))

            self.state.last_monitor_at = datetime.utcnow()

        finally:
            ibkr.disconnect()

    # ------------------------------------------------------------------ #
    #                     HEALTH MONITOR                                    #
    # ------------------------------------------------------------------ #

    async def _health_loop(self) -> None:
        while not self._shutdown.is_set():
            try:
                await self._health_check()
                await self._interruptible_sleep(3600)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")
                await self._interruptible_sleep(600)

    async def _health_check(self) -> None:
        self.state.last_health_check_at = datetime.utcnow()
        uptime = datetime.utcnow() - self.state.started_at
        hours = uptime.total_seconds() / 3600

        print(
            f"  [Health] Uptime: {hours:.1f}h | "
            f"Phase: {self.state.phase.value} | "
            f"Scans: {self.state.total_scans} | "
            f"Executions: {self.state.total_executions} | "
            f"Errors today: {self.state.errors_today}"
        )

        if self.state.errors_today > 10:
            await self._send_telegram(
                f"\U0001F6A8 <b>Health Warning</b>\n\n"
                f"שגיאות היום: {self.state.errors_today}\n"
                f"המערכת עדיין פעילה אבל ייתכן שיש בעיה."
            )

    # ------------------------------------------------------------------ #
    #                     DAILY REPORT                                      #
    # ------------------------------------------------------------------ #

    async def _send_daily_report(self) -> None:
        self.state.phase = AutomationPhase.END_OF_DAY_REPORT

        uptime = datetime.utcnow() - self.state.started_at
        hours = uptime.total_seconds() / 3600

        lines = [
            "\U0001F4CB <b>דוח יומי — AI Broker</b>",
            f"{datetime.utcnow().strftime('%Y-%m-%d')}",
            "",
            f"\U0001F50D סריקות: {self.state.scans_today}",
            f"\U0001F4E1 ביצועים: {self.state.executions_today}",
            f"\U0001F6A8 שגיאות: {self.state.errors_today}",
            f"\U0001F4B0 P&L יומי: ${self.state.daily_pnl:+,.2f}",
            "",
            f"⏱ Uptime: {hours:.1f} שעות",
            f"\U0001F4CA סריקות מצטבר: {self.state.total_scans}",
            f"\U0001F4E1 ביצועים מצטבר: {self.state.total_executions}",
        ]

        executed_today = [r for r in self.state.execution_results if r.bracket]
        if executed_today:
            lines.append("")
            lines.append("<b>עסקאות היום:</b>")
            for r in executed_today:
                b = r.bracket
                lines.append(
                    f"  \U0001f4cc {_esc(b.ticker)} — "
                    f"{b.side} {b.quantity} מניות"
                )

        lines.append(f"\n\U0001F552 {datetime.utcnow().strftime('%H:%M UTC')}")
        await self._send_telegram("\n".join(lines))

    async def _daily_reset(self) -> None:
        self.state.scans_today = 0
        self.state.executions_today = 0
        self.state.errors_today = 0
        self.state.execution_results.clear()
        self.state.pending_signals.clear()
        self.state.daily_pnl = 0.0
        logger.info("Daily counters reset")

    # ------------------------------------------------------------------ #
    #                     TIME HELPERS                                      #
    # ------------------------------------------------------------------ #

    def _is_weekend(self, now: datetime) -> bool:
        return now.weekday() >= 5

    def _is_market_hours(self, now: datetime) -> bool:
        return 13 <= now.hour <= 21 and now.weekday() < 5

    def _is_scan_time(self, now: datetime) -> bool:
        if self.state.last_scan_at:
            diff = (now - self.state.last_scan_at).total_seconds()
            if diff < 3600:
                return False

        return (
            now.hour == self.settings.scan_hour
            and now.minute <= 5
        )

    def _is_end_of_day(self, now: datetime) -> bool:
        return now.hour == 21 and now.minute <= 5 and now.weekday() < 5

    def _should_reset_daily(self, now: datetime) -> bool:
        return now.hour == 0 and now.minute <= 1

    async def _sleep_until_next_weekday(self) -> None:
        now = datetime.utcnow()
        days_until_monday = (7 - now.weekday()) % 7
        if days_until_monday == 0:
            days_until_monday = 1
        next_monday = now.replace(hour=6, minute=0, second=0, microsecond=0) + timedelta(days=days_until_monday)
        wait = (next_monday - now).total_seconds()
        print(f"  [Weekend] שינה עד יום שני ({wait/3600:.1f} שעות)")
        await self._interruptible_sleep(wait)

    async def _interruptible_sleep(self, seconds: float) -> None:
        try:
            await asyncio.wait_for(self._shutdown.wait(), timeout=seconds)
        except asyncio.TimeoutError:
            pass

    # ------------------------------------------------------------------ #
    #                     TELEGRAM                                          #
    # ------------------------------------------------------------------ #

    async def _send_telegram(self, text: str) -> None:
        if not self.settings.telegram_enabled:
            return
        if not self.settings.telegram_bot_token or not self.settings.telegram_chat_id:
            return

        url = f"https://api.telegram.org/bot{self.settings.telegram_bot_token}/sendMessage"
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
                await client.post(url, json={
                    "chat_id": self.settings.telegram_chat_id,
                    "text": text,
                    "parse_mode": "HTML",
                    "disable_web_page_preview": True,
                })
        except Exception as e:
            logger.error(f"Telegram send failed: {e}")

    # ------------------------------------------------------------------ #
    #                     LOGGING                                           #
    # ------------------------------------------------------------------ #

    def _log_phase(self, msg: str) -> None:
        now = datetime.utcnow().strftime("%H:%M:%S")
        phase = self.state.phase.value
        print(f"  [{now}] [{phase}] {msg}")


# ===================================================================== #
#                   MULTI-SCHEDULE AUTOMATION                              #
# ===================================================================== #

class ScheduleConfig:
    """Multiple scan schedules throughout the trading day."""

    SCHEDULES = [
        {"name": "pre_market", "hour": 8, "minute": 0, "description": "סריקת טרום-שוק"},
        {"name": "market_open", "hour": 14, "minute": 35, "description": "סריקת פתיחה (30 דק אחרי)"},
        {"name": "midday", "hour": 17, "minute": 0, "description": "סריקת אמצע יום"},
        {"name": "pre_close", "hour": 20, "minute": 30, "description": "סריקת טרום-סגירה"},
    ]

    @classmethod
    def get_next_schedule(cls, now: datetime) -> dict | None:
        for sched in cls.SCHEDULES:
            if now.hour == sched["hour"] and now.minute <= 5:
                return sched
        return None


class AdvancedAutomation(AutonomousEngine):
    """Extended automation with multiple daily scan schedules."""

    async def _scheduler_loop(self) -> None:
        while not self._shutdown.is_set():
            try:
                now = datetime.utcnow()

                if self._is_weekend(now):
                    self.state.phase = AutomationPhase.WEEKEND_SLEEP
                    self._log_phase("סוף שבוע — שינה")
                    await self._sleep_until_next_weekday()
                    continue

                if self._should_reset_daily(now):
                    await self._daily_reset()

                schedule = ScheduleConfig.get_next_schedule(now)
                if schedule and self._can_run_schedule(now):
                    self._log_phase(f"{schedule['description']}")
                    await self._run_pre_market_scan()

                    if self.state.pending_signals and self._is_market_hours(now):
                        await self._run_execution()

                if self._is_end_of_day(now):
                    await self._send_daily_report()

                await self._interruptible_sleep(60)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.state.errors_today += 1
                logger.error(f"Scheduler error: {e}", exc_info=True)
                await self._send_telegram(
                    f"\U0001F6A8 <b>שגיאת מתזמן</b>\n\n{_esc(str(e)[:200])}"
                )
                await self._interruptible_sleep(300)

    def _can_run_schedule(self, now: datetime) -> bool:
        if not self.state.last_scan_at:
            return True
        diff = (now - self.state.last_scan_at).total_seconds()
        return diff > 1800


# ===================================================================== #
#                         CLI ENTRY POINT                                  #
# ===================================================================== #

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="AI Broker — Autonomous Trading Automation"
    )
    parser.add_argument(
        "--mode",
        choices=["basic", "advanced", "once"],
        default="advanced",
        help="basic: single daily scan | advanced: multiple scans | once: scan now and exit",
    )
    parser.add_argument(
        "--tickers",
        type=str,
        help="Override watchlist (comma-separated)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Enable IBKR live trading (port 7496)",
    )
    parser.add_argument(
        "--scan-now",
        action="store_true",
        help="Run an immediate scan before entering the schedule loop",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("automation.log", encoding="utf-8"),
        ],
    )

    settings = Settings()

    if args.tickers:
        settings.watchlist = args.tickers

    if args.live:
        settings.ibkr_port = 7496
        settings.ibkr_trading_enabled = True
        print("\U0001F534 LIVE TRADING MODE — real money!")

    if args.mode == "once":
        print("\nRunning single scan...")
        asyncio.run(run_scan(settings))
        return

    if args.mode == "advanced":
        engine = AdvancedAutomation(settings)
    else:
        engine = AutonomousEngine(settings)

    if args.scan_now:
        async def _run_with_immediate_scan():
            engine.graph = compile_graph()
            await engine._run_pre_market_scan()
            if engine.state.pending_signals:
                await engine._run_execution()
            await engine.start()

        asyncio.run(_run_with_immediate_scan())
    else:
        asyncio.run(engine.start())


if __name__ == "__main__":
    main()
