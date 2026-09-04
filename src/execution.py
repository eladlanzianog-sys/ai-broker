"""Execution engine — connects AI signals to IBKR order placement."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime

from src.agents.graph import compile_graph
from src.config.settings import Settings
from src.models.schemas import AnalysisRequest, FinalRecommendation, Signal
from src.services.ibkr import (
    AccountSummary,
    BracketResult,
    IBKRService,
    OrderSide,
)
from src.services.telegram import send_telegram_alert, _esc

logger = logging.getLogger(__name__)

ACTIONABLE_BUY = {Signal.STRONG_BUY, Signal.BUY}
ACTIONABLE_SELL = {Signal.SELL, Signal.STRONG_SELL}
ACTIONABLE = ACTIONABLE_BUY | ACTIONABLE_SELL


@dataclass
class ExecutionResult:
    ticker: str
    action: str
    success: bool
    bracket: BracketResult | None = None
    reason: str = ""
    recommendation: FinalRecommendation | None = None


def calculate_position_size(
    account_value: float,
    risk_pct: float,
    entry_price: float,
    stop_loss_price: float,
    max_position_pct: float = 0.20,
) -> int:
    if entry_price <= 0 or stop_loss_price <= 0:
        return 0
    r = abs(entry_price - stop_loss_price)
    if r < 0.01:
        return 0
    risk_dollars = account_value * (risk_pct / 100)
    shares = int(risk_dollars / r)
    max_shares = int((account_value * max_position_pct) / entry_price)
    return max(0, min(shares, max_shares))


def calculate_stop_loss(
    entry_price: float,
    atr: float | None,
    atr_multiplier: float = 1.5,
    fallback_pct: float = 0.02,
) -> float:
    if atr and atr > 0:
        return round(entry_price - atr * atr_multiplier, 2)
    return round(entry_price * (1 - fallback_pct), 2)


def calculate_take_profit(
    entry_price: float,
    stop_loss_price: float,
    reward_ratio: float = 2.0,
) -> float:
    r = entry_price - stop_loss_price
    return round(entry_price + r * reward_ratio, 2)


class ExecutionEngine:
    """Analyzes stocks and executes trades on IBKR with full risk management."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self.ibkr = IBKRService(
            host=self.settings.ibkr_host,
            port=self.settings.ibkr_port,
            client_id=self.settings.ibkr_client_id,
            readonly=not self.settings.ibkr_trading_enabled,
        )
        self.graph = None

    async def start(self) -> bool:
        connected = await self.ibkr.connect()
        if not connected:
            logger.error("Cannot start execution engine — IBKR connection failed")
            return False
        self.graph = compile_graph()
        logger.info("Execution engine started")
        return True

    def stop(self):
        self.ibkr.disconnect()
        logger.info("Execution engine stopped")

    async def analyze_and_execute(self, ticker: str) -> ExecutionResult:
        if not self.ibkr.is_connected:
            return ExecutionResult(ticker=ticker, action="none", success=False,
                                  reason="Not connected to IBKR")

        request = AnalysisRequest(ticker=ticker)
        try:
            result = await self.graph.ainvoke({"request": request})
        except Exception as e:
            return ExecutionResult(ticker=ticker, action="none", success=False,
                                  reason=f"Analysis failed: {e}")

        rec: FinalRecommendation | None = result.get("recommendation")
        if rec is None:
            return ExecutionResult(ticker=ticker, action="none", success=False,
                                  reason="No recommendation produced")

        if rec.action not in ACTIONABLE:
            return ExecutionResult(ticker=ticker, action=rec.action.value, success=True,
                                  reason="Signal is HOLD — no trade", recommendation=rec)

        if rec.confidence < self.settings.ibkr_min_confidence:
            return ExecutionResult(
                ticker=ticker, action=rec.action.value, success=True,
                reason=f"Confidence {rec.confidence:.0%} below threshold "
                       f"{self.settings.ibkr_min_confidence:.0%}",
                recommendation=rec,
            )

        return await self._execute_signal(rec)

    async def _execute_signal(self, rec: FinalRecommendation) -> ExecutionResult:
        if not self.settings.ibkr_trading_enabled:
            await self._notify(rec, None, "DRY RUN — trading disabled")
            return ExecutionResult(
                ticker=rec.ticker, action=rec.action.value, success=True,
                reason="Dry run — ibkr_trading_enabled=false", recommendation=rec,
            )

        account = await self.ibkr.get_account_summary()
        existing = await self.ibkr.get_position(rec.ticker)

        if rec.action in ACTIONABLE_BUY:
            return await self._execute_buy(rec, account, existing)
        else:
            return await self._execute_sell(rec, account, existing)

    async def _execute_buy(
        self, rec: FinalRecommendation, account: AccountSummary,
        existing,
    ) -> ExecutionResult:
        if existing and existing.quantity > 0:
            return ExecutionResult(
                ticker=rec.ticker, action=rec.action.value, success=True,
                reason=f"Already long {existing.quantity} shares — skipping",
                recommendation=rec,
            )

        if not self.ibkr.pdt.can_day_trade:
            logger.warning(
                f"PDT limit reached ({self.ibkr.pdt.trades_used}/3) — "
                f"trade will be swing (hold overnight)"
            )

        entry = rec.entry_price or 0
        if entry <= 0:
            price = await self.ibkr.get_market_price(rec.ticker)
            if price is None:
                return ExecutionResult(
                    ticker=rec.ticker, action=rec.action.value, success=False,
                    reason="Cannot get market price", recommendation=rec,
                )
            entry = price

        sl = rec.stop_loss_price or calculate_stop_loss(entry, None)
        tp = rec.take_profit_price or calculate_take_profit(
            entry, sl, self.settings.ibkr_reward_ratio,
        )

        qty = calculate_position_size(
            account_value=account.net_liquidation,
            risk_pct=self.settings.ibkr_risk_per_trade,
            entry_price=entry,
            stop_loss_price=sl,
            max_position_pct=self.settings.ibkr_max_position_pct,
        )

        if qty <= 0:
            return ExecutionResult(
                ticker=rec.ticker, action=rec.action.value, success=False,
                reason="Position size is 0 — risk too high or account too small",
                recommendation=rec,
            )

        cost = qty * entry
        if cost > account.available_funds:
            return ExecutionResult(
                ticker=rec.ticker, action=rec.action.value, success=False,
                reason=f"Insufficient funds: need ${cost:.0f}, have ${account.available_funds:.0f}",
                recommendation=rec,
            )

        try:
            bracket = await self.ibkr.place_bracket_order(
                ticker=rec.ticker,
                side=OrderSide.BUY,
                quantity=qty,
                entry_price=None,
                stop_loss_price=sl,
                take_profit_price=tp,
            )
            logger.info(
                f"BUY {rec.ticker}: {qty} shares @ ~${entry:.2f} "
                f"SL=${sl:.2f} TP=${tp:.2f}"
            )
            await self._notify(rec, bracket, "EXECUTED")
            return ExecutionResult(
                ticker=rec.ticker, action=rec.action.value, success=True,
                bracket=bracket, recommendation=rec,
            )
        except Exception as e:
            logger.error(f"Order failed for {rec.ticker}: {e}")
            return ExecutionResult(
                ticker=rec.ticker, action=rec.action.value, success=False,
                reason=f"Order failed: {e}", recommendation=rec,
            )

    async def _execute_sell(
        self, rec: FinalRecommendation, account: AccountSummary,
        existing,
    ) -> ExecutionResult:
        if existing is None or existing.quantity <= 0:
            return ExecutionResult(
                ticker=rec.ticker, action=rec.action.value, success=True,
                reason="No long position to close — skipping (no short selling)",
                recommendation=rec,
            )

        try:
            trade = await self.ibkr.close_position(rec.ticker)
            await self.ibkr.cancel_all_orders(rec.ticker)
            logger.info(f"SELL {rec.ticker}: closed {existing.quantity} shares")
            await self._notify(rec, None, f"CLOSED {existing.quantity} shares")
            return ExecutionResult(
                ticker=rec.ticker, action=rec.action.value, success=True,
                reason=f"Closed {existing.quantity} shares", recommendation=rec,
            )
        except Exception as e:
            logger.error(f"Close position failed for {rec.ticker}: {e}")
            return ExecutionResult(
                ticker=rec.ticker, action=rec.action.value, success=False,
                reason=f"Close failed: {e}", recommendation=rec,
            )

    async def _notify(
        self, rec: FinalRecommendation, bracket: BracketResult | None, status: str,
    ):
        if not self.settings.telegram_enabled:
            return

        import httpx
        lines = [
            f"\U0001f4e1 <b>ביצוע — {status}</b>",
            "",
            f"\U0001f4cc {_esc(rec.ticker)} — "
            f"{'קניה' if rec.action in ACTIONABLE_BUY else 'מכירה'}",
            f"\U0001f3af ביטחון: {rec.confidence:.0%}",
        ]

        if bracket:
            lines.extend([
                "",
                f"\U0001f4ca כמות: {bracket.quantity} מניות",
                f"\U0001f4b5 כניסה: ~${bracket.entry_price or 0:.2f} (שוק)",
                f"\U0001f6d1 SL: ${bracket.stop_loss_price:.2f}",
                f"✅ TP: ${bracket.take_profit_price:.2f}",
                f"\U0001f4b0 סיכון: ~${abs(bracket.quantity * ((bracket.entry_price or 0) - bracket.stop_loss_price)):.0f}",
            ])

        lines.extend([
            "",
            f"PDT: {self.ibkr.pdt.trades_used}/3 עסקאות יום",
            f"\U0001f552 {datetime.utcnow().strftime('%H:%M UTC')}",
        ])

        url = f"https://api.telegram.org/bot{self.settings.telegram_bot_token}/sendMessage"
        try:
            async with httpx.AsyncClient() as client:
                await client.post(url, json={
                    "chat_id": self.settings.telegram_chat_id,
                    "text": "\n".join(lines),
                    "parse_mode": "HTML",
                    "disable_web_page_preview": True,
                })
        except Exception as e:
            logger.error(f"Telegram execution notification failed: {e}")

    async def run_scan_and_execute(self) -> list[ExecutionResult]:
        tickers = self.settings.get_watchlist()
        results = []

        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        print(f"\n{'='*55}")
        print(f"  IBKR Execution Scan — {len(tickers)} tickers")
        print(f"  {now}")
        print(f"  Mode: {'LIVE' if self.settings.ibkr_trading_enabled else 'DRY RUN'}")
        print(f"  PDT: {self.ibkr.pdt.trades_used}/3 day trades used")
        print(f"{'='*55}\n")

        for i, ticker in enumerate(tickers, 1):
            print(f"[{i}/{len(tickers)}] {ticker}...", end=" ", flush=True)
            result = await self.analyze_and_execute(ticker)
            results.append(result)

            status = "OK" if result.success else "FAIL"
            print(f"{status} — {result.action} — {result.reason or 'executed'}")

            if i < len(tickers):
                await asyncio.sleep(2)

        executed = [r for r in results if r.bracket]
        skipped = [r for r in results if r.success and not r.bracket]
        failed = [r for r in results if not r.success]

        print(f"\n{'='*55}")
        print(f"  Results: {len(executed)} executed, {len(skipped)} skipped, {len(failed)} failed")
        if executed:
            for r in executed:
                b = r.bracket
                print(f"    {b.ticker}: {b.side} {b.quantity} @ SL=${b.stop_loss_price} TP=${b.take_profit_price}")
        print(f"{'='*55}\n")

        return results


async def run_execution(settings: Settings | None = None):
    engine = ExecutionEngine(settings)
    if not await engine.start():
        print("Failed to connect to IBKR. Make sure TWS/Gateway is running.")
        print("  Paper Trading port: 7497")
        print("  Live Trading port:  7496")
        return
    try:
        await engine.run_scan_and_execute()
    finally:
        engine.stop()


def main():
    import sys
    logging.basicConfig(level=logging.INFO)
    settings = Settings()

    if "--tickers" in sys.argv:
        idx = sys.argv.index("--tickers")
        if idx + 1 < len(sys.argv):
            settings.watchlist = sys.argv[idx + 1]

    if "--live" in sys.argv:
        settings.ibkr_port = 7496
        print("⚠️  LIVE TRADING MODE — real money!")

    asyncio.run(run_execution(settings))


if __name__ == "__main__":
    main()
