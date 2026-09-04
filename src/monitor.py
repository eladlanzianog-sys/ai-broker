"""Portfolio monitor — shows live IBKR account status and positions."""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime

from src.config.settings import Settings
from src.services.ibkr import IBKRService

logger = logging.getLogger(__name__)


async def show_status(settings: Settings | None = None):
    if settings is None:
        settings = Settings()

    ibkr = IBKRService(
        host=settings.ibkr_host,
        port=settings.ibkr_port,
        client_id=settings.ibkr_client_id + 10,
        readonly=True,
    )

    if not await ibkr.connect():
        print("Cannot connect to IBKR. Make sure TWS/Gateway is running.")
        return

    try:
        account = await ibkr.get_account_summary()
        positions = await ibkr.get_positions()
        orders = await ibkr.get_open_orders()

        mode = "PAPER" if settings.ibkr_port == 7497 else "LIVE"
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

        print(f"\n{'='*55}")
        print(f"  IBKR Portfolio Monitor [{mode}]")
        print(f"  {now}")
        print(f"{'='*55}")

        print(f"\n  Account")
        print(f"  {'─'*40}")
        print(f"  Net Liquidation:  ${account.net_liquidation:>12,.2f}")
        print(f"  Available Funds:  ${account.available_funds:>12,.2f}")
        print(f"  Buying Power:     ${account.buying_power:>12,.2f}")
        print(f"  Cash:             ${account.total_cash:>12,.2f}")
        print(f"  Unrealized P&L:   ${account.unrealized_pnl:>12,.2f}")
        print(f"  Realized P&L:     ${account.realized_pnl:>12,.2f}")

        if positions:
            print(f"\n  Positions ({len(positions)})")
            print(f"  {'─'*40}")
            print(f"  {'Ticker':<8} {'Qty':>6} {'Avg Cost':>10} {'P&L':>12}")
            for p in positions:
                pnl_str = f"${p.unrealized_pnl:>+,.2f}" if p.unrealized_pnl else "—"
                print(f"  {p.ticker:<8} {p.quantity:>6} ${p.avg_cost:>9,.2f} {pnl_str:>12}")
        else:
            print(f"\n  No open positions")

        if orders:
            print(f"\n  Open Orders ({len(orders)})")
            print(f"  {'─'*40}")
            for t in orders:
                o = t.order
                c = t.contract
                print(f"  {c.symbol:<8} {o.action:<5} {int(o.totalQuantity):>6} "
                      f"{o.orderType:<6} ${o.lmtPrice or o.auxPrice or 0:>9,.2f}")
        else:
            print(f"\n  No open orders")

        print(f"\n  Settings")
        print(f"  {'─'*40}")
        print(f"  Risk per trade:    {settings.ibkr_risk_per_trade}%")
        print(f"  Max position:      {settings.ibkr_max_position_pct:.0%}")
        print(f"  Reward ratio:      1:{settings.ibkr_reward_ratio}")
        print(f"  Min confidence:    {settings.ibkr_min_confidence:.0%}")
        print(f"  Trading enabled:   {settings.ibkr_trading_enabled}")
        print(f"  PDT trades used:   {ibkr.pdt.trades_used}/3")

        print(f"\n{'='*55}\n")

    finally:
        ibkr.disconnect()


def main():
    logging.basicConfig(level=logging.WARNING)
    asyncio.run(show_status())


if __name__ == "__main__":
    main()
