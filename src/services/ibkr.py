"""Interactive Brokers connection and order management via ib_insync."""
from __future__ import annotations

import asyncio
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from ib_insync import (
    IB,
    Contract,
    LimitOrder,
    MarketOrder,
    Order,
    Stock,
    Trade,
    util,
)

logger = logging.getLogger(__name__)


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class Position:
    ticker: str
    quantity: int
    avg_cost: float
    market_price: float
    unrealized_pnl: float
    realized_pnl: float


@dataclass
class AccountSummary:
    net_liquidation: float
    available_funds: float
    buying_power: float
    total_cash: float
    unrealized_pnl: float
    realized_pnl: float


@dataclass
class BracketResult:
    parent_order_id: int
    stop_loss_order_id: int
    take_profit_order_id: int
    ticker: str
    side: str
    quantity: int
    entry_price: float | None
    stop_loss_price: float
    take_profit_price: float
    status: str = "submitted"


@dataclass
class PDTTracker:
    """Tracks day trades to stay under the Pattern Day Trader limit."""
    day_trades: deque = field(default_factory=deque)
    max_day_trades: int = 3
    window_days: int = 5

    def record_day_trade(self, dt: datetime | None = None):
        if dt is None:
            dt = datetime.utcnow()
        self.day_trades.append(dt)
        self._prune()

    def _prune(self):
        cutoff = datetime.utcnow() - timedelta(days=self.window_days)
        while self.day_trades and self.day_trades[0] < cutoff:
            self.day_trades.popleft()

    @property
    def trades_used(self) -> int:
        self._prune()
        return len(self.day_trades)

    @property
    def trades_remaining(self) -> int:
        return max(0, self.max_day_trades - self.trades_used)

    @property
    def can_day_trade(self) -> bool:
        return self.trades_remaining > 0


class IBKRService:
    """Manages connection to IBKR TWS/Gateway and order execution."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 7497,
        client_id: int = 1,
        timeout: int = 30,
        readonly: bool = False,
    ):
        self.host = host
        self.port = port
        self.client_id = client_id
        self.timeout = timeout
        self.readonly = readonly
        self.ib = IB()
        self.pdt = PDTTracker()
        self._connected = False

    async def connect(self) -> bool:
        try:
            await self.ib.connectAsync(
                self.host, self.port, clientId=self.client_id, timeout=self.timeout,
                readonly=self.readonly,
            )
            self._connected = self.ib.isConnected()
            if self._connected:
                logger.info(
                    f"Connected to IBKR at {self.host}:{self.port} "
                    f"(paper={self.port == 7497})"
                )
            return self._connected
        except Exception as e:
            logger.error(f"IBKR connection failed: {e}")
            self._connected = False
            return False

    def disconnect(self):
        if self._connected:
            self.ib.disconnect()
            self._connected = False
            logger.info("Disconnected from IBKR")

    @property
    def is_connected(self) -> bool:
        return self._connected and self.ib.isConnected()

    async def get_account_summary(self) -> AccountSummary:
        self._require_connected()
        summary = self.ib.accountSummary()
        vals: dict[str, float] = {}
        for item in summary:
            if item.tag in (
                "NetLiquidation", "AvailableFunds", "BuyingPower",
                "TotalCashValue", "UnrealizedPnL", "RealizedPnL",
            ):
                try:
                    vals[item.tag] = float(item.value)
                except (ValueError, TypeError):
                    vals[item.tag] = 0.0
        return AccountSummary(
            net_liquidation=vals.get("NetLiquidation", 0),
            available_funds=vals.get("AvailableFunds", 0),
            buying_power=vals.get("BuyingPower", 0),
            total_cash=vals.get("TotalCashValue", 0),
            unrealized_pnl=vals.get("UnrealizedPnL", 0),
            realized_pnl=vals.get("RealizedPnL", 0),
        )

    async def get_positions(self) -> list[Position]:
        self._require_connected()
        positions = self.ib.positions()
        result = []
        for pos in positions:
            if pos.position == 0:
                continue
            result.append(Position(
                ticker=pos.contract.symbol,
                quantity=int(pos.position),
                avg_cost=pos.avgCost,
                market_price=pos.marketPrice if hasattr(pos, "marketPrice") else 0,
                unrealized_pnl=pos.unrealizedPNL if hasattr(pos, "unrealizedPNL") else 0,
                realized_pnl=pos.realizedPNL if hasattr(pos, "realizedPNL") else 0,
            ))
        return result

    async def get_position(self, ticker: str) -> Position | None:
        positions = await self.get_positions()
        for p in positions:
            if p.ticker.upper() == ticker.upper():
                return p
        return None

    async def get_market_price(self, ticker: str) -> float | None:
        self._require_connected()
        contract = Stock(ticker, "SMART", "USD")
        self.ib.qualifyContracts(contract)
        ticker_data = self.ib.reqMktData(contract, "", False, False)
        await asyncio.sleep(2)
        price = ticker_data.marketPrice()
        self.ib.cancelMktData(contract)
        if price and price > 0 and not util.isNan(price):
            return float(price)
        return None

    async def place_bracket_order(
        self,
        ticker: str,
        side: OrderSide,
        quantity: int,
        entry_price: float | None,
        stop_loss_price: float,
        take_profit_price: float,
    ) -> BracketResult:
        self._require_connected()
        if self.readonly:
            raise PermissionError("Cannot place orders in readonly mode")
        if quantity <= 0:
            raise ValueError("Quantity must be positive")

        contract = Stock(ticker, "SMART", "USD")
        self.ib.qualifyContracts(contract)

        bracket = self.ib.bracketOrder(
            action=side.value,
            quantity=quantity,
            limitPrice=entry_price if entry_price else 0,
            takeProfitPrice=take_profit_price,
            stopLossPrice=stop_loss_price,
        )

        if entry_price is None:
            bracket[0].orderType = "MKT"
            bracket[0].lmtPrice = 0

        trades = []
        for order in bracket:
            trade = self.ib.placeOrder(contract, order)
            trades.append(trade)

        await asyncio.sleep(1)

        return BracketResult(
            parent_order_id=bracket[0].orderId,
            stop_loss_order_id=bracket[1].orderId,
            take_profit_order_id=bracket[2].orderId,
            ticker=ticker,
            side=side.value,
            quantity=quantity,
            entry_price=entry_price,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
            status="submitted",
        )

    async def place_market_order(
        self, ticker: str, side: OrderSide, quantity: int,
    ) -> Trade:
        self._require_connected()
        if self.readonly:
            raise PermissionError("Cannot place orders in readonly mode")
        contract = Stock(ticker, "SMART", "USD")
        self.ib.qualifyContracts(contract)
        order = MarketOrder(side.value, quantity)
        trade = self.ib.placeOrder(contract, order)
        await asyncio.sleep(1)
        return trade

    async def close_position(self, ticker: str) -> Trade | None:
        pos = await self.get_position(ticker)
        if pos is None or pos.quantity == 0:
            logger.info(f"No position to close for {ticker}")
            return None
        side = OrderSide.SELL if pos.quantity > 0 else OrderSide.BUY
        qty = abs(pos.quantity)
        return await self.place_market_order(ticker, side, qty)

    async def cancel_all_orders(self, ticker: str | None = None):
        self._require_connected()
        open_orders = self.ib.openOrders()
        for order in open_orders:
            if ticker and hasattr(order, "contract"):
                if order.contract.symbol.upper() != ticker.upper():
                    continue
            self.ib.cancelOrder(order)
        logger.info(f"Cancelled orders{f' for {ticker}' if ticker else ''}")

    async def get_open_orders(self) -> list[Trade]:
        self._require_connected()
        return self.ib.openTrades()

    def _require_connected(self):
        if not self.is_connected:
            raise ConnectionError(
                "Not connected to IBKR. Start TWS/Gateway and call connect()"
            )
