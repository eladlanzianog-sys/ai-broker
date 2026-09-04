"""yfinance wrappers for market data collection."""
import asyncio
from datetime import datetime

import yfinance as yf

from src.models.schemas import OHLCVBar, FinancialStatements


def _sync_fetch_ohlcv(ticker: str, start_date: datetime, end_date: datetime) -> list[OHLCVBar]:
    stock = yf.Ticker(ticker)
    df = stock.history(start=start_date, end=end_date, auto_adjust=True)
    if df.empty:
        return []
    return [
        OHLCVBar(
            date=idx.to_pydatetime(),
            open=row["Open"],
            high=row["High"],
            low=row["Low"],
            close=row["Close"],
            volume=int(row["Volume"]),
        )
        for idx, row in df.iterrows()
    ]


def _sync_fetch_financials(ticker: str) -> FinancialStatements:
    stock = yf.Ticker(ticker)
    info = stock.info or {}
    return FinancialStatements(
        revenue_ttm=info.get("totalRevenue"),
        net_income_ttm=info.get("netIncomeToCommon"),
        total_debt=info.get("totalDebt"),
        total_cash=info.get("totalCash"),
        free_cash_flow_ttm=info.get("freeCashflow"),
        pe_ratio=info.get("trailingPE"),
        pb_ratio=info.get("priceToBook"),
        ps_ratio=info.get("priceToSalesTrailing12Months"),
        debt_to_equity=info.get("debtToEquity"),
        current_ratio=info.get("currentRatio"),
        roe=info.get("returnOnEquity"),
        revenue_growth_yoy=info.get("revenueGrowth"),
        earnings_growth_yoy=info.get("earningsGrowth"),
        market_cap=info.get("marketCap"),
        sector=info.get("sector"),
        industry=info.get("industry"),
    )


def _sync_fetch_company_info(ticker: str) -> dict:
    stock = yf.Ticker(ticker)
    info = stock.info or {}
    return {
        "fifty_two_week_high": info.get("fiftyTwoWeekHigh", 0.0),
        "fifty_two_week_low": info.get("fiftyTwoWeekLow", 0.0),
        "average_volume_10d": info.get("averageDailyVolume10Day", 0),
        "beta": info.get("beta"),
    }


async def fetch_ohlcv(ticker: str, start_date: datetime, end_date: datetime) -> list[OHLCVBar]:
    return await asyncio.to_thread(_sync_fetch_ohlcv, ticker, start_date, end_date)


async def fetch_financials(ticker: str) -> FinancialStatements:
    return await asyncio.to_thread(_sync_fetch_financials, ticker)


async def fetch_company_info(ticker: str) -> dict:
    return await asyncio.to_thread(_sync_fetch_company_info, ticker)
