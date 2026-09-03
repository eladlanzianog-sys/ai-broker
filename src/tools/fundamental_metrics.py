"""Fundamental analysis metric calculations.

Pure deterministic functions for financial health, growth, and valuation scoring.
"""
from src.models.schemas import FinancialStatements


def score_financial_health(fin: FinancialStatements) -> float:
    score = 0.5
    checks = 0

    if fin.current_ratio is not None:
        checks += 1
        if fin.current_ratio >= 2.0:
            score += 0.15
        elif fin.current_ratio >= 1.5:
            score += 0.10
        elif fin.current_ratio < 1.0:
            score -= 0.15

    if fin.debt_to_equity is not None:
        checks += 1
        if fin.debt_to_equity < 0.5:
            score += 0.15
        elif fin.debt_to_equity < 1.0:
            score += 0.05
        elif fin.debt_to_equity > 2.0:
            score -= 0.15

    if fin.total_cash is not None and fin.total_debt is not None and fin.total_debt > 0:
        checks += 1
        cash_to_debt = fin.total_cash / fin.total_debt
        if cash_to_debt > 1.0:
            score += 0.10
        elif cash_to_debt < 0.2:
            score -= 0.10

    if fin.free_cash_flow_ttm is not None:
        checks += 1
        if fin.free_cash_flow_ttm > 0:
            score += 0.10
        else:
            score -= 0.10

    return max(0.0, min(1.0, score))


def score_growth(fin: FinancialStatements) -> float:
    score = 0.5

    if fin.revenue_growth_yoy is not None:
        if fin.revenue_growth_yoy > 0.20:
            score += 0.25
        elif fin.revenue_growth_yoy > 0.10:
            score += 0.15
        elif fin.revenue_growth_yoy > 0:
            score += 0.05
        else:
            score -= 0.15

    if fin.earnings_growth_yoy is not None:
        if fin.earnings_growth_yoy > 0.20:
            score += 0.20
        elif fin.earnings_growth_yoy > 0:
            score += 0.10
        else:
            score -= 0.10

    return max(0.0, min(1.0, score))


def score_profitability(fin: FinancialStatements) -> float:
    score = 0.5

    if fin.roe is not None:
        if fin.roe > 0.20:
            score += 0.20
        elif fin.roe > 0.10:
            score += 0.10
        elif fin.roe < 0:
            score -= 0.20

    if (
        fin.net_income_ttm is not None
        and fin.revenue_ttm is not None
        and fin.revenue_ttm > 0
    ):
        margin = fin.net_income_ttm / fin.revenue_ttm
        if margin > 0.20:
            score += 0.15
        elif margin > 0.10:
            score += 0.08
        elif margin < 0:
            score -= 0.15

    return max(0.0, min(1.0, score))


def estimate_intrinsic_value(fin: FinancialStatements) -> float | None:
    if fin.free_cash_flow_ttm is None or fin.free_cash_flow_ttm <= 0:
        return None
    if fin.market_cap is None or fin.market_cap <= 0:
        return None

    growth_rate = fin.revenue_growth_yoy if fin.revenue_growth_yoy is not None else 0.05
    growth_rate = max(0.0, min(growth_rate, 0.25))
    discount_rate = 0.10
    terminal_growth = 0.03
    projection_years = 10

    fcf = fin.free_cash_flow_ttm
    total_pv = 0.0
    for year in range(1, projection_years + 1):
        fcf *= 1 + growth_rate
        total_pv += fcf / ((1 + discount_rate) ** year)

    terminal_value = fcf * (1 + terminal_growth) / (discount_rate - terminal_growth)
    total_pv += terminal_value / ((1 + discount_rate) ** projection_years)

    return total_pv


def compute_margin_of_safety(
    intrinsic_value: float | None,
    current_price: float,
) -> float | None:
    if intrinsic_value is None or intrinsic_value <= 0 or current_price <= 0:
        return None
    shares_implied = 1.0
    return (intrinsic_value - current_price) / intrinsic_value
