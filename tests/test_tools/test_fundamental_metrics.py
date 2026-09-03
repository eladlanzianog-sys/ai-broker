"""Unit tests for fundamental metric calculations."""
import pytest

from src.models.schemas import FinancialStatements
from src.tools.fundamental_metrics import (
    compute_margin_of_safety,
    estimate_intrinsic_value,
    score_financial_health,
    score_growth,
    score_profitability,
)


@pytest.fixture
def healthy_company():
    return FinancialStatements(
        revenue_ttm=100_000_000,
        net_income_ttm=25_000_000,
        total_debt=20_000_000,
        total_cash=50_000_000,
        free_cash_flow_ttm=30_000_000,
        current_ratio=2.5,
        debt_to_equity=0.3,
        roe=0.25,
        revenue_growth_yoy=0.15,
        earnings_growth_yoy=0.20,
        market_cap=500_000_000,
    )


@pytest.fixture
def weak_company():
    return FinancialStatements(
        revenue_ttm=50_000_000,
        net_income_ttm=-5_000_000,
        total_debt=100_000_000,
        total_cash=5_000_000,
        free_cash_flow_ttm=-10_000_000,
        current_ratio=0.8,
        debt_to_equity=3.0,
        roe=-0.10,
        revenue_growth_yoy=-0.05,
        earnings_growth_yoy=-0.20,
        market_cap=100_000_000,
    )


class TestFinancialHealth:
    def test_healthy_scores_high(self, healthy_company):
        score = score_financial_health(healthy_company)
        assert score > 0.7

    def test_weak_scores_low(self, weak_company):
        score = score_financial_health(weak_company)
        assert score < 0.4

    def test_range_0_1(self, healthy_company):
        score = score_financial_health(healthy_company)
        assert 0.0 <= score <= 1.0


class TestGrowth:
    def test_growing_company(self, healthy_company):
        score = score_growth(healthy_company)
        assert score > 0.6

    def test_declining_company(self, weak_company):
        score = score_growth(weak_company)
        assert score < 0.4


class TestProfitability:
    def test_profitable_company(self, healthy_company):
        score = score_profitability(healthy_company)
        assert score > 0.7

    def test_unprofitable_company(self, weak_company):
        score = score_profitability(weak_company)
        assert score < 0.3


class TestIntrinsicValue:
    def test_positive_fcf(self, healthy_company):
        value = estimate_intrinsic_value(healthy_company)
        assert value is not None
        assert value > 0

    def test_negative_fcf_returns_none(self, weak_company):
        value = estimate_intrinsic_value(weak_company)
        assert value is None


class TestMarginOfSafety:
    def test_undervalued(self):
        mos = compute_margin_of_safety(200.0, 150.0)
        assert mos is not None
        assert mos > 0

    def test_overvalued(self):
        mos = compute_margin_of_safety(100.0, 150.0)
        assert mos is not None
        assert mos < 0
