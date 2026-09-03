"""Unit tests for risk calculation functions."""
import numpy as np
import pandas as pd
import pytest

from src.tools.risk_calculations import (
    compute_max_drawdown,
    compute_sharpe_ratio,
    compute_var_95,
    compute_volatility_percentile,
)


@pytest.fixture
def price_series():
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.015, 252)
    prices = 100 * np.cumprod(1 + returns)
    return pd.Series(prices)


class TestVolatilityPercentile:
    def test_range_0_1(self, price_series):
        result = compute_volatility_percentile(price_series)
        assert 0.0 <= result <= 1.0

    def test_short_series_returns_default(self):
        result = compute_volatility_percentile(pd.Series([100, 101]))
        assert result == 0.5


class TestVaR95:
    def test_negative_value(self, price_series):
        result = compute_var_95(price_series)
        assert result is not None
        assert result < 0

    def test_none_for_short_series(self):
        assert compute_var_95(pd.Series([100, 101, 102])) is None


class TestMaxDrawdown:
    def test_negative_value(self, price_series):
        result = compute_max_drawdown(price_series)
        assert result is not None
        assert result <= 0

    def test_no_drawdown_for_monotonic(self):
        prices = pd.Series(range(100, 200))
        result = compute_max_drawdown(prices)
        assert result == pytest.approx(0.0)


class TestSharpeRatio:
    def test_returns_float(self, price_series):
        result = compute_sharpe_ratio(price_series)
        assert result is not None
        assert isinstance(result, float)

    def test_none_for_short_series(self):
        assert compute_sharpe_ratio(pd.Series([100, 101])) is None
