"""Unit tests for technical indicator functions."""
import numpy as np
import pandas as pd
import pytest

from src.tools.technical_indicators import (
    compute_atr,
    compute_bollinger_bands,
    compute_ema,
    compute_macd,
    compute_obv,
    compute_rsi,
    compute_sma,
    compute_stochastic,
    detect_patterns,
    identify_support_resistance,
)


@pytest.fixture
def rising_prices():
    return pd.Series(np.linspace(100, 200, 300))


@pytest.fixture
def flat_prices():
    return pd.Series([100.0] * 300)


@pytest.fixture
def ohlcv_df():
    np.random.seed(42)
    n = 300
    close = np.cumsum(np.random.normal(0.1, 1.0, n)) + 150
    return pd.DataFrame(
        {
            "open": close + np.random.normal(0, 0.5, n),
            "high": close + abs(np.random.normal(0, 1.0, n)),
            "low": close - abs(np.random.normal(0, 1.0, n)),
            "close": close,
            "volume": np.random.randint(1_000_000, 10_000_000, n),
        }
    )


class TestSMA:
    def test_returns_float(self, rising_prices):
        result = compute_sma(rising_prices, 20)
        assert isinstance(result, float)

    def test_returns_none_for_short_series(self):
        assert compute_sma(pd.Series([1, 2, 3]), 20) is None

    def test_flat_series_equals_value(self, flat_prices):
        result = compute_sma(flat_prices, 20)
        assert result == pytest.approx(100.0)


class TestEMA:
    def test_returns_float(self, rising_prices):
        assert isinstance(compute_ema(rising_prices, 12), float)

    def test_returns_none_for_short_series(self):
        assert compute_ema(pd.Series([1, 2]), 12) is None


class TestRSI:
    def test_rising_prices_high_rsi(self, rising_prices):
        result = compute_rsi(rising_prices, 14)
        assert result is not None
        assert result > 70

    def test_returns_none_for_short_series(self):
        assert compute_rsi(pd.Series([1, 2, 3]), 14) is None

    def test_range_0_100(self, rising_prices):
        result = compute_rsi(rising_prices, 14)
        assert 0 <= result <= 100


class TestMACD:
    def test_returns_dict_with_keys(self, rising_prices):
        result = compute_macd(rising_prices)
        assert "macd_line" in result
        assert "macd_signal" in result
        assert "macd_histogram" in result

    def test_returns_nones_for_short(self):
        result = compute_macd(pd.Series([1, 2, 3]))
        assert result["macd_line"] is None


class TestBollinger:
    def test_upper_above_lower(self, rising_prices):
        result = compute_bollinger_bands(rising_prices)
        assert result["bollinger_upper"] > result["bollinger_lower"]

    def test_middle_between(self, rising_prices):
        result = compute_bollinger_bands(rising_prices)
        assert result["bollinger_lower"] < result["bollinger_middle"] < result["bollinger_upper"]


class TestATR:
    def test_positive(self, ohlcv_df):
        result = compute_atr(ohlcv_df, 14)
        assert result is not None
        assert result > 0


class TestOBV:
    def test_returns_float(self, ohlcv_df):
        result = compute_obv(ohlcv_df)
        assert isinstance(result, float)


class TestStochastic:
    def test_returns_values(self, ohlcv_df):
        result = compute_stochastic(ohlcv_df)
        assert result["stochastic_k"] is not None
        assert 0 <= result["stochastic_k"] <= 100


class TestPatterns:
    def test_doji_detection(self):
        df = pd.DataFrame(
            {
                "open": [100, 101, 102, 103, 100.01],
                "high": [105, 106, 107, 108, 105],
                "low": [95, 96, 97, 98, 95],
                "close": [101, 102, 103, 104, 100.0],
                "volume": [1000] * 5,
            }
        )
        patterns = detect_patterns(df)
        assert "doji" in patterns


class TestSupportResistance:
    def test_returns_tuple(self, ohlcv_df):
        support, resistance = identify_support_resistance(ohlcv_df)
        assert support is not None
        assert resistance is not None
        assert support < resistance
