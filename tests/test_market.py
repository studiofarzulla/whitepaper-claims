"""Tests for market summary statistics (src/market/summary_statistics.py)."""
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import tempfile

from market.summary_statistics import SummaryStatistics, STATISTICS


def _make_ohlcv(prices, volumes=None):
    """Helper: create a minimal OHLCV DataFrame from close prices."""
    n = len(prices)
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    if volumes is None:
        volumes = np.ones(n) * 1e6
    return pd.DataFrame({
        "timestamp": dates,
        "open": prices,
        "high": prices * 1.01,
        "low": prices * 0.99,
        "close": prices,
        "volume": volumes,
    })


@pytest.fixture
def ss():
    """SummaryStatistics instance with temp dirs."""
    with tempfile.TemporaryDirectory() as market, tempfile.TemporaryDirectory() as out:
        yield SummaryStatistics(market_dir=market, output_dir=out)


# ── compute_returns ─────────────────────────────────────────────────

class TestComputeReturns:
    def test_constant_prices_zero_returns(self, ss):
        """Constant price → returns = 0."""
        df = _make_ohlcv(np.full(100, 50.0))
        ret = ss.compute_returns(df)
        np.testing.assert_allclose(ret.values, 0.0, atol=1e-14)

    def test_known_values(self, ss):
        """Log return of 100 → 110 = ln(1.1)."""
        df = _make_ohlcv(np.array([100.0, 110.0, 110.0]))
        ret = ss.compute_returns(df)
        assert ret.iloc[0] == pytest.approx(np.log(1.1), rel=1e-10)
        assert ret.iloc[1] == pytest.approx(0.0, abs=1e-14)

    def test_length(self, ss):
        """Returns series has len(prices) - 1 entries."""
        df = _make_ohlcv(np.arange(1.0, 51.0))
        ret = ss.compute_returns(df)
        assert len(ret) == 49

    def test_formula_is_log_returns(self, ss):
        """Verify log(p_t / p_{t-1}) formula."""
        np.random.seed(7)
        prices = 100 * np.cumprod(1 + np.random.normal(0, 0.02, 50))
        df = _make_ohlcv(prices)
        ret = ss.compute_returns(df)
        expected = np.diff(np.log(prices))
        np.testing.assert_allclose(ret.values, expected, rtol=1e-10)


# ── compute_max_drawdown ────────────────────────────────────────────

class TestComputeMaxDrawdown:
    def test_monotonic_increase_zero(self, ss):
        """Monotonically increasing prices → drawdown = 0."""
        prices = pd.Series(np.arange(1.0, 101.0))
        assert ss.compute_max_drawdown(prices) == pytest.approx(0.0, abs=1e-14)

    def test_known_drawdown(self, ss):
        """Peak 100, trough 60 → max drawdown = 0.4."""
        prices = pd.Series([80.0, 100.0, 60.0, 90.0])
        assert ss.compute_max_drawdown(prices) == pytest.approx(0.4, rel=1e-10)

    def test_nonnegative(self, ss, sample_price_series):
        """Drawdown is non-negative."""
        dd = ss.compute_max_drawdown(sample_price_series["close"])
        assert dd >= 0

    def test_at_most_one(self, ss):
        """Drawdown ≤ 1 (can't lose more than 100%)."""
        prices = pd.Series([100.0, 1.0, 50.0])
        dd = ss.compute_max_drawdown(prices)
        assert dd <= 1.0 + 1e-10


# ── compute_trend ───────────────────────────────────────────────────

class TestComputeTrend:
    def test_uptrend_positive(self, ss):
        """Linearly increasing prices → positive trend."""
        prices = pd.Series(np.linspace(10, 20, 100))
        assert ss.compute_trend(prices) > 0

    def test_downtrend_negative(self, ss):
        """Linearly decreasing prices → negative trend."""
        prices = pd.Series(np.linspace(20, 10, 100))
        assert ss.compute_trend(prices) < 0

    def test_flat_near_zero(self, ss):
        """Flat prices → trend ≈ 0."""
        prices = pd.Series(np.full(100, 50.0))
        assert ss.compute_trend(prices) == pytest.approx(0.0, abs=1e-10)


# ── compute_stats ───────────────────────────────────────────────────

class TestComputeStats:
    def test_returns_all_keys(self, ss, sample_price_series):
        """Result contains all 7 expected statistic keys."""
        result = ss.compute_stats(sample_price_series.copy())
        for key in STATISTICS:
            assert key in result, f"Missing key: {key}"

    def test_values_are_finite(self, ss, sample_price_series):
        """All computed statistics are finite numbers."""
        result = ss.compute_stats(sample_price_series.copy())
        for key, val in result.items():
            assert np.isfinite(val), f"{key} is not finite: {val}"

    def test_max_drawdown_consistent(self, ss, sample_price_series):
        """max_drawdown from compute_stats is non-negative."""
        result = ss.compute_stats(sample_price_series.copy())
        assert result["max_drawdown"] >= 0
