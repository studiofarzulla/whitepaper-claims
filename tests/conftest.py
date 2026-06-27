"""Shared fixtures for whitepaper-claims tests."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def rng():
    """Seeded random generator for reproducibility."""
    return np.random.RandomState(42)


@pytest.fixture
def random_matrix(rng):
    """Factory fixture: returns a seeded random (n, p) matrix."""
    def _make(n=50, p=10):
        return rng.randn(n, p)
    return _make


@pytest.fixture
def identity_matrix():
    """Factory fixture: returns an (n, n) identity matrix."""
    def _make(n=10):
        return np.eye(n)
    return _make


@pytest.fixture
def sample_price_series():
    """Realistic price series with DatetimeIndex for market tests."""
    np.random.seed(42)
    n_days = 252
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    # Geometric Brownian Motion prices
    log_returns = np.random.normal(0.0003, 0.02, n_days)
    log_returns[0] = 0
    prices = 100.0 * np.exp(np.cumsum(log_returns))
    volumes = np.random.lognormal(mean=15, sigma=1, size=n_days)
    df = pd.DataFrame({
        "timestamp": dates,
        "open": prices * (1 + np.random.uniform(-0.005, 0.005, n_days)),
        "high": prices * (1 + np.abs(np.random.normal(0, 0.01, n_days))),
        "low": prices * (1 - np.abs(np.random.normal(0, 0.01, n_days))),
        "close": prices,
        "volume": volumes,
    })
    return df


@pytest.fixture
def sample_returns_series(sample_price_series):
    """Log returns corresponding to sample_price_series."""
    close = sample_price_series["close"]
    return np.log(close / close.shift(1)).dropna()
