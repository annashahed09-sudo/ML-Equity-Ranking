"""
Pytest configuration and shared fixtures for the ML-Equity-Ranking test suite.

This conftest.py provides:
- Shared fixtures used across all test modules
- pytest configuration hooks
- Coverage configuration
- Import path setup
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure project root is on sys.path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Set random seed for reproducibility
np.random.seed(42)


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Return the project root directory."""
    return PROJECT_ROOT


@pytest.fixture
def rng() -> np.random.Generator:
    """Return a seeded random number generator."""
    return np.random.default_rng(42)


@pytest.fixture
def sample_dates() -> pd.DatetimeIndex:
    """Return a sample date range."""
    return pd.date_range("2023-01-01", periods=252, freq="B")


@pytest.fixture
def sample_tickers() -> list[str]:
    """Return a list of sample tickers."""
    return [f"ASSET{i:02d}" for i in range(10)]


@pytest.fixture
def sample_ohlcv(sample_dates, sample_tickers, rng) -> pd.DataFrame:
    """Generate sample OHLCV data."""
    rows = []
    for ticker in sample_tickers:
        price = 100 + np.cumsum(rng.normal(0, 1.5, len(sample_dates)))
        price = np.maximum(price, 10)
        for i, d in enumerate(sample_dates):
            rows.append({
                "date": d,
                "ticker": ticker,
                "open": float(price[i]),
                "high": float(price[i] * (1 + abs(rng.normal(0, 0.01)))),
                "low": float(price[i] * (1 - abs(rng.normal(0, 0.01)))),
                "close": float(price[i]),
                "volume": int(rng.integers(1_000_000, 10_000_000)),
            })
    return pd.DataFrame(rows).sort_values(["date", "ticker"]).reset_index(drop=True)


# Register custom markers
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "gpu: marks tests that require GPU")
