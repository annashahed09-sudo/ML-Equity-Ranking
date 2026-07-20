"""
Tests for data loading and quality modules.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.quality import DataQualityChecker, DataQualityReport
from data.loader import DataCache


class TestDataQuality:
    """Data quality checker tests."""

    @pytest.fixture
    def clean_data(self) -> pd.DataFrame:
        dates = pd.date_range("2024-01-01", periods=100, freq="B")
        data = []
        for d in dates:
            for t in range(5):
                data.append({
                    "date": d,
                    "ticker": f"T{t}",
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.5,
                    "volume": 1_000_000,
                })
        return pd.DataFrame(data)

    @pytest.fixture
    def dirty_data(self) -> pd.DataFrame:
        dates = pd.date_range("2024-01-01", periods=100, freq="B")
        data = []
        for d in dates:
            for t in range(5):
                data.append({
                    "date": d,
                    "ticker": f"T{t}",
                    "open": 100.0 if t != 0 else float("nan"),
                    "high": 99.0 if t != 1 else 101.0,
                    "low": 101.0 if t != 1 else 99.0,
                    "close": 100.5,
                    "volume": 1_000_000 if t != 2 else 0,
                })
        # Add duplicate
        data.append(data[0])
        return pd.DataFrame(data)

    def test_clean_data_passes(self, clean_data):
        checker = DataQualityChecker()
        report = checker.check(clean_data)
        assert report.passed or report.n_issues < 3

    def test_dirty_data_has_issues(self, dirty_data):
        checker = DataQualityChecker()
        report = checker.check(dirty_data)
        assert report.n_issues > 0

    def test_quality_report_structure(self, clean_data):
        checker = DataQualityChecker()
        report = checker.check(clean_data)
        assert isinstance(report, DataQualityReport)
        assert report.n_rows > 0
        assert report.n_tickers > 0
        assert len(report.date_range) == 2

    def test_clean_removes_invalid(self, dirty_data):
        checker = DataQualityChecker()
        cleaned = checker.clean(dirty_data)
        assert len(cleaned) < len(dirty_data)
        assert (cleaned["high"] >= cleaned["low"]).all()
        assert cleaned["open"].notna().all()

    def test_extreme_returns_detected(self, clean_data):
        # Inject extreme return in the second row of first ticker (row 5)
        # This ensures pct_change within the ticker group catches it
        clean_data.loc[5, "close"] = 300.0  # 200% return from 100.5 -> exceeds 0.5 threshold
        checker = DataQualityChecker(extreme_return_threshold=0.5)
        report = checker.check(clean_data)
        assert report.extreme_returns > 0


class TestDataCache:
    """Data cache tests."""

    def test_cache_key_consistency(self, tmp_path):
        cache = DataCache(cache_dir=tmp_path)
        key1 = cache._key("test", 123)
        key2 = cache._key("test", 123)
        assert key1 == key2

    def test_cache_key_different(self, tmp_path):
        cache = DataCache(cache_dir=tmp_path)
        key1 = cache._key("test", 123)
        key2 = cache._key("test", 456)
        assert key1 != key2

    def test_set_and_get(self, tmp_path):
        cache = DataCache(cache_dir=tmp_path)
        df = pd.DataFrame({"a": [1, 2, 3]})
        cache.set(df, "test_key")
        result = cache.get("test_key")
        assert result is not None
        assert result.equals(df)
