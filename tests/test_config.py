"""
Tests for the configuration module.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from config.settings import Settings, Environment, LogLevel


class TestSettings:
    """Settings validation tests."""

    def test_default_settings(self):
        settings = Settings()
        assert settings.ENVIRONMENT == Environment.DEVELOPMENT
        assert settings.LOG_LEVEL == LogLevel.INFO

    def test_environment_detection(self):
        dev = Settings(ENVIRONMENT="development")
        assert dev.is_development()
        assert not dev.is_production()

        prod = Settings(ENVIRONMENT="production")
        assert prod.is_production()
        assert not prod.is_development()

    def test_default_paths_exist(self, tmp_path):
        data_dir = tmp_path / "test_data"
        settings = Settings(
            DATA_DIR=str(data_dir),
            CACHE_DIR=str(tmp_path / "test_cache"),
            OUTPUT_DIR=str(tmp_path / "test_output"),
        )
        assert data_dir.exists()

    def test_factor_settings(self):
        settings = Settings()
        assert settings.FACTOR_WINSORIZATION_LIMITS == (0.01, 0.99)
        assert settings.FACTOR_NEUTRALIZATION in ["sector", "market", "none"]

    def test_model_settings(self):
        settings = Settings()
        assert settings.LIGHTGBM_NUM_LEAVES == 31
        assert settings.XGBOOST_MAX_DEPTH == 6
        assert 0 < settings.CATBOOST_LEARNING_RATE <= 1

    def test_portfolio_settings(self):
        settings = Settings()
        assert 0 < settings.MAX_POSITION_WEIGHT <= 1
        assert settings.PORTFOLIO_OBJECTIVE.value in ["mean_variance", "risk_parity",
                                                       "minimum_variance", "max_diversification",
                                                       "equal_risk_contribution", "black_litterman"]

    def test_risk_settings(self):
        settings = Settings()
        assert 0 <= settings.RISK_FREE_RATE <= 1
        assert 0 < settings.CONFIDENCE_LEVEL < 1

    def test_backtest_settings(self):
        settings = Settings()
        assert settings.BT_INITIAL_CAPITAL > 0
        assert 0 <= settings.BT_COMMISSION_PCT <= 1


class TestEnums:
    """Enum validation tests."""

    def test_environment_values(self):
        assert Environment.DEVELOPMENT.value == "development"
        assert Environment.STAGING.value == "staging"
        assert Environment.PRODUCTION.value == "production"

    def test_log_level_values(self):
        assert LogLevel.DEBUG.value == "DEBUG"
        assert LogLevel.INFO.value == "INFO"
        assert LogLevel.ERROR.value == "ERROR"
