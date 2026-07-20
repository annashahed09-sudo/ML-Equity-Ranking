"""
Custom exceptions for the quantitative equity research platform.

Provides a hierarchy of typed exceptions for granular error handling
across all platform modules.
"""

from __future__ import annotations


class QuantsError(Exception):
    """Base exception for all platform errors."""
    def __init__(self, message: str, detail: str | None = None):
        self.detail = detail
        super().__init__(message)


class DataError(QuantsError):
    """Raised when data loading, validation, or transformation fails."""
    pass


class ModelError(QuantsError):
    """Raised when model training, prediction, or serialization fails."""
    pass


class ValidationError(QuantsError):
    """Raised when validation checks fail."""
    pass


class PortfolioError(QuantsError):
    """Raised when portfolio construction or optimization fails."""
    pass


class RiskError(QuantsError):
    """Raised when risk computation fails."""
    pass


class ConfigError(QuantsError):
    """Raised when configuration is invalid or missing."""
    pass


class FactorError(QuantsError):
    """Raised when factor computation fails."""
    pass


class SignalError(QuantsError):
    """Raised when signal processing fails."""
    pass


class BacktestError(QuantsError):
    """Raised when backtesting fails."""
    pass


class ConvergenceError(QuantsError):
    """Raised when optimization fails to converge."""
    pass


class NumericError(QuantsError):
    """Raised when numerical computation produces invalid values."""
    pass


class DataQualityError(DataError):
    """Raised when data fails quality checks."""
    pass


class LookAheadBiasError(ValidationError):
    """Raised when look-ahead bias is detected."""
    pass


class LeakageError(ValidationError):
    """Raised when data leakage is detected."""
    pass
