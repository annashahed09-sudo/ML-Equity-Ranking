"""
Data layer for quantitative equity research.

Handles:
- Data loading (Yahoo Finance, CSV, databases)
- Universe construction (S&P 500, custom lists)
- Data quality checks and validation
- Data cleaning and normalization
- Cache management
"""

from .loader import DataLoader, load_yfinance_data, load_csv_data
from .quality import DataQualityChecker

__all__ = [
    "DataLoader",
    "DataQualityChecker",
    "load_yfinance_data",
    "load_csv_data",
]
