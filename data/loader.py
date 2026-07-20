"""
Data loading module with caching, retry logic, and validation.

Supports multiple data sources with a unified interface.
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from config import settings
from core.exceptions import DataError

logger = logging.getLogger(__name__)


class DataCache:
    """Simple disk-based cache for loaded data."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or settings.CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _key(self, *args) -> str:
        raw = ":".join(str(a) for a in args)
        return hashlib.md5(raw.encode()).hexdigest()
    
    def get(self, *args) -> Optional[pd.DataFrame]:
        if not settings.DATA_CACHE_ENABLED:
            return None
        
        key = self._key(*args)
        cache_file = self.cache_dir / f"{key}.pkl"
        
        if not cache_file.exists():
            return None
        
        # Check TTL
        age = time.time() - cache_file.stat().st_mtime
        if age > settings.DATA_CACHE_TTL_HOURS * 3600:
            cache_file.unlink(missing_ok=True)
            return None
        
        try:
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None
    
    def set(self, data: pd.DataFrame, *args) -> None:
        if not settings.DATA_CACHE_ENABLED:
            return
        
        key = self._key(*args)
        cache_file = self.cache_dir / f"{key}.pkl"
        
        with open(cache_file, "wb") as f:
            pickle.dump(data, f)


class DataLoader:
    """Unified data loader with caching and validation."""
    
    def __init__(self, cache: Optional[DataCache] = None):
        self.cache = cache or DataCache()
    
    def load_tickers(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        interval: str = "1d",
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Load OHLCV data for a list of tickers with caching.
        
        Parameters
        ----------
        tickers : List[str]
            Stock ticker symbols
        start_date : str
            Start date (YYYY-MM-DD)
        end_date : str
            End date (YYYY-MM-DD)
        interval : str
            Data interval ('1d', '1wk', '1mo')
        use_cache : bool
            Whether to use disk cache
        
        Returns
        -------
        pd.DataFrame
            OHLCV data with columns: date, ticker, open, high, low, close, volume
        """
        cache_key = (tuple(sorted(tickers)), start_date, end_date, interval)
        
        if use_cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                logger.info(f"Loaded {len(tickers)} tickers from cache")
                return cached
        
        data = load_yfinance_data(tickers, start_date, end_date, interval)
        
        if use_cache:
            self.cache.set(data, cache_key)
        
        return data
    
    def get_sp500_universe(
        self,
        limit: Optional[int] = None,
        use_yahoo_screener: bool = True,
    ) -> List[str]:
        """Get S&P 500 universe."""
        from src.sp500 import get_sp500_universe as _get
        return _get(limit=limit, use_yahoo_screener=use_yahoo_screener)


def load_yfinance_data(
    tickers: List[str],
    start_date: str,
    end_date: str,
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Load OHLCV data from Yahoo Finance.
    Includes retry logic and progress logging.
    """
    data_list: List[pd.DataFrame] = []
    
    for ticker in tickers:
        for attempt in range(settings.YFINANCE_MAX_RETRIES):
            try:
                logger.debug(f"Loading {ticker} (attempt {attempt + 1})")
                df = yf.download(
                    ticker,
                    start=start_date,
                    end=end_date,
                    interval=interval,
                    progress=False,
                    auto_adjust=True,
                )
                break
            except Exception as e:
                if attempt == settings.YFINANCE_MAX_RETRIES - 1:
                    logger.warning(f"Failed to load {ticker}: {e}")
                    df = pd.DataFrame()
                else:
                    time.sleep(2 ** attempt)
        
        if df.empty:
            continue
        
        df = df.reset_index()
        
        # Handle multi-level columns from yfinance
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        df.columns = df.columns.str.lower()
        df["ticker"] = ticker.upper()
        
        # Normalize column names
        rename = {
            "adj close": "close",
            "adj_close": "close",
        }
        df = df.rename(columns=rename)
        
        required = {"date", "open", "high", "low", "close", "volume"}
        if not required.issubset(df.columns):
            logger.warning(f"Missing columns for {ticker}: {required - set(df.columns)}")
            continue
        
        df = df[["date", "ticker", "open", "high", "low", "close", "volume"]]
        df["date"] = pd.to_datetime(df["date"])
        df = df.dropna(subset=["close"])
        
        data_list.append(df)
    
    if not data_list:
        raise DataError("No data loaded for any tickers")
    
    combined = pd.concat(data_list, ignore_index=True)
    combined = combined.sort_values(["date", "ticker"]).reset_index(drop=True)
    
    logger.info(
        f"Loaded {combined['ticker'].nunique()} tickers, "
        f"{len(combined)} rows, "
        f"{combined['date'].min().date()} to {combined['date'].max().date()}"
    )
    
    return combined


def load_csv_data(filepath: str) -> pd.DataFrame:
    """Load OHLCV data from CSV file."""
    df = pd.read_csv(filepath)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)
    return df
