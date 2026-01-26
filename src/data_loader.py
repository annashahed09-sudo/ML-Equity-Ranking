"""
Data loader for OHLCV equity market data.

This module handles ingestion of time-series equity data from public sources (yfinance)
or local CSV files. Data is returned as a time-indexed pandas DataFrame with columns:
date, ticker, open, high, low, close, volume.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple
import yfinance as yf


def load_yfinance_data(
    tickers: List[str],
    start_date: str,
    end_date: str,
    interval: str = "1d"
) -> pd.DataFrame:
    """
    Load OHLCV data from yfinance for a list of tickers.

    Parameters
    ----------
    tickers : List[str]
        List of ticker symbols (e.g., ['AAPL', 'MSFT', 'GOOGL'])
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
    interval : str, default '1d'
        Data interval ('1d', '1wk', '1mo')

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: date, ticker, open, high, low, close, volume
        Index is reset (date is a column, not index)
    """
    data_list = []
    
    for ticker in tickers:
        print(f"Loading {ticker}...")
        df = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False)
        
        if df.empty:
            print(f"  Warning: No data for {ticker}")
            continue
        
        df = df.reset_index()
        df['Ticker'] = ticker.upper()
        df.columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'ticker']
        
        # Ensure date is datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Drop rows with missing close prices
        df = df.dropna(subset=['close'])
        
        data_list.append(df[['date', 'ticker', 'open', 'high', 'low', 'close', 'volume']])
    
    if not data_list:
        raise ValueError("No data loaded for any tickers")
    
    combined = pd.concat(data_list, ignore_index=True)
    combined = combined.sort_values(['date', 'ticker']).reset_index(drop=True)
    
    return combined


def load_csv_data(filepath: str) -> pd.DataFrame:
    """
    Load OHLCV data from CSV file.

    Expected columns: date, ticker, open, high, low, close, volume
    date should be parseable to datetime.

    Parameters
    ----------
    filepath : str
        Path to CSV file

    Returns
    -------
    pd.DataFrame
        DataFrame with OHLCV data
    """
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['date', 'ticker']).reset_index(drop=True)
    
    return df


def validate_data_integrity(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Check for common data quality issues.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame

    Returns
    -------
    Tuple[bool, List[str]]
        (is_valid, list_of_issues)
    """
    issues = []
    
    # Check required columns
    required_cols = {'date', 'ticker', 'open', 'high', 'low', 'close', 'volume'}
    if not required_cols.issubset(df.columns):
        issues.append(f"Missing columns: {required_cols - set(df.columns)}")
    
    # Check for null prices
    price_cols = ['open', 'high', 'low', 'close']
    null_counts = df[price_cols].isnull().sum()
    if null_counts.sum() > 0:
        issues.append(f"Null prices found: {null_counts[null_counts > 0].to_dict()}")
    
    # Check date ordering
    if not df['date'].is_monotonic_increasing:
        issues.append("Dates not monotonically increasing")
    
    # Check OHLC relationships
    invalid_ohlc = (df['high'] < df['low']).sum()
    if invalid_ohlc > 0:
        issues.append(f"Invalid OHLC: {invalid_ohlc} rows where high < low")
    
    return len(issues) == 0, issues


def get_asset_universe(df: pd.DataFrame) -> List[str]:
    """Get sorted list of unique tickers in dataset."""
    return sorted(df['ticker'].unique().tolist())


def get_date_range(df: pd.DataFrame) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Get min and max dates in dataset."""
    return df['date'].min(), df['date'].max()
