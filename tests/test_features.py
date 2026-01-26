import pandas as pd
import numpy as np
from src.features import compute_rolling_momentum, compute_rolling_volatility, compute_trend_distance, compute_features, compute_forward_returns

def test_momentum():
    df = pd.DataFrame({'close': np.arange(1, 11)})
    mom = compute_rolling_momentum(df, window=2)
    assert np.allclose(mom.iloc[2], (3-1)/1)

def test_volatility():
    df = pd.DataFrame({'close': np.arange(1, 11)})
    vol = compute_rolling_volatility(df, window=2)
    assert vol.isnull().sum() > 0

def test_trend_distance():
    df = pd.DataFrame({'close': np.arange(1, 11)})
    trend = compute_trend_distance(df, window=2)
    assert trend.isnull().sum() > 0

def test_compute_features():
    df = pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=10),
        'ticker': ['A']*10,
        'open': np.arange(1, 11),
        'high': np.arange(1, 11),
        'low': np.arange(1, 11),
        'close': np.arange(1, 11),
        'volume': np.arange(1, 11)
    })
    feats = compute_features(df)
    assert 'momentum_5_mom' in feats.columns
    assert 'volatility_vol' in feats.columns
    assert 'trend_distance_trend' in feats.columns

def test_forward_returns():
    df = pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=10),
        'ticker': ['A']*10,
        'close': np.arange(1, 11)
    })
    out = compute_forward_returns(df, forward_periods=1)
    assert 'forward_return' in out.columns
