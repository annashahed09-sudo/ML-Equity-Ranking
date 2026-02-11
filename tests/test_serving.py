import numpy as np
import pandas as pd
import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from src.serving import app


def _rows(n_days=120, n_tickers=3):
    rng = np.random.default_rng(11)
    dates = pd.date_range('2022-01-01', periods=n_days)
    rows = []
    for t in range(n_tickers):
        ticker = f'T{t:02d}'
        close = 100 + np.cumsum(rng.normal(0, 1, n_days))
        for i, d in enumerate(dates):
            rows.append(
                {
                    'date': str(d.date()),
                    'ticker': ticker,
                    'open': float(close[i] * 0.995),
                    'high': float(close[i] * 1.01),
                    'low': float(close[i] * 0.99),
                    'close': float(close[i]),
                    'volume': float(1_000_000 + i),
                }
            )
    return rows


def test_health():
    client = TestClient(app)
    r = client.get('/health')
    assert r.status_code == 200
    assert r.json()['status'] == 'ok'


def test_predict_from_rows():
    client = TestClient(app)
    payload = {
        'rows': _rows(),
        'model_type': 'ridge',
        'reviews_by_ticker': {'T00': ['Strong growth and bullish guidance.']},
    }
    r = client.post('/predict_from_rows', json=payload)
    assert r.status_code == 200
    body = r.json()
    assert 'ranking' in body and 'report' in body
    assert len(body['ranking']) > 0


def test_predict_from_rows_invalid_model_type():
    client = TestClient(app)
    payload = {'rows': _rows(), 'model_type': 'not_a_model'}
    r = client.post('/predict_from_rows', json=payload)
    assert r.status_code == 400
    assert 'Unknown model_type' in r.json()['detail']
