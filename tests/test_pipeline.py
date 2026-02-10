import numpy as np
import pandas as pd

from src.pipeline import run_walk_forward_pipeline


def _synthetic_data(n_days=140, n_tickers=6):
    rng = np.random.default_rng(7)
    dates = pd.date_range('2021-01-01', periods=n_days)
    rows = []
    for t in range(n_tickers):
        ticker = f'T{t:02d}'
        close = 100 + np.cumsum(rng.normal(0, 1, n_days))
        for i, d in enumerate(dates):
            rows.append(
                {
                    'date': d,
                    'ticker': ticker,
                    'open': close[i] * 0.995,
                    'high': close[i] * 1.01,
                    'low': close[i] * 0.99,
                    'close': close[i],
                    'volume': 1_000_000 + i,
                }
            )
    return pd.DataFrame(rows).sort_values(['date', 'ticker']).reset_index(drop=True)


def test_run_walk_forward_pipeline_quantum_inspired():
    df = _synthetic_data()
    result = run_walk_forward_pipeline(
        df,
        model_type='quantum_inspired',
        n_splits=3,
        test_size=120,
        min_train_size=240,
        model_kwargs={'n_components': 32, 'prefer_gpu': False},
    )

    assert not result.predictions.empty
    assert set(['date', 'ticker', 'forward_return', 'model_score', 'fold_id']).issubset(result.predictions.columns)
    assert len(result.fold_metrics) > 0
    assert 'mean_net_return' in result.portfolio_summary
