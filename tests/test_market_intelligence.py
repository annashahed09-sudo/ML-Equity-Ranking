import numpy as np
import pandas as pd

from src.market_intelligence import MarketIntelligenceService


def _synthetic_data(n_days=160, n_tickers=5):
    rng = np.random.default_rng(9)
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


def test_market_report_from_ranking_and_reviews():
    service = MarketIntelligenceService()
    ranking = pd.DataFrame(
        {
            'ticker': ['A', 'B'],
            'rank': [1, 2],
            'model_score': [0.8, -0.2],
        }
    )
    reviews = {
        'A': ['Strong growth and bullish guidance.'],
        'B': ['Weak results and downgrade risk.'],
    }
    report = service.build_market_report(ranking, reviews)
    assert set(['ticker', 'rank', 'expected_direction', 'review_sentiment']).issubset(report.columns)


def test_rank_tickers_advanced_ensemble():
    service = MarketIntelligenceService()
    df = _synthetic_data()
    from src.features import compute_features, compute_forward_returns

    feat = compute_features(df)
    feat = compute_forward_returns(feat).dropna().reset_index(drop=True)

    ranked = service.rank_tickers(feat, model_type='advanced_ensemble', model_kwargs={'prefer_gpu': False, 'prefer_numba': False})
    assert not ranked.empty
    assert set(['ticker', 'model_score', 'rank']).issubset(ranked.columns)
