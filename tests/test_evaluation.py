import pandas as pd
import numpy as np
from src.evaluation import information_coefficient, compute_ic_by_date, long_short_portfolio_returns, summarize_portfolio_performance

def test_ic():
    scores = pd.Series([1, 2, 3, 4])
    rets = pd.Series([0.1, 0.2, 0.3, 0.4])
    ic = information_coefficient(scores, rets)
    assert abs(ic - 1.0) < 1e-6

def test_ic_by_date():
    df = pd.DataFrame({
        'date': [1, 1, 2, 2],
        'model_score': [1, 2, 1, 2],
        'forward_return': [0.1, 0.2, 0.2, 0.1]
    })
    ic = compute_ic_by_date(df)
    assert len(ic) == 2

def test_long_short_returns():
    df = pd.DataFrame({
        'date': [1, 1, 1, 2, 2, 2],
        'ticker': ['A', 'B', 'C', 'A', 'B', 'C'],
        'model_score': [1, 2, 3, 2, 1, 3],
        'forward_return': [0.1, 0.2, 0.3, 0.2, 0.1, 0.3]
    })
    returns = long_short_portfolio_returns(df)
    assert 'gross_return' in returns.columns
    summary = summarize_portfolio_performance(returns)
    assert 'mean_gross_return' in summary

# Initial full project structure for ML-Equity-Ranking
#
# - Modular src/ with data loading, feature engineering, models, validation, evaluation, and portfolio logic
# - All code follows cross-sectional, walk-forward, and normalization methodology
# - Unit tests for all major modules in tests/
# - requirements.txt for reproducible dependencies
# - .github/copilot-instructions.md for AI agent guidance
# - Ready for end-to-end experimentation and extension
