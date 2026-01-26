import pandas as pd
from src.portfolio import run_portfolio_backtest

def test_portfolio_backtest():
    df = pd.DataFrame({
        'date': [1, 1, 1, 2, 2, 2],
        'ticker': ['A', 'B', 'C', 'A', 'B', 'C'],
        'model_score': [1, 2, 3, 2, 1, 3],
        'forward_return': [0.1, 0.2, 0.3, 0.2, 0.1, 0.3]
    })
    returns, summary = run_portfolio_backtest(df)
    assert 'gross_return' in returns.columns
    assert 'mean_gross_return' in summary
