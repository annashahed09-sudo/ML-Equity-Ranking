# Cross-Sectional Equity Return Prediction with Machine Learning

## Overview

This project builds a production-style research framework for **cross-sectional stock ranking**, combining:

- advanced ML models (linear, tree-based, neural, stacking ensemble)
- quantum-inspired kernel approximation
- accelerated computing paths (NumPy, optional CuPy, optional Numba/CUDA)
- NLP-driven review sentiment for qualitative context

It predicts **relative signal strength** (ranking) rather than guaranteed absolute returns.

---

## Implemented capabilities

### 1) Advanced ML model stack

- Ridge baseline
- Gradient boosting
- Random forest
- Histogram gradient boosting
- Neural MLP regressor
- Quantum-inspired model (RFF + Ridge)
- Advanced stacked ensemble (`advanced_ensemble`)

### 2) Accelerated computing

- NumPy default execution
- CuPy auto-switch when GPU/CUDA environment is available
- Numba JIT acceleration for critical numerical routines
- Optional Numba CUDA kernel path for score normalization

### 3) NLP / review intelligence

- Financial sentiment analyzer with trainable TF-IDF + Logistic Regression
- Lexicon fallback for zero-shot review scoring
- Review summarization (top themes)
- Unified ticker report with model direction + review sentiment

### 4) Validation and portfolio analytics

- Walk-forward validation (time-safe)
- Information Coefficient (IC)
- Long-short backtest with transaction costs and turnover
- Model leaderboard benchmarking across model families

---

## Quickstart

```bash
pip install -r requirements.txt
pytest tests/ -q
python run_all.py
```

---

## Example: model pipeline

```python
from src.pipeline import run_walk_forward_pipeline
from src.data_loader import load_yfinance_data

raw = load_yfinance_data(
    tickers=["AAPL", "MSFT", "NVDA", "AMZN", "META"],
    start_date="2019-01-01",
    end_date="2024-12-31",
)

result = run_walk_forward_pipeline(
    raw_df=raw,
    model_type="advanced_ensemble",
    n_splits=5,
    model_kwargs={"prefer_gpu": True, "prefer_numba": True},
)

print(result.fold_metrics)
print(result.portfolio_summary)
```

---

## Example: high-level market report with reviews

```python
from src.market_intelligence import MarketIntelligenceService

service = MarketIntelligenceService()
df = service.prepare_data(["AAPL", "MSFT", "NVDA"], "2021-01-01", "2024-12-31")
ranking = service.rank_tickers(df, model_type="advanced_ensemble")

reviews = {
    "AAPL": ["Strong product momentum and earnings beat."],
    "MSFT": ["Cloud growth remains healthy but valuation risk persists."],
}
report = service.build_market_report(ranking, reviews)
print(report)
```

---

## Important limitations

- No model can predict markets with perfect accuracy.
- Performance can degrade under regime shifts and macro shocks.
- This framework is for research/education and not investment advice.
- Always perform additional risk controls and out-of-sample validation.
