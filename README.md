# Cross-Sectional Equity Return Prediction with Machine Learning

## Overview

This project examines whether machine learning models can extract **useful cross-sectional signals** from equity market data to predict **relative next-period performance**. Rather than forecasting absolute returns, the focus is on ranking assets and evaluating whether those rankings translate into economically meaningful long–short portfolios.

The project prioritizes **sound problem formulation, statistical discipline, and realistic validation** over headline performance.

---

## What is implemented

- Modular data loading (`yfinance` and CSV)
- Feature engineering for momentum, volatility, and trend-distance
- Walk-forward time-series validation
- Model layer with:
  - Ridge baseline
  - Gradient boosting regressor
  - **Quantum-inspired regressor** using random Fourier feature mapping + ridge
- **Accelerated computing abstraction**:
  - NumPy backend by default
  - Optional CuPy backend when CUDA is available
- Evaluation metrics:
  - Information Coefficient (IC)
  - Long-short returns with turnover-aware transaction costs
  - Portfolio performance summary
- End-to-end walk-forward training pipeline in `src/pipeline.py`
- Unit tests across all major modules

---

## Methodology

- **Target**: Cross-sectional ranking of next-period equity returns
- **Features**: Momentum, volatility, and trend-distance indicators
- **Normalization**: Cross-sectional standardization at each time step
- **Validation**: Walk-forward folds only (no random CV)
- **Portfolio rule**: Long top quantile, short bottom quantile per rebalance date

---

## Quickstart

```bash
pip install -r requirements.txt
pytest tests/ -q
python run_all.py
```

`run_all.py` executes tests and then runs a synthetic-data demonstration of the full walk-forward pipeline with the quantum-inspired model.

---

## Programmatic usage

```python
from src.pipeline import run_walk_forward_pipeline
from src.data_loader import load_yfinance_data

raw = load_yfinance_data(
    tickers=["AAPL", "MSFT", "NVDA", "AMZN"],
    start_date="2019-01-01",
    end_date="2024-01-01",
)

result = run_walk_forward_pipeline(
    raw_df=raw,
    model_type="quantum_inspired",
    n_splits=5,
    model_kwargs={"n_components": 256, "alpha": 0.5},
)

print(result.fold_metrics)
print(result.portfolio_summary)
```

---

## Limitations

This is a research and educational project:

- Uses public historical data and simplified execution assumptions
- Does not model intraday dynamics or microstructure
- Results are not intended to represent deployable trading strategies

---

## Disclaimer

This project is for educational and research purposes only and does not constitute investment advice.
