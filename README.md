# Cross-Sectional Equity Return Prediction with Machine Learning

## Overview

This project is now a **usable market-intelligence tool** with:

- advanced cross-sectional ML ranking models,
- quantum-inspired and accelerated computing paths,
- NLP review analysis,
- a deployable API,
- and a polished interactive dashboard.

It is designed for research and decision-support (not guaranteed market prediction).

---

## What’s included

### Advanced ML
- Ridge, Gradient Boosting, Random Forest, HistGradientBoosting, Neural MLP
- Quantum-inspired model (RFF + Ridge)
- Advanced stacked ensemble

### Accelerated computing
- NumPy default backend
- Optional CuPy GPU backend
- Numba JIT accelerated routines
- Optional explicit CUDA kernel path for score normalization

### NLP + market intelligence
- Financial sentiment analyzer (TF-IDF + Logistic Regression + lexicon fallback)
- Review summarization and confidence scoring
- Unified ticker report: rank, score, expected direction, review sentiment

### Product interfaces
- **FastAPI service** (`src/serving.py`)
- **Streamlit dashboard** (`dashboard.py`)
- **CLI tool** (`src/cli.py`)
- launch helper script (`launch.sh`)

---

## Install

```bash
pip install -r requirements.txt
```

---

## Run tests

```bash
pytest tests/ -q
```

---

## Launch as a tool

### 1) Dashboard (interactive + aesthetics)

```bash
./launch.sh dashboard
```

Open: `http://localhost:8501`

### 2) API service

```bash
./launch.sh api
```

Open: `http://localhost:8000/docs`

### 3) CLI

```bash
python -m src.cli --tickers AAPL,MSFT,NVDA,AMZN --start 2021-01-01 --end 2024-12-31 --model advanced_ensemble
```

---

## API quick example

`POST /predict_from_tickers`

```json
{
  "tickers": ["AAPL", "MSFT", "NVDA", "AMZN"],
  "start_date": "2021-01-01",
  "end_date": "2024-12-31",
  "model_type": "advanced_ensemble",
  "reviews_by_ticker": {
    "AAPL": ["Strong growth and earnings beat."]
  }
}
```

---

## Important legitimacy notes

- This tool provides model-based rankings and sentiment context; it is **not financial advice**.
- No system can predict markets with perfect accuracy.
- Always apply risk controls, diversification, and independent validation.
- Best use: research workflow, scenario analysis, and signal triage.
