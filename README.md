# Cross-Sectional Equity Return Prediction with Machine Learning

## Overview

This project examines whether machine learning models can extract **useful cross-sectional signals** from equity market data to predict **relative next-period performance**. Rather than forecasting absolute returns, the focus is on ranking assets and evaluating whether those rankings translate into economically meaningful long–short portfolios.

The project prioritizes **sound problem formulation, statistical discipline, and realistic validation** over headline performance.

---

## Motivation

Financial returns are noisy, weakly predictable, and non-stationary. While absolute return prediction is unreliable, empirical finance suggests that **relative relationships across assets**—such as momentum, volatility, and trend effects—can be more stable and actionable.

This project asks whether machine learning can capture such cross-sectional structure under realistic assumptions.

---

## Methodology

* **Target**: Cross-sectional ranking of next-period equity returns
* **Features**: Momentum, volatility, and trend-distance indicators
* **Normalization**: Cross-sectional standardization at each time step
* **Models**:

  * Ridge regression as a linear baseline
  * Gradient boosting to capture nonlinear effects

Model complexity is deliberately constrained to emphasize robustness and interpretability.

---

## Validation and Evaluation

Models are evaluated using **walk-forward validation** to respect the temporal structure of financial data. Randomized cross-validation is avoided.

Key evaluation metrics include:

* Information Coefficient (rank correlation)
* Stability of predictive signal over time
* Performance of simple long–short portfolios, before and after transaction costs

---

## Portfolio Construction

At each rebalance, assets are ranked by model score. The top quantile is held long and the bottom quantile short, forming a simple, market-neutral portfolio directly tied to the model’s ranking output.

---

## Limitations

This is a research and educational project:

* Uses public historical data and simplified execution assumptions
* Does not model intraday dynamics or microstructure
* Results are not intended to represent deployable trading strategies

---

## Purpose

The goal of this project is to demonstrate a careful application of machine learning in a domain where signal is weak, assumptions matter, and validation is critical.

---

## Disclaimer

This project is for educational and research purposes only and does not constitute investment advice.
