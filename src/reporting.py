"""PDF report generation for simulations and market-intelligence evidence."""

from __future__ import annotations

from pathlib import Path
from textwrap import wrap
from typing import Iterable, Optional

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from .compliance import combined_disclaimer
from .news import NewsEvidence, build_evidence_narrative
from .sp500 import SP500SimulationResult


def _write_lines(ax, lines: Iterable[str], y_start: float = 0.95, line_height: float = 0.04, fontsize: int = 10) -> None:
    y = y_start
    for line in lines:
        for wrapped in wrap(str(line), width=105) or [""]:
            ax.text(0.03, y, wrapped, fontsize=fontsize, va="top", family="monospace")
            y -= line_height
            if y < 0.05:
                return


def build_prediction_reasoning(result: SP500SimulationResult, max_tickers: int = 10) -> list[str]:
    """Explain why the model produced the ranking using available diagnostic evidence."""
    lines = [
        "Prediction reasoning (research explanation, not financial advice):",
        "1. The model ranks tickers cross-sectionally by engineered momentum, volatility, and trend-distance features.",
        "2. Walk-forward validation is used so training data precedes test windows, reducing look-ahead bias.",
        "3. Higher model_score values indicate stronger relative ranking in the latest simulated rebalance window.",
        "4. Portfolio metrics summarize how long-top / short-bottom ranks behaved in the historical simulation.",
        "",
        "Top ranked tickers:",
    ]
    for _, row in result.ranking.head(max_tickers).iterrows():
        lines.append(f"- Rank {int(row['rank'])}: {row['ticker']} with model_score={float(row['model_score']):.4f}")
    return lines


def generate_pdf_report(
    result: SP500SimulationResult,
    output_path: str | Path,
    evidence: Optional[list[NewsEvidence]] = None,
) -> Path:
    """Generate a PDF report with rankings, reasoning, evidence, and disclaimers."""
    output = Path(output_path)
    evidence = evidence or []

    with PdfPages(output) as pdf:
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        summary_lines = [
            "ML Equity Intelligence - S&P 500 Simulation Report",
            "",
            f"Universe size: {len(result.universe)}",
            f"Top ticker: {result.ranking.iloc[0]['ticker'] if not result.ranking.empty else 'N/A'}",
            "",
            "Portfolio summary:",
        ]
        for key, value in result.portfolio_summary.items():
            summary_lines.append(f"- {key}: {float(value):.6f}")
        _write_lines(ax, summary_lines)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        _write_lines(ax, build_prediction_reasoning(result), fontsize=9)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        evidence_lines = ["External evidence (NYT / Economist RSS source links):", ""]
        evidence_lines.extend(build_evidence_narrative(evidence).splitlines())
        _write_lines(ax, evidence_lines, fontsize=8)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        _write_lines(ax, ["Privacy and regulatory notices:", "", *combined_disclaimer().splitlines()], fontsize=8)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    return output
