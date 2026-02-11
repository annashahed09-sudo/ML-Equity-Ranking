"""CLI entrypoint for market intelligence tooling."""

from __future__ import annotations

import argparse
from typing import List

from .market_intelligence import MarketIntelligenceService


def _parse_tickers(raw: str) -> List[str]:
    return [t.strip().upper() for t in raw.split(",") if t.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="ML Equity Intelligence CLI")
    parser.add_argument("--tickers", required=True, help="Comma-separated tickers, e.g. AAPL,MSFT,NVDA")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument(
        "--model",
        default="advanced_ensemble",
        choices=["advanced_ensemble", "quantum_inspired", "ridge", "random_forest", "histgb", "mlp"],
    )

    args = parser.parse_args()
    tickers = _parse_tickers(args.tickers)
    if len(tickers) < 2:
        raise SystemExit("Provide at least 2 tickers.")

    service = MarketIntelligenceService()
    feat = service.prepare_data(tickers, args.start, args.end)
    ranking = service.rank_tickers(feat, model_type=args.model)
    report = service.build_market_report(ranking)

    print("\n=== Ranking ===")
    print(ranking[["rank", "ticker", "model_score"]].to_string(index=False))
    print("\n=== Market report ===")
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
