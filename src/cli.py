"""CLI entrypoint for market intelligence tooling."""

from __future__ import annotations

import argparse
from typing import List

from .market_intelligence import MarketIntelligenceService
from .sp500 import run_sp500_simulation


def _parse_tickers(raw: str) -> List[str]:
    return [t.strip().upper() for t in raw.split(",") if t.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="ML Equity Intelligence CLI")
    parser.add_argument("--tickers", help="Comma-separated tickers, e.g. AAPL,MSFT,NVDA")
    parser.add_argument("--sp500", action="store_true", help="Run an S&P 500 simulation instead of custom tickers")
    parser.add_argument("--sp500-limit", type=int, default=25, help="Number of S&P 500 constituents to simulate")
    parser.add_argument("--offline-sp500", action="store_true", help="Use bundled fallback S&P 500 sample")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument(
        "--model",
        default="advanced_ensemble",
        choices=["advanced_ensemble", "quantum_inspired", "ridge", "random_forest", "histgb", "mlp"],
    )

    args = parser.parse_args()

    if args.sp500:
        simulation = run_sp500_simulation(
            start_date=args.start,
            end_date=args.end,
            model_type=args.model,
            limit=args.sp500_limit,
            use_live_wikipedia=not args.offline_sp500,
        )
        print("\n=== S&P 500 Simulation Ranking ===")
        print(simulation.ranking[["rank", "ticker", "model_score"]].to_string(index=False))
        print("\n=== Portfolio Summary ===")
        for key, value in simulation.portfolio_summary.items():
            print(f"{key}: {value:.6f}")
        return

    if not args.tickers:
        raise SystemExit("Provide --tickers or use --sp500.")

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
