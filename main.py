"""
Unified entry point for the Quantitative Equity Research Platform.

Supports four modes:
    main.py api         Start the FastAPI research API server
    main.py dashboard   Launch the Streamlit institutional dashboard
    main.py research    Run the research pipeline with synthetic or live data
    main.py test        Run the full test suite

Usage:
    python main.py api            # API at http://localhost:8000
    python main.py dashboard      # Dashboard at http://localhost:8501
    python main.py research       # Demo with synthetic data
    python main.py research --sp500 --start 2020-01-01  # Live S&P 500
    python main.py test           # Run all tests
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger("main")


def run_api(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Start the FastAPI research API server."""
    import uvicorn
    logger.info(f"Starting API server on {host}:{port}")
    uvicorn.run(
        "api.app:app",
        host=host,
        port=port,
        reload=True,
        log_level="info",
    )


def run_dashboard() -> None:
    """Start the Streamlit institutional dashboard."""
    import subprocess
    dashboard_path = PROJECT_ROOT / "dashboard" / "app.py"
    logger.info(f"Starting dashboard: {dashboard_path}")
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", str(dashboard_path),
        "--server.port", "8501",
        "--server.address", "0.0.0.0",
    ])


def run_research(mode: str = "demo", start: str = "2020-01-01",
                 end: str = "2024-01-01") -> None:
    """Run the research pipeline."""
    if mode == "demo":
        logger.info("Running demo with synthetic data...")
        from research.run import run_demo
        run_demo()
    elif mode == "sp500":
        logger.info(f"Running with S&P 500 data from {start} to {end}...")
        from research.run import run_sp500
        run_sp500(start, end)


def run_tests(verbose: bool = True, target: str = "tests/") -> int:
    """Run the test suite."""
    import pytest
    args = [target]
    if verbose:
        args.extend(["-v", "--tb=short"])
    logger.info(f"Running tests: {' '.join(args)}")
    return pytest.main(args)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quantitative Equity Research Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python main.py api                        Start API server
  python main.py dashboard                  Launch dashboard
  python main.py research                    Demo pipeline
  python main.py research --sp500           S&P 500 pipeline
  python main.py test                       Run tests
  python main.py test --target tests/test_factors.py  Specific test
        """,
    )

    parser.add_argument(
        "command",
        choices=["api", "dashboard", "research", "test"],
        help="Command to execute",
    )
    parser.add_argument("--host", default="0.0.0.0", help="API server host")
    parser.add_argument("--port", type=int, default=8000, help="API server port")
    parser.add_argument("--start", default="2020-01-01", help="Start date for research")
    parser.add_argument("--end", default="2024-01-01", help="End date for research")
    parser.add_argument("--sp500", action="store_true", help="Use S&P 500 data")
    parser.add_argument("--target", default="tests/", help="Test target path")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if args.command == "api":
        run_api(host=args.host, port=args.port)
    elif args.command == "dashboard":
        run_dashboard()
    elif args.command == "research":
        mode = "sp500" if args.sp500 else "demo"
        run_research(mode=mode, start=args.start, end=args.end)
    elif args.command == "test":
        sys.exit(run_tests(verbose=args.verbose, target=args.target))


if __name__ == "__main__":
    main()
