.PHONY: help setup test lint format typecheck run backtest report dashboard api precommit clean

PYTHON ?= python
START ?= 2022-01-01
END ?= 2023-01-01
MODEL ?= advanced_ensemble
SP500_LIMIT ?= 25

help:
	@echo "Available targets:"
	@echo "  setup      Install runtime + dev dependencies and pre-commit hooks"
	@echo "  test       Run the test suite with coverage"
	@echo "  lint       Run ruff + isort + black checks (no changes)"
	@echo "  format     Auto-format with isort + black"
	@echo "  typecheck  Run mypy on src/"
	@echo "  run        Run an end-to-end demo pipeline on synthetic data"
	@echo "  backtest   Run an S&P 500 walk-forward backtest (fallback universe)"
	@echo "  report     Run a backtest and write a PDF report to reports/"
	@echo "  dashboard  Launch the Streamlit research dashboard"
	@echo "  api        Launch the FastAPI service"
	@echo "  precommit  Run all pre-commit hooks on all files"

setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev]"
	pre-commit install

test:
	$(PYTHON) -m pytest --cov=src --cov-report=term-missing

lint:
	ruff check src tests
	isort --check-only src tests
	black --check src tests

format:
	isort src tests
	black src tests

typecheck:
	mypy src

run:
	$(PYTHON) run_all.py

backtest:
	$(PYTHON) -m src.cli --sp500 --fallback-sp500 --sp500-limit $(SP500_LIMIT) \
		--model $(MODEL) --start $(START) --end $(END)

report:
	mkdir -p reports
	$(PYTHON) -m src.cli --sp500 --fallback-sp500 --sp500-limit $(SP500_LIMIT) \
		--model $(MODEL) --start $(START) --end $(END) \
		--pdf-report reports/simulation_report.pdf

dashboard:
	./launch.sh dashboard

api:
	./launch.sh api

precommit:
	pre-commit run --all-files

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
