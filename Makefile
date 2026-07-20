# ML-Equity-Ranking: Quantitative Equity Research Platform
# ================================================================

.PHONY: install install-dev test lint clean api research dashboard help

help:
	@echo "ML Equity Ranking - Quantitative Research Platform"
	@echo ""
	@echo "Usage:"
	@echo "  make install       Install production dependencies"
	@echo "  make install-dev   Install development dependencies"
	@echo "  make test          Run all tests"
	@echo "  make test-cov      Run tests with coverage report"
	@echo "  make lint          Run linters (flake8, black --check, isort --check)"
	@echo "  make format        Auto-format code (black, isort)"
	@echo "  make api           Start the FastAPI server"
	@echo "  make research      Run the research pipeline with synthetic data"
	@echo "  make dashboard     Start the Streamlit dashboard"
	@echo "  make clean         Clean build artifacts"

# Dependencies
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install pytest pytest-cov black isort flake8

# Testing
test:
	python -m pytest tests/ -v --tb=short

test-cov:
	python -m pytest tests/ -v --tb=short --cov=. --cov-report=term --cov-report=html

# Linting
lint:
	flake8 config/ core/ data/ factors/ models/ validation/ risk/ portfolio/ signal_processing/ nlp/ news/ explainability/ research/ api/
	black --check config/ core/ data/ factors/ models/ validation/ risk/ portfolio/ signal_processing/
	isort --check-only config/ core/ data/ factors/ models/ validation/ risk/ portfolio/ signal_processing/

format:
	black config/ core/ data/ factors/ models/ validation/ risk/ portfolio/ signal_processing/ nlp/ news/ explainability/ research/ api/
	isort config/ core/ data/ factors/ models/ validation/ risk/ portfolio/ signal_processing/ nlp/ news/ explainability/ research/ api/

# Services
api:
	uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload

dashboard:
	streamlit run dashboard/app.py --server.port 8501 --server.address 0.0.0.0

# Research
research:
	python -m research.run --mode demo

# Cleanup
clean:
	rm -rf .pytest_cache/
	rm -rf __pycache__/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
