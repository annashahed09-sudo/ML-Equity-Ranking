# Contributing to ML-Equity-Ranking

## Welcome

Thank you for considering contributing to ML-Equity-Ranking. This project aims to be an institutional-grade quantitative equity research platform, and contributions from the community help make it better.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/annashahed09-sudo/ML-Equity-Ranking.git
cd ML-Equity-Ranking

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest pytest-cov black isort flake8 pre-commit mypy

# Install pre-commit hooks
pre-commit install
```

## Code Standards

### Python

- **Type hints**: All functions must have complete type annotations
- **Docstrings**: NumPy-style docstrings for all public APIs
- **Formatting**: Black (line length 88), isort (compatible settings)
- **Linting**: flake8 with reasonable limits (max-line-length=100)
- **Naming**: `snake_case` for functions/variables, `PascalCase` for classes, `UPPER_CASE` for constants

### Testing

- All new modules must have corresponding test files in `tests/`
- Tests must be deterministic (seed all random operations)
- Use pytest fixtures for shared test data
- Aim for >80% coverage on new code

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=term

# Run specific test file
pytest tests/test_factors.py -v
```

### Quantitative Code

- All mathematical implementations must cite academic references in docstrings
- Algorithms must be validated against known test cases or literature benchmarks
- Numerical stability is critical: use `np.errstate` guards, log-sum-exp tricks, etc.
- All assumptions must be documented (e.g., "assumes daily returns are log-normal")

## Pull Request Process

1. Create a feature branch from `Development`: `git checkout -b feature/my-feature`
2. Make your changes with clear commit messages
3. Run the full test suite: `pytest tests/ -v`
4. Run linters: `make lint`
5. Update documentation if needed
6. Open a PR against the `Development` branch
7. Ensure CI passes (all tests, linting, type checking)

## Architecture Guidelines

### Module Structure

```
module_name/
    __init__.py      # Public API exports
    base.py          # Abstract base classes (if applicable)
    implementation.py # Concrete implementations
```

### Dependency Rules

- `core/` has zero internal dependencies
- `config/` depends only on `core/`
- `data/` depends on `config/` and `core/`
- No circular imports allowed
- Optional dependencies should be guarded with `try/except ImportError`

## Adding New Factors

1. Create a new class inheriting from `FactorComputer` in the appropriate submodule
2. Implement `compute(self, df, **kwargs) -> FactorResult`
3. Register in the `FactorCatalog.create_default()` method
4. Add tests in `tests/test_factors.py`
5. Document the academic reference in the class docstring

## Adding New Models

1. Create a new class inheriting from `BaseModel`
2. Implement `_fit(self, X, y)` and `_predict(self, X)`
3. Register in `ModelFactory` via `_MODEL_REGISTRY`
4. Add tests in `tests/test_models.py`

## Questions?

Open an issue on GitHub for discussions, bug reports, or feature requests.
