.PHONY: lint fix fmt check test install dev clean

# Lint with ruff
lint:
	uv run ruff check src tests examples scripts

# Auto-fix lint issues
fix:
	uv run ruff check src tests examples scripts --fix

# Format code
fmt:
	uv run ruff format src tests examples scripts

# Check lint + format (CI)
check:
	uv run ruff check src tests examples scripts
	uv run ruff format --check src tests examples scripts

# Run tests
test:
	uv run pytest 

# Install in dev mode
install:
	uv pip install -e ".[dev]"

# Install and setup pre-commit hooks
dev: install
	uv run pre-commit install

# Clean build artifacts
clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
