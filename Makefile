.PHONY: help install install-dev test lint format clean build

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install the package
	pip install -e .

install-dev: ## Install the package with development dependencies
	pip install -e ".[dev]"

test: ## Run tests with pytest
	pytest tests/ -v

test-env: ## Run tests with pytest in specific environment (usage: make test-env-pytest ENV=myenv)
	conda run --no-capture-output -n $(ENV) pytest tests/ -v

clean: ## Clean build, test, coverage, cache and temporary artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf .eggs/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf .hypothesis/
	rm -rf .tox/
	rm -rf .nox/
	rm -rf htmlcov/
	rm -rf pip-wheel-metadata/
	rm -rf wheelhouse/
	rm -rf .coverage
	rm -rf .coverage.*
	rm -rf coverage.xml
	rm -rf junit.xml
	find . -type d -name ".pytest_cache" -prune -exec rm -rf {} +
	find . -type d -name "*.egg-info" -prune -exec rm -rf {} +
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type d -name ".ipynb_checkpoints" -prune -exec rm -rf {} +
	find . -type f \( -name "*.pyc" -o -name "*.pyo" -o -name "*$$py.class" \) -delete

build: ## Build the package
	python -m build
