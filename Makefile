.PHONY: help install install-dev test lint format clean build

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install the package
	pip install -e .

install-dev: ## Install the package with development dependencies
	pip install -e ".[dev]"

test: ## Run tests
	conda run -n cosmo python -m unittest discover tests/ -v

test-cov: ## Run tests with coverage
	conda run -n cosmo pytest --cov=CosmoFuse --cov-report=html --cov-report=term-missing

test-pytest: ## Run tests with pytest
	conda run -n cosmo pytest tests/ -v

test-env: ## Run tests in specific environment (usage: make test-env ENV=myenv)
	conda run -n $(ENV) python -m unittest discover tests/ -v

clean: ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

build: ## Build the package
	python -m build
