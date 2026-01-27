# ============================================================================
# georeader Makefile
# ============================================================================
# Dependencies are managed via Poetry with optional groups:
#   - dev:      Development tools (pytest, mypy, pre-commit, tox)
#   - docs:     Documentation tools (mkdocs, mkdocstrings, jupyter)
#   - tutorial: Additional packages for running tutorial notebooks
#
# Quick start:
#   make install      # Install core + dev dependencies
#   make install-all  # Install everything (dev + docs + tutorial)
#   make test         # Run tests
#   make docs         # Serve documentation locally
# ============================================================================

.PHONY: install
install: ## Install core + dev dependencies
	@echo "🚀 Installing core and dev dependencies with Poetry"
	@poetry install --with dev

.PHONY: install-docs
install-docs: ## Install docs dependencies (for building documentation)
	@echo "📚 Installing docs dependencies"
	@poetry install --with docs

.PHONY: install-tutorial
install-tutorial: ## Install tutorial dependencies (for running notebooks)
	@echo "📓 Installing tutorial dependencies"
	@poetry install --with tutorial

.PHONY: install-all
install-all: ## Install all dependency groups (dev + docs + tutorial)
	@echo "🚀 Installing all dependencies"
	@poetry install --with dev,docs,tutorial

.PHONY: check
check: ## Run code quality tools (type check)
	@echo "🚀 Checking Poetry lock file consistency with 'pyproject.toml'"
	@poetry check --lock
	@echo "🚀 Static type checking: Running mypy"
	@poetry run mypy

.PHONY: typecheck
typecheck: ## Run type checking only (mypy)
	@echo "🔍 Running mypy"
	@poetry run mypy

.PHONY: test
test: ## Run tests with pytest
	@echo "🧪 Running tests"
	@poetry run pytest tests/ -v

.PHONY: test-cov
test-cov: ## Run tests with coverage report
	@echo "🧪 Running tests with coverage"
	@poetry run pytest tests/ -v --cov=georeader --cov-report=term-missing --cov-report=html

.PHONY: build
build: clean-build ## Build wheel file using poetry
	@echo "📦 Building wheel file"
	@poetry build

.PHONY: clean-build
clean-build: ## Clean build artifacts
	@echo "🧹 Cleaning build artifacts"
	@rm -rf dist build *.egg-info

.PHONY: clean
clean: clean-build ## Clean all generated files
	@echo "🧹 Cleaning all generated files"
	@rm -rf site htmlcov .pytest_cache .mypy_cache .coverage
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true

.PHONY: publish
publish: ## Publish release to PyPI
	@echo "🚀 Publishing: Dry run"
	@poetry config pypi-token.pypi $(PYPI_TOKEN)
	@poetry publish --dry-run
	@echo "🚀 Publishing to PyPI"
	@poetry publish

.PHONY: build-and-publish
build-and-publish: build publish ## Build and publish to PyPI

.PHONY: docs-test
docs-test: ## Test if documentation builds without errors (strict mode)
	@echo "📚 Testing documentation build"
	@poetry run mkdocs build -s

.PHONY: docs
docs: ## Build and serve documentation locally (http://127.0.0.1:8000)
	@echo "📚 Serving documentation at http://127.0.0.1:8000"
	@poetry run mkdocs serve

.PHONY: docs-build
docs-build: ## Build documentation to site/ directory
	@echo "📚 Building documentation"
	@poetry run mkdocs build

.PHONY: docs-publish
docs-publish: ## Build and publish documentation to GitHub Pages
	@echo "📚 Publishing documentation to GitHub Pages"
	@poetry run mkdocs build
	@poetry run ghp-import -n -p -f site

.PHONY: update
update: ## Update dependencies to latest versions
	@echo "⬆️  Updating dependencies"
	@poetry update

.PHONY: lock
lock: ## Regenerate poetry.lock file
	@echo "🔒 Regenerating lock file"
	@poetry lock

.PHONY: shell
shell: ## Activate poetry virtual environment shell
	@poetry shell

.PHONY: help
help:
	@echo "georeader-spaceml development commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help