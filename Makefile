.PHONY: help setup download-data eda train tune evaluate explain serve ui docker-build docker-run test lint format clean

PYTHON := python
PIP := pip
PROJECT_ROOT := $(shell pwd)

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

setup: ## Install dependencies and create directories
	$(PIP) install -r requirements.txt
	mkdir -p data/raw data/processed data/external data/sample_payloads
	mkdir -p models/registry
	mkdir -p docs/figures
	mkdir -p logs
	mkdir -p mlruns
	mkdir -p optuna_studies
	@echo "Setup complete. Download data with: make download-data"

download-data: ## Download dataset from Kaggle (requires kaggle CLI configured)
	kaggle datasets download -d mlg-ulb/creditcardfraud -p data/raw --unzip
	@echo "Dataset downloaded to data/raw/creditcard.csv"

split-data: ## Run data splitting (time-based or stratified per config)
	$(PYTHON) -m src.data.split

eda: ## Run EDA analysis and generate plots
	$(PYTHON) -m src.eda.run_eda

train-baselines: ## Train baseline models and imbalance comparison
	$(PYTHON) -m src.models.baselines

tune: ## Run Optuna hyperparameter tuning
	$(PYTHON) -m src.models.tune

evaluate: ## Run full evaluation pipeline
	$(PYTHON) -m src.evaluation.evaluate

explain: ## Generate SHAP explanations
	$(PYTHON) -m src.explain.generate

save-artifacts: ## Save final model artifacts
	$(PYTHON) -m src.artifacts.save

serve: ## Start FastAPI server
	uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload

ui: ## Start Streamlit UI
	streamlit run app/ui/streamlit_app.py --server.port 7860

docker-build: ## Build Docker image
	docker build -t fraud-detection:latest .

docker-run: ## Run Docker container
	docker run -p 7860:7860 -p 8000:8000 fraud-detection:latest

test: ## Run test suite
	$(PYTHON) -m pytest tests/ -v --tb=short --cov=src --cov=app --cov-report=term-missing

test-unit: ## Run unit tests only
	$(PYTHON) -m pytest tests/unit/ -v --tb=short

test-integration: ## Run integration tests only
	$(PYTHON) -m pytest tests/integration/ -v --tb=short

lint: ## Run linters
	$(PYTHON) -m ruff check src/ app/ tests/
	$(PYTHON) -m mypy src/ app/ --ignore-missing-imports

format: ## Format code with black and ruff
	$(PYTHON) -m black src/ app/ tests/
	$(PYTHON) -m ruff check --fix src/ app/ tests/

clean: ## Remove artifacts, caches, and generated files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf htmlcov .coverage coverage.xml
	rm -rf mlruns mlflow.db
	rm -rf optuna_studies
	rm -rf data/processed/*
	rm -rf logs/*
	@echo "Cleaned all generated artifacts."

mlflow-ui: ## Start MLflow tracking UI
	mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

all: setup split-data train-baselines tune evaluate explain save-artifacts ## Run full pipeline
	@echo "Full pipeline complete."
