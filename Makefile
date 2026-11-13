.PHONY: help setup lint format test data eda baselines mlflow-ui clean

ifeq ($(OS),Windows_NT)
PYTHON ?= .venv/Scripts/python.exe
else
PYTHON ?= .venv/bin/python
endif

help:
	@echo "Available commands:"
	@echo "  make setup      Install development dependencies"
	@echo "  make lint       Check code style and lint with ruff"
	@echo "  make format     Format code with ruff"
	@echo "  make test       Run test suite with pytest"
	@echo "  make data       Convert raw CSVs and generate sampled parquet datasets"
	@echo "  make eda        Execute exploratory data analysis and candidate ceilings"
	@echo "  make baselines  Compute and log baseline MAP@12 metrics to MLflow"
	@echo "  make candidates Generate validation candidates and update recall results"
	@echo "  make candidates-train Generate ranker-training-week candidates"
	@echo "  make kaggle-input Export both cutoffs for the GPU handoff"
	@echo "  make mlflow-ui  Start local MLflow UI dashboard"

setup:
	$(PYTHON) -m pip install -e ".[dev,modeling]"

lint:
	$(PYTHON) -m ruff check .

format:
	$(PYTHON) -m ruff format .
	$(PYTHON) -m ruff check --fix .

test:
	$(PYTHON) -m pytest tests/

data:
	$(PYTHON) data/ingest.py

eda:
	$(PYTHON) data/eda.py

baselines:
	$(PYTHON) -m ranking.baselines

.PHONY: candidates candidates-train kaggle-input
candidates:
	$(PYTHON) -m candidates.pipeline --split val
	$(PYTHON) -m candidates.report

candidates-train:
	$(PYTHON) -m candidates.pipeline --split train

kaggle-input:
	$(PYTHON) -m candidates.prepare_kaggle --split train
	$(PYTHON) -m candidates.prepare_kaggle --split val
	$(PYTHON) -m candidates.prepare_kaggle --bundle

mlflow-ui:
	$(PYTHON) -m mlflow ui --backend-store-uri sqlite:///artifacts/mlflow.db

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +

