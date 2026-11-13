# H&M Two-Stage Recommender System

A production-grade, two-stage recommender system (multi-source candidate retrieval + LightGBM LambdaRank) built on the H&M Personalized Fashion Recommendations dataset.

Engineered with full MLOps lifecycle: Feast feature store, training-serving skew testing, MLflow model registry, deterministic A/B routing, Evidently drift monitoring, Prometheus/Grafana observability, and a local Docker Compose stack.

---

## Quickstart & Execution Guide

### Windows (PowerShell)

```powershell
# 1. Ingest raw CSVs and generate sampled Parquet files (10% customer sample)
python data/ingest.py

# 2. Run Exploratory Data Analysis & Candidate Ceiling evaluation
python data/eda.py

# 3. Run unit tests
python -m pytest tests/

# 4. Compute and evaluate Phase 1 Baseline Models (MAP@12)
python -m ranking.baselines

# 5. Launch the local MLflow Tracking UI
python -m mlflow ui --backend-store-uri sqlite:///artifacts/mlflow.db
```
> After launching the MLflow UI, navigate to **`http://localhost:5000`** in your browser to view experiment runs and metrics.

### Linux / WSL / macOS (Makefile)

```bash
make data       # Ingest data and create Parquet sample
make eda        # Run EDA and compute retrieval ceilings
make test       # Run unit test suite
make baselines  # Evaluate baselines and log to MLflow
make mlflow-ui  # Start MLflow UI
```

### Phase 2 candidate generation (Windows 11)

Use the existing virtual environment directly; activation is optional:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev,modeling]"
.\.venv\Scripts\python.exe -m candidates.pipeline --split val
.\.venv\Scripts\python.exe -m candidates.report
.\.venv\Scripts\python.exe -m candidates.pipeline --split train
.\.venv\Scripts\python.exe -m pytest tests -p no:cacheprovider
```

With GNU Make installed, `make candidates` and `make candidates-train` use
`.venv/Scripts/python.exe` on Windows. Each split writes source Parquets, an ALS
model, a deduplicated top-200 union with source scores/ranks, and measured recall
under `artifacts/candidates/{split}/`. See [results](docs/results.md).

Phase 2 acceptance passed: all six generators and the union were evaluated on V.
The merged pool achieves **18.75% Recall@200**; see the
[acceptance audit](docs/phase2_acceptance.md) and [full results](docs/results.md).
Both Google Colab cutoff exports were returned and validated. For reruns, use the
[Colab runbook](docs/colab_runbook.md). This project uses offline purchase data;
later online experiments are replay simulations.

---

## Phase 1 Baseline Results (Validation Week V)

Evaluated on **6,837 active customers** in Validation Week V (`2020-09-16` to `2020-09-22`) using historical interactions strictly $\le$ `2020-09-15` (no future leakage). Logged in MLflow under experiment `hm-recsys-baselines`:

| Baseline Model | MAP@12 (V) | Notes |
| :--- | :---: | :--- |
| **Repurchase + Global Fallback** | **0.022248** | **Best Baseline**: User repeat purchases heavily dominate early rank slots |
| **Global Top-12 (Last 7 Days)** | **0.007848** | Popularity heuristic over trailing 7 days (`2020-09-09` to `2020-09-15`) |
| **Popularity per Age Bucket** | **0.007686** | 7-day sales stratified by customer age demographics |

---

## Documentation Links

- [H&M Two-Stage Recommender Agent Handover Plan.md](file:///d:/Projects/hm-recsys/H&M%20Two-Stage%20Recommender%20Agent%20Handover%20Plan.md)
- [System Architecture](file:///d:/Projects/hm-recsys/docs/architecture.md)
- [Architecture Decisions (ADR-01 to ADR-06)](file:///d:/Projects/hm-recsys/docs/decisions.md)
- [Data Card](file:///d:/Projects/hm-recsys/docs/data_card.md)
- [Operational Runbook](file:///d:/Projects/hm-recsys/docs/runbook.md)
- [Kaggle GPU Training Runbook](file:///d:/Projects/hm-recsys/docs/kaggle_runbook.md)
- [Experimental Results](file:///d:/Projects/hm-recsys/docs/results.md)
