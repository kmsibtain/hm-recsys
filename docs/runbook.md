# Operational Runbook

Standard operating procedures for managing the local production stack, handling alerts, and performing operational maintenance.

## 1. Quick Start
```bash
make setup
make data
make train
docker compose up -d
```

## 2. Model Rollback Procedure
If the challenger or champion experiences high error rates or degradation detected by the A/B test or Evidently:
1. Check current active model aliases in MLflow:
   ```bash
   python -m serving.model_registry list-aliases
   ```
2. Revert the `champion` alias to the previous known good version:
   ```bash
   python -m serving.model_registry set-alias --model hm-ranker --version <KNOWN_GOOD_VERSION> --alias champion
   ```
3. Restart or reload API workers:
   ```bash
   curl -X POST http://localhost:8000/model/reload
   ```

## 3. Redis Flush & Feature Re-Materialization
If Redis cache is corrupted or out of sync with offline Parquet:
1. Flush Redis feature cache:
   ```bash
   docker compose exec redis redis-cli FLUSHDB
   ```
2. Re-materialize Feast features from offline Parquet:
   ```bash
   cd feature_repo
   feast materialize 2019-09-23T00:00:00 2020-09-16T00:00:00
   ```

## 4. Drift Alert Response (Evidently + Prometheus)
When `recsys_drift_share` exceeds the alert threshold:
1. Inspect the latest Evidently drift report under `artifacts/reports/drift_report.html`.
2. Determine which features drifted (e.g. seasonal price or category shifts).
3. Trigger drift-mitigation retraining pipeline:
   ```bash
   make retrain
   ```

## 5. Routine Retraining
Run the automated retraining workflow:
```bash
make retrain
```
This target rebuilds candidate generation, refreshes offline features, trains a challenger LightGBM ranker, logs to MLflow, and executes the A/B evaluation against the current champion.

