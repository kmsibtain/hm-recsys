# System Architecture

## Overview

The H&M Recommender System is a production-grade two-stage recommender architecture (Candidate Retrieval $\to$ LightGBM LambdaRank) with feature store integration, automated skew verification, real-time A/B routing, and continuous drift monitoring.

```mermaid
flowchart TD
    subgraph Data_Pipeline["Data & Features"]
        CSV["Raw CSVs (transactions, articles, customers)"] --> Polars["Polars / Parquet Conversion & 10% Sampling"]
        Polars --> Parquet["Parquet Artifacts (Split R & V)"]
        Parquet --> FeastOffline["Feast Offline Store (Parquet)"]
        FeastOffline --> FeastRedis["Feast Online Store (Redis)"]
    end

    subgraph Candidate_Retrieval["Multi-Source Candidate Generation"]
        Parquet --> C1["Repurchase (Recency weighted)"]
        Parquet --> C2["Recent Popularity (7d, 14d, age bucket)"]
        Parquet --> C3["Product Code Variants"]
        Parquet --> C4["Item-Item Co-purchase (14-day window)"]
        Parquet --> C5["ALS Matrix Factorization (CPU fallback)"]
        Parquet --> C6["Two-Tower Embeddings (Kaggle GPU) + FAISS"]
        C1 & C2 & C3 & C4 & C5 & C6 --> Dedup["Merge & Deduplicate (Top 100-200 Candidates)"]
    end

    subgraph Ranking["Ranking Stage"]
        Dedup --> RankDataset["Candidate-Feature Matrix Builder"]
        FeastRedis -.->|Online Feature Retrieval| RankDataset
        RankDataset --> LightGBM["LightGBM LambdaRank (eval_at=[12])"]
        LightGBM --> MLflow["MLflow Model Registry (champion / challenger)"]
    end

    subgraph Serving_Layer["Serving & Experimentation"]
        UserReq["User Request: /recommend"] --> FastAPI["FastAPI Service"]
        FastAPI --> Router["Deterministic A/B Router (SHA256 Hash)"]
        Router -->|Arm A (Champion)| ModelA["LightGBM Champion"]
        Router -->|Arm B (Challenger)| ModelB["LightGBM Challenger"]
        ModelA & ModelB --> Top12["Top 12 Recommendations"]
        Top12 --> RedisStreams["Redis Streams (Exposure & Interaction Logging)"]
    end

    subgraph Observability["Monitoring & MLOps"]
        FastAPI --> Prometheus["Prometheus Metrics (/metrics)"]
        RedisStreams --> Simulator["Replay Simulator & Stat Evaluation"]
        Prometheus --> Grafana["Grafana Dashboards & Alerts"]
        Parquet --> Evidently["Evidently AI Drift Monitoring (HTML & Exporter)"]
        Evidently --> RetrainTrigger["Drift-Triggered Retraining Workflow"]
    end
```

## Non-Negotiables & Guarantees
1. **Time Leakage Contract**: Features for target week $T$ strictly use data before $T_{\text{start}}$.
2. **Training-Serving Equality**: Shared feature transformation logic between ranking training and online FastAPI serving; validated by automated skew tests (`tests/test_skew.py`).
3. **Hardware Efficiency**: Designed for local CPU execution with tight memory limits (< 6 GB Compose stack).

