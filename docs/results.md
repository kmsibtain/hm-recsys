# Experimental Results and Benchmarks

> **Rule**: No fabricated numbers. All metrics reported here are produced by actual runs executed in this workspace or tracked in MLflow.

## 1. Baselines (Validation Week V)

| Method | MAP@12 (V) | Notes |
| :--- | :--- | :--- |
| Global Top-12 (Last 7d) | - | Fallback popularity |
| Repurchase + Popularity fallback | - | User repeat purchase heuristic |
| Age-bucket Popularity | - | Demographic popularity |

## 2. Candidate Retrieval: Recall@K (Validation Week V)

| Generator | K | Recall@K | Incremental Recall | Notes |
| :--- | :--- | :--- | :--- | :--- |
| Repurchase | 30 | - | - | Recency-weighted repeat buys |
| Recent Popularity | 50 | - | - | Global & age-bucketed |
| Product Code Variants | 30 | - | - | Color variants of bought products |
| Item-Item Co-purchase | 30 | - | - | 14-day co-occurrence matrix |
| ALS (`implicit`) | 50 | - | - | Latent factor collaborative filtering |
| Two-Tower (FAISS) | 50 | - | - | Dual-encoder semantic embeddings |
| **Union (Merged & Deduped)** | **100–200** | - | - | Combined candidate pool |

## 3. Ranker Performance (LightGBM LambdaRank)

| Model / Feature Configuration | MAP@12 (V) | Lift vs Best Baseline |
| :--- | :--- | :--- |
| Best Baseline | - | - |
| Best Single Generator Ordering | - | - |
| Full LightGBM Ranker (`champion`) | - | - |

### Feature Group Ablation
| Configuration | MAP@12 (V) | Delta |
| :--- | :--- | :--- |
| Full Model | - | Ref |
| w/o Customer Features | - | - |
| w/o Article Features | - | - |
| w/o Interaction Features | - | - |
| w/o Text Embeddings (PCA) | - | - |

## 4. A/B Replay Simulation Experiment

| Arm | Description | Hit Rate | MAP@12 | p-value (z-test) | 95% Bootstrap CI Lift |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Arm A (Champion) | Full LightGBM Ranker | - | - | - | - |
| Arm B (Challenger) | Challenger Model | - | - | - | - |
| Arm B (Bad Model Demo) | Degraded Ranker (shuffled labels) | - | - | - | - |

## 5. Serving Latency & System Benchmark

- **Machine**: Intel i7 7th Gen, 16 GB RAM, Local CPU
- **Tool**: Locust load test

| Concurrency / RPS | p50 (ms) | p95 (ms) | p99 (ms) | Stage Breakdown (p95) |
| :--- | :--- | :--- | :--- | :--- |
| 10 RPS | - | - | - | Candidates: - ms, Feast: - ms, Ranker: - ms |
| 20 RPS | - | - | - | Candidates: - ms, Feast: - ms, Ranker: - ms |
| Peak Sustainable RPS | - | - | - | - |

