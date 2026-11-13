# Experimental Results and Benchmarks

> **Rule**: No fabricated numbers. All metrics reported here are produced by actual runs executed in this workspace or tracked in MLflow.

## 1. Baselines (Validation Week V)

Evaluated on 6,837 active customers who made at least one purchase in Validation Week V (`2020-09-16` to `2020-09-22`), using features and history strictly $\le$ `2020-09-15`. Logged in MLflow experiment `hm-recsys-baselines`.

| Method | MAP@12 (V) | Notes |
| :--- | :--- | :--- |
| **Repurchase + Popularity fallback** | **0.022248** | Best baseline; user repeat purchases heavily dominate early rank slots |
| **Global Top-12 (Last 7d)** | **0.007848** | Baseline popularity heuristic (`2020-09-09` to `2020-09-15`) |
| **Popularity per Age Bucket** | **0.007686** | 7d sales stratified by age bins (`<20`, `20-29`, `30-39`, `40-49`, `50-59`, `60+`) |

## 2. Candidate Retrieval: Recall@K (Validation Week V)

Measured on the sampled dataset with source dates through `2020-09-15`. Macro recall averages distinct held-out item recall over every validation buyer, including cold and zero-hit customers.

| Generator | K | Recall@K | Incremental recall |
| :--- | ---: | ---: | ---: |
| repurchase | 30 | 0.041914 | 0.041914 |
| popularity | 200 | 0.126781 | 0.118527 |
| variants | 30 | 0.027043 | 0.017531 |
| copurchase | 100 | 0.036209 | 0.009477 |
| als | 50 | 0.037883 | 0.014846 |
| two_tower | 50 | 0.032601 | 0.008616 |
| union | 200 | 0.187523 | — |

Incremental recall uses an **uncapped** cumulative union in table order, so it measures unique added coverage. The final union is capped at 200 using reciprocal-rank fusion (constant 60); its recall may be lower. Popularity combines six top-50 lists: 7/14-day global, age-bucket and preferred-channel popularity, fused to 200. Co-purchase takes up to 20 neighbors for each of the last five distinct items. All generators permit previously purchased items. The union retains per-source scores and ranks.

Machine: Windows 11, Intel Core i7-7600U, four logical CPUs; DuckDB capped at 4 GB and four threads, ALS 64 factors / 15 iterations. Artifacts: `artifacts/candidates/val/`, including `evaluation.json`, `evaluation.parquet`, source Parquets and `union.parquet`.

All six generators were evaluated locally on V, including FAISS retrieval from the human-run Colab exports. Notebook internal Recall@100 is a separate metric and is not substituted for this table. See [the Phase 2 acceptance audit](phase2_acceptance.md).

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

