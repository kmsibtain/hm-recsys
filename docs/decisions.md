# Architecture and Tool Decisions (ADR)

Single source of truth for architectural choices, hardware constraints, tool justifications, and deviations.

## Hardware Baseline
- **CPU**: Intel Core i7 7th Gen (assume 4 cores / 8 threads)
- **RAM**: 16 GB (WSL2 capped at 8–10 GB if running via WSL2)
- **GPU**: None (Local CPU only; Kaggle/Colab for GPU two-tower training)
- **Constraint**: Peak compose stack target < 6 GB RAM. Never run model training and the full serving stack simultaneously.

## Component Justifications

| Component | Choice | Why it exists | What breaks without it |
| :--- | :--- | :--- | :--- |
| **Data Engine** | Polars / DuckDB + Parquet | Efficient memory-mapped queries, lazy evaluation | Loading 31M CSV rows in pandas crashes with Out-Of-Memory (OOM) |
| **Candidate Generators** | Multi-source (Repurchase, Popularity, Co-purchase, Product Code Variants, ALS, Two-Tower) | High candidate recall across known and cold users | A single retrieval method misses either repetitive purchases, trending items, or latent preferences |
| **Ranker** | LightGBM LambdaRank | Fast CPU training, handles heterogeneous tabular & text features, directly optimizes NDCG/MAP | Pointwise classifiers fail to rank relevant items to the top-12 |
| **Feature Store** | Feast (Redis online, Parquet offline) | Low-latency feature retrieval, unified entity schema, point-in-time joins | Training-serving skew and slow real-time joins in FastAPI |
| **Model Registry** | MLflow | Model versioning, alias management (`champion`, `challenger`), reproducibility | Untracked weights, ambiguous production deployments, manual file management |
| **Serving API** | FastAPI + Uvicorn | Async request handling, Pydantic contracts, Prometheus instrumentation | Inflexible model serving, lack of standard endpoints (`/recommend`, `/health`, `/metrics`) |
| **Event Logging** | Redis Streams | Fast in-memory append-only log for replay exposures | Disk bottleneck or heavy Kafka infrastructure overhead |
| **Monitoring** | Prometheus + Grafana + Evidently | Automated drift detection (data/prediction), latency alerts, dashboard visualization | Undetected concept/data drift and silent production degradation |
| **CI/CD** | GitHub Actions | Automated linting, unit tests, skew tests on fixture data | Regressions and leakage creeping into main |

## Decisions Derived from Exploratory Data Analysis (EDA)

The following design decisions were established directly from empirical findings on the 10% sampled 12-month dataset:

| # | Empirical EDA Finding | Impacted Component | Confirmed Modeling Decision |
| :--- | :--- | :--- | :--- |
| **ADR-01** | **51.8% of articles never sold** in the 12-month window (54,650 out of 105,542 items). | Candidate Retrieval & FAISS Index | **Prune inactive catalog items**: Only index and retrieve articles active in the trailing 12 months. Do not pollute candidate slots with dead stock. |
| **ADR-02** | **7-day popularity beats 14-day** (Recall: 6.41% vs 5.98%). Article median circulation is 89 days. | Candidate Generator (`recent_popularity`) | **Prioritize short-term recency**: Use 7-day sales window as primary popularity generator with age-bucket stratification. |
| **ADR-03** | **Repurchase rate is 13.53%**; **Product-code variant repeat rate is 22.12%**. | Candidate Retrieval (`repurchase`, `variants`) | **Mandate dual repurchase generators**: Keep distinct historical items (recency-weighted) AND color variants of purchased styles. |
| **ADR-04** | **11.31% of Week V buyers are completely cold**; **52.01% had zero purchases in preceding 30 days**. | Serving & Cold-Start Fallback | **Age-stratified popularity fallback**: Customers with no purchase history in 30 days automatically receive age-bucketed popular items. |
| **ADR-05** | **Heuristic retrieval ceiling is 11.52%** (Repurchase + Pop 7d + Variants + Co-purchase combined). | Two-Tower Embeddings & ALS (`implicit`) | **Latent semantic retrieval is essential**: Rule-based generators cap at 11.52% recall; collaborative filtering and neural Two-Tower are required to uncover non-obvious affinities. |
| **ADR-06** | **9.84% of rows are multi-unit buys** on the same day (`same customer, article, t_dat`). | Ranker Dataset Builder | **Binary ranking formulation**: Aggregate multi-unit transactions to binary positive labels ($y \in \{0, 1\}$) grouped by customer for LightGBM LambdaRank. |


