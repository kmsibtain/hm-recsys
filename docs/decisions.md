# Architecture and Tool Decisions (ADR)

Single source of truth for architectural choices, hardware constraints, tool justifications, and deviations.

## Hardware Baseline
- Phase 2 detected on Windows 11: Intel Core i7-7600U @ 2.80 GHz, **4 logical CPUs** via Python/Windows registry. Physical-core CIM query was denied by the sandbox; no physical-core claim is inferred. Use native `.venv\Scripts\python.exe`, not WSL. The WSL memory setting does not apply.
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

## Phase 2 retrieval decisions

- All behavioral retrieval reads one cutoff-filtered `history` table, shared by every local generator. Active articles are restricted using only that history, never sales after cutoff. Generate for all sampled customer IDs; target-week purchases select the evaluation cohort only. The inherited sample itself was selected over the full configured window, as required by the original plan.
- DuckDB uses four threads / 4 GB. Merge source scores and ranks in batches of 2,000 customer indices: the initial full aggregation caused severe paging and was stopped before producing metrics. Do not run the serving stack concurrently.
- Repurchase uses inverse-square-root days-since-purchase weighting, up to 30 distinct items. Popularity uses six top-50 lists (7/14-day global, age and preferred historical channel). Unknown/cold users receive global and applicable age popularity; a channel is not invented for customers without history.
- Variants require a matching product code and a different colour, restricted to pre-cutoff-active articles. Up to 30 are scored by reciprocal rank of historical items. Without this source, alternate colours of a familiar product are missed.
- Co-purchase counts distinct customers buying a pair within 14 days (inclusive), normalized by the geometric mean of distinct-customer item frequencies. Multi-unit buys cannot inflate counts. Keep top 20 neighbors per item, apply to the last five distinct purchases and aggregate up to 100 candidates. Without temporal restriction, unrelated distant purchases create noisy associations.
- `implicit` CPU ALS: 64 factors, 15 iterations, seed 42, regularization 0.05 and log-scaled purchase counts. Retrieve top 50, including already-liked items because repeat purchases are legitimate. Restrict retrieval to active items. Save model alongside candidate artifacts. FAISS CPU uses normalized vectors with `IndexFlatIP`; it is used only when cutoff-specific GPU exports are available.
- Reciprocal-rank fusion (constant 60) avoids comparing incompatible raw ALS/count/co-purchase scores. Keep source lists, score maps and rank maps, deduplicate pairs, and cap at 200 with deterministic article-index ties. A catalogue with fewer eligible articles can produce fewer than 100 candidates; no fabricated padding is used.
- Report macro recall over distinct held-out purchases, including cold users and zero-hit users. Incremental recall uses the uncapped cumulative union; the final capped union is measured separately. This separates added coverage from the effect of truncation.
- Two-tower GPU execution is a human action. Both cutoff exports and a reviewable notebook are prepared. The human completed both cutoffs in Google Colab; returned vectors and V retrieval passed Phase 2 checks. Local vectors are batch-precomputed, never inferred live.
- The human executed GPU training in Colab; local checks cover the returned artifacts and retrieval. It evaluates the last pre-cutoff week, then refits all allowed history; text vectors are computed in Colab with MiniLM. It uses age/club attributes and five item category embeddings in addition to ID embeddings, pooled prior history and projected text. A training example excludes purchases on its own day from user history.

## Decisions Derived from Exploratory Data Analysis (EDA)

The following design decisions were established directly from empirical findings on the 10% sampled 12-month dataset:

| # | Empirical EDA Finding | Impacted Component | Confirmed Modeling Decision |
| :--- | :--- | :--- | :--- |
| **ADR-01** | **51.8% of articles never sold** in the 12-month window (54,650 out of 105,542 items). | Candidate Retrieval & FAISS Index | **Prune inactive catalog items**: Only index and retrieve articles active in the trailing 12 months. Do not pollute candidate slots with dead stock. |
| **ADR-02** | **7-day popularity beats 14-day** (Recall: 6.41% vs 5.98%). Article median circulation is 89 days. | Candidate Generator (`recent_popularity`) | **Prioritize short-term recency**: Use 7-day sales window as primary popularity generator with age-bucket stratification. |
| **ADR-03** | **Repurchase rate is 13.53%**; **Product-code variant repeat rate is 22.12%**. | Candidate Retrieval (`repurchase`, `variants`) | **Mandate dual repurchase generators**: Keep distinct historical items (recency-weighted) AND color variants of purchased styles. |
| **ADR-04** | **11.31% of Week V buyers are completely cold**; **52.01% had zero purchases in preceding 30 days**. | Serving & Cold-Start Fallback | **Age-stratified popularity fallback**: Customers with no purchase history in 30 days automatically receive age-bucketed popular items. |
| **ADR-05** | **Earlier EDA heuristic setup achieved 11.52% recall** (Repurchase + Pop 7d + Variants + Co-purchase combined). | Two-Tower Embeddings & ALS (`implicit`) | **Measure latent retrieval contribution**: 11.52% was specific to that EDA setup and candidate budget, not a hard ceiling. Phase 2 compares ALS and two-tower against the expanded heuristic union. |
| **ADR-06** | **9.84% of rows are multi-unit buys** on the same day (`same customer, article, t_dat`). | Ranker Dataset Builder | **Binary ranking formulation**: Aggregate multi-unit transactions to binary positive labels ($y \in \{0, 1\}$) grouped by customer for LightGBM LambdaRank. |



## GPU handoff update: Google Colab

The human selected Google Colab instead of Kaggle. The same notebook now uploads a prepared ZIP using Colab file helpers, discovers both cutoffs, trains them sequentially, and downloads one ZIP. Input exports retain their existing local `artifacts/kaggle_input/` paths for compatibility; `artifacts/colab/two_tower_inputs.zip` is the upload bundle. Both cutoff exports have been returned and verified; see the Phase 2 acceptance audit.

## Phase 2 measured retrieval contribution

The human-run Colab exports passed exact local cutoff/ID/shape checks. Two-tower adds 0.8616 percentage points of uncapped recall and raises the final top-200 union from 18.5547% to 18.7523%. This supports retaining it as a measured complementary source; it does not establish ranking lift. Full V candidate integrity checks and 28 tests passed. See [the acceptance audit](phase2_acceptance.md).
