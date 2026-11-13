# H&M Two-Stage Recommender: Agent Handover Plan

Owner: Cypher (Muhammad Sibtain). Purpose: a production-grade, CV-ready ML engineering project. This document is the single source of truth for the building agent.

## 1. Mission and non-negotiables

Build a two-stage recommender (multi-source candidate generation, then LightGBM LambdaRank) on the H&M Personalized Fashion Recommendations dataset, wrapped in real MLOps: a feature store, a training-serving skew test, a model registry, A/B routing with significance testing, drift monitoring, dashboards, CI, and a load test.

Rules the agent must follow:

- **No fabricated numbers.** Every metric in the README comes from a run the agent actually executed. If something was not run, say so.
- **No image data. Ever.** Use only `transactions_train.csv`, `customers.csv`, `articles.csv`. Do not download the images folder (about 30 GB).
- **Be honest about simulation.** The dataset has purchases only, no clicks or impressions. All 'online' traffic is an offline replay simulator. Label it that way everywhere.
- **Time-based splits only.** Never split randomly. Any feature, candidate or embedding used for a target week may only use data strictly before that week starts.
- **Every tool must earn its place.** In `docs/decisions.md`, record why each component exists and what breaks without it.
- **Work in phases.** Do not start phase N+1 until phase N's acceptance criteria pass. Commit per phase with a clear message.
- **Re-runnable stages.** Each pipeline stage reads and writes Parquet under `artifacts/`, is idempotent, and is invoked by a `make` target.
- **Ask the human** (see section 12) rather than guessing when credentials, GPU notebooks or account actions are needed.

## 2. Hardware and budget constraints

- Laptop: Intel i7 7th gen, 16 GB RAM, **no GPU**. Assume 4 cores; check with `lscpu` or Task Manager and note the result in `docs/decisions.md`.
- No paid services. Allowed: Kaggle free account (about 30 GPU hours per week) and Google Colab free tier for GPU work only.
- Memory discipline: use Polars lazy scans or DuckDB over Parquet. Never load the full CSV into pandas. Convert CSV to Parquet once.
- Disk: keep 10 to 15 GB free. If on Windows, cap WSL2 memory at 8 to 10 GB in `.wslconfig`.
- Never run model training and the full serving stack at the same time.
- No public deployment. Deliverable is a one-command local stack (`docker compose up`), screenshots, an architecture diagram and a short demo video.

## 3. Tech stack (final)

| Concern | Choice | Notes |
| --- | --- | --- |
| Data | Polars, DuckDB, Parquet | CSV converted once |
| Candidates | pandas/Polars logic, `implicit` (ALS, CPU), FAISS (CPU), PyTorch two-tower (trained on Kaggle) | ALS is the local fallback |
| Ranker | LightGBM LambdaRank | CPU |
| Text features | sentence-transformers `all-MiniLM-L6-v2` on article text | 384-dim, cached; run on Kaggle if CPU is slow |
| Features | Feast (Redis online, Parquet offline) |  |
| Registry/tracking | MLflow (SQLite backend, local artifact store) |  |
| Serving | FastAPI + Uvicorn |  |
| Events | Redis Streams by default; optional Redpanda compose profile | Kafka is already on the CV, keep it light |
| Monitoring | Prometheus, Grafana, Evidently | dashboards provisioned as code |
| CI/CD | GitHub Actions | lint, tests, small-data smoke test, image build |
| Load test | Locust |  |
| Deploy | Docker Compose (primary); K3s manifests optional, last |  |

## 4. Dataset specification

Source: Kaggle competition `h-and-m-personalized-fashion-recommendations`. The human must join the competition and accept the rules, then provide a Kaggle API token. Download individual files only:

```
kaggle competitions download -c h-and-m-personalized-fashion-recommendations -f transactions_train.csv
kaggle competitions download -c h-and-m-personalized-fashion-recommendations -f customers.csv.zip
kaggle competitions download -c h-and-m-personalized-fashion-recommendations -f articles.csv
```

Files and key columns:

- `transactions_train.csv` (about 31M rows): `t_dat`, `customer_id`, `article_id`, `price`, `sales_channel_id`. Dates run from 2018-09-20 to 2020-09-22.
- `customers.csv` (about 1.4M rows): `customer_id`, `FN`, `Active`, `club_member_status`, `fashion_news_frequency`, `age`, `postal_code` (hashed).
- `articles.csv` (about 105k rows): `article_id`, `product_code`, `prod_name`, `product_type_name`, `product_group_name`, `graphical_appearance_name`, `colour_group_name`, `perceived_colour_value_name`, `perceived_colour_master_name`, `department_name`, `index_name`, `index_group_name`, `section_name`, `garment_group_name`, `detail_desc` (plus numeric code columns).

Note: `article_id` and `customer_id` must be read as strings (article ids have leading zeros; customer ids are long hashes). Map customer ids to int32 indices early to save memory.

### Sampling strategy (config-driven)

- Keep transactions from the last 12 months only (2019-09-23 onward). Make the window a config value.
- Sample customers, never rows: take a random 10% of customers with at least one transaction in the window (seed 42), and keep their full history inside the window. `SAMPLE_FRAC` is configurable; if the machine is struggling, drop to 5%.
- Keep all articles. Optionally restrict to articles sold in the window.
- Write the chosen config and resulting row counts to `docs/data_card.md`.

### Time split (the leakage contract)

Let the last data day be 2020-09-22.

- **Validation week (V):** 2020-09-16 to 2020-09-22. Final reported metrics.
- **Ranker-train week (R):** 2020-09-09 to 2020-09-15. Ranker labels come from this week.
- Candidates and features for R use data up to 2020-09-08. Candidates and features for V use data up to 2020-09-15.
- Optional extra ranker-train weeks (R-1, R-2) with the same pattern improve the ranker. Add them if time and RAM allow.
- A test must assert that `max(feature_source_date) < target_week_start` for every target week.

## 5. Repo structure

```
hm-recsys/
├── Makefile
├── pyproject.toml
├── configs/            # sampling, split dates, model params (YAML)
├── data/               # ingest, parquet conversion, sampling, split
├── candidates/         # one module per generator + recall eval
├── ranking/            # features, dataset builder, training, eval
├── feature_repo/       # Feast definitions + materialization
├── serving/            # FastAPI app, router, metrics, model loader
├── simulator/          # replay producer + exposure logger
├── experiments/        # assignment + analysis (stats)
├── monitoring/         # Evidently jobs, Prometheus rules, Grafana JSON
├── notebooks/          # Kaggle/Colab notebooks only
├── deploy/             # Dockerfiles, compose, optional k3s manifests
├── tests/
├── artifacts/          # gitignored: parquet, embeddings, models
└── docs/               # architecture, decisions, data_card, runbook, results
```

## 6. Phase plan (about 7 weeks part-time)

### Phase 1: Data and baselines (week 1)

Tasks:

1. Set up repo, `pyproject.toml`, pre-commit (ruff, black), Makefile, CI skeleton.
2. Download the three files. Convert to Parquet with correct dtypes. Apply sampling. Write the split definitions to config.
3. Implement `map_at_k` (k=12) from scratch. Competition definition: average precision at 12 per customer with denominator `min(len(actual), 12)`, then mean over customers. Evaluate on customers who have at least one purchase in the target week, and document this choice.
4. Baselines: (a) global top-12 by sales in the last 7 days before the target week; (b) repurchase (last 12 distinct items per customer, padded with popularity); (c) popularity per age bucket.

Acceptance: MAP@12 unit tests pass on hand-computed examples; baseline MAP@12 on V is logged to MLflow; `docs/data_card.md` exists.

### Phase 2: Candidate generation (week 2)

Build each generator as a function returning `(customer_idx, article_idx, source, score, rank)` Parquet. Generators:

1. **Repurchase:** distinct items the customer bought, recency-weighted (up to 30).
2. **Recent popularity:** global top 50 over the last 7 and 14 days, plus per age bucket and per `sales_channel_id`.
3. **Same product-code variants:** other colours of products the customer bought (`product_code` match).
4. **Item-item co-purchase:** items bought by the same customer within 14 days of each other, top 20 neighbours per item, scored by normalised co-count. Apply to the customer's last 5 items.
5. **ALS (`implicit`):** CPU, factors 64, trained on the pre-cutoff matrix. This is the local fallback and a legitimate generator.
6. **Two-tower + FAISS:** trained on Kaggle (section 8). Local side loads embeddings, builds a FAISS `IndexFlatIP` (or HNSW), retrieves top 50 per customer.

Merge and dedupe to 100 to 200 candidates per customer. Keep the `source` as a list so each source's contribution is traceable.

Acceptance: `docs/results.md` contains a table of recall@K per generator and for the union, on V. Include the incremental recall each generator adds. This table proves what each component earns.

**Status (2026-11-13): Phase 2 acceptance passed.** Both Colab cutoff exports were verified; all six generators and the capped union were measured on V, with incremental contributions recorded. All 28 tests and full V candidate integrity checks passed. See [the acceptance audit](docs/phase2_acceptance.md). Phase 3 has not started.

### Phase 3: Ranking (week 3)

Build the training set for week R (and optionally extra weeks): one row per (customer, candidate), label 1 if purchased in the target week, grouped by customer. Drop customers with no positive in candidates for training (keep all for evaluation).

Feature groups (all computed with data before the target week):

- Customer: age, club status, news frequency, purchase count (7/30/90 days), days since last purchase, mean and std of price paid, preferred channel, share of purchases by product group.
- Article: sales count (1/7/28 days), popularity trend (ratio of 7-day to 28-day), price, mean price vs its product type, categorical attributes (product type, colour, department, index, garment group) as LightGBM categoricals, article age (days since first sale), MiniLM text embedding reduced to about 16 dims by PCA.
- Interaction: times customer bought this article, days since customer last bought it, customer bought same product code, customer share of purchases in this product type, price relative to customer's average, candidate source flags and scores, rank within each source, co-purchase score, two-tower and ALS similarity.

Train LightGBM `lambdarank` with `eval_at=[12]`, `learning_rate=0.05`, `num_leaves=63`, `min_data_in_leaf=50`, `feature_fraction=0.8`, early stopping on V (rounds=50, up to 1000). Tune lightly with a documented small search.

Log to MLflow: params, MAP@12 (V), feature importance, the dataset hash and config. Register the model as `hm-ranker` with alias `champion`.

Acceptance: ranker MAP@12 on V clearly beats the best baseline and the best single generator ordering; an ablation table (drop each feature group) is in `docs/results.md`.

### Phase 4: Feature store and serving (week 4)

Feast:

- Entities: `customer`, `article`.
- Feature views: `customer_stats` and `article_stats` backed by Parquet, with event timestamps. Online store: Redis. Materialise the features as of the V cutoff.
- Serving reads customer and candidate features through `get_online_features`. Batch the candidate lookups.

FastAPI:

- `POST /recommend` with `{customer_id, k=12}` returns items, model version, arm and per-stage timings.
- `GET /health`, `GET /metrics` (Prometheus), `GET /model` (loaded version).
- Flow: load precomputed candidate sets and user vectors (artifact or Redis), fetch features, build the feature matrix with the same code used in training, score with LightGBM, return top k.
- Cold start: unknown customers or customers without history get recent-popularity candidates with the age bucket default.
- Share feature-building code between training and serving in one module. Do not duplicate it.

**Skew test (the key artifact):** `tests/test_skew.py` samples 200 customers and 200 articles, compares online features from Feast with offline features computed by the training pipeline for the same cutoff, and asserts equality within tolerance. Run in CI on a tiny fixture dataset, and locally on the real sample.

Acceptance: skew test passes; `/recommend` returns valid output for known and cold-start users; single-request p95 measured and logged.

### Phase 5: Experimentation (week 5)

- **Router:** deterministic assignment, `bucket = int(sha256(customer_id + experiment_salt)) % 100`; below the traffic percentage goes to arm B. Same customer always gets the same arm. Test this.
- **Arms:** A = champion; B = challenger model loaded by registry alias.
- **Simulator:** replays validation-week purchases in time order. For each customer-day with purchases, request recommendations as of the start of that day (features frozen at the cutoff, documented), log exposure `{customer, arm, items, ts}` to Redis Streams, and mark a **hit** if any purchased item that day is in the shown top 12.
- **Analysis:** per arm hit rate and MAP@12; two-proportion z-test; user-level bootstrap confidence interval on MAP@12 lift; a sample-size/power note.
- **Bad-model demo:** deploy a deliberately degraded challenger (for example, a ranker trained with shuffled labels, or popularity-only). Show the analysis flags it as significantly worse and the rollback script reverts the alias.

Acceptance: `docs/results.md` has the experiment table for a good challenger (a real improvement or an honest null result) and for the bad model.

### Phase 6: Monitoring and deployment (week 6)

Prometheus metrics (prefix `recsys_`):

- `request_latency_seconds` (histogram, label `stage`: candidates, features, ranking, total) and `request_total` (labels `arm`, `status`)
- `candidate_count` (histogram), `cold_start_total`, `feature_store_miss_total`
- `model_version_info` (gauge with labels)
- `exposure_hit_total` from the simulator

Evidently:

- Reference window vs current window feature drift (customer and article features, prediction score distribution). Simulate seasonality: reference = a September 2019 window, current = a December 2019 window (documented as a seasonal shift, not an incident).
- Output HTML reports to `artifacts/reports/` and export drift share as a Prometheus metric via a small exporter job.
- Drift-triggered retrain: a `make retrain` pipeline that rebuilds features, retrains the ranker, registers the challenger and runs the A/B analysis. Triggered by a GitHub Actions `workflow_dispatch` or a script that checks the drift metric. Do not add Airflow unless time remains.

Grafana: provision datasource and dashboards from JSON under `monitoring/grafana/`. Dashboards: serving (latency per stage, RPS, errors), model and experiment (arm hit rates, cold starts), drift (drift share, top drifted features). Alert rules: p95 total latency above threshold, error rate above 1%, drift share above threshold, hit rate drop vs baseline.

**Docker Compose** services and memory limits (`mem_limit`):

| Service | Limit |
| --- | --- |
| redis | 512m |
| mlflow | 512m |
| api (champion and challenger can be one process with two models loaded) | 1.5g |
| simulator (one-shot) | 768m |
| prometheus | 512m |
| grafana | 384m |
| evidently job (one-shot) | 2g |
| redpanda (optional profile) | 1g |

Target: whole stack under about 6 GB when running. Document the measured peak.

CI (GitHub Actions): ruff, pytest on a 1,000-customer fixture dataset committed to the repo, skew test on the fixture, MAP@12 smoke test, Docker build. No big data in CI.

Optional last: K3s manifests and an Argo/GitOps layer only if everything else is done.

### Phase 6b (optional): Streamlit demo page (1 to 2 days)

Do this only after phases 1 to 6 pass. It is a thin UI over the API, so a recruiter can see the two-stage design working in 30 seconds. It contains no model or feature code: it only calls the FastAPI service.

Placement and deployment:

- Code in `demo/app.py` (single file, plus `demo/requirements.txt` with `streamlit` and `httpx`). Add `streamlit` as an approved dependency.
- Compose service `demo`, port 8501, `mem_limit` 384m, `depends_on` the API. Reads the API URL from an env var (default `http://api:8000`).
- No images anywhere. Items render as text cards.

API additions needed in `serving/` (small):

- `GET /demo/customers` returns about 50 curated sample customers from the validation set, labelled by profile (heavy buyer, light buyer, one category loyalist, one cold-start 'new user'). Pick them with a fixed seed and document the rule in `docs/decisions.md`.
- `GET /demo/customers/{id}/history?n=10` returns the customer's last purchases (before the validation cutoff) with article metadata.
- `POST /recommend?explain=true` additionally returns, per item: `prod_name`, `product_type_name`, `colour_group_name`, `department_name`, `price`, candidate `sources` list, ranker score, and `purchased_in_val_week` (bool, from held-out data, used only by the demo).

Page layout:

1. **Banner:** 'Offline replay simulation on a sampled subset of the H&M dataset. Not live traffic.'
2. **Sidebar:** customer picker (curated list plus a free-text customer id), experiment mode (side by side A vs B by default, or a single arm), and `k` slider (default 12).
3. **Left column, 'Recent purchases':** the customer's last 10 items as text cards with date.
4. **Main area, 'Recommendations':** two columns, arm A and arm B, each with top-k cards showing name, type, colour, department, price, source badges (for example `repurchase`, `co-purchase`, `two-tower`, `popular`) and ranker score. Items marked `purchased_in_val_week` get a visible 'bought later' tag.
5. **Footer:** per-stage latency (candidates, features, ranking, total) for each arm, model version and arm name, plus a short note on what the cold-start path does when 'new user' is picked.

Implementation notes:

- Use `st.cache_data` for the customer list and histories. Never cache `/recommend` responses.
- Handle API errors with a clear message instead of a stack trace.
- Keep the file under about 250 lines; no custom CSS beyond small badges.
- Use the same arm assignment as the router for the single-arm mode, and show which arm the router would assign that customer.

Acceptance:

- `docker compose up` serves the page on `localhost:8501`, and it works for a heavy buyer, a light buyer and the cold-start user.
- Both arms render side by side with source badges and per-stage latency.
- A screenshot and a 30-second GIF of the page are added to the README, and the page is included in the demo video.
- Peak memory of the `demo` container is recorded in `docs/results.md`.

### Phase 7: Polish (week 7)

- Locust load test at several request rates; record p50/p95/p99 and max sustainable RPS, with the laptop spec listed beside the numbers.
- Architecture diagram (Mermaid or draw.io export), runbook, Grafana screenshots, demo video script.
- Final README (section 11) and `docs/results.md` with all tables.

## 7. Architecture (reference)

```
simulator (replay of held-out purchases) --> Redis Streams (exposures/events)

request --> FastAPI --> router (hash of customer_id)
              |-- arm A: candidates --> Feast (Redis) features --> LightGBM champion
              |-- arm B: candidates --> Feast (Redis) features --> LightGBM challenger
            --> top 12 + exposure log

candidates = repurchase + recent popularity + product-code variants
           + co-purchase + ALS + two-tower/FAISS (precomputed embeddings)

MLflow (registry, alias champion/challenger)
Prometheus <-- /metrics ; Grafana ; Evidently (drift) --> metrics + HTML reports
```

## 8. Kaggle/Colab handoff protocol (GPU work)

The agent cannot run Kaggle or Colab notebooks itself. It prepares everything and the human runs them.

Agent prepares:

1. `artifacts/kaggle_input/` containing: sampled interactions Parquet (customer idx, article idx, date), article metadata Parquet with categorical codes, article text embeddings (if computed), and `config.json` with the cutoff date.
2. `notebooks/two_tower_train.ipynb` that reads from `/kaggle/input/<dataset>` and writes to `/kaggle/working/`.
3. Clear instructions in `docs/kaggle_runbook.md` for uploading as a private Kaggle Dataset and running with GPU enabled.

Model spec:

- Item tower: embeddings for article idx, product type, colour group, department, index, garment group, plus the 384-dim text vector projected down. Output 64-dim, L2-normalised.
- User tower: customer idx embedding, age bucket, club status, and mean-pooled embeddings of the last 20 purchased items. Output 64-dim, L2-normalised.
- Training: in-batch sampled softmax with temperature about 0.05 and logQ frequency correction, batch size 4096, Adam, 5 to 10 epochs, positives only, time-ordered with the last week held out for recall@100 evaluation.
- Train twice if time allows: cutoff 2020-09-08 (for ranker-train week R) and cutoff 2020-09-15 (for validation week V). If limited, train once at 2020-09-08 and document the simplification in `docs/decisions.md`.

Human returns (to `artifacts/two_tower/`): `item_embeddings.npy`, `user_embeddings.npy` (precomputed for all sampled customers at the cutoff), `id_maps.json`, `train_log.json` (loss, recall@100 per epoch).

Serving note: user vectors are precomputed in batch, not computed live from history. Document that.

If the human cannot use the GPU, fall back to ALS-only retrieval and record that in `docs/results.md`.

## 9. Testing requirements

- `test_metrics.py`: MAP@12 on hand-computed cases, including fewer than 12 relevant items and empty predictions.
- `test_no_leakage.py`: for each target week, assert all feature and candidate sources end before the week starts.
- `test_skew.py`: online vs offline feature equality (section 6, phase 4).
- `test_router.py`: determinism, correct split proportions within tolerance, stable across restarts.
- `test_api.py`: contract tests for known, unknown and malformed ids.
- Smoke test: whole pipeline on the fixture dataset in under 5 minutes in CI.

## 10. Performance targets (aspirations, report actuals)

These are targets, not claims. Report what is measured, with hardware noted.

- Total `/recommend` p95 under 300 ms at 10 to 20 RPS on the laptop; candidates plus features plus ranking breakdown by stage.
- Ranker MAP@12 on V materially above the best baseline.
- Whole compose stack under about 6 GB RAM.

## 11. README and CV deliverables

README sections: problem and honest scope (replay simulator, sampled data, laptop numbers), architecture diagram, quick start (`make setup && make data && make train && docker compose up`), results tables (baselines, per-generator recall, ranker, ablation, experiment, latency), monitoring screenshots, decisions and trade-offs, limitations, runbook link.

CV bullets (fill with real numbers only):

- Built a two-stage recommender (multi-source candidate generation plus LightGBM LambdaRank) on a sampled subset of 31M H&M transactions, improving MAP@12 from \[baseline\] to \[final\] on a time-based holdout.
- Served features with Feast/Redis and an automated training-serving skew test; measured p95 latency of \[x\] ms at \[y\] req/s on commodity hardware.
- Implemented hash-based A/B routing with significance testing, drift detection (Evidently) and Prometheus/Grafana alerting, with a drift-triggered retraining workflow and CI/CD in GitHub Actions.

## 12. Human-in-the-loop checklist

The agent must stop and ask the human for:

1. Joining the Kaggle competition, accepting the rules, and providing the API token (`~/.kaggle/kaggle.json`). Never commit it.
2. Uploading the Kaggle input dataset and running the GPU notebook, then returning the output files.
3. Confirming CPU core count and OS (Windows/WSL2 vs Linux) if not detectable.
4. Approving any new dependency that is not in section 3.
5. Recording the demo video and capturing Grafana screenshots.

## 13. Known pitfalls

- **Time leakage** from random splits or from features computed over the target week. This is the most common failure in recsys projects.
- **Loading the CSV in pandas.** Use Polars lazy or DuckDB and convert to Parquet first.
- **Leading zeros** in `article_id` lost by a numeric dtype.
- **Overbuilding retrieval.** Co-purchase, recency and repurchase carry much of the lift; report honestly what the two-tower adds.
- **Making Kafka the centerpiece.** Keep events light; spend effort on features, ranking and experimentation.
- **Duplicated feature code** between training and serving, which causes skew. One module, two callers.
- **Overclaiming.** Call the A/B part a replay simulation; call laptop latency laptop latency.
- **RAM blowups** when joining candidates with features. Process by customer chunks and write Parquet partitions.

## 14. Definition of done

- All phases' acceptance criteria met and committed.
- `docker compose up` brings up a working stack on the stated laptop.
- CI is green.
- `docs/results.md` contains every table listed above, all from real runs.
- Runbook covers: model rollback, Redis flush and re-materialise, drift alert response, retrain.
- `docs/decisions.md` justifies each tool and lists every deviation from this plan.
