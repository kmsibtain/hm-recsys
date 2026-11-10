"""Baseline models for H&M Recommendation System.

Implements and evaluates three non-trivial baseline models on Validation Week (V):
  1. Global Top-12 Popularity (last 7 days before target week)
  2. Repurchase (last 12 distinct items per customer, padded with global popularity)
  3. Popularity per Age Bucket (last 7 days popularity by demographic age bucket)

Evaluates on active customers in Week V using the competition MAP@12 metric,
and logs all parameters, metrics, and models to MLflow.
"""

from pathlib import Path

import duckdb
import yaml

from ranking.metrics import mapk

try:
    import mlflow
except ImportError:
    mlflow = None


def get_project_root() -> Path:
    """Dynamically resolves the project root containing artifacts."""
    p = Path.cwd().resolve()
    for parent in [p] + list(p.parents):
        if (parent / "artifacts" / "parquet" / "transactions_sample.parquet").exists():
            return parent
    return p


def run_baselines(config_path: str = "configs/sampling.yaml"):
    root = get_project_root()
    full_config_path = root / config_path
    with open(full_config_path) as f:
        config = yaml.safe_load(f)

    tx_path = (root / config["artifacts"]["sampled_transactions"]).as_posix()
    (root / config["artifacts"]["articles_parquet"]).as_posix()
    cust_path = (root / config["artifacts"]["sampled_customers"]).as_posix()

    val_start = config["splits"]["val_start"]
    val_end = config["splits"]["val_end"]
    val_cutoff = config["splits"]["val_feature_cutoff"]

    print("=================================================================")
    print("PHASE 1 BASELINE EVALUATION (Validation Week V)")
    print(f"Validation Target Period: {val_start} to {val_end}")
    print(f"Feature & History Cutoff: Strictly <= {val_cutoff} (No leakage)")
    print("=================================================================")

    con = duckdb.connect()
    con.execute("PRAGMA memory_limit='4GB'")
    con.execute("PRAGMA threads=4")

    # Step 1: Extract Ground Truth for Active Customers in Validation Week V
    print("\n--- 1. Extracting Validation Ground Truth ---")
    val_gt_df = con.execute(f"""
        SELECT
            customer_id,
            LIST(DISTINCT article_id) AS actual_items
        FROM '{tx_path}'
        WHERE t_dat >= DATE '{val_start}' AND t_dat <= DATE '{val_end}'
        GROUP BY customer_id
        ORDER BY customer_id;
    """).df()

    val_customers = val_gt_df["customer_id"].tolist()
    actuals = val_gt_df["actual_items"].tolist()
    n_eval_customers = len(val_customers)
    print(f"Active validation customers to evaluate: {n_eval_customers:,}")

    # Register temp tables for efficient querying
    con.register("val_gt", val_gt_df)

    # Step 2: Baseline (a) - Global Top-12 by Sales in Last 7 Days (2020-09-09 to 2020-09-15)
    print("\n--- 2. Computing Baseline (a): Global Top-12 (Last 7 Days) ---")
    top12_global = (
        con.execute(f"""
        SELECT article_id
        FROM '{tx_path}'
        WHERE t_dat >= DATE '{val_cutoff}' - INTERVAL 6 DAY
          AND t_dat <= DATE '{val_cutoff}'
        GROUP BY article_id
        ORDER BY count(*) DESC
        LIMIT 12;
    """)
        .df()["article_id"]
        .tolist()
    )
    print(f"Top 12 global popular items: {top12_global}")

    # Predict identical top-12 for all active customers
    preds_global = [top12_global] * n_eval_customers
    map12_global = mapk(actuals, preds_global, k=12, filter_empty_actuals=True)
    print(f">> Baseline (a) MAP@12: {map12_global:.6f}")

    # Step 3: Baseline (b) - User Repurchase (Recency Weighted + Popularity Fallback)
    print("\n--- 3. Computing Baseline (b): User Repurchase + Popularity Fallback ---")
    # For each customer, get their distinct previously bought items ordered by latest purchase date
    user_repurchases = (
        con.execute(f"""
        WITH recent_buys AS (
            SELECT
                customer_id,
                article_id,
                max(t_dat) AS last_buy,
                row_number() OVER (PARTITION BY customer_id ORDER BY max(t_dat) DESC) AS rn
            FROM '{tx_path}'
            WHERE t_dat <= DATE '{val_cutoff}'
            GROUP BY customer_id, article_id
        )
        SELECT
            customer_id,
            LIST(article_id ORDER BY rn) AS repurchased_items
        FROM recent_buys
        WHERE rn <= 12
        GROUP BY customer_id;
    """)
        .df()
        .set_index("customer_id")["repurchased_items"]
        .to_dict()
    )

    # Build predictions padded with global popularity
    preds_repurchase = []
    for cust_id in val_customers:
        items = list(user_repurchases.get(cust_id, []))
        if len(items) < 12:
            seen = set(items)
            for pop_item in top12_global:
                if pop_item not in seen:
                    items.append(pop_item)
                    seen.add(pop_item)
                if len(items) == 12:
                    break
        preds_repurchase.append(items[:12])

    map12_repurchase = mapk(actuals, preds_repurchase, k=12, filter_empty_actuals=True)
    print(f">> Baseline (b) MAP@12: {map12_repurchase:.6f}")

    # Step 4: Baseline (c) - Popularity per Demographic Age Bucket
    print("\n--- 4. Computing Baseline (c): Popularity per Age Bucket ---")
    # Define age buckets: <20, 20-29, 30-39, 40-49, 50-59, 60+, and Unknown
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE customer_age_buckets AS
        SELECT
            customer_id,
            CASE
                WHEN age IS NULL THEN 'Unknown'
                WHEN age < 20 THEN '<20'
                WHEN age < 30 THEN '20-29'
                WHEN age < 40 THEN '30-39'
                WHEN age < 50 THEN '40-49'
                WHEN age < 60 THEN '50-59'
                ELSE '60+'
            END AS age_bucket
        FROM '{cust_path}';
    """)

    # Top 12 per age bucket over the last 7 days before cutoff
    age_bucket_pops = (
        con.execute(f"""
        WITH bucket_sales AS (
            SELECT
                b.age_bucket,
                t.article_id,
                count(*) AS sales,
                row_number() OVER (PARTITION BY b.age_bucket ORDER BY count(*) DESC) AS rn
            FROM '{tx_path}' t
            JOIN customer_age_buckets b ON t.customer_id = b.customer_id
            WHERE t.t_dat >= DATE '{val_cutoff}' - INTERVAL 6 DAY
              AND t.t_dat <= DATE '{val_cutoff}'
            GROUP BY b.age_bucket, t.article_id
        )
        SELECT
            age_bucket,
            LIST(article_id ORDER BY rn) AS bucket_top12
        FROM bucket_sales
        WHERE rn <= 12
        GROUP BY age_bucket;
    """)
        .df()
        .set_index("age_bucket")["bucket_top12"]
        .to_dict()
    )

    # Customer to age bucket mapping
    cust_to_bucket = (
        con.execute("""
        SELECT customer_id, age_bucket FROM customer_age_buckets;
    """)
        .df()
        .set_index("customer_id")["age_bucket"]
        .to_dict()
    )

    # Build predictions by age bucket (fallback to global top-12 if needed)
    preds_age_bucket = []
    for cust_id in val_customers:
        bucket = cust_to_bucket.get(cust_id, "Unknown")
        bucket_items = list(age_bucket_pops.get(bucket, []))
        if len(bucket_items) < 12:
            seen = set(bucket_items)
            for pop_item in top12_global:
                if pop_item not in seen:
                    bucket_items.append(pop_item)
                    seen.add(pop_item)
                if len(bucket_items) == 12:
                    break
        preds_age_bucket.append(bucket_items[:12])

    map12_age_bucket = mapk(actuals, preds_age_bucket, k=12, filter_empty_actuals=True)
    print(f">> Baseline (c) MAP@12: {map12_age_bucket:.6f}")

    con.close()

    # Step 5: Summary Results Dictionary
    results = {
        "baseline_global_top12": float(map12_global),
        "baseline_repurchase": float(map12_repurchase),
        "baseline_age_bucket": float(map12_age_bucket),
        "val_customers": n_eval_customers,
    }

    # Step 6: Log to MLflow if available
    print("\n--- 5. Logging to MLflow ---")
    if mlflow:
        # Per Handover Plan Section 3: SQLite backend, local artifact store
        mlflow_db = (root / "artifacts" / "mlflow.db").as_posix()
        mlflow.set_tracking_uri(f"sqlite:///{mlflow_db}")
        mlflow.set_experiment("hm-recsys-baselines")

        # Run 1: Global Top-12
        with mlflow.start_run(run_name="baseline_global_top12"):
            mlflow.log_params(
                {"model": "global_top12", "k": 12, "window_days": 7, "cutoff": val_cutoff}
            )
            mlflow.log_metric("val_map12", map12_global)
            mlflow.log_metric("eval_customers", n_eval_customers)

        # Run 2: Repurchase + Global Fallback
        with mlflow.start_run(run_name="baseline_repurchase"):
            mlflow.log_params(
                {"model": "repurchase_with_pop_fallback", "k": 12, "cutoff": val_cutoff}
            )
            mlflow.log_metric("val_map12", map12_repurchase)
            mlflow.log_metric("eval_customers", n_eval_customers)

        # Run 3: Age Bucket Popularity
        with mlflow.start_run(run_name="baseline_age_bucket_popularity"):
            mlflow.log_params(
                {"model": "age_bucket_popularity", "k": 12, "window_days": 7, "cutoff": val_cutoff}
            )
            mlflow.log_metric("val_map12", map12_age_bucket)
            mlflow.log_metric("eval_customers", n_eval_customers)

        print("Logged all three baseline runs to MLflow experiment 'hm-recsys-baselines'.")
    else:
        print("MLflow not installed; skipped MLflow logging.")

    print("\n=================================================================")
    print("PHASE 1 BASELINE RESULTS SUMMARY:")
    print(f"  1. Global Top-12 (Last 7d):       MAP@12 = {map12_global:.6f}")
    print(f"  2. Repurchase + Pop Fallback:     MAP@12 = {map12_repurchase:.6f}")
    print(f"  3. Popularity per Age Bucket:     MAP@12 = {map12_age_bucket:.6f}")
    print("=================================================================")

    return results


if __name__ == "__main__":
    run_baselines()
