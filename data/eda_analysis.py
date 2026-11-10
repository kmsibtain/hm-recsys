"""Comprehensive Exploratory Data Analysis (EDA) and Candidate Ceiling Evaluation.

Performs:
1. Structural Profiling & Relational Integrity
2. Distributions & Long-Tail Skew (Pareto, Gini, Quantiles)
3. Temporal Dynamics & Catalog Turnover
4. Customer Behavior & Cold-Start Analysis
5. Data Quality & Anomalies
6. Retrieval Candidate Upper-Bound Testing (Recall@K on Validation Week V)
"""

import json
from pathlib import Path

import duckdb


def get_project_root() -> Path:
    p = Path.cwd().resolve()
    for parent in [p] + list(p.parents):
        if (parent / "artifacts" / "parquet" / "transactions_sample.parquet").exists():
            return parent
    return p


def run_eda(parquet_dir: str = None):
    root = get_project_root()
    if parquet_dir is None:
        parquet_dir = (root / "artifacts" / "parquet").as_posix()
    else:
        parquet_dir = Path(parquet_dir).resolve().as_posix()

    con = duckdb.connect()
    con.execute("PRAGMA memory_limit='4GB'")
    con.execute("PRAGMA threads=4")

    tx_path = f"{parquet_dir}/transactions_sample.parquet"
    art_path = f"{parquet_dir}/articles.parquet"
    cust_path = f"{parquet_dir}/customers_sample.parquet"
    out_json = (root / "artifacts" / "eda_results.json").as_posix()

    print("=================================================================")
    print("STEP 2: STRUCTURAL PROFILING & RELATIONAL INTEGRITY")
    print("=================================================================")

    # 1. Row counts and column checks
    tx_count = con.execute(f"SELECT count(*) FROM '{tx_path}'").fetchone()[0]
    art_count = con.execute(f"SELECT count(*) FROM '{art_path}'").fetchone()[0]
    cust_count = con.execute(f"SELECT count(*) FROM '{cust_path}'").fetchone()[0]
    print(f"Transactions rows: {tx_count:,}")
    print(f"Articles rows:     {art_count:,}")
    print(f"Customers rows:    {cust_count:,}")

    # 2. Date ranges
    date_bounds = con.execute(f"SELECT min(t_dat), max(t_dat) FROM '{tx_path}'").fetchall()[0]
    print(f"Transaction date range: {date_bounds[0]} to {date_bounds[1]}")

    # 3. Foreign key integrity
    orphan_cust = con.execute(f"""
        SELECT count(distinct t.customer_id)
        FROM '{tx_path}' t
        LEFT JOIN '{cust_path}' c ON t.customer_id = c.customer_id
        WHERE c.customer_id IS NULL
    """).fetchone()[0]

    orphan_art = con.execute(f"""
        SELECT count(distinct t.article_id)
        FROM '{tx_path}' t
        LEFT JOIN '{art_path}' a ON t.article_id = a.article_id
        WHERE a.article_id IS NULL
    """).fetchone()[0]
    print(f"Orphaned customer transactions: {orphan_cust}")
    print(f"Orphaned article transactions:  {orphan_art}")

    # 4. Null value checks
    print("\nCustomer missing values:")
    cust_nulls = con.execute(f"""
        SELECT
            count(*) - count(FN) AS fn_null,
            count(*) - count(Active) AS active_null,
            count(*) - count(club_member_status) AS club_null,
            count(*) - count(fashion_news_frequency) AS news_null,
            count(*) - count(age) AS age_null
        FROM '{cust_path}'
    """).fetchall()[0]
    print(f"  FN: {cust_nulls[0]:,} ({cust_nulls[0] / cust_count * 100:.1f}%)")
    print(f"  Active: {cust_nulls[1]:,} ({cust_nulls[1] / cust_count * 100:.1f}%)")
    print(f"  club_member_status: {cust_nulls[2]:,} ({cust_nulls[2] / cust_count * 100:.2f}%)")
    print(f"  fashion_news_frequency: {cust_nulls[3]:,} ({cust_nulls[3] / cust_count * 100:.2f}%)")
    print(f"  age: {cust_nulls[4]:,} ({cust_nulls[4] / cust_count * 100:.2f}%)")

    # 5. Article string length checks
    art_len_check = con.execute(f"""
        SELECT min(length(article_id)), max(length(article_id)) FROM '{art_path}'
    """).fetchall()[0]
    print(
        f"Article ID string length min/max: {art_len_check[0]} / {art_len_check[1]} (strictly 10 digits)"
    )

    print("\n=================================================================")
    print("STEP 3: DISTRIBUTIONS & RECOMMENDER LONG-TAIL SKEW")
    print("=================================================================")

    # 1. Item popularity Pareto
    con.execute(f"""
        CREATE TEMP TABLE item_sales AS
        SELECT article_id, count(*) AS sales
        FROM '{tx_path}'
        GROUP BY article_id
        ORDER BY sales DESC;
    """)
    total_sales = tx_count
    top_1pct = con.execute(
        "SELECT sum(sales) FROM (SELECT sales FROM item_sales LIMIT (SELECT CAST(count(*)*0.01 AS INT) FROM item_sales))"
    ).fetchone()[0]
    top_5pct = con.execute(
        "SELECT sum(sales) FROM (SELECT sales FROM item_sales LIMIT (SELECT CAST(count(*)*0.05 AS INT) FROM item_sales))"
    ).fetchone()[0]
    top_10pct = con.execute(
        "SELECT sum(sales) FROM (SELECT sales FROM item_sales LIMIT (SELECT CAST(count(*)*0.10 AS INT) FROM item_sales))"
    ).fetchone()[0]
    top_20pct = con.execute(
        "SELECT sum(sales) FROM (SELECT sales FROM item_sales LIMIT (SELECT CAST(count(*)*0.20 AS INT) FROM item_sales))"
    ).fetchone()[0]

    print("Item Pareto distribution:")
    print(
        f"  Top 1% of active items account for:  {top_1pct:,} sales ({top_1pct / total_sales * 100:.2f}%)"
    )
    print(
        f"  Top 5% of active items account for:  {top_5pct:,} sales ({top_5pct / total_sales * 100:.2f}%)"
    )
    print(
        f"  Top 10% of active items account for: {top_10pct:,} sales ({top_10pct / total_sales * 100:.2f}%)"
    )
    print(
        f"  Top 20% of active items account for: {top_20pct:,} sales ({top_20pct / total_sales * 100:.2f}%)"
    )

    # 2. Customer activity quantiles
    cust_quantiles = con.execute(f"""
        SELECT
            quantile_cont(tx_per_cust, 0.10),
            quantile_cont(tx_per_cust, 0.25),
            quantile_cont(tx_per_cust, 0.50),
            quantile_cont(tx_per_cust, 0.75),
            quantile_cont(tx_per_cust, 0.90),
            quantile_cont(tx_per_cust, 0.95),
            quantile_cont(tx_per_cust, 0.99),
            max(tx_per_cust),
            avg(tx_per_cust)
        FROM (
            SELECT customer_id, count(*) AS tx_per_cust
            FROM '{tx_path}'
            GROUP BY customer_id
        )
    """).fetchall()[0]
    print("\nCustomer purchase volume percentiles:")
    print(
        f"  p10: {cust_quantiles[0]:.1f}, p25: {cust_quantiles[1]:.1f}, Median (p50): {cust_quantiles[2]:.1f}, "
        f"p75: {cust_quantiles[3]:.1f}, p90: {cust_quantiles[4]:.1f}, p95: {cust_quantiles[5]:.1f}, p99: {cust_quantiles[6]:.1f}, Max: {cust_quantiles[7]}, Mean: {cust_quantiles[8]:.2f}"
    )

    # Customer activity buckets
    cust_buckets = con.execute(f"""
        SELECT
            sum(CASE WHEN tx_per_cust = 1 THEN 1 ELSE 0 END),
            sum(CASE WHEN tx_per_cust BETWEEN 2 AND 5 THEN 1 ELSE 0 END),
            sum(CASE WHEN tx_per_cust BETWEEN 6 AND 15 THEN 1 ELSE 0 END),
            sum(CASE WHEN tx_per_cust > 15 THEN 1 ELSE 0 END)
        FROM (
            SELECT customer_id, count(*) AS tx_per_cust
            FROM '{tx_path}'
            GROUP BY customer_id
        )
    """).fetchall()[0]
    print("Customer activity segment share:")
    print(f"  1 purchase only:    {cust_buckets[0]:,} ({cust_buckets[0] / cust_count * 100:.1f}%)")
    print(f"  2 to 5 purchases:   {cust_buckets[1]:,} ({cust_buckets[1] / cust_count * 100:.1f}%)")
    print(f"  6 to 15 purchases:  {cust_buckets[2]:,} ({cust_buckets[2] / cust_count * 100:.1f}%)")
    print(f"  >15 purchases:      {cust_buckets[3]:,} ({cust_buckets[3] / cust_count * 100:.1f}%)")

    # 3. Price distribution
    price_stats = con.execute(f"""
        SELECT
            min(price),
            quantile_cont(price, 0.25),
            quantile_cont(price, 0.50),
            avg(price),
            quantile_cont(price, 0.75),
            quantile_cont(price, 0.99),
            max(price)
        FROM '{tx_path}'
    """).fetchall()[0]
    print("\nPrice distribution:")
    print(
        f"  Min: {price_stats[0]:.4f}, p25: {price_stats[1]:.4f}, Median: {price_stats[2]:.4f}, "
        f"Mean: {price_stats[3]:.4f}, p75: {price_stats[4]:.4f}, p99: {price_stats[5]:.4f}, Max: {price_stats[6]:.4f}"
    )

    print("\n=================================================================")
    print("STEP 4: TEMPORAL DYNAMICS & CATALOG TURNOVER")
    print("=================================================================")

    # 1. Product lifespan in circulation
    lifespans = con.execute(f"""
        SELECT
            avg(datediff('day', first_sale, last_sale)),
            median(datediff('day', first_sale, last_sale)),
            quantile_cont(datediff('day', first_sale, last_sale), 0.75),
            max(datediff('day', first_sale, last_sale))
        FROM (
            SELECT article_id, min(t_dat) AS first_sale, max(t_dat) AS last_sale
            FROM '{tx_path}'
            GROUP BY article_id
        )
    """).fetchall()[0]
    print("Article circulation lifespan (days between first and last sale in 12-month window):")
    print(f"  Average: {lifespans[0]:.1f} days")
    print(f"  Median:  {lifespans[1]:.1f} days")
    print(f"  p75:     {lifespans[2]:.1f} days")
    print(f"  Max:     {lifespans[3]} days")

    # 2. Proportion of sales in last 7 days vs older
    recency_breakdown = con.execute(f"""
        SELECT
            sum(CASE WHEN t_dat >= DATE '2020-09-09' THEN 1 ELSE 0 END) AS last_14d_sales,
            sum(CASE WHEN t_dat >= DATE '2020-09-16' THEN 1 ELSE 0 END) AS val_week_sales
        FROM '{tx_path}'
    """).fetchall()[0]
    print(f"Sales in last 14 days (Ranker + Val weeks): {recency_breakdown[0]:,}")
    print(f"Sales in Validation week (V):             {recency_breakdown[1]:,}")

    print("\n=================================================================")
    print("STEP 5: CUSTOMER BEHAVIOR & REPURCHASE DYNAMICS")
    print("=================================================================")

    # 1. Overall repurchase rate (buying same article_id again)
    repurchase_stats = con.execute(f"""
        SELECT
            count(*),
            sum(CASE WHEN purchase_seq > 1 THEN 1 ELSE 0 END) AS repurchases
        FROM (
            SELECT
                customer_id,
                article_id,
                row_number() OVER (PARTITION BY customer_id, article_id ORDER BY t_dat) AS purchase_seq
            FROM '{tx_path}'
        )
    """).fetchall()[0]
    print(
        f"Overall repurchase rate: {repurchase_stats[1]:,} / {repurchase_stats[0]:,} ({repurchase_stats[1] / repurchase_stats[0] * 100:.2f}%)"
    )

    # 2. Product-code repeat rate (buying different color/variant of same style)
    prod_code_repeat = con.execute(f"""
        SELECT
            count(*),
            sum(CASE WHEN code_seq > 1 THEN 1 ELSE 0 END) AS code_repeats
        FROM (
            SELECT
                t.customer_id,
                a.product_code,
                row_number() OVER (PARTITION BY t.customer_id, a.product_code ORDER BY t.t_dat) AS code_seq
            FROM '{tx_path}' t
            JOIN '{art_path}' a ON t.article_id = a.article_id
        )
    """).fetchall()[0]
    print(
        f"Product-code variant repeat rate: {prod_code_repeat[1]:,} / {prod_code_repeat[0]:,} ({prod_code_repeat[1] / prod_code_repeat[0] * 100:.2f}%)"
    )

    # 3. Duplicate transactions on same day (basket size / multiple units)
    basket_duplicates = con.execute(f"""
        SELECT
            count(*) AS total_rows,
            count(distinct concat(customer_id, '_', article_id, '_', CAST(t_dat AS VARCHAR))) AS distinct_tuples
        FROM '{tx_path}'
    """).fetchall()[0]
    dup_rows = basket_duplicates[0] - basket_duplicates[1]
    print(
        f"Multi-unit purchases on same day (same customer, article, date): {dup_rows:,} ({dup_rows / basket_duplicates[0] * 100:.2f}%)"
    )

    # 4. Cold start in Validation Week V (2020-09-16 to 2020-09-22)
    val_cust_stats = con.execute(f"""
        WITH val_customers AS (
            SELECT DISTINCT customer_id
            FROM '{tx_path}'
            WHERE t_dat >= DATE '2020-09-16' AND t_dat <= DATE '2020-09-22'
        ),
        hist_counts AS (
            SELECT
                vc.customer_id,
                count(t.article_id) AS prior_purchases,
                sum(CASE WHEN t.t_dat >= DATE '2020-08-16' THEN 1 ELSE 0 END) AS last_30d_purchases
            FROM val_customers vc
            LEFT JOIN '{tx_path}' t
              ON vc.customer_id = t.customer_id
             AND t.t_dat < DATE '2020-09-16'
            GROUP BY vc.customer_id
        )
        SELECT
            count(*),
            sum(CASE WHEN prior_purchases = 0 THEN 1 ELSE 0 END) AS zero_prior_purchases,
            sum(CASE WHEN last_30d_purchases = 0 THEN 1 ELSE 0 END) AS zero_last_30d_purchases
        FROM hist_counts
    """).fetchall()[0]
    total_val_custs = val_cust_stats[0]
    print(f"\nValidation Week Active Customers: {total_val_custs:,}")
    print(
        f"  Zero prior purchases in 12-month window: {val_cust_stats[1]:,} ({val_cust_stats[1] / total_val_custs * 100:.2f}%)"
    )
    print(
        f"  Zero purchases in preceding 30 days:     {val_cust_stats[2]:,} ({val_cust_stats[2] / total_val_custs * 100:.2f}%)"
    )

    print("\n=================================================================")
    print("STEP 6: DATA QUALITY & ANOMALIES")
    print("=================================================================")

    # 1. Zero price transactions
    zero_prices = con.execute(f"SELECT count(*) FROM '{tx_path}' WHERE price <= 0").fetchone()[0]
    print(f"Zero or negative price transactions: {zero_prices}")

    # 2. Articles in catalog with 0 sales
    unsold_articles = con.execute(f"""
        SELECT count(*)
        FROM '{art_path}' a
        LEFT JOIN (SELECT DISTINCT article_id FROM '{tx_path}') t ON a.article_id = t.article_id
        WHERE t.article_id IS NULL
    """).fetchone()[0]
    print(
        f"Catalog articles never sold in 12-month window: {unsold_articles:,} / {art_count:,} ({unsold_articles / art_count * 100:.2f}%)"
    )

    # 3. Channel breakdown
    channels = con.execute(f"""
        SELECT sales_channel_id, count(*), count(*)*100.0/(SELECT count(*) FROM '{tx_path}')
        FROM '{tx_path}'
        GROUP BY sales_channel_id
    """).fetchall()
    print("Sales channel split:")
    for ch, cnt, pct in channels:
        name = "Online (2)" if ch == 2 else "In-Store (1)"
        print(f"  Channel {ch} ({name}): {cnt:,} ({pct:.2f}%)")

    print("\n=================================================================")
    print("STEP 7: RETRIEVAL CANDIDATE UPPER-BOUND (RECALL CEILING ON WEEK V)")
    print("=================================================================")

    # Target: Customers who made purchases in Week V (2020-09-16 to 2020-09-22)
    # Ground Truth: Actual items bought in Week V
    # Condition: Candidates generated ONLY using data strictly before 2020-09-16 (t_dat <= '2020-09-15')

    con.execute(f"""
        CREATE TEMP TABLE val_ground_truth AS
        SELECT DISTINCT customer_id, article_id
        FROM '{tx_path}'
        WHERE t_dat >= DATE '2020-09-16' AND t_dat <= DATE '2020-09-22';
    """)
    total_val_pairs = con.execute("SELECT count(*) FROM val_ground_truth").fetchone()[0]
    val_customers_count = con.execute(
        "SELECT count(distinct customer_id) FROM val_ground_truth"
    ).fetchone()[0]
    print(
        f"Validation Week target: {val_customers_count:,} customers, {total_val_pairs:,} positive (customer, article) pairs."
    )

    # Generator 1: Repurchase (Top 30 distinct items previously bought by user)
    con.execute(f"""
        CREATE TEMP TABLE cand_repurchase AS
        SELECT customer_id, article_id
        FROM (
            SELECT
                customer_id,
                article_id,
                max(t_dat) AS last_buy,
                row_number() OVER (PARTITION BY customer_id ORDER BY max(t_dat) DESC) AS rn
            FROM '{tx_path}'
            WHERE t_dat < DATE '2020-09-16'
            GROUP BY customer_id, article_id
        )
        WHERE rn <= 30;
    """)
    hits_repurchase = con.execute("""
        SELECT count(*)
        FROM val_ground_truth gt
        JOIN cand_repurchase c ON gt.customer_id = c.customer_id AND gt.article_id = c.article_id
    """).fetchone()[0]
    recall_repurchase = hits_repurchase / total_val_pairs
    print("\n1. Repurchase Generator (Top 30):")
    print(f"   Hits: {hits_repurchase:,} / {total_val_pairs:,}")
    print(f"   Recall@30: {recall_repurchase * 100:.2f}%")

    # Generator 2: Recent Popularity (Global Top 50 in last 7 days: 2020-09-09 to 2020-09-15)
    con.execute(f"""
        CREATE TEMP TABLE cand_popular_7d AS
        SELECT article_id
        FROM '{tx_path}'
        WHERE t_dat >= DATE '2020-09-09' AND t_dat <= DATE '2020-09-15'
        GROUP BY article_id
        ORDER BY count(*) DESC
        LIMIT 50;
    """)
    hits_popular_7d = con.execute("""
        SELECT count(*)
        FROM val_ground_truth gt
        JOIN cand_popular_7d c ON gt.article_id = c.article_id
    """).fetchone()[0]
    recall_pop_7d = hits_popular_7d / total_val_pairs
    print("\n2. Global Popularity Generator (Last 7 Days, Top 50):")
    print(f"   Hits: {hits_popular_7d:,} / {total_val_pairs:,}")
    print(f"   Recall@50: {recall_pop_7d * 100:.2f}%")

    # Generator 2b: Recent Popularity (Global Top 50 in last 14 days: 2020-09-02 to 2020-09-15)
    con.execute(f"""
        CREATE TEMP TABLE cand_popular_14d AS
        SELECT article_id
        FROM '{tx_path}'
        WHERE t_dat >= DATE '2020-09-02' AND t_dat <= DATE '2020-09-15'
        GROUP BY article_id
        ORDER BY count(*) DESC
        LIMIT 50;
    """)
    hits_popular_14d = con.execute("""
        SELECT count(*)
        FROM val_ground_truth gt
        JOIN cand_popular_14d c ON gt.article_id = c.article_id
    """).fetchone()[0]
    recall_pop_14d = hits_popular_14d / total_val_pairs
    print("\n3. Global Popularity Generator (Last 14 Days, Top 50):")
    print(f"   Hits: {hits_popular_14d:,} / {total_val_pairs:,}")
    print(f"   Recall@50: {recall_pop_14d * 100:.2f}%")

    # Generator 3: Product-Code Variants (Other articles with same product_code as customer's bought items, up to 30)
    con.execute(f"""
        CREATE TEMP TABLE user_product_codes AS
        SELECT DISTINCT t.customer_id, a.product_code
        FROM '{tx_path}' t
        JOIN '{art_path}' a ON t.article_id = a.article_id
        WHERE t.t_dat < DATE '2020-09-16';

        CREATE TEMP TABLE cand_variants AS
        SELECT customer_id, article_id
        FROM (
            SELECT
                u.customer_id,
                a.article_id,
                row_number() OVER (PARTITION BY u.customer_id ORDER BY a.article_id) AS rn
            FROM user_product_codes u
            JOIN '{art_path}' a ON u.product_code = a.product_code
        )
        WHERE rn <= 30;
    """)
    hits_variants = con.execute("""
        SELECT count(*)
        FROM val_ground_truth gt
        JOIN cand_variants c ON gt.customer_id = c.customer_id AND gt.article_id = c.article_id
    """).fetchone()[0]
    recall_variants = hits_variants / total_val_pairs
    print("\n4. Product-Code Variant Generator (Top 30):")
    print(f"   Hits: {hits_variants:,} / {total_val_pairs:,}")
    print(f"   Recall@30: {recall_variants * 100:.2f}%")

    # Generator 4: Item-Item Co-purchase (Co-purchased within 14 days, applied to user's last 5 items)
    print("\n5. Computing Item-Item Co-purchase Co-occurrences...")
    con.execute(f"""
        CREATE TEMP TABLE recent_user_items AS
        SELECT customer_id, article_id
        FROM (
            SELECT
                customer_id,
                article_id,
                row_number() OVER (PARTITION BY customer_id ORDER BY t_dat DESC) AS rn
            FROM '{tx_path}'
            WHERE t_dat < DATE '2020-09-16'
        )
        WHERE rn <= 5;

        -- Fast co-purchase pairs in last 30 days
        CREATE TEMP TABLE top_co_purchases AS
        SELECT a1, a2
        FROM (
            SELECT
                t1.article_id AS a1,
                t2.article_id AS a2,
                count(*) AS pair_count,
                row_number() OVER (PARTITION BY t1.article_id ORDER BY count(*) DESC) AS rn
            FROM '{tx_path}' t1
            JOIN '{tx_path}' t2
              ON t1.customer_id = t2.customer_id
             AND t1.article_id != t2.article_id
             AND abs(datediff('day', t1.t_dat, t2.t_dat)) <= 14
            WHERE t1.t_dat >= DATE '2020-08-16' AND t1.t_dat < DATE '2020-09-16'
              AND t2.t_dat >= DATE '2020-08-16' AND t2.t_dat < DATE '2020-09-16'
            GROUP BY t1.article_id, t2.article_id
        )
        WHERE rn <= 10;

        CREATE TEMP TABLE cand_copurchase AS
        SELECT DISTINCT u.customer_id, cp.a2 AS article_id
        FROM recent_user_items u
        JOIN top_co_purchases cp ON u.article_id = cp.a1;
    """)
    hits_copurchase = con.execute("""
        SELECT count(*)
        FROM val_ground_truth gt
        JOIN cand_copurchase c ON gt.customer_id = c.customer_id AND gt.article_id = c.article_id
    """).fetchone()[0]
    recall_copurchase = hits_copurchase / total_val_pairs
    print(f"   Hits: {hits_copurchase:,} / {total_val_pairs:,}")
    print(f"   Recall@CoPurchase: {recall_copurchase * 100:.2f}%")

    # Union Recall (Combining Repurchase + Popularity 7d + Product Variants + Co-purchase)
    con.execute("""
        CREATE TEMP TABLE cand_union AS
        SELECT customer_id, article_id FROM cand_repurchase
        UNION
        SELECT vc.customer_id, p.article_id FROM cand_popular_7d p CROSS JOIN (SELECT DISTINCT customer_id FROM val_ground_truth) vc
        UNION
        SELECT customer_id, article_id FROM cand_variants
        UNION
        SELECT customer_id, article_id FROM cand_copurchase;
    """)
    hits_union = con.execute("""
        SELECT count(*)
        FROM val_ground_truth gt
        JOIN cand_union c ON gt.customer_id = c.customer_id AND gt.article_id = c.article_id
    """).fetchone()[0]
    recall_union = hits_union / total_val_pairs
    print("\n=================================================================")
    print("UNION RETRIEVAL CEILING (Repurchase + Pop 7d + Variants + Co-purchase):")
    print(f"  Total Union Hits: {hits_union:,} / {total_val_pairs:,}")
    print(f"  Max Possible Recall: {recall_union * 100:.2f}%")
    print("=================================================================")

    # Prepare structured summary dict
    results = {
        "dataset": {
            "transactions": tx_count,
            "articles": art_count,
            "customers": cust_count,
            "val_customers": val_customers_count,
            "val_positive_pairs": total_val_pairs,
        },
        "retrieval_ceilings": {
            "repurchase_recall": float(recall_repurchase),
            "pop_7d_recall": float(recall_pop_7d),
            "pop_14d_recall": float(recall_pop_14d),
            "variants_recall": float(recall_variants),
            "copurchase_recall": float(recall_copurchase),
            "union_recall": float(recall_union),
        },
    }
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    con.close()
    return results


if __name__ == "__main__":
    run_eda()
