"""Data ingestion, parquet conversion, sampling, and ID mapping.

Converts raw CSV files to Parquet, enforces string types on IDs, applies 10%
customer sampling on the 12-month window, and saves outputs to artifacts/parquet/.
"""

import os
import yaml
import duckdb


def run_ingestion(config_path: str = "configs/sampling.yaml"):
    print(f"Loading configuration from {config_path}...")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    raw_transactions = config["data"]["raw_transactions"]
    raw_customers = config["data"]["raw_customers"]
    raw_articles = config["data"]["raw_articles"]

    window_start = config["data"]["window_start_date"]
    window_end = config["data"]["window_end_date"]
    sample_frac = config["data"]["sample_frac"]
    seed = config["data"]["seed"]

    parquet_dir = config["artifacts"]["parquet_dir"]
    os.makedirs(parquet_dir, exist_ok=True)

    out_articles = config["artifacts"]["articles_parquet"]
    out_transactions = config["artifacts"]["sampled_transactions"]
    out_customers = config["artifacts"]["sampled_customers"]
    out_cust_map = os.path.join(parquet_dir, "customer_id_map.parquet")
    out_article_map = os.path.join(parquet_dir, "article_id_map.parquet")

    con = duckdb.connect()
    con.execute("PRAGMA memory_limit='6GB'")
    con.execute("PRAGMA threads=4")

    print("\n--- Step 1: Converting articles.csv to Parquet ---")
    con.execute(f"""
        COPY (
            SELECT 
                CAST(article_id AS VARCHAR) AS article_id,
                CAST(product_code AS VARCHAR) AS product_code,
                prod_name,
                CAST(product_type_no AS INT) AS product_type_no,
                product_type_name,
                product_group_name,
                CAST(graphical_appearance_no AS INT) AS graphical_appearance_no,
                graphical_appearance_name,
                CAST(colour_group_code AS INT) AS colour_group_code,
                colour_group_name,
                CAST(perceived_colour_value_id AS INT) AS perceived_colour_value_id,
                perceived_colour_value_name,
                CAST(perceived_colour_master_id AS INT) AS perceived_colour_master_id,
                perceived_colour_master_name,
                CAST(department_no AS INT) AS department_no,
                department_name,
                index_code,
                index_name,
                CAST(index_group_no AS INT) AS index_group_no,
                index_group_name,
                CAST(section_no AS INT) AS section_no,
                section_name,
                CAST(garment_group_no AS INT) AS garment_group_no,
                garment_group_name,
                detail_desc
            FROM read_csv('{raw_articles}', 
                          types={{'article_id': 'VARCHAR', 'product_code': 'VARCHAR'}}, 
                          header=True)
        ) TO '{out_articles}' (FORMAT PARQUET, COMPRESSION 'ZSTD');
    """)
    article_count = con.execute(f"SELECT count(*) FROM '{out_articles}'").fetchone()[0]
    print(f"Saved {article_count:,} articles to {out_articles}")

    # Build article_idx mapping
    con.execute(f"""
        COPY (
            SELECT 
                article_id,
                CAST(ROW_NUMBER() OVER (ORDER BY article_id) - 1 AS INTEGER) AS article_idx
            FROM '{out_articles}'
        ) TO '{out_article_map}' (FORMAT PARQUET, COMPRESSION 'ZSTD');
    """)
    print(f"Saved article ID map to {out_article_map}")

    print("\n--- Step 2: Customer Sampling on 12-Month Window ---")
    print(f"Window: {window_start} to {window_end}")
    print(f"Sampling fraction: {sample_frac*100:.1f}%, Seed: {seed}")

    # Active customers in window
    con.execute(f"""
        CREATE TEMP TABLE active_window_customers AS
        SELECT DISTINCT customer_id
        FROM read_csv('{raw_transactions}', 
                      types={{'article_id': 'VARCHAR', 'customer_id': 'VARCHAR'}}, 
                      header=True)
        WHERE CAST(t_dat AS DATE) >= DATE '{window_start}'
          AND CAST(t_dat AS DATE) <= DATE '{window_end}';
    """)
    total_active_cust = con.execute("SELECT count(*) FROM active_window_customers").fetchone()[0]
    print(f"Total active customers in 12-month window: {total_active_cust:,}")

    # Deterministic sampling: abs(hash(customer_id || seed)) % 1000000 < sample_frac * 1000000
    threshold = int(sample_frac * 1_000_000)
    con.execute(f"""
        CREATE TEMP TABLE sampled_customer_ids AS
        SELECT 
            customer_id,
            CAST(ROW_NUMBER() OVER (ORDER BY customer_id) - 1 AS INTEGER) AS customer_idx
        FROM active_window_customers
        WHERE abs(hash(concat(customer_id, '{seed}'))) % 1000000 < {threshold};
    """)
    sampled_cust_count = con.execute("SELECT count(*) FROM sampled_customer_ids").fetchone()[0]
    print(f"Sampled customers: {sampled_cust_count:,} ({sampled_cust_count/total_active_cust*100:.2f}%)")

    # Save customer ID map
    con.execute(f"""
        COPY sampled_customer_ids TO '{out_cust_map}' (FORMAT PARQUET, COMPRESSION 'ZSTD');
    """)
    print(f"Saved customer ID mapping to {out_cust_map}")

    print("\n--- Step 3: Filtering & Ingesting Sampled Transactions ---")
    con.execute(f"""
        COPY (
            SELECT 
                CAST(t.t_dat AS DATE) AS t_dat,
                t.customer_id,
                m.customer_idx,
                t.article_id,
                am.article_idx,
                CAST(t.price AS FLOAT) AS price,
                CAST(t.sales_channel_id AS TINYINT) AS sales_channel_id
            FROM read_csv('{raw_transactions}', 
                          types={{'article_id': 'VARCHAR', 'customer_id': 'VARCHAR'}}, 
                          header=True) t
            JOIN sampled_customer_ids m ON t.customer_id = m.customer_id
            JOIN '{out_article_map}' am ON t.article_id = am.article_id
            WHERE CAST(t.t_dat AS DATE) >= DATE '{window_start}'
              AND CAST(t.t_dat AS DATE) <= DATE '{window_end}'
            ORDER BY t.t_dat, m.customer_idx
        ) TO '{out_transactions}' (FORMAT PARQUET, COMPRESSION 'ZSTD');
    """)
    sampled_trans_count = con.execute(f"SELECT count(*) FROM '{out_transactions}'").fetchone()[0]
    print(f"Saved {sampled_trans_count:,} sampled transactions to {out_transactions}")

    print("\n--- Step 4: Converting Sampled Customers ---")
    con.execute(f"""
        COPY (
            SELECT 
                c.customer_id,
                m.customer_idx,
                c.FN,
                c.Active,
                c.club_member_status,
                c.fashion_news_frequency,
                CAST(c.age AS TINYINT) AS age,
                c.postal_code
            FROM read_csv('{raw_customers}', 
                          types={{'customer_id': 'VARCHAR'}}, 
                          header=True) c
            JOIN sampled_customer_ids m ON c.customer_id = m.customer_id
            ORDER BY m.customer_idx
        ) TO '{out_customers}' (FORMAT PARQUET, COMPRESSION 'ZSTD');
    """)
    saved_cust_count = con.execute(f"SELECT count(*) FROM '{out_customers}'").fetchone()[0]
    print(f"Saved {saved_cust_count:,} customer profiles to {out_customers}")

    print("\n--- Summary Statistics ---")
    stats = con.execute(f"""
        SELECT 
            min(t_dat) AS min_date,
            max(t_dat) AS max_date,
            count(*) AS total_tx,
            count(distinct customer_id) AS distinct_customers,
            count(distinct article_id) AS distinct_articles
        FROM '{out_transactions}'
    """).fetchall()[0]
    print(f"Window span: {stats[0]} to {stats[1]}")
    print(f"Total transactions: {stats[2]:,}")
    print(f"Distinct customers: {stats[3]:,}")
    print(f"Distinct articles: {stats[4]:,}")

    con.close()
    return stats


if __name__ == "__main__":
    run_ingestion()
