"""Repurchase candidate generator.

Retrieves distinct articles previously purchased by each customer prior to the cutoff date,
scored by recency-weighted frequency. Returns up to `max_candidates` items per customer.
"""

from pathlib import Path
import duckdb


def generate_repurchase_candidates(
    con: duckdb.DuckDBPyConnection,
    tx_path: str,
    cutoff_date: str,
    output_path: str | None = None,
    customer_indices: list[int] | None = None,
    max_candidates: int = 30,
) -> str:
    """Generates repurchase candidates.

    Args:
        con: Active DuckDB connection.
        tx_path: Path to transactions Parquet file.
        cutoff_date: Date string (YYYY-MM-DD); data strictly <= cutoff_date is used.
        output_path: Destination path for candidates Parquet file.
        customer_indices: Optional list of customer_idx to restrict candidate generation to.
        max_candidates: Maximum items per customer (default 30).

    Returns:
        output_path where candidate Parquet was written.
    """
    if output_path is None:
        output_path = f"artifacts/parquet/candidates_repurchase_{cutoff_date}.parquet"

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    cust_filter_sql = ""
    if customer_indices is not None:
        con.register("target_cust_filter", duckdb.from_df({"customer_idx": customer_indices}))
        cust_filter_sql = "JOIN target_cust_filter f ON t.customer_idx = f.customer_idx"

    query = f"""
        COPY (
            WITH scored_items AS (
                SELECT 
                    t.customer_idx,
                    t.article_idx,
                    'repurchase' AS source,
                    CAST(
                        sum(1.0 / sqrt(datediff('day', t.t_dat, DATE '{cutoff_date}') + 1.0))
                        AS FLOAT
                    ) AS score
                FROM '{tx_path}' t
                {cust_filter_sql}
                WHERE t.t_dat <= DATE '{cutoff_date}'
                GROUP BY t.customer_idx, t.article_idx
            ),
            ranked_items AS (
                SELECT 
                    customer_idx,
                    article_idx,
                    source,
                    score,
                    CAST(row_number() OVER (PARTITION BY customer_idx ORDER BY score DESC, article_idx) AS INTEGER) AS rank
                FROM scored_items
            )
            SELECT 
                customer_idx,
                article_idx,
                source,
                score,
                rank
            FROM ranked_items
            WHERE rank <= {max_candidates}
            ORDER BY customer_idx, rank
        ) TO '{output_path}' (FORMAT PARQUET, COMPRESSION 'ZSTD');
    """
    con.execute(query)
    return output_path
