"""Retrieval contracts on deliberately small, hand-checkable histories."""

import json
from datetime import date

import duckdb
import numpy as np
import polars as pl
import pytest

from candidates.pipeline import (
    als,
    copurchase,
    merge,
    popularity,
    ranked,
    recall,
    repurchase,
    variants,
)
from candidates.two_tower import retrieve


@pytest.fixture
def con():
    connection = duckdb.connect()
    connection.execute("""CREATE TABLE history(customer_idx INTEGER,article_idx INTEGER,t_dat DATE,sales_channel_id INTEGER);
        INSERT INTO history VALUES (0,0,'2020-09-01',1),(0,0,'2020-09-01',1),
        (0,1,'2020-09-15',2),(1,0,'2020-08-01',1),(1,2,'2020-09-15',1);
        CREATE TABLE users(customer_idx INTEGER,bucket INTEGER);
        INSERT INTO users VALUES (0,1),(1,2),(2,-1);
        CREATE TABLE catalog(article_idx INTEGER,product_code VARCHAR,colour_group_code INTEGER);
        INSERT INTO catalog VALUES (0,'p',1),(1,'p',2),(2,'q',1);""")
    connection.execute("""CREATE TABLE recent AS SELECT customer_idx,article_idx,max(t_dat) last_date,
        row_number() OVER(PARTITION BY customer_idx ORDER BY max(t_dat) DESC,article_idx) rn
        FROM history GROUP BY customer_idx,article_idx""")
    yield connection
    connection.close()


def test_repurchase_dedup_and_recency(con, tmp_path):
    path = repurchase(con, "2020-09-15", tmp_path / "repurchase.parquet")
    frame = pl.read_parquet(path)
    assert frame.filter(pl.col("customer_idx") == 0)["article_idx"].to_list() == [1, 0]
    assert frame.select("customer_idx", "article_idx").unique().height == frame.height


def test_popularity_cold_user_and_windows(con, tmp_path):
    paths = popularity(con, "2020-09-15", tmp_path)
    global7 = pl.read_parquet(paths[0])
    assert set(global7["article_idx"]) == {1, 2}
    assert 2 in global7["customer_idx"]
    assert all(pl.read_parquet(p)["rank"].max() <= 50 for p in paths)


def test_variants_exclude_same_item_and_colour(con, tmp_path):
    frame = pl.read_parquet(variants(con, tmp_path / "variants.parquet"))
    assert set(frame.filter(pl.col("customer_idx") == 1)["article_idx"]) == {1}


def test_copurchase_window_and_distinct_customers(con, tmp_path):
    copurchase(con, tmp_path / "copurchase.parquet")
    neighbors = pl.read_parquet(tmp_path / "item_neighbors.parquet")
    assert set(zip(neighbors["seed"], neighbors["article_idx"], strict=True)) == {(0, 1), (1, 0)}
    assert neighbors["score"].to_list() == pytest.approx([1 / np.sqrt(2)] * 2)


def test_merge_provenance_cap_and_recall_with_zero_hits(con, tmp_path):
    a = ranked(
        con, "SELECT 0 customer_idx,0 article_idx,1.0 score", "a", 30, tmp_path / "a.parquet"
    )
    b = ranked(
        con,
        "SELECT 0 customer_idx,0 article_idx,2.0 score UNION ALL SELECT 0,1,1.0",
        "b",
        30,
        tmp_path / "b.parquet",
    )
    out = merge(con, [a, b], tmp_path / "union.parquet", 1)
    row = pl.read_parquet(out).row(0, named=True)
    assert row["article_idx"] == 0
    assert row["source"] == ["a", "b"]
    truth = tmp_path / "truth.parquet"
    pl.DataFrame(
        {"customer_idx": [0, 0, 1], "article_idx": [0, 1, 2], "t_dat": [date(2020, 9, 16)] * 3}
    ).write_parquet(truth)
    assert recall(con, out, "2020-09-16", "2020-09-22", str(truth), 1) == 0.25


def test_two_tower_mapping_and_leakage(con, tmp_path):
    np.save(tmp_path / "item_embeddings.npy", np.eye(3, dtype=np.float32))
    np.save(tmp_path / "user_embeddings.npy", np.array([[0, 1, 0]], dtype=np.float32))
    (tmp_path / "id_maps.json").write_text(
        json.dumps({"article_idx": [2, 0, 1], "customer_idx": [1]})
    )
    log = {"cutoff": "2020-09-15", "max_source_date": "2020-09-15"}
    (tmp_path / "train_log.json").write_text(json.dumps(log))
    result = retrieve(con, tmp_path, "2020-09-15", tmp_path / "tt.parquet")
    assert pl.read_parquet(result).row(0, named=True)["article_idx"] == 0
    log["max_source_date"] = "2020-09-16"
    (tmp_path / "train_log.json").write_text(json.dumps(log))
    with pytest.raises(ValueError, match="after cutoff"):
        retrieve(con, tmp_path, "2020-09-15", tmp_path / "tt.parquet")


def test_als_cpu_schema_and_seen_item_eligibility(con, tmp_path):
    result = als(con, tmp_path / "als.parquet", {"als_iterations": 2})
    frame = pl.read_parquet(result)
    assert set(frame["customer_idx"]) == {0, 1, 2}
    assert set(frame["article_idx"]) == {0, 1, 2}
    assert frame["score"].is_finite().all()
    assert frame.group_by("customer_idx").len()["len"].to_list() == [3] * 3
    assert (tmp_path / "als_model.npz").exists()
