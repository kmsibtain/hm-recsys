"""Future purchases must not alter any history-based retrieval source."""

from datetime import date

import duckdb
import polars as pl
import pytest
import yaml

from candidates.pipeline import prepare


@pytest.mark.parametrize("split", ["train", "val"])
def test_source_dates_before_target_week(split):
    config = yaml.safe_load(open("configs/sampling.yaml"))
    assert date.fromisoformat(config["splits"][f"{split}_feature_cutoff"]) < date.fromisoformat(
        config["splits"][f"{split}_start"]
    )


def test_prepare_rejects_invalid_cutoff():
    with pytest.raises(ValueError, match="strictly before"):
        prepare(
            duckdb.connect(),
            {
                "artifacts": {},
                "splits": {"val_start": "2020-09-16", "val_feature_cutoff": "2020-09-16"},
            },
            "val",
        )


@pytest.mark.parametrize("split", ["train", "val"])
def test_future_rows_are_excluded(tmp_path, split):
    settings = yaml.safe_load(open("configs/sampling.yaml"))["splits"]
    cutoff = date.fromisoformat(settings[f"{split}_feature_cutoff"])
    target = date.fromisoformat(settings[f"{split}_start"])
    tx = tmp_path / "transactions.parquet"
    customers = tmp_path / "customers.parquet"
    articles = tmp_path / "articles.parquet"
    pl.DataFrame(
        {
            "customer_idx": [0, 0],
            "article_idx": [0, 1],
            "t_dat": [cutoff, target],
        }
    ).write_parquet(tx)
    pl.DataFrame({"customer_idx": [0], "age": [25]}).write_parquet(customers)
    pl.DataFrame({"article_id": ["001", "002"]}).write_parquet(articles)
    pl.DataFrame({"article_id": ["001", "002"], "article_idx": [0, 1]}).write_parquet(
        tmp_path / "article_id_map.parquet"
    )
    config = {
        "artifacts": {
            "sampled_transactions": str(tx),
            "sampled_customers": str(customers),
            "articles_parquet": str(articles),
            "parquet_dir": str(tmp_path),
        },
        "splits": {f"{split}_start": str(target), f"{split}_feature_cutoff": str(cutoff)},
    }
    con = duckdb.connect()
    prepare(con, config, split)
    assert con.execute("SELECT article_idx FROM history").fetchall() == [(0,)]
    assert con.execute("SELECT article_idx FROM catalog").fetchall() == [(0,)]
    con.close()
