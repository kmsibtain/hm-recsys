"""Point-in-time candidate retrieval. Run with ``python -m candidates.pipeline``."""

import argparse
import json
import os
import tempfile
from datetime import date
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import duckdb
import numpy as np
import polars as pl
import yaml


def literal(value):
    return "'" + str(value).replace("'", "''") + "'"


def write(con, query, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    con.execute(f"COPY ({query}) TO {literal(path.as_posix())} (FORMAT PARQUET, COMPRESSION ZSTD)")
    return path


def ranked(con, query, source, k, path):
    return write(
        con,
        f"""
        SELECT customer_idx, article_idx, {literal(source)} source, score,
            row_number() OVER (PARTITION BY customer_idx ORDER BY score DESC, article_idx)::INTEGER rank
        FROM ({query}) q QUALIFY rank <= {int(k)}
    """,
        path,
    )


def prepare(con, config, split):
    a, s = config["artifacts"], config["splits"]
    cutoff = s[f"{split}_feature_cutoff"]
    start = s[f"{split}_start"]
    if date.fromisoformat(cutoff) >= date.fromisoformat(start):
        raise ValueError("Candidate cutoff must be strictly before target week")
    con.execute(
        f"CREATE OR REPLACE TEMP TABLE history AS SELECT * FROM {literal(a['sampled_transactions'])} WHERE t_dat <= DATE {literal(cutoff)}"
    )
    # All sampled customers, never a cohort selected using held-out purchases.
    con.execute(f"""CREATE OR REPLACE TEMP TABLE users AS SELECT customer_idx,
        CASE WHEN age IS NULL THEN -1 WHEN age < 20 THEN 0 ELSE least(age//10-1,5) END bucket
        FROM {literal(a["sampled_customers"])}""")
    con.execute(f"""CREATE OR REPLACE TEMP TABLE catalog AS SELECT m.article_idx,a.*
        FROM {literal(a["articles_parquet"])} a
        JOIN {literal(str(Path(a["parquet_dir"]) / "article_id_map.parquet"))} m USING(article_id)
        WHERE m.article_idx IN (SELECT article_idx FROM history)""")
    con.execute("""CREATE OR REPLACE TEMP TABLE recent AS SELECT customer_idx,article_idx,max(t_dat) last_date,
        row_number() OVER(PARTITION BY customer_idx ORDER BY max(t_dat) DESC,article_idx) rn
        FROM history GROUP BY customer_idx,article_idx""")
    return cutoff


def repurchase(con, cutoff, out):
    return ranked(
        con,
        f"""SELECT customer_idx, article_idx,
        sum(1.0/sqrt(date_diff('day',t_dat,DATE {literal(cutoff)})+1)) score
        FROM history GROUP BY customer_idx, article_idx""",
        "repurchase",
        30,
        out,
    )


def popularity(con, cutoff, out):
    paths = []
    for days in (7, 14):
        sales = (
            f"SELECT * FROM history WHERE t_dat >= DATE {literal(cutoff)} - INTERVAL {days - 1} DAY"
        )
        for segment in ("global", "age", "channel"):
            source = f"popular_{segment}_{days}d"
            if segment == "global":
                query = f"SELECT u.customer_idx,p.article_idx,p.score FROM users u CROSS JOIN (SELECT article_idx,count(*) score FROM ({sales}) GROUP BY article_idx ORDER BY score DESC,article_idx LIMIT 50) p"
            elif segment == "age":
                query = f"""WITH p AS (SELECT bucket,article_idx,count(*) score,
                    row_number() OVER(PARTITION BY bucket ORDER BY count(*) DESC,article_idx) rn
                    FROM ({sales}) h JOIN users USING(customer_idx) GROUP BY bucket,article_idx)
                    SELECT u.customer_idx,p.article_idx,p.score FROM users u JOIN p USING(bucket) WHERE rn<=50"""
            else:
                query = f"""WITH preferred AS (SELECT customer_idx,sales_channel_id,
                    row_number() OVER(PARTITION BY customer_idx ORDER BY count(*) DESC,sales_channel_id) rn
                    FROM history GROUP BY customer_idx,sales_channel_id),
                    p AS (SELECT sales_channel_id,article_idx,count(*) score,
                    row_number() OVER(PARTITION BY sales_channel_id ORDER BY count(*) DESC,article_idx) rn
                    FROM ({sales}) GROUP BY sales_channel_id,article_idx)
                    SELECT u.customer_idx,p.article_idx,p.score FROM preferred u JOIN p USING(sales_channel_id)
                    WHERE u.rn=1 AND p.rn<=50"""
            paths.append(ranked(con, query, source, 50, out / f"{source}.parquet"))
    return paths


def variants(con, out):
    return ranked(
        con,
        """SELECT r.customer_idx,b.article_idx,sum(1.0/r.rn) score
        FROM recent r JOIN catalog a ON r.article_idx=a.article_idx
        JOIN catalog b ON a.product_code=b.product_code AND a.article_idx<>b.article_idx
        AND a.colour_group_code<>b.colour_group_code
        GROUP BY r.customer_idx,b.article_idx""",
        "variants",
        30,
        out,
    )


def copurchase(con, out):
    # Count distinct customers per pair, avoiding multi-unit transaction inflation.
    con.execute("""CREATE OR REPLACE TEMP TABLE neighbors AS
        WITH events AS (SELECT DISTINCT customer_idx,article_idx,t_dat FROM history),
        counts AS (SELECT a.article_idx seed,b.article_idx,count(DISTINCT a.customer_idx) n
            FROM events a JOIN events b ON a.customer_idx=b.customer_idx
            AND a.article_idx<>b.article_idx AND abs(date_diff('day',a.t_dat,b.t_dat))<=14
            GROUP BY a.article_idx,b.article_idx),
        freq AS (SELECT article_idx,count(DISTINCT customer_idx) n FROM history GROUP BY article_idx)
        SELECT c.seed,c.article_idx,c.n/sqrt(a.n::DOUBLE*b.n) score,
            row_number() OVER(PARTITION BY seed ORDER BY score DESC,c.article_idx) rn
        FROM counts c JOIN freq a ON c.seed=a.article_idx JOIN freq b ON c.article_idx=b.article_idx
        QUALIFY rn<=20""")
    write(con, "SELECT * FROM neighbors", out.parent / "item_neighbors.parquet")
    return ranked(
        con,
        """SELECT r.customer_idx,n.article_idx,sum(n.score/r.rn) score
        FROM recent r JOIN neighbors n ON r.article_idx=n.seed WHERE r.rn<=5
        GROUP BY r.customer_idx,n.article_idx""",
        "copurchase",
        100,
        out,
    )


def als(con, out, settings):
    from implicit.cpu.als import AlternatingLeastSquares
    from scipy.sparse import csr_matrix
    from threadpoolctl import threadpool_limits

    triples = con.execute(
        "SELECT customer_idx,article_idx,count(*) FROM history GROUP BY customer_idx,article_idx"
    ).fetchnumpy()
    rows, cols, counts = triples.values()
    n_users = con.execute("SELECT coalesce(max(customer_idx),-1)+1 FROM users").fetchone()[0]
    n_items = con.execute("SELECT coalesce(max(article_idx),-1)+1 FROM catalog").fetchone()[0]
    matrix = csr_matrix(
        (1 + np.log1p(counts), (rows, cols)), shape=(n_users, n_items), dtype=np.float32
    )
    with threadpool_limits(limits=1, user_api="blas"):
        model = AlternatingLeastSquares(
            factors=64,
            iterations=settings.get("als_iterations", 15),
            regularization=0.05,
            random_state=42,
            num_threads=4,
        )
        model.fit(matrix, show_progress=False)
    model.save(str(out.parent / "als_model.npz"))
    active_items = np.unique(cols).astype(np.int32)
    parts = []
    for batch in (
        con.execute("SELECT customer_idx FROM users ORDER BY customer_idx").fetchnumpy().values()
    ):
        for chunk in np.array_split(batch, max(1, (len(batch) + 511) // 512)):
            if not len(chunk):
                continue
            with threadpool_limits(limits=1, user_api="blas"):
                ids, scores = model.recommend(
                    chunk,
                    matrix[chunk],
                    N=min(50, len(active_items)),
                    filter_already_liked_items=False,
                    items=active_items,
                )
            parts.append(
                pl.DataFrame(
                    {
                        "customer_idx": np.repeat(chunk, ids.shape[1]),
                        "article_idx": ids.ravel(),
                        "source": "als",
                        "score": scores.ravel(),
                        "rank": np.tile(np.arange(1, ids.shape[1] + 1), len(chunk)),
                    }
                )
            )
    pl.concat(parts).write_parquet(out)
    return out


def merge(con, paths, out, k=200):
    files = "[" + ",".join(literal(p.as_posix()) for p in paths) + "]"
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    maximum = con.execute(
        f"SELECT coalesce(max(customer_idx),0) FROM read_parquet({files})"
    ).fetchone()[0]
    # Bound memory used by the nested provenance aggregates on the 16 GB laptop.
    with tempfile.TemporaryDirectory(prefix="merge_", dir=out.parent) as temporary:
        parts = []
        for start in range(0, int(maximum) + 1, 2000):
            parts.append(
                write(
                    con,
                    f"""WITH grouped AS (SELECT customer_idx,article_idx,
        list(source ORDER BY source) source,
        map(list(source ORDER BY source),list(score::DOUBLE ORDER BY source)) source_scores,
        map(list(source ORDER BY source),list(rank::INTEGER ORDER BY source)) source_ranks,
        sum(1.0/(60+rank)) score FROM read_parquet({files})
        WHERE customer_idx BETWEEN {start} AND {start + 1999} GROUP BY customer_idx,article_idx)
        SELECT *,row_number() OVER(PARTITION BY customer_idx ORDER BY score DESC,article_idx)::INTEGER rank
        FROM grouped QUALIFY rank<={int(k)}""",
                    Path(temporary) / f"{start}.parquet",
                )
            )
        part_files = "[" + ",".join(literal(p.as_posix()) for p in parts) + "]"
        return write(con, f"SELECT * FROM read_parquet({part_files})", out)


def recall(con, path, start, end, tx, k):
    # Macro recall over all target-week buyers, including cold users and zero-hit users.
    return con.execute(f"""WITH truth AS (SELECT DISTINCT customer_idx,article_idx
        FROM {literal(tx)} WHERE t_dat BETWEEN DATE {literal(start)} AND DATE {literal(end)}),
        pred AS (SELECT customer_idx,article_idx FROM {literal(str(path))} WHERE rank<={int(k)})
        SELECT avg(r) FROM (SELECT t.customer_idx,count(p.article_idx)::DOUBLE/count(*) r
        FROM truth t LEFT JOIN pred p USING(customer_idx,article_idx) GROUP BY t.customer_idx)""").fetchone()[
        0
    ]


def evaluation_pool(con, paths, out, k):
    files = "[" + ",".join(literal(p.as_posix()) for p in paths) + "]"
    return ranked(
        con,
        f"""SELECT p.customer_idx,article_idx,sum(1.0/(60+rank)) score
        FROM read_parquet({files}) p JOIN eval_users u USING(customer_idx)
        GROUP BY p.customer_idx,article_idx""",
        "evaluation",
        k,
        out,
    )


def run(config_path="configs/sampling.yaml", split="val"):
    config = yaml.safe_load(Path(config_path).read_text())
    settings = config.get("candidates", {})
    out = Path(settings.get("output_dir", "artifacts/candidates")) / split
    out.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("SET memory_limit='4GB'")
    con.execute("SET threads=4")
    con.execute(f"SET temp_directory={literal((out / 'duckdb_tmp').as_posix())}")
    cutoff = prepare(con, config, split)
    groups = {}
    for name, generate in [
        ("repurchase", lambda: [repurchase(con, cutoff, out / "repurchase.parquet")]),
        ("popularity", lambda: popularity(con, cutoff, out)),
        ("variants", lambda: [variants(con, out / "variants.parquet")]),
        ("copurchase", lambda: [copurchase(con, out / "copurchase.parquet")]),
        ("als", lambda: [als(con, out / "als.parquet", settings)]),
    ]:
        print(f"Generating {split}: {name}", flush=True)
        groups[name] = generate()
    from candidates.two_tower import retrieve

    embedding_dir = Path(settings.get("two_tower_dir", "artifacts/two_tower")) / split
    if (embedding_dir / "train_log.json").exists():
        groups["two_tower"] = [retrieve(con, embedding_dir, cutoff, out / "two_tower.parquet")]
    s = config["splits"]
    con.execute(f"""CREATE TEMP TABLE eval_users AS SELECT DISTINCT customer_idx
        FROM {literal(config["artifacts"]["sampled_transactions"])}
        WHERE t_dat BETWEEN DATE {literal(s[f"{split}_start"])} AND DATE {literal(s[f"{split}_end"])}""")
    rows, cumulative, previous = [], [], 0.0
    for name, paths in groups.items():
        print(f"Evaluating {split}: {name}", flush=True)
        standalone = evaluation_pool(con, paths, out / f"eval_{name}.parquet", 200)
        k = (
            200
            if name == "popularity"
            else (
                100 if name == "copurchase" else (30 if name in ("repurchase", "variants") else 50)
            )
        )
        value = recall(
            con,
            standalone,
            s[f"{split}_start"],
            s[f"{split}_end"],
            config["artifacts"]["sampled_transactions"],
            k,
        )
        cumulative.extend(paths)
        union = evaluation_pool(con, cumulative, out / "uncapped_union.parquet", 10000)
        total = recall(
            con,
            union,
            s[f"{split}_start"],
            s[f"{split}_end"],
            config["artifacts"]["sampled_transactions"],
            10000,
        )
        rows.append(
            {"generator": name, "k": k, "recall": value, "incremental_recall": total - previous}
        )
        previous = total
    print(f"Merging {split} candidates in customer batches", flush=True)
    union = merge(con, cumulative, out / "union.parquet", settings.get("max_candidates", 200))
    rows.append(
        {
            "generator": "union",
            "k": settings.get("max_candidates", 200),
            "recall": recall(
                con,
                union,
                s[f"{split}_start"],
                s[f"{split}_end"],
                config["artifacts"]["sampled_transactions"],
                settings.get("max_candidates", 200),
            ),
            "incremental_recall": None,
        }
    )
    report = {
        "split": split,
        "cutoff": cutoff,
        "max_source_date": str(con.execute("SELECT max(t_dat) FROM history").fetchone()[0]),
        "two_tower": "executed" if "two_tower" in groups else "pending GPU handoff; ALS fallback",
        "results": rows,
    }
    (out / "evaluation.json").write_text(json.dumps(report, indent=2))
    pl.DataFrame(rows).write_parquet(out / "evaluation.parquet")
    con.close()
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/sampling.yaml")
    parser.add_argument("--split", choices=["train", "val"], default="val")
    args = parser.parse_args()
    run(args.config, args.split)
