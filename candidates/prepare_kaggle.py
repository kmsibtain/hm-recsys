"""Export leakage-safe, cutoff-specific GPU inputs without account actions."""

import argparse
import json
import zipfile
from pathlib import Path

import duckdb
import yaml

from candidates.pipeline import literal, prepare, write


def export(split, config_path="configs/sampling.yaml"):
    config = yaml.safe_load(Path(config_path).read_text())
    con = duckdb.connect()
    con.execute("SET memory_limit='4GB'")
    con.execute("SET threads=4")
    cutoff = prepare(con, config, split)
    out = Path("artifacts/kaggle_input") / split
    out.mkdir(parents=True, exist_ok=True)
    write(
        con,
        "SELECT customer_idx,article_idx,t_dat FROM history ORDER BY t_dat,customer_idx,article_idx",
        out / "interactions.parquet",
    )
    categories = [
        "product_type_no",
        "colour_group_code",
        "department_no",
        "index_code",
        "garment_group_no",
    ]
    codes = ",".join(
        f"dense_rank() OVER(ORDER BY {c} NULLS FIRST)::INTEGER-1 AS {c}_idx" for c in categories
    )
    write(
        con,
        f"SELECT article_idx,{codes},concat_ws(' ',prod_name,product_type_name,colour_group_name,detail_desc) text FROM catalog ORDER BY article_idx",
        out / "articles_metadata.parquet",
    )
    write(
        con,
        f"""SELECT u.customer_idx,u.bucket+1 age_idx,
        dense_rank() OVER(ORDER BY c.club_member_status NULLS FIRST)::INTEGER-1 club_idx
        FROM users u JOIN {literal(config["artifacts"]["sampled_customers"])} c USING(customer_idx)
        ORDER BY customer_idx""",
        out / "customers.parquet",
    )
    (out / "config.json").write_text(
        json.dumps(
            {
                "cutoff": cutoff,
                "max_source_date": str(con.execute("SELECT max(t_dat) FROM history").fetchone()[0]),
                "split": split,
                "seed": 42,
                "epochs": 5,
                "batch_size": 4096,
                "temperature": 0.05,
            },
            indent=2,
        )
    )
    con.close()
    print(f"Prepared {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument(
        "--bundle", action="store_true", help="Package both existing exports for Colab"
    )
    args = parser.parse_args()
    if args.bundle:
        destination = Path("artifacts/colab/two_tower_inputs.zip")
        destination.parent.mkdir(parents=True, exist_ok=True)
        inputs = Path("artifacts/kaggle_input")
        names = (
            "config.json",
            "interactions.parquet",
            "articles_metadata.parquet",
            "customers.parquet",
        )
        files = [inputs / split / name for split in ("train", "val") for name in names]
        if not all(path.is_file() for path in files):
            raise FileNotFoundError("Export both train and val cutoffs before creating the bundle")
        with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in files:
                archive.write(path, path.relative_to(inputs).as_posix())
        print(f"Upload this file when Colab prompts: {destination.resolve()}")
    else:
        export(args.split)
