"""Render measured retrieval results without touching other phase tables."""

import argparse
import json
from pathlib import Path


def render(artifact="artifacts/candidates/val/evaluation.json", output="docs/results.md"):
    report = json.loads(Path(artifact).read_text())
    if report["split"] != "val":
        raise ValueError("Final results must use validation week")
    section = "## 2. Candidate Retrieval: Recall@K (Validation Week V)\n\n"
    section += (
        f"Measured on the sampled dataset with source dates through `{report['max_source_date']}`. "
    )
    section += "Macro recall averages distinct held-out item recall over every validation buyer, including cold and zero-hit customers.\n\n"
    section += "| Generator | K | Recall@K | Incremental recall |\n| :--- | ---: | ---: | ---: |\n"
    for row in report["results"]:
        delta = "—" if row["incremental_recall"] is None else f"{row['incremental_recall']:.6f}"
        section += f"| {row['generator']} | {row['k']} | {row['recall']:.6f} | {delta} |\n"
    if report["two_tower"] != "executed":
        section += "| two_tower | 50 | Pending GPU run | Pending GPU run |\n"
    section += "\nIncremental recall uses an **uncapped** cumulative union in table order, so it measures unique added coverage. "
    section += "The final union is capped at 200 using reciprocal-rank fusion (constant 60); its recall may be lower. "
    section += "Popularity combines six top-50 lists: 7/14-day global, age-bucket and preferred-channel popularity, fused to 200. "
    section += "Co-purchase takes up to 20 neighbors for each of the last five distinct items. "
    section += "All generators permit previously purchased items. The union retains per-source scores and ranks.\n\n"
    section += "Machine: Windows 11, Intel Core i7-7600U, four logical CPUs; DuckDB capped at 4 GB and four threads, ALS 64 factors / 15 iterations. "
    section += "Artifacts: `artifacts/candidates/val/`, including `evaluation.json`, `evaluation.parquet`, source Parquets and `union.parquet`.\n\n"
    if report["two_tower"] != "executed":
        section += "**Phase 2 acceptance remains pending** the human Colab run and the measured two-tower/final-union comparison. See [the Colab runbook](colab_runbook.md).\n\n"
    else:
        section += "All six generators were evaluated locally on V, including FAISS retrieval from the human-run Colab exports. Notebook internal Recall@100 is a separate metric and is not substituted for this table. See [the Phase 2 acceptance audit](phase2_acceptance.md).\n\n"
    path = Path(output)
    text = path.read_text(encoding="utf-8")
    start = text.index("## 2.")
    end = text.index("## 3.", start)
    path.write_text(text[:start] + section + text[end:], encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", default="artifacts/candidates/val/evaluation.json")
    args = parser.parse_args()
    render(args.artifact)
