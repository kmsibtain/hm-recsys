"""Validate cutoff-specific GPU exports and retrieve via CPU FAISS."""

import json
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl


def retrieve(con, directory, cutoff, output):
    import faiss

    directory = Path(directory)
    log = json.loads((directory / "train_log.json").read_text())
    if date.fromisoformat(log["max_source_date"]) > date.fromisoformat(cutoff):
        raise ValueError("Two-tower embeddings include data after cutoff")
    if log["cutoff"] != cutoff:
        raise ValueError("Two-tower cutoff does not match candidate cutoff")
    maps = json.loads((directory / "id_maps.json").read_text())
    items = np.load(directory / "item_embeddings.npy", allow_pickle=False)
    users = np.load(directory / "user_embeddings.npy", allow_pickle=False)
    item_ids = np.asarray(maps["article_idx"], dtype=np.int32)
    user_ids = np.asarray(maps["customer_idx"], dtype=np.int32)
    if (
        items.ndim != 2
        or users.ndim != 2
        or items.shape[1] != users.shape[1]
        or len(items) != len(item_ids)
        or len(users) != len(user_ids)
        or len(set(item_ids)) != len(item_ids)
        or len(set(user_ids)) != len(user_ids)
        or not np.isfinite(items).all()
        or not np.isfinite(users).all()
    ):
        raise ValueError("Invalid embedding shapes, IDs or values")
    active = set(con.execute("SELECT article_idx FROM catalog").fetchnumpy()["article_idx"])
    known = set(con.execute("SELECT customer_idx FROM users").fetchnumpy()["customer_idx"])
    if not set(user_ids).issubset(known):
        raise ValueError("Embedding customer IDs are not in the local map")
    keep = np.array([i in active for i in item_ids])
    items = np.ascontiguousarray(items[keep], dtype=np.float32)
    item_ids = item_ids[keep]
    if not len(items):
        raise ValueError("No active items in embedding export")
    faiss.normalize_L2(items)
    index = faiss.IndexFlatIP(items.shape[1])
    index.add(items)
    faiss.omp_set_num_threads(4)
    frames = []
    for start in range(0, len(users), 512):
        batch = np.ascontiguousarray(users[start : start + 512], dtype=np.float32)
        faiss.normalize_L2(batch)
        scores, positions = index.search(batch, min(50, len(items)))
        frames.append(
            pl.DataFrame(
                {
                    "customer_idx": np.repeat(
                        user_ids[start : start + len(batch)], positions.shape[1]
                    ),
                    "article_idx": item_ids[positions.ravel()],
                    "source": "two_tower",
                    "score": scores.ravel(),
                    "rank": np.tile(np.arange(1, positions.shape[1] + 1), len(batch)),
                }
            )
        )
    pl.concat(frames).write_parquet(output)
    return Path(output)
