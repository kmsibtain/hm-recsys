# ruff: noqa: E402
"""Run All on Google Colab: upload inputs, train both cutoffs, download exports."""

# %%
import importlib.util
import subprocess
import sys

# Colab supplies most of these; install only missing approved dependencies.
required = {
    "numpy": "numpy",
    "pandas": "pandas",
    "pyarrow": "pyarrow",
    "torch": "torch",
    "sentence_transformers": "sentence-transformers",
}
missing = [
    package for module, package in required.items() if importlib.util.find_spec(module) is None
]
if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", *missing])

# %%
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def discover_inputs(root):
    """Find exactly one complete input folder for each required cutoff."""
    expected = {"train": "2020-09-08", "val": "2020-09-15"}
    found = {}
    required = ("interactions.parquet", "articles_metadata.parquet", "customers.parquet")
    for path in sorted(Path(root).rglob("config.json")):
        try:
            config = json.loads(path.read_text())
        except (ValueError, OSError):
            continue
        split = config.get("split")
        if split not in expected or not all((path.parent / name).is_file() for name in required):
            continue
        if config.get("cutoff") != expected[split]:
            raise ValueError(f"Unexpected {split} cutoff in {path}")
        if split in found:
            raise ValueError(f"Multiple {split} inputs found. Upload only one input bundle.")
        found[split] = path.parent
    absent = set(expected) - set(found)
    if absent:
        raise FileNotFoundError(
            f"Missing complete input folders: {sorted(absent)}. "
            "Upload two_tower_inputs.zip containing BOTH exported train and val folders, "
            "then Run All. No path edits are needed."
        )
    return found


def run_split(INPUT, OUTPUT):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    config = json.loads((INPUT / "config.json").read_text())
    assert torch.cuda.is_available(), "Enable a Colab GPU before running"
    device = torch.device("cuda")
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    events = pd.read_parquet(INPUT / "interactions.parquet").sort_values(
        ["t_dat", "customer_idx", "article_idx"]
    )
    items = pd.read_parquet(INPUT / "articles_metadata.parquet")
    users = pd.read_parquet(INPUT / "customers.parquet")
    events["t_dat"] = pd.to_datetime(events["t_dat"])
    assert events.t_dat.max() <= pd.Timestamp(config["cutoff"])
    item_map = {int(v): i for i, v in enumerate(items.article_idx)}
    user_map = {int(v): i for i, v in enumerate(users.customer_idx)}
    events["item"] = events.article_idx.map(item_map)
    events["user"] = events.customer_idx.map(user_map)
    # Slot zero is exclusively padding; actual items occupy slots 1..N.
    events["item"] += 1
    categories = [c for c in items.columns if c.endswith("_idx") and c != "article_idx"]
    category_ids = torch.tensor(items[categories].to_numpy(), dtype=torch.long, device=device)
    user_features = torch.tensor(
        users[["age_idx", "club_idx"]].to_numpy(), dtype=torch.long, device=device
    )
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda")
    text = encoder.encode(items.text.fillna("").tolist(), batch_size=256, show_progress_bar=True)
    np.save(OUTPUT / "article_text_embeddings.npy", text)
    del encoder
    torch.cuda.empty_cache()
    text = torch.tensor(text, dtype=torch.float32, device=device)

    class TwoTower(nn.Module):
        def __init__(self):
            super().__init__()
            self.item_id = nn.Embedding(len(items) + 1, 64, padding_idx=0)
            self.categories = nn.ModuleList(
                [
                    nn.Embedding(int(category_ids[:, j].max()) + 1, 16)
                    for j in range(len(categories))
                ]
            )
            self.text_projection = nn.Linear(384, 32)
            self.item_projection = nn.Sequential(
                nn.Linear(64 + 16 * len(categories) + 32, 128), nn.ReLU(), nn.Linear(128, 64)
            )
            self.user_id = nn.Embedding(len(users), 64)
            self.age = nn.Embedding(int(user_features[:, 0].max()) + 1, 8)
            self.club = nn.Embedding(int(user_features[:, 1].max()) + 1, 8)
            self.user_projection = nn.Sequential(
                nn.Linear(64 + 8 + 8 + 64, 128), nn.ReLU(), nn.Linear(128, 64)
            )

        def item(self, ids):
            idx = (ids - 1).clamp_min(0)
            parts = [self.item_id(ids), self.text_projection(text[idx])]
            parts += [emb(category_ids[idx, j]) for j, emb in enumerate(self.categories)]
            result = nn.functional.normalize(self.item_projection(torch.cat(parts, dim=-1)), dim=-1)
            return result * (ids != 0).unsqueeze(-1)

        def user(self, ids, history):
            pooled = self.item(history).sum(1) / (history != 0).sum(1, keepdim=True).clamp_min(1)
            attrs = user_features[ids]
            parts = [self.user_id(ids), self.age(attrs[:, 0]), self.club(attrs[:, 1]), pooled]
            return nn.functional.normalize(self.user_projection(torch.cat(parts, dim=-1)), dim=-1)

    def examples(frame):
        # Each positive sees only earlier DAYS, excluding other purchases on its own day.
        result_u, result_i, result_h, result_dates = [], [], [], []
        for user, group in frame.groupby("user", sort=False):
            history = []
            for day_date, day in group.groupby("t_dat", sort=True):
                padded = [0] * (20 - len(history[-20:])) + history[-20:]
                for item in day.item:
                    result_u.append(user)
                    result_i.append(item)
                    result_h.append(padded)
                    result_dates.append(day_date.value)
                history.extend(day.item.tolist())
        order = np.argsort(result_dates, kind="stable")
        return TensorDataset(
            torch.tensor(np.array(result_u)[order], dtype=torch.long),
            torch.tensor(np.array(result_i)[order], dtype=torch.long),
            torch.tensor(np.array(result_h)[order], dtype=torch.long),
        )

    def histories(frame):
        result = np.zeros((len(users), 20), dtype=np.int64)
        for user, group in frame.groupby("user", sort=False):
            last = group.item.to_numpy()[-20:]
            result[user, -len(last) :] = last
        return result

    @torch.no_grad()
    def vectors(model, frame):
        model.eval()
        item_vectors = torch.cat(
            [
                model.item(torch.arange(s, min(s + 2048, len(items) + 1), device=device))
                for s in range(1, len(items) + 1, 2048)
            ]
        )
        history = histories(frame)
        user_vectors = torch.cat(
            [
                model.user(
                    torch.arange(s, min(s + 512, len(users)), device=device),
                    torch.tensor(history[s : s + 512], device=device),
                )
                for s in range(0, len(users), 512)
            ]
        )
        return item_vectors, user_vectors

    @torch.no_grad()
    def evaluate(model, before, heldout):
        item_vectors, user_vectors = vectors(model, before)
        actual = heldout.groupby("user").item.apply(set).to_dict()
        values = []
        for start in range(0, len(actual), 256):
            ids = list(actual)[start : start + 256]
            top = (user_vectors[ids] @ item_vectors.T).topk(
                min(100, len(items)), dim=1
            ).indices.cpu().numpy() + 1
            values += [len(actual[u] & set(pred)) / len(actual[u]) for u, pred in zip(ids, top)]
        return float(np.mean(values))

    def train(frame, heldout=None):
        model = TwoTower().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        dataset = examples(frame)
        loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False, num_workers=0)
        counts = np.bincount(frame.item, minlength=len(items) + 1)
        logq = torch.tensor(
            np.log(np.maximum(counts, 1) / len(frame)), dtype=torch.float32, device=device
        )
        logs = []
        for epoch in range(config["epochs"]):
            model.train()
            losses = []
            for u, i, h in loader:
                u, i, h = u.to(device), i.to(device), h.to(device)
                logits = (
                    model.user(u, h) @ model.item(i).T / config["temperature"] - logq[i][None, :]
                )
                duplicate = i[:, None] == i[None, :]
                duplicate.fill_diagonal_(False)
                logits = logits.masked_fill(duplicate, float("-inf"))
                loss = nn.functional.cross_entropy(logits, torch.arange(len(i), device=device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach()))
            record = {"epoch": epoch + 1, "loss": float(np.mean(losses))}
            if heldout is not None:
                record["recall_at_100"] = evaluate(model, frame, heldout)
            logs.append(record)
            print(record, flush=True)
        return model, logs

    # Last pre-cutoff week is internal evaluation, never the external target week.
    internal_start = pd.Timestamp(config["cutoff"]) - pd.Timedelta(days=6)
    before = events[events.t_dat < internal_start]
    heldout = events[events.t_dat >= internal_start]
    model, evaluation_log = train(before, heldout)
    del model
    torch.cuda.empty_cache()
    # Refit all permitted interactions before exporting cutoff-specific vectors.
    model, refit_log = train(events)
    item_vectors, user_vectors = vectors(model, events)
    np.save(OUTPUT / "item_embeddings.npy", item_vectors.cpu().numpy())
    np.save(OUTPUT / "user_embeddings.npy", user_vectors.cpu().numpy())
    (OUTPUT / "id_maps.json").write_text(
        json.dumps(
            {"article_idx": items.article_idx.tolist(), "customer_idx": users.customer_idx.tolist()}
        )
    )
    (OUTPUT / "train_log.json").write_text(
        json.dumps(
            {
                "cutoff": config["cutoff"],
                "max_source_date": str(events.t_dat.max().date()),
                "internal_holdout_start": str(internal_start.date()),
                "evaluation_epochs": evaluation_log,
                "refit_epochs": refit_log,
                "user_vectors": "precomputed using last 20 pre-cutoff items",
            },
            indent=2,
        )
    )
    print(f"Finished {config['split']}: {OUTPUT}", flush=True)


# %%
def main():
    import gc
    import io
    import shutil
    import zipfile

    from google.colab import files

    if not torch.cuda.is_available():
        raise RuntimeError("In Colab, select Runtime > Change runtime type > T4 GPU, then Run All.")
    input_root = Path("/content/hm_recsys_inputs")
    input_root.mkdir(parents=True, exist_ok=True)
    if not any(input_root.rglob("config.json")):
        print("Select the prepared two_tower_inputs.zip from your computer.", flush=True)
        uploaded = files.upload()
        if len(uploaded) != 1:
            raise ValueError("Upload exactly one file: two_tower_inputs.zip")
        filename, payload = next(iter(uploaded.items()))
        if not filename.lower().endswith(".zip"):
            raise ValueError("Please upload the prepared ZIP input bundle.")
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            root = input_root.resolve()
            for member in archive.infolist():
                destination = (root / member.filename).resolve()
                if not destination.is_relative_to(root):
                    raise ValueError("Input ZIP contains a path outside its destination")
            archive.extractall(root)
        del uploaded, payload
    inputs = discover_inputs(input_root)
    output = Path("/content/two_tower")
    for split in ("train", "val"):
        print(f"Running {split} from {inputs[split]}", flush=True)
        run_split(inputs[split], output / split)
        gc.collect()
        torch.cuda.empty_cache()
    archive = shutil.make_archive("/content/two_tower", "zip", root_dir=output)
    print(f"DONE. Download {archive} and extract it into artifacts/two_tower/ locally.")
    files.download(archive)


if __name__ == "__main__":
    main()
