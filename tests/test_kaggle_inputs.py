"""Validate notebook input discovery without installing GPU packages locally."""

import ast
import io
import json
import sys
import types
import zipfile
from pathlib import Path

import pytest


def discovery():
    tree = ast.parse(Path("notebooks/two_tower_train.py").read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "discover_inputs"
    )
    namespace = {"Path": Path, "json": json}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "discovery", "exec"), namespace)
    return namespace["discover_inputs"]


def folder(root, split, cutoff):
    path = root / split
    path.mkdir(parents=True)
    (path / "config.json").write_text(json.dumps({"split": split, "cutoff": cutoff}))
    for filename in ("interactions.parquet", "articles_metadata.parquet", "customers.parquet"):
        (path / filename).touch()
    return path


def test_discovers_both_cutoffs_under_arbitrary_dataset_name(tmp_path):
    root = tmp_path / "datasets" / "arbitrary-slug" / "nested"
    train = folder(root, "train", "2020-09-08")
    val = folder(root, "val", "2020-09-15")
    assert discovery()(tmp_path) == {"train": train, "val": val}


def test_missing_input_explains_attachment(tmp_path):
    with pytest.raises(FileNotFoundError, match="No path edits"):
        discovery()(tmp_path)


def test_duplicate_dataset_is_rejected(tmp_path):
    folder(tmp_path / "a", "train", "2020-09-08")
    folder(tmp_path / "b", "train", "2020-09-08")
    with pytest.raises(ValueError, match="Multiple train"):
        discovery()(tmp_path)


def test_wrong_cutoff_is_rejected(tmp_path):
    folder(tmp_path, "train", "2020-09-15")
    with pytest.raises(ValueError, match="Unexpected train cutoff"):
        discovery()(tmp_path)


def test_colab_upload_runs_both_cutoffs_and_downloads_zip(tmp_path, monkeypatch):
    """Exercise the Colab orchestration with training and browser helpers stubbed."""
    bundle = io.BytesIO()
    with zipfile.ZipFile(bundle, "w") as archive:
        for split, cutoff in (("train", "2020-09-08"), ("val", "2020-09-15")):
            archive.writestr(f"{split}/config.json", json.dumps({"split": split, "cutoff": cutoff}))
            for name in ("interactions.parquet", "articles_metadata.parquet", "customers.parquet"):
                archive.writestr(f"{split}/{name}", b"fixture")
    downloaded, runs = [], []
    colab = types.ModuleType("google.colab")
    colab.files = types.SimpleNamespace(
        upload=lambda: {"inputs.zip": bundle.getvalue()}, download=downloaded.append
    )
    monkeypatch.setitem(sys.modules, "google.colab", colab)

    def fake_run(source, output):
        output.mkdir(parents=True)
        (output / "train_log.json").write_text("{}")
        runs.append(source.name)

    def local_path(path):
        return tmp_path / str(path).removeprefix("/content/")

    # main also calls shutil.make_archive with an absolute Colab output path.
    import shutil

    real_archive = shutil.make_archive
    monkeypatch.setattr(
        shutil,
        "make_archive",
        lambda base, fmt, root_dir: real_archive(str(local_path(base)), fmt, root_dir=root_dir),
    )
    namespace = {
        "Path": local_path,
        "discover_inputs": discovery(),
        "run_split": fake_run,
        "torch": types.SimpleNamespace(
            cuda=types.SimpleNamespace(is_available=lambda: True, empty_cache=lambda: None)
        ),
    }
    tree = ast.parse(Path("notebooks/two_tower_train.py").read_text())
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    exec(compile(ast.Module(body=[function], type_ignores=[]), "colab-main", "exec"), namespace)
    namespace["main"]()
    assert runs == ["train", "val"]
    assert len(downloaded) == 1
    with zipfile.ZipFile(downloaded[0]) as archive:
        assert {"train/train_log.json", "val/train_log.json"}.issubset(archive.namelist())
