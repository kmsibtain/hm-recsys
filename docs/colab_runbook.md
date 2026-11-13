# Google Colab GPU runbook

The notebook needs no path edits. Inputs for both cutoffs are packaged in
`artifacts/colab/two_tower_inputs.zip` on your Windows machine.

1. Open Google Colab and upload `notebooks/two_tower_train.ipynb` via **File > Upload notebook**.
2. Choose **Runtime > Change runtime type**, select a GPU, and save.
3. Choose **Runtime > Run all**. Missing dependencies are installed automatically.
4. When the upload picker appears, select `artifacts/colab/two_tower_inputs.zip`.
5. Leave the browser session connected. Both `train` and `val` run automatically.
6. At completion, the notebook starts downloading `two_tower.zip`. If the browser
   blocks the download, use Colab's Files panel to download `/content/two_tower.zip`.
7. Extract the ZIP **into** the local `artifacts/two_tower/` directory. Its top-level
   folders are already `train/` and `val/`; do not add an extra `two_tower/` level.

Each output folder contains `item_embeddings.npy`, `user_embeddings.npy`,
`id_maps.json`, `train_log.json`, and cached `article_text_embeddings.npy`.
The train cutoff is 2020-09-08; validation is 2020-09-15. Each run evaluates
the final pre-cutoff week, refits all permitted history, then exports vectors.
It uses MiniLM text features and 64-dimensional two-tower embeddings. No images
are used. The local loader verifies cutoff provenance before FAISS retrieval.

Regenerate the upload bundle if the input exports change:

```powershell
.\.venv\Scripts\python.exe -m candidates.prepare_kaggle --split train
.\.venv\Scripts\python.exe -m candidates.prepare_kaggle --split val
.\.venv\Scripts\python.exe -m candidates.prepare_kaggle --bundle
```

After extracting results, run locally:

```powershell
.\.venv\Scripts\python.exe -m candidates.pipeline --split val
.\.venv\Scripts\python.exe -m candidates.report
.\.venv\Scripts\python.exe -m candidates.pipeline --split train
```

Both human-run Colab cutoff exports have passed local Phase 2 checks. For reruns, Colab's temporary
`/content` storage can be lost when the runtime ends; download the results before
disconnecting. Rerunning in the same runtime reuses the extracted inputs but
restarts training. No Google Drive connection or account credentials are needed.
