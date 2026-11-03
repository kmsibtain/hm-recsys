# Kaggle / Colab GPU Handoff Runbook

This guide describes how to run the Two-Tower embedding model on a free Kaggle GPU notebook and return the trained vectors to the local repository.

## 1. Local Preparation
The local pipeline prepares inputs under `artifacts/kaggle_input/`:
- `interactions.parquet` (sampled customer idx, article idx, timestamp)
- `articles_metadata.parquet` (article idx, categoricals, projected text features)
- `config.json` (cutoff date, model hyperparameters)

## 2. Uploading Data to Kaggle
1. Create a new private dataset on Kaggle:
   - Title: `hm-recsys-two-tower-inputs`
   - Files: upload everything in `artifacts/kaggle_input/`
2. Create a Kaggle Notebook:
   - File -> Upload Notebook -> Select `notebooks/two_tower_train.ipynb`
   - Settings -> Accelerator: **GPU P100** (or T4 x2)
   - Add Data -> Select your uploaded `hm-recsys-two-tower-inputs` dataset

## 3. Running Training
1. Run all cells in `two_tower_train.ipynb`.
2. Model trains in-batch sampled softmax for 5–10 epochs.
3. Verification: Check the final cell outputs:
   - `loss` decreasing across epochs
   - Validation `recall@100` logged

## 4. Returning Artifacts
Download generated outputs from `/kaggle/working/` and place them locally into `artifacts/two_tower/`:
- `item_embeddings.npy`
- `user_embeddings.npy`
- `id_maps.json`
- `train_log.json`

## 5. Local Ingestion
Run local verification:
```bash
python -m candidates.two_tower --verify
```
If GPU training is unavailable, the candidate generation pipeline falls back to CPU ALS retrieval (`implicit`).

