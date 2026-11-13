# Phase 2 acceptance audit

**Passed on 2026-11-13.** The handover plan's Phase 2 acceptance criterion is met:
`docs/results.md` contains measured validation Recall@K for all six generators and
the union, plus each generator's incremental contribution. No Phase 3 work was started.

| Check | Evidence | Result |
| --- | --- | --- |
| Both GPU cutoff exports | R: 2020-09-08; V: 2020-09-15, both before their target weeks | Pass |
| ID coverage | Exactly 99,338 sampled users in each export; all 49,550 R and 50,287 V pre-cutoff-active articles | Pass |
| Embedding contract | User/item vectors have 64 dimensions, text vectors 384; finite values, unique ID maps, unit-normalized user/item vectors | Pass |
| V retrieval | Six source groups plus capped union measured on all 6,837 validation-week buyers | Pass |
| Candidate counts | 19,844,995 rows, 99,338 customers, 164?200 candidates per customer | Pass |
| Dedupe and ranks | Full-file checks found zero repeated pairs, duplicate ranks or non-contiguous ranks | Pass |
| Source traceability | Full-file checks found zero missing/inconsistent source lists, score maps or rank maps | Pass |
| FAISS export | 4,966,900 candidates, exactly 50 per sampled customer; finite scores | Pass |
| Tests | 28 tests passed using the Windows virtual environment | Pass |
| Code quality | Scoped Ruff lint and format checks passed for Phase 2 code/tests/notebook | Pass |

Two-tower Recall@50 on V is **3.2601%**. It adds
**0.8616 percentage points** to the uncapped
union in the documented source order. The capped union improves from
**18.5547% to 18.7523% Recall@200**
(**+0.1976 percentage points**). These are recall values,
not MAP@12, and do not establish ranker quality.

The Colab logs report internal Recall@100 of 0.056215 for R and 0.057828 for V's
pre-cutoff holdout. Those human-run training diagnostics are not the final V
retrieval metric. Provenance checks use the returned logs and exact matches to
local pre-cutoff ID sets; the agent executed FAISS retrieval and final evaluation locally.

Evidence retained under ignored artifacts:

- `artifacts/two_tower/validation_audit.json`: shapes, cutoff checks, ID coverage and SHA-256 hashes of returned files.
- `artifacts/candidates/val/evaluation.json` and `evaluation.parquet`: final source and union metrics.
- `artifacts/candidates/val/evaluation_cpu_reference.json`: earlier measured CPU-only comparison.
- `artifacts/candidates/val/phase2_acceptance.json`: full candidate integrity-check results.
- `artifacts/candidates/val/union.parquet`: complete candidate pool with provenance.

Reproduce the main checks from PowerShell:

```powershell
.\.venv\Scripts\python.exe -m candidates.pipeline --split val
.\.venv\Scripts\python.exe -m candidates.report
.\.venv\Scripts\python.exe -m pytest tests -p no:cacheprovider
```

Scope: the full candidate stage was executed on V; both R/V GPU exports were
validated. The R candidate stage is available via `make candidates-train` and has
not been run yet. Repository-wide lint/CI was not certified: unrelated existing
worktree edits in `notebooks/01_eda.ipynb` and the untracked `candidates/repurchase.py`
were left untouched. They are excluded from the Phase 2 change scope.

The phase commit has not been created: writing the Git index required elevated permission, and the staging request was declined. Acceptance checks and documentation are complete; changes remain in the worktree.
