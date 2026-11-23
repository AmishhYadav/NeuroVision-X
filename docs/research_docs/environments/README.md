# Environment snapshots (MasterPlan W7.2)

Exact package versions of the three local virtual environments, taken with `uv pip freeze` on
2026-11-24 (macOS, Apple M4, **Python 3.11.15** in all three). Regenerate the same way after any
dependency change and commit the diff with that change.

| File | Environment | Built from | Used for |
|---|---|---|---|
| `venv_freeze.txt` | `.venv` | `requirements.txt` + `app/backend/requirements.txt` + `-e .` | training code, tests, most analysis |
| `venv-analysis_freeze.txt` | `.venv-analysis` | `requirements-analysis.txt` | lesion-wise metrics (`panoptica`), `statsmodels`, `nnunetv2` glue |
| `venv-clinical_freeze.txt` | `.venv-clinical` | `requirements-clinical.txt` | DICOM ingest, registration, HD-BET, the clinical pipeline |

These are records, not install files: the requirement files stay the source of truth. The W7.2
"done when" (fresh-clone rehearsal green) is still open.
