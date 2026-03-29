# EVOO Adulteration AI (FTIR/Raman)

## Authors
- George David Tsitlauri
- Theodora Anna Lasithiotaki

This repository contains a reproducible machine-learning pipeline for olive-oil authenticity and adulteration screening using FTIR and Raman spectroscopy.

## Scope of this project
- FTIR main task: `pure EVOO` (label `0`) vs `adulterated EVOO blends` (label `1`)
- Raman secondary task: `EVOO` (label `0`) vs `non-EVOO oils` (label `1`) on Raman2

Compute is GPU-aware via `--compute {auto,cpu,gpu}` with safe backend tracking in metadata.

## Repository structure

- `ml_pipeline.py`: main training/evaluation pipeline
- `scripts/build_dataset_from_confirmed_labels.py`: legacy helper from earlier dataset phase
- `data/raw_evoo_ftir_raman/`
  - `ATRPure3.csv`
  - `ATRAdulteration3.csv`
  - `Raman1A.csv`
  - `Raman2.csv`
  - `OilClassKey.csv`
- `data/processed_evoo_ftir_raman/`
  - `ftir_evoo_authenticity.csv`
  - `raman1a_evoo_vs_other.csv`
  - `raman2_evoo_vs_other.csv`
  - `dataset_summary.json`

## Environment setup (Python 3.12)

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

Optional GPU extras (XGBoost CUDA path):

```bash
pip install -r requirements-gpu.txt
```

## Final reproducible runs

Run from repository root (`C:\Users\TOM\Documents\FST_DIT`):

```bash
python ml_pipeline.py --mode real --data-path data/processed_evoo_ftir_raman/ftir_evoo_authenticity.csv --compute gpu --output-dir results_final_ftir --cv-folds 5 --random-state 42 --shap-max-samples 80 --shap-nsamples 80

python ml_pipeline.py --mode real --data-path data/processed_evoo_ftir_raman/raman2_evoo_vs_other.csv --compute gpu --output-dir results_final_raman2 --cv-folds 5 --random-state 42 --shap-max-samples 80 --shap-nsamples 80
```

Optional strict GPU-model run:

```bash
python ml_pipeline.py --mode real --data-path data/processed_evoo_ftir_raman/raman2_evoo_vs_other.csv --compute gpu --gpu-models-only --output-dir results_final_raman2_gpuonly --cv-folds 5 --random-state 42 --shap-max-samples 80 --shap-nsamples 80
```

## Output contract

Each run generates:
- `metrics.csv`
- `cv_results.csv`
- `run_metadata.json`
- figures:
  - `01_spectra_<preprocess>.png`
  - `02_pca_<preprocess>.png`
  - `03_mean_spectra_diff.png`
  - `04_model_comparison.png`
  - `05_confusion_matrix.png`
  - `06_shap_bar.png` (when SHAP succeeds)
  - `07_cv_distribution.png`

`metrics.csv` includes `ExecutionBackend` per model.

`run_metadata.json` includes:
- requested/effective compute mode
- GPU probe summary
- fallback events (if any)
- model backend mapping

## Input CSV contract (`--mode real`)

Required:
- One binary `label` column with values `{0,1}`
- Spectral feature columns only (numeric, no missing values)
- Feature names must match one of:
  - `nm_<value>`
  - `wl_<value>`
  - `w_<value>`
  - `<value>`

Examples: `nm_1000`, `nm_1002.14`, `wl_1720`, `2310`

Additional checks:
- Wavelength columns must be strictly increasing
- Both classes must be present

## Latest final run snapshot (2026-03-29)

- FTIR (`results_final_ftir`):
  - samples: `199` (79 pure EVOO, 120 adulterated)
  - features: `6921`
  - best model: `LogisticRegression`
  - test Accuracy/F1/ROC-AUC/MCC: `1.000 / 1.000 / 1.000 / 1.000`

- Raman2 (`results_final_raman2`):
  - samples: `215` (44 EVOO, 171 non-EVOO)
  - features: `1044`
  - best model: `MLP`
  - test Accuracy/F1/ROC-AUC/MCC: `1.000 / 1.000 / 1.000 / 1.000`

## Troubleshooting

- `ModuleNotFoundError`: install dependencies from `requirements.txt`
- `xgboost is not installed`: install `requirements-gpu.txt`
- `Missing required 'label' column`: fix CSV schema
- `Feature naming pattern is invalid`: rename spectral columns to accepted pattern
- SHAP skipped: check `run_metadata.json -> shap.reason`
- GPU fallback occurred: check `run_metadata.json -> compute.fallback_events`

## License

Code is released under the MIT License (see `LICENSE`).
External datasets keep their original licenses and citation requirements.
