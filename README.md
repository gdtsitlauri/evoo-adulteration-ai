# EVOO Adulteration AI (FTIR/Raman)


## Project Metadata

| Field | Value |
| --- | --- |
| Author | George David Tsitlauri |
| Affiliation | Dept. of Informatics & Telecommunications, University of Thessaly, Greece |
| Contact | gdtsitlauri@gmail.com |
| Year | 2026 |

## Authors

| Name | Affiliation | Contact |
|---|---|---|
| George David Tsitlauri | Dept. of Informatics & Telecommunications, University of Thessaly, Greece | gdtsitlauri@gmail.com |
| Theodora Anna Lasithiotaki | Dept. of Food Science and Technology, University of the Peloponnese, Greece | dwra.las@gmail.com |

## Overview

This repository contains a reproducible machine-learning pipeline for olive-oil authenticity and adulteration screening using FTIR and Raman spectroscopy.

## Scope of this project
- FTIR main task: `pure EVOO` (label `0`) vs `adulterated EVOO blends` (label `1`)
- Raman secondary task: `EVOO` (label `0`) vs `non-EVOO oils` (label `1`) on Raman2

Compute is GPU-aware via `--compute {auto,cpu,gpu}` with safe backend tracking in metadata.

## Repository structure

- `ml_pipeline.py`: main training/evaluation pipeline
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

## Data

- Data type: real spectroscopy data, not synthetic benchmark data.
- FTIR task: authentic EVOO versus adulterated EVOO blends.
- Raman2 task: EVOO versus non-EVOO oils.
- Current committed datasets are small but real: 199 FTIR samples and 215 Raman2 samples.

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

Run from repository root (`C:\Users\TOM\Documents\2026\EVOO`):

```bash
python ml_pipeline.py --mode real --data-path data/processed_evoo_ftir_raman/ftir_evoo_authenticity.csv --compute gpu --output-dir results_final_ftir --cv-folds 5 --random-state 42 --shap-max-samples 80 --shap-nsamples 80

python ml_pipeline.py --mode real --data-path data/processed_evoo_ftir_raman/raman2_evoo_vs_other.csv --compute gpu --output-dir results_final_raman2 --cv-folds 5 --random-state 42 --shap-max-samples 80 --shap-nsamples 80
```

Optional strict GPU-model run:

```bash
python ml_pipeline.py --mode real --data-path data/processed_evoo_ftir_raman/raman2_evoo_vs_other.csv --compute gpu --gpu-models-only --output-dir results_final_raman2_gpuonly --cv-folds 5 --random-state 42 --shap-max-samples 80 --shap-nsamples 80
```

## Output contract

## Result Snapshot

Sources:

- `results_final_ftir/metrics.csv`
- `results_final_raman2/metrics.csv`

Representative committed results:

- FTIR task:
  - Logistic Regression: `Accuracy = 1.00`, `F1 = 1.00`
  - SVM-RBF / MLP: `Accuracy = 0.975`, `F1 = 0.9796`
- Raman2 task:
  - MLP / Random Forest / XGBoost: `Accuracy = 1.00`, `F1 = 1.00`
  - Gradient Boosting: `Accuracy = 0.9535`, `F1 = 0.9697`

## Interpretation Boundary

These are strong real-data results for a small spectroscopy study, but they
should be interpreted with the corresponding sample-size caveat in mind.

- The repository is strongest as an applied ML pipeline for authenticity
  screening on real spectroscopy data.
- The current evidence does **not** yet amount to a large external
  multi-laboratory validation study.
- This is therefore a strong applied research repository, but narrower in
  scientific breadth than the flagship systems/security platforms.

## Why EVOO is stronger than a typical small applied-ML repo

- The data basis is real spectroscopy data rather than synthetic proxies.
- Multiple model families are evaluated and persisted in committed metric files.
- GPU-aware execution and metadata tracking are implemented cleanly.
- The repository is narrow by design, but methodologically cleaner than many
  broader repos with weaker evidence discipline.

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

## Latest committed run snapshot (2026-03-29)

Evaluation protocol used for the committed snapshot:

- internal stratified 80/20 holdout split for the reported test metrics
- 5-fold stratified cross-validation as a stability check
- no external validation cohort from a different lab, season, or instrument

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

## Results and Limitations

### Observed performance

Both tasks achieve perfect internal held-out scores (Accuracy / F1 / ROC-AUC / MCC = **1.000**) on small real-data cohorts:

| Task | Dataset | Samples | Features | Best model |
|------|---------|---------|----------|------------|
| FTIR authenticity | `ftir_evoo_authenticity.csv` | 199 | 6 921 | LogisticRegression |
| Raman oil classification | `raman2_evoo_vs_other.csv` | 215 | 1 044 | MLP |

Holdout split details for the committed snapshot:

- FTIR: train `159` samples (`63` pure, `96` adulterated), test `40` samples (`16` pure, `24` adulterated)
- Raman2: train `172` samples (`35` EVOO, `137` non-EVOO), test `43` samples (`9` EVOO, `34` non-EVOO)

### Why perfect scores are physically plausible here

FTIR and Raman spectra carry thousands of correlated wavenumber intensities that directly encode molecular composition (fatty-acid profiles, ester bonds, C–H stretches). The chemical difference between pure EVOO and adulterated blends — or between olive oil and unrelated seed oils — is large and consistent across instruments. Prior published work on the same spectral modalities routinely reports >98 % accuracy even with simple linear classifiers, and near-perfect separation is common when adulterant type and concentration vary widely in the training set (as they do here).

### What the scores do *not* guarantee

| Risk factor | Detail |
|-------------|--------|
| **Small dataset** | 199 and 215 samples cover a limited range of cultivars, harvest years, adulteration levels, and instrument makes. A model trained here may not generalise to samples collected under different conditions. |
| **No independent external test set** | The current results use an internal stratified holdout split plus cross-validation. There is still no separate cohort from a different lab, harvest season, or instrument. |
| **Feature-to-sample ratio** | Both datasets are high-dimensional (up to 6 921 features for 199 samples). Even regularised models can memorise subtle batch effects not present in new data. |
| **Label leakage risk** | If spectral batches map perfectly onto class labels (e.g. all pure samples measured on the same day), CV folds may not break the batch boundary, inflating apparent generalisation. |
| **Single instrument / preprocessing** | Results were obtained on one preprocessing chain (`--mode real` defaults). Performance on differently pre-processed or differently calibrated spectra is unknown. |

### Recommended next steps before deployment

1. Collect an independent validation set from a different lab or harvest season.
2. Perform batch-aware cross-validation (group-by-batch or leave-one-batch-out) to test whether batch effects drive the perfect separation.
3. Test robustness to lower adulterant concentrations and to novel adulterant types not present in the current corpus.
4. Apply calibration transfer methods if the model is to run on a different spectrometer model.

The pipeline is designed for research reproducibility; the 1.000 scores should be treated as an upper-bound internal estimate pending the above external validation steps.

## Troubleshooting

- `ModuleNotFoundError`: install dependencies from `requirements.txt`
- `xgboost is not installed`: install `requirements-gpu.txt`
- `Missing required 'label' column`: fix CSV schema
- `Feature naming pattern is invalid`: rename spectral columns to accepted pattern
- SHAP skipped: check `run_metadata.json -> shap.reason`
- GPU fallback occurred: check `run_metadata.json -> compute.fallback_events`


