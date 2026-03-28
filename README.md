# EVOO Adulteration - FTIR/Raman Integrity Pipeline

## Authors
- George David Tsitlauri
- Theodora Anna Lasithiotaki

This repository is structured as a reproducible pre-submission package for EVOO adulteration detection with FTIR/Raman spectroscopy and ML.

Current status:
- `demo` mode is fully executable with synthetic spectra (workflow validation only).
- `real` mode is supported and enforces strict CSV schema validation.
- Hybrid compute is supported through `--compute {auto,cpu,gpu}` with safe GPU fallback.
- No publication claims should be made from `demo` outputs.
- Current project outputs are in a provisional-label phase pending official sample-code mapping.

## 1. Environment setup (Python 3.12)

CPU baseline:

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

GPU extras (optional, for XGBoost CUDA):

```bash
pip install -r requirements-gpu.txt
```

## 2. Run commands

Demo mode (no external dataset required):

```bash
python ml_pipeline.py --mode demo --compute auto --output-dir results_demo
```

Real mode with the current processed dataset in this repository:

```bash
python ml_pipeline.py \
  --mode real \
  --data-path data/processed_14651816/olive_oil_ftir_confirmed_provisional.csv \
  --compute auto \
  --output-dir results_14651816_confirmed_provisional
```

GPU-only benchmark (XGBoost GPU path):

```bash
python ml_pipeline.py \
  --mode real \
  --data-path data/processed_14651816/olive_oil_ftir_confirmed_provisional.csv \
  --compute auto \
  --gpu-models-only \
  --output-dir results_14651816_confirmed_provisional_gpuonly
```

Run with your own CSV:

```bash
python ml_pipeline.py --mode real --data-path <your_csv> --compute auto --output-dir results_real
```

Note:
- The current workflow is FTIR-based (classification sheet from Zenodo 14651816).
- Raw Raman files are present in `data/raw_14651816` for future extension.
- Filenames containing `_nir_` are legacy naming from an earlier draft; they currently store FTIR-derived tabular data.
- Confirm labels in `label_suggestion_template.csv` using `sample_index` as the unique key
  (some `sample_id` values are duplicated in the source workbook).

If you update labels in `label_suggestion_template.csv`, rebuild a model-ready CSV:

```bash
python scripts/build_dataset_from_confirmed_labels.py \
  --spectra-csv data/processed_14651816/ftir_classification_spectra.csv \
  --labels-csv data/processed_14651816/label_suggestion_template.csv \
  --output-csv data/processed_14651816/olive_oil_ftir_confirmed.csv
```

Then run the pipeline on the rebuilt CSV:

```bash
python ml_pipeline.py \
  --mode real \
  --data-path data/processed_14651816/olive_oil_ftir_confirmed.csv \
  --compute auto \
  --output-dir results_14651816_ftir_confirmed
```

Current working run folders in this repository:
- `results_14651816_confirmed_provisional` (full benchmark set)
- `results_14651816_confirmed_provisional_gpuonly` (GPU-only, SHAP-ready)

Force GPU policy (auto-fallback to CPU if CUDA probe fails):

```bash
python ml_pipeline.py --mode real --data-path data/processed_14651816/olive_oil_ftir_confirmed_provisional.csv --compute gpu --output-dir results_real_gpu
```

Run only GPU-capable models:

```bash
python ml_pipeline.py --mode real --data-path data/processed_14651816/olive_oil_ftir_confirmed_provisional.csv --compute auto --gpu-models-only --output-dir results_gpu_only
```

Optional key arguments:
- `--compute auto|cpu|gpu`
- `--gpu-models-only`
- `--cv-folds 10`
- `--random-state 42`
- `--preprocess snv` (`raw|snv|sg1|sg2|msc`)
- `--pca-components 50`
- `--shap-max-samples 100`

## 3. Dataset files in this repo

- Raw files:
  - `data/raw_14651816/Data_set_FTIR_Raman/FTIR/FTIR_spectra_Classification Study.xlsx`
  - `data/raw_14651816/Data_set_FTIR_Raman/Raman/Raman_spectra_Classification Study.xlsx`
- Processed files:
  - `data/processed_14651816/ftir_classification_spectra.csv`
  - `data/processed_14651816/label_suggestion_template.csv`
  - `data/processed_14651816/olive_oil_ftir_confirmed_provisional.csv`
  - `data/processed_14651816/preparation_summary.json`

## 4. Input CSV contract (`--mode real`)

Required:
- A `label` column with binary values `{0,1}`.
- Spectral feature columns only (numeric values, no missing values).
- Feature naming pattern per column:
  - `nm_<wavelength>`
  - `wl_<wavelength>`
  - `w_<wavelength>`
  - or `<wavelength>` directly

Examples:
- `nm_1000`, `nm_1002.14`, `wl_1720`, `2310`

Additional rules:
- Wavelength columns must be strictly increasing.
- Data must include both classes.

## 5. Standard output contract

Each run writes:
- `metrics.csv`
- `cv_results.csv`
- `run_metadata.json`
- `metrics.csv` now includes `ExecutionBackend` per model.
- `run_metadata.json` now includes `compute` block:
  - requested/effective mode
  - GPU probe details
  - fallback events
  - model-to-backend mapping
- Figures:
  - `01_spectra_<preprocess>.png`
  - `02_pca_<preprocess>.png`
  - `03_mean_spectra_diff.png`
  - `04_model_comparison.png`
  - `05_confusion_matrix.png`
  - `07_cv_distribution.png`
  - `06_shap_bar.png` (only if SHAP is compatible and successful)

## 6. Troubleshooting

- `ModuleNotFoundError`: install dependencies from `requirements.txt`.
- `xgboost is not installed`: install `requirements-gpu.txt` for GPU-capable model support.
- `Missing required 'label' column`: fix CSV schema.
- `Feature naming pattern is invalid`: rename spectral columns to accepted pattern.
- SHAP skipped: check `run_metadata.json -> shap.reason` (this is expected for incompatible runtime/model states).
- If SHAP is skipped for an MLP best model, run with `--gpu-models-only` to generate SHAP from `XGBoost_GPU`.
- GPU fallback happened: check `run_metadata.json -> compute.fallback_events` for exact reason.

## 7. License

Code in this repository is licensed under the MIT License (see `LICENSE`).
External datasets keep their original licenses and citation requirements.
