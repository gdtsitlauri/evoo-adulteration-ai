"""
Build final model-ready CSV from confirmed labels.

Inputs:
- ftir_classification_spectra.csv (must include sample_index, sample_id, spectral columns)
- label_suggestion_template.csv (must include sample_index, confirmed_label)

Output:
- CSV with spectral columns + label, compatible with ml_pipeline.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build final dataset from confirmed labels.")
    parser.add_argument(
        "--spectra-csv",
        default="data/processed_14651816/ftir_classification_spectra.csv",
        help="Path to FTIR spectra CSV (must include sample_index, sample_id, spectral columns)",
    )
    parser.add_argument(
        "--labels-csv",
        default="data/processed_14651816/label_suggestion_template.csv",
        help="Path to label template with confirmed_label filled",
    )
    parser.add_argument(
        "--output-csv",
        default="data/processed_14651816/olive_oil_ftir_confirmed.csv",
        help="Output model-ready CSV path",
    )
    return parser.parse_args()


def ensure_required_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def main() -> None:
    args = parse_args()
    spectra_path = Path(args.spectra_csv)
    labels_path = Path(args.labels_csv)
    out_path = Path(args.output_csv)

    if not spectra_path.exists():
        raise FileNotFoundError(f"Spectra CSV not found: {spectra_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels CSV not found: {labels_path}")

    spectra = pd.read_csv(spectra_path)
    labels = pd.read_csv(labels_path)

    ensure_required_columns(spectra, ["sample_index", "sample_id"], "Spectra CSV")
    ensure_required_columns(labels, ["sample_index", "confirmed_label"], "Labels CSV")

    if spectra["sample_index"].duplicated().any():
        raise ValueError("Spectra CSV contains duplicated sample_index values.")
    if labels["sample_index"].duplicated().any():
        raise ValueError("Labels CSV contains duplicated sample_index values.")

    labels_local = labels[["sample_index", "confirmed_label"]].copy()
    labels_local["confirmed_label"] = pd.to_numeric(labels_local["confirmed_label"], errors="coerce")
    if labels_local["confirmed_label"].isna().any():
        missing_n = int(labels_local["confirmed_label"].isna().sum())
        raise ValueError(
            f"confirmed_label has {missing_n} missing/invalid rows. Fill all with 0 or 1."
        )

    invalid = ~labels_local["confirmed_label"].isin([0, 1])
    if invalid.any():
        bad_values = sorted(labels_local.loc[invalid, "confirmed_label"].unique().tolist())
        raise ValueError(f"confirmed_label contains invalid values: {bad_values}. Allowed: 0, 1.")

    merged = spectra.merge(labels_local, on="sample_index", how="left")
    if merged["confirmed_label"].isna().any():
        raise ValueError("Some spectra rows have no confirmed_label after merge.")

    model_df = merged.drop(columns=["sample_index", "sample_id"]).copy()
    model_df = model_df.rename(columns={"confirmed_label": "label"})
    model_df["label"] = model_df["label"].astype(int)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    model_df.to_csv(out_path, index=False)

    print(f"[OK] Final dataset written: {out_path.resolve()}")
    print(f"[OK] Samples: {len(model_df)}")
    print(f"[OK] Features: {model_df.shape[1] - 1}")
    print(f"[OK] Label distribution: {model_df['label'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
