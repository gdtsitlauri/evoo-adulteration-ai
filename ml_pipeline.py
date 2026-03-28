"""
EVOO adulteration detection pipeline (integrity rebuild).

This script provides a reproducible, submission-safe workflow with:
- explicit execution modes: demo (synthetic) and real (CSV input)
- strict input schema validation
- leakage-safe model evaluation using sklearn Pipeline + Stratified CV
- hybrid compute orchestration (CPU/GPU policy with safe fallback)
- SHAP analysis with compatibility checks and graceful fallback
- standardized outputs: metrics.csv, cv_results.csv, run_metadata.json, figures
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

# Keep matplotlib fully headless/reproducible in restricted environments.
if "MPLCONFIGDIR" not in os.environ:
    mpl_cache_dir = Path(".mpl_cache")
    mpl_cache_dir.mkdir(exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_cache_dir.resolve())

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.signal import savgol_filter
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    import shap  # type: ignore
except Exception:  # pragma: no cover
    shap = None

try:
    import xgboost as xgb  # type: ignore
except Exception:  # pragma: no cover
    xgb = None


LABEL_COL = "label"
FEATURE_NAME_PATTERN = re.compile(r"^(?:nm_|w_|wl_)?(\d+(?:\.\d+)?)$")


@dataclass(frozen=True)
class RunConfig:
    mode: str
    data_path: str
    output_dir: str
    compute: str
    gpu_models_only: bool
    cv_folds: int
    random_state: int
    test_size: float
    preprocess: str
    pca_components: int
    shap_max_samples: int
    shap_nsamples: int


class IdentityTransformer(BaseEstimator, TransformerMixin):
    """No-op transformer."""

    def fit(self, x: np.ndarray, y: np.ndarray | None = None) -> "IdentityTransformer":
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=float)


class SNVTransformer(BaseEstimator, TransformerMixin):
    """Standard Normal Variate per spectrum (row-wise)."""

    def fit(self, x: np.ndarray, y: np.ndarray | None = None) -> "SNVTransformer":
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        row_mean = x.mean(axis=1, keepdims=True)
        row_std = x.std(axis=1, keepdims=True)
        row_std[row_std == 0.0] = 1.0
        return (x - row_mean) / row_std


class SavgolDerivativeTransformer(BaseEstimator, TransformerMixin):
    """Savitzky-Golay derivative transformer."""

    def __init__(self, deriv: int, window_length: int = 11, polyorder: int = 2) -> None:
        self.deriv = deriv
        self.window_length = window_length
        self.polyorder = polyorder

    def fit(self, x: np.ndarray, y: np.ndarray | None = None) -> "SavgolDerivativeTransformer":
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.shape[1] < self.window_length:
            raise ValueError(
                f"Savitzky-Golay requires at least {self.window_length} features, "
                f"but got {x.shape[1]}."
            )
        return savgol_filter(
            x,
            window_length=self.window_length,
            polyorder=self.polyorder,
            deriv=self.deriv,
            axis=1,
        )


class MSCTransformer(BaseEstimator, TransformerMixin):
    """Multiplicative Scatter Correction."""

    def __init__(self) -> None:
        self.reference_: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray | None = None) -> "MSCTransformer":
        x = np.asarray(x, dtype=float)
        self.reference_ = x.mean(axis=0)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.reference_ is None:
            raise RuntimeError("MSCTransformer must be fitted before transform.")
        x = np.asarray(x, dtype=float)
        corrected = np.zeros_like(x)
        ref = self.reference_
        for idx, spectrum in enumerate(x):
            slope, intercept = np.polyfit(ref, spectrum, 1)
            if slope == 0:
                slope = 1.0
            corrected[idx] = (spectrum - intercept) / slope
        return corrected


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(
        description="Reproducible EVOO adulteration detection pipeline"
    )
    parser.add_argument(
        "--mode",
        choices=["demo", "real"],
        default="demo",
        help="demo: synthetic data for workflow check, real: required CSV input",
    )
    parser.add_argument(
        "--data-path",
        default="data/processed_14651816/olive_oil_ftir_confirmed_provisional.csv",
        help="Path to CSV used in --mode real",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory for metrics, metadata, and figures",
    )
    parser.add_argument(
        "--compute",
        choices=["auto", "cpu", "gpu"],
        default="auto",
        help="Compute backend policy. auto: GPU when ready, else CPU.",
    )
    parser.add_argument(
        "--gpu-models-only",
        action="store_true",
        help="Run only GPU-capable models (with CPU fallback policy).",
    )
    parser.add_argument("--cv-folds", type=int, default=10, help="Stratified CV folds")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument("--test-size", type=float, default=0.2, help="Holdout test size")
    parser.add_argument(
        "--preprocess",
        choices=["raw", "snv", "sg1", "sg2", "msc"],
        default="snv",
        help="Spectral preprocessing method",
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        default=50,
        help="Requested PCA components (auto-capped by data dimensions)",
    )
    parser.add_argument(
        "--shap-max-samples",
        type=int,
        default=100,
        help="Max test samples for SHAP explainability",
    )
    parser.add_argument(
        "--shap-nsamples",
        type=int,
        default=100,
        help="KernelExplainer sampling budget",
    )

    args = parser.parse_args()
    if args.cv_folds < 2:
        raise ValueError("--cv-folds must be >= 2")
    if not (0.05 <= args.test_size <= 0.5):
        raise ValueError("--test-size must be in [0.05, 0.5]")
    if args.pca_components < 2:
        raise ValueError("--pca-components must be >= 2")
    if args.shap_max_samples < 10:
        raise ValueError("--shap-max-samples must be >= 10")

    return RunConfig(
        mode=args.mode,
        data_path=args.data_path,
        output_dir=args.output_dir,
        compute=args.compute,
        gpu_models_only=args.gpu_models_only,
        cv_folds=args.cv_folds,
        random_state=args.random_state,
        test_size=args.test_size,
        preprocess=args.preprocess,
        pca_components=args.pca_components,
        shap_max_samples=args.shap_max_samples,
        shap_nsamples=args.shap_nsamples,
    )


def create_demo_dataframe(n_samples: int, n_features: int, random_state: int) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    wavelengths = np.linspace(1000, 2500, n_features)

    def pure_profile(wl: np.ndarray) -> np.ndarray:
        signal = (
            np.exp(-((wl - 1210) ** 2) / (2 * 40**2)) * 0.35
            + np.exp(-((wl - 1730) ** 2) / (2 * 50**2)) * 0.25
            + np.exp(-((wl - 2310) ** 2) / (2 * 60**2)) * 0.40
        )
        return signal / signal.max()

    def adulterated_profile(wl: np.ndarray) -> np.ndarray:
        signal = (
            np.exp(-((wl - 1200) ** 2) / (2 * 45**2)) * 0.30
            + np.exp(-((wl - 1750) ** 2) / (2 * 55**2)) * 0.35
            + np.exp(-((wl - 2290) ** 2) / (2 * 65**2)) * 0.35
            + np.exp(-((wl - 1650) ** 2) / (2 * 30**2)) * 0.10
        )
        return signal / signal.max()

    per_class = n_samples // 2
    x_pure = np.array(
        [
            pure_profile(wavelengths) + rng.normal(0, 0.02, n_features)
            for _ in range(per_class)
        ]
    )
    x_adulterated = np.array(
        [
            adulterated_profile(wavelengths) + rng.normal(0, 0.02, n_features)
            for _ in range(per_class)
        ]
    )
    x = np.vstack([x_pure, x_adulterated])
    y = np.array([0] * per_class + [1] * per_class)

    df = pd.DataFrame(x, columns=[f"nm_{int(w)}" for w in wavelengths])
    df[LABEL_COL] = y
    return df


def load_input_dataframe(mode: str, data_path: str, random_state: int) -> tuple[pd.DataFrame, str]:
    if mode == "demo":
        df = create_demo_dataframe(n_samples=300, n_features=700, random_state=random_state)
        source = "synthetic_demo"
        return df, source

    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Real mode requires an existing CSV file, but '{path}' was not found."
        )
    df = pd.read_csv(path)
    return df, str(path.resolve())


def _validate_feature_name_pattern(feature_cols: list[str]) -> list[str]:
    invalid = []
    for col in feature_cols:
        if FEATURE_NAME_PATTERN.match(str(col).strip()) is None:
            invalid.append(col)
    return invalid


def _extract_wavelengths(feature_cols: list[str]) -> np.ndarray:
    wavelengths: list[float] = []
    for col in feature_cols:
        match = FEATURE_NAME_PATTERN.match(str(col).strip())
        if match is None:
            raise ValueError(
                "Feature naming pattern is invalid. Expected names like "
                "'nm_1100', 'wl_1720.5', 'w_2310', or '1210'."
            )
        wavelengths.append(float(match.group(1)))
    return np.asarray(wavelengths, dtype=float)


def validate_dataframe(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    if LABEL_COL not in df.columns:
        raise ValueError(f"Missing required '{LABEL_COL}' column in input CSV.")

    feature_cols = [str(c) for c in df.columns if str(c) != LABEL_COL]
    if len(feature_cols) < 10:
        raise ValueError("At least 10 spectral feature columns are required.")

    invalid_names = _validate_feature_name_pattern(feature_cols)
    if invalid_names:
        example = ", ".join(invalid_names[:5])
        raise ValueError(
            "Feature columns must follow a stable wavelength naming pattern. "
            f"Invalid examples: {example}"
        )

    x_df = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    if x_df.isna().any().any():
        na_cols = list(x_df.columns[x_df.isna().any()][:5])
        raise ValueError(
            "Found missing or non-numeric values in spectral features. "
            f"First problematic columns: {na_cols}"
        )

    y = pd.to_numeric(df[LABEL_COL], errors="coerce")
    if y.isna().any():
        raise ValueError("Label column contains non-numeric values.")

    labels = sorted(set(y.astype(int).tolist()))
    if labels != [0, 1]:
        raise ValueError(
            f"Label column must be binary with values {{0,1}}, but got {labels}."
        )

    x = x_df.to_numpy(dtype=float)
    y_arr = y.to_numpy(dtype=int)
    wavelengths = _extract_wavelengths(feature_cols)

    if len(np.unique(y_arr)) != 2:
        raise ValueError("Input data must include both classes (0 and 1).")
    if np.any(np.diff(wavelengths) <= 0):
        raise ValueError(
            "Wavelength columns must be strictly increasing to ensure spectral order."
        )
    if np.isnan(x).any() or np.isinf(x).any():
        raise ValueError("Features contain NaN or infinite values.")

    return x, y_arr, feature_cols, wavelengths


def build_preprocessor(name: str) -> TransformerMixin:
    registry: dict[str, TransformerMixin] = {
        "raw": IdentityTransformer(),
        "snv": SNVTransformer(),
        "sg1": SavgolDerivativeTransformer(deriv=1),
        "sg2": SavgolDerivativeTransformer(deriv=2),
        "msc": MSCTransformer(),
    }
    if name not in registry:
        raise ValueError(f"Unsupported preprocess method '{name}'.")
    return registry[name]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    gpu_capable: bool
    preferred_backend: str
    classifier_builder: Callable[[str], ClassifierMixin]


def probe_gpu_environment(random_state: int) -> dict[str, Any]:
    probe: dict[str, Any] = {
        "nvidia_smi_available": False,
        "nvidia_smi_summary": "",
        "xgboost_installed": xgb is not None,
        "xgboost_version": getattr(xgb, "__version__", None) if xgb is not None else None,
        "cuda_ready": False,
        "cuda_probe_reason": "",
    }

    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        probe["nvidia_smi_available"] = True
        probe["nvidia_smi_summary"] = proc.stdout.strip().splitlines()[0] if proc.stdout else ""
    except Exception as exc:
        probe["cuda_probe_reason"] = f"nvidia-smi unavailable or failed: {exc}"

    if xgb is None:
        if not probe["cuda_probe_reason"]:
            probe["cuda_probe_reason"] = "xgboost is not installed."
        return probe

    try:
        toy_x = np.array([[0.0], [1.0], [0.5], [1.5]], dtype=float)
        toy_y = np.array([0, 1, 0, 1], dtype=int)
        model = xgb.XGBClassifier(
            n_estimators=1,
            max_depth=1,
            learning_rate=0.3,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            device="cuda",
            random_state=random_state,
            verbosity=0,
        )
        model.fit(toy_x, toy_y)
        probe["cuda_ready"] = True
        probe["cuda_probe_reason"] = "CUDA training probe succeeded with XGBoost."
    except Exception as exc:
        probe["cuda_ready"] = False
        probe["cuda_probe_reason"] = f"XGBoost CUDA probe failed: {exc}"

    return probe


def resolve_effective_compute_mode(
    requested_mode: str,
    gpu_probe: dict[str, Any],
) -> tuple[str, list[dict[str, str]]]:
    fallback_events: list[dict[str, str]] = []
    cuda_ready = bool(gpu_probe.get("cuda_ready", False))

    if requested_mode == "cpu":
        return "cpu", fallback_events

    if requested_mode == "gpu":
        if cuda_ready:
            return "gpu", fallback_events
        fallback_events.append(
            {
                "event": "compute_fallback",
                "from": "gpu",
                "to": "cpu",
                "reason": str(gpu_probe.get("cuda_probe_reason", "GPU unavailable.")),
            }
        )
        return "cpu", fallback_events

    # requested auto
    if cuda_ready:
        return "gpu", fallback_events
    fallback_events.append(
        {
            "event": "compute_auto_cpu",
            "from": "auto",
            "to": "cpu",
            "reason": str(gpu_probe.get("cuda_probe_reason", "GPU unavailable.")),
        }
    )
    return "cpu", fallback_events


def _xgboost_classifier(random_state: int, backend: str) -> ClassifierMixin:
    if xgb is None:
        raise RuntimeError("xgboost is not installed. Install requirements-gpu.txt.")
    device = "cuda" if backend == "gpu" else "cpu"
    return xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        device=device,
        random_state=random_state,
        verbosity=0,
    )


def model_templates(random_state: int) -> list[ModelSpec]:
    specs: list[ModelSpec] = [
        ModelSpec(
            name="SVM_RBF",
            gpu_capable=False,
            preferred_backend="cpu",
            classifier_builder=lambda _backend: SVC(
                kernel="rbf",
                probability=True,
                random_state=random_state,
            ),
        ),
        ModelSpec(
            name="RandomForest",
            gpu_capable=False,
            preferred_backend="cpu",
            classifier_builder=lambda _backend: RandomForestClassifier(
                n_estimators=300,
                random_state=random_state,
            ),
        ),
        ModelSpec(
            name="GradientBoosting",
            gpu_capable=False,
            preferred_backend="cpu",
            classifier_builder=lambda _backend: GradientBoostingClassifier(
                random_state=random_state
            ),
        ),
        ModelSpec(
            name="KNN",
            gpu_capable=False,
            preferred_backend="cpu",
            classifier_builder=lambda _backend: KNeighborsClassifier(n_neighbors=7),
        ),
        ModelSpec(
            name="MLP",
            gpu_capable=False,
            preferred_backend="cpu",
            classifier_builder=lambda _backend: MLPClassifier(
                hidden_layer_sizes=(128, 64),
                random_state=random_state,
                max_iter=600,
            ),
        ),
        ModelSpec(
            name="LogisticRegression",
            gpu_capable=False,
            preferred_backend="cpu",
            classifier_builder=lambda _backend: LogisticRegression(
                max_iter=2000,
                random_state=random_state,
            ),
        ),
        ModelSpec(
            name="NaiveBayes",
            gpu_capable=False,
            preferred_backend="cpu",
            classifier_builder=lambda _backend: GaussianNB(),
        ),
    ]
    if xgb is not None:
        specs.insert(
            0,
            ModelSpec(
                name="XGBoost_GPU",
                gpu_capable=True,
                preferred_backend="gpu",
                classifier_builder=lambda backend: _xgboost_classifier(random_state, backend),
            ),
        )
        specs.insert(
            1,
            ModelSpec(
                name="XGBoost_CPU",
                gpu_capable=False,
                preferred_backend="cpu",
                classifier_builder=lambda backend: _xgboost_classifier(random_state, "cpu"),
            ),
        )
    return specs


def select_model_specs(specs: list[ModelSpec], gpu_models_only: bool) -> list[ModelSpec]:
    if not gpu_models_only:
        return specs
    selected = [spec for spec in specs if spec.gpu_capable]
    if not selected:
        raise ValueError(
            "gpu-models-only requested, but no GPU-capable models are available. "
            "Install requirements-gpu.txt and ensure XGBoost is importable."
        )
    return selected


def build_model_pipeline(
    preprocess_name: str,
    classifier: ClassifierMixin,
    pca_components: int,
    random_state: int,
) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor(preprocess_name)),
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=pca_components, random_state=random_state)),
            ("classifier", clone(classifier)),
        ]
    )


def probability_scores(model: Pipeline, x: np.ndarray) -> np.ndarray:
    classifier = model.named_steps["classifier"]
    if hasattr(classifier, "predict_proba"):
        probs = model.predict_proba(x)[:, 1]
        return np.asarray(probs, dtype=float)

    if hasattr(classifier, "decision_function"):
        decision = model.decision_function(x)
        decision = np.asarray(decision, dtype=float)
        decision = (decision - decision.min()) / (decision.max() - decision.min() + 1e-12)
        return decision

    preds = model.predict(x).astype(float)
    return preds


def evaluate_models(
    x: np.ndarray,
    y: np.ndarray,
    cfg: RunConfig,
    effective_compute_mode: str,
    upstream_fallback_events: list[dict[str, str]],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    str,
    Pipeline,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, str],
    list[dict[str, str]],
]:
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=cfg.test_size,
        stratify=y,
        random_state=cfg.random_state,
    )

    max_components = min(cfg.pca_components, x_train.shape[1], x_train.shape[0] - 1)
    if max_components < 2:
        raise ValueError(
            "PCA components became <2 after data-size capping. "
            "Increase sample size or reduce test split."
        )

    cv = StratifiedKFold(n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.random_state)
    specs = select_model_specs(model_templates(cfg.random_state), cfg.gpu_models_only)

    metric_rows: list[dict[str, Any]] = []
    cv_rows: list[dict[str, Any]] = []
    fitted_models: dict[str, Pipeline] = {}
    model_backend_map: dict[str, str] = {}
    fallback_events: list[dict[str, str]] = list(upstream_fallback_events)

    for spec in specs:
        model_name = spec.name
        requested_backend = spec.preferred_backend
        execution_backend = requested_backend

        if spec.gpu_capable and execution_backend == "gpu" and effective_compute_mode != "gpu":
            fallback_events.append(
                {
                    "event": "model_fallback",
                    "model": model_name,
                    "from": "gpu",
                    "to": "cpu",
                    "reason": "Effective compute mode is CPU.",
                }
            )
            execution_backend = "cpu"

        try:
            clf = spec.classifier_builder(execution_backend)
        except Exception as exc:
            if spec.gpu_capable and execution_backend == "gpu":
                fallback_events.append(
                    {
                        "event": "model_fallback",
                        "model": model_name,
                        "from": "gpu",
                        "to": "cpu",
                        "reason": f"GPU model construction failed: {exc}",
                    }
                )
                execution_backend = "cpu"
                clf = spec.classifier_builder(execution_backend)
            else:
                fallback_events.append(
                    {
                        "event": "model_skipped",
                        "model": model_name,
                        "from": execution_backend,
                        "to": "skipped",
                        "reason": f"Model construction failed: {exc}",
                    }
                )
                continue

        model = build_model_pipeline(
            preprocess_name=cfg.preprocess,
            classifier=clf,
            pca_components=max_components,
            random_state=cfg.random_state,
        )
        try:
            model.fit(x_train, y_train)
        except Exception as exc:
            if spec.gpu_capable and execution_backend == "gpu":
                fallback_events.append(
                    {
                        "event": "model_fallback",
                        "model": model_name,
                        "from": "gpu",
                        "to": "cpu",
                        "reason": f"GPU training failed: {exc}",
                    }
                )
                execution_backend = "cpu"
                clf = spec.classifier_builder(execution_backend)
                model = build_model_pipeline(
                    preprocess_name=cfg.preprocess,
                    classifier=clf,
                    pca_components=max_components,
                    random_state=cfg.random_state,
                )
                model.fit(x_train, y_train)
            else:
                fallback_events.append(
                    {
                        "event": "model_skipped",
                        "model": model_name,
                        "from": execution_backend,
                        "to": "skipped",
                        "reason": f"Model training failed: {exc}",
                    }
                )
                continue

        y_pred = model.predict(x_test)
        y_score = probability_scores(model, x_test)

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_score)
        mcc = matthews_corrcoef(y_test, y_pred)

        cv_model = build_model_pipeline(
            preprocess_name=cfg.preprocess,
            classifier=clf,
            pca_components=max_components,
            random_state=cfg.random_state,
        )
        try:
            cv_scores = cross_val_score(cv_model, x, y, cv=cv, scoring="accuracy")
        except Exception as exc:
            if spec.gpu_capable and execution_backend == "gpu":
                fallback_events.append(
                    {
                        "event": "model_fallback",
                        "model": model_name,
                        "from": "gpu",
                        "to": "cpu",
                        "reason": f"GPU CV failed: {exc}",
                    }
                )
                execution_backend = "cpu"
                clf = spec.classifier_builder(execution_backend)
                model = build_model_pipeline(
                    preprocess_name=cfg.preprocess,
                    classifier=clf,
                    pca_components=max_components,
                    random_state=cfg.random_state,
                )
                model.fit(x_train, y_train)
                y_pred = model.predict(x_test)
                y_score = probability_scores(model, x_test)
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                auc = roc_auc_score(y_test, y_score)
                mcc = matthews_corrcoef(y_test, y_pred)
                cv_model = build_model_pipeline(
                    preprocess_name=cfg.preprocess,
                    classifier=clf,
                    pca_components=max_components,
                    random_state=cfg.random_state,
                )
                cv_scores = cross_val_score(cv_model, x, y, cv=cv, scoring="accuracy")
            else:
                fallback_events.append(
                    {
                        "event": "model_skipped",
                        "model": model_name,
                        "from": execution_backend,
                        "to": "skipped",
                        "reason": f"Model CV failed: {exc}",
                    }
                )
                continue

        for fold_idx, fold_score in enumerate(cv_scores, start=1):
            cv_rows.append(
                {
                    "Model": model_name,
                    "Fold": fold_idx,
                    "Accuracy": round(float(fold_score), 6),
                    "ExecutionBackend": execution_backend.upper(),
                }
            )

        metric_rows.append(
            {
                "Model": model_name,
                "ExecutionBackend": execution_backend.upper(),
                "Accuracy": round(float(accuracy), 6),
                "F1": round(float(f1), 6),
                "ROC_AUC": round(float(auc), 6),
                "MCC": round(float(mcc), 6),
                "CV_Accuracy_Mean": round(float(cv_scores.mean()), 6),
                "CV_Accuracy_STD": round(float(cv_scores.std()), 6),
            }
        )
        fitted_models[model_name] = model
        model_backend_map[model_name] = execution_backend.upper()

    if not metric_rows:
        raise RuntimeError("No models were successfully evaluated.")

    metrics_df = pd.DataFrame(metric_rows).sort_values(
        by=["MCC", "CV_Accuracy_Mean"],
        ascending=False,
    )
    cv_df = pd.DataFrame(cv_rows)
    best_model_name = str(metrics_df.iloc[0]["Model"])
    best_model = fitted_models[best_model_name]

    return (
        metrics_df,
        cv_df,
        best_model_name,
        best_model,
        x_train,
        x_test,
        y_train,
        y_test,
        model_backend_map,
        fallback_events,
    )


def transform_for_plotting(model: Pipeline, x: np.ndarray) -> np.ndarray:
    x_proc = model.named_steps["preprocess"].transform(x)
    x_scaled = model.named_steps["scaler"].transform(x_proc)
    x_pca = model.named_steps["pca"].transform(x_scaled)
    return x_pca


def preprocess_for_plotting(x: np.ndarray, preprocess_name: str) -> np.ndarray:
    transformer = build_preprocessor(preprocess_name)
    transformer.fit(x)
    return transformer.transform(x)


def plot_spectra(x: np.ndarray, y: np.ndarray, wavelengths: np.ndarray, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    palette = {0: "#1f77b4", 1: "#d62728"}
    legend = {0: "Pure EVOO", 1: "Adulterated"}

    max_lines = min(40, len(x))
    for idx in range(max_lines):
        ax.plot(wavelengths, x[idx], color=palette[int(y[idx])], linewidth=0.8, alpha=0.4)

    for class_id, color in palette.items():
        ax.plot([], [], color=color, label=legend[class_id], linewidth=2)

    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Absorbance (a.u.)")
    ax.set_title("Spectral Profiles")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_pca_projection(x_pca: np.ndarray, y: np.ndarray, output_path: Path) -> None:
    if x_pca.shape[1] < 2:
        raise ValueError("PCA output has fewer than 2 components. Cannot create PCA projection plot.")

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        x_pca[:, 0],
        x_pca[:, 1],
        c=y,
        cmap="coolwarm",
        edgecolor="k",
        linewidths=0.3,
        s=35,
        alpha=0.85,
    )
    color_bar = fig.colorbar(scatter, ax=ax)
    color_bar.set_ticks([0, 1])
    color_bar.set_ticklabels(["Pure EVOO", "Adulterated"])
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA Score Plot")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_mean_difference(
    x: np.ndarray,
    y: np.ndarray,
    wavelengths: np.ndarray,
    output_path: Path,
) -> None:
    pure_mean = x[y == 0].mean(axis=0)
    adulterated_mean = x[y == 1].mean(axis=0)
    delta = adulterated_mean - pure_mean

    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax_top.plot(wavelengths, pure_mean, color="#1f77b4", label="Pure EVOO", linewidth=1.5)
    ax_top.plot(
        wavelengths,
        adulterated_mean,
        color="#d62728",
        label="Adulterated",
        linewidth=1.5,
    )
    ax_top.set_ylabel("Mean Absorbance")
    ax_top.set_title("Class Mean Spectra")
    ax_top.grid(alpha=0.3)
    ax_top.legend()

    ax_bottom.axhline(0.0, color="black", linewidth=0.8)
    ax_bottom.fill_between(
        wavelengths,
        delta,
        0,
        where=(delta > 0),
        color="#d62728",
        alpha=0.4,
        label="Higher in adulterated",
    )
    ax_bottom.fill_between(
        wavelengths,
        delta,
        0,
        where=(delta <= 0),
        color="#1f77b4",
        alpha=0.4,
        label="Higher in pure",
    )
    ax_bottom.set_xlabel("Wavelength (nm)")
    ax_bottom.set_ylabel("Difference")
    ax_bottom.set_title("Adulterated - Pure")
    ax_bottom.grid(alpha=0.3)
    ax_bottom.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_model_comparison(metrics_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    metrics = ["Accuracy", "F1", "ROC_AUC", "MCC"]
    ordered = metrics_df.copy().sort_values("MCC", ascending=False)

    for ax, metric in zip(axes, metrics):
        sns.barplot(
            data=ordered,
            x=metric,
            y="Model",
            hue="Model",
            legend=False,
            orient="h",
            palette="viridis",
            ax=ax,
        )
        ax.set_xlim(0.0, 1.05)
        ax.set_title(metric)
        ax.grid(axis="x", alpha=0.3)

    plt.suptitle("Model Benchmark")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, model_name: str, output_path: Path) -> None:
    cm_arr = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm_arr, display_labels=["Pure EVOO", "Adulterated"])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix ({model_name})")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_cv_distribution(cv_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=cv_df, x="Model", y="Accuracy", ax=ax)
    sns.stripplot(data=cv_df, x="Model", y="Accuracy", ax=ax, color="black", alpha=0.35, size=3)
    ax.set_title("Cross-Validation Accuracy Distribution")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close(fig)


def _extract_binary_shap_array(shap_values: Any) -> np.ndarray:
    if isinstance(shap_values, list):
        selected = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        return np.asarray(selected, dtype=float)

    arr = np.asarray(shap_values, dtype=float)
    if arr.ndim == 3:
        class_axis = 2
        class_idx = 1 if arr.shape[class_axis] > 1 else 0
        return arr[:, :, class_idx]
    return arr


def run_shap_analysis(
    best_model: Pipeline,
    model_name: str,
    x_train: np.ndarray,
    x_test: np.ndarray,
    output_dir: Path,
    random_state: int,
    max_samples: int,
    nsamples: int,
) -> dict[str, Any]:
    status: dict[str, Any] = {
        "enabled": False,
        "model": model_name,
        "reason": "",
        "plot_path": "",
    }

    if shap is None:
        status["reason"] = "shap library is not available in the current environment."
        return status

    classifier = best_model.named_steps["classifier"]
    if not hasattr(classifier, "predict_proba"):
        status["reason"] = "Best model does not expose predict_proba; SHAP skipped."
        return status

    x_train_pca = transform_for_plotting(best_model, x_train)
    x_test_pca = transform_for_plotting(best_model, x_test)
    if len(x_test_pca) == 0:
        status["reason"] = "No test samples available for SHAP."
        return status

    rng = np.random.default_rng(random_state)
    test_take = min(max_samples, len(x_test_pca))
    test_idx = rng.choice(len(x_test_pca), size=test_take, replace=False)
    x_eval = x_test_pca[test_idx]
    background_take = min(50, len(x_train_pca))
    background_idx = rng.choice(len(x_train_pca), size=background_take, replace=False)
    x_background = x_train_pca[background_idx]

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            is_tree_model = isinstance(classifier, (RandomForestClassifier, GradientBoostingClassifier))
            if xgb is not None:
                is_tree_model = is_tree_model or isinstance(classifier, xgb.XGBModel)
            if is_tree_model:
                explainer = shap.TreeExplainer(classifier)
                raw_values = explainer.shap_values(x_eval)
            else:
                explainer = shap.KernelExplainer(classifier.predict_proba, x_background)
                raw_values = explainer.shap_values(
                    x_eval,
                    nsamples=nsamples,
                    silent=True,
                )

        shap_arr = _extract_binary_shap_array(raw_values)
        mean_abs = np.abs(shap_arr).mean(axis=0)
        top = np.argsort(mean_abs)[-15:]

        feature_names = [f"PC{i + 1}" for i in range(shap_arr.shape[1])]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(
            [feature_names[i] for i in top],
            mean_abs[top],
            color="#ff7f0e",
            edgecolor="black",
            linewidth=0.4,
        )
        ax.set_title(f"SHAP Importance ({model_name})")
        ax.set_xlabel("Mean |SHAP|")
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()

        shap_plot = output_dir / "06_shap_bar.png"
        plt.savefig(shap_plot, dpi=160)
        plt.close(fig)

        status["enabled"] = True
        status["reason"] = "SHAP computed successfully."
        status["plot_path"] = str(shap_plot.resolve())
        status["samples_used"] = int(test_take)
        return status
    except Exception as exc:
        status["reason"] = f"SHAP skipped due to runtime error: {exc}"
        return status


def save_metadata(
    cfg: RunConfig,
    source: str,
    x: np.ndarray,
    y: np.ndarray,
    best_model_name: str,
    effective_compute_mode: str,
    gpu_probe: dict[str, Any],
    model_backend_map: dict[str, str],
    fallback_events: list[dict[str, str]],
    shap_status: dict[str, Any],
    output_dir: Path,
) -> None:
    class_vals, class_counts = np.unique(y, return_counts=True)
    class_distribution = {int(k): int(v) for k, v in zip(class_vals, class_counts)}

    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "mode": cfg.mode,
        "data_source": source,
        "sample_count": int(x.shape[0]),
        "feature_count": int(x.shape[1]),
        "class_distribution": class_distribution,
        "config": asdict(cfg),
        "compute": {
            "requested_mode": cfg.compute,
            "effective_mode": effective_compute_mode,
            "gpu_models_only": cfg.gpu_models_only,
            "gpu_probe": gpu_probe,
            "fallback_events": fallback_events,
            "model_backend_map": model_backend_map,
        },
        "best_model": best_model_name,
        "shap": shap_status,
        "artifacts": {
            "metrics_csv": str((output_dir / "metrics.csv").resolve()),
            "cv_results_csv": str((output_dir / "cv_results.csv").resolve()),
            "figures_dir": str(output_dir.resolve()),
        },
    }

    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as fp:
        json.dump(metadata, fp, indent=2)


def run(cfg: RunConfig) -> None:
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gpu_probe = probe_gpu_environment(cfg.random_state)
    effective_compute_mode, compute_events = resolve_effective_compute_mode(
        requested_mode=cfg.compute,
        gpu_probe=gpu_probe,
    )

    df, source = load_input_dataframe(mode=cfg.mode, data_path=cfg.data_path, random_state=cfg.random_state)
    x, y, _, wavelengths = validate_dataframe(df)

    (
        metrics_df,
        cv_df,
        best_model_name,
        best_model,
        x_train,
        x_test,
        y_train,
        y_test,
        model_backend_map,
        fallback_events,
    ) = evaluate_models(
        x=x,
        y=y,
        cfg=cfg,
        effective_compute_mode=effective_compute_mode,
        upstream_fallback_events=compute_events,
    )

    metrics_df.to_csv(output_dir / "metrics.csv", index=False)
    cv_df.to_csv(output_dir / "cv_results.csv", index=False)

    x_plot = preprocess_for_plotting(x, cfg.preprocess)
    plot_spectra(x_plot, y, wavelengths, output_dir / f"01_spectra_{cfg.preprocess}.png")

    plot_model = clone(best_model)
    plot_model.fit(x, y)
    x_pca_full = transform_for_plotting(plot_model, x)
    plot_pca_projection(x_pca_full, y, output_dir / f"02_pca_{cfg.preprocess}.png")

    plot_mean_difference(x_plot, y, wavelengths, output_dir / "03_mean_spectra_diff.png")
    plot_model_comparison(metrics_df, output_dir / "04_model_comparison.png")

    y_pred_best = best_model.predict(x_test)
    plot_confusion(y_test, y_pred_best, best_model_name, output_dir / "05_confusion_matrix.png")
    plot_cv_distribution(cv_df, output_dir / "07_cv_distribution.png")

    shap_status = run_shap_analysis(
        best_model=best_model,
        model_name=best_model_name,
        x_train=x_train,
        x_test=x_test,
        output_dir=output_dir,
        random_state=cfg.random_state,
        max_samples=cfg.shap_max_samples,
        nsamples=cfg.shap_nsamples,
    )

    save_metadata(
        cfg=cfg,
        source=source,
        x=x,
        y=y,
        best_model_name=best_model_name,
        effective_compute_mode=effective_compute_mode,
        gpu_probe=gpu_probe,
        model_backend_map=model_backend_map,
        fallback_events=fallback_events,
        shap_status=shap_status,
        output_dir=output_dir,
    )

    print("=" * 72)
    print("EVOO integrity pipeline completed")
    print("=" * 72)
    print(f"Mode: {cfg.mode}")
    print(f"Compute requested/effective: {cfg.compute} -> {effective_compute_mode}")
    print(f"Source: {source}")
    print(f"Samples: {x.shape[0]} | Features: {x.shape[1]}")
    print(f"Best model: {best_model_name}")
    print(f"Outputs: {output_dir.resolve()}")
    print("Required artifacts:")
    print(f"  - {(output_dir / 'metrics.csv').name}")
    print(f"  - {(output_dir / 'cv_results.csv').name}")
    print(f"  - {(output_dir / 'run_metadata.json').name}")
    if shap_status.get("enabled"):
        print("SHAP: generated")
    else:
        print(f"SHAP: skipped ({shap_status.get('reason', 'unknown reason')})")
    if fallback_events:
        print("Fallback events:")
        for event in fallback_events:
            print(
                f"  - {event.get('event')}: {event.get('model', 'compute')} "
                f"{event.get('from')} -> {event.get('to')} ({event.get('reason')})"
            )


def main() -> None:
    cfg = parse_args()
    run(cfg)


if __name__ == "__main__":
    main()
