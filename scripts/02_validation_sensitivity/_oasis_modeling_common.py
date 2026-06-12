# -*- coding: utf-8 -*-
from __future__ import annotations

import gzip
import json
import logging
import math
import os
import re
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight


PROJ_DIR = Path(r"C:\Users\linjingwu\anaconda3\Library\share\proj")
if PROJ_DIR.exists():
    os.environ.setdefault("PROJ_LIB", str(PROJ_DIR))

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_ROOT / "logs"
DATA_ROOT = Path(r"D:\绿洲未来适宜区预测数据")
FUTURE_WORLDCLIM_DIR = DATA_ROOT / "raw" / "worldclim" / "future_30s"

CURRENT_MODELING_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "stage05_current_worldclim_model_ready"
    / "modeling_samples_current_worldclim_complete_cases.csv"
)
STAGE30_TABLE_DIR = PROJECT_ROOT / "outputs" / "stage30_environment_factor_collinearity" / "tables"
SELECTED_FEATURE_LIST = STAGE30_TABLE_DIR / "stage30_selected_predictor_list.txt"

ALL_FEATURES = [f"wc_bio{i:02d}" for i in range(1, 20)] + ["wc_elev_m"]
BASE_METADATA_COLUMNS = [
    "SampleID",
    "Response",
    "SampleType",
    "Region",
    "DrylandStratum",
    "SpatialCVFold",
    "PointLon",
    "PointLat",
    "SpatialBlockID",
    "PatchID",
    "AreaStratum",
]

FUTURE_RE = re.compile(
    r"wc2\.1_30s_bioc_(?P<gcm>.+?)_(?P<ssp>ssp\d+)_(?P<period>\d{4}-\d{4})\.tif$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    estimator: Any
    use_sample_weight: bool = False


@dataclass(frozen=True)
class FutureScenario:
    gcm: str
    ssp: str
    period: str
    path: Path

    @property
    def key(self) -> str:
        return f"{self.gcm}__{self.ssp}__{self.period}"


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def safe_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return safe.strip("._-") or "value"


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(data, ensure_ascii=False, indent=2))


def atomic_write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8-sig")
    tmp.replace(path)


def atomic_write_csv_gz(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8-sig", compression="gzip")
    tmp.replace(path)


def setup_logging(log_path: Path) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def exception_text(exc: BaseException) -> str:
    return f"{repr(exc)}\n{traceback.format_exc()}"


def read_selected_features() -> list[str]:
    if not SELECTED_FEATURE_LIST.exists():
        raise FileNotFoundError(f"Missing selected predictor list: {SELECTED_FEATURE_LIST}")
    features = [line.strip() for line in SELECTED_FEATURE_LIST.read_text(encoding="utf-8").splitlines() if line.strip()]
    missing_from_known = [col for col in features if col not in ALL_FEATURES]
    if missing_from_known:
        raise ValueError(f"Selected predictor list has unknown features: {missing_from_known}")
    if not features:
        raise ValueError(f"Selected predictor list is empty: {SELECTED_FEATURE_LIST}")
    return features


def load_modeling_samples(features: list[str], include_all_features: bool = False) -> pd.DataFrame:
    if not CURRENT_MODELING_CSV.exists():
        raise FileNotFoundError(f"Missing modeling sample table: {CURRENT_MODELING_CSV}")
    needed = set(BASE_METADATA_COLUMNS) | set(features)
    if include_all_features:
        needed |= set(ALL_FEATURES)
    df = pd.read_csv(CURRENT_MODELING_CSV, usecols=lambda c: c in needed, low_memory=False)
    missing = sorted(set(features + ["SampleID", "Response", "SpatialCVFold"]) - set(df.columns))
    if missing:
        raise ValueError(f"Modeling sample table missing required columns: {missing}")
    for col in features:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if include_all_features:
        for col in ALL_FEATURES:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
    df["Response"] = pd.to_numeric(df["Response"], errors="coerce")
    df["SpatialCVFold"] = pd.to_numeric(df["SpatialCVFold"], errors="coerce")
    df = df.dropna(subset=features + ["Response", "SpatialCVFold"]).copy()
    df["Response"] = df["Response"].astype(int)
    df["SpatialCVFold"] = df["SpatialCVFold"].astype(int)
    for col in ["PointLon", "PointLat"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def build_model_specs(seed: int, n_jobs: int, include_extended: bool = False) -> list[ModelSpec]:
    specs = [
        ModelSpec(
            name="glm_logistic_balanced",
            estimator=Pipeline(
                steps=[
                    ("scale", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            max_iter=2000,
                            class_weight="balanced",
                            solver="lbfgs",
                            random_state=seed,
                        ),
                    ),
                ]
            ),
        ),
        ModelSpec(
            name="random_forest_balanced",
            estimator=RandomForestClassifier(
                n_estimators=300,
                max_features="sqrt",
                min_samples_leaf=5,
                class_weight="balanced_subsample",
                n_jobs=n_jobs,
                random_state=seed,
            ),
        ),
        ModelSpec(
            name="hist_gradient_boosting_balanced",
            estimator=HistGradientBoostingClassifier(
                max_iter=300,
                learning_rate=0.05,
                max_leaf_nodes=31,
                l2_regularization=0.05,
                early_stopping=True,
                random_state=seed,
            ),
            use_sample_weight=True,
        ),
    ]
    if include_extended:
        specs.extend(
            [
                ModelSpec(
                    name="extra_trees_balanced",
                    estimator=ExtraTreesClassifier(
                        n_estimators=300,
                        max_features="sqrt",
                        min_samples_leaf=5,
                        class_weight="balanced",
                        n_jobs=n_jobs,
                        random_state=seed,
                    ),
                ),
                ModelSpec(
                    name="gradient_boosting_brt_balanced",
                    estimator=GradientBoostingClassifier(
                        n_estimators=250,
                        learning_rate=0.05,
                        max_depth=3,
                        subsample=0.8,
                        random_state=seed,
                    ),
                    use_sample_weight=True,
                ),
            ]
        )
        try:
            from xgboost import XGBClassifier

            specs.append(
                ModelSpec(
                    name="xgboost_balanced",
                    estimator=XGBClassifier(
                        n_estimators=300,
                        learning_rate=0.05,
                        max_depth=5,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        objective="binary:logistic",
                        eval_metric="logloss",
                        tree_method="hist",
                        n_jobs=n_jobs,
                        random_state=seed,
                    ),
                    use_sample_weight=True,
                )
            )
        except Exception:
            logging.info("xgboost is not available; skipping xgboost_balanced model.")
        try:
            from lightgbm import LGBMClassifier

            specs.append(
                ModelSpec(
                    name="lightgbm_balanced",
                    estimator=LGBMClassifier(
                        n_estimators=300,
                        learning_rate=0.05,
                        num_leaves=31,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        objective="binary",
                        n_jobs=n_jobs,
                        random_state=seed,
                        verbose=-1,
                    ),
                    use_sample_weight=True,
                )
            )
        except Exception:
            logging.info("lightgbm is not available; skipping lightgbm_balanced model.")
    return specs


def best_tss_threshold(y_true: np.ndarray, probability: np.ndarray) -> float:
    valid = np.isfinite(probability)
    y_true = y_true[valid]
    probability = probability[valid]
    if len(np.unique(y_true)) < 2:
        return 0.5
    fpr, tpr, thresholds = roc_curve(y_true, probability)
    scores = tpr - fpr
    finite = np.isfinite(thresholds)
    if not finite.any():
        return 0.5
    best_idx = int(np.nanargmax(np.where(finite, scores, np.nan)))
    return float(thresholds[best_idx])


def evaluate_predictions(y_true: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, Any]:
    valid = np.isfinite(probability) & np.isfinite(y_true)
    y_true = y_true[valid].astype(int)
    probability = probability[valid]
    if len(y_true) == 0:
        raise ValueError("No valid predictions to evaluate.")
    y_pred = (probability >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    roc_auc = float(roc_auc_score(y_true, probability)) if len(np.unique(y_true)) == 2 else math.nan
    pr_auc = float(average_precision_score(y_true, probability)) if len(np.unique(y_true)) == 2 else math.nan
    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "tss": float(sensitivity + specificity - 1.0),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "valid_n": int(len(y_true)),
    }


def summarize_metrics(metrics: pd.DataFrame, group_cols: list[str] | None = None) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    group_cols = group_cols or ["model"]
    metric_cols = ["roc_auc", "pr_auc", "precision", "recall", "f1", "balanced_accuracy", "tss"]
    rows: list[dict[str, Any]] = []
    for keys, group in metrics.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row: dict[str, Any] = {col: key for col, key in zip(group_cols, keys)}
        row["run_count"] = int(len(group))
        for col in metric_cols:
            row[f"{col}_mean"] = float(group[col].mean())
            row[f"{col}_std"] = float(group[col].std(ddof=0))
        rows.append(row)
    sort_col = "pr_auc_mean" if "pr_auc_mean" in rows[0] else group_cols[0]
    return pd.DataFrame(rows).sort_values(sort_col, ascending=False)


def fit_model(spec: ModelSpec, x_train: np.ndarray, y_train: np.ndarray) -> Any:
    estimator = clone(spec.estimator)
    if spec.use_sample_weight:
        weights = compute_sample_weight(class_weight="balanced", y=y_train)
        estimator.fit(x_train, y_train, sample_weight=weights)
    else:
        estimator.fit(x_train, y_train)
    return estimator


def model_feature_importance(model_name: str, fold: Any, estimator: Any, features: list[str]) -> pd.DataFrame:
    if model_name in {"random_forest_balanced", "extra_trees_balanced"}:
        return pd.DataFrame(
            {
                "model": model_name,
                "fold": fold,
                "feature": features,
                "importance": estimator.feature_importances_,
                "importance_type": "gini_importance",
            }
        )
    if model_name in {"xgboost_balanced", "lightgbm_balanced"}:
        return pd.DataFrame(
            {
                "model": model_name,
                "fold": fold,
                "feature": features,
                "importance": estimator.feature_importances_,
                "importance_type": "tree_feature_importance",
            }
        )
    if model_name == "gradient_boosting_brt_balanced":
        return pd.DataFrame(
            {
                "model": model_name,
                "fold": fold,
                "feature": features,
                "importance": estimator.feature_importances_,
                "importance_type": "gbm_impurity_importance",
            }
        )
    if model_name == "glm_logistic_balanced":
        coefs = estimator.named_steps["model"].coef_[0]
        return pd.DataFrame(
            {
                "model": model_name,
                "fold": fold,
                "feature": features,
                "importance": coefs,
                "importance_type": "standardized_logistic_coefficient",
            }
        )
    return pd.DataFrame(columns=["model", "fold", "feature", "importance", "importance_type"])


def train_cv_job(
    df: pd.DataFrame,
    features: list[str],
    spec: ModelSpec,
    fold: int,
    model_path: Path,
    metrics_path: Path,
    predictions_path: Path,
    importance_path: Path,
    extra_prediction_cols: list[str] | None = None,
) -> dict[str, Any]:
    train_df = df.loc[df["SpatialCVFold"] != fold]
    test_df = df.loc[df["SpatialCVFold"] == fold]
    x_train = train_df[features].to_numpy(dtype="float64")
    y_train = train_df["Response"].to_numpy(dtype="int64")
    x_test = test_df[features].to_numpy(dtype="float64")
    y_test = test_df["Response"].to_numpy(dtype="int64")

    estimator = fit_model(spec, x_train, y_train)
    train_prob = estimator.predict_proba(x_train)[:, 1]
    test_prob = estimator.predict_proba(x_test)[:, 1]
    threshold = best_tss_threshold(y_train, train_prob)
    metrics = {
        "model": spec.name,
        "fold": int(fold),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "test_presence": int(y_test.sum()),
        "test_background": int(len(y_test) - y_test.sum()),
        **evaluate_predictions(y_test, test_prob, threshold),
    }

    base_cols = [
        col
        for col in [
            "SampleID",
            "Response",
            "SpatialCVFold",
            "SampleType",
            "Region",
            "DrylandStratum",
            "PointLon",
            "PointLat",
        ]
        + (extra_prediction_cols or [])
        if col in test_df.columns
    ]
    pred = test_df[base_cols].copy()
    pred["model"] = spec.name
    pred["probability"] = test_prob.astype("float32")
    pred["threshold"] = threshold
    pred["prediction"] = (test_prob >= threshold).astype(int)

    importance = model_feature_importance(spec.name, fold, estimator, features)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": estimator, "features": features, "threshold": threshold}, model_path)
    atomic_write_csv(metrics_path, pd.DataFrame([metrics]))
    atomic_write_csv(predictions_path, pred)
    atomic_write_csv(importance_path, importance)
    return metrics


def valid_csv(path: Path, required_columns: list[str] | None = None) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        df = pd.read_csv(path, nrows=3)
    except Exception:
        return False
    return all(col in df.columns for col in (required_columns or []))


def valid_job_outputs(paths: list[Path]) -> bool:
    return all(path.exists() and path.stat().st_size > 0 for path in paths)


def discover_future_scenarios(
    gcms: list[str] | None = None,
    ssps: list[str] | None = None,
    periods: list[str] | None = None,
    limit: int | None = None,
) -> list[FutureScenario]:
    if not FUTURE_WORLDCLIM_DIR.exists():
        raise FileNotFoundError(f"Missing future WorldClim directory: {FUTURE_WORLDCLIM_DIR}")
    scenarios: list[FutureScenario] = []
    for path in sorted(FUTURE_WORLDCLIM_DIR.rglob("*.tif")):
        match = FUTURE_RE.match(path.name)
        if not match:
            continue
        scenario = FutureScenario(match.group("gcm"), match.group("ssp"), match.group("period"), path)
        if gcms and scenario.gcm not in set(gcms):
            continue
        if ssps and scenario.ssp not in set(ssps):
            continue
        if periods and scenario.period not in set(periods):
            continue
        scenarios.append(scenario)
    if limit is not None:
        scenarios = scenarios[:limit]
    if not scenarios:
        raise FileNotFoundError(f"No future WorldClim scenarios found under {FUTURE_WORLDCLIM_DIR}")
    return scenarios


def sample_future_bioclim(path: Path, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    import rasterio
    from rasterio.transform import rowcol

    with rasterio.open(path) as src:
        if src.count != 19:
            raise ValueError(f"Expected 19 bioclim bands, got {src.count}: {path}")
        rows, cols = rowcol(src.transform, lon, lat, op=np.floor)
        rows = np.asarray(rows, dtype="int64")
        cols = np.asarray(cols, dtype="int64")
        inside = (rows >= 0) & (rows < src.height) & (cols >= 0) & (cols < src.width)
        order = np.lexsort((cols, rows))
        coords = list(zip(lon[order], lat[order]))
        values_sorted = np.empty((len(coords), 19), dtype="float32")
        for idx, sampled in enumerate(src.sample(coords, masked=True)):
            if np.ma.is_masked(sampled):
                arr = np.full(19, np.nan, dtype="float32")
            else:
                arr = np.asarray(sampled, dtype="float32")
                arr[~np.isfinite(arr)] = np.nan
            values_sorted[idx, :] = arr
        values = np.empty_like(values_sorted)
        values[order, :] = values_sorted
        values[~inside, :] = np.nan
    return values


def future_matrix_for_features(
    scenario_path: Path,
    samples: pd.DataFrame,
    features: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    lon = samples["PointLon"].to_numpy(dtype="float64")
    lat = samples["PointLat"].to_numpy(dtype="float64")
    future_bio = sample_future_bioclim(scenario_path, lon, lat)
    cols: list[np.ndarray] = []
    for feature in features:
        if feature == "wc_elev_m":
            cols.append(samples[feature].to_numpy(dtype="float32"))
        else:
            band_idx = int(feature.replace("wc_bio", "")) - 1
            cols.append(future_bio[:, band_idx])
    x = np.column_stack(cols).astype("float32")
    valid = np.isfinite(x).all(axis=1)
    return x, valid


def read_csv_maybe_gzip(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".gz":
        with gzip.open(path, "rt", encoding="utf-8-sig") as f:
            return pd.read_csv(f, low_memory=False)
    return pd.read_csv(path, low_memory=False)


def write_status_table(path: Path, rows: list[dict[str, Any]]) -> None:
    atomic_write_csv(path, pd.DataFrame(rows))
