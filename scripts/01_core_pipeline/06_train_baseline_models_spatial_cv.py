# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import logging
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
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


WORKSPACE = Path(r"C:\Users\linjingwu\Desktop\python")
PROJECT_ROOT = WORKSPACE / "绿洲未来适宜区预测"
LOG_DIR = PROJECT_ROOT / "logs"
OUT_DIR = PROJECT_ROOT / "outputs" / "stage06_current_worldclim_baseline_models"
MODEL_DIR = OUT_DIR / "models"

INPUT_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "stage05_current_worldclim_model_ready"
    / "modeling_samples_current_worldclim_complete_cases.csv"
)

LOG_PATH = LOG_DIR / "stage06_current_worldclim_baseline_model_training.log"
STATE_PATH = LOG_DIR / "stage06_current_worldclim_baseline_model_training_state.json"
STATUS_CSV = LOG_DIR / "stage06_current_worldclim_baseline_model_training_status.csv"
MODEL_STATUS_CSV = LOG_DIR / "stage06_current_worldclim_baseline_model_status.csv"

METRICS_CSV = OUT_DIR / "current_worldclim_spatial_cv_metrics.csv"
MODEL_SUMMARY_CSV = OUT_DIR / "current_worldclim_spatial_cv_model_summary.csv"
PREDICTIONS_CSV = OUT_DIR / "current_worldclim_spatial_cv_predictions.csv"
FEATURE_IMPORTANCE_CSV = OUT_DIR / "current_worldclim_feature_importance.csv"
REPORT_MD = OUT_DIR / "stage06_current_worldclim_baseline_model_report.md"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    estimator: Any
    use_sample_weight: bool = False


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def setup_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(LOG_PATH, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(data, ensure_ascii=False, indent=2))


def atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8-sig")
    tmp.replace(path)


def write_status(status: str, message: str, extra: dict[str, Any] | None = None) -> None:
    row: dict[str, Any] = {"updated_at": now_iso(), "status": status, "message": message}
    if extra:
        row.update(extra)
    atomic_write_csv(pd.DataFrame([row]), STATUS_CSV)


def write_model_status(rows: list[dict[str, Any]]) -> None:
    if rows:
        atomic_write_csv(pd.DataFrame(rows), MODEL_STATUS_CSV)


def feature_columns(df: pd.DataFrame) -> list[str]:
    cols = [f"wc_bio{i:02d}" for i in range(1, 20)] + ["wc_elev_m"]
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing model feature columns: {missing}")
    return cols


def build_model_specs(seed: int, n_jobs: int) -> list[ModelSpec]:
    return [
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


def best_tss_threshold(y_true: np.ndarray, probability: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, probability)
    scores = tpr - fpr
    finite = np.isfinite(thresholds)
    if not finite.any():
        return 0.5
    best_idx = int(np.nanargmax(np.where(finite, scores, np.nan)))
    return float(thresholds[best_idx])


def evaluate_predictions(y_true: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, Any]:
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
    return {
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "pr_auc": float(average_precision_score(y_true, probability)),
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
    }


def model_feature_importance(model_name: str, fold: int, estimator: Any, features: list[str]) -> pd.DataFrame:
    if model_name == "random_forest_balanced":
        return pd.DataFrame(
            {
                "model": model_name,
                "fold": fold,
                "feature": features,
                "importance": estimator.feature_importances_,
                "importance_type": "gini_importance",
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


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    metric_cols = ["roc_auc", "pr_auc", "precision", "recall", "f1", "balanced_accuracy", "tss"]
    rows: list[dict[str, Any]] = []
    for model_name, group in metrics.groupby("model"):
        row: dict[str, Any] = {"model": model_name, "fold_count": int(group["fold"].nunique())}
        for col in metric_cols:
            row[f"{col}_mean"] = float(group[col].mean())
            row[f"{col}_std"] = float(group[col].std(ddof=0))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("pr_auc_mean", ascending=False)


def write_report(summary: dict[str, Any], model_summary: pd.DataFrame, metrics: pd.DataFrame) -> None:
    lines = [
        "# Stage06 当前 WorldClim 基线模型空间交叉验证",
        "",
        f"- 生成时间: {now_iso()}",
        f"- 输入表: {INPUT_CSV}",
        f"- 特征数: {summary['feature_count']}",
        f"- 样本数: {summary['sample_count']}",
        f"- Fold 数: {summary['fold_count']}",
        f"- 成功模型-Fold: {summary['success_runs']}",
        f"- 失败模型-Fold: {summary['failed_runs']}",
        "",
        "## 模型均值指标",
        "",
        model_summary.to_markdown(index=False),
        "",
        "## Fold 级指标",
        "",
        metrics.sort_values(["model", "fold"]).to_markdown(index=False),
        "",
        "## 输出文件",
        "",
    ]
    for key, value in summary["outputs"].items():
        lines.append(f"- {key}: {value}")
    atomic_write_text(REPORT_MD, "\n".join(lines))


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV missing: {INPUT_CSV}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    atomic_write_json(
        STATE_PATH,
        {
            "status": "running",
            "started_at": now_iso(),
            "message": "stage06 current WorldClim baseline model training started",
            "input_csv": str(INPUT_CSV),
        },
    )
    write_status("running", "stage06 model training started")

    df = pd.read_csv(INPUT_CSV, low_memory=False)
    features = feature_columns(df)
    required = {"SampleID", "Response", "SpatialCVFold"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Input missing required columns: {missing}")
    df = df.dropna(subset=features + ["Response", "SpatialCVFold"]).copy()
    df["Response"] = df["Response"].astype(int)
    folds = sorted(int(f) for f in df["SpatialCVFold"].dropna().unique())
    specs = build_model_specs(args.seed, args.n_jobs)

    status_rows: list[dict[str, Any]] = []
    metrics_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    importance_frames: list[pd.DataFrame] = []

    for spec in specs:
        for fold in folds:
            row = {
                "updated_at": now_iso(),
                "model": spec.name,
                "fold": fold,
                "status": "running",
                "message": "",
            }
            status_rows.append(row)
            write_model_status(status_rows)
            try:
                train_mask = df["SpatialCVFold"] != fold
                test_mask = df["SpatialCVFold"] == fold
                train_df = df.loc[train_mask]
                test_df = df.loc[test_mask]
                x_train = train_df[features].to_numpy(dtype="float64")
                y_train = train_df["Response"].to_numpy(dtype="int64")
                x_test = test_df[features].to_numpy(dtype="float64")
                y_test = test_df["Response"].to_numpy(dtype="int64")

                estimator = clone(spec.estimator)
                logging.info("Training %s fold=%s train=%s test=%s", spec.name, fold, len(train_df), len(test_df))
                if spec.use_sample_weight:
                    weights = compute_sample_weight(class_weight="balanced", y=y_train)
                    estimator.fit(x_train, y_train, sample_weight=weights)
                else:
                    estimator.fit(x_train, y_train)

                train_prob = estimator.predict_proba(x_train)[:, 1]
                test_prob = estimator.predict_proba(x_test)[:, 1]
                threshold = best_tss_threshold(y_train, train_prob)
                metrics = evaluate_predictions(y_test, test_prob, threshold)
                metrics_rows.append(
                    {
                        "model": spec.name,
                        "fold": fold,
                        "train_rows": int(len(train_df)),
                        "test_rows": int(len(test_df)),
                        "test_presence": int(y_test.sum()),
                        "test_background": int(len(y_test) - y_test.sum()),
                        **metrics,
                    }
                )

                pred = test_df[["SampleID", "Response", "SpatialCVFold", "SampleType", "Region", "PointLon", "PointLat"]].copy()
                pred["model"] = spec.name
                pred["probability"] = test_prob
                pred["threshold"] = threshold
                pred["prediction"] = (test_prob >= threshold).astype(int)
                prediction_frames.append(pred)

                importance = model_feature_importance(spec.name, fold, estimator, features)
                if not importance.empty:
                    importance_frames.append(importance)

                model_path = MODEL_DIR / f"{spec.name}_fold{fold}.joblib"
                joblib.dump({"model": estimator, "features": features, "threshold": threshold}, model_path)
                row.update({"updated_at": now_iso(), "status": "success", "message": str(model_path)})
                write_model_status(status_rows)
            except Exception as exc:
                row.update(
                    {
                        "updated_at": now_iso(),
                        "status": "failed",
                        "message": repr(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
                write_model_status(status_rows)
                logging.exception("Failed model=%s fold=%s", spec.name, fold)

    metrics_df = pd.DataFrame(metrics_rows)
    if metrics_df.empty:
        raise RuntimeError(f"No successful model runs. See {MODEL_STATUS_CSV}")
    predictions_df = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    importance_df = pd.concat(importance_frames, ignore_index=True) if importance_frames else pd.DataFrame()
    summary_df = summarize_metrics(metrics_df)

    atomic_write_csv(metrics_df, METRICS_CSV)
    atomic_write_csv(summary_df, MODEL_SUMMARY_CSV)
    atomic_write_csv(predictions_df, PREDICTIONS_CSV)
    atomic_write_csv(importance_df, FEATURE_IMPORTANCE_CSV)

    failed_runs = sum(1 for row in status_rows if row.get("status") == "failed")
    summary = {
        "status": "success" if failed_runs == 0 else "partial_success",
        "sample_count": int(len(df)),
        "feature_count": int(len(features)),
        "fold_count": int(len(folds)),
        "model_count": int(len(specs)),
        "success_runs": int(len(metrics_df)),
        "failed_runs": int(failed_runs),
        "outputs": {
            "metrics_csv": str(METRICS_CSV),
            "model_summary_csv": str(MODEL_SUMMARY_CSV),
            "predictions_csv": str(PREDICTIONS_CSV),
            "feature_importance_csv": str(FEATURE_IMPORTANCE_CSV),
            "report_md": str(REPORT_MD),
            "model_status_csv": str(MODEL_STATUS_CSV),
            "model_dir": str(MODEL_DIR),
        },
    }
    write_report(summary, summary_df, metrics_df)
    state = {
        **summary,
        "started_at": json.loads(STATE_PATH.read_text(encoding="utf-8")).get("started_at", ""),
        "finished_at": now_iso(),
    }
    atomic_write_json(STATE_PATH, state)
    write_status(summary["status"], "stage06 model training completed", summary)
    logging.info("Stage06 model training completed: %s", json.dumps(summary, ensure_ascii=False))
    return state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用当前 WorldClim+高程变量训练空间 CV 基线模型。")
    parser.add_argument("--seed", type=int, default=20260530)
    parser.add_argument("--n-jobs", type=int, default=-1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging()
    try:
        state = run(args)
        return 0 if state.get("status") in {"success", "partial_success"} else 1
    except Exception as exc:
        err = {
            "status": "failed",
            "failed_at": now_iso(),
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }
        atomic_write_json(STATE_PATH, err)
        write_status("failed", repr(exc))
        logging.exception("Stage06 model training failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
