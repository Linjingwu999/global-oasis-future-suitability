# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from _oasis_modeling_common import (
    LOG_DIR,
    PROJECT_ROOT,
    atomic_write_csv,
    atomic_write_json,
    atomic_write_text,
    build_model_specs,
    exception_text,
    load_modeling_samples,
    now_iso,
    read_selected_features,
    setup_logging,
    summarize_metrics,
    train_cv_job,
    valid_job_outputs,
    write_status_table,
)


OUT_DIR = PROJECT_ROOT / "outputs" / "stage31_selected_predictor_models"
MODEL_DIR = OUT_DIR / "models"
TABLE_DIR = OUT_DIR / "tables"
JOB_DIR = OUT_DIR / "job_outputs"
PRED_DIR = JOB_DIR / "predictions"
METRICS_DIR = JOB_DIR / "metrics"
IMPORTANCE_DIR = JOB_DIR / "feature_importance"

LOG_PATH = LOG_DIR / "stage31_selected_predictor_models.log"
STATE_JSON = LOG_DIR / "stage31_selected_predictor_models_state.json"
STATUS_CSV = LOG_DIR / "stage31_selected_predictor_models_status.csv"
JOB_STATUS_CSV = LOG_DIR / "stage31_selected_predictor_model_job_status.csv"

METRICS_CSV = TABLE_DIR / "stage31_selected_predictor_spatial_cv_metrics.csv"
SUMMARY_CSV = TABLE_DIR / "stage31_selected_predictor_model_summary.csv"
PREDICTIONS_CSV = TABLE_DIR / "stage31_selected_predictor_spatial_cv_predictions.csv"
IMPORTANCE_CSV = TABLE_DIR / "stage31_selected_predictor_feature_importance.csv"
COMPARISON_CSV = TABLE_DIR / "stage31_selected_vs_full_model_metric_comparison.csv"
REPORT_MD = OUT_DIR / "Stage31_筛选变量模型重训与原模型对比报告.md"

BASELINE_SUMMARY_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "stage06_current_worldclim_baseline_models"
    / "current_worldclim_spatial_cv_model_summary.csv"
)


def ensure_dirs() -> None:
    for path in [OUT_DIR, MODEL_DIR, TABLE_DIR, JOB_DIR, PRED_DIR, METRICS_DIR, IMPORTANCE_DIR, LOG_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def write_stage_status(status: str, message: str, extra: dict[str, Any] | None = None) -> None:
    row: dict[str, Any] = {"updated_at": now_iso(), "status": status, "message": message}
    if extra:
        row.update(extra)
    atomic_write_csv(STATUS_CSV, pd.DataFrame([row]))


def load_existing_or_run(
    df: pd.DataFrame,
    features: list[str],
    spec: Any,
    fold: int,
    overwrite: bool,
    status_rows: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model_path = MODEL_DIR / f"{spec.name}_fold{fold}.joblib"
    metrics_path = METRICS_DIR / f"{spec.name}_fold{fold}_metrics.csv"
    pred_path = PRED_DIR / f"{spec.name}_fold{fold}_predictions.csv"
    importance_path = IMPORTANCE_DIR / f"{spec.name}_fold{fold}_feature_importance.csv"
    row = {
        "updated_at": now_iso(),
        "model": spec.name,
        "fold": fold,
        "status": "running",
        "message": "",
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "predictions_path": str(pred_path),
    }
    status_rows.append(row)
    write_status_table(JOB_STATUS_CSV, status_rows)

    try:
        paths = [model_path, metrics_path, pred_path, importance_path]
        if not overwrite and valid_job_outputs(paths):
            row.update({"updated_at": now_iso(), "status": "skipped", "message": "existing successful job outputs"})
            write_status_table(JOB_STATUS_CSV, status_rows)
        else:
            logging.info("Training selected-variable model=%s fold=%s", spec.name, fold)
            train_cv_job(df, features, spec, fold, model_path, metrics_path, pred_path, importance_path)
            row.update({"updated_at": now_iso(), "status": "success", "message": "trained"})
            write_status_table(JOB_STATUS_CSV, status_rows)
        return pd.read_csv(metrics_path), pd.read_csv(pred_path), pd.read_csv(importance_path)
    except Exception as exc:
        row.update({"updated_at": now_iso(), "status": "failed", "message": repr(exc), "traceback": exception_text(exc)})
        write_status_table(JOB_STATUS_CSV, status_rows)
        logging.exception("Stage31 job failed: model=%s fold=%s", spec.name, fold)
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


def compare_with_stage06(selected_summary: pd.DataFrame) -> pd.DataFrame:
    if not BASELINE_SUMMARY_CSV.exists() or selected_summary.empty:
        return pd.DataFrame()
    baseline = pd.read_csv(BASELINE_SUMMARY_CSV)
    selected = selected_summary.rename(columns={"run_count": "fold_count"}).copy()
    common_cols = [
        "model",
        "roc_auc_mean",
        "pr_auc_mean",
        "balanced_accuracy_mean",
        "tss_mean",
        "f1_mean",
        "recall_mean",
        "precision_mean",
    ]
    baseline = baseline[[col for col in common_cols if col in baseline.columns]].copy()
    selected = selected[[col for col in common_cols if col in selected.columns]].copy()
    merged = baseline.merge(selected, on="model", suffixes=("_full20", "_selected10"))
    for metric in ["roc_auc", "pr_auc", "balanced_accuracy", "tss", "f1", "recall", "precision"]:
        a = f"{metric}_mean_selected10"
        b = f"{metric}_mean_full20"
        if a in merged.columns and b in merged.columns:
            merged[f"{metric}_mean_delta_selected_minus_full"] = merged[a] - merged[b]
    return merged


def write_report(summary: dict[str, Any], model_summary: pd.DataFrame, comparison: pd.DataFrame) -> None:
    lines = [
        "# Stage31 筛选变量模型重训与原模型对比报告",
        "",
        f"- 生成时间: {now_iso()}",
        f"- 输入样本数: {summary['sample_count']}",
        f"- 使用变量数: {summary['feature_count']}",
        f"- 使用变量: `{', '.join(summary['features'])}`",
        f"- 成功模型-折数: {summary['success_jobs']}",
        f"- 失败模型-折数: {summary['failed_jobs']}",
        f"- 跳过已有模型-折数: {summary['skipped_jobs']}",
        "",
        "## 筛选变量模型空间 CV 指标",
        "",
        model_summary.to_markdown(index=False) if not model_summary.empty else "暂无成功模型。",
        "",
        "## 与 Stage06 全变量模型对比",
        "",
        comparison.to_markdown(index=False) if not comparison.empty else "未找到可比较的 Stage06 全变量模型摘要。",
        "",
        "## 解释",
        "",
        "本阶段使用 Stage30 筛选后的变量重新训练与 Stage06 相同的三类基线模型，并保留相同 SpatialCVFold，因此可直接比较全变量模型与筛选变量模型的空间交叉验证表现。若后续决定以筛选变量模型作为主线，未来样本预测、栅格投影、约束面积和论文图表均应重做筛选变量版。",
        "",
        "## 输出文件",
        "",
    ]
    for key, value in summary["outputs"].items():
        lines.append(f"- {key}: `{value}`")
    atomic_write_text(REPORT_MD, "\n".join(lines) + "\n")


def run(args: argparse.Namespace) -> dict[str, Any]:
    ensure_dirs()
    features = read_selected_features()
    atomic_write_json(
        STATE_JSON,
        {
            "status": "running",
            "started_at": now_iso(),
            "message": "Stage31 selected predictor model retraining started",
            "features": features,
        },
    )
    write_stage_status("running", "Stage31 started", {"feature_count": len(features)})

    df = load_modeling_samples(features)
    folds = sorted(df["SpatialCVFold"].dropna().astype(int).unique().tolist())
    specs = build_model_specs(args.seed, args.n_jobs, include_extended=False)
    status_rows: list[dict[str, Any]] = []
    metrics_frames: list[pd.DataFrame] = []
    pred_frames: list[pd.DataFrame] = []
    importance_frames: list[pd.DataFrame] = []

    for spec in specs:
        for fold in folds:
            metrics, pred, importance = load_existing_or_run(df, features, spec, fold, args.overwrite, status_rows)
            if not metrics.empty:
                metrics_frames.append(metrics)
            if not pred.empty:
                pred_frames.append(pred)
            if not importance.empty:
                importance_frames.append(importance)

    metrics_df = pd.concat(metrics_frames, ignore_index=True) if metrics_frames else pd.DataFrame()
    if metrics_df.empty:
        raise RuntimeError(f"Stage31 produced no successful model-fold results. See {JOB_STATUS_CSV}")
    predictions_df = pd.concat(pred_frames, ignore_index=True) if pred_frames else pd.DataFrame()
    importance_df = pd.concat(importance_frames, ignore_index=True) if importance_frames else pd.DataFrame()
    model_summary = summarize_metrics(metrics_df, ["model"])
    comparison = compare_with_stage06(model_summary)

    atomic_write_csv(METRICS_CSV, metrics_df)
    atomic_write_csv(SUMMARY_CSV, model_summary)
    atomic_write_csv(PREDICTIONS_CSV, predictions_df)
    atomic_write_csv(IMPORTANCE_CSV, importance_df)
    atomic_write_csv(COMPARISON_CSV, comparison)

    status_df = pd.DataFrame(status_rows)
    failed_jobs = int((status_df["status"] == "failed").sum()) if not status_df.empty else 0
    success_jobs = int((status_df["status"] == "success").sum()) if not status_df.empty else 0
    skipped_jobs = int((status_df["status"] == "skipped").sum()) if not status_df.empty else 0
    state = {
        "status": "success" if failed_jobs == 0 else "partial_success",
        "sample_count": int(len(df)),
        "feature_count": len(features),
        "features": features,
        "fold_count": len(folds),
        "model_count": len(specs),
        "success_jobs": success_jobs,
        "skipped_jobs": skipped_jobs,
        "failed_jobs": failed_jobs,
        "outputs": {
            "report_md": str(REPORT_MD),
            "model_summary_csv": str(SUMMARY_CSV),
            "metrics_csv": str(METRICS_CSV),
            "predictions_csv": str(PREDICTIONS_CSV),
            "feature_importance_csv": str(IMPORTANCE_CSV),
            "comparison_csv": str(COMPARISON_CSV),
            "model_dir": str(MODEL_DIR),
            "job_status_csv": str(JOB_STATUS_CSV),
            "stage_status_csv": str(STATUS_CSV),
            "log_path": str(LOG_PATH),
        },
        "started_at": json.loads(STATE_JSON.read_text(encoding="utf-8")).get("started_at", ""),
        "finished_at": now_iso(),
    }
    write_report(state, model_summary, comparison)
    atomic_write_json(STATE_JSON, state)
    write_stage_status(state["status"], "Stage31 completed", state)
    logging.info("Stage31 completed: %s", json.dumps(state, ensure_ascii=False))
    return state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用 Stage30 筛选变量重训空间 CV 模型并对比 Stage06 全变量模型。")
    parser.add_argument("--seed", type=int, default=20260531)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_dirs()
    setup_logging(LOG_PATH)
    try:
        state = run(args)
        return 0 if state.get("status") in {"success", "partial_success"} else 1
    except Exception as exc:
        err = {"status": "failed", "failed_at": now_iso(), "error": repr(exc), "traceback": exception_text(exc)}
        atomic_write_json(STATE_JSON, err)
        write_stage_status("failed", repr(exc))
        logging.exception("Stage31 failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

