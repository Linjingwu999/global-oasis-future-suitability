# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _oasis_modeling_common import (
    LOG_DIR,
    PROJECT_ROOT,
    ModelSpec,
    atomic_write_csv,
    atomic_write_json,
    atomic_write_text,
    best_tss_threshold,
    build_model_specs,
    evaluate_predictions,
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


OUT_DIR = PROJECT_ROOT / "outputs" / "stage33_multimodel_ensemble_background_sensitivity"
TABLE_DIR = OUT_DIR / "tables"
FIG_DIR = OUT_DIR / "figures"
MODEL_DIR = OUT_DIR / "models"
JOB_DIR = OUT_DIR / "job_outputs"
METRICS_DIR = JOB_DIR / "metrics"
PRED_DIR = JOB_DIR / "predictions"
IMPORTANCE_DIR = JOB_DIR / "feature_importance"
BG_DIR = OUT_DIR / "background_sensitivity_jobs"

LOG_PATH = LOG_DIR / "stage33_multimodel_ensemble_background_sensitivity.log"
STATE_JSON = LOG_DIR / "stage33_multimodel_ensemble_background_sensitivity_state.json"
STATUS_CSV = LOG_DIR / "stage33_multimodel_ensemble_background_sensitivity_status.csv"
MODEL_JOB_STATUS_CSV = LOG_DIR / "stage33_multimodel_cv_job_status.csv"
BG_STATUS_CSV = LOG_DIR / "stage33_background_sensitivity_status.csv"

MODEL_METRICS_CSV = TABLE_DIR / "stage33_multimodel_spatial_cv_metrics.csv"
MODEL_SUMMARY_CSV = TABLE_DIR / "stage33_multimodel_spatial_cv_summary.csv"
ENSEMBLE_PREDICTIONS_CSV = TABLE_DIR / "stage33_ensemble_spatial_cv_predictions.csv"
ENSEMBLE_METRICS_CSV = TABLE_DIR / "stage33_ensemble_spatial_cv_metrics.csv"
BG_METRICS_CSV = TABLE_DIR / "stage33_background_sensitivity_metrics.csv"
BG_SUMMARY_CSV = TABLE_DIR / "stage33_background_sensitivity_summary.csv"
REPORT_MD = OUT_DIR / "Stage33_多模型集成与背景敏感性分析报告.md"


def ensure_dirs() -> None:
    for path in [OUT_DIR, TABLE_DIR, FIG_DIR, MODEL_DIR, JOB_DIR, METRICS_DIR, PRED_DIR, IMPORTANCE_DIR, BG_DIR, LOG_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def write_stage_status(status: str, message: str, extra: dict[str, Any] | None = None) -> None:
    row: dict[str, Any] = {"updated_at": now_iso(), "status": status, "message": message}
    if extra:
        row.update(extra)
    atomic_write_csv(STATUS_CSV, pd.DataFrame([row]))


def package_availability() -> dict[str, bool]:
    return {
        "xgboost": importlib.util.find_spec("xgboost") is not None,
        "lightgbm": importlib.util.find_spec("lightgbm") is not None,
    }


def run_multimodel_cv(
    df: pd.DataFrame,
    features: list[str],
    seed: int,
    n_jobs: int,
    overwrite: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    specs = build_model_specs(seed, n_jobs, include_extended=True)
    folds = sorted(df["SpatialCVFold"].dropna().astype(int).unique().tolist())
    status_rows: list[dict[str, Any]] = []
    metrics_frames: list[pd.DataFrame] = []
    pred_frames: list[pd.DataFrame] = []
    importance_frames: list[pd.DataFrame] = []
    for spec in specs:
        for fold in folds:
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
                "metrics_path": str(metrics_path),
                "predictions_path": str(pred_path),
            }
            status_rows.append(row)
            write_status_table(MODEL_JOB_STATUS_CSV, status_rows)
            try:
                paths = [model_path, metrics_path, pred_path, importance_path]
                if not overwrite and valid_job_outputs(paths):
                    row.update({"updated_at": now_iso(), "status": "skipped", "message": "existing successful job outputs"})
                    write_status_table(MODEL_JOB_STATUS_CSV, status_rows)
                else:
                    logging.info("Stage33 multimodel CV model=%s fold=%s", spec.name, fold)
                    train_cv_job(df, features, spec, fold, model_path, metrics_path, pred_path, importance_path)
                    row.update({"updated_at": now_iso(), "status": "success", "message": "trained"})
                    write_status_table(MODEL_JOB_STATUS_CSV, status_rows)
                metrics_frames.append(pd.read_csv(metrics_path))
                pred_frames.append(pd.read_csv(pred_path))
                importance_frames.append(pd.read_csv(importance_path))
            except Exception as exc:
                row.update({"updated_at": now_iso(), "status": "failed", "message": repr(exc), "traceback": exception_text(exc)})
                write_status_table(MODEL_JOB_STATUS_CSV, status_rows)
                logging.exception("Stage33 multimodel job failed model=%s fold=%s", spec.name, fold)
    metrics = pd.concat(metrics_frames, ignore_index=True) if metrics_frames else pd.DataFrame()
    predictions = pd.concat(pred_frames, ignore_index=True) if pred_frames else pd.DataFrame()
    importance = pd.concat(importance_frames, ignore_index=True) if importance_frames else pd.DataFrame()
    return metrics, predictions, importance


def ensemble_from_oof_predictions(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if predictions.empty:
        return pd.DataFrame(), pd.DataFrame()
    id_cols = [
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
        if col in predictions.columns
    ]
    ens = (
        predictions.groupby(id_cols, dropna=False)["probability"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "ensemble_probability", "std": "model_probability_sd", "count": "model_count"})
    )
    y = ens["Response"].to_numpy(dtype="int64")
    p = ens["ensemble_probability"].to_numpy(dtype="float64")
    threshold = best_tss_threshold(y, p)
    ens["threshold"] = threshold
    ens["prediction"] = (ens["ensemble_probability"] >= threshold).astype(int)
    metric_rows: list[dict[str, Any]] = []
    for fold, group in ens.groupby("SpatialCVFold"):
        metric_rows.append(
            {
                "model": "mean_probability_ensemble",
                "fold": int(fold),
                "test_rows": int(len(group)),
                "test_presence": int(group["Response"].sum()),
                "test_background": int(len(group) - group["Response"].sum()),
                **evaluate_predictions(
                    group["Response"].to_numpy(dtype="int64"),
                    group["ensemble_probability"].to_numpy(dtype="float64"),
                    threshold,
                ),
            }
        )
    metric_rows.append(
        {
            "model": "mean_probability_ensemble",
            "fold": "all_oof",
            "test_rows": int(len(ens)),
            "test_presence": int(ens["Response"].sum()),
            "test_background": int(len(ens) - ens["Response"].sum()),
            **evaluate_predictions(y, p, threshold),
        }
    )
    return ens, pd.DataFrame(metric_rows)


def make_background_strategy(df: pd.DataFrame, strategy: str, seed: int) -> pd.DataFrame:
    presence = df[df["Response"] == 1].copy()
    background = df[df["Response"] == 0].copy()
    if strategy == "all_background_current_pool":
        return df.copy()

    ratio_map = {
        "fold_balanced_background_2to1": 2,
        "fold_balanced_background_1to1": 1,
    }
    if strategy not in ratio_map:
        raise ValueError(f"Unknown background sensitivity strategy: {strategy}")
    ratio = ratio_map[strategy]
    sampled_parts: list[pd.DataFrame] = [presence]
    for fold, p_fold in presence.groupby("SpatialCVFold"):
        bg_fold = background[background["SpatialCVFold"] == fold]
        n = min(len(bg_fold), int(len(p_fold) * ratio))
        if n > 0:
            sampled_parts.append(bg_fold.sample(n=n, random_state=seed + int(fold) * 17))
    return pd.concat(sampled_parts, ignore_index=True)


def run_background_sensitivity(
    df: pd.DataFrame,
    features: list[str],
    seed: int,
    overwrite: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    hgb_spec = [s for s in build_model_specs(seed, -1, include_extended=False) if s.name == "hist_gradient_boosting_balanced"][0]
    strategies = ["all_background_current_pool", "fold_balanced_background_2to1", "fold_balanced_background_1to1"]
    status_rows: list[dict[str, Any]] = []
    metrics_rows: list[dict[str, Any]] = []
    for strategy in strategies:
        sdf = make_background_strategy(df, strategy, seed)
        folds = sorted(sdf["SpatialCVFold"].dropna().astype(int).unique().tolist())
        for fold in folds:
            metrics_path = BG_DIR / f"{strategy}__{hgb_spec.name}_fold{fold}_metrics.csv"
            pred_path = BG_DIR / f"{strategy}__{hgb_spec.name}_fold{fold}_predictions.csv"
            model_path = BG_DIR / f"{strategy}__{hgb_spec.name}_fold{fold}.joblib"
            importance_path = BG_DIR / f"{strategy}__{hgb_spec.name}_fold{fold}_feature_importance.csv"
            row = {
                "updated_at": now_iso(),
                "strategy": strategy,
                "model": hgb_spec.name,
                "fold": fold,
                "status": "running",
                "message": "",
                "train_sample_pool_rows": int(len(sdf)),
                "pool_presence": int((sdf["Response"] == 1).sum()),
                "pool_background": int((sdf["Response"] == 0).sum()),
            }
            status_rows.append(row)
            write_status_table(BG_STATUS_CSV, status_rows)
            try:
                if not overwrite and valid_job_outputs([metrics_path, pred_path, model_path, importance_path]):
                    metric = pd.read_csv(metrics_path)
                    row.update({"updated_at": now_iso(), "status": "skipped", "message": "existing successful job outputs"})
                    write_status_table(BG_STATUS_CSV, status_rows)
                else:
                    logging.info("Stage33 background sensitivity strategy=%s fold=%s rows=%s", strategy, fold, len(sdf))
                    train_cv_job(sdf, features, hgb_spec, fold, model_path, metrics_path, pred_path, importance_path)
                    metric = pd.read_csv(metrics_path)
                    row.update({"updated_at": now_iso(), "status": "success", "message": "trained"})
                    write_status_table(BG_STATUS_CSV, status_rows)
                metric["background_strategy"] = strategy
                metric["pool_rows"] = int(len(sdf))
                metric["pool_presence"] = int((sdf["Response"] == 1).sum())
                metric["pool_background"] = int((sdf["Response"] == 0).sum())
                metrics_rows.extend(metric.to_dict(orient="records"))
            except Exception as exc:
                row.update({"updated_at": now_iso(), "status": "failed", "message": repr(exc), "traceback": exception_text(exc)})
                write_status_table(BG_STATUS_CSV, status_rows)
                logging.exception("Stage33 background sensitivity failed strategy=%s fold=%s", strategy, fold)
    metrics = pd.DataFrame(metrics_rows)
    summary = summarize_metrics(metrics, ["background_strategy", "model"]) if not metrics.empty else pd.DataFrame()
    return metrics, summary


def base_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
            "savefig.dpi": 600,
        }
    )


def write_figures(model_summary: pd.DataFrame, bg_summary: pd.DataFrame) -> dict[str, str]:
    base_style()
    figures: dict[str, str] = {}
    if not model_summary.empty:
        plot_df = model_summary.sort_values("pr_auc_mean", ascending=True)
        fig, ax = plt.subplots(figsize=(6.6, 3.5))
        ax.barh(plot_df["model"], plot_df["pr_auc_mean"], color="#4C78A8")
        ax.set_xlabel("Spatial CV PR-AUC")
        ax.grid(axis="x", color="#D9D9D9", linewidth=0.5)
        png = FIG_DIR / "fig_stage33_multimodel_pr_auc.png"
        svg = FIG_DIR / "fig_stage33_multimodel_pr_auc.svg"
        fig.savefig(png, bbox_inches="tight", pad_inches=0.02)
        fig.savefig(svg, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        figures["multimodel_pr_auc_png"] = str(png)
        figures["multimodel_pr_auc_svg"] = str(svg)
    if not bg_summary.empty:
        plot_df = bg_summary.sort_values("background_strategy")
        fig, ax = plt.subplots(figsize=(7.0, 3.3))
        x = np.arange(len(plot_df))
        ax.bar(x, plot_df["pr_auc_mean"], color="#6B9E6B")
        ax.set_xticks(x)
        ax.set_xticklabels(plot_df["background_strategy"], rotation=30, ha="right")
        ax.set_ylabel("HGB PR-AUC")
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.5)
        png = FIG_DIR / "fig_stage33_background_sensitivity_pr_auc.png"
        svg = FIG_DIR / "fig_stage33_background_sensitivity_pr_auc.svg"
        fig.savefig(png, bbox_inches="tight", pad_inches=0.02)
        fig.savefig(svg, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        figures["background_sensitivity_png"] = str(png)
        figures["background_sensitivity_svg"] = str(svg)
    return figures


def write_report(summary: dict[str, Any], model_summary: pd.DataFrame, ensemble_metrics: pd.DataFrame, bg_summary: pd.DataFrame) -> None:
    lines = [
        "# Stage33 多模型集成与背景敏感性分析报告",
        "",
        f"- 生成时间: {now_iso()}",
        f"- 样本数: {summary['sample_count']}",
        f"- 使用变量: `{', '.join(summary['features'])}`",
        f"- 模型 CV 成功/跳过/失败: {summary['model_success_jobs']} / {summary['model_skipped_jobs']} / {summary['model_failed_jobs']}",
        f"- 背景敏感性成功/跳过/失败: {summary['background_success_jobs']} / {summary['background_skipped_jobs']} / {summary['background_failed_jobs']}",
        f"- xgboost 可用: {summary['package_availability']['xgboost']}",
        f"- lightgbm 可用: {summary['package_availability']['lightgbm']}",
        "",
        "## 多模型空间 CV 摘要",
        "",
        model_summary.to_markdown(index=False) if not model_summary.empty else "暂无成功模型。",
        "",
        "## OOF 均值概率集成指标",
        "",
        ensemble_metrics.to_markdown(index=False) if not ensemble_metrics.empty else "暂无集成结果。",
        "",
        "## 背景点敏感性摘要",
        "",
        bg_summary.to_markdown(index=False) if not bg_summary.empty else "暂无背景敏感性结果。",
        "",
        "## 方法说明",
        "",
        "多模型集成使用 Stage30 筛选变量，并在相同 SpatialCVFold 上训练 GLM、RF、HGB、ExtraTrees 与 GBM/BRT-like 模型；如当前 Python 环境可用，也纳入 XGBoost 与 LightGBM。背景点敏感性使用现有干旱区背景点池进行 3:1 当前池、2:1 抽样、1:1 抽样对比，属于背景点抽样敏感性，不等同于重新生成不同地理假设下的背景点。",
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
            "message": "Stage33 multimodel ensemble and background sensitivity started",
            "features": features,
        },
    )
    write_stage_status("running", "Stage33 started", {"feature_count": len(features)})
    df = load_modeling_samples(features)

    model_metrics, model_predictions, model_importance = run_multimodel_cv(df, features, args.seed, args.n_jobs, args.overwrite)
    model_summary = summarize_metrics(model_metrics, ["model"]) if not model_metrics.empty else pd.DataFrame()
    ensemble_predictions, ensemble_metrics = ensemble_from_oof_predictions(model_predictions)
    bg_metrics, bg_summary = run_background_sensitivity(df, features, args.seed, args.overwrite)

    if not ensemble_metrics.empty:
        model_summary_with_ensemble = pd.concat(
            [model_summary, summarize_metrics(ensemble_metrics[ensemble_metrics["fold"] != "all_oof"], ["model"])],
            ignore_index=True,
        )
    else:
        model_summary_with_ensemble = model_summary

    atomic_write_csv(MODEL_METRICS_CSV, model_metrics)
    atomic_write_csv(MODEL_SUMMARY_CSV, model_summary_with_ensemble)
    atomic_write_csv(ENSEMBLE_PREDICTIONS_CSV, ensemble_predictions)
    atomic_write_csv(ENSEMBLE_METRICS_CSV, ensemble_metrics)
    atomic_write_csv(TABLE_DIR / "stage33_multimodel_feature_importance.csv", model_importance)
    atomic_write_csv(BG_METRICS_CSV, bg_metrics)
    atomic_write_csv(BG_SUMMARY_CSV, bg_summary)
    figures = write_figures(model_summary_with_ensemble, bg_summary)

    model_status = pd.read_csv(MODEL_JOB_STATUS_CSV) if MODEL_JOB_STATUS_CSV.exists() else pd.DataFrame()
    bg_status = pd.read_csv(BG_STATUS_CSV) if BG_STATUS_CSV.exists() else pd.DataFrame()
    state = {
        "status": "success"
        if (not model_status.empty and not bg_status.empty and (model_status["status"].eq("failed").sum() + bg_status["status"].eq("failed").sum()) == 0)
        else "partial_success",
        "sample_count": int(len(df)),
        "feature_count": len(features),
        "features": features,
        "package_availability": package_availability(),
        "model_success_jobs": int(model_status["status"].eq("success").sum()) if not model_status.empty else 0,
        "model_skipped_jobs": int(model_status["status"].eq("skipped").sum()) if not model_status.empty else 0,
        "model_failed_jobs": int(model_status["status"].eq("failed").sum()) if not model_status.empty else 0,
        "background_success_jobs": int(bg_status["status"].eq("success").sum()) if not bg_status.empty else 0,
        "background_skipped_jobs": int(bg_status["status"].eq("skipped").sum()) if not bg_status.empty else 0,
        "background_failed_jobs": int(bg_status["status"].eq("failed").sum()) if not bg_status.empty else 0,
        "outputs": {
            "report_md": str(REPORT_MD),
            "model_summary_csv": str(MODEL_SUMMARY_CSV),
            "model_metrics_csv": str(MODEL_METRICS_CSV),
            "ensemble_predictions_csv": str(ENSEMBLE_PREDICTIONS_CSV),
            "ensemble_metrics_csv": str(ENSEMBLE_METRICS_CSV),
            "background_metrics_csv": str(BG_METRICS_CSV),
            "background_summary_csv": str(BG_SUMMARY_CSV),
            "model_job_status_csv": str(MODEL_JOB_STATUS_CSV),
            "background_status_csv": str(BG_STATUS_CSV),
            "figures": figures,
            "log_path": str(LOG_PATH),
        },
        "started_at": json.loads(STATE_JSON.read_text(encoding="utf-8")).get("started_at", ""),
        "finished_at": now_iso(),
    }
    write_report(state, model_summary_with_ensemble, ensemble_metrics, bg_summary)
    atomic_write_json(STATE_JSON, state)
    write_stage_status(state["status"], "Stage33 completed", state)
    logging.info("Stage33 completed: %s", json.dumps(state, ensure_ascii=False))
    return state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="补做筛选变量多模型集成与背景点抽样敏感性分析。")
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
        logging.exception("Stage33 failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
