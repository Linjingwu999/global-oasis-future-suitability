# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _oasis_modeling_common import (
    LOG_DIR,
    PROJECT_ROOT,
    atomic_write_csv,
    atomic_write_csv_gz,
    atomic_write_json,
    atomic_write_text,
    best_tss_threshold,
    build_model_specs,
    discover_future_scenarios,
    evaluate_predictions,
    exception_text,
    fit_model,
    future_matrix_for_features,
    load_modeling_samples,
    now_iso,
    read_selected_features,
    setup_logging,
    summarize_metrics,
    valid_job_outputs,
    write_status_table,
)


OUT_DIR = PROJECT_ROOT / "outputs" / "stage32_independent_validation_extrapolation"
TABLE_DIR = OUT_DIR / "tables"
FIG_DIR = OUT_DIR / "figures"
LORO_DIR = OUT_DIR / "loro_job_outputs"
EXTRAP_DIR = OUT_DIR / "future_extrapolation_sample_risk"

LOG_PATH = LOG_DIR / "stage32_independent_validation_extrapolation.log"
STATE_JSON = LOG_DIR / "stage32_independent_validation_extrapolation_state.json"
STATUS_CSV = LOG_DIR / "stage32_independent_validation_extrapolation_status.csv"
LORO_STATUS_CSV = LOG_DIR / "stage32_loro_validation_status.csv"
EXTRAP_STATUS_CSV = LOG_DIR / "stage32_future_extrapolation_status.csv"

LORO_METRICS_CSV = TABLE_DIR / "stage32_leave_one_region_out_metrics.csv"
LORO_SUMMARY_CSV = TABLE_DIR / "stage32_leave_one_region_out_model_summary.csv"
REGION_ASSIGNMENT_CSV = TABLE_DIR / "stage32_background_validation_region_assignment_summary.csv"
EXTRAP_SUMMARY_CSV = TABLE_DIR / "stage32_future_extrapolation_scenario_summary.csv"
EXTRAP_GROUP_SUMMARY_CSV = TABLE_DIR / "stage32_future_extrapolation_group_summary.csv"
REPORT_MD = OUT_DIR / "Stage32_独立区域验证与外推风险诊断报告.md"


def ensure_dirs() -> None:
    for path in [OUT_DIR, TABLE_DIR, FIG_DIR, LORO_DIR, EXTRAP_DIR, LOG_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def write_stage_status(status: str, message: str, extra: dict[str, Any] | None = None) -> None:
    row: dict[str, Any] = {"updated_at": now_iso(), "status": status, "message": message}
    if extra:
        row.update(extra)
    atomic_write_csv(STATUS_CSV, pd.DataFrame([row]))


def assign_validation_regions(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    presence = out[(out["Response"] == 1) & out["Region"].notna()].copy()
    centroids = (
        presence.groupby("Region", dropna=False)[["PointLon", "PointLat"]]
        .mean()
        .dropna()
        .reset_index()
    )
    if centroids.empty:
        raise RuntimeError("Cannot assign validation regions because presence Region centroids are empty.")
    out["ValidationRegion"] = out["Region"].where(out["Region"].notna(), "")
    missing_mask = out["ValidationRegion"].eq("")
    bg = out.loc[missing_mask, ["PointLon", "PointLat"]].to_numpy(dtype="float64")
    cxy = centroids[["PointLon", "PointLat"]].to_numpy(dtype="float64")
    region_values = centroids["Region"].astype(str).to_numpy()
    if len(bg):
        d2 = ((bg[:, None, :] - cxy[None, :, :]) ** 2).sum(axis=2)
        out.loc[missing_mask, "ValidationRegion"] = region_values[np.argmin(d2, axis=1)]

    summary = (
        out.groupby(["ValidationRegion", "Response"], dropna=False)
        .size()
        .unstack(fill_value=0)
        .reset_index()
        .rename(columns={0: "background_n", 1: "presence_n"})
    )
    summary["total_n"] = summary.get("background_n", 0) + summary.get("presence_n", 0)
    return out, summary


def run_loro_validation(
    df: pd.DataFrame,
    features: list[str],
    seed: int,
    n_jobs: int,
    overwrite: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    specs = build_model_specs(seed, n_jobs, include_extended=False)
    regions = sorted(str(v) for v in df["ValidationRegion"].dropna().unique())
    status_rows: list[dict[str, Any]] = []
    metrics_rows: list[dict[str, Any]] = []
    for spec in specs:
        for region in regions:
            safe_region = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in region)
            metrics_path = LORO_DIR / f"{spec.name}__{safe_region}_metrics.csv"
            pred_path = LORO_DIR / f"{spec.name}__{safe_region}_predictions.csv"
            model_path = LORO_DIR / f"{spec.name}__{safe_region}.joblib"
            row = {
                "updated_at": now_iso(),
                "task": "leave_one_region_out",
                "model": spec.name,
                "heldout_region": region,
                "status": "running",
                "message": "",
                "metrics_path": str(metrics_path),
                "predictions_path": str(pred_path),
            }
            status_rows.append(row)
            write_status_table(LORO_STATUS_CSV, status_rows)
            try:
                if not overwrite and valid_job_outputs([metrics_path, pred_path, model_path]):
                    row.update({"updated_at": now_iso(), "status": "skipped", "message": "existing successful job outputs"})
                    write_status_table(LORO_STATUS_CSV, status_rows)
                    metrics_rows.extend(pd.read_csv(metrics_path).to_dict(orient="records"))
                    continue

                train_df = df[df["ValidationRegion"] != region].copy()
                test_df = df[df["ValidationRegion"] == region].copy()
                if train_df["Response"].nunique() < 2 or test_df["Response"].nunique() < 2:
                    raise RuntimeError("Train or held-out region test set has only one class.")
                x_train = train_df[features].to_numpy(dtype="float64")
                y_train = train_df["Response"].to_numpy(dtype="int64")
                x_test = test_df[features].to_numpy(dtype="float64")
                y_test = test_df["Response"].to_numpy(dtype="int64")

                logging.info("LORO training model=%s heldout=%s train=%s test=%s", spec.name, region, len(train_df), len(test_df))
                estimator = fit_model(spec, x_train, y_train)
                train_prob = estimator.predict_proba(x_train)[:, 1]
                test_prob = estimator.predict_proba(x_test)[:, 1]
                threshold = best_tss_threshold(y_train, train_prob)
                metrics = {
                    "model": spec.name,
                    "heldout_region": region,
                    "train_rows": int(len(train_df)),
                    "test_rows": int(len(test_df)),
                    "test_presence": int(y_test.sum()),
                    "test_background": int(len(y_test) - y_test.sum()),
                    **evaluate_predictions(y_test, test_prob, threshold),
                }
                pred_cols = [
                    col
                    for col in [
                        "SampleID",
                        "Response",
                        "SampleType",
                        "Region",
                        "ValidationRegion",
                        "DrylandStratum",
                        "SpatialCVFold",
                        "PointLon",
                        "PointLat",
                    ]
                    if col in test_df.columns
                ]
                pred = test_df[pred_cols].copy()
                pred["model"] = spec.name
                pred["heldout_region"] = region
                pred["probability"] = test_prob.astype("float32")
                pred["threshold"] = threshold
                pred["prediction"] = (test_prob >= threshold).astype(int)

                import joblib

                model_path.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump({"model": estimator, "features": features, "threshold": threshold}, model_path)
                atomic_write_csv(metrics_path, pd.DataFrame([metrics]))
                atomic_write_csv(pred_path, pred)
                metrics_rows.append(metrics)
                row.update({"updated_at": now_iso(), "status": "success", "message": "validated"})
                write_status_table(LORO_STATUS_CSV, status_rows)
            except Exception as exc:
                row.update({"updated_at": now_iso(), "status": "failed", "message": repr(exc), "traceback": exception_text(exc)})
                write_status_table(LORO_STATUS_CSV, status_rows)
                logging.exception("LORO validation failed model=%s region=%s", spec.name, region)
    metrics_df = pd.DataFrame(metrics_rows)
    summary = summarize_metrics(metrics_df, ["model"]) if not metrics_df.empty else pd.DataFrame()
    return metrics_df, summary


def training_range_table(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature in features:
        values = pd.to_numeric(df[feature], errors="coerce").dropna()
        rows.append(
            {
                "feature": feature,
                "min": float(values.min()),
                "p05": float(values.quantile(0.05)),
                "p95": float(values.quantile(0.95)),
                "max": float(values.max()),
                "range": float(values.max() - values.min()),
            }
        )
    return pd.DataFrame(rows)


def compute_mess_like(x: np.ndarray, range_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mins = range_df["min"].to_numpy(dtype="float32")
    maxs = range_df["max"].to_numpy(dtype="float32")
    p05 = range_df["p05"].to_numpy(dtype="float32")
    p95 = range_df["p95"].to_numpy(dtype="float32")
    ranges = np.maximum(maxs - mins, 1e-6)
    below = x < mins
    above = x > maxs
    central = (x < p05) | (x > p95)
    similarity = np.minimum((x - mins) / ranges, (maxs - x) / ranges) * 100.0
    similarity[below] = ((x[below] - np.take(mins, np.where(below)[1])) / np.take(ranges, np.where(below)[1])) * 100.0
    similarity[above] = ((np.take(maxs, np.where(above)[1]) - x[above]) / np.take(ranges, np.where(above)[1])) * 100.0
    strict_count = (below | above).sum(axis=1)
    central_count = central.sum(axis=1)
    mess_min = np.nanmin(similarity, axis=1)
    return strict_count, central_count, mess_min, below | above


def summarize_extrapolation(risk: pd.DataFrame, scenario: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    group_sets = [
        ("overall", []),
        ("response", ["Response"]),
        ("sample_type", ["SampleType"]),
        ("validation_region", ["ValidationRegion"]),
    ]
    for group_type, cols in group_sets:
        grouped = [((), risk)] if not cols else risk.groupby(cols, dropna=False)
        for keys, group in grouped:
            if not isinstance(keys, tuple):
                keys = (keys,)
            row: dict[str, Any] = {
                "gcm": scenario.gcm,
                "ssp": scenario.ssp,
                "period": scenario.period,
                "group_type": group_type,
                "n": int(len(group)),
                "valid_future_feature_rate": float(group["valid_future_features"].mean()),
                "strict_extrapolation_rate": float(group["strict_extrapolation"].mean()),
                "central_5_95_outside_rate": float(group["central_5_95_outside"].mean()),
                "mean_strict_outside_count": float(group["strict_outside_count"].mean()),
                "mean_central_outside_count": float(group["central_outside_count"].mean()),
                "mean_mess_min": float(group["mess_min"].mean()),
                "p10_mess_min": float(group["mess_min"].quantile(0.10)),
            }
            for col, value in zip(cols, keys):
                row[col] = value
            rows.append(row)
    return rows


def run_extrapolation(
    df: pd.DataFrame,
    features: list[str],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scenarios = discover_future_scenarios(args.gcm, args.ssp, args.period, args.limit_scenarios)
    range_df = training_range_table(df, features)
    atomic_write_csv(TABLE_DIR / "stage32_training_selected_predictor_ranges.csv", range_df)
    range_df = range_df.set_index("feature").loc[features].reset_index()

    base_cols = [
        col
        for col in [
            "SampleID",
            "Response",
            "SampleType",
            "Region",
            "ValidationRegion",
            "DrylandStratum",
            "SpatialCVFold",
            "PointLon",
            "PointLat",
        ]
        if col in df.columns
    ]
    status_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        out_path = EXTRAP_DIR / f"{scenario.key}_sample_extrapolation_risk.csv.gz"
        row = {
            "updated_at": now_iso(),
            "gcm": scenario.gcm,
            "ssp": scenario.ssp,
            "period": scenario.period,
            "source_tif": str(scenario.path),
            "output_csv_gz": str(out_path),
            "status": "running",
            "message": "",
        }
        status_rows.append(row)
        write_status_table(EXTRAP_STATUS_CSV, status_rows)
        try:
            if out_path.exists() and out_path.stat().st_size > 0 and not args.overwrite:
                risk = pd.read_csv(out_path, low_memory=False)
                summary_rows.extend(summarize_extrapolation(risk, scenario))
                row.update({"updated_at": now_iso(), "status": "skipped", "message": "existing output"})
                write_status_table(EXTRAP_STATUS_CSV, status_rows)
                continue

            logging.info("Sampling future predictors for extrapolation risk: %s", scenario.path)
            x, valid = future_matrix_for_features(scenario.path, df, features)
            strict_count = np.full(len(df), np.nan, dtype="float32")
            central_count = np.full(len(df), np.nan, dtype="float32")
            mess_min = np.full(len(df), np.nan, dtype="float32")
            if valid.any():
                s_count, c_count, m_min, _ = compute_mess_like(x[valid], range_df)
                strict_count[valid] = s_count
                central_count[valid] = c_count
                mess_min[valid] = m_min
            risk = df[base_cols].copy()
            risk.insert(0, "gcm", scenario.gcm)
            risk.insert(1, "ssp", scenario.ssp)
            risk.insert(2, "period", scenario.period)
            risk["valid_future_features"] = valid
            risk["strict_outside_count"] = strict_count
            risk["central_outside_count"] = central_count
            risk["strict_outside_fraction"] = strict_count / len(features)
            risk["central_outside_fraction"] = central_count / len(features)
            risk["mess_min"] = mess_min
            risk["strict_extrapolation"] = risk["strict_outside_count"].fillna(0).gt(0)
            risk["central_5_95_outside"] = risk["central_outside_count"].fillna(0).gt(0)
            atomic_write_csv_gz(out_path, risk)
            summary_rows.extend(summarize_extrapolation(risk[risk["valid_future_features"]], scenario))
            row.update(
                {
                    "updated_at": now_iso(),
                    "status": "success",
                    "message": "risk computed",
                    "n_samples": int(len(risk)),
                    "valid_samples": int(valid.sum()),
                }
            )
            write_status_table(EXTRAP_STATUS_CSV, status_rows)
        except Exception as exc:
            row.update({"updated_at": now_iso(), "status": "failed", "message": repr(exc), "traceback": exception_text(exc)})
            write_status_table(EXTRAP_STATUS_CSV, status_rows)
            logging.exception("Extrapolation risk failed: %s", scenario.key)

    group_summary = pd.DataFrame(summary_rows)
    overall = group_summary[group_summary["group_type"].eq("overall")].copy() if not group_summary.empty else pd.DataFrame()
    return overall, group_summary


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


def write_figures(loro_summary: pd.DataFrame, extrap_overall: pd.DataFrame) -> dict[str, str]:
    base_style()
    figures: dict[str, str] = {}
    if not loro_summary.empty:
        plot_df = loro_summary.sort_values("pr_auc_mean", ascending=True)
        fig, ax = plt.subplots(figsize=(6.4, 3.2))
        ax.barh(plot_df["model"], plot_df["pr_auc_mean"], color="#4C78A8")
        ax.set_xlabel("Leave-one-region-out PR-AUC")
        ax.grid(axis="x", color="#D9D9D9", linewidth=0.5)
        png = FIG_DIR / "fig_stage32_loro_pr_auc_by_model.png"
        svg = FIG_DIR / "fig_stage32_loro_pr_auc_by_model.svg"
        fig.savefig(png, bbox_inches="tight", pad_inches=0.02)
        fig.savefig(svg, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        figures["loro_pr_auc_png"] = str(png)
        figures["loro_pr_auc_svg"] = str(svg)
    if not extrap_overall.empty:
        plot_df = extrap_overall.sort_values(["gcm", "ssp", "period"]).copy()
        plot_df["scenario"] = plot_df["gcm"] + "\n" + plot_df["ssp"] + "\n" + plot_df["period"]
        fig, ax = plt.subplots(figsize=(max(8.0, len(plot_df) * 0.24), 3.5))
        ax.plot(np.arange(len(plot_df)), plot_df["strict_extrapolation_rate"], color="#B8554E", marker="o", markersize=2.5, linewidth=1)
        ax.set_ylabel("Strict extrapolation rate")
        ax.set_xticks(np.arange(len(plot_df)))
        ax.set_xticklabels(plot_df["scenario"], rotation=75, ha="right", fontsize=5.5)
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.5)
        png = FIG_DIR / "fig_stage32_future_strict_extrapolation_rate.png"
        svg = FIG_DIR / "fig_stage32_future_strict_extrapolation_rate.svg"
        fig.savefig(png, bbox_inches="tight", pad_inches=0.02)
        fig.savefig(svg, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        figures["future_extrapolation_png"] = str(png)
        figures["future_extrapolation_svg"] = str(svg)
    return figures


def write_report(summary: dict[str, Any], loro_summary: pd.DataFrame, extrap_overall: pd.DataFrame) -> None:
    lines = [
        "# Stage32 独立区域验证与外推风险诊断报告",
        "",
        f"- 生成时间: {now_iso()}",
        f"- 样本数: {summary['sample_count']}",
        f"- 使用变量: `{', '.join(summary['features'])}`",
        f"- 留一区域验证模型-区域任务成功/跳过/失败: {summary['loro_success_jobs']} / {summary['loro_skipped_jobs']} / {summary['loro_failed_jobs']}",
        f"- 未来情景外推风险成功/跳过/失败: {summary['extrap_success_scenarios']} / {summary['extrap_skipped_scenarios']} / {summary['extrap_failed_scenarios']}",
        "",
        "## 留一区域独立验证摘要",
        "",
        loro_summary.to_markdown(index=False) if not loro_summary.empty else "暂无成功的留一区域验证结果。",
        "",
        "## 未来外推风险 Overall 摘要",
        "",
        extrap_overall.to_markdown(index=False) if not extrap_overall.empty else "暂无成功的未来外推风险结果。",
        "",
        "## 方法说明与限制",
        "",
        "由于原背景点没有 Region 字段，本阶段将背景点按经纬度分配到最近的 presence 区域质心，形成 ValidationRegion 后做 leave-one-region-out 二分类验证。这是对空间独立性的补强诊断，不等同于全新的野外独立样本验证。外推风险使用 Stage30 筛选变量的当前训练范围与未来 WorldClim 样本值比较，报告严格超出当前 min/max 的比例以及 MESS-like 最小相似度。",
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
            "message": "Stage32 independent validation and extrapolation started",
            "features": features,
        },
    )
    write_stage_status("running", "Stage32 started", {"feature_count": len(features)})

    df = load_modeling_samples(features)
    df, region_summary = assign_validation_regions(df)
    atomic_write_csv(REGION_ASSIGNMENT_CSV, region_summary)

    loro_metrics, loro_summary = run_loro_validation(df, features, args.seed, args.n_jobs, args.overwrite)
    atomic_write_csv(LORO_METRICS_CSV, loro_metrics)
    atomic_write_csv(LORO_SUMMARY_CSV, loro_summary)

    extrap_overall, extrap_groups = run_extrapolation(df, features, args)
    atomic_write_csv(EXTRAP_SUMMARY_CSV, extrap_overall)
    atomic_write_csv(EXTRAP_GROUP_SUMMARY_CSV, extrap_groups)
    figures = write_figures(loro_summary, extrap_overall)

    loro_status = pd.read_csv(LORO_STATUS_CSV) if LORO_STATUS_CSV.exists() else pd.DataFrame()
    extrap_status = pd.read_csv(EXTRAP_STATUS_CSV) if EXTRAP_STATUS_CSV.exists() else pd.DataFrame()
    state = {
        "status": "success"
        if (not loro_status.empty and not extrap_status.empty and (loro_status["status"].eq("failed").sum() + extrap_status["status"].eq("failed").sum()) == 0)
        else "partial_success",
        "sample_count": int(len(df)),
        "feature_count": len(features),
        "features": features,
        "loro_success_jobs": int(loro_status["status"].eq("success").sum()) if not loro_status.empty else 0,
        "loro_skipped_jobs": int(loro_status["status"].eq("skipped").sum()) if not loro_status.empty else 0,
        "loro_failed_jobs": int(loro_status["status"].eq("failed").sum()) if not loro_status.empty else 0,
        "extrap_success_scenarios": int(extrap_status["status"].eq("success").sum()) if not extrap_status.empty else 0,
        "extrap_skipped_scenarios": int(extrap_status["status"].eq("skipped").sum()) if not extrap_status.empty else 0,
        "extrap_failed_scenarios": int(extrap_status["status"].eq("failed").sum()) if not extrap_status.empty else 0,
        "outputs": {
            "report_md": str(REPORT_MD),
            "loro_metrics_csv": str(LORO_METRICS_CSV),
            "loro_summary_csv": str(LORO_SUMMARY_CSV),
            "region_assignment_csv": str(REGION_ASSIGNMENT_CSV),
            "extrapolation_summary_csv": str(EXTRAP_SUMMARY_CSV),
            "extrapolation_group_summary_csv": str(EXTRAP_GROUP_SUMMARY_CSV),
            "extrapolation_sample_dir": str(EXTRAP_DIR),
            "loro_status_csv": str(LORO_STATUS_CSV),
            "extrapolation_status_csv": str(EXTRAP_STATUS_CSV),
            "figures": figures,
            "log_path": str(LOG_PATH),
        },
        "started_at": json.loads(STATE_JSON.read_text(encoding="utf-8")).get("started_at", ""),
        "finished_at": now_iso(),
    }
    write_report(state, loro_summary, extrap_overall)
    atomic_write_json(STATE_JSON, state)
    write_stage_status(state["status"], "Stage32 completed", state)
    logging.info("Stage32 completed: %s", json.dumps(state, ensure_ascii=False))
    return state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="补做筛选变量模型的留一区域验证与未来外推风险诊断。")
    parser.add_argument("--seed", type=int, default=20260531)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--gcm", action="append")
    parser.add_argument("--ssp", action="append")
    parser.add_argument("--period", action="append")
    parser.add_argument("--limit-scenarios", type=int, default=None)
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
        logging.exception("Stage32 failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

