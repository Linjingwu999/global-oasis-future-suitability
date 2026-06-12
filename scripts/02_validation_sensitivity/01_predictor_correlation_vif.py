# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import logging
import math
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "stage05_current_worldclim_model_ready"
    / "modeling_samples_current_worldclim_complete_cases.csv"
)
FEATURE_IMPORTANCE_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "stage08_manuscript_figures_and_draft"
    / "tables"
    / "feature_importance_summary_for_manuscript.csv"
)

OUT_DIR = PROJECT_ROOT / "outputs" / "stage30_environment_factor_collinearity"
FIG_DIR = OUT_DIR / "figures"
TABLE_DIR = OUT_DIR / "tables"
LOG_DIR = PROJECT_ROOT / "logs"
LOG_PATH = LOG_DIR / "stage30_environment_factor_collinearity.log"
STATUS_CSV = LOG_DIR / "stage30_environment_factor_collinearity_status.csv"
STATE_JSON = LOG_DIR / "stage30_environment_factor_collinearity_state.json"
SUMMARY_JSON = OUT_DIR / "stage30_environment_factor_collinearity_summary.json"
REPORT_MD = OUT_DIR / "Stage30_环境因子相关性与共线性诊断报告.md"

FEATURES = [f"wc_bio{i:02d}" for i in range(1, 20)] + ["wc_elev_m"]

FEATURE_LABELS = {
    "wc_bio01": "Bio1: annual mean temperature",
    "wc_bio02": "Bio2: mean diurnal range",
    "wc_bio03": "Bio3: isothermality",
    "wc_bio04": "Bio4: temperature seasonality",
    "wc_bio05": "Bio5: max temperature warmest month",
    "wc_bio06": "Bio6: min temperature coldest month",
    "wc_bio07": "Bio7: temperature annual range",
    "wc_bio08": "Bio8: mean temperature wettest quarter",
    "wc_bio09": "Bio9: mean temperature driest quarter",
    "wc_bio10": "Bio10: mean temperature warmest quarter",
    "wc_bio11": "Bio11: mean temperature coldest quarter",
    "wc_bio12": "Bio12: annual precipitation",
    "wc_bio13": "Bio13: precipitation wettest month",
    "wc_bio14": "Bio14: precipitation driest month",
    "wc_bio15": "Bio15: precipitation seasonality",
    "wc_bio16": "Bio16: precipitation wettest quarter",
    "wc_bio17": "Bio17: precipitation driest quarter",
    "wc_bio18": "Bio18: precipitation warmest quarter",
    "wc_bio19": "Bio19: precipitation coldest quarter",
    "wc_elev_m": "Elevation",
}

CLIMATE_GROUPS = {
    "wc_bio01": "temperature_mean",
    "wc_bio02": "temperature_range",
    "wc_bio03": "temperature_ratio",
    "wc_bio04": "temperature_seasonality",
    "wc_bio05": "temperature_extreme",
    "wc_bio06": "temperature_extreme",
    "wc_bio07": "temperature_range",
    "wc_bio08": "temperature_quarter",
    "wc_bio09": "temperature_quarter",
    "wc_bio10": "temperature_quarter",
    "wc_bio11": "temperature_quarter",
    "wc_bio12": "precipitation_total",
    "wc_bio13": "precipitation_extreme",
    "wc_bio14": "precipitation_extreme",
    "wc_bio15": "precipitation_seasonality",
    "wc_bio16": "precipitation_quarter",
    "wc_bio17": "precipitation_quarter",
    "wc_bio18": "precipitation_quarter",
    "wc_bio19": "precipitation_quarter",
    "wc_elev_m": "topography",
}

TASKS = [
    "validate_inputs",
    "load_samples",
    "compute_descriptive_statistics",
    "compute_correlation_matrices",
    "compute_vif_and_selection",
    "write_figures",
    "write_report_and_summary",
    "integrity_check",
]


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


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


def setup_logging() -> None:
    ensure_dirs()
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(LOG_PATH, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


@dataclass
class TaskTracker:
    rows: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        for task in TASKS:
            self.rows.append(
                {
                    "task": task,
                    "status": "pending",
                    "updated_at": "",
                    "message": "",
                    "error": "",
                }
            )
        self.flush()

    def update(self, task: str, status: str, message: str = "", error: str = "") -> None:
        for row in self.rows:
            if row["task"] == task:
                row.update(
                    {
                        "status": status,
                        "updated_at": now_iso(),
                        "message": message,
                        "error": error,
                    }
                )
                break
        else:
            self.rows.append(
                {
                    "task": task,
                    "status": status,
                    "updated_at": now_iso(),
                    "message": message,
                    "error": error,
                }
            )
        self.flush()

    def flush(self) -> None:
        df = pd.DataFrame(self.rows)
        atomic_write_csv(STATUS_CSV, df)
        atomic_write_json(
            STATE_JSON,
            {
                "updated_at": now_iso(),
                "tasks": self.rows,
                "status_counts": df["status"].value_counts().to_dict(),
            },
        )


def base_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "legend.frameon": False,
            "figure.dpi": 150,
            "savefig.dpi": 600,
        }
    )


def save_figure(fig: plt.Figure, stem: str) -> dict[str, str]:
    png = FIG_DIR / f"{stem}.png"
    svg = FIG_DIR / f"{stem}.svg"
    fig.savefig(png, dpi=600, bbox_inches="tight", pad_inches=0.02, transparent=True)
    fig.savefig(svg, bbox_inches="tight", pad_inches=0.02, transparent=True)
    plt.close(fig)

    img = Image.open(png)
    if not (img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info)):
        raise RuntimeError(f"PNG does not contain transparency: {png}")
    return {"png": str(png), "svg": str(svg)}


def validate_inputs() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Missing input modeling sample table: {INPUT_CSV}")
    if INPUT_CSV.stat().st_size <= 0:
        raise RuntimeError(f"Input modeling sample table is empty: {INPUT_CSV}")

    header = pd.read_csv(INPUT_CSV, nrows=0).columns.tolist()
    missing = [col for col in FEATURES if col not in header]
    if missing:
        raise RuntimeError(f"Input table is missing environmental predictors: {missing}")


def load_samples() -> pd.DataFrame:
    usecols = [col for col in ["SampleID", "Response", "SampleType", "Region", "SpatialCVFold"] if col]
    usecols.extend(FEATURES)
    df = pd.read_csv(INPUT_CSV, usecols=lambda c: c in set(usecols), low_memory=False)
    missing = [col for col in FEATURES if col not in df.columns]
    if missing:
        raise RuntimeError(f"Loaded table is missing environmental predictors: {missing}")

    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    before = len(df)
    df = df.dropna(subset=FEATURES).copy()
    if df.empty:
        raise RuntimeError("No complete predictor rows remain after dropping missing values.")
    logging.info("Loaded %s complete predictor rows from %s rows.", len(df), before)
    return df


def read_rf_importance() -> pd.DataFrame:
    if not FEATURE_IMPORTANCE_CSV.exists():
        return pd.DataFrame(
            {
                "feature": FEATURES,
                "rf_mean_importance": np.nan,
                "rf_sd_importance": np.nan,
                "rf_rank": np.nan,
            }
        )

    fi = pd.read_csv(FEATURE_IMPORTANCE_CSV)
    fi = fi[
        (fi["model"] == "random_forest_balanced")
        & (fi["importance_type"] == "gini_importance")
        & (fi["feature"].isin(FEATURES))
    ].copy()
    if fi.empty:
        return pd.DataFrame(
            {
                "feature": FEATURES,
                "rf_mean_importance": np.nan,
                "rf_sd_importance": np.nan,
                "rf_rank": np.nan,
            }
        )

    out = fi.rename(
        columns={
            "mean_importance": "rf_mean_importance",
            "sd_importance": "rf_sd_importance",
        }
    )[["feature", "rf_mean_importance", "rf_sd_importance"]]
    out["rf_rank"] = out["rf_mean_importance"].rank(method="dense", ascending=False).astype(int)
    return out


def descriptive_statistics(df: pd.DataFrame) -> pd.DataFrame:
    stats = df[FEATURES].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).T.reset_index()
    stats = stats.rename(columns={"index": "feature", "50%": "median"})
    stats.insert(1, "feature_label", stats["feature"].map(FEATURE_LABELS))
    stats.insert(2, "group", stats["feature"].map(CLIMATE_GROUPS))
    stats["missing_rows_after_complete_case_filter"] = df[FEATURES].isna().sum().reindex(stats["feature"]).values
    atomic_write_csv(TABLE_DIR / "stage30_predictor_descriptive_stats.csv", stats)
    return stats


def correlation_matrices(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pearson = df[FEATURES].corr(method="pearson")
    spearman = df[FEATURES].corr(method="spearman")
    pearson_out = pearson.reset_index().rename(columns={"index": "feature"})
    spearman_out = spearman.reset_index().rename(columns={"index": "feature"})
    atomic_write_csv(TABLE_DIR / "stage30_pearson_correlation_matrix.csv", pearson_out)
    atomic_write_csv(TABLE_DIR / "stage30_spearman_correlation_matrix.csv", spearman_out)
    return pearson, spearman


def high_correlation_pairs(
    corr: pd.DataFrame,
    method: str,
    threshold: float,
    importance: pd.DataFrame,
) -> pd.DataFrame:
    importance_map = importance.set_index("feature")["rf_mean_importance"].to_dict()
    rows: list[dict[str, Any]] = []
    for i, a in enumerate(FEATURES):
        for b in FEATURES[i + 1 :]:
            r = float(corr.loc[a, b])
            if abs(r) >= threshold:
                ia = importance_map.get(a, np.nan)
                ib = importance_map.get(b, np.nan)
                rows.append(
                    {
                        "method": method,
                        "feature_a": a,
                        "feature_a_label": FEATURE_LABELS[a],
                        "feature_b": b,
                        "feature_b_label": FEATURE_LABELS[b],
                        "correlation": r,
                        "abs_correlation": abs(r),
                        "feature_a_rf_mean_importance": ia,
                        "feature_b_rf_mean_importance": ib,
                        "lower_rf_importance_feature": (
                            a
                            if pd.notna(ia)
                            and pd.notna(ib)
                            and ia < ib
                            else b
                            if pd.notna(ia)
                            and pd.notna(ib)
                            and ib < ia
                            else ""
                        ),
                    }
                )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["abs_correlation", "feature_a", "feature_b"], ascending=[False, True, True])
    return out


def compute_vif_from_correlation(corr: pd.DataFrame, ridge: float = 1e-8) -> pd.DataFrame:
    cols = corr.columns.tolist()
    matrix = corr.to_numpy(dtype="float64")
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    matrix = (matrix + matrix.T) / 2
    np.fill_diagonal(matrix, 1.0)

    condition_number = float(np.linalg.cond(matrix))
    used_pinv = False
    try:
        inv = np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(matrix + np.eye(matrix.shape[0]) * ridge)
        used_pinv = True

    vif = np.diag(inv)
    vif = np.where(vif < 0, np.nan, vif)
    out = pd.DataFrame(
        {
            "feature": cols,
            "feature_label": [FEATURE_LABELS.get(c, c) for c in cols],
            "group": [CLIMATE_GROUPS.get(c, "") for c in cols],
            "vif": vif,
            "condition_number": condition_number,
            "used_pseudoinverse": used_pinv,
        }
    )
    return out.sort_values("vif", ascending=False, na_position="last")


def compute_vif_from_frame(data: pd.DataFrame) -> pd.DataFrame:
    cols = data.columns.tolist()
    x = data.to_numpy(dtype="float64")
    valid = np.isfinite(x).all(axis=1)
    x = x[valid]
    if x.shape[0] <= len(cols) + 1:
        raise RuntimeError("Not enough complete rows to compute VIF.")

    means = x.mean(axis=0)
    stds = x.std(axis=0, ddof=0)
    if np.any(stds == 0):
        zero_cols = [cols[i] for i, value in enumerate(stds) if value == 0]
        raise RuntimeError(f"Cannot compute VIF for zero-variance predictors: {zero_cols}")
    z = (x - means) / stds

    corr = np.corrcoef(z, rowvar=False)
    condition_number = float(np.linalg.cond(np.nan_to_num(corr, nan=0.0)))
    rows: list[dict[str, Any]] = []
    for idx, feature in enumerate(cols):
        y = z[:, idx]
        others = np.delete(z, idx, axis=1)
        coef, *_ = np.linalg.lstsq(others, y, rcond=None)
        pred = others @ coef
        residual = y - pred
        sse = float(np.dot(residual, residual))
        sst = float(np.dot(y - y.mean(), y - y.mean()))
        if sst <= 0:
            r2 = np.nan
            vif = np.nan
        else:
            r2 = max(0.0, min(1.0, 1.0 - sse / sst))
            denom = 1.0 - r2
            vif = math.inf if denom <= 1e-12 else 1.0 / denom
        rows.append(
            {
                "feature": feature,
                "feature_label": FEATURE_LABELS.get(feature, feature),
                "group": CLIMATE_GROUPS.get(feature, ""),
                "vif": vif,
                "r2_against_other_predictors": r2,
                "condition_number": condition_number,
                "used_pseudoinverse": False,
                "vif_method": "least_squares_regression",
            }
        )
    return pd.DataFrame(rows).sort_values("vif", ascending=False, na_position="last")


def choose_drop_candidate(vif_df: pd.DataFrame, importance: pd.DataFrame, threshold: float) -> tuple[str, str]:
    imp = importance.set_index("feature")["rf_mean_importance"].to_dict()
    candidates = vif_df[vif_df["vif"] > threshold].copy()
    if candidates.empty:
        candidates = vif_df.head(1).copy()

    max_vif = candidates["vif"].max()
    near_max = candidates[candidates["vif"] >= max_vif * 0.95].copy()
    near_max["rf_mean_importance"] = near_max["feature"].map(imp).fillna(-1.0)
    near_max = near_max.sort_values(["rf_mean_importance", "vif"], ascending=[True, False])
    feature = str(near_max.iloc[0]["feature"])
    reason = (
        f"VIF={float(near_max.iloc[0]['vif']):.2f} exceeds threshold {threshold:g}; "
        "within the highest-VIF candidates this feature has lower RF importance."
    )
    return feature, reason


def iterative_vif_selection(
    df: pd.DataFrame,
    importance: pd.DataFrame,
    vif_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    remaining = FEATURES.copy()
    history_rows: list[dict[str, Any]] = []
    dropped: dict[str, dict[str, Any]] = {}
    iteration = 0

    while len(remaining) > 2:
        iteration += 1
        vif_df = compute_vif_from_frame(df[remaining])
        max_vif = float(vif_df["vif"].max())
        history_rows.append(
            {
                "iteration": iteration,
                "remaining_feature_count": len(remaining),
                "max_vif": max_vif,
                "max_vif_feature": str(vif_df.iloc[0]["feature"]),
                "action": "stop" if max_vif <= vif_threshold else "drop",
                "dropped_feature": "",
                "reason": "",
            }
        )
        if max_vif <= vif_threshold:
            break

        drop_feature, reason = choose_drop_candidate(vif_df, importance, vif_threshold)
        history_rows[-1]["dropped_feature"] = drop_feature
        history_rows[-1]["reason"] = reason
        dropped[drop_feature] = {"drop_iteration": iteration, "drop_reason": reason}
        remaining.remove(drop_feature)

    final_vif = compute_vif_from_frame(df[remaining])
    final_vif_map = final_vif.set_index("feature")["vif"].to_dict()
    all_vif = compute_vif_from_frame(df[FEATURES])
    all_vif_map = all_vif.set_index("feature")["vif"].to_dict()
    imp = importance.set_index("feature").to_dict(orient="index")

    rec_rows: list[dict[str, Any]] = []
    for feature in FEATURES:
        info = imp.get(feature, {})
        rec_rows.append(
            {
                "feature": feature,
                "feature_label": FEATURE_LABELS[feature],
                "group": CLIMATE_GROUPS[feature],
                "decision": "selected" if feature in remaining else "dropped",
                "drop_iteration": dropped.get(feature, {}).get("drop_iteration", ""),
                "drop_reason": dropped.get(feature, {}).get("drop_reason", ""),
                "vif_all_predictors": all_vif_map.get(feature, np.nan),
                "vif_final_selected_set": final_vif_map.get(feature, np.nan) if feature in remaining else np.nan,
                "rf_mean_importance": info.get("rf_mean_importance", np.nan),
                "rf_sd_importance": info.get("rf_sd_importance", np.nan),
                "rf_rank": info.get("rf_rank", np.nan),
            }
        )

    history = pd.DataFrame(history_rows)
    recommendation = pd.DataFrame(rec_rows).sort_values(
        ["decision", "rf_mean_importance"], ascending=[False, False], na_position="last"
    )
    return history, recommendation


def write_selected_sample(df: pd.DataFrame, recommendation: pd.DataFrame) -> Path:
    selected = recommendation.loc[recommendation["decision"] == "selected", "feature"].tolist()
    metadata_cols = [
        col
        for col in ["SampleID", "Response", "SampleType", "Region", "SpatialCVFold"]
        if col in df.columns
    ]
    out_path = TABLE_DIR / "stage30_modeling_samples_selected_predictors.csv"
    atomic_write_csv(out_path, df[metadata_cols + selected])
    atomic_write_text(TABLE_DIR / "stage30_selected_predictor_list.txt", "\n".join(selected) + "\n")
    return out_path


def plot_correlation_heatmap(corr: pd.DataFrame, stem: str, method_label: str) -> dict[str, str]:
    base_style()
    labels = corr.columns.tolist()
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    image = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=60, ha="right", rotation_mode="anchor")
    ax.set_yticklabels(labels)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    cbar = fig.colorbar(image, ax=ax, shrink=0.82, pad=0.02)
    cbar.set_label(f"{method_label} r")
    return save_figure(fig, stem)


def plot_vif(vif_df: pd.DataFrame, stem: str, threshold: float) -> dict[str, str]:
    base_style()
    plot_df = vif_df.sort_values("vif", ascending=True).copy()
    finite_vif = plot_df["vif"].replace([np.inf, -np.inf], np.nan).dropna()
    cap = max(threshold * 3, float(finite_vif.quantile(0.9))) if not finite_vif.empty else threshold * 3
    plot_df["vif_plot"] = plot_df["vif"].map(lambda value: min(float(value), cap) if np.isfinite(value) else cap)
    colors = np.where(plot_df["vif"] > threshold, "#B8554E", "#4C78A8")

    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    ax.barh(plot_df["feature"], plot_df["vif_plot"], color=colors, alpha=0.9)
    ax.axvline(threshold, color="#333333", linestyle="--", linewidth=0.9)
    ax.set_xlabel("Variance inflation factor (VIF)")
    ax.set_ylabel("")
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.5)
    for y, row in enumerate(plot_df.itertuples(index=False)):
        vif_value = getattr(row, "vif")
        if pd.notna(vif_value) and (not np.isfinite(vif_value) or vif_value > plot_df["vif_plot"].max() * 0.96):
            label = " inf" if not np.isfinite(vif_value) else f" {vif_value:.0f}"
            ax.text(
                getattr(row, "vif_plot"),
                y,
                label,
                va="center",
                ha="left",
                fontsize=6.5,
                color="#333333",
            )
    return save_figure(fig, stem)


def plot_selected_vif(recommendation: pd.DataFrame, stem: str, threshold: float) -> dict[str, str]:
    selected = recommendation[recommendation["decision"] == "selected"].copy()
    selected = selected.sort_values("vif_final_selected_set", ascending=True)
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.barh(selected["feature"], selected["vif_final_selected_set"], color="#4E8F5B", alpha=0.9)
    ax.axvline(threshold, color="#333333", linestyle="--", linewidth=0.9)
    ax.set_xlabel("Final selected-set VIF")
    ax.set_ylabel("")
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.5)
    return save_figure(fig, stem)


def write_report(
    df: pd.DataFrame,
    pearson_pairs: pd.DataFrame,
    spearman_pairs: pd.DataFrame,
    vif_all: pd.DataFrame,
    history: pd.DataFrame,
    recommendation: pd.DataFrame,
    figures: dict[str, dict[str, str]],
    corr_threshold: float,
    vif_threshold: float,
    selected_sample_path: Path,
) -> dict[str, Any]:
    selected = recommendation[recommendation["decision"] == "selected"].copy()
    dropped = recommendation[recommendation["decision"] == "dropped"].copy()

    top_pairs = pearson_pairs.head(12)
    top_vif = vif_all.head(12)
    selected_features = selected["feature"].tolist()
    dropped_features = dropped["feature"].tolist()
    max_vif = float(vif_all["vif"].max())
    max_vif_text = "infinite" if not np.isfinite(max_vif) else f"{max_vif:.2f}"

    report_lines = [
        "# Stage30 环境因子相关性与共线性诊断报告",
        "",
        f"- 生成时间: {now_iso()}",
        f"- 输入样本表: `{INPUT_CSV}`",
        f"- 样本数: {len(df)}",
        f"- 环境因子数: {len(FEATURES)} (`wc_bio01`-`wc_bio19` + `wc_elev_m`)",
        f"- 强相关阈值: `|r| >= {corr_threshold:g}`",
        f"- VIF 阈值: `{vif_threshold:g}`",
        "",
        "## 结论",
        "",
        f"1. 全变量方案存在明显共线性：Pearson 强相关变量对共 {len(pearson_pairs)} 对，Spearman 强相关变量对共 {len(spearman_pairs)} 对。",
        f"2. 全变量 VIF 最高为 {max_vif_text}，说明 19 个 bioclim 变量之间存在强冗余，原全变量模型不宜直接用于精细机制解释。",
        f"3. 按 VIF 迭代筛选后，推荐保留 {len(selected_features)} 个变量: `{', '.join(selected_features)}`。",
        "4. 已有全变量适宜面积结果建议保留为 baseline/sensitivity；正式论文主线应使用筛选变量模型重训后再重新计算适宜面积，并与全变量面积做稳健性对比。",
        "",
        "## 关于土地覆盖变量",
        "",
        "本阶段不把土地覆盖数据并入 `WorldClim + elevation` 的 Pearson/VIF 诊断，原因是当前 Stage06 建模表中的预测变量只有连续气候因子和高程；Stage20 土地覆盖目前用于空间约束或后处理，不是同一个基线模型的输入变量。若后续要把土地覆盖比例、耕地/水体频率、兼容度百分比等连续变量作为模型预测因子，则应另建增强版建模表，并对这些连续变量一起做相关性和 VIF；若使用离散土地覆盖类别，则更适合补充类别分布、Cramer's V、卡方检验或基于面积的约束敏感性分析，而不是直接纳入 Pearson/VIF。",
        "",
        "## 推荐保留变量",
        "",
        selected[
            [
                "feature",
                "feature_label",
                "group",
                "vif_final_selected_set",
                "rf_mean_importance",
                "rf_rank",
            ]
        ].to_markdown(index=False),
        "",
        "## 被剔除变量",
        "",
    ]
    if dropped.empty:
        report_lines.append("无。")
    else:
        report_lines.append(
            dropped[
                [
                    "feature",
                    "feature_label",
                    "group",
                    "vif_all_predictors",
                    "rf_mean_importance",
                    "drop_iteration",
                    "drop_reason",
                ]
            ].to_markdown(index=False)
        )

    report_lines.extend(
        [
            "",
            "## Pearson 强相关变量对 Top 12",
            "",
        ]
    )
    report_lines.append(
        top_pairs[
            [
                "feature_a",
                "feature_b",
                "correlation",
                "feature_a_rf_mean_importance",
                "feature_b_rf_mean_importance",
                "lower_rf_importance_feature",
            ]
        ].to_markdown(index=False)
        if not top_pairs.empty
        else "未发现达到阈值的强相关变量对。"
    )

    report_lines.extend(
        [
            "",
            "## 全变量 VIF Top 12",
            "",
            top_vif[
                ["feature", "feature_label", "group", "vif", "condition_number", "used_pseudoinverse"]
            ].to_markdown(index=False),
            "",
            "## VIF 迭代筛选过程",
            "",
            history.to_markdown(index=False),
            "",
            "## 输出文件",
            "",
            f"- 推荐变量样本表: `{selected_sample_path}`",
            f"- Pearson 相关矩阵: `{TABLE_DIR / 'stage30_pearson_correlation_matrix.csv'}`",
            f"- Spearman 相关矩阵: `{TABLE_DIR / 'stage30_spearman_correlation_matrix.csv'}`",
            f"- 强相关变量对: `{TABLE_DIR / 'stage30_high_correlation_pairs.csv'}`",
            f"- 全变量 VIF: `{TABLE_DIR / 'stage30_vif_all_predictors.csv'}`",
            f"- VIF 筛选过程: `{TABLE_DIR / 'stage30_iterative_vif_selection.csv'}`",
            f"- 推荐变量表: `{TABLE_DIR / 'stage30_recommended_predictors.csv'}`",
            f"- 推荐变量清单: `{TABLE_DIR / 'stage30_selected_predictor_list.txt'}`",
            f"- Pearson 热图 PNG/SVG: `{figures.get('pearson_heatmap', {}).get('png')}` / `{figures.get('pearson_heatmap', {}).get('svg')}`",
            f"- 全变量 VIF 图 PNG/SVG: `{figures.get('vif_all', {}).get('png')}` / `{figures.get('vif_all', {}).get('svg')}`",
            f"- 筛选后 VIF 图 PNG/SVG: `{figures.get('vif_selected', {}).get('png')}` / `{figures.get('vif_selected', {}).get('svg')}`",
            "",
            "## 后续建模建议",
            "",
            "1. 用 `stage30_modeling_samples_selected_predictors.csv` 重训 Stage06 空间交叉验证模型，保留相同 SpatialCVFold，以便和原模型公平比较。",
            "2. 若筛选变量模型的 AUC/TSS 和空间格局接近原模型，可将原模型作为全变量敏感性结果，正式面积以筛选变量模型为主。",
            "3. 重训后再执行未来样本预测、栅格投影、地形/土地覆盖约束和适宜面积汇总，生成筛选变量版面积结果。",
            "4. 土地覆盖约束建议单独补一张 Stage20/Stage29 敏感性表：报告约束前面积、土地覆盖约束后面积、剔除面积比例、各土地覆盖兼容等级贡献。",
        ]
    )

    atomic_write_text(REPORT_MD, "\n".join(report_lines) + "\n")

    summary = {
        "status": "success",
        "generated_at": now_iso(),
        "input_csv": str(INPUT_CSV),
        "sample_count": int(len(df)),
        "feature_count_before": len(FEATURES),
        "feature_count_selected": int(len(selected_features)),
        "correlation_threshold_abs": corr_threshold,
        "vif_threshold": vif_threshold,
        "pearson_high_pair_count": int(len(pearson_pairs)),
        "spearman_high_pair_count": int(len(spearman_pairs)),
        "max_vif_all_predictors": max_vif_text,
        "selected_features": selected_features,
        "dropped_features": dropped_features,
        "report_md": str(REPORT_MD),
        "selected_sample_csv": str(selected_sample_path),
        "tables_dir": str(TABLE_DIR),
        "figures": figures,
        "status_csv": str(STATUS_CSV),
        "state_json": str(STATE_JSON),
        "log_path": str(LOG_PATH),
    }
    atomic_write_json(SUMMARY_JSON, summary)
    return summary


def integrity_check(summary: dict[str, Any]) -> None:
    required_paths = [
        REPORT_MD,
        SUMMARY_JSON,
        TABLE_DIR / "stage30_predictor_descriptive_stats.csv",
        TABLE_DIR / "stage30_pearson_correlation_matrix.csv",
        TABLE_DIR / "stage30_spearman_correlation_matrix.csv",
        TABLE_DIR / "stage30_high_correlation_pairs.csv",
        TABLE_DIR / "stage30_vif_all_predictors.csv",
        TABLE_DIR / "stage30_iterative_vif_selection.csv",
        TABLE_DIR / "stage30_recommended_predictors.csv",
        TABLE_DIR / "stage30_modeling_samples_selected_predictors.csv",
        TABLE_DIR / "stage30_selected_predictor_list.txt",
        STATUS_CSV,
        STATE_JSON,
        LOG_PATH,
    ]
    for path in required_paths:
        if not path.exists() or path.stat().st_size <= 0:
            raise RuntimeError(f"Required output missing or empty: {path}")

    for figure in summary.get("figures", {}).values():
        for path_text in figure.values():
            path = Path(path_text)
            if not path.exists() or path.stat().st_size <= 0:
                raise RuntimeError(f"Figure output missing or empty: {path}")
            if path.suffix.lower() == ".png":
                img = Image.open(path)
                if not (
                    img.mode in ("RGBA", "LA")
                    or (img.mode == "P" and "transparency" in img.info)
                ):
                    raise RuntimeError(f"PNG lacks alpha channel: {path}")


def run(force: bool, corr_threshold: float, vif_threshold: float) -> dict[str, Any]:
    ensure_dirs()
    setup_logging()
    tracker = TaskTracker()

    if SUMMARY_JSON.exists() and not force:
        existing = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))
        if existing.get("status") == "success":
            logging.info("Existing successful Stage30 summary found; use --force to recompute.")
            return existing

    try:
        tracker.update("validate_inputs", "running", "Checking input table and predictor columns.")
        validate_inputs()
        tracker.update("validate_inputs", "success", "Input table and predictor columns are available.")

        tracker.update("load_samples", "running", "Loading complete current WorldClim modeling samples.")
        df = load_samples()
        tracker.update("load_samples", "success", f"Loaded {len(df)} complete rows.")

        tracker.update("compute_descriptive_statistics", "running", "Writing predictor summary statistics.")
        stats = descriptive_statistics(df)
        tracker.update(
            "compute_descriptive_statistics",
            "success",
            f"Wrote statistics for {len(stats)} predictors.",
        )

        tracker.update("compute_correlation_matrices", "running", "Computing Pearson and Spearman correlations.")
        pearson, spearman = correlation_matrices(df)
        importance = read_rf_importance()
        pearson_pairs = high_correlation_pairs(pearson, "pearson", corr_threshold, importance)
        spearman_pairs = high_correlation_pairs(spearman, "spearman", corr_threshold, importance)
        high_pairs = pd.concat([pearson_pairs, spearman_pairs], ignore_index=True)
        atomic_write_csv(TABLE_DIR / "stage30_high_correlation_pairs.csv", high_pairs)
        tracker.update(
            "compute_correlation_matrices",
            "success",
            f"Pearson pairs={len(pearson_pairs)}, Spearman pairs={len(spearman_pairs)}.",
        )

        tracker.update("compute_vif_and_selection", "running", "Computing VIF and iterative variable selection.")
        vif_all = compute_vif_from_frame(df[FEATURES])
        vif_all = vif_all.merge(importance, on="feature", how="left")
        atomic_write_csv(TABLE_DIR / "stage30_vif_all_predictors.csv", vif_all)
        history, recommendation = iterative_vif_selection(df, importance, vif_threshold)
        atomic_write_csv(TABLE_DIR / "stage30_iterative_vif_selection.csv", history)
        atomic_write_csv(TABLE_DIR / "stage30_recommended_predictors.csv", recommendation)
        selected_sample_path = write_selected_sample(df, recommendation)
        tracker.update(
            "compute_vif_and_selection",
            "success",
            f"Selected {(recommendation['decision'] == 'selected').sum()} predictors.",
        )

        figures: dict[str, dict[str, str]] = {}
        tracker.update("write_figures", "running", "Exporting transparent PNG/SVG diagnostic figures.")
        try:
            figures["pearson_heatmap"] = plot_correlation_heatmap(
                pearson,
                "fig_stage30_pearson_correlation_heatmap",
                "Pearson",
            )
            figures["spearman_heatmap"] = plot_correlation_heatmap(
                spearman,
                "fig_stage30_spearman_correlation_heatmap",
                "Spearman",
            )
            figures["vif_all"] = plot_vif(vif_all, "fig_stage30_vif_all_predictors", vif_threshold)
            figures["vif_selected"] = plot_selected_vif(
                recommendation,
                "fig_stage30_vif_selected_predictors",
                vif_threshold,
            )
            tracker.update("write_figures", "success", f"Exported {len(figures)} figures.")
        except Exception as exc:
            tracker.update("write_figures", "failed", "Figure export failed; tables will still be reported.", traceback.format_exc())
            logging.exception("Figure export failed: %s", exc)

        tracker.update("write_report_and_summary", "running", "Writing Markdown report and JSON summary.")
        summary = write_report(
            df=df,
            pearson_pairs=pearson_pairs,
            spearman_pairs=spearman_pairs,
            vif_all=vif_all,
            history=history,
            recommendation=recommendation,
            figures=figures,
            corr_threshold=corr_threshold,
            vif_threshold=vif_threshold,
            selected_sample_path=selected_sample_path,
        )
        tracker.update("write_report_and_summary", "success", "Report and summary written.")

        tracker.update("integrity_check", "running", "Checking required output files and PNG alpha channels.")
        integrity_check(summary)
        tracker.update("integrity_check", "success", "All required outputs are present and non-empty.")
        logging.info("Stage30 completed successfully: %s", REPORT_MD)
        return summary

    except Exception:
        error = traceback.format_exc()
        logging.error("Stage30 failed:\n%s", error)
        atomic_write_json(
            SUMMARY_JSON,
            {
                "status": "failed",
                "generated_at": now_iso(),
                "error": error,
                "status_csv": str(STATUS_CSV),
                "state_json": str(STATE_JSON),
                "log_path": str(LOG_PATH),
            },
        )
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Supplement environmental predictor correlation, VIF, and collinearity diagnostics."
    )
    parser.add_argument("--force", action="store_true", help="Recompute even if a successful Stage30 summary exists.")
    parser.add_argument("--corr-threshold", type=float, default=0.85, help="Absolute correlation threshold for high-pair reporting.")
    parser.add_argument("--vif-threshold", type=float, default=10.0, help="VIF threshold for iterative predictor filtering.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run(
        force=bool(args.force),
        corr_threshold=float(args.corr_threshold),
        vif_threshold=float(args.vif_threshold),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary.get("status") == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
