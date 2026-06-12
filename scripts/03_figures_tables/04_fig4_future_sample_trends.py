#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Create Fig4 v06 candidate from selected10 HGB/RF future sample predictions.

This version replaces the old 2x2 HGB-only candidate with a differentiated
two-column layout using the workstation-returned selected10 HGB and RF outputs.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGE_DIR = PROJECT_ROOT / "outputs" / "stage39_manuscript_main_figures"
FIG_DIR = STAGE_DIR / "figures"
TABLE_DIR = STAGE_DIR / "tables"
LOG_DIR = PROJECT_ROOT / "logs"

INPUT_DIR = PROJECT_ROOT / "outputs" / "stage34_selected10_future_worldclim_sample_predictions"
INPUTS = {
    "HGB": INPUT_DIR / "future_worldclim_sample_prediction_summary_selected10_hgb.csv",
    "RF": INPUT_DIR / "future_worldclim_sample_prediction_summary_selected10_rf.csv",
}
STATUS_INPUTS = {
    "HGB": LOG_DIR / "stage07_future_worldclim_sample_predictions_selected10_hgb_status.csv",
    "RF": LOG_DIR / "stage07_future_worldclim_sample_predictions_selected10_rf_status.csv",
}

FIG_BASENAME = "fig_stage39_fig4_future_sample_suitability_trends_v06"
STATUS_JSON = STAGE_DIR / f"{FIG_BASENAME}_status.json"
README_MD = STAGE_DIR / f"{FIG_BASENAME}_README.md"
LOG_PATH = LOG_DIR / "stage39_fig4_future_sample_suitability_trends_v06.log"

REQUIRED_COLUMNS = {
    "gcm",
    "ssp",
    "period",
    "model_group",
    "n",
    "mean_probability",
    "median_probability",
    "p10_probability",
    "p90_probability",
    "suitable_rate",
}
COPY_COLUMNS = [
    "model_short",
    "gcm",
    "ssp",
    "period",
    "model_group",
    "n",
    "mean_probability",
    "median_probability",
    "p10_probability",
    "p90_probability",
    "suitable_rate",
    "suitable_rate_pct",
]

SSP_ORDER = ["ssp126", "ssp245", "ssp370", "ssp585"]
PERIOD_ORDER = ["2021-2040", "2041-2060", "2061-2080", "2081-2100"]
GCM_ORDER = ["ACCESS-CM2", "MPI-ESM1-2-HR", "MRI-ESM2-0"]
MODEL_ORDER = ["HGB", "RF"]
SSP_LABELS = {
    "ssp126": "SSP1-2.6",
    "ssp245": "SSP2-4.5",
    "ssp370": "SSP3-7.0",
    "ssp585": "SSP5-8.5",
}
PERIOD_SHORT = {
    "2021-2040": "2021-40",
    "2041-2060": "2041-60",
    "2061-2080": "2061-80",
    "2081-2100": "2081-2100",
}

SSP_COLORS = {
    "ssp126": "#3D74A9",
    "ssp245": "#278178",
    "ssp370": "#C47A2B",
    "ssp585": "#C75A61",
}
MODEL_STYLES = {
    "HGB": {
        "linestyle": "-",
        "marker": "o",
        "linewidth": 2.1,
        "markersize": 4.8,
        "alpha": 0.96,
    },
    "RF": {
        "linestyle": "--",
        "marker": "s",
        "linewidth": 1.85,
        "markersize": 4.1,
        "alpha": 0.92,
    },
}
MODEL_OFFSETS = {"HGB": 0.16, "RF": -0.16}

TEXT = "#25313A"
MUTED = "#66737C"
GRID = "#DDE6E4"
SPINE = "#D5DFDA"
RANGE_LINE = "#BAC7C9"


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def setup_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=LOG_PATH,
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        encoding="utf-8",
    )


def write_status(status: str, **payload: Any) -> None:
    STAGE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = STATUS_JSON.with_suffix(".json.tmp")
    data = {
        "status": status,
        "updated_at": now_iso(),
        "script": str(Path(__file__).resolve()),
        "figure_basename": FIG_BASENAME,
        "word_insertion_allowed": False,
        "user_confirmed_final": False,
        **payload,
    }
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(STATUS_JSON)


def read_status(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "status": "missing"}
    df = pd.read_csv(path)
    if df.empty:
        return {"exists": True, "status": "empty"}
    row = df.iloc[-1].to_dict()
    row["exists"] = True
    return row


def validate_run_statuses() -> dict[str, dict[str, Any]]:
    statuses: dict[str, dict[str, Any]] = {}
    for model, path in STATUS_INPUTS.items():
        status = read_status(path)
        statuses[model] = status
        if str(status.get("status", "")).lower() != "success":
            raise RuntimeError(f"{model} selected10 sample prediction is not success: {status}")
        total = int(float(status.get("total_scenarios", 0) or 0))
        success = int(float(status.get("success_scenarios", 0) or 0))
        failed = int(float(status.get("failed_scenarios", 0) or 0))
        if total != 48 or success != 48 or failed != 0:
            raise RuntimeError(f"{model} selected10 scenario count is unexpected: {status}")
    return statuses


def load_one_summary(model_short: str, path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing selected10 summary for {model_short}: {path}")
    df = pd.read_csv(path)
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"{model_short} summary is missing columns: {missing}")
    expected = len(GCM_ORDER) * len(SSP_ORDER) * len(PERIOD_ORDER)
    if len(df) != expected:
        raise ValueError(f"{model_short} summary expected {expected} rows, found {len(df)}")

    df = df.copy()
    df["model_short"] = model_short
    df["gcm"] = pd.Categorical(df["gcm"], categories=GCM_ORDER, ordered=True)
    df["ssp"] = pd.Categorical(df["ssp"], categories=SSP_ORDER, ordered=True)
    df["period"] = pd.Categorical(df["period"], categories=PERIOD_ORDER, ordered=True)
    if df[["gcm", "ssp", "period"]].isna().any().any():
        bad = df[df[["gcm", "ssp", "period"]].isna().any(axis=1)].head(10)
        raise ValueError(f"Unknown labels in {model_short}: {bad.to_dict(orient='records')}")

    numeric_cols = [
        "n",
        "mean_probability",
        "median_probability",
        "p10_probability",
        "p90_probability",
        "suitable_rate",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="raise")
    if (df["suitable_rate"] < 0).any() or (df["suitable_rate"] > 1).any():
        raise ValueError(f"{model_short} suitable_rate is outside [0, 1].")
    df["period_index"] = df["period"].cat.codes.astype(float)
    df["suitable_rate_pct"] = df["suitable_rate"] * 100.0

    coverage = (
        df.groupby(["gcm", "ssp"], observed=True)["period"]
        .nunique()
        .reset_index(name="n_period")
    )
    incomplete = coverage[coverage["n_period"] != len(PERIOD_ORDER)]
    if not incomplete.empty:
        raise ValueError(
            f"Incomplete period coverage in {model_short}: "
            + json.dumps(incomplete.to_dict(orient="records"), ensure_ascii=False)
        )
    return df


def load_data() -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    statuses = validate_run_statuses()
    frames = [load_one_summary(model, path) for model, path in INPUTS.items()]
    df = pd.concat(frames, ignore_index=True)
    sample_counts = df.groupby("model_short", observed=True)["n"].nunique().to_dict()
    if any(count != 1 for count in sample_counts.values()):
        raise ValueError(f"Unexpected sample count variation: {sample_counts}")
    return df, statuses


def make_aggregates(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    agg = (
        df.groupby(["model_short", "ssp", "period"], observed=True)
        .agg(
            period_index=("period_index", "first"),
            rate_mean_pct=("suitable_rate_pct", "mean"),
            rate_min_pct=("suitable_rate_pct", "min"),
            rate_max_pct=("suitable_rate_pct", "max"),
            prob_mean=("mean_probability", "mean"),
            prob_min=("mean_probability", "min"),
            prob_max=("mean_probability", "max"),
            n_gcm=("gcm", "nunique"),
            n_samples=("n", "max"),
        )
        .reset_index()
        .sort_values(["model_short", "ssp", "period"])
    )

    early = df[df["period"].astype(str) == PERIOD_ORDER[0]][
        ["model_short", "gcm", "ssp", "suitable_rate_pct", "mean_probability"]
    ].rename(
        columns={
            "suitable_rate_pct": "early_rate_pct",
            "mean_probability": "early_mean_probability",
        }
    )
    late = df[df["period"].astype(str) == PERIOD_ORDER[-1]][
        ["model_short", "gcm", "ssp", "suitable_rate_pct", "mean_probability"]
    ].rename(
        columns={
            "suitable_rate_pct": "late_rate_pct",
            "mean_probability": "late_mean_probability",
        }
    )
    change_by_gcm = early.merge(late, on=["model_short", "gcm", "ssp"], how="inner")
    change_by_gcm["rate_change_pp"] = change_by_gcm["late_rate_pct"] - change_by_gcm["early_rate_pct"]
    change_by_gcm["probability_change"] = (
        change_by_gcm["late_mean_probability"] - change_by_gcm["early_mean_probability"]
    )
    change_by_gcm = change_by_gcm.sort_values(["model_short", "ssp", "gcm"])

    change_mean = (
        change_by_gcm.groupby(["model_short", "ssp"], observed=True)
        .agg(
            rate_change_mean_pp=("rate_change_pp", "mean"),
            rate_change_min_pp=("rate_change_pp", "min"),
            rate_change_max_pp=("rate_change_pp", "max"),
            probability_change_mean=("probability_change", "mean"),
            late_rate_mean_pct=("late_rate_pct", "mean"),
            late_rate_min_pct=("late_rate_pct", "min"),
            late_rate_max_pct=("late_rate_pct", "max"),
            n_gcm=("gcm", "nunique"),
        )
        .reset_index()
        .sort_values(["model_short", "ssp"])
    )
    return {
        "scenario_period_model_aggregate": agg,
        "end_century_change_by_gcm": change_by_gcm,
        "end_century_change_model_mean": change_mean,
    }


def clean_for_csv(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.select_dtypes(include=["category"]).columns:
        out[col] = out[col].astype(str)
    return out


def export_tables(df: pd.DataFrame, aggregates: dict[str, pd.DataFrame]) -> dict[str, str]:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}
    for model_short in MODEL_ORDER:
        path = TABLE_DIR / f"fig4_v06_input_selected10_{model_short.lower()}_summary.csv"
        clean_for_csv(df[df["model_short"] == model_short][COPY_COLUMNS]).to_csv(
            path, index=False, encoding="utf-8-sig"
        )
        outputs[f"input_selected10_{model_short.lower()}_summary"] = rel(path)
    for name, table in aggregates.items():
        path = TABLE_DIR / f"fig4_v06_{name}.csv"
        clean_for_csv(table).to_csv(path, index=False, encoding="utf-8-sig")
        outputs[name] = rel(path)
    return outputs


def style_axis(ax: plt.Axes, grid_axis: str = "both") -> None:
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_color(SPINE)
        spine.set_linewidth(0.9)
    if grid_axis in {"both", "x"}:
        ax.grid(True, axis="x", color=GRID, linewidth=0.75, alpha=0.85)
    if grid_axis in {"both", "y"}:
        ax.grid(True, axis="y", color=GRID, linewidth=0.75, alpha=0.85)
    ax.tick_params(axis="both", labelsize=8.2, colors=TEXT, length=3)
    ax.set_axisbelow(True)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.055,
        1.04,
        label,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=8.8,
        fontweight="bold",
        color="white",
        bbox=dict(boxstyle="round,pad=0.18,rounding_size=0.04", facecolor="#2B837D", edgecolor="none"),
        clip_on=False,
    )


def signed_one_decimal(value: float) -> str:
    display = 0.0 if abs(value) < 0.05 else value
    return f"{display:+.1f}" if display != 0 else "0.0"


def plot_trajectory(ax: plt.Axes, agg: pd.DataFrame) -> None:
    x = np.arange(len(PERIOD_ORDER), dtype=float)
    for model_short in MODEL_ORDER:
        model_df = agg[agg["model_short"] == model_short]
        style = MODEL_STYLES[model_short]
        for ssp in SSP_ORDER:
            sub = model_df[model_df["ssp"].astype(str) == ssp].sort_values("period")
            color = SSP_COLORS[ssp]
            y = sub["rate_mean_pct"].to_numpy()
            yerr_low = y - sub["rate_min_pct"].to_numpy()
            yerr_high = sub["rate_max_pct"].to_numpy() - y
            ax.errorbar(
                x,
                y,
                yerr=np.vstack([yerr_low, yerr_high]),
                color=color,
                linestyle=style["linestyle"],
                marker=style["marker"],
                linewidth=style["linewidth"],
                markersize=style["markersize"],
                markerfacecolor="white" if model_short == "RF" else color,
                markeredgecolor=color,
                markeredgewidth=1.0,
                ecolor=color,
                elinewidth=0.75,
                capsize=2.2,
                alpha=style["alpha"],
                zorder=3 if model_short == "HGB" else 2,
            )

    style_axis(ax)
    ax.set_xlim(-0.18, 3.18)
    ax.set_ylim(10.0, 31.5)
    ax.set_xticks(x, [PERIOD_SHORT[p] for p in PERIOD_ORDER])
    ax.set_yticks([10, 15, 20, 25, 30])
    ax.set_xlabel("Projection period", fontsize=8.8, color=TEXT, labelpad=5)
    ax.set_ylabel("Suitable sample rate (%)", fontsize=8.8, color=TEXT, labelpad=5)
    panel_label(ax, "a")
    ax.text(
        0.02,
        0.97,
        "GCM range",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.4,
        color=MUTED,
    )

    ssp_handles = [
        plt.Line2D([0], [0], color=SSP_COLORS[ssp], lw=2.2, marker="o", markersize=4, label=SSP_LABELS[ssp])
        for ssp in SSP_ORDER
    ]
    model_handles = [
        plt.Line2D(
            [0],
            [0],
            color="#4B565F",
            lw=MODEL_STYLES[model]["linewidth"],
            linestyle=MODEL_STYLES[model]["linestyle"],
            marker=MODEL_STYLES[model]["marker"],
            markerfacecolor="white" if model == "RF" else "#4B565F",
            markersize=4.2,
            label=model,
        )
        for model in MODEL_ORDER
    ]
    leg1 = ax.legend(
        handles=ssp_handles,
        loc="lower left",
        bbox_to_anchor=(0.015, 0.03),
        frameon=True,
        framealpha=0.96,
        edgecolor=SPINE,
        facecolor="white",
        fontsize=7.4,
        ncol=2,
        handlelength=1.35,
        columnspacing=0.9,
        borderpad=0.35,
        labelspacing=0.35,
    )
    ax.add_artist(leg1)
    ax.legend(
        handles=model_handles,
        loc="lower right",
        bbox_to_anchor=(0.985, 1.012),
        frameon=True,
        framealpha=0.96,
        edgecolor=SPINE,
        facecolor="white",
        fontsize=7.4,
        handlelength=1.55,
        borderpad=0.35,
        labelspacing=0.35,
    )


def plot_endpoint_forest(ax: plt.Axes, change_mean: pd.DataFrame) -> None:
    style_axis(ax, grid_axis="x")
    base_positions = np.arange(len(SSP_ORDER))[::-1].astype(float)
    ytick_positions = base_positions
    ax.axvline(0, color="#8B9AA2", lw=1.0, linestyle=(0, (3, 3)), zorder=1)
    ax.text(
        0.08,
        base_positions[0] + 0.38,
        "no change",
        ha="left",
        va="center",
        fontsize=7.2,
        color="#7A8790",
    )
    for model_short in MODEL_ORDER:
        offset = MODEL_OFFSETS[model_short]
        for idx, ssp in enumerate(SSP_ORDER):
            y = base_positions[idx] + offset
            row = change_mean[
                (change_mean["model_short"] == model_short)
                & (change_mean["ssp"].astype(str) == ssp)
            ].iloc[0]
            mean = float(row["rate_change_mean_pp"])
            min_v = float(row["rate_change_min_pp"])
            max_v = float(row["rate_change_max_pp"])
            color = SSP_COLORS[ssp]
            marker = MODEL_STYLES[model_short]["marker"]
            ax.plot([min_v, max_v], [y, y], color=color, lw=2.05, alpha=0.36, solid_capstyle="round", zorder=2)
            ax.plot([min_v, min_v], [y - 0.055, y + 0.055], color=color, lw=0.95, alpha=0.62, zorder=2)
            ax.plot([max_v, max_v], [y - 0.055, y + 0.055], color=color, lw=0.95, alpha=0.62, zorder=2)
            ax.scatter(
                [mean],
                [y],
                s=34 if model_short == "HGB" else 28,
                marker=marker,
                facecolor=color if model_short == "HGB" else "white",
                edgecolor=color,
                linewidth=1.05,
                zorder=3,
            )
            label_x = mean - 0.34 if mean <= -0.3 else mean + 0.24
            ha = "right" if mean <= -0.3 else "left"
            ax.text(
                label_x,
                y,
                signed_one_decimal(mean),
                ha=ha,
                va="center",
                fontsize=7.25,
                color=TEXT,
            )

    ax.set_yticks(ytick_positions, [SSP_LABELS[ssp] for ssp in SSP_ORDER])
    ax.set_ylim(base_positions[-1] - 0.58, base_positions[0] + 0.58)
    ax.set_xlim(-10.4, 1.15)
    ax.set_xticks([-10, -8, -6, -4, -2, 0])
    ax.set_xlabel("End-century change (percentage points)", fontsize=8.8, color=TEXT, labelpad=5)
    ax.tick_params(axis="y", labelsize=8.2, colors=TEXT)
    panel_label(ax, "b")

    model_handles = [
        plt.Line2D(
            [0],
            [0],
            color="#4B565F",
            lw=0,
            marker=MODEL_STYLES[model]["marker"],
            markerfacecolor="#4B565F" if model == "HGB" else "white",
            markeredgecolor="#4B565F",
            markersize=5.2,
            label=model,
        )
        for model in MODEL_ORDER
    ]
    ax.legend(
        handles=model_handles,
        loc="upper left",
        bbox_to_anchor=(0.035, 0.98),
        frameon=True,
        framealpha=0.96,
        edgecolor=SPINE,
        facecolor="white",
        fontsize=7.4,
        borderpad=0.35,
        labelspacing=0.35,
        handletextpad=0.5,
    )


def make_figure(aggregates: dict[str, pd.DataFrame]) -> dict[str, str | list[int]]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )

    fig = plt.figure(figsize=(8.95, 4.45), constrained_layout=False)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.68, 1.0], wspace=0.30)
    ax_traj = fig.add_subplot(gs[0, 0])
    ax_change = fig.add_subplot(gs[0, 1])

    plot_trajectory(ax_traj, aggregates["scenario_period_model_aggregate"])
    plot_endpoint_forest(ax_change, aggregates["end_century_change_model_mean"])
    fig.subplots_adjust(left=0.074, right=0.988, top=0.965, bottom=0.145, wspace=0.32)

    png_path = FIG_DIR / f"{FIG_BASENAME}.png"
    svg_path = FIG_DIR / f"{FIG_BASENAME}.svg"
    pdf_path = FIG_DIR / f"{FIG_BASENAME}.pdf"
    preview_path = FIG_DIR / f"{FIG_BASENAME}_white_preview.png"

    fig.savefig(png_path, dpi=600, bbox_inches="tight", pad_inches=0.05, transparent=True)
    fig.savefig(svg_path, bbox_inches="tight", pad_inches=0.05, transparent=True)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.05, transparent=True)
    plt.close(fig)

    with Image.open(png_path) as img:
        if img.mode not in ("RGBA", "LA") and not (img.mode == "P" and "transparency" in img.info):
            raise RuntimeError(f"Transparent PNG lacks alpha channel: mode={img.mode}")
        rgba = img.convert("RGBA")
        white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        white.alpha_composite(rgba)
        white.convert("RGB").save(preview_path)
        png_size = list(rgba.size)

    return {
        "png": rel(png_path),
        "svg": rel(svg_path),
        "pdf": rel(pdf_path),
        "white_preview": rel(preview_path),
        "png_mode": "RGBA",
        "png_size": png_size,
    }


def write_readme(
    meta: dict[str, Any],
    figure_outputs: dict[str, Any],
    table_outputs: dict[str, str],
) -> None:
    lines = [
        "# Fig4 v06 candidate: future sample suitability trends",
        "",
        "- Status: success",
        "- Version: v06",
        "- Layout family: future-response-timeline-plus-endpoint-forest",
        "- Word status: not inserted into Word; user confirmation is required before manuscript insertion.",
        "- Data boundary: selected10 sample-level HGB/RF future predictions; full-grid and land-cover constrained area outputs are handled in separate area/constrained figures.",
        "- Visual design: non-2x2 two-column layout, with trajectory panel plus endpoint-change forest panel to preserve within-paper figure diversity.",
        "",
        "## Inputs",
        "",
    ]
    for key, path in INPUTS.items():
        lines.append(f"- {key}: `{rel(path)}`")
    lines.extend(["", "## Run status", ""])
    for key, status in meta["run_statuses"].items():
        lines.append(
            f"- {key}: status `{status.get('status')}`, scenarios {status.get('success_scenarios')}/{status.get('total_scenarios')}"
        )
    lines.extend(["", "## Outputs", ""])
    for key, path in figure_outputs.items():
        lines.append(f"- {key}: `{path}`")
    lines.extend(["", "## Plot-data tables", ""])
    for key, path in table_outputs.items():
        lines.append(f"- {key}: `{path}`")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Error bars and horizontal ranges summarize the three GCMs.",
            "- Values are sample-level suitable-rate percentages, not mapped area.",
            "- Complete caption/title should be placed below the figure in Word after final confirmation.",
        ]
    )
    README_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    setup_logging()
    write_status("running", message="stage39 fig4 v06 candidate generation started")
    try:
        df, statuses = load_data()
        aggregates = make_aggregates(df)
        table_outputs = export_tables(df, aggregates)
        figure_outputs = make_figure(aggregates)
        meta = {
            "run_statuses": statuses,
            "n_rows": int(len(df)),
            "n_samples": int(df["n"].max()),
            "n_models": int(df["model_short"].nunique()),
            "n_gcm": int(df["gcm"].nunique()),
            "n_ssp": int(df["ssp"].nunique()),
            "n_period": int(df["period"].nunique()),
        }
        write_readme(meta, figure_outputs, table_outputs)
        write_status(
            "success",
            message="stage39 fig4 v06 candidate generated",
            layout_family="future-response-timeline-plus-endpoint-forest",
            data_inputs={key: rel(path) for key, path in INPUTS.items()},
            data_boundary="selected10 sample-level HGB/RF future predictions; mapped area outputs are separate figures",
            meta=meta,
            figure_outputs=figure_outputs,
            table_outputs=table_outputs,
            readme=rel(README_MD),
            log=rel(LOG_PATH),
        )
        logging.info("Fig4 v06 candidate generated successfully.")
        return 0
    except Exception as exc:
        logging.exception("Fig4 v06 candidate generation failed.")
        write_status("failed", message=str(exc), log=rel(LOG_PATH))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
