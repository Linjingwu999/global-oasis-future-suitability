from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGE_DIR = PROJECT_ROOT / "outputs" / "stage42_manuscript_supplementary_figures"
FIG_DIR = STAGE_DIR / "figures"
TABLE_DIR = STAGE_DIR / "tables"
LOG_DIR = PROJECT_ROOT / "logs"

INPUT_DIR = PROJECT_ROOT / "outputs" / "stage32_independent_validation_extrapolation" / "tables"
LORO_METRICS_CSV = INPUT_DIR / "stage32_leave_one_region_out_metrics.csv"
SCENARIO_SUMMARY_CSV = INPUT_DIR / "stage32_future_extrapolation_scenario_summary.csv"
GROUP_SUMMARY_CSV = INPUT_DIR / "stage32_future_extrapolation_group_summary.csv"
STAGE32_STATUS_JSON = PROJECT_ROOT / "logs" / "stage32_independent_validation_extrapolation_state.json"

FIG_BASENAME = "fig_stage42_supp_independent_validation_extrapolation_v03"
STATUS_JSON = STAGE_DIR / f"{FIG_BASENAME}_status.json"
README_MD = STAGE_DIR / f"{FIG_BASENAME}_README.md"
LOG_PATH = LOG_DIR / "stage42_supp_independent_validation_extrapolation_v03.log"

TEXT = "#25313B"
MUTED = "#6C7783"
GRID = "#DCE5E4"
SPINE = "#D6DFDB"
BLUE = "#3B75AF"
TEAL = "#2C837E"
RUST = "#BF5B5B"
OCHRE = "#C27A28"
PURPLE = "#7761A8"
LIGHT_BLUE = "#A7BED8"
LIGHT_TEAL = "#A7CEC5"
LIGHT_RUST = "#E0B2AF"
LIGHT_OCHRE = "#E4C48C"

MODEL_LABELS = {
    "hist_gradient_boosting_balanced": "HGB",
    "random_forest_balanced": "RF",
    "glm_logistic_balanced": "GLM",
}
MODEL_ORDER = ["HGB", "RF", "GLM"]
MODEL_COLORS = {"HGB": BLUE, "RF": TEAL, "GLM": RUST}
SSP_ORDER = ["ssp126", "ssp245", "ssp370", "ssp585"]
SSP_LABELS = {
    "ssp126": "SSP1-2.6",
    "ssp245": "SSP2-4.5",
    "ssp370": "SSP3-7.0",
    "ssp585": "SSP5-8.5",
}
SSP_COLORS = {
    "ssp126": BLUE,
    "ssp245": TEAL,
    "ssp370": OCHRE,
    "ssp585": RUST,
}
PERIOD_ORDER = ["2021-2040", "2041-2060", "2061-2080", "2081-2100"]
PERIOD_POS = {period: idx for idx, period in enumerate(PERIOD_ORDER)}
GCM_MARKERS = {"ACCESS-CM2": "o", "MPI-ESM1-2-HR": "s", "MRI-ESM2-0": "^"}
REGION_LABELS = {
    "\u4e9a\u6d32\u4e1c\u90e8": "E Asia",
    "\u4e9a\u6d32\u4e2d\u90e8": "C Asia",
    "\u4e9a\u6d32\u897f\u5357\u90e8": "SW Asia",
    "\u5317\u7f8e\u6d32": "N America",
    "\u5357\u7f8e\u6d32": "S America",
    "\u963f\u62c9\u4f2f\u534a\u5c9b": "Arabia",
    "\u975e\u6d32\u5317\u90e8": "N Africa",
    "\u975e\u6d32\u5357\u90e8": "S Africa",
    "\u5927\u6d0b\u6d32": "Oceania",
}

REQUIRED_LORO_COLUMNS = {
    "model",
    "heldout_region",
    "pr_auc",
    "roc_auc",
    "tss",
    "test_rows",
    "test_presence",
    "test_background",
}
REQUIRED_SCENARIO_COLUMNS = {
    "gcm",
    "ssp",
    "period",
    "group_type",
    "n",
    "strict_extrapolation_rate",
    "central_5_95_outside_rate",
    "mean_central_outside_count",
}
REQUIRED_GROUP_COLUMNS = REQUIRED_SCENARIO_COLUMNS | {"Response", "ValidationRegion"}


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def setup_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=LOG_PATH,
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        encoding="utf-8",
    )


def write_status(status: str, **payload: object) -> None:
    STAGE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = STATUS_JSON.with_suffix(".json.tmp")
    data = {
        "status": status,
        "updated_at": now_iso(),
        "script": str(Path(__file__).resolve()),
        "figure_basename": FIG_BASENAME,
        **payload,
    }
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(STATUS_JSON)


def read_csv_checked(path: Path, required_columns: set[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required input: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"{path.name} is empty")
    missing = sorted(required_columns - set(df.columns))
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {missing}")
    return df


def ensure_stage32_complete() -> None:
    if not STAGE32_STATUS_JSON.exists():
        raise FileNotFoundError(f"Missing Stage32 status file: {STAGE32_STATUS_JSON}")
    status = json.loads(STAGE32_STATUS_JSON.read_text(encoding="utf-8"))
    if status.get("status") != "success":
        raise RuntimeError(f"Stage32 is not complete: {status.get('status')}")
    failed = [
        status.get("loro_failed_jobs", 0),
        status.get("extrap_failed_scenarios", 0),
    ]
    if any(int(x or 0) for x in failed):
        raise RuntimeError(f"Stage32 has failed jobs/scenarios: {failed}")


def prepare_data() -> dict[str, pd.DataFrame]:
    ensure_stage32_complete()
    loro = read_csv_checked(LORO_METRICS_CSV, REQUIRED_LORO_COLUMNS).copy()
    scenario = read_csv_checked(SCENARIO_SUMMARY_CSV, REQUIRED_SCENARIO_COLUMNS).copy()
    group = read_csv_checked(GROUP_SUMMARY_CSV, REQUIRED_GROUP_COLUMNS).copy()

    loro["model_label"] = loro["model"].map(MODEL_LABELS)
    loro = loro[loro["model_label"].notna()].copy()
    loro["region_label"] = loro["heldout_region"].map(REGION_LABELS).fillna(loro["heldout_region"])
    loro["pr_auc"] = pd.to_numeric(loro["pr_auc"], errors="coerce")
    loro = loro.dropna(subset=["pr_auc"]).copy()
    loro["region_mean_pr_auc"] = loro.groupby("region_label")["pr_auc"].transform("mean")
    loro["region_label"] = pd.Categorical(
        loro["region_label"],
        categories=loro.groupby("region_label")["pr_auc"].mean().sort_values().index.tolist(),
        ordered=True,
    )
    loro["model_label"] = pd.Categorical(loro["model_label"], categories=MODEL_ORDER, ordered=True)
    loro = loro.sort_values(["region_label", "model_label"]).reset_index(drop=True)

    scenario = scenario[scenario["group_type"].eq("overall")].copy()
    scenario["period"] = pd.Categorical(scenario["period"], categories=PERIOD_ORDER, ordered=True)
    scenario["ssp"] = pd.Categorical(scenario["ssp"], categories=SSP_ORDER, ordered=True)
    for col in [
        "strict_extrapolation_rate",
        "central_5_95_outside_rate",
        "mean_central_outside_count",
    ]:
        scenario[col] = pd.to_numeric(scenario[col], errors="coerce")
    scenario = scenario.dropna(subset=["strict_extrapolation_rate", "central_5_95_outside_rate"]).copy()

    scenario_period = (
        scenario.groupby(["ssp", "period"], observed=True)
        .agg(
            strict_mean=("strict_extrapolation_rate", "mean"),
            strict_min=("strict_extrapolation_rate", "min"),
            strict_max=("strict_extrapolation_rate", "max"),
            central_mean=("central_5_95_outside_rate", "mean"),
            central_min=("central_5_95_outside_rate", "min"),
            central_max=("central_5_95_outside_rate", "max"),
            central_count_mean=("mean_central_outside_count", "mean"),
            gcm_count=("gcm", "nunique"),
        )
        .reset_index()
    )

    end_century_gcm = scenario[scenario["period"].astype(str).eq("2081-2100")].copy()

    response = group[
        group["group_type"].eq("response") & group["period"].eq("2081-2100")
    ].copy()
    response["Response"] = pd.to_numeric(response["Response"], errors="coerce")
    response["response_label"] = response["Response"].map({0.0: "background", 1.0: "presence"})
    response["ssp"] = pd.Categorical(response["ssp"], categories=SSP_ORDER, ordered=True)
    response_summary = (
        response.dropna(subset=["response_label"])
        .groupby(["ssp", "response_label"], observed=True)
        .agg(
            strict_mean=("strict_extrapolation_rate", "mean"),
            strict_min=("strict_extrapolation_rate", "min"),
            strict_max=("strict_extrapolation_rate", "max"),
            central_mean=("central_5_95_outside_rate", "mean"),
            n_mean=("n", "mean"),
        )
        .reset_index()
    )

    region = group[
        group["group_type"].eq("validation_region")
        & group["period"].eq("2081-2100")
        & group["ssp"].eq("ssp585")
    ].copy()
    region["region_label"] = region["ValidationRegion"].map(REGION_LABELS).fillna(region["ValidationRegion"])
    region_summary = (
        region.groupby("region_label", observed=True)
        .agg(
            strict_mean=("strict_extrapolation_rate", "mean"),
            strict_min=("strict_extrapolation_rate", "min"),
            strict_max=("strict_extrapolation_rate", "max"),
            central_mean=("central_5_95_outside_rate", "mean"),
            central_min=("central_5_95_outside_rate", "min"),
            central_max=("central_5_95_outside_rate", "max"),
            central_count_mean=("mean_central_outside_count", "mean"),
            n_mean=("n", "mean"),
        )
        .reset_index()
        .sort_values("central_mean", ascending=True)
    )
    region_summary["region_label"] = pd.Categorical(
        region_summary["region_label"],
        categories=region_summary["region_label"].tolist(),
        ordered=True,
    )

    expected_scenarios = len(SSP_ORDER) * len(PERIOD_ORDER) * scenario["gcm"].nunique()
    if len(scenario) != expected_scenarios:
        raise ValueError(f"Expected {expected_scenarios} overall scenario rows, got {len(scenario)}")
    if response_summary.empty or region_summary.empty:
        raise ValueError("Missing response or region extrapolation summaries")

    return {
        "loro": loro,
        "scenario": scenario,
        "scenario_period": scenario_period,
        "end_century_gcm": end_century_gcm,
        "response_summary": response_summary,
        "region_summary": region_summary,
    }


def style_axis(ax: plt.Axes, *, y_grid: bool = True, x_grid: bool = True) -> None:
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_color(SPINE)
        spine.set_linewidth(0.9)
    ax.tick_params(axis="both", labelsize=7.8, colors=TEXT, length=3)
    ax.set_axisbelow(True)
    if x_grid:
        ax.grid(True, axis="x", color=GRID, linewidth=0.75, alpha=0.9)
    if y_grid:
        ax.grid(True, axis="y", color=GRID, linewidth=0.55, alpha=0.52)


def panel_label(ax: plt.Axes, label: str, color: str = TEAL) -> None:
    ax.text(
        -0.082,
        1.04,
        label,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=8.8,
        fontweight="bold",
        color="white",
        bbox=dict(boxstyle="round,pad=0.18,rounding_size=0.04", fc=color, ec="none"),
        clip_on=False,
    )


def plot_loro_panel(ax: plt.Axes, loro: pd.DataFrame) -> None:
    style_axis(ax)
    region_order = list(loro["region_label"].cat.categories)
    y_pos = np.arange(len(region_order))
    region_ranges = loro.groupby("region_label", observed=True)["pr_auc"].agg(["min", "max", "mean"])

    for idx, region in enumerate(region_order):
        row = region_ranges.loc[region]
        ax.hlines(idx, row["min"], row["max"], color="#B9C5C7", linewidth=1.9, zorder=1)

    offsets = {"HGB": -0.16, "RF": 0.0, "GLM": 0.16}
    for model in MODEL_ORDER:
        sub = loro[loro["model_label"].astype(str).eq(model)]
        xs = sub["pr_auc"].to_numpy()
        ys = np.array([region_order.index(str(region)) + offsets[model] for region in sub["region_label"]])
        ax.scatter(
            xs,
            ys,
            s=18,
            color=MODEL_COLORS[model],
            edgecolor="white",
            linewidth=0.45,
            zorder=3,
            label=model,
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(region_order)
    ax.set_xlim(0.04, 0.68)
    ax.set_xlabel("PR-AUC", fontsize=8.6, color=TEXT, labelpad=4)
    ax.set_title("Independent-region validation", fontsize=9.5, fontweight="bold", color=TEXT, pad=9)
    ax.legend(
        loc="lower right",
        ncol=3,
        frameon=False,
        fontsize=7.4,
        handletextpad=0.35,
        columnspacing=0.75,
        borderaxespad=0.2,
    )
    panel_label(ax, "a", BLUE)


def plot_trajectory_panel(ax: plt.Axes, scenario_period: pd.DataFrame) -> None:
    style_axis(ax, y_grid=True, x_grid=False)
    ax.grid(True, axis="y", color=GRID, linewidth=0.75, alpha=0.9)
    x = np.arange(len(PERIOD_ORDER))
    for ssp in SSP_ORDER:
        sub = scenario_period[scenario_period["ssp"].astype(str).eq(ssp)].sort_values("period")
        y = sub["strict_mean"].to_numpy() * 100
        ymin = sub["strict_min"].to_numpy() * 100
        ymax = sub["strict_max"].to_numpy() * 100
        color = SSP_COLORS[ssp]
        ax.fill_between(x, ymin, ymax, color=color, alpha=0.12, linewidth=0)
        ax.plot(x, y, color=color, linewidth=1.8, marker="o", markersize=3.5, label=SSP_LABELS[ssp])
        ax.text(
            x[-1] + 0.05,
            y[-1],
            SSP_LABELS[ssp].replace("SSP", ""),
            ha="left",
            va="center",
            fontsize=7.2,
            color=color,
            clip_on=False,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(["2021\n2040", "2041\n2060", "2061\n2080", "2081\n2100"])
    ax.set_xlim(-0.15, len(PERIOD_ORDER) - 0.32)
    ax.set_ylim(0, 25)
    ax.set_ylabel("Strict outside rate (%)", fontsize=8.6, color=TEXT, labelpad=4)
    ax.set_title("Future strict extrapolation", fontsize=9.5, fontweight="bold", color=TEXT, pad=9)
    panel_label(ax, "b", TEAL)


def plot_response_panel(ax: plt.Axes, response_summary: pd.DataFrame) -> None:
    style_axis(ax, y_grid=True, x_grid=False)
    x = np.arange(len(SSP_ORDER))
    offsets = {"background": -0.12, "presence": 0.12}
    colors = {"background": MUTED, "presence": RUST}
    labels = {"background": "Background", "presence": "Presence"}
    for ssp_idx, ssp in enumerate(SSP_ORDER):
        sub = response_summary[response_summary["ssp"].astype(str).eq(ssp)]
        vals = {
            row["response_label"]: row["strict_mean"] * 100
            for _, row in sub.iterrows()
        }
        if {"background", "presence"}.issubset(vals):
            ax.plot(
                [ssp_idx + offsets["background"], ssp_idx + offsets["presence"]],
                [vals["background"], vals["presence"]],
                color="#C8D1D2",
                linewidth=1.5,
                zorder=1,
            )
            gap = vals["presence"] - vals["background"]
            ax.text(
                ssp_idx,
                max(vals.values()) + 1.0,
                f"+{gap:.1f}",
                ha="center",
                va="bottom",
                fontsize=7.0,
                color=RUST,
            )
    for response_label in ["background", "presence"]:
        sub = response_summary[response_summary["response_label"].eq(response_label)].copy()
        xs = np.array([SSP_ORDER.index(str(ssp)) + offsets[response_label] for ssp in sub["ssp"]])
        y = sub["strict_mean"].to_numpy() * 100
        yerr = np.vstack(
            [
                y - sub["strict_min"].to_numpy() * 100,
                sub["strict_max"].to_numpy() * 100 - y,
            ]
        )
        ax.errorbar(
            xs,
            y,
            yerr=yerr,
            fmt="o",
            color=colors[response_label],
            ecolor=colors[response_label],
            elinewidth=1.0,
            capsize=2.2,
            markersize=4.0,
            label=labels[response_label],
            zorder=3,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([SSP_LABELS[ssp].replace("SSP", "") for ssp in SSP_ORDER])
    ax.set_ylim(0, 32)
    ax.set_ylabel("Strict outside rate (%)", fontsize=8.6, color=TEXT, labelpad=4)
    ax.set_title("End-century sample-type contrast", fontsize=9.5, fontweight="bold", color=TEXT, pad=9)
    ax.legend(
        loc="upper left",
        frameon=False,
        fontsize=7.4,
        handletextpad=0.35,
        borderaxespad=0.15,
    )
    panel_label(ax, "c", OCHRE)


def plot_region_panel(ax: plt.Axes, region_summary: pd.DataFrame) -> None:
    style_axis(ax)
    region_order = list(region_summary["region_label"].astype(str))
    y = np.arange(len(region_order))
    strict = region_summary["strict_mean"].to_numpy() * 100
    central = region_summary["central_mean"].to_numpy() * 100
    for idx, (s_val, c_val) in enumerate(zip(strict, central)):
        ax.hlines(idx, s_val, c_val, color="#C8D1D2", linewidth=1.8, zorder=1)
    ax.scatter(strict, y, s=23, color=RUST, edgecolor="white", linewidth=0.45, zorder=3, clip_on=False, label="Strict")
    ax.scatter(
        central,
        y,
        s=23,
        color=PURPLE,
        edgecolor="white",
        linewidth=0.45,
        zorder=3,
        clip_on=False,
        label="Central 5-95",
    )
    for idx, c_val in enumerate(central):
        if c_val >= 84:
            ax.text(c_val - 1.2, idx, f"{c_val:.0f}", ha="right", va="center", fontsize=6.7, color=PURPLE)
        else:
            ax.text(c_val + 1.2, idx, f"{c_val:.0f}", ha="left", va="center", fontsize=6.7, color=PURPLE)
    ax.set_yticks(y)
    ax.set_yticklabels(region_order)
    ax.set_xlim(-3, 93)
    ax.set_xlabel("Outside training envelope (%)", fontsize=8.6, color=TEXT, labelpad=4)
    ax.set_title("SSP5-8.5 regional profile", fontsize=9.5, fontweight="bold", color=TEXT, pad=9)
    ax.text(
        0.33,
        0.965,
        "Strict",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.3,
        color=RUST,
        fontweight="bold",
    )
    ax.text(
        0.73,
        0.965,
        "Central 5-95",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.3,
        color=PURPLE,
        fontweight="bold",
    )
    panel_label(ax, "d", PURPLE)


def save_plot_tables(data: dict[str, pd.DataFrame]) -> dict[str, str]:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {
        "loro_region_model_pr_auc": TABLE_DIR / "stage42_v03_loro_region_model_pr_auc.csv",
        "future_extrapolation_scenario_period": TABLE_DIR / "stage42_v03_future_extrapolation_scenario_period.csv",
        "end_century_gcm_strict_extrapolation": TABLE_DIR / "stage42_v03_end_century_gcm_strict_extrapolation.csv",
        "end_century_response_risk": TABLE_DIR / "stage42_v03_end_century_response_risk.csv",
        "ssp585_region_extrapolation_profile": TABLE_DIR / "stage42_v03_ssp585_region_extrapolation_profile.csv",
    }
    data["loro"].to_csv(outputs["loro_region_model_pr_auc"], index=False, encoding="utf-8-sig")
    data["scenario_period"].to_csv(
        outputs["future_extrapolation_scenario_period"], index=False, encoding="utf-8-sig"
    )
    data["end_century_gcm"].to_csv(
        outputs["end_century_gcm_strict_extrapolation"], index=False, encoding="utf-8-sig"
    )
    data["response_summary"].to_csv(outputs["end_century_response_risk"], index=False, encoding="utf-8-sig")
    data["region_summary"].to_csv(
        outputs["ssp585_region_extrapolation_profile"], index=False, encoding="utf-8-sig"
    )
    return {key: str(path) for key, path in outputs.items()}


def verify_alpha(path: Path) -> None:
    img = Image.open(path)
    if not (img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info)):
        raise ValueError(f"PNG does not contain alpha transparency: {path}")


def make_white_preview(png_path: Path, preview_path: Path) -> None:
    img = Image.open(png_path).convert("RGBA")
    white = Image.new("RGBA", img.size, (255, 255, 255, 255))
    white.alpha_composite(img)
    white.convert("RGB").save(preview_path, dpi=(300, 300))


def build_figure(data: dict[str, pd.DataFrame]) -> dict[str, str]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.unicode_minus": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig = plt.figure(figsize=(11.2, 7.0), dpi=160)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.0], height_ratios=[1.0, 1.0])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    plot_loro_panel(ax_a, data["loro"])
    plot_trajectory_panel(ax_b, data["scenario_period"])
    plot_response_panel(ax_c, data["response_summary"])
    plot_region_panel(ax_d, data["region_summary"])

    fig.text(
        0.01,
        0.012,
        "Data boundary: Stage32 selected10 sample diagnostics only; full-grid area and land-cover-constrained suitability are not used.",
        ha="left",
        va="bottom",
        fontsize=7.1,
        color=MUTED,
    )
    fig.subplots_adjust(left=0.07, right=0.985, top=0.945, bottom=0.085, wspace=0.27, hspace=0.36)

    png = FIG_DIR / f"{FIG_BASENAME}.png"
    svg = FIG_DIR / f"{FIG_BASENAME}.svg"
    pdf = FIG_DIR / f"{FIG_BASENAME}.pdf"
    preview = FIG_DIR / f"{FIG_BASENAME}_white_preview.png"
    fig.savefig(png, dpi=600, bbox_inches="tight", pad_inches=0.08, transparent=True)
    fig.savefig(svg, bbox_inches="tight", pad_inches=0.08, transparent=True)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.08, transparent=True)
    plt.close(fig)
    verify_alpha(png)
    make_white_preview(png, preview)
    return {
        "png": str(png),
        "svg": str(svg),
        "pdf": str(pdf),
        "white_preview": str(preview),
    }


def write_readme(outputs: dict[str, str], table_outputs: dict[str, str]) -> None:
    lines = [
        f"# {FIG_BASENAME}",
        "",
        "Candidate supplementary figure for independent-region validation and future predictor extrapolation risk.",
        "",
        "## Inputs",
        f"- `{LORO_METRICS_CSV}`",
        f"- `{SCENARIO_SUMMARY_CSV}`",
        f"- `{GROUP_SUMMARY_CSV}`",
        f"- `{STAGE32_STATUS_JSON}`",
        "",
        "## Data boundary",
        "- Uses completed Stage32 selected10 sample diagnostics.",
        "- Does not use selected10 full-grid suitability, constrained area, or land-cover overlays.",
        "",
        "## Figure outputs",
        *[f"- `{path}`" for path in outputs.values()],
        "",
        "## Plot-data tables",
        *[f"- `{path}`" for path in table_outputs.values()],
    ]
    README_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    setup_logging()
    try:
        write_status("running")
        logging.info("Loading Stage32 inputs")
        data = prepare_data()
        table_outputs = save_plot_tables(data)
        outputs = build_figure(data)
        write_readme(outputs, table_outputs)
        write_status(
            "success",
            outputs=outputs,
            plot_tables=table_outputs,
            source_inputs={
                "loro_metrics_csv": str(LORO_METRICS_CSV),
                "scenario_summary_csv": str(SCENARIO_SUMMARY_CSV),
                "group_summary_csv": str(GROUP_SUMMARY_CSV),
                "stage32_status_json": str(STAGE32_STATUS_JSON),
            },
            data_boundary="Stage32 selected10 sample diagnostics only; no full-grid area or land-cover constraints.",
        )
        logging.info("Figure completed: %s", outputs)
        print(json.dumps({"status": "success", "outputs": outputs, "tables": table_outputs}, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        logging.exception("Figure generation failed")
        write_status("failed", error=repr(exc))
        print(json.dumps({"status": "failed", "error": repr(exc)}, ensure_ascii=False, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
