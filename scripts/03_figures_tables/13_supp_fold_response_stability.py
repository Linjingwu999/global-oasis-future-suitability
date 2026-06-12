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
STAGE_DIR = PROJECT_ROOT / "outputs" / "stage45_manuscript_supplementary_figures"
FIG_DIR = STAGE_DIR / "figures"
TABLE_DIR = STAGE_DIR / "tables"
LOG_DIR = PROJECT_ROOT / "logs"

INPUT_DIR = PROJECT_ROOT / "outputs" / "stage34_selected10_future_worldclim_sample_predictions"
HGB_SCENARIO_SUMMARY_CSV = INPUT_DIR / "future_worldclim_sample_prediction_scenario_summary_selected10_hgb.csv"
RF_SCENARIO_SUMMARY_CSV = INPUT_DIR / "future_worldclim_sample_prediction_scenario_summary_selected10_rf.csv"
HGB_STATE_JSON = LOG_DIR / "stage07_future_worldclim_sample_predictions_selected10_hgb_state.json"
RF_STATE_JSON = LOG_DIR / "stage07_future_worldclim_sample_predictions_selected10_rf_state.json"

FIG_BASENAME = "fig_stage45_supp_future_fold_response_stability_v01"
STATUS_JSON = STAGE_DIR / f"{FIG_BASENAME}_status.json"
README_MD = STAGE_DIR / f"{FIG_BASENAME}_README.md"
LOG_PATH = LOG_DIR / "stage45_supp_future_fold_response_stability_v01.log"

REQUIRED_COLUMNS = {
    "gcm",
    "ssp",
    "period",
    "model_group",
    "group_type",
    "n",
    "mean_probability",
    "median_probability",
    "p10_probability",
    "p90_probability",
    "suitable_rate",
    "Response",
    "SpatialCVFold",
}

MODEL_LABELS = {
    "hist_gradient_boosting_balanced": "HGB",
    "random_forest_balanced": "RF",
}
MODEL_ORDER = ["HGB", "RF"]
SSP_ORDER = ["ssp126", "ssp245", "ssp370", "ssp585"]
PERIOD_ORDER = ["2021-2040", "2041-2060", "2061-2080", "2081-2100"]
GCM_ORDER = ["ACCESS-CM2", "MPI-ESM1-2-HR", "MRI-ESM2-0"]
FOLD_ORDER = [1, 2, 3, 4, 5]
SSP_LABELS = {
    "ssp126": "SSP1-2.6",
    "ssp245": "SSP2-4.5",
    "ssp370": "SSP3-7.0",
    "ssp585": "SSP5-8.5",
}
PERIOD_LABELS = {
    "2021-2040": "2030s",
    "2041-2060": "2050s",
    "2061-2080": "2070s",
    "2081-2100": "2090s",
}
RESPONSE_LABELS = {0.0: "background", 1.0: "presence"}

TEXT = "#26323A"
MUTED = "#687785"
GRID = "#DDE7E5"
SPINE = "#D7E0DD"
BLUE = "#3D72A8"
TEAL = "#2B837D"
RUST = "#BF5E62"
OCHRE = "#C57A27"
GOLD = "#DDBA52"
MODEL_COLORS = {"HGB": BLUE, "RF": TEAL}
MODEL_LIGHT = {"HGB": "#BFD0E4", "RF": "#BFDCD5"}
SSP_COLORS = {
    "ssp126": BLUE,
    "ssp245": TEAL,
    "ssp370": OCHRE,
    "ssp585": RUST,
}


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path)


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
    tmp_path = STATUS_JSON.with_suffix(".json.tmp")
    data = {
        "status": status,
        "updated_at": now_iso(),
        "script": str(Path(__file__).resolve()),
        "figure_basename": FIG_BASENAME,
        **payload,
    }
    tmp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(STATUS_JSON)


def atomic_to_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp_path, index=False, encoding="utf-8-sig")
    tmp_path.replace(path)


def quantile_25(values: pd.Series) -> float:
    return float(values.quantile(0.25))


def quantile_75(values: pd.Series) -> float:
    return float(values.quantile(0.75))


def check_state_file(path: Path, expected_group: str) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Missing state file: {path}")
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("status") != "success":
        raise RuntimeError(f"{path.name} is not success: {state.get('status')}")
    if state.get("model_group") != expected_group:
        raise RuntimeError(
            f"{path.name} has unexpected model_group {state.get('model_group')!r}"
        )
    if int(state.get("success_scenarios", 0) or 0) != 48:
        raise RuntimeError(f"{path.name} does not record all 48 successful scenarios")
    if int(state.get("failed_scenarios", 0) or 0) != 0:
        raise RuntimeError(f"{path.name} records failed scenarios")
    return state


def read_csv_checked(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input CSV: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"{path.name} is empty")
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {missing}")
    return df


def load_inputs() -> tuple[pd.DataFrame, dict[str, object]]:
    hgb_state = check_state_file(HGB_STATE_JSON, "hist_gradient_boosting_balanced")
    rf_state = check_state_file(RF_STATE_JSON, "random_forest_balanced")

    frames = []
    for path in [HGB_SCENARIO_SUMMARY_CSV, RF_SCENARIO_SUMMARY_CSV]:
        df = read_csv_checked(path)
        frames.append(df)
        logging.info("Loaded %s rows from %s", len(df), path)

    raw = pd.concat(frames, ignore_index=True)
    raw["model_label"] = raw["model_group"].map(MODEL_LABELS)
    raw["ssp"] = pd.Categorical(raw["ssp"], categories=SSP_ORDER, ordered=True)
    raw["period"] = pd.Categorical(raw["period"], categories=PERIOD_ORDER, ordered=True)
    raw["gcm"] = pd.Categorical(raw["gcm"], categories=GCM_ORDER, ordered=True)
    raw["Response"] = pd.to_numeric(raw["Response"], errors="coerce")
    raw["SpatialCVFold"] = pd.to_numeric(raw["SpatialCVFold"], errors="coerce").astype("Int64")

    numeric_cols = [
        "n",
        "mean_probability",
        "median_probability",
        "p10_probability",
        "p90_probability",
        "suitable_rate",
    ]
    for col in numeric_cols:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    raw["suitable_rate_pct"] = raw["suitable_rate"] * 100.0

    fold = raw[raw["group_type"].eq("fold_response")].copy()
    fold["response_label"] = fold["Response"].map(RESPONSE_LABELS)
    required = [
        "model_label",
        "gcm",
        "ssp",
        "period",
        "SpatialCVFold",
        "response_label",
        "suitable_rate_pct",
    ]
    bad = fold[fold[required].isna().any(axis=1)]
    if not bad.empty:
        raise ValueError(f"Unexpected fold rows: {bad.head(10).to_dict('records')}")

    expected_rows = (
        len(MODEL_ORDER)
        * len(GCM_ORDER)
        * len(SSP_ORDER)
        * len(PERIOD_ORDER)
        * len(FOLD_ORDER)
        * 2
    )
    if len(fold) != expected_rows:
        raise ValueError(f"Expected {expected_rows} fold_response rows, found {len(fold)}")
    if sorted(int(x) for x in fold["SpatialCVFold"].dropna().unique()) != FOLD_ORDER:
        raise ValueError("SpatialCVFold does not contain the expected five folds")

    meta = {
        "hgb_state": hgb_state,
        "rf_state": rf_state,
        "scenario_summary_rows": int(len(raw)),
        "fold_response_rows": int(len(fold)),
        "folds": FOLD_ORDER,
        "gcm_count": len(GCM_ORDER),
        "replicates_per_model_ssp_period_response": len(FOLD_ORDER) * len(GCM_ORDER),
        "data_scope": (
            "sample-level fold_response summaries from selected10 HGB/RF future "
            "WorldClim predictions only; full-grid areas and land-cover constraints "
            "are not used"
        ),
    }
    return fold, meta


def prepare_tables(fold: pd.DataFrame) -> dict[str, pd.DataFrame]:
    keep_cols = [
        "model_label",
        "gcm",
        "ssp",
        "period",
        "SpatialCVFold",
        "response_label",
        "n",
        "mean_probability",
        "suitable_rate_pct",
    ]
    detail = fold[keep_cols].copy()
    detail["ssp_label"] = detail["ssp"].astype(str).map(SSP_LABELS)
    detail["period_label"] = detail["period"].astype(str).map(PERIOD_LABELS)

    pivot = detail.pivot_table(
        index=["model_label", "gcm", "ssp", "period", "SpatialCVFold"],
        columns="response_label",
        values=["suitable_rate_pct", "mean_probability", "n"],
        observed=True,
    )
    pivot.columns = [f"{metric}_{label}" for metric, label in pivot.columns]
    separation = pivot.reset_index()
    separation["separation_pp"] = (
        separation["suitable_rate_pct_presence"]
        - separation["suitable_rate_pct_background"]
    )
    separation["mean_probability_gap"] = (
        separation["mean_probability_presence"]
        - separation["mean_probability_background"]
    )
    separation["ssp_label"] = separation["ssp"].astype(str).map(SSP_LABELS)
    separation["period_label"] = separation["period"].astype(str).map(PERIOD_LABELS)

    stability = (
        separation.groupby(["model_label", "ssp", "period"], observed=True)
        .agg(
            separation_mean_pp=("separation_pp", "mean"),
            separation_sd_pp=("separation_pp", "std"),
            separation_min_pp=("separation_pp", "min"),
            separation_q25_pp=("separation_pp", quantile_25),
            separation_q75_pp=("separation_pp", quantile_75),
            separation_max_pp=("separation_pp", "max"),
            background_mean_pct=("suitable_rate_pct_background", "mean"),
            background_sd_pct=("suitable_rate_pct_background", "std"),
            presence_mean_pct=("suitable_rate_pct_presence", "mean"),
            presence_sd_pct=("suitable_rate_pct_presence", "std"),
            replicate_count=("separation_pp", "count"),
        )
        .reset_index()
    )
    stability["separation_iqr_pp"] = (
        stability["separation_q75_pp"] - stability["separation_q25_pp"]
    )
    stability["ssp_label"] = stability["ssp"].astype(str).map(SSP_LABELS)
    stability["period_label"] = stability["period"].astype(str).map(PERIOD_LABELS)

    late_distribution = separation[separation["period"].astype(str).eq("2081-2100")].copy()

    spread = (
        stability[stability["period"].astype(str).isin(["2021-2040", "2081-2100"])]
        .pivot_table(
            index=["model_label", "ssp"],
            columns="period",
            values="separation_iqr_pp",
            observed=True,
        )
        .reset_index()
    )
    spread.columns = [str(c) for c in spread.columns]
    spread["iqr_change_2090s_minus_2030s_pp"] = (
        spread["2081-2100"] - spread["2021-2040"]
    )
    spread["ssp_label"] = spread["ssp"].astype(str).map(SSP_LABELS)

    return {
        "fold_response_detail": detail,
        "fold_separation": separation,
        "fold_stability_summary": stability,
        "late_century_fold_distribution": late_distribution,
        "spread_change": spread,
    }


def apply_axis_style(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.set_axisbelow(True)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_color(SPINE)
        spine.set_linewidth(1.0)
    ax.tick_params(colors=TEXT, length=0)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)


def add_panel_label(ax: plt.Axes, label: str, color: str, x: float = -0.075, y: float = 1.06) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold",
        color="white",
        bbox=dict(boxstyle="round,pad=0.22,rounding_size=0.04", facecolor=color, edgecolor="none"),
        clip_on=False,
    )


def plot_model_envelope(ax: plt.Axes, stability: pd.DataFrame, model: str) -> None:
    model_df = stability[stability["model_label"].eq(model)].copy()
    x = np.arange(len(PERIOD_ORDER))
    for ssp in SSP_ORDER:
        ssp_df = (
            model_df[model_df["ssp"].astype(str).eq(ssp)]
            .set_index("period")
            .reindex(PERIOD_ORDER)
            .reset_index()
        )
        mean = ssp_df["separation_mean_pp"].to_numpy(dtype=float)
        q25 = ssp_df["separation_q25_pp"].to_numpy(dtype=float)
        q75 = ssp_df["separation_q75_pp"].to_numpy(dtype=float)
        color = SSP_COLORS[ssp]
        ax.fill_between(x, q25, q75, color=color, alpha=0.13, linewidth=0)
        ax.plot(
            x,
            mean,
            color=color,
            linewidth=1.9,
            marker="o",
            markersize=4.2,
            markeredgecolor="white",
            markeredgewidth=0.7,
            label=SSP_LABELS[ssp],
        )
    ax.set_xticks(x)
    ax.set_xticklabels([PERIOD_LABELS[p] for p in PERIOD_ORDER])
    ax.set_ylim(48, 82)
    ax.set_ylabel("Presence-background separation (pp)")
    ax.set_title(f"{model} fold envelope", fontsize=11, fontweight="bold", pad=8)
    apply_axis_style(ax, grid_axis="y")
    if model == "HGB":
        ax.legend(
            loc="lower left",
            frameon=False,
            fontsize=7.6,
            ncol=2,
            handlelength=1.6,
            handletextpad=0.45,
            columnspacing=0.8,
            borderpad=0.2,
        )


def row_order() -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for ssp in SSP_ORDER:
        for model in MODEL_ORDER:
            rows.append((ssp, model))
    return rows


def plot_late_violin(ax: plt.Axes, late: pd.DataFrame) -> None:
    rows = row_order()
    positions = np.arange(len(rows))[::-1]
    distributions = []
    labels = []
    row_colors = []
    for ssp, model in rows:
        values = (
            late[
                late["ssp"].astype(str).eq(ssp)
                & late["model_label"].eq(model)
            ]["separation_pp"]
            .dropna()
            .to_numpy(dtype=float)
        )
        distributions.append(values)
        labels.append(f"{SSP_LABELS[ssp]}  {model}")
        row_colors.append(MODEL_COLORS[model])

    parts = ax.violinplot(
        distributions,
        positions=positions,
        vert=False,
        widths=0.72,
        showmeans=False,
        showextrema=False,
        showmedians=False,
    )
    for body, color in zip(parts["bodies"], row_colors):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.18)
        body.set_linewidth(0.8)

    for pos, values, color in zip(positions, distributions, row_colors):
        values_sorted = np.sort(values)
        if len(values_sorted) == 0:
            continue
        jitter = np.linspace(-0.18, 0.18, len(values_sorted))
        ax.scatter(
            values_sorted,
            pos + jitter,
            s=14,
            color=color,
            alpha=0.62,
            edgecolor="white",
            linewidth=0.35,
            zorder=3,
        )
        q25, median, q75 = np.percentile(values_sorted, [25, 50, 75])
        ax.hlines(pos, q25, q75, color=color, linewidth=3.2, zorder=4)
        ax.scatter(
            median,
            pos,
            s=34,
            marker="D",
            color=color,
            edgecolor="white",
            linewidth=0.7,
            zorder=5,
        )
    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlim(45, 84)
    ax.set_xlabel("2090s separation across folds and GCMs (pp)")
    ax.set_title("Late-century fold distribution", fontsize=11, fontweight="bold", pad=8)
    apply_axis_style(ax, grid_axis="x")


def plot_spread_change(ax: plt.Axes, spread: pd.DataFrame) -> None:
    rows = row_order()
    positions = np.arange(len(rows))[::-1]
    labels = []
    max_value = 0.0
    for pos, (ssp, model) in zip(positions, rows):
        row = spread[
            spread["ssp"].astype(str).eq(ssp)
            & spread["model_label"].eq(model)
        ]
        if row.empty:
            continue
        start = float(row["2021-2040"].iloc[0])
        end = float(row["2081-2100"].iloc[0])
        max_value = max(max_value, start, end)
        color = MODEL_COLORS[model]
        ax.hlines(pos, start, end, color="#C6D1D4", linewidth=2.2, zorder=1)
        ax.scatter(
            start,
            pos,
            s=34,
            color="white",
            edgecolor=color,
            linewidth=1.4,
            label="2030s" if pos == positions[0] else None,
            zorder=3,
        )
        ax.scatter(
            end,
            pos,
            s=42,
            color=color,
            edgecolor="white",
            linewidth=0.7,
            label="2090s" if pos == positions[0] else None,
            zorder=4,
        )
        labels.append(f"{SSP_LABELS[ssp]}  {model}")
    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlim(0, max(10.5, max_value + 1.2))
    ax.set_xlabel("Interquartile spread of separation (pp)")
    ax.set_title("Fold spread change", fontsize=11, fontweight="bold", pad=8)
    ax.legend(
        loc="lower right",
        frameon=False,
        fontsize=8,
        handletextpad=0.45,
        borderpad=0.2,
    )
    apply_axis_style(ax, grid_axis="x")


def make_figure(tables: dict[str, pd.DataFrame], meta: dict[str, object]) -> dict[str, str]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": ["Arial", "DejaVu Sans"],
            "axes.labelsize": 9,
            "axes.titlesize": 11,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "savefig.dpi": 600,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )

    stability = tables["fold_stability_summary"]
    late = tables["late_century_fold_distribution"]
    spread = tables["spread_change"]

    fig = plt.figure(figsize=(11.3, 7.0), facecolor="none")
    gs = fig.add_gridspec(
        2,
        2,
        left=0.075,
        right=0.985,
        top=0.91,
        bottom=0.13,
        wspace=0.30,
        hspace=0.46,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    plot_model_envelope(ax_a, stability, "HGB")
    plot_model_envelope(ax_b, stability, "RF")
    plot_late_violin(ax_c, late)
    plot_spread_change(ax_d, spread)

    add_panel_label(ax_a, "a", BLUE)
    add_panel_label(ax_b, "b", TEAL)
    add_panel_label(ax_c, "c", RUST)
    add_panel_label(ax_d, "d", OCHRE)

    footnote = (
        "Sample-level fold_response summaries only: each model-SSP-period cell uses "
        f"{meta['replicates_per_model_ssp_period_response']} fold x GCM values; "
        "full-grid area and land-cover constraints are not used."
    )
    fig.text(0.075, 0.052, footnote, ha="left", va="center", fontsize=8, color=MUTED)

    png_path = FIG_DIR / f"{FIG_BASENAME}.png"
    svg_path = FIG_DIR / f"{FIG_BASENAME}.svg"
    pdf_path = FIG_DIR / f"{FIG_BASENAME}.pdf"
    preview_path = FIG_DIR / f"{FIG_BASENAME}_white_preview.png"

    fig.savefig(png_path, dpi=600, bbox_inches="tight", pad_inches=0.03, transparent=True)
    fig.savefig(svg_path, bbox_inches="tight", pad_inches=0.03, transparent=True)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.03, transparent=True)
    fig.savefig(preview_path, dpi=300, bbox_inches="tight", pad_inches=0.03, facecolor="white")
    plt.close(fig)

    image = Image.open(png_path)
    if image.mode not in ("RGBA", "LA") and not (
        image.mode == "P" and "transparency" in image.info
    ):
        raise RuntimeError(f"PNG does not contain transparency: {png_path}")

    return {
        "png": str(png_path),
        "svg": str(svg_path),
        "pdf": str(pdf_path),
        "white_preview": str(preview_path),
    }


def write_readme(outputs: dict[str, str], tables: dict[str, pd.DataFrame], meta: dict[str, object]) -> None:
    lines = [
        "# Stage45 supplementary figure candidate: future fold-response stability",
        "",
        f"- Figure basename: `{FIG_BASENAME}`",
        f"- Generated at: {now_iso()}",
        f"- Script: `{rel(Path(__file__).resolve())}`",
        "- Status: candidate; not inserted into Word until user confirms.",
        f"- Data scope: {meta['data_scope']}.",
        "- Completed input status: HGB and RF sample-level selected10 future WorldClim predictions both record `success` with 48/48 scenarios and 0 failures.",
        "",
        "## Scientific content",
        "",
        "- Panels a-b show fold and GCM envelopes of presence-background suitability separation through time for HGB and RF.",
        "- Panel c shows the 2090s distribution of separation across folds and GCMs.",
        "- Panel d shows how the interquartile spread of fold/GCM separation changes from the 2030s to the 2090s.",
        "- This is not a full-grid suitable-area figure and does not use land-cover constraints.",
        "",
        "## Outputs",
        "",
    ]
    for key, path in outputs.items():
        lines.append(f"- {key}: `{rel(Path(path))}`")
    lines.extend(["", "## Tables", ""])
    for name in tables:
        lines.append(f"- {name}: `{rel(TABLE_DIR / f'stage45_v01_{name}.csv')}`")
    README_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    setup_logging()
    write_status("pending")
    try:
        write_status("running", started_at=now_iso())
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        TABLE_DIR.mkdir(parents=True, exist_ok=True)
        fold, meta = load_inputs()
        tables = prepare_tables(fold)
        for name, table in tables.items():
            atomic_to_csv(table, TABLE_DIR / f"stage45_v01_{name}.csv")
        outputs = make_figure(tables, meta)
        write_readme(outputs, tables, meta)
        write_status(
            "success",
            finished_at=now_iso(),
            inputs={
                "hgb_scenario_summary_csv": str(HGB_SCENARIO_SUMMARY_CSV),
                "rf_scenario_summary_csv": str(RF_SCENARIO_SUMMARY_CSV),
                "hgb_state_json": str(HGB_STATE_JSON),
                "rf_state_json": str(RF_STATE_JSON),
            },
            outputs=outputs,
            tables={name: str(TABLE_DIR / f"stage45_v01_{name}.csv") for name in tables},
            readme=str(README_MD),
            log=str(LOG_PATH),
            checks=meta,
        )
        logging.info("Completed %s", FIG_BASENAME)
    except Exception as exc:
        logging.exception("Failed %s", FIG_BASENAME)
        write_status("failed", failed_at=now_iso(), error=repr(exc), log=str(LOG_PATH))
        raise


if __name__ == "__main__":
    main()
