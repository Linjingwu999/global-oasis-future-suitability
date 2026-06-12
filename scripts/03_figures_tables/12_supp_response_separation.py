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
from matplotlib.colors import LinearSegmentedColormap, Normalize
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGE_DIR = PROJECT_ROOT / "outputs" / "stage44_manuscript_supplementary_figures"
FIG_DIR = STAGE_DIR / "figures"
TABLE_DIR = STAGE_DIR / "tables"
LOG_DIR = PROJECT_ROOT / "logs"

INPUT_DIR = PROJECT_ROOT / "outputs" / "stage34_selected10_future_worldclim_sample_predictions"
HGB_SCENARIO_SUMMARY_CSV = INPUT_DIR / "future_worldclim_sample_prediction_scenario_summary_selected10_hgb.csv"
RF_SCENARIO_SUMMARY_CSV = INPUT_DIR / "future_worldclim_sample_prediction_scenario_summary_selected10_rf.csv"
HGB_STATE_JSON = LOG_DIR / "stage07_future_worldclim_sample_predictions_selected10_hgb_state.json"
RF_STATE_JSON = LOG_DIR / "stage07_future_worldclim_sample_predictions_selected10_rf_state.json"

FIG_BASENAME = "fig_stage44_supp_future_sample_response_separation_v04"
STATUS_JSON = STAGE_DIR / f"{FIG_BASENAME}_status.json"
README_MD = STAGE_DIR / f"{FIG_BASENAME}_README.md"
LOG_PATH = LOG_DIR / "stage44_supp_future_sample_response_separation_v04.log"

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
    "SampleType",
}

MODEL_LABELS = {
    "hist_gradient_boosting_balanced": "HGB",
    "random_forest_balanced": "RF",
}
MODEL_ORDER = ["HGB", "RF"]
SSP_ORDER = ["ssp126", "ssp245", "ssp370", "ssp585"]
PERIOD_ORDER = ["2021-2040", "2041-2060", "2061-2080", "2081-2100"]
GCM_ORDER = ["ACCESS-CM2", "MPI-ESM1-2-HR", "MRI-ESM2-0"]
SAMPLE_TYPE_ORDER = ["background", "presence"]
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

TEXT = "#26323A"
MUTED = "#687785"
GRID = "#DDE7E5"
SPINE = "#D7E0DD"
BLUE = "#3D72A8"
TEAL = "#2B837D"
RUST = "#BF5E62"
OCHRE = "#C57A27"
LIGHT_BLUE = "#BFD0E4"
LIGHT_TEAL = "#BFDCD5"

MODEL_COLORS = {"HGB": BLUE, "RF": TEAL}
LOSS_COLORS = {"HGB": "#6F97BE", "RF": "#68A79E"}


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
    raw["SampleType"] = pd.Categorical(
        raw["SampleType"], categories=SAMPLE_TYPE_ORDER, ordered=True
    )
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

    sample = raw[raw["group_type"].eq("sample_type")].copy()
    required = ["model_label", "gcm", "ssp", "period", "SampleType", "suitable_rate_pct"]
    bad = sample[sample[required].isna().any(axis=1)]
    if not bad.empty:
        raise ValueError(f"Unexpected labels or numeric values: {bad.head(10).to_dict('records')}")

    expected_rows = len(MODEL_ORDER) * len(GCM_ORDER) * len(SSP_ORDER) * len(PERIOD_ORDER) * 2
    if len(sample) != expected_rows:
        raise ValueError(f"Expected {expected_rows} sample_type rows, found {len(sample)}")

    meta = {
        "hgb_state": hgb_state,
        "rf_state": rf_state,
        "scenario_summary_rows": int(len(raw)),
        "sample_type_rows": int(len(sample)),
        "n_samples": int(sample.groupby("SampleType", observed=True)["n"].first().sum()),
        "n_background": int(sample[sample["SampleType"].eq("background")]["n"].max()),
        "n_presence": int(sample[sample["SampleType"].eq("presence")]["n"].max()),
        "data_scope": (
            "sample-level future WorldClim predictions only; "
            "full-grid area projections and land-cover constraints are not used"
        ),
    }
    return sample, meta


def prepare_tables(sample: pd.DataFrame) -> dict[str, pd.DataFrame]:
    index_cols = ["model_label", "gcm", "ssp", "period"]
    pivot = sample.pivot_table(
        index=index_cols,
        columns="SampleType",
        values=["suitable_rate_pct", "mean_probability", "n"],
        observed=True,
    )
    pivot.columns = [f"{metric}_{sample_type}" for metric, sample_type in pivot.columns]
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

    summary = (
        separation.groupby(["model_label", "ssp", "period"], observed=True)
        .agg(
            separation_mean_pp=("separation_pp", "mean"),
            separation_min_pp=("separation_pp", "min"),
            separation_max_pp=("separation_pp", "max"),
            separation_sd_pp=("separation_pp", "std"),
            presence_mean_pct=("suitable_rate_pct_presence", "mean"),
            background_mean_pct=("suitable_rate_pct_background", "mean"),
            background_min_pct=("suitable_rate_pct_background", "min"),
            background_max_pct=("suitable_rate_pct_background", "max"),
            n_gcm=("gcm", "nunique"),
        )
        .reset_index()
    )
    summary["ssp_label"] = summary["ssp"].astype(str).map(SSP_LABELS)
    summary["period_label"] = summary["period"].astype(str).map(PERIOD_LABELS)

    early = summary[summary["period"].astype(str).eq("2021-2040")][
        ["model_label", "ssp", "separation_mean_pp"]
    ].rename(columns={"separation_mean_pp": "separation_2030s_pp"})
    late = summary[summary["period"].astype(str).eq("2081-2100")][
        ["model_label", "ssp", "separation_mean_pp"]
    ].rename(columns={"separation_mean_pp": "separation_2090s_pp"})
    change = early.merge(late, on=["model_label", "ssp"], how="inner")
    change["change_2090s_minus_2030s_pp"] = (
        change["separation_2090s_pp"] - change["separation_2030s_pp"]
    )
    change["ssp_label"] = change["ssp"].astype(str).map(SSP_LABELS)

    background_2090s = summary[summary["period"].astype(str).eq("2081-2100")][
        [
            "model_label",
            "ssp",
            "ssp_label",
            "background_mean_pct",
            "background_min_pct",
            "background_max_pct",
            "n_gcm",
        ]
    ].copy()

    return {
        "sample_separation_by_scenario": separation,
        "sample_separation_summary": summary,
        "separation_change_2090s": change,
        "background_risk_2090s": background_2090s,
    }


def apply_common_axis_style(ax: plt.Axes, grid_axis: str = "x") -> None:
    ax.tick_params(axis="both", colors=TEXT, labelsize=9, length=0)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    for spine in ax.spines.values():
        spine.set_color(SPINE)
        spine.set_linewidth(1.0)
    if grid_axis:
        ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.8, alpha=0.85)
        ax.set_axisbelow(True)


def add_panel_label(
    ax: plt.Axes,
    label: str,
    color: str = TEAL,
    y: float = 1.05,
) -> None:
    ax.text(
        -0.08,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=10,
        fontweight="bold",
        color="white",
        bbox=dict(boxstyle="round,pad=0.2,rounding_size=0.04", fc=color, ec="none"),
        clip_on=False,
    )


def plot_heatmap(
    ax: plt.Axes,
    summary: pd.DataFrame,
    model: str,
    cmap: LinearSegmentedColormap,
    norm: Normalize,
) -> None:
    model_df = summary[summary["model_label"].eq(model)].copy()
    matrix = model_df.pivot_table(
        index="ssp",
        columns="period",
        values="separation_mean_pp",
        observed=True,
    ).reindex(index=SSP_ORDER, columns=PERIOD_ORDER)
    image = ax.imshow(matrix.values, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(np.arange(len(PERIOD_ORDER)))
    ax.set_xticklabels([PERIOD_LABELS[p] for p in PERIOD_ORDER], fontsize=8)
    ax.set_yticks(np.arange(len(SSP_ORDER)))
    ax.set_yticklabels([SSP_LABELS[s] for s in SSP_ORDER], fontsize=8)
    ax.set_title(f"{model} separation", fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks(np.arange(-0.5, len(PERIOD_ORDER), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(SSP_ORDER), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    for row_idx, ssp in enumerate(SSP_ORDER):
        for col_idx, period in enumerate(PERIOD_ORDER):
            value = matrix.loc[ssp, period]
            if pd.isna(value):
                label = "NA"
                color = MUTED
            else:
                label = f"{value:.1f}"
                color = "white" if value >= 68 else TEXT
            ax.text(
                col_idx,
                row_idx,
                label,
                ha="center",
                va="center",
                fontsize=8,
                color=color,
                fontweight="bold" if not pd.isna(value) else "normal",
            )
    for spine in ax.spines.values():
        spine.set_color(SPINE)
        spine.set_linewidth(1.0)
    return image


def plot_change(ax: plt.Axes, change: pd.DataFrame) -> None:
    y_base = np.arange(len(SSP_ORDER))
    offsets = {"HGB": -0.16, "RF": 0.16}
    ax.axvline(0, color="#9AA7B2", linewidth=1.0)
    all_values: list[float] = []
    for model in MODEL_ORDER:
        model_df = (
            change[change["model_label"].eq(model)]
            .set_index("ssp")
            .reindex(SSP_ORDER)
            .reset_index()
        )
        values = model_df["change_2090s_minus_2030s_pp"].to_numpy(dtype=float)
        all_values.extend([float(v) for v in values if np.isfinite(v)])
        y = y_base + offsets[model]
        bars = ax.barh(
            y,
            values,
            color=LOSS_COLORS[model],
            edgecolor="white",
            linewidth=0.6,
            height=0.26,
            label=model,
        )
        for bar, value in zip(bars, values):
            if not np.isfinite(value):
                continue
            x = value - 0.35 if value < 0 else value + 0.25
            ha = "right" if value < 0 else "left"
            ax.text(
                x,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.1f}",
                ha=ha,
                va="center",
                fontsize=8,
                color=TEXT,
            )
    ax.set_yticks(y_base)
    ax.set_yticklabels([SSP_LABELS[s] for s in SSP_ORDER], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Change in separation, 2090s minus 2030s (pp)", fontsize=9)
    ax.set_title("Separation loss by late century", fontsize=11, fontweight="bold", pad=8)
    min_value = min(all_values) if all_values else -1.0
    ax.set_xlim(min_value - 1.8, 1.0)
    ax.legend(
        loc="upper left",
        frameon=False,
        fontsize=8,
        handlelength=1.2,
        handletextpad=0.4,
        borderpad=0.2,
    )
    apply_common_axis_style(ax, grid_axis="x")


def plot_background_risk(ax: plt.Axes, risk: pd.DataFrame) -> None:
    y_base = np.arange(len(SSP_ORDER))
    offsets = {"HGB": -0.15, "RF": 0.15}
    for model in MODEL_ORDER:
        model_df = (
            risk[risk["model_label"].eq(model)]
            .set_index("ssp")
            .reindex(SSP_ORDER)
            .reset_index()
        )
        y = y_base + offsets[model]
        x = model_df["background_mean_pct"].to_numpy(dtype=float)
        lo = model_df["background_min_pct"].to_numpy(dtype=float)
        hi = model_df["background_max_pct"].to_numpy(dtype=float)
        ax.hlines(y, lo, hi, color=MODEL_COLORS[model], linewidth=2.0, alpha=0.55)
        ax.scatter(
            x,
            y,
            s=34,
            color=MODEL_COLORS[model],
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
            label=model,
        )
        for xi, yi in zip(x, y):
            ax.text(
                xi + 0.25,
                yi,
                f"{xi:.1f}",
                ha="left",
                va="center",
                fontsize=8,
                color=TEXT,
            )
    ax.set_yticks(y_base)
    ax.set_yticklabels([SSP_LABELS[s] for s in SSP_ORDER], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Background samples classified suitable in 2090s (%)", fontsize=9)
    ax.set_title("False-suitability risk", fontsize=11, fontweight="bold", pad=8)
    ax.set_xlim(0, max(15.5, float(risk["background_max_pct"].max()) + 2.0))
    ax.legend(
        loc="lower right",
        frameon=False,
        fontsize=8,
        handlelength=1.2,
        handletextpad=0.4,
        borderpad=0.2,
    )
    apply_common_axis_style(ax, grid_axis="x")


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

    summary = tables["sample_separation_summary"]
    change = tables["separation_change_2090s"]
    risk = tables["background_risk_2090s"]
    vmin = float(summary["separation_mean_pp"].min()) - 1.0
    vmax = float(summary["separation_mean_pp"].max()) + 1.0
    cmap = LinearSegmentedColormap.from_list(
        "sample_sep",
        ["#F2EBD8", "#C9DCCF", "#78B5A7", "#23776F"],
    )
    norm = Normalize(vmin=vmin, vmax=vmax)

    fig = plt.figure(figsize=(11.2, 7.2), facecolor="none")
    gs = fig.add_gridspec(
        2,
        3,
        width_ratios=[1.0, 1.0, 1.35],
        height_ratios=[1.0, 1.05],
        left=0.055,
        right=0.982,
        top=0.91,
        bottom=0.12,
        wspace=0.34,
        hspace=0.44,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0:2])
    ax_d = fig.add_subplot(gs[:, 2])

    image = plot_heatmap(ax_a, summary, "HGB", cmap, norm)
    plot_heatmap(ax_b, summary, "RF", cmap, norm)
    plot_change(ax_c, change)
    plot_background_risk(ax_d, risk)

    add_panel_label(ax_a, "a", TEAL)
    add_panel_label(ax_b, "b", TEAL)
    add_panel_label(ax_c, "c", RUST)
    add_panel_label(ax_d, "d", BLUE, y=1.015)

    cbar = fig.colorbar(
        image,
        ax=[ax_a, ax_b],
        orientation="horizontal",
        fraction=0.055,
        pad=0.12,
        aspect=30,
    )
    cbar.set_label("Presence-background suitability separation (percentage points)", fontsize=8, color=TEXT)
    cbar.ax.tick_params(labelsize=8, colors=TEXT, length=0)
    cbar.outline.set_edgecolor(SPINE)
    cbar.outline.set_linewidth(0.8)

    footnote = (
        f"Sample-level only: {meta['n_presence']:,} presence and "
        f"{meta['n_background']:,} background samples; full-grid area and "
        "land-cover constraints are not used in this figure."
    )
    fig.text(0.055, 0.045, footnote, ha="left", va="center", fontsize=8, color=MUTED)

    png_path = FIG_DIR / f"{FIG_BASENAME}.png"
    svg_path = FIG_DIR / f"{FIG_BASENAME}.svg"
    pdf_path = FIG_DIR / f"{FIG_BASENAME}.pdf"
    preview_path = FIG_DIR / f"{FIG_BASENAME}_white_preview.png"

    fig.savefig(png_path, dpi=600, bbox_inches="tight", pad_inches=0.03, transparent=True)
    fig.savefig(svg_path, bbox_inches="tight", pad_inches=0.03, transparent=True)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.03, transparent=True)
    fig.savefig(preview_path, dpi=300, bbox_inches="tight", pad_inches=0.03, facecolor="white")
    plt.close(fig)

    image_obj = Image.open(png_path)
    if image_obj.mode not in ("RGBA", "LA") and not (
        image_obj.mode == "P" and "transparency" in image_obj.info
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
        "# Stage44 supplementary figure candidate: future sample response separation",
        "",
        f"- Figure basename: `{FIG_BASENAME}`",
        f"- Generated at: {now_iso()}",
        f"- Script: `{rel(Path(__file__).resolve())}`",
        f"- Status: candidate; not inserted into Word until user confirms.",
        f"- Data scope: {meta['data_scope']}.",
        f"- Completed input status: HGB and RF sample-level selected10 future WorldClim predictions both record `success` with 48/48 scenarios and 0 failures.",
        "",
        "## Scientific content",
        "",
        "- Panels a-b show the mean presence-background suitable-rate separation across GCMs for HGB and RF.",
        "- Panel c shows how that separation changes from the 2030s to the 2090s.",
        "- Panel d shows the 2090s background-sample false-suitability rate; points are GCM means and line spans show min-max across GCMs.",
        "- This is not a full-grid suitable-area figure and does not use land-cover constraints.",
        "",
        "## Outputs",
        "",
    ]
    for key, path in outputs.items():
        lines.append(f"- {key}: `{rel(Path(path))}`")
    lines.extend(["", "## Tables", ""])
    for name in tables:
        lines.append(f"- {name}: `{rel(TABLE_DIR / f'stage44_v04_{name}.csv')}`")
    README_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    setup_logging()
    write_status("pending")
    try:
        write_status("running", started_at=now_iso())
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        TABLE_DIR.mkdir(parents=True, exist_ok=True)
        sample, meta = load_inputs()
        tables = prepare_tables(sample)
        for name, table in tables.items():
            atomic_to_csv(table, TABLE_DIR / f"stage44_v04_{name}.csv")
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
            tables={name: str(TABLE_DIR / f"stage44_v04_{name}.csv") for name in tables},
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
