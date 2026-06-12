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
STAGE_DIR = PROJECT_ROOT / "outputs" / "stage43_manuscript_supplementary_figures"
FIG_DIR = STAGE_DIR / "figures"
TABLE_DIR = STAGE_DIR / "tables"
LOG_DIR = PROJECT_ROOT / "logs"

INPUT_DIR = PROJECT_ROOT / "outputs" / "stage34_selected10_future_worldclim_sample_predictions"
HGB_SUMMARY_CSV = INPUT_DIR / "future_worldclim_sample_prediction_summary_selected10_hgb.csv"
RF_SUMMARY_CSV = INPUT_DIR / "future_worldclim_sample_prediction_summary_selected10_rf.csv"
HGB_SCENARIO_SUMMARY_CSV = INPUT_DIR / "future_worldclim_sample_prediction_scenario_summary_selected10_hgb.csv"
RF_SCENARIO_SUMMARY_CSV = INPUT_DIR / "future_worldclim_sample_prediction_scenario_summary_selected10_rf.csv"
HGB_STATE_JSON = LOG_DIR / "stage07_future_worldclim_sample_predictions_selected10_hgb_state.json"
RF_STATE_JSON = LOG_DIR / "stage07_future_worldclim_sample_predictions_selected10_rf_state.json"

FIG_BASENAME = "fig_stage43_supp_hgb_rf_future_sample_agreement_v02"
STATUS_JSON = STAGE_DIR / f"{FIG_BASENAME}_status.json"
README_MD = STAGE_DIR / f"{FIG_BASENAME}_README.md"
LOG_PATH = LOG_DIR / "stage43_supp_hgb_rf_future_sample_agreement_v02.log"

REQUIRED_SUMMARY_COLUMNS = {
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
REQUIRED_SCENARIO_COLUMNS = REQUIRED_SUMMARY_COLUMNS | {
    "group_type",
    "Response",
    "Region",
}

MODEL_LABELS = {
    "hist_gradient_boosting_balanced": "HGB",
    "random_forest_balanced": "RF",
}
MODEL_ORDER = ["HGB", "RF"]
SSP_ORDER = ["ssp126", "ssp245", "ssp370", "ssp585"]
PERIOD_ORDER = ["2021-2040", "2041-2060", "2061-2080", "2081-2100"]
GCM_ORDER = ["ACCESS-CM2", "MPI-ESM1-2-HR", "MRI-ESM2-0"]
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

TEXT = "#26323A"
MUTED = "#687785"
GRID = "#DDE7E5"
SPINE = "#D7E0DD"
BLUE = "#3D72A8"
TEAL = "#2B837D"
RUST = "#BF5E62"
OCHRE = "#C57A27"
GOLD = "#DDBA52"
LIGHT_BLUE = "#9EB9D5"
LIGHT_TEAL = "#A8D0C7"
LIGHT_RUST = "#E3B3B2"
LIGHT_OCHRE = "#E6CA8D"
SSP_COLORS = {
    "ssp126": BLUE,
    "ssp245": TEAL,
    "ssp370": OCHRE,
    "ssp585": RUST,
}
PERIOD_MARKERS = {
    "2021-2040": "o",
    "2041-2060": "s",
    "2061-2080": "^",
    "2081-2100": "D",
}


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
    if int(state.get("failed_scenarios", 0) or 0) != 0:
        raise RuntimeError(f"{path.name} records failed scenarios: {state}")
    if int(state.get("success_scenarios", 0) or 0) != 48:
        raise RuntimeError(f"{path.name} does not record all 48 scenarios: {state}")
    return state


def read_csv_checked(path: Path, required_columns: set[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input CSV: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"{path.name} is empty")
    missing = sorted(required_columns - set(df.columns))
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {missing}")
    return df


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    hgb_state = check_state_file(HGB_STATE_JSON, "hist_gradient_boosting_balanced")
    rf_state = check_state_file(RF_STATE_JSON, "random_forest_balanced")

    hgb = read_csv_checked(HGB_SUMMARY_CSV, REQUIRED_SUMMARY_COLUMNS)
    rf = read_csv_checked(RF_SUMMARY_CSV, REQUIRED_SUMMARY_COLUMNS)
    hgb_scenario = read_csv_checked(HGB_SCENARIO_SUMMARY_CSV, REQUIRED_SCENARIO_COLUMNS)
    rf_scenario = read_csv_checked(RF_SCENARIO_SUMMARY_CSV, REQUIRED_SCENARIO_COLUMNS)

    summary = pd.concat([hgb, rf], ignore_index=True)
    scenario = pd.concat([hgb_scenario, rf_scenario], ignore_index=True)
    for df in [summary, scenario]:
        df["model_label"] = df["model_group"].map(MODEL_LABELS)
        df["ssp"] = pd.Categorical(df["ssp"], categories=SSP_ORDER, ordered=True)
        df["period"] = pd.Categorical(df["period"], categories=PERIOD_ORDER, ordered=True)
        df["gcm"] = pd.Categorical(df["gcm"], categories=GCM_ORDER, ordered=True)
        for col in [
            "mean_probability",
            "median_probability",
            "p10_probability",
            "p90_probability",
            "suitable_rate",
            "Response",
        ]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df["suitable_rate_pct"] = df["suitable_rate"] * 100.0

    bad = summary[
        summary[["model_label", "ssp", "period", "gcm", "suitable_rate_pct"]].isna().any(axis=1)
    ]
    if not bad.empty:
        raise ValueError(f"Unexpected labels or numeric values: {bad.head(10).to_dict('records')}")

    expected_per_model = len(GCM_ORDER) * len(SSP_ORDER) * len(PERIOD_ORDER)
    counts = summary.groupby("model_label", observed=True).size().to_dict()
    for model in MODEL_ORDER:
        if int(counts.get(model, 0)) != expected_per_model:
            raise ValueError(f"{model} expected {expected_per_model} rows, found {counts.get(model, 0)}")

    meta = {
        "hgb_state": hgb_state,
        "rf_state": rf_state,
        "summary_rows": int(len(summary)),
        "scenario_summary_rows": int(len(scenario)),
        "n_samples": int(summary["n"].max()),
        "n_gcm": int(summary["gcm"].nunique()),
        "n_ssp": int(summary["ssp"].nunique()),
        "n_period": int(summary["period"].nunique()),
        "data_scope": "sample-level future WorldClim predictions only; full-grid area and land-cover constraints are not used",
    }
    return summary, scenario, meta


def prepare_tables(summary: pd.DataFrame, scenario: pd.DataFrame) -> dict[str, pd.DataFrame]:
    pivot = summary.pivot_table(
        index=["gcm", "ssp", "period"],
        columns="model_label",
        values=["suitable_rate_pct", "mean_probability"],
        observed=True,
    )
    pivot.columns = [f"{metric}_{model}" for metric, model in pivot.columns]
    pair = pivot.reset_index()
    pair["suitable_rate_diff_pp"] = pair["suitable_rate_pct_HGB"] - pair["suitable_rate_pct_RF"]
    pair["mean_probability_diff"] = pair["mean_probability_HGB"] - pair["mean_probability_RF"]
    pair["period_label"] = pair["period"].astype(str).map(PERIOD_LABELS)
    pair["ssp_label"] = pair["ssp"].astype(str).map(SSP_LABELS)

    diff_period = (
        pair.groupby(["ssp", "period"], observed=True)
        .agg(
            diff_mean_pp=("suitable_rate_diff_pp", "mean"),
            diff_min_pp=("suitable_rate_diff_pp", "min"),
            diff_max_pp=("suitable_rate_diff_pp", "max"),
            rf_mean_pct=("suitable_rate_pct_RF", "mean"),
            hgb_mean_pct=("suitable_rate_pct_HGB", "mean"),
            n_gcm=("gcm", "nunique"),
        )
        .reset_index()
    )
    diff_period["period_index"] = diff_period["period"].cat.codes.astype(float)
    diff_period["ssp_label"] = diff_period["ssp"].astype(str).map(SSP_LABELS)
    diff_period["period_label"] = diff_period["period"].astype(str).map(PERIOD_LABELS)

    end_century = summary[summary["period"].astype(str).eq("2081-2100")].copy()
    end_century["ssp_label"] = end_century["ssp"].astype(str).map(SSP_LABELS)

    regional = scenario[
        scenario["group_type"].eq("region_response")
        & scenario["period"].astype(str).eq("2081-2100")
        & scenario["ssp"].astype(str).eq("ssp585")
        & scenario["Response"].eq(1.0)
        & scenario["Region"].notna()
    ].copy()
    regional["region_label"] = regional["Region"].map(REGION_LABELS).fillna(regional["Region"])
    region_pivot = regional.pivot_table(
        index=["Region", "region_label", "gcm"],
        columns="model_label",
        values="suitable_rate_pct",
        observed=True,
    ).reset_index()
    if not {"HGB", "RF"}.issubset(region_pivot.columns):
        raise ValueError("Regional panel needs both HGB and RF values")
    region_pivot["diff_pp"] = region_pivot["HGB"] - region_pivot["RF"]
    region_diff = (
        region_pivot.groupby(["Region", "region_label"], observed=True)
        .agg(
            diff_mean_pp=("diff_pp", "mean"),
            diff_min_pp=("diff_pp", "min"),
            diff_max_pp=("diff_pp", "max"),
            hgb_mean_pct=("HGB", "mean"),
            rf_mean_pct=("RF", "mean"),
            n_gcm=("gcm", "nunique"),
        )
        .reset_index()
        .sort_values("diff_mean_pp")
    )
    return {
        "pair": pair,
        "diff_period": diff_period,
        "end_century": end_century,
        "region_pivot": region_pivot,
        "region_diff": region_diff,
    }


def style_axes(ax: plt.Axes) -> None:
    ax.tick_params(colors=TEXT, labelsize=8.6, length=3, width=0.8)
    for spine in ax.spines.values():
        spine.set_color(SPINE)
        spine.set_linewidth(0.9)
    ax.grid(True, color=GRID, linewidth=0.75, alpha=0.85)
    ax.set_axisbelow(True)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.075,
        1.045,
        label,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=9.5,
        fontweight="bold",
        color="white",
        bbox=dict(boxstyle="round,pad=0.18,rounding_size=0.04", fc=TEAL, ec="none"),
        clip_on=False,
        zorder=10,
    )


def plot_scatter(ax: plt.Axes, pair: pd.DataFrame) -> None:
    style_axes(ax)
    for ssp in SSP_ORDER:
        sub = pair[pair["ssp"].astype(str).eq(ssp)]
        for period in PERIOD_ORDER:
            sp = sub[sub["period"].astype(str).eq(period)]
            ax.scatter(
                sp["suitable_rate_pct_RF"],
                sp["suitable_rate_pct_HGB"],
                s=32,
                marker=PERIOD_MARKERS[period],
                color=SSP_COLORS[ssp],
                edgecolor="white",
                linewidth=0.6,
                alpha=0.9,
                zorder=3,
            )
    lim_min = 10.0
    lim_max = 32.0
    ax.plot([lim_min, lim_max], [lim_min, lim_max], color="#AAB5B8", lw=1.1, ls=(0, (3, 3)))
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_xlabel("RF suitable sample rate (%)", fontsize=9.5, color=TEXT)
    ax.set_ylabel("HGB suitable sample rate (%)", fontsize=9.5, color=TEXT)
    ax.text(
        12.1,
        30.4,
        "HGB > RF",
        fontsize=8.6,
        color=MUTED,
        ha="left",
        va="center",
    )
    add_panel_label(ax, "a")


def plot_diff_ribbon(ax: plt.Axes, diff_period: pd.DataFrame) -> None:
    style_axes(ax)
    x = np.arange(len(PERIOD_ORDER), dtype=float)
    for ssp in SSP_ORDER:
        sub = diff_period[diff_period["ssp"].astype(str).eq(ssp)].sort_values("period")
        ax.fill_between(
            x,
            sub["diff_min_pp"].to_numpy(float),
            sub["diff_max_pp"].to_numpy(float),
            color=SSP_COLORS[ssp],
            alpha=0.13,
            lw=0,
        )
        ax.plot(
            x,
            sub["diff_mean_pp"].to_numpy(float),
            color=SSP_COLORS[ssp],
            lw=1.8,
            marker="o",
            markersize=4.2,
            markeredgecolor="white",
            markeredgewidth=0.6,
            label=SSP_LABELS[ssp],
        )
    ax.set_xticks(x)
    ax.set_xticklabels([PERIOD_LABELS[p] for p in PERIOD_ORDER])
    ax.set_ylim(6.2, 12.6)
    ax.set_ylabel("HGB - RF suitable rate (pp)", fontsize=9.5, color=TEXT)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.015, 0.985),
        frameon=False,
        ncol=2,
        fontsize=8.0,
        handlelength=1.7,
        columnspacing=0.9,
        borderaxespad=0.0,
    )
    add_panel_label(ax, "b")


def plot_end_century_distribution(ax: plt.Axes, end_century: pd.DataFrame) -> None:
    style_axes(ax)
    positions = np.arange(len(SSP_ORDER), dtype=float)
    offsets = {"RF": -0.16, "HGB": 0.16}
    colors = {"RF": LIGHT_TEAL, "HGB": LIGHT_BLUE}
    edge_colors = {"RF": TEAL, "HGB": BLUE}
    rng = np.random.default_rng(20260601)
    for model in MODEL_ORDER:
        data = []
        pos = []
        for idx, ssp in enumerate(SSP_ORDER):
            vals = end_century[
                end_century["model_label"].eq(model) & end_century["ssp"].astype(str).eq(ssp)
            ]["suitable_rate_pct"].to_numpy(float)
            data.append(vals)
            pos.append(positions[idx] + offsets[model])
        violin = ax.violinplot(
            data,
            positions=pos,
            widths=0.26,
            showextrema=False,
            showmeans=False,
            showmedians=False,
        )
        for body in violin["bodies"]:
            body.set_facecolor(colors[model])
            body.set_edgecolor(edge_colors[model])
            body.set_alpha(0.78)
            body.set_linewidth(0.8)
        for p, vals in zip(pos, data):
            jitter = rng.normal(0, 0.018, len(vals))
            ax.scatter(
                np.full(len(vals), p) + jitter,
                vals,
                s=22,
                color=edge_colors[model],
                edgecolor="white",
                linewidth=0.55,
                zorder=3,
            )
            ax.plot([p - 0.08, p + 0.08], [np.mean(vals), np.mean(vals)], color=TEXT, lw=1.1)
    ax.set_xticks(positions)
    ax.set_xticklabels([SSP_LABELS[s] for s in SSP_ORDER])
    ax.set_ylim(10.0, 32.5)
    ax.set_ylabel("End-century suitable sample rate (%)", fontsize=9.5, color=TEXT)
    handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=BLUE, markeredgecolor="white", markersize=6, label="HGB"),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=TEAL, markeredgecolor="white", markersize=6, label="RF"),
    ]
    ax.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(0.995, 0.99),
        frameon=False,
        fontsize=8.2,
        handletextpad=0.4,
    )
    add_panel_label(ax, "c")


def plot_region_diff(ax: plt.Axes, region_diff: pd.DataFrame, region_pivot: pd.DataFrame) -> None:
    style_axes(ax)
    region_diff = region_diff.copy().sort_values("diff_mean_pp")
    y = np.arange(len(region_diff), dtype=float)
    x_max = max(40.0, float(region_pivot["diff_pp"].max()) + 2.5)
    ax.barh(
        y,
        region_diff["diff_mean_pp"],
        color=LIGHT_OCHRE,
        edgecolor=OCHRE,
        linewidth=0.8,
        height=0.58,
        zorder=2,
    )
    for idx, row in region_diff.reset_index(drop=True).iterrows():
        vals = region_pivot[region_pivot["Region"].eq(row["Region"])]["diff_pp"].to_numpy(float)
        ax.scatter(
            vals,
            np.full(len(vals), idx),
            s=18,
            color=OCHRE,
            edgecolor="white",
            linewidth=0.55,
            alpha=0.9,
            zorder=3,
        )
        label_x = min(float(row["diff_mean_pp"]) + 1.0, x_max - 2.8)
        ax.annotate(
            f"{row['diff_mean_pp']:.1f}",
            xy=(float(row["diff_mean_pp"]), idx),
            xytext=(label_x, idx + 0.28),
            textcoords="data",
            fontsize=7.2,
            color=TEXT,
            ha="left",
            va="bottom",
            arrowprops=dict(arrowstyle="-", color=OCHRE, lw=0.55, shrinkA=1.5, shrinkB=2.5),
            clip_on=False,
            zorder=4,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(region_diff["region_label"])
    ax.set_xlim(0, x_max)
    ax.set_xlabel("HGB - RF suitable rate (pp)", fontsize=9.5, color=TEXT)
    ax.text(
        0.99,
        1.035,
        "SSP5-8.5, 2081-2100",
        transform=ax.transAxes,
        fontsize=7.8,
        color=MUTED,
        ha="right",
        va="bottom",
        clip_on=False,
    )
    add_panel_label(ax, "d")


def draw_figure(tables: dict[str, pd.DataFrame]) -> tuple[plt.Figure, np.ndarray]:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Microsoft YaHei"],
            "axes.titleweight": "bold",
            "axes.labelcolor": TEXT,
            "text.color": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(11.3, 7.25), constrained_layout=False)
    fig.patch.set_alpha(0.0)
    for ax in axes.ravel():
        ax.set_facecolor((1, 1, 1, 0))

    plot_scatter(axes[0, 0], tables["pair"])
    plot_diff_ribbon(axes[0, 1], tables["diff_period"])
    plot_end_century_distribution(axes[1, 0], tables["end_century"])
    plot_region_diff(axes[1, 1], tables["region_diff"], tables["region_pivot"])

    plt.subplots_adjust(left=0.075, right=0.985, bottom=0.095, top=0.975, wspace=0.36, hspace=0.42)
    return fig, axes


def write_tables(tables: dict[str, pd.DataFrame]) -> dict[str, str]:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    out: dict[str, str] = {}
    table_map = {
        "scenario_model_pair": tables["pair"],
        "period_model_difference": tables["diff_period"],
        "end_century_model_distribution": tables["end_century"],
        "ssp585_end_century_region_difference": tables["region_diff"],
    }
    for name, df in table_map.items():
        path = TABLE_DIR / f"stage43_v02_{name}.csv"
        df.to_csv(path, index=False, encoding="utf-8-sig")
        out[name] = str(path)
    return out


def save_outputs(fig: plt.Figure) -> dict[str, str]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    png = FIG_DIR / f"{FIG_BASENAME}.png"
    svg = FIG_DIR / f"{FIG_BASENAME}.svg"
    pdf = FIG_DIR / f"{FIG_BASENAME}.pdf"
    white = FIG_DIR / f"{FIG_BASENAME}_white_preview.png"

    fig.savefig(png, dpi=600, bbox_inches="tight", pad_inches=0.02, transparent=True)
    fig.savefig(svg, bbox_inches="tight", pad_inches=0.02, transparent=True)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.02, transparent=True)
    fig.savefig(white, dpi=220, bbox_inches="tight", pad_inches=0.02, transparent=False, facecolor="white")

    img = Image.open(png)
    has_alpha = img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info)
    if not has_alpha:
        raise RuntimeError(f"PNG export lacks alpha channel: {png}")
    return {"png": str(png), "svg": str(svg), "pdf": str(pdf), "white_preview": str(white)}


def write_readme(meta: dict[str, object], outputs: dict[str, str], tables: dict[str, str]) -> None:
    lines = [
        "# Stage43 supplementary figure: HGB-RF future sample agreement",
        "",
        f"- Created: {now_iso()}",
        f"- Script: `{Path(__file__).resolve()}`",
        "- Scope: selected10 sample-level future WorldClim predictions only.",
        "- Excluded by design: full-grid suitable area, constrained suitable area, and land-cover constraint outputs because those long-running outputs may still be updating.",
        "- Data completion check: HGB and RF sample prediction state files are both `success` with 48/48 scenarios and 0 failed scenarios.",
        "- Figure status: candidate; do not insert into Word until explicitly confirmed by the user.",
        "",
        "## Inputs",
        f"- `{HGB_SUMMARY_CSV}`",
        f"- `{RF_SUMMARY_CSV}`",
        f"- `{HGB_SCENARIO_SUMMARY_CSV}`",
        f"- `{RF_SCENARIO_SUMMARY_CSV}`",
        "",
        "## Outputs",
    ]
    lines.extend(f"- {key}: `{value}`" for key, value in outputs.items())
    lines.append("")
    lines.append("## Tables")
    lines.extend(f"- {key}: `{value}`" for key, value in tables.items())
    lines.append("")
    lines.append("## Metadata")
    lines.append("```json")
    lines.append(json.dumps(meta, ensure_ascii=False, indent=2))
    lines.append("```")
    README_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    setup_logging()
    write_status("running", started_at=now_iso())
    try:
        logging.info("Loading completed HGB and RF sample-prediction summaries")
        summary, scenario, meta = load_inputs()
        tables = prepare_tables(summary, scenario)
        table_outputs = write_tables(tables)
        fig, _axes = draw_figure(tables)
        figure_outputs = save_outputs(fig)
        plt.close(fig)
        write_readme(meta, figure_outputs, table_outputs)
        write_status(
            "success",
            finished_at=now_iso(),
            inputs={
                "hgb_summary_csv": str(HGB_SUMMARY_CSV),
                "rf_summary_csv": str(RF_SUMMARY_CSV),
                "hgb_scenario_summary_csv": str(HGB_SCENARIO_SUMMARY_CSV),
                "rf_scenario_summary_csv": str(RF_SCENARIO_SUMMARY_CSV),
            },
            outputs=figure_outputs,
            tables=table_outputs,
            metadata=meta,
            candidate_note="Do not insert into Word until user confirms this candidate figure.",
        )
        logging.info("Completed %s", FIG_BASENAME)
    except Exception as exc:
        logging.exception("Stage43 figure failed")
        write_status("failed", failed_at=now_iso(), error=repr(exc))
        raise


if __name__ == "__main__":
    main()
