from __future__ import annotations

import json
import logging
import math
import re
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patches
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGE_DIR = PROJECT_ROOT / "outputs" / "stage41_manuscript_supplementary_figures"
FIG_DIR = STAGE_DIR / "figures"
TABLE_DIR = STAGE_DIR / "tables"
LOG_DIR = PROJECT_ROOT / "logs"

INPUT_DIR = PROJECT_ROOT / "outputs" / "stage30_environment_factor_collinearity" / "tables"
RECOMMENDED_CSV = INPUT_DIR / "stage30_recommended_predictors.csv"
PEARSON_CSV = INPUT_DIR / "stage30_pearson_correlation_matrix.csv"
HIGH_CORR_CSV = INPUT_DIR / "stage30_high_correlation_pairs.csv"
VIF_ALL_CSV = INPUT_DIR / "stage30_vif_all_predictors.csv"
SELECTED_TXT = INPUT_DIR / "stage30_selected_predictor_list.txt"

FIG_BASENAME = "fig_stage41_supp_predictor_correlation_vif_v05"
STATUS_JSON = STAGE_DIR / f"{FIG_BASENAME}_status.json"
README_MD = STAGE_DIR / f"{FIG_BASENAME}_README.md"
LOG_PATH = LOG_DIR / "stage41_supp_predictor_correlation_vif_v05.log"

TEXT = "#25313B"
MUTED = "#6C7783"
GRID = "#DCE5E4"
SPINE = "#D6DFDB"
TEAL = "#2C837E"
TEAL_LIGHT = "#A8D0C5"
BLUE = "#3B75AF"
RUST = "#BF5B5B"
RUST_LIGHT = "#E9C7C4"
OCHRE = "#C27A28"
NOTE_BG = "#F8FBFA"

REQUIRED_RECOMMENDED_COLUMNS = {
    "feature",
    "feature_label",
    "group",
    "decision",
    "vif_all_predictors",
    "vif_final_selected_set",
}
REQUIRED_HIGH_CORR_COLUMNS = {
    "method",
    "feature_a",
    "feature_a_label",
    "feature_b",
    "feature_b_label",
    "correlation",
}
REQUIRED_VIF_COLUMNS = {"feature", "feature_label", "group", "vif"}

FAMILY_LABELS = {
    "temperature_mean": "Temperature",
    "temperature_range": "Temperature",
    "temperature_ratio": "Temperature",
    "temperature_quarter": "Temperature",
    "temperature_extreme": "Temperature",
    "temperature_seasonality": "Temperature",
    "precipitation_total": "Precipitation",
    "precipitation_quarter": "Precipitation",
    "precipitation_extreme": "Precipitation",
    "precipitation_seasonality": "Precipitation",
    "topography": "Terrain",
}
FAMILY_ORDER = ["Temperature", "Precipitation", "Terrain"]


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


def read_csv_checked(path: Path, required_columns: set[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required input: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"{path.name} is empty")
    if required_columns:
        missing = sorted(required_columns - set(df.columns))
        if missing:
            raise ValueError(f"{path.name} is missing required columns: {missing}")
    return df


def short_label(feature: str, fallback: str | None = None) -> str:
    if feature == "wc_elev_m":
        return "Elevation"
    match = re.match(r"wc_bio0?(\d+)$", feature)
    if match:
        return f"Bio{int(match.group(1))}"
    if fallback:
        if fallback.lower().startswith("bio"):
            return fallback.split(":")[0].replace("Bio0", "Bio")
        return fallback.split(":")[0]
    return feature


def as_numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.replace({"inf": np.inf, "Inf": np.inf, "INF": np.inf}), errors="coerce")


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    recommended = read_csv_checked(RECOMMENDED_CSV, REQUIRED_RECOMMENDED_COLUMNS)
    high_corr = read_csv_checked(HIGH_CORR_CSV, REQUIRED_HIGH_CORR_COLUMNS)
    vif_all = read_csv_checked(VIF_ALL_CSV, REQUIRED_VIF_COLUMNS)
    pearson = read_csv_checked(PEARSON_CSV)
    if "feature" not in pearson.columns:
        raise ValueError("Pearson matrix must contain a feature column")
    if not SELECTED_TXT.exists():
        raise FileNotFoundError(f"Missing selected predictor list: {SELECTED_TXT}")
    selected = [
        line.strip()
        for line in SELECTED_TXT.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(selected) != 10:
        raise ValueError(f"Expected 10 selected predictors, got {len(selected)}")
    missing_selected = [feature for feature in selected if feature not in set(recommended["feature"])]
    if missing_selected:
        raise ValueError(f"Selected predictors missing from recommended table: {missing_selected}")
    matrix_features = set(pearson["feature"])
    missing_matrix = [feature for feature in selected if feature not in matrix_features or feature not in pearson.columns]
    if missing_matrix:
        raise ValueError(f"Selected predictors missing from Pearson matrix: {missing_matrix}")
    return recommended, pearson, high_corr, vif_all, selected


def prepare_data() -> dict[str, pd.DataFrame | list[str] | float]:
    recommended, pearson, high_corr, vif_all, selected = load_inputs()
    rec = recommended.copy()
    rec["short_label"] = [short_label(f, lbl) for f, lbl in zip(rec["feature"], rec["feature_label"])]
    rec["family"] = rec["group"].map(FAMILY_LABELS).fillna("Other")
    rec["decision"] = rec["decision"].str.lower()
    selected_from_decision = rec.loc[rec["decision"].eq("selected"), "feature"].tolist()
    if set(selected_from_decision) != set(selected):
        raise ValueError("Selected predictor list and recommended table do not match")

    pearson_selected = pearson.set_index("feature").loc[selected, selected].apply(pd.to_numeric)
    pearson_selected.index = [short_label(feature) for feature in pearson_selected.index]
    pearson_selected.columns = [short_label(feature) for feature in pearson_selected.columns]

    high = high_corr.copy()
    high = high[high["method"].str.lower().eq("pearson")].copy()
    if high.empty:
        raise ValueError("No Pearson rows in high-correlation pair table")
    high["correlation"] = pd.to_numeric(high["correlation"], errors="coerce")
    high = high.dropna(subset=["correlation"]).copy()
    high["abs_correlation"] = high["correlation"].abs()
    high["feature_a_short"] = [short_label(f, lbl) for f, lbl in zip(high["feature_a"], high["feature_a_label"])]
    high["feature_b_short"] = [short_label(f, lbl) for f, lbl in zip(high["feature_b"], high["feature_b_label"])]
    high["pair_label"] = high["feature_a_short"] + " - " + high["feature_b_short"]
    high = high.sort_values("abs_correlation", ascending=False).head(10).sort_values("abs_correlation")

    vif = vif_all.copy()
    vif["short_label"] = [short_label(f, lbl) for f, lbl in zip(vif["feature"], vif["feature_label"])]
    vif["vif_numeric"] = as_numeric_series(vif["vif"])
    vif["decision"] = vif["feature"].map(dict(zip(rec["feature"], rec["decision"]))).fillna("unknown")
    vif["family"] = vif["feature"].map(dict(zip(rec["feature"], rec["family"]))).fillna("Other")
    finite = vif.loc[np.isfinite(vif["vif_numeric"]), "vif_numeric"]
    if finite.empty:
        raise ValueError("No finite VIF values are available")
    cap_value = max(10000.0, float(finite.max()) * 1.15)
    vif["vif_plot"] = vif["vif_numeric"].replace(np.inf, cap_value)
    vif["is_infinite"] = np.isinf(vif["vif_numeric"])
    vif = vif.sort_values("vif_plot", ascending=True).reset_index(drop=True)

    tiles = rec.copy()
    tiles["family"] = pd.Categorical(tiles["family"], categories=FAMILY_ORDER + ["Other"], ordered=True)
    bio_num = tiles["feature"].str.extract(r"wc_bio0?(\d+)")[0].astype(float)
    tiles["_sort"] = bio_num.fillna(99)
    tiles = tiles.sort_values(["family", "_sort", "feature"]).drop(columns=["_sort"]).reset_index(drop=True)

    family_counts = (
        tiles.groupby(["family", "decision"], observed=False)
        .size()
        .reset_index(name="count")
        .query("count > 0")
    )

    return {
        "recommended": rec,
        "pearson_selected": pearson_selected,
        "high_pairs": high,
        "vif": vif,
        "tiles": tiles,
        "family_counts": family_counts,
        "selected": selected,
        "cap_value": cap_value,
    }


def style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_color(SPINE)
        spine.set_linewidth(0.9)
    ax.tick_params(axis="both", labelsize=7.8, colors=TEXT, length=3)
    ax.grid(True, axis="x", color=GRID, linewidth=0.75, alpha=0.9)
    ax.grid(True, axis="y", color=GRID, linewidth=0.55, alpha=0.5)
    ax.set_axisbelow(True)


def panel_label(ax: plt.Axes, label: str, color: str = TEAL) -> None:
    ax.text(
        -0.065,
        1.035,
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


def plot_selected_heatmap(ax: plt.Axes, matrix: pd.DataFrame) -> None:
    labels = matrix.index.tolist()
    values = matrix.to_numpy(dtype=float)
    mask = np.triu(np.ones_like(values, dtype=bool), k=1)
    values_masked = np.ma.array(values, mask=mask)
    im = ax.imshow(values_masked, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7.1, color=TEXT)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=7.1, color=TEXT)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_color(SPINE)
        spine.set_linewidth(0.9)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if i <= j:
                continue
            val = values[i, j]
            if abs(val) >= 0.70:
                ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=5.8,
                    color="white" if abs(val) >= 0.78 else TEXT,
                )
    ax.set_title("Selected-predictor Pearson r", fontsize=10.2, fontweight="bold", color=TEXT, pad=9)
    ax.set_xlim(-0.5, len(labels) - 0.5)
    ax.set_ylim(len(labels) - 0.5, -0.5)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.025)
    cbar.outline.set_edgecolor(SPINE)
    cbar.ax.tick_params(labelsize=7.2, colors=TEXT, length=2)
    cbar.set_label("r", fontsize=7.4, color=TEXT, rotation=0, labelpad=7)


def plot_high_corr_pairs(ax: plt.Axes, high: pd.DataFrame) -> None:
    style_axis(ax)
    y = np.arange(len(high))
    vals = high["abs_correlation"].to_numpy(dtype=float)
    signed = high["correlation"].to_numpy(dtype=float)
    colors = np.where(signed >= 0, TEAL, RUST)
    ax.hlines(y, 0.86, vals, color="#D6E2DF", linewidth=6, zorder=1)
    ax.scatter(vals, y, s=36, c=colors, edgecolor="white", linewidth=0.8, zorder=3)
    for yi, val, corr in zip(y, vals, signed):
        label = f"{corr:+.2f}"
        if val >= 0.985:
            x_text = val - 0.006
            ha = "right"
        else:
            x_text = min(val + 0.006, 0.994)
            ha = "left"
        ax.text(
            x_text,
            yi,
            label,
            ha=ha,
            va="center",
            fontsize=7.2,
            color=TEXT,
            clip_on=False,
        )
    ax.axvline(0.90, color=OCHRE, linestyle=(0, (3, 2)), linewidth=1.0)
    ax.text(
        0.908,
        len(high) - 0.62,
        "|r| threshold = 0.90",
        ha="left",
        va="top",
        fontsize=6.9,
        color=OCHRE,
        bbox=dict(boxstyle="round,pad=0.12,rounding_size=0.03", fc="white", ec="none", alpha=0.88),
    )
    ax.set_yticks(y)
    ax.set_yticklabels(high["pair_label"], fontsize=7.3, color=TEXT)
    ax.set_xlim(0.86, 1.00)
    ax.set_xticks(np.arange(0.86, 1.001, 0.02))
    ax.set_xlabel("Absolute Pearson correlation", fontsize=8.4, color=TEXT, labelpad=5)
    ax.set_title("Original-pool high-correlation pairs", fontsize=10.2, fontweight="bold", color=TEXT, pad=9)


def plot_initial_vif(ax: plt.Axes, vif: pd.DataFrame, cap_value: float) -> None:
    style_axis(ax)
    y = np.arange(len(vif))
    is_selected = vif["decision"].eq("selected").to_numpy()
    colors = np.where(is_selected, TEAL, RUST)
    stem_colors = np.where(is_selected, TEAL_LIGHT, RUST_LIGHT)
    vals = vif["vif_plot"].astype(float).to_numpy()
    ax.hlines(y, 1.0, vals, color=stem_colors, linewidth=4.8, zorder=1)
    ax.scatter(vals, y, s=32, c=colors, edgecolor="white", linewidth=0.7, zorder=3)
    ax.axvline(10, color=OCHRE, linestyle=(0, (3, 2)), linewidth=1.0)
    ax.text(
        10.6,
        len(vif) - 0.35,
        "VIF = 10",
        ha="left",
        va="top",
        fontsize=7.0,
        color=OCHRE,
    )
    for yi, row in vif.iterrows():
        value = row["vif_numeric"]
        plot_value = float(row["vif_plot"])
        if bool(row["is_infinite"]):
            text = "inf"
        elif value >= 100:
            text = f"{value:.0f}"
        elif value >= 10:
            text = f"{value:.1f}"
        else:
            text = f"{value:.2f}"
        if value >= 50 or row["decision"] == "selected":
            ax.text(
                plot_value * 1.05,
                yi,
                text,
                ha="left",
                va="center",
                fontsize=6.3,
                color=MUTED,
                clip_on=False,
            )
    ax.set_xscale("log")
    ax.set_xlim(1, cap_value * 1.65)
    ax.set_yticks(y)
    ax.set_yticklabels(vif["short_label"], fontsize=6.8, color=TEXT)
    ax.set_xlabel("Initial VIF before screening (log scale)", fontsize=8.4, color=TEXT, labelpad=5)
    ax.set_title("Initial multicollinearity burden", fontsize=10.2, fontweight="bold", color=TEXT, pad=9)
    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=TEAL, markeredgecolor="white", markersize=6, label="selected"),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=RUST, markeredgecolor="white", markersize=6, label="dropped"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower right",
        frameon=False,
        fontsize=7.2,
        handletextpad=0.35,
        borderaxespad=0.2,
    )


def tile_positions(row_count: int, row_y: float) -> list[tuple[float, float]]:
    width = 0.064
    gap = 0.008
    total_width = row_count * width + max(row_count - 1, 0) * gap
    start = max(0.16, (1.0 - total_width) / 2)
    return [(start + i * (width + gap), row_y) for i in range(row_count)]


def plot_selection_tiles(ax: plt.Axes, tiles: pd.DataFrame) -> None:
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(
        0.5,
        1.03,
        "Screening outcome by predictor family",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=10.2,
        fontweight="bold",
        color=TEXT,
        clip_on=False,
    )
    row_y = {"Temperature": 0.66, "Precipitation": 0.38, "Terrain": 0.13}
    tile_w = 0.064
    tile_h = 0.15
    tile_records = []
    for family in FAMILY_ORDER:
        sub = tiles[tiles["family"].astype(str).eq(family)].copy()
        if sub.empty:
            continue
        ax.text(
            0.02,
            row_y[family] + tile_h / 2,
            family,
            ha="left",
            va="center",
            fontsize=8.0,
            fontweight="bold",
            color=TEXT,
        )
        positions = tile_positions(len(sub), row_y[family])
        for (_, row), (x, y0) in zip(sub.iterrows(), positions):
            selected = row["decision"] == "selected"
            fc = "#DCEFEB" if selected else "#F2DCDC"
            ec = TEAL if selected else RUST
            rect = patches.FancyBboxPatch(
                (x, y0),
                tile_w,
                tile_h,
                boxstyle="round,pad=0.006,rounding_size=0.018",
                linewidth=1.0,
                edgecolor=ec,
                facecolor=fc,
            )
            ax.add_patch(rect)
            ax.text(
                x + tile_w / 2,
                y0 + tile_h / 2,
                row["short_label"],
                ha="center",
                va="center",
                fontsize=6.2,
                fontweight="bold" if selected else "normal",
                color=TEXT,
            )
            tile_records.append(
                {
                    "family": family,
                    "feature": row["feature"],
                    "short_label": row["short_label"],
                    "decision": row["decision"],
                    "x": x,
                    "y": y0,
                }
            )
    ax.scatter([0.70, 0.84], [0.925, 0.925], s=46, c=[TEAL, RUST], edgecolors="white", linewidths=0.8)
    ax.text(0.715, 0.925, "selected", ha="left", va="center", fontsize=7.4, color=TEXT)
    ax.text(0.855, 0.925, "dropped", ha="left", va="center", fontsize=7.4, color=TEXT)
    ax.text(
        0.02,
        0.015,
        "Tiles represent the original 20-factor climate + terrain pool.",
        ha="left",
        va="bottom",
        fontsize=7.0,
        color=MUTED,
    )
    ax._tile_records = tile_records  # type: ignore[attr-defined]


def make_figure(data: dict[str, pd.DataFrame | list[str] | float]) -> plt.Figure:
    plt.rcParams.update(
        {
            "font.family": ["Arial", "DejaVu Sans", "sans-serif"],
            "axes.unicode_minus": False,
            "svg.fonttype": "none",
        }
    )
    fig = plt.figure(figsize=(12.1, 8.35), constrained_layout=False)
    gs = fig.add_gridspec(
        2,
        2,
        left=0.055,
        right=0.985,
        top=0.94,
        bottom=0.14,
        wspace=0.29,
        hspace=0.40,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    plot_selected_heatmap(ax_a, data["pearson_selected"])  # type: ignore[arg-type]
    plot_high_corr_pairs(ax_b, data["high_pairs"])  # type: ignore[arg-type]
    plot_initial_vif(ax_c, data["vif"], float(data["cap_value"]))  # type: ignore[arg-type]
    plot_selection_tiles(ax_d, data["tiles"])  # type: ignore[arg-type]

    panel_label(ax_a, "a", TEAL)
    panel_label(ax_b, "b", RUST)
    panel_label(ax_c, "c", BLUE)
    panel_label(ax_d, "d", TEAL)

    fig.text(
        0.055,
        0.065,
        "Data boundary: completed Stage30 predictor collinearity diagnostics only; future projection and land-cover outputs are not used.",
        ha="left",
        va="bottom",
        fontsize=7.1,
        color=MUTED,
        bbox=dict(boxstyle="round,pad=0.22,rounding_size=0.04", fc=NOTE_BG, ec=SPINE, lw=0.6),
    )
    return fig


def save_outputs(fig: plt.Figure) -> dict[str, object]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    png_path = FIG_DIR / f"{FIG_BASENAME}.png"
    svg_path = FIG_DIR / f"{FIG_BASENAME}.svg"
    pdf_path = FIG_DIR / f"{FIG_BASENAME}.pdf"
    preview_path = FIG_DIR / f"{FIG_BASENAME}_white_preview.png"
    fig.savefig(png_path, dpi=600, bbox_inches="tight", pad_inches=0.12, transparent=True)
    fig.savefig(svg_path, bbox_inches="tight", pad_inches=0.12, transparent=True)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.12, transparent=True)
    plt.close(fig)

    img = Image.open(png_path)
    if img.mode not in ("RGBA", "LA") and not (img.mode == "P" and "transparency" in img.info):
        raise ValueError("PNG does not contain transparency")
    rgba = img.convert("RGBA")
    white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    white.alpha_composite(rgba)
    white.convert("RGB").save(preview_path, quality=95)
    return {
        "png": str(png_path),
        "svg": str(svg_path),
        "pdf": str(pdf_path),
        "white_preview": str(preview_path),
        "png_mode": img.mode,
        "png_size": list(img.size),
    }


def export_tables(data: dict[str, pd.DataFrame | list[str] | float]) -> dict[str, str]:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    pearson_matrix = data["pearson_selected"]  # type: ignore[assignment]
    high_pairs = data["high_pairs"]  # type: ignore[assignment]
    vif = data["vif"]  # type: ignore[assignment]
    tiles = data["tiles"]  # type: ignore[assignment]
    family_counts = data["family_counts"]  # type: ignore[assignment]

    matrix_path = TABLE_DIR / "stage41_v05_selected10_pearson_matrix.csv"
    pearson_matrix.to_csv(matrix_path, encoding="utf-8-sig")
    paths["selected10_pearson_matrix"] = str(matrix_path)

    high_path = TABLE_DIR / "stage41_v05_high_correlation_pairs_top10.csv"
    high_pairs.to_csv(high_path, index=False, encoding="utf-8-sig")
    paths["high_correlation_pairs_top10"] = str(high_path)

    vif_path = TABLE_DIR / "stage41_v05_initial_vif_plot_data.csv"
    vif.to_csv(vif_path, index=False, encoding="utf-8-sig")
    paths["initial_vif_plot_data"] = str(vif_path)

    tile_path = TABLE_DIR / "stage41_v05_predictor_selection_tiles.csv"
    tiles.to_csv(tile_path, index=False, encoding="utf-8-sig")
    paths["predictor_selection_tiles"] = str(tile_path)

    family_path = TABLE_DIR / "stage41_v05_selection_counts_by_family.csv"
    family_counts.to_csv(family_path, index=False, encoding="utf-8-sig")
    paths["selection_counts_by_family"] = str(family_path)
    return paths


def write_readme(outputs: dict[str, object], table_paths: dict[str, str], data: dict[str, pd.DataFrame | list[str] | float]) -> None:
    selected = data["selected"]  # type: ignore[assignment]
    rec = data["recommended"]  # type: ignore[assignment]
    selected_count = int((rec["decision"] == "selected").sum())
    dropped_count = int((rec["decision"] == "dropped").sum())
    lines = [
        f"# {FIG_BASENAME}",
        "",
        "Purpose: supplementary predictor-audit figure for the oasis suitability manuscript.",
        "",
        "Panels:",
        "- a: Pearson correlation matrix for the retained 10 predictors.",
        "- b: strongest Pearson-correlated predictor pairs in the original 20-factor pool.",
        "- c: initial VIF values before iterative screening, with final selected/dropped state.",
        "- d: tile-based screening outcome by predictor family.",
        "",
        "Data boundary:",
        "- Uses completed Stage30 environmental-factor collinearity diagnostics only.",
        "- Does not use future projection, full-grid area, RF future rerun, or land-cover constraint outputs.",
        "",
        f"Selected predictors ({selected_count}): {', '.join(selected)}",
        f"Dropped predictors: {dropped_count}",
        "",
        "Figure outputs:",
        *[f"- {key}: `{value}`" for key, value in outputs.items()],
        "",
        "Plot-data tables:",
        *[f"- {key}: `{value}`" for key, value in table_paths.items()],
        "",
        f"Generated at: {now_iso()}",
        f"Script: `{Path(__file__).resolve()}`",
    ]
    README_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    setup_logging()
    write_status("running", message="stage41 predictor correlation/VIF supplementary figure started")
    try:
        for directory in (STAGE_DIR, FIG_DIR, TABLE_DIR):
            directory.mkdir(parents=True, exist_ok=True)
        logging.info("Preparing Stage30 predictor diagnostics")
        data = prepare_data()
        logging.info("Rendering figure")
        fig = make_figure(data)
        logging.info("Saving outputs")
        outputs = save_outputs(fig)
        table_paths = export_tables(data)
        write_readme(outputs, table_paths, data)
        required = [
            str(outputs[key])
            for key in ("png", "svg", "pdf", "white_preview")
        ] + list(table_paths.values()) + [str(README_MD), str(LOG_PATH)]
        missing = [path for path in required if not Path(path).exists()]
        if missing:
            raise RuntimeError(f"Missing expected outputs after export: {missing}")
        write_status(
            "success",
            message="stage41 predictor correlation/VIF supplementary figure completed",
            outputs=outputs,
            tables=table_paths,
            readme=str(README_MD),
            log=str(LOG_PATH),
            data_boundary="Stage30 collinearity diagnostics only; no future projection or land-cover outputs used.",
            selected_predictor_count=10,
            original_predictor_count=20,
        )
        logging.info("Completed successfully")
    except Exception as exc:
        logging.exception("Stage41 figure generation failed")
        write_status("failed", message=str(exc), log=str(LOG_PATH))
        raise


if __name__ == "__main__":
    main()
