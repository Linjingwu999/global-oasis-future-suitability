#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Create Stage50 Fig7 candidate: q10 tile contribution concentration.

This q10 replacement figure uses the completed selected10 HGB main-chain
HydroRIVERS >=10 m3/s land-cover constrained tile table. It is intentionally a
single cumulative concentration curve with a ranked-tile rug so the manuscript
does not repeat the earlier 2x2, slopegraph, waterfall, or ranked-bar forms.
"""

from __future__ import annotations

import json
import logging
import traceback
from datetime import datetime
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGE_DIR = PROJECT_ROOT / "outputs" / "stage50_ijaeog_q10_figure_updates"
FIG_DIR = STAGE_DIR / "figures"
TABLE_DIR = STAGE_DIR / "tables"
LOG_DIR = STAGE_DIR / "logs"
QC_DIR = STAGE_DIR / "qc"

FIGURE_STEM = "fig_stage50_fig7_q10_tile_contribution_concentration_v03"
FIGURE_ID = "Fig7_stage50_v03"
LAYOUT_FAMILY = "single-cumulative-area-concentration-curve-with-tile-rug"

Q10_TILE_SUMMARY_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "stage36_hydrology_landcover_sensitivity"
    / "q10cms"
    / "tables"
    / "stage20_landcover_spatial_constraint_selected10_hgb_hydrorivers_q10cms_landcover_summary.csv"
)
Q10_INPUT_CHECK_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "stage36_hydrology_landcover_sensitivity"
    / "q10cms"
    / "tables"
    / "stage20_input_file_check.csv"
)
LEDGER_JSON = PROJECT_ROOT / "docs" / "figure_style_ledger.json"

STATUS_JSON = STAGE_DIR / f"{FIGURE_STEM}_status.json"
README_MD = STAGE_DIR / f"README_{FIGURE_STEM}.md"

PALETTE = {
    "curve": "#284E77",
    "curve_light": "#DCE9F2",
    "equality": "#A8B4BC",
    "top10": "#E5B84D",
    "top20": "#2D827D",
    "milestone": "#B55F5F",
    "rug_low": "#CFE4DF",
    "rug_high": "#255F73",
    "ink": "#26313D",
    "muted": "#6E7B85",
    "grid": "#E3EAE7",
    "frame": "#D7E0DC",
}


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def setup_dirs() -> None:
    for path in [STAGE_DIR, FIG_DIR, TABLE_DIR, LOG_DIR, QC_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def setup_logging() -> Path:
    log_path = LOG_DIR / f"{FIGURE_STEM}.log"
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root.addHandler(console)
    return log_path


def update_layout_ledger(status: str, notes: str, reviewed: dict | None = None) -> None:
    if LEDGER_JSON.exists():
        ledger = json.loads(LEDGER_JSON.read_text(encoding="utf-8"))
    else:
        ledger = {
            "paper": "03_oasis_future_suitability",
            "purpose": "Track figure layout families so figures in the same paper remain visually differentiated.",
            "entries": [],
        }

    entry = {
        "figure_id": FIGURE_ID,
        "source": rel(STAGE_DIR),
        "status": status,
        "layout_family": LAYOUT_FAMILY,
        "avoid_repeating_as_default": True,
        "recorded_at": now_iso(),
        "notes": notes,
    }
    if reviewed:
        entry.update(reviewed)

    entries = ledger.setdefault("entries", [])
    for idx, old in enumerate(entries):
        if old.get("figure_id") == FIGURE_ID:
            entries[idx] = entry
            break
    else:
        entries.append(entry)
    write_json_atomic(LEDGER_JSON, ledger)


def init_status(log_path: Path) -> None:
    payload = {
        "figure_id": FIGURE_ID,
        "figure_stem": FIGURE_STEM,
        "layout_family": LAYOUT_FAMILY,
        "status": "running",
        "started_at": now_iso(),
        "inputs": {
            "q10_tile_summary": rel(Q10_TILE_SUMMARY_CSV),
            "q10_input_check": rel(Q10_INPUT_CHECK_CSV),
        },
        "log": rel(log_path),
        "outputs": {},
        "warnings": [],
        "errors": [],
        "data_notes": [
            "Rows with status='skipped' are checkpointed Stage20 rows whose outputs already existed; they are treated as completed when paired with input existence checks.",
            "The candidate figure is not inserted into Word and requires explicit user confirmation before manuscript insertion.",
        ],
        "word_insertion_allowed": False,
        "word_insertion": "not_inserted_candidate",
        "user_confirmed_final": False,
    }
    write_json_atomic(STATUS_JSON, payload)


def update_status(**kwargs: object) -> None:
    payload = json.loads(STATUS_JSON.read_text(encoding="utf-8"))
    payload.update(kwargs)
    write_json_atomic(STATUS_JSON, payload)


def format_lon(value: float) -> str:
    value = float(value)
    if value == 0:
        return "0"
    suffix = "E" if value > 0 else "W"
    return f"{abs(value):.0f}{suffix}"


def format_lat(value: float) -> str:
    value = float(value)
    if value == 0:
        return "0"
    suffix = "N" if value > 0 else "S"
    return f"{abs(value):.0f}{suffix}"


def tile_label(tile_id: str) -> str:
    stem = str(tile_id).replace("dryland_", "")
    try:
        min_lon, min_lat, max_lon, max_lat = [float(part) for part in stem.split("_")]
    except Exception:
        return str(tile_id)
    return f"{format_lon(min_lon)}-{format_lon(max_lon)}, {format_lat(min_lat)}-{format_lat(max_lat)}"


def validate_inputs(tile_df: pd.DataFrame, input_check: pd.DataFrame) -> dict:
    required = {
        "tile_id",
        "status",
        "weighted_compatible_area_km2",
        "binary_suitable_area_km2",
        "stage17_suitable_area_km2",
        "excluded_by_landcover_area_km2",
        "weighted_retention_pct",
        "binary_retention_pct",
    }
    missing = sorted(required.difference(tile_df.columns))
    if missing:
        raise ValueError(f"q10 tile summary missing required columns: {missing}")

    if "tile_id" not in input_check.columns:
        raise ValueError("q10 input check is missing tile_id column.")
    for col in ["stage17_exists", "landcover_exists"]:
        if col not in input_check.columns:
            raise ValueError(f"q10 input check is missing {col} column.")

    row_count = int(len(tile_df))
    if row_count != 171:
        raise ValueError(f"Expected 171 q10 tile rows, found {row_count}.")

    allowed_status = {"success", "skipped"}
    bad_status = sorted(set(tile_df["status"].astype(str)) - allowed_status)
    if bad_status:
        raise ValueError(f"q10 tile summary contains non-completed statuses: {bad_status}")

    if input_check["stage17_exists"].astype(bool).sum() != 171:
        raise ValueError("Not all q10 Stage17 input rasters exist according to input check.")
    if input_check["landcover_exists"].astype(bool).sum() != 171:
        raise ValueError("Not all q10 land-cover input rasters exist according to input check.")

    if set(tile_df["tile_id"]) != set(input_check["tile_id"]):
        raise ValueError("q10 tile summary and input check tile IDs do not match.")

    area_cols = [
        "weighted_compatible_area_km2",
        "binary_suitable_area_km2",
        "stage17_suitable_area_km2",
        "excluded_by_landcover_area_km2",
    ]
    for col in area_cols:
        values = pd.to_numeric(tile_df[col], errors="coerce")
        if values.isna().any():
            raise ValueError(f"q10 tile summary contains non-numeric values in {col}.")
        if (values < -1e-9).any():
            raise ValueError(f"q10 tile summary contains negative area values in {col}.")

    status_counts = tile_df["status"].value_counts(dropna=False).rename_axis("status").reset_index(name="rows")
    return {
        "row_count": row_count,
        "status_counts": status_counts,
        "skipped_rows": int((tile_df["status"].astype(str) == "skipped").sum()),
        "success_rows": int((tile_df["status"].astype(str) == "success").sum()),
    }


def load_plot_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    if not Q10_TILE_SUMMARY_CSV.exists():
        raise FileNotFoundError(f"Missing q10 tile summary: {Q10_TILE_SUMMARY_CSV}")
    if not Q10_INPUT_CHECK_CSV.exists():
        raise FileNotFoundError(f"Missing q10 input check: {Q10_INPUT_CHECK_CSV}")

    tile_df = pd.read_csv(Q10_TILE_SUMMARY_CSV)
    input_check = pd.read_csv(Q10_INPUT_CHECK_CSV)
    validation = validate_inputs(tile_df, input_check)

    work = tile_df.copy()
    numeric_cols = [
        "weighted_compatible_area_km2",
        "binary_suitable_area_km2",
        "stage17_suitable_area_km2",
        "excluded_by_landcover_area_km2",
        "weighted_retention_pct",
        "binary_retention_pct",
    ]
    for col in numeric_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    work["retention_pct_undefined"] = work["weighted_retention_pct"].isna() | work["binary_retention_pct"].isna()
    work["weighted_retention_pct"] = work["weighted_retention_pct"].fillna(0.0)
    work["binary_retention_pct"] = work["binary_retention_pct"].fillna(0.0)
    work["tile_label"] = work["tile_id"].map(tile_label)
    work = work.sort_values("weighted_compatible_area_km2", ascending=False).reset_index(drop=True)

    work["rank"] = np.arange(1, len(work) + 1)
    work["ranked_tile_share_pct"] = work["rank"] / len(work) * 100.0
    total_weighted = float(work["weighted_compatible_area_km2"].sum())
    if total_weighted <= 0:
        raise ValueError("Total q10 weighted compatible area is zero.")

    work["area_share_pct"] = work["weighted_compatible_area_km2"] / total_weighted * 100.0
    work["cumulative_area_km2"] = work["weighted_compatible_area_km2"].cumsum()
    work["cumulative_area_share_pct"] = work["cumulative_area_km2"] / total_weighted * 100.0
    work["area_1000_km2"] = work["weighted_compatible_area_km2"] / 1000.0

    thresholds = []
    for threshold in [50, 80, 90, 95]:
        row = work.loc[work["cumulative_area_share_pct"].ge(threshold)].iloc[0]
        thresholds.append(
            {
                "cumulative_area_share_threshold_pct": threshold,
                "rank_reached": int(row["rank"]),
                "ranked_tile_share_pct": float(row["ranked_tile_share_pct"]),
                "actual_cumulative_area_share_pct": float(row["cumulative_area_share_pct"]),
                "tile_id_at_threshold": row["tile_id"],
                "tile_label_at_threshold": row["tile_label"],
            }
        )
    threshold_df = pd.DataFrame(thresholds)

    top10_share = float(work.head(10)["weighted_compatible_area_km2"].sum() / total_weighted * 100.0)
    top20_share = float(work.head(20)["weighted_compatible_area_km2"].sum() / total_weighted * 100.0)
    summary_df = pd.DataFrame(
        [
            {
                "figure_id": FIGURE_ID,
                "tile_count": int(len(work)),
                "nonzero_tile_count": int((work["weighted_compatible_area_km2"] > 0).sum()),
                "total_weighted_compatible_area_km2": total_weighted,
                "total_weighted_compatible_area_million_km2": total_weighted / 1_000_000.0,
                "total_binary_suitable_area_km2": float(work["binary_suitable_area_km2"].sum()),
                "total_stage17_suitable_area_km2": float(work["stage17_suitable_area_km2"].sum()),
                "total_excluded_by_landcover_area_km2": float(work["excluded_by_landcover_area_km2"].sum()),
                "top_10_tile_share_pct": top10_share,
                "top_20_tile_share_pct": top20_share,
                "rows_with_status_success": validation["success_rows"],
                "rows_with_status_skipped": validation["skipped_rows"],
                "stage17_inputs_exist": int(input_check["stage17_exists"].astype(bool).sum()),
                "landcover_inputs_exist": int(input_check["landcover_exists"].astype(bool).sum()),
            }
        ]
    )

    return work, threshold_df, summary_df, validation["status_counts"], validation


def write_plot_tables(
    plot_df: pd.DataFrame,
    threshold_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    status_counts: pd.DataFrame,
) -> dict[str, Path]:
    paths = {
        "plot_data": TABLE_DIR / f"{FIGURE_STEM}_plot_data.csv",
        "thresholds": TABLE_DIR / f"{FIGURE_STEM}_threshold_ranks.csv",
        "summary": TABLE_DIR / f"{FIGURE_STEM}_summary.csv",
        "top20_tiles": TABLE_DIR / f"{FIGURE_STEM}_top20_tiles.csv",
        "status_counts": TABLE_DIR / f"{FIGURE_STEM}_input_status_counts.csv",
    }
    plot_cols = [
        "tile_id",
        "tile_label",
        "status",
        "rank",
        "ranked_tile_share_pct",
        "weighted_compatible_area_km2",
        "area_1000_km2",
        "area_share_pct",
        "cumulative_area_km2",
        "cumulative_area_share_pct",
        "binary_suitable_area_km2",
        "stage17_suitable_area_km2",
        "excluded_by_landcover_area_km2",
        "weighted_retention_pct",
        "binary_retention_pct",
        "retention_pct_undefined",
    ]
    plot_df[plot_cols].to_csv(paths["plot_data"], index=False, encoding="utf-8-sig")
    threshold_df.to_csv(paths["thresholds"], index=False, encoding="utf-8-sig")
    summary_df.to_csv(paths["summary"], index=False, encoding="utf-8-sig")
    plot_df[plot_cols].head(20).to_csv(paths["top20_tiles"], index=False, encoding="utf-8-sig")
    status_counts.to_csv(paths["status_counts"], index=False, encoding="utf-8-sig")
    return paths


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.6,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.4,
            "ytick.labelsize": 8.4,
            "figure.dpi": 160,
            "savefig.dpi": 600,
            "axes.linewidth": 0.8,
            "axes.edgecolor": PALETTE["frame"],
            "xtick.color": PALETTE["ink"],
            "ytick.color": PALETTE["ink"],
            "text.color": PALETTE["ink"],
            "axes.labelcolor": PALETTE["ink"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def draw_figure(plot_df: pd.DataFrame, threshold_df: pd.DataFrame, summary_df: pd.DataFrame) -> dict[str, Path]:
    configure_style()
    paths = {
        "png": FIG_DIR / f"{FIGURE_STEM}.png",
        "svg": FIG_DIR / f"{FIGURE_STEM}.svg",
        "pdf": FIG_DIR / f"{FIGURE_STEM}.pdf",
        "white_preview": FIG_DIR / f"{FIGURE_STEM}_white_preview.png",
    }

    fig, ax = plt.subplots(figsize=(7.2, 4.55))
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("white")

    x = plot_df["ranked_tile_share_pct"].to_numpy()
    y = plot_df["cumulative_area_share_pct"].to_numpy()
    equality = x.copy()

    top10_x = float(plot_df.loc[plot_df["rank"].eq(10), "ranked_tile_share_pct"].iloc[0])
    top20_x = float(plot_df.loc[plot_df["rank"].eq(20), "ranked_tile_share_pct"].iloc[0])
    top10_y = float(plot_df.loc[plot_df["rank"].eq(10), "cumulative_area_share_pct"].iloc[0])
    top20_y = float(plot_df.loc[plot_df["rank"].eq(20), "cumulative_area_share_pct"].iloc[0])

    ax.axvspan(0, top10_x, color=PALETTE["top10"], alpha=0.08, lw=0)
    ax.axvspan(top10_x, top20_x, color=PALETTE["top20"], alpha=0.06, lw=0)
    ax.fill_between(x, equality, y, where=y >= equality, color=PALETTE["curve_light"], alpha=0.75, zorder=1)
    ax.plot([0, 100], [0, 100], color=PALETTE["equality"], lw=1.1, ls=(0, (3, 3)), zorder=2)
    ax.plot(x, y, color=PALETTE["curve"], lw=2.8, zorder=5)

    ax.scatter([top10_x, top20_x], [top10_y, top20_y], s=[44, 44], color=[PALETTE["top10"], PALETTE["top20"]], edgecolor="white", linewidth=0.8, zorder=7)
    ax.vlines([top10_x, top20_x], ymin=0, ymax=[top10_y, top20_y], colors=[PALETTE["top10"], PALETTE["top20"]], linestyles=(0, (3, 3)), lw=1.2, zorder=4)

    milestone_df = threshold_df[threshold_df["cumulative_area_share_threshold_pct"].isin([50, 80, 90])].copy()
    ax.scatter(
        milestone_df["ranked_tile_share_pct"],
        milestone_df["actual_cumulative_area_share_pct"],
        s=30,
        color=PALETTE["milestone"],
        edgecolor="white",
        linewidth=0.6,
        zorder=8,
    )

    # The rug encodes individual ranked tile contributions without repeating a
    # ranked bar chart. Zero-area tiles remain nearly invisible at the far right.
    shares = plot_df["area_share_pct"].to_numpy()
    max_share = max(float(np.nanmax(shares)), 1e-9)
    norm = np.sqrt(np.clip(shares / max_share, 0, 1))
    cmap = mpl.colors.LinearSegmentedColormap.from_list("tile_rug", [PALETTE["rug_low"], PALETTE["rug_high"]])
    rug_colors = [cmap(v) for v in norm]
    ax.vlines(x, ymin=-4.2, ymax=-2.2, colors=rug_colors, lw=0.9, alpha=0.92, zorder=3)

    total_million = float(summary_df["total_weighted_compatible_area_million_km2"].iloc[0])
    nonzero = int(summary_df["nonzero_tile_count"].iloc[0])
    tile_count = int(summary_df["tile_count"].iloc[0])
    top10_share = float(summary_df["top_10_tile_share_pct"].iloc[0])
    top20_share = float(summary_df["top_20_tile_share_pct"].iloc[0])
    rank50 = int(threshold_df.loc[threshold_df["cumulative_area_share_threshold_pct"].eq(50), "rank_reached"].iloc[0])
    rank80 = int(threshold_df.loc[threshold_df["cumulative_area_share_threshold_pct"].eq(80), "rank_reached"].iloc[0])
    rank90 = int(threshold_df.loc[threshold_df["cumulative_area_share_threshold_pct"].eq(90), "rank_reached"].iloc[0])

    ax.annotate(
        f"Top 10 tiles: {top10_share:.1f}%",
        xy=(top10_x, top10_y),
        xytext=(17.5, 70.5),
        arrowprops=dict(arrowstyle="-", color=PALETTE["top10"], lw=1.1, shrinkA=3, shrinkB=3),
        fontsize=8.4,
        color="#8F6B19",
        ha="left",
        va="center",
        zorder=9,
    )
    ax.annotate(
        f"Top 20: {top20_share:.1f}%",
        xy=(top20_x, top20_y),
        xytext=(26.5, 89.6),
        arrowprops=dict(arrowstyle="-", color=PALETTE["top20"], lw=1.1, shrinkA=3, shrinkB=3),
        fontsize=8.4,
        color=PALETTE["top20"],
        ha="left",
        va="center",
        zorder=9,
    )

    summary_text = (
        f"q10 weighted area: {total_million:.3f} million km$^2$\n"
        f"50% area reached by {rank50} tiles\n"
        f"80% by {rank80}; 90% by {rank90}\n"
        f"Non-zero tiles: {nonzero} of {tile_count}"
    )
    ax.text(
        55.5,
        31.0,
        summary_text,
        ha="left",
        va="center",
        fontsize=8.2,
        linespacing=1.25,
        color=PALETTE["ink"],
        bbox=dict(boxstyle="round,pad=0.35,rounding_size=0.8", facecolor="white", edgecolor=PALETTE["frame"], linewidth=0.9),
        zorder=10,
    )
    ax.text(
        64.0,
        62.0,
        "equal contribution line",
        ha="left",
        va="center",
        fontsize=7.8,
        color=PALETTE["muted"],
        rotation=37,
        zorder=6,
    )
    ax.set_xlim(0, 100)
    ax.set_ylim(-5.5, 106.0)
    ax.set_xlabel("Ranked dryland tile share (%)")
    ax.set_ylabel("Cumulative q10 compatible area share (%)")
    ax.set_xticks(np.arange(0, 101, 20))
    ax.set_yticks(np.arange(0, 101, 20))
    ax.grid(axis="both", color=PALETTE["grid"], lw=0.8, zorder=0)
    ax.tick_params(length=3.5, width=0.8)
    for spine in ax.spines.values():
        spine.set_color(PALETTE["frame"])
        spine.set_linewidth(0.9)

    fig.subplots_adjust(left=0.105, right=0.982, bottom=0.16, top=0.965)
    fig.savefig(paths["png"], dpi=600, bbox_inches="tight", pad_inches=0.03, transparent=True)
    fig.savefig(paths["svg"], bbox_inches="tight", pad_inches=0.03, transparent=True)
    fig.savefig(paths["pdf"], bbox_inches="tight", pad_inches=0.03, transparent=True)
    plt.close(fig)

    with Image.open(paths["png"]) as image:
        if image.mode not in ("RGBA", "LA"):
            image = image.convert("RGBA")
        background = Image.new("RGBA", image.size, "WHITE")
        background.alpha_composite(image)
        background.convert("RGB").save(paths["white_preview"], quality=95)

    with Image.open(paths["png"]) as image:
        has_alpha = image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info)
    if not has_alpha:
        raise ValueError(f"Exported PNG lacks alpha channel: {paths['png']}")

    return paths


def write_readme(summary_df: pd.DataFrame, threshold_df: pd.DataFrame, table_paths: dict[str, Path], figure_paths: dict[str, Path]) -> None:
    summary = summary_df.iloc[0]
    rank50 = int(threshold_df.loc[threshold_df["cumulative_area_share_threshold_pct"].eq(50), "rank_reached"].iloc[0])
    rank80 = int(threshold_df.loc[threshold_df["cumulative_area_share_threshold_pct"].eq(80), "rank_reached"].iloc[0])
    rank90 = int(threshold_df.loc[threshold_df["cumulative_area_share_threshold_pct"].eq(90), "rank_reached"].iloc[0])
    text = f"""# {FIGURE_ID} q10 tile contribution concentration candidate

Status: success candidate, not inserted into Word.

This figure replaces the older broad-reference tile concentration concept with the selected10 HGB main-chain HydroRIVERS >=10 m3/s q10 result. It remains outside the manuscript Word file until the user explicitly confirms the figure as final.

## Data lineage

- q10 tile summary: `{rel(Q10_TILE_SUMMARY_CSV)}`
- q10 input check: `{rel(Q10_INPUT_CHECK_CSV)}`
- Stage20 row statuses include `{int(summary['rows_with_status_skipped'])}` skipped rows and `{int(summary['rows_with_status_success'])}` success rows. In this project run, `skipped` means the resumable Stage20 task found existing completed outputs; it is not treated as unfinished. The input check confirms 171/171 Stage17 rasters and 171/171 land-cover rasters exist.

## Main values shown

- Total q10 weighted compatible area: {float(summary['total_weighted_compatible_area_million_km2']):.3f} million km2.
- Top 10 ranked tiles: {float(summary['top_10_tile_share_pct']):.1f}% of the final q10 weighted compatible area.
- Top 20 ranked tiles: {float(summary['top_20_tile_share_pct']):.1f}% of the final q10 weighted compatible area.
- 50% area threshold reached by {rank50} tiles; 80% by {rank80} tiles; 90% by {rank90} tiles.

## Figure package

- PNG: `{rel(figure_paths['png'])}`
- SVG: `{rel(figure_paths['svg'])}`
- PDF: `{rel(figure_paths['pdf'])}`
- White preview: `{rel(figure_paths['white_preview'])}`
- Plot data: `{rel(table_paths['plot_data'])}`
- Threshold table: `{rel(table_paths['thresholds'])}`
- Summary table: `{rel(table_paths['summary'])}`
- Top 20 tiles: `{rel(table_paths['top20_tiles'])}`
- Input status counts: `{rel(table_paths['status_counts'])}`

## Layout rule

Layout family: `{LAYOUT_FAMILY}`. This is a single cumulative contribution curve with a ranked-tile rug, selected to avoid repeating the 2x2, threshold slopegraph, waterfall, and older ranked-bar figure rhythms.
"""
    README_MD.write_text(text, encoding="utf-8")


def main() -> int:
    setup_dirs()
    log_path = setup_logging()
    init_status(log_path)
    update_layout_ledger(
        "candidate_generated_not_inserted",
        "q10 tile contribution concentration v01 generated as a single cumulative curve with tile rug; candidate is not inserted into Word.",
    )

    try:
        logging.info("Loading q10 tile data.")
        plot_df, threshold_df, summary_df, status_counts, validation = load_plot_data()
        table_paths = write_plot_tables(plot_df, threshold_df, summary_df, status_counts)
        logging.info("Plot data tables written.")
        figure_paths = draw_figure(plot_df, threshold_df, summary_df)
        logging.info("Figure exports written.")
        write_readme(summary_df, threshold_df, table_paths, figure_paths)

        outputs = {
            "figures": {key: rel(path) for key, path in figure_paths.items()},
            "tables": {key: rel(path) for key, path in table_paths.items()},
            "readme": rel(README_MD),
        }
        update_status(
            status="success",
            finished_at=now_iso(),
            outputs=outputs,
            validation={
                "row_count": validation["row_count"],
                "success_rows": validation["success_rows"],
                "skipped_rows": validation["skipped_rows"],
                "input_status_note": "Skipped rows are documented checkpointed outputs, not failed or unfinished rows.",
            },
            summary=summary_df.iloc[0].to_dict(),
            threshold_ranks=threshold_df.to_dict(orient="records"),
            word_insertion_allowed=False,
            word_insertion="not_inserted_candidate",
            user_confirmed_final=False,
        )
        logging.info("Completed %s.", FIGURE_STEM)
        return 0
    except Exception as exc:
        logging.error("Failed to create %s: %s", FIGURE_STEM, exc)
        logging.error(traceback.format_exc())
        update_status(
            status="failed",
            finished_at=now_iso(),
            errors=[{"message": str(exc), "traceback": traceback.format_exc()}],
            word_insertion_allowed=False,
            user_confirmed_final=False,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
