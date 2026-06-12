#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Create Stage48 Fig7 candidate: selected10 tile contribution concentration.

This manuscript candidate is a single ranked-contribution figure. It avoids
another 2x2 panel plate and avoids repeating the map/funnel and area-mosaic
forms already used in nearby main figures.
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
STAGE_DIR = PROJECT_ROOT / "outputs" / "stage48_manuscript_main_figures"
FIG_DIR = STAGE_DIR / "figures"
TABLE_DIR = STAGE_DIR / "tables"
LOG_DIR = STAGE_DIR / "logs"
QC_DIR = STAGE_DIR / "qc"

FIGURE_STEM = "fig_stage48_fig7_selected10_tile_contribution_concentration_v01"
FIGURE_ID = "Fig7_stage48_v01"
LAYOUT_FAMILY = "single-ranked-tile-contribution-concentration"

STAGE17_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "stage34_selected10_constrained_suitability"
    / "tables"
    / "stage17_constrained_suitability_selected10_hgb_main_summary.csv"
)
STAGE20_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "stage34_selected10_landcover_spatial_constraint"
    / "tables"
    / "stage20_landcover_spatial_constraint_selected10_hgb_main_summary.csv"
)
LEDGER_JSON = PROJECT_ROOT / "docs" / "figure_style_ledger.json"


PALETTE = {
    "bar": "#8fc8b6",
    "bar_edge": "#5ba694",
    "bar_top": "#2c827d",
    "curve": "#c77b25",
    "threshold": "#a6b5b8",
    "ink": "#24313d",
    "muted": "#6f7d85",
    "grid": "#e3e9e7",
    "axis": "#cfd9d6",
    "frame": "#d9e2de",
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
    for path in [FIG_DIR, TABLE_DIR, LOG_DIR, QC_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def setup_logging() -> Path:
    log_path = LOG_DIR / f"{FIGURE_STEM}.log"
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root.addHandler(console)
    return log_path


def update_layout_ledger(status: str, notes: str) -> None:
    if LEDGER_JSON.exists():
        ledger = json.loads(LEDGER_JSON.read_text(encoding="utf-8"))
    else:
        ledger = {
            "paper": "03_oasis_future_suitability",
            "purpose": "Track figure layout families so figures in the same paper remain visually differentiated.",
            "entries": [],
        }
    entries = ledger.setdefault("entries", [])
    entry = {
        "figure_id": FIGURE_ID,
        "source": rel(STAGE_DIR),
        "status": status,
        "layout_family": LAYOUT_FAMILY,
        "avoid_repeating_as_default": True,
        "recorded_at": now_iso(),
        "notes": notes,
    }
    for idx, old in enumerate(entries):
        if old.get("figure_id") == FIGURE_ID:
            entries[idx] = entry
            break
    else:
        entries.append(entry)
    write_json_atomic(LEDGER_JSON, ledger)


def init_status(log_path: Path) -> Path:
    status_path = STAGE_DIR / f"{FIGURE_STEM}_status.json"
    payload = {
        "figure_id": FIGURE_ID,
        "figure_stem": FIGURE_STEM,
        "layout_family": LAYOUT_FAMILY,
        "status": "running",
        "started_at": now_iso(),
        "inputs": {
            "stage17": rel(STAGE17_CSV),
            "stage20": rel(STAGE20_CSV),
        },
        "log": rel(log_path),
        "outputs": {},
        "warnings": [],
        "errors": [],
        "word_insertion": "not_inserted_candidate",
    }
    write_json_atomic(status_path, payload)
    return status_path


def update_status(status_path: Path, **kwargs: object) -> None:
    payload = json.loads(status_path.read_text(encoding="utf-8"))
    payload.update(kwargs)
    write_json_atomic(status_path, payload)


def validate_inputs(stage17: pd.DataFrame, stage20: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required17 = {
        "tile_id",
        "status",
        "min_lon",
        "min_lat",
        "max_lon",
        "max_lat",
        "constrained_suitable_area_km2",
    }
    required20 = {
        "tile_id",
        "status",
        "weighted_compatible_area_km2",
        "binary_suitable_area_km2",
        "stage17_suitable_area_km2",
        "excluded_by_landcover_area_km2",
        "weighted_retention_pct",
        "binary_retention_pct",
    }
    missing17 = sorted(required17.difference(stage17.columns))
    missing20 = sorted(required20.difference(stage20.columns))
    if missing17:
        raise ValueError(f"Stage17 missing required columns: {missing17}")
    if missing20:
        raise ValueError(f"Stage20 missing required columns: {missing20}")

    stage17_ok = stage17[stage17["status"].eq("success")].copy()
    stage20_done = stage20[stage20["status"].isin(["success", "skipped"])].copy()
    stage20_failed = stage20[~stage20["status"].isin(["success", "skipped"])].copy()

    if len(stage17_ok) != 171:
        raise ValueError(f"Stage17 expected 171 success rows, found {len(stage17_ok)}")
    if len(stage20_done) != 171 or not stage20_failed.empty:
        raise ValueError(
            "Stage20 expected 171 completed rows and 0 failed rows; "
            f"completed={len(stage20_done)}, failed={len(stage20_failed)}"
        )
    return stage17_ok, stage20_done


def load_plot_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not STAGE17_CSV.exists():
        raise FileNotFoundError(f"Missing Stage17 summary: {STAGE17_CSV}")
    if not STAGE20_CSV.exists():
        raise FileNotFoundError(f"Missing Stage20 summary: {STAGE20_CSV}")

    stage17 = pd.read_csv(STAGE17_CSV)
    stage20 = pd.read_csv(STAGE20_CSV)
    stage17_ok, stage20_done = validate_inputs(stage17, stage20)

    coords = stage17_ok[["tile_id", "min_lon", "min_lat", "max_lon", "max_lat"]].drop_duplicates("tile_id")
    merged = stage20_done.merge(coords, on="tile_id", how="left", validate="one_to_one")
    if merged[["min_lon", "min_lat", "max_lon", "max_lat"]].isna().any().any():
        raise ValueError("Some Stage20 tiles did not match Stage17 coordinates.")

    merged["center_lon"] = (merged["min_lon"] + merged["max_lon"]) / 2.0
    merged["center_lat"] = (merged["min_lat"] + merged["max_lat"]) / 2.0
    merged["retention_pct_undefined"] = (
        merged["weighted_retention_pct"].isna() | merged["binary_retention_pct"].isna()
    )
    merged["weighted_retention_pct"] = merged["weighted_retention_pct"].fillna(0.0)
    merged["binary_retention_pct"] = merged["binary_retention_pct"].fillna(0.0)
    merged = merged.sort_values("weighted_compatible_area_km2", ascending=False).reset_index(drop=True)
    merged["rank"] = np.arange(1, len(merged) + 1)
    total_weighted = float(merged["weighted_compatible_area_km2"].sum())
    if total_weighted <= 0:
        raise ValueError("Total weighted compatible area is zero.")
    merged["area_1000_km2"] = merged["weighted_compatible_area_km2"] / 1000.0
    merged["area_share_pct"] = merged["weighted_compatible_area_km2"] / total_weighted * 100.0
    merged["cumulative_area_km2"] = merged["weighted_compatible_area_km2"].cumsum()
    merged["cumulative_share_pct"] = merged["cumulative_area_km2"] / total_weighted * 100.0

    thresholds = []
    for share in [50, 80, 90, 95]:
        row = merged[merged["cumulative_share_pct"].ge(share)].iloc[0]
        thresholds.append(
            {
                "cumulative_share_threshold_pct": share,
                "rank_reached": int(row["rank"]),
                "actual_cumulative_share_pct": float(row["cumulative_share_pct"]),
                "tile_id_at_threshold": row["tile_id"],
            }
        )
    threshold_df = pd.DataFrame(thresholds)

    summary_df = pd.DataFrame(
        [
            {
                "tile_count": int(len(merged)),
                "nonzero_tile_count": int((merged["weighted_compatible_area_km2"] > 0).sum()),
                "total_weighted_compatible_area_km2": total_weighted,
                "total_binary_suitable_area_km2": float(merged["binary_suitable_area_km2"].sum()),
                "total_stage17_suitable_area_km2": float(merged["stage17_suitable_area_km2"].sum()),
                "total_excluded_by_landcover_area_km2": float(merged["excluded_by_landcover_area_km2"].sum()),
                "top_10_tile_share_pct": float(merged.head(10)["weighted_compatible_area_km2"].sum() / total_weighted * 100.0),
                "top_20_tile_share_pct": float(merged.head(20)["weighted_compatible_area_km2"].sum() / total_weighted * 100.0),
            }
        ]
    )
    return merged, threshold_df, summary_df


def write_plot_tables(plot_df: pd.DataFrame, threshold_df: pd.DataFrame, summary_df: pd.DataFrame) -> dict[str, Path]:
    outputs = {
        "plot_data_csv": TABLE_DIR / f"{FIGURE_STEM}_plot_data.csv",
        "thresholds_csv": TABLE_DIR / f"{FIGURE_STEM}_thresholds.csv",
        "summary_csv": TABLE_DIR / f"{FIGURE_STEM}_summary.csv",
        "top_tiles_csv": TABLE_DIR / f"{FIGURE_STEM}_top_tiles.csv",
    }
    table_cols = [
        "rank",
        "tile_id",
        "center_lon",
        "center_lat",
        "weighted_compatible_area_km2",
        "binary_suitable_area_km2",
        "stage17_suitable_area_km2",
        "excluded_by_landcover_area_km2",
        "weighted_retention_pct",
        "binary_retention_pct",
        "retention_pct_undefined",
        "area_share_pct",
        "cumulative_area_km2",
        "cumulative_share_pct",
    ]
    plot_df[table_cols].to_csv(outputs["plot_data_csv"], index=False, encoding="utf-8-sig")
    threshold_df.to_csv(outputs["thresholds_csv"], index=False, encoding="utf-8-sig")
    summary_df.to_csv(outputs["summary_csv"], index=False, encoding="utf-8-sig")
    plot_df[table_cols].head(30).to_csv(outputs["top_tiles_csv"], index=False, encoding="utf-8-sig")
    return outputs


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.edgecolor": PALETTE["axis"],
            "axes.linewidth": 0.8,
            "axes.facecolor": "white",
            "figure.facecolor": "none",
            "savefig.dpi": 600,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def draw_figure(plot_df: pd.DataFrame, threshold_df: pd.DataFrame, summary_df: pd.DataFrame) -> plt.Figure:
    set_style()
    fig, ax = plt.subplots(figsize=(7.7, 4.25), constrained_layout=False)
    fig.subplots_adjust(left=0.075, right=0.915, top=0.95, bottom=0.16)

    x = plot_df["rank"].to_numpy()
    y = plot_df["area_1000_km2"].to_numpy()
    cum = plot_df["cumulative_share_pct"].to_numpy()
    ranks = threshold_df.set_index("cumulative_share_threshold_pct")["rank_reached"].to_dict()

    norm = mpl.colors.Normalize(vmin=float(y.min()), vmax=float(y.max()))
    cmap = mpl.colors.LinearSegmentedColormap.from_list("tile_area_teal", ["#d8eee6", PALETTE["bar"], PALETTE["bar_top"]])
    colors = cmap(norm(y))
    ax.bar(x, y, width=0.88, color=colors, edgecolor="none", zorder=2, label="Tile area")

    ax2 = ax.twinx()
    ax2.plot(x, cum, color=PALETTE["curve"], lw=2.2, zorder=5, label="Cumulative share")
    ax2.scatter([x[0], x[9], x[19], x[-1]], [cum[0], cum[9], cum[19], cum[-1]], s=18, color=PALETTE["curve"], edgecolor="white", linewidth=0.6, zorder=6)

    ax.set_xlim(0, len(plot_df) + 2)
    ax.set_ylim(0, max(y) * 1.18)
    ax2.set_ylim(0, 114)

    ax.set_xlabel("Ranked dryland tiles")
    ax.set_ylabel("Area per tile (10$^3$ km$^2$)")
    ax2.set_ylabel("Cumulative share of final area (%)", color=PALETTE["curve"])
    ax2.tick_params(axis="y", colors=PALETTE["curve"])

    xticks = [1, 10, 20, 30, 50, 75, 100, 125, 150, 171]
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(v) for v in xticks])
    ax.grid(axis="y", color=PALETTE["grid"], lw=0.8, zorder=1)
    ax.grid(axis="x", visible=False)

    for spine in ax.spines.values():
        spine.set_color(PALETTE["frame"])
    for spine in ax2.spines.values():
        spine.set_color(PALETTE["frame"])

    for share, rank in ranks.items():
        rank = int(rank)
        ax2.axvline(rank, ymin=0.02, ymax=0.84, color=PALETTE["threshold"], lw=0.9, ls=(0, (3, 3)), zorder=3)

    top_10_share = float(summary_df.loc[0, "top_10_tile_share_pct"])
    total_mkm2 = float(summary_df.loc[0, "total_weighted_compatible_area_km2"]) / 1_000_000.0
    nonzero = int(summary_df.loc[0, "nonzero_tile_count"])
    milestone_text = "\n".join(
        f"{int(row.cumulative_share_threshold_pct)}%: {int(row.rank_reached)} tiles"
        for row in threshold_df.itertuples(index=False)
    )
    ax.text(
        0.56,
        0.55,
        "Milestones\n" + milestone_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        color=PALETTE["muted"],
        fontsize=7.6,
        linespacing=1.22,
        bbox={"facecolor": "white", "edgecolor": PALETTE["frame"], "linewidth": 0.4, "alpha": 0.86, "pad": 3.2},
        zorder=8,
    )
    ax.text(
        0.56,
        0.32,
        f"Final area: {total_mkm2:.2f} million km$^2$\nTop 10 tiles: {top_10_share:.1f}%\nNon-zero tiles: {nonzero}/171",
        transform=ax.transAxes,
        ha="left",
        va="top",
        color=PALETTE["ink"],
        fontsize=7.8,
        linespacing=1.25,
        bbox={"facecolor": "white", "edgecolor": PALETTE["frame"], "linewidth": 0.4, "alpha": 0.86, "pad": 3.2},
        zorder=8,
    )

    handles = [
        mpl.patches.Patch(facecolor=PALETTE["bar"], edgecolor="none", label="Ranked tile area"),
        mpl.lines.Line2D([0], [0], color=PALETTE["curve"], lw=2.2, label="Cumulative share"),
    ]
    legend = ax.legend(
        handles=handles,
        loc="lower right",
        frameon=False,
        handlelength=1.7,
        borderaxespad=0.9,
    )
    for text in legend.get_texts():
        text.set_color(PALETTE["ink"])

    ax.tick_params(colors=PALETTE["ink"])
    ax.yaxis.label.set_color(PALETTE["ink"])
    ax.xaxis.label.set_color(PALETTE["ink"])
    ax2.yaxis.label.set_color(PALETTE["curve"])
    return fig


def export_figure(fig: plt.Figure) -> dict[str, Path]:
    png = FIG_DIR / f"{FIGURE_STEM}.png"
    svg = FIG_DIR / f"{FIGURE_STEM}.svg"
    pdf = FIG_DIR / f"{FIGURE_STEM}.pdf"
    preview = FIG_DIR / f"{FIGURE_STEM}_white_preview.png"
    export_pad = 0.055

    fig.savefig(png, dpi=600, bbox_inches="tight", pad_inches=export_pad, transparent=True)
    fig.savefig(svg, bbox_inches="tight", pad_inches=export_pad, transparent=True)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=export_pad, transparent=True)
    fig.savefig(preview, dpi=300, bbox_inches="tight", pad_inches=export_pad, facecolor="white", transparent=False)
    plt.close(fig)

    with Image.open(png) as img:
        has_alpha = img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info)
        if not has_alpha:
            raise ValueError(f"Transparent PNG check failed: {png}")

    return {"png": png, "svg": svg, "pdf": pdf, "white_preview_png": preview}


def write_readme(plot_outputs: dict[str, Path], figure_outputs: dict[str, Path], summary_df: pd.DataFrame, threshold_df: pd.DataFrame) -> Path:
    readme = STAGE_DIR / "README.md"
    total_mkm2 = float(summary_df.loc[0, "total_weighted_compatible_area_km2"]) / 1_000_000.0
    top10 = float(summary_df.loc[0, "top_10_tile_share_pct"])
    undefined_retention = int(pd.read_csv(plot_outputs["plot_data_csv"])["retention_pct_undefined"].sum())
    threshold_lines = "\n".join(
        f"- {int(row.cumulative_share_threshold_pct)}% cumulative area: rank {int(row.rank_reached)} "
        f"({row.actual_cumulative_share_pct:.2f}%)."
        for row in threshold_df.itertuples(index=False)
    )
    text = f"""# Stage48 Fig7 candidate

Figure stem: `{FIGURE_STEM}`

Figure id: `{FIGURE_ID}`

Layout family: `{LAYOUT_FAMILY}`

Candidate status: not inserted into Word. Insert only after explicit user confirmation and add the below-figure Word caption/title at that time.

## Scientific message

This single ranked-contribution figure summarizes how concentrated the final selected10 HGB land-cover constrained suitable area is across the 171 dryland tiles. It is intentionally not a 2x2 plate.

## Inputs

- `{rel(STAGE17_CSV)}`
- `{rel(STAGE20_CSV)}`

## Key values

- Final land-cover weighted compatible area: {total_mkm2:.3f} million km2.
- Top 10 tile contribution: {top10:.2f}%.
- Non-zero contributing tiles: {int(summary_df.loc[0, "nonzero_tile_count"])} of {int(summary_df.loc[0, "tile_count"])}.
- Retention percentages are undefined for {undefined_retention} zero-stage17-suitable tiles; plot-data tables flag them in `retention_pct_undefined` and encode the display value as 0.

## Cumulative thresholds

{threshold_lines}

## Outputs

- Transparent PNG: `{rel(figure_outputs["png"])}`
- SVG: `{rel(figure_outputs["svg"])}`
- PDF: `{rel(figure_outputs["pdf"])}`
- White preview: `{rel(figure_outputs["white_preview_png"])}`
- Plot data: `{rel(plot_outputs["plot_data_csv"])}`
- Threshold table: `{rel(plot_outputs["thresholds_csv"])}`
- Summary table: `{rel(plot_outputs["summary_csv"])}`
- Top tiles: `{rel(plot_outputs["top_tiles_csv"])}`
"""
    readme.write_text(text, encoding="utf-8")
    return readme


def main() -> int:
    setup_dirs()
    log_path = setup_logging()
    status_path = init_status(log_path)
    try:
        logging.info("Loading selected10 Stage17 and Stage20 summaries.")
        plot_df, threshold_df, summary_df = load_plot_data()
        plot_outputs = write_plot_tables(plot_df, threshold_df, summary_df)
        logging.info("Wrote plot-data tables.")

        fig = draw_figure(plot_df, threshold_df, summary_df)
        figure_outputs = export_figure(fig)
        logging.info("Exported figure package.")

        readme = write_readme(plot_outputs, figure_outputs, summary_df, threshold_df)
        update_layout_ledger(
            status="candidate_generated_local_pending_review",
            notes=(
                "New non-2x2 single ranked tile-contribution figure. "
                "It uses selected10 Stage20 land-cover constrained area and remains outside Word until user confirmation."
            ),
        )
        update_status(
            status_path,
            status="success",
            finished_at=now_iso(),
            outputs={k: rel(v) for k, v in {**plot_outputs, **figure_outputs, "readme": readme}.items()},
            key_values=summary_df.iloc[0].to_dict(),
            threshold_values=threshold_df.to_dict(orient="records"),
        )
        print(json.dumps({"status": "success", "figure": rel(figure_outputs["white_preview_png"])}, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:  # noqa: BLE001
        logging.error("Failed to create Stage48 Fig7 candidate: %s", exc)
        logging.error(traceback.format_exc())
        update_status(
            status_path,
            status="failed",
            finished_at=now_iso(),
            errors=[{"message": str(exc), "traceback": traceback.format_exc()}],
        )
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
