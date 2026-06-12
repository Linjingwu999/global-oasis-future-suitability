#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Create Stage46 Fig5 candidate: selected10 land-cover constrained footprint.

This figure deliberately uses a single global tile map plus a compact area
cascade so it does not repeat the existing 2x2 statistical-plate rhythm.
"""

from __future__ import annotations

import json
import logging
import traceback
from datetime import datetime
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Patch, Polygon, Rectangle
from matplotlib.cm import ScalarMappable
import numpy as np
import pandas as pd
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGE_DIR = PROJECT_ROOT / "outputs" / "stage46_manuscript_main_figures"
FIG_DIR = STAGE_DIR / "figures"
TABLE_DIR = STAGE_DIR / "tables"
LOG_DIR = STAGE_DIR / "logs"
QC_DIR = STAGE_DIR / "qc"

FIGURE_STEM = "fig_stage46_fig5_selected10_landcover_constraint_footprint_v01"
FIGURE_ID = "Fig5_stage46_v01"
LAYOUT_FAMILY = "single-global-tile-map-with-area-funnel"

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
WORLD_SHP = (
    PROJECT_ROOT
    / "outputs"
    / "stage28_stage20_landcover_distribution_map"
    / "reference"
    / "ne_110m_admin_0_countries"
    / "ne_110m_admin_0_countries.shp"
)
LEDGER_JSON = PROJECT_ROOT / "docs" / "figure_style_ledger.json"


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
    logging.basicConfig(
        filename=log_path,
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        encoding="utf-8",
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logging.getLogger().addHandler(console)
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


def load_world():
    try:
        import geopandas as gpd

        if WORLD_SHP.exists():
            return gpd.read_file(WORLD_SHP)
        logging.warning("World shapefile missing: %s", WORLD_SHP)
    except Exception as exc:
        logging.warning("Could not load world shapefile; map will use tile grid only: %s", exc)
    return None


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    if not STAGE17_CSV.exists():
        raise FileNotFoundError(f"Missing Stage17 summary: {STAGE17_CSV}")
    if not STAGE20_CSV.exists():
        raise FileNotFoundError(f"Missing Stage20 summary: {STAGE20_CSV}")

    stage17 = pd.read_csv(STAGE17_CSV)
    stage20 = pd.read_csv(STAGE20_CSV)

    stage17_ok = stage17[stage17["status"].eq("success")].copy()
    stage20_done = stage20[stage20["status"].isin(["success", "skipped"])].copy()
    failed_stage20 = stage20[~stage20["status"].isin(["success", "skipped"])].copy()

    if len(stage17_ok) != 171:
        raise ValueError(f"Stage17 expected 171 success rows, found {len(stage17_ok)}")
    if len(stage20_done) != 171 or not failed_stage20.empty:
        raise ValueError(
            f"Stage20 expected 171 completed rows and 0 failed rows; "
            f"completed={len(stage20_done)}, failed={len(failed_stage20)}"
        )

    coord_cols = [
        "tile_id",
        "min_lon",
        "min_lat",
        "max_lon",
        "max_lat",
        "original_suitable_area_km2_recomputed",
        "constrained_suitable_area_km2",
        "excluded_by_any_area_km2",
    ]
    metric_cols = [
        "tile_id",
        "status",
        "weighted_compatible_area_km2",
        "binary_suitable_area_km2",
        "excluded_by_landcover_area_km2",
        "weighted_retention_pct",
        "binary_retention_pct",
    ]
    merged = stage17_ok[coord_cols].merge(stage20_done[metric_cols], on="tile_id", how="left")
    if merged[metric_cols[1:]].isna().all(axis=1).any():
        missing = merged.loc[merged[metric_cols[1:]].isna().all(axis=1), "tile_id"].tolist()
        raise ValueError(f"Stage20 metrics missing for tile(s): {missing[:10]}")

    totals = {
        "original_suitable_area_km2": float(stage17_ok["original_suitable_area_km2_recomputed"].sum()),
        "spatially_constrained_area_km2": float(stage17_ok["constrained_suitable_area_km2"].sum()),
        "landcover_weighted_area_km2": float(stage20_done["weighted_compatible_area_km2"].sum()),
        "landcover_binary_area_km2": float(stage20_done["binary_suitable_area_km2"].sum()),
        "terrain_oasis_river_excluded_area_km2": float(stage17_ok["excluded_by_any_area_km2"].sum()),
        "landcover_excluded_area_km2": float(stage20_done["excluded_by_landcover_area_km2"].sum()),
        "stage17_success_rows": int(len(stage17_ok)),
        "stage20_completed_rows": int(len(stage20_done)),
        "stage20_status_counts": {str(k): int(v) for k, v in stage20["status"].value_counts(dropna=False).items()},
    }

    cascade = pd.DataFrame(
        [
            {
                "step": "Climate threshold suitable",
                "area_km2": totals["original_suitable_area_km2"],
                "retained_vs_original_pct": 100.0,
            },
            {
                "step": "Terrain, oasis and river constraints",
                "area_km2": totals["spatially_constrained_area_km2"],
                "retained_vs_original_pct": 100.0
                * totals["spatially_constrained_area_km2"]
                / totals["original_suitable_area_km2"],
            },
            {
                "step": "Land-cover weighted compatibility",
                "area_km2": totals["landcover_weighted_area_km2"],
                "retained_vs_original_pct": 100.0
                * totals["landcover_weighted_area_km2"]
                / totals["original_suitable_area_km2"],
            },
        ]
    )
    return stage17_ok, stage20_done, merged, cascade, totals


def draw_map(ax: plt.Axes, merged: pd.DataFrame, world) -> None:
    ax.set_facecolor("#fbfaf7")
    if world is not None:
        try:
            world.plot(ax=ax, color="#f2efe8", edgecolor="#cbd5d0", linewidth=0.28, zorder=0)
        except Exception as exc:
            logging.warning("World map plot failed; continuing with tile grid only: %s", exc)

    ax.set_xlim(-180, 180)
    ax.set_ylim(-58, 82)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Longitude", labelpad=9)
    ax.set_ylabel("Latitude", labelpad=4)
    ax.set_xticks(np.arange(-180, 181, 60))
    ax.set_yticks(np.arange(-40, 81, 40))
    ax.grid(color="#dfe7e3", linewidth=0.6, zorder=0)
    for spine in ax.spines.values():
        spine.set_color("#cbd8d2")
        spine.set_linewidth(0.8)

    values = merged["weighted_compatible_area_km2"].fillna(0.0) / 1000.0
    max_value = max(float(values.max()), 1.0)
    norm = Normalize(vmin=0.0, vmax=max_value)
    cmap = LinearSegmentedColormap.from_list(
        "oasis_constraint_teal",
        ["#edf6ef", "#b9dbc6", "#63aa99", "#1e716d"],
    )

    for _, row in merged.iterrows():
        rect = Rectangle(
            (row["min_lon"], row["min_lat"]),
            row["max_lon"] - row["min_lon"],
            row["max_lat"] - row["min_lat"],
            facecolor="none",
            edgecolor="#b8c7c0",
            linewidth=0.25,
            zorder=1,
        )
        ax.add_patch(rect)

    nonzero = merged[merged["weighted_compatible_area_km2"].fillna(0.0) > 0].copy()
    for _, row in nonzero.iterrows():
        val = row["weighted_compatible_area_km2"] / 1000.0
        rect = Rectangle(
            (row["min_lon"], row["min_lat"]),
            row["max_lon"] - row["min_lon"],
            row["max_lat"] - row["min_lat"],
            facecolor=cmap(norm(val)),
            edgecolor="#1d6f68",
            linewidth=0.62,
            alpha=0.92,
            zorder=3,
        )
        ax.add_patch(rect)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    cbar = plt.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.035, pad=0.13)
    cbar.set_label("Land-cover compatible area per tile (10$^3$ km$^2$)", labelpad=4)
    cbar.outline.set_edgecolor("#cbd8d2")
    cbar.ax.tick_params(labelsize=7.4, length=2.5, colors="#2d3941")

    legend_handles = [
        Patch(facecolor="#63aa99", edgecolor="#1d6f68", label="Weighted compatible area"),
        Patch(facecolor="none", edgecolor="#b8c7c0", label="Processed dryland tile"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower left",
        bbox_to_anchor=(0.01, 0.02),
        frameon=False,
        fontsize=7.8,
        handlelength=1.4,
        borderaxespad=0.0,
    )
    ax.text(
        0.01,
        1.015,
        "a",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.5,
        fontweight="bold",
        color="white",
        bbox={"boxstyle": "round,pad=0.18,rounding_size=0.02", "fc": "#2b817c", "ec": "none"},
    )


def draw_cascade(ax: plt.Axes, cascade: pd.DataFrame, totals: dict) -> None:
    ax.set_axis_off()
    ax.set_xlim(0, 1.04)
    ax.set_ylim(-0.9, 2.75)
    ax.text(
        0.02,
        2.63,
        "b",
        ha="left",
        va="center",
        fontsize=9.5,
        fontweight="bold",
        color="white",
        bbox={"boxstyle": "round,pad=0.18,rounding_size=0.02", "fc": "#2b817c", "ec": "none"},
    )
    ax.text(0.15, 2.62, "Constraint cascade", ha="left", va="center", fontsize=10.2, fontweight="bold")

    colors = ["#6d9ec1", "#d2a546", "#2f8f7f"]
    edge_colors = ["#416c8a", "#9c7520", "#1e6c63"]
    max_area = float(cascade["area_km2"].max())
    bar_h = 0.34
    y_positions = [1.95, 1.05, 0.15]
    bar_specs: list[tuple[float, float, float]] = []

    for i, (_, row) in enumerate(cascade.iterrows()):
        area = float(row["area_km2"])
        width = 0.70 * area / max_area
        left = 0.19 + (0.70 - width) / 2
        y = y_positions[i]
        bar_specs.append((left, width, y))

        rect = Rectangle(
            (left, y - bar_h / 2),
            width,
            bar_h,
            facecolor=colors[i],
            edgecolor=edge_colors[i],
            linewidth=0.8,
            alpha=0.96,
        )
        ax.add_patch(rect)

        step_label = str(row["step"])
        if step_label.startswith("Terrain"):
            step_label = "Spatial constraints"
        elif step_label.startswith("Land-cover"):
            step_label = "Land-cover weighted"
        elif step_label.startswith("Climate"):
            step_label = "Climate threshold"

        ax.text(left, y + 0.285, step_label, ha="left", va="bottom", fontsize=7.9, color="#26343d")
        ax.text(
            min(left + width + 0.025, 0.99),
            y,
            f"{area / 1_000_000:.2f} M km$^2$",
            ha="left",
            va="center",
            fontsize=8.4,
            fontweight="bold",
            color="#26343d",
        )
        ax.text(
            left + width / 2,
            y - 0.285,
            f"{row['retained_vs_original_pct']:.0f}% of original",
            ha="center",
            va="top",
            fontsize=7.2,
            color="#63727b",
        )

    for (left0, width0, y0), (left1, width1, y1) in zip(bar_specs[:-1], bar_specs[1:]):
        poly = Polygon(
            [
                (left0, y0 - bar_h / 2),
                (left0 + width0, y0 - bar_h / 2),
                (left1 + width1, y1 + bar_h / 2),
                (left1, y1 + bar_h / 2),
            ],
            closed=True,
            facecolor="#dce5e0",
            edgecolor="none",
            alpha=0.55,
            zorder=-1,
        )
        ax.add_patch(poly)

    constrained = totals["spatially_constrained_area_km2"]
    weighted = totals["landcover_weighted_area_km2"]
    binary = totals["landcover_binary_area_km2"]
    ax.text(
        0.14,
        -0.43,
        f"Land-cover retention: {100 * weighted / constrained:.1f}% of spatially constrained area",
        ha="left",
        va="center",
        fontsize=7.5,
        color="#2f5f59",
    )
    ax.text(
        0.14,
        -0.68,
        f"Binary 50% compatible area: {binary / 1_000_000:.2f} M km$^2$",
        ha="left",
        va="center",
        fontsize=7.3,
        color="#6b757a",
    )


def make_white_preview(png_path: Path, preview_path: Path) -> None:
    with Image.open(png_path).convert("RGBA") as img:
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        bg.alpha_composite(img)
        bg.convert("RGB").save(preview_path, quality=95)


def verify_png_alpha(path: Path) -> dict:
    with Image.open(path) as img:
        has_alpha = img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info)
        return {"mode": img.mode, "size": img.size, "has_alpha": bool(has_alpha)}


def write_readme(outputs: dict, totals: dict) -> Path:
    readme = STAGE_DIR / f"{FIGURE_STEM}_README.md"
    lines = [
        "# Stage46 Fig5 candidate: selected10 land-cover constrained footprint",
        "",
        "- Status: candidate figure, not inserted into Word.",
        "- Figure family: single global tile map with area funnel; this intentionally avoids repeating the 2x2 plate format.",
        "- Input Stage17: `outputs/stage34_selected10_constrained_suitability/tables/stage17_constrained_suitability_selected10_hgb_main_summary.csv`.",
        "- Input Stage20: `outputs/stage34_selected10_landcover_spatial_constraint/tables/stage20_landcover_spatial_constraint_selected10_hgb_main_summary.csv`.",
        "- Scenario: HGB selected10, ACCESS-CM2, SSP5-8.5, 2081-2100.",
        "- Stage20 status interpretation: `success` and `skipped` both mean completed output rasters/tables are present; skipped rows were produced by the final completeness pass after the single failed state-write tile was repaired.",
        "",
        "## Area values",
        "",
        f"- Climate-threshold suitable area: {totals['original_suitable_area_km2']:.2f} km2.",
        f"- Spatially constrained suitable area: {totals['spatially_constrained_area_km2']:.2f} km2.",
        f"- Land-cover weighted compatible area: {totals['landcover_weighted_area_km2']:.2f} km2.",
        f"- Binary 50% compatible area: {totals['landcover_binary_area_km2']:.2f} km2.",
        "",
        "## Outputs",
        "",
    ]
    for key, value in outputs.items():
        lines.append(f"- {key}: `{rel(Path(value)) if isinstance(value, str) else value}`")
    readme.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return readme


def run() -> dict:
    setup_dirs()
    log_path = setup_logging()
    status_path = STAGE_DIR / f"{FIGURE_STEM}_status.json"
    running = {
        "status": "running",
        "started_at": now_iso(),
        "figure_id": FIGURE_ID,
        "layout_family": LAYOUT_FAMILY,
    }
    write_json_atomic(status_path, running)
    update_layout_ledger(
        "candidate_pending_review",
        "New non-2x2 global footprint map plus side area funnel; visual judge still required.",
    )

    try:
        logging.info("Loading Stage17 and Stage20 summaries.")
        stage17, stage20, merged, cascade, totals = load_data()
        world = load_world()

        tile_table = TABLE_DIR / f"{FIGURE_STEM}_tile_data.csv"
        cascade_table = TABLE_DIR / f"{FIGURE_STEM}_area_cascade.csv"
        status_table = TABLE_DIR / f"{FIGURE_STEM}_input_status_counts.csv"
        merged.to_csv(tile_table, index=False, encoding="utf-8-sig")
        cascade.to_csv(cascade_table, index=False, encoding="utf-8-sig")
        pd.DataFrame(
            [
                {"source": "Stage17", "status": "success", "rows": len(stage17)},
                *[
                    {"source": "Stage20", "status": str(status), "rows": int(count)}
                    for status, count in stage20["status"].value_counts(dropna=False).items()
                ],
            ]
        ).to_csv(status_table, index=False, encoding="utf-8-sig")

        mpl.rcParams.update(
            {
                "font.family": "DejaVu Sans",
                "font.size": 8.5,
                "axes.labelsize": 8.4,
                "xtick.labelsize": 7.5,
                "ytick.labelsize": 7.5,
                "legend.fontsize": 7.6,
                "svg.fonttype": "none",
                "pdf.fonttype": 42,
                "axes.unicode_minus": False,
            }
        )

        fig = plt.figure(figsize=(12.4, 5.35), dpi=180)
        grid = fig.add_gridspec(
            nrows=1,
            ncols=2,
            width_ratios=[3.4, 1.16],
            left=0.045,
            right=0.985,
            top=0.94,
            bottom=0.18,
            wspace=0.12,
        )
        ax_map = fig.add_subplot(grid[0, 0])
        ax_cascade = fig.add_subplot(grid[0, 1])

        draw_map(ax_map, merged, world)
        draw_cascade(ax_cascade, cascade, totals)

        png_path = FIG_DIR / f"{FIGURE_STEM}.png"
        svg_path = FIG_DIR / f"{FIGURE_STEM}.svg"
        pdf_path = FIG_DIR / f"{FIGURE_STEM}.pdf"
        preview_path = FIG_DIR / f"{FIGURE_STEM}_white_preview.png"

        fig.savefig(png_path, dpi=600, bbox_inches="tight", pad_inches=0.02, transparent=True)
        fig.savefig(svg_path, bbox_inches="tight", pad_inches=0.02, transparent=True)
        fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02, transparent=True)
        plt.close(fig)
        make_white_preview(png_path, preview_path)
        alpha_check = verify_png_alpha(png_path)
        if not alpha_check["has_alpha"]:
            raise RuntimeError(f"Transparent PNG alpha check failed: {alpha_check}")

        outputs = {
            "transparent_png": str(png_path),
            "svg": str(svg_path),
            "pdf": str(pdf_path),
            "white_preview": str(preview_path),
            "tile_data_csv": str(tile_table),
            "area_cascade_csv": str(cascade_table),
            "status_counts_csv": str(status_table),
            "log": str(log_path),
        }
        readme_path = write_readme(outputs, totals)
        outputs["readme"] = str(readme_path)

        payload = {
            "status": "success",
            "started_at": running["started_at"],
            "finished_at": now_iso(),
            "figure_id": FIGURE_ID,
            "figure_stem": FIGURE_STEM,
            "layout_family": LAYOUT_FAMILY,
            "user_confirmed_final": False,
            "word_insertion_allowed": False,
            "inputs": {
                "stage17_summary": rel(STAGE17_CSV),
                "stage20_summary": rel(STAGE20_CSV),
                "world_shapefile": rel(WORLD_SHP),
            },
            "totals": totals,
            "outputs": {key: rel(Path(value)) for key, value in outputs.items()},
            "alpha_check": alpha_check,
            "notes": [
                "Candidate figure only; not inserted into Word.",
                "All Stage20 rows are completed as skipped after completeness pass; no failed rows remain.",
            ],
        }
        write_json_atomic(status_path, payload)
        logging.info("Figure package completed: %s", json.dumps(payload["outputs"], ensure_ascii=False))
        return payload
    except Exception as exc:
        payload = {
            "status": "failed",
            "started_at": running["started_at"],
            "finished_at": now_iso(),
            "figure_id": FIGURE_ID,
            "figure_stem": FIGURE_STEM,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        write_json_atomic(status_path, payload)
        logging.exception("Stage46 figure generation failed.")
        raise


def main() -> int:
    payload = run()
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
