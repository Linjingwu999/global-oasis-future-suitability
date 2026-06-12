#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Create Stage47 Fig6 candidate: selected10 area-budget mosaic.

The figure uses one integrated area-budget mosaic plus a compact overlap
diagnostic. It deliberately avoids another 2x2 statistical plate.
"""

from __future__ import annotations

import json
import logging
import traceback
from datetime import datetime
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGE_DIR = PROJECT_ROOT / "outputs" / "stage47_manuscript_main_figures"
FIG_DIR = STAGE_DIR / "figures"
TABLE_DIR = STAGE_DIR / "tables"
LOG_DIR = STAGE_DIR / "logs"
QC_DIR = STAGE_DIR / "qc"

FIGURE_STEM = "fig_stage47_fig6_selected10_area_budget_mosaic_v01"
FIGURE_ID = "Fig6_stage47_v01"
LAYOUT_FAMILY = "single-waffle-area-budget-with-overlap-diagnostics"

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
    "retained": "#2f817b",
    "spatial_excluded": "#bf6f4b",
    "landcover_excluded": "#d6a53a",
    "residual": "#cfd8d3",
    "ink": "#24313d",
    "muted": "#6d7984",
    "grid": "#e1e8e5",
    "frame": "#d8e1dc",
    "axis": "#a9b5b3",
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


def pct(value: float, denominator: float) -> float:
    return 100.0 * float(value) / float(denominator) if denominator else 0.0


def million(value: float) -> float:
    return float(value) / 1_000_000.0


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    if not STAGE17_CSV.exists():
        raise FileNotFoundError(f"Missing Stage17 summary: {STAGE17_CSV}")
    if not STAGE20_CSV.exists():
        raise FileNotFoundError(f"Missing Stage20 summary: {STAGE20_CSV}")

    stage17 = pd.read_csv(STAGE17_CSV)
    stage20 = pd.read_csv(STAGE20_CSV)
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

    original = float(stage17_ok["original_suitable_area_km2_recomputed"].sum())
    spatial_retained = float(stage17_ok["constrained_suitable_area_km2"].sum())
    spatial_excluded = float(stage17_ok["excluded_by_any_area_km2"].sum())
    landcover_weighted = float(stage20_done["weighted_compatible_area_km2"].sum())
    landcover_binary = float(stage20_done["binary_suitable_area_km2"].sum())
    landcover_excluded = float(stage20_done["excluded_by_landcover_area_km2"].sum())
    residual = original - spatial_excluded - landcover_excluded - landcover_weighted

    if residual < -0.001 * original:
        raise ValueError(f"Area budget residual is negative and too large: {residual:.3f} km²")
    residual = max(0.0, residual)

    area_budget = pd.DataFrame(
        [
            {
                "component": "Land-cover weighted retained",
                "short_label": "Retained",
                "area_km2": landcover_weighted,
                "percent_of_original": pct(landcover_weighted, original),
                "color_key": "retained",
                "plot_order": 3,
            },
            {
                "component": "Excluded by terrain, oasis, or river union",
                "short_label": "Spatial filters",
                "area_km2": spatial_excluded,
                "percent_of_original": pct(spatial_excluded, original),
                "color_key": "spatial_excluded",
                "plot_order": 1,
            },
            {
                "component": "Excluded by land-cover compatibility",
                "short_label": "Land cover",
                "area_km2": landcover_excluded,
                "percent_of_original": pct(landcover_excluded, original),
                "color_key": "landcover_excluded",
                "plot_order": 2,
            },
            {
                "component": "Small accounting residual",
                "short_label": "Residual",
                "area_km2": residual,
                "percent_of_original": pct(residual, original),
                "color_key": "residual",
                "plot_order": 4,
            },
        ]
    ).sort_values("plot_order")

    spatial_filters = pd.DataFrame(
        [
            {
                "filter": "Elevation",
                "area_km2": float(stage17_ok["excluded_by_elevation_area_km2"].sum()),
            },
            {
                "filter": "Slope",
                "area_km2": float(stage17_ok["excluded_by_slope_area_km2"].sum()),
            },
            {
                "filter": "Oasis buffer",
                "area_km2": float(stage17_ok["excluded_by_oasis_area_km2"].sum()),
            },
            {
                "filter": "River buffer",
                "area_km2": float(stage17_ok["excluded_by_river_area_km2"].sum()),
            },
        ]
    )
    spatial_filters["percent_of_original"] = spatial_filters["area_km2"].map(lambda value: pct(value, original))
    spatial_filters["non_additive_note"] = "overlap possible; do not sum individual filters"

    totals = {
        "input_stage17_rows": int(len(stage17)),
        "input_stage20_rows": int(len(stage20)),
        "stage17_success_rows": int(len(stage17_ok)),
        "stage20_completed_rows": int(len(stage20_done)),
        "stage20_status_counts": {str(k): int(v) for k, v in stage20["status"].value_counts(dropna=False).items()},
        "model_group": str(stage17_ok["model_group"].iloc[0]),
        "gcm": str(stage17_ok["gcm"].iloc[0]),
        "ssp": str(stage17_ok["ssp"].iloc[0]),
        "period": str(stage17_ok["period"].iloc[0]),
        "original_suitable_area_km2": original,
        "spatially_constrained_area_km2": spatial_retained,
        "excluded_by_spatial_union_area_km2": spatial_excluded,
        "landcover_weighted_compatible_area_km2": landcover_weighted,
        "landcover_binary_suitable_area_km2": landcover_binary,
        "excluded_by_landcover_area_km2": landcover_excluded,
        "accounting_residual_area_km2": residual,
        "accounting_residual_percent_of_original": pct(residual, original),
    }
    return stage17_ok, stage20_done, area_budget, spatial_filters, totals


def assign_waffle_counts(area_budget: pd.DataFrame, total_cells: int = 200) -> pd.DataFrame:
    budget = area_budget.copy()
    raw = budget["percent_of_original"].to_numpy(dtype=float) * total_cells / 100.0
    floors = np.floor(raw).astype(int)
    remainder = int(total_cells - floors.sum())
    fractions = raw - floors
    if remainder > 0:
        for idx in np.argsort(fractions)[::-1][:remainder]:
            floors[idx] += 1
    elif remainder < 0:
        for idx in np.argsort(fractions)[: abs(remainder)]:
            floors[idx] = max(0, floors[idx] - 1)
    budget["waffle_cells"] = floors
    budget["waffle_cell_percent"] = 100.0 * budget["waffle_cells"] / float(total_cells)
    return budget


def add_text(ax, x: float, y: float, text: str, **kwargs):
    kwargs.setdefault("transform", ax.transAxes)
    kwargs.setdefault("color", PALETTE["ink"])
    kwargs.setdefault("ha", "left")
    kwargs.setdefault("va", "center")
    return ax.text(x, y, text, **kwargs)


def draw_pipeline(ax, totals: dict) -> None:
    y = 0.840
    milestones = [
        ("Climate threshold", totals["original_suitable_area_km2"], 0.105),
        ("Spatially constrained", totals["spatially_constrained_area_km2"], 0.355),
        ("Land-cover weighted", totals["landcover_weighted_compatible_area_km2"], 0.625),
    ]
    for idx, (label, area, x) in enumerate(milestones):
        box = FancyBboxPatch(
            (x, y - 0.055),
            0.19,
            0.11,
            boxstyle="round,pad=0.008,rounding_size=0.015",
            linewidth=1.0,
            edgecolor=PALETTE["frame"],
            facecolor="#f5f8f6",
            transform=ax.transAxes,
        )
        ax.add_patch(box)
        add_text(ax, x + 0.018, y + 0.018, f"{million(area):.2f}M km²", fontsize=15, fontweight="bold")
        add_text(ax, x + 0.018, y - 0.028, label, fontsize=8.3, color=PALETTE["muted"])
        if idx < len(milestones) - 1:
            start = (x + 0.202, y)
            end = (milestones[idx + 1][2] - 0.016, y)
            ax.add_patch(
                FancyArrowPatch(
                    start,
                    end,
                    transform=ax.transAxes,
                    arrowstyle="-|>",
                    mutation_scale=13,
                    linewidth=1.5,
                    color=PALETTE["axis"],
                )
            )


def draw_waffle(ax, area_budget: pd.DataFrame, totals: dict) -> None:
    budget = assign_waffle_counts(area_budget)
    grid_x = 0.065
    grid_y = 0.235
    cols = 20
    fig_w, fig_h = ax.figure.get_size_inches()
    aspect = fig_w / fig_h
    cell_w = 0.0192
    gap_w = 0.0022
    cell_h = cell_w * aspect
    gap_h = gap_w * aspect
    total_h = 10 * cell_h + 9 * gap_h
    total_w = cols * cell_w + (cols - 1) * gap_w

    outline = FancyBboxPatch(
        (grid_x - 0.014, grid_y - 0.020),
        total_w + 0.028,
        total_h + 0.040,
        boxstyle="round,pad=0.003,rounding_size=0.012",
        linewidth=1.0,
        edgecolor=PALETTE["frame"],
        facecolor="#ffffff",
        alpha=0.20,
        transform=ax.transAxes,
    )
    ax.add_patch(outline)

    cells: list[str] = []
    for _, row in budget.iterrows():
        cells.extend([str(row["color_key"])] * int(row["waffle_cells"]))
    cells = (cells + ["residual"] * 200)[:200]

    for idx, key in enumerate(cells):
        row = idx // cols
        col = idx % cols
        x = grid_x + col * (cell_w + gap_w)
        y = grid_y + (9 - row) * (cell_h + gap_h)
        rect = Rectangle(
            (x, y),
            cell_w,
            cell_h,
            transform=ax.transAxes,
            facecolor=PALETTE[key],
            edgecolor="white",
            linewidth=0.32,
            alpha=0.95 if key != "residual" else 0.78,
        )
        ax.add_patch(rect)

    add_text(ax, grid_x, grid_y + total_h + 0.072, "Area budget mosaic", fontsize=12, fontweight="bold")
    add_text(
        ax,
        grid_x,
        grid_y + total_h + 0.035,
        "Each square represents 0.5% of the original climate-threshold suitable area.",
        fontsize=8.0,
        color=PALETTE["muted"],
    )

    budget_out = budget.sort_values("plot_order").copy()
    legend_x = 0.595
    legend_y = 0.675
    add_text(ax, legend_x, legend_y + 0.075, "Partition of 7.30M km²", fontsize=11.3, fontweight="bold")
    add_text(ax, legend_x, legend_y + 0.040, "Values are percent of original suitable area.", fontsize=7.8, color=PALETTE["muted"])
    for i, row in enumerate(budget_out.itertuples(index=False)):
        y = legend_y - i * 0.078
        key = str(row.color_key)
        ax.add_patch(
            Rectangle(
                (legend_x, y - 0.014),
                0.018,
                0.028,
                transform=ax.transAxes,
                facecolor=PALETTE[key],
                edgecolor="white",
                linewidth=0.5,
            )
        )
        add_text(ax, legend_x + 0.030, y + 0.008, str(row.short_label), fontsize=9.6, fontweight="bold")
        add_text(
            ax,
            legend_x + 0.030,
            y - 0.020,
            str(row.component),
            fontsize=7.45,
            color=PALETTE["muted"],
        )
        add_text(
            ax,
            0.905,
            y + 0.006,
            f"{float(row.percent_of_original):.1f}%",
            fontsize=10.5,
            fontweight="bold",
            ha="right",
        )
        add_text(
            ax,
            0.975,
            y + 0.006,
            f"{million(float(row.area_km2)):.2f}M",
            fontsize=8.6,
            color=PALETTE["muted"],
            ha="right",
        )


def draw_spatial_filter_diagnostic(ax, spatial_filters: pd.DataFrame) -> None:
    x0, x1 = 0.595, 0.952
    y0 = 0.265
    add_text(ax, x0, y0 + 0.085, "Spatial filters, shown separately", fontsize=10.2, fontweight="bold")
    add_text(
        ax,
        x0,
        y0 + 0.052,
        "Individual filters overlap, so these bars are not additive.",
        fontsize=7.7,
        color=PALETTE["muted"],
    )
    max_pct = 36.0
    for tick in [0, 10, 20, 30]:
        x = x0 + (x1 - x0) * tick / max_pct
        ax.plot([x, x], [y0 - 0.115, y0 + 0.020], transform=ax.transAxes, color=PALETTE["grid"], lw=0.8)
        add_text(ax, x, y0 - 0.145, f"{tick}", fontsize=7.0, color=PALETTE["muted"], ha="center")
    add_text(ax, x1 + 0.010, y0 - 0.145, "%", fontsize=7.0, color=PALETTE["muted"], ha="left")

    order = ["Oasis buffer", "River buffer", "Slope", "Elevation"]
    color_map = {
        "Oasis buffer": "#a85f44",
        "River buffer": "#ba7953",
        "Slope": "#ce986d",
        "Elevation": "#e0b489",
    }
    table = spatial_filters.set_index("filter").loc[order].reset_index()
    for idx, row in enumerate(table.itertuples(index=False)):
        y = y0 - idx * 0.040
        bar_end = x0 + (x1 - x0) * float(row.percent_of_original) / max_pct
        ax.plot([x0, bar_end], [y, y], transform=ax.transAxes, color="#cbd5d2", lw=5.0, solid_capstyle="round")
        ax.plot([bar_end], [y], transform=ax.transAxes, marker="o", ms=5.8, color=color_map[str(row.filter)], mec="white", mew=0.7)
        add_text(ax, x0 - 0.012, y, str(row.filter), fontsize=8.1, color=PALETTE["ink"], ha="right")
        label = f"{float(row.percent_of_original):.1f}%  ({million(float(row.area_km2)):.2f}M km²)"
        add_text(ax, bar_end + 0.012, y, label, fontsize=7.7, color=PALETTE["muted"])


def draw_figure(area_budget: pd.DataFrame, spatial_filters: pd.DataFrame, totals: dict) -> plt.Figure:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )
    fig = plt.figure(figsize=(11.4, 5.55), dpi=180)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    add_text(
        ax,
        0.065,
        0.970,
        f"{totals['gcm']} {totals['ssp']} {totals['period']} | HGB selected10",
        fontsize=8.4,
        color=PALETTE["muted"],
    )
    add_text(
        ax,
        0.065,
        0.938,
        "Future suitable-area budget after spatial and land-cover constraints",
        fontsize=14.2,
        fontweight="bold",
    ).set_path_effects([pe.withStroke(linewidth=2.6, foreground="white", alpha=0.75)])
    draw_pipeline(ax, totals)
    draw_waffle(ax, area_budget, totals)
    draw_spatial_filter_diagnostic(ax, spatial_filters)
    add_text(
        ax,
        0.065,
        0.116,
        "Grey cell = 0.4% accounting residual; spatial filter bars are non-additive.",
        fontsize=7.2,
        color=PALETTE["muted"],
    )
    return fig


def make_white_preview(png_path: Path, preview_path: Path) -> None:
    with Image.open(png_path).convert("RGBA") as img:
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        bg.alpha_composite(img)
        bg.convert("RGB").save(preview_path, "PNG")


def verify_png_alpha(png_path: Path) -> dict:
    with Image.open(png_path) as img:
        has_alpha = img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info)
        if not has_alpha:
            return {"has_alpha": False, "mode": img.mode, "size": list(img.size)}
        alpha = img.convert("RGBA").getchannel("A")
        return {
            "has_alpha": True,
            "mode": img.mode,
            "size": list(img.size),
            "alpha_min": int(np.asarray(alpha).min()),
            "alpha_max": int(np.asarray(alpha).max()),
        }


def write_readme(outputs: dict, totals: dict) -> Path:
    path = STAGE_DIR / f"{FIGURE_STEM}_README.md"
    text = f"""# {FIGURE_ID}: Selected10 area-budget mosaic v01

Status: candidate, not inserted into Word.

## Purpose

This figure summarizes how the HGB selected10 future suitable area changes after
terrain/oasis/river constraints and land-cover compatibility weighting. The
layout is a single area-budget mosaic plus a compact overlap diagnostic, so it
does not repeat the Fig3 2x2 statistical plate or the Fig5 global map plus area
funnel.

## Inputs

- Stage17 constrained suitability summary: `{rel(STAGE17_CSV)}`
- Stage20 land-cover spatial constraint summary: `{rel(STAGE20_CSV)}`
- Scenario: `{totals['gcm']} / {totals['ssp']} / {totals['period']}`
- Model: `{totals['model_group']}`

## Main numbers

- Original climate-threshold suitable area: {totals['original_suitable_area_km2']:.3f} km²
- Spatially constrained suitable area: {totals['spatially_constrained_area_km2']:.3f} km²
- Land-cover weighted compatible area: {totals['landcover_weighted_compatible_area_km2']:.3f} km²
- Land-cover excluded area: {totals['excluded_by_landcover_area_km2']:.3f} km²
- Accounting residual: {totals['accounting_residual_area_km2']:.3f} km² ({totals['accounting_residual_percent_of_original']:.3f}% of original)

## Caveats

- Spatial filter components overlap and must not be summed as independent
  exclusions.
- Stage20 `skipped` rows indicate existing completed raster/table outputs were
  reused during the final completeness pass. There are no failed Stage20 rows.
- The grey residual is retained in the plot because weighted compatibility,
  binary compatibility, and excluded-area accounting do not form a perfectly
  closed partition at sub-tile precision.

## Outputs

- Transparent PNG: `{rel(Path(outputs['transparent_png']))}`
- SVG: `{rel(Path(outputs['svg']))}`
- PDF: `{rel(Path(outputs['pdf']))}`
- White preview: `{rel(Path(outputs['white_preview']))}`
- Area budget table: `{rel(Path(outputs['area_budget_csv']))}`
- Spatial filter diagnostics: `{rel(Path(outputs['spatial_filter_csv']))}`
- Input status counts: `{rel(Path(outputs['status_counts_csv']))}`
"""
    path.write_text(text, encoding="utf-8")
    return path


def run() -> dict:
    setup_dirs()
    log_path = setup_logging()
    status_path = STAGE_DIR / f"{FIGURE_STEM}_status.json"
    running = {
        "status": "running",
        "started_at": now_iso(),
        "figure_id": FIGURE_ID,
        "figure_stem": FIGURE_STEM,
        "layout_family": LAYOUT_FAMILY,
        "message": "Generating candidate figure package.",
    }
    write_json_atomic(status_path, running)
    logging.info("Stage47 Fig6 generation started.")

    try:
        stage17, stage20, area_budget, spatial_filters, totals = load_data()
        area_budget_plot = assign_waffle_counts(area_budget)

        area_budget_csv = TABLE_DIR / f"{FIGURE_STEM}_area_budget.csv"
        spatial_filter_csv = TABLE_DIR / f"{FIGURE_STEM}_spatial_filter_overlap_diagnostics.csv"
        status_counts_csv = TABLE_DIR / f"{FIGURE_STEM}_input_status_counts.csv"
        area_budget_plot.to_csv(area_budget_csv, index=False, encoding="utf-8-sig")
        spatial_filters.to_csv(spatial_filter_csv, index=False, encoding="utf-8-sig")
        pd.DataFrame(
            [
                {"source": "Stage17", "status": "success", "rows": int(len(stage17))},
                *[
                    {"source": "Stage20", "status": str(status), "rows": int(count)}
                    for status, count in stage20["status"].value_counts(dropna=False).items()
                ],
            ]
        ).to_csv(status_counts_csv, index=False, encoding="utf-8-sig")

        fig = draw_figure(area_budget, spatial_filters, totals)
        png_path = FIG_DIR / f"{FIGURE_STEM}.png"
        svg_path = FIG_DIR / f"{FIGURE_STEM}.svg"
        pdf_path = FIG_DIR / f"{FIGURE_STEM}.pdf"
        preview_path = FIG_DIR / f"{FIGURE_STEM}_white_preview.png"
        fig.savefig(png_path, dpi=600, bbox_inches="tight", pad_inches=0.015, transparent=True)
        fig.savefig(svg_path, bbox_inches="tight", pad_inches=0.015, transparent=True)
        fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.015, transparent=True)
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
            "area_budget_csv": str(area_budget_csv),
            "spatial_filter_csv": str(spatial_filter_csv),
            "status_counts_csv": str(status_counts_csv),
            "log": str(log_path),
        }
        readme_path = write_readme(outputs, totals)
        outputs["readme"] = str(readme_path)

        update_layout_ledger(
            "candidate_needs_review",
            "Non-2x2 selected10 HGB area-budget mosaic. Awaiting local and browser review; not inserted into Word.",
        )
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
            },
            "totals": totals,
            "outputs": {key: rel(Path(value)) for key, value in outputs.items()},
            "alpha_check": alpha_check,
            "notes": [
                "Candidate figure only; not inserted into Word.",
                "Spatial filter components overlap and are marked as non-additive.",
                "Stage20 skipped rows are treated as completed because existing output rasters/tables were reused.",
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
            "layout_family": LAYOUT_FAMILY,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        write_json_atomic(status_path, payload)
        logging.exception("Stage47 figure generation failed.")
        raise


def main() -> int:
    payload = run()
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
