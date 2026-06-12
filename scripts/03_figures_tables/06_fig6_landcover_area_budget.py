#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Create Stage50 Fig6 candidate: q10 land-cover area budget.

This figure replaces the older Stage47 broad-reference area-budget figure with
the current selected10 HGB main-chain q10 hydrological threshold. It uses a
single waterfall-style area accounting plot to avoid repeating the earlier
2x2, slopegraph, and waffle-mosaic figure forms.
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
from matplotlib.patches import Rectangle
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGE_DIR = PROJECT_ROOT / "outputs" / "stage50_ijaeog_q10_figure_updates"
FIG_DIR = STAGE_DIR / "figures"
TABLE_DIR = STAGE_DIR / "tables"
LOG_DIR = STAGE_DIR / "logs"
QC_DIR = STAGE_DIR / "qc"

FIGURE_STEM = "fig_stage50_fig6_q10_landcover_area_budget_v03"
FIGURE_ID = "Fig6_stage50_v03"
LAYOUT_FAMILY = "single-waterfall-area-budget-with-reference-line"

SUMMARY_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "stage36_hydrology_landcover_sensitivity"
    / "tables"
    / "stage36_hydrology_landcover_sensitivity_summary.csv"
)
STAGE36_STATUS_JSON = (
    PROJECT_ROOT
    / "outputs"
    / "stage36_hydrology_landcover_sensitivity"
    / "stage36_hydrology_landcover_sensitivity_summary.json"
)
Q10_STATUS_JSON = (
    PROJECT_ROOT
    / "outputs"
    / "stage36_hydrology_landcover_sensitivity"
    / "q10cms"
    / "stage20_landcover_spatial_constraint_selected10_hgb_hydrorivers_q10cms_landcover_summary.json"
)
LEDGER_JSON = PROJECT_ROOT / "docs" / "figure_style_ledger.json"

# Current manuscript-guidance comparator. This is not recalculated by this
# figure script and should be cited from the 2020 oasis mapping source in text.
OASIS_2020_REFERENCE_MILLION_KM2 = 2.20

PALETTE = {
    "envelope": "#8FC2B3",
    "landcover_loss": "#C07A37",
    "weighting_loss": "#D9B84F",
    "final": "#167F7A",
    "reference": "#7F6AA3",
    "connector": "#AEBABD",
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


def million(value: float) -> float:
    return float(value) / 1_000_000.0


def load_q10_values() -> tuple[pd.DataFrame, dict, dict]:
    if not SUMMARY_CSV.exists():
        raise FileNotFoundError(f"Missing Stage36 hydrology summary: {SUMMARY_CSV}")

    summary = pd.read_csv(SUMMARY_CSV)
    required = {"q1_current_reference", "q10_main_candidate", "q25_strict_backup"}
    observed = set(summary["scenario"].astype(str))
    missing = sorted(required - observed)
    if missing:
        raise ValueError(f"Stage36 summary is missing required scenarios: {missing}")

    q10 = summary.loc[summary["scenario"].eq("q10_main_candidate")].iloc[0].to_dict()
    status = str(q10.get("summary_status", ""))
    if status != "success":
        raise ValueError(f"q10 row is not successful: summary_status={status}")
    if int(q10.get("rows", -1)) != 171 or int(q10.get("ok_rows", -1)) != 171:
        raise ValueError(
            "q10 row does not report 171/171 successful tiles: "
            f"rows={q10.get('rows')}, ok_rows={q10.get('ok_rows')}"
        )
    if int(q10.get("failed_rows", -1)) != 0:
        raise ValueError(f"q10 row reports failed rows: {q10.get('failed_rows')}")

    stage17 = float(q10["stage17_area_km2"])
    binary50 = float(q10["binary50_area_km2"])
    weighted = float(q10["weighted_area_km2"])
    landcover_excluded = float(q10["landcover_excluded_km2"])
    weighting_discount = binary50 - weighted

    if not (stage17 > binary50 >= weighted > 0):
        raise ValueError(
            "q10 area ordering is invalid: "
            f"stage17={stage17}, binary50={binary50}, weighted={weighted}"
        )
    if abs((stage17 - binary50) - landcover_excluded) > 1.0:
        raise ValueError(
            "q10 land-cover exclusion does not balance stage17 - binary50: "
            f"stage17-binary50={stage17 - binary50}, reported={landcover_excluded}"
        )

    stage36_status = {}
    if STAGE36_STATUS_JSON.exists():
        stage36_status = json.loads(STAGE36_STATUS_JSON.read_text(encoding="utf-8"))
    q10_status = {}
    if Q10_STATUS_JSON.exists():
        q10_status = json.loads(Q10_STATUS_JSON.read_text(encoding="utf-8"))

    rows = pd.DataFrame(
        [
            {
                "step": "q10_hydro_spatial_envelope",
                "label": "q10 hydro-spatial\nenvelope",
                "operation": "initial",
                "start_million_km2": 0.0,
                "end_million_km2": million(stage17),
                "delta_million_km2": million(stage17),
                "color_key": "envelope",
            },
            {
                "step": "land_cover_below_50_excluded",
                "label": "Land-cover\n<50% excluded",
                "operation": "loss",
                "start_million_km2": million(stage17),
                "end_million_km2": million(binary50),
                "delta_million_km2": -million(landcover_excluded),
                "color_key": "landcover_loss",
            },
            {
                "step": "compatibility_weighting_discount",
                "label": "Compatibility\nweighting",
                "operation": "loss",
                "start_million_km2": million(binary50),
                "end_million_km2": million(weighted),
                "delta_million_km2": -million(weighting_discount),
                "color_key": "weighting_loss",
            },
            {
                "step": "weighted_compatible_area",
                "label": "Weighted\ncompatible area",
                "operation": "final",
                "start_million_km2": 0.0,
                "end_million_km2": million(weighted),
                "delta_million_km2": million(weighted),
                "color_key": "final",
            },
        ]
    )

    totals = {
        "q10_stage17_area_million_km2": million(stage17),
        "q10_binary50_area_million_km2": million(binary50),
        "q10_weighted_area_million_km2": million(weighted),
        "q10_landcover_excluded_million_km2": million(landcover_excluded),
        "q10_weighting_discount_million_km2": million(weighting_discount),
        "q10_weighted_area_wan_km2": float(q10["weighted_area_wan_km2"]),
        "q10_binary50_area_wan_km2": float(q10["binary50_area_wan_km2"]),
        "q10_stage17_area_wan_km2": float(q10["stage17_area_wan_km2"]),
        "q10_tiles_success": int(q10["ok_rows"]),
        "q10_tiles_total": int(q10["rows"]),
        "q10_failed_rows": int(q10["failed_rows"]),
        "q10_weighted_vs_q1_pct": float(q10["weighted_pct_vs_q1"]),
        "oasis_2020_reference_million_km2": OASIS_2020_REFERENCE_MILLION_KM2,
        "source_summary_csv": rel(SUMMARY_CSV),
        "source_stage36_status_json": rel(STAGE36_STATUS_JSON),
        "source_q10_status_json": rel(Q10_STATUS_JSON),
        "stage36_status": stage36_status.get("status", "missing"),
        "q10_stage20_status": q10_status.get("status", "missing"),
    }
    return rows, totals, q10


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.2,
            "axes.labelsize": 8.6,
            "xtick.labelsize": 7.9,
            "ytick.labelsize": 8.0,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )


def add_waterfall(ax: plt.Axes, rows: pd.DataFrame, totals: dict) -> None:
    x = np.arange(len(rows))
    bar_width = 0.52

    for idx, row in rows.iterrows():
        start = float(row["start_million_km2"])
        end = float(row["end_million_km2"])
        low = min(start, end)
        high = max(start, end)
        height = max(high - low, 0.006)
        color = PALETTE[str(row["color_key"])]
        ax.add_patch(
            Rectangle(
                (idx - bar_width / 2, low),
                bar_width,
                height,
                facecolor=color,
                edgecolor="white",
                linewidth=0.9,
                zorder=3,
            )
        )

        if idx < len(rows) - 1:
            connector_y = end
            ax.plot(
                [idx + bar_width / 2, idx + 1 - bar_width / 2],
                [connector_y, connector_y],
                color=PALETTE["connector"],
                lw=1.15,
                zorder=2,
            )

    labels = [
        (0, rows.loc[0, "end_million_km2"] - 0.12, "3.02", "white", "bold", "center"),
        (1, (rows.loc[1, "start_million_km2"] + rows.loc[1, "end_million_km2"]) / 2, "-0.64", "white", "bold", "center"),
        (3, rows.loc[3, "end_million_km2"] + 0.09, "2.36", PALETTE["final"], "bold", "center"),
    ]
    for x_pos, y_pos, text, color, weight, va in labels:
        ax.text(
            x_pos,
            y_pos,
            text,
            ha="center",
            va=va,
            color=color,
            fontsize=8.7,
            fontweight=weight,
            zorder=5,
        )

    # The weighting discount is deliberately labelled outside its very small bar.
    ax.annotate(
        "-0.02",
        xy=(2, rows.loc[2, "end_million_km2"] + 0.012),
        xytext=(2.23, 2.68),
        textcoords="data",
        ha="left",
        va="center",
        fontsize=8.0,
        color=PALETTE["muted"],
        arrowprops={
            "arrowstyle": "-",
            "color": PALETTE["connector"],
            "lw": 0.9,
            "shrinkA": 0,
            "shrinkB": 0,
            "connectionstyle": "angle3,angleA=0,angleB=90",
        },
        zorder=5,
    )

    reference = float(totals["oasis_2020_reference_million_km2"])
    ax.axhline(reference, color=PALETTE["reference"], lw=1.1, ls=(0, (4, 3)), zorder=1)
    ax.text(
        3.40,
        reference - 0.030,
        "2020 oasis\nreference",
        ha="left",
        va="top",
        fontsize=7.6,
        color=PALETTE["reference"],
        linespacing=1.1,
        zorder=5,
    )

    ax.text(
        0.01,
        1.045,
        "HydroRIVERS $\\geq$10 m$^3$ s$^{-1}$ main chain",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.9,
        color=PALETTE["muted"],
    )

    ax.set_xlim(-0.55, 4.18)
    ax.set_ylim(0, 3.36)
    ax.set_xticks(x)
    ax.set_xticklabels(rows["label"].tolist())
    ax.set_ylabel("Area (million km$^2$)")
    ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.8, zorder=0)
    ax.grid(axis="x", visible=False)
    for spine in ax.spines.values():
        spine.set_color(PALETTE["frame"])
        spine.set_linewidth(0.9)
    ax.tick_params(axis="x", length=0, colors=PALETTE["ink"], pad=6)
    ax.tick_params(axis="y", colors=PALETTE["ink"])


def export_figure(rows: pd.DataFrame, totals: dict) -> dict[str, str]:
    configure_matplotlib()
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=180)
    add_waterfall(ax, rows, totals)
    fig.subplots_adjust(left=0.105, right=0.955, bottom=0.20, top=0.88)

    png_path = FIG_DIR / f"{FIGURE_STEM}.png"
    svg_path = FIG_DIR / f"{FIGURE_STEM}.svg"
    pdf_path = FIG_DIR / f"{FIGURE_STEM}.pdf"
    white_path = FIG_DIR / f"{FIGURE_STEM}_white_preview.png"
    fig.savefig(png_path, dpi=600, bbox_inches="tight", pad_inches=0.025, transparent=True)
    fig.savefig(svg_path, bbox_inches="tight", pad_inches=0.025, transparent=True)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.025, transparent=True)
    fig.savefig(white_path, dpi=220, bbox_inches="tight", pad_inches=0.025, facecolor="white", transparent=False)
    plt.close(fig)

    with Image.open(png_path) as img:
        if not (img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info)):
            raise ValueError(f"Transparent PNG does not expose alpha channel, mode={img.mode}")
        alpha_info = {"mode": img.mode, "size": list(img.size)}

    return {
        "transparent_png": rel(png_path),
        "svg": rel(svg_path),
        "pdf": rel(pdf_path),
        "white_preview": rel(white_path),
        "alpha_check": json.dumps(alpha_info, ensure_ascii=False),
    }


def write_tables_and_readme(rows: pd.DataFrame, totals: dict, outputs: dict[str, str]) -> dict[str, str]:
    plot_data = TABLE_DIR / f"{FIGURE_STEM}_plot_data.csv"
    source_summary = TABLE_DIR / f"{FIGURE_STEM}_source_q10_summary.csv"
    rows.to_csv(plot_data, index=False, encoding="utf-8-sig")
    pd.DataFrame([totals]).to_csv(source_summary, index=False, encoding="utf-8-sig")

    readme_path = STAGE_DIR / f"README_{FIGURE_STEM}.md"
    readme = f"""# {FIGURE_ID} candidate: q10 land-cover area budget

## Status

- Candidate only; not inserted into Word.
- Layout family: `{LAYOUT_FAMILY}`.
- Main chain: selected10 HGB with HydroRIVERS `DIS_AV_CMS >= 10 m3 s-1`.
- Completeness: q10 Stage20 reports {totals["q10_tiles_success"]}/{totals["q10_tiles_total"]} tiles successful and {totals["q10_failed_rows"]} failed rows.

## Scientific message

This figure shows how the q10 hydrology-constrained suitability envelope changes after ESA WorldCover land-cover compatibility filtering:

- q10 hydro-spatial envelope: {totals["q10_stage17_area_million_km2"]:.4f} million km2.
- compatible_pct >= 50% binary land-cover envelope: {totals["q10_binary50_area_million_km2"]:.4f} million km2.
- weighted land-cover-compatible area: {totals["q10_weighted_area_million_km2"]:.4f} million km2 ({totals["q10_weighted_area_wan_km2"]:.2f} wan km2).
- 2020 oasis reference line: approximately {totals["oasis_2020_reference_million_km2"]:.2f} million km2. This reference is used only as manuscript context and should be cited from the source 2020 oasis mapping dataset in text.

## Data lineage

- Stage36 hydrology summary: `{totals["source_summary_csv"]}`.
- Stage36 status: `{totals["source_stage36_status_json"]}`.
- q10 Stage20 status: `{totals["source_q10_status_json"]}`.

## Outputs

- Transparent PNG: `{outputs["transparent_png"]}`.
- SVG: `{outputs["svg"]}`.
- PDF: `{outputs["pdf"]}`.
- White preview: `{outputs["white_preview"]}`.
- Plot data: `{rel(plot_data)}`.
- Source summary table: `{rel(source_summary)}`.

## Notes for caption context

The q10 result is the manuscript main area estimate. q25 remains the strict hydrological sensitivity bound, and q1 is retained only as a broad reference. The 2020 reference line is approximate and should not be described as a recalculated output of this script.
"""
    readme_path.write_text(readme, encoding="utf-8")
    return {
        "plot_data": rel(plot_data),
        "source_summary": rel(source_summary),
        "readme": rel(readme_path),
    }


def main() -> int:
    setup_dirs()
    log_path = setup_logging()
    status_path = STAGE_DIR / f"{FIGURE_STEM}_status.json"
    write_json_atomic(
        status_path,
        {
            "status": "running",
            "figure_id": FIGURE_ID,
            "layout_family": LAYOUT_FAMILY,
            "started_at": now_iso(),
            "log": rel(log_path),
        },
    )

    try:
        logging.info("Loading q10 Stage36 values")
        rows, totals, q10_source = load_q10_values()
        outputs = export_figure(rows, totals)
        tables = write_tables_and_readme(rows, totals, outputs)

        status = {
            "status": "success",
            "figure_id": FIGURE_ID,
            "figure_stem": FIGURE_STEM,
            "layout_family": LAYOUT_FAMILY,
            "generated_at": now_iso(),
            "outputs": outputs,
            "tables": tables,
            "source": {
                "stage36_summary_csv": rel(SUMMARY_CSV),
                "stage36_status_json": rel(STAGE36_STATUS_JSON),
                "q10_status_json": rel(Q10_STATUS_JSON),
                "q10_source_row": q10_source,
            },
            "key_values": totals,
            "checks": {
                "q10_status_success": True,
                "q10_tile_completeness": f"{totals['q10_tiles_success']}/{totals['q10_tiles_total']}",
                "png_alpha": json.loads(outputs["alpha_check"]),
                "candidate_not_inserted_into_word": True,
            },
            "warnings": [
                "The 2020 oasis reference is an approximate manuscript comparator and should be cited in text."
            ],
            "log": rel(log_path),
        }
        write_json_atomic(status_path, status)
        update_layout_ledger(
            "candidate_generated_not_inserted",
            "q10 land-cover area budget uses a single waterfall chart with a 2020 reference line; candidate only, not inserted into Word.",
        )
        logging.info("Generated %s", FIGURE_STEM)
        print(json.dumps(status, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        tb = traceback.format_exc()
        logging.error("Failed to generate %s: %s\n%s", FIGURE_STEM, exc, tb)
        write_json_atomic(
            status_path,
            {
                "status": "failed",
                "figure_id": FIGURE_ID,
                "figure_stem": FIGURE_STEM,
                "layout_family": LAYOUT_FAMILY,
                "failed_at": now_iso(),
                "error": str(exc),
                "traceback": tb,
                "log": rel(log_path),
            },
        )
        update_layout_ledger(
            "failed",
            f"q10 land-cover area budget generation failed: {exc}",
        )
        print(tb)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
