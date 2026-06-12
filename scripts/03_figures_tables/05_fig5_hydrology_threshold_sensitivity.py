#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Create Stage50 Fig5 candidate: q10 hydrology threshold sensitivity.

This figure uses the completed Stage36 q1/q10/q25 hydrology plus land-cover
summary. It is intentionally a single-panel threshold response chart so the
manuscript does not repeat the previous 2x2 or four-panel figure rhythm.
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

FIGURE_STEM = "fig_stage50_fig5_q10_hydrology_threshold_sensitivity_v03"
FIGURE_ID = "Fig5_stage50_v03"
LAYOUT_FAMILY = "single-threshold-response-slopegraph"

STAGE36_SUMMARY_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "stage36_hydrology_landcover_sensitivity"
    / "tables"
    / "stage36_hydrology_landcover_sensitivity_summary.csv"
)

STATUS_JSON = STAGE_DIR / f"{FIGURE_STEM}_status.json"
README_MD = STAGE_DIR / f"README_{FIGURE_STEM}.md"


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
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    root_logger.addHandler(file_handler)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    root_logger.addHandler(console)
    return log_path


def load_stage36_summary() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if not STAGE36_SUMMARY_CSV.exists():
        raise FileNotFoundError(f"Missing Stage36 summary table: {STAGE36_SUMMARY_CSV}")

    summary = pd.read_csv(STAGE36_SUMMARY_CSV)
    required_scenarios = ["q1_current_reference", "q10_main_candidate", "q25_strict_backup"]
    required_columns = [
        "scenario",
        "role",
        "min_discharge_cms",
        "summary_status",
        "rows",
        "ok_rows",
        "failed_rows",
        "stage17_area_km2",
        "binary50_area_km2",
        "weighted_area_km2",
        "weighted_pct_vs_q1",
    ]
    missing_columns = [col for col in required_columns if col not in summary.columns]
    if missing_columns:
        raise ValueError(f"Stage36 summary is missing required columns: {missing_columns}")

    present = set(summary["scenario"].astype(str))
    missing_scenarios = [name for name in required_scenarios if name not in present]
    if missing_scenarios:
        raise ValueError(f"Stage36 summary is missing scenarios: {missing_scenarios}")

    selected = summary[summary["scenario"].isin(required_scenarios)].copy()
    selected["scenario"] = pd.Categorical(
        selected["scenario"], categories=required_scenarios, ordered=True
    )
    selected = selected.sort_values("scenario").reset_index(drop=True)

    if not selected["summary_status"].eq("success").all():
        failed = selected.loc[~selected["summary_status"].eq("success"), "scenario"].tolist()
        raise ValueError(f"Stage36 contains non-success scenario rows: {failed}")
    if not selected["failed_rows"].fillna(0).eq(0).all():
        raise ValueError("Stage36 summary contains failed tile rows; figure cannot be treated as clean.")
    if not selected["rows"].eq(171).all() or not selected["ok_rows"].eq(171).all():
        raise ValueError("Stage36 summary does not report 171/171 completed tiles for all scenarios.")

    scenario_meta = {
        "q1_current_reference": {
            "label": "q1 reference",
            "short": "q1",
            "threshold": ">=1 m3/s",
            "role": "broad reference only",
            "color": "#8A9AA5",
            "marker": "o",
        },
        "q10_main_candidate": {
            "label": "q10 main",
            "short": "q10",
            "threshold": ">=10 m3/s",
            "role": "main hydrology threshold",
            "color": "#167F7A",
            "marker": "D",
        },
        "q25_strict_backup": {
            "label": "q25 strict",
            "short": "q25",
            "threshold": ">=25 m3/s",
            "role": "strict sensitivity backup",
            "color": "#B7654A",
            "marker": "s",
        },
    }
    area_meta = [
        ("stage17_area_km2", "Hydro-spatial constraint", "before land-cover filter"),
        ("binary50_area_km2", "Binary land-cover envelope", "compatible pct >= 50"),
        ("weighted_area_km2", "Weighted compatible area", "land-cover weighted main result"),
    ]

    records: list[dict] = []
    for _, row in selected.iterrows():
        scenario = str(row["scenario"])
        for area_col, area_label, area_note in area_meta:
            records.append(
                {
                    "scenario": scenario,
                    "scenario_label": scenario_meta[scenario]["label"],
                    "scenario_short": scenario_meta[scenario]["short"],
                    "min_discharge_cms": float(row["min_discharge_cms"]),
                    "threshold_label": scenario_meta[scenario]["threshold"],
                    "scenario_role": scenario_meta[scenario]["role"],
                    "area_metric": area_col,
                    "area_label": area_label,
                    "area_note": area_note,
                    "area_km2": float(row[area_col]),
                    "area_million_km2": float(row[area_col]) / 1_000_000,
                    "area_wan_km2": float(row[area_col]) / 10_000,
                    "weighted_pct_vs_q1": float(row["weighted_pct_vs_q1"]),
                    "status": str(row["summary_status"]),
                    "source_csv": rel(STAGE36_SUMMARY_CSV),
                }
            )
    long_df = pd.DataFrame.from_records(records)

    totals = {
        "q10_weighted_area_million_km2": float(
            selected.loc[selected["scenario"].eq("q10_main_candidate"), "weighted_area_km2"].iloc[0]
        )
        / 1_000_000,
        "q10_weighted_area_wan_km2": float(
            selected.loc[selected["scenario"].eq("q10_main_candidate"), "weighted_area_km2"].iloc[0]
        )
        / 10_000,
        "q25_weighted_area_million_km2": float(
            selected.loc[selected["scenario"].eq("q25_strict_backup"), "weighted_area_km2"].iloc[0]
        )
        / 1_000_000,
        "q1_weighted_area_million_km2": float(
            selected.loc[selected["scenario"].eq("q1_current_reference"), "weighted_area_km2"].iloc[0]
        )
        / 1_000_000,
    }
    return selected, long_df, totals


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.2,
            "figure.dpi": 160,
            "savefig.dpi": 600,
            "axes.linewidth": 0.8,
            "axes.edgecolor": "#D7DFDA",
            "xtick.color": "#28343D",
            "ytick.color": "#28343D",
            "text.color": "#28343D",
            "axes.labelcolor": "#28343D",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def draw_threshold_response(long_df: pd.DataFrame, totals: dict) -> dict[str, Path]:
    configure_style()
    paths = {
        "png": FIG_DIR / f"{FIGURE_STEM}.png",
        "svg": FIG_DIR / f"{FIGURE_STEM}.svg",
        "pdf": FIG_DIR / f"{FIGURE_STEM}.pdf",
        "white_preview": FIG_DIR / f"{FIGURE_STEM}_white_preview.png",
    }

    scenario_order = ["q1_current_reference", "q10_main_candidate", "q25_strict_backup"]
    area_order = [
        "Hydro-spatial constraint",
        "Binary land-cover envelope",
        "Weighted compatible area",
    ]
    y_positions = {name: idx for idx, name in enumerate(reversed(area_order))}
    colors = {
        "q1_current_reference": "#8A9AA5",
        "q10_main_candidate": "#167F7A",
        "q25_strict_backup": "#B7654A",
    }
    marker = {
        "q1_current_reference": "o",
        "q10_main_candidate": "D",
        "q25_strict_backup": "s",
    }
    labels = {
        "q1_current_reference": "q1 reference",
        "q10_main_candidate": "q10 main",
        "q25_strict_backup": "q25 strict",
    }

    fig, ax = plt.subplots(figsize=(7.35, 3.85))
    fig.patch.set_alpha(0)
    ax.set_facecolor("white")

    for area_label in area_order:
        subset = long_df[long_df["area_label"].eq(area_label)].set_index("scenario").loc[scenario_order]
        y = y_positions[area_label]
        xs = subset["area_million_km2"].to_numpy(dtype=float)
        ax.plot(xs, [y] * len(xs), color="#C8D2D0", linewidth=2.4, zorder=1, solid_capstyle="round")
        for scenario in scenario_order:
            row = subset.loc[scenario]
            x = float(row["area_million_km2"])
            size = 72 if scenario == "q10_main_candidate" else 54
            ax.scatter(
                [x],
                [y],
                s=size,
                color=colors[scenario],
                marker=marker[scenario],
                edgecolor="white",
                linewidth=1.0,
                zorder=3,
                label=labels[scenario] if area_label == area_order[0] else None,
            )

    # Direct labels use fixed offsets by scenario so they stay near markers without touching them.
    label_offsets = {
        "q1_current_reference": (0.045, 0.055),
        "q10_main_candidate": (0.050, -0.010),
        "q25_strict_backup": (-0.060, -0.060),
    }
    horizontal_alignment = {
        "q1_current_reference": "left",
        "q10_main_candidate": "left",
        "q25_strict_backup": "right",
    }
    for _, row in long_df.iterrows():
        scenario = str(row["scenario"])
        area_label = str(row["area_label"])
        x = float(row["area_million_km2"])
        y = y_positions[area_label]
        dx, dy = label_offsets[scenario]
        ha = horizontal_alignment[scenario]
        if scenario == "q25_strict_backup" and area_label == "Hydro-spatial constraint":
            dx, dy = 0.055, -0.055
            ha = "left"
        value_text = f"{x:.2f}"
        ax.text(
            x + dx,
            y + dy,
            value_text,
            ha=ha,
            va="center",
            fontsize=8.1,
            color=colors[scenario],
            fontweight="bold" if scenario == "q10_main_candidate" else "normal",
            zorder=4,
        )

    q10_x = float(totals["q10_weighted_area_million_km2"])
    ax.axvspan(q10_x - 0.035, q10_x + 0.035, color="#167F7A", alpha=0.08, zorder=0)
    ax.axvline(q10_x, color="#167F7A", linewidth=1.0, linestyle=(0, (3, 3)), alpha=0.85, zorder=0)
    ax.text(
        q10_x + 0.040,
        -0.28,
        "q10 main threshold",
        ha="left",
        va="bottom",
        fontsize=7.6,
        color="#167F7A",
        fontweight="bold",
    )

    ax.set_yticks([y_positions[name] for name in area_order])
    ax.set_yticklabels(area_order)
    ax.set_xlabel("Future suitable area after hydrological thresholding (million km2)")
    ax.set_xlim(1.72, 4.18)
    ax.set_ylim(-0.56, 2.50)
    ax.set_xticks(np.arange(1.8, 4.21, 0.4))
    ax.grid(axis="x", color="#E4EAEA", linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend = ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.025),
        ncol=3,
        frameon=False,
        handletextpad=0.45,
        columnspacing=1.4,
        borderaxespad=0.0,
    )
    for text in legend.get_texts():
        text.set_color("#28343D")

    fig.subplots_adjust(left=0.24, right=0.985, bottom=0.21, top=0.81)
    fig.savefig(paths["png"], dpi=600, bbox_inches="tight", pad_inches=0.055, transparent=True)
    fig.savefig(paths["svg"], bbox_inches="tight", pad_inches=0.055, transparent=True)
    fig.savefig(paths["pdf"], bbox_inches="tight", pad_inches=0.055, transparent=True)

    fig.patch.set_alpha(1)
    fig.patch.set_facecolor("white")
    fig.savefig(paths["white_preview"], dpi=300, bbox_inches="tight", pad_inches=0.065, facecolor="white")
    plt.close(fig)
    return paths


def verify_alpha_png(path: Path) -> dict:
    img = Image.open(path)
    info = {
        "mode": img.mode,
        "size": img.size,
        "has_alpha": img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info),
    }
    if not info["has_alpha"]:
        raise ValueError(f"PNG does not contain transparency: {path}")
    return info


def write_readme(paths: dict[str, Path], selected: pd.DataFrame, totals: dict) -> Path:
    table_text = selected[
        [
            "scenario",
            "min_discharge_cms",
            "stage17_area_wan_km2",
            "binary50_area_wan_km2",
            "weighted_area_wan_km2",
            "weighted_pct_vs_q1",
        ]
    ].to_markdown(index=False, floatfmt=".2f")
    readme = f"""# {FIGURE_ID} candidate: q10 hydrology threshold sensitivity

Status: candidate, not inserted into Word.

This package updates the manuscript figure set for the selected10 HGB main chain after the hydrological threshold decision. It uses `DIS_AV_CMS >= 10 m3/s` as the main HydroRIVERS threshold and keeps `>=25 m3/s` as the strict sensitivity backup. `>=1 m3/s` is shown only as the broad reference.

## Source data

- Stage36 summary: `{rel(STAGE36_SUMMARY_CSV)}`
- Completion condition: all three scenarios are `success`, all report 171/171 completed rows, and failed rows are zero.

## Values used

{table_text}

## Interpretation guardrails

- q10 weighted compatible area: {totals["q10_weighted_area_million_km2"]:.4f} million km2 ({totals["q10_weighted_area_wan_km2"]:.2f} wan km2).
- q25 strict backup weighted compatible area: {totals["q25_weighted_area_million_km2"]:.4f} million km2.
- q1 is retained as a broad reference and must not be written as the main conclusion.
- The long q1/q10/q25 role note is kept in this README and should be handled in the figure caption, not inside the figure, to avoid manuscript-size text collisions.
- Candidate figure only: do not insert into Word until the user confirms the final figure.

## Exports

- Transparent PNG: `{rel(paths["png"])}`
- SVG: `{rel(paths["svg"])}`
- PDF: `{rel(paths["pdf"])}`
- White preview: `{rel(paths["white_preview"])}`
- Plot data: `{rel(TABLE_DIR / (FIGURE_STEM + "_plot_data.csv"))}`
- Source summary copy: `{rel(TABLE_DIR / (FIGURE_STEM + "_source_summary.csv"))}`

## Layout note

Layout family: `{LAYOUT_FAMILY}`. This is a single-panel threshold response chart and is intentionally different from the existing 2x2 statistical plates and prior map/funnel figures.
"""
    README_MD.write_text(readme, encoding="utf-8")
    return README_MD


def main() -> int:
    setup_dirs()
    log_path = setup_logging()
    status = {
        "figure_id": FIGURE_ID,
        "figure_stem": FIGURE_STEM,
        "layout_family": LAYOUT_FAMILY,
        "stage_dir": rel(STAGE_DIR),
        "status": "running",
        "started_at": now_iso(),
        "source_script": rel(Path(__file__)),
        "source_data": rel(STAGE36_SUMMARY_CSV),
        "word_insertion": "not_inserted_candidate_waiting_for_user_confirmation",
        "outputs": {},
        "checks": {},
        "warnings": [],
        "errors": [],
    }
    write_json_atomic(STATUS_JSON, status)

    try:
        logging.info("Loading Stage36 hydrology-landcover sensitivity summary")
        selected, long_df, totals = load_stage36_summary()
        source_summary_path = TABLE_DIR / f"{FIGURE_STEM}_source_summary.csv"
        plot_data_path = TABLE_DIR / f"{FIGURE_STEM}_plot_data.csv"
        selected.to_csv(source_summary_path, index=False, encoding="utf-8-sig")
        long_df.to_csv(plot_data_path, index=False, encoding="utf-8-sig")

        logging.info("Rendering threshold response figure")
        paths = draw_threshold_response(long_df, totals)
        alpha_info = verify_alpha_png(paths["png"])
        readme_path = write_readme(paths, selected, totals)

        status.update(
            {
                "status": "success",
                "finished_at": now_iso(),
                "outputs": {key: rel(value) for key, value in paths.items()}
                | {
                    "plot_data_csv": rel(plot_data_path),
                    "source_summary_csv": rel(source_summary_path),
                    "readme": rel(readme_path),
                    "log": rel(log_path),
                },
                "checks": {
                    "alpha_png": alpha_info,
                    "scenario_count": int(selected.shape[0]),
                    "plot_rows": int(long_df.shape[0]),
                    "all_summary_status_success": bool(selected["summary_status"].eq("success").all()),
                    "failed_rows_total": int(selected["failed_rows"].fillna(0).sum()),
                    "q10_weighted_area_million_km2": round(
                        totals["q10_weighted_area_million_km2"], 4
                    ),
                    "q25_weighted_area_million_km2": round(
                        totals["q25_weighted_area_million_km2"], 4
                    ),
                },
            }
        )
        write_json_atomic(STATUS_JSON, status)
        logging.info("Figure package created successfully: %s", paths["png"])
        return 0
    except Exception as exc:
        status.update(
            {
                "status": "failed",
                "finished_at": now_iso(),
                "errors": [str(exc)],
                "traceback": traceback.format_exc(),
                "outputs": status.get("outputs", {}) | {"log": rel(log_path)},
            }
        )
        write_json_atomic(STATUS_JSON, status)
        logging.exception("Figure package creation failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
