# -*- coding: utf-8 -*-
"""Create Fig. 1 v72 by minimally updating the original Stage24 workflow figure.

This script deliberately reuses the original Stage24 drawing code and only
overrides small text/data modules that became outdated after the selected10 +
q10 hydrology decision. It preserves earlier outputs and writes a new v72 package.
This revision aligns row and module titles within their header bands and shortens
the second row heading for manual review. It also adds a standalone
field-survey presence-check step after model validation, not as a model-training
input. The v65 pass applies the user's manual alignment notes: the three sample
row boxes share the Presence box geometry, model-validation nodes are enlarged
and centered, and the Field survey icon/text are enlarged.
The v66 pass applies the user's next manual notes: sample-row text is moved up,
model-validation nodes are moved down, Field survey icon/text are enlarged again,
and the Main chain icon is changed so it is no longer the same check icon.
The v67 pass removes the standalone upper-left panel letter and enlarges the key
row-two icons. The v68 pass followed the previous manual markup. The v69 pass
corrects the reversed instruction: yellow-marked validation/check pictorial
elements are enlarged by 20%, while the red-marked Presence, Background, and
Training icons are reduced by 5% with text spacing adjusted.
The v70 pass applies the latest browser annotation: the yellow-marked
Presence/Background/Training pictorial icons are reduced by 10%, the
blue-circled model-validation nodes are lowered with more label spacing, and
the green Field survey check icon is enlarged while its text is shifted left
and aligned.
The v71 pass applies the latest flowchart annotation: the yellow-marked result
labels are enlarged, row-two arrows are matched to the third arrow length/style,
the first three row-two boxes are narrowed by 5% to preserve spacing, and the
red-marked CV + LORO text is moved upward.
The v72 pass applies the next manual markups: the yellow-marked CMIP6 caption is
moved upward, the green Field survey icon is lowered, its text is shifted left,
and the local alignment is retuned.
"""

from __future__ import annotations

import csv
import importlib.util
import json
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_SCRIPT = PROJECT_ROOT / "scripts" / "24_生成图标增强版Nature方法框架图.py"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "stage24_icon_enhanced_method_framework"
FIGURE_DIR = OUTPUT_ROOT / "figures"
TABLE_DIR = OUTPUT_ROOT / "tables"
LOG_DIR = OUTPUT_ROOT / "logs"

FIGURE_STEM = "fig_stage24_icon_enhanced_method_framework_v72_text_icon_alignment"

AREA_ROWS = [
    {
        "label": "q10 hydro",
        "value_wan_km2": 302.46,
        "meaning": "selected10 HGB hydro-spatial envelope after HydroRIVERS DIS_AV_CMS >= 10 m3 s-1",
    },
    {
        "label": "LC >=50%",
        "value_wan_km2": 238.74,
        "meaning": "binary envelope where ESA WorldCover compatible_pct >= 50%",
    },
    {
        "label": "weighted",
        "value_wan_km2": 236.34,
        "meaning": "land-cover weighted compatible area; manuscript main area result",
    },
]


def load_original_module():
    spec = importlib.util.spec_from_file_location("stage24_original_workflow", ORIGINAL_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load original Stage24 script: {ORIGINAL_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def patch_stage24(stage24):
    original_save = stage24.save
    original_process_box_large_icon = stage24.process_box_large_icon
    original_text = stage24.text
    original_icon_raster = stage24.icon_raster

    def save_v56(fig, _stem):
        return original_save(fig, FIGURE_STEM)

    def text_v60(ax, x, y, s, *args, **kwargs):
        replacements = {
            "Analysis inputs": "20 factors",
            "Sample construction and model selection": "Samples, VIF and models",
            "Scenario projection and spatial constraints": "Projection and constraints",
            "Constrained map": "Constrained map",
        }
        title_y_offsets = {
            "Input data layers": -0.06,
            "Sample construction and model selection": -0.08,
            "Scenario projection and spatial constraints": -0.05,
            "Manuscript-ready products": -0.07,
            "Presence": -0.035,
            "Background": -0.035,
            "Training": -0.035,
            "Model validation": -0.03,
            "Main chain": -0.03,
            "CMIP6 futures": -0.035,
            "Raster projection": -0.03,
            "Thresholding": -0.03,
            "Geo-hydro + LC": -0.025,
            "Constrained map": -0.025,
            "Area results": -0.035,
            "Spatial products": -0.03,
            "Sensitivity": -0.03,
            "Paper package": -0.03,
        }
        new_s = replacements.get(s, s)
        y += title_y_offsets.get(s, 0.0)
        if s == "Analysis inputs":
            kwargs["size"] = 5.65
        return original_text(ax, x, y, new_s, *args, **kwargs)

    def process_box_large_icon_v56(ax, x, y, w, h, title, body, fc, icon):
        if title == "Training":
            body = "selected10\nspatial CV"
        return original_process_box_large_icon(ax, x, y, w, h, title, body, fc, icon)

    def row_process_box_v65(ax, x, y, w, h, title, lines, icon):
        stage24.box(ax, x, y, w, h, fc=stage24.COL["sample"], ec=stage24.COL["line"], lw=0.55, radius=0.035)
        stage24.text(ax, x + w / 2, y + h - 0.13, title, size=5.35, weight="bold")
        icon(ax, x + 0.23, y + 0.16, 0.530, 0.530)
        if isinstance(lines, str):
            lines = lines.splitlines()
        text_x = x + 0.88
        if title == "Training":
            text_x = x + 0.91
        if len(lines) == 1:
            stage24.text(ax, text_x, y + 0.43, lines[0], size=4.15, ha="left")
        else:
            stage24.text(ax, text_x, y + 0.50, lines[0], size=4.25, ha="left")
            stage24.text(ax, text_x, y + 0.33, lines[1], size=4.25, ha="left")

    def icon_model_v65(ax, x, y, w, h):
        stage24.box(ax, x, y, w, h, fc=stage24.COL["model"], ec=stage24.COL["line"], lw=0.55, radius=0.035)
        stage24.text(ax, x + w / 2, y + h - 0.19, "Model validation", size=4.55, weight="bold")
        nodes = [("HGB", "#B2DF8A", 0.23), ("RF", "#A6CEE3", 0.50), ("GLM", "#FDBF6F", 0.77)]
        cy = y + h * 0.47
        ax.plot([x + w * px for _, _, px in nodes], [cy] * len(nodes), color="#888888", lw=0.24, zorder=0)
        for lab, col, px in nodes:
            ax.add_patch(stage24.Circle((x + w * px, cy), 0.18, facecolor=col, edgecolor=stage24.COL["line"], lw=0.42))
            stage24.text(ax, x + w * px, cy, lab, size=4.25)
        stage24.text(ax, x + w / 2, y + 0.165, "CV + LORO", size=4.25)

    def field_survey_box_v72(ax, x, y, w, h):
        stage24.box(ax, x, y, w, h, fc="#E8F3E7", ec=stage24.COL["line"], lw=0.55, radius=0.035)
        stage24.text(ax, x + w / 2, y + h - 0.15, "Field survey", size=4.95, weight="bold")
        if not stage24.draw_user_png_icon(ax, "mainmode_user_20260531_165204_transparent.png", x + 0.00, y + 0.09, 0.72, 0.72, z=6):
            stage24.svg_inbox_check_icon(ax, x + 0.04, y + 0.14, 0.62, 0.62)
        text_x = x + 0.705
        stage24.text(ax, text_x, y + 0.490, "presence", size=4.25, ha="left")
        stage24.text(ax, text_x, y + 0.320, "check", size=4.25, ha="left")

    def mini_main_chain_icon_v66(ax, x, y, w, h):
        ax.add_patch(stage24.Rectangle((x, y), w, h, facecolor="#E7F5F3", edgecolor="#00897B", lw=0.42, zorder=6))
        pts = [
            (x + w * 0.18, y + h * 0.68),
            (x + w * 0.42, y + h * 0.50),
            (x + w * 0.66, y + h * 0.63),
            (x + w * 0.83, y + h * 0.35),
        ]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color="#00796B", lw=0.70, zorder=7, solid_capstyle="round")
        for px, py in pts[:-1]:
            ax.add_patch(stage24.Circle((px, py), min(w, h) * 0.075, facecolor="#21A389", edgecolor="#00695C", lw=0.18, zorder=8))
        ax.add_patch(
            stage24.Polygon(
                [
                    (pts[-1][0], pts[-1][1]),
                    (pts[-1][0] - w * 0.10, pts[-1][1] + h * 0.10),
                    (pts[-1][0] - w * 0.03, pts[-1][1] - h * 0.12),
                ],
                closed=True,
                facecolor="#00796B",
                edgecolor="none",
                zorder=8,
            )
        )

    def main_model_box_v56(ax, x, y, w, h):
        stage24.box(ax, x, y, w, h, fc=stage24.COL["model"], ec=stage24.COL["line"], lw=0.55, radius=0.035)
        stage24.text(ax, x + w / 2, y + h - 0.18, "Main chain", size=4.75, weight="bold")
        mini_main_chain_icon_v66(ax, x + 0.10, y + 0.15, 0.52, 0.47)
        stage24.text(ax, x + 0.69, y + 0.45, "HGB main", size=4.25, ha="left")
        stage24.text(ax, x + 0.69, y + 0.29, "RF sens.", size=4.25, ha="left")

    def icon_scenarios_v72(ax, x, y, w, h):
        stage24.box(ax, x, y, w, h, fc=stage24.COL["future"], ec=stage24.COL["line"], lw=0.55, radius=0.035)
        stage24.text(ax, x + w / 2, y + h - 0.15, "CMIP6 futures", size=5.7, weight="bold")
        ssps = [("126", "#7BC8A4"), ("245", "#74A9CF"), ("370", "#E9C46A"), ("585", "#D95F02")]
        for i, (lab, col) in enumerate(ssps):
            bx = x + 0.18 + i * (w - 0.46) / 4
            stage24.box(ax, bx, y + 0.35, 0.28, 0.32, fc=col, ec="white", lw=0.15, radius=0.02)
            stage24.text(ax, bx + 0.14, y + 0.51, lab, size=4.25, color="#222")
        stage24.text(ax, x + w / 2, y + 0.17, "GCM x SSP x period", size=4.7)

    def constrained_map_thumbnail_v59(ax, x, y, w, h):
        ax.add_patch(stage24.Rectangle((x, y), w, h, facecolor="#F7F1DF", edgecolor="#BBBBBB", lw=0.18))
        for frac, width, col in [
            (0.00, 0.27, "#E4D1A4"),
            (0.27, 0.22, "#D7C291"),
            (0.49, 0.25, "#D1E2BC"),
            (0.74, 0.26, "#EFE7D1"),
        ]:
            ax.add_patch(
                stage24.Rectangle(
                    (x + w * frac, y),
                    w * width,
                    h,
                    facecolor=col,
                    edgecolor="white",
                    lw=0.055,
                    alpha=0.88,
                )
            )
        foothill = [
            (x + w * 0.00, y + h * 0.70),
            (x + w * 0.16, y + h * 1.00),
            (x + w * 0.42, y + h * 0.76),
            (x + w * 0.27, y + h * 0.58),
        ]
        ax.add_patch(stage24.Polygon(foothill, closed=True, facecolor="#BFA77F", edgecolor="none", alpha=0.42))
        halo = [
            (x + w * 0.12, y + h * 0.36),
            (x + w * 0.42, y + h * 0.67),
            (x + w * 0.73, y + h * 0.48),
            (x + w * 0.64, y + h * 0.23),
            (x + w * 0.25, y + h * 0.20),
        ]
        core = [
            (x + w * 0.20, y + h * 0.41),
            (x + w * 0.44, y + h * 0.58),
            (x + w * 0.63, y + h * 0.49),
            (x + w * 0.55, y + h * 0.33),
            (x + w * 0.29, y + h * 0.30),
        ]
        secondary = [
            (x + w * 0.67, y + h * 0.35),
            (x + w * 0.92, y + h * 0.50),
            (x + w * 0.86, y + h * 0.32),
            (x + w * 0.70, y + h * 0.23),
        ]
        ax.add_patch(stage24.Polygon(halo, closed=True, facecolor="#B9DCA8", edgecolor="none", alpha=0.82, zorder=2))
        ax.add_patch(stage24.Polygon(core, closed=True, facecolor="#3F9A5B", edgecolor="#2F7D48", lw=0.12, alpha=0.98, zorder=3))
        ax.add_patch(stage24.Polygon(secondary, closed=True, facecolor="#58B676", edgecolor="#2F7D48", lw=0.10, alpha=0.95, zorder=3))
        river_x = [x + w * 0.07, x + w * 0.25, x + w * 0.42, x + w * 0.58, x + w * 0.78, x + w * 0.95]
        river_y = [y + h * 0.27, y + h * 0.38, y + h * 0.34, y + h * 0.46, y + h * 0.39, y + h * 0.55]
        ax.plot(river_x, river_y, color="#9DD3E8", lw=1.00, solid_capstyle="round", zorder=4)
        ax.plot(river_x, river_y, color=stage24.COL["blue"], lw=0.42, solid_capstyle="round", zorder=5)
        branch_x = [x + w * 0.43, x + w * 0.57, x + w * 0.69]
        branch_y = [y + h * 0.35, y + h * 0.27, y + h * 0.24]
        ax.plot(branch_x, branch_y, color="#9DD3E8", lw=0.72, solid_capstyle="round", zorder=4)
        ax.plot(branch_x, branch_y, color=stage24.COL["blue"], lw=0.30, solid_capstyle="round", zorder=5)

    def icon_raster_v59(ax, x, y, w, h, title, fc, variant="projection"):
        if title == "Constrained map" and variant == "constrained":
            stage24.box(ax, x, y, w, h, fc=fc, ec=stage24.COL["line"], lw=0.55, radius=0.035)
            stage24.text(ax, x + w / 2, y + h - 0.18, title, size=5.05, weight="bold")
            mx, my, mw, mh = x + 0.20, y + 0.29, w - 0.40, h - 0.66
            constrained_map_thumbnail_v59(ax, mx, my, mw, mh)
            stage24.text(ax, x + w / 2, y + 0.13, "masked suitability", size=4.25)
            return
        return original_icon_raster(ax, x, y, w, h, title, fc, variant=variant)

    def icon_constraint_stack_v56(ax, x, y, w, h):
        stage24.box(ax, x, y, w, h, fc=stage24.COL["constraint"], ec=stage24.COL["line"], lw=0.55, radius=0.035)
        stage24.text(ax, x + w / 2, y + h - 0.17, "Geo-hydro + LC", size=4.95, weight="bold")
        layers = [
            ("terrain mask", "#D9EAD3"),
            ("oasis proximity", "#E6F1DA"),
            ("HydroRIVERS q10", "#CFE2F3"),
            ("WorldCover weight", "#B6D7A8"),
        ]
        for i, (lab, col) in enumerate(layers):
            yy = y + h - 0.50 - i * 0.22
            stage24.box(ax, x + 0.20, yy, w - 0.40, 0.17, fc=col, ec="#888888", lw=0.25, radius=0.015)
            stage24.text(ax, x + w / 2, yy + 0.085, lab, size=4.25)

    def icon_area_v56(ax, x, y, w, h):
        stage24.box(ax, x, y, w, h, fc="#FBFAF7", ec=stage24.COL["line"], lw=0.55, radius=0.035)
        stage24.text(ax, x + w / 2, y + h - 0.16, "Area results", size=5.15, weight="bold")
        vals = [row["value_wan_km2"] for row in AREA_ROWS]
        colors = ["#5A9EC2", "#69B37B", "#4F9A57"]
        maxv = max(vals)
        row_icons = [stage24.mini_geohydro_icon, stage24.mini_lc_icon, stage24.mini_lc_icon]
        for i, (row, val, col, draw_icon) in enumerate(zip(AREA_ROWS, vals, colors, row_icons)):
            yy = y + h - 0.47 - i * 0.19
            draw_icon(ax, x + 0.18, yy - 0.02, 0.23, 0.16, frame=True)
            stage24.text(ax, x + 0.52, yy + 0.045, row["label"], size=4.25, ha="left", color="#555")
            bar_x = x + 1.22
            bar_w = (w - 1.95) * val / maxv
            ax.add_patch(stage24.Rectangle((bar_x, yy), bar_w, 0.09, facecolor=col, edgecolor="none"))
            stage24.text(ax, x + w - 0.26, yy + 0.045, f"{val:.0f}", size=4.25, ha="right")

    def result_dual_icon_box_v71(ax, x, y, w, h, title, items):
        if title == "Sensitivity":
            items = [("RF", stage24.mini_model_icon), ("q25", stage24.mini_threshold_icon)]
        stage24.box(ax, x, y, w, h, fc=stage24.COL["output"], ec=stage24.COL["line"], lw=0.55, radius=0.035)
        title_size = 4.85 if w <= 1.62 else 5.35
        stage24.text(ax, x + w / 2, y + h - 0.18, title, size=title_size, weight="bold")
        icon_size = min(0.48, w * 0.27)
        centers = [x + w * 0.32, x + w * 0.68]
        for cx, (lab, draw_func) in zip(centers, items):
            draw_func(ax, cx - icon_size / 2, y + 0.27, icon_size, icon_size, frame=False)
            stage24.text(ax, cx, y + 0.10, lab, size=4.25)

    def paper_package_box_v71(ax, x, y, w, h):
        stage24.box(ax, x, y, w, h, fc=stage24.COL["output"], ec=stage24.COL["line"], lw=0.55, radius=0.035)
        stage24.text(ax, x + w / 2, y + h - 0.18, "Paper package", size=5.35, weight="bold")
        inner_x, inner_w = x + 0.20, w - 0.40
        stage24.tiny_paper(ax, inner_x, y + 0.30, inner_w, 0.42)
        icon_centers = [inner_x + inner_w * p for p in (0.185, 0.500, 0.815)]
        for cx, lab in zip(icon_centers, ["figures", "tables", "text"]):
            stage24.text(ax, cx, y + 0.17, lab, size=4.25)

    def row_gap_arrow(ax, left_edge, right_edge, y=5.55, length=0.28):
        center = (left_edge + right_edge) / 2
        half = length / 2
        stage24.arrow(ax, (center - half, y), (center + half, y))

    def draw_framework_v65():
        stage24.plt.rcParams.update(
            {
                "font.family": "DejaVu Sans",
                "font.size": 5.5,
                "svg.fonttype": "none",
                "pdf.fonttype": 42,
                "axes.linewidth": 0.5,
            }
        )
        fig, ax = stage24.plt.subplots(figsize=(183 * stage24.MM, 145 * stage24.MM))
        ax.set_xlim(0, 11.6)
        ax.set_ylim(0, 9.7)
        ax.axis("off")
        panel_x, panel_w = 0.85, 10.25
        rows = [
            ("Data\npreparation", 7.00, 1.85, stage24.COL["stage1"], "Input data layers"),
            ("Model\nvalidation", 4.80, 1.75, stage24.COL["stage2"], "Sample construction and model selection"),
            ("Future\nprojection", 2.55, 1.75, stage24.COL["stage3"], "Scenario projection and spatial constraints"),
            ("Result\nsynthesis", 0.40, 1.70, stage24.COL["stage4"], "Manuscript-ready products"),
        ]
        for stage, y, h, c, title in rows:
            stage_x, stage_w = 0.16, 0.56
            ax.add_patch(stage24.Rectangle((stage_x, y + 0.08), stage_w, h - 0.16, facecolor=c, edgecolor=stage24.COL["line"], lw=0.55))
            stage24.stage_label(ax, stage_x + stage_w / 2, y + h / 2, stage)
            stage24.box(ax, panel_x, y, panel_w, h, fc="#FFFFFF", ec=stage24.COL["dash"], lw=0.55, radius=0.06, ls=(0, (3, 2)), z=0)
            stage24.text(ax, panel_x + 0.18, y + h - 0.18, title, size=5.8, weight="bold", ha="left")

        icons = [
            (1.12, stage24.icon_oasis),
            (2.45, stage24.icon_dryland),
            (3.78, stage24.icon_climate),
            (5.11, stage24.icon_terrain_vector),
            (6.44, stage24.icon_rivers_vector),
            (7.77, stage24.icon_landcover),
        ]
        for x, func in icons:
            func(ax, x, 7.24, 1.20, 1.14)
        stage24.box(ax, 9.45, 7.28, 1.55, 1.06, fc="#F8F8F8", ec=stage24.COL["line"], lw=0.50, radius=0.03)
        stage24.text(ax, 10.225, 8.08, "Analysis inputs", size=4.85, weight="bold")
        stage24.tiny_analysis_inputs_icon(ax, 9.55, 7.43, 1.35, 0.52)
        stage24.arrow(ax, (8.97, 7.81), (9.45, 7.81))

        row_process_box_v65(ax, 1.05, 5.07, 1.406, 0.95, "Presence", ["oasis", "patches"], stage24.tiny_points)
        row_process_box_v65(ax, 2.76, 5.07, 1.406, 0.95, "Background", ["dryland", "points"], stage24.tiny_background)
        row_process_box_v65(ax, 4.47, 5.07, 1.596, 0.95, "Training", ["selected10", "spatial CV"], stage24.tiny_table)
        stage24.icon_model(ax, 6.43, 4.98, 1.48, 1.12)
        field_survey_box_v72(ax, 8.19, 5.07, 1.28, 0.95)
        stage24.main_model_box(ax, 9.75, 5.07, 1.35, 0.95)
        row_gap_arrow(ax, 2.456, 2.76)
        row_gap_arrow(ax, 4.166, 4.47)
        row_gap_arrow(ax, 6.066, 6.43)
        row_gap_arrow(ax, 7.91, 8.19)
        row_gap_arrow(ax, 9.47, 9.75)
        stage24.stage_arrow(ax, 5.98, 6.775, length=0.34)

        stage24.icon_scenarios(ax, 1.18, 2.87, 1.65, 0.98)
        stage24.icon_raster(ax, 3.11, 2.87, 1.75, 0.98, "Raster projection", stage24.COL["future"], variant="projection")
        stage24.process_box(ax, 5.14, 2.87, 1.55, 0.98, "Thresholding", "potential\nsuitability", stage24.COL["future"], stage24.tiny_threshold, icon_size=0.52, icon_x=0.15, icon_y=0.14, text_x=0.78, text_y=0.39)
        stage24.icon_constraint_stack(ax, 6.97, 2.72, 1.70, 1.25)
        stage24.icon_raster(ax, 9.13, 2.87, 1.72, 0.98, "Constrained map", stage24.COL["constraint"], variant="constrained")
        stage24.arrow(ax, (2.83, 3.36), (3.11, 3.36))
        stage24.arrow(ax, (4.86, 3.36), (5.14, 3.36))
        stage24.arrow(ax, (6.69, 3.36), (6.97, 3.36))
        stage24.arrow(ax, (8.67, 3.36), (9.13, 3.36))
        stage24.stage_arrow(ax, 5.98, 4.55, length=0.34)
        stage24.stage_arrow(ax, 5.98, 2.325, length=0.34)

        result_y, result_h = 0.55, 1.06
        stage24.icon_area(ax, 1.18, result_y, 2.42, result_h)
        stage24.result_dual_icon_box(ax, 4.23, result_y, 1.61, result_h, "Spatial products", [("maps", stage24.mini_map_sheet_icon), ("rasters", stage24.mini_raster_grid_icon)])
        stage24.result_dual_icon_box(ax, 6.46, result_y, 1.61, result_h, "Sensitivity", [("model", stage24.mini_model_icon), ("thresholds", stage24.mini_threshold_icon)])
        stage24.paper_package_box(ax, 8.70, result_y, 2.15, result_h)
        stage24.arrow(ax, (3.62, 1.08), (4.21, 1.08))
        stage24.arrow(ax, (5.86, 1.08), (6.44, 1.08))
        stage24.arrow(ax, (8.09, 1.08), (8.68, 1.08))

        return stage24.save(fig, FIGURE_STEM)

    stage24.save = save_v56
    stage24.text = text_v60
    stage24.process_box_large_icon = process_box_large_icon_v56
    stage24.icon_model = icon_model_v65
    stage24.main_model_box = main_model_box_v56
    stage24.icon_scenarios = icon_scenarios_v72
    stage24.icon_raster = icon_raster_v59
    stage24.icon_constraint_stack = icon_constraint_stack_v56
    stage24.icon_area = icon_area_v56
    stage24.result_dual_icon_box = result_dual_icon_box_v71
    stage24.paper_package_box = paper_package_box_v71
    stage24.draw_framework = draw_framework_v65
    return stage24


def write_plot_data():
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    path = TABLE_DIR / f"{FIGURE_STEM}_area_rows.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "value_wan_km2", "value_million_km2", "meaning"])
        writer.writeheader()
        for row in AREA_ROWS:
            writer.writerow(
                {
                    "label": row["label"],
                    "value_wan_km2": row["value_wan_km2"],
                    "value_million_km2": round(row["value_wan_km2"] / 100, 4),
                    "meaning": row["meaning"],
                }
            )
    return path


def write_readme(paths, plot_data):
    readme = OUTPUT_ROOT / f"README_{FIGURE_STEM}.md"
    readme.write_text(
        "\n".join(
            [
                f"# {FIGURE_STEM}",
                "",
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "",
                "This is a minimal update of the original Stage24 v55 workflow figure. The layout, visual style, icons, panel structure, and original figure lineage are preserved.",
                "",
                "Manual-review mode: this candidate is for user visual review before Word insertion.",
                "",
                "Main scientific edits:",
                "- Removed the outdated Stage13/Stage17/Stage18 area chain `669.51 -> 378.22 -> 276.48` from the figure.",
                "- Updated the main workflow wording to selected10, HGB main chain, RF sensitivity, HydroRIVERS q10 main threshold, q25 strict sensitivity, and WorldCover land-cover weighting.",
                "- Replaced the area module with q10 results: hydro-spatial envelope 302.46 x 10^4 km2, LC >=50% envelope 238.74 x 10^4 km2, and weighted compatible area 236.34 x 10^4 km2.",
                "- Updated only the Constrained map thumbnail to a clearer schematic masked-suitability map; it is an illustrative icon rather than a data map.",
                "- Aligned row/module titles within their header bands and shortened the model row heading to `Samples, VIF and models`.",
                "- Added a standalone `Field survey / presence check` module between `Model validation` and `Main chain` to show the new field-survey presence-point check as independent validation evidence.",
                "- Fixed v64 manual-review issues: Presence, Background, and Training now use the same module geometry; model-validation nodes are enlarged and centered; Field survey icon/text are enlarged.",
                "- Fixed v65 manual-review issues: sample-row text moved upward, model-validation nodes moved downward, Field survey icon/text enlarged again, and the Main chain icon changed to a distinct route/chain symbol.",
                "- Fixed v66 manual-review issues: removed the standalone upper-left panel letter and enlarged key row-two icons with adjusted text spacing.",
                "- Fixed v68 manual-review correction: enlarged yellow-marked validation/check pictorial elements by 20%, reduced the red-marked Presence/Background/Training icons by 5%, and retuned adjacent text spacing.",
                "- Fixed v69 browser-annotation issues: reduced yellow-marked Presence/Background/Training pictorial icons by 10%, lowered and spaced the blue-circled model-validation nodes/text, enlarged the green Field survey check icon, and shifted/aligned its text left.",
                "- Fixed v70 browser-annotation issues: enlarged yellow-marked result labels, matched row-two arrows to the third arrow, narrowed the first three row-two boxes by 5% to preserve arrow spacing, and moved the red-marked `CV + LORO` text upward.",
                "- Fixed v71 manual-review issues: moved the CMIP6 caption upward, lowered the Field survey icon, shifted its text left, and retuned local alignment.",
                "",
                "Outputs:",
                f"- Transparent PNG: `{paths['png']}`",
                f"- White preview PNG: `{paths['white_preview_png']}`",
                f"- SVG: `{paths['svg']}`",
                f"- PDF: `{paths['pdf']}`",
                f"- Plot data: `{plot_data}`",
                "",
                "Candidate status: not inserted into Word. Waiting for user manual approval.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return readme


def main():
    status = {
        "figure_id": "Fig1_stage24_v72_text_icon_alignment",
        "figure_stem": FIGURE_STEM,
        "status": "running",
        "manual_review_required": True,
        "automated_review": "not_run_after_manual_revision",
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "source_script": str(__file__),
        "original_script": str(ORIGINAL_SCRIPT),
        "previous_candidate_preserved": "fig_stage24_icon_enhanced_method_framework_v71_annotation_text_arrows.*",
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    status_path = OUTPUT_ROOT / f"{FIGURE_STEM}_status.json"
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        stage24 = patch_stage24(load_original_module())
        paths = stage24.draw_framework()
        plot_data = write_plot_data()
        readme = write_readme(paths, plot_data)
        status.update(
            {
                "status": "success",
                "finished_at": datetime.now().isoformat(timespec="seconds"),
                "outputs": paths,
                "plot_data": str(plot_data),
                "readme": str(readme),
            }
        )
        status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(status, ensure_ascii=False, indent=2))
    except Exception as exc:
        status.update({"status": "failed", "finished_at": datetime.now().isoformat(timespec="seconds"), "error": repr(exc)})
        status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
