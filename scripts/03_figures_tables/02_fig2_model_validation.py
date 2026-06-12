from __future__ import annotations

import colorsys
import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
STAGE30 = ROOT / "outputs" / "stage30_environment_factor_collinearity"
STAGE31 = ROOT / "outputs" / "stage31_selected_predictor_models"
STAGE32 = ROOT / "outputs" / "stage32_independent_validation_extrapolation"
OUT_DIR = ROOT / "outputs" / "stage36_manuscript_main_figures"
FIG_DIR = OUT_DIR / "figures"
TABLE_DIR = OUT_DIR / "tables"
STAGE21_FIG_DIR = ROOT / "outputs" / "stage21_future_suitability_manuscript_synthesis" / "figures"

FIG_VERSION = "v10"
FIG_BASENAME = f"fig_stage36_fig2_variable_screening_model_validation_{FIG_VERSION}"
STAGE21_BASENAME = f"fig_stage21_variable_screening_model_validation_{FIG_VERSION}"
TABLE_PREFIX = f"fig2_{FIG_VERSION}"

SELECTED_ORDER = [
    "wc_elev_m",
    "wc_bio13",
    "wc_bio18",
    "wc_bio03",
    "wc_bio08",
    "wc_bio02",
    "wc_bio09",
    "wc_bio19",
    "wc_bio15",
    "wc_bio14",
]

FEATURE_LABELS = {
    "wc_elev_m": "Elevation",
    "wc_bio13": "Bio13",
    "wc_bio18": "Bio18",
    "wc_bio03": "Bio3",
    "wc_bio08": "Bio8",
    "wc_bio02": "Bio2",
    "wc_bio09": "Bio9",
    "wc_bio19": "Bio19",
    "wc_bio15": "Bio15",
    "wc_bio14": "Bio14",
}

MODEL_LABELS = {
    "hist_gradient_boosting_balanced": "HGB",
    "random_forest_balanced": "RF",
    "glm_logistic_balanced": "GLM",
}

MODEL_COLORS = {
    "HGB": "#2B7A78",
    "RF": "#3B6EA8",
    "GLM": "#C0792B",
}

PALETTE = {
    "teal": "#2B7A78",
    "teal_dark": "#165A57",
    "teal_soft": "#DDEFEA",
    "blue": "#3B6EA8",
    "blue_soft": "#E1EAF4",
    "amber": "#C0792B",
    "amber_soft": "#F4E6D5",
    "rose": "#B65C5C",
    "rose_soft": "#F2DDDC",
    "ink": "#1F2A33",
    "muted": "#65727F",
    "grid": "#D8DEE3",
    "panel_bg": "#FBFCF8",
    "panel_edge": "#D7DED8",
    "white": "#FFFFFF",
}


def boost_saturation(hex_color: str, factor: float = 1.05) -> str:
    r, g, b = mpl.colors.to_rgb(hex_color)
    h, lightness, saturation = colorsys.rgb_to_hls(r, g, b)
    boosted = min(1.0, saturation * factor)
    return mpl.colors.to_hex(colorsys.hls_to_rgb(h, lightness, boosted))


def darken_color(hex_color: str, factor: float = 0.90) -> str:
    r, g, b = mpl.colors.to_rgb(hex_color)
    h, lightness, saturation = colorsys.rgb_to_hls(r, g, b)
    return mpl.colors.to_hex(colorsys.hls_to_rgb(h, max(0.0, lightness * factor), saturation))


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required input file is missing: {path}")
    return path


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(require_file(path), low_memory=False)


def compute_vif(samples: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    missing = [f for f in features if f not in samples.columns]
    if missing:
        raise ValueError(f"Selected predictor columns missing from modeling sample table: {missing}")

    x = samples[features].apply(pd.to_numeric, errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
    if len(x) <= len(features) + 1:
        raise ValueError("Not enough complete rows to compute selected-predictor VIF.")

    values = x.to_numpy(dtype=float)
    values = (values - values.mean(axis=0)) / values.std(axis=0, ddof=0)
    rows = []
    for idx, feature in enumerate(features):
        y = values[:, idx]
        others = np.delete(values, idx, axis=1)
        design = np.column_stack([np.ones(len(others)), others])
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        pred = design @ coef
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot else np.nan
        r2 = min(r2, 0.999999999999)
        vif = 1.0 / (1.0 - r2)
        rows.append({"feature": feature, "label": FEATURE_LABELS.get(feature, feature), "final_vif": vif})
    return pd.DataFrame(rows)


def model_sort_key(model: str) -> int:
    order = ["hist_gradient_boosting_balanced", "random_forest_balanced", "glm_logistic_balanced"]
    return order.index(model) if model in order else len(order)


def prepare_inputs() -> dict[str, pd.DataFrame]:
    iterative_vif = read_csv(STAGE30 / "tables" / "stage30_iterative_vif_selection.csv")
    recommended = read_csv(STAGE30 / "tables" / "stage30_recommended_predictors.csv")
    samples = read_csv(STAGE30 / "tables" / "stage30_modeling_samples_selected_predictors.csv")
    selected_vif = compute_vif(samples, SELECTED_ORDER)

    comparison = read_csv(STAGE31 / "tables" / "stage31_selected_vs_full_model_metric_comparison.csv")
    selected_summary = read_csv(STAGE31 / "tables" / "stage31_selected_predictor_model_summary.csv")
    cv_metrics = read_csv(STAGE31 / "tables" / "stage31_selected_predictor_spatial_cv_metrics.csv")
    loro_metrics = read_csv(STAGE32 / "tables" / "stage32_leave_one_region_out_metrics.csv")
    loro_summary = read_csv(STAGE32 / "tables" / "stage32_leave_one_region_out_model_summary.csv")

    selected_rows = recommended[recommended["decision"].eq("selected")].copy()
    selected_set = selected_rows["feature"].tolist()
    if selected_set != SELECTED_ORDER:
        raise ValueError(f"Unexpected selected predictor order: {selected_set}")

    return {
        "iterative_vif": iterative_vif,
        "recommended": recommended,
        "selected_vif": selected_vif,
        "comparison": comparison,
        "selected_summary": selected_summary,
        "cv_metrics": cv_metrics,
        "loro_metrics": loro_metrics,
        "loro_summary": loro_summary,
    }


def add_round_box(
    ax: mpl.axes.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    facecolor: str,
    edgecolor: str = "none",
    lw: float = 0.8,
    radius: float = 0.04,
    zorder: int = 1,
) -> patches.FancyBboxPatch:
    box = patches.FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        transform=ax.transAxes,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=lw,
        zorder=zorder,
        clip_on=False,
    )
    ax.add_patch(box)
    return box


def add_story_strip(ax: mpl.axes.Axes, iterative_vif: pd.DataFrame) -> None:
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    initial_count = int(pd.to_numeric(iterative_vif["remaining_feature_count"], errors="coerce").max())
    final_count = int(pd.to_numeric(iterative_vif["remaining_feature_count"], errors="coerce").min())
    removed_count = initial_count - final_count

    strip_y = 0.02
    box_y = 0.14
    box_h = 0.58
    big_y = 0.55
    label_y = 0.38
    note_y = 0.25
    arrow_y = 0.43

    add_round_box(ax, (0.005, strip_y), 0.99, 0.86, "#F5F8F4", PALETTE["panel_edge"], lw=0.8, radius=0.035)
    stages = [
        {
            "x": 0.035,
            "w": 0.27,
            "color": PALETTE["blue_soft"],
            "accent": PALETTE["blue"],
            "big": f"{initial_count}",
            "label": "candidate factors",
            "note": "climate + terrain pool",
        },
        {
            "x": 0.365,
            "w": 0.27,
            "color": PALETTE["teal_soft"],
            "accent": PALETTE["teal"],
            "big": f"{final_count}",
            "label": "selected predictors",
            "note": f"{removed_count} high-VIF factors removed",
        },
        {
            "x": 0.695,
            "w": 0.27,
            "color": PALETTE["amber_soft"],
            "accent": PALETTE["amber"],
            "big": "CV + LORO",
            "label": "model validation",
            "note": "within-region fit and transferability",
        },
    ]
    for item in stages:
        add_round_box(ax, (item["x"], box_y), item["w"], box_h, item["color"], "none", radius=0.025)
        ax.add_patch(
            patches.Rectangle(
                (item["x"], box_y),
                0.012,
                box_h,
                transform=ax.transAxes,
                facecolor=item["accent"],
                edgecolor="none",
                zorder=3,
                clip_on=False,
            )
        )
        ax.text(
            item["x"] + 0.035,
            big_y,
            item["big"],
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=11.5,
            fontweight="bold",
            color=item["accent"],
        )
        ax.text(
            item["x"] + 0.035,
            label_y,
            item["label"],
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=8.1,
            fontweight="bold",
            color=PALETTE["ink"],
        )
        ax.text(
            item["x"] + 0.035,
            note_y,
            item["note"],
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=6.9,
            color=PALETTE["muted"],
        )

    for x0, x1 in [(0.315, 0.355), (0.645, 0.685)]:
        ax.annotate(
            "",
            xy=(x1, arrow_y),
            xytext=(x0, arrow_y),
            xycoords=ax.transAxes,
            textcoords=ax.transAxes,
            arrowprops=dict(arrowstyle="-|>", lw=1.3, color=PALETTE["muted"], shrinkA=0, shrinkB=0),
        )


def style_axes(ax: mpl.axes.Axes, grid_axis: str = "y") -> None:
    ax.set_facecolor(PALETTE["panel_bg"])
    ax.tick_params(axis="both", labelsize=7.3, colors=PALETTE["ink"], length=3)
    for spine in ax.spines.values():
        spine.set_color(PALETTE["panel_edge"])
        spine.set_linewidth(0.8)
    if grid_axis:
        ax.grid(True, axis=grid_axis, color=PALETTE["grid"], linewidth=0.55, alpha=0.78)
    ax.set_axisbelow(True)


def add_panel_label(ax: mpl.axes.Axes, label: str, title: str, accent: str) -> None:
    ax.text(
        -0.075,
        1.095,
        label,
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=8.2,
        fontweight="bold",
        color=PALETTE["white"],
        bbox=dict(boxstyle="round,pad=0.22,rounding_size=0.08", facecolor=accent, edgecolor="none"),
    )
    ax.text(
        0.065,
        1.095,
        title,
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=8.6,
        fontweight="bold",
        color=PALETTE["ink"],
    )


def parse_vif(value: object) -> float:
    text = str(value).strip().lower()
    if text in {"inf", "infinity"}:
        return math.inf
    return float(text)


def plot_vif_path(ax: mpl.axes.Axes, iterative_vif: pd.DataFrame) -> None:
    df = iterative_vif.copy()
    df["iteration"] = pd.to_numeric(df["iteration"], errors="coerce")
    df["remaining_feature_count"] = pd.to_numeric(df["remaining_feature_count"], errors="coerce")
    df["max_vif_numeric"] = df["max_vif"].map(parse_vif)
    finite_max = df.loc[np.isfinite(df["max_vif_numeric"]), "max_vif_numeric"].max()
    cap = max(10_000.0, finite_max * 1.35)
    df["plot_vif"] = df["max_vif_numeric"].replace(math.inf, cap)

    ax.fill_between(
        df["iteration"],
        10,
        df["plot_vif"],
        where=df["plot_vif"] >= 10,
        color=PALETTE["amber_soft"],
        alpha=0.75,
        zorder=0,
    )
    ax.plot(
        df["iteration"],
        df["plot_vif"],
        color=PALETTE["teal"],
        marker="o",
        markersize=4.4,
        markerfacecolor=PALETTE["white"],
        markeredgewidth=1.2,
        linewidth=1.9,
        zorder=4,
    )
    ax.axhline(10, color=PALETTE["amber"], linestyle=(0, (3, 2)), linewidth=1.05, zorder=2)
    ax.text(
        0.985,
        0.255,
        "VIF = 10",
        transform=ax.transAxes,
        fontsize=6.2,
        ha="right",
        va="center",
        color=PALETTE["amber"],
        zorder=6,
    )
    ax.set_yscale("log")
    ax.set_xlabel("Screening iteration", fontsize=8.0)
    ax.set_ylabel("Maximum VIF (log scale)", fontsize=8.0)
    ax.set_xticks(df["iteration"])
    ax.set_xticklabels([str(int(v)) for v in df["iteration"]], fontsize=6.7)
    ax.set_ylim(1.25, cap * 1.28)
    ax.set_xlim(df["iteration"].min() - 0.35, df["iteration"].max() + 1.2)

    final_row = df.iloc[-1]
    ax.text(
        final_row["iteration"] - 0.42,
        max(final_row["plot_vif"] * 0.64, 1.62),
        "10 retained",
        fontsize=7.0,
        fontweight="bold",
        color=PALETTE["teal"],
        ha="right",
        va="center",
        zorder=6,
    )

    style_axes(ax, grid_axis="y")
    ax.grid(True, which="both", axis="y", color=PALETTE["grid"], linewidth=0.5, alpha=0.68)
    add_panel_label(ax, "a", "Collinearity screening path", PALETTE["teal"])


def plot_selected_vif(ax: mpl.axes.Axes, selected_vif: pd.DataFrame) -> None:
    df = selected_vif.sort_values("final_vif", ascending=True).copy()
    y = np.arange(len(df))
    ax.axvspan(0, 10, color=PALETTE["teal_soft"], alpha=0.35, zorder=0)
    ax.hlines(y, 0, df["final_vif"], color="#9BC9BB", linewidth=4.2, zorder=2)
    ax.scatter(df["final_vif"], y, s=42, color=PALETTE["teal"], edgecolor=PALETTE["white"], linewidth=0.8, zorder=3)
    ax.axvline(10, color=PALETTE["amber"], linestyle=(0, (3, 2)), linewidth=1.05, zorder=2)
    ax.text(
        10.08,
        len(df) - 0.12,
        "threshold",
        fontsize=7.0,
        color=PALETTE["amber"],
        va="center",
        ha="left",
        clip_on=False,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(df["label"], fontsize=7.2)
    ax.set_xlabel("Final VIF", fontsize=8.0)
    ax.set_xlim(0, 10.8)
    for yy, val in zip(y, df["final_vif"]):
        ax.text(val + 0.16, yy, f"{val:.2f}", va="center", ha="left", fontsize=6.8, color=PALETTE["ink"])
    style_axes(ax, grid_axis="x")
    ax.grid(False, axis="y")
    add_panel_label(ax, "b", "Selected predictor VIF values", PALETTE["teal"])


def plot_delta_heatmap(ax: mpl.axes.Axes, comparison: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in sorted(comparison["model"].tolist(), key=model_sort_key):
        row = comparison[comparison["model"].eq(model)].iloc[0]
        rows.append(
            {
                "model": MODEL_LABELS.get(model, model),
                "ROC-AUC": float(row["roc_auc_mean_delta_selected_minus_full"]),
                "PR-AUC": float(row["pr_auc_mean_delta_selected_minus_full"]),
                "TSS": float(row["tss_mean_delta_selected_minus_full"]),
            }
        )
    delta = pd.DataFrame(rows).set_index("model")
    values = delta.to_numpy(dtype=float)
    vmax = max(0.04, float(np.nanmax(np.abs(values))) * 1.05)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    cmap = LinearSegmentedColormap.from_list(
        "selected_delta_vivid5",
        [boost_saturation(PALETTE["rose"], 1.05), "#F8F5EF", boost_saturation(PALETTE["teal"], 1.05)],
    )
    im = ax.imshow(values, cmap=cmap, norm=norm, aspect="auto")

    ax.set_facecolor(PALETTE["panel_bg"])
    ax.set_xticks(np.arange(delta.shape[1]))
    ax.set_xticklabels(delta.columns, fontsize=7.4)
    ax.set_yticks(np.arange(delta.shape[0]))
    ax.set_yticklabels(delta.index, fontsize=7.4)
    ax.tick_params(length=0)
    for i in range(delta.shape[0]):
        for j in range(delta.shape[1]):
            val = values[i, j]
            ax.text(j, i, f"{val:+.3f}", ha="center", va="center", fontsize=7.1, color=PALETTE["ink"])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, delta.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, delta.shape[0], 1), minor=True)
    ax.grid(which="minor", color=PALETTE["white"], linewidth=1.45)
    ax.tick_params(which="minor", bottom=False, left=False)
    cbar = plt.colorbar(im, ax=ax, fraction=0.050, pad=0.035)
    cbar.outline.set_linewidth(0.4)
    cbar.outline.set_edgecolor(PALETTE["panel_edge"])
    cbar.ax.tick_params(labelsize=6.6, length=2, colors=PALETTE["ink"])
    cbar.set_label("selected10 - full20", fontsize=7.0, color=PALETTE["ink"])
    add_panel_label(ax, "c", "Compact-model metric change", PALETTE["rose"])
    return delta.reset_index()


def draw_validation_legend(
    ax: mpl.axes.Axes,
    colors: dict[str, tuple[float, float, float]],
    edge_colors: dict[str, tuple[float, float, float]],
) -> None:
    x_handle = 0.735
    x_text = 0.780
    y_start = 0.942
    row_step = 0.074
    cap_half_width = 0.012
    err_half_height = 0.019

    ax.add_patch(
        patches.Rectangle(
            (0.705, 0.828),
            0.285,
            0.148,
            transform=ax.transAxes,
            facecolor=PALETTE["panel_bg"],
            edgecolor="none",
            clip_on=False,
            zorder=11,
        )
    )

    for row, label in enumerate(["Spatial CV", "LORO"]):
        y = y_start - row * row_step
        ax.plot(
            [x_handle, x_handle],
            [y - err_half_height, y + err_half_height],
            transform=ax.transAxes,
            color=edge_colors[label],
            linewidth=1.05,
            solid_capstyle="butt",
            clip_on=False,
            zorder=12,
        )
        ax.plot(
            [x_handle - cap_half_width, x_handle + cap_half_width],
            [y - err_half_height, y - err_half_height],
            transform=ax.transAxes,
            color=edge_colors[label],
            linewidth=1.05,
            solid_capstyle="butt",
            clip_on=False,
            zorder=12,
        )
        ax.plot(
            [x_handle - cap_half_width, x_handle + cap_half_width],
            [y + err_half_height, y + err_half_height],
            transform=ax.transAxes,
            color=edge_colors[label],
            linewidth=1.05,
            solid_capstyle="butt",
            clip_on=False,
            zorder=12,
        )
        ax.scatter(
            [x_handle],
            [y],
            transform=ax.transAxes,
            s=27,
            color=colors[label],
            edgecolor=PALETTE["white"],
            linewidth=0.45,
            clip_on=False,
            zorder=13,
        )
        ax.text(
            x_text,
            y,
            label,
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=7.3,
            color=PALETTE["ink"],
            zorder=13,
        )


def plot_validation_pr_auc(
    ax: mpl.axes.Axes,
    selected_summary: pd.DataFrame,
    loro_summary: pd.DataFrame,
    cv_metrics: pd.DataFrame,
    loro_metrics: pd.DataFrame,
) -> pd.DataFrame:
    models = ["hist_gradient_boosting_balanced", "random_forest_balanced", "glm_logistic_balanced"]
    labels = [MODEL_LABELS[m] for m in models]
    x = np.arange(len(models))
    offsets = {"Spatial CV": -0.18, "LORO": 0.18}
    colors = {"Spatial CV": darken_color(PALETTE["blue"], 0.90), "LORO": darken_color(PALETTE["amber"], 0.90)}
    edge_colors = {"Spatial CV": darken_color("#2D4B6D", 0.90), "LORO": darken_color("#744116", 0.90)}
    sources = {
        "Spatial CV": cv_metrics,
        "LORO": loro_metrics,
    }

    rows = []
    y_max = 0.0
    for model_index, model in enumerate(models):
        for group_name, source in sources.items():
            vals = pd.to_numeric(source.loc[source["model"].eq(model), "pr_auc"], errors="coerce").dropna().to_numpy(dtype=float)
            if vals.size == 0:
                raise ValueError(f"No PR-AUC values found for {group_name} model {model}.")
            pos = x[model_index] + offsets[group_name]
            mean = float(vals.mean())
            sd = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
            jitter = np.linspace(-0.055, 0.055, vals.size) if vals.size > 1 else np.array([0.0])

            ax.scatter(
                pos + jitter,
                vals,
                s=10,
                color=colors[group_name],
                edgecolor=PALETTE["white"],
                linewidth=0.35,
                alpha=0.48,
                zorder=3,
            )
            ax.errorbar(
                pos,
                mean,
                yerr=sd,
                fmt="o",
                markersize=4.8,
                color=colors[group_name],
                ecolor=edge_colors[group_name],
                elinewidth=1.25,
                capsize=3.2,
                capthick=1.1,
                markeredgecolor=PALETTE["white"],
                markeredgewidth=0.55,
                label=group_name if model_index == 0 else None,
                zorder=5,
            )
            ax.text(
                pos,
                mean + sd + 0.026,
                f"{mean:.2f}",
                ha="center",
                va="bottom",
                fontsize=6.6,
                color=PALETTE["ink"],
                zorder=6,
            )
            y_max = max(y_max, float(vals.max()), mean + sd)
            rows.append(
                {
                    "model": MODEL_LABELS[model],
                    "validation_type": group_name,
                    "n": int(vals.size),
                    "pr_auc_mean": mean,
                    "pr_auc_std": sd,
                    "pr_auc_min": float(vals.min()),
                    "pr_auc_max": float(vals.max()),
                }
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.8)
    ax.set_ylabel("PR-AUC", fontsize=8.0)
    ax.set_xlim(-0.55, len(models) - 0.45)
    ax.set_ylim(0, y_max + 0.095)
    style_axes(ax, grid_axis="y")
    draw_validation_legend(ax, colors, edge_colors)
    add_panel_label(ax, "d", "Spatial CV versus independent regions", PALETTE["blue"])
    return pd.DataFrame(rows)


def export_with_checks(fig: mpl.figure.Figure, base_path: Path) -> dict[str, str]:
    png_path = base_path.with_suffix(".png")
    svg_path = base_path.with_suffix(".svg")
    pdf_path = base_path.with_suffix(".pdf")
    white_preview_path = base_path.with_name(base_path.name + "_white_preview").with_suffix(".png")

    export_pad = 0.012
    fig.savefig(png_path, dpi=600, bbox_inches="tight", pad_inches=export_pad, transparent=True)
    fig.savefig(svg_path, bbox_inches="tight", pad_inches=export_pad, transparent=True)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=export_pad, transparent=True)

    with Image.open(png_path) as img:
        if not (img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info)):
            raise RuntimeError(f"PNG does not contain transparency: {png_path}")
        rgba = img.convert("RGBA")
        white = Image.new("RGBA", rgba.size, "WHITE")
        white.alpha_composite(rgba)
        white.convert("RGB").save(white_preview_path, quality=95)

    return {
        "png": str(png_path),
        "svg": str(svg_path),
        "pdf": str(pdf_path),
        "white_preview": str(white_preview_path),
    }


def copy_to_stage21(paths: dict[str, str]) -> dict[str, str]:
    STAGE21_FIG_DIR.mkdir(parents=True, exist_ok=True)
    copied = {}
    suffix_map = {
        "png": ".png",
        "svg": ".svg",
        "pdf": ".pdf",
        "white_preview": "_white_preview.png",
    }
    for key, suffix in suffix_map.items():
        src = Path(paths[key])
        dest = STAGE21_FIG_DIR / f"{STAGE21_BASENAME}{suffix}"
        dest.write_bytes(src.read_bytes())
        copied[key] = str(dest)
    return copied


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    data = prepare_inputs()

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.labelcolor": PALETTE["ink"],
            "xtick.color": PALETTE["ink"],
            "ytick.color": PALETTE["ink"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )

    fig = plt.figure(figsize=(7.95, 7.20), constrained_layout=False, facecolor=(1, 1, 1, 0))
    gs = fig.add_gridspec(
        3,
        2,
        height_ratios=[0.36, 1.0, 1.0],
        left=0.075,
        right=0.982,
        top=0.975,
        bottom=0.075,
        hspace=0.62,
        wspace=0.46,
    )
    ax_story = fig.add_subplot(gs[0, :])
    ax_a = fig.add_subplot(gs[1, 0])
    ax_b = fig.add_subplot(gs[1, 1])
    ax_c = fig.add_subplot(gs[2, 0])
    ax_d = fig.add_subplot(gs[2, 1])

    ax_story.set_position([0.075, 0.800, 0.907, 0.097])

    add_story_strip(ax_story, data["iterative_vif"])
    plot_vif_path(ax_a, data["iterative_vif"])
    plot_selected_vif(ax_b, data["selected_vif"])
    delta_table = plot_delta_heatmap(ax_c, data["comparison"])
    validation_table = plot_validation_pr_auc(
        ax_d,
        data["selected_summary"],
        data["loro_summary"],
        data["cv_metrics"],
        data["loro_metrics"],
    )

    base = FIG_DIR / FIG_BASENAME
    exports = export_with_checks(fig, base)
    stage21_exports = copy_to_stage21(exports)
    plt.close(fig)

    data["selected_vif"].to_csv(
        TABLE_DIR / f"{TABLE_PREFIX}_selected_predictor_final_vif.csv",
        index=False,
        encoding="utf-8-sig",
    )
    delta_table.to_csv(
        TABLE_DIR / f"{TABLE_PREFIX}_selected10_minus_full20_metric_delta.csv",
        index=False,
        encoding="utf-8-sig",
    )
    validation_table.to_csv(
        TABLE_DIR / f"{TABLE_PREFIX}_validation_pr_auc_plot_data.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "figure": "Fig. 2 variable screening and model validation",
        "version": FIG_VERSION,
        "status": "candidate_success",
        "design_scope": [
            "Fig. 2 v10 revises the v09 candidate after user-directed legend layout QA.",
            "The top story strip is moved upward by about 5 percent of its strip height from the v07 placement.",
            "Panel-d legend markers are manually centered between their error-bar caps, shifted left from v09, and separated from grid lines.",
            "The exported figure padding is tightened to reduce excess white space above the story strip.",
            "The panel-a VIF label is moved downward and reduced in size to avoid crossing nearby grid and threshold lines.",
            "Panel-c heatmap endpoints use 5 percent higher color saturation while preserving the same metric values and color scale range.",
            "Decorative elements are limited to a story strip and light panel backgrounds.",
            "Panel d keeps raw validation points plus mean +/- SD error bars, with smaller circles and 10 percent darker circle colors.",
            "The panel-a retained-count annotation is changed from an arrow callout to direct text near the final point.",
            "No external icon assets are used in this statistical figure.",
        ],
        "inputs": {
            "stage30_iterative_vif": str(STAGE30 / "tables" / "stage30_iterative_vif_selection.csv"),
            "stage30_selected_samples": str(STAGE30 / "tables" / "stage30_modeling_samples_selected_predictors.csv"),
            "stage31_selected_vs_full": str(STAGE31 / "tables" / "stage31_selected_vs_full_model_metric_comparison.csv"),
            "stage31_selected_summary": str(STAGE31 / "tables" / "stage31_selected_predictor_model_summary.csv"),
            "stage31_spatial_cv_metrics": str(STAGE31 / "tables" / "stage31_selected_predictor_spatial_cv_metrics.csv"),
            "stage32_loro_metrics": str(STAGE32 / "tables" / "stage32_leave_one_region_out_metrics.csv"),
            "stage32_loro_summary": str(STAGE32 / "tables" / "stage32_leave_one_region_out_model_summary.csv"),
        },
        "outputs": exports,
        "stage21_copies": stage21_exports,
        "notes": [
            "Only completed Stage30-32 tabular results are plotted.",
            "No selected10 spatial area, land-cover overlay, or workstation rerun results are embedded.",
            "LORO means leave-one-region-out independent regional validation.",
            "The current manuscript DOCX is not modified by this candidate script.",
        ],
    }
    (OUT_DIR / f"stage36_fig2_variable_screening_model_validation_{FIG_VERSION}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (OUT_DIR / f"Stage36_Fig2_variable_screening_model_validation_{FIG_VERSION}_README.md").write_text(
        "\n".join(
            [
                f"# Stage36 Fig. 2 variable screening and model validation {FIG_VERSION}",
                "",
                "This candidate figure combines completed Stage30-32 diagnostics for the manuscript main text.",
                "",
                "Panels:",
                "- Story strip: candidate factors -> selected predictors -> model validation.",
                "- (a) VIF screening path from 20 candidate predictors to the selected10 set.",
                "- (b) Final selected-predictor VIF values computed from the Stage30 selected-predictor modeling table.",
                "- (c) Metric changes for selected10 relative to the full20 baseline in Stage31.",
                "- (d) PR-AUC comparison between spatial cross-validation and leave-one-region-out validation in Stage32.",
                "",
                "User-directed v10 edits:",
                "- Move the top story strip upward by about 5 percent of its strip height from the v07 placement.",
                "- Center the Spatial CV and LORO legend markers between their error-bar cap lines, shift the legend left, and mask grid lines behind the labels.",
                "- Tighten export padding to reduce excess white space above the story strip.",
                "- Keep the panel-a retained-count annotation as direct text without an arrow.",
                "",
                "Important scope note:",
                "- The figure does not include selected10 future area, land-cover overlay, or workstation rerun outputs.",
                "- Therefore it should remain stable after the selected10 grid rerun, unless Stage30-32 tables are recomputed.",
                "- The current manuscript DOCX is not modified by this candidate script.",
            ]
        ),
        encoding="utf-8",
    )

    for value in exports.values():
        print(value)


if __name__ == "__main__":
    main()
