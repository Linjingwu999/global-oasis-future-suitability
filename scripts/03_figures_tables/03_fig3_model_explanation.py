from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
STAGE31 = ROOT / "outputs" / "stage31_selected_predictor_models"
STAGE32 = ROOT / "outputs" / "stage32_independent_validation_extrapolation"
OUT_DIR = ROOT / "outputs" / "stage38_manuscript_main_figures" / "combined_decided"

VERSION = "v06"
BASENAME = f"fig_stage38_model_explanation_loro_decided_{VERSION}"

MODEL_LABELS = {
    "hist_gradient_boosting_balanced": "HGB",
    "random_forest_balanced": "RF",
    "glm_logistic_balanced": "GLM",
}
MODEL_ORDER = ["HGB", "RF", "GLM"]

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

REGION_NAMES = [
    "E Asia",
    "C Asia",
    "SW Asia",
    "N America",
    "S America",
    "Oceania",
    "Arabia",
    "N Africa",
    "S Africa",
]

PALETTE = {
    "ink": "#1F2A33",
    "muted": "#65727F",
    "grid": "#E1E7E8",
    "border": "#D4DEDA",
    "teal": "#2B7A78",
    "teal_soft": "#A4CFC4",
    "blue": "#3B6EA8",
    "amber": "#C0792B",
    "rose": "#B95E60",
}


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 600,
            "font.family": "DejaVu Sans",
            "axes.titlesize": 10.0,
            "axes.titleweight": "bold",
            "axes.labelsize": 8.4,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.4,
            "axes.edgecolor": PALETTE["border"],
            "axes.linewidth": 0.8,
            "text.color": PALETTE["ink"],
            "axes.labelcolor": PALETTE["ink"],
            "xtick.color": PALETTE["ink"],
            "ytick.color": PALETTE["ink"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    importance = pd.read_csv(
        require_file(STAGE31 / "tables" / "stage31_selected_predictor_feature_importance.csv")
    )
    spatial_cv = pd.read_csv(
        require_file(STAGE31 / "tables" / "stage31_selected_predictor_spatial_cv_metrics.csv")
    )
    loro = pd.read_csv(require_file(STAGE32 / "tables" / "stage32_leave_one_region_out_metrics.csv"))

    importance["model_label"] = importance["model"].map(MODEL_LABELS).fillna(importance["model"])
    importance["feature_label"] = importance["feature"].map(FEATURE_LABELS).fillna(importance["feature"])
    spatial_cv["model_label"] = spatial_cv["model"].map(MODEL_LABELS).fillna(spatial_cv["model"])
    spatial_cv["source"] = "Spatial CV"
    loro["model_label"] = loro["model"].map(MODEL_LABELS).fillna(loro["model"])
    loro["source"] = "LORO"
    region_map = {
        raw: REGION_NAMES[i] if i < len(REGION_NAMES) else f"Region {i + 1}"
        for i, raw in enumerate(pd.unique(loro["heldout_region"]))
    }
    loro["region_label"] = loro["heldout_region"].map(region_map).fillna(loro["heldout_region"])

    combined = pd.concat(
        [
            spatial_cv[["model_label", "source", "pr_auc"]],
            loro[["model_label", "source", "pr_auc"]],
        ],
        ignore_index=True,
    )
    combined = combined[combined["model_label"].isin(MODEL_ORDER)].copy()
    if combined.empty:
        raise RuntimeError("No validation data loaded for decided figure.")
    return importance, spatial_cv, loro, combined


def panel_label(ax: plt.Axes, label: str, color: str = PALETTE["teal"]) -> None:
    ax.text(
        -0.055,
        1.075,
        label,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=8.4,
        fontweight="bold",
        color="white",
        bbox=dict(boxstyle="round,pad=0.20,rounding_size=0.04", fc=color, ec="none"),
        clip_on=False,
    )


def values_for(data: pd.DataFrame, model: str, source: str) -> np.ndarray:
    vals = data[(data["model_label"] == model) & (data["source"] == source)]["pr_auc"].to_numpy(dtype=float)
    return vals[np.isfinite(vals)]


def summarize_importance(importance: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rf = (
        importance[(importance["model_label"] == "RF") & (importance["importance_type"] == "gini_importance")]
        .groupby(["feature", "feature_label"], as_index=False)["importance"]
        .agg(mean="mean", sd="std", min="min", max="max", n="count")
        .sort_values("mean", ascending=False)
    )
    glm = (
        importance[
            (importance["model_label"] == "GLM")
            & (importance["importance_type"] == "standardized_logistic_coefficient")
        ]
        .groupby(["feature", "feature_label"], as_index=False)["importance"]
        .agg(mean="mean", sd="std", min="min", max="max", n="count")
    )
    glm["abs_mean"] = glm["mean"].abs()
    glm = glm.sort_values("abs_mean", ascending=False)

    if rf.empty:
        raise RuntimeError("No RF gini_importance rows were found in the feature-importance table.")
    if glm.empty:
        raise RuntimeError("No GLM standardized_logistic_coefficient rows were found in the feature-importance table.")
    return rf, glm


def panel_rf_importance(ax: plt.Axes, rf: pd.DataFrame) -> None:
    plot_df = rf.sort_values("mean", ascending=True).reset_index(drop=True)
    y = np.arange(len(plot_df))
    sd = plot_df["sd"].fillna(0)
    label_gap = 0.006
    label_positions = plot_df["mean"] + sd + label_gap
    ax.barh(
        y,
        plot_df["mean"],
        xerr=sd,
        height=0.62,
        color=PALETTE["teal_soft"],
        edgecolor="none",
        error_kw={"ecolor": PALETTE["teal"], "elinewidth": 1.0, "capsize": 2.5, "capthick": 1.0},
        zorder=2,
    )
    ax.scatter(plot_df["mean"], y, s=26, color=PALETTE["teal"], edgecolor="white", linewidth=0.7, zorder=3)
    for yi, value, label_x in zip(y, plot_df["mean"], label_positions):
        ax.text(label_x, yi, f"{value:.3f}", ha="left", va="center", fontsize=7.2, color=PALETTE["ink"])

    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["feature_label"])
    ax.set_xlabel("Mean Gini importance across folds")
    ax.set_title("RF predictor importance", loc="left", pad=12)
    ax.grid(axis="x", color=PALETTE["grid"], lw=0.75, zorder=1)
    ax.set_axisbelow(True)
    ax.set_xlim(0, max(0.225, float(label_positions.max()) + 0.030))
    panel_label(ax, "a", PALETTE["teal"])


def panel_glm_effect(ax: plt.Axes, glm: pd.DataFrame) -> None:
    plot_df = glm.sort_values("mean", ascending=True).reset_index(drop=True)
    y = np.arange(len(plot_df))
    colors = np.where(plot_df["mean"] >= 0, PALETTE["blue"], PALETTE["rose"])
    ax.barh(y, plot_df["mean"], height=0.62, color=colors, alpha=0.84, edgecolor="none", zorder=2)
    ax.errorbar(
        plot_df["mean"],
        y,
        xerr=plot_df["sd"].fillna(0),
        fmt="none",
        ecolor=PALETTE["ink"],
        elinewidth=0.8,
        capsize=2.4,
        alpha=0.70,
        zorder=3,
    )
    ax.axvline(0, color=PALETTE["ink"], lw=0.8, alpha=0.55, zorder=1)

    label_positions = []
    for yi, value, err in zip(y, plot_df["mean"], plot_df["sd"].fillna(0)):
        if value < 0:
            x = value - err - 0.055
            ha = "right"
        else:
            x = value + err + 0.045
            ha = "left"
        label_positions.append(x)
        ax.text(x, yi, f"{value:+.2f}", ha=ha, va="center", fontsize=7.1, color=PALETTE["ink"])

    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["feature_label"])
    ax.set_xlabel("Standardized logistic coefficient")
    ax.set_title("GLM effect direction", loc="left", pad=12)
    ax.grid(axis="x", color=PALETTE["grid"], lw=0.75, zorder=1)
    ax.set_axisbelow(True)
    left = min(-1.72, min(label_positions) - 0.08)
    right = max(0.78, max(label_positions) + 0.08)
    ax.set_xlim(left, right)
    panel_label(ax, "b", PALETTE["rose"])


def panel_mean_gap(ax: plt.Axes, data: pd.DataFrame) -> None:
    y_lookup = {model: len(MODEL_ORDER) - 1 - i for i, model in enumerate(MODEL_ORDER)}
    for model in MODEL_ORDER:
        cv_mean = float(np.mean(values_for(data, model, "Spatial CV")))
        loro_mean = float(np.mean(values_for(data, model, "LORO")))
        y = y_lookup[model]
        ax.plot([cv_mean, loro_mean], [y, y], color="#AEB8BD", lw=2.1, zorder=1)
        ax.scatter(cv_mean, y, s=34, color=PALETTE["blue"], edgecolor="white", linewidth=0.8, zorder=3)
        ax.scatter(loro_mean, y, s=34, color=PALETTE["amber"], edgecolor="white", linewidth=0.8, zorder=3)
        ax.text(cv_mean + 0.012, y, f"{cv_mean:.2f}", ha="left", va="center", fontsize=6.8)
        ax.text(loro_mean - 0.012, y, f"{loro_mean:.2f}", ha="right", va="center", fontsize=6.8)
        ax.text(
            min(cv_mean, loro_mean) + abs(cv_mean - loro_mean) / 2,
            y + 0.18,
            f"{loro_mean - cv_mean:+.2f}",
            ha="center",
            va="bottom",
            fontsize=6.5,
            color=PALETTE["muted"],
        )
    ax.set_yticks([y_lookup[m] for m in MODEL_ORDER])
    ax.set_yticklabels(MODEL_ORDER)
    ax.set_xlim(0.25, 0.82)
    ax.set_ylim(-0.35, 2.42)
    ax.set_xlabel("PR-AUC")
    ax.set_title("Mean validation gap")
    ax.grid(axis="x", color=PALETTE["grid"], lw=0.7)
    ax.scatter([], [], s=30, color=PALETTE["blue"], label="Spatial CV")
    ax.scatter([], [], s=30, color=PALETTE["amber"], label="LORO")
    ax.legend(frameon=False, loc="lower right", handletextpad=0.35, borderpad=0.1)
    panel_label(ax, "c", PALETTE["rose"])


def panel_regional_profile(ax: plt.Axes, loro: pd.DataFrame) -> None:
    profile = loro.pivot_table(index="region_label", columns="model_label", values="pr_auc", aggfunc="mean")
    profile = profile.reindex(columns=MODEL_ORDER)
    order = profile.mean(axis=1).sort_values(ascending=False).index.tolist()
    profile = profile.loc[order]
    model_colors = {"HGB": PALETTE["blue"], "RF": PALETTE["teal"], "GLM": PALETTE["rose"]}
    overall_mean = float(loro["pr_auc"].mean())

    for y, region in enumerate(order):
        vals = profile.loc[region, MODEL_ORDER].dropna().to_numpy(dtype=float)
        if vals.size == 0:
            continue
        ax.hlines(y, float(np.min(vals)), float(np.max(vals)), color="#C9D3D4", lw=2.0, zorder=1)
        for model in MODEL_ORDER:
            value = profile.loc[region, model]
            if pd.isna(value):
                continue
            ax.scatter(
                value,
                y,
                s=28,
                color=model_colors[model],
                edgecolor="white",
                linewidth=0.75,
                zorder=3,
                label=model if y == 0 else None,
            )

    ax.axvline(overall_mean, color=PALETTE["amber"], lw=1.05, ls=(0, (3, 3)), alpha=0.72, zorder=0)
    ax.text(
        overall_mean + 0.012,
        0.985,
        "LORO mean",
        transform=ax.get_xaxis_transform(),
        ha="left",
        va="top",
        fontsize=7.4,
        fontweight="bold",
        color=PALETTE["amber"],
    )
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels(order)
    ax.invert_yaxis()
    ax.set_xlabel("PR-AUC")
    ax.set_title("Independent-region profile")
    ax.grid(axis="x", color=PALETTE["grid"], lw=0.7)
    ax.set_xlim(0.0, 0.70)
    ax.legend(frameon=False, loc="lower right", ncol=3, handletextpad=0.35, columnspacing=0.8, borderpad=0.1)
    panel_label(ax, "d", PALETTE["teal"])


def save_preview(png_path: Path, preview_path: Path) -> None:
    image = Image.open(png_path).convert("RGBA")
    background = Image.new("RGBA", image.size, "WHITE")
    background.alpha_composite(image)
    background.convert("RGB").save(preview_path, quality=95)


def assert_transparent_png(png_path: Path) -> None:
    image = Image.open(png_path)
    has_alpha = image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info)
    if not has_alpha:
        raise RuntimeError(f"PNG has no alpha channel: {png_path}")


def write_source_notes() -> None:
    notes_path = OUT_DIR / f"{BASENAME}_source_notes.md"
    notes_path.write_text(
        "\n".join(
            [
                "# Decided four-panel model explanation and validation figure",
                "",
                "- Version rule: this decided combined figure starts at v01.",
                "- Included forms: RF predictor importance, GLM effect direction, mean validation gap, and independent-region profile.",
                "- Data source: local stage31 selected-predictor outputs and stage32 LORO validation outputs only.",
                "- Visual direction: OWID-inspired clean statistical charts with light grids, restrained colors, direct labels, and minimal decoration.",
                "- Output package: transparent PNG, SVG, PDF, and white-background preview.",
                "- No figure was inserted into Word.",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    apply_style()
    importance, spatial_cv, loro, combined = load_data()
    rf, glm = summarize_importance(importance)

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.2))
    fig.subplots_adjust(left=0.075, right=0.96, top=0.93, bottom=0.105, wspace=0.36, hspace=0.58)
    fig.patch.set_alpha(0)

    panel_rf_importance(axes[0, 0], rf)
    panel_glm_effect(axes[0, 1], glm)
    panel_mean_gap(axes[1, 0], combined)
    panel_regional_profile(axes[1, 1], loro)

    png_path = OUT_DIR / f"{BASENAME}.png"
    svg_path = OUT_DIR / f"{BASENAME}.svg"
    pdf_path = OUT_DIR / f"{BASENAME}.pdf"
    preview_path = OUT_DIR / f"{BASENAME}_white_preview.png"
    fig.savefig(png_path, transparent=True, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(svg_path, transparent=True, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(pdf_path, transparent=True, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

    assert_transparent_png(png_path)
    save_preview(png_path, preview_path)
    write_source_notes()
    print(preview_path)


if __name__ == "__main__":
    main()
