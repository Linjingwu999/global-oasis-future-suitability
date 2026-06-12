# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import logging
import math
import re
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from PIL import Image


WORKSPACE = Path(r"C:\Users\linjingwu\Desktop\python")
PROJECT_ROOT = WORKSPACE / "绿洲未来适宜区预测"

STAGE11_DRAFT = PROJECT_ROOT / "outputs" / "stage11_submission_enhancement" / "未来绿洲适宜区预测_投稿增强稿.md"
STAGE12_STATUS_CSV = PROJECT_ROOT / "logs" / "stage12_region_tile_grid_projection_status.csv"
STAGE12_MANIFEST_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "stage12_region_tile_grid_projection"
    / "tables"
    / "stage12_region_tile_manifest.csv"
)

OUT_DIR = PROJECT_ROOT / "outputs" / "stage13_global_dryland_tile_summary"
FIG_DIR = OUT_DIR / "figures"
TABLE_DIR = OUT_DIR / "tables"
LOG_DIR = PROJECT_ROOT / "logs"
LOG_PATH = LOG_DIR / "stage13_global_dryland_tile_summary.log"
STATUS_CSV = LOG_DIR / "stage13_global_dryland_tile_summary_status.csv"
STATE_JSON = LOG_DIR / "stage13_global_dryland_tile_summary_state.json"
SUMMARY_JSON = OUT_DIR / "stage13_global_dryland_tile_summary.json"
REPORT_MD = OUT_DIR / "Stage13_全球干旱区tile投影汇总报告.md"
UPDATED_DRAFT_MD = OUT_DIR / "未来绿洲适宜区预测_加入Stage12全干旱区试投影稿.md"

TILE_METRICS_CSV = TABLE_DIR / "stage13_stage12_tile_metrics.csv"
GLOBAL_SUMMARY_CSV = TABLE_DIR / "stage13_global_summary.csv"
MACRO_REGION_SUMMARY_CSV = TABLE_DIR / "stage13_macro_region_summary.csv"
TOP_TILES_CSV = TABLE_DIR / "stage13_top_suitable_tiles.csv"


OWID_COLORS = {
    "blue": "#4C78A8",
    "green": "#4E8F5B",
    "teal": "#79AEB2",
    "sand": "#D8C08C",
    "orange": "#F28E2B",
    "red": "#B8554E",
    "gray": "#6E6E6E",
    "light_grid": "#D9D9D9",
}


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def setup_logging() -> None:
    ensure_dirs()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(LOG_PATH, encoding="utf-8"), logging.StreamHandler()],
    )


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(data, ensure_ascii=False, indent=2))


def atomic_write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8-sig")
    tmp.replace(path)


def write_status(status: str, message: str, **extra: Any) -> None:
    row = {"updated_at": now_iso(), "status": status, "message": message}
    row.update(extra)
    atomic_write_csv(STATUS_CSV, pd.DataFrame([row]))
    state = dict(row)
    atomic_write_json(STATE_JSON, state)


def base_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "legend.frameon": False,
            "figure.dpi": 150,
            "savefig.dpi": 600,
        }
    )


def save_figure(fig: plt.Figure, stem: str) -> dict[str, str]:
    png = FIG_DIR / f"{stem}.png"
    svg = FIG_DIR / f"{stem}.svg"
    fig.savefig(png, dpi=600, bbox_inches="tight", pad_inches=0.02, transparent=True)
    fig.savefig(svg, bbox_inches="tight", pad_inches=0.02, transparent=True)
    img = Image.open(png)
    if not (img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info)):
        raise RuntimeError(f"PNG lacks alpha channel: {png}")
    return {"png": str(png), "svg": str(svg)}


def extract_stage10_json(message: str) -> dict[str, Any] | None:
    if not isinstance(message, str) or not message.strip():
        return None
    marker = "Stage10 completed:"
    idx = message.find(marker)
    text = message[idx + len(marker) :] if idx >= 0 else message
    brace_idx = text.find("{")
    if brace_idx < 0:
        return None
    text = text[brace_idx:].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"(\{.*\})", text, flags=re.S)
        if not match:
            return None
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            return None


def macro_region_from_center(lon: float, lat: float) -> str:
    # Coarse reporting bins by tile centroid; these are for exploratory summaries,
    # not administrative or eco-region boundaries.
    if -170 <= lon < -30 and lat >= 0:
        return "North America drylands"
    if -90 <= lon < -30 and lat < 0:
        return "South America drylands"
    if -20 <= lon < 35 and lat >= 0:
        return "North Africa and Mediterranean"
    if -20 <= lon < 55 and lat < 0:
        return "Sub-Saharan Africa drylands"
    if 35 <= lon < 80:
        return "Middle East and Central Asia"
    if 80 <= lon < 115 and lat >= 0:
        return "South and East Asian drylands"
    if 115 <= lon <= 180 and lat >= 0:
        return "East Asia and northern drylands"
    if 110 <= lon <= 180 and lat < 0:
        return "Australia and Oceania drylands"
    return "Other drylands"


def load_tile_metrics() -> pd.DataFrame:
    if not STAGE12_STATUS_CSV.exists():
        raise FileNotFoundError(f"Stage12 status CSV not found: {STAGE12_STATUS_CSV}")

    status_df = pd.read_csv(STAGE12_STATUS_CSV)
    if status_df.empty:
        raise RuntimeError(f"Stage12 status CSV is empty: {STAGE12_STATUS_CSV}")

    rows: list[dict[str, Any]] = []
    parse_failures: list[str] = []
    for _, row in status_df.iterrows():
        payload = extract_stage10_json(str(row.get("message", "")))
        if payload is None:
            parse_failures.append(str(row.get("job_id", "")))
            continue
        out = {
            "updated_at": row.get("updated_at"),
            "job_id": row.get("job_id"),
            "status": row.get("status"),
            "model_group": row.get("model_group"),
            "gcm": row.get("gcm"),
            "ssp": row.get("ssp"),
            "period": row.get("period"),
            "tile_id": row.get("tile_id"),
            "region": row.get("region"),
            "min_lon": float(row.get("min_lon")),
            "min_lat": float(row.get("min_lat")),
            "max_lon": float(row.get("max_lon")),
            "max_lat": float(row.get("max_lat")),
        }
        for key in [
            "threshold",
            "width",
            "height",
            "valid_pixels",
            "suitable_pixels",
            "valid_area_km2",
            "suitable_area_km2",
            "mean_probability",
            "min_probability",
            "max_probability",
            "suitable_rate",
            "probability_tif",
            "suitable_tif",
        ]:
            out[key] = payload.get(key)
        out["center_lon"] = (out["min_lon"] + out["max_lon"]) / 2
        out["center_lat"] = (out["min_lat"] + out["max_lat"]) / 2
        out["macro_region"] = macro_region_from_center(out["center_lon"], out["center_lat"])
        rows.append(out)

    if parse_failures:
        logging.warning("Failed to parse %s Stage12 message rows.", len(parse_failures))
    if not rows:
        raise RuntimeError("No Stage10 JSON payloads could be parsed from Stage12 status CSV.")

    df = pd.DataFrame(rows)
    numeric_cols = [
        "threshold",
        "width",
        "height",
        "valid_pixels",
        "suitable_pixels",
        "valid_area_km2",
        "suitable_area_km2",
        "mean_probability",
        "min_probability",
        "max_probability",
        "suitable_rate",
        "center_lon",
        "center_lat",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if STAGE12_MANIFEST_CSV.exists():
        manifest = pd.read_csv(STAGE12_MANIFEST_CSV)
        manifest = manifest[["tile_id", "source_file", "region_bounds"]].drop_duplicates("tile_id")
        df = df.merge(manifest, on="tile_id", how="left")

    atomic_write_csv(TILE_METRICS_CSV, df)
    return df


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    valid = values.notna() & weights.notna() & (weights > 0)
    if not valid.any():
        return float("nan")
    return float(np.average(values[valid], weights=weights[valid]))


def build_summaries(tile_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    group_cols = ["model_group", "gcm", "ssp", "period"]
    global_rows = []
    for keys, g in tile_df.groupby(group_cols, dropna=False):
        valid_area = float(g["valid_area_km2"].sum(skipna=True))
        suitable_area = float(g["suitable_area_km2"].sum(skipna=True))
        global_rows.append(
            {
                **dict(zip(group_cols, keys)),
                "tile_count": int(len(g)),
                "success_tile_count": int((g["status"] == "success").sum()),
                "valid_tile_count": int((g["valid_area_km2"].fillna(0) > 0).sum()),
                "suitable_tile_count": int((g["suitable_area_km2"].fillna(0) > 0).sum()),
                "valid_area_km2": valid_area,
                "suitable_area_km2": suitable_area,
                "area_weighted_suitable_rate": suitable_area / valid_area if valid_area > 0 else float("nan"),
                "area_weighted_mean_probability": weighted_mean(g["mean_probability"], g["valid_area_km2"]),
                "max_tile_probability": float(g["max_probability"].max(skipna=True)),
                "threshold": float(g["threshold"].dropna().iloc[0]) if g["threshold"].notna().any() else float("nan"),
            }
        )
    global_summary = pd.DataFrame(global_rows)

    macro_rows = []
    for keys, g in tile_df.groupby(group_cols + ["macro_region"], dropna=False):
        valid_area = float(g["valid_area_km2"].sum(skipna=True))
        suitable_area = float(g["suitable_area_km2"].sum(skipna=True))
        macro_rows.append(
            {
                **dict(zip(group_cols + ["macro_region"], keys)),
                "tile_count": int(len(g)),
                "valid_area_km2": valid_area,
                "suitable_area_km2": suitable_area,
                "area_weighted_suitable_rate": suitable_area / valid_area if valid_area > 0 else float("nan"),
                "area_weighted_mean_probability": weighted_mean(g["mean_probability"], g["valid_area_km2"]),
                "max_tile_probability": float(g["max_probability"].max(skipna=True)),
            }
        )
    macro_summary = pd.DataFrame(macro_rows).sort_values("suitable_area_km2", ascending=False)

    top_tiles = (
        tile_df.sort_values(["suitable_area_km2", "suitable_rate", "mean_probability"], ascending=False)
        .head(30)
        .reset_index(drop=True)
    )

    atomic_write_csv(GLOBAL_SUMMARY_CSV, global_summary)
    atomic_write_csv(MACRO_REGION_SUMMARY_CSV, macro_summary)
    atomic_write_csv(TOP_TILES_CSV, top_tiles)
    return global_summary, macro_summary, top_tiles


def plot_tile_suitable_rate(tile_df: pd.DataFrame) -> dict[str, str]:
    plot_df = tile_df[tile_df["valid_area_km2"].fillna(0) > 0].copy()
    patches: list[Rectangle] = []
    values: list[float] = []
    for _, row in plot_df.iterrows():
        patches.append(
            Rectangle(
                (row["min_lon"], row["min_lat"]),
                row["max_lon"] - row["min_lon"],
                row["max_lat"] - row["min_lat"],
            )
        )
        values.append(float(row["suitable_rate"] or 0) * 100)

    fig, ax = plt.subplots(figsize=(8.4, 3.9))
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "dryland_suitable_rate",
        ["#F2EFE7", "#D8C08C", "#A6B957", "#4E8F5B"],
    )
    collection = PatchCollection(patches, cmap=cmap, edgecolor="#B8B8B8", linewidth=0.18)
    collection.set_array(np.asarray(values))
    collection.set_clim(0, max(1.0, np.nanpercentile(values, 98) if values else 1.0))
    ax.add_collection(collection)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 75)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xticks(np.arange(-180, 181, 60))
    ax.set_yticks(np.arange(-60, 76, 30))
    ax.grid(color=OWID_COLORS["light_grid"], linewidth=0.45, alpha=0.7)
    cbar = fig.colorbar(collection, ax=ax, fraction=0.024, pad=0.015)
    cbar.set_label("Suitable proportion (%)")
    cbar.outline.set_linewidth(0.5)
    return save_figure(fig, "fig_stage13_global_dryland_tile_suitable_rate")


def plot_macro_region_suitable_area(macro_summary: pd.DataFrame) -> dict[str, str]:
    df = macro_summary.copy()
    df = df[df["valid_area_km2"].fillna(0) > 0]
    df = df.sort_values("suitable_area_km2", ascending=True)
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    y = np.arange(len(df))
    ax.barh(y, df["suitable_area_km2"] / 1000, color=OWID_COLORS["green"], alpha=0.92)
    ax.set_yticks(y)
    ax.set_yticklabels(df["macro_region"])
    ax.set_xlabel("Threshold-suitable area (10^3 km2)")
    ax.grid(axis="x", color=OWID_COLORS["light_grid"], linewidth=0.5)
    for idx, value in enumerate(df["suitable_area_km2"] / 1000):
        if value > 0:
            ax.text(value, idx, f" {value:,.0f}", va="center", ha="left", fontsize=7.5, color="#333333")
    return save_figure(fig, "fig_stage13_macro_region_suitable_area")


def plot_probability_area_scatter(tile_df: pd.DataFrame) -> dict[str, str]:
    df = tile_df[tile_df["valid_area_km2"].fillna(0) > 0].copy()
    sizes = np.clip(np.sqrt(df["valid_area_km2"].fillna(0)) * 0.22, 8, 90)
    colors = df["suitable_rate"].fillna(0) * 100
    fig, ax = plt.subplots(figsize=(5.2, 3.7))
    sc = ax.scatter(
        df["mean_probability"],
        df["suitable_area_km2"],
        s=sizes,
        c=colors,
        cmap=mcolors.LinearSegmentedColormap.from_list("rate", ["#D8C08C", "#A6B957", "#4E8F5B"]),
        edgecolor="#4F4F4F",
        linewidth=0.25,
        alpha=0.86,
    )
    ax.set_xlabel("Mean suitability probability")
    ax.set_ylabel("Threshold-suitable area (km2)")
    ax.grid(color=OWID_COLORS["light_grid"], linewidth=0.5)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.02)
    cbar.set_label("Suitable proportion (%)")
    return save_figure(fig, "fig_stage13_tile_probability_area_scatter")


def write_report(
    global_summary: pd.DataFrame,
    macro_summary: pd.DataFrame,
    top_tiles: pd.DataFrame,
    figures: dict[str, dict[str, str]],
) -> None:
    g = global_summary.iloc[0].to_dict()
    valid_area = float(g["valid_area_km2"])
    suitable_area = float(g["suitable_area_km2"])
    suitable_rate = float(g["area_weighted_suitable_rate"])
    mean_prob = float(g["area_weighted_mean_probability"])
    threshold = float(g["threshold"])

    top = top_tiles.head(10).copy()
    top_table = top[
        [
            "tile_id",
            "macro_region",
            "valid_area_km2",
            "suitable_area_km2",
            "suitable_rate",
            "mean_probability",
        ]
    ].copy()
    for col in ["valid_area_km2", "suitable_area_km2", "suitable_rate", "mean_probability"]:
        top_table[col] = top_table[col].astype(float)

    macro_table = macro_summary[
        [
            "macro_region",
            "tile_count",
            "valid_area_km2",
            "suitable_area_km2",
            "area_weighted_suitable_rate",
            "area_weighted_mean_probability",
        ]
    ].copy()

    lines = [
        "# Stage13 全球干旱区 tile 投影汇总报告",
        "",
        f"生成时间：{now_iso()}",
        "",
        "## 数据来源",
        "",
        f"- 输入状态表：`{STAGE12_STATUS_CSV}`",
        f"- 输入 tile 清单：`{STAGE12_MANIFEST_CSV}`",
        f"- 模型组：`{g.get('model_group')}`",
        f"- 情景：`{g.get('gcm')} / {g.get('ssp')} / {g.get('period')}`",
        "- 空间范围：全球干旱区掩膜内 10° tile。",
        "",
        "## 核心统计",
        "",
        f"- tile 总数：{int(g['tile_count'])}，成功 tile：{int(g['success_tile_count'])}，有效面积 tile：{int(g['valid_tile_count'])}。",
        f"- 干旱区有效面积估算：{valid_area:,.2f} km²。",
        f"- 阈值适宜面积估算：{suitable_area:,.2f} km²。",
        f"- 面积加权适宜比例：{suitable_rate * 100:.2f}%。",
        f"- 面积加权平均适宜概率：{mean_prob:.4f}。",
        f"- 使用阈值：{threshold:.4f}。",
        "",
        "## 分区汇总",
        "",
        macro_table.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## 适宜面积最高的 tile",
        "",
        top_table.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## 图件",
        "",
    ]
    for name, outputs in figures.items():
        lines.append(f"- {name}: PNG `{outputs['png']}`；SVG `{outputs['svg']}`")
    lines.extend(
        [
            "",
            "## 投稿解释边界",
            "",
            "- 这是 HGB 气候-高程基线模型在单一未来情景（ACCESS-CM2 / SSP585 / 2081-2100）下的全球干旱区 tile 投影汇总。",
            "- 该结果可作为全图投影链条和末世纪高排放情景空间格局的阶段性证据。",
            "- 不能单独代表多 GCM、多 SSP、多时期的稳健未来结论；正式面积结论仍需扩展到全情景集合并给出不确定性。",
            "- 宏区域分组基于 tile 中心经纬度的启发式归类，仅用于快速汇总展示，不等同于行政区或严格生态区边界。",
        ]
    )
    atomic_write_text(REPORT_MD, "\n".join(lines) + "\n")


def write_updated_draft(global_summary: pd.DataFrame, figures: dict[str, dict[str, str]]) -> None:
    if STAGE11_DRAFT.exists():
        base = STAGE11_DRAFT.read_text(encoding="utf-8").rstrip()
    else:
        base = "# 未来绿洲潜在适宜区预测\n"
    g = global_summary.iloc[0].to_dict()
    section = f"""

## 全球干旱区 tile 投影阶段性结果

在完成北美窗口试跑后，本研究进一步将 HGB 气候-高程基线模型扩展到全球干旱区 10° tile。该阶段使用 ACCESS-CM2 / SSP585 / 2081-2100 情景，针对 171 个干旱区 tile 运行分块预测，其中 171 个任务完成且失败为 0。根据 tile 级像元面积汇总，干旱区有效面积约 {float(g['valid_area_km2']):,.2f} km²，阈值适宜面积约 {float(g['suitable_area_km2']):,.2f} km²，面积加权适宜比例为 {float(g['area_weighted_suitable_rate']) * 100:.2f}%，面积加权平均适宜概率为 {float(g['area_weighted_mean_probability']):.4f}。

该结果表明，从点位样本预测到全球干旱区栅格投影的计算链条已经贯通，可支撑后续多 GCM、多 SSP 和多时期批量投影。当前结果仍应表述为单一高排放末世纪情景下的阶段性空间投影，不应直接替代完整情景集合的稳健面积结论。图件和表格见 `{OUT_DIR}`。
"""
    atomic_write_text(UPDATED_DRAFT_MD, base + section)


def run() -> dict[str, Any]:
    ensure_dirs()
    write_status("running", "Stage13 summary started")
    base_style()
    tile_df = load_tile_metrics()
    global_summary, macro_summary, top_tiles = build_summaries(tile_df)
    figures = {
        "global_tile_suitable_rate": plot_tile_suitable_rate(tile_df),
        "macro_region_suitable_area": plot_macro_region_suitable_area(macro_summary),
        "tile_probability_area_scatter": plot_probability_area_scatter(tile_df),
    }
    write_report(global_summary, macro_summary, top_tiles, figures)
    write_updated_draft(global_summary, figures)
    summary = {
        "status": "success",
        "generated_at": now_iso(),
        "input_status_csv": str(STAGE12_STATUS_CSV),
        "tile_metrics_csv": str(TILE_METRICS_CSV),
        "global_summary_csv": str(GLOBAL_SUMMARY_CSV),
        "macro_region_summary_csv": str(MACRO_REGION_SUMMARY_CSV),
        "top_tiles_csv": str(TOP_TILES_CSV),
        "report_md": str(REPORT_MD),
        "updated_draft_md": str(UPDATED_DRAFT_MD),
        "figures": figures,
        "global_summary": global_summary.iloc[0].to_dict(),
    }
    atomic_write_json(SUMMARY_JSON, summary)
    write_status("success", "Stage13 summary completed", report=str(REPORT_MD))
    logging.info("Stage13 completed: %s", json.dumps(summary, ensure_ascii=False))
    return summary


def main() -> int:
    setup_logging()
    try:
        run()
        return 0
    except Exception as exc:
        logging.error("Stage13 failed: %s", exc)
        logging.error(traceback.format_exc())
        write_status("failed", str(exc), traceback=traceback.format_exc()[-3500:])
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
