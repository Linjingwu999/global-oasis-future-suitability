# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
from pyproj import Geod
from rasterio.features import geometry_mask
from rasterio.windows import Window, from_bounds
from shapely.validation import make_valid


PROJ_DIR = Path(r"C:\Users\linjingwu\anaconda3\Library\share\proj")
if PROJ_DIR.exists():
    os.environ["PROJ_LIB"] = str(PROJ_DIR)

WORKSPACE = Path(r"C:\Users\linjingwu\Desktop\python")
PROJECT_ROOT = WORKSPACE / "绿洲未来适宜区预测"
DATA_ROOT = Path(r"D:\绿洲未来适宜区预测数据")

FUTURE_DIR = DATA_ROOT / "raw" / "worldclim" / "future_30s"
ELEV_TIF = DATA_ROOT / "processed" / "worldclim" / "current_30s" / "elev" / "wc2.1_30s_elev.tif"
MODEL_DIR = PROJECT_ROOT / "outputs" / "stage06_current_worldclim_baseline_models" / "models"
DEFAULT_DRYLAND_MASK = Path.home() / "Desktop" / "会议相关" / "世界绿洲合并" / "世界绿洲" / "全球干旱区" / "AI0-0.65干旱区.shp"

OUT_DIR = PROJECT_ROOT / "outputs" / "stage10_future_grid_projection_pilot"
RASTER_DIR = OUT_DIR / "rasters"
FIG_DIR = OUT_DIR / "figures"
TABLE_DIR = OUT_DIR / "tables"
LOG_DIR = PROJECT_ROOT / "logs"

LOG_PATH = LOG_DIR / "stage10_future_grid_projection_pilot.log"
STATE_PATH = LOG_DIR / "stage10_future_grid_projection_pilot_state.json"
STATUS_CSV = LOG_DIR / "stage10_future_grid_projection_pilot_status.csv"
SUMMARY_JSON = OUT_DIR / "stage10_future_grid_projection_pilot_summary.json"
SUMMARY_CSV = TABLE_DIR / "stage10_future_grid_projection_pilot_summary.csv"
REPORT_MD = OUT_DIR / "stage10_future_grid_projection_pilot_report.md"

FUTURE_RE = re.compile(
    r"wc2\.1_30s_bioc_(?P<gcm>.+?)_(?P<ssp>ssp\d+)_(?P<period>\d{4}-\d{4})\.tif$",
    re.IGNORECASE,
)

DEFAULT_BBOX = (-130.0, 20.0, -90.0, 55.0)
DEFAULT_BBOX_NAME = "north_america_west_pilot"


@dataclass(frozen=True)
class Scenario:
    gcm: str
    ssp: str
    period: str
    path: Path

    @property
    def key(self) -> str:
        return f"{self.gcm}__{self.ssp}__{self.period}"


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def setup_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
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


def atomic_write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        atomic_write_text(path, "")
        return
    import pandas as pd

    tmp = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(rows).to_csv(tmp, index=False, encoding="utf-8-sig")
    tmp.replace(path)


def write_status(status: str, message: str, extra: dict[str, Any] | None = None) -> None:
    row = {"updated_at": now_iso(), "status": status, "message": message}
    if extra:
        row.update(extra)
    atomic_write_csv(STATUS_CSV, [row])


def discover_scenario(gcm: str, ssp: str, period: str) -> Scenario:
    matches: list[Scenario] = []
    for path in FUTURE_DIR.rglob("*.tif"):
        match = FUTURE_RE.match(path.name)
        if not match:
            continue
        scenario = Scenario(match.group("gcm"), match.group("ssp"), match.group("period"), path)
        if scenario.gcm == gcm and scenario.ssp == ssp and scenario.period == period:
            matches.append(scenario)
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one scenario for {gcm}/{ssp}/{period}, found {len(matches)}")
    return matches[0]


def load_model_group(model_group: str) -> tuple[list[Any], list[str], float]:
    files = sorted(MODEL_DIR.glob(f"{model_group}_fold*.joblib"))
    if not files:
        raise FileNotFoundError(f"No model files found for model_group={model_group}: {MODEL_DIR}")
    models: list[Any] = []
    thresholds: list[float] = []
    features: list[str] | None = None
    for file in files:
        obj = joblib.load(file)
        models.append(obj["model"])
        thresholds.append(float(obj["threshold"]))
        this_features = list(obj["features"])
        if features is None:
            features = this_features
        elif this_features != features:
            raise ValueError(f"Feature mismatch in {file}")
    return models, features or [], float(np.mean(thresholds))


def expected_features() -> list[str]:
    return [f"wc_bio{i:02d}" for i in range(1, 20)] + ["wc_elev_m"]


def assemble_block_feature_matrix(bio: np.ndarray, elev: np.ndarray, features: list[str]) -> np.ndarray:
    if not features:
        raise ValueError("Model feature list is empty.")
    columns: list[np.ndarray] = []
    for feature in features:
        if feature == "wc_elev_m":
            columns.append(elev.reshape(-1))
            continue
        match = re.fullmatch(r"wc_bio(\d{2})", feature)
        if not match:
            raise ValueError(f"Unsupported model feature for grid projection: {feature}")
        band_index = int(match.group(1)) - 1
        if band_index < 0 or band_index >= bio.shape[0]:
            raise ValueError(f"Feature {feature} is outside future bioclim band range.")
        columns.append(bio[band_index].reshape(-1))
    return np.column_stack(columns).astype("float32", copy=False)


def safe_name(text: str) -> str:
    out = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return out.strip("._-") or "run"


def configure_paths(args: argparse.Namespace) -> None:
    global MODEL_DIR, OUT_DIR, RASTER_DIR, FIG_DIR, TABLE_DIR, LOG_PATH, STATE_PATH, STATUS_CSV, SUMMARY_JSON, SUMMARY_CSV, REPORT_MD
    MODEL_DIR = Path(args.model_dir)
    OUT_DIR = Path(args.output_dir)
    RASTER_DIR = OUT_DIR / "rasters"
    FIG_DIR = OUT_DIR / "figures"
    TABLE_DIR = OUT_DIR / "tables"
    suffix = safe_name(args.run_label) if args.run_label else "stage10_future_grid_projection_pilot"
    LOG_PATH = LOG_DIR / f"{suffix}.log"
    STATE_PATH = LOG_DIR / f"{suffix}_state.json"
    STATUS_CSV = LOG_DIR / f"{suffix}_status.csv"
    SUMMARY_JSON = OUT_DIR / f"{suffix}_summary.json"
    SUMMARY_CSV = TABLE_DIR / f"{suffix}_summary.csv"
    REPORT_MD = OUT_DIR / f"{suffix}_report.md"


def rasters_aligned(src: rasterio.DatasetReader, other: rasterio.DatasetReader, tol: float = 1e-10) -> bool:
    if src.width != other.width or src.height != other.height or src.crs != other.crs:
        return False
    return all(abs(a - b) <= tol for a, b in zip(src.transform, other.transform))


def load_mask_geometries(mask_vector: str | None, dst_crs: Any) -> list[Any]:
    if not mask_vector:
        return []
    path = Path(mask_vector)
    if not path.exists():
        raise FileNotFoundError(f"Mask vector not found: {path}")
    gdf = gpd.read_file(path)
    if gdf.empty:
        raise ValueError(f"Mask vector is empty: {path}")
    if gdf.crs is None:
        raise ValueError(f"Mask vector CRS is missing: {path}")
    if gdf.crs != dst_crs:
        gdf = gdf.to_crs(dst_crs)
    fixed = []
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        if not geom.is_valid:
            geom = make_valid(geom)
        if geom is not None and not geom.is_empty:
            fixed.append(geom)
    if not fixed:
        raise ValueError(f"No valid geometries after repair: {path}")
    return fixed


def align_window(window: Window, width: int, height: int) -> Window:
    col_off = max(0, int(np.floor(window.col_off)))
    row_off = max(0, int(np.floor(window.row_off)))
    col_stop = min(width, int(np.ceil(window.col_off + window.width)))
    row_stop = min(height, int(np.ceil(window.row_off + window.height)))
    if col_stop <= col_off or row_stop <= row_off:
        raise ValueError(f"Invalid clipped window: {window}")
    return Window(col_off, row_off, col_stop - col_off, row_stop - row_off)


def window_from_args(src: rasterio.DatasetReader, args: argparse.Namespace) -> tuple[Window, tuple[float, float, float, float], str]:
    if args.global_grid:
        return Window(0, 0, src.width, src.height), (src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top), "global"
    bbox = tuple(args.bbox) if args.bbox else DEFAULT_BBOX
    raw = from_bounds(*bbox, transform=src.transform)
    window = align_window(raw, src.width, src.height)
    bounds = rasterio.windows.bounds(window, src.transform)
    name = safe_name(args.bbox_name or DEFAULT_BBOX_NAME)
    return window, (bounds[0], bounds[1], bounds[2], bounds[3]), name


def iter_block_windows(window: Window, block_size: int) -> list[Window]:
    out: list[Window] = []
    row0 = int(window.row_off)
    col0 = int(window.col_off)
    row1 = int(window.row_off + window.height)
    col1 = int(window.col_off + window.width)
    for row in range(row0, row1, block_size):
        for col in range(col0, col1, block_size):
            out.append(Window(col, row, min(block_size, col1 - col), min(block_size, row1 - row)))
    return out


def predict_ensemble(models: list[Any], x: np.ndarray) -> np.ndarray:
    probs = np.zeros((x.shape[0], len(models)), dtype="float32")
    for idx, model in enumerate(models):
        probs[:, idx] = model.predict_proba(x)[:, 1].astype("float32")
    return probs.mean(axis=1)


def row_cell_areas_km2(transform: rasterio.Affine, window: Window) -> np.ndarray:
    geod = Geod(ellps="WGS84")
    col0 = int(window.col_off)
    col1 = int(window.col_off + window.width)
    lon_left = transform.c + col0 * transform.a
    lon_right = transform.c + col1 * transform.a
    rows = np.arange(int(window.row_off), int(window.row_off + window.height))
    lat_top = transform.f + rows * transform.e
    lat_bottom = lat_top + transform.e
    areas = np.empty(len(rows), dtype="float64")
    for idx, (top, bottom) in enumerate(zip(lat_top, lat_bottom)):
        area, _ = geod.polygon_area_perimeter([lon_left, lon_right, lon_right, lon_left], [top, top, bottom, bottom])
        areas[idx] = abs(area) / 1_000_000.0
    return areas


def make_figure(probability_tif: Path, scenario: Scenario, bbox_name: str) -> dict[str, str]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    with rasterio.open(probability_tif) as src:
        arr = src.read(1, masked=True)
        extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
    cmap = LinearSegmentedColormap.from_list("oasis_suitability", ["#F2F0E6", "#9CCB86", "#2E7D59"])
    fig, ax = plt.subplots(figsize=(5.2, 3.5))
    im = ax.imshow(arr, extent=extent, origin="upper", cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(color="#D9D9D9", linewidth=0.35)
    cbar = fig.colorbar(im, ax=ax, fraction=0.032, pad=0.02)
    cbar.set_label("Suitability probability")
    ax.text(0.01, 0.99, f"{scenario.gcm} {scenario.ssp.upper()} {scenario.period}", transform=ax.transAxes, va="top", ha="left", fontsize=8)
    png = FIG_DIR / f"fig_stage10_{bbox_name}_{scenario.key}_probability.png"
    svg = FIG_DIR / f"fig_stage10_{bbox_name}_{scenario.key}_probability.svg"
    fig.savefig(png, dpi=600, bbox_inches="tight", pad_inches=0.02, transparent=True)
    fig.savefig(svg, bbox_inches="tight", pad_inches=0.02, transparent=True)
    plt.close(fig)
    img = Image.open(png)
    assert img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info), "PNG lacks transparency"
    return {"png": str(png), "svg": str(svg)}


def run(args: argparse.Namespace) -> dict[str, Any]:
    RASTER_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    scenario = discover_scenario(args.gcm, args.ssp, args.period)
    models, features, threshold = load_model_group(args.model_group)

    out_prefix = f"{safe_name(args.model_group)}__{scenario.key}__{safe_name(args.bbox_name or DEFAULT_BBOX_NAME if not args.global_grid else 'global')}"
    probability_tif = RASTER_DIR / f"{out_prefix}_probability.tif"
    suitable_tif = RASTER_DIR / f"{out_prefix}_suitable_threshold.tif"
    if probability_tif.exists() and suitable_tif.exists() and not args.overwrite:
        state = {"status": "success", "message": "existing outputs skipped", "probability_tif": str(probability_tif), "suitable_tif": str(suitable_tif)}
        atomic_write_json(STATE_PATH, state)
        write_status("success", "existing outputs skipped", state)
        return state

    atomic_write_json(STATE_PATH, {"status": "running", "started_at": now_iso(), "scenario": scenario.key, "model_group": args.model_group})
    write_status("running", "stage10 grid projection started", {"scenario": scenario.key, "model_group": args.model_group})

    with rasterio.open(scenario.path) as src, rasterio.open(ELEV_TIF) as elev_src:
        if src.count != 19:
            raise ValueError(f"Expected 19 bands, got {src.count}: {scenario.path}")
        if not rasters_aligned(src, elev_src):
            raise ValueError("Future bioclim and elevation rasters are not aligned.")

        window, bbox, bbox_name = window_from_args(src, args)
        block_windows = iter_block_windows(window, args.block_size)
        mask_geometries = load_mask_geometries(args.mask_vector, src.crs)
        window_transform = src.window_transform(window)
        height = int(window.height)
        width = int(window.width)
        profile_base = src.profile.copy()
        profile_base.update(
            driver="GTiff",
            height=height,
            width=width,
            count=1,
            crs=src.crs,
            transform=window_transform,
            tiled=True,
            blockxsize=256,
            blockysize=256,
            compress="DEFLATE",
            BIGTIFF="YES",
        )
        prob_profile = profile_base.copy()
        prob_profile.update(dtype="float32", nodata=-9999.0, predictor=2)
        suitable_profile = profile_base.copy()
        suitable_profile.update(dtype="uint8", nodata=255)

        tmp_prob = probability_tif.with_suffix(probability_tif.suffix + ".tmp")
        tmp_suit = suitable_tif.with_suffix(suitable_tif.suffix + ".tmp")
        for tmp in [tmp_prob, tmp_suit]:
            if tmp.exists():
                tmp.unlink()

        valid_pixels = 0
        suitable_pixels = 0
        probability_sum = 0.0
        probability_min = np.inf
        probability_max = -np.inf
        valid_area_km2 = 0.0
        suitable_area_km2 = 0.0
        row_areas = row_cell_areas_km2(src.transform, window)

        with rasterio.open(tmp_prob, "w", **prob_profile) as prob_dst, rasterio.open(tmp_suit, "w", **suitable_profile) as suit_dst:
            for idx, src_win in enumerate(block_windows, 1):
                out_win = Window(src_win.col_off - window.col_off, src_win.row_off - window.row_off, src_win.width, src_win.height)
                bio = src.read(indexes=list(range(1, 20)), window=src_win).astype("float32")
                elev = elev_src.read(1, window=src_win).astype("float32")
                elev[elev == elev_src.nodata] = np.nan
                x = assemble_block_feature_matrix(bio, elev, features)
                valid = np.isfinite(x).all(axis=1).reshape(elev.shape)
                if mask_geometries:
                    dryland = geometry_mask(
                        mask_geometries,
                        out_shape=(int(src_win.height), int(src_win.width)),
                        transform=src.window_transform(src_win),
                        invert=True,
                        all_touched=args.mask_all_touched,
                    )
                    valid &= dryland
                prob = np.full((int(src_win.height), int(src_win.width)), -9999.0, dtype="float32")
                suitable = np.full((int(src_win.height), int(src_win.width)), 255, dtype="uint8")
                if valid.any():
                    flat_valid = valid.reshape(-1)
                    pred = predict_ensemble(models, x[flat_valid])
                    flat_prob = prob.reshape(-1)
                    flat_suit = suitable.reshape(-1)
                    flat_prob[flat_valid] = pred
                    flat_suit[flat_valid] = (pred >= threshold).astype("uint8")

                    block_valid = int(flat_valid.sum())
                    block_suitable = int(flat_suit[flat_valid].sum())
                    valid_pixels += block_valid
                    suitable_pixels += block_suitable
                    probability_sum += float(pred.sum())
                    probability_min = min(probability_min, float(pred.min()))
                    probability_max = max(probability_max, float(pred.max()))
                    rel_rows = np.arange(int(src_win.row_off - window.row_off), int(src_win.row_off - window.row_off + src_win.height))
                    per_pixel_area = row_areas[rel_rows] / float(window.width)
                    valid_area_km2 += float((valid * per_pixel_area[:, None]).sum())
                    suitable_area_km2 += float(((suitable == 1) * per_pixel_area[:, None]).sum())
                prob_dst.write(prob, 1, window=out_win)
                suit_dst.write(suitable, 1, window=out_win)
                if idx == 1 or idx % args.status_every == 0 or idx == len(block_windows):
                    write_status(
                        "running",
                        "processing windows",
                        {"completed_windows": idx, "total_windows": len(block_windows), "valid_pixels": valid_pixels, "suitable_pixels": suitable_pixels},
                    )
        tmp_prob.replace(probability_tif)
        tmp_suit.replace(suitable_tif)

    mean_probability = probability_sum / valid_pixels if valid_pixels else float("nan")
    suitable_rate = suitable_pixels / valid_pixels if valid_pixels else float("nan")
    summary = {
        "status": "success",
        "generated_at": now_iso(),
        "scenario": scenario.key,
        "gcm": scenario.gcm,
        "ssp": scenario.ssp,
        "period": scenario.period,
        "model_group": args.model_group,
        "threshold": threshold,
        "bbox_name": bbox_name,
        "bbox": bbox,
        "width": width,
        "height": height,
        "block_size": args.block_size,
        "mask_vector": args.mask_vector,
        "mask_all_touched": args.mask_all_touched,
        "valid_pixels": int(valid_pixels),
        "suitable_pixels": int(suitable_pixels),
        "valid_area_km2": valid_area_km2,
        "suitable_area_km2": suitable_area_km2,
        "mean_probability": mean_probability,
        "min_probability": None if not np.isfinite(probability_min) else probability_min,
        "max_probability": None if not np.isfinite(probability_max) else probability_max,
        "suitable_rate": suitable_rate,
        "probability_tif": str(probability_tif),
        "suitable_tif": str(suitable_tif),
    }
    figure_paths = make_figure(probability_tif, scenario, bbox_name) if args.make_figure else {}
    summary["figure"] = figure_paths
    atomic_write_json(SUMMARY_JSON, summary)
    atomic_write_json(STATE_PATH, summary)
    atomic_write_csv(SUMMARY_CSV, [summary])
    report_lines = [
        "# Stage10 未来适宜性栅格投影试跑报告",
        "",
        f"- 生成时间: {summary['generated_at']}",
        f"- 情景: {summary['scenario']}",
        f"- 模型组: {summary['model_group']}",
        f"- 阈值: {threshold:.6f}",
        f"- 区域: {bbox_name}; bbox={bbox}",
        f"- 栅格尺寸: {width} x {height}",
        f"- 有效像元: {valid_pixels}",
        f"- 适宜像元: {suitable_pixels}",
        f"- 平均适宜概率: {mean_probability:.4f}",
        f"- 适宜像元比例: {suitable_rate:.4f}",
        f"- 有效面积估算: {valid_area_km2:.2f} km2",
        f"- 适宜面积估算: {suitable_area_km2:.2f} km2",
        "",
        "## 输出",
        "",
        f"- 连续概率 GeoTIFF: `{probability_tif}`",
        f"- 阈值适宜区 GeoTIFF: `{suitable_tif}`",
    ]
    if figure_paths:
        report_lines.extend(["", "## 图件", "", f"- PNG: `{figure_paths['png']}`", f"- SVG: `{figure_paths['svg']}`"])
    atomic_write_text(REPORT_MD, "\n".join(report_lines))
    write_status("success", "stage10 grid projection completed", summary)
    logging.info("Stage10 completed: %s", json.dumps(summary, ensure_ascii=False))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="未来 WorldClim 情景下全图/区域适宜性栅格投影，默认只做安全 bbox 试跑。")
    parser.add_argument("--model-group", default="hist_gradient_boosting_balanced")
    parser.add_argument("--model-dir", default=str(MODEL_DIR), help="模型目录，默认使用 Stage06 20 因子基线模型。")
    parser.add_argument("--output-dir", default=str(OUT_DIR), help="输出目录，默认使用 Stage10 试跑目录。")
    parser.add_argument("--run-label", default=None, help="日志、状态、摘要文件前缀；用于 selected10 等独立重跑。")
    parser.add_argument("--gcm", default="ACCESS-CM2")
    parser.add_argument("--ssp", default="ssp585")
    parser.add_argument("--period", default="2081-2100")
    parser.add_argument("--bbox", nargs=4, type=float, metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"), default=None)
    parser.add_argument("--bbox-name", default=DEFAULT_BBOX_NAME)
    parser.add_argument("--global-grid", action="store_true", help="Run the full global 30 arc-second grid. This can create multi-GB outputs.")
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--status-every", type=int, default=10)
    parser.add_argument("--mask-vector", default=None, help="Optional polygon vector mask. Pixels outside the mask are written as NoData.")
    parser.add_argument("--use-default-dryland-mask", action="store_true", help="Use the AI0-0.65 global dryland polygon as mask.")
    parser.add_argument("--mask-all-touched", action="store_true")
    parser.add_argument("--make-figure", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.use_default_dryland_mask and not args.mask_vector:
        args.mask_vector = str(DEFAULT_DRYLAND_MASK)
    return args


def main() -> int:
    args = parse_args()
    configure_paths(args)
    setup_logging()
    try:
        state = run(args)
        return 0 if state.get("status") == "success" else 1
    except Exception as exc:
        err = {"status": "failed", "failed_at": now_iso(), "error": repr(exc), "traceback": traceback.format_exc()}
        atomic_write_json(STATE_PATH, err)
        write_status("failed", repr(exc))
        logging.exception("Stage10 failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
