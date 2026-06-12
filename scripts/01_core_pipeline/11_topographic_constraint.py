# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from pyproj import CRS
from rasterio.features import geometry_mask
from pyproj import Geod
from rasterio.windows import Window, from_bounds
from shapely.ops import unary_union
from shapely.validation import make_valid


PROJ_DIR = Path(r"C:\Users\linjingwu\anaconda3\Library\share\proj")
if PROJ_DIR.exists():
    os.environ["PROJ_LIB"] = str(PROJ_DIR)

def env_path(name: str, default: str | Path) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser()


WORKSPACE = env_path("OASIS_WORKSPACE", r"C:\Users\linjingwu\Desktop\python")
PROJECT_ROOT = env_path("OASIS_PROJECT_ROOT", WORKSPACE / "绿洲未来适宜区预测")
DATA_ROOT = env_path("OASIS_DATA_ROOT", r"D:\绿洲未来适宜区预测数据")

STAGE12_STATUS_CSV = PROJECT_ROOT / "logs" / "stage12_region_tile_grid_projection_status.csv"
ELEV_TIF = DATA_ROOT / "processed" / "worldclim" / "current_30s" / "elev" / "wc2.1_30s_elev.tif"
DEFAULT_OASIS_VECTOR = env_path(
    "OASIS_OASIS_VECTOR",
    Path.home() / "Desktop" / "会议相关" / "世界绿洲合并" / "世界绿洲" / "世界绿洲-崔.shp",
)
DEFAULT_RIVER_VECTOR = env_path(
    "OASIS_RIVER_VECTOR",
    Path.home() / "Desktop" / "会议相关" / "世界绿洲合并" / "世界绿洲" / "HydroRIVERS_v10.gdb",
)
DEFAULT_RIVER_LAYER = "HydroRIVERS_v10"

OUT_DIR = PROJECT_ROOT / "outputs" / "stage17_constrained_suitability"
RASTER_DIR = OUT_DIR / "rasters"
TABLE_DIR = OUT_DIR / "tables"
LOG_DIR = PROJECT_ROOT / "logs"
LOG_PATH = LOG_DIR / "stage17_constrained_suitability.log"
STATUS_CSV = LOG_DIR / "stage17_constrained_suitability_status.csv"
STATE_JSON = LOG_DIR / "stage17_constrained_suitability_state.json"
SUMMARY_CSV = TABLE_DIR / "stage17_constrained_suitability_summary.csv"
SUMMARY_JSON = OUT_DIR / "stage17_constrained_suitability_summary.json"
REPORT_MD = OUT_DIR / "Stage17_多约束适宜区后处理报告.md"

NODATA_UINT8 = 255


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RASTER_DIR.mkdir(parents=True, exist_ok=True)
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


def safe_name(text: str) -> str:
    out = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text))
    return out.strip("._-") or "unnamed"


def configure_paths(args: argparse.Namespace) -> None:
    global STAGE12_STATUS_CSV, OUT_DIR, RASTER_DIR, TABLE_DIR, LOG_PATH, STATUS_CSV, STATE_JSON, SUMMARY_CSV, SUMMARY_JSON, REPORT_MD
    STAGE12_STATUS_CSV = Path(args.stage12_status_csv)
    OUT_DIR = Path(args.output_dir)
    RASTER_DIR = OUT_DIR / "rasters"
    TABLE_DIR = OUT_DIR / "tables"
    prefix = "stage17_constrained_suitability"
    if args.run_label:
        prefix = f"{prefix}_{safe_name(args.run_label)}"
    LOG_PATH = LOG_DIR / f"{prefix}.log"
    STATUS_CSV = LOG_DIR / f"{prefix}_status.csv"
    STATE_JSON = LOG_DIR / f"{prefix}_state.json"
    SUMMARY_CSV = TABLE_DIR / f"{prefix}_summary.csv"
    SUMMARY_JSON = OUT_DIR / f"{prefix}_summary.json"
    REPORT_MD = OUT_DIR / f"{prefix}_report.md"


def number_token(value: Any) -> str:
    if value is None:
        return "none"
    try:
        value_float = float(value)
    except (TypeError, ValueError):
        return safe_name(str(value))
    if math.isfinite(value_float) and value_float.is_integer():
        return str(int(value_float))
    return safe_name(f"{value_float:g}")


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


def rasters_aligned(src: rasterio.DatasetReader, other: rasterio.DatasetReader, tol: float = 1e-10) -> bool:
    if src.width != other.width or src.height != other.height or src.crs != other.crs:
        return False
    return all(abs(a - b) <= tol for a, b in zip(src.transform, other.transform))


def grids_have_same_resolution(src: rasterio.DatasetReader, other: rasterio.DatasetReader, tol: float = 1e-10) -> bool:
    if src.crs != other.crs:
        return False
    return abs(src.transform.a - other.transform.a) <= tol and abs(src.transform.e - other.transform.e) <= tol


def align_window(window: Window, width: int, height: int) -> Window:
    col_off = max(0, int(round(window.col_off)))
    row_off = max(0, int(round(window.row_off)))
    col_stop = min(width, int(round(window.col_off + window.width)))
    row_stop = min(height, int(round(window.row_off + window.height)))
    if col_stop <= col_off or row_stop <= row_off:
        raise ValueError(f"Invalid clipped window: {window}")
    return Window(col_off, row_off, col_stop - col_off, row_stop - row_off)


def matching_elevation_window(tile_src: rasterio.DatasetReader, elev_src: rasterio.DatasetReader) -> Window:
    if not grids_have_same_resolution(tile_src, elev_src):
        raise ValueError("Suitable raster and elevation raster have different CRS or pixel size.")
    raw = from_bounds(
        tile_src.bounds.left,
        tile_src.bounds.bottom,
        tile_src.bounds.right,
        tile_src.bounds.top,
        transform=elev_src.transform,
    )
    win = align_window(raw, elev_src.width, elev_src.height)
    if int(win.width) != tile_src.width or int(win.height) != tile_src.height:
        raise ValueError(
            "Matched elevation window has different shape: "
            f"tile={tile_src.width}x{tile_src.height}, elev_window={int(win.width)}x{int(win.height)}"
        )
    return win


def load_oasis_buffer_geometries(
    oasis_vector: str | None,
    tile_crs: Any,
    tile_bounds: rasterio.coords.BoundingBox,
    buffer_km: float | None,
) -> list[Any]:
    if not oasis_vector or not buffer_km or buffer_km <= 0:
        return []
    path = Path(oasis_vector)
    if not path.exists():
        raise FileNotFoundError(f"Oasis vector not found: {path}")

    center_lon = (tile_bounds.left + tile_bounds.right) / 2
    center_lat = (tile_bounds.bottom + tile_bounds.top) / 2
    cos_lat = max(abs(math.cos(math.radians(center_lat))), 0.15)
    buffer_deg_lon = buffer_km / (111.32 * cos_lat)
    buffer_deg_lat = buffer_km / 111.32
    bbox = (
        tile_bounds.left - buffer_deg_lon,
        tile_bounds.bottom - buffer_deg_lat,
        tile_bounds.right + buffer_deg_lon,
        tile_bounds.top + buffer_deg_lat,
    )

    gdf = gpd.read_file(path, bbox=bbox)
    if gdf.empty:
        return []
    if gdf.crs is None:
        raise ValueError(f"Oasis vector CRS is missing: {path}")
    if gdf.crs != tile_crs:
        gdf = gdf.to_crs(tile_crs)
    fixed = []
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        if not geom.is_valid:
            geom = make_valid(geom)
        if geom is not None and not geom.is_empty:
            fixed.append(geom)
    if not fixed:
        return []

    local_crs = CRS.from_proj4(
        f"+proj=aeqd +lat_0={center_lat:.8f} +lon_0={center_lon:.8f} +datum=WGS84 +units=m +no_defs"
    )
    local = gpd.GeoSeries(fixed, crs=tile_crs).to_crs(local_crs)
    buffered = local.buffer(float(buffer_km) * 1000.0)
    buffered = buffered[~buffered.is_empty]
    if buffered.empty:
        return []
    merged = unary_union(list(buffered.geometry))
    out = gpd.GeoSeries([merged], crs=local_crs).to_crs(tile_crs)
    return [geom for geom in out.geometry if geom is not None and not geom.is_empty]


def load_river_buffer_geometries(
    river_vector: str | None,
    river_layer: str | None,
    tile_crs: Any,
    tile_bounds: rasterio.coords.BoundingBox,
    buffer_km: float | None,
    min_discharge_cms: float | None,
    min_upstream_area_km2: float | None,
    min_strahler_order: float | None,
) -> tuple[list[Any], int, int]:
    if not river_vector or not buffer_km or buffer_km <= 0:
        return [], 0, 0
    path = Path(river_vector)
    if not path.exists():
        raise FileNotFoundError(f"River vector not found: {path}")

    center_lon = (tile_bounds.left + tile_bounds.right) / 2
    center_lat = (tile_bounds.bottom + tile_bounds.top) / 2
    cos_lat = max(abs(math.cos(math.radians(center_lat))), 0.15)
    buffer_deg_lon = buffer_km / (111.32 * cos_lat)
    buffer_deg_lat = buffer_km / 111.32
    bbox = (
        tile_bounds.left - buffer_deg_lon,
        tile_bounds.bottom - buffer_deg_lat,
        tile_bounds.right + buffer_deg_lon,
        tile_bounds.top + buffer_deg_lat,
    )

    read_kwargs: dict[str, Any] = {"bbox": bbox}
    if river_layer:
        read_kwargs["layer"] = river_layer
    gdf = gpd.read_file(path, **read_kwargs)
    raw_count = int(len(gdf))
    if gdf.empty:
        return [], raw_count, 0

    if min_discharge_cms is not None and "DIS_AV_CMS" in gdf.columns:
        gdf = gdf[pd.to_numeric(gdf["DIS_AV_CMS"], errors="coerce") >= float(min_discharge_cms)]
    elif min_discharge_cms is not None:
        logging.warning("River field DIS_AV_CMS not found; discharge filter ignored.")

    if min_upstream_area_km2 is not None and "UPLAND_SKM" in gdf.columns:
        gdf = gdf[pd.to_numeric(gdf["UPLAND_SKM"], errors="coerce") >= float(min_upstream_area_km2)]
    elif min_upstream_area_km2 is not None:
        logging.warning("River field UPLAND_SKM not found; upstream-area filter ignored.")

    if min_strahler_order is not None and "ORD_STRA" in gdf.columns:
        gdf = gdf[pd.to_numeric(gdf["ORD_STRA"], errors="coerce") >= float(min_strahler_order)]
    elif min_strahler_order is not None:
        logging.warning("River field ORD_STRA not found; Strahler-order filter ignored.")

    filtered_count = int(len(gdf))
    if gdf.empty:
        return [], raw_count, filtered_count
    if gdf.crs is None:
        raise ValueError(f"River vector CRS is missing: {path}")
    if gdf.crs != tile_crs:
        gdf = gdf.to_crs(tile_crs)

    fixed = []
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        if not geom.is_valid:
            geom = make_valid(geom)
        if geom is not None and not geom.is_empty:
            fixed.append(geom)
    if not fixed:
        return [], raw_count, filtered_count

    local_crs = CRS.from_proj4(
        f"+proj=aeqd +lat_0={center_lat:.8f} +lon_0={center_lon:.8f} +datum=WGS84 +units=m +no_defs"
    )
    local = gpd.GeoSeries(fixed, crs=tile_crs).to_crs(local_crs)
    buffered = local.buffer(float(buffer_km) * 1000.0)
    buffered = buffered[~buffered.is_empty]
    if buffered.empty:
        return [], raw_count, filtered_count
    merged = unary_union(list(buffered.geometry))
    out = gpd.GeoSeries([merged], crs=local_crs).to_crs(tile_crs)
    return [geom for geom in out.geometry if geom is not None and not geom.is_empty], raw_count, filtered_count


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


def iter_block_windows(width: int, height: int, block_size: int) -> list[Window]:
    windows: list[Window] = []
    for row in range(0, height, block_size):
        for col in range(0, width, block_size):
            windows.append(Window(col, row, min(block_size, width - col), min(block_size, height - row)))
    return windows


def padded_window(window: Window, width: int, height: int, pad: int = 1) -> tuple[Window, tuple[slice, slice]]:
    col0 = max(0, int(window.col_off) - pad)
    row0 = max(0, int(window.row_off) - pad)
    col1 = min(width, int(window.col_off + window.width) + pad)
    row1 = min(height, int(window.row_off + window.height) + pad)
    padded = Window(col0, row0, col1 - col0, row1 - row0)
    row_slice = slice(int(window.row_off) - row0, int(window.row_off) - row0 + int(window.height))
    col_slice = slice(int(window.col_off) - col0, int(window.col_off) - col0 + int(window.width))
    return padded, (row_slice, col_slice)


def slope_degrees_from_elevation(elev: np.ndarray, transform: rasterio.Affine, window: Window) -> np.ndarray:
    rows = np.arange(int(window.row_off), int(window.row_off + window.height))
    lat = transform.f + (rows + 0.5) * transform.e
    dy_m = abs(transform.e) * 111_320.0
    dx_m = abs(transform.a) * 111_320.0 * np.maximum(np.cos(np.deg2rad(lat)), 0.05)
    grad_y = np.gradient(elev, dy_m, axis=0)
    grad_x = np.empty_like(elev, dtype="float32")
    for row_idx, row_dx in enumerate(dx_m):
        grad_x[row_idx, :] = np.gradient(elev[row_idx, :], float(row_dx))
    slope = np.rad2deg(np.arctan(np.sqrt(grad_x * grad_x + grad_y * grad_y)))
    return slope.astype("float32")


def load_stage12_jobs(args: argparse.Namespace) -> pd.DataFrame:
    if not STAGE12_STATUS_CSV.exists():
        raise FileNotFoundError(f"Stage12 status CSV not found: {STAGE12_STATUS_CSV}")
    status_df = pd.read_csv(STAGE12_STATUS_CSV)
    if status_df.empty:
        raise RuntimeError(f"Stage12 status CSV is empty: {STAGE12_STATUS_CSV}")

    rows: list[dict[str, Any]] = []
    for _, row in status_df.iterrows():
        if str(row.get("status", "")).lower() != "success":
            continue
        if args.model_group and str(row.get("model_group")) != args.model_group:
            continue
        if args.gcm and str(row.get("gcm")) != args.gcm:
            continue
        if args.ssp and str(row.get("ssp")) != args.ssp:
            continue
        if args.period and str(row.get("period")) != args.period:
            continue
        if args.tile_id and str(row.get("tile_id")) != args.tile_id:
            continue

        payload = extract_stage10_json(str(row.get("message", "")))
        if payload is None:
            logging.warning("Cannot parse Stage10 payload for job_id=%s", row.get("job_id"))
            continue
        suitable_tif = Path(str(payload.get("suitable_tif", "")))
        probability_tif = Path(str(payload.get("probability_tif", "")))
        if not suitable_tif.exists():
            logging.warning("Suitable tif missing for tile=%s: %s", row.get("tile_id"), suitable_tif)
            continue
        rows.append(
            {
                "job_id": row.get("job_id"),
                "tile_id": row.get("tile_id"),
                "model_group": row.get("model_group"),
                "gcm": row.get("gcm"),
                "ssp": row.get("ssp"),
                "period": row.get("period"),
                "min_lon": float(row.get("min_lon")),
                "min_lat": float(row.get("min_lat")),
                "max_lon": float(row.get("max_lon")),
                "max_lat": float(row.get("max_lat")),
                "threshold": payload.get("threshold"),
                "valid_area_km2_stage12": payload.get("valid_area_km2"),
                "suitable_area_km2_stage12": payload.get("suitable_area_km2"),
                "suitable_rate_stage12": payload.get("suitable_rate"),
                "mean_probability_stage12": payload.get("mean_probability"),
                "probability_tif": str(probability_tif) if probability_tif.exists() else "",
                "suitable_tif": str(suitable_tif),
            }
        )
    jobs = pd.DataFrame(rows)
    if jobs.empty:
        raise RuntimeError("No Stage12 jobs matched the requested filters.")
    if args.sort_by == "suitable_area_desc":
        jobs["_sort_suitable_area"] = pd.to_numeric(jobs["suitable_area_km2_stage12"], errors="coerce").fillna(-1)
        jobs = jobs.sort_values(["_sort_suitable_area", "gcm", "ssp", "period", "tile_id"], ascending=[False, True, True, True, True])
        jobs = jobs.drop(columns=["_sort_suitable_area"])
    else:
        jobs = jobs.sort_values(["gcm", "ssp", "period", "tile_id"])
    jobs = jobs.reset_index(drop=True)
    if args.limit and args.limit > 0:
        jobs = jobs.head(args.limit).copy()
    return jobs


def load_existing_status() -> pd.DataFrame:
    if STATUS_CSV.exists():
        try:
            return pd.read_csv(STATUS_CSV)
        except Exception:
            logging.warning("Existing status CSV could not be read and will be replaced: %s", STATUS_CSV)
    return pd.DataFrame()


def update_status_row(row: dict[str, Any]) -> None:
    existing = load_existing_status()
    row_df = pd.DataFrame([row])
    if existing.empty or "output_key" not in existing.columns:
        out = row_df
    else:
        existing = existing[existing["output_key"].astype(str) != str(row["output_key"])]
        out = pd.concat([existing, row_df], ignore_index=True)
    atomic_write_csv(STATUS_CSV, out)
    atomic_write_json(STATE_JSON, row)


def area_sum(mask: np.ndarray, row_areas_for_window: np.ndarray, width: int) -> float:
    per_pixel = row_areas_for_window / float(width)
    return float((mask * per_pixel[:, None]).sum())


def constraint_suffix_from_args(args: argparse.Namespace) -> str:
    constraint_suffix = "terrain"
    if args.oasis_buffer_km and args.oasis_buffer_km > 0:
        constraint_suffix += f"_oasis{number_token(args.oasis_buffer_km)}km"
    if args.river_buffer_km and args.river_buffer_km > 0:
        constraint_suffix += f"_river{number_token(args.river_buffer_km)}km"
        if args.min_river_discharge_cms is not None:
            constraint_suffix += f"_q{number_token(args.min_river_discharge_cms)}cms"
        if args.min_river_upstream_km2 is not None:
            constraint_suffix += f"_up{number_token(args.min_river_upstream_km2)}km2"
        if args.min_river_strahler_order is not None:
            constraint_suffix += f"_str{number_token(args.min_river_strahler_order)}"
    return constraint_suffix


def process_one(job: pd.Series, args: argparse.Namespace) -> dict[str, Any]:
    base_output_key = (
        f"{safe_name(job['model_group'])}__{safe_name(job['gcm'])}__{safe_name(job['ssp'])}"
        f"__{safe_name(job['period'])}__{safe_name(job['tile_id'])}"
    )
    constraint_suffix = constraint_suffix_from_args(args)
    output_key = f"{base_output_key}__{constraint_suffix}"
    out_tif = RASTER_DIR / f"{base_output_key}_{constraint_suffix}_constrained_suitable.tif"
    if out_tif.exists() and not args.overwrite:
        existing_status = load_existing_status()
        prior_rows = pd.DataFrame()
        if not existing_status.empty and "output_key" in existing_status.columns:
            prior_rows = existing_status[existing_status["output_key"].astype(str) == output_key].copy()
        if not prior_rows.empty:
            result = prior_rows.iloc[-1].to_dict()
            result.update(
                {
                    "updated_at": now_iso(),
                    "output_key": output_key,
                    "stage12_output_key": base_output_key,
                    "constraint_suffix": constraint_suffix,
                    "status": "skipped",
                    "message": "existing constrained raster skipped; reused previous metrics",
                    "output_tif": str(out_tif),
                }
            )
            update_status_row(result)
            return result
        result = {
            "updated_at": now_iso(),
            "output_key": output_key,
            "stage12_output_key": base_output_key,
            "constraint_suffix": constraint_suffix,
            "status": "skipped",
            "message": "existing constrained raster skipped",
            "output_tif": str(out_tif),
            **job.to_dict(),
            "max_elevation_m": args.max_elevation_m,
            "max_slope_deg": args.max_slope_deg,
            "oasis_vector": args.oasis_vector,
            "oasis_buffer_km": args.oasis_buffer_km,
            "river_vector": args.river_vector,
            "river_layer": args.river_layer,
            "river_buffer_km": args.river_buffer_km,
            "min_river_discharge_cms": args.min_river_discharge_cms,
            "min_river_upstream_km2": args.min_river_upstream_km2,
            "min_river_strahler_order": args.min_river_strahler_order,
        }
        update_status_row(result)
        return result

    update_status_row(
        {
            "updated_at": now_iso(),
            "output_key": output_key,
            "stage12_output_key": base_output_key,
            "constraint_suffix": constraint_suffix,
            "status": "running",
            "message": "constrained suitability processing started",
            **job.to_dict(),
            "output_tif": str(out_tif),
            "max_elevation_m": args.max_elevation_m,
            "max_slope_deg": args.max_slope_deg,
            "oasis_vector": args.oasis_vector,
            "oasis_buffer_km": args.oasis_buffer_km,
            "river_vector": args.river_vector,
            "river_layer": args.river_layer,
            "river_buffer_km": args.river_buffer_km,
            "min_river_discharge_cms": args.min_river_discharge_cms,
            "min_river_upstream_km2": args.min_river_upstream_km2,
            "min_river_strahler_order": args.min_river_strahler_order,
        }
    )

    suitable_tif = Path(str(job["suitable_tif"]))
    tmp_tif = out_tif.with_suffix(out_tif.suffix + ".tmp")
    if tmp_tif.exists():
        tmp_tif.unlink()

    with rasterio.open(suitable_tif) as src, rasterio.open(ELEV_TIF) as elev_src:
        elev_tile_window = matching_elevation_window(src, elev_src)
        oasis_geometries = load_oasis_buffer_geometries(args.oasis_vector, src.crs, src.bounds, args.oasis_buffer_km)
        river_geometries, river_raw_count, river_filtered_count = load_river_buffer_geometries(
            args.river_vector,
            args.river_layer,
            src.crs,
            src.bounds,
            args.river_buffer_km,
            args.min_river_discharge_cms,
            args.min_river_upstream_km2,
            args.min_river_strahler_order,
        )
        profile = src.profile.copy()
        profile.update(
            driver="GTiff",
            dtype="uint8",
            count=1,
            nodata=NODATA_UINT8,
            tiled=True,
            blockxsize=256,
            blockysize=256,
            compress="DEFLATE",
            BIGTIFF="YES",
        )

        windows = iter_block_windows(src.width, src.height, args.block_size)
        row_areas = row_cell_areas_km2(src.transform, Window(0, 0, src.width, src.height))

        valid_pixels = 0
        original_suitable_pixels = 0
        constrained_suitable_pixels = 0
        excluded_by_elevation_pixels = 0
        excluded_by_slope_pixels = 0
        excluded_by_oasis_pixels = 0
        excluded_by_river_pixels = 0
        excluded_by_any_pixels = 0
        valid_area_km2 = 0.0
        original_suitable_area_km2 = 0.0
        constrained_suitable_area_km2 = 0.0
        excluded_by_elevation_area_km2 = 0.0
        excluded_by_slope_area_km2 = 0.0
        excluded_by_oasis_area_km2 = 0.0
        excluded_by_river_area_km2 = 0.0
        excluded_by_any_area_km2 = 0.0

        with rasterio.open(tmp_tif, "w", **profile) as dst:
            for idx, win in enumerate(windows, 1):
                suitable = src.read(1, window=win)
                valid = suitable != NODATA_UINT8
                base_suitable = valid & (suitable == 1)

                elev_win = Window(
                    int(elev_tile_window.col_off + win.col_off),
                    int(elev_tile_window.row_off + win.row_off),
                    int(win.width),
                    int(win.height),
                )
                pad_win, crop = padded_window(elev_win, elev_src.width, elev_src.height, pad=1)
                elev_pad = elev_src.read(1, window=pad_win, masked=True).astype("float32").filled(np.nan)
                if elev_src.nodata is not None:
                    elev_pad[elev_pad == elev_src.nodata] = np.nan
                slope_pad = slope_degrees_from_elevation(elev_pad, elev_src.transform, pad_win)
                elev = elev_pad[crop]
                slope = slope_pad[crop]

                elev_ok = np.isfinite(elev)
                if args.min_elevation_m is not None:
                    elev_ok &= elev >= args.min_elevation_m
                if args.max_elevation_m is not None:
                    elev_ok &= elev <= args.max_elevation_m
                slope_ok = np.isfinite(slope)
                if args.max_slope_deg is not None:
                    slope_ok &= slope <= args.max_slope_deg
                if oasis_geometries:
                    oasis_ok = geometry_mask(
                        oasis_geometries,
                        out_shape=suitable.shape,
                        transform=src.window_transform(win),
                        invert=True,
                        all_touched=args.oasis_all_touched,
                    )
                else:
                    oasis_ok = np.ones(suitable.shape, dtype=bool)
                if river_geometries:
                    river_ok = geometry_mask(
                        river_geometries,
                        out_shape=suitable.shape,
                        transform=src.window_transform(win),
                        invert=True,
                        all_touched=args.river_all_touched,
                    )
                else:
                    river_ok = np.ones(suitable.shape, dtype=bool)

                constrained = base_suitable & elev_ok & slope_ok & oasis_ok & river_ok
                excluded_elev = base_suitable & (~elev_ok)
                excluded_slope = base_suitable & (~slope_ok)
                excluded_oasis = base_suitable & (~oasis_ok)
                excluded_river = base_suitable & (~river_ok)
                excluded_any = base_suitable & (~(elev_ok & slope_ok & oasis_ok & river_ok))

                out = np.full(suitable.shape, NODATA_UINT8, dtype="uint8")
                out[valid] = 0
                out[constrained] = 1
                dst.write(out, 1, window=win)

                rel_rows = np.arange(int(win.row_off), int(win.row_off + win.height))
                block_row_areas = row_areas[rel_rows]
                valid_pixels += int(valid.sum())
                original_suitable_pixels += int(base_suitable.sum())
                constrained_suitable_pixels += int(constrained.sum())
                excluded_by_elevation_pixels += int(excluded_elev.sum())
                excluded_by_slope_pixels += int(excluded_slope.sum())
                excluded_by_oasis_pixels += int(excluded_oasis.sum())
                excluded_by_river_pixels += int(excluded_river.sum())
                excluded_by_any_pixels += int(excluded_any.sum())
                valid_area_km2 += area_sum(valid, block_row_areas, src.width)
                original_suitable_area_km2 += area_sum(base_suitable, block_row_areas, src.width)
                constrained_suitable_area_km2 += area_sum(constrained, block_row_areas, src.width)
                excluded_by_elevation_area_km2 += area_sum(excluded_elev, block_row_areas, src.width)
                excluded_by_slope_area_km2 += area_sum(excluded_slope, block_row_areas, src.width)
                excluded_by_oasis_area_km2 += area_sum(excluded_oasis, block_row_areas, src.width)
                excluded_by_river_area_km2 += area_sum(excluded_river, block_row_areas, src.width)
                excluded_by_any_area_km2 += area_sum(excluded_any, block_row_areas, src.width)

                if idx == 1 or idx % args.status_every == 0 or idx == len(windows):
                    update_status_row(
                        {
                            "updated_at": now_iso(),
                            "output_key": output_key,
                            "stage12_output_key": base_output_key,
                            "constraint_suffix": constraint_suffix,
                            "status": "running",
                            "message": "processing windows",
                            "completed_windows": idx,
                            "total_windows": len(windows),
                            **job.to_dict(),
                            "output_tif": str(out_tif),
                            "constrained_suitable_pixels": constrained_suitable_pixels,
                            "max_elevation_m": args.max_elevation_m,
                            "max_slope_deg": args.max_slope_deg,
                            "oasis_buffer_km": args.oasis_buffer_km,
                            "river_buffer_km": args.river_buffer_km,
                        }
                    )

    tmp_tif.replace(out_tif)
    retention_rate = (
        constrained_suitable_area_km2 / original_suitable_area_km2 if original_suitable_area_km2 > 0 else float("nan")
    )
    result = {
        "updated_at": now_iso(),
        "output_key": output_key,
        "stage12_output_key": base_output_key,
        "constraint_suffix": constraint_suffix,
        "status": "success",
        "message": "constrained suitability processing completed",
        **job.to_dict(),
        "output_tif": str(out_tif),
        "max_elevation_m": args.max_elevation_m,
        "min_elevation_m": args.min_elevation_m,
        "max_slope_deg": args.max_slope_deg,
        "oasis_vector": args.oasis_vector,
        "oasis_buffer_km": args.oasis_buffer_km,
        "oasis_feature_count_in_tile_buffer": len(oasis_geometries),
        "river_vector": args.river_vector,
        "river_layer": args.river_layer,
        "river_buffer_km": args.river_buffer_km,
        "min_river_discharge_cms": args.min_river_discharge_cms,
        "min_river_upstream_km2": args.min_river_upstream_km2,
        "min_river_strahler_order": args.min_river_strahler_order,
        "river_feature_count_raw_in_tile_buffer": river_raw_count,
        "river_feature_count_after_filters": river_filtered_count,
        "river_buffer_geometry_count": len(river_geometries),
        "valid_pixels_recomputed": valid_pixels,
        "original_suitable_pixels_recomputed": original_suitable_pixels,
        "constrained_suitable_pixels": constrained_suitable_pixels,
        "excluded_by_elevation_pixels": excluded_by_elevation_pixels,
        "excluded_by_slope_pixels": excluded_by_slope_pixels,
        "excluded_by_oasis_pixels": excluded_by_oasis_pixels,
        "excluded_by_river_pixels": excluded_by_river_pixels,
        "excluded_by_any_pixels": excluded_by_any_pixels,
        "valid_area_km2_recomputed": valid_area_km2,
        "original_suitable_area_km2_recomputed": original_suitable_area_km2,
        "constrained_suitable_area_km2": constrained_suitable_area_km2,
        "excluded_by_elevation_area_km2": excluded_by_elevation_area_km2,
        "excluded_by_slope_area_km2": excluded_by_slope_area_km2,
        "excluded_by_oasis_area_km2": excluded_by_oasis_area_km2,
        "excluded_by_river_area_km2": excluded_by_river_area_km2,
        "excluded_by_any_area_km2": excluded_by_any_area_km2,
        "constraint_retention_rate": retention_rate,
        "constrained_suitable_rate": constrained_suitable_area_km2 / valid_area_km2 if valid_area_km2 > 0 else float("nan"),
    }
    update_status_row(result)
    return result


def write_report(summary_df: pd.DataFrame, args: argparse.Namespace) -> None:
    success = summary_df[summary_df["status"].isin(["success", "skipped"])].copy()
    if "constrained_suitable_area_km2" in success.columns:
        has_area = pd.to_numeric(success["constrained_suitable_area_km2"], errors="coerce").notna()
        success_area = success[has_area].copy()
    else:
        success_area = success.iloc[0:0].copy()
    total_original = (
        pd.to_numeric(success_area["original_suitable_area_km2_recomputed"], errors="coerce").sum()
        if "original_suitable_area_km2_recomputed" in success_area.columns
        else 0.0
    )
    total_constrained = (
        pd.to_numeric(success_area["constrained_suitable_area_km2"], errors="coerce").sum()
        if "constrained_suitable_area_km2" in success_area.columns
        else 0.0
    )
    total_excluded = (
        pd.to_numeric(success_area["excluded_by_any_area_km2"], errors="coerce").sum()
        if "excluded_by_any_area_km2" in success_area.columns
        else 0.0
    )
    retention = total_constrained / total_original if total_original > 0 else math.nan
    lines = [
        "# Stage17 多约束适宜区后处理报告",
        "",
        f"- 生成时间: {now_iso()}",
        f"- 输入状态表: `{STAGE12_STATUS_CSV}`",
        f"- 输出状态表: `{STATUS_CSV}`",
        f"- 汇总表: `{SUMMARY_CSV}`",
        f"- 高程栅格: `{ELEV_TIF}`",
        f"- 筛选情景: `{args.model_group} / {args.gcm or 'ALL'} / {args.ssp or 'ALL'} / {args.period or 'ALL'}`",
        f"- 约束: min_elevation={args.min_elevation_m}, max_elevation={args.max_elevation_m}, max_slope={args.max_slope_deg} deg",
        f"- 现有绿洲邻近约束: oasis_vector=`{args.oasis_vector}`, buffer_km={args.oasis_buffer_km}",
        f"- 水系邻近约束: river_vector=`{args.river_vector}`, layer={args.river_layer}, buffer_km={args.river_buffer_km}, min_discharge_cms={args.min_river_discharge_cms}, min_upstream_km2={args.min_river_upstream_km2}, min_strahler={args.min_river_strahler_order}",
        "",
        "## 当前批次结果",
        "",
        f"- 成功或跳过 tile 数: {len(success)}",
        f"- 具备面积统计 tile 数: {len(success_area)}",
        f"- 原始阈值适宜面积重算: {total_original:,.2f} km²",
        f"- 约束后适宜面积: {total_constrained:,.2f} km²",
        f"- 被约束剔除面积: {total_excluded:,.2f} km²",
        f"- 约束后保留比例: {retention * 100:.2f}%" if math.isfinite(retention) else "- 约束后保留比例: NA",
        "",
        "## 解释边界",
        "",
        "- 该结果是在原 Stage12 阈值适宜区基础上叠加地形、水系和现有绿洲邻近约束，属于“潜在适宜区收缩版”，仍不等同于真实未来绿洲面积。",
        "- 当前版本可加入高程、坡度、现有绿洲邻近缓冲和 HydroRIVERS 水系邻近缓冲约束；灌溉、土地覆盖仍需要作为下一版约束继续叠加。",
        "- 坡度由 WorldClim 30 arc-second 高程近似计算，适合做全球尺度筛查，不适合替代高分辨率 DEM 地貌判读。",
    ]
    atomic_write_text(REPORT_MD, "\n".join(lines) + "\n")


def run(args: argparse.Namespace) -> dict[str, Any]:
    ensure_dirs()
    if not ELEV_TIF.exists():
        raise FileNotFoundError(f"Elevation raster not found: {ELEV_TIF}")
    jobs = load_stage12_jobs(args)
    logging.info("Matched %s Stage12 tile jobs.", len(jobs))

    results: list[dict[str, Any]] = []
    for idx, job in jobs.iterrows():
        tile_id = job["tile_id"]
        try:
            logging.info("Processing %s/%s tile=%s scenario=%s/%s/%s", idx + 1, len(jobs), tile_id, job["gcm"], job["ssp"], job["period"])
            results.append(process_one(job, args))
        except Exception as exc:
            base_output_key = f"{safe_name(job['model_group'])}__{safe_name(job['gcm'])}__{safe_name(job['ssp'])}__{safe_name(job['period'])}__{safe_name(tile_id)}"
            constraint_suffix = constraint_suffix_from_args(args)
            failure = {
                "updated_at": now_iso(),
                "output_key": f"{base_output_key}__{constraint_suffix}",
                "stage12_output_key": base_output_key,
                "constraint_suffix": constraint_suffix,
                "status": "failed",
                "message": str(exc),
                **job.to_dict(),
                "traceback": traceback.format_exc(),
                "max_elevation_m": args.max_elevation_m,
                "min_elevation_m": args.min_elevation_m,
                "max_slope_deg": args.max_slope_deg,
                "oasis_vector": args.oasis_vector,
                "oasis_buffer_km": args.oasis_buffer_km,
                "river_vector": args.river_vector,
                "river_layer": args.river_layer,
                "river_buffer_km": args.river_buffer_km,
                "min_river_discharge_cms": args.min_river_discharge_cms,
                "min_river_upstream_km2": args.min_river_upstream_km2,
                "min_river_strahler_order": args.min_river_strahler_order,
            }
            logging.error("Tile failed: %s\n%s", tile_id, failure["traceback"])
            update_status_row(failure)
            results.append(failure)
            if args.stop_on_error:
                raise

    summary_df = pd.DataFrame(results)
    atomic_write_csv(SUMMARY_CSV, summary_df)
    summary = {
        "status": "success" if not (summary_df["status"] == "failed").any() else "partial_success",
        "generated_at": now_iso(),
        "matched_jobs": int(len(jobs)),
        "success_jobs": int((summary_df["status"] == "success").sum()),
        "skipped_jobs": int((summary_df["status"] == "skipped").sum()),
        "failed_jobs": int((summary_df["status"] == "failed").sum()),
        "summary_csv": str(SUMMARY_CSV),
        "status_csv": str(STATUS_CSV),
        "report_md": str(REPORT_MD),
        "max_elevation_m": args.max_elevation_m,
        "min_elevation_m": args.min_elevation_m,
        "max_slope_deg": args.max_slope_deg,
        "oasis_vector": args.oasis_vector,
        "oasis_buffer_km": args.oasis_buffer_km,
        "river_vector": args.river_vector,
        "river_layer": args.river_layer,
        "river_buffer_km": args.river_buffer_km,
        "min_river_discharge_cms": args.min_river_discharge_cms,
        "min_river_upstream_km2": args.min_river_upstream_km2,
        "min_river_strahler_order": args.min_river_strahler_order,
    }
    atomic_write_json(SUMMARY_JSON, summary)
    write_report(summary_df, args)
    atomic_write_json(STATE_JSON, summary)
    logging.info("Stage17 completed: %s", json.dumps(summary, ensure_ascii=False))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="在 Stage12 阈值适宜栅格基础上叠加地形约束，生成更保守的未来绿洲潜在适宜区。"
    )
    parser.add_argument("--model-group", default="hist_gradient_boosting_balanced")
    parser.add_argument("--stage12-status-csv", default=str(STAGE12_STATUS_CSV), help="Stage12 批处理状态表。")
    parser.add_argument("--output-dir", default=str(OUT_DIR), help="Stage17 输出目录。")
    parser.add_argument("--run-label", default=None, help="Stage17 日志、状态、摘要和报告文件标签。")
    parser.add_argument("--gcm", default="ACCESS-CM2")
    parser.add_argument("--ssp", default="ssp585")
    parser.add_argument("--period", default="2081-2100")
    parser.add_argument("--tile-id", default=None, help="只处理指定 tile，例如 dryland_40_20_50_30。")
    parser.add_argument("--limit", type=int, default=None, help="只处理前 N 个匹配 tile，用于试跑。")
    parser.add_argument("--sort-by", choices=["tile_id", "suitable_area_desc"], default="tile_id", help="批量试跑时的 tile 排序方式。")
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--status-every", type=int, default=5)
    parser.add_argument("--min-elevation-m", type=float, default=-500.0)
    parser.add_argument("--max-elevation-m", type=float, default=2500.0)
    parser.add_argument("--max-slope-deg", type=float, default=5.0)
    parser.add_argument("--oasis-vector", default=None, help="现有绿洲矢量，用于邻近缓冲约束。")
    parser.add_argument("--use-default-oasis-vector", action="store_true", help="使用本机默认世界绿洲-崔.shp。")
    parser.add_argument("--oasis-buffer-km", type=float, default=None, help="只保留距现有绿洲 N km 内的阈值适宜区。")
    parser.add_argument("--oasis-all-touched", action="store_true")
    parser.add_argument("--river-vector", default=None, help="水系矢量或 FileGDB，用于水源邻近缓冲约束。")
    parser.add_argument("--use-default-river-vector", action="store_true", help="使用本机默认 HydroRIVERS_v10.gdb。")
    parser.add_argument("--river-layer", default=DEFAULT_RIVER_LAYER, help="FileGDB 中的水系图层名。")
    parser.add_argument("--river-buffer-km", type=float, default=None, help="只保留距筛选后水系 N km 内的阈值适宜区。")
    parser.add_argument("--min-river-discharge-cms", type=float, default=None, help="按 HydroRIVERS DIS_AV_CMS 过滤过小河流。")
    parser.add_argument("--min-river-upstream-km2", type=float, default=None, help="按 HydroRIVERS UPLAND_SKM 过滤过小河流。")
    parser.add_argument("--min-river-strahler-order", type=float, default=None, help="按 HydroRIVERS ORD_STRA 过滤过小河流。")
    parser.add_argument("--river-all-touched", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    args = parser.parse_args()
    if args.use_default_oasis_vector and not args.oasis_vector:
        args.oasis_vector = str(DEFAULT_OASIS_VECTOR)
    if args.use_default_river_vector and not args.river_vector:
        args.river_vector = str(DEFAULT_RIVER_VECTOR)
    if args.river_buffer_km is not None and args.river_buffer_km > 0 and not args.river_vector:
        parser.error("--river-buffer-km requires --river-vector or --use-default-river-vector.")
    return args


def main() -> int:
    args = parse_args()
    configure_paths(args)
    setup_logging()
    try:
        run(args)
        return 0
    except Exception as exc:
        logging.error("Stage17 failed: %s", exc)
        logging.error(traceback.format_exc())
        atomic_write_json(
            STATE_JSON,
            {
                "status": "failed",
                "updated_at": now_iso(),
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
