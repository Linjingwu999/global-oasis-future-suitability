# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import math
import re
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
from pyproj import Geod
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window


WORKSPACE = Path(r"C:\Users\linjingwu\Desktop\python")
SYNC_ROOT = Path(r"C:\Users\linjingwu\Desktop\跨电脑同步_工作站传输") / "stage17_future_suitability_workstation_20260530"
PROJECT_ROOT = SYNC_ROOT / "绿洲未来适宜区预测"

DEFAULT_STAGE17_SUMMARY = SYNC_ROOT / "家里电脑查看_STAGE17结果" / "stage17_constrained_suitability_summary.csv"
DEFAULT_STAGE17_RASTER_DIR = PROJECT_ROOT / "outputs" / "stage17_constrained_suitability" / "rasters"
DEFAULT_LANDCOVER_DIR = PROJECT_ROOT / "data" / "landcover" / "stage19_esa_core_30s"
DEFAULT_CONSTRAINT_SUFFIX = "terrain_oasis100km_river100km_q1cms_up1000km2"

OUT_DIR = PROJECT_ROOT / "outputs" / "stage20_landcover_spatial_constraint"
RASTER_DIR = OUT_DIR / "rasters"
TABLE_DIR = OUT_DIR / "tables"
LOG_DIR = PROJECT_ROOT / "logs"
STATUS_CSV = LOG_DIR / "stage20_landcover_spatial_constraint_status.csv"
STATE_JSON = LOG_DIR / "stage20_landcover_spatial_constraint_state.json"
SUMMARY_CSV = TABLE_DIR / "stage20_landcover_spatial_constraint_summary.csv"
SUMMARY_JSON = OUT_DIR / "stage20_landcover_spatial_constraint_summary.json"
REPORT_MD = OUT_DIR / "Stage20_土地覆盖空间约束报告.md"

NODATA_UINT8 = 255
GEOD = Geod(ellps="WGS84")


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def safe_name(text: Any) -> str:
    out = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text))
    return out.strip("._-") or "unnamed"


def configure_paths(args: argparse.Namespace) -> None:
    global OUT_DIR, RASTER_DIR, TABLE_DIR, LOG_DIR, STATUS_CSV, STATE_JSON, SUMMARY_CSV, SUMMARY_JSON, REPORT_MD
    OUT_DIR = Path(args.output_dir)
    RASTER_DIR = OUT_DIR / "rasters"
    TABLE_DIR = OUT_DIR / "tables"
    LOG_DIR = Path(args.log_dir)
    prefix = "stage20_landcover_spatial_constraint"
    if args.run_label:
        prefix = f"{prefix}_{safe_name(args.run_label)}"
    STATUS_CSV = LOG_DIR / f"{prefix}_status.csv"
    STATE_JSON = LOG_DIR / f"{prefix}_state.json"
    SUMMARY_CSV = TABLE_DIR / f"{prefix}_summary.csv"
    SUMMARY_JSON = OUT_DIR / f"{prefix}_summary.json"
    REPORT_MD = OUT_DIR / f"{prefix}_report.md"


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


def ensure_dirs() -> None:
    RASTER_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def load_existing_status() -> pd.DataFrame:
    if not STATUS_CSV.exists():
        return pd.DataFrame()
    return pd.read_csv(STATUS_CSV)


def update_status_row(row: dict[str, Any]) -> None:
    row = dict(row)
    row["updated_at"] = now_iso()
    existing = load_existing_status()
    row_df = pd.DataFrame([row])
    key = str(row["output_key"])
    if existing.empty or "output_key" not in existing.columns:
        out = row_df
    else:
        existing = existing[existing["output_key"].astype(str) != key]
        out = pd.concat([existing, row_df], ignore_index=True)
    atomic_write_csv(STATUS_CSV, out)
    atomic_write_json(STATE_JSON, row)


def rasters_same_grid(a: rasterio.io.DatasetReader, b: rasterio.io.DatasetReader) -> bool:
    return (
        a.crs == b.crs
        and a.width == b.width
        and a.height == b.height
        and all(abs(x - y) <= 1e-9 for x, y in zip(a.transform, b.transform))
    )


def iter_windows(width: int, height: int, block_size: int) -> list[Window]:
    wins: list[Window] = []
    for row in range(0, height, block_size):
        for col in range(0, width, block_size):
            wins.append(Window(col, row, min(block_size, width - col), min(block_size, height - row)))
    return wins


def row_cell_areas_km2(transform, width: int, height: int) -> np.ndarray:
    left = transform.c
    right = transform.c + transform.a * width
    areas = np.zeros(height, dtype="float64")
    for row in range(height):
        y0 = transform.f + transform.e * row
        y1 = transform.f + transform.e * (row + 1)
        lons = [left, right, right, left]
        lats = [y0, y0, y1, y1]
        area_m2, _ = GEOD.polygon_area_perimeter(lons, lats)
        areas[row] = abs(area_m2) / 1_000_000.0
    return areas


def area_sum(mask_or_weight: np.ndarray, row_areas_for_window: np.ndarray, width: int) -> float:
    per_pixel = row_areas_for_window / float(width)
    return float((mask_or_weight * per_pixel[:, None]).sum())


def lc_path_for_tile(landcover_dir: Path, tile_id: str) -> Path:
    return landcover_dir / f"LC_ESA_CORE_30s_{safe_name(tile_id)}.tif"


def resolve_stage17_tif(path_text: str, stage17_raster_dir: Path) -> Path:
    path = Path(str(path_text))
    if path.exists():
        return path
    fallback = stage17_raster_dir / path.name
    return fallback


def load_jobs(args: argparse.Namespace) -> pd.DataFrame:
    summary_path = Path(args.stage17_summary)
    if not summary_path.exists():
        raise FileNotFoundError(f"Stage17 summary not found: {summary_path}")
    df = pd.read_csv(summary_path)
    df = df[
        (df["constraint_suffix"].astype(str) == args.constraint_suffix)
        & (df["status"].astype(str).isin(["success", "skipped"]))
    ].copy()
    if args.tile_id:
        df = df[df["tile_id"].astype(str) == args.tile_id].copy()
    df = df.sort_values("tile_id").reset_index(drop=True)
    if args.limit:
        df = df.head(int(args.limit)).copy()
    if df.empty:
        raise ValueError("No Stage17 river100 rows matched.")
    return df


def check_inputs(args: argparse.Namespace, jobs: pd.DataFrame) -> pd.DataFrame:
    stage17_dir = Path(args.stage17_raster_dir)
    landcover_dir = Path(args.landcover_dir)
    rows: list[dict[str, Any]] = []
    for _, job in jobs.iterrows():
        tile_id = str(job["tile_id"])
        stage17_tif = resolve_stage17_tif(str(job.get("output_tif", "")), stage17_dir)
        lc_tif = lc_path_for_tile(landcover_dir, tile_id)
        rows.append(
            {
                "tile_id": tile_id,
                "stage17_tif": str(stage17_tif),
                "stage17_exists": stage17_tif.exists(),
                "landcover_tif": str(lc_tif),
                "landcover_exists": lc_tif.exists(),
            }
        )
    return pd.DataFrame(rows)


def process_one(job: pd.Series, args: argparse.Namespace) -> dict[str, Any]:
    stage17_dir = Path(args.stage17_raster_dir)
    landcover_dir = Path(args.landcover_dir)
    tile_id = str(job["tile_id"])
    stage17_tif = resolve_stage17_tif(str(job.get("output_tif", "")), stage17_dir)
    lc_tif = lc_path_for_tile(landcover_dir, tile_id)

    base_key = (
        f"{safe_name(job['model_group'])}__{safe_name(job['gcm'])}__{safe_name(job['ssp'])}"
        f"__{safe_name(job['period'])}__{safe_name(tile_id)}"
    )
    suffix = f"{args.constraint_suffix}_esa_core_lc_pct_binary{int(args.min_compatible_pct)}"
    output_key = f"{base_key}__{suffix}"
    binary_tif = RASTER_DIR / f"{base_key}_{suffix}_suitable.tif"
    pct_tif = RASTER_DIR / f"{base_key}_{suffix}_compatible_pct.tif"

    if binary_tif.exists() and pct_tif.exists() and not args.overwrite:
        prior = load_existing_status()
        if not prior.empty and "output_key" in prior.columns:
            hit = prior[prior["output_key"].astype(str) == output_key]
            if not hit.empty:
                row = hit.iloc[-1].to_dict()
                row.update({"status": "skipped", "message": "existing outputs skipped", "updated_at": now_iso()})
                update_status_row(row)
                return row

    if not stage17_tif.exists():
        raise FileNotFoundError(f"Stage17 raster missing: {stage17_tif}")
    if not lc_tif.exists():
        raise FileNotFoundError(f"land-cover raster missing: {lc_tif}")

    update_status_row(
        {
            "output_key": output_key,
            "status": "running",
            "message": "land-cover spatial constraint started",
            "tile_id": tile_id,
            "stage17_tif": str(stage17_tif),
            "landcover_tif": str(lc_tif),
            "binary_tif": str(binary_tif),
            "compatible_pct_tif": str(pct_tif),
            "min_compatible_pct": args.min_compatible_pct,
        }
    )

    tmp_binary = binary_tif.with_suffix(binary_tif.suffix + ".tmp")
    tmp_pct = pct_tif.with_suffix(pct_tif.suffix + ".tmp")
    for tmp in [tmp_binary, tmp_pct]:
        if tmp.exists():
            tmp.unlink()

    with rasterio.open(stage17_tif) as src, rasterio.open(lc_tif) as lc_src:
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
        windows = iter_windows(src.width, src.height, args.block_size)
        row_areas = row_cell_areas_km2(src.transform, src.width, src.height)

        lc_reader_ctx = (
            lc_src
            if rasters_same_grid(src, lc_src)
            else WarpedVRT(
                lc_src,
                crs=src.crs,
                transform=src.transform,
                width=src.width,
                height=src.height,
                resampling=Resampling.bilinear,
                src_nodata=lc_src.nodata,
                nodata=NODATA_UINT8,
            )
        )

        stage17_suitable_pixels = 0
        binary_suitable_pixels = 0
        lc_valid_pixels_on_stage17 = 0
        stage17_suitable_area_km2 = 0.0
        binary_suitable_area_km2 = 0.0
        weighted_compatible_area_km2 = 0.0
        excluded_by_landcover_area_km2 = 0.0

        with lc_reader_ctx as lc_reader, rasterio.open(tmp_binary, "w", **profile) as dst_bin, rasterio.open(tmp_pct, "w", **profile) as dst_pct:
            for idx, win in enumerate(windows, 1):
                arr = src.read(1, window=win)
                lc_pct = lc_reader.read(1, window=win, out_shape=(int(win.height), int(win.width))).astype("float32")
                if lc_reader.nodata is not None:
                    lc_valid = lc_pct != float(lc_reader.nodata)
                else:
                    lc_valid = np.isfinite(lc_pct)
                lc_pct = np.clip(lc_pct, 0.0, 100.0)
                frac = lc_pct / 100.0

                valid = arr != NODATA_UINT8
                stage17_suitable = valid & (arr == 1)
                binary_suitable = stage17_suitable & lc_valid & (lc_pct >= float(args.min_compatible_pct))
                weighted = np.where(stage17_suitable & lc_valid, frac, 0.0)
                excluded = stage17_suitable & (~binary_suitable)

                out_bin = np.full(arr.shape, NODATA_UINT8, dtype="uint8")
                out_bin[valid] = 0
                out_bin[binary_suitable] = 1
                dst_bin.write(out_bin, 1, window=win)

                out_pct = np.full(arr.shape, NODATA_UINT8, dtype="uint8")
                out_pct[valid] = 0
                out_pct[stage17_suitable & lc_valid] = np.rint(lc_pct[stage17_suitable & lc_valid]).astype("uint8")
                dst_pct.write(out_pct, 1, window=win)

                rel_rows = np.arange(int(win.row_off), int(win.row_off + win.height))
                block_row_areas = row_areas[rel_rows]
                stage17_suitable_pixels += int(stage17_suitable.sum())
                binary_suitable_pixels += int(binary_suitable.sum())
                lc_valid_pixels_on_stage17 += int((stage17_suitable & lc_valid).sum())
                stage17_suitable_area_km2 += area_sum(stage17_suitable, block_row_areas, src.width)
                binary_suitable_area_km2 += area_sum(binary_suitable, block_row_areas, src.width)
                weighted_compatible_area_km2 += area_sum(weighted, block_row_areas, src.width)
                excluded_by_landcover_area_km2 += area_sum(excluded, block_row_areas, src.width)

                if idx == 1 or idx % args.status_every == 0 or idx == len(windows):
                    update_status_row(
                        {
                            "output_key": output_key,
                            "status": "running",
                            "message": "processing windows",
                            "tile_id": tile_id,
                            "completed_windows": idx,
                            "total_windows": len(windows),
                            "weighted_compatible_area_km2": weighted_compatible_area_km2,
                            "binary_suitable_area_km2": binary_suitable_area_km2,
                        }
                    )

    tmp_binary.replace(binary_tif)
    tmp_pct.replace(pct_tif)
    result = {
        "output_key": output_key,
        "status": "success",
        "message": "land-cover spatial constraint completed",
        "tile_id": tile_id,
        "model_group": job["model_group"],
        "gcm": job["gcm"],
        "ssp": job["ssp"],
        "period": job["period"],
        "stage17_tif": str(stage17_tif),
        "landcover_tif": str(lc_tif),
        "binary_tif": str(binary_tif),
        "compatible_pct_tif": str(pct_tif),
        "min_compatible_pct": args.min_compatible_pct,
        "stage17_suitable_pixels": stage17_suitable_pixels,
        "binary_suitable_pixels": binary_suitable_pixels,
        "lc_valid_pixels_on_stage17": lc_valid_pixels_on_stage17,
        "stage17_suitable_area_km2": stage17_suitable_area_km2,
        "binary_suitable_area_km2": binary_suitable_area_km2,
        "weighted_compatible_area_km2": weighted_compatible_area_km2,
        "excluded_by_landcover_area_km2": excluded_by_landcover_area_km2,
        "weighted_retention_pct": weighted_compatible_area_km2 / stage17_suitable_area_km2 * 100 if stage17_suitable_area_km2 > 0 else float("nan"),
        "binary_retention_pct": binary_suitable_area_km2 / stage17_suitable_area_km2 * 100 if stage17_suitable_area_km2 > 0 else float("nan"),
    }
    update_status_row(result)
    return result


def write_report(summary: pd.DataFrame, args: argparse.Namespace) -> None:
    success = summary[summary["status"].isin(["success", "skipped"])].copy()
    for col in ["stage17_suitable_area_km2", "binary_suitable_area_km2", "weighted_compatible_area_km2", "excluded_by_landcover_area_km2"]:
        if col in success.columns:
            success[col] = pd.to_numeric(success[col], errors="coerce")
    total_stage17 = float(success["stage17_suitable_area_km2"].sum()) if "stage17_suitable_area_km2" in success.columns else 0.0
    total_binary = float(success["binary_suitable_area_km2"].sum()) if "binary_suitable_area_km2" in success.columns else 0.0
    total_weighted = float(success["weighted_compatible_area_km2"].sum()) if "weighted_compatible_area_km2" in success.columns else 0.0
    total_excluded = float(success["excluded_by_landcover_area_km2"].sum()) if "excluded_by_landcover_area_km2" in success.columns else 0.0
    lines = [
        "# Stage20 土地覆盖空间约束报告",
        "",
        f"- 生成时间: {now_iso()}",
        f"- Stage17 summary: `{args.stage17_summary}`",
        f"- Stage17 raster dir: `{args.stage17_raster_dir}`",
        f"- Land-cover raster dir: `{args.landcover_dir}`",
        f"- Binary threshold: compatible_pct >= {args.min_compatible_pct}",
        "",
        "## 汇总",
        "",
        f"- 成功或跳过 tile 数: {len(success)}",
        f"- Stage17 river100 面积: {total_stage17:,.2f} km² ({total_stage17/10000.0:.2f} 万 km²)",
        f"- 土地覆盖加权兼容面积: {total_weighted:,.2f} km² ({total_weighted/10000.0:.2f} 万 km²)",
        f"- 二值阈值兼容面积: {total_binary:,.2f} km² ({total_binary/10000.0:.2f} 万 km²)",
        f"- 二值阈值剔除面积: {total_excluded:,.2f} km² ({total_excluded/10000.0:.2f} 万 km²)",
        "",
        "## 解释",
        "",
        "- 加权兼容面积使用 30 arc-second 像元内 ESA WorldCover 核心兼容类比例作为面积权重，避免人为设置土地覆盖二值阈值。",
        "- 二值阈值结果主要用于制图；面积讨论建议优先使用加权兼容面积，并用二值结果做空间表达。",
    ]
    atomic_write_text(REPORT_MD, "\n".join(lines) + "\n")


def run(args: argparse.Namespace) -> dict[str, Any]:
    ensure_dirs()
    jobs = load_jobs(args)
    input_check = check_inputs(args, jobs)
    input_check_path = TABLE_DIR / "stage20_input_file_check.csv"
    atomic_write_csv(input_check_path, input_check)
    missing_stage17 = int((~input_check["stage17_exists"]).sum())
    missing_landcover = int((~input_check["landcover_exists"]).sum())

    if args.check_inputs_only:
        summary = {
            "status": "input_check",
            "matched_jobs": int(len(jobs)),
            "missing_stage17": missing_stage17,
            "missing_landcover": missing_landcover,
            "input_check_csv": str(input_check_path),
        }
        atomic_write_json(STATE_JSON, summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return summary

    results: list[dict[str, Any]] = []
    for _, job in jobs.iterrows():
        try:
            results.append(process_one(job, args))
        except Exception as exc:
            failure = {
                "output_key": f"{safe_name(job['model_group'])}__{safe_name(job['gcm'])}__{safe_name(job['ssp'])}__{safe_name(job['period'])}__{safe_name(job['tile_id'])}__{args.constraint_suffix}_esa_core_lc",
                "status": "failed",
                "message": str(exc),
                "traceback": traceback.format_exc(),
                "tile_id": job["tile_id"],
            }
            update_status_row(failure)
            results.append(failure)
            if args.stop_on_error:
                raise

    summary_df = pd.DataFrame(results)
    atomic_write_csv(SUMMARY_CSV, summary_df)
    write_report(summary_df, args)
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
    }
    atomic_write_json(SUMMARY_JSON, summary)
    atomic_write_json(STATE_JSON, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply Stage19 ESA core-compatible land-cover percentage rasters to Stage17 river100 suitability rasters.")
    parser.add_argument("--stage17-summary", default=str(DEFAULT_STAGE17_SUMMARY))
    parser.add_argument("--stage17-raster-dir", default=str(DEFAULT_STAGE17_RASTER_DIR))
    parser.add_argument("--landcover-dir", default=str(DEFAULT_LANDCOVER_DIR))
    parser.add_argument("--output-dir", default=str(OUT_DIR), help="Stage20 输出目录。")
    parser.add_argument("--log-dir", default=str(LOG_DIR), help="Stage20 状态文件目录。")
    parser.add_argument("--run-label", default=None, help="Stage20 状态、摘要和报告文件标签。")
    parser.add_argument("--constraint-suffix", default=DEFAULT_CONSTRAINT_SUFFIX)
    parser.add_argument("--min-compatible-pct", type=float, default=50.0)
    parser.add_argument("--tile-id", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--status-every", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--check-inputs-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_paths(args)
    try:
        run(args)
        return 0
    except Exception as exc:
        failure = {"status": "failed", "updated_at": now_iso(), "message": str(exc), "traceback": traceback.format_exc()}
        atomic_write_json(STATE_JSON, failure)
        print(json.dumps(failure, ensure_ascii=False, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
