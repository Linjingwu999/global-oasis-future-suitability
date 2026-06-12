# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import logging
import math
import subprocess
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import geopandas as gpd
import pyogrio
from shapely.geometry import box
from shapely.ops import unary_union
from shapely.validation import make_valid


WORKSPACE = Path(r"C:\Users\linjingwu\Desktop\python")
PROJECT_ROOT = WORKSPACE / "绿洲未来适宜区预测"
SCRIPT12 = PROJECT_ROOT / "scripts" / "12_未来WorldClim全图适宜性栅格投影_断点试跑.py"
REGION_INDEX = PROJECT_ROOT / "data" / "region_polygon_index.csv"
DEFAULT_DRYLAND_MASK = Path.home() / "Desktop" / "会议相关" / "世界绿洲合并" / "世界绿洲" / "全球干旱区" / "AI0-0.65干旱区.shp"

OUT_DIR = PROJECT_ROOT / "outputs" / "stage12_region_tile_grid_projection"
TABLE_DIR = OUT_DIR / "tables"
LOG_DIR = PROJECT_ROOT / "logs"
LOG_PATH = LOG_DIR / "stage12_region_tile_grid_projection.log"
STATE_JSON = LOG_DIR / "stage12_region_tile_grid_projection_state.json"
STATUS_CSV = LOG_DIR / "stage12_region_tile_grid_projection_status.csv"
TILE_MANIFEST_CSV = TABLE_DIR / "stage12_region_tile_manifest.csv"
RUN_SUMMARY_JSON = OUT_DIR / "stage12_region_tile_grid_projection_summary.json"
REPORT_MD = OUT_DIR / "stage12_region_tile_grid_projection_report.md"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "outputs" / "stage06_current_worldclim_baseline_models" / "models"
DEFAULT_STAGE10_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "stage10_future_grid_projection_pilot"

PERIODS = ["2021-2040", "2041-2060", "2061-2080", "2081-2100"]
SSPS = ["ssp126", "ssp245", "ssp370", "ssp585"]
GCMS = ["ACCESS-CM2", "MPI-ESM1-2-HR", "MRI-ESM2-0"]


@dataclass(frozen=True)
class Tile:
    tile_id: str
    region: str
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float


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


def atomic_write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8-sig")
    tmp.replace(path)


def safe_name(value: str) -> str:
    out = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value.strip())
    return out.strip("._-") or "item"


def configure_paths(args: argparse.Namespace) -> None:
    global OUT_DIR, TABLE_DIR, LOG_PATH, STATE_JSON, STATUS_CSV, TILE_MANIFEST_CSV, RUN_SUMMARY_JSON, REPORT_MD
    OUT_DIR = Path(args.output_dir)
    TABLE_DIR = OUT_DIR / "tables"
    prefix = "stage12_region_tile_grid_projection"
    if args.run_label:
        prefix = f"{prefix}_{safe_name(args.run_label)}"
    LOG_PATH = LOG_DIR / f"{prefix}.log"
    STATE_JSON = LOG_DIR / f"{prefix}_state.json"
    STATUS_CSV = LOG_DIR / f"{prefix}_status.csv"
    TILE_MANIFEST_CSV = TABLE_DIR / f"{prefix}_tile_manifest.csv"
    RUN_SUMMARY_JSON = OUT_DIR / f"{prefix}_summary.json"
    REPORT_MD = OUT_DIR / f"{prefix}_report.md"
    if args.stage10_output_dir is None:
        args.stage10_output_dir = str(DEFAULT_STAGE10_OUTPUT_DIR)
    if args.stage10_run_label is None and args.run_label:
        args.stage10_run_label = args.run_label


def snap_floor(value: float, step: float) -> float:
    return math.floor(value / step) * step


def snap_ceil(value: float, step: float) -> float:
    return math.ceil(value / step) * step


def build_dryland_tiles(tile_deg: float, mask_vector: Path) -> pd.DataFrame:
    if not mask_vector.exists():
        raise FileNotFoundError(f"Dryland mask not found: {mask_vector}")
    gdf = gpd.read_file(mask_vector)
    if gdf.crs is None:
        raise ValueError(f"Dryland mask CRS is missing: {mask_vector}")
    if str(gdf.crs).upper() != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")
    geometries = []
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        if not geom.is_valid:
            geom = make_valid(geom)
        if geom is not None and not geom.is_empty:
            geometries.append(geom)
    if not geometries:
        raise ValueError(f"No valid dryland mask geometries: {mask_vector}")
    mask_union = unary_union(geometries)
    minx, miny, maxx, maxy = mask_union.bounds
    min_lon = max(-180.0, snap_floor(float(minx), tile_deg))
    max_lon = min(180.0, snap_ceil(float(maxx), tile_deg))
    min_lat = max(-90.0, snap_floor(float(miny), tile_deg))
    max_lat = min(90.0, snap_ceil(float(maxy), tile_deg))
    rows: list[dict[str, Any]] = []
    lon = min_lon
    while lon < max_lon:
        lat = min_lat
        while lat < max_lat:
            max_tile_lon = min(lon + tile_deg, 180.0)
            max_tile_lat = min(lat + tile_deg, 90.0)
            tile_geom = box(lon, lat, max_tile_lon, max_tile_lat)
            if tile_geom.intersects(mask_union):
                rows.append(
                    {
                        "tile_id": f"dryland_{lon:g}_{lat:g}_{max_tile_lon:g}_{max_tile_lat:g}",
                        "region": "global_dryland",
                        "min_lon": lon,
                        "min_lat": lat,
                        "max_lon": max_tile_lon,
                        "max_lat": max_tile_lat,
                        "source_file": str(mask_vector),
                        "region_bounds": f"{minx},{miny},{maxx},{maxy}",
                        "status": "candidate",
                    }
                )
            lat += tile_deg
        lon += tile_deg
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No dryland tiles generated.")
    atomic_write_csv(TILE_MANIFEST_CSV, df)
    return df


def build_region_tiles(tile_deg: float, margin_deg: float) -> pd.DataFrame:
    idx = pd.read_csv(REGION_INDEX)
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, float, float, float, float]] = set()
    for _, row in idx.iterrows():
        path = Path(row["Polygon_file"])
        if not path.exists():
            rows.append({"region": row["Region"], "status": "missing_region_file", "source_file": str(path)})
            continue
        info = pyogrio.read_info(path)
        minx, miny, maxx, maxy = info["total_bounds"]
        min_lon = max(-180.0, snap_floor(float(minx) - margin_deg, tile_deg))
        max_lon = min(180.0, snap_ceil(float(maxx) + margin_deg, tile_deg))
        min_lat = max(-90.0, snap_floor(float(miny) - margin_deg, tile_deg))
        max_lat = min(90.0, snap_ceil(float(maxy) + margin_deg, tile_deg))
        lon = min_lon
        while lon < max_lon:
            lat = min_lat
            while lat < max_lat:
                tile = (row["Region"], lon, lat, min(lon + tile_deg, 180.0), min(lat + tile_deg, 90.0))
                if tile not in seen:
                    seen.add(tile)
                    rows.append(
                        {
                            "tile_id": f"{safe_name(row['Region'])}_{lon:g}_{lat:g}_{min(lon + tile_deg, 180.0):g}_{min(lat + tile_deg, 90.0):g}",
                            "region": row["Region"],
                            "min_lon": lon,
                            "min_lat": lat,
                            "max_lon": min(lon + tile_deg, 180.0),
                            "max_lat": min(lat + tile_deg, 90.0),
                            "source_file": str(path),
                            "region_bounds": f"{minx},{miny},{maxx},{maxy}",
                            "status": "candidate",
                        }
                    )
                lat += tile_deg
            lon += tile_deg
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No tiles generated.")
    atomic_write_csv(TILE_MANIFEST_CSV, df)
    return df


def build_tiles(args: argparse.Namespace) -> pd.DataFrame:
    if args.tile_source == "dryland":
        return build_dryland_tiles(args.tile_deg, Path(args.mask_vector))
    return build_region_tiles(args.tile_deg, args.margin_deg)


def scenarios_from_args(args: argparse.Namespace) -> list[dict[str, str]]:
    gcms = args.gcm or ([args.default_gcm] if not args.all_scenarios else GCMS)
    ssps = args.ssp or ([args.default_ssp] if not args.all_scenarios else SSPS)
    periods = args.period or ([args.default_period] if not args.all_scenarios else PERIODS)
    return [{"gcm": g, "ssp": s, "period": p} for g in gcms for s in ssps for p in periods]


def load_existing_status() -> pd.DataFrame:
    if STATUS_CSV.exists():
        return pd.read_csv(STATUS_CSV)
    return pd.DataFrame()


def make_job_id(model_group: str, scenario: dict[str, str], tile: pd.Series) -> str:
    return "__".join([safe_name(model_group), scenario["gcm"], scenario["ssp"], scenario["period"], str(tile["tile_id"])])


def run_one(args: argparse.Namespace, scenario: dict[str, str], tile: pd.Series) -> tuple[str, str]:
    cmd = [
        sys.executable,
        "-X",
        "utf8",
        str(SCRIPT12),
        "--model-group",
        args.model_group,
        "--model-dir",
        args.model_dir,
        "--output-dir",
        args.stage10_output_dir,
        "--gcm",
        scenario["gcm"],
        "--ssp",
        scenario["ssp"],
        "--period",
        scenario["period"],
        "--bbox",
        str(tile["min_lon"]),
        str(tile["min_lat"]),
        str(tile["max_lon"]),
        str(tile["max_lat"]),
        "--bbox-name",
        str(tile["tile_id"]),
        "--block-size",
        str(args.block_size),
    ]
    if args.stage10_run_label:
        cmd.extend(["--run-label", args.stage10_run_label])
    if args.make_figure:
        cmd.append("--make-figure")
    if args.mask_vector:
        cmd.extend(["--mask-vector", args.mask_vector])
    if args.mask_all_touched:
        cmd.append("--mask-all-touched")
    if args.overwrite:
        cmd.append("--overwrite")
    logging.info("Running tile job: %s", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(WORKSPACE), capture_output=True, text=True, encoding="utf-8", errors="replace")
    message = (result.stdout + "\n" + result.stderr).strip()
    return ("success" if result.returncode == 0 else "failed", message[-4000:])


def run(args: argparse.Namespace) -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    tiles = build_tiles(args)
    if args.region:
        tiles = tiles[tiles["region"].isin(args.region)].copy()
    if args.limit_tiles:
        tiles = tiles.head(args.limit_tiles).copy()
    scenarios = scenarios_from_args(args)
    if args.limit_scenarios:
        scenarios = scenarios[: args.limit_scenarios]
    if tiles.empty or not scenarios:
        raise RuntimeError("No tile jobs left after filters.")

    old = load_existing_status()
    done_ids = set(old.loc[old["status"].eq("success"), "job_id"].astype(str)) if not old.empty and "job_id" in old else set()
    status_rows = [] if old.empty else old.to_dict("records")

    total = len(tiles) * len(scenarios)
    success = failed = skipped = 0
    atomic_write_json(
        STATE_JSON,
        {
            "status": "running",
            "started_at": now_iso(),
            "model_group": args.model_group,
            "tile_count": int(len(tiles)),
            "scenario_count": int(len(scenarios)),
            "total_jobs": int(total),
        },
    )

    for scenario in scenarios:
        for _, tile in tiles.iterrows():
            job_id = make_job_id(args.model_group, scenario, tile)
            if job_id in done_ids and not args.overwrite:
                skipped += 1
                continue
            row = {
                "updated_at": now_iso(),
                "job_id": job_id,
                "status": "running",
                "model_group": args.model_group,
                "gcm": scenario["gcm"],
                "ssp": scenario["ssp"],
                "period": scenario["period"],
                "tile_id": tile["tile_id"],
                "region": tile["region"],
                "min_lon": tile["min_lon"],
                "min_lat": tile["min_lat"],
                "max_lon": tile["max_lon"],
                "max_lat": tile["max_lat"],
                "message": "",
            }
            status_rows.append(row)
            atomic_write_csv(STATUS_CSV, pd.DataFrame(status_rows))
            try:
                job_status, message = run_one(args, scenario, tile)
                row["updated_at"] = now_iso()
                row["status"] = job_status
                row["message"] = message
                if job_status == "success":
                    success += 1
                    done_ids.add(job_id)
                else:
                    failed += 1
            except Exception as exc:
                failed += 1
                row["updated_at"] = now_iso()
                row["status"] = "failed"
                row["message"] = repr(exc)
                row["traceback"] = traceback.format_exc()
                logging.exception("Tile job failed: %s", job_id)
            atomic_write_csv(STATUS_CSV, pd.DataFrame(status_rows))

    final_status = "success" if failed == 0 else ("partial_success" if success or skipped else "failed")
    state = {
        "status": final_status,
        "finished_at": now_iso(),
        "model_group": args.model_group,
        "tile_count": int(len(tiles)),
        "scenario_count": int(len(scenarios)),
        "total_jobs": int(total),
        "success_jobs": int(success),
        "skipped_jobs": int(skipped),
        "failed_jobs": int(failed),
        "tile_manifest": str(TILE_MANIFEST_CSV),
        "status_csv": str(STATUS_CSV),
    }
    atomic_write_json(STATE_JSON, state)
    atomic_write_json(RUN_SUMMARY_JSON, state)
    lines = [
        "# Stage12 区域 tile 栅格投影批处理报告",
        "",
        f"- 生成时间: {state['finished_at']}",
        f"- 状态: {state['status']}",
        f"- 模型组: {state['model_group']}",
        f"- tile 数: {state['tile_count']}",
        f"- 情景数: {state['scenario_count']}",
        f"- 总任务: {state['total_jobs']}",
        f"- 成功: {state['success_jobs']}",
        f"- 跳过: {state['skipped_jobs']}",
        f"- 失败: {state['failed_jobs']}",
        f"- tile 清单: `{TILE_MANIFEST_CSV}`",
        f"- 状态表: `{STATUS_CSV}`",
    ]
    atomic_write_text(REPORT_MD, "\n".join(lines))
    return state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按区域 tile 批量调用 Stage10 栅格投影脚本，支持断点续跑。")
    parser.add_argument("--model-group", default="hist_gradient_boosting_balanced")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR), help="模型目录，默认使用 Stage06 20 因子基线模型。")
    parser.add_argument("--output-dir", default=str(OUT_DIR), help="Stage12 批处理状态、报告和 tile 清单输出目录。")
    parser.add_argument("--run-label", default=None, help="Stage12 批处理日志、状态和报告文件标签。")
    parser.add_argument("--stage10-output-dir", default=None, help="Stage10 单 tile 栅格输出目录；selected10 重跑时应设为独立目录。")
    parser.add_argument("--stage10-run-label", default=None, help="传给 Stage10 的日志和状态文件标签；默认沿用 --run-label。")
    parser.add_argument("--default-gcm", default="ACCESS-CM2")
    parser.add_argument("--default-ssp", default="ssp585")
    parser.add_argument("--default-period", default="2081-2100")
    parser.add_argument("--gcm", action="append")
    parser.add_argument("--ssp", action="append")
    parser.add_argument("--period", action="append")
    parser.add_argument("--all-scenarios", action="store_true")
    parser.add_argument("--region", action="append", help="只运行指定中文分区名，可重复。仅 tile-source=regions 时使用。")
    parser.add_argument("--tile-source", choices=["dryland", "regions"], default="dryland")
    parser.add_argument("--mask-vector", default=str(DEFAULT_DRYLAND_MASK), help="用于 tile 筛选和 Stage10 栅格掩膜的干旱区矢量。")
    parser.add_argument("--mask-all-touched", action="store_true")
    parser.add_argument("--tile-deg", type=float, default=10.0)
    parser.add_argument("--margin-deg", type=float, default=2.0)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--limit-tiles", type=int, default=None)
    parser.add_argument("--limit-scenarios", type=int, default=None)
    parser.add_argument("--make-figure", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_paths(args)
    setup_logging()
    try:
        state = run(args)
        return 0 if state["status"] in {"success", "partial_success"} else 1
    except Exception as exc:
        err = {"status": "failed", "failed_at": now_iso(), "error": repr(exc), "traceback": traceback.format_exc()}
        atomic_write_json(STATE_JSON, err)
        logging.exception("Stage12 batch failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
