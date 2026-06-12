# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import force_2d
from shapely.geometry import Point
from shapely.prepared import prep
from shapely.validation import make_valid


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DESKTOP = Path.home() / "Desktop"
OASIS_BASE = DESKTOP / "绿洲编码最终版"
DRYLAND_SHP = DESKTOP / "会议相关" / "世界绿洲合并" / "世界绿洲" / "全球干旱区" / "AI0-0.65干旱区.shp"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "stage02_background_points"
LOG_DIR = PROJECT_ROOT / "logs"
LOG_PATH = LOG_DIR / "stage02_background_sampling_fast.log"
STATUS_CSV = LOG_DIR / "stage02_background_sampling_fast_status.csv"
STATE_JSON = LOG_DIR / "stage02_background_sampling_fast_state.json"

BACKGROUND_GPKG = OUTPUT_DIR / "background_points_combined.gpkg"
BACKGROUND_CSV = OUTPUT_DIR / "background_points_combined.csv"
BACKGROUND_SUMMARY_CSV = OUTPUT_DIR / "background_sampling_summary.csv"
BACKGROUND_SUMMARY_XLSX = OUTPUT_DIR / "background_sampling_summary.xlsx"
BACKGROUND_VALIDATION_CSV = OUTPUT_DIR / "background_spatial_validation.csv"
BUFFER_CACHE_GPKG = OUTPUT_DIR / "oasis_exclusion_buffer_3000m.gpkg"

SOURCE_CRS = "EPSG:4326"
AREA_CRS = "EPSG:6933"

REGION_NAMES = [
    "阿拉伯半岛",
    "北美洲",
    "大洋洲",
    "非洲北部",
    "非洲南部",
    "南美洲",
    "亚洲东部",
    "亚洲西南部",
    "亚洲中部",
]
REGION_FILES = {name: OASIS_BASE / name / f"{name}.shp" for name in REGION_NAMES}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="分块生成 AI<0.65 干旱区背景点，并用空间索引排除绿洲缓冲区。"
    )
    parser.add_argument("--background-ratio", type=float, default=3.0)
    parser.add_argument("--total-background", type=int, default=0)
    parser.add_argument("--exclude-buffer-m", type=float, default=3000.0)
    parser.add_argument("--seed", type=int, default=20260529)
    parser.add_argument("--candidate-factor", type=float, default=2.5)
    parser.add_argument("--max-rounds", type=int, default=120)
    parser.add_argument("--min-per-stratum", type=int, default=1000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--reuse-buffer-cache", action="store_true")
    return parser.parse_args()


def setup_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(LOG_PATH, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def load_state() -> dict[str, Any]:
    if STATE_JSON.exists():
        return json.loads(STATE_JSON.read_text(encoding="utf-8"))
    return {}


def save_state(state: dict[str, Any]) -> None:
    tmp = STATE_JSON.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(STATE_JSON)


def write_status(row: dict[str, Any]) -> None:
    pd.DataFrame([row]).to_csv(STATUS_CSV, index=False, encoding="utf-8-sig")


def clean_geometries(gdf: gpd.GeoDataFrame, label: str) -> gpd.GeoDataFrame:
    before = len(gdf)
    gdf = gdf[gdf.geometry.notna()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()
    dropped = before - len(gdf)
    if dropped:
        logging.warning("%s dropped empty geometries: %s", label, dropped)
    try:
        gdf["geometry"] = gdf.geometry.apply(force_2d)
    except Exception:
        logging.warning("%s force_2d failed; continuing.", label)
    invalid = ~gdf.geometry.is_valid
    invalid_count = int(invalid.sum())
    if invalid_count:
        logging.info("%s invalid geometries: %s; repairing with make_valid.", label, invalid_count)
        gdf.loc[invalid, "geometry"] = gdf.loc[invalid, "geometry"].apply(make_valid)
        gdf = gdf[gdf.geometry.notna()].copy()
        gdf = gdf[~gdf.geometry.is_empty].copy()
    return gdf


def read_presence_count() -> int:
    presence_csv = PROJECT_ROOT / "outputs" / "stage01_presence_samples" / "presence_points_combined.csv"
    if not presence_csv.exists():
        raise FileNotFoundError(f"缺少阶段 01 presence 点: {presence_csv}")
    return int(sum(1 for _ in presence_csv.open("r", encoding="utf-8-sig")) - 1)


def read_dryland() -> gpd.GeoDataFrame:
    if not DRYLAND_SHP.exists():
        raise FileNotFoundError(DRYLAND_SHP)
    dry = gpd.read_file(DRYLAND_SHP)
    if dry.crs is None:
        raise ValueError(f"干旱区文件缺少 CRS: {DRYLAND_SHP}")
    if str(dry.crs).upper() not in {"EPSG:4326", "OGC:CRS84"}:
        dry = dry.to_crs(SOURCE_CRS)
    dry = clean_geometries(dry, "AI0-0.65 dryland").to_crs(AREA_CRS)
    dry["DrylandAreaKm2"] = dry.geometry.area / 1_000_000.0
    dry["DrylandStratum"] = dry.apply(
        lambda r: str(r.get("HIX_DESC") or r.get("HIX_ZONE") or f"dryland_{r.name + 1}"),
        axis=1,
    )
    dry["DrylandStratum"] = dry["DrylandStratum"].str.replace(r"\s+", "_", regex=True)
    return dry


def build_oasis_buffers(buffer_m: float, reuse_cache: bool) -> gpd.GeoDataFrame:
    if reuse_cache and BUFFER_CACHE_GPKG.exists():
        logging.info("Loading existing buffer cache: %s", BUFFER_CACHE_GPKG)
        buffers = gpd.read_file(BUFFER_CACHE_GPKG)
        if buffers.crs is None:
            buffers = buffers.set_crs(AREA_CRS)
        elif str(buffers.crs) != AREA_CRS:
            buffers = buffers.to_crs(AREA_CRS)
        return buffers[["Region", "geometry"]].copy()

    parts: list[gpd.GeoDataFrame] = []
    for region, path in REGION_FILES.items():
        if not path.exists():
            raise FileNotFoundError(path)
        logging.info("Reading oasis polygons for exclusion: %s", region)
        gdf = gpd.read_file(path)
        if gdf.crs is None:
            raise ValueError(f"{path} 缺少 CRS")
        if str(gdf.crs).upper() not in {"EPSG:4326", "OGC:CRS84"}:
            gdf = gdf.to_crs(SOURCE_CRS)
        gdf = clean_geometries(gdf, f"oasis_{region}").to_crs(AREA_CRS)
        logging.info("Buffering %s features for %s by %.1f m", len(gdf), region, buffer_m)
        buffered = gdf.geometry.buffer(buffer_m, resolution=8)
        part = gpd.GeoDataFrame(
            {"Region": [region] * len(buffered)},
            geometry=buffered,
            crs=AREA_CRS,
        )
        part = part[part.geometry.notna()].copy()
        part = part[~part.geometry.is_empty].copy()
        parts.append(part)
        logging.info("Buffered exclusion polygons: %s features for %s", len(part), region)

    buffers = pd.concat(parts, ignore_index=True)
    buffers = gpd.GeoDataFrame(buffers, geometry="geometry", crs=AREA_CRS)
    logging.info("Writing buffer cache: %s", BUFFER_CACHE_GPKG)
    buffers.to_file(BUFFER_CACHE_GPKG, driver="GPKG")
    logging.info("Buffer cache finished; total buffered features: %s", len(buffers))
    return buffers[["Region", "geometry"]].copy()


def allocate_by_area(dry: gpd.GeoDataFrame, total: int, min_per_stratum: int) -> dict[int, int]:
    area = dry["DrylandAreaKm2"].astype(float)
    raw = area / area.sum() * total
    alloc = raw.round().astype(int)
    if min_per_stratum > 0 and total >= min_per_stratum * len(dry):
        alloc = alloc.clip(lower=min_per_stratum)
    diff = int(total - alloc.sum())
    order = list((raw - np.floor(raw)).sort_values(ascending=diff > 0).index)
    step = 1 if diff > 0 else -1
    k = 0
    while diff != 0 and order:
        idx = order[k % len(order)]
        if step < 0 and alloc.loc[idx] <= 1:
            k += 1
            continue
        alloc.loc[idx] += step
        diff -= step
        k += 1
    return {int(i): int(v) for i, v in alloc.items()}


def filter_outside_buffers(points: gpd.GeoDataFrame, buffers: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if points.empty:
        return points
    joined = gpd.sjoin(points[["geometry"]], buffers[["geometry"]], how="left", predicate="intersects")
    blocked = joined.index[joined["index_right"].notna()].unique()
    if len(blocked) == 0:
        return points
    return points.loc[~points.index.isin(blocked)].copy()


def sample_stratum_points(
    geom,
    target: int,
    buffers: gpd.GeoDataFrame,
    rng: np.random.Generator,
    candidate_factor: float,
    max_rounds: int,
    stratum: str,
) -> tuple[list[Point], int]:
    prepared_domain = prep(geom)
    minx, miny, maxx, maxy = geom.bounds
    accepted: list[Point] = []
    total_candidates = 0
    batch_size = max(10_000, min(250_000, int(math.ceil(max(target, 1) * candidate_factor))))

    for round_i in range(1, max_rounds + 1):
        needed = target - len(accepted)
        if needed <= 0:
            break
        this_batch = max(batch_size, min(250_000, int(math.ceil(needed * candidate_factor))))
        xs = rng.uniform(minx, maxx, this_batch)
        ys = rng.uniform(miny, maxy, this_batch)
        total_candidates += this_batch

        inside = []
        for x, y in zip(xs, ys):
            point = Point(float(x), float(y))
            if prepared_domain.covers(point):
                inside.append(point)
        if not inside:
            logging.info(
                "Stratum=%s round=%s no dryland candidates; total_candidates=%s",
                stratum,
                round_i,
                total_candidates,
            )
            continue

        cand = gpd.GeoDataFrame(geometry=inside, crs=AREA_CRS)
        cand = cand.reset_index(drop=True)
        cand = filter_outside_buffers(cand, buffers)
        take_n = min(needed, len(cand))
        if take_n > 0:
            accepted.extend(list(cand.geometry.iloc[:take_n]))

        logging.info(
            "Stratum=%s round=%s accepted=%s/%s inside_batch=%s outside_buffer_batch=%s total_candidates=%s",
            stratum,
            round_i,
            len(accepted),
            target,
            len(inside),
            len(cand),
            total_candidates,
        )
    return accepted, total_candidates


def validate_background(points_area: gpd.GeoDataFrame, dry: gpd.GeoDataFrame, buffers: gpd.GeoDataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    blocked_all = gpd.sjoin(points_area[["DrylandStratum", "geometry"]], buffers[["geometry"]], how="left", predicate="intersects")
    blocked_index = set(blocked_all.index[blocked_all["index_right"].notna()].unique())

    for stratum, sub in points_area.groupby("DrylandStratum"):
        dry_sub = dry[dry["DrylandStratum"] == stratum]
        prepared_domains = [prep(geom) for geom in dry_sub.geometry]
        inside_dryland = [
            any(prepared_domain.covers(geom) for prepared_domain in prepared_domains)
            for geom in sub.geometry
        ]
        inside_buffer = [idx in blocked_index for idx in sub.index]
        rows.append(
            {
                "DrylandStratum": stratum,
                "Checked_points": int(len(sub)),
                "Inside_dryland": int(sum(inside_dryland)),
                "Outside_dryland": int(len(sub) - sum(inside_dryland)),
                "Outside_oasis_buffer": int(len(sub) - sum(inside_buffer)),
                "Inside_oasis_buffer": int(sum(inside_buffer)),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    setup_logging()
    state = load_state()
    status = {
        "Task": "stage02_background_sampling_fast",
        "Status": "running",
        "Started_at": datetime.now().isoformat(timespec="seconds"),
        "Finished_at": "",
        "Presence_points": "",
        "Target_background_points": "",
        "Output_background_points": "",
        "Exclude_buffer_m": args.exclude_buffer_m,
        "Message": "",
    }
    write_status(status)

    if (
        not args.overwrite
        and state.get("status") == "success"
        and BACKGROUND_GPKG.exists()
        and BACKGROUND_CSV.exists()
    ):
        status.update(
            {
                "Status": "success",
                "Finished_at": state.get("finished_at", ""),
                "Presence_points": state.get("presence_points", ""),
                "Target_background_points": state.get("target_background_points", ""),
                "Output_background_points": state.get("output_background_points", ""),
                "Message": "skipped_existing_success",
            }
        )
        write_status(status)
        logging.info("Existing successful stage02 output found; skipped.")
        return 0

    try:
        presence_count = read_presence_count()
        total_background = (
            args.total_background
            if args.total_background > 0
            else int(round(presence_count * args.background_ratio))
        )
        status.update(
            {
                "Presence_points": presence_count,
                "Target_background_points": total_background,
            }
        )
        write_status(status)
        logging.info("Presence points: %s", presence_count)
        logging.info("Target background points: %s", total_background)

        dry = read_dryland()
        logging.info("Dryland strata: %s", dry[["DrylandStratum", "DrylandAreaKm2"]].to_dict("records"))

        buffers = build_oasis_buffers(args.exclude_buffer_m, args.reuse_buffer_cache)
        _ = buffers.sindex
        logging.info("Oasis buffer spatial index ready: %s features", len(buffers))

        allocation = allocate_by_area(dry, total_background, args.min_per_stratum)
        rng = np.random.default_rng(args.seed)

        point_rows: list[dict[str, Any]] = []
        point_geoms: list[Point] = []
        summary_rows: list[dict[str, Any]] = []

        for idx, row in dry.iterrows():
            target = int(allocation.get(int(idx), 0))
            stratum = row["DrylandStratum"]
            logging.info("Sampling dryland stratum=%s target=%s", stratum, target)
            points, candidates = sample_stratum_points(
                row.geometry,
                target,
                buffers,
                rng,
                args.candidate_factor,
                args.max_rounds,
                stratum,
            )
            if len(points) < target:
                logging.warning("Stratum %s only generated %s/%s background points", stratum, len(points), target)
            for seq, point in enumerate(points, start=1):
                point_rows.append(
                    {
                        "SampleType": "background",
                        "DrylandStratum": stratum,
                        "DrylandFeatureIndex": int(idx),
                        "BackgroundSeqInStratum": seq,
                        "TargetInStratum": target,
                        "ExcludeBufferM": args.exclude_buffer_m,
                        "RandomSeed": args.seed,
                        "SourceDrylandFile": str(DRYLAND_SHP),
                        "SamplingScript": Path(__file__).name,
                    }
                )
                point_geoms.append(point)
            summary_rows.append(
                {
                    "DrylandStratum": stratum,
                    "DrylandFeatureIndex": int(idx),
                    "DrylandAreaKm2": float(row["DrylandAreaKm2"]),
                    "TargetBackgroundPoints": target,
                    "GeneratedBackgroundPoints": len(points),
                    "CandidatePointsTried": candidates,
                    "ExcludeBufferM": args.exclude_buffer_m,
                }
            )
            logging.info("Finished stratum=%s generated=%s candidates=%s", stratum, len(points), candidates)

        bg_area = gpd.GeoDataFrame(point_rows, geometry=point_geoms, crs=AREA_CRS)
        bg_area = bg_area.reset_index(drop=True)
        validation = validate_background(bg_area, dry, buffers)

        inside_buffer = int(validation["Inside_oasis_buffer"].sum()) if not validation.empty else 0
        outside_dryland = int(validation["Outside_dryland"].sum()) if not validation.empty else 0
        if inside_buffer or outside_dryland:
            raise RuntimeError(
                f"背景点空间校验未通过: inside_oasis_buffer={inside_buffer}, outside_dryland={outside_dryland}"
            )

        bg = bg_area.to_crs(SOURCE_CRS)
        bg["PointLon"] = bg.geometry.x
        bg["PointLat"] = bg.geometry.y
        bg["geometry_wkt"] = bg.geometry.to_wkt()

        summary = pd.DataFrame(summary_rows)
        bg.to_file(BACKGROUND_GPKG, driver="GPKG")
        bg.drop(columns="geometry").to_csv(BACKGROUND_CSV, index=False, encoding="utf-8-sig")
        summary.to_csv(BACKGROUND_SUMMARY_CSV, index=False, encoding="utf-8-sig")
        validation.to_csv(BACKGROUND_VALIDATION_CSV, index=False, encoding="utf-8-sig")
        try:
            summary.to_excel(BACKGROUND_SUMMARY_XLSX, index=False)
        except Exception as exc:
            logging.warning("Writing xlsx failed; CSV is available: %r", exc)

        status.update(
            {
                "Status": "success",
                "Finished_at": datetime.now().isoformat(timespec="seconds"),
                "Output_background_points": len(bg),
                "Message": "done",
            }
        )
        state.update(
            {
                "status": "success",
                "finished_at": status["Finished_at"],
                "presence_points": presence_count,
                "target_background_points": total_background,
                "output_background_points": int(len(bg)),
                "exclude_buffer_m": args.exclude_buffer_m,
                "message": "done",
            }
        )
        logging.info("Stage02 fast background sampling finished: %s points", len(bg))
        return 0
    except Exception as exc:
        status.update(
            {
                "Status": "failed",
                "Finished_at": datetime.now().isoformat(timespec="seconds"),
                "Message": repr(exc),
            }
        )
        state.update(
            {
                "status": "failed",
                "finished_at": status["Finished_at"],
                "message": repr(exc),
                "traceback": traceback.format_exc()[-6000:],
            }
        )
        logging.error("Stage02 fast failed: %r", exc)
        logging.error(traceback.format_exc())
        return 1
    finally:
        save_state(state)
        write_status(status)


if __name__ == "__main__":
    raise SystemExit(main())
