# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import traceback
import zlib
from datetime import datetime
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import force_2d
from shapely.geometry import Point
from shapely.validation import make_valid


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DESKTOP = Path.home() / "Desktop"
OASIS_BASE = DESKTOP / "绿洲编码最终版"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "stage01_presence_samples"
LOG_DIR = PROJECT_ROOT / "logs"
DATA_DIR = PROJECT_ROOT / "data"

LOG_PATH = LOG_DIR / "stage01_presence_sampling.log"
STATUS_CSV = LOG_DIR / "stage01_presence_sampling_status.csv"
STATE_JSON = LOG_DIR / "stage01_presence_sampling_state.json"
REGION_INDEX_CSV = DATA_DIR / "region_polygon_index.csv"

COMBINED_GPKG = OUTPUT_DIR / "presence_points_combined.gpkg"
COMBINED_CSV = OUTPUT_DIR / "presence_points_combined.csv"
PATCH_SUMMARY_CSV = OUTPUT_DIR / "patch_sampling_summary.csv"
PATCH_SUMMARY_XLSX = OUTPUT_DIR / "patch_sampling_summary.xlsx"
STRATA_SUMMARY_CSV = OUTPUT_DIR / "strata_sampling_summary.csv"
STRATA_SUMMARY_XLSX = OUTPUT_DIR / "strata_sampling_summary.xlsx"

SOURCE_CRS = "EPSG:4326"
AREA_CRS = "EPSG:6933"

DEFAULT_REGIONS = ["阿拉伯半岛"]
REGION_FILES = {
    "阿拉伯半岛": OASIS_BASE / "阿拉伯半岛" / "阿拉伯半岛.shp",
    "北美洲": OASIS_BASE / "北美洲" / "北美洲.shp",
    "大洋洲": OASIS_BASE / "大洋洲" / "大洋洲.shp",
    "非洲北部": OASIS_BASE / "非洲北部" / "非洲北部.shp",
    "非洲南部": OASIS_BASE / "非洲南部" / "非洲南部.shp",
    "南美洲": OASIS_BASE / "南美洲" / "南美洲.shp",
    "亚洲东部": OASIS_BASE / "亚洲东部" / "亚洲东部.shp",
    "亚洲西南部": OASIS_BASE / "亚洲西南部" / "亚洲西南部.shp",
    "亚洲中部": OASIS_BASE / "亚洲中部" / "亚洲中部.shp",
}

POLYGON_FIELDS = [
    "OasisID",
    "ContinentI",
    "CountryID",
    "BasinID",
    "AreaID",
    "Area",
    "Longitude",
    "Latitude",
    "Perimeter",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从非简化绿洲面矢量生成斑块感知 presence 样本点。"
    )
    parser.add_argument(
        "--regions",
        default=",".join(DEFAULT_REGIONS),
        help="逗号分隔的区域名。默认只跑阿拉伯半岛，避免误跑全球。",
    )
    parser.add_argument(
        "--all-regions",
        action="store_true",
        help="运行全部非简化分区。",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="重新处理已成功区域并覆盖该区域输出。",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="采样配额系数。n=max(1,min(n_max,round(alpha*sqrt(area_km2/cell_km2))))。",
    )
    parser.add_argument(
        "--n-max",
        type=int,
        default=10,
        help="单个斑块最多采样点数。",
    )
    parser.add_argument(
        "--cell-km2",
        type=float,
        default=1.0,
        help="目标网格面积，1 km 分辨率取 1。",
    )
    parser.add_argument(
        "--min-distance-m",
        type=float,
        default=1000.0,
        help="样本点最小距离。每个斑块的第一个强制样本会保留，额外样本会执行距离约束。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260529,
        help="随机种子。",
    )
    parser.add_argument(
        "--candidate-factor",
        type=int,
        default=80,
        help="拒绝采样候选点倍数。复杂细长斑块可增大。",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=250,
        help="每个斑块最多候选批次数。",
    )
    parser.add_argument(
        "--max-patches",
        type=int,
        default=0,
        help="调试用：每个区域最多处理前 N 个斑块。0 表示不限制。",
    )
    return parser.parse_args()


def setup_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(LOG_PATH, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def selected_regions(args: argparse.Namespace) -> list[str]:
    if args.all_regions:
        return list(REGION_FILES.keys())
    regions = [r.strip() for r in args.regions.split(",") if r.strip()]
    unknown = sorted(set(regions) - set(REGION_FILES))
    if unknown:
        raise ValueError(f"未知区域: {unknown}; 可选: {list(REGION_FILES)}")
    return regions


def save_region_index() -> None:
    rows = []
    for region, path in REGION_FILES.items():
        rows.append(
            {
                "Region": region,
                "Polygon_file": str(path),
                "Exists": path.exists(),
                "Size_MB": round(path.stat().st_size / 1024 / 1024, 3)
                if path.exists()
                else None,
                "Is_simplified": "simplified10m" in str(path),
            }
        )
    pd.DataFrame(rows).to_csv(REGION_INDEX_CSV, index=False, encoding="utf-8-sig")


def load_state() -> dict[str, Any]:
    if STATE_JSON.exists():
        return json.loads(STATE_JSON.read_text(encoding="utf-8"))
    return {}


def save_state(state: dict[str, Any]) -> None:
    tmp = STATE_JSON.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(STATE_JSON)


def write_status(rows: list[dict[str, Any]]) -> None:
    pd.DataFrame(rows).to_csv(STATUS_CSV, index=False, encoding="utf-8-sig")


def region_slug(region: str) -> str:
    mapping = {
        "阿拉伯半岛": "Arabian_Peninsula",
        "北美洲": "North_America",
        "大洋洲": "Oceania",
        "非洲北部": "North_Africa",
        "非洲南部": "Southern_Africa",
        "南美洲": "South_America",
        "亚洲东部": "East_Asia",
        "亚洲西南部": "Southwest_Asia",
        "亚洲中部": "Central_Asia",
    }
    return mapping[region]


def clean_geometries(gdf: gpd.GeoDataFrame, label: str) -> gpd.GeoDataFrame:
    before = len(gdf)
    gdf = gdf[gdf.geometry.notna()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()
    dropped_empty = before - len(gdf)
    if dropped_empty:
        logging.warning("%s dropped empty geometries: %s", label, dropped_empty)

    try:
        gdf["geometry"] = gdf.geometry.apply(force_2d)
    except Exception:
        logging.warning("%s force_2d failed; continuing with original geometries.", label)

    invalid = ~gdf.geometry.is_valid
    invalid_count = int(invalid.sum())
    if invalid_count:
        logging.info("%s invalid geometries: %s; repairing with make_valid.", label, invalid_count)
        gdf.loc[invalid, "geometry"] = gdf.loc[invalid, "geometry"].apply(make_valid)
        gdf = gdf[gdf.geometry.notna()].copy()
        gdf = gdf[~gdf.geometry.is_empty].copy()

    return gdf


def allocate_sample_count(
    area_km2: float,
    alpha: float,
    n_max: int,
    cell_km2: float,
) -> int:
    if not math.isfinite(area_km2) or area_km2 <= 0:
        return 0
    n = round(alpha * math.sqrt(area_km2 / cell_km2))
    return int(max(1, min(n_max, n)))


def classify_area_stratum(area_km2: float) -> str:
    if not math.isfinite(area_km2) or area_km2 <= 0:
        return "invalid"
    if area_km2 < 1:
        return "small_lt_1km2"
    if area_km2 < 10:
        return "medium_1_10km2"
    if area_km2 < 100:
        return "large_10_100km2"
    return "very_large_ge_100km2"


def random_points_in_polygon(
    polygon,
    n_target: int,
    min_distance_m: float,
    rng: np.random.Generator,
    candidate_factor: int,
    max_batches: int,
) -> tuple[list[Point], str]:
    if n_target <= 0 or polygon.is_empty:
        return [], "empty"

    minx, miny, maxx, maxy = polygon.bounds
    if maxx <= minx or maxy <= miny:
        point = polygon.representative_point()
        return [point], "representative_degenerate_bounds"

    accepted: list[Point] = []
    status = "random"
    batch_size = max(200, n_target * candidate_factor)

    for _ in range(max_batches):
        xs = rng.uniform(minx, maxx, batch_size)
        ys = rng.uniform(miny, maxy, batch_size)
        for x, y in zip(xs, ys):
            point = Point(float(x), float(y))
            if not polygon.covers(point):
                continue
            if accepted:
                nearest = min(point.distance(old) for old in accepted)
                if nearest < min_distance_m:
                    continue
            accepted.append(point)
            if len(accepted) >= n_target:
                return accepted, status

    if not accepted:
        accepted = [polygon.representative_point()]
        status = "representative_fallback"
    else:
        status = f"partial_random_{len(accepted)}_of_{n_target}"
    return accepted, status


def thin_extra_samples_keep_first(
    points_gdf: gpd.GeoDataFrame,
    min_distance_m: float,
) -> tuple[gpd.GeoDataFrame, dict[int, int]]:
    if points_gdf.empty:
        return points_gdf, {}

    mandatory = points_gdf[points_gdf["SampleSeqInPatch"] == 1].copy()
    extra = points_gdf[points_gdf["SampleSeqInPatch"] > 1].copy()

    accepted_geoms = list(mandatory.geometry)
    keep_extra_indexes: list[Any] = []

    extra = extra.sort_values(["PatchAreaKm2", "PatchID", "SampleSeqInPatch"], ascending=[False, True, True])
    for idx, row in extra.iterrows():
        point = row.geometry
        if not accepted_geoms:
            keep = True
        else:
            keep = min(point.distance(old) for old in accepted_geoms) >= min_distance_m
        if keep:
            keep_extra_indexes.append(idx)
            accepted_geoms.append(point)

    kept = pd.concat([mandatory, extra.loc[keep_extra_indexes]], ignore_index=False)
    kept = kept.sort_values(["PatchID", "SampleSeqInPatch"]).copy()
    counts = kept.groupby("PatchID").size().astype(int).to_dict()
    return kept, counts


def read_region_polygons(region: str, max_patches: int) -> gpd.GeoDataFrame:
    path = REGION_FILES[region]
    if not path.exists():
        raise FileNotFoundError(path)
    if "simplified10m" in str(path):
        raise ValueError(f"拒绝使用简化文件: {path}")

    logging.info("Reading region %s: %s", region, path)
    gdf = gpd.read_file(path)
    logging.info("Loaded %s features for %s", len(gdf), region)

    if gdf.crs is None:
        raise ValueError(f"{path} 缺少 CRS")
    if str(gdf.crs).upper() not in {"EPSG:4326", "OGC:CRS84"}:
        logging.info("%s CRS=%s; reprojecting to EPSG:4326 before area projection.", region, gdf.crs)
        gdf = gdf.to_crs(SOURCE_CRS)

    gdf = clean_geometries(gdf, region)
    gdf = gdf.reset_index(drop=False).rename(columns={"index": "SourceFeatureIndex"})

    if max_patches and max_patches > 0:
        logging.warning("%s debug limit active: only first %s patches.", region, max_patches)
        gdf = gdf.head(max_patches).copy()

    return gdf


def sample_region(region: str, args: argparse.Namespace) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    gdf = read_region_polygons(region, args.max_patches)
    area_gdf = gdf.to_crs(AREA_CRS)
    area_gdf["PatchAreaKm2"] = area_gdf.geometry.area / 1_000_000.0
    area_gdf["PatchID"] = [f"{region_slug(region)}_{i + 1:06d}" for i in range(len(area_gdf))]

    region_seed = args.seed + zlib.crc32(region.encode("utf-8")) % 1_000_000
    rng = np.random.default_rng(region_seed)

    point_rows: list[dict[str, Any]] = []
    point_geoms: list[Point] = []
    patch_rows: list[dict[str, Any]] = []

    attribute_fields = ["SourceFeatureIndex", "PatchID", "PatchAreaKm2"] + [
        field for field in POLYGON_FIELDS if field in area_gdf.columns
    ]

    for ordinal, row in area_gdf.iterrows():
        area_km2 = float(row["PatchAreaKm2"])
        n_target = allocate_sample_count(
            area_km2=area_km2,
            alpha=args.alpha,
            n_max=args.n_max,
            cell_km2=args.cell_km2,
        )
        geom = row.geometry
        area_stratum = classify_area_stratum(area_km2)
        raw_points, method = random_points_in_polygon(
            geom,
            n_target,
            args.min_distance_m,
            rng,
            args.candidate_factor,
            args.max_batches,
        )

        patch_id = row["PatchID"]
        base_attrs = {field: row.get(field) for field in attribute_fields}

        for seq, point in enumerate(raw_points, start=1):
            rec = {
                "SourceRegion": region,
                "PatchID": patch_id,
                "SourceFeatureIndex": int(row["SourceFeatureIndex"]),
                "SampleSeqInPatch": seq,
                "PatchAreaKm2": area_km2,
                "AreaStratum": area_stratum,
                "TargetSamples": n_target,
                "RawGeneratedSamples": len(raw_points),
                "IsMandatoryPatchSample": seq == 1,
                "SamplingMethod": method,
                "MinDistanceM": args.min_distance_m,
                "Alpha": args.alpha,
                "NMax": args.n_max,
                "CellKm2": args.cell_km2,
                "RandomSeed": args.seed,
            }
            for key, value in base_attrs.items():
                if key not in rec:
                    rec[key] = value
            point_rows.append(rec)
            point_geoms.append(point)

        patch_rows.append(
            {
                "SourceRegion": region,
                "PatchID": patch_id,
                "SourceFeatureIndex": int(row["SourceFeatureIndex"]),
                "OasisID": row.get("OasisID") if "OasisID" in row.index else None,
                "CountryID": row.get("CountryID") if "CountryID" in row.index else None,
                "BasinID": row.get("BasinID") if "BasinID" in row.index else None,
                "AreaID": row.get("AreaID") if "AreaID" in row.index else None,
                "PatchAreaKm2": area_km2,
                "AreaStratum": area_stratum,
                "TargetSamples": n_target,
                "RawGeneratedSamples": len(raw_points),
                "SamplingMethod": method,
            }
        )

        if (ordinal + 1) % 50 == 0 or (ordinal + 1) == len(area_gdf):
            logging.info(
                "%s sampled patches: %s/%s, raw_points=%s",
                region,
                ordinal + 1,
                len(area_gdf),
                len(point_rows),
            )

    raw_points_gdf = gpd.GeoDataFrame(point_rows, geometry=point_geoms, crs=AREA_CRS)
    kept_points_gdf, kept_counts = thin_extra_samples_keep_first(
        raw_points_gdf,
        args.min_distance_m,
    )

    patch_summary = pd.DataFrame(patch_rows)
    patch_summary["KeptSamplesAfterThinning"] = patch_summary["PatchID"].map(kept_counts).fillna(0).astype(int)
    patch_summary["DroppedByRegionThinning"] = (
        patch_summary["RawGeneratedSamples"] - patch_summary["KeptSamplesAfterThinning"]
    )
    patch_summary["RunAlpha"] = args.alpha
    patch_summary["RunNMax"] = args.n_max
    patch_summary["RunCellKm2"] = args.cell_km2
    patch_summary["RunMinDistanceM"] = args.min_distance_m

    kept_points_gdf = kept_points_gdf.to_crs(SOURCE_CRS)
    kept_points_gdf["PointLon"] = kept_points_gdf.geometry.x
    kept_points_gdf["PointLat"] = kept_points_gdf.geometry.y
    kept_points_gdf["geometry_wkt"] = kept_points_gdf.geometry.to_wkt()
    kept_points_gdf = kept_points_gdf.reset_index(drop=True)

    return kept_points_gdf, patch_summary


def write_region_outputs(region: str, points: gpd.GeoDataFrame, patch_summary: pd.DataFrame) -> None:
    slug = region_slug(region)
    region_dir = OUTPUT_DIR / slug
    region_dir.mkdir(parents=True, exist_ok=True)

    points_gpkg = region_dir / f"{slug}_presence_points.gpkg"
    points_csv = region_dir / f"{slug}_presence_points.csv"
    patch_csv = region_dir / f"{slug}_patch_sampling_summary.csv"

    points.to_file(points_gpkg, driver="GPKG")
    points.drop(columns="geometry").to_csv(points_csv, index=False, encoding="utf-8-sig")
    patch_summary.to_csv(patch_csv, index=False, encoding="utf-8-sig")

    logging.info("Wrote region outputs: %s", region_dir)


def write_combined_outputs(regions: list[str]) -> None:
    point_parts: list[gpd.GeoDataFrame] = []
    patch_parts: list[pd.DataFrame] = []

    for region in regions:
        slug = region_slug(region)
        region_dir = OUTPUT_DIR / slug
        points_gpkg = region_dir / f"{slug}_presence_points.gpkg"
        patch_csv = region_dir / f"{slug}_patch_sampling_summary.csv"
        if points_gpkg.exists():
            point_parts.append(gpd.read_file(points_gpkg))
        if patch_csv.exists():
            patch_parts.append(pd.read_csv(patch_csv, encoding="utf-8-sig"))

    if point_parts:
        combined = pd.concat(point_parts, ignore_index=True)
        combined = gpd.GeoDataFrame(combined, geometry="geometry", crs=point_parts[0].crs)
        combined.to_file(COMBINED_GPKG, driver="GPKG")
        combined.drop(columns="geometry").to_csv(COMBINED_CSV, index=False, encoding="utf-8-sig")
        logging.info("Wrote combined presence samples: %s records", len(combined))

    if patch_parts:
        patch_summary = pd.concat(patch_parts, ignore_index=True)
        patch_summary.to_csv(PATCH_SUMMARY_CSV, index=False, encoding="utf-8-sig")
        group_cols = ["SourceRegion", "AreaStratum"]
        if "BasinID" in patch_summary.columns:
            patch_summary["BasinID_for_strata"] = patch_summary["BasinID"].fillna("NA").astype(str)
            group_cols.append("BasinID_for_strata")
        strata_summary = (
            patch_summary.groupby(group_cols, dropna=False)
            .agg(
                Patch_count=("PatchID", "count"),
                Patch_area_km2_sum=("PatchAreaKm2", "sum"),
                Target_samples_sum=("TargetSamples", "sum"),
                Kept_samples_sum=("KeptSamplesAfterThinning", "sum"),
                Dropped_by_thinning_sum=("DroppedByRegionThinning", "sum"),
            )
            .reset_index()
        )
        strata_summary.to_csv(STRATA_SUMMARY_CSV, index=False, encoding="utf-8-sig")
        try:
            patch_summary.to_excel(PATCH_SUMMARY_XLSX, index=False)
            strata_summary.to_excel(STRATA_SUMMARY_XLSX, index=False)
        except Exception as exc:
            logging.warning("Writing patch summary xlsx failed; CSV is still available: %r", exc)
        logging.info("Wrote combined patch summary: %s records", len(patch_summary))
        logging.info("Wrote strata sampling summary: %s records", len(strata_summary))


def main() -> int:
    args = parse_args()
    setup_logging()
    save_region_index()

    regions = selected_regions(args)
    logging.info("Stage 01 presence sampling started")
    logging.info("Selected regions: %s", regions)
    logging.info(
        "Parameters: alpha=%s, n_max=%s, cell_km2=%s, min_distance_m=%s, seed=%s",
        args.alpha,
        args.n_max,
        args.cell_km2,
        args.min_distance_m,
        args.seed,
    )

    state = load_state()
    status_rows: list[dict[str, Any]] = []
    failed = 0

    for region in regions:
        slug = region_slug(region)
        region_state = state.get(region, {})
        region_dir = OUTPUT_DIR / slug
        region_points = region_dir / f"{slug}_presence_points.gpkg"

        status = {
            "Region": region,
            "Region_slug": slug,
            "Polygon_file": str(REGION_FILES[region]),
            "Status": "pending",
            "Started_at": "",
            "Finished_at": "",
            "Input_patches": "",
            "Output_presence_points": "",
            "Message": "",
        }
        status_rows.append(status)
        write_status(status_rows)

        if (
            not args.overwrite
            and region_state.get("status") == "success"
            and region_points.exists()
        ):
            logging.info("Skip completed region: %s", region)
            status.update(
                {
                    "Status": "success",
                    "Started_at": region_state.get("started_at", ""),
                    "Finished_at": region_state.get("finished_at", ""),
                    "Input_patches": region_state.get("input_patches", ""),
                    "Output_presence_points": region_state.get("output_presence_points", ""),
                    "Message": "skipped_existing_success",
                }
            )
            write_status(status_rows)
            continue

        status.update({"Status": "running", "Started_at": datetime.now().isoformat(timespec="seconds")})
        state[region] = {"status": "running", "started_at": status["Started_at"]}
        save_state(state)
        write_status(status_rows)

        try:
            points, patch_summary = sample_region(region, args)
            write_region_outputs(region, points, patch_summary)

            status.update(
                {
                    "Status": "success",
                    "Finished_at": datetime.now().isoformat(timespec="seconds"),
                    "Input_patches": len(patch_summary),
                    "Output_presence_points": len(points),
                    "Message": "done",
                }
            )
            state[region] = {
                "status": "success",
                "started_at": status["Started_at"],
                "finished_at": status["Finished_at"],
                "input_patches": int(len(patch_summary)),
                "output_presence_points": int(len(points)),
                "message": "done",
            }
            logging.info(
                "Finished %s: patches=%s, presence_points=%s",
                region,
                len(patch_summary),
                len(points),
            )
        except Exception as exc:
            failed += 1
            status.update(
                {
                    "Status": "failed",
                    "Finished_at": datetime.now().isoformat(timespec="seconds"),
                    "Message": repr(exc),
                }
            )
            state[region] = {
                "status": "failed",
                "started_at": status["Started_at"],
                "finished_at": status["Finished_at"],
                "message": repr(exc),
                "traceback": traceback.format_exc()[-5000:],
            }
            logging.error("Region failed: %s", region)
            logging.error(traceback.format_exc())
        finally:
            save_state(state)
            write_status(status_rows)

    write_combined_outputs(regions)
    logging.info("Stage 01 presence sampling finished. failed=%s", failed)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
