# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import shutil
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import numpy as np
import pandas as pd

PROJ_DIR = Path(r"C:\Users\linjingwu\anaconda3\Library\share\proj")
if PROJ_DIR.exists():
    os.environ["PROJ_LIB"] = str(PROJ_DIR)

import rasterio
from pyproj import Transformer


WORKSPACE = Path(r"C:\Users\linjingwu\Desktop\python")
PROJECT_ROOT = WORKSPACE / "绿洲未来适宜区预测"
DATA_ROOT = Path(r"D:\绿洲未来适宜区预测数据")

RAW_CURRENT_DIR = DATA_ROOT / "raw" / "worldclim" / "current_30s"
PROCESSED_DIR = DATA_ROOT / "processed" / "worldclim" / "current_30s"
OUT_DIR = PROJECT_ROOT / "outputs" / "stage04_current_worldclim_features"
LOG_DIR = PROJECT_ROOT / "logs"

SAMPLES_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "stage03_modeling_samples_spatial_cv"
    / "modeling_samples_with_spatial_folds.csv"
)

BIO_ZIP = RAW_CURRENT_DIR / "wc2.1_30s_bio.zip"
ELEV_ZIP = RAW_CURRENT_DIR / "wc2.1_30s_elev.zip"

LOG_PATH = LOG_DIR / "stage04_current_worldclim_feature_extraction.log"
STATE_PATH = LOG_DIR / "stage04_current_worldclim_feature_extraction_state.json"
STATUS_CSV = LOG_DIR / "stage04_current_worldclim_feature_extraction_status.csv"
VARIABLE_STATUS_CSV = LOG_DIR / "stage04_current_worldclim_variable_status.csv"

OUTPUT_CSV = OUT_DIR / "modeling_samples_with_current_worldclim.csv"
VARIABLE_SUMMARY_CSV = OUT_DIR / "current_worldclim_variable_summary.csv"
REPORT_MD = OUT_DIR / "stage04_current_worldclim_feature_extraction_report.md"


@dataclass(frozen=True)
class RasterVariable:
    name: str
    path: Path
    source: str


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def setup_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(LOG_PATH, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(data, ensure_ascii=False, indent=2))


def atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8-sig")
    tmp.replace(path)


def write_status(status: str, message: str, extra: dict[str, Any] | None = None) -> None:
    row: dict[str, Any] = {"updated_at": now_iso(), "status": status, "message": message}
    if extra:
        row.update(extra)
    atomic_write_csv(pd.DataFrame([row]), STATUS_CSV)


def write_variable_status(rows: list[dict[str, Any]]) -> None:
    if rows:
        atomic_write_csv(pd.DataFrame(rows), VARIABLE_STATUS_CSV)


def load_state() -> dict[str, Any]:
    if not STATE_PATH.exists():
        return {}
    return json.loads(STATE_PATH.read_text(encoding="utf-8"))


def output_is_complete(expected_rows: int, expected_columns: list[str]) -> bool:
    if not OUTPUT_CSV.exists():
        return False
    try:
        header = pd.read_csv(OUTPUT_CSV, nrows=0).columns.tolist()
        if any(col not in header for col in expected_columns):
            return False
        row_count = sum(1 for _ in open(OUTPUT_CSV, "rb")) - 1
        return row_count == expected_rows
    except Exception:
        return False


def extract_zip_member(zip_file: ZipFile, member_name: str, target_path: Path, expected_size: int, overwrite: bool) -> str:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists() and target_path.stat().st_size == expected_size and not overwrite:
        return "skipped_existing"

    tmp = target_path.with_suffix(target_path.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()

    with zip_file.open(member_name) as src, tmp.open("wb") as dst:
        shutil.copyfileobj(src, dst, length=16 * 1024 * 1024)

    actual_size = tmp.stat().st_size
    if actual_size != expected_size:
        raise IOError(f"Extracted size mismatch for {member_name}: {actual_size} != {expected_size}")
    tmp.replace(target_path)
    return "extracted"


def extract_current_worldclim(overwrite: bool) -> list[dict[str, Any]]:
    tasks = [
        (BIO_ZIP, PROCESSED_DIR / "bio"),
        (ELEV_ZIP, PROCESSED_DIR / "elev"),
    ]
    rows: list[dict[str, Any]] = []
    for zip_path, target_dir in tasks:
        if not zip_path.exists():
            raise FileNotFoundError(f"WorldClim zip missing: {zip_path}")
        logging.info("Checking zip: %s", zip_path)
        with ZipFile(zip_path) as zf:
            for info in zf.infolist():
                if info.is_dir() or not info.filename.lower().endswith(".tif"):
                    continue
                target = target_dir / Path(info.filename).name
                row = {
                    "updated_at": now_iso(),
                    "step": "extract",
                    "item": info.filename,
                    "target": str(target),
                    "status": "running",
                    "message": "",
                }
                rows.append(row)
                write_variable_status(rows)
                try:
                    result = extract_zip_member(zf, info.filename, target, info.file_size, overwrite)
                    row.update(
                        {
                            "updated_at": now_iso(),
                            "status": "success",
                            "message": result,
                            "bytes": info.file_size,
                        }
                    )
                    logging.info("%s %s -> %s", result, info.filename, target)
                except Exception as exc:
                    row.update(
                        {
                            "updated_at": now_iso(),
                            "status": "failed",
                            "message": repr(exc),
                            "traceback": traceback.format_exc(),
                        }
                    )
                    logging.exception("Failed extracting %s", info.filename)
                write_variable_status(rows)
    return rows


def list_raster_variables() -> list[RasterVariable]:
    variables: list[RasterVariable] = []
    bio_dir = PROCESSED_DIR / "bio"
    elev_dir = PROCESSED_DIR / "elev"

    bio_re = re.compile(r"wc2\.1_30s_bio_(\d+)\.tif$", re.IGNORECASE)
    for path in sorted(bio_dir.glob("wc2.1_30s_bio_*.tif")):
        match = bio_re.match(path.name)
        if not match:
            continue
        bio_num = int(match.group(1))
        variables.append(RasterVariable(name=f"wc_bio{bio_num:02d}", path=path, source="WorldClim_current_bioclim_30s"))

    elev_path = elev_dir / "wc2.1_30s_elev.tif"
    if elev_path.exists():
        variables.append(RasterVariable(name="wc_elev_m", path=elev_path, source="WorldClim_current_elevation_30s"))

    variables = sorted(variables, key=lambda v: (0 if v.name.startswith("wc_bio") else 1, v.name))
    expected = {f"wc_bio{i:02d}" for i in range(1, 20)} | {"wc_elev_m"}
    found = {v.name for v in variables}
    missing = sorted(expected - found)
    if missing:
        raise FileNotFoundError(f"Missing extracted raster variables: {missing}")
    return variables


def read_samples(max_rows: int | None = None) -> pd.DataFrame:
    if not SAMPLES_CSV.exists():
        raise FileNotFoundError(f"Modeling sample CSV missing: {SAMPLES_CSV}")
    df = pd.read_csv(SAMPLES_CSV, nrows=max_rows)
    required = {"SampleID", "Response", "PointLon", "PointLat", "SpatialCVFold"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Modeling sample CSV missing required columns: {missing}")
    df["PointLon"] = pd.to_numeric(df["PointLon"], errors="coerce")
    df["PointLat"] = pd.to_numeric(df["PointLat"], errors="coerce")
    bad_coords = df[["PointLon", "PointLat"]].isna().any(axis=1)
    out_of_range = (df["PointLon"] < -180) | (df["PointLon"] > 180) | (df["PointLat"] < -90) | (df["PointLat"] > 90)
    if bool((bad_coords | out_of_range).any()):
        raise ValueError(f"Invalid coordinate rows: {int((bad_coords | out_of_range).sum())}")
    return df


def coordinates_for_raster(src: rasterio.io.DatasetReader, lon: np.ndarray, lat: np.ndarray) -> list[tuple[float, float]]:
    if src.crs is None:
        logging.warning("Raster has no CRS, assuming lon/lat coordinates: %s", src.name)
        return list(zip(lon, lat))
    if src.crs.to_epsg() == 4326:
        return list(zip(lon, lat))
    transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
    x, y = transformer.transform(lon, lat)
    return list(zip(x, y))


def sample_raster(variable: RasterVariable, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    values: list[float] = []
    with rasterio.open(variable.path) as src:
        coords = coordinates_for_raster(src, lon, lat)
        nodata = src.nodata
        for sampled in src.sample(coords, masked=True):
            value = sampled[0]
            if np.ma.is_masked(value):
                values.append(np.nan)
                continue
            value_float = float(value)
            if nodata is not None and math.isclose(value_float, float(nodata), rel_tol=0.0, abs_tol=1e-12):
                values.append(np.nan)
            elif not math.isfinite(value_float):
                values.append(np.nan)
            else:
                values.append(value_float)
    return np.asarray(values, dtype="float64")


def summarize_variable(name: str, values: np.ndarray) -> dict[str, Any]:
    finite = values[np.isfinite(values)]
    return {
        "variable": name,
        "valid_count": int(finite.size),
        "missing_count": int(values.size - finite.size),
        "min": float(np.min(finite)) if finite.size else np.nan,
        "max": float(np.max(finite)) if finite.size else np.nan,
        "mean": float(np.mean(finite)) if finite.size else np.nan,
        "std": float(np.std(finite)) if finite.size else np.nan,
    }


def write_report(summary: dict[str, Any], variable_summary: pd.DataFrame, extraction_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Stage04 当前 WorldClim 环境变量提取",
        "",
        f"- 生成时间: {now_iso()}",
        f"- 样本输入: {SAMPLES_CSV}",
        f"- 原始数据目录: {RAW_CURRENT_DIR}",
        f"- 解压目录: {PROCESSED_DIR}",
        f"- 输出样本表: {OUTPUT_CSV}",
        f"- 总样本数: {summary['sample_count']}",
        f"- 成功变量数: {summary['success_variables']}",
        f"- 失败变量数: {summary['failed_variables']}",
        f"- 解压失败数: {summary['failed_extractions']}",
        "",
        "## 变量统计",
        "",
        variable_summary.to_markdown(index=False),
        "",
        "## 解压状态",
        "",
    ]
    extract_df = pd.DataFrame(extraction_rows)
    if not extract_df.empty:
        lines.append(extract_df.groupby(["status"]).size().reset_index(name="count").to_markdown(index=False))
    else:
        lines.append("No extraction records.")
    atomic_write_text(REPORT_MD, "\n".join(lines))


def run(args: argparse.Namespace) -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        STATE_PATH,
        {
            "status": "running",
            "started_at": now_iso(),
            "message": "stage04 current WorldClim feature extraction started",
            "samples_csv": str(SAMPLES_CSV),
        },
    )
    write_status("running", "stage04 started")

    samples = read_samples(args.max_rows)
    logging.info("Loaded modeling samples: rows=%s columns=%s", len(samples), len(samples.columns))

    expected_columns = [f"wc_bio{i:02d}" for i in range(1, 20)] + ["wc_elev_m"]
    state = load_state()
    if (
        state.get("status") == "success"
        and not args.overwrite
        and args.max_rows is None
        and output_is_complete(len(samples), expected_columns)
    ):
        logging.info("Stage04 already completed. Use --overwrite to rerun.")
        write_status("success", "stage04 already completed", {"total_samples": len(samples)})
        return state

    extraction_rows = extract_current_worldclim(args.overwrite_extract or args.overwrite)
    failed_extractions = [row for row in extraction_rows if row.get("status") == "failed"]
    if failed_extractions:
        raise RuntimeError(f"Extraction failed for {len(failed_extractions)} raster files. See {VARIABLE_STATUS_CSV}")

    variables = list_raster_variables()
    lon = samples["PointLon"].to_numpy(dtype="float64")
    lat = samples["PointLat"].to_numpy(dtype="float64")

    variable_rows: list[dict[str, Any]] = [
        {
            "updated_at": now_iso(),
            "step": "sample",
            "item": var.name,
            "target": str(var.path),
            "status": "pending",
            "message": "",
        }
        for var in variables
    ]
    write_variable_status(extraction_rows + variable_rows)

    variable_summary_rows: list[dict[str, Any]] = []
    failed_variables = 0
    enriched = samples.copy()
    for row, variable in zip(variable_rows, variables):
        row.update({"updated_at": now_iso(), "status": "running"})
        write_variable_status(extraction_rows + variable_rows)
        logging.info("Sampling %s from %s", variable.name, variable.path)
        try:
            values = sample_raster(variable, lon, lat)
            if values.shape[0] != len(enriched):
                raise RuntimeError(f"Unexpected sampled value count for {variable.name}: {values.shape[0]} != {len(enriched)}")
            enriched[variable.name] = values
            summary = summarize_variable(variable.name, values)
            summary["source"] = variable.source
            summary["raster_path"] = str(variable.path)
            variable_summary_rows.append(summary)
            row.update(
                {
                    "updated_at": now_iso(),
                    "status": "success",
                    "message": "sampled",
                    "valid_count": summary["valid_count"],
                    "missing_count": summary["missing_count"],
                }
            )
        except Exception as exc:
            failed_variables += 1
            enriched[variable.name] = np.nan
            row.update(
                {
                    "updated_at": now_iso(),
                    "status": "failed",
                    "message": repr(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            logging.exception("Failed sampling variable %s", variable.name)
        write_variable_status(extraction_rows + variable_rows)

    variable_summary = pd.DataFrame(variable_summary_rows)
    atomic_write_csv(enriched, OUTPUT_CSV)
    atomic_write_csv(variable_summary, VARIABLE_SUMMARY_CSV)

    summary = {
        "status": "success" if failed_variables == 0 else "failed",
        "sample_count": int(len(enriched)),
        "input_columns": int(samples.shape[1]),
        "output_columns": int(enriched.shape[1]),
        "success_variables": int(len(variables) - failed_variables),
        "failed_variables": int(failed_variables),
        "failed_extractions": int(len(failed_extractions)),
        "outputs": {
            "samples_with_features_csv": str(OUTPUT_CSV),
            "variable_summary_csv": str(VARIABLE_SUMMARY_CSV),
            "report_md": str(REPORT_MD),
            "variable_status_csv": str(VARIABLE_STATUS_CSV),
        },
    }
    write_report(summary, variable_summary, extraction_rows)
    state_out = {
        **summary,
        "started_at": load_state().get("started_at", ""),
        "finished_at": now_iso(),
    }
    atomic_write_json(STATE_PATH, state_out)
    write_status(summary["status"], "stage04 completed" if failed_variables == 0 else "stage04 completed with failed variables", summary)
    if failed_variables:
        raise RuntimeError(f"Failed sampling {failed_variables} variables. See {VARIABLE_STATUS_CSV}")
    return state_out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="解压当前 WorldClim bio/elev，并为建模样本提取当前气候和高程变量。"
    )
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有 stage04 输出并重新提取。")
    parser.add_argument("--overwrite-extract", action="store_true", help="重新解压已存在且大小匹配的 GeoTIFF。")
    parser.add_argument("--max-rows", type=int, default=None, help="仅用于调试的小样本行数；正式运行不要设置。")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging()
    try:
        state = run(args)
        logging.info("Stage04 finished: %s", json.dumps(state, ensure_ascii=False))
        return 0 if state.get("status") == "success" else 1
    except Exception as exc:
        err = {
            "status": "failed",
            "failed_at": now_iso(),
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }
        previous = load_state()
        if previous.get("started_at"):
            err["started_at"] = previous["started_at"]
        atomic_write_json(STATE_PATH, err)
        write_status("failed", repr(exc))
        logging.exception("Stage04 failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
