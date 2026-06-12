# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import math
import os
import re
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


WORKSPACE = Path(r"C:\Users\linjingwu\Desktop\python")
SYNC_ROOT = Path(r"C:\Users\linjingwu\Desktop\跨电脑同步_工作站传输") / "stage17_future_suitability_workstation_20260530"
PROJECT_ROOT = SYNC_ROOT / "绿洲未来适宜区预测"

DEFAULT_STAGE12_STATUS = PROJECT_ROOT / "logs" / "stage12_region_tile_grid_projection_status.csv"
DEFAULT_STAGE17_SUMMARY = SYNC_ROOT / "家里电脑查看_STAGE17结果" / "stage17_constrained_suitability_summary.csv"
DEFAULT_CONSTRAINT_SUFFIX = "terrain_oasis100km_river100km_q1cms_up1000km2"

LOG_DIR = PROJECT_ROOT / "logs"
STATUS_CSV = LOG_DIR / "stage19_gee_landcover_esa_core_export_status.csv"
STATE_JSON = LOG_DIR / "stage19_gee_landcover_esa_core_export_state.json"
DEFAULT_DRIVE_FOLDER = "oasis_stage19_landcover_esa_core_30s"

RES_DEG = 1.0 / 120.0


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def safe_name(text: Any) -> str:
    out = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text))
    return out.strip("._-") or "unnamed"


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


def load_existing_status() -> pd.DataFrame:
    if not STATUS_CSV.exists():
        return pd.DataFrame()
    return pd.read_csv(STATUS_CSV)


def update_status_row(row: dict[str, Any]) -> None:
    row = dict(row)
    row["updated_at"] = now_iso()
    existing = load_existing_status()
    row_df = pd.DataFrame([row])
    key = str(row["export_key"])
    if existing.empty or "export_key" not in existing.columns:
        out = row_df
    else:
        existing = existing[existing["export_key"].astype(str) != key]
        out = pd.concat([existing, row_df], ignore_index=True)
    atomic_write_csv(STATUS_CSV, out)
    atomic_write_json(STATE_JSON, row)


def normalize_float(value: Any) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"non-finite coordinate: {value}")
    return out


def fallback_bbox(row: pd.Series) -> tuple[list[float], int, int]:
    left = normalize_float(row["min_lon"])
    bottom = normalize_float(row["min_lat"])
    right = normalize_float(row["max_lon"])
    top = normalize_float(row["max_lat"])
    width = max(1, int(round((right - left) / RES_DEG)))
    height = max(1, int(round((top - bottom) / RES_DEG)))
    return [left, bottom, left + width * RES_DEG, bottom + height * RES_DEG], width, height


def load_jobs(args: argparse.Namespace) -> pd.DataFrame:
    stage12_path = Path(args.stage12_status)
    stage17_path = Path(args.stage17_summary)
    if not stage12_path.exists():
        raise FileNotFoundError(f"Stage12 status not found: {stage12_path}")
    if not stage17_path.exists():
        raise FileNotFoundError(f"Stage17 summary not found: {stage17_path}")

    stage17 = pd.read_csv(stage17_path)
    stage17 = stage17[
        (stage17["constraint_suffix"].astype(str) == args.constraint_suffix)
        & (stage17["status"].astype(str).isin(["success", "skipped"]))
    ].copy()
    if args.tile_id:
        stage17 = stage17[stage17["tile_id"].astype(str) == args.tile_id].copy()
    key_cols = ["model_group", "gcm", "ssp", "period", "tile_id"]
    stage17["_keep"] = True

    stage12 = pd.read_csv(stage12_path)
    stage12 = stage12.merge(stage17[key_cols + ["_keep"]], on=key_cols, how="inner")
    if stage12.empty:
        raise ValueError("No Stage12 jobs matched Stage17 river100 summary.")

    records: list[dict[str, Any]] = []
    for _, row in stage12.iterrows():
        parsed = extract_stage10_json(str(row.get("message", ""))) or {}
        bbox = parsed.get("bbox")
        width = parsed.get("width")
        height = parsed.get("height")
        if not bbox or width is None or height is None:
            bbox, width, height = fallback_bbox(row)
        left, bottom, right, top = [float(x) for x in bbox]
        width = int(width)
        height = int(height)
        tile_id = str(row["tile_id"])
        prefix = f"LC_ESA_CORE_30s_{safe_name(tile_id)}"
        export_key = f"{safe_name(row['model_group'])}__{safe_name(row['gcm'])}__{safe_name(row['ssp'])}__{safe_name(row['period'])}__{safe_name(tile_id)}__esa_core_compatible_pct_30s"
        records.append(
            {
                "export_key": export_key,
                "file_prefix": prefix,
                "description": f"Stage19_ESA_CORE_30s_{safe_name(tile_id)}",
                "tile_id": tile_id,
                "model_group": row["model_group"],
                "gcm": row["gcm"],
                "ssp": row["ssp"],
                "period": row["period"],
                "left": left,
                "bottom": bottom,
                "right": right,
                "top": top,
                "width": width,
                "height": height,
                "crs_transform": [RES_DEG, 0, left, 0, -RES_DEG, top],
            }
        )
    jobs = pd.DataFrame(records).sort_values(["tile_id"]).reset_index(drop=True)
    if args.limit:
        jobs = jobs.head(int(args.limit)).copy()
    return jobs


def import_and_initialize_ee(project: str | None = None):
    try:
        import ee  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Earth Engine Python API is not installed in this Python environment. "
            "Install with `pip install earthengine-api`, then run `earthengine authenticate` once."
        ) from exc
    if project:
        ee.Initialize(project=project)
    else:
        ee.Initialize()
    return ee


def current_tasks_by_description(ee) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for task in ee.batch.Task.list():
        try:
            status = task.status()
        except Exception:
            continue
        desc = str(status.get("description", ""))
        if desc:
            out[desc] = status
    return out


def build_esa_core_pct_image(ee, transform: list[float]):
    esa = ee.ImageCollection("ESA/WorldCover/v100").first().select("Map")
    compatible = esa.eq(40).Or(esa.eq(50)).Or(esa.eq(60)).rename("compatible")
    pct = (
        compatible.unmask(0)
        .reduceResolution(reducer=ee.Reducer.mean(), maxPixels=20000)
        .multiply(100)
        .round()
        .toUint8()
        .rename("esa_core_compatible_pct")
        .reproject(crs="EPSG:4326", crsTransform=transform)
    )
    return pct


def submit_one(ee, row: pd.Series, args: argparse.Namespace, task_lookup: dict[str, dict[str, Any]]) -> dict[str, Any]:
    existing_task = task_lookup.get(str(row["description"]))
    if existing_task and not args.overwrite:
        state = existing_task.get("state", "UNKNOWN")
        return {
            **row.to_dict(),
            "status": "already_submitted",
            "task_id": existing_task.get("id"),
            "task_state": state,
            "drive_folder": args.drive_folder,
            "message": "task with same description already exists in Earth Engine task list",
        }

    geom = ee.Geometry.Rectangle(
        [row["left"], row["bottom"], row["right"], row["top"]],
        proj="EPSG:4326",
        geodesic=False,
    )
    transform = list(row["crs_transform"])
    image = build_esa_core_pct_image(ee, transform).clip(geom)
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=str(row["description"]),
        folder=args.drive_folder,
        fileNamePrefix=str(row["file_prefix"]),
        region=geom,
        crs="EPSG:4326",
        crsTransform=transform,
        maxPixels=args.max_pixels,
        fileFormat="GeoTIFF",
    )
    task.start()
    status = task.status()
    return {
        **row.to_dict(),
        "status": "submitted",
        "task_id": status.get("id"),
        "task_state": status.get("state"),
        "drive_folder": args.drive_folder,
        "message": "GEE export task submitted",
    }


def monitor_only(ee) -> None:
    lookup = current_tasks_by_description(ee)
    status = load_existing_status()
    rows: list[dict[str, Any]] = []
    for _, row in status.iterrows():
        desc = str(row.get("description", ""))
        task = lookup.get(desc, {})
        out = row.to_dict()
        if task:
            out["task_id"] = task.get("id", out.get("task_id"))
            out["task_state"] = task.get("state")
            out["message"] = task.get("error_message", out.get("message", ""))
            out["status"] = "completed" if task.get("state") == "COMPLETED" else out.get("status", "submitted")
        rows.append(out)
    if rows:
        atomic_write_csv(STATUS_CSV, pd.DataFrame(rows))
        atomic_write_json(STATE_JSON, {"status": "monitored", "updated_at": now_iso(), "rows": len(rows)})
    print(json.dumps({"status": "monitored", "tracked_rows": len(rows)}, ensure_ascii=False, indent=2))


def run(args: argparse.Namespace) -> dict[str, Any]:
    jobs = load_jobs(args)
    if args.dry_run:
        preview = jobs.head(10).to_dict(orient="records")
        summary = {
            "status": "dry_run",
            "matched_jobs": int(len(jobs)),
            "first_jobs": preview,
            "status_csv": str(STATUS_CSV),
            "drive_folder": args.drive_folder,
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return summary

    ee = import_and_initialize_ee(args.ee_project)
    if args.monitor_only:
        monitor_only(ee)
        return {"status": "monitored", "status_csv": str(STATUS_CSV)}

    task_lookup = current_tasks_by_description(ee)
    results: list[dict[str, Any]] = []
    for _, row in jobs.iterrows():
        try:
            result = submit_one(ee, row, args, task_lookup)
        except Exception as exc:
            result = {
                **row.to_dict(),
                "status": "failed",
                "task_id": "",
                "task_state": "",
                "drive_folder": args.drive_folder,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
        update_status_row(result)
        results.append(result)
    summary = {
        "status": "success" if not any(r["status"] == "failed" for r in results) else "partial_success",
        "generated_at": now_iso(),
        "matched_jobs": int(len(jobs)),
        "submitted": sum(1 for r in results if r["status"] == "submitted"),
        "already_submitted": sum(1 for r in results if r["status"] == "already_submitted"),
        "failed": sum(1 for r in results if r["status"] == "failed"),
        "status_csv": str(STATUS_CSV),
        "drive_folder": args.drive_folder,
        "ee_project": args.ee_project,
    }
    atomic_write_json(STATE_JSON, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submit GEE tile exports for ESA WorldCover core-compatible land-cover percentage at 30 arc-second grid.")
    parser.add_argument("--stage12-status", default=str(DEFAULT_STAGE12_STATUS))
    parser.add_argument("--stage17-summary", default=str(DEFAULT_STAGE17_SUMMARY))
    parser.add_argument("--constraint-suffix", default=DEFAULT_CONSTRAINT_SUFFIX)
    parser.add_argument("--tile-id", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--drive-folder", default=DEFAULT_DRIVE_FOLDER)
    parser.add_argument("--ee-project", default=os.environ.get("EARTHENGINE_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT"))
    parser.add_argument("--max-pixels", type=float, default=2e8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--monitor-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    try:
        run(parse_args())
        return 0
    except Exception as exc:
        failure = {"status": "failed", "updated_at": now_iso(), "message": str(exc), "traceback": traceback.format_exc()}
        atomic_write_json(STATE_JSON, failure)
        print(json.dumps(failure, ensure_ascii=False, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
