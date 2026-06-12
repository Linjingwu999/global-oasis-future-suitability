# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer
from sklearn.model_selection import StratifiedGroupKFold
from shapely.geometry import Point


WORKSPACE = Path(r"C:\Users\linjingwu\Desktop\python")
PROJECT_ROOT = WORKSPACE / "绿洲未来适宜区预测"
LOG_DIR = PROJECT_ROOT / "logs"
OUT_DIR = PROJECT_ROOT / "outputs" / "stage03_modeling_samples_spatial_cv"

PRESENCE_CSV = PROJECT_ROOT / "outputs" / "stage01_presence_samples" / "presence_points_combined.csv"
BACKGROUND_CSV = PROJECT_ROOT / "outputs" / "stage02_background_points" / "background_points_combined.csv"

LOG_PATH = LOG_DIR / "stage03_modeling_samples_spatial_cv.log"
STATE_PATH = LOG_DIR / "stage03_modeling_samples_spatial_cv_state.json"
STATUS_CSV = LOG_DIR / "stage03_modeling_samples_spatial_cv_status.csv"


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


def write_status(status: str, message: str, extra: dict[str, Any] | None = None) -> None:
    row = {"updated_at": now_iso(), "status": status, "message": message}
    if extra:
        row.update(extra)
    STATUS_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(STATUS_CSV, index=False, encoding="utf-8-sig")


def stable_int(value: str) -> int:
    return int(hashlib.md5(value.encode("utf-8")).hexdigest()[:12], 16)


def read_input_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{label} input missing: {path}")
    df = pd.read_csv(path)
    required = {"PointLon", "PointLat"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")
    logging.info("%s rows=%s columns=%s", label, len(df), len(df.columns))
    return df


def normalize_presence(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.insert(0, "SampleID", [f"PRES_{i:08d}" for i in range(1, len(out) + 1)])
    out.insert(1, "Response", 1)
    out.insert(2, "SampleType", "presence")
    out["Region"] = out.get("SourceRegion", "")
    out["DrylandStratum"] = ""
    out["OriginalRow"] = np.arange(1, len(out) + 1)
    return out


def normalize_background(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.insert(0, "SampleID", [f"BACK_{i:08d}" for i in range(1, len(out) + 1)])
    out.insert(1, "Response", 0)
    out["SampleType"] = "background"
    out["Region"] = ""
    out["PatchID"] = ""
    out["AreaStratum"] = ""
    out["PatchAreaKm2"] = np.nan
    out["OriginalRow"] = np.arange(1, len(out) + 1)
    return out


def keep_modeling_columns(df: pd.DataFrame) -> pd.DataFrame:
    preferred = [
        "SampleID",
        "Response",
        "SampleType",
        "Region",
        "DrylandStratum",
        "PatchID",
        "SourceFeatureIndex",
        "DrylandFeatureIndex",
        "AreaStratum",
        "PatchAreaKm2",
        "OasisID",
        "ContinentI",
        "CountryID",
        "BasinID",
        "AreaID",
        "Area",
        "Perimeter",
        "PointLon",
        "PointLat",
        "OriginalRow",
        "geometry_wkt",
    ]
    for col in preferred:
        if col not in df.columns:
            df[col] = np.nan
    return df[preferred].copy()


def add_projected_coordinates_and_blocks(df: pd.DataFrame, block_size_km: float) -> pd.DataFrame:
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:6933", always_xy=True)
    x, y = transformer.transform(df["PointLon"].to_numpy(), df["PointLat"].to_numpy())
    block_size_m = block_size_km * 1000.0
    out = df.copy()
    out["X_EPSG6933_m"] = x
    out["Y_EPSG6933_m"] = y
    out["SpatialBlockX"] = np.floor(out["X_EPSG6933_m"] / block_size_m).astype("int64")
    out["SpatialBlockY"] = np.floor(out["Y_EPSG6933_m"] / block_size_m).astype("int64")
    out["SpatialBlockID"] = out["SpatialBlockX"].astype(str) + "_" + out["SpatialBlockY"].astype(str)
    out["SpatialBlockSizeKm"] = block_size_km
    return out


def assign_spatial_folds(df: pd.DataFrame, n_folds: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    block_stats = (
        df.groupby("SpatialBlockID", dropna=False)
        .agg(
            BlockX=("SpatialBlockX", "first"),
            BlockY=("SpatialBlockY", "first"),
            TotalSamples=("SampleID", "size"),
            PresenceSamples=("Response", "sum"),
        )
        .reset_index()
    )
    block_stats["BackgroundSamples"] = block_stats["TotalSamples"] - block_stats["PresenceSamples"]
    out = df.copy()
    out["SpatialCVFold"] = 0
    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    y = out["Response"].astype(int).to_numpy()
    groups = out["SpatialBlockID"].astype(str).to_numpy()
    dummy_x = np.zeros((len(out), 1))
    for fold_idx, (_, test_idx) in enumerate(splitter.split(dummy_x, y, groups), start=1):
        out.loc[out.index[test_idx], "SpatialCVFold"] = fold_idx
    if int((out["SpatialCVFold"] == 0).sum()) != 0:
        raise RuntimeError("Some samples were not assigned to a spatial CV fold.")

    block_fold_counts = (
        out.groupby("SpatialBlockID")["SpatialCVFold"].nunique().reset_index(name="FoldCount")
    )
    split_blocks = int((block_fold_counts["FoldCount"] > 1).sum())
    if split_blocks:
        raise RuntimeError(f"{split_blocks} spatial blocks were split across folds.")

    block_fold = out.groupby("SpatialBlockID")["SpatialCVFold"].first().reset_index()
    block_stats = block_stats.merge(block_fold, on="SpatialBlockID", how="left")
    return out, block_stats


def quality_summary(df: pd.DataFrame, block_stats: pd.DataFrame) -> dict[str, Any]:
    coord_missing = int(df[["PointLon", "PointLat"]].isna().any(axis=1).sum())
    coord_out_of_range = int(((df["PointLon"] < -180) | (df["PointLon"] > 180) | (df["PointLat"] < -90) | (df["PointLat"] > 90)).sum())
    exact_duplicate_coords = int(df.duplicated(subset=["PointLon", "PointLat", "Response"]).sum())
    fold_counts = (
        df.groupby(["SpatialCVFold", "Response"])
        .size()
        .unstack(fill_value=0)
        .rename(columns={0: "Background", 1: "Presence"})
        .reset_index()
    )
    for col in ["Presence", "Background"]:
        if col not in fold_counts.columns:
            fold_counts[col] = 0
    fold_counts["Total"] = fold_counts["Presence"] + fold_counts["Background"]
    return {
        "total_samples": int(len(df)),
        "presence_samples": int((df["Response"] == 1).sum()),
        "background_samples": int((df["Response"] == 0).sum()),
        "coordinate_missing_rows": coord_missing,
        "coordinate_out_of_range_rows": coord_out_of_range,
        "exact_duplicate_coordinate_rows_by_response": exact_duplicate_coords,
        "spatial_block_count": int(block_stats["SpatialBlockID"].nunique()),
        "fold_summary": fold_counts.to_dict(orient="records"),
    }


def write_markdown_report(summary: dict[str, Any], args: argparse.Namespace, outputs: dict[str, str]) -> None:
    lines = [
        "# Stage03 建模样本与空间分块交叉验证",
        "",
        f"- 生成时间: {now_iso()}",
        f"- presence 输入: {PRESENCE_CSV}",
        f"- background 输入: {BACKGROUND_CSV}",
        f"- 空间块大小: {args.block_size_km} km",
        f"- 折数: {args.n_folds}",
        f"- 随机性: 无随机分配；同一空间块不会被拆到不同 fold",
        "",
        "## 样本数量",
        "",
        f"- 总样本: {summary['total_samples']}",
        f"- Presence: {summary['presence_samples']}",
        f"- Background: {summary['background_samples']}",
        f"- 空间块数量: {summary['spatial_block_count']}",
        "",
        "## 质量检查",
        "",
        f"- 坐标缺失行: {summary['coordinate_missing_rows']}",
        f"- 坐标超范围行: {summary['coordinate_out_of_range_rows']}",
        f"- 同类别精确重复坐标行: {summary['exact_duplicate_coordinate_rows_by_response']}",
        "",
        "## Fold 样本平衡",
        "",
        pd.DataFrame(summary["fold_summary"]).to_markdown(index=False),
        "",
        "## 输出文件",
        "",
    ]
    for key, value in outputs.items():
        lines.append(f"- {key}: {value}")
    atomic_write_text(OUT_DIR / "stage03_modeling_samples_spatial_cv_report.md", "\n".join(lines))


def run(args: argparse.Namespace) -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        STATE_PATH,
        {
            "status": "running",
            "started_at": now_iso(),
            "message": "stage03 modeling sample construction started",
        },
    )
    write_status("running", "stage03 started")

    presence = normalize_presence(read_input_csv(PRESENCE_CSV, "presence"))
    background = normalize_background(read_input_csv(BACKGROUND_CSV, "background"))
    modeling = pd.concat([keep_modeling_columns(presence), keep_modeling_columns(background)], ignore_index=True)
    modeling["PointLon"] = pd.to_numeric(modeling["PointLon"], errors="coerce")
    modeling["PointLat"] = pd.to_numeric(modeling["PointLat"], errors="coerce")
    modeling = add_projected_coordinates_and_blocks(modeling, args.block_size_km)
    modeling, block_stats = assign_spatial_folds(modeling, args.n_folds, args.seed)

    summary = quality_summary(modeling, block_stats)
    outputs = {
        "samples_csv": str(OUT_DIR / "modeling_samples_with_spatial_folds.csv"),
        "samples_gpkg": str(OUT_DIR / "modeling_samples_with_spatial_folds.gpkg"),
        "block_summary_csv": str(OUT_DIR / "spatial_block_summary.csv"),
        "fold_summary_csv": str(OUT_DIR / "fold_balance_summary.csv"),
        "report_md": str(OUT_DIR / "stage03_modeling_samples_spatial_cv_report.md"),
    }

    samples_csv = Path(outputs["samples_csv"])
    samples_gpkg = Path(outputs["samples_gpkg"])
    block_csv = Path(outputs["block_summary_csv"])
    fold_csv = Path(outputs["fold_summary_csv"])

    modeling.to_csv(samples_csv, index=False, encoding="utf-8-sig")
    block_stats.to_csv(block_csv, index=False, encoding="utf-8-sig")
    fold_df = pd.DataFrame(summary["fold_summary"])
    fold_df.to_csv(fold_csv, index=False, encoding="utf-8-sig")

    geometry = [Point(xy) for xy in zip(modeling["PointLon"], modeling["PointLat"])]
    gdf = gpd.GeoDataFrame(modeling.drop(columns=["geometry_wkt"], errors="ignore"), geometry=geometry, crs="EPSG:4326")
    gdf.to_file(samples_gpkg, layer="modeling_samples_spatial_cv", driver="GPKG")

    with pd.ExcelWriter(OUT_DIR / "stage03_summary_tables.xlsx", engine="openpyxl") as writer:
        fold_df.to_excel(writer, sheet_name="fold_balance", index=False)
        block_stats.to_excel(writer, sheet_name="spatial_blocks", index=False)

    write_markdown_report(summary, args, outputs)

    state = {
        "status": "success",
        "started_at": read_state_started_at(),
        "finished_at": now_iso(),
        "summary": summary,
        "outputs": outputs,
    }
    atomic_write_json(STATE_PATH, state)
    write_status("success", "stage03 completed", {"total_samples": summary["total_samples"]})
    return state


def read_state_started_at() -> str:
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8")).get("started_at", "")
    except Exception:
        return ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="合并 presence/background 样本，并生成空间分块交叉验证 fold。")
    parser.add_argument("--block-size-km", type=float, default=500.0)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260529)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging()
    if STATE_PATH.exists() and not args.overwrite:
        state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        if state.get("status") == "success":
            logging.info("Stage03 already completed. Use --overwrite to rerun.")
            return 0
    try:
        state = run(args)
        logging.info("Stage03 completed: %s", json.dumps(state.get("summary", {}), ensure_ascii=False))
        return 0
    except Exception as exc:
        err = {
            "status": "failed",
            "failed_at": now_iso(),
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }
        atomic_write_json(STATE_PATH, err)
        write_status("failed", repr(exc))
        logging.exception("Stage03 failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
