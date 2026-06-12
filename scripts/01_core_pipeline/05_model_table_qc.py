# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


WORKSPACE = Path(r"C:\Users\linjingwu\Desktop\python")
PROJECT_ROOT = WORKSPACE / "绿洲未来适宜区预测"
LOG_DIR = PROJECT_ROOT / "logs"
OUT_DIR = PROJECT_ROOT / "outputs" / "stage05_current_worldclim_model_ready"

INPUT_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "stage04_current_worldclim_features"
    / "modeling_samples_with_current_worldclim.csv"
)

LOG_PATH = LOG_DIR / "stage05_current_worldclim_model_ready_qc.log"
STATE_PATH = LOG_DIR / "stage05_current_worldclim_model_ready_qc_state.json"
STATUS_CSV = LOG_DIR / "stage05_current_worldclim_model_ready_qc_status.csv"

COMPLETE_CASES_CSV = OUT_DIR / "modeling_samples_current_worldclim_complete_cases.csv"
MISSING_ROWS_CSV = OUT_DIR / "modeling_samples_current_worldclim_missing_rows.csv"
FOLD_BALANCE_CSV = OUT_DIR / "current_worldclim_complete_case_fold_balance.csv"
REGION_MISSING_CSV = OUT_DIR / "current_worldclim_missing_by_region_sample_type.csv"
VARIABLE_MISSING_CSV = OUT_DIR / "current_worldclim_missing_by_variable.csv"
REPORT_MD = OUT_DIR / "stage05_current_worldclim_model_ready_qc_report.md"


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


def variable_columns(df: pd.DataFrame) -> list[str]:
    cols = [col for col in df.columns if col.startswith("wc_bio") or col == "wc_elev_m"]
    expected = [f"wc_bio{i:02d}" for i in range(1, 20)] + ["wc_elev_m"]
    missing = [col for col in expected if col not in cols]
    if missing:
        raise ValueError(f"Missing expected WorldClim columns: {missing}")
    return expected


def fold_balance(df: pd.DataFrame) -> pd.DataFrame:
    balance = (
        df.groupby(["SpatialCVFold", "Response"], dropna=False)
        .size()
        .unstack(fill_value=0)
        .rename(columns={0: "Background", 1: "Presence"})
        .reset_index()
    )
    for col in ["Presence", "Background"]:
        if col not in balance.columns:
            balance[col] = 0
    balance["Total"] = balance["Presence"] + balance["Background"]
    balance["PresenceRate"] = balance["Presence"] / balance["Total"]
    return balance[["SpatialCVFold", "Presence", "Background", "Total", "PresenceRate"]]


def missing_by_region(missing_df: pd.DataFrame) -> pd.DataFrame:
    if missing_df.empty:
        return pd.DataFrame(columns=["Region", "SampleType", "Response", "MissingRows"])
    return (
        missing_df.groupby(["Region", "SampleType", "Response"], dropna=False)
        .size()
        .reset_index(name="MissingRows")
        .sort_values(["MissingRows", "Region", "SampleType"], ascending=[False, True, True])
    )


def write_report(summary: dict[str, Any], fold_df: pd.DataFrame, region_missing_df: pd.DataFrame, variable_missing_df: pd.DataFrame) -> None:
    lines = [
        "# Stage05 当前 WorldClim 建模表质量检查",
        "",
        f"- 生成时间: {now_iso()}",
        f"- 输入表: {INPUT_CSV}",
        f"- 完整案例输出: {COMPLETE_CASES_CSV}",
        f"- 缺失样本输出: {MISSING_ROWS_CSV}",
        "",
        "## 总览",
        "",
        f"- 原始样本数: {summary['total_rows']}",
        f"- 完整案例样本数: {summary['complete_case_rows']}",
        f"- 缺失样本数: {summary['missing_rows']}",
        f"- 变量数: {summary['variable_count']}",
        f"- Presence 完整案例: {summary['complete_presence_rows']}",
        f"- Background 完整案例: {summary['complete_background_rows']}",
        "",
        "## 完整案例 Fold 平衡",
        "",
        fold_df.to_markdown(index=False),
        "",
        "## 缺失变量统计",
        "",
        variable_missing_df.to_markdown(index=False),
        "",
        "## 缺失样本分组",
        "",
        region_missing_df.to_markdown(index=False) if not region_missing_df.empty else "无缺失样本。",
    ]
    atomic_write_text(REPORT_MD, "\n".join(lines))


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV missing: {INPUT_CSV}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        STATE_PATH,
        {
            "status": "running",
            "started_at": now_iso(),
            "message": "stage05 current WorldClim model-ready QC started",
            "input_csv": str(INPUT_CSV),
        },
    )
    write_status("running", "stage05 QC started")

    df = pd.read_csv(INPUT_CSV, low_memory=False)
    vars_ = variable_columns(df)
    missing_mask = df[vars_].isna().any(axis=1)
    complete = df.loc[~missing_mask].copy()
    missing = df.loc[missing_mask].copy()

    if complete.empty:
        raise RuntimeError("No complete-case samples remain after WorldClim missing-value filter.")

    variable_missing_df = (
        df[vars_]
        .isna()
        .sum()
        .reset_index()
        .rename(columns={"index": "Variable", 0: "MissingRows"})
    )
    variable_missing_df["MissingRate"] = variable_missing_df["MissingRows"] / len(df)
    region_missing_df = missing_by_region(missing)
    fold_df = fold_balance(complete)

    atomic_write_csv(complete, COMPLETE_CASES_CSV)
    atomic_write_csv(missing, MISSING_ROWS_CSV)
    atomic_write_csv(fold_df, FOLD_BALANCE_CSV)
    atomic_write_csv(region_missing_df, REGION_MISSING_CSV)
    atomic_write_csv(variable_missing_df, VARIABLE_MISSING_CSV)

    summary = {
        "status": "success",
        "total_rows": int(len(df)),
        "complete_case_rows": int(len(complete)),
        "missing_rows": int(len(missing)),
        "variable_count": int(len(vars_)),
        "complete_presence_rows": int((complete["Response"] == 1).sum()),
        "complete_background_rows": int((complete["Response"] == 0).sum()),
        "outputs": {
            "complete_cases_csv": str(COMPLETE_CASES_CSV),
            "missing_rows_csv": str(MISSING_ROWS_CSV),
            "fold_balance_csv": str(FOLD_BALANCE_CSV),
            "region_missing_csv": str(REGION_MISSING_CSV),
            "variable_missing_csv": str(VARIABLE_MISSING_CSV),
            "report_md": str(REPORT_MD),
        },
    }

    write_report(summary, fold_df, region_missing_df, variable_missing_df)
    state = {
        **summary,
        "started_at": json.loads(STATE_PATH.read_text(encoding="utf-8")).get("started_at", ""),
        "finished_at": now_iso(),
    }
    atomic_write_json(STATE_PATH, state)
    write_status("success", "stage05 QC completed", summary)
    logging.info("Stage05 QC completed: %s", json.dumps(summary, ensure_ascii=False))
    return state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检查当前 WorldClim 建模表缺失值，并输出完整案例训练表。")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging()
    try:
        run(args)
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
        logging.exception("Stage05 QC failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
