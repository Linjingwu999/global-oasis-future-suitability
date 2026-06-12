# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import gzip
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
import numpy as np
import pandas as pd

PROJ_DIR = Path(r"C:\Users\linjingwu\anaconda3\Library\share\proj")
if PROJ_DIR.exists():
    os.environ["PROJ_LIB"] = str(PROJ_DIR)

import rasterio
from rasterio.transform import rowcol


WORKSPACE = Path(r"C:\Users\linjingwu\Desktop\python")
PROJECT_ROOT = WORKSPACE / "绿洲未来适宜区预测"
DATA_ROOT = Path(r"D:\绿洲未来适宜区预测数据")

FUTURE_DIR = DATA_ROOT / "raw" / "worldclim" / "future_30s"
INPUT_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "stage05_current_worldclim_model_ready"
    / "modeling_samples_current_worldclim_complete_cases.csv"
)
MODEL_DIR = PROJECT_ROOT / "outputs" / "stage06_current_worldclim_baseline_models" / "models"

OUT_DIR = PROJECT_ROOT / "outputs" / "stage07_future_worldclim_sample_predictions"
PRED_DIR = OUT_DIR / "scenario_predictions"
LOG_DIR = PROJECT_ROOT / "logs"

LOG_PATH = LOG_DIR / "stage07_future_worldclim_sample_predictions.log"
STATE_PATH = LOG_DIR / "stage07_future_worldclim_sample_predictions_state.json"
STATUS_CSV = LOG_DIR / "stage07_future_worldclim_sample_predictions_status.csv"
SCENARIO_STATUS_CSV = LOG_DIR / "stage07_future_worldclim_scenario_status.csv"

SUMMARY_CSV = OUT_DIR / "future_worldclim_sample_prediction_summary.csv"
SCENARIO_SUMMARY_CSV = OUT_DIR / "future_worldclim_sample_prediction_scenario_summary.csv"
REPORT_MD = OUT_DIR / "stage07_future_worldclim_sample_predictions_report.md"

FUTURE_RE = re.compile(
    r"wc2\.1_30s_bioc_(?P<gcm>.+?)_(?P<ssp>ssp\d+)_(?P<period>\d{4}-\d{4})\.tif$",
    re.IGNORECASE,
)


def safe_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return safe.strip("._-") or "model"


def use_named_outputs(label: str, separate_prediction_dir: bool = False) -> None:
    global PRED_DIR, LOG_PATH, STATE_PATH, STATUS_CSV, SCENARIO_STATUS_CSV, SUMMARY_CSV, SCENARIO_SUMMARY_CSV, REPORT_MD
    suffix = safe_name(label)
    if separate_prediction_dir:
        PRED_DIR = OUT_DIR / "scenario_predictions" / suffix
    LOG_PATH = LOG_DIR / f"stage07_future_worldclim_sample_predictions_{suffix}.log"
    STATE_PATH = LOG_DIR / f"stage07_future_worldclim_sample_predictions_{suffix}_state.json"
    STATUS_CSV = LOG_DIR / f"stage07_future_worldclim_sample_predictions_{suffix}_status.csv"
    SCENARIO_STATUS_CSV = LOG_DIR / f"stage07_future_worldclim_scenario_status_{suffix}.csv"
    SUMMARY_CSV = OUT_DIR / f"future_worldclim_sample_prediction_summary_{suffix}.csv"
    SCENARIO_SUMMARY_CSV = OUT_DIR / f"future_worldclim_sample_prediction_scenario_summary_{suffix}.csv"
    REPORT_MD = OUT_DIR / f"stage07_future_worldclim_sample_predictions_report_{suffix}.md"


def use_model_specific_outputs(model_group: str) -> None:
    use_named_outputs(model_group)


def configure_paths(args: argparse.Namespace) -> None:
    global MODEL_DIR, OUT_DIR, PRED_DIR
    MODEL_DIR = Path(args.model_dir)
    OUT_DIR = Path(args.output_dir)
    PRED_DIR = OUT_DIR / "scenario_predictions"
    if args.run_label:
        use_named_outputs(args.run_label, separate_prediction_dir=True)
    elif args.model_specific_outputs:
        use_model_specific_outputs(args.model_group)


@dataclass(frozen=True)
class FutureScenario:
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


def write_scenario_status(rows: list[dict[str, Any]]) -> None:
    if rows:
        atomic_write_csv(pd.DataFrame(rows), SCENARIO_STATUS_CSV)


def feature_columns() -> list[str]:
    return [f"wc_bio{i:02d}" for i in range(1, 20)] + ["wc_elev_m"]


def load_samples() -> pd.DataFrame:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input sample table missing: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV, low_memory=False)
    required = {"SampleID", "Response", "SampleType", "Region", "SpatialCVFold", "PointLon", "PointLat", "wc_elev_m"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Input table missing required columns: {missing}")
    df["PointLon"] = pd.to_numeric(df["PointLon"], errors="coerce")
    df["PointLat"] = pd.to_numeric(df["PointLat"], errors="coerce")
    if df[["PointLon", "PointLat", "wc_elev_m"]].isna().any(axis=1).any():
        raise ValueError("Input sample table still has missing coordinates or elevation.")
    return df


def discover_scenarios() -> list[FutureScenario]:
    if not FUTURE_DIR.exists():
        raise FileNotFoundError(f"Future WorldClim directory missing: {FUTURE_DIR}")
    scenarios: list[FutureScenario] = []
    for path in sorted(FUTURE_DIR.rglob("*.tif")):
        match = FUTURE_RE.match(path.name)
        if not match:
            continue
        scenarios.append(
            FutureScenario(
                gcm=match.group("gcm"),
                ssp=match.group("ssp"),
                period=match.group("period"),
                path=path,
            )
        )
    if not scenarios:
        raise FileNotFoundError(f"No future WorldClim tif files found under {FUTURE_DIR}")
    return scenarios


def filter_scenarios(scenarios: list[FutureScenario], args: argparse.Namespace) -> list[FutureScenario]:
    out = scenarios
    if args.gcm:
        keep = set(args.gcm)
        out = [s for s in out if s.gcm in keep]
    if args.ssp:
        keep = set(args.ssp)
        out = [s for s in out if s.ssp in keep]
    if args.period:
        keep = set(args.period)
        out = [s for s in out if s.period in keep]
    if args.limit_scenarios:
        out = out[: args.limit_scenarios]
    return out


def load_model_group(model_group: str) -> tuple[list[Any], list[str], float]:
    files = sorted(MODEL_DIR.glob(f"{model_group}_fold*.joblib"))
    if not files:
        raise FileNotFoundError(f"No model files found for model_group={model_group}: {MODEL_DIR}")
    models: list[Any] = []
    thresholds: list[float] = []
    expected_features: list[str] | None = None
    for file in files:
        obj = joblib.load(file)
        models.append(obj["model"])
        thresholds.append(float(obj["threshold"]))
        features = list(obj["features"])
        if expected_features is None:
            expected_features = features
        elif features != expected_features:
            raise ValueError(f"Feature mismatch in model file: {file}")
    return models, expected_features or feature_columns(), float(np.mean(thresholds))


def prediction_output_path(scenario: FutureScenario, model_group: str) -> Path:
    return PRED_DIR / model_group / f"{scenario.key}_sample_predictions.csv.gz"


def output_exists(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with gzip.open(path, "rt", encoding="utf-8-sig") as f:
            header = f.readline()
        return "SampleID" in header and "probability" in header
    except Exception:
        return False


def sample_future_bioclim(path: Path, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    with rasterio.open(path) as src:
        if src.count != 19:
            raise ValueError(f"Expected 19 bioclim bands, got {src.count}: {path}")
        rows, cols = rowcol(src.transform, lon, lat, op=np.floor)
        rows = np.asarray(rows, dtype="int64")
        cols = np.asarray(cols, dtype="int64")
        inside = (rows >= 0) & (rows < src.height) & (cols >= 0) & (cols < src.width)
        order = np.lexsort((cols, rows))
        coords = list(zip(lon[order], lat[order]))
        values_sorted = np.empty((len(coords), 19), dtype="float32")
        for idx, sampled in enumerate(src.sample(coords, masked=True)):
            if np.ma.is_masked(sampled):
                arr = np.full(19, np.nan, dtype="float32")
            else:
                arr = np.asarray(sampled, dtype="float32")
                arr[~np.isfinite(arr)] = np.nan
            values_sorted[idx, :] = arr
        values = np.empty_like(values_sorted)
        values[order, :] = values_sorted
        values[~inside, :] = np.nan
    return values


def assemble_future_feature_matrix(future_bio: np.ndarray, elev: np.ndarray, features: list[str]) -> np.ndarray:
    if not features:
        raise ValueError("Model feature list is empty.")
    columns: list[np.ndarray] = []
    for feature in features:
        if feature == "wc_elev_m":
            columns.append(elev[:, 0])
            continue
        match = re.fullmatch(r"wc_bio(\d{2})", feature)
        if not match:
            raise ValueError(f"Unsupported model feature for future prediction: {feature}")
        band_index = int(match.group(1)) - 1
        if band_index < 0 or band_index >= future_bio.shape[1]:
            raise ValueError(f"Feature {feature} is outside future bioclim band range.")
        columns.append(future_bio[:, band_index])
    return np.column_stack(columns).astype("float32", copy=False)


def predict_ensemble(models: list[Any], x: np.ndarray) -> np.ndarray:
    probs = np.zeros((x.shape[0], len(models)), dtype="float32")
    for idx, model in enumerate(models):
        probs[:, idx] = model.predict_proba(x)[:, 1].astype("float32")
    return probs.mean(axis=1)


def summarize_predictions(pred: pd.DataFrame, scenario: FutureScenario, model_group: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    group_sets = [
        ("overall", []),
        ("response", ["Response"]),
        ("sample_type", ["SampleType"]),
        ("region_response", ["Region", "Response"]),
        ("fold_response", ["SpatialCVFold", "Response"]),
    ]
    for group_name, cols in group_sets:
        grouped = [((), pred)] if not cols else pred.groupby(cols, dropna=False)
        for keys, group in grouped:
            if not isinstance(keys, tuple):
                keys = (keys,)
            row: dict[str, Any] = {
                "gcm": scenario.gcm,
                "ssp": scenario.ssp,
                "period": scenario.period,
                "model_group": model_group,
                "group_type": group_name,
                "n": int(len(group)),
                "mean_probability": float(group["probability"].mean()),
                "median_probability": float(group["probability"].median()),
                "p10_probability": float(group["probability"].quantile(0.10)),
                "p90_probability": float(group["probability"].quantile(0.90)),
                "suitable_rate": float(group["predicted_suitable"].mean()),
            }
            for col, value in zip(cols, keys):
                row[col] = value
            rows.append(row)
    return rows


def write_report(summary: dict[str, Any], scenario_summary: pd.DataFrame) -> None:
    overall = scenario_summary[scenario_summary["group_type"] == "overall"].copy()
    lines = [
        "# Stage07 未来 WorldClim 样本点适宜性预测",
        "",
        f"- 生成时间: {now_iso()}",
        f"- 输入样本: {INPUT_CSV}",
        f"- 模型组: {summary['model_group']}",
        f"- 阈值: {summary['threshold']}",
        f"- 计划情景数: {summary['total_scenarios']}",
        f"- 成功情景数: {summary['success_scenarios']}",
        f"- 失败情景数: {summary['failed_scenarios']}",
        f"- 跳过情景数: {summary['skipped_scenarios']}",
        "",
        "## Overall 情景摘要",
        "",
        overall.sort_values(["gcm", "ssp", "period"]).to_markdown(index=False) if not overall.empty else "暂无摘要。",
        "",
        "## 输出文件",
        "",
    ]
    for key, value in summary["outputs"].items():
        lines.append(f"- {key}: {value}")
    atomic_write_text(REPORT_MD, "\n".join(lines))


def run(args: argparse.Namespace) -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        STATE_PATH,
        {
            "status": "running",
            "started_at": now_iso(),
            "message": "stage07 future WorldClim sample prediction started",
            "model_group": args.model_group,
        },
    )
    write_status("running", "stage07 started", {"model_group": args.model_group})

    samples = load_samples()
    scenarios = filter_scenarios(discover_scenarios(), args)
    if not scenarios:
        raise ValueError("No future scenarios left after filters.")
    models, model_features, threshold = load_model_group(args.model_group)

    base_cols = ["SampleID", "Response", "SampleType", "Region", "SpatialCVFold", "PointLon", "PointLat"]
    lon = samples["PointLon"].to_numpy(dtype="float64")
    lat = samples["PointLat"].to_numpy(dtype="float64")
    elev = samples["wc_elev_m"].to_numpy(dtype="float32").reshape(-1, 1)

    status_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    success = failed = skipped = 0

    for scenario in scenarios:
        out_path = prediction_output_path(scenario, args.model_group)
        row = {
            "updated_at": now_iso(),
            "gcm": scenario.gcm,
            "ssp": scenario.ssp,
            "period": scenario.period,
            "model_group": args.model_group,
            "source_tif": str(scenario.path),
            "output_csv_gz": str(out_path),
            "status": "running",
            "message": "",
        }
        status_rows.append(row)
        write_scenario_status(status_rows)

        try:
            if output_exists(out_path) and not args.overwrite:
                skipped += 1
                row.update({"updated_at": now_iso(), "status": "skipped", "message": "existing output"})
                pred = pd.read_csv(out_path, low_memory=False)
                summary_rows.extend(summarize_predictions(pred, scenario, args.model_group))
                write_scenario_status(status_rows)
                continue

            logging.info("Sampling and predicting %s", scenario.path)
            future_bio = sample_future_bioclim(scenario.path, lon, lat)
            x = assemble_future_feature_matrix(future_bio, elev, model_features)
            valid = np.isfinite(x).all(axis=1)
            probability = np.full(len(samples), np.nan, dtype="float32")
            if valid.any():
                probability[valid] = predict_ensemble(models, x[valid])
            pred = samples[base_cols].copy()
            pred.insert(0, "gcm", scenario.gcm)
            pred.insert(1, "ssp", scenario.ssp)
            pred.insert(2, "period", scenario.period)
            pred.insert(3, "model_group", args.model_group)
            pred["probability"] = probability
            pred["threshold"] = threshold
            pred["predicted_suitable"] = (pred["probability"] >= threshold).astype("Int64")
            pred.loc[pred["probability"].isna(), "predicted_suitable"] = pd.NA
            pred["valid_future_features"] = valid

            out_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
            pred.to_csv(tmp_path, index=False, encoding="utf-8-sig", compression="gzip")
            tmp_path.replace(out_path)

            scenario_summary = summarize_predictions(pred.dropna(subset=["probability"]), scenario, args.model_group)
            summary_rows.extend(scenario_summary)
            success += 1
            row.update(
                {
                    "updated_at": now_iso(),
                    "status": "success",
                    "message": "predicted",
                    "n_samples": int(len(pred)),
                    "valid_samples": int(valid.sum()),
                    "missing_samples": int((~valid).sum()),
                }
            )
        except Exception as exc:
            failed += 1
            row.update(
                {
                    "updated_at": now_iso(),
                    "status": "failed",
                    "message": repr(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            logging.exception("Failed scenario %s", scenario.key)
        write_scenario_status(status_rows)

    scenario_summary_df = pd.DataFrame(summary_rows)
    atomic_write_csv(pd.DataFrame(status_rows), SCENARIO_STATUS_CSV)
    atomic_write_csv(scenario_summary_df, SCENARIO_SUMMARY_CSV)
    atomic_write_csv(
        scenario_summary_df[scenario_summary_df.get("group_type", pd.Series(dtype=str)).eq("overall")]
        if not scenario_summary_df.empty
        else pd.DataFrame(),
        SUMMARY_CSV,
    )

    state_status = "success" if failed == 0 else ("partial_success" if success or skipped else "failed")
    state = {
        "status": state_status,
        "model_group": args.model_group,
        "threshold": threshold,
        "sample_count": int(len(samples)),
        "total_scenarios": int(len(scenarios)),
        "success_scenarios": int(success),
        "failed_scenarios": int(failed),
        "skipped_scenarios": int(skipped),
        "outputs": {
            "summary_csv": str(SUMMARY_CSV),
            "scenario_summary_csv": str(SCENARIO_SUMMARY_CSV),
            "scenario_status_csv": str(SCENARIO_STATUS_CSV),
            "prediction_dir": str(PRED_DIR / args.model_group),
            "report_md": str(REPORT_MD),
        },
        "started_at": json.loads(STATE_PATH.read_text(encoding="utf-8")).get("started_at", ""),
        "finished_at": now_iso(),
    }
    write_report(state, scenario_summary_df)
    atomic_write_json(STATE_PATH, state)
    write_status(state_status, "stage07 completed", state)
    logging.info("Stage07 completed: %s", json.dumps(state, ensure_ascii=False))
    return state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="用已训练模型预测未来 WorldClim 情景下样本点适宜性。")
    parser.add_argument("--model-group", default="hist_gradient_boosting_balanced")
    parser.add_argument("--model-dir", default=str(MODEL_DIR), help="模型目录，默认使用 Stage06 20 因子基线模型。")
    parser.add_argument("--output-dir", default=str(OUT_DIR), help="输出目录，默认使用 Stage07 目录。")
    parser.add_argument("--run-label", default=None, help="独立重跑标签；会隔离日志、状态、摘要和预测子目录。")
    parser.add_argument("--gcm", action="append", help="只运行指定 GCM，可重复。")
    parser.add_argument("--ssp", action="append", help="只运行指定 SSP，可重复。")
    parser.add_argument("--period", action="append", help="只运行指定时期，可重复。")
    parser.add_argument("--limit-scenarios", type=int, default=None, help="仅运行前 N 个情景，用于试跑。")
    parser.add_argument(
        "--model-specific-outputs",
        action="store_true",
        help="Use model-group-specific state, status, summary, report, and log files to avoid overwriting the main Stage07 outputs.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_paths(args)
    setup_logging()
    try:
        state = run(args)
        return 0 if state.get("status") in {"success", "partial_success"} else 1
    except Exception as exc:
        err = {
            "status": "failed",
            "failed_at": now_iso(),
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }
        atomic_write_json(STATE_PATH, err)
        write_status("failed", repr(exc))
        logging.exception("Stage07 failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
