from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import math
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
from shapely.geometry import Point
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "outputs" / "stage59_field_survey_control_validation"
TABLE_DIR = OUT_DIR / "tables"
LOG_DIR = PROJECT_ROOT / "logs"
LOG_PATH = LOG_DIR / "stage59_field_survey_control_validation.log"
STATE_PATH = OUT_DIR / "stage59_field_survey_control_validation_state.json"
STATUS_CSV = LOG_DIR / "stage59_field_survey_control_validation_status.csv"
REPORT_MD = OUT_DIR / "stage59_field_survey_control_validation_report.md"
SUMMARY_JSON = OUT_DIR / "stage59_field_survey_control_validation_summary.json"

MODEL_DIR = PROJECT_ROOT / "outputs" / "stage31_selected_predictor_models" / "models"
STAGE02_DIR = PROJECT_ROOT / "outputs" / "stage02_background_points"
FIELD_SEARCH_ROOT = Path.home() / "Desktop" / "ESSD"

DEFAULT_SEED = 20260602
DEFAULT_CONTROLS_PER_PRESENCE = 3
DEFAULT_EXCLUDE_BUFFER_M = 3000.0
SOURCE_CRS = "EPSG:4326"
REGION_NAME_MAP = {
    "中国": "China",
    "埃及": "Egypt",
    "沙特": "Saudi Arabia",
}


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2))


def write_status(status: str, step: str, message: str, extra: dict[str, Any] | None = None) -> None:
    payload = {
        "timestamp": now_iso(),
        "task": "stage59_field_survey_control_validation",
        "status": status,
        "step": step,
        "message": message,
    }
    if extra:
        payload.update(extra)
    STATUS_CSV.parent.mkdir(parents=True, exist_ok=True)
    exists = STATUS_CSV.exists()
    pd.DataFrame([payload]).to_csv(STATUS_CSV, mode="a", index=False, header=not exists, encoding="utf-8-sig")
    atomic_write_json(STATE_PATH, payload)


def setup_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(LOG_PATH, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )


def import_stage_script(prefix: str, largest: bool = False):
    candidates = [p for p in (PROJECT_ROOT / "scripts").iterdir() if p.is_file() and p.name.startswith(prefix)]
    if not candidates:
        raise FileNotFoundError(f"No script found with prefix {prefix!r}")
    if largest:
        path = sorted(candidates, key=lambda p: p.stat().st_size, reverse=True)[0]
    else:
        path = sorted(candidates)[0]
    module_name = f"stage59_import_{prefix.strip('_')}_{abs(hash(path.name))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module, path


def find_field_csv() -> Path:
    explicit = FIELD_SEARCH_ROOT / "验证点数据" / "绿洲实地调查点位_面内筛选.csv"
    if explicit.exists():
        return explicit
    required = {"SourceRegion", "PointLon", "PointLat"}
    if not FIELD_SEARCH_ROOT.exists():
        raise FileNotFoundError(f"Field search root does not exist: {FIELD_SEARCH_ROOT}")
    for path in FIELD_SEARCH_ROOT.rglob("*.csv"):
        try:
            header = pd.read_csv(path, nrows=0, encoding="utf-8-sig")
        except Exception:
            continue
        if required.issubset(set(header.columns)):
            return path
    raise FileNotFoundError(f"No field survey CSV with columns {sorted(required)} found under {FIELD_SEARCH_ROOT}")


def read_field_presence(field_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(field_csv, encoding="utf-8-sig")
    required = {"SourceRegion", "PointLon", "PointLat"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Field CSV missing required columns: {missing}")
    df = df.copy()
    df["SourceRegion"] = df["SourceRegion"].map(lambda value: REGION_NAME_MAP.get(str(value).strip(), str(value).strip()))
    df["PointLon"] = pd.to_numeric(df["PointLon"], errors="coerce")
    df["PointLat"] = pd.to_numeric(df["PointLat"], errors="coerce")
    bad = df[["PointLon", "PointLat"]].isna().any(axis=1)
    out = (df["PointLon"] < -180) | (df["PointLon"] > 180) | (df["PointLat"] < -90) | (df["PointLat"] > 90)
    if bool((bad | out).any()):
        raise ValueError(f"Invalid field survey coordinates: {int((bad | out).sum())}")
    if "value" in df.columns:
        df["OriginalValue"] = df["value"]
    else:
        df["OriginalValue"] = np.nan
    df["ValidationSource"] = "field_survey_presence"
    df["ValidationClass"] = "presence"
    df["Response"] = 1
    df["ValidationID"] = [f"field_presence_{i + 1:04d}" for i in range(len(df))]
    return df


def generate_dryland_controls(
    target_n: int,
    seed: int,
    exclude_buffer_m: float,
    candidate_factor: float,
    max_rounds: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    stage02, stage02_path = import_stage_script("02_", largest=True)
    logging.info("Using background helper script: %s", stage02_path)
    dry = stage02.read_dryland()
    buffers = stage02.build_oasis_buffers(exclude_buffer_m, reuse_cache=True)
    allocation = stage02.allocate_by_area(dry, target_n, min_per_stratum=0)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    controls_parts: list[gpd.GeoDataFrame] = []
    control_seq = 0

    for idx, target in allocation.items():
        if target <= 0:
            continue
        row = dry.loc[idx]
        stratum = str(row["DrylandStratum"])
        points, candidates = stage02.sample_stratum_points(
            row.geometry,
            target=target,
            buffers=buffers,
            rng=rng,
            candidate_factor=candidate_factor,
            max_rounds=max_rounds,
            stratum=stratum,
        )
        if len(points) < target:
            logging.warning("Only sampled %s/%s controls for stratum %s", len(points), target, stratum)
        if points:
            part = gpd.GeoDataFrame(
                {
                    "ValidationSource": ["generated_dryland_control"] * len(points),
                    "ValidationClass": ["generated_control"] * len(points),
                    "Response": [0] * len(points),
                    "DrylandStratum": [stratum] * len(points),
                    "ControlTargetInStratum": [target] * len(points),
                    "ExcludeBufferM": [exclude_buffer_m] * len(points),
                    "RandomSeed": [seed] * len(points),
                    "SourceStage02Script": [stage02_path.name] * len(points),
                },
                geometry=points,
                crs=stage02.AREA_CRS,
            )
            controls_parts.append(part)
        rows.append(
            {
                "DrylandStratum": stratum,
                "Target": int(target),
                "Sampled": int(len(points)),
                "CandidatePointsDrawn": int(candidates),
            }
        )

    if not controls_parts:
        raise RuntimeError("No dryland control points were generated.")
    controls_area = pd.concat(controls_parts, ignore_index=True)
    controls_area = gpd.GeoDataFrame(controls_area, geometry="geometry", crs=stage02.AREA_CRS)
    controls_area = controls_area.iloc[:target_n].copy()
    controls_area["ValidationID"] = [f"generated_control_{i + 1:04d}" for i in range(len(controls_area))]
    validation = stage02.validate_background(controls_area, dry, buffers)

    controls_ll = controls_area.to_crs(SOURCE_CRS)
    controls_ll["PointLon"] = controls_ll.geometry.x
    controls_ll["PointLat"] = controls_ll.geometry.y
    controls_ll["geometry_wkt"] = controls_ll.geometry.to_wkt()
    controls_df = pd.DataFrame(controls_ll.drop(columns="geometry"))
    controls_df["ControlSeq"] = range(1, len(controls_df) + 1)
    controls_df["OriginalValue"] = np.nan

    meta = {
        "helper_script": str(stage02_path),
        "dryland_shp": str(getattr(stage02, "DRYLAND_SHP", "")),
        "buffer_cache": str(getattr(stage02, "BUFFER_CACHE_GPKG", "")),
        "target_controls": int(target_n),
        "output_controls": int(len(controls_df)),
        "seed": int(seed),
        "exclude_buffer_m": float(exclude_buffer_m),
    }
    return controls_df, validation, {"allocation": rows, "meta": meta}


def load_current_worldclim_helpers():
    stage06, stage06_path = import_stage_script("06_", largest=False)
    variables = stage06.list_raster_variables()
    variable_map = {v.name: v for v in variables}
    return stage06, stage06_path, variable_map


def extract_selected_features(points: pd.DataFrame, selected_features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    stage06, stage06_path, variable_map = load_current_worldclim_helpers()
    lon = points["PointLon"].to_numpy(dtype="float64")
    lat = points["PointLat"].to_numpy(dtype="float64")
    feature_df = points.copy()
    rows: list[dict[str, Any]] = []
    missing = sorted(set(selected_features) - set(variable_map))
    if missing:
        raise FileNotFoundError(f"Selected features missing from current rasters: {missing}")

    for feature in selected_features:
        variable = variable_map[feature]
        logging.info("Extracting %s from %s", feature, variable.path)
        values = stage06.sample_raster(variable, lon, lat)
        feature_df[feature] = values
        finite = np.isfinite(values)
        rows.append(
            {
                "Feature": feature,
                "RasterPath": str(variable.path),
                "ValidCount": int(finite.sum()),
                "MissingCount": int((~finite).sum()),
                "Min": float(np.nanmin(values)) if finite.any() else np.nan,
                "Max": float(np.nanmax(values)) if finite.any() else np.nan,
                "Mean": float(np.nanmean(values)) if finite.any() else np.nan,
            }
        )

    meta = {
        "helper_script": str(stage06_path),
        "processed_worldclim_dir": str(getattr(stage06, "PROCESSED_DIR", "")),
    }
    return feature_df, pd.DataFrame(rows), meta


def load_model_payloads(model_prefix: str) -> list[dict[str, Any]]:
    paths = sorted(MODEL_DIR.glob(f"{model_prefix}_fold*.joblib"))
    if not paths:
        raise FileNotFoundError(f"No model files found for prefix: {model_prefix}")
    payloads: list[dict[str, Any]] = []
    for path in paths:
        obj = joblib.load(path)
        if isinstance(obj, dict):
            model = obj.get("model")
            features = list(obj.get("features", []))
            threshold = float(obj.get("threshold", 0.5))
        else:
            model = obj
            features = []
            threshold = 0.5
        if model is None or not features:
            raise ValueError(f"Invalid model payload: {path}")
        payloads.append({"path": path, "model": model, "features": features, "threshold": threshold})
    return payloads


def predict_model_group(df: pd.DataFrame, model_name: str, model_prefix: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    payloads = load_model_payloads(model_prefix)
    features = payloads[0]["features"]
    for payload in payloads[1:]:
        if payload["features"] != features:
            raise ValueError(f"Feature mismatch across {model_name} model folds")
    pred = df[["ValidationID", "ValidationSource", "ValidationClass", "Response", "PointLon", "PointLat"]].copy()
    for optional in ["SourceRegion", "DrylandStratum", "OriginalValue"]:
        if optional in df.columns:
            pred[optional] = df[optional]
    valid = df[features].notna().all(axis=1).to_numpy()
    prob_cols: list[str] = []
    X = df.loc[valid, features]
    for i, payload in enumerate(payloads, start=1):
        col = f"{model_name}_fold{i}_probability"
        prob_cols.append(col)
        pred[col] = np.nan
        model = payload["model"]
        if hasattr(model, "predict_proba"):
            values = model.predict_proba(X)[:, 1]
        else:
            values = model.predict(X)
        pred.loc[valid, col] = values
    pred[f"{model_name}_probability"] = pred[prob_cols].mean(axis=1)
    mean_threshold = float(np.mean([p["threshold"] for p in payloads]))
    pred[f"{model_name}_threshold"] = mean_threshold
    pred[f"{model_name}_predicted_presence"] = np.where(
        pred[f"{model_name}_probability"].notna(),
        (pred[f"{model_name}_probability"] >= mean_threshold).astype(int),
        np.nan,
    )
    meta = {
        "model_name": model_name,
        "model_prefix": model_prefix,
        "fold_count": len(payloads),
        "model_files": [str(p["path"]) for p in payloads],
        "features": features,
        "mean_threshold": mean_threshold,
        "valid_prediction_count": int(valid.sum()),
        "missing_feature_count": int((~valid).sum()),
    }
    return pred, meta


def metric_or_nan(func, *args, **kwargs) -> float:
    try:
        value = func(*args, **kwargs)
    except Exception:
        return float("nan")
    try:
        return float(value)
    except Exception:
        return float("nan")


def summarize_model_predictions(pred: pd.DataFrame, model_name: str) -> dict[str, Any]:
    prob_col = f"{model_name}_probability"
    yhat_col = f"{model_name}_predicted_presence"
    valid = pred[prob_col].notna() & pred[yhat_col].notna()
    sub = pred.loc[valid].copy()
    y_true = sub["Response"].astype(int).to_numpy()
    y_prob = sub[prob_col].astype(float).to_numpy()
    y_pred = sub[yhat_col].astype(int).to_numpy()
    labels = sorted(set(y_true.tolist()))
    if labels == [0, 1]:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    else:
        tn = fp = fn = tp = 0
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    false_positive_rate = fp / (tn + fp) if (tn + fp) else float("nan")
    tss = recall_score(y_true, y_pred, zero_division=0) + specificity - 1 if labels == [0, 1] else float("nan")
    presence = sub[sub["Response"] == 1]
    control = sub[sub["Response"] == 0]
    return {
        "Model": model_name,
        "ValidationDesign": "field_presence_vs_generated_dryland_control",
        "ValidN": int(len(sub)),
        "PresenceN": int(len(presence)),
        "GeneratedControlN": int(len(control)),
        "MeanPresenceProbability": float(presence[prob_col].mean()) if len(presence) else float("nan"),
        "MeanControlProbability": float(control[prob_col].mean()) if len(control) else float("nan"),
        "PresenceHitRate": float((presence[yhat_col] == 1).mean()) if len(presence) else float("nan"),
        "PresenceOmissionRate": float((presence[yhat_col] == 0).mean()) if len(presence) else float("nan"),
        "ControlSpecificity": float(specificity),
        "ControlFalsePositiveRate": float(false_positive_rate),
        "ROC_AUC": metric_or_nan(roc_auc_score, y_true, y_prob) if labels == [0, 1] else float("nan"),
        "PR_AUC": metric_or_nan(average_precision_score, y_true, y_prob) if labels == [0, 1] else float("nan"),
        "Precision": metric_or_nan(precision_score, y_true, y_pred, zero_division=0) if labels == [0, 1] else float("nan"),
        "Recall": metric_or_nan(recall_score, y_true, y_pred, zero_division=0) if labels == [0, 1] else float("nan"),
        "BalancedAccuracy": metric_or_nan(balanced_accuracy_score, y_true, y_pred) if labels == [0, 1] else float("nan"),
        "TSS": float(tss),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def summarize_presence_by_region(pred: pd.DataFrame, model_name: str) -> pd.DataFrame:
    prob_col = f"{model_name}_probability"
    yhat_col = f"{model_name}_predicted_presence"
    presence = pred[pred["Response"] == 1].copy()
    rows: list[dict[str, Any]] = []
    for region, sub in presence.groupby("SourceRegion", dropna=False):
        valid = sub[prob_col].notna() & sub[yhat_col].notna()
        v = sub.loc[valid]
        rows.append(
            {
                "Model": model_name,
                "SourceRegion": region,
                "PresenceN": int(len(sub)),
                "ValidN": int(len(v)),
                "MeanProbability": float(v[prob_col].mean()) if len(v) else np.nan,
                "MedianProbability": float(v[prob_col].median()) if len(v) else np.nan,
                "HitRate": float((v[yhat_col] == 1).mean()) if len(v) else np.nan,
                "OmissionRate": float((v[yhat_col] == 0).mean()) if len(v) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def write_report(summary: dict[str, Any], metric_df: pd.DataFrame, region_df: pd.DataFrame) -> None:
    lines = [
        "# Stage59 Field Survey Presence and Generated Dryland Control Validation",
        "",
        f"- Generated at: {now_iso()}",
        f"- Field survey CSV: {summary['field_csv']}",
        f"- Field presence points: {summary['field_presence_count']}",
        f"- Generated dryland controls: {summary['generated_control_count']}",
        f"- Control seed: {summary['control_generation']['meta']['seed']}",
        f"- Oasis exclusion buffer: {summary['control_generation']['meta']['exclude_buffer_m']} m",
        "",
        "## Interpretation boundary",
        "",
        "The field survey records are treated as presence-only oasis observations, following the user's clarification that all field points represent oasis occurrence. Dryland control samples were drawn from the dryland background outside the existing oasis exclusion buffer to provide a contrastive reference set for independent model checking.",
        "",
        "This stage validates current selected10 model predictions using current climate and terrain predictors. HydroRIVERS q10/q25 are spatial post-processing constraints for final suitability products, so they are discussed as constraints rather than as field-observed absence labels.",
        "",
        "## Model-level summary",
        "",
        metric_df.to_markdown(index=False),
        "",
        "## Field presence validation by source region",
        "",
        region_df.to_markdown(index=False),
        "",
        "## Outputs",
        "",
        "- tables/field_presence_points.csv",
        "- tables/generated_dryland_control_points.csv",
        "- tables/combined_validation_features.csv",
        "- tables/combined_validation_predictions.csv",
        "- tables/model_validation_summary.csv",
        "- tables/presence_validation_by_region.csv",
        "- stage59_field_survey_control_validation_summary.json",
    ]
    atomic_write_text(REPORT_MD, "\n".join(lines) + "\n")


def run(args: argparse.Namespace) -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    if (
        not args.force
        and STATE_PATH.exists()
        and SUMMARY_JSON.exists()
        and pd.read_json(STATE_PATH, typ="series").get("status") == "success"
    ):
        write_status("success", "skip", "Existing successful output found; use --force to rerun.")
        return json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))

    write_status("running", "start", "Starting field survey plus generated control validation.")
    field_csv = Path(args.field_csv) if args.field_csv else find_field_csv()
    field_df = read_field_presence(field_csv)
    control_n = int(args.control_count or len(field_df) * args.controls_per_presence)
    write_status("running", "generate_controls", f"Generating {control_n} dryland controls.")
    controls_df, control_check_df, control_meta = generate_dryland_controls(
        target_n=control_n,
        seed=args.seed,
        exclude_buffer_m=args.exclude_buffer_m,
        candidate_factor=args.candidate_factor,
        max_rounds=args.max_rounds,
    )

    base_cols = sorted(set(field_df.columns) | set(controls_df.columns))
    combined = pd.concat(
        [field_df.reindex(columns=base_cols), controls_df.reindex(columns=base_cols)],
        ignore_index=True,
    )
    field_out = TABLE_DIR / "field_presence_points.csv"
    control_out = TABLE_DIR / "generated_dryland_control_points.csv"
    combined_base_out = TABLE_DIR / "combined_validation_points.csv"
    field_df.to_csv(field_out, index=False, encoding="utf-8-sig")
    controls_df.to_csv(control_out, index=False, encoding="utf-8-sig")
    combined.to_csv(combined_base_out, index=False, encoding="utf-8-sig")
    control_check_df.to_csv(TABLE_DIR / "generated_control_integrity_check.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(control_meta["allocation"]).to_csv(TABLE_DIR / "generated_control_allocation.csv", index=False, encoding="utf-8-sig")

    write_status("running", "load_models", "Loading HGB and RF selected10 models.")
    hgb_payloads = load_model_payloads("hist_gradient_boosting_balanced")
    rf_payloads = load_model_payloads("random_forest_balanced")
    selected_features = hgb_payloads[0]["features"]
    if rf_payloads[0]["features"] != selected_features:
        raise ValueError("HGB and RF selected features differ; stop validation to avoid mixed feature semantics.")

    write_status("running", "extract_features", "Extracting current WorldClim selected10 features.")
    feature_df, variable_summary, raster_meta = extract_selected_features(combined, selected_features)
    feature_df.to_csv(TABLE_DIR / "combined_validation_features.csv", index=False, encoding="utf-8-sig")
    variable_summary.to_csv(TABLE_DIR / "current_worldclim_feature_extraction_summary.csv", index=False, encoding="utf-8-sig")

    write_status("running", "predict", "Predicting with HGB and RF fold ensembles.")
    hgb_pred, hgb_meta = predict_model_group(feature_df, "HGB", "hist_gradient_boosting_balanced")
    rf_pred, rf_meta = predict_model_group(feature_df, "RF", "random_forest_balanced")
    pred = hgb_pred.merge(
        rf_pred.drop(columns=["ValidationSource", "ValidationClass", "Response", "PointLon", "PointLat"], errors="ignore"),
        on="ValidationID",
        how="left",
        suffixes=("", "_rfdup"),
    )
    pred.to_csv(TABLE_DIR / "combined_validation_predictions.csv", index=False, encoding="utf-8-sig")

    metric_df = pd.DataFrame(
        [
            summarize_model_predictions(pred, "HGB"),
            summarize_model_predictions(pred, "RF"),
        ]
    )
    region_df = pd.concat(
        [
            summarize_presence_by_region(pred, "HGB"),
            summarize_presence_by_region(pred, "RF"),
        ],
        ignore_index=True,
    )
    metric_df.to_csv(TABLE_DIR / "model_validation_summary.csv", index=False, encoding="utf-8-sig")
    region_df.to_csv(TABLE_DIR / "presence_validation_by_region.csv", index=False, encoding="utf-8-sig")

    summary = {
        "status": "success",
        "finished_at": now_iso(),
        "field_csv": str(field_csv),
        "field_presence_count": int(len(field_df)),
        "generated_control_count": int(len(controls_df)),
        "selected_features": selected_features,
        "control_generation": control_meta,
        "raster_extraction": raster_meta,
        "model_meta": {"HGB": hgb_meta, "RF": rf_meta},
        "outputs": {
            "report_md": str(REPORT_MD),
            "summary_json": str(SUMMARY_JSON),
            "field_presence_points": str(field_out),
            "generated_controls": str(control_out),
            "combined_features": str(TABLE_DIR / "combined_validation_features.csv"),
            "combined_predictions": str(TABLE_DIR / "combined_validation_predictions.csv"),
            "model_validation_summary": str(TABLE_DIR / "model_validation_summary.csv"),
            "presence_validation_by_region": str(TABLE_DIR / "presence_validation_by_region.csv"),
            "log": str(LOG_PATH),
            "status_csv": str(STATUS_CSV),
        },
    }
    atomic_write_json(SUMMARY_JSON, summary)
    write_report(summary, metric_df, region_df)
    write_status("success", "complete", "Stage59 validation completed.", {"summary_json": str(SUMMARY_JSON)})
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate selected10 models against field presences and generated dryland controls.")
    parser.add_argument("--field-csv", default="", help="Optional field survey CSV path. If omitted, the script auto-detects it.")
    parser.add_argument("--controls-per-presence", type=int, default=DEFAULT_CONTROLS_PER_PRESENCE)
    parser.add_argument("--control-count", type=int, default=0, help="Override generated control count.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--exclude-buffer-m", type=float, default=DEFAULT_EXCLUDE_BUFFER_M)
    parser.add_argument("--candidate-factor", type=float, default=5.0)
    parser.add_argument("--max-rounds", type=int, default=100)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    setup_logging()
    try:
        args = parse_args()
        run(args)
        return 0
    except Exception as exc:
        logging.exception("Stage59 validation failed.")
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        error_payload = {
            "status": "failed",
            "failed_at": now_iso(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "log": str(LOG_PATH),
        }
        atomic_write_json(STATE_PATH, error_payload)
        write_status("failed", "exception", str(exc), {"error_type": type(exc).__name__})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
