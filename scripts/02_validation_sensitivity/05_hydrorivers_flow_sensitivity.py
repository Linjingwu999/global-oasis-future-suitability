# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_ROOT / "logs"
DEFAULT_STAGE12_STATUS = LOG_DIR / "stage12_region_tile_grid_projection_selected10_hgb_main_status.csv"
DEFAULT_BASELINE_SUMMARY_JSON = (
    PROJECT_ROOT
    / "outputs"
    / "stage34_selected10_constrained_suitability"
    / "stage17_constrained_suitability_selected10_hgb_main_summary.json"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "stage35_hydrorivers_flow_sensitivity"
DEFAULT_RUN_LABEL = "selected10_hgb_hydrorivers_flow_sensitivity"


def resolve_default_stage17_script() -> Path:
    candidates = sorted((PROJECT_ROOT / "scripts").glob("17_*.py"))
    if not candidates:
        return PROJECT_ROOT / "scripts" / "17_约束未来适宜区_地形约束后处理.py"
    if len(candidates) == 1:
        return candidates[0]
    for candidate in candidates:
        if "约束" in candidate.name and "地形" in candidate.name:
            return candidate
    return candidates[0]


DEFAULT_STAGE17_SCRIPT = resolve_default_stage17_script()


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def number_token(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:g}".replace(".", "p")


def q_label(value: float) -> str:
    return f"q{number_token(value)}cms"


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


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_existing_status(status_csv: Path) -> pd.DataFrame:
    if status_csv.exists():
        try:
            return pd.read_csv(status_csv)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def update_threshold_status(status_csv: Path, row: dict[str, Any]) -> None:
    existing = load_existing_status(status_csv)
    row_df = pd.DataFrame([row])
    if existing.empty or "threshold_label" not in existing.columns:
        out = row_df
    else:
        existing = existing[existing["threshold_label"].astype(str) != str(row["threshold_label"])]
        out = pd.concat([existing, row_df], ignore_index=True)
    atomic_write_csv(status_csv, out)


def normalize_thresholds(values: list[float]) -> list[float]:
    out: list[float] = []
    seen: set[str] = set()
    for value in values:
        token = number_token(value)
        if token not in seen:
            seen.add(token)
            out.append(float(value))
    return sorted(out)


def summarize_stage17(summary_csv: Path, threshold: float, threshold_status: str, source_kind: str) -> dict[str, Any]:
    if not summary_csv.exists():
        raise FileNotFoundError(f"Stage17 summary CSV not found: {summary_csv}")
    df = pd.read_csv(summary_csv)
    if df.empty:
        raise RuntimeError(f"Stage17 summary CSV is empty: {summary_csv}")
    numeric_cols = [
        "valid_area_km2_recomputed",
        "original_suitable_area_km2_recomputed",
        "constrained_suitable_area_km2",
        "excluded_by_elevation_area_km2",
        "excluded_by_slope_area_km2",
        "excluded_by_oasis_area_km2",
        "excluded_by_river_area_km2",
        "excluded_by_any_area_km2",
    ]
    sums: dict[str, float] = {}
    for col in numeric_cols:
        if col in df.columns:
            sums[col] = float(pd.to_numeric(df[col], errors="coerce").fillna(0.0).sum())
        else:
            sums[col] = float("nan")

    original = sums["original_suitable_area_km2_recomputed"]
    constrained = sums["constrained_suitable_area_km2"]
    retention = constrained / original if original else float("nan")

    statuses = {}
    if "status" in df.columns:
        statuses = df["status"].astype(str).value_counts().to_dict()

    return {
        "threshold_cms": threshold,
        "threshold_label": q_label(threshold),
        "threshold_status": threshold_status,
        "source_kind": source_kind,
        "summary_csv": str(summary_csv),
        "tile_rows": int(len(df)),
        "success_or_skipped_rows": int(
            df["status"].astype(str).isin(["success", "skipped"]).sum() if "status" in df.columns else len(df)
        ),
        "failed_rows": int((df["status"].astype(str) == "failed").sum() if "status" in df.columns else 0),
        "tile_status_counts": json.dumps(statuses, ensure_ascii=False, sort_keys=True),
        "valid_area_km2_recomputed": sums["valid_area_km2_recomputed"],
        "original_suitable_area_km2_recomputed": original,
        "constrained_suitable_area_km2": constrained,
        "excluded_by_elevation_area_km2": sums["excluded_by_elevation_area_km2"],
        "excluded_by_slope_area_km2": sums["excluded_by_slope_area_km2"],
        "excluded_by_oasis_area_km2": sums["excluded_by_oasis_area_km2"],
        "excluded_by_river_area_km2": sums["excluded_by_river_area_km2"],
        "excluded_by_any_area_km2": sums["excluded_by_any_area_km2"],
        "constraint_retention_rate": retention,
    }


def append_deltas(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.sort_values("threshold_cms").reset_index(drop=True).copy()
    baseline = out[out["threshold_cms"] == out["threshold_cms"].min()].iloc[0]
    base_area = float(baseline["constrained_suitable_area_km2"])
    base_ret = float(baseline["constraint_retention_rate"])
    out["delta_constrained_area_from_lowest_threshold_km2"] = (
        pd.to_numeric(out["constrained_suitable_area_km2"], errors="coerce") - base_area
    )
    out["additional_area_removed_from_lowest_threshold_km2"] = -out[
        "delta_constrained_area_from_lowest_threshold_km2"
    ]
    out["delta_retention_rate_from_lowest_threshold_pp"] = (
        pd.to_numeric(out["constraint_retention_rate"], errors="coerce") - base_ret
    ) * 100.0
    return out


def write_report(path: Path, summary: pd.DataFrame, args: argparse.Namespace, completeness: dict[str, Any]) -> None:
    lines = [
        "# Stage35 HydroRIVERS flow-threshold sensitivity",
        "",
        f"- Generated at: {now_iso()}",
        f"- Stage12 status: `{args.stage12_status_csv}`",
        f"- Baseline Stage17 summary JSON: `{args.baseline_summary_json}`",
        f"- Output directory: `{args.output_dir}`",
        f"- River buffer: {args.river_buffer_km:g} km",
        f"- Upstream-area filter: {args.min_river_upstream_km2:g} km2",
        f"- Thresholds: {', '.join(q_label(v) for v in args.thresholds)}",
        "",
        "## Result summary",
        "",
        "| DIS_AV_CMS threshold | Area after constraints (10k km2) | Change vs lowest threshold (10k km2) | Retention (%) | Status |",
        "|---:|---:|---:|---:|---|",
    ]
    for _, row in summary.iterrows():
        area_10k = float(row["constrained_suitable_area_km2"]) / 10000.0
        delta_10k = float(row["delta_constrained_area_from_lowest_threshold_km2"]) / 10000.0
        retention_pct = float(row["constraint_retention_rate"]) * 100.0
        lines.append(
            f"| >= {row['threshold_cms']:g} m3/s | {area_10k:,.2f} | {delta_10k:,.2f} | {retention_pct:.2f} | {row['threshold_status']} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "- This is a hydrological-neighborhood sensitivity analysis built on the selected10 HGB spatial projection and the existing Stage17 terrain/oasis/river constraint chain.",
            "- It does not retrain the ecological suitability model; it tests how much the constrained suitability envelope changes when small HydroRIVERS streams are removed by higher long-term discharge thresholds.",
            "- The lowest-threshold row corresponds to the current main Stage17 hydrology setting and is treated as the reference envelope. Higher thresholds are conservative sensitivity checks, not automatic replacements for the main result.",
            "- Areas are potential suitability envelopes after constraints, not observed or guaranteed future oasis extent.",
            "",
            "## Completeness check",
            "",
            f"- Overall status: `{completeness.get('status')}`",
            f"- Completed thresholds: {completeness.get('success_thresholds')} / {completeness.get('total_thresholds')}",
            f"- Failed thresholds: {completeness.get('failed_thresholds')}",
            f"- Threshold status CSV: `{completeness.get('threshold_status_csv')}`",
            f"- Summary CSV: `{completeness.get('summary_csv')}`",
            f"- Detail CSV: `{completeness.get('detail_csv')}`",
        ]
    )
    atomic_write_text(path, "\n".join(lines) + "\n")


def build_stage17_command(
    args: argparse.Namespace,
    threshold: float,
    run_dir: Path,
    baseline: dict[str, Any],
) -> list[str]:
    label = f"{args.run_label}_{q_label(threshold)}"
    cmd = [
        sys.executable,
        "-X",
        "utf8",
        str(args.stage17_script),
        "--model-group",
        args.model_group,
        "--stage12-status-csv",
        str(args.stage12_status_csv),
        "--output-dir",
        str(run_dir),
        "--run-label",
        label,
        "--gcm",
        args.gcm,
        "--ssp",
        args.ssp,
        "--period",
        args.period,
        "--oasis-buffer-km",
        str(args.oasis_buffer_km),
        "--river-buffer-km",
        str(args.river_buffer_km),
        "--min-river-discharge-cms",
        str(threshold),
        "--min-river-upstream-km2",
        str(args.min_river_upstream_km2),
        "--block-size",
        str(args.block_size),
    ]
    oasis_vector = args.oasis_vector or baseline.get("oasis_vector")
    river_vector = args.river_vector or baseline.get("river_vector")
    river_layer = args.river_layer or baseline.get("river_layer")
    if oasis_vector:
        cmd.extend(["--oasis-vector", str(oasis_vector)])
    else:
        cmd.append("--use-default-oasis-vector")
    if river_vector:
        cmd.extend(["--river-vector", str(river_vector)])
    else:
        cmd.append("--use-default-river-vector")
    if river_layer:
        cmd.extend(["--river-layer", str(river_layer)])
    if args.limit:
        cmd.extend(["--limit", str(args.limit)])
    if args.overwrite_stage17:
        cmd.append("--overwrite")
    return cmd


def run_stage17_threshold(
    args: argparse.Namespace,
    threshold: float,
    run_dir: Path,
    baseline: dict[str, Any],
    threshold_status_csv: Path,
) -> dict[str, Any]:
    run_label = f"{args.run_label}_{q_label(threshold)}"
    summary_json = run_dir / f"stage17_constrained_suitability_{run_label}_summary.json"
    summary_csv = run_dir / "tables" / f"stage17_constrained_suitability_{run_label}_summary.csv"

    if summary_json.exists() and summary_csv.exists() and not args.rerun:
        try:
            existing = read_json(summary_json)
            if existing.get("status") == "success":
                row = {
                    "updated_at": now_iso(),
                    "threshold_cms": threshold,
                    "threshold_label": q_label(threshold),
                    "status": "skipped",
                    "message": "existing Stage17 threshold run reused",
                    "exit_code": 0,
                    "summary_json": str(summary_json),
                    "summary_csv": str(summary_csv),
                    "run_dir": str(run_dir),
                    "command": "",
                }
                update_threshold_status(threshold_status_csv, row)
                return row
        except Exception:
            pass

    cmd = build_stage17_command(args, threshold, run_dir, baseline)
    stdout_log = LOG_DIR / f"stage35_hydrorivers_flow_sensitivity_{q_label(threshold)}_stdout.log"
    stderr_log = LOG_DIR / f"stage35_hydrorivers_flow_sensitivity_{q_label(threshold)}_stderr.log"
    row = {
        "updated_at": now_iso(),
        "threshold_cms": threshold,
        "threshold_label": q_label(threshold),
        "status": "running",
        "message": "Stage17 threshold run started",
        "exit_code": "",
        "summary_json": str(summary_json),
        "summary_csv": str(summary_csv),
        "run_dir": str(run_dir),
        "stdout_log": str(stdout_log),
        "stderr_log": str(stderr_log),
        "command": " ".join(f'"{part}"' if " " in str(part) else str(part) for part in cmd),
        "started_at": now_iso(),
        "finished_at": "",
    }
    update_threshold_status(threshold_status_csv, row)

    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        with stdout_log.open("w", encoding="utf-8") as out, stderr_log.open("w", encoding="utf-8") as err:
            proc = subprocess.run(
                cmd,
                cwd=str(PROJECT_ROOT),
                stdout=out,
                stderr=err,
                text=True,
                env={**os.environ, "PYTHONUTF8": "1", "OASIS_PROJECT_ROOT": str(PROJECT_ROOT)},
            )
        row["exit_code"] = proc.returncode
        row["finished_at"] = now_iso()
        if proc.returncode == 0 and summary_csv.exists():
            summary_status = ""
            if summary_json.exists():
                try:
                    summary_status = str(read_json(summary_json).get("status", ""))
                except Exception:
                    summary_status = ""
            if summary_status == "success":
                row["status"] = "success"
                row["message"] = "Stage17 threshold run completed"
            elif summary_status == "partial_success":
                row["status"] = "partial_success"
                row["message"] = "Stage17 threshold run completed with tile-level failures"
            else:
                row["status"] = "success"
                row["message"] = "Stage17 threshold run completed; summary status unavailable"
        else:
            row["status"] = "failed"
            row["message"] = f"Stage17 threshold run failed or summary missing; exit_code={proc.returncode}"
    except Exception as exc:
        row["status"] = "failed"
        row["message"] = str(exc)
        row["traceback"] = traceback.format_exc()
        row["finished_at"] = now_iso()
    update_threshold_status(threshold_status_csv, row)
    return row


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    table_dir = output_dir / "tables"
    run_root = output_dir / "runs"
    output_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    threshold_status_csv = LOG_DIR / f"stage35_hydrorivers_flow_sensitivity_{args.run_label}_threshold_status.csv"
    state_json = LOG_DIR / f"stage35_hydrorivers_flow_sensitivity_{args.run_label}_state.json"
    summary_csv = table_dir / f"stage35_hydrorivers_flow_sensitivity_{args.run_label}_summary.csv"
    detail_csv = table_dir / f"stage35_hydrorivers_flow_sensitivity_{args.run_label}_tile_details.csv"
    report_md = output_dir / f"stage35_hydrorivers_flow_sensitivity_{args.run_label}_report.md"

    baseline = read_json(Path(args.baseline_summary_json))
    baseline_summary_csv = Path(baseline["summary_csv"])
    args.thresholds = normalize_thresholds(args.thresholds)

    threshold_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    detail_frames: list[pd.DataFrame] = []

    atomic_write_json(
        state_json,
        {
            "status": "running",
            "updated_at": now_iso(),
            "message": "Stage35 HydroRIVERS flow sensitivity started",
            "thresholds": args.thresholds,
            "output_dir": str(output_dir),
        },
    )

    for threshold in args.thresholds:
        if abs(threshold - float(args.baseline_threshold_cms)) < 1e-9 and baseline_summary_csv.exists():
            row = {
                "updated_at": now_iso(),
                "threshold_cms": threshold,
                "threshold_label": q_label(threshold),
                "status": "success",
                "message": "baseline Stage17 summary reused",
                "exit_code": 0,
                "summary_json": str(args.baseline_summary_json),
                "summary_csv": str(baseline_summary_csv),
                "run_dir": str(Path(args.baseline_summary_json).parent),
                "command": "",
                "started_at": "",
                "finished_at": now_iso(),
            }
            update_threshold_status(threshold_status_csv, row)
        else:
            run_dir = run_root / q_label(threshold)
            row = run_stage17_threshold(args, threshold, run_dir, baseline, threshold_status_csv)
        threshold_rows.append(row)

        try:
            if threshold == float(args.baseline_threshold_cms):
                source_kind = "baseline_reused"
            elif row.get("status") == "skipped":
                source_kind = "stage17_reused"
            else:
                source_kind = "stage17_rerun"
            summary_status = "success" if row.get("status") == "skipped" else str(row["status"])
            summary_row = summarize_stage17(Path(row["summary_csv"]), threshold, summary_status, source_kind)
            summary_rows.append(summary_row)
            detail = pd.read_csv(row["summary_csv"])
            detail.insert(0, "threshold_cms", threshold)
            detail.insert(1, "threshold_label", q_label(threshold))
            detail_frames.append(detail)
        except Exception as exc:
            summary_rows.append(
                {
                    "threshold_cms": threshold,
                    "threshold_label": q_label(threshold),
                    "threshold_status": "failed",
                    "source_kind": "summary_failed",
                    "summary_csv": row.get("summary_csv", ""),
                    "tile_rows": 0,
                    "success_or_skipped_rows": 0,
                    "failed_rows": 0,
                    "tile_status_counts": "{}",
                    "error": str(exc),
                }
            )

    summary = pd.DataFrame(summary_rows)
    ok_mask = pd.to_numeric(summary.get("constrained_suitable_area_km2"), errors="coerce").notna()
    successful_summary = summary[ok_mask].copy()
    if not successful_summary.empty:
        successful_summary = append_deltas(successful_summary)
        failed_summary = summary[~ok_mask].copy()
        summary_out = pd.concat([successful_summary, failed_summary], ignore_index=True, sort=False)
    else:
        summary_out = summary

    atomic_write_csv(summary_csv, summary_out)
    if detail_frames:
        atomic_write_csv(detail_csv, pd.concat(detail_frames, ignore_index=True, sort=False))

    success_thresholds = [row["threshold_label"] for row in threshold_rows if row.get("status") in {"success", "skipped"}]
    partial_thresholds = [row["threshold_label"] for row in threshold_rows if row.get("status") == "partial_success"]
    failed_thresholds = [row["threshold_label"] for row in threshold_rows if row.get("status") == "failed"]
    problem_thresholds = partial_thresholds + failed_thresholds
    completeness = {
        "status": "success" if not problem_thresholds else "partial_success",
        "updated_at": now_iso(),
        "total_thresholds": len(args.thresholds),
        "success_thresholds": len(success_thresholds),
        "partial_thresholds": partial_thresholds,
        "failed_thresholds": failed_thresholds,
        "threshold_status_csv": str(threshold_status_csv),
        "summary_csv": str(summary_csv),
        "detail_csv": str(detail_csv),
        "report_md": str(report_md),
        "output_dir": str(output_dir),
    }
    write_report(report_md, summary_out, args, completeness)
    atomic_write_json(state_json, completeness)
    return completeness


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run HydroRIVERS discharge-threshold sensitivity on the selected10 HGB Stage17 constraint chain."
    )
    parser.add_argument("--stage17-script", type=Path, default=DEFAULT_STAGE17_SCRIPT)
    parser.add_argument("--stage12-status-csv", type=Path, default=DEFAULT_STAGE12_STATUS)
    parser.add_argument("--baseline-summary-json", type=Path, default=DEFAULT_BASELINE_SUMMARY_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-label", default=DEFAULT_RUN_LABEL)
    parser.add_argument("--model-group", default="hist_gradient_boosting_balanced")
    parser.add_argument("--gcm", default="ACCESS-CM2")
    parser.add_argument("--ssp", default="ssp585")
    parser.add_argument("--period", default="2081-2100")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[1.0, 5.0, 10.0, 25.0, 50.0])
    parser.add_argument("--baseline-threshold-cms", type=float, default=1.0)
    parser.add_argument("--oasis-vector", default=None)
    parser.add_argument("--river-vector", default=None)
    parser.add_argument("--river-layer", default=None)
    parser.add_argument("--oasis-buffer-km", type=float, default=100.0)
    parser.add_argument("--river-buffer-km", type=float, default=100.0)
    parser.add_argument("--min-river-upstream-km2", type=float, default=1000.0)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--rerun", action="store_true", help="Rerun Stage35 wrapper even when prior threshold summaries exist.")
    parser.add_argument("--overwrite-stage17", action="store_true", help="Pass --overwrite to Stage17 threshold runs.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = run(args)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result.get("status") == "success" else 2
    except Exception as exc:
        run_label = getattr(locals().get("args", None), "run_label", DEFAULT_RUN_LABEL)
        state_json = LOG_DIR / f"stage35_hydrorivers_flow_sensitivity_{run_label}_state.json"
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            state_json,
            {
                "status": "failed",
                "updated_at": now_iso(),
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        print(traceback.format_exc(), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
