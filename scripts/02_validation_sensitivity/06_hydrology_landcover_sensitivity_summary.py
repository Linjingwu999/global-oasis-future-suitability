# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "outputs" / "stage36_hydrology_landcover_sensitivity"
TABLE_DIR = OUT_DIR / "tables"
LOG_DIR = PROJECT_ROOT / "logs"

SCENARIOS = [
    {
        "scenario": "q1_current_reference",
        "role": "current broad hydrology reference",
        "min_discharge_cms": 1.0,
        "summary_csv": PROJECT_ROOT
        / "outputs"
        / "stage34_selected10_landcover_spatial_constraint"
        / "tables"
        / "stage20_landcover_spatial_constraint_selected10_hgb_main_summary.csv",
        "status_json": PROJECT_ROOT
        / "outputs"
        / "stage34_selected10_landcover_spatial_constraint"
        / "stage20_landcover_spatial_constraint_selected10_hgb_main_summary.json",
    },
    {
        "scenario": "q10_main_candidate",
        "role": "recommended main hydrological constraint",
        "min_discharge_cms": 10.0,
        "summary_csv": OUT_DIR
        / "q10cms"
        / "tables"
        / "stage20_landcover_spatial_constraint_selected10_hgb_hydrorivers_q10cms_landcover_summary.csv",
        "status_json": OUT_DIR
        / "q10cms"
        / "stage20_landcover_spatial_constraint_selected10_hgb_hydrorivers_q10cms_landcover_summary.json",
    },
    {
        "scenario": "q25_strict_backup",
        "role": "strict hydrological sensitivity backup",
        "min_discharge_cms": 25.0,
        "summary_csv": OUT_DIR
        / "q25cms"
        / "tables"
        / "stage20_landcover_spatial_constraint_selected10_hgb_hydrorivers_q25cms_landcover_summary.csv",
        "status_json": OUT_DIR
        / "q25cms"
        / "stage20_landcover_spatial_constraint_selected10_hgb_hydrorivers_q25cms_landcover_summary.json",
    },
]


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "missing", "path": str(path)}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def summarize_one(config: dict[str, Any]) -> dict[str, Any]:
    csv_path = Path(config["summary_csv"])
    status_json = Path(config["status_json"])
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing Stage20 summary CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    ok = df[df["status"].astype(str).isin(["success", "skipped"])].copy()
    failed = df[df["status"].astype(str).eq("failed")].copy()
    stage17_area = float(ok["stage17_suitable_area_km2"].sum())
    binary_area = float(ok["binary_suitable_area_km2"].sum())
    weighted_area = float(ok["weighted_compatible_area_km2"].sum())
    excluded_area = float(ok["excluded_by_landcover_area_km2"].sum())
    return {
        "scenario": config["scenario"],
        "role": config["role"],
        "min_discharge_cms": config["min_discharge_cms"],
        "summary_status": read_json(status_json).get("status", "unknown"),
        "rows": int(len(df)),
        "ok_rows": int(len(ok)),
        "failed_rows": int(len(failed)),
        "stage17_area_km2": stage17_area,
        "binary50_area_km2": binary_area,
        "weighted_area_km2": weighted_area,
        "landcover_excluded_km2": excluded_area,
        "stage17_area_wan_km2": stage17_area / 10000.0,
        "binary50_area_wan_km2": binary_area / 10000.0,
        "weighted_area_wan_km2": weighted_area / 10000.0,
        "landcover_excluded_wan_km2": excluded_area / 10000.0,
        "binary_retention_pct_total": (binary_area / stage17_area * 100.0) if stage17_area else None,
        "weighted_retention_pct_total": (weighted_area / stage17_area * 100.0) if stage17_area else None,
        "summary_csv": str(csv_path.relative_to(PROJECT_ROOT)),
        "status_json": str(status_json.relative_to(PROJECT_ROOT)),
    }


def build_report(summary: pd.DataFrame, output_csv: Path) -> str:
    generated_at = now_iso()
    q1 = summary[summary["scenario"].eq("q1_current_reference")].iloc[0]
    q10 = summary[summary["scenario"].eq("q10_main_candidate")].iloc[0]
    q25 = summary[summary["scenario"].eq("q25_strict_backup")].iloc[0]

    q10_drop = q1["weighted_area_wan_km2"] - q10["weighted_area_wan_km2"]
    q25_drop = q10["weighted_area_wan_km2"] - q25["weighted_area_wan_km2"]

    lines = [
        "# Stage36 水文阈值与土地覆盖约束敏感性汇总",
        "",
        f"- 生成时间：{generated_at}",
        f"- 汇总表：`{output_csv.relative_to(PROJECT_ROOT)}`",
        "- 说明：q10 被作为正文主候选水文阈值；q25 保留为严格水文备选/敏感性结果；q1 仅作为原主链宽松水文参照。",
        "- q20 曾按候选档启动，但根据最新方案已停止，不纳入本轮主线汇总。",
        "",
        "## 结果总览",
        "",
        "| 情景 | 角色 | HydroRIVERS 最小多年平均流量 | Stage17 面积(万 km²) | 土地覆盖二值>=50%(万 km²) | 土地覆盖加权面积(万 km²) | 完整性 |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            "| {scenario} | {role} | {q:.0f} m³/s | {stage17:.2f} | {binary:.2f} | {weighted:.2f} | {ok}/{rows} ok, {failed} failed |".format(
                scenario=row["scenario"],
                role=row["role"],
                q=row["min_discharge_cms"],
                stage17=row["stage17_area_wan_km2"],
                binary=row["binary50_area_wan_km2"],
                weighted=row["weighted_area_wan_km2"],
                ok=int(row["ok_rows"]),
                rows=int(row["rows"]),
                failed=int(row["failed_rows"]),
            )
        )

    lines.extend(
        [
            "",
            "## 写作解释建议",
            "",
            (
                f"q10 约束后，Stage17 水文-地形-绿洲邻近约束面积为 {q10['stage17_area_wan_km2']:.2f} 万 km²，"
                f"叠加 ESA WorldCover 兼容度后的加权面积为 {q10['weighted_area_wan_km2']:.2f} 万 km²，"
                f"二值兼容面积为 {q10['binary50_area_wan_km2']:.2f} 万 km²。"
            ),
            (
                f"相对于 q1 原宽松水文参照，q10 的土地覆盖加权面积减少 {q10_drop:.2f} 万 km²，"
                "主要体现为低流量河段和季节性/小支流邻近区被剔除。"
            ),
            (
                f"q25 严格备选的加权面积为 {q25['weighted_area_wan_km2']:.2f} 万 km²，"
                f"比 q10 再减少 {q25_drop:.2f} 万 km²，适合用于说明更严格河流阈值下的保守范围。"
            ),
            "",
            "建议正文表述：选择 q10 并不是为了使面积接近某个预期值，而是因为它在排除小流量支流导致的潜在虚高和保留主要河流补给廊道之间取得折中；q25 作为严格备选，用于检验结果对水文阈值提高的敏感性。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    rows = [summarize_one(config) for config in SCENARIOS]
    summary = pd.DataFrame(rows)

    q1_weighted = float(summary.loc[summary["scenario"].eq("q1_current_reference"), "weighted_area_km2"].iloc[0])
    summary["weighted_delta_vs_q1_wan_km2"] = (summary["weighted_area_km2"] - q1_weighted) / 10000.0
    summary["weighted_pct_vs_q1"] = summary["weighted_area_km2"] / q1_weighted * 100.0

    output_csv = TABLE_DIR / "stage36_hydrology_landcover_sensitivity_summary.csv"
    output_json = OUT_DIR / "stage36_hydrology_landcover_sensitivity_summary.json"
    report_md = OUT_DIR / "stage36_hydrology_landcover_sensitivity_report.md"
    state_json = LOG_DIR / "stage36_hydrology_landcover_sensitivity_state.json"

    tmp_csv = output_csv.with_suffix(output_csv.suffix + ".tmp")
    summary.to_csv(tmp_csv, index=False, encoding="utf-8-sig")
    tmp_csv.replace(output_csv)

    report_md.write_text(build_report(summary, output_csv), encoding="utf-8")
    status = "success" if int(summary["failed_rows"].sum()) == 0 else "partial_success"
    state = {
        "status": status,
        "generated_at": now_iso(),
        "scenarios": int(len(summary)),
        "failed_rows_total": int(summary["failed_rows"].sum()),
        "summary_csv": str(output_csv.relative_to(PROJECT_ROOT)),
        "summary_json": str(output_json.relative_to(PROJECT_ROOT)),
        "report_md": str(report_md.relative_to(PROJECT_ROOT)),
    }
    write_json(output_json, state)
    write_json(state_json, state)
    print(json.dumps(state, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
