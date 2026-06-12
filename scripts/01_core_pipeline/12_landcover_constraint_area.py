# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


WORKSPACE = Path(r"C:\Users\linjingwu\Desktop\python")
PROJECT_ROOT = WORKSPACE / "绿洲未来适宜区预测"
SYNC_STAGE17_ROOT = (
    Path(r"C:\Users\linjingwu\Desktop\跨电脑同步_工作站传输")
    / "stage17_future_suitability_workstation_20260530"
)

DEFAULT_STAGE17_COMPARISON = (
    SYNC_STAGE17_ROOT
    / "家里电脑查看_STAGE17结果"
    / "stage17_constraint_sensitivity_comparison.csv"
)
DEFAULT_LANDCOVER_SUMMARY = (
    WORKSPACE
    / "论文项目_三篇"
    / "01_全球绿洲智能提取与动态监测"
    / "03_结果输出"
    / "GEE_Global_LC_ESA_DW_2020_combined_with_NA"
    / "global_with_existing_NA_summary_by_scope_product_class.csv"
)

OUT_DIR = PROJECT_ROOT / "outputs" / "stage18_landcover_constraint_estimate"
POLICY_CSV = OUT_DIR / "stage18_landcover_policy_ratios.csv"
ESTIMATE_CSV = OUT_DIR / "stage18_landcover_area_estimates.csv"
SUMMARY_JSON = OUT_DIR / "stage18_landcover_area_estimate_summary.json"
REPORT_MD = OUT_DIR / "Stage18_土地覆盖约束面积估算与下一步.md"


CLASS_NAMES = {
    1: "Cropland",
    2: "Tree_shrub_grass_vegetation",
    3: "Wetland_mangrove_flooded_vegetation",
    4: "Built_up",
    5: "Bare_or_sparse_vegetation",
    6: "Water",
    7: "Snow_ice",
    8: "Other",
}

POLICIES = [
    {
        "policy_id": "recommended_esa_core_compatible",
        "product": "ESA_WorldCover_2020",
        "class_codes": [1, 4, 5],
        "role": "recommended_main",
        "interpretation": "Cropland + Built-up + Bare/sparse; removes mountain forest/grass/wetland/water/snow while retaining arid bare or sparse oasis corridors.",
    },
    {
        "policy_id": "upper_dw_core_compatible",
        "product": "Dynamic_World_2020_mode",
        "class_codes": [1, 4, 5],
        "role": "upper_sensitivity",
        "interpretation": "Same compatible classes under Dynamic World; kept as an upper land-cover product sensitivity bound.",
    },
    {
        "policy_id": "lower_esa_cropland_built",
        "product": "ESA_WorldCover_2020",
        "class_codes": [1, 4],
        "role": "lower_reference_too_strict",
        "interpretation": "Cropland + Built-up only; useful as a lower reference but too strict for arid sparse vegetation oases.",
    },
    {
        "policy_id": "soft_esa_keep_tree_shrub_grass",
        "product": "ESA_WorldCover_2020",
        "class_codes": [1, 2, 4, 5],
        "role": "soft_reference_not_main",
        "interpretation": "Keeps tree/shrub/grass vegetation; useful to show why the main result should not retain mountain vegetation broadly.",
    },
]


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


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


def read_csv_checked(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return pd.read_csv(path)


def class_label(codes: list[int]) -> str:
    return " + ".join(CLASS_NAMES.get(int(code), str(code)) for code in codes)


def build_policy_ratios(landcover_df: pd.DataFrame, scope: str) -> pd.DataFrame:
    scoped = landcover_df[landcover_df["Summary_scope"].astype(str) == scope].copy()
    if scoped.empty:
        available = sorted(landcover_df["Summary_scope"].dropna().astype(str).unique())
        raise ValueError(f"land-cover scope not found: {scope}; available={available}")

    rows: list[dict[str, Any]] = []
    for policy in POLICIES:
        product = policy["product"]
        class_codes = [int(x) for x in policy["class_codes"]]
        product_df = scoped[scoped["Product"].astype(str) == product].copy()
        if product_df.empty:
            raise ValueError(f"land-cover product not found in scope {scope}: {product}")
        product_df["Class_code"] = pd.to_numeric(product_df["Class_code"], errors="coerce").astype("Int64")
        product_df["Area_km2"] = pd.to_numeric(product_df["Area_km2"], errors="coerce")
        product_df["Product_total_area_km2"] = pd.to_numeric(product_df["Product_total_area_km2"], errors="coerce")
        total_area = float(product_df["Product_total_area_km2"].dropna().iloc[0])
        compatible_area = float(product_df[product_df["Class_code"].isin(class_codes)]["Area_km2"].sum())
        ratio = compatible_area / total_area if total_area > 0 else float("nan")
        rows.append(
            {
                "policy_id": policy["policy_id"],
                "role": policy["role"],
                "product": product,
                "scope": scope,
                "compatible_class_codes": ",".join(str(x) for x in class_codes),
                "compatible_classes": class_label(class_codes),
                "compatible_area_km2_current_oasis": compatible_area,
                "product_total_area_km2_current_oasis": total_area,
                "compatible_ratio": ratio,
                "compatible_percent": ratio * 100 if math.isfinite(ratio) else float("nan"),
                "interpretation": policy["interpretation"],
            }
        )
    return pd.DataFrame(rows)


def build_estimates(stage17_df: pd.DataFrame, policy_df: pd.DataFrame) -> pd.DataFrame:
    stage = stage17_df.copy()
    stage["constrained_area_km2"] = pd.to_numeric(stage["constrained_area_km2"], errors="coerce")
    stage = stage[stage["constrained_area_km2"].notna()].copy()
    rows: list[dict[str, Any]] = []
    for _, stage_row in stage.iterrows():
        for _, policy in policy_df.iterrows():
            ratio = float(policy["compatible_ratio"])
            base_area = float(stage_row["constrained_area_km2"])
            estimate = base_area * ratio if math.isfinite(ratio) else float("nan")
            rows.append(
                {
                    "stage17_label": stage_row.get("label"),
                    "stage17_constraint_suffix": stage_row.get("constraint_suffix"),
                    "stage17_area_km2": base_area,
                    "stage17_area_10k_km2": base_area / 10000.0,
                    "policy_id": policy["policy_id"],
                    "role": policy["role"],
                    "landcover_product": policy["product"],
                    "compatible_classes": policy["compatible_classes"],
                    "compatible_percent": policy["compatible_percent"],
                    "estimated_area_km2": estimate,
                    "estimated_area_10k_km2": estimate / 10000.0 if math.isfinite(estimate) else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def find_recommended(estimate_df: pd.DataFrame) -> dict[str, Any]:
    candidates = estimate_df[
        (estimate_df["stage17_label"].astype(str) == "main_river100_q1_up1000")
        & (estimate_df["policy_id"].astype(str) == "recommended_esa_core_compatible")
    ].copy()
    if candidates.empty:
        return {}
    row = candidates.iloc[0].to_dict()
    return {
        "stage17_label": row["stage17_label"],
        "policy_id": row["policy_id"],
        "estimated_area_km2": float(row["estimated_area_km2"]),
        "estimated_area_10k_km2": float(row["estimated_area_10k_km2"]),
        "compatible_percent": float(row["compatible_percent"]),
    }


def write_report(
    stage17_path: Path,
    landcover_path: Path,
    scope: str,
    policy_df: pd.DataFrame,
    estimate_df: pd.DataFrame,
    recommended: dict[str, Any],
) -> None:
    river100 = estimate_df[estimate_df["stage17_label"].astype(str) == "main_river100_q1_up1000"].copy()
    show = river100[
        [
            "policy_id",
            "role",
            "landcover_product",
            "compatible_classes",
            "compatible_percent",
            "estimated_area_10k_km2",
        ]
    ].copy()
    show["compatible_percent"] = show["compatible_percent"].map(lambda x: f"{x:.2f}%")
    show["estimated_area_10k_km2"] = show["estimated_area_10k_km2"].map(lambda x: f"{x:.2f}")

    policy_show = policy_df[
        ["policy_id", "role", "product", "compatible_classes", "compatible_percent"]
    ].copy()
    policy_show["compatible_percent"] = policy_show["compatible_percent"].map(lambda x: f"{x:.2f}%")

    rec_area = recommended.get("estimated_area_10k_km2")
    rec_pct = recommended.get("compatible_percent")
    rec_line = (
        f"- 推荐主口径: Stage17 `river100 + q1 + upstream1000` × ESA 核心兼容土地覆盖比例 "
        f"({rec_pct:.2f}%) = **约 {rec_area:.2f} 万 km²**。"
        if rec_area is not None and rec_pct is not None
        else "- 推荐主口径: NA"
    )

    lines = [
        "# Stage18 土地覆盖约束面积估算与下一步",
        "",
        f"- 生成时间: {now_iso()}",
        f"- Stage17 对比表: `{stage17_path}`",
        f"- 现有绿洲土地覆盖统计表: `{landcover_path}`",
        f"- 土地覆盖统计范围: `{scope}`",
        "",
        "## 结论",
        "",
        rec_line,
        "- 这个口径比直接使用 `river75` 更好解释：`river100` 保留为水文邻近尺度，面积收缩由独立的土地覆盖兼容性完成，不像人为调小河流缓冲区。",
        "- 当前结果是面积层面的约束估算，不是逐像元空间遮罩成果；最终出图还需要下载或导出可与 Stage17 对齐的土地覆盖兼容性栅格。",
        "",
        "## 土地覆盖兼容比例",
        "",
        policy_show.to_markdown(index=False),
        "",
        "## River100 主结果下的面积估算",
        "",
        show.to_markdown(index=False),
        "",
        "## 推荐解释",
        "",
        "- 主结果建议使用 `river100 + ESA WorldCover compatible core`。",
        "- 兼容类包括 Cropland、Built-up、Bare_or_sparse_vegetation；其中 Bare/sparse 保留是为了避免把干旱区裸地和稀疏植被绿洲廊道误删。",
        "- Tree/shrub/grass vegetation 不放进主结果，是为了回应山区植被可能被模型误判为绿洲适生区的问题。",
        "- Dynamic World 同类口径作为上界敏感性；Cropland+Built-up 作为过严下界参考。",
        "",
        "## 后续空间化步骤",
        "",
        "1. 获取或导出 ESA WorldCover / Dynamic World 的 30 arc-second 兼容性栅格，类别重采样必须使用 nearest 或先计算兼容比例后阈值化。",
        "2. 用 Stage17 `river100` 栅格逐 tile 叠加土地覆盖兼容性遮罩，输出真正的 `river100 + landcover` 空间结果。",
        "3. 重新汇总面积、出图和论文表述；面积层面应重点看是否落在 250-300 万 km²附近，若偏离，优先检查土地覆盖比例阈值和类别定义，而不是调水文缓冲。",
    ]
    atomic_write_text(REPORT_MD, "\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate Stage17 future oasis suitability area after land-cover compatibility constraints.")
    parser.add_argument("--stage17-comparison", default=str(DEFAULT_STAGE17_COMPARISON))
    parser.add_argument("--landcover-summary", default=str(DEFAULT_LANDCOVER_SUMMARY))
    parser.add_argument("--scope", default="Global_combined_current_inputs")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stage17_path = Path(args.stage17_comparison)
    landcover_path = Path(args.landcover_summary)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    stage17_df = read_csv_checked(stage17_path, "Stage17 comparison CSV")
    landcover_df = read_csv_checked(landcover_path, "Land-cover summary CSV")
    policy_df = build_policy_ratios(landcover_df, args.scope)
    estimate_df = build_estimates(stage17_df, policy_df)
    recommended = find_recommended(estimate_df)

    atomic_write_csv(POLICY_CSV, policy_df)
    atomic_write_csv(ESTIMATE_CSV, estimate_df)
    summary = {
        "status": "success",
        "generated_at": now_iso(),
        "stage17_comparison": str(stage17_path),
        "landcover_summary": str(landcover_path),
        "scope": args.scope,
        "policy_csv": str(POLICY_CSV),
        "estimate_csv": str(ESTIMATE_CSV),
        "report_md": str(REPORT_MD),
        "recommended": recommended,
        "warning": "This is an area-level estimate, not a pixel-level land-cover mask result.",
    }
    atomic_write_json(SUMMARY_JSON, summary)
    write_report(stage17_path, landcover_path, args.scope, policy_df, estimate_df, recommended)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
