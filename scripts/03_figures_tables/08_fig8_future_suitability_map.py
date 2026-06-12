from __future__ import annotations

import json
import logging
import math
import sys
import traceback
import colorsys
from datetime import datetime
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Circle, Ellipse, Polygon, Rectangle
from PIL import Image
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from pyproj import Transformer
from shapely.geometry import Point as ShapelyPoint
from shapely.geometry import Polygon as ShapelyPolygon


SCRIPT_NAME = Path(__file__).name
PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGE_DIR = PROJECT_ROOT / "outputs" / "stage58_q10_future_suitability_map"
FIG_DIR = STAGE_DIR / "figures"
TABLE_DIR = STAGE_DIR / "tables"
LOG_DIR = STAGE_DIR / "logs"

FIGURE_ID = "Fig8_stage58_v49"
FIGURE_STEM = "fig_stage58_q10_future_suitability_map_v49"
LAYOUT_FAMILY = "web-mercator-circular-magnifier-round-connected"


def _scale_hex_saturation(hex_color: str, factor: float) -> str:
    rgb = tuple(int(hex_color[i : i + 2], 16) / 255.0 for i in (1, 3, 5))
    hue, lightness, saturation = colorsys.rgb_to_hls(*rgb)
    adjusted_rgb = colorsys.hls_to_rgb(hue, lightness, min(1.0, saturation * factor))
    return "#" + "".join(f"{round(channel * 255):02x}" for channel in adjusted_rgb)

Q10_SUMMARY_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "stage36_hydrology_landcover_sensitivity"
    / "q10cms"
    / "tables"
    / "stage20_landcover_spatial_constraint_selected10_hgb_hydrorivers_q10cms_landcover_summary.csv"
)
COUNTRIES_SHP = (
    PROJECT_ROOT
    / "outputs"
    / "stage28_stage20_landcover_distribution_map"
    / "reference"
    / "ne_110m_admin_0_countries"
    / "ne_110m_admin_0_countries.shp"
)
DISPLAY_CRS = "EPSG:3857"
WGS84_CRS = "EPSG:4326"
WGS84_TO_WEB = Transformer.from_crs(WGS84_CRS, DISPLAY_CRS, always_xy=True)
WEB_TO_WGS84 = Transformer.from_crs(DISPLAY_CRS, WGS84_CRS, always_xy=True)

DEFAULT_BASEMAP_NAME = "default_global_basemap_stage29b_natural_earth_gray_relief.png"


def _find_default_basemap_png() -> Path:
    candidates = sorted((PROJECT_ROOT / "assets").rglob(DEFAULT_BASEMAP_NAME))
    for candidate in candidates:
        if "Stage29b" in str(candidate):
            return candidate
    if candidates:
        return candidates[0]
    return PROJECT_ROOT / "assets" / DEFAULT_BASEMAP_NAME


DEFAULT_BASEMAP_PNG = _find_default_basemap_png()
DEFAULT_BASEMAP_SVG = DEFAULT_BASEMAP_PNG.with_suffix(".svg")
DRYLAND_AI_SHP = Path(r"C:\Users\linjingwu\Desktop\会议相关\世界绿洲合并\世界绿洲\全球干旱区\AI0-0.65干旱区.shp")

LC_THRESHOLD = 50
HIGH_THRESHOLD = 75
OPTIMAL_THRESHOLD = 90
CLASS_BOUNDS = [LC_THRESHOLD, HIGH_THRESHOLD, OPTIMAL_THRESHOLD, 100.01]
CLASS_LABELS = ["Suitable", "High", "Optimal"]
CLASS_COLORS = ["#0072b2", "#d55e00", "#009e73"]
CLASS_FOCUS_PANELS = [
    ("Suitable", 50.0, 75.0, "#0072b2"),
    ("High", 75.0, 90.0, "#d55e00"),
    ("Optimal", 90.0, 100.0, "#009e73"),
]
MINOR_CLASS_ZOOM_WINDOW_LON_DEG = 7.5
MINOR_CLASS_ZOOM_WINDOW_LAT_DEG = 6.0
MINOR_CLASS_ZOOM_SELECTION_SHAPE = "circle"
ZOOM_CONNECTOR_TARGET_OUTSET_PX = 0.8
ZOOM_CONNECTOR_CIRCLE_CLIP_OUTSET_PX = 1.6
ZOOM_CONTENT_CLIP_RADIUS_AXES = 0.486


def _zoom_extent_from_center(lon: float, lat: float) -> tuple[float, float, float, float]:
    half_lon = MINOR_CLASS_ZOOM_WINDOW_LON_DEG / 2.0
    half_lat = MINOR_CLASS_ZOOM_WINDOW_LAT_DEG / 2.0
    return (lon - half_lon, lon + half_lon, lat - half_lat, lat + half_lat)


MINOR_CLASS_ZOOM_CANDIDATE_CENTERS = {
    "Suitable": [
        (-112.5, 35.0),
        (-114.5, 36.0),
        (-110.5, 34.5),
        (13.0, 36.0),
        (15.5, 37.0),
        (133.0, -29.0),
    ],
    "High": [
        (45.5, 31.0),
        (48.0, 30.0),
        (51.5, 31.5),
        (58.0, 32.5),
        (76.0, 36.0),
        (-112.5, 35.0),
    ],
}
MINOR_CLASS_ZOOM_CANDIDATES = {
    label: [_zoom_extent_from_center(lon, lat) for lon, lat in centers]
    for label, centers in MINOR_CLASS_ZOOM_CANDIDATE_CENTERS.items()
}
MINOR_CLASS_ZOOM_PANELS = [
    {
        "target_label": "Suitable",
        "low": 50.0,
        "high": 75.0,
        "axes_bounds": [0.251, 0.541, 0.112, 0.246],
        "connector_source_angles_deg": [67.3, -67.3],
        "connector_target_axes": [[0.18, 0.907], [0.18, 0.093]],
        "color": "#0072b2",
        "label": "Suitable",
    },
    {
        "target_label": "High",
        "low": 75.0,
        "high": 90.0,
        "axes_bounds": [0.690, 0.165, 0.112, 0.246],
        "connector_source_angles_deg": [-16.4, -95.6],
        "connector_target_axes": [[0.544, 0.981], [0.016, 0.529]],
        "color": "#d55e00",
        "label": "High",
    },
]
DRYLAND_CLASS_ORDER = ["hyperarid", "arid", "semiarid", "dry_subhumid"]
DRYLAND_SATURATION_FACTOR = 1.10
DRYLAND_CLASS_STYLES = {
    "hyperarid": {
        "label": "Hyperarid",
        "color": _scale_hex_saturation("#a66f5f", DRYLAND_SATURATION_FACTOR),
        "hix_desc": "Hyperarid",
    },
    "arid": {
        "label": "Arid",
        "color": _scale_hex_saturation("#c89568", DRYLAND_SATURATION_FACTOR),
        "hix_desc": "Arid",
    },
    "semiarid": {
        "label": "Semiarid",
        "color": _scale_hex_saturation("#d8bd7a", DRYLAND_SATURATION_FACTOR),
        "hix_desc": "Semiarid",
    },
    "dry_subhumid": {
        "label": "Dry Subhumid",
        "color": _scale_hex_saturation("#eadbad", DRYLAND_SATURATION_FACTOR),
        "hix_desc": "Dry Subhumid",
    },
}
DRYLAND_ZONE_TO_CLASS = {5: "hyperarid", 4: "arid", 3: "semiarid", 2: "dry_subhumid"}
DRYLAND_DESC_TO_CLASS = {
    "hyperarid": "hyperarid",
    "arid": "arid",
    "semiarid": "semiarid",
    "dry subhumid": "dry_subhumid",
}
DRYLAND_ALPHA = 0.52
OCEAN_COLOR_HEX = "#d9e5e8"
LAND_COLOR_HEX = "#fbfbf7"
CONTINENT_LINE_HEX = "#adb7b9"
OCEAN_RGB = np.array([228, 240, 247], dtype=np.float32)
DOWNSAMPLE = 3
DEFAULT_BASEMAP_EXTENT = (-180.0, 180.0, -55.0, 72.0)
GLOBAL_EXTENT = (-135.0, 155.0, -47.0, 62.0)


def _setup_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(FIGURE_STEM)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(LOG_DIR / f"{FIGURE_STEM}.log", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def _write_status(status: str, **extra: object) -> None:
    payload = {
        "status": status,
        "figure_id": FIGURE_ID,
        "figure_stem": FIGURE_STEM,
        "layout_family": LAYOUT_FAMILY,
        "script": SCRIPT_NAME,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "word_insertion_allowed": False,
        "user_confirmed_final": False,
        "candidate_only": True,
        "manual_review_required": True,
    }
    payload.update(extra)
    (STAGE_DIR / f"{FIGURE_STEM}_status.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _atomic_save(fig: mpl.figure.Figure, path: Path, *, transparent: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp" + path.suffix)
    facecolor = "none" if transparent else "white"
    fig.savefig(
        tmp,
        dpi=450,
        bbox_inches="tight",
        pad_inches=0.055,
        transparent=transparent,
        facecolor=facecolor,
    )
    if not transparent and path.suffix.lower() == ".png":
        with Image.open(tmp) as saved:
            rgba = saved.convert("RGBA")
        white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        white.alpha_composite(rgba)
        white.save(tmp)
    tmp.replace(path)


def _resolve_path(value: str) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def _intersects(bounds: tuple[float, float, float, float], extent: tuple[float, float, float, float]) -> bool:
    left, right, bottom, top = bounds
    xmin, xmax, ymin, ymax = extent
    return not (right < xmin or left > xmax or top < ymin or bottom > ymax)


def _read_class_patch(path: Path) -> tuple[np.ma.MaskedArray, tuple[float, float, float, float], dict[str, int]]:
    with rasterio.open(path) as src:
        out_h = max(1, math.ceil(src.height / DOWNSAMPLE))
        out_w = max(1, math.ceil(src.width / DOWNSAMPLE))
        arr = src.read(
            1,
            out_shape=(out_h, out_w),
            masked=True,
            resampling=Resampling.nearest,
        ).astype("float32")
        data = np.asarray(arr)
        mask = np.ma.getmaskarray(arr)
        if src.nodata is not None:
            mask = mask | np.isclose(data, float(src.nodata))
        mask = mask | (data < LC_THRESHOLD)
        masked = np.ma.array(data, mask=mask)
        bounds = (float(src.bounds.left), float(src.bounds.right), float(src.bounds.bottom), float(src.bounds.top))

    values = masked.compressed()
    counts = {
        "cells_50_74_downsampled": int(np.sum((values >= 50) & (values < 75))),
        "cells_75_89_downsampled": int(np.sum((values >= 75) & (values < 90))),
        "cells_ge90_downsampled": int(np.sum(values >= 90)),
    }
    return masked, bounds, counts


def _raster_bounds(path: Path) -> tuple[float, float, float, float]:
    with rasterio.open(path) as src:
        return (float(src.bounds.left), float(src.bounds.right), float(src.bounds.bottom), float(src.bounds.top))


def _load_country_layer(logger: logging.Logger):
    import geopandas as gpd

    if not COUNTRIES_SHP.exists():
        raise FileNotFoundError(f"Country boundary file is missing: {COUNTRIES_SHP}")
    world = gpd.read_file(COUNTRIES_SHP)
    if world.crs is not None and str(world.crs).lower() not in {"epsg:4326", "wgs84"}:
        world = world.to_crs("EPSG:4326")
    logger.info("Loaded country layer with %s features.", len(world))
    return world


def _load_ai_dryland_layer(logger: logging.Logger):
    import geopandas as gpd

    if not DRYLAND_AI_SHP.exists():
        raise FileNotFoundError(f"Local dryland AI shapefile is missing: {DRYLAND_AI_SHP}")
    dryland = gpd.read_file(DRYLAND_AI_SHP)
    if dryland.empty:
        raise ValueError(f"Local dryland AI shapefile has no features: {DRYLAND_AI_SHP}")
    if dryland.crs is not None and str(dryland.crs).lower() not in {"epsg:4326", "wgs84"}:
        dryland = dryland.to_crs("EPSG:4326")
    dryland = dryland[dryland.geometry.notna() & ~dryland.geometry.is_empty].copy()
    if dryland.empty:
        raise ValueError("Local dryland AI shapefile has no valid geometries after filtering.")
    invalid_count = int((~dryland.geometry.is_valid).sum())
    if invalid_count:
        logger.info("Repairing %s invalid dryland geometries with buffer(0).", invalid_count)
        dryland["geometry"] = dryland.geometry.buffer(0)
        dryland = dryland[dryland.geometry.notna() & ~dryland.geometry.is_empty].copy()

    dryland["ai_class"] = None
    if "HIX_ZONE" in dryland.columns:
        zones = pd.to_numeric(dryland["HIX_ZONE"], errors="coerce")
        dryland["ai_class"] = zones.round().astype("Int64").map(DRYLAND_ZONE_TO_CLASS)
    if dryland["ai_class"].isna().any() and "HIX_DESC" in dryland.columns:
        desc_classes = dryland["HIX_DESC"].astype(str).str.strip().str.lower().map(DRYLAND_DESC_TO_CLASS)
        dryland["ai_class"] = dryland["ai_class"].fillna(desc_classes)

    missing_classes = sorted(str(v) for v in dryland.loc[dryland["ai_class"].isna(), "HIX_DESC"].unique())
    if missing_classes:
        raise ValueError(f"Could not map all dryland AI classes from shapefile fields: {missing_classes}")

    dryland["ai_label"] = dryland["ai_class"].map(lambda key: DRYLAND_CLASS_STYLES[str(key)]["label"])
    dryland["plot_color"] = dryland["ai_class"].map(lambda key: DRYLAND_CLASS_STYLES[str(key)]["color"])
    logger.info("Loaded local dryland AI shapefile with %s features: %s", len(dryland), DRYLAND_AI_SHP)
    return dryland


def _project_point(lon: float, lat: float) -> tuple[float, float]:
    x, y = WGS84_TO_WEB.transform(lon, lat)
    return float(x), float(y)


def _project_extent(extent: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    xmin, xmax, ymin, ymax = extent
    x0, y0 = _project_point(xmin, ymin)
    x1, y1 = _project_point(xmax, ymax)
    return (x0, x1, y0, y1)


def _source_circle_from_extent(
    extent: tuple[float, float, float, float]
) -> tuple[float, float, float]:
    x0, x1, y0, y1 = _project_extent(extent)
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    radius = min(abs(x1 - x0), abs(y1 - y0)) / 2.0
    return cx, cy, radius


def _format_lon(value: float) -> str:
    if value == 0:
        return "0"
    return f"{abs(int(value))}°{'W' if value < 0 else 'E'}"


def _format_lat(value: float) -> str:
    if value == 0:
        return "0"
    return f"{abs(int(value))}°{'S' if value < 0 else 'N'}"


def _set_lonlat_ticks(ax: plt.Axes, extent: tuple[float, float, float, float]) -> None:
    xmin, xmax, ymin, ymax = extent
    lon_candidates = [-120, -80, -40, 0, 40, 80, 120]
    lat_candidates = [-40, -20, 0, 20, 40, 60]
    lon_ticks = [lon for lon in lon_candidates if xmin <= lon <= xmax]
    lat_ticks = [lat for lat in lat_candidates if ymin <= lat <= ymax]
    ax.set_xticks([_project_point(lon, 0)[0] for lon in lon_ticks])
    ax.set_xticklabels([_format_lon(lon) for lon in lon_ticks])
    ax.set_yticks([_project_point(0, lat)[1] for lat in lat_ticks])
    ax.set_yticklabels([_format_lat(lat) for lat in lat_ticks])


def _load_continent_layer(logger: logging.Logger):
    world = _load_country_layer(logger)
    world = world[world.geometry.notna() & ~world.geometry.is_empty].copy()
    world = world[world["CONTINENT"].notna()].copy()
    world = world[~world["CONTINENT"].astype(str).str.contains("Seven seas", case=False, na=False)].copy()
    invalid_count = int((~world.geometry.is_valid).sum())
    if invalid_count:
        logger.info("Repairing %s invalid country geometries before continent dissolve.", invalid_count)
        world["geometry"] = world.geometry.buffer(0)
        world = world[world.geometry.notna() & ~world.geometry.is_empty].copy()

    continents = world[["CONTINENT", "geometry"]].dissolve(by="CONTINENT", as_index=False)
    continents = continents[continents.geometry.notna() & ~continents.geometry.is_empty].copy()
    logger.info(
        "Built continent outline layer from %s countries into %s dissolved continent features.",
        len(world),
        len(continents),
    )
    return continents


def _plot_base(
    ax: plt.Axes,
    continents,
    continents_web,
    extent: tuple[float, float, float, float],
    *,
    label: str | None = None,
    show_ticks: bool = False,
) -> None:
    ax.set_facecolor(OCEAN_COLOR_HEX)
    x0, x1, y0, y1 = _project_extent(extent)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal", adjustable="box")
    subset = _subset_layer_for_extent(continents, extent)
    if not subset.empty:
        part = continents_web.loc[subset.index]
        part.plot(
            ax=ax,
            facecolor=LAND_COLOR_HEX,
            edgecolor="none",
            linewidth=0,
            alpha=1.0,
            zorder=0,
        )
        part.boundary.plot(
            ax=ax,
            color=CONTINENT_LINE_HEX,
            linewidth=0.46,
            alpha=0.78,
            zorder=1,
        )
    ax.grid(color="#cfd8d6", linewidth=0.34, linestyle="-", alpha=0.30, zorder=1.2)
    ax.tick_params(labelsize=6.8, colors="#48515a", length=2.5, width=0.58)
    if show_ticks:
        _set_lonlat_ticks(ax, extent)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#bfc8c0")
        spine.set_linewidth(0.58)
    if label:
        ax.text(
            0.02,
            0.96,
            label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6.7,
            fontweight="bold",
            color="#24313b",
            bbox={
                "boxstyle": "round,pad=0.15,rounding_size=0.035",
                "facecolor": "white",
                "edgecolor": "#cbd6cf",
                "linewidth": 0.42,
                "alpha": 0.9,
            },
            zorder=8,
        )


def _subset_layer_for_extent(layer, extent: tuple[float, float, float, float]):
    xmin, xmax, ymin, ymax = extent
    bounds = layer.bounds
    return layer[
        (bounds["maxx"] >= xmin)
        & (bounds["minx"] <= xmax)
        & (bounds["maxy"] >= ymin)
        & (bounds["miny"] <= ymax)
    ]


def _plot_ai_dryland(
    ax: plt.Axes,
    dryland,
    dryland_web,
    extent: tuple[float, float, float, float],
    *,
    inset: bool = False,
) -> None:
    subset = _subset_layer_for_extent(dryland, extent)
    if subset.empty:
        return
    for class_key in DRYLAND_CLASS_ORDER:
        part = dryland_web.loc[subset[subset["ai_class"] == class_key].index]
        if part.empty:
            continue
        part.plot(
            ax=ax,
            facecolor=DRYLAND_CLASS_STYLES[class_key]["color"],
            edgecolor="none",
            linewidth=0,
            alpha=DRYLAND_ALPHA if not inset else min(DRYLAND_ALPHA + 0.08, 0.55),
            zorder=2,
        )


def _plot_raster_patches(
    ax: plt.Axes,
    patches: list[tuple[np.ma.MaskedArray, tuple[float, float, float, float]]],
    cmap: ListedColormap,
    norm: BoundaryNorm,
    extent: tuple[float, float, float, float],
    *,
    inset: bool = False,
) -> None:
    for data, bounds in patches:
        if not _intersects(bounds, extent):
            continue
        left, right, bottom, top = bounds
        web_extent = _project_extent((left, right, bottom, top))
        ax.imshow(
            data,
            extent=web_extent,
            origin="upper",
            cmap=cmap,
            norm=norm,
            interpolation="nearest",
            alpha=0.96 if not inset else 0.98,
            zorder=4,
        )


def _plot_raster_class_focus(
    ax: plt.Axes,
    patches: list[tuple[np.ma.MaskedArray, tuple[float, float, float, float]]],
    extent: tuple[float, float, float, float],
    low: float,
    high: float,
    color: str,
) -> None:
    class_cmap = ListedColormap([color])
    class_cmap.set_bad((0, 0, 0, 0))
    for data, bounds in patches:
        if not _intersects(bounds, extent):
            continue
        values = np.asarray(data)
        upper_mask = values <= high if high >= 100.0 else values < high
        mask = np.ma.getmaskarray(data) | (values < low) | ~upper_mask
        if not bool((~mask).any()):
            continue
        focused = np.ma.array(np.ones(data.shape, dtype=np.float32), mask=mask)
        left, right, bottom, top = bounds
        web_extent = _project_extent((left, right, bottom, top))
        ax.imshow(
            focused,
            extent=web_extent,
            origin="upper",
            cmap=class_cmap,
            vmin=0,
            vmax=1,
            interpolation="nearest",
            alpha=0.98,
            zorder=4,
        )


def _class_focus_summary(
    patches: list[tuple[np.ma.MaskedArray, tuple[float, float, float, float]]]
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for label, low, high, color in CLASS_FOCUS_PANELS:
        cell_count = 0
        for data, _bounds in patches:
            values = data.compressed()
            upper_mask = values <= high if high >= 100.0 else values < high
            cell_count += int(np.sum((values >= low) & upper_mask))
        rows.append(
            {
                "panel": label,
                "low_inclusive_pct": low,
                "upper_bound_pct": high,
                "upper_bound_rule": "<= upper_bound_pct" if high >= 100.0 else "< upper_bound_pct",
                "color": color,
                "visible_cells_downsampled": cell_count,
                "extent_lonlat": GLOBAL_EXTENT,
            }
        )
    total = sum(int(row["visible_cells_downsampled"]) for row in rows)
    for row in rows:
        row["share_pct"] = (
            100.0 * int(row["visible_cells_downsampled"]) / total if total > 0 else 0.0
        )
    return rows


def _count_class_cells_in_extent(
    patches: list[tuple[np.ma.MaskedArray, tuple[float, float, float, float]]],
    extent: tuple[float, float, float, float],
    low: float,
    high: float,
    *,
    circular: bool = False,
) -> int:
    lon_min, lon_max, lat_min, lat_max = extent
    circle_x, circle_y, circle_radius = _source_circle_from_extent(extent) if circular else (0.0, 0.0, 0.0)
    cell_count = 0
    for data, bounds in patches:
        if not _intersects(bounds, extent):
            continue
        left, right, bottom, top = bounds
        values = np.ma.getdata(data)
        mask = np.ma.getmaskarray(data)
        rows, cols = values.shape
        lon_centers = left + (np.arange(cols, dtype=float) + 0.5) * (right - left) / cols
        lat_centers = top - (np.arange(rows, dtype=float) + 0.5) * (top - bottom) / rows
        col_mask = (lon_centers >= lon_min) & (lon_centers <= lon_max)
        row_mask = (lat_centers >= lat_min) & (lat_centers <= lat_max)
        if not bool(col_mask.any() and row_mask.any()):
            continue
        sub_values = values[np.ix_(row_mask, col_mask)]
        sub_mask = mask[np.ix_(row_mask, col_mask)]
        upper_mask = sub_values <= high if high >= 100.0 else sub_values < high
        valid_mask = (~sub_mask) & (sub_values >= low) & upper_mask
        if circular:
            lon_grid, lat_grid = np.meshgrid(lon_centers[col_mask], lat_centers[row_mask])
            x_grid, y_grid = WGS84_TO_WEB.transform(lon_grid, lat_grid)
            circle_mask = ((x_grid - circle_x) ** 2 + (y_grid - circle_y) ** 2) <= circle_radius**2
            valid_mask = valid_mask & circle_mask
        cell_count += int(np.sum(valid_mask))
    return cell_count


def _count_suitability_class_mix_in_extent(
    patches: list[tuple[np.ma.MaskedArray, tuple[float, float, float, float]]],
    extent: tuple[float, float, float, float],
    *,
    circular: bool = False,
) -> dict[str, int]:
    counts = {
        "cells_50_74_downsampled": _count_class_cells_in_extent(
            patches, extent, 50.0, 75.0, circular=circular
        ),
        "cells_75_89_downsampled": _count_class_cells_in_extent(
            patches, extent, 75.0, 90.0, circular=circular
        ),
        "cells_ge90_downsampled": _count_class_cells_in_extent(
            patches, extent, 90.0, 100.0, circular=circular
        ),
    }
    counts["nonzero_suitability_class_count"] = sum(1 for value in counts.values() if int(value) > 0)
    counts["total_suitability_cells_downsampled"] = sum(int(value) for value in counts.values())
    return counts


def _minor_class_zoom_summary(
    patches: list[tuple[np.ma.MaskedArray, tuple[float, float, float, float]]]
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for panel in MINOR_CLASS_ZOOM_PANELS:
        target_label = str(panel["target_label"])
        low = float(panel["low"])
        high = float(panel["high"])
        candidates = MINOR_CLASS_ZOOM_CANDIDATES[target_label]
        scored_candidates = []
        for candidate in candidates:
            class_mix = _count_suitability_class_mix_in_extent(patches, candidate, circular=True)
            scored_candidates.append(
                {
                    "extent_lonlat": list(candidate),
                    "target_cells_downsampled": _count_class_cells_in_extent(
                        patches, candidate, low, high, circular=True
                    ),
                    **class_mix,
                }
            )
        best = max(
            scored_candidates,
            key=lambda row: (
                int(row["nonzero_suitability_class_count"]),
                int(row["target_cells_downsampled"]),
            ),
        )
        rows.append(
            {
                "panel": target_label,
                "label": str(panel["label"]),
                "low_inclusive_pct": low,
                "upper_bound_pct": high,
                "upper_bound_rule": "<= upper_bound_pct" if high >= 100.0 else "< upper_bound_pct",
                "color": str(panel["color"]),
                "axes_bounds": list(panel["axes_bounds"]),
                "connector_source_angles_deg": list(panel["connector_source_angles_deg"]),
                "connector_target_axes": list(panel["connector_target_axes"]),
                "source_selection_shape": MINOR_CLASS_ZOOM_SELECTION_SHAPE,
                "source_window_lon_deg": MINOR_CLASS_ZOOM_WINDOW_LON_DEG,
                "source_window_lat_deg": MINOR_CLASS_ZOOM_WINDOW_LAT_DEG,
                "extent_lonlat": best["extent_lonlat"],
                "source_circle_radius_web_m": float(
                    _source_circle_from_extent(tuple(float(value) for value in best["extent_lonlat"]))[2]
                ),
                "target_cells_downsampled": int(best["target_cells_downsampled"]),
                "cells_50_74_downsampled": int(best["cells_50_74_downsampled"]),
                "cells_75_89_downsampled": int(best["cells_75_89_downsampled"]),
                "cells_ge90_downsampled": int(best["cells_ge90_downsampled"]),
                "nonzero_suitability_class_count": int(best["nonzero_suitability_class_count"]),
                "total_suitability_cells_downsampled": int(best["total_suitability_cells_downsampled"]),
                "candidate_scores": scored_candidates,
            }
        )
    return rows


def _draw_selected_source_circle(
    ax: plt.Axes,
    cx: float,
    cy: float,
    radius: float,
    color: str,
) -> None:
    shadow_offset = radius * 0.035
    ax.add_patch(
        Circle(
            (cx + shadow_offset, cy - shadow_offset),
            radius,
            transform=ax.transData,
            facecolor="none",
            edgecolor=(0.08, 0.11, 0.13, 0.23),
            linewidth=1.08,
            zorder=8,
        )
    )
    ax.add_patch(
        Circle(
            (cx, cy),
            radius * 1.055,
            transform=ax.transData,
            facecolor="none",
            edgecolor=mpl.colors.to_rgba(color, 0.28),
            linewidth=1.65,
            zorder=8.2,
        )
    )
    ax.add_patch(
        Circle(
            (cx, cy),
            radius * 1.025,
            transform=ax.transData,
            facecolor="none",
            edgecolor=mpl.colors.to_rgba(color, 0.36),
            linewidth=0.92,
            zorder=8.4,
        )
    )
    ax.add_patch(
        Circle(
            (cx, cy),
            radius,
            transform=ax.transData,
            facecolor=mpl.colors.to_rgba(color, 0.055),
            edgecolor=(1, 1, 1, 0.88),
            linewidth=1.38,
            zorder=9,
        )
    )
    ax.add_patch(
        Circle(
            (cx - shadow_offset * 0.24, cy + shadow_offset * 0.24),
            radius * 0.972,
            transform=ax.transData,
            facecolor="none",
            edgecolor=(1, 1, 1, 0.86),
            linewidth=0.62,
            zorder=9.4,
        )
    )
    ax.add_patch(
        Circle(
            (cx, cy),
            radius,
            transform=ax.transData,
            facecolor="none",
            edgecolor=color,
            linewidth=0.59,
            alpha=0.98,
            zorder=10,
        )
    )


def _display_width_to_figure_polygons(
    fig: plt.Figure,
    source_display: np.ndarray,
    target_display: np.ndarray,
    source_width_pt: float,
    target_width_pt: float,
    *,
    clip_center_display: np.ndarray | None = None,
    clip_radius_px: float | None = None,
) -> list[np.ndarray]:
    direction = target_display - source_display
    length = float(np.hypot(direction[0], direction[1]))
    if length <= 0:
        return []
    normal = np.array([-direction[1], direction[0]], dtype=float) / length
    px_per_pt = fig.dpi / 72.0
    source_half_width = source_width_pt * px_per_pt * 0.5
    target_half_width = target_width_pt * px_per_pt * 0.5
    display_vertices = np.vstack(
        [
            source_display + normal * source_half_width,
            target_display + normal * target_half_width,
            target_display - normal * target_half_width,
            source_display - normal * source_half_width,
        ]
    )
    if clip_center_display is None or clip_radius_px is None:
        return [fig.transFigure.inverted().transform(display_vertices)]

    connector_poly = ShapelyPolygon(display_vertices)
    if connector_poly.is_empty or not connector_poly.is_valid:
        return []
    clip_disk = ShapelyPoint(
        float(clip_center_display[0]),
        float(clip_center_display[1]),
    ).buffer(float(clip_radius_px), resolution=96)
    clipped = connector_poly.difference(clip_disk)
    if clipped.is_empty:
        return []

    geoms = getattr(clipped, "geoms", [clipped])
    figure_polygons: list[np.ndarray] = []
    for geom in geoms:
        if geom.geom_type != "Polygon" or geom.is_empty:
            continue
        coords = np.asarray(geom.exterior.coords, dtype=float)
        if coords.shape[0] < 4:
            continue
        figure_polygons.append(fig.transFigure.inverted().transform(coords))
    return figure_polygons


def _preferred_tangent_point_on_circle(
    source_display: np.ndarray,
    center_display: np.ndarray,
    radius_px: float,
    preferred_display: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    source_vector = source_display - center_display
    source_distance = float(np.hypot(source_vector[0], source_vector[1]))
    if source_distance <= radius_px:
        return None
    source_angle = math.atan2(float(source_vector[1]), float(source_vector[0]))
    tangent_delta = math.acos(max(-1.0, min(1.0, radius_px / source_distance)))
    candidates: list[tuple[float, np.ndarray, np.ndarray]] = []
    for sign in (-1.0, 1.0):
        tangent_angle = source_angle + sign * tangent_delta
        radial_unit = np.array(
            [math.cos(tangent_angle), math.sin(tangent_angle)],
            dtype=float,
        )
        point = center_display + radial_unit * radius_px
        candidates.append(
            (
                float(np.hypot(*(point - preferred_display))),
                point,
                radial_unit,
            )
        )
    _distance, point, radial_unit = min(candidates, key=lambda item: item[0])
    return point, radial_unit


def _draw_tapered_connector(
    ax: plt.Axes,
    zoom_ax: plt.Axes,
    source_xy: tuple[float, float],
    target_axes_xy: tuple[float, float],
    color: str,
) -> None:
    fig = ax.figure
    source_display = np.asarray(ax.transData.transform(source_xy), dtype=float)
    target_axes = np.asarray(target_axes_xy, dtype=float)
    center_axes = np.array([0.5, 0.5], dtype=float)
    center_display = np.asarray(zoom_ax.transAxes.transform(center_axes), dtype=float)
    target_hint_display = np.asarray(zoom_ax.transAxes.transform(target_axes), dtype=float)
    target_vector_display = target_hint_display - center_display
    target_vector_length = float(
        np.hypot(target_vector_display[0], target_vector_display[1])
    )
    if target_vector_length <= 0:
        return
    zoom_bbox = zoom_ax.get_window_extent()
    circle_radius_px = min(float(zoom_bbox.width), float(zoom_bbox.height)) * 0.5
    tangent = _preferred_tangent_point_on_circle(
        source_display,
        center_display,
        circle_radius_px,
        target_hint_display,
    )
    if tangent is None:
        target_unit = target_vector_display / target_vector_length
        target_edge_display = center_display + target_unit * circle_radius_px
    else:
        target_edge_display, target_unit = tangent
    target_display = target_edge_display + target_unit * min(
        ZOOM_CONNECTOR_TARGET_OUTSET_PX, circle_radius_px * 0.04
    )
    for source_width_pt, target_width_pt, facecolor, zorder in [
        (0.70, 1.32, (1, 1, 1, 0.56), 15.2),
        (0.30, 0.82, mpl.colors.to_rgba(color, 0.92), 15.8),
    ]:
        figure_polygons = _display_width_to_figure_polygons(
            fig,
            source_display,
            target_display,
            source_width_pt=source_width_pt,
            target_width_pt=target_width_pt,
            clip_center_display=center_display,
            clip_radius_px=circle_radius_px + ZOOM_CONNECTOR_CIRCLE_CLIP_OUTSET_PX,
        )
        for vertices in figure_polygons:
            fig.add_artist(
                Polygon(
                    vertices,
                    closed=True,
                    transform=fig.transFigure,
                    facecolor=facecolor,
                    edgecolor="none",
                    zorder=zorder,
                    clip_on=False,
                )
            )


def _square_inset_axes_bounds(
    ax: plt.Axes, bounds: list[float] | tuple[float, float, float, float]
) -> list[float]:
    x, y, width, height = [float(value) for value in bounds]
    parent_bbox = ax.get_position()
    if parent_bbox.width <= 0 or parent_bbox.height <= 0:
        return [x, y, width, height]
    square_width = height * float(parent_bbox.height) / float(parent_bbox.width)
    center_x = x + width * 0.5
    return [center_x - square_width * 0.5, y, square_width, height]


def _draw_zoom_frame_on_figure(zoom_ax: plt.Axes, color: str) -> None:
    fig = zoom_ax.figure
    fig.canvas.draw()
    center_display = np.asarray(zoom_ax.transAxes.transform((0.5, 0.5)), dtype=float)
    zoom_bbox = zoom_ax.get_window_extent()
    radius_px = min(float(zoom_bbox.width), float(zoom_bbox.height)) * 0.5
    center_fig = fig.transFigure.inverted().transform(center_display)
    frame_width_fig = 2.0 * radius_px / float(fig.bbox.width)
    frame_height_fig = 2.0 * radius_px / float(fig.bbox.height)
    fig.add_artist(
        Ellipse(
            center_fig,
            width=frame_width_fig,
            height=frame_height_fig,
            transform=fig.transFigure,
            facecolor="none",
            edgecolor=color,
            linewidth=1.05,
            alpha=0.98,
            zorder=13.2,
            clip_on=False,
        )
    )


def _draw_zoom_edge_cleanup_on_figure(zoom_ax: plt.Axes) -> None:
    fig = zoom_ax.figure
    fig.canvas.draw()
    center_display = np.asarray(zoom_ax.transAxes.transform((0.5, 0.5)), dtype=float)
    zoom_bbox = zoom_ax.get_window_extent()
    radius_px = min(float(zoom_bbox.width), float(zoom_bbox.height)) * 0.5
    cleanup_linewidth_pt = 2.15
    cleanup_linewidth_px = cleanup_linewidth_pt * float(fig.dpi) / 72.0
    cleanup_radius_px = radius_px + cleanup_linewidth_px * 0.54
    center_fig = fig.transFigure.inverted().transform(center_display)
    frame_width_fig = 2.0 * cleanup_radius_px / float(fig.bbox.width)
    frame_height_fig = 2.0 * cleanup_radius_px / float(fig.bbox.height)
    fig.add_artist(
        Ellipse(
            center_fig,
            width=frame_width_fig,
            height=frame_height_fig,
            transform=fig.transFigure,
            facecolor="none",
            edgecolor=OCEAN_COLOR_HEX,
            linewidth=cleanup_linewidth_pt,
            alpha=0.96,
            zorder=12.95,
            clip_on=False,
        )
    )


def _apply_circular_zoom_clip(zoom_ax: plt.Axes) -> None:
    clip_circle = Circle(
        (0.5, 0.5),
        ZOOM_CONTENT_CLIP_RADIUS_AXES,
        transform=zoom_ax.transAxes,
    )
    for artist in [
        *zoom_ax.images,
        *zoom_ax.collections,
        *zoom_ax.lines,
        *zoom_ax.patches,
    ]:
        artist.set_clip_path(clip_circle)


def _draw_zoom_connectors(
    ax: plt.Axes,
    zoom_ax: plt.Axes,
    extent: tuple[float, float, float, float],
    color: str,
    source_angles_deg: list[float],
    target_axes: list[list[float]],
) -> None:
    cx, cy, radius = _source_circle_from_extent(extent)
    _draw_selected_source_circle(ax, cx, cy, radius, color)
    for angle_deg, axes_xy in zip(source_angles_deg, target_axes):
        angle = math.radians(float(angle_deg))
        source_xy = (cx + radius * math.cos(angle), cy + radius * math.sin(angle))
        _draw_tapered_connector(
            ax,
            zoom_ax,
            source_xy,
            (float(axes_xy[0]), float(axes_xy[1])),
            color,
        )


def _plot_minor_class_zoom_insets(
    ax: plt.Axes,
    continents,
    continents_web,
    dryland,
    dryland_web,
    patches: list[tuple[np.ma.MaskedArray, tuple[float, float, float, float]]],
    cmap: ListedColormap,
    norm: BoundaryNorm,
    zoom_rows: list[dict[str, object]],
) -> None:
    ax.figure.canvas.draw()
    for row in zoom_rows:
        extent = tuple(float(value) for value in row["extent_lonlat"])
        color = str(row["color"])
        zoom_bounds = _square_inset_axes_bounds(ax, list(row["axes_bounds"]))
        zoom_ax = ax.inset_axes(zoom_bounds, zorder=12)
        zoom_ax.set_box_aspect(1.0)
        zoom_ax.set_facecolor((1, 1, 1, 0))
        zoom_ax.patch.set_alpha(0)
        zoom_ax.add_patch(
            Circle(
                (0.5, 0.5),
                0.5,
                transform=zoom_ax.transAxes,
                facecolor=mpl.colors.to_rgba(OCEAN_COLOR_HEX, 0.94),
                edgecolor="none",
                linewidth=0,
                zorder=-20,
                clip_on=False,
            )
        )
        _plot_base(zoom_ax, continents, continents_web, extent, show_ticks=False)
        zoom_ax.set_facecolor((1, 1, 1, 0))
        zoom_ax.patch.set_alpha(0)
        _plot_ai_dryland(zoom_ax, dryland, dryland_web, extent, inset=True)
        _plot_raster_patches(zoom_ax, patches, cmap, norm, extent, inset=True)
        _apply_circular_zoom_clip(zoom_ax)
        for spine in zoom_ax.spines.values():
            spine.set_visible(False)
        ax.figure.canvas.draw()
        _draw_zoom_edge_cleanup_on_figure(zoom_ax)
        _draw_zoom_connectors(
            ax,
            zoom_ax,
            extent,
            color,
            list(row["connector_source_angles_deg"]),
            list(row["connector_target_axes"]),
        )
        _draw_zoom_frame_on_figure(zoom_ax, color)


def _plot_suitability_share_bar_inset(ax: plt.Axes, class_rows: list[dict[str, object]]) -> None:
    values = np.array([float(row["visible_cells_downsampled"]) for row in class_rows], dtype=float)
    total = float(values.sum())
    if total <= 0:
        raise ValueError("Cannot draw share bar because all class share counts are zero.")

    ax.set_facecolor((1, 1, 1, 0.82))
    ax.patch.set_alpha(0.82)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#49535a")
        spine.set_linewidth(0.48)

    ax.text(
        0.055,
        0.84,
        "Suitable Area",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=5.75,
        fontweight="bold",
        color="#4f5b64",
    )

    bar_x = 0.055
    bar_y = 0.48
    bar_w = 0.89
    bar_h = 0.22
    left = bar_x
    for row in class_rows:
        share = float(row["share_pct"])
        width = bar_w * share / 100.0
        ax.add_patch(
            Rectangle(
                (left, bar_y),
                width,
                bar_h,
                transform=ax.transAxes,
                facecolor=str(row["color"]),
                edgecolor="#ffffff",
                linewidth=0.38,
            )
        )
        left += width
    ax.add_patch(
        Rectangle(
            (bar_x, bar_y),
            bar_w,
            bar_h,
            transform=ax.transAxes,
            facecolor="none",
            edgecolor="#3f4a4f",
            linewidth=0.32,
        )
    )

    label_positions = [(0.055, 0.18), (0.365, 0.18), (0.665, 0.18)]
    for row, (x, y) in zip(class_rows, label_positions):
        ax.add_patch(
            Rectangle(
                (x, y - 0.028),
                0.034,
                0.052,
                transform=ax.transAxes,
                facecolor=str(row["color"]),
                edgecolor="#ffffff",
                linewidth=0.28,
            )
        )
        ax.text(
            x + 0.047,
            y,
            f"{float(row['share_pct']):.1f}%",
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=5.65,
            color="#24313b",
        )


def _plot_grouped_legend_inset(ax: plt.Axes) -> None:
    legend_ax = ax.inset_axes([0.040, 0.065, 0.125, 0.242], zorder=13)
    legend_ax.set_facecolor((1, 1, 1, 0.82))
    legend_ax.patch.set_alpha(0.82)
    legend_ax.set_xlim(0, 1)
    legend_ax.set_ylim(0, 1)
    legend_ax.set_xticks([])
    legend_ax.set_yticks([])
    for spine in legend_ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#49535a")
        spine.set_linewidth(0.50)

    text_color = "#24313b"
    heading_color = "#4f5b64"
    legend_ax.text(
        0.075,
        0.86,
        "Dryland",
        transform=legend_ax.transAxes,
        ha="left",
        va="center",
        fontsize=5.75,
        fontweight="bold",
        color=heading_color,
    )

    dryland_y = [0.680, 0.520, 0.360, 0.200]
    for class_key, y in zip(DRYLAND_CLASS_ORDER, dryland_y):
        style = DRYLAND_CLASS_STYLES[class_key]
        legend_ax.add_patch(
            Rectangle(
                (0.080, y - 0.032),
                0.115,
                0.064,
                transform=legend_ax.transAxes,
                facecolor=style["color"],
                edgecolor="none",
                alpha=DRYLAND_ALPHA,
                linewidth=0,
            )
        )
        legend_ax.text(
            0.245,
            y,
            str(style["label"]),
            transform=legend_ax.transAxes,
            ha="left",
            va="center",
            fontsize=5.65,
            color=text_color,
        )



def _make_readme(summary: dict[str, float], outputs: dict[str, str], dryland_feature_count: int) -> None:
    text = f"""# {FIGURE_ID} candidate package

This is a manuscript candidate figure and has not been inserted into Word.

## Scientific content

- Main chain: selected10 HGB.
- Hydrology rule: HydroRIVERS DIS_AV_CMS >= 10 m3/s.
- Displayed suitability classes are derived from the q10 land-cover compatibility raster:
  - 50-74%: suitable.
  - 75-89%: high suitability.
  - >=90%: optimal suitability.
- Dryland overlay uses the local shapefile `{DRYLAND_AI_SHP}` ({dryland_feature_count} features). The polygons are filled by AI/P/PET class from `HIX_ZONE`, `HIX_DESC`, and `UNCCDDESC`, with no polygon outline stroke.
- Basemap: local Natural Earth country polygons dissolved to continents, drawn without country borders; displayed in Web Mercator (`{DISPLAY_CRS}`) for visual alignment only.
- Weighted land-cover compatible area: {summary['weighted_compatible_area_km2'] / 1e6:.2f} million km2.
- Binary land-cover compatible area: {summary['binary_suitable_area_km2'] / 1e6:.2f} million km2.
- Hydrology-constrained area before land-cover weighting: {summary['stage17_suitable_area_km2'] / 1e6:.2f} million km2.

## Review state

- Status: candidate for user visual review.
- Word insertion: not inserted into Word; not allowed until the user explicitly confirms the figure is final.
- Uncertain data: none identified in the local q10 summary used here.
- Design response to review: v49 keeps the no-country-border local continent-outline base, the compact dryland legend, the suitability-share component, and the non-purple/non-pink high-contrast palette from v48: blue for Suitable, vermillion/orange-red for High, and green for Optimal (`#0072b2`, `#d55e00`, `#009e73`). It keeps the two circular in-map local zoom insets, selected source-circle treatment, and top-layer tapered connector bands from v48. Compared with v48, it raises the smallest share-bar and dryland-legend fonts above the local SVG small-font warning boundary without changing map geometry, class colors, or data content. The red-marked inset labels remain removed so the enlarged circles contain only map content. The magnifier treatment follows the standard inset-axes plus connector-line pattern used by Matplotlib's zoom-inset tools, with a custom circular clip path and edge-snapped tapered connector polygons because the official helper defaults are rectangle-oriented and equal-width. Each main-map source circle remains a circular selection bounded by a {MINOR_CLASS_ZOOM_WINDOW_LON_DEG:.1f} by {MINOR_CLASS_ZOOM_WINDOW_LAT_DEG:.1f} degree lon/lat window. The source circles are selected from candidate areas with high local target-class density. The share bar keeps `Suitable Area`, color, and percentage information without English class-name labels. The 0-49% low-suitability context layer remains removed. The dryland polygons remain filled by local AI class without outlines.

## Outputs

{json.dumps(outputs, ensure_ascii=False, indent=2)}
"""
    (STAGE_DIR / f"{FIGURE_STEM}_README.md").write_text(text, encoding="utf-8")


def main() -> int:
    for d in (STAGE_DIR, FIG_DIR, TABLE_DIR, LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)
    logger = _setup_logging()
    _write_status("running")

    try:
        if not Q10_SUMMARY_CSV.exists():
            raise FileNotFoundError(f"Missing q10 summary table: {Q10_SUMMARY_CSV}")

        logger.info("Reading q10 summary: %s", Q10_SUMMARY_CSV)
        df = pd.read_csv(Q10_SUMMARY_CSV)
        needed_cols = {
            "tile_id",
            "compatible_pct_tif",
            "weighted_compatible_area_km2",
            "binary_suitable_area_km2",
            "stage17_suitable_area_km2",
        }
        missing_cols = sorted(needed_cols - set(df.columns))
        if missing_cols:
            raise ValueError(f"q10 summary is missing columns: {missing_cols}")

        used = df[df["binary_suitable_area_km2"].fillna(0) > 0].copy()
        if used.empty:
            raise ValueError("No q10 binary suitable tiles were found.")

        summary = {
            "weighted_compatible_area_km2": float(df["weighted_compatible_area_km2"].sum()),
            "binary_suitable_area_km2": float(df["binary_suitable_area_km2"].sum()),
            "stage17_suitable_area_km2": float(df["stage17_suitable_area_km2"].sum()),
        }
        logger.info("Area summary: %s", summary)

        continents = _load_continent_layer(logger)
        continents_web = continents.to_crs(DISPLAY_CRS)

        dryland = _load_ai_dryland_layer(logger)
        dryland_web = dryland.to_crs(DISPLAY_CRS)
        dryland_bounds = dryland.geometry.bounds
        dryland_records = dryland.drop(columns="geometry").copy()
        dryland_records["left"] = dryland_bounds["minx"].astype(float)
        dryland_records["right"] = dryland_bounds["maxx"].astype(float)
        dryland_records["bottom"] = dryland_bounds["miny"].astype(float)
        dryland_records["top"] = dryland_bounds["maxy"].astype(float)
        dryland_records["source_shp"] = str(DRYLAND_AI_SHP)
        dryland_records = dryland_records.fillna("not_available")
        dryland_records.to_csv(
            TABLE_DIR / f"{FIGURE_STEM}_ai_dryland_shp_plot_data.csv",
            index=False,
            encoding="utf-8-sig",
        )

        tile_table = used[
            [
                "tile_id",
                "weighted_compatible_area_km2",
                "binary_suitable_area_km2",
                "stage17_suitable_area_km2",
                "compatible_pct_tif",
            ]
        ].copy()
        tile_table = tile_table.sort_values("weighted_compatible_area_km2", ascending=False)
        tile_table.to_csv(TABLE_DIR / f"{FIGURE_STEM}_tile_area_plot_data.csv", index=False, encoding="utf-8-sig")

        logger.info("Loading raster patches from %s positive tiles.", len(used))
        patches: list[tuple[np.ma.MaskedArray, tuple[float, float, float, float]]] = []
        patch_records = []
        for _, row in used.iterrows():
            tif = _resolve_path(str(row["compatible_pct_tif"]))
            if not tif.exists():
                raise FileNotFoundError(f"Missing compatibility raster for tile {row['tile_id']}: {tif}")
            data, bounds, counts = _read_class_patch(tif)
            if np.ma.count(data) == 0:
                continue
            patches.append((data, bounds))
            patch_records.append(
                {
                    "tile_id": row["tile_id"],
                    "left": bounds[0],
                    "right": bounds[1],
                    "bottom": bounds[2],
                    "top": bounds[3],
                    "visible_cells_downsampled": int(np.ma.count(data)),
                    **counts,
                }
            )
        pd.DataFrame(patch_records).to_csv(
            TABLE_DIR / f"{FIGURE_STEM}_raster_patch_plot_data.csv",
            index=False,
            encoding="utf-8-sig",
        )
        if not patches:
            raise ValueError("All compatibility rasters were empty after the >=50% display threshold.")

        class_share_rows = _class_focus_summary(patches)
        pd.DataFrame(class_share_rows).to_csv(
            TABLE_DIR / f"{FIGURE_STEM}_share_bar_plot_data.csv",
            index=False,
            encoding="utf-8-sig",
        )
        minor_zoom_rows = _minor_class_zoom_summary(patches)
        pd.DataFrame(minor_zoom_rows).to_csv(
            TABLE_DIR / f"{FIGURE_STEM}_minor_class_zoom_plot_data.csv",
            index=False,
            encoding="utf-8-sig",
        )
        logger.info("Minor-class zoom insets: %s", minor_zoom_rows)
        logger.info(
            "Using local Natural Earth continent outlines; individual country boundaries are not plotted."
        )

        mpl.rcParams.update(
            {
                "font.family": "DejaVu Sans",
                "font.size": 7.7,
                "axes.labelsize": 8.0,
                "xtick.labelsize": 6.9,
                "ytick.labelsize": 6.9,
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
                "svg.fonttype": "none",
            }
        )
        cmap = ListedColormap(CLASS_COLORS, name="q10_suitability_classes")
        cmap.set_bad((0, 0, 0, 0))
        norm = BoundaryNorm(CLASS_BOUNDS, cmap.N)

        fig = plt.figure(figsize=(7.65, 3.78), constrained_layout=False)
        fig.patch.set_alpha(0)
        gs = fig.add_gridspec(
            nrows=1,
            ncols=1,
            left=0.055,
            right=0.985,
            top=0.985,
            bottom=0.105,
        )

        main_ax = fig.add_subplot(gs[0, 0])
        _plot_base(main_ax, continents, continents_web, GLOBAL_EXTENT, show_ticks=True)
        _plot_ai_dryland(main_ax, dryland, dryland_web, GLOBAL_EXTENT)
        _plot_raster_patches(main_ax, patches, cmap, norm, GLOBAL_EXTENT)
        _plot_minor_class_zoom_insets(
            main_ax,
            continents,
            continents_web,
            dryland,
            dryland_web,
            patches,
            cmap,
            norm,
            minor_zoom_rows,
        )
        main_ax.set_xlabel("")
        main_ax.set_ylabel("")

        _plot_grouped_legend_inset(main_ax)

        share_ax = main_ax.inset_axes([0.570, 0.018, 0.285, 0.105], zorder=12)
        _plot_suitability_share_bar_inset(share_ax, class_share_rows)

        outputs = {
            "transparent_png": str(FIG_DIR / f"{FIGURE_STEM}.png"),
            "white_preview_png": str(FIG_DIR / f"{FIGURE_STEM}_white_preview.png"),
            "svg": str(FIG_DIR / f"{FIGURE_STEM}.svg"),
            "pdf": str(FIG_DIR / f"{FIGURE_STEM}.pdf"),
            "tile_area_plot_data": str(TABLE_DIR / f"{FIGURE_STEM}_tile_area_plot_data.csv"),
            "raster_patch_plot_data": str(TABLE_DIR / f"{FIGURE_STEM}_raster_patch_plot_data.csv"),
            "ai_dryland_shp_plot_data": str(TABLE_DIR / f"{FIGURE_STEM}_ai_dryland_shp_plot_data.csv"),
            "share_bar_plot_data": str(TABLE_DIR / f"{FIGURE_STEM}_share_bar_plot_data.csv"),
            "minor_class_zoom_plot_data": str(TABLE_DIR / f"{FIGURE_STEM}_minor_class_zoom_plot_data.csv"),
            "continent_outline_source": str(COUNTRIES_SHP),
            "continent_outline_method": "Natural Earth admin-0 countries dissolved by CONTINENT; no country boundaries plotted.",
            "local_dryland_ai_shp": str(DRYLAND_AI_SHP),
        }
        _atomic_save(fig, Path(outputs["transparent_png"]), transparent=True)
        _atomic_save(fig, Path(outputs["white_preview_png"]), transparent=False)
        _atomic_save(fig, Path(outputs["svg"]), transparent=True)
        _atomic_save(fig, Path(outputs["pdf"]), transparent=True)
        plt.close(fig)

        _make_readme(summary, outputs, len(dryland))
        _write_status(
            "success",
            area_summary=summary,
            source_summary_csv=str(Q10_SUMMARY_CSV),
            local_dryland_ai_shp=str(DRYLAND_AI_SHP),
            continent_outline_source=str(COUNTRIES_SHP),
            continent_outline_method="Natural Earth admin-0 countries dissolved by CONTINENT; no country boundaries plotted.",
            display_crs=DISPLAY_CRS,
            ocean_color_hex=OCEAN_COLOR_HEX,
            land_color_hex=LAND_COLOR_HEX,
            continent_line_hex=CONTINENT_LINE_HEX,
            dryland_alpha=DRYLAND_ALPHA,
            global_extent_lonlat=list(GLOBAL_EXTENT),
            share_bar_classes=[
                {
                    "label": label,
                    "low_inclusive_pct": low,
                    "upper_bound_pct": high,
                    "upper_bound_rule": "<= upper_bound_pct" if high >= 100.0 else "< upper_bound_pct",
                    "color": color,
                }
                for label, low, high, color in CLASS_FOCUS_PANELS
            ],
            minor_class_zoom_source_selection_shape=MINOR_CLASS_ZOOM_SELECTION_SHAPE,
            minor_class_zoom_source_bounding_window_lonlat_degrees=[
                MINOR_CLASS_ZOOM_WINDOW_LON_DEG,
                MINOR_CLASS_ZOOM_WINDOW_LAT_DEG,
            ],
            minor_class_zoom_panels=minor_zoom_rows,
            suitability_classes={
                "suitable": "50-74% Land-cover compatibility",
                "high": "75-89% Land-cover compatibility",
                "optimal": ">=90% Land-cover compatibility",
            },
            dryland_ai_classes={
                class_key: DRYLAND_CLASS_STYLES[class_key]["label"] for class_key in DRYLAND_CLASS_ORDER
            },
            dryland_domain_note="Dryland overlay uses the local AI0-0.65 dryland shapefile, filled by AI/P/PET class, with edgecolor='none' and linewidth=0.",
            output_files=outputs,
            notes=[
                "Candidate figure only; do not insert into Word before user confirmation.",
                "Mapped pixels use compatible_pct >= 50 from the q10 HGB main chain.",
                "v22 keeps the no-country-border land/ocean base and dissolved continent outlines from v21.",
                "v22 restores dryland-related English labels while removing only generic legend-title English text.",
                "v22 adds compact semi-transparent background frames to the dryland legend and share bar.",
                "v22 keeps the share bar label-free except for the numeric percentages.",
                "v26 replaces the v25 magenta/purple-adjacent suitability color with a non-purple, non-pink blue-orange-red-green palette to improve salience of the two small-share classes.",
                "v27 previously added two local zoom insets for the two small-share classes.",
                "v28 supersedes the v27 locator-box treatment: it removes the main-map boxes, shrinks the two zoom source extents, and draws class-colored arrows from each source-area center to its enlarged inset.",
                "v28 records the local class mixture for each chosen zoom extent and prioritizes candidates containing multiple displayed suitability classes.",
                "v29 supersedes v28 by matching the zoom inset display boxes to the user-marked red-circle positions and by forcing both source windows to the same blue-circle-sized 15 by 12 degree selection extent.",
                "v29 restores a thin source-window outline on the main map so the fixed-size selection range remains visible.",
                "v30 supersedes v29 by using circular zoom insets and circular main-map source selections, while keeping the red-circle inset positions and the blue-circle-sized source bounding window.",
                "v31 supersedes v30 by halving each circular source selection to a 7.5 by 6 degree bounding window and replacing arrow connectors with two edge lines for a magnifier-style link.",
                "v32 supersedes v31 by removing the red-marked Suitable/High text inside the circular zoom insets while keeping the circular magnifier and two-line connector treatment.",
                "v33 supersedes v32 by adding a restrained selected-state treatment to each main-map source circle: subtle shadow, pale rim, class-colored outline, and very light fill.",
                "v34 supersedes v33 by moving the selected/raised treatment onto the enlarged circular zoom images: subtle lens shadow, pale rim, inner edge shade, class-colored outline, and a light upper-left highlight.",
                "v35 supersedes v34 by moving the selected treatment back to the source/viewfinder circles, halving the source-circle stroke widths, and expanding the two connector-line opening angles by 50%.",
                "v36 supersedes v35 by making the source/viewfinder-circle selected treatment visibly stronger, converting the equal-width connector lines into tapered vector bands that grow thicker toward the enlarged circles, and expanding each connector pair another 20% symmetrically around its center direction.",
                "v37 supersedes v36 by reducing connector-band weight and snapping each connector endpoint to the enlarged circular inset frame, so the lines meet the big-circle outer border without overlap or gaps.",
                "v38 supersedes v37 by forcing the enlarged inset axes to square display boxes, expanding the connector-pair opening by another 10%, and extending connector endpoints fractionally under the circular frame so the visible line edge connects cleanly without a gap.",
                "v39 supersedes v38 by moving enlarged-circle connector endpoints outside the circle instead of into it, lowering connector z-order below zoom content, and clipping all zoom-map artists to a slightly smaller circular mask so orange inset content does not leak outside the round frame.",
                "v40 supersedes v39 by calculating the enlarged-circle connector endpoints as true external tangent points and tightening the circular zoom clip further, preventing connector intrusion into the enlarged circles and map-content bleed outside the orange frame.",
                "v41 supersedes v40 by subtracting the enlarged-circle disk from each tapered connector polygon in display-pixel space, so connector end caps cannot enter the large circular insets.",
                "v42 supersedes v41 by expanding the connector exclusion disk outside the enlarged-circle frame and tightening the circular map-content clip, addressing residual connector intrusion and orange-inset content bleed.",
                "v43 supersedes v42 by reducing the connector exclusion disk to remove the visible line-circle gap while retaining outside-circle clipping.",
                "v44 supersedes v43 by making a smaller edge-snapping adjustment so connector tips sit under the circle frame with less visible gap.",
                "v45 supersedes v44 by replacing the enlarged-circle white underlay with ocean-tone fill and relaxing the content clip slightly inward from the circular frame, removing the visible white ring while retaining bleed protection.",
                "v46 supersedes v45 by tightening the enlarged-circle content clip, adding an ocean-tone edge cleanup stroke outside the zoom content, and raising the connector bands to the top layer while keeping the outside-circle clipping.",
                "v47 supersedes v46 by shifting the ocean-tone cleanup stroke outward from the enlarged-circle frame so the cleanup masks only exterior bleed rather than covering the inset map content.",
                "v48 supersedes v47 by increasing the smallest share-bar and dryland-legend label font sizes to clear the local SVG font-size QC warning without changing map geometry, class colors, or data content.",
                "v49 supersedes v48 by raising the smallest map-embedded text above the local small-font warning boundary while preserving the v48 geometry and palette.",
                "In-map stacked share bar percentages are based on the visible downsampled raster cells used for this figure.",
                "Dryland context comes from the local AI0-0.65 dryland shapefile rather than q10 tile bounds.",
            ],
        )
        logger.info("Completed %s.", FIGURE_STEM)
        return 0
    except Exception as exc:
        logger.error("Failed to build %s: %s", FIGURE_STEM, exc)
        logger.error(traceback.format_exc())
        _write_status("failed", error=str(exc), traceback=traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
