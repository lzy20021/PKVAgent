from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MPLBACKEND", "Agg")

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import Rectangle

from src.agent import tools as agent_tools
from src.agent.core import MapGenAgent
from src.agent.generalization_planner import GeneralizationPlanner, PlanningRequest
from src.agent.providers import BaseLLMProvider, LLMResponse
from src.agent.vlm_verifier import VerifyInput, VLMVerifier
from src.ui.point_symbol_core import POI_SYMBOL_RULES, canonicalize_poi_category


CATEGORY_ORDER = list(POI_SYMBOL_RULES.keys())
CATEGORY_ALIASES = {
    "餐厅": "餐饮",
    "咖啡厅": "餐饮",
    "餐饮": "餐饮",
    "公交站": "交通设施",
    "地铁站": "交通设施",
    "交通设施": "交通设施",
}
ICON_FILES = {
    "便利店": "convenience_store.png",
    "学校": "school.png",
    "停车场": "parking.png",
    "酒店": "hotel.png",
    "银行": "bank.png",
    "医院": "hospital.png",
    "餐饮": "restaurant.png",
    "交通设施": "transit.png",
}
CATEGORY_FIELDS = [
    "query_category",
    "poi_category",
    "category",
    "type",
    "类别",
    "类型",
    "class",
    "kind",
    "fclass",
]


class NullProvider(BaseLLMProvider):
    def __init__(self) -> None:
        super().__init__(api_key="", model="null")

    def chat(self, messages, tools=None) -> LLMResponse:  # type: ignore[override]
        return LLMResponse("")

    def stream_chat(self, messages, tools=None):  # type: ignore[override]
        if False:
            yield ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a four-stage PointGenAI point-symbol generalization process sequence."
    )
    parser.add_argument(
        "--input",
        default="TestDatasets/Scale_1_86000/sample_13.geojson",
        help="Selected sample-area POI GeoJSON.",
    )
    parser.add_argument(
        "--output-dir",
        default="output/pointgenai_process_sequence_sample13_1_250000",
        help="Output directory for R0-R3 maps, layers, logs, and metrics.",
    )
    parser.add_argument("--source-scale", type=int, default=86000)
    parser.add_argument("--target-scale", type=int, default=250000)
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove the output directory before running. Only allowed below project output/.",
    )
    return parser.parse_args()


def choose_cjk_font() -> str:
    candidates = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC", "PingFang SC"]
    available = {font.name for font in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            return name
    return "DejaVu Sans"


def configure_plot() -> None:
    font = choose_cjk_font()
    plt.rcParams.update(
        {
            "font.family": font,
            "font.sans-serif": [font],
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def canonical_category(value: Any) -> str:
    text = str(value or "").strip()
    if not text or text.lower() == "nan":
        return ""
    if text in CATEGORY_ALIASES:
        return CATEGORY_ALIASES[text]
    category = canonicalize_poi_category(text)
    if category in CATEGORY_ALIASES:
        return CATEGORY_ALIASES[category]
    return category if category in POI_SYMBOL_RULES else ""


def find_category_field(gdf: gpd.GeoDataFrame) -> str:
    for field in CATEGORY_FIELDS:
        if field in gdf.columns:
            mapped = gdf[field].dropna().astype(str).map(canonical_category)
            if mapped.isin(CATEGORY_ORDER).any():
                return field
    for field in gdf.columns:
        if field == gdf.geometry.name:
            continue
        mapped = gdf[field].dropna().astype(str).head(300).map(canonical_category)
        if mapped.isin(CATEGORY_ORDER).any():
            return str(field)
    raise ValueError("No recognizable 8-class POI category field was found.")


def assign_poi_symbols(gdf: gpd.GeoDataFrame, category_field: str) -> gpd.GeoDataFrame:
    out = gdf.copy()
    raw = out[category_field].fillna("").astype(str)
    out["source_category"] = raw
    out["poi_category"] = raw.map(canonical_category)
    out = out[out["poi_category"].isin(CATEGORY_ORDER)].copy()
    out["point_symbol"] = out["poi_category"].map(lambda c: POI_SYMBOL_RULES[c]["symbol"])
    out["point_symbol_color"] = out["poi_category"].map(lambda c: POI_SYMBOL_RULES[c]["color"])
    out["point_symbol_label"] = out["poi_category"].map(lambda c: POI_SYMBOL_RULES[c]["label"])

    # Force MapGenAgent to use the canonical 8-class field instead of raw query aliases.
    for raw_field in ["query_category", "category", "type", "类别", "类型", "class", "kind", "fclass"]:
        if raw_field in out.columns and raw_field != "poi_category":
            out.rename(columns={raw_field: f"source_{raw_field}"}, inplace=True)
    return out


def prepare_agent_input(input_path: Path, output_dir: Path) -> Tuple[Path, gpd.GeoDataFrame, str]:
    gdf = gpd.read_file(input_path)
    if len(gdf) == 0:
        raise ValueError(f"Input is empty: {input_path}")
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326", allow_override=True)
    category_field = find_category_field(gdf)
    normalized = assign_poi_symbols(gdf, category_field)
    if len(normalized) == 0:
        raise ValueError("No supported 8-class POI features remain after normalization.")
    out = output_dir / "agent_input_sample13_8class.geojson"
    normalized.to_file(out, driver="GeoJSON")
    return out, normalized, category_field


def projected(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326", allow_override=True)
    if gdf.crs.is_projected:
        return gdf
    try:
        crs = gdf.estimate_utm_crs()
        if crs:
            return gdf.to_crs(crs)
    except Exception:
        pass
    lon = float(gdf.to_crs("EPSG:4326").geometry.x.mean())
    lat = float(gdf.to_crs("EPSG:4326").geometry.y.mean())
    zone = int((lon + 180.0) // 6.0) + 1
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    return gdf.to_crs(epsg=epsg)


def min_sep_m(target_scale: int) -> float:
    return 0.30 * float(target_scale) / 1000.0


def conflict_pairs(gdf: gpd.GeoDataFrame, target_scale: int, max_pairs: int = 600) -> Tuple[int, List[Tuple[int, int]]]:
    if len(gdf) <= 1:
        return 0, []
    work = projected(gdf).reset_index(drop=True)
    coords = [(float(p.x), float(p.y)) for p in work.geometry]
    threshold = min_sep_m(target_scale)
    count = 0
    pairs: List[Tuple[int, int]] = []
    for i in range(len(coords)):
        x1, y1 = coords[i]
        for j in range(i + 1, len(coords)):
            x2, y2 = coords[j]
            if math.hypot(x1 - x2, y1 - y2) < threshold:
                count += 1
                if len(pairs) < max_pairs:
                    pairs.append((i, j))
    return count, pairs


def category_coverage(gdf: gpd.GeoDataFrame, reference: gpd.GeoDataFrame) -> float:
    ref = set(reference["poi_category"].dropna().astype(str).tolist())
    cur = set(gdf["poi_category"].dropna().astype(str).tolist()) if "poi_category" in gdf.columns else set()
    return len(ref & cur) / max(1, len(ref))


def region_coverage(gdf: gpd.GeoDataFrame, reference: gpd.GeoDataFrame, grid_n: int = 6) -> float:
    if len(gdf) == 0 or len(reference) == 0:
        return 0.0
    ref = projected(reference).reset_index(drop=True)
    cur = projected(gdf).reset_index(drop=True)
    minx, miny, maxx, maxy = [float(v) for v in ref.total_bounds]
    dx = max((maxx - minx) / grid_n, 1.0)
    dy = max((maxy - miny) / grid_n, 1.0)

    def occupied(frame: gpd.GeoDataFrame) -> set[Tuple[int, int]]:
        cells: set[Tuple[int, int]] = set()
        for geom in frame.geometry:
            ix = max(0, min(grid_n - 1, int((geom.x - minx) / dx)))
            iy = max(0, min(grid_n - 1, int((geom.y - miny) / dy)))
            cells.add((ix, iy))
        return cells

    ref_cells = occupied(ref)
    cur_cells = occupied(cur)
    return len(ref_cells & cur_cells) / max(1, len(ref_cells))


def structural_metrics(gdf: gpd.GeoDataFrame, reference: gpd.GeoDataFrame) -> Dict[str, float | int]:
    if len(reference) == 0:
        return {
            "point_axis_coverage_ratio": 0.0,
            "point_middle_gap_ratio": 1.0,
            "point_hotspot_coverage_ratio": 0.0,
            "point_hotspot_rep_count": 0,
            "point_region_coverage_ratio": 0.0,
            "point_sparse_region_coverage_ratio": 0.0,
            "point_dense_region_singleton_ratio": 1.0,
        }
    ref = projected(reference).reset_index(drop=True)
    cur = projected(gdf).reset_index(drop=True)
    if len(cur) == 0:
        return {
            "point_axis_coverage_ratio": 0.0,
            "point_middle_gap_ratio": 1.0,
            "point_hotspot_coverage_ratio": 0.0,
            "point_hotspot_rep_count": 0,
            "point_region_coverage_ratio": 0.0,
            "point_sparse_region_coverage_ratio": 0.0,
            "point_dense_region_singleton_ratio": 1.0,
        }

    in_x = ref.geometry.x
    in_y = ref.geometry.y
    mx = float(in_x.mean())
    my = float(in_y.mean())
    dx = in_x - mx
    dy = in_y - my
    sxx = float((dx * dx).mean())
    syy = float((dy * dy).mean())
    sxy = float((dx * dy).mean())
    theta = 0.5 * math.atan2(2.0 * sxy, (sxx - syy))
    ux = math.cos(theta)
    uy = math.sin(theta)
    in_t = (in_x - mx) * ux + (in_y - my) * uy
    out_t = (cur.geometry.x - mx) * ux + (cur.geometry.y - my) * uy
    tmin = float(in_t.min())
    tmax = float(in_t.max())
    axis_cov = 1.0
    middle_gap = 0.0
    if tmax > tmin:
        axis_bins = max(8, min(32, int(round((len(ref) ** 0.5) * 2.5))))
        in_bin = ((in_t - tmin) / (tmax - tmin + 1e-9) * axis_bins).astype(int).clip(lower=0, upper=axis_bins)
        out_bin = ((out_t - tmin) / (tmax - tmin + 1e-9) * axis_bins).astype(int).clip(lower=0, upper=axis_bins)
        in_set = set(int(v) for v in in_bin.dropna().unique().tolist())
        out_set = set(int(v) for v in out_bin.dropna().unique().tolist())
        axis_cov = len(in_set & out_set) / max(1, len(in_set))
        run = 0
        longest = 0
        for bi in range(axis_bins + 1):
            if bi in in_set and bi not in out_set:
                run += 1
                longest = max(longest, run)
            elif bi in in_set:
                run = 0
        middle_gap = longest / max(1, len(in_set))

    minx, miny, maxx, maxy = [float(v) for v in ref.total_bounds]
    width = max(1e-9, maxx - minx)
    height = max(1e-9, maxy - miny)

    hot_bins = max(6, min(20, int(round((len(ref) ** 0.5) * 0.8))))
    in_hx = ((ref.geometry.x - minx) / width * hot_bins).astype(int).clip(lower=0, upper=hot_bins)
    in_hy = ((ref.geometry.y - miny) / height * hot_bins).astype(int).clip(lower=0, upper=hot_bins)
    out_hx = ((cur.geometry.x - minx) / width * hot_bins).astype(int).clip(lower=0, upper=hot_bins)
    out_hy = ((cur.geometry.y - miny) / height * hot_bins).astype(int).clip(lower=0, upper=hot_bins)
    in_keys = pd.Series([f"{int(x)}_{int(y)}" for x, y in zip(in_hx.tolist(), in_hy.tolist())])
    out_keys = pd.Series([f"{int(x)}_{int(y)}" for x, y in zip(out_hx.tolist(), out_hy.tolist())])
    in_freq = in_keys.value_counts(dropna=False)
    hotspot_cells: List[str] = []
    if len(in_freq) > 0:
        q90 = float(in_freq.quantile(0.90))
        min_hot = max(3.0, q90)
        hotspot_cells_all = [str(k) for k, v in in_freq.items() if float(v) >= min_hot]
        hotspot_cells = hotspot_cells_all[: max(6, min(18, len(hotspot_cells_all)))]
    out_hot = set(str(v) for v in out_keys.tolist())
    hotspot_cov = len([k for k in hotspot_cells if k in out_hot]) / max(1, len(hotspot_cells))
    hotspot_rep = int(sum(1 for k in out_keys.tolist() if str(k) in set(hotspot_cells))) if hotspot_cells else int(len(cur))

    region_bins = max(3, min(10, int(round((max(6, len(cur)) ** 0.5) * 1.15))))
    in_rx = ((ref.geometry.x - minx) / width * region_bins).astype(int).clip(lower=0, upper=region_bins)
    in_ry = ((ref.geometry.y - miny) / height * region_bins).astype(int).clip(lower=0, upper=region_bins)
    out_rx = ((cur.geometry.x - minx) / width * region_bins).astype(int).clip(lower=0, upper=region_bins)
    out_ry = ((cur.geometry.y - miny) / height * region_bins).astype(int).clip(lower=0, upper=region_bins)
    in_regions = pd.Series([f"{int(x)}_{int(y)}" for x, y in zip(in_rx.tolist(), in_ry.tolist())])
    out_regions = pd.Series([f"{int(x)}_{int(y)}" for x, y in zip(out_rx.tolist(), out_ry.tolist())])
    in_region_freq = in_regions.value_counts(dropna=False)
    out_region_freq = out_regions.value_counts(dropna=False)
    in_region_set = set(str(v) for v in in_region_freq.index.tolist())
    out_region_set = set(str(v) for v in out_region_freq.index.tolist())
    reg_cov = len(in_region_set & out_region_set) / max(1, len(in_region_set))
    sparse_keys = [str(k) for k, v in in_region_freq.items() if int(v) <= 2]
    if sparse_keys:
        sparse_cov = len([k for k in sparse_keys if k in out_region_set]) / max(1, len(sparse_keys))
    else:
        sparse_cov = 1.0
    dense_threshold = max(8, int(round(float(in_region_freq.quantile(0.75))))) if len(in_region_freq) > 0 else 8
    dense_keys = [str(k) for k, v in in_region_freq.items() if int(v) >= dense_threshold]
    dense_singletons = sum(1 for key in dense_keys if int(out_region_freq.get(key, 0)) <= 1)
    dense_singleton_ratio = dense_singletons / max(1, len(dense_keys))

    return {
        "point_axis_coverage_ratio": round(float(axis_cov), 4),
        "point_middle_gap_ratio": round(float(middle_gap), 4),
        "point_hotspot_coverage_ratio": round(float(hotspot_cov), 4),
        "point_hotspot_rep_count": int(hotspot_rep),
        "point_region_coverage_ratio": round(float(reg_cov), 4),
        "point_sparse_region_coverage_ratio": round(float(sparse_cov), 4),
        "point_dense_region_singleton_ratio": round(float(dense_singleton_ratio), 4),
    }


def stage_qa_metrics(
    gdf: gpd.GeoDataFrame,
    reference: gpd.GeoDataFrame,
    target_scale: int,
) -> Dict[str, Any]:
    conflicts, _ = conflict_pairs(gdf, target_scale, max_pairs=0)
    struct = structural_metrics(gdf, reference)
    cats = sorted(set(gdf.get("poi_category", pd.Series(dtype=str)).dropna().astype(str).tolist()))
    out: Dict[str, Any] = {
        "input_count": int(len(reference)),
        "output_count": int(len(gdf)),
        "target_scale": int(target_scale),
        "geometry_type": "point",
        "topology_error_count": int((~gdf.geometry.is_valid).sum()),
        "retention_rate": float(len(gdf)) / float(max(1, len(reference))),
        "poi_symbol_field": "point_symbol" if "point_symbol" in gdf.columns else "",
        "poi_symbol_rule_count": int(len(POI_SYMBOL_RULES)),
        "poi_symbol_categories": cats,
        "conflict_count_total": int(conflicts),
        "min_symbol_separation_m": min_sep_m(target_scale),
    }
    out.update(struct)
    return out


def build_display_extent(reference: gpd.GeoDataFrame) -> Tuple[float, float, float, float]:
    ref = reference.to_crs("EPSG:4326") if str(reference.crs).upper() != "EPSG:4326" else reference
    minx, miny, maxx, maxy = [float(v) for v in ref.total_bounds]
    dx = max(maxx - minx, 0.002)
    dy = max(maxy - miny, 0.002)
    pad = 0.08
    return (minx - dx * pad, miny - dy * pad, maxx + dx * pad, maxy + dy * pad)


def load_icons() -> Dict[str, Any]:
    icon_dir = PROJECT_ROOT / "output" / "arcgis_10_2_poi_symbols"
    icons = {}
    for category, name in ICON_FILES.items():
        path = icon_dir / name
        if not path.is_file():
            raise FileNotFoundError(path)
        icons[category] = mpimg.imread(path)
    return icons


def draw_density_cells(ax: plt.Axes, gdf: gpd.GeoDataFrame, extent: Tuple[float, float, float, float]) -> None:
    work = gdf.to_crs("EPSG:4326").reset_index(drop=True)
    if len(work) == 0:
        return
    minx, miny, maxx, maxy = extent
    bins = 9
    width = maxx - minx
    height = maxy - miny
    gx = np.floor((work.geometry.x - minx) / max(width, 1e-9) * bins).clip(0, bins - 1).astype(int)
    gy = np.floor((work.geometry.y - miny) / max(height, 1e-9) * bins).clip(0, bins - 1).astype(int)
    cells = pd.Series([f"{x}_{y}" for x, y in zip(gx, gy)]).value_counts()
    if cells.empty:
        return
    threshold = max(float(cells.quantile(0.90)), 8.0)
    for key, value in cells.items():
        if float(value) < threshold:
            continue
        x, y = [int(v) for v in str(key).split("_")]
        ax.add_patch(
            Rectangle(
                (minx + width * x / bins, miny + height * y / bins),
                width / bins,
                height / bins,
                facecolor="#d32f2f",
                edgecolor="#7f1d1d",
                linewidth=0.6,
                alpha=0.10,
                zorder=1,
            )
        )


def render_map(
    gdf: gpd.GeoDataFrame,
    output_png: Path,
    output_svg: Path,
    title: str,
    target_scale: int,
    reference: gpd.GeoDataFrame,
    icons: Dict[str, Any],
    highlight_conflicts: bool,
) -> None:
    work = gdf.to_crs("EPSG:4326").reset_index(drop=True)
    extent = build_display_extent(reference)
    conflicts, pairs = conflict_pairs(gdf, target_scale, max_pairs=300)

    fig = plt.figure(figsize=(7.2, 7.8))
    ax = fig.add_axes([0.055, 0.145, 0.89, 0.78])
    ax.set_facecolor("#F8FAFC")
    ax.set_xlim(extent[0], extent[2])
    ax.set_ylim(extent[1], extent[3])
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("#CBD5E1")
        spine.set_linewidth(0.8)

    ax.add_patch(
        Rectangle(
            (extent[0], extent[1]),
            extent[2] - extent[0],
            extent[3] - extent[1],
            facecolor="#F8FAFC",
            edgecolor="#CBD5E1",
            linewidth=0.8,
            zorder=0,
        )
    )
    if highlight_conflicts:
        draw_density_cells(ax, work, extent)
        for i, j in pairs:
            p0 = work.geometry.iloc[i]
            p1 = work.geometry.iloc[j]
            ax.plot(
                [p0.x, p1.x],
                [p0.y, p1.y],
                color="#991B1B",
                linewidth=0.32,
                alpha=0.13,
                zorder=2,
            )

    order = {cat: idx for idx, cat in enumerate(CATEGORY_ORDER)}
    work["_draw_order"] = work["poi_category"].map(order).fillna(99)
    work = work.sort_values("_draw_order")
    for _, row in work.iterrows():
        cat = str(row.get("poi_category", ""))
        if cat not in icons:
            continue
        image = OffsetImage(icons[cat], zoom=0.105, interpolation="hanning")
        ax.add_artist(
            AnnotationBbox(
                image,
                (row.geometry.x, row.geometry.y),
                xycoords="data",
                frameon=False,
                box_alignment=(0.5, 0.5),
                pad=0,
                annotation_clip=True,
                zorder=8,
            )
        )

    fig.text(0.055, 0.955, title, ha="left", va="top", fontsize=15, fontweight="bold", color="#111827")
    fig.text(
        0.055,
        0.925,
        f"目标比例尺 1:{target_scale:,} | 点数 {len(gdf)} | 符号冲突点对 {conflicts}",
        ha="left",
        va="top",
        fontsize=9.5,
        color="#4B5563",
    )

    legend_ax = fig.add_axes([0.06, 0.025, 0.88, 0.085])
    legend_ax.set_axis_off()
    xs = np.linspace(0.035, 0.965, len(CATEGORY_ORDER))
    for x, cat in zip(xs, CATEGORY_ORDER):
        legend_ax.add_artist(
            AnnotationBbox(
                OffsetImage(icons[cat], zoom=0.14, interpolation="hanning"),
                (x, 0.62),
                xycoords=legend_ax.transAxes,
                frameon=False,
                box_alignment=(0.5, 0.5),
                pad=0,
            )
        )
        legend_ax.text(x, 0.10, cat, transform=legend_ax.transAxes, ha="center", va="bottom", fontsize=8.2, color="#1F2937")

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=320, facecolor="white")
    fig.savefig(output_svg, facecolor="white")
    plt.close(fig)


def write_layer_files(gdf: gpd.GeoDataFrame, geojson_path: Path, csv_path: Path) -> None:
    geojson_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(geojson_path, driver="GeoJSON")
    out = gdf.to_crs("EPSG:4326") if str(gdf.crs).upper() != "EPSG:4326" else gdf.copy()
    table = out.drop(columns=[out.geometry.name]).copy()
    table["lon"] = out.geometry.x
    table["lat"] = out.geometry.y
    table.to_csv(csv_path, index=False, encoding="utf-8-sig")


def summarize_event(event_type: str, data: Any) -> Dict[str, Any]:
    if event_type == "tool_start" and isinstance(data, dict):
        args = data.get("args", {}) if isinstance(data.get("args", {}), dict) else {}
        code = str(args.get("code", ""))
        return {"type": event_type, "name": data.get("name"), "code_chars": len(code)}
    if event_type == "tool_end" and isinstance(data, dict):
        result = data.get("result", {}) if isinstance(data.get("result", {}), dict) else {}
        stdout = str(result.get("stdout", "") or "")
        metrics = parse_metrics_from_stdout(stdout)
        return {
            "type": event_type,
            "name": data.get("name"),
            "status": result.get("status"),
            "stdout_metrics": metrics,
            "stdout_tail": "\n".join(stdout.splitlines()[-8:]),
        }
    return {"type": event_type, "data": str(data)}


def parse_metrics_from_stdout(stdout: str) -> Dict[str, Any]:
    for line in reversed(str(stdout or "").splitlines()):
        if line.startswith("metrics:"):
            text = line.split("metrics:", 1)[1].strip()
            try:
                return json.loads(text)
            except Exception:
                return {}
    return {}


def run_pointgenai(input_path: Path, output_dir: Path, source_scale: int, target_scale: int) -> Tuple[str, List[Dict[str, Any]]]:
    agent_tools.local_executor.workspace_path = str(PROJECT_ROOT)
    agent_tools.ui_callbacks.pop("vlm_verify", None)
    agent_tools.register_ui_callback("load_layer", lambda *_a, **_k: None)
    agent_tools.register_ui_callback("zoom_to_layer", lambda *_a, **_k: None)

    events: List[Dict[str, Any]] = []
    agent = MapGenAgent(provider=NullProvider())
    agent.set_workspace(str(output_dir))
    agent.set_event_callback(lambda event_type, data: events.append(summarize_event(event_type, data)))
    task = {
        "parameters": {
            "input_file": str(input_path),
            "source_scale": int(source_scale),
            "target_scale": int(target_scale),
            "feature_count": int(len(gpd.read_file(input_path))),
            "geometry_type": "point",
            "layer_name": f"sample13_process_1_{target_scale}",
            "output_formats": ["geojson", "png"],
            "require_real_vlm": False,
        }
    }
    reply = agent._run_scale_generalization(task) or ""
    (output_dir / "agent_reply.txt").write_text(reply, encoding="utf-8")
    (output_dir / "agent_events.json").write_text(json.dumps(events, ensure_ascii=False, indent=2), encoding="utf-8")
    return reply, events


def generated_rounds(input_path: Path, target_scale: int) -> Dict[int, Dict[str, Path]]:
    base = input_path.with_suffix("")
    pattern = f"{base.name}_gen_1_{target_scale}_r*.geojson"
    rounds: Dict[int, Dict[str, Path]] = {}
    for geojson in sorted(input_path.parent.glob(pattern)):
        m = re.search(r"_r(\d+)\.geojson$", geojson.name)
        if not m:
            continue
        idx = int(m.group(1))
        rounds[idx] = {
            "geojson": geojson,
            "png": geojson.with_suffix(".png"),
            "pdf": geojson.with_suffix(".pdf"),
        }
    return rounds


def stage_names() -> Dict[str, Dict[str, str]]:
    return {
        "R0": {
            "label": "R0_input",
            "title": "R0 原始输入：符号压盖与高密度聚集",
            "main_feedback": "原始输入存在符号压盖、局部拥挤和高密度聚集。",
            "next_strategy": "执行重要点优先保留、比例尺驱动删减、密集区点筛选、聚类中心保持、多核结构保持和符号层级配置。",
        },
        "R1": {
            "label": "R1_iter1",
            "title": "R1 第一轮：重要点保留与比例尺驱动删减",
            "main_feedback": "第一轮已完成密集区点筛选，但需继续检查热点过绘、局部失衡和格局保持。",
            "next_strategy": "调用邻近冲突检测、邻近冲突消解、均衡优化、稀疏格局保持、符号压盖校核和结果反馈修正。",
        },
        "R2": {
            "label": "R2_iter2",
            "title": "R2 第二轮：邻近冲突消解与均衡优化",
            "main_feedback": "第二轮继续降低局部符号负载，仍需通过视觉可读性评估复核热点区和外围离散点。",
            "next_strategy": "再次执行视觉可读性评估、符号压盖校核和结果反馈修正，保留主要聚集中心与外围离散点。",
        },
        "R3": {
            "label": "R3_final",
            "title": "R3 最终结果：反馈修正后的可读表达",
            "main_feedback": "最终结果符号冲突明显减少，图面负载更均衡，并保留 8 类 POI、主要聚集中心和外围离散点。",
            "next_strategy": "作为过程示意图最终图层；后续只进行论文图排版，不改变 POI 类别符号。",
        },
    }


def skill_list_for_stage(stage: str) -> List[str]:
    if stage == "R0":
        return ["符号层级配置", "符号压盖校核", "视觉可读性评估"]
    if stage == "R1":
        return ["重要点优先保留", "比例尺驱动删减", "密集区点筛选", "聚类中心保持", "多核结构保持", "符号层级配置", "视觉可读性评估"]
    if stage == "R2":
        return ["邻近冲突检测", "邻近冲突消解", "均衡优化", "稀疏格局保持", "符号压盖校核", "视觉可读性评估", "结果反馈修正"]
    return ["视觉可读性评估", "结果反馈修正", "符号压盖校核", "稀疏格局保持", "均衡优化"]


def reconstruct_plans_and_verify(
    stages: Dict[str, Dict[str, Any]],
    reference: gpd.GeoDataFrame,
    profile: Dict[str, Any],
    source_scale: int,
    target_scale: int,
) -> None:
    planner = GeneralizationPlanner()
    verifier = VLMVerifier()
    qa_prev: Dict[str, Any] = {}
    last_fail_reason = ""
    before_image = str(stages["R0"]["png"])
    point_retention_min = 0.12 if target_scale >= 200000 else 0.25
    layer_stats = {
        "geometry_type": "point",
        "feature_count": int(len(reference)),
        "semantic_type": profile.get("semantic_type", "commercial_poi_points"),
        "structure_type": profile.get("structure_type", "mixed"),
    }
    for stage in ["R1", "R2", "R3"]:
        round_index = int(stages[stage]["agent_round"])
        plan = planner.plan(
            PlanningRequest(
                source_scale=source_scale,
                target_scale=target_scale,
                layer_stats=layer_stats,
                qa_metrics=qa_prev,
                round_index=round_index,
                last_fail_reason=last_fail_reason,
            )
        )
        qa = stages[stage]["qa_metrics"]
        verify = verifier.verify(
            VerifyInput(
                target_scale=target_scale,
                metrics=qa,
                stop_conditions=plan.get("stop_conditions", {}),
                retention_min={"point": point_retention_min},
                before_image=before_image,
                after_image=str(stages[stage]["png"]),
                last_fail_reason=last_fail_reason,
                symbol_context={
                    "expected_poi_categories": CATEGORY_ORDER,
                    "poi_symbol_rules": POI_SYMBOL_RULES,
                    "poi_symbol_field": "point_symbol",
                    "poi_symbol_categories": qa.get("poi_symbol_categories", []),
                },
                require_real_vlm=False,
            )
        )
        stages[stage]["plan"] = plan
        stages[stage]["verify"] = verify
        qa_prev = qa
        failed = verify.get("failed_rules", [])
        last_fail_reason = "; ".join(str(x) for x in failed)[:200] if isinstance(failed, list) else str(failed)[:200]
        before_image = str(stages[stage]["png"])

    r0_plan = planner.plan(
        PlanningRequest(
            source_scale=source_scale,
            target_scale=target_scale,
            layer_stats=layer_stats,
            qa_metrics={},
            round_index=1,
            last_fail_reason="",
        )
    )
    r0_verify = verifier.verify(
        VerifyInput(
            target_scale=target_scale,
            metrics=stages["R0"]["qa_metrics"],
            stop_conditions=r0_plan.get("stop_conditions", {}),
            retention_min={"point": point_retention_min},
            before_image="",
            after_image=str(stages["R0"]["png"]),
            symbol_context={
                "expected_poi_categories": CATEGORY_ORDER,
                "poi_symbol_rules": POI_SYMBOL_RULES,
                "poi_symbol_field": "point_symbol",
                "poi_symbol_categories": stages["R0"]["qa_metrics"].get("poi_symbol_categories", []),
            },
            require_real_vlm=False,
        )
    )
    stages["R0"]["plan"] = r0_plan
    stages["R0"]["verify"] = r0_verify


def compute_public_metrics(stages: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    r0_conflicts = max(1, int(stages["R0"]["qa_metrics"].get("conflict_count_total", 0) or 0))
    rows: List[Dict[str, Any]] = []
    labels = stage_names()
    for stage in ["R0", "R1", "R2", "R3"]:
        qa = stages[stage]["qa_metrics"]
        verify = stages[stage]["verify"]
        conflicts = int(qa.get("conflict_count_total", 0) or 0)
        conflict_score = 1.0 if conflicts == 0 else max(0.0, 1.0 - conflicts / r0_conflicts)
        resolution_rate = 0.0 if stage == "R0" else max(0.0, (r0_conflicts - conflicts) / r0_conflicts)
        category_cov = category_coverage(stages[stage]["gdf"], stages["R0"]["gdf"])
        region_cov = float(qa.get("point_region_coverage_ratio", 0.0) or 0.0)
        hotspot_cov = float(qa.get("point_hotspot_coverage_ratio", 0.0) or 0.0)
        sparse_cov = float(qa.get("point_sparse_region_coverage_ratio", 0.0) or 0.0)
        axis_cov = float(qa.get("point_axis_coverage_ratio", 0.0) or 0.0)
        pattern = round((region_cov + hotspot_cov + sparse_cov + axis_cov) / 4.0, 4)
        visual_score = float(verify.get("score", 0) or 0.0)
        constraint = round(0.35 * conflict_score + 0.20 * category_cov + 0.20 * region_cov + 0.25 * (visual_score / 100.0), 4)
        rows.append(
            {
                "stage": stage,
                "stage_label": labels[stage]["label"],
                "point_count": int(qa.get("output_count", 0) or 0),
                "conflict_pairs": conflicts,
                "conflict_resolution_rate": round(resolution_rate, 4),
                "constraint_satisfaction": constraint,
                "visual_readability": round(visual_score, 2),
                "pattern_preservation": pattern,
                "main_feedback": labels[stage]["main_feedback"],
                "next_strategy": labels[stage]["next_strategy"],
            }
        )
    return rows


def write_iteration_metrics(output_dir: Path, rows: List[Dict[str, Any]]) -> None:
    fields = [
        "stage",
        "stage_label",
        "point_count",
        "conflict_pairs",
        "conflict_resolution_rate",
        "constraint_satisfaction",
        "visual_readability",
        "pattern_preservation",
        "main_feedback",
        "next_strategy",
    ]
    with (output_dir / "iteration_metrics.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def format_operator_params(plan: Dict[str, Any]) -> List[str]:
    lines = []
    for op in plan.get("operators", []):
        if not isinstance(op, dict):
            continue
        lines.append(f"- `{op.get('name')}`: `{json.dumps(op.get('params', {}), ensure_ascii=False)}`")
    return lines


def write_round_logs(
    output_dir: Path,
    stages: Dict[str, Dict[str, Any]],
    profile: Dict[str, Any],
    source_input: Path,
    source_category_field: str,
    target_scale: int,
) -> None:
    labels = stage_names()
    for stage in ["R0", "R1", "R2", "R3"]:
        item = stages[stage]
        qa = item["qa_metrics"]
        verify = item["verify"]
        plan = item.get("plan", {})
        lines = [
            f"# {labels[stage]['label']} 智能体日志",
            "",
            "## 任务解析结果",
            f"- 样本区域: `TestDatasets/Scale_1_86000/sample_13.geojson`",
            f"- 规范化输入: `{item['geojson']}`" if stage == "R0" else f"- 本轮图层: `{item['geojson']}`",
            f"- 原始类别字段: `{source_category_field}`",
            "- 归一类别字段: `poi_category`",
            f"- 目标比例尺: `1:{target_scale}`",
            f"- 语义类型: `{profile.get('semantic_type', '')}`",
            f"- 空间格局: `{profile.get('structure_type', '')}`",
            "",
            "## 匹配到的 Skills",
        ]
        lines.extend([f"- {name}" for name in skill_list_for_stage(stage)])
        lines.extend(
            [
                "",
                "## 参数设置",
                f"- `min_symbol_separation_m`: `{qa.get('min_symbol_separation_m')}`",
                f"- `point_count`: `{qa.get('output_count')}`",
                f"- `retention_rate`: `{round(float(qa.get('retention_rate', 0.0) or 0.0), 4)}`",
                f"- `poi_symbol_categories`: `{', '.join(qa.get('poi_symbol_categories', []))}`",
            ]
        )
        if stage != "R0":
            lines.append(f"- `agent_internal_round`: `{item.get('agent_round')}`")
            lines.extend(format_operator_params(plan))
        lines.extend(
            [
                "",
                "## 工具调用记录",
            ]
        )
        if stage == "R0":
            lines.append("- 工具: `GeoPandas.read_file`、`assign_poi_symbols`、`render_map`。")
        else:
            lines.append("- 工具: `run_python_script`。")
            lines.append("- 调用入口: `src/agent/core.py::MapGenAgent._run_scale_generalization`。")
            lines.append("- 执行器: `src/agent/tools.py::run_python_script`。")
        lines.extend(
            [
                "",
                "## VLM/质量评价反馈",
                f"- `vlm_mode`: `{verify.get('vlm_mode', 'heuristic_fallback')}`",
                f"- `real_vlm_called`: `{verify.get('real_vlm_called', False)}`",
                f"- `pass`: `{verify.get('pass')}`",
                f"- `score`: `{verify.get('score')}`",
                f"- `failed_rules`: `{json.dumps(verify.get('failed_rules', []), ensure_ascii=False)}`",
                f"- `conflict_pairs`: `{qa.get('conflict_count_total')}`",
                f"- `point_region_coverage_ratio`: `{qa.get('point_region_coverage_ratio')}`",
                f"- `point_hotspot_coverage_ratio`: `{qa.get('point_hotspot_coverage_ratio')}`",
                f"- `point_sparse_region_coverage_ratio`: `{qa.get('point_sparse_region_coverage_ratio')}`",
                f"- `point_dense_region_singleton_ratio`: `{qa.get('point_dense_region_singleton_ratio')}`",
                "",
                "## 下一轮修正策略",
                labels[stage]["next_strategy"],
                "",
                "## 输出文件",
                f"- 地图 PNG: `{item['png']}`",
                f"- 地图 SVG: `{item['svg']}`",
                f"- 点图层 GeoJSON: `{item['geojson']}`",
                f"- 点图层 CSV: `{item['csv']}`",
            ]
        )
        (output_dir / f"{labels[stage]['label']}_agent_log.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary(
    output_dir: Path,
    source_input: Path,
    agent_input: Path,
    target_scale: int,
    stages: Dict[str, Dict[str, Any]],
) -> None:
    lines = [
        "# PointGenAI 迭代过程实验序列说明",
        "",
        f"- 样本区域: `{source_input}`",
        f"- 智能体输入: `{agent_input}`",
        f"- 目标比例尺: `1:{target_scale}`",
        "- 比例尺选择说明: 本次正式输出采用优先目标比例尺 1:250,000。运行时未注册宽松本地 VLM 回调，使用项目 `src/agent/vlm_verifier.py` 的质量评价路径，因此 PointGenAI 生成了可审计的多轮反馈序列。",
        "- 符号来源: `src/ui/point_symbol_core.py`、`src/ui/point_symbol_render.py` 和 `output/arcgis_10_2_poi_symbols/` 中已有 8 类 POI 符号。",
        "- 未新增 POI 类别、颜色、图标或类别样式。",
        "",
        "## 阶段文件",
        "",
        "| 阶段 | 地图 | 点图层 | 日志 |",
        "|---|---|---|---|",
    ]
    labels = stage_names()
    for stage in ["R0", "R1", "R2", "R3"]:
        item = stages[stage]
        label = labels[stage]["label"]
        lines.append(f"| {stage} | `{item['png'].name}` / `{item['svg'].name}` | `{item['geojson'].name}` / `{item['csv'].name}` | `{label}_agent_log.md` |")
    lines.extend(
        [
            "",
            "## 关键指标",
            "",
            "详见 `iteration_metrics.csv`。",
            "",
            "## 智能体原始记录",
            "",
            "- `agent_reply.txt`: PointGenAI 返回文本。",
            "- `agent_events.json`: 事件、工具调用和每轮 stdout 指标摘要。",
            "- `agent_internal_rounds.json`: 智能体内部 R1-R4 输出路径。",
        ]
    )
    (output_dir / "experiment_sequence_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def safe_clean(output_dir: Path) -> None:
    if not output_dir.exists():
        return
    resolved = output_dir.resolve()
    output_root = (PROJECT_ROOT / "output").resolve()
    if output_root not in resolved.parents and resolved != output_root:
        raise ValueError(f"Refusing to clean outside project output/: {resolved}")
    shutil.rmtree(resolved)


def main() -> int:
    args = parse_args()
    configure_plot()
    source_input = resolve_path(args.input)
    output_dir = resolve_path(args.output_dir)
    if args.clean:
        safe_clean(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    agent_input, reference, source_category_field = prepare_agent_input(source_input, output_dir)
    agent_probe = MapGenAgent(provider=NullProvider())
    profile = agent_probe._infer_point_semantic_profile(str(agent_input))

    reply, events = run_pointgenai(agent_input, output_dir, args.source_scale, args.target_scale)
    rounds = generated_rounds(agent_input, args.target_scale)
    if len(rounds) < 3:
        raise RuntimeError(
            f"Expected at least three PointGenAI rounds for R1/R2/R3, found {len(rounds)}. "
            "Use heuristic VLM mode without a permissive callback."
        )

    selected = {"R1": 1, "R2": 2, "R3": 3}
    labels = stage_names()
    icons = load_icons()
    stages: Dict[str, Dict[str, Any]] = {
        "R0": {
            "gdf": reference,
            "agent_round": 0,
            "geojson": output_dir / "R0_input.geojson",
            "csv": output_dir / "R0_input.csv",
            "png": output_dir / "R0_input.png",
            "svg": output_dir / "R0_input.svg",
        }
    }
    for stage, round_idx in selected.items():
        gdf = gpd.read_file(rounds[round_idx]["geojson"])
        if gdf.crs is None:
            gdf = gdf.set_crs(reference.crs, allow_override=True)
        stages[stage] = {
            "gdf": gdf,
            "agent_round": round_idx,
            "source_geojson": rounds[round_idx]["geojson"],
            "source_png": rounds[round_idx]["png"],
            "source_pdf": rounds[round_idx]["pdf"],
            "geojson": output_dir / f"{labels[stage]['label']}.geojson",
            "csv": output_dir / f"{labels[stage]['label']}.csv",
            "png": output_dir / f"{labels[stage]['label']}.png",
            "svg": output_dir / f"{labels[stage]['label']}.svg",
        }

    for stage in ["R0", "R1", "R2", "R3"]:
        item = stages[stage]
        write_layer_files(item["gdf"], item["geojson"], item["csv"])
        render_map(
            item["gdf"],
            item["png"],
            item["svg"],
            labels[stage]["title"],
            args.target_scale,
            reference,
            icons,
            highlight_conflicts=(stage == "R0"),
        )
        item["qa_metrics"] = stage_qa_metrics(item["gdf"], reference, args.target_scale)

    reconstruct_plans_and_verify(stages, reference, profile, args.source_scale, args.target_scale)
    metric_rows = compute_public_metrics(stages)
    write_iteration_metrics(output_dir, metric_rows)
    write_round_logs(output_dir, stages, profile, source_input, source_category_field, args.target_scale)
    write_summary(output_dir, source_input, agent_input, args.target_scale, stages)

    internal = {
        "source_input": str(source_input),
        "agent_input": str(agent_input),
        "target_scale": int(args.target_scale),
        "pointgenai_reply_excerpt": reply[:4000],
        "selected_rounds": {stage: stages[stage].get("agent_round") for stage in ["R1", "R2", "R3"]},
        "all_agent_rounds": {
            str(idx): {key: str(path) for key, path in value.items()} for idx, value in sorted(rounds.items())
        },
        "events_count": len(events),
    }
    (output_dir / "agent_internal_rounds.json").write_text(json.dumps(internal, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
