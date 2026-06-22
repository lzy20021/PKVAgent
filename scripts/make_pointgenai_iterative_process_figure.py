from __future__ import annotations

import csv
import json
import math
import sys
import textwrap
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.offsetbox import AnnotationBbox, DrawingArea
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

from src.ui.point_symbol_core import POI_SYMBOL_RULES


PROCESS_DIR = PROJECT_ROOT / "output" / "pointgenai_process_sequence_sample13_1_250000"
OUT_SVG = PROCESS_DIR / "PointGenAI_iterative_process.svg"
OUT_PNG = PROCESS_DIR / "PointGenAI_iterative_process.png"
OUT_TIF = PROCESS_DIR / "PointGenAI_iterative_process.tif"
OUT_CAPTION = PROCESS_DIR / "PointGenAI_iterative_process_caption.txt"
OUT_README = PROCESS_DIR / "PointGenAI_iterative_process_readme.md"

CATEGORY_ORDER = list(POI_SYMBOL_RULES.keys())
STAGE_ORDER = ["R0", "R1", "R2", "R3"]
STAGE_FILES = {
    "R0": "R0_input",
    "R1": "R1_iter1",
    "R2": "R2_iter2",
    "R3": "R3_final",
}
STAGE_TITLES = {
    "R0": "R0 原始输入",
    "R1": "R1 第一轮综合",
    "R2": "R2 第二轮修正",
    "R3": "R3 最终结果",
}
CARD_TITLES = {
    "R0": "任务解析与问题识别",
    "R1": "初始策略生成与 Skills 调用",
    "R2": "VLM 反馈驱动的冲突消解与均衡优化",
    "R3": "视觉可读性评估与最终校核",
}
FLOW_LABELS = [
    "任务解析 → Skills 匹配 → 初始参数配置",
    "VLM 反馈：残留符号冲突 / 局部图面失衡",
    "结果反馈修正：调整最小视觉间距、保留比例、局部位移与均衡约束",
]
METRIC_LABELS = {
    "conflict_pairs": "conflict_pairs",
    "conflict_resolution_rate": "conflict_resolution_rate",
    "visual_readability": "visual_readability",
    "constraint_satisfaction": "constraint_satisfaction",
}


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


def wrap_zh(text: str, width: int) -> str:
    parts: List[str] = []
    for raw in str(text).splitlines():
        raw = raw.strip()
        if not raw:
            parts.append("")
            continue
        current = ""
        current_width = 0
        for ch in raw:
            ch_width = 2 if unicodedata.east_asian_width(ch) in {"W", "F"} else 1
            if current and current_width + ch_width > width:
                parts.append(current.rstrip())
                current = "" if ch.isspace() else ch
                current_width = 0 if ch.isspace() else ch_width
            else:
                current += ch
                current_width += ch_width
        if current:
            parts.append(current.rstrip())
    return "\n".join(parts)


def read_metrics() -> Dict[str, Dict[str, Any]]:
    path = PROCESS_DIR / "iteration_metrics.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        stage = row["stage"]
        row["point_count"] = int(float(row["point_count"]))
        row["conflict_pairs"] = int(float(row["conflict_pairs"]))
        row["conflict_resolution_rate"] = float(row["conflict_resolution_rate"])
        row["constraint_satisfaction"] = float(row["constraint_satisfaction"])
        row["visual_readability"] = float(row["visual_readability"])
        row["pattern_preservation"] = float(row["pattern_preservation"])
        out[stage] = row
    return out


def read_layers() -> Dict[str, gpd.GeoDataFrame]:
    layers: Dict[str, gpd.GeoDataFrame] = {}
    for stage, stem in STAGE_FILES.items():
        gdf = gpd.read_file(PROCESS_DIR / f"{stem}.geojson")
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326", allow_override=True)
        layers[stage] = gdf
    return layers


def extract_skills(stage: str) -> List[str]:
    stem = STAGE_FILES[stage]
    path = PROCESS_DIR / f"{stem}_agent_log.md"
    text = path.read_text(encoding="utf-8")
    skills: List[str] = []
    in_block = False
    for line in text.splitlines():
        if line.strip() == "## 匹配到的 Skills":
            in_block = True
            continue
        if in_block and line.startswith("## "):
            break
        if in_block and line.strip().startswith("- "):
            skills.append(line.strip()[2:].strip())
    return skills


def display_extent(layers: Dict[str, gpd.GeoDataFrame]) -> Tuple[float, float, float, float]:
    ref = layers["R0"].to_crs("EPSG:4326")
    minx, miny, maxx, maxy = [float(v) for v in ref.total_bounds]
    dx = max(maxx - minx, 0.002)
    dy = max(maxy - miny, 0.002)
    pad = 0.07
    return (minx - dx * pad, miny - dy * pad, maxx + dx * pad, maxy + dy * pad)


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
    return gdf.to_crs("EPSG:3857")


def conflict_pairs(gdf: gpd.GeoDataFrame, target_scale: int = 250000, max_pairs: int = 260) -> List[Tuple[int, int]]:
    if len(gdf) <= 1:
        return []
    work = projected(gdf).reset_index(drop=True)
    coords = np.column_stack([work.geometry.x.to_numpy(), work.geometry.y.to_numpy()])
    threshold = 0.30 * float(target_scale) / 1000.0
    pairs: List[Tuple[int, int]] = []
    for i in range(len(coords)):
        xi, yi = coords[i]
        for j in range(i + 1, len(coords)):
            xj, yj = coords[j]
            if math.hypot(float(xi - xj), float(yi - yj)) < threshold:
                pairs.append((i, j))
                if len(pairs) >= max_pairs:
                    return pairs
    return pairs


def icon_box(category: str, size: int) -> DrawingArea:
    color = POI_SYMBOL_RULES[category]["color"]
    da = DrawingArea(size, size, 0, 0)
    bg = FancyBboxPatch(
        (0.5, 0.5),
        size - 1,
        size - 1,
        boxstyle=f"round,pad=0,rounding_size={max(2, size * 0.16)}",
        facecolor=color,
        edgecolor="#1F2933",
        linewidth=max(0.45, size * 0.040),
    )
    da.add_artist(bg)
    w = "white"
    lw = max(0.72, size * 0.070)
    s = float(size)

    if category == "便利店":
        da.add_artist(Rectangle((s * 0.25, s * 0.39), s * 0.50, s * 0.34, fill=False, edgecolor=w, linewidth=lw))
        da.add_artist(
            Polygon(
                [[s * 0.22, s * 0.40], [s * 0.50, s * 0.22], [s * 0.78, s * 0.40]],
                closed=False,
                edgecolor=w,
                linewidth=lw,
                fill=False,
            )
        )
        da.add_artist(Rectangle((s * 0.37, s * 0.52), s * 0.12, s * 0.21, facecolor=w, edgecolor=w, linewidth=0))
        da.add_artist(Rectangle((s * 0.55, s * 0.52), s * 0.12, s * 0.21, facecolor=w, edgecolor=w, linewidth=0))
    elif category == "学校":
        da.add_artist(
            Polygon(
                [[s * 0.15, s * 0.43], [s * 0.50, s * 0.25], [s * 0.85, s * 0.43], [s * 0.50, s * 0.60]],
                closed=True,
                facecolor=w,
                edgecolor=w,
                linewidth=0,
            )
        )
        da.add_artist(Rectangle((s * 0.32, s * 0.55), s * 0.36, s * 0.09, facecolor=w, edgecolor=w, linewidth=0))
        da.add_artist(Rectangle((s * 0.76, s * 0.43), s * 0.045, s * 0.28, facecolor=w, edgecolor=w, linewidth=0))
        da.add_artist(Circle((s * 0.785, s * 0.74), s * 0.035, facecolor=w, edgecolor=w, linewidth=0))
    elif category == "停车场":
        da.add_artist(
            plt.Text(
                s * 0.50,
                s * 0.51,
                "P",
                color=w,
                ha="center",
                va="center",
                fontsize=s * 0.78,
                fontweight="bold",
            )
        )
    elif category == "酒店":
        da.add_artist(Rectangle((s * 0.18, s * 0.51), s * 0.64, s * 0.20, facecolor=w, edgecolor=w, linewidth=0))
        da.add_artist(Rectangle((s * 0.18, s * 0.36), s * 0.12, s * 0.35, facecolor=w, edgecolor=w, linewidth=0))
        da.add_artist(Rectangle((s * 0.31, s * 0.39), s * 0.22, s * 0.11, facecolor=w, edgecolor=w, linewidth=0))
        da.add_artist(Rectangle((s * 0.18, s * 0.72), s * 0.64, s * 0.06, facecolor=w, edgecolor=w, linewidth=0))
    elif category == "银行":
        da.add_artist(
            Polygon(
                [[s * 0.16, s * 0.39], [s * 0.50, s * 0.18], [s * 0.84, s * 0.39]],
                closed=True,
                facecolor=w,
                edgecolor=w,
                linewidth=0,
            )
        )
        for x in [0.27, 0.43, 0.59, 0.75]:
            da.add_artist(Rectangle((s * x - s * 0.035, s * 0.43), s * 0.07, s * 0.31, facecolor=w, edgecolor=w, linewidth=0))
        da.add_artist(Rectangle((s * 0.17, s * 0.77), s * 0.66, s * 0.07, facecolor=w, edgecolor=w, linewidth=0))
    elif category == "医院":
        da.add_artist(Rectangle((s * 0.43, s * 0.20), s * 0.14, s * 0.60, facecolor=w, edgecolor=w, linewidth=0))
        da.add_artist(Rectangle((s * 0.20, s * 0.43), s * 0.60, s * 0.14, facecolor=w, edgecolor=w, linewidth=0))
    elif category == "餐饮":
        da.add_artist(Rectangle((s * 0.31, s * 0.24), s * 0.055, s * 0.54, facecolor=w, edgecolor=w, linewidth=0))
        for x in [0.24, 0.31, 0.38]:
            da.add_artist(Rectangle((s * x, s * 0.24), s * 0.035, s * 0.25, facecolor=w, edgecolor=w, linewidth=0))
        da.add_artist(Rectangle((s * 0.62, s * 0.25), s * 0.06, s * 0.55, facecolor=w, edgecolor=w, linewidth=0))
        da.add_artist(
            Polygon(
                [[s * 0.58, s * 0.25], [s * 0.73, s * 0.25], [s * 0.68, s * 0.48], [s * 0.62, s * 0.48]],
                closed=True,
                facecolor=w,
                edgecolor=w,
                linewidth=0,
            )
        )
    elif category == "交通设施":
        da.add_artist(Rectangle((s * 0.22, s * 0.31), s * 0.56, s * 0.36, facecolor=w, edgecolor=w, linewidth=0))
        da.add_artist(Rectangle((s * 0.28, s * 0.37), s * 0.18, s * 0.13, facecolor=color, edgecolor=color, linewidth=0))
        da.add_artist(Rectangle((s * 0.54, s * 0.37), s * 0.18, s * 0.13, facecolor=color, edgecolor=color, linewidth=0))
        da.add_artist(Circle((s * 0.33, s * 0.70), s * 0.055, facecolor=w, edgecolor=w, linewidth=0))
        da.add_artist(Circle((s * 0.67, s * 0.70), s * 0.055, facecolor=w, edgecolor=w, linewidth=0))
    return da


def draw_context_grid(ax: plt.Axes, extent: Tuple[float, float, float, float]) -> None:
    minx, miny, maxx, maxy = extent
    width = maxx - minx
    height = maxy - miny
    ax.add_patch(Rectangle((minx, miny), width, height, facecolor="#F8FAFC", edgecolor="#D7DEE8", linewidth=0.75, zorder=0))
    for frac in np.linspace(0.18, 0.82, 4):
        ax.plot([minx + width * 0.04, maxx - width * 0.04], [miny + height * frac, miny + height * frac], color="#E8EEF5", linewidth=0.55, zorder=1)
        ax.plot([minx + width * frac, minx + width * frac], [miny + height * 0.04, maxy - height * 0.04], color="#E8EEF5", linewidth=0.55, zorder=1)


def draw_density_overlay(ax: plt.Axes, gdf: gpd.GeoDataFrame, extent: Tuple[float, float, float, float]) -> None:
    work = gdf.to_crs("EPSG:4326").reset_index(drop=True)
    minx, miny, maxx, maxy = extent
    width = maxx - minx
    height = maxy - miny
    bins = 9
    gx = np.floor((work.geometry.x - minx) / max(width, 1e-9) * bins).clip(0, bins - 1).astype(int)
    gy = np.floor((work.geometry.y - miny) / max(height, 1e-9) * bins).clip(0, bins - 1).astype(int)
    cells = pd.Series([f"{int(x)}_{int(y)}" for x, y in zip(gx.tolist(), gy.tolist())]).value_counts()
    if cells.empty:
        return
    threshold = max(float(cells.quantile(0.88)), 8.0)
    for key, value in cells.items():
        if float(value) < threshold:
            continue
        x, y = [int(v) for v in str(key).split("_")]
        ax.add_patch(
            Rectangle(
                (minx + width * x / bins, miny + height * y / bins),
                width / bins,
                height / bins,
                facecolor="#DC2626",
                edgecolor="#991B1B",
                linewidth=0.45,
                alpha=0.10,
                zorder=2,
            )
        )


def draw_map_panel(
    ax: plt.Axes,
    gdf: gpd.GeoDataFrame,
    extent: Tuple[float, float, float, float],
    stage: str,
    metrics: Dict[str, Any],
) -> None:
    work = gdf.to_crs("EPSG:4326").reset_index(drop=True)
    ax.set_xlim(extent[0], extent[2])
    ax.set_ylim(extent[1], extent[3])
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    draw_context_grid(ax, extent)
    if stage == "R0":
        draw_density_overlay(ax, work, extent)
        pairs = conflict_pairs(gdf, max_pairs=280)
        for i, j in pairs:
            p0 = work.geometry.iloc[i]
            p1 = work.geometry.iloc[j]
            ax.plot([p0.x, p1.x], [p0.y, p1.y], color="#991B1B", linewidth=0.25, alpha=0.10, zorder=3)

    order = {cat: idx for idx, cat in enumerate(CATEGORY_ORDER)}
    work["_draw_order"] = work["poi_category"].map(order).fillna(99)
    work = work.sort_values("_draw_order")
    size = 7 if len(work) > 600 else (8 if len(work) > 200 else 9)
    for _, row in work.iterrows():
        cat = str(row.get("poi_category", ""))
        if cat not in POI_SYMBOL_RULES:
            continue
        ax.add_artist(
            AnnotationBbox(
                icon_box(cat, size=size),
                (row.geometry.x, row.geometry.y),
                xycoords="data",
                frameon=False,
                box_alignment=(0.5, 0.5),
                pad=0,
                annotation_clip=True,
                zorder=8,
            )
        )

    ax.text(
        0.02,
        0.96,
        f"n={metrics['point_count']}  conflicts={metrics['conflict_pairs']}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.2,
        color="#263545",
        bbox=dict(boxstyle="round,pad=0.20", facecolor="white", edgecolor="#D7DEE8", linewidth=0.45, alpha=0.92),
    )


def draw_card(fig: plt.Figure, rect: Tuple[float, float, float, float], stage: str, metric: Dict[str, Any], skills: List[str]) -> None:
    ax = fig.add_axes(rect)
    ax.set_axis_off()
    bg = FancyBboxPatch(
        (0, 0),
        1,
        1,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        transform=ax.transAxes,
        facecolor="#FFFFFF",
        edgecolor="#D7DEE8",
        linewidth=0.9,
    )
    ax.add_patch(bg)
    ax.text(0.045, 0.90, CARD_TITLES[stage], ha="left", va="top", fontsize=8.6, fontweight="bold", color="#1F2937", transform=ax.transAxes)

    skills_text = "、".join(skills)
    lines = [
        f"点数 {metric['point_count']}；冲突 {metric['conflict_pairs']}；可读性 {metric['visual_readability']:.0f}",
        "Skills: " + skills_text,
    ]
    if stage == "R0":
        lines.append("识别：符号压盖、局部拥挤、高密度聚集")
    elif stage == "R1":
        lines.append("反馈：热点过绘与局部图面负载仍需校核")
    elif stage == "R2":
        lines.append("修正：降低保留比例并加强视觉间距约束")
    else:
        lines.append("校核：8 类符号保留，冲突消解率 1.0000")

    y = 0.73
    for idx, line in enumerate(lines):
        wrapped = wrap_zh(line, 45)
        ax.text(
            0.055,
            y,
            wrapped,
            ha="left",
            va="top",
            fontsize=7.0 if idx == 1 else 7.4,
            color="#374151",
            linespacing=1.18,
            transform=ax.transAxes,
        )
        line_count = max(1, wrapped.count("\n") + 1)
        y -= 0.105 * line_count + (0.050 if idx == 1 else 0.030)


def draw_flow_arrows(fig: plt.Figure, centers: List[float], y: float) -> None:
    for i, label in enumerate(FLOW_LABELS):
        x0 = centers[i] + 0.091
        x1 = centers[i + 1] - 0.091
        arrow = FancyArrowPatch(
            (x0, y),
            (x1, y),
            transform=fig.transFigure,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.25,
            color="#607D99",
            shrinkA=0,
            shrinkB=0,
            zorder=20,
        )
        fig.add_artist(arrow)
        fig.text(
            (x0 + x1) / 2,
            y + 0.030,
            wrap_zh(label, 30 if i < 2 else 36),
            ha="center",
            va="bottom",
            fontsize=6.4,
            color="#36516A",
            linespacing=1.08,
            bbox=dict(boxstyle="round,pad=0.22", facecolor="#F8FAFC", edgecolor="#D7DEE8", linewidth=0.55, alpha=0.95),
            zorder=21,
        )


def normalize_metric(values: List[float], invert: bool = False) -> List[float]:
    arr = np.array(values, dtype=float)
    if invert:
        arr = arr.max() - arr
    mn = float(arr.min())
    mx = float(arr.max())
    if abs(mx - mn) < 1e-12:
        return [0.70 for _ in arr]
    return ((arr - mn) / (mx - mn) * 0.70 + 0.15).tolist()


def draw_metrics_band(fig: plt.Figure, metrics: Dict[str, Dict[str, Any]], centers: List[float]) -> None:
    ax = fig.add_axes([0.055, 0.052, 0.89, 0.125])
    ax.set_axis_off()
    ax.add_patch(
        FancyBboxPatch(
            (0, 0),
            1,
            1,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            transform=ax.transAxes,
            facecolor="#FFFFFF",
            edgecolor="#D7DEE8",
            linewidth=0.9,
        )
    )
    ax.text(0.020, 0.88, "指标变化带 / Metric trend", ha="left", va="top", fontsize=8.5, fontweight="bold", color="#1F2937", transform=ax.transAxes)

    xs = np.linspace(0.23, 0.95, 4)
    for x, stage in zip(xs, STAGE_ORDER):
        ax.text(x, 0.90, stage, ha="center", va="top", fontsize=7.4, color="#4B5563", transform=ax.transAxes)
        ax.plot([x, x], [0.10, 0.80], color="#EEF2F7", linewidth=0.7, transform=ax.transAxes, zorder=1)

    rows = [
        ("conflict_pairs", [metrics[s]["conflict_pairs"] for s in STAGE_ORDER], False, "#D35B5B", "↓"),
        ("conflict_resolution_rate", [metrics[s]["conflict_resolution_rate"] for s in STAGE_ORDER], False, "#2E7D32", "↑"),
        ("visual_readability", [metrics[s]["visual_readability"] for s in STAGE_ORDER], False, "#2F5D8A", "↑"),
        ("constraint_satisfaction", [metrics[s]["constraint_satisfaction"] for s in STAGE_ORDER], False, "#7B4FA1", "↑"),
    ]
    y_rows = [0.68, 0.51, 0.34, 0.17]
    for (name, values, invert, color, arrow), y0 in zip(rows, y_rows):
        yn = normalize_metric([float(v) for v in values], invert=invert)
        y_points = [y0 - 0.055 + v * 0.12 for v in yn]
        ax.text(0.020, y0, f"{name} {arrow}", ha="left", va="center", fontsize=7.1, color="#374151", transform=ax.transAxes)
        ax.plot(xs, y_points, color=color, linewidth=1.35, marker="o", markersize=3.2, transform=ax.transAxes, zorder=5)
        for x, yp, val in zip(xs, y_points, values):
            if isinstance(val, float):
                label = f"{val:.2f}" if name == "visual_readability" else f"{val:.3f}"
            else:
                label = str(val)
            ax.text(x, yp + 0.028, label, ha="center", va="bottom", fontsize=5.9, color=color, transform=ax.transAxes)


def draw_legend(fig: plt.Figure) -> None:
    ax = fig.add_axes([0.065, 0.185, 0.87, 0.035])
    ax.set_axis_off()
    xs = np.linspace(0.025, 0.975, len(CATEGORY_ORDER))
    for x, cat in zip(xs, CATEGORY_ORDER):
        ax.add_artist(
            AnnotationBbox(
                icon_box(cat, size=9),
                (x, 0.56),
                xycoords=ax.transAxes,
                frameon=False,
                box_alignment=(0.5, 0.5),
                pad=0,
            )
        )
        ax.text(x, 0.08, cat, transform=ax.transAxes, ha="center", va="bottom", fontsize=6.2, color="#374151")


def build_figure() -> plt.Figure:
    configure_plot()
    metrics = read_metrics()
    layers = read_layers()
    skills = {stage: extract_skills(stage) for stage in STAGE_ORDER}
    extent = display_extent(layers)

    fig = plt.figure(figsize=(16.0, 9.0), facecolor="#F5F7FA")
    fig.text(0.055, 0.965, "PointGenAI 点符号综合迭代优化过程示意图", ha="left", va="top", fontsize=17, fontweight="bold", color="#111827")
    fig.text(
        0.055,
        0.931,
        "Iterative optimization process of PointGenAI for point symbol generalization",
        ha="left",
        va="top",
        fontsize=11.2,
        color="#4B5563",
    )
    fig.text(0.945, 0.954, "样本：南京市建邺区 / 目标比例尺 1:250,000", ha="right", va="top", fontsize=8.5, color="#526272")

    centers = [0.145, 0.385, 0.625, 0.865]
    map_w = 0.205
    map_h = 0.315
    map_y = 0.535
    card_y = 0.255
    card_h = 0.205
    card_w = 0.205

    for center, stage in zip(centers, STAGE_ORDER):
        fig.text(center, 0.872, STAGE_TITLES[stage], ha="center", va="center", fontsize=10.6, fontweight="bold", color="#1F2937")
        ax = fig.add_axes([center - map_w / 2, map_y, map_w, map_h])
        draw_map_panel(ax, layers[stage], extent, stage, metrics[stage])
        draw_card(fig, (center - card_w / 2, card_y, card_w, card_h), stage, metrics[stage], skills[stage])

    draw_flow_arrows(fig, centers, y=0.492)
    draw_legend(fig)
    draw_metrics_band(fig, metrics, centers)

    fig.text(
        0.055,
        0.018,
        "Note: stage maps are redrawn from R0-R3 GeoJSON layers; POI symbols reuse project category rules and existing classic POI symbol design.",
        ha="left",
        va="bottom",
        fontsize=6.6,
        color="#6B7280",
    )
    return fig


def write_caption(metrics: Dict[str, Dict[str, Any]]) -> None:
    r0 = metrics["R0"]
    r1 = metrics["R1"]
    r2 = metrics["R2"]
    r3 = metrics["R3"]
    text = f"""图名：PointGenAI 点符号综合迭代优化过程示意图
English title: Iterative optimization process of PointGenAI for point symbol generalization

图注：
图X展示了 PointGenAI 在同一样本区域内的点符号综合迭代优化过程。系统首先根据目标比例尺、点要素属性、空间分布特征和符号约束识别原始图面中的符号压盖与局部拥挤问题，并匹配重要点优先保留、比例尺驱动删减、密集区点筛选和结构保持等 Skills 生成第一轮综合结果。随后，VLM/质量评价从符号冲突、图面均衡、格局保持和符号表达合理性等方面对第一轮结果进行评价，并将残留热点过绘和局部失衡问题反馈至决策模块。智能体据此调整最小视觉间距、保留比例、局部位移和均衡约束，进一步调用邻近冲突检测、邻近冲突消解、均衡优化、符号压盖校核和结果反馈修正等 Skills，最终生成符号冲突更少、图面负载更均衡且主要空间格局得到保持的综合结果。本实验中，点数由 R0 的 {r0['point_count']} 个逐轮减少至 R1 的 {r1['point_count']} 个、R2 的 {r2['point_count']} 个和 R3 的 {r3['point_count']} 个；符号冲突点对由 {r0['conflict_pairs']} 降至 {r3['conflict_pairs']}，冲突消解率由 {r0['conflict_resolution_rate']:.4f} 提高至 {r3['conflict_resolution_rate']:.4f}；视觉可读性评分由 {r0['visual_readability']:.0f} 提高至 {r3['visual_readability']:.0f}；约束满足率由 {r0['constraint_satisfaction']:.4f} 提高至最终 {r3['constraint_satisfaction']:.4f}。

Caption:
Figure X illustrates the iterative optimization process of PointGenAI for point symbol generalization in the same sample area. The system first identifies symbol overlap, local congestion, and dense clusters from the target scale, point attributes, spatial distribution, and symbol constraints, then generates the first-round generalized result by matching Skills such as important-point retention, scale-driven reduction, dense-area point filtering, and structure preservation. VLM/quality evaluation then assesses symbol conflicts, visual balance, pattern preservation, and symbol validity, and feeds hotspot overdraw and local imbalance back to the decision module. The agent adjusts the minimum visual spacing, retention ratio, local displacement, and balance constraints, and further invokes proximity conflict detection, proximity conflict resolution, balance optimization, symbol-overlap checking, and result feedback correction. In this experiment, the point count decreases from {r0['point_count']} in R0 to {r1['point_count']} in R1, {r2['point_count']} in R2, and {r3['point_count']} in R3. Symbol conflict pairs decrease from {r0['conflict_pairs']} to {r3['conflict_pairs']}, the conflict resolution rate increases from {r0['conflict_resolution_rate']:.4f} to {r3['conflict_resolution_rate']:.4f}, the visual readability score increases from {r0['visual_readability']:.0f} to {r3['visual_readability']:.0f}, and constraint satisfaction increases from {r0['constraint_satisfaction']:.4f} to {r3['constraint_satisfaction']:.4f}.
"""
    OUT_CAPTION.write_text(text, encoding="utf-8")


def write_readme(metrics: Dict[str, Dict[str, Any]], skills: Dict[str, List[str]]) -> None:
    lines = [
        "# PointGenAI Iterative Process Figure",
        "",
        "## Outputs",
        "",
        f"- `PointGenAI_iterative_process.svg`: editable vector layout.",
        f"- `PointGenAI_iterative_process.png`: 600 dpi preview.",
        f"- `PointGenAI_iterative_process.tif`: 600 dpi TIFF preview.",
        f"- `PointGenAI_iterative_process_caption.txt`: Chinese/English title and caption.",
        "",
        "## Data Source",
        "",
        "- Sample area: `TestDatasets/Scale_1_86000/sample_13.geojson`.",
        "- Iteration sequence: `output/pointgenai_process_sequence_sample13_1_250000/`.",
        "- Stage layers: `R0_input.geojson`, `R1_iter1.geojson`, `R2_iter2.geojson`, `R3_final.geojson`.",
        "- Target scale: `1:250,000`.",
        "",
        "## Symbol Source",
        "",
        "- Category rules: `src/ui/point_symbol_core.py::POI_SYMBOL_RULES`.",
        "- Symbol rendering logic: `src/ui/point_symbol_render.py` and the existing classic-symbol drawing style used by `scripts/render_classic_symbol_two_figures.py`.",
        "- Raster symbol reference from step 2: `output/arcgis_10_2_poi_symbols/`.",
        "- No new POI category, color, icon, or category style was introduced.",
        "",
        "## Per-Round Strategy",
        "",
    ]
    stage_desc = {
        "R0": "任务解析与问题识别",
        "R1": "初始策略生成与 Skills 调用",
        "R2": "VLM 反馈驱动的冲突消解与均衡优化",
        "R3": "视觉可读性评估与最终校核",
    }
    for stage in STAGE_ORDER:
        lines.extend(
            [
                f"### {stage} {stage_desc[stage]}",
                "",
                "- Skills: " + "、".join(skills[stage]),
                f"- point_count: {metrics[stage]['point_count']}",
                f"- conflict_pairs: {metrics[stage]['conflict_pairs']}",
                f"- conflict_resolution_rate: {metrics[stage]['conflict_resolution_rate']:.4f}",
                f"- visual_readability: {metrics[stage]['visual_readability']:.0f}",
                f"- constraint_satisfaction: {metrics[stage]['constraint_satisfaction']:.4f}",
                f"- pattern_preservation: {metrics[stage]['pattern_preservation']:.4f}",
                "",
            ]
        )
    lines.extend(
        [
            "## Generation Script",
            "",
            "- Figure script: `scripts/make_pointgenai_iterative_process_figure.py`.",
            "- Input sequence script from step 2: `scripts/export_pointgenai_process_sequence.py`.",
            "- Metrics table: `output/pointgenai_process_sequence_sample13_1_250000/iteration_metrics.csv`.",
        ]
    )
    OUT_README.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    metrics = read_metrics()
    skills = {stage: extract_skills(stage) for stage in STAGE_ORDER}
    fig = build_figure()
    fig.savefig(OUT_SVG, facecolor=fig.get_facecolor())
    fig.savefig(OUT_PNG, dpi=600, facecolor=fig.get_facecolor())
    fig.savefig(OUT_TIF, dpi=600, facecolor=fig.get_facecolor(), pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    write_caption(metrics)
    write_readme(metrics, skills)
    print(OUT_SVG)
    print(OUT_PNG)
    print(OUT_TIF)
    print(OUT_CAPTION)
    print(OUT_README)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
