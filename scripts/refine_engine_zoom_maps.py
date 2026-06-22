from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import geopandas as gpd
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.ticker import MaxNLocator

PROJECT_ROOT = Path(r"E:\GradOpening\MapGenAI")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.skills.point_generalization_dlm_dcm import PointGeneralizationEngine, SymbolRule

BASE_DIR = PROJECT_ROOT / "output" / "engine_maps_20260329_175753"
DATA_DIR = BASE_DIR / "data"
FIG_DIR = BASE_DIR / "figures"
SUMMARY_DIR = BASE_DIR / "summary"

SCALES = [84000, 253000, 532000]
CJK_FONT_PROP: FontProperties | None = None
CHOSEN_FONT_NAME = ""
CATEGORY_ORDER = [
    "\u4ea4\u901a\u8bbe\u65bd",
    "\u533b\u9662",
    "\u5b66\u6821",
    "\u9910\u996e",
    "\u4fbf\u5229\u5e97",
    "\u505c\u8f66\u573a",
    "\u9152\u5e97",
    "\u94f6\u884c",
]
SHAPE_RADIUS_FACTOR = {"circle": 1.00, "square": 1.15, "triangle": 1.22, "diamond": 1.20, "hexagon": 1.10}
RAW_SHRINK_BY_SCALE = {84000: 0.60, 253000: 0.56, 532000: 0.52}
GEN_SHRINK_BY_SCALE = {84000: 0.98, 253000: 0.96, 532000: 0.94}
CATEGORY_STYLE = {
    "\u4ea4\u901a\u8bbe\u65bd": {"color": "#4c78a8", "marker": "circle", "size": 1.00, "min_distance": 0.64, "priority": 0.96, "preserve_floor": 1},
    "\u533b\u9662": {"color": "#d1495b", "marker": "triangle", "size": 1.18, "min_distance": 0.76, "priority": 1.20, "preserve_floor": 2},
    "\u5b66\u6821": {"color": "#2f5d62", "marker": "square", "size": 1.12, "min_distance": 0.72, "priority": 1.12, "preserve_floor": 2},
    "\u9910\u996e": {"color": "#f4a261", "marker": "circle", "size": 0.96, "min_distance": 0.60, "priority": 0.90, "preserve_floor": 1},
    "\u4fbf\u5229\u5e97": {"color": "#e76f51", "marker": "circle", "size": 0.88, "min_distance": 0.58, "priority": 0.86, "preserve_floor": 1},
    "\u505c\u8f66\u573a": {"color": "#8d99ae", "marker": "square", "size": 0.92, "min_distance": 0.65, "priority": 0.88, "preserve_floor": 1},
    "\u9152\u5e97": {"color": "#7b6d8d", "marker": "hexagon", "size": 1.00, "min_distance": 0.70, "priority": 0.95, "preserve_floor": 1},
    "\u94f6\u884c": {"color": "#2a9d8f", "marker": "diamond", "size": 1.00, "min_distance": 0.70, "priority": 1.00, "preserve_floor": 1},
}


def choose_cjk_font() -> FontProperties:
    candidates = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "PingFang SC",
        "WenQuanYi Zen Hei",
    ]
    available = {font.name: font.fname for font in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            return FontProperties(fname=available[name])
    return FontProperties(family="DejaVu Sans")


def configure_plotting() -> None:
    global CJK_FONT_PROP, CHOSEN_FONT_NAME
    CJK_FONT_PROP = choose_cjk_font()
    CHOSEN_FONT_NAME = CJK_FONT_PROP.get_name()
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.family": CHOSEN_FONT_NAME,
        "font.sans-serif": [CHOSEN_FONT_NAME],
        "font.serif": [CHOSEN_FONT_NAME],
        "axes.unicode_minus": False,
        "axes.titlesize": 11,
        "axes.labelsize": 8.0,
        "xtick.labelsize": 7.1,
        "ytick.labelsize": 7.1,
        "legend.fontsize": 7.0,
        "axes.edgecolor": "#111111",
        "axes.labelcolor": "#30363d",
        "xtick.color": "#111111",
        "ytick.color": "#111111",
        "patch.antialiased": True,
        "lines.antialiased": True,
        "savefig.dpi": 500,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "path",
    })


def apply_text_font(obj) -> None:
    if CJK_FONT_PROP is not None:
        obj.set_fontproperties(CJK_FONT_PROP)


def apply_axis_font(ax) -> None:
    if CJK_FONT_PROP is None:
        return
    ax.title.set_fontproperties(CJK_FONT_PROP)
    ax.xaxis.label.set_fontproperties(CJK_FONT_PROP)
    ax.yaxis.label.set_fontproperties(CJK_FONT_PROP)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(CJK_FONT_PROP)


def strip_svg_comments(svg_path: Path) -> None:
    text = svg_path.read_text(encoding="utf-8")
    chunks: List[str] = []
    start = 0
    while True:
        i = text.find("<!--", start)
        if i == -1:
            chunks.append(text[start:])
            break
        chunks.append(text[start:i])
        j = text.find("-->", i + 4)
        if j == -1:
            break
        start = j + 3
    svg_path.write_text("".join(chunks), encoding="utf-8")


def normalize_category_labels(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    mapped = gdf.copy()
    mapped["category"] = mapped["category"].replace({
        "\u516c\u4ea4\u7ad9": "\u4ea4\u901a\u8bbe\u65bd",
        "\u5730\u94c1\u7ad9": "\u4ea4\u901a\u8bbe\u65bd",
        "\u9910\u5385": "\u9910\u996e",
        "\u5496\u5561\u5385": "\u9910\u996e",
    })
    mapped = mapped[mapped["category"].isin(CATEGORY_ORDER)].copy()
    return mapped


def ordered_categories(categories: List[str]) -> List[str]:
    existing = set(categories)
    return [cat for cat in CATEGORY_ORDER if cat in existing]


def build_symbol_rules(categories: List[str]) -> Dict[str, SymbolRule]:
    rules: Dict[str, SymbolRule] = {}
    for cat in CATEGORY_ORDER:
        if cat in categories:
            style = CATEGORY_STYLE[cat]
            rules[cat] = SymbolRule(cat, style["size"], style["min_distance"], style["priority"], style["preserve_floor"], style["color"], style["marker"])
    return rules


def metric_radius(rule: SymbolRule, scale: int) -> float:
    return (rule.symbol_size * scale / 1000.0) * 0.5


def symbol_half_extent(rule: SymbolRule, scale: int, shrink: float = 1.0) -> float:
    return metric_radius(rule, scale) * SHAPE_RADIUS_FACTOR.get(rule.marker, 1.0) * shrink


def style_ax(ax) -> None:
    ax.set_facecolor("white")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.8)
        spine.set_edgecolor("#111111")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.tick_params(axis="both", which="major", length=3.4, width=1.15, colors="#111111", direction="out", pad=2.2)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    apply_axis_font(ax)


def add_symbol(ax, x: float, y: float, rule: SymbolRule, scale: int, alpha: float = 0.95, shrink: float = 1.0) -> None:
    edge = "#1f2933"
    radius = metric_radius(rule, scale) * shrink
    if rule.marker == "square":
        patch = RegularPolygon((x, y), 4, radius=radius * 1.15, orientation=np.pi / 4, facecolor=rule.color, edgecolor=edge, linewidth=0.45, alpha=alpha)
    elif rule.marker == "triangle":
        patch = RegularPolygon((x, y), 3, radius=radius * 1.22, orientation=0, facecolor=rule.color, edgecolor=edge, linewidth=0.45, alpha=alpha)
    elif rule.marker == "diamond":
        patch = RegularPolygon((x, y), 4, radius=radius * 1.20, orientation=0, facecolor=rule.color, edgecolor=edge, linewidth=0.45, alpha=alpha)
    elif rule.marker == "hexagon":
        patch = RegularPolygon((x, y), 6, radius=radius * 1.10, orientation=0, facecolor=rule.color, edgecolor=edge, linewidth=0.45, alpha=alpha)
    else:
        patch = Circle((x, y), radius=radius, facecolor=rule.color, edgecolor=edge, linewidth=0.45, alpha=alpha)
    ax.add_patch(patch)


def render_map(ax, gdf: gpd.GeoDataFrame, symbol_rules: Dict[str, SymbolRule], scale: int, raw_mode: bool = False) -> None:
    shrink_map = RAW_SHRINK_BY_SCALE if raw_mode else GEN_SHRINK_BY_SCALE
    shrink = shrink_map.get(scale, 0.60 if raw_mode else 0.96)
    alpha = 0.80 if raw_mode else 0.96
    fallback_rule = next(iter(symbol_rules.values()))
    for row in gdf.itertuples():
        rule = symbol_rules.get(getattr(row, "category"), fallback_rule)
        add_symbol(ax, row.geometry.x, row.geometry.y, rule, scale, alpha=alpha, shrink=shrink)


def legend_handles(symbol_rules: Dict[str, SymbolRule], categories: List[str]) -> List[Line2D]:
    shape_map = {"circle": "o", "square": "s", "triangle": "^", "diamond": "D", "hexagon": "h"}
    handles: List[Line2D] = []
    for cat in categories:
        rule = symbol_rules[cat]
        handles.append(
            Line2D(
                [0], [0],
                marker=shape_map.get(rule.marker, "o"),
                linestyle="",
                label=cat,
                markerfacecolor=rule.color,
                markeredgecolor="#222222",
                markeredgewidth=0.65,
                markersize=max(5.2, rule.symbol_size * 3.35),
            )
        )
    return handles


def choose_zoom_bounds(raw_metric: gpd.GeoDataFrame, gen_metric: gpd.GeoDataFrame, engine: PointGeneralizationEngine, scale: int, symbol_rules: Dict[str, SymbolRule]) -> Tuple[float, float, float, float]:
    selected_counts = gen_metric.groupby("grid_id").size().to_dict() if "grid_id" in gen_metric.columns else {}
    scored: List[Tuple[float, str]] = []
    for grid_id, group in raw_metric.groupby("grid_id"):
        raw = len(group)
        cats = group["category"].nunique()
        sel = int(selected_counts.get(grid_id, 0))
        score = raw * 2.4 + cats * 8.5 + (raw - sel) * 2.0
        scored.append((score, grid_id))
    scored.sort(reverse=True)
    group = raw_metric[raw_metric["grid_id"] == scored[0][1]]
    minx, miny, maxx, maxy = group.total_bounds
    fallback_rule = next(iter(symbol_rules.values()))
    max_radius = max(metric_radius(symbol_rules.get(cat, fallback_rule), scale) for cat in group["category"].unique())
    padx = max((maxx - minx) * 0.50, max_radius * 2.8)
    pady = max((maxy - miny) * 0.50, max_radius * 2.8)
    return minx - padx, miny - pady, maxx + padx, maxy + pady


def expand_bounds_for_symbols(
    bounds: Tuple[float, float, float, float],
    raw_metric: gpd.GeoDataFrame,
    gen_metric: gpd.GeoDataFrame,
    scale: int,
    symbol_rules: Dict[str, SymbolRule],
) -> Tuple[Tuple[float, float, float, float], gpd.GeoDataFrame, gpd.GeoDataFrame]:
    bx0, by0, bx1, by1 = bounds
    fallback_rule = next(iter(symbol_rules.values()))
    max_extent = max(symbol_half_extent(rule, scale, 1.0) for rule in symbol_rules.values())
    min_margin = max(80.0, max_extent * 1.10)
    for _ in range(10):
        raw_zoom = raw_metric.cx[bx0:bx1, by0:by1]
        gen_zoom = gen_metric.cx[bx0:bx1, by0:by1]
        expanded = False
        for gdf, shrink in ((raw_zoom, RAW_SHRINK_BY_SCALE.get(scale, 0.60)), (gen_zoom, GEN_SHRINK_BY_SCALE.get(scale, 0.96))):
            for row in gdf.itertuples():
                rule = symbol_rules.get(getattr(row, "category"), fallback_rule)
                extent = symbol_half_extent(rule, scale, shrink) + min_margin
                if row.geometry.x - extent < bx0:
                    bx0 = row.geometry.x - extent
                    expanded = True
                if row.geometry.x + extent > bx1:
                    bx1 = row.geometry.x + extent
                    expanded = True
                if row.geometry.y - extent < by0:
                    by0 = row.geometry.y - extent
                    expanded = True
                if row.geometry.y + extent > by1:
                    by1 = row.geometry.y + extent
                    expanded = True
        if not expanded:
            return (bx0, by0, bx1, by1), raw_zoom, gen_zoom
    return (bx0, by0, bx1, by1), raw_metric.cx[bx0:bx1, by0:by1], gen_metric.cx[bx0:bx1, by0:by1]


def build_scale_bounds(
    anchor_bounds: Tuple[float, float, float, float],
    raw_metric: gpd.GeoDataFrame,
    gen_metric: gpd.GeoDataFrame,
    scale: int,
    symbol_rules: Dict[str, SymbolRule],
) -> Tuple[Tuple[float, float, float, float], gpd.GeoDataFrame, gpd.GeoDataFrame]:
    ax0, ay0, ax1, ay1 = anchor_bounds
    center_x = (ax0 + ax1) * 0.5
    center_y = (ay0 + ay1) * 0.5
    width = ax1 - ax0
    height = ay1 - ay0
    pad = max(symbol_half_extent(rule, scale, RAW_SHRINK_BY_SCALE.get(scale, 0.60)) for rule in symbol_rules.values()) * 2.6
    initial = (
        center_x - width * 0.5 - pad,
        center_y - height * 0.5 - pad,
        center_x + width * 0.5 + pad,
        center_y + height * 0.5 + pad,
    )
    bounds, _, _ = expand_bounds_for_symbols(initial, raw_metric, gen_metric, scale, symbol_rules)
    final_pad = max(120.0, max(symbol_half_extent(rule, scale, 1.0) for rule in symbol_rules.values()) * 2.8)
    bx0, by0, bx1, by1 = bounds
    bounds = (bx0 - final_pad, by0 - final_pad, bx1 + final_pad, by1 + final_pad)
    bounds, raw_zoom, gen_zoom = expand_bounds_for_symbols(bounds, raw_metric, gen_metric, scale, symbol_rules)
    return bounds, raw_zoom, gen_zoom


def verify_bounds(bounds: Tuple[float, float, float, float], gdf: gpd.GeoDataFrame, symbol_rules: Dict[str, SymbolRule], scale: int, raw_mode: bool) -> None:
    bx0, by0, bx1, by1 = bounds
    shrink = RAW_SHRINK_BY_SCALE.get(scale, 0.60) if raw_mode else GEN_SHRINK_BY_SCALE.get(scale, 0.96)
    fallback_rule = next(iter(symbol_rules.values()))
    tolerance = 1.0
    for row in gdf.itertuples():
        rule = symbol_rules.get(getattr(row, "category"), fallback_rule)
        extent = symbol_half_extent(rule, scale, shrink)
        if row.geometry.x - extent < bx0 - tolerance or row.geometry.x + extent > bx1 + tolerance or row.geometry.y - extent < by0 - tolerance or row.geometry.y + extent > by1 + tolerance:
            raise RuntimeError(f"Symbol would be clipped for scale 1:{scale} around {getattr(row, 'category', 'unknown')}")


def export_zoom_final_v5(scale: int, anchor_bounds: Tuple[float, float, float, float]) -> None:
    raw_wgs = normalize_category_labels(gpd.read_file(DATA_DIR / f"raw_{scale}.geojson"))
    gen_wgs = normalize_category_labels(gpd.read_file(DATA_DIR / f"generalized_{scale}.geojson"))
    city = str(raw_wgs["city"].iloc[0])
    district = str(raw_wgs["district"].iloc[0])
    metric_crs = raw_wgs.estimate_utm_crs() or "EPSG:3857"
    raw_metric = raw_wgs.to_crs(metric_crs)
    gen_metric = gen_wgs.to_crs(metric_crs)
    categories = ordered_categories(raw_wgs["category"].dropna().astype(str).unique().tolist())
    symbol_rules = build_symbol_rules(categories)
    engine = PointGeneralizationEngine(symbol_rules)
    raw_metric = engine.analyze_dlm(raw_metric)
    if "grid_id" not in gen_metric.columns:
        gen_metric = gpd.sjoin_nearest(gen_metric, raw_metric[["grid_id", "geometry"]], how="left", distance_col="join_dist").drop(columns=["index_right", "join_dist"])
    bounds, raw_zoom, gen_zoom = build_scale_bounds(anchor_bounds, raw_metric, gen_metric, scale, symbol_rules)
    verify_bounds(bounds, raw_zoom, symbol_rules, scale, raw_mode=True)
    verify_bounds(bounds, gen_zoom, symbol_rules, scale, raw_mode=False)
    minx, miny, maxx, maxy = bounds

    fig = plt.figure(figsize=(14.4, 6.55))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.06, 1.06, 0.24], left=0.055, right=0.978, top=0.885, bottom=0.118, wspace=0.065)
    ax_raw = fig.add_subplot(gs[0, 0])
    ax_gen = fig.add_subplot(gs[0, 1])
    ax_leg = fig.add_subplot(gs[0, 2])
    for ax in (ax_raw, ax_gen):
        style_ax(ax)
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)

    render_map(ax_raw, raw_zoom, symbol_rules, scale, raw_mode=True)
    render_map(ax_gen, gen_zoom, symbol_rules, scale, raw_mode=False)
    ax_raw.set_title("(a) Raw zoom", pad=8, fontproperties=CJK_FONT_PROP)
    ax_gen.set_title("(b) Generalized zoom", pad=8, fontproperties=CJK_FONT_PROP)
    ax_leg.set_facecolor("white")
    ax_leg.set_axis_off()
    ax_leg.set_frame_on(False)
    for spine in ax_leg.spines.values():
        spine.set_visible(False)
    ax_leg.set_xticks([])
    ax_leg.set_yticks([])
    legend = ax_leg.legend(
        handles=legend_handles(symbol_rules, categories),
        labels=categories,
        loc="center",
        frameon=False,
        ncol=2,
        borderaxespad=0.0,
        columnspacing=1.0,
        handletextpad=0.42,
        labelspacing=0.52,
        fontsize=7.05,
        title="POI\u7c7b\u522b",
        title_fontsize=8.0,
        handlelength=1.0,
        handleheight=0.9,
    )
    legend.get_title().set_fontproperties(CJK_FONT_PROP)
    for text_obj in legend.get_texts():
        apply_text_font(text_obj)

    title = fig.suptitle(f"\u5c40\u90e8\u9ad8\u5bc6\u5ea6\u533a\u57df\u653e\u5927\u5bf9\u6bd4\uff08{city}{district}\uff0c1:{scale}\uff09", fontsize=12.8, fontweight="bold", y=0.955, fontproperties=CJK_FONT_PROP)
    apply_text_font(title)

    out_dir = FIG_DIR / str(scale)
    base = out_dir / f"zoom_map_{scale}_final_v5"
    fig.savefig(base.with_suffix(".png"), dpi=500, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), dpi=500, bbox_inches="tight")
    svg_path = base.with_suffix(".svg")
    fig.savefig(svg_path, dpi=500, bbox_inches="tight")
    plt.close(fig)
    strip_svg_comments(svg_path)




def export_clean_compare(scale: int, anchor_bounds: Tuple[float, float, float, float]) -> None:
    raw_wgs = normalize_category_labels(gpd.read_file(DATA_DIR / f"raw_{scale}.geojson"))
    gen_wgs = normalize_category_labels(gpd.read_file(DATA_DIR / f"generalized_{scale}.geojson"))
    city = str(raw_wgs["city"].iloc[0])
    district = str(raw_wgs["district"].iloc[0])
    metric_crs = raw_wgs.estimate_utm_crs() or "EPSG:3857"
    raw_metric = raw_wgs.to_crs(metric_crs)
    gen_metric = gen_wgs.to_crs(metric_crs)
    categories = ordered_categories(raw_wgs["category"].dropna().astype(str).unique().tolist())
    symbol_rules = build_symbol_rules(categories)
    engine = PointGeneralizationEngine(symbol_rules)
    raw_metric = engine.analyze_dlm(raw_metric)
    if "grid_id" not in gen_metric.columns:
        gen_metric = gpd.sjoin_nearest(gen_metric, raw_metric[["grid_id", "geometry"]], how="left", distance_col="join_dist").drop(columns=["index_right", "join_dist"])
    bounds, raw_zoom, gen_zoom = build_scale_bounds(anchor_bounds, raw_metric, gen_metric, scale, symbol_rules)
    verify_bounds(bounds, raw_zoom, symbol_rules, scale, raw_mode=True)
    verify_bounds(bounds, gen_zoom, symbol_rules, scale, raw_mode=False)
    minx, miny, maxx, maxy = bounds

    fig = plt.figure(figsize=(12.6, 6.5))
    gs = fig.add_gridspec(1, 2, left=0.075, right=0.965, top=0.87, bottom=0.11, wspace=0.16)
    ax_before = fig.add_subplot(gs[0, 0])
    ax_after = fig.add_subplot(gs[0, 1])
    for ax in (ax_before, ax_after):
        style_ax(ax)
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)
    render_map(ax_before, raw_zoom, symbol_rules, scale, raw_mode=True)
    render_map(ax_after, gen_zoom, symbol_rules, scale, raw_mode=False)
    ax_before.set_title("???", pad=8, fontproperties=CJK_FONT_PROP)
    ax_after.set_title("???", pad=8, fontproperties=CJK_FONT_PROP)
    title = fig.suptitle(f"{city}{district}??POI???????", fontsize=13.2, fontweight="bold", y=0.95, fontproperties=CJK_FONT_PROP)
    apply_text_font(title)

    out_dir = FIG_DIR / str(scale)
    base = out_dir / "compare_before_after_clean"
    fig.savefig(base.with_suffix(".png"), dpi=500, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), dpi=500, bbox_inches="tight")
    svg_path = base.with_suffix(".svg")
    fig.savefig(svg_path, dpi=500, bbox_inches="tight")
    plt.close(fig)
    strip_svg_comments(svg_path)


def export_clean_single(scale: int, anchor_bounds: Tuple[float, float, float, float], before: bool) -> None:
    raw_wgs = normalize_category_labels(gpd.read_file(DATA_DIR / f"raw_{scale}.geojson"))
    gen_wgs = normalize_category_labels(gpd.read_file(DATA_DIR / f"generalized_{scale}.geojson"))
    metric_crs = raw_wgs.estimate_utm_crs() or "EPSG:3857"
    raw_metric = raw_wgs.to_crs(metric_crs)
    gen_metric = gen_wgs.to_crs(metric_crs)
    categories = ordered_categories(raw_wgs["category"].dropna().astype(str).unique().tolist())
    symbol_rules = build_symbol_rules(categories)
    engine = PointGeneralizationEngine(symbol_rules)
    raw_metric = engine.analyze_dlm(raw_metric)
    if "grid_id" not in gen_metric.columns:
        gen_metric = gpd.sjoin_nearest(gen_metric, raw_metric[["grid_id", "geometry"]], how="left", distance_col="join_dist").drop(columns=["index_right", "join_dist"])
    bounds, raw_zoom, gen_zoom = build_scale_bounds(anchor_bounds, raw_metric, gen_metric, scale, symbol_rules)
    target = raw_zoom if before else gen_zoom
    verify_bounds(bounds, target, symbol_rules, scale, raw_mode=before)
    minx, miny, maxx, maxy = bounds

    fig, ax = plt.subplots(figsize=(6.45, 6.5))
    fig.subplots_adjust(left=0.12, right=0.96, top=0.90, bottom=0.11)
    style_ax(ax)
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    render_map(ax, target, symbol_rules, scale, raw_mode=before)
    ax.set_title("???" if before else "???", pad=8, fontproperties=CJK_FONT_PROP)

    out_dir = FIG_DIR / str(scale)
    base = out_dir / ("before_only_clean" if before else "after_only_clean")
    fig.savefig(base.with_suffix(".png"), dpi=500, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), dpi=500, bbox_inches="tight")
    svg_path = base.with_suffix(".svg")
    fig.savefig(svg_path, dpi=500, bbox_inches="tight")
    plt.close(fig)
    strip_svg_comments(svg_path)


def export_clean_suite(scale: int, anchor_bounds: Tuple[float, float, float, float]) -> None:
    export_clean_compare(scale, anchor_bounds)
    export_clean_single(scale, anchor_bounds, before=True)
    export_clean_single(scale, anchor_bounds, before=False)

def rewrite_summary_files() -> None:
    summary_text = r"""# Engine Map Summary

- ee388602153357df1a	e9ae02929daf33a
- 	0092e939f6e05a7b22b POI 51b5860e63e00139f9cbbc6	6c61b2a81f3a001efc40840e53958460e63e02d408f5ce3ab263f7868fbef18316c5593a33a002
- 57036e765e90587ef6E:\GradOpening\MapGenAI\amap_poi_dataset_expanded\all_poi_expanded.geojson
- f7f5286847306fef154ce/5b9848GeoPandas + shapely symbol-footprint renderer
- b9e	645f931fadeff84E:\GradOpening\MapGenAI\output\engine_maps_20260329_175753

## clean exports
- compare_before_after_clean.png / pdf / svg
- before_only_clean.png / pdf / svg
- after_only_clean.png / pdf / svg

## v4-v5 zoom maps
- zoom_map_84000_final_v4.png / pdf / svg
- zoom_map_253000_final_v4.png / pdf / svg
- zoom_map_532000_final_v4.png / pdf / svg
- zoom_map_84000_final_v5.png / pdf / svg
- zoom_map_253000_final_v5.png / pdf / svg
- zoom_map_532000_final_v5.png / pdf / svg
"""
    (SUMMARY_DIR / "engine_map_summary.md").write_text(summary_text, encoding="utf-8-sig")

    payload = {
        "region": "\u4e09\u4e9a\u5e02\u5929\u6daf\u533a",
        "source_file": r"E:\GradOpening\MapGenAI\amap_poi_dataset_expanded\all_poi_expanded.geojson",
        "engine": "GeoPandas + shapely symbol-footprint renderer",
        "legend_categories": CATEGORY_ORDER,
    }
    (SUMMARY_DIR / "scale_symbol_rules.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8-sig")


def main() -> None:
    configure_plotting()
    anchor_raw_wgs = normalize_category_labels(gpd.read_file(DATA_DIR / "raw_84000.geojson"))
    anchor_gen_wgs = normalize_category_labels(gpd.read_file(DATA_DIR / "generalized_84000.geojson"))
    anchor_metric_crs = anchor_raw_wgs.estimate_utm_crs() or "EPSG:3857"
    anchor_raw_metric = anchor_raw_wgs.to_crs(anchor_metric_crs)
    anchor_gen_metric = anchor_gen_wgs.to_crs(anchor_metric_crs)
    anchor_categories = ordered_categories(anchor_raw_wgs["category"].dropna().astype(str).unique().tolist())
    anchor_rules = build_symbol_rules(anchor_categories)
    anchor_engine = PointGeneralizationEngine(anchor_rules)
    anchor_raw_metric = anchor_engine.analyze_dlm(anchor_raw_metric)
    if "grid_id" not in anchor_gen_metric.columns:
        anchor_gen_metric = gpd.sjoin_nearest(anchor_gen_metric, anchor_raw_metric[["grid_id", "geometry"]], how="left", distance_col="join_dist").drop(columns=["index_right", "join_dist"])
    anchor_bounds = choose_zoom_bounds(anchor_raw_metric, anchor_gen_metric, anchor_engine, 84000, anchor_rules)
    for scale in SCALES:
        export_zoom_final_v5(scale, anchor_bounds)
        export_clean_suite(scale, anchor_bounds)
    rewrite_summary_files()
    print(BASE_DIR)


if __name__ == "__main__":
    main()
