from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(r"E:\GradOpening\MapGenAI")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.skills.point_generalization_dlm_dcm import PointGeneralizationEngine, build_default_symbol_rules

OUTPUT_DIR = Path(r"E:\GradOpening\MapGenAI\output\dlm_dcm_exports\dlm_dcm_generalization_20260329_102542")
DATA_PATH = PROJECT_ROOT / "amap_poi_dataset_expanded" / "all_poi_expanded.geojson"
PUB_DIR = OUTPUT_DIR / "figures" / "publication_panels"
PUB_DIR.mkdir(parents=True, exist_ok=True)


CJK_FONT_PROP: FontProperties | None = None

DISPLAY_CATEGORY_ORDER = ["????", "???", "??", "???", "??", "??", "??", "??"]
DISPLAY_CATEGORY_COLORS = {
    "????": "#F3F2C7",
    "??": "#55E0C3",
    "??": "#D8A7BC",
    "??": "#DCCFD6",
    "???": "#C89C92",
    "???": "#2D6DCC",
    "??": "#F4EED9",
    "??": "#C7FF66",
    "??": "#C9CED6",
}
DISPLAY_CATEGORY_ALIASES = {
    "???": "????",
    "???": "????",
}
LEGEND_EDGE_COLOR = "#1F3B63"
LEGEND_TITLE_COLOR = "#4C78B8"


def choose_cjk_font() -> FontProperties:
    candidates = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC", "PingFang SC"]
    available = {font.name: font.fname for font in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            return FontProperties(fname=available[name])
    return FontProperties(family="DejaVu Sans")


def configure_plotting() -> None:
    global CJK_FONT_PROP
    CJK_FONT_PROP = choose_cjk_font()
    chosen_name = CJK_FONT_PROP.get_name()
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": chosen_name,
            "font.sans-serif": [chosen_name],
            "font.serif": [chosen_name],
            "axes.unicode_minus": False,
            "axes.titlesize": 11,
            "axes.labelsize": 8.8,
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 7.8,
            "legend.fontsize": 7.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "path",
        }
    )


def apply_axis_font(ax) -> None:
    if CJK_FONT_PROP is None:
        return
    ax.title.set_fontproperties(CJK_FONT_PROP)
    ax.xaxis.label.set_fontproperties(CJK_FONT_PROP)
    ax.yaxis.label.set_fontproperties(CJK_FONT_PROP)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(CJK_FONT_PROP)


def apply_text_font(text_obj) -> None:
    if CJK_FONT_PROP is not None:
        text_obj.set_fontproperties(CJK_FONT_PROP)


def marker_area(size_mm: float, factor: float = 8.5) -> float:
    return (size_mm * factor) ** 2


def display_category(category: str) -> str:
    text = str(category or "").strip()
    return DISPLAY_CATEGORY_ALIASES.get(text, text or "??")


def display_color(category: str, fallback: str | None = None) -> str:
    label = display_category(category)
    return DISPLAY_CATEGORY_COLORS.get(label, fallback or DISPLAY_CATEGORY_COLORS["??"])


def ordered_display_categories(categories: List[str]) -> List[str]:
    deduped = []
    seen = set()
    for category in categories:
        label = display_category(category)
        if label in seen:
            continue
        seen.add(label)
        deduped.append(label)
    ranked = [label for label in DISPLAY_CATEGORY_ORDER if label in seen]
    tail = [label for label in deduped if label not in ranked]
    return ranked + tail


def prepare_axes(ax) -> None:
    ax.grid(True, color="#d9d9d9", linewidth=0.35, alpha=0.12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.tick_params(length=2.5, width=0.5)
    apply_axis_font(ax)


def add_panel_label(ax, label: str, title: str) -> None:
    label_text = ax.text(0.012, 0.982, label, transform=ax.transAxes, va="top", ha="left", fontsize=10.8, fontweight="bold")
    apply_text_font(label_text)
    ax.set_title(title, loc="center", pad=6, fontproperties=CJK_FONT_PROP)


def plot_colored_points(ax, gdf: gpd.GeoDataFrame, symbol_rules: Dict[str, object], alpha: float = 0.88, edge: str = "#1f2933") -> None:
    for category in sorted(gdf["category"].unique().tolist()):
        subset = gdf[gdf["category"] == category]
        rule = symbol_rules[category]
        ax.scatter(
            subset.geometry.x,
            subset.geometry.y,
            s=marker_area(rule.symbol_size),
            c=display_color(category, rule.color),
            marker=rule.marker,
            alpha=alpha,
            linewidths=0.35,
            edgecolors=edge,
            zorder=3,
        )


def plot_original_faded(ax, gdf: gpd.GeoDataFrame, size: float = 12, alpha: float = 0.20) -> None:
    ax.scatter(
        gdf.geometry.x,
        gdf.geometry.y,
        s=size,
        c="#c7c7c7",
        marker="o",
        alpha=alpha,
        linewidths=0,
        zorder=1,
    )


def build_legend(symbol_rules: Dict[str, object], categories: List[str]) -> List[Line2D]:
    handles = []
    for category in ordered_display_categories(categories):
        handles.append(
            Line2D(
                [0], [0], marker="o", linestyle="", label=category,
                markerfacecolor=DISPLAY_CATEGORY_COLORS.get(category, DISPLAY_CATEGORY_COLORS["??"]),
                markeredgecolor=LEGEND_EDGE_COLOR,
                markeredgewidth=1.4, markersize=14.0
            )
        )
    return handles




def canonical_name(value: str) -> str:
    text = str(value or "").strip()
    for suffix in ["?", "?", "?", "???", "??", "??"]:
        if text.endswith(suffix) and len(text) > len(suffix) + 1:
            text = text[:-len(suffix)]
            break
    return text


def subset_sample(source: gpd.GeoDataFrame, city: str, district: str) -> gpd.GeoDataFrame:
    subset = source[(source["city"] == city) & (source["district"] == district)].copy()
    if len(subset) > 0:
        return subset
    city_key = canonical_name(city)
    district_key = canonical_name(district)
    return source[
        (source["city"].map(canonical_name) == city_key) &
        (source["district"].map(canonical_name) == district_key)
    ].copy()

def compute_conflict_pairs(engine: PointGeneralizationEngine, metric_gdf: gpd.GeoDataFrame, target_scale: int) -> List[Tuple[int, int]]:
    pairs = []
    metric_gdf = metric_gdf.reset_index(drop=True)
    for i in range(len(metric_gdf)):
        for j in range(i + 1, len(metric_gdf)):
            a = metric_gdf.iloc[i]
            b = metric_gdf.iloc[j]
            if a.geometry.distance(b.geometry) < engine._conflict_threshold(a, b, target_scale):
                pairs.append((i, j))
    return pairs


def choose_zoom_window(
    original_metric: gpd.GeoDataFrame,
    selected_metric: gpd.GeoDataFrame,
    engine: PointGeneralizationEngine,
    target_scale: int,
) -> Tuple[Tuple[float, float, float, float], Tuple[float, float, float, float], Dict[str, int]]:
    scores = []
    selected_counts = selected_metric.groupby("grid_id").size().to_dict() if "grid_id" in selected_metric.columns else {}
    for grid_id, group in original_metric.groupby("grid_id"):
        raw_count = len(group)
        raw_cat = group["category"].nunique()
        sel_count = int(selected_counts.get(grid_id, 0))
        reduction = raw_count - sel_count
        conflict_count = len(compute_conflict_pairs(engine, group.reset_index(drop=True), target_scale))
        score = raw_count * 2.2 + raw_cat * 7.5 + reduction * 1.8 + conflict_count * 2.4
        scores.append((score, grid_id, raw_count, sel_count, raw_cat, conflict_count))
    scores.sort(reverse=True)
    best = scores[0]
    grid_id = best[1]
    window_metric = original_metric[original_metric["grid_id"] == grid_id]
    minx, miny, maxx, maxy = window_metric.total_bounds
    pad_x = max((maxx - minx) * 0.35, 120)
    pad_y = max((maxy - miny) * 0.35, 120)
    metric_bounds = (minx - pad_x, miny - pad_y, maxx + pad_x, maxy + pad_y)
    wgs_bounds = window_metric.to_crs(4326).total_bounds
    pad_lon = max((wgs_bounds[2] - wgs_bounds[0]) * 0.25, 0.0035)
    pad_lat = max((wgs_bounds[3] - wgs_bounds[1]) * 0.25, 0.0035)
    geographic_bounds = (wgs_bounds[0] - pad_lon, wgs_bounds[1] - pad_lat, wgs_bounds[2] + pad_lon, wgs_bounds[3] + pad_lat)
    return metric_bounds, geographic_bounds, {"raw": best[2], "selected": best[3], "categories": best[4], "conflicts": best[5]}


def representative_conflict_clusters(
    metric_gdf: gpd.GeoDataFrame,
    pairs: List[Tuple[int, int]],
    limit: int = 5,
) -> List[Tuple[float, float, float, int]]:
    hotspot_counter: Dict[int, int] = {}
    for i, j in pairs:
        hotspot_counter[i] = hotspot_counter.get(i, 0) + 1
        hotspot_counter[j] = hotspot_counter.get(j, 0) + 1
    ranked = sorted(hotspot_counter.items(), key=lambda x: x[1], reverse=True)
    clusters: List[Tuple[float, float, float, int]] = []
    used_points: List[Tuple[float, float]] = []
    for idx, weight in ranked:
        row = metric_gdf.iloc[idx]
        x = float(row.geometry.x)
        y = float(row.geometry.y)
        if any(((x - ux) ** 2 + (y - uy) ** 2) ** 0.5 < 260 for ux, uy in used_points):
            continue
        used_points.append((x, y))
        radius = 120 + min(weight, 12) * 24
        clusters.append((x, y, radius, weight))
        if len(clusters) >= limit:
            break
    return clusters


def annotate_bar_values(ax, bars, values: List[float]) -> None:
    for bar, value in zip(bars, values):
        t = ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
        apply_text_font(t)


def render_case(sample_row: pd.Series, scale: int, source: gpd.GeoDataFrame, symbol_rules: Dict[str, object], engine: PointGeneralizationEngine) -> bool:
    sample_id = sample_row["sample_id"]
    city = sample_row["city"]
    district = sample_row["district"]
    selected_path = next((OUTPUT_DIR / "geojson").glob(f"{sample_id}_{scale}.geojson"))
    selected_wgs = gpd.read_file(selected_path)
    original_wgs = subset_sample(source, city, district)
    if len(original_wgs) == 0:
        print(f"skip {sample_id}: source sample not found in current dataset")
        return False

    metric_crs = original_wgs.estimate_utm_crs() or "EPSG:3857"
    original_metric = engine.analyze_dlm(original_wgs.to_crs(metric_crs))
    selected_metric = selected_wgs.to_crs(metric_crs)
    if "grid_id" not in selected_metric.columns:
        selected_metric = gpd.sjoin(
            selected_metric,
            original_metric[["grid_id", "geometry"]].rename_geometry("geometry"),
            predicate="within",
            how="left",
        ).drop(columns=[c for c in ["index_right"] if c in selected_metric.columns])

    categories = sorted(original_wgs["category"].unique().tolist())
    fig, axes = plt.subplots(2, 3, figsize=(15.0, 9.5))
    fig.subplots_adjust(left=0.055, right=0.985, top=0.90, bottom=0.205, wspace=0.18, hspace=0.24)

    for ax in axes.ravel():
        prepare_axes(ax)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

    ax_a, ax_b, ax_c, ax_d, ax_e, ax_f = axes.ravel()

    plot_colored_points(ax_a, original_wgs, symbol_rules)
    add_panel_label(ax_a, "(a)", "原始点分布")

    plot_colored_points(ax_b, selected_wgs, symbol_rules)
    add_panel_label(ax_b, "(b)", "综合后点分布")

    plot_original_faded(ax_c, original_wgs, size=9, alpha=0.18)
    plot_colored_points(ax_c, selected_wgs, symbol_rules, alpha=0.95)
    add_panel_label(ax_c, "(c)", "前后叠加对比")
    t = ax_c.text(0.99, 0.02, "灰色: 原始点  彩色: 综合后", transform=ax_c.transAxes, ha="right", va="bottom", fontsize=7.4, color="#666666")
    apply_text_font(t)

    metric_bounds, geographic_bounds, zoom_meta = choose_zoom_window(original_metric, selected_metric, engine, scale)
    gminx, gminy, gmaxx, gmaxy = geographic_bounds
    zoom_rect = Rectangle((gminx, gminy), gmaxx - gminx, gmaxy - gminy, fill=False, linestyle="--", linewidth=1.0, edgecolor="#4d4d4d", zorder=4)
    ax_c.add_patch(zoom_rect)
    t = ax_c.text(0.02, 0.04, "密集冲突窗口", transform=ax_c.transAxes, fontsize=7.4, color="#4d4d4d")
    apply_text_font(t)

    minx, miny, maxx, maxy = metric_bounds
    original_zoom = original_metric.cx[minx:maxx, miny:maxy].to_crs(4326)
    selected_zoom = selected_metric.cx[minx:maxx, miny:maxy].to_crs(4326)
    plot_original_faded(ax_d, original_zoom, size=10, alpha=0.20)
    plot_colored_points(ax_d, selected_zoom, symbol_rules, alpha=0.98)
    if len(original_zoom) > 0:
        ax_d.set_xlim(gminx, gmaxx)
        ax_d.set_ylim(gminy, gmaxy)
    add_panel_label(ax_d, "(d)", "局部区域放大对比")
    t = ax_d.text(0.98, 0.03, f"raw={zoom_meta['raw']}  gen={zoom_meta['selected']}  cat={zoom_meta['categories']}  conflict={zoom_meta['conflicts']}", transform=ax_d.transAxes, ha="right", va="bottom", fontsize=7.2, bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#cccccc", linewidth=0.4))
    apply_text_font(t)

    plot_original_faded(ax_e, original_wgs)
    plot_colored_points(ax_e, selected_wgs, symbol_rules, alpha=0.94)
    conflict_pairs = compute_conflict_pairs(engine, original_metric.reset_index(drop=True), scale)
    hotspot_counter: Dict[int, int] = {}
    for i, j in conflict_pairs:
        hotspot_counter[i] = hotspot_counter.get(i, 0) + 1
        hotspot_counter[j] = hotspot_counter.get(j, 0) + 1
    top_hotspots = sorted(hotspot_counter.items(), key=lambda x: x[1], reverse=True)[:10]
    if top_hotspots:
        hotspot_idx = [idx for idx, _ in top_hotspots]
        hotspot_wgs = original_metric.reset_index(drop=True).iloc[hotspot_idx].to_crs(4326)
        sizes = [140 + hotspot_counter[idx] * 42 for idx in hotspot_idx]
        ax_e.scatter(hotspot_wgs.geometry.x, hotspot_wgs.geometry.y, s=sizes, facecolors="none", edgecolors="#d62828", linewidths=1.3, zorder=5)
        for (idx, weight), row in zip(top_hotspots, hotspot_wgs.itertuples()):
            ann = ax_e.annotate(f"{weight}", (row.geometry.x, row.geometry.y), xytext=(3, 3), textcoords="offset points", fontsize=7, color="#8d1f1f")
            apply_text_font(ann)
    add_panel_label(ax_e, "(e)", "符号冲突消解示意")
    t = ax_e.text(0.98, 0.02, f"代表性冲突对: {len(conflict_pairs)}", transform=ax_e.transAxes, ha="right", va="bottom", fontsize=7.8, color="#8d1f1f")
    apply_text_font(t)

    summary = (
        original_wgs.groupby("category").size().rename("before").to_frame()
        .join(selected_wgs.groupby("category").size().rename("after"), how="left")
        .fillna(0)
        .reset_index()
    )
    summary["display_category"] = summary["category"].map(display_category)
    summary = summary.groupby("display_category", as_index=False)[["before", "after"]].sum()
    summary["sort_key"] = summary["display_category"].map(
        lambda value: DISPLAY_CATEGORY_ORDER.index(value) if value in DISPLAY_CATEGORY_ORDER else len(DISPLAY_CATEGORY_ORDER)
    )
    summary = summary.sort_values(["sort_key", "before"], ascending=[True, False]).drop(columns=["sort_key"])
    summary["ratio"] = summary["after"] / summary["before"]
    colors = [DISPLAY_CATEGORY_COLORS.get(c, DISPLAY_CATEGORY_COLORS["??"]) for c in summary["display_category"]]
    bars = ax_f.bar(summary["display_category"], summary["ratio"], color=colors, edgecolor="#374151", linewidth=0.5, zorder=3)
    annotate_bar_values(ax_f, bars, summary["ratio"].tolist())
    ax_f.set_ylim(0, 1.15)
    ax_f.set_ylabel("Retention ratio")
    ax_f.set_xlabel("POI category")
    ax_f.tick_params(axis="x", rotation=35)
    add_panel_label(ax_f, "(f)", "类别保留率统计")

    legend_categories = ordered_display_categories(categories)
    legend = fig.legend(
        handles=build_legend(symbol_rules, categories),
        labels=legend_categories,
        loc="lower center",
        ncol=2,
        frameon=True,
        bbox_to_anchor=(0.5, 0.037),
        handletextpad=0.6,
        columnspacing=2.2,
        handlelength=1.0,
        borderpad=1.05,
        labelspacing=1.25,
        fancybox=False,
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor(LEGEND_EDGE_COLOR)
    legend.get_frame().set_linewidth(1.4)
    legend.set_title("Poi??", prop=CJK_FONT_PROP)
    legend.get_title().set_color(LEGEND_TITLE_COLOR)
    legend.get_title().set_fontsize(16)
    for text in legend.get_texts():
        text.set_fontsize(12.5)
        apply_text_font(text)

    title_city = city.replace("?", "") if city.endswith("?") else city
    supt = fig.suptitle(f"顾及符号表达的POI点综合结果（{title_city}，1:{scale}）", fontsize=13, fontweight="bold", y=0.965)
    apply_text_font(supt)

    base = PUB_DIR / f"{sample_id}_{scale}_publication_panel"
    fig.savefig(base.with_suffix('.png'), dpi=360, bbox_inches='tight')
    fig.savefig(base.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(base.with_suffix('.svg'), bbox_inches='tight')
    plt.close(fig)
    return True


def main() -> None:
    configure_plotting()
    source = gpd.read_file(DATA_PATH)
    source = source.rename(columns={"query_category": "category"})
    source = source[["poi_uid", "name", "type", "category", "city", "district", "geometry"]].copy()
    source["category"] = source["category"].fillna("其他").astype(str)
    symbol_rules = build_default_symbol_rules(sorted(source["category"].unique().tolist()))
    engine = PointGeneralizationEngine(symbol_rules)
    metrics = pd.read_csv(OUTPUT_DIR / "metrics_summary.csv")
    rendered = 0
    skipped = 0
    for row in metrics[["sample_id", "city", "district", "target_scale"]].itertuples(index=False):
        ok = render_case(pd.Series({"sample_id": row.sample_id, "city": row.city, "district": row.district}), int(row.target_scale), source, symbol_rules, engine)
        if ok:
            rendered += 1
        else:
            skipped += 1
    (PUB_DIR / "render_log.txt").write_text(f"rendered={rendered}\nskipped={skipped}\n", encoding="utf-8")
    print(PUB_DIR)
    print(f"font={CJK_FONT_PROP.get_name() if CJK_FONT_PROP else 'unknown'}")


if __name__ == "__main__":
    main()
