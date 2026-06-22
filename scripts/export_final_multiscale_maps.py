from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import pandas as pd

PROJECT_ROOT = Path(r"E:\GradOpening\MapGenAI")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.skills.point_generalization_dlm_dcm import PointGeneralizationEngine, SymbolRule

DATA_PATH = PROJECT_ROOT / "amap_poi_dataset_expanded" / "all_poi_expanded.geojson"
LEGACY_METRICS = PROJECT_ROOT / "output" / "dlm_dcm_exports" / "dlm_dcm_generalization_20260329_102542" / "metrics_summary.csv"
OUTPUT_ROOT = Path(r"E:\GradOpening\outputs")
FALLBACK_OUTPUT_ROOT = PROJECT_ROOT / "output"
SCALES = [84000, 253000, 532000]

CJK_FONT_PROP: FontProperties | None = None

CATEGORY_STYLE = {
    "便利店": {"color": "#d9485f", "marker": "o", "size": 0.88, "min_distance": 0.58, "priority": 0.86, "preserve_floor": 1},
    "公交站": {"color": "#3f72af", "marker": "o", "size": 0.95, "min_distance": 0.62, "priority": 0.92, "preserve_floor": 1},
    "学校": {"color": "#264653", "marker": "s", "size": 1.15, "min_distance": 0.72, "priority": 1.15, "preserve_floor": 2},
    "停车场": {"color": "#8d99ae", "marker": "s", "size": 0.92, "min_distance": 0.65, "priority": 0.88, "preserve_floor": 1},
    "酒店": {"color": "#6d597a", "marker": "h", "size": 1.00, "min_distance": 0.70, "priority": 0.95, "preserve_floor": 1},
    "银行": {"color": "#2a9d8f", "marker": "P", "size": 1.00, "min_distance": 0.70, "priority": 1.00, "preserve_floor": 1},
    "医院": {"color": "#d62828", "marker": "^", "size": 1.20, "min_distance": 0.75, "priority": 1.22, "preserve_floor": 2},
    "餐厅": {"color": "#f4a261", "marker": "o", "size": 0.96, "min_distance": 0.60, "priority": 0.90, "preserve_floor": 1},
    "咖啡厅": {"color": "#b56576", "marker": "o", "size": 0.90, "min_distance": 0.58, "priority": 0.89, "preserve_floor": 1},
    "餐饮": {"color": "#f77f00", "marker": "o", "size": 0.94, "min_distance": 0.60, "priority": 0.88, "preserve_floor": 1},
    "地铁站": {"color": "#1d3557", "marker": "D", "size": 1.25, "min_distance": 0.80, "priority": 1.30, "preserve_floor": 2},
    "其他": {"color": "#7f8c8d", "marker": "o", "size": 0.92, "min_distance": 0.60, "priority": 0.82, "preserve_floor": 1},
}


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
    chosen = CJK_FONT_PROP.get_name()
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": chosen,
            "font.sans-serif": [chosen],
            "font.serif": [chosen],
            "axes.unicode_minus": False,
            "axes.titlesize": 12,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "path",
        }
    )


def apply_text_font(text_obj) -> None:
    if CJK_FONT_PROP is not None:
        text_obj.set_fontproperties(CJK_FONT_PROP)


def apply_axis_font(ax) -> None:
    if CJK_FONT_PROP is None:
        return
    ax.title.set_fontproperties(CJK_FONT_PROP)
    ax.xaxis.label.set_fontproperties(CJK_FONT_PROP)
    ax.yaxis.label.set_fontproperties(CJK_FONT_PROP)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(CJK_FONT_PROP)


def make_output_dir() -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    target_name = f"final_maps_{ts}"
    for root in [OUTPUT_ROOT, FALLBACK_OUTPUT_ROOT]:
        try:
            root.mkdir(parents=True, exist_ok=True)
            out = root / target_name
            out.mkdir(parents=True, exist_ok=False)
            for sub in ["figures/84000", "figures/253000", "figures/532000", "data", "summary", "logs"]:
                (out / sub).mkdir(parents=True, exist_ok=True)
            return out
        except Exception:
            continue
    raise RuntimeError("Unable to create output directory")


def build_logger(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger("final_multiscale_maps")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(output_dir / "logs" / "map_export_log.txt", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))
    logger.addHandler(sh)
    return logger


def build_symbol_rules(categories: List[str]) -> Dict[str, SymbolRule]:
    rules: Dict[str, SymbolRule] = {}
    for cat in categories:
        style = CATEGORY_STYLE.get(cat, CATEGORY_STYLE["其他"])
        rules[cat] = SymbolRule(cat, style["size"], style["min_distance"], style["priority"], style["preserve_floor"], style["color"], style["marker"])
    if "其他" not in rules:
        style = CATEGORY_STYLE["其他"]
        rules["其他"] = SymbolRule("其他", style["size"], style["min_distance"], style["priority"], style["preserve_floor"], style["color"], style["marker"])
    return rules


def load_source() -> gpd.GeoDataFrame:
    gdf = gpd.read_file(DATA_PATH)
    gdf = gdf.rename(columns={"query_category": "category"})
    gdf = gdf[["poi_uid", "name", "type", "category", "city", "district", "geometry"]].copy()
    gdf["category"] = gdf["category"].fillna("其他").astype(str)
    return gdf


def choose_representative_region(source: gpd.GeoDataFrame, logger: logging.Logger) -> Tuple[str, str, str]:
    metrics = pd.read_csv(LEGACY_METRICS)
    candidates = []
    for sample_id, group in metrics.groupby("sample_id"):
        city = group["city"].iloc[0]
        district = group["district"].iloc[0]
        subset = source[(source["city"] == city) & (source["district"] == district)]
        if len(subset) == 0:
            continue
        score = float(group["symbol_conflicts_before"].max()) + float(group["raw_category_count"].max()) * 100 + float(group["raw_count"].max())
        candidates.append((score, city, district, sample_id))
    if candidates:
        candidates.sort(reverse=True)
        score, city, district, sample_id = candidates[0]
        logger.info("Selected representative region from prior experiment: %s (%s, score %.2f)", sample_id, DATA_PATH.name, score)
        return city, district, sample_id

    grouped = source.groupby(["city", "district"]).agg(raw_count=("poi_uid", "size"), category_count=("category", "nunique")).reset_index()
    grouped = grouped[(grouped["raw_count"] >= 160) & (grouped["category_count"] >= 6) & (grouped["raw_count"] <= 400)]
    top = grouped.sort_values(["category_count", "raw_count"], ascending=False).iloc[0]
    sample_id = f"{top['city']}_{top['district']}"
    logger.info("Fallback representative region selected: %s", sample_id)
    return top["city"], top["district"], sample_id


def marker_area(size_mm: float, factor: float = 8.2) -> float:
    return (size_mm * factor) ** 2


def style_axes(ax, with_labels: bool = True) -> None:
    ax.set_facecolor("white")
    ax.grid(True, color="#d9d9d9", linewidth=0.3, alpha=0.10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)
    if with_labels:
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
    apply_axis_font(ax)


def legend_handles(symbol_rules: Dict[str, SymbolRule], categories: List[str]) -> List[Line2D]:
    handles = []
    for category in categories:
        rule = symbol_rules[category]
        handle = Line2D([0], [0], marker=rule.marker, linestyle="", label=category, markerfacecolor=rule.color, markeredgecolor="#1f2933", markeredgewidth=0.45, markersize=max(5, rule.symbol_size * 4.0))
        handles.append(handle)
    return handles


def add_shared_legend(fig, symbol_rules: Dict[str, SymbolRule], categories: List[str]) -> None:
    legend = fig.legend(handles=legend_handles(symbol_rules, categories), labels=categories, loc="lower center", bbox_to_anchor=(0.5, 0.02), frameon=False, ncol=min(5, max(3, len(categories))), columnspacing=0.9, handletextpad=0.3)
    legend.set_title("POI类别", prop=CJK_FONT_PROP)
    for txt in legend.get_texts():
        apply_text_font(txt)
        txt.set_fontsize(7.6)


def plot_points(ax, gdf: gpd.GeoDataFrame, symbol_rules: Dict[str, SymbolRule], alpha: float = 0.90, faded: bool = False) -> None:
    for cat in sorted(gdf["category"].unique().tolist()):
        subset = gdf[gdf["category"] == cat]
        rule = symbol_rules.get(cat, symbol_rules["其他"])
        if faded:
            ax.scatter(subset.geometry.x, subset.geometry.y, s=12, c="#c7c7c7", alpha=0.25, marker="o", linewidths=0, zorder=1)
        else:
            ax.scatter(subset.geometry.x, subset.geometry.y, s=marker_area(rule.symbol_size), c=rule.color, marker=rule.marker, alpha=alpha, linewidths=0.35, edgecolors="#1f2933", zorder=3)


def add_map_title(fig, title: str, subtitle: str | None = None) -> None:
    supt = fig.suptitle(title, fontsize=13, fontweight="bold", y=0.965)
    apply_text_font(supt)
    if subtitle:
        txt = fig.text(0.5, 0.935, subtitle, ha="center", va="center", fontsize=9, color="#555555")
        apply_text_font(txt)


def export_figure(fig, base_path: Path) -> None:
    fig.savefig(base_path.with_suffix(".png"), dpi=360, bbox_inches="tight")
    fig.savefig(base_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


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


def choose_zoom_window(original_metric: gpd.GeoDataFrame, selected_metric: gpd.GeoDataFrame, engine: PointGeneralizationEngine, scale: int) -> Tuple[Tuple[float, float, float, float], Tuple[float, float, float, float]]:
    selected_counts = selected_metric.groupby("grid_id").size().to_dict() if "grid_id" in selected_metric.columns else {}
    scores = []
    for grid_id, group in original_metric.groupby("grid_id"):
        raw_count = len(group)
        cat_count = group["category"].nunique()
        sel_count = int(selected_counts.get(grid_id, 0))
        conflict_count = len(compute_conflict_pairs(engine, group.reset_index(drop=True), scale))
        score = raw_count * 2.0 + cat_count * 7.0 + conflict_count * 2.5 + (raw_count - sel_count) * 1.8
        scores.append((score, grid_id))
    scores.sort(reverse=True)
    grid_id = scores[0][1]
    window_metric = original_metric[original_metric["grid_id"] == grid_id]
    minx, miny, maxx, maxy = window_metric.total_bounds
    pad_x = max((maxx - minx) * 0.38, 120)
    pad_y = max((maxy - miny) * 0.38, 120)
    metric_bounds = (minx - pad_x, miny - pad_y, maxx + pad_x, maxy + pad_y)
    wgs_bounds = window_metric.to_crs(4326).total_bounds
    pad_lon = max((wgs_bounds[2] - wgs_bounds[0]) * 0.28, 0.0035)
    pad_lat = max((wgs_bounds[3] - wgs_bounds[1]) * 0.28, 0.0035)
    geo_bounds = (wgs_bounds[0] - pad_lon, wgs_bounds[1] - pad_lat, wgs_bounds[2] + pad_lon, wgs_bounds[3] + pad_lat)
    return metric_bounds, geo_bounds


def export_raw_map(sample_wgs: gpd.GeoDataFrame, categories: List[str], symbol_rules: Dict[str, SymbolRule], scale: int, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 7.2))
    style_axes(ax)
    plot_points(ax, sample_wgs, symbol_rules)
    add_map_title(fig, f"原始点分布图", f"目标比例尺：1:{scale}")
    add_shared_legend(fig, symbol_rules, categories)
    export_figure(fig, out_dir / f"map_raw_{scale}")


def export_generalized_map(selected_wgs: gpd.GeoDataFrame, categories: List[str], symbol_rules: Dict[str, SymbolRule], scale: int, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 7.2))
    style_axes(ax)
    plot_points(ax, selected_wgs, symbol_rules)
    add_map_title(fig, f"综合后点分布图", f"目标比例尺：1:{scale}")
    add_shared_legend(fig, symbol_rules, categories)
    export_figure(fig, out_dir / f"map_generalized_{scale}")


def export_comparison_map(sample_wgs: gpd.GeoDataFrame, selected_wgs: gpd.GeoDataFrame, categories: List[str], symbol_rules: Dict[str, SymbolRule], scale: int, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 6.6))
    for ax in axes:
        style_axes(ax)
    plot_points(axes[0], sample_wgs, symbol_rules)
    t0 = axes[0].set_title("原始点分布", pad=8, fontproperties=CJK_FONT_PROP)
    plot_points(axes[1], selected_wgs, symbol_rules)
    t1 = axes[1].set_title("综合后点分布", pad=8, fontproperties=CJK_FONT_PROP)
    add_map_title(fig, f"原始/综合后对比图", f"目标比例尺：1:{scale}")
    add_shared_legend(fig, symbol_rules, categories)
    export_figure(fig, out_dir / f"map_comparison_{scale}")


def export_zoom_comparison(original_metric: gpd.GeoDataFrame, selected_metric: gpd.GeoDataFrame, categories: List[str], symbol_rules: Dict[str, SymbolRule], engine: PointGeneralizationEngine, scale: int, out_dir: Path) -> None:
    metric_bounds, geo_bounds = choose_zoom_window(original_metric, selected_metric, engine, scale)
    minx, miny, maxx, maxy = metric_bounds
    gminx, gminy, gmaxx, gmaxy = geo_bounds
    raw_zoom = original_metric.cx[minx:maxx, miny:maxy].to_crs(4326)
    gen_zoom = selected_metric.cx[minx:maxx, miny:maxy].to_crs(4326)
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 6.6))
    for ax in axes:
        style_axes(ax)
        ax.set_xlim(gminx, gmaxx)
        ax.set_ylim(gminy, gmaxy)
    plot_points(axes[0], raw_zoom, symbol_rules)
    axes[0].set_title("局部原始图", pad=8, fontproperties=CJK_FONT_PROP)
    plot_points(axes[1], gen_zoom, symbol_rules)
    axes[1].set_title("局部综合后图", pad=8, fontproperties=CJK_FONT_PROP)
    add_map_title(fig, "局部放大对比图", f"目标比例尺：1:{scale}")
    add_shared_legend(fig, symbol_rules, categories)
    export_figure(fig, out_dir / f"map_zoom_comparison_{scale}")


def main() -> None:
    configure_plotting()
    output_dir = make_output_dir()
    logger = build_logger(output_dir)
    logger.info("Output directory: %s", output_dir)
    logger.info("Using font: %s", CJK_FONT_PROP.get_name() if CJK_FONT_PROP else "unknown")

    source = load_source()
    city, district, sample_id = choose_representative_region(source, logger)
    sample_wgs = source[(source["city"] == city) & (source["district"] == district)].copy()
    metric_crs = sample_wgs.estimate_utm_crs() or "EPSG:3857"
    categories = sorted(sample_wgs["category"].unique().tolist())
    symbol_rules = build_symbol_rules(categories)
    engine = PointGeneralizationEngine(symbol_rules)
    sample_metric = engine.analyze_dlm(sample_wgs.to_crs(metric_crs))

    scale_settings = {
        "region": sample_id,
        "source_file": str(DATA_PATH),
        "font": CJK_FONT_PROP.get_name() if CJK_FONT_PROP else "unknown",
        "scales": {},
    }
    generated_files: List[str] = []

    for scale in SCALES:
        scale_dir = output_dir / "figures" / str(scale)
        logger.info("Processing scale 1:%s", scale)
        selected_metric = engine.optimize_dcm(sample_metric, scale)
        selected_wgs = selected_metric.to_crs(4326)
        selected_wgs.to_file(output_dir / "data" / f"generalized_{scale}.geojson", driver="GeoJSON")
        sample_wgs.to_file(output_dir / "data" / f"raw_{scale}.geojson", driver="GeoJSON")

        export_raw_map(sample_wgs, categories, symbol_rules, scale, scale_dir)
        export_generalized_map(selected_wgs, categories, symbol_rules, scale, scale_dir)
        export_comparison_map(sample_wgs, selected_wgs, categories, symbol_rules, scale, scale_dir)
        export_zoom_comparison(sample_metric, selected_metric, categories, symbol_rules, engine, scale, scale_dir)

        for stem in [f"map_raw_{scale}", f"map_generalized_{scale}", f"map_comparison_{scale}", f"map_zoom_comparison_{scale}"]:
            for ext in ["png", "pdf", "svg"]:
                generated_files.append(str(scale_dir / f"{stem}.{ext}"))
                logger.info("Exported %s", scale_dir / f"{stem}.{ext}")

        scale_settings["scales"][str(scale)] = {
            "symbol_rules": {cat: {
                "symbol_size": symbol_rules[cat].symbol_size,
                "min_distance": symbol_rules[cat].min_distance,
                "priority": symbol_rules[cat].priority,
                "preserve_floor": symbol_rules[cat].preserve_floor,
                "color": symbol_rules[cat].color,
                "marker": symbol_rules[cat].marker,
            } for cat in categories},
            "raw_count": int(len(sample_metric)),
            "generalized_count": int(len(selected_metric)),
            "retention_rate": round(len(selected_metric) / max(len(sample_metric), 1), 4),
            "category_count": int(sample_metric["category"].nunique()),
            "conflicts_before": int(engine.count_symbol_conflicts(sample_metric, scale)),
            "conflicts_after": int(engine.count_symbol_conflicts(selected_metric, scale)),
        }

    (output_dir / "summary" / "scale_settings.json").write_text(json.dumps(scale_settings, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_lines = [
        "# Final Map Summary",
        "",
        f"- 代表性区域：{sample_id}",
        f"- 数据来源文件：{DATA_PATH}",
        f"- 目标比例尺：{', '.join(f'1:{s}' for s in SCALES)}",
        f"- 真实输出路径：{output_dir}",
        "",
        "## 比例尺规则概述",
    ]
    for scale in SCALES:
        info = scale_settings["scales"][str(scale)]
        summary_lines.append(f"- 1:{scale}：保留率 {info['retention_rate']}，冲突 {info['conflicts_before']} -> {info['conflicts_after']}，类别数 {info['category_count']}")
    summary_lines.extend(["", "## 输出图件清单"])
    summary_lines.extend([f"- {Path(item).name}" for item in generated_files])
    (output_dir / "summary" / "final_map_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")

    print(output_dir)


if __name__ == "__main__":
    main()
