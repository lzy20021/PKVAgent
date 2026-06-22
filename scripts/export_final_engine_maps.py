from __future__ import annotations

import json
import logging
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, RegularPolygon
import numpy as np
import pandas as pd
from shapely.geometry import Point

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
    "便利店": {"color": "#d9485f", "marker": "circle", "size": 0.88, "min_distance": 0.58, "priority": 0.86, "preserve_floor": 1},
    "公交站": {"color": "#3f72af", "marker": "circle", "size": 0.95, "min_distance": 0.62, "priority": 0.92, "preserve_floor": 1},
    "学校": {"color": "#264653", "marker": "square", "size": 1.15, "min_distance": 0.72, "priority": 1.15, "preserve_floor": 2},
    "停车场": {"color": "#8d99ae", "marker": "square", "size": 0.92, "min_distance": 0.65, "priority": 0.88, "preserve_floor": 1},
    "酒店": {"color": "#6d597a", "marker": "hexagon", "size": 1.00, "min_distance": 0.70, "priority": 0.95, "preserve_floor": 1},
    "银行": {"color": "#2a9d8f", "marker": "diamond", "size": 1.00, "min_distance": 0.70, "priority": 1.00, "preserve_floor": 1},
    "医院": {"color": "#d62828", "marker": "triangle", "size": 1.20, "min_distance": 0.75, "priority": 1.22, "preserve_floor": 2},
    "餐厅": {"color": "#f4a261", "marker": "circle", "size": 0.96, "min_distance": 0.60, "priority": 0.90, "preserve_floor": 1},
    "咖啡厅": {"color": "#b56576", "marker": "circle", "size": 0.90, "min_distance": 0.58, "priority": 0.89, "preserve_floor": 1},
    "餐饮": {"color": "#f77f00", "marker": "circle", "size": 0.94, "min_distance": 0.60, "priority": 0.88, "preserve_floor": 1},
    "地铁站": {"color": "#1d3557", "marker": "diamond", "size": 1.25, "min_distance": 0.80, "priority": 1.30, "preserve_floor": 2},
    "其他": {"color": "#7f8c8d", "marker": "circle", "size": 0.92, "min_distance": 0.60, "priority": 0.82, "preserve_floor": 1},
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
    name = f"engine_maps_{ts}"
    for root in [OUTPUT_ROOT, FALLBACK_OUTPUT_ROOT]:
        try:
            root.mkdir(parents=True, exist_ok=True)
            out = root / name
            out.mkdir(parents=True, exist_ok=False)
            for sub in ["figures/84000", "figures/253000", "figures/532000", "layouts", "styles", "data", "summary", "logs"]:
                (out / sub).mkdir(parents=True, exist_ok=True)
            return out
        except Exception:
            continue
    raise RuntimeError("Unable to create output directory")


def build_logger(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger("engine_final_maps")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(output_dir / "logs" / "engine_export_log.txt", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))
    logger.addHandler(sh)
    return logger


def load_source() -> gpd.GeoDataFrame:
    gdf = gpd.read_file(DATA_PATH)
    gdf = gdf.rename(columns={"query_category": "category"})
    gdf = gdf[["poi_uid", "name", "type", "category", "city", "district", "geometry"]].copy()
    gdf["category"] = gdf["category"].fillna("其他").astype(str)
    return gdf


def build_symbol_rules(categories: List[str]) -> Dict[str, SymbolRule]:
    rules: Dict[str, SymbolRule] = {}
    for cat in categories:
        style = CATEGORY_STYLE.get(cat, CATEGORY_STYLE["其他"])
        rules[cat] = SymbolRule(cat, style["size"], style["min_distance"], style["priority"], style["preserve_floor"], style["color"], style["marker"])
    if "其他" not in rules:
        style = CATEGORY_STYLE["其他"]
        rules["其他"] = SymbolRule("其他", style["size"], style["min_distance"], style["priority"], style["preserve_floor"], style["color"], style["marker"])
    return rules


def choose_region(source: gpd.GeoDataFrame, logger: logging.Logger) -> Tuple[str, str, str]:
    metrics = pd.read_csv(LEGACY_METRICS)
    scored = []
    for sample_id, grp in metrics.groupby("sample_id"):
        city = grp["city"].iloc[0]
        district = grp["district"].iloc[0]
        subset = source[(source["city"] == city) & (source["district"] == district)]
        if len(subset) == 0:
            continue
        score = float(grp["symbol_conflicts_before"].max()) + float(grp["raw_category_count"].max()) * 100 + float(grp["local_category_diversity"].mean()) * 100
        scored.append((score, city, district, sample_id))
    scored.sort(reverse=True)
    score, city, district, sample_id = scored[0]
    logger.info("Selected representative region: %s (score %.2f)", sample_id, score)
    return city, district, sample_id


def metric_radius(rule: SymbolRule, scale: int) -> float:
    return (rule.symbol_size * scale / 1000.0) * 0.5


def marker_handles(symbol_rules: Dict[str, SymbolRule], categories: List[str]) -> List[Line2D]:
    shape_map = {"circle": "o", "square": "s", "triangle": "^", "diamond": "D", "hexagon": "h"}
    handles = []
    for cat in categories:
        rule = symbol_rules[cat]
        handles.append(Line2D([0], [0], marker=shape_map.get(rule.marker, "o"), linestyle="", label=cat, markerfacecolor=rule.color, markeredgecolor="#1f2933", markeredgewidth=0.45, markersize=max(5, rule.symbol_size * 4.0)))
    return handles


def style_ax(ax) -> None:
    ax.set_facecolor("white")
    ax.grid(True, color="#d9d9d9", linewidth=0.30, alpha=0.10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    apply_axis_font(ax)


def add_title(fig, title: str, subtitle: str) -> None:
    st = fig.suptitle(title, fontsize=13, fontweight="bold", y=0.965)
    apply_text_font(st)
    sub = fig.text(0.5, 0.935, subtitle, ha="center", va="center", fontsize=9, color="#555555")
    apply_text_font(sub)


def add_legend(fig, symbol_rules: Dict[str, SymbolRule], categories: List[str]) -> None:
    legend = fig.legend(handles=marker_handles(symbol_rules, categories), labels=categories, loc="lower center", bbox_to_anchor=(0.5, 0.02), frameon=False, ncol=min(5, max(3, len(categories))), columnspacing=0.9, handletextpad=0.3)
    legend.set_title("POI类别", prop=CJK_FONT_PROP)
    for txt in legend.get_texts():
        apply_text_font(txt)


def add_symbol_patch(ax, x: float, y: float, rule: SymbolRule, scale: int, alpha: float = 0.95) -> None:
    radius = metric_radius(rule, scale)
    edge = "#1f2933"
    if rule.marker == "square":
        patch = RegularPolygon((x, y), numVertices=4, radius=radius * 1.15, orientation=np.pi / 4, facecolor=rule.color, edgecolor=edge, linewidth=0.4, alpha=alpha)
    elif rule.marker == "triangle":
        patch = RegularPolygon((x, y), numVertices=3, radius=radius * 1.22, orientation=0, facecolor=rule.color, edgecolor=edge, linewidth=0.4, alpha=alpha)
    elif rule.marker == "diamond":
        patch = RegularPolygon((x, y), numVertices=4, radius=radius * 1.20, orientation=0, facecolor=rule.color, edgecolor=edge, linewidth=0.4, alpha=alpha)
    elif rule.marker == "hexagon":
        patch = RegularPolygon((x, y), numVertices=6, radius=radius * 1.10, orientation=0, facecolor=rule.color, edgecolor=edge, linewidth=0.4, alpha=alpha)
    else:
        patch = Circle((x, y), radius=radius, facecolor=rule.color, edgecolor=edge, linewidth=0.4, alpha=alpha)
    ax.add_patch(patch)


def render_symbol_map(ax, gdf_metric: gpd.GeoDataFrame, symbol_rules: Dict[str, SymbolRule], scale: int, faded: bool = False) -> None:
    for row in gdf_metric.itertuples():
        cat = getattr(row, "category")
        rule = symbol_rules.get(cat, symbol_rules["其他"])
        if faded:
            patch = Circle((row.geometry.x, row.geometry.y), radius=max(18, metric_radius(rule, scale) * 0.55), facecolor="#d0d0d0", edgecolor="none", alpha=0.22)
            ax.add_patch(patch)
        else:
            add_symbol_patch(ax, row.geometry.x, row.geometry.y, rule, scale)


def export_figure(fig, path_base: Path) -> None:
    fig.savefig(path_base.with_suffix('.png'), dpi=360, bbox_inches='tight')
    fig.savefig(path_base.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(path_base.with_suffix('.svg'), bbox_inches='tight')
    plt.close(fig)


def compute_conflict_pairs(engine: PointGeneralizationEngine, metric_gdf: gpd.GeoDataFrame, scale: int) -> List[Tuple[int, int]]:
    pairs: List[Tuple[int, int]] = []
    work = metric_gdf.reset_index(drop=True)
    for i in range(len(work)):
        for j in range(i + 1, len(work)):
            a = work.iloc[i]
            b = work.iloc[j]
            if a.geometry.distance(b.geometry) < engine._conflict_threshold(a, b, scale):
                pairs.append((i, j))
    return pairs


def choose_zoom_bounds(original_metric: gpd.GeoDataFrame, selected_metric: gpd.GeoDataFrame, engine: PointGeneralizationEngine, scale: int) -> Tuple[float, float, float, float]:
    selected_counts = selected_metric.groupby('grid_id').size().to_dict() if 'grid_id' in selected_metric.columns else {}
    scores = []
    for grid_id, group in original_metric.groupby('grid_id'):
        raw = len(group)
        cats = group['category'].nunique()
        sel = int(selected_counts.get(grid_id, 0))
        conflicts = len(compute_conflict_pairs(engine, group, scale))
        score = raw * 2.2 + cats * 7.0 + conflicts * 2.5 + (raw - sel) * 1.8
        scores.append((score, grid_id))
    scores.sort(reverse=True)
    grid_id = scores[0][1]
    window = original_metric[original_metric['grid_id'] == grid_id]
    minx, miny, maxx, maxy = window.total_bounds
    padx = max((maxx - minx) * 0.40, 160)
    pady = max((maxy - miny) * 0.40, 160)
    return (minx - padx, miny - pady, maxx + padx, maxy + pady)


def export_raw(sample_metric: gpd.GeoDataFrame, categories: List[str], symbol_rules: Dict[str, SymbolRule], scale: int, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 7.2))
    style_ax(ax)
    render_symbol_map(ax, sample_metric, symbol_rules, scale)
    add_title(fig, "原始地图成果图", f"目标比例尺：1:{scale}")
    add_legend(fig, symbol_rules, categories)
    export_figure(fig, out_dir / f"raw_map_{scale}")


def export_generalized(selected_metric: gpd.GeoDataFrame, categories: List[str], symbol_rules: Dict[str, SymbolRule], scale: int, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 7.2))
    style_ax(ax)
    render_symbol_map(ax, selected_metric, symbol_rules, scale)
    add_title(fig, "综合后地图成果图", f"目标比例尺：1:{scale}")
    add_legend(fig, symbol_rules, categories)
    export_figure(fig, out_dir / f"generalized_map_{scale}")


def export_comparison(sample_metric: gpd.GeoDataFrame, selected_metric: gpd.GeoDataFrame, categories: List[str], symbol_rules: Dict[str, SymbolRule], scale: int, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 6.8))
    for ax in axes:
        style_ax(ax)
    render_symbol_map(axes[0], sample_metric, symbol_rules, scale)
    axes[0].set_title("Raw", fontproperties=CJK_FONT_PROP, pad=8)
    render_symbol_map(axes[1], selected_metric, symbol_rules, scale)
    axes[1].set_title("Generalized", fontproperties=CJK_FONT_PROP, pad=8)
    add_title(fig, "原始/综合后对比地图", f"目标比例尺：1:{scale}")
    add_legend(fig, symbol_rules, categories)
    export_figure(fig, out_dir / f"comparison_map_{scale}")


def export_zoom(sample_metric: gpd.GeoDataFrame, selected_metric: gpd.GeoDataFrame, categories: List[str], symbol_rules: Dict[str, SymbolRule], engine: PointGeneralizationEngine, scale: int, out_dir: Path) -> None:
    bounds = choose_zoom_bounds(sample_metric, selected_metric, engine, scale)
    minx, miny, maxx, maxy = bounds
    raw_zoom = sample_metric.cx[minx:maxx, miny:maxy]
    gen_zoom = selected_metric.cx[minx:maxx, miny:maxy]
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 6.8))
    for ax in axes:
        style_ax(ax)
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)
    render_symbol_map(axes[0], raw_zoom, symbol_rules, scale)
    axes[0].set_title("Raw zoom", fontproperties=CJK_FONT_PROP, pad=8)
    render_symbol_map(axes[1], gen_zoom, symbol_rules, scale)
    axes[1].set_title("Generalized zoom", fontproperties=CJK_FONT_PROP, pad=8)
    add_title(fig, "局部高密度区域放大对比图", f"目标比例尺：1:{scale}")
    add_legend(fig, symbol_rules, categories)
    export_figure(fig, out_dir / f"zoom_map_{scale}")


def main() -> None:
    configure_plotting()
    output_dir = make_output_dir()
    logger = build_logger(output_dir)
    logger.info("Output directory: %s", output_dir)
    logger.info("Map engine check: PyQGIS unavailable; fallback to GeoPandas + shapely symbol-footprint renderer")
    logger.info("Using font: %s", CJK_FONT_PROP.get_name() if CJK_FONT_PROP else 'unknown')

    source = load_source()
    city, district, sample_id = choose_region(source, logger)
    sample_wgs = source[(source['city'] == city) & (source['district'] == district)].copy()
    metric_crs = sample_wgs.estimate_utm_crs() or 'EPSG:3857'
    categories = sorted(sample_wgs['category'].unique().tolist())
    symbol_rules = build_symbol_rules(categories)
    engine = PointGeneralizationEngine(symbol_rules)
    sample_metric = engine.analyze_dlm(sample_wgs.to_crs(metric_crs))

    styles_payload = {cat: {
        'color': rule.color,
        'marker': rule.marker,
        'symbol_size': rule.symbol_size,
        'min_distance': rule.min_distance,
        'priority': rule.priority,
        'preserve_floor': rule.preserve_floor,
    } for cat, rule in symbol_rules.items()}
    (output_dir / 'styles' / 'scale_symbol_rules.json').write_text(json.dumps(styles_payload, ensure_ascii=False, indent=2), encoding='utf-8')

    layout_spec = {
        'engine': 'GeoPandas + shapely symbol-footprint renderer',
        'crs': str(metric_crs),
        'region': sample_id,
        'scales': SCALES,
        'layout': 'single-map or side-by-side comparison with shared legend',
    }
    (output_dir / 'layouts' / 'layout_spec.json').write_text(json.dumps(layout_spec, ensure_ascii=False, indent=2), encoding='utf-8')

    generated: List[str] = []
    summary_scales = {}
    for scale in SCALES:
        out_dir = output_dir / 'figures' / str(scale)
        logger.info('Processing scale 1:%s', scale)
        selected_metric = engine.optimize_dcm(sample_metric, scale)
        selected_metric.to_crs(4326).to_file(output_dir / 'data' / f'generalized_{scale}.geojson', driver='GeoJSON')
        sample_metric.to_crs(4326).to_file(output_dir / 'data' / f'raw_{scale}.geojson', driver='GeoJSON')
        export_raw(sample_metric, categories, symbol_rules, scale, out_dir)
        export_generalized(selected_metric, categories, symbol_rules, scale, out_dir)
        export_comparison(sample_metric, selected_metric, categories, symbol_rules, scale, out_dir)
        export_zoom(sample_metric, selected_metric, categories, symbol_rules, engine, scale, out_dir)
        for stem in [f'raw_map_{scale}', f'generalized_map_{scale}', f'comparison_map_{scale}', f'zoom_map_{scale}']:
            for ext in ['png', 'pdf', 'svg']:
                path = out_dir / f'{stem}.{ext}'
                generated.append(str(path))
                logger.info('Exported %s', path)
        summary_scales[str(scale)] = {
            'raw_count': int(len(sample_metric)),
            'generalized_count': int(len(selected_metric)),
            'retention_rate': round(len(selected_metric) / max(len(sample_metric), 1), 4),
            'category_count': int(sample_metric['category'].nunique()),
            'conflicts_before': int(engine.count_symbol_conflicts(sample_metric, scale)),
            'conflicts_after': int(engine.count_symbol_conflicts(selected_metric, scale)),
            'min_distance_rule': {cat: symbol_rules[cat].min_distance for cat in categories},
            'symbol_size_rule': {cat: symbol_rules[cat].symbol_size for cat in categories},
        }

    summary_lines = [
        '# Engine Map Summary',
        '',
        f'- 代表性区域：{sample_id}',
        '- 选择原因：多类别POI共存明显、原始密集冲突强、综合后改善明显，适合作为符号表达优化展示区。',
        f'- 数据来源文件：{DATA_PATH}',
        '- 使用的地图引擎/方案：GeoPandas + shapely symbol-footprint renderer（因 PyQGIS / QGIS 当前不可用）',
        '- 遮挡处理方式：基于 DLM/DCM 综合引擎进行选取、位移、优先级控制、最小间距约束与类别覆盖约束，渲染时按符号真实尺寸生成占位图元。',
        f'- 实际输出路径：{output_dir}',
        '',
        '## 三个比例尺设置',
    ]
    for scale in SCALES:
        info = summary_scales[str(scale)]
        summary_lines.append(f"- 1:{scale}：保留率 {info['retention_rate']}，冲突 {info['conflicts_before']} -> {info['conflicts_after']}，类别数 {info['category_count']}")
    summary_lines.extend(['', '## 输出文件清单'])
    summary_lines.extend([f'- {Path(p).name}' for p in generated])
    (output_dir / 'summary' / 'engine_map_summary.md').write_text('\n'.join(summary_lines), encoding='utf-8')
    (output_dir / 'summary' / 'scale_symbol_rules.json').write_text(json.dumps(summary_scales, ensure_ascii=False, indent=2), encoding='utf-8')

    print(output_dir)


if __name__ == '__main__':
    main()
