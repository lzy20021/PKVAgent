from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.offsetbox import AnnotationBbox, DrawingArea
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle


PROJECT_ROOT = Path(r"E:\GradOpening\MapGenAI")
PAIR_DIR = PROJECT_ROOT / "output" / "classic_symbol_figures" / "agent_intermediate"
DATA_PATH = PROJECT_ROOT / "amap_poi_dataset_expanded" / "all_poi_expanded.geojson"
OUTPUT_DIR = PROJECT_ROOT / "output" / "standalone_before_after_symbol_compare_v2"

CATEGORY_ORDER = ["便利店", "学校", "停车场", "酒店", "银行", "医院", "餐饮", "交通设施"]
COLORS = {
    "便利店": "#D84B8A",
    "学校": "#2E7D32",
    "停车场": "#2F80ED",
    "酒店": "#7B4FA3",
    "银行": "#00897B",
    "医院": "#D32F2F",
    "餐饮": "#F28E2B",
    "交通设施": "#4E79A7",
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


def normalize_category(value: str) -> str | None:
    text = str(value or "").strip()
    mapping = {
        "餐厅": "餐饮",
        "咖啡厅": "餐饮",
        "公交站": "交通设施",
        "地铁站": "交通设施",
    }
    return mapping.get(text, text if text in CATEGORY_ORDER else None)


def normalize_source(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    out = gdf.copy()
    if "category" not in out.columns:
        out["category"] = out["query_category"] if "query_category" in out.columns else ""
    out["category"] = out["category"].map(normalize_category)
    out = out.dropna(subset=["category"]).copy()
    out = out[out.geometry.notna() & ~out.geometry.is_empty].copy()
    if out.crs is None:
        out = out.set_crs("EPSG:4326")
    elif str(out.crs).upper() != "EPSG:4326":
        out = out.to_crs("EPSG:4326")
    return out


def load_pair(before_path: Path, after_path: Path) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    before = gpd.read_file(before_path)
    after = gpd.read_file(after_path)
    for gdf in [before, after]:
        if "category" not in gdf.columns:
            if "query_category" in gdf.columns:
                gdf["category"] = gdf["query_category"]
            else:
                gdf["category"] = ""
        gdf["category"] = gdf["category"].map(normalize_category)
        gdf.dropna(subset=["category"], inplace=True)
        if gdf.crs is None:
            gdf.set_crs("EPSG:4326", inplace=True)
        elif str(gdf.crs).upper() != "EPSG:4326":
            gdf.to_crs("EPSG:4326", inplace=True)
    return before, after


def icon_box(category: str, size: int, edge: str = "#1F2933") -> DrawingArea:
    color = COLORS[category]
    da = DrawingArea(size, size, 0, 0)
    bg = FancyBboxPatch(
        (0.5, 0.5),
        size - 1,
        size - 1,
        boxstyle=f"round,pad=0,rounding_size={max(2, size * 0.16)}",
        facecolor=color,
        edgecolor=edge,
        linewidth=max(0.55, size * 0.045),
    )
    da.add_artist(bg)
    w = "white"
    lw = max(0.8, size * 0.075)
    s = float(size)

    if category == "便利店":
        da.add_artist(Rectangle((s * 0.25, s * 0.39), s * 0.50, s * 0.34, fill=False, edgecolor=w, linewidth=lw))
        da.add_artist(Polygon([[s * 0.22, s * 0.40], [s * 0.50, s * 0.22], [s * 0.78, s * 0.40]], closed=False, edgecolor=w, linewidth=lw, fill=False))
        da.add_artist(Rectangle((s * 0.37, s * 0.52), s * 0.12, s * 0.21, facecolor=w, edgecolor=w, linewidth=0))
        da.add_artist(Rectangle((s * 0.55, s * 0.52), s * 0.12, s * 0.21, facecolor=w, edgecolor=w, linewidth=0))
    elif category == "学校":
        da.add_artist(Polygon([[s * 0.15, s * 0.43], [s * 0.50, s * 0.25], [s * 0.85, s * 0.43], [s * 0.50, s * 0.60]], closed=True, facecolor=w, edgecolor=w, linewidth=0))
        da.add_artist(Rectangle((s * 0.32, s * 0.55), s * 0.36, s * 0.09, facecolor=w, edgecolor=w, linewidth=0))
        da.add_artist(Rectangle((s * 0.76, s * 0.43), s * 0.045, s * 0.28, facecolor=w, edgecolor=w, linewidth=0))
        da.add_artist(Circle((s * 0.785, s * 0.74), s * 0.035, facecolor=w, edgecolor=w, linewidth=0))
    elif category == "停车场":
        da.add_artist(plt.Text(s * 0.50, s * 0.51, "P", color=w, ha="center", va="center", fontsize=s * 0.82, fontweight="bold"))
    elif category == "酒店":
        da.add_artist(Rectangle((s * 0.18, s * 0.51), s * 0.64, s * 0.20, facecolor=w, edgecolor=w, linewidth=0))
        da.add_artist(Rectangle((s * 0.18, s * 0.36), s * 0.12, s * 0.35, facecolor=w, edgecolor=w, linewidth=0))
        da.add_artist(Rectangle((s * 0.31, s * 0.39), s * 0.22, s * 0.11, facecolor=w, edgecolor=w, linewidth=0))
        da.add_artist(Rectangle((s * 0.18, s * 0.72), s * 0.64, s * 0.06, facecolor=w, edgecolor=w, linewidth=0))
    elif category == "银行":
        da.add_artist(Polygon([[s * 0.16, s * 0.39], [s * 0.50, s * 0.18], [s * 0.84, s * 0.39]], closed=True, facecolor=w, edgecolor=w, linewidth=0))
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
        da.add_artist(Polygon([[s * 0.58, s * 0.25], [s * 0.73, s * 0.25], [s * 0.68, s * 0.48], [s * 0.62, s * 0.48]], closed=True, facecolor=w, edgecolor=w, linewidth=0))
    elif category == "交通设施":
        da.add_artist(Rectangle((s * 0.22, s * 0.31), s * 0.56, s * 0.36, facecolor=w, edgecolor=w, linewidth=0))
        da.add_artist(Rectangle((s * 0.28, s * 0.37), s * 0.18, s * 0.13, facecolor=color, edgecolor=color, linewidth=0))
        da.add_artist(Rectangle((s * 0.54, s * 0.37), s * 0.18, s * 0.13, facecolor=color, edgecolor=color, linewidth=0))
        da.add_artist(Circle((s * 0.33, s * 0.70), s * 0.055, facecolor=w, edgecolor=w, linewidth=0))
        da.add_artist(Circle((s * 0.67, s * 0.70), s * 0.055, facecolor=w, edgecolor=w, linewidth=0))
    return da


def add_symbol(ax, x: float, y: float, category: str, size: int, edge: str) -> None:
    ax.add_artist(
        AnnotationBbox(
            icon_box(category, size=size, edge=edge),
            (x, y),
            xycoords="data",
            frameon=False,
            box_alignment=(0.5, 0.5),
            pad=0,
            annotation_clip=True,
            zorder=8,
        )
    )


def display_filter(ax, gdf: gpd.GeoDataFrame, size: int, gap: float) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf.copy()
    work = gdf.copy()
    work["_order"] = work["category"].map({c: i for i, c in enumerate(CATEGORY_ORDER)}).fillna(99)
    work["_x"] = work.geometry.x
    work["_y"] = work.geometry.y
    center_x = float((work["_x"].min() + work["_x"].max()) / 2)
    center_y = float((work["_y"].min() + work["_y"].max()) / 2)
    work["_dist"] = (work["_x"] - center_x) ** 2 + (work["_y"] - center_y) ** 2
    work = work.sort_values(["_order", "_dist"]).copy()
    xy = ax.transData.transform(np.column_stack([work.geometry.x.to_numpy(), work.geometry.y.to_numpy()]))
    half = float(size) * gap * 0.5
    keep: List[int] = []
    kept_boxes: List[Tuple[float, float, float, float]] = []
    for idx, (px, py) in zip(work.index.tolist(), xy):
        box = (float(px) - half, float(py) - half, float(px) + half, float(py) + half)
        overlaps = any(
            not (box[2] <= old[0] or box[0] >= old[2] or box[3] <= old[1] or box[1] >= old[3])
            for old in kept_boxes
        )
        if overlaps:
            continue
        keep.append(idx)
        kept_boxes.append(box)
    return gdf.loc[keep].copy()


def generalize_for_comparison(gdf: gpd.GeoDataFrame, scale: int, seed: int) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf.copy()
    ratio = 0.54 if scale <= 120000 else (0.42 if scale <= 300000 else 0.34)
    target = max(24, min(len(gdf), int(round(len(gdf) * ratio))))
    rng = np.random.default_rng(seed)
    protected = []
    rest = []
    for category in CATEGORY_ORDER:
        subset = gdf[gdf["category"] == category].copy()
        if subset.empty:
            continue
        take = min(2, len(subset))
        protected.append(subset.sample(take, random_state=seed))
        rest.append(subset.drop(index=protected[-1].index))
    protected_gdf = gpd.GeoDataFrame(pd.concat(protected), geometry="geometry", crs=gdf.crs) if protected else gdf.iloc[0:0].copy()
    rest_gdf = gpd.GeoDataFrame(pd.concat(rest), geometry="geometry", crs=gdf.crs) if rest else gdf.iloc[0:0].copy()
    remaining = max(target - len(protected_gdf), 0)
    if remaining <= 0 or rest_gdf.empty:
        return protected_gdf.head(target).copy()

    b = gdf.total_bounds
    width = max(float(b[2] - b[0]), 1e-6)
    height = max(float(b[3] - b[1]), 1e-6)
    nx = max(5, int(np.sqrt(remaining) * 1.5))
    ny = max(5, int(np.sqrt(remaining) * 1.2))
    work = rest_gdf.copy()
    work["_gx"] = np.floor((work.geometry.x - b[0]) / width * nx).clip(0, nx - 1).astype(int)
    work["_gy"] = np.floor((work.geometry.y - b[1]) / height * ny).clip(0, ny - 1).astype(int)
    work["_rand"] = rng.random(len(work))
    picked = work.sort_values(["category", "_gx", "_gy", "_rand"]).groupby(["category", "_gx", "_gy"], as_index=False).head(1)
    if len(picked) > remaining:
        picked = picked.sample(remaining, random_state=seed)
    picked = picked.drop(columns=["_gx", "_gy", "_rand"])
    return gpd.GeoDataFrame(pd.concat([protected_gdf, picked]), geometry="geometry", crs=gdf.crs)


def set_extent(axes: Iterable[plt.Axes], before: gpd.GeoDataFrame, after: gpd.GeoDataFrame) -> Tuple[float, float, float, float]:
    b0 = before.total_bounds
    b1 = after.total_bounds
    minx = min(float(b0[0]), float(b1[0]))
    miny = min(float(b0[1]), float(b1[1]))
    maxx = max(float(b0[2]), float(b1[2]))
    maxy = max(float(b0[3]), float(b1[3]))
    dx = max(maxx - minx, 0.002)
    dy = max(maxy - miny, 0.002)
    pad = 0.12
    ex = (minx - dx * pad, miny - dy * pad, maxx + dx * pad, maxy + dy * pad)
    for ax in axes:
        ax.set_xlim(ex[0], ex[2])
        ax.set_ylim(ex[1], ex[3])
        ax.set_aspect("equal", adjustable="box")
    return ex


def draw_context_basemap(ax, extent: Tuple[float, float, float, float], style: str, seed: int) -> None:
    minx, miny, maxx, maxy = extent
    rng = np.random.default_rng(seed)
    dark = style.startswith("dark")
    ax.set_facecolor("#111820" if dark else "#F8FAFC")
    road_major = "#536273" if dark else "#D8DEE7"
    road_minor = "#394655" if dark else "#E8EDF3"
    building = "#1E2A34" if dark else "#EEF2F6"
    building_edge = "#2E3B47" if dark else "#DDE4EC"
    water = "#0D2A3A" if dark else "#E8F3FA"

    width = maxx - minx
    height = maxy - miny
    ax.add_patch(Rectangle((minx, miny), width, height, facecolor=ax.get_facecolor(), edgecolor="none", zorder=0))
    ax.add_patch(Rectangle((minx + width * 0.03, miny + height * 0.05), width * 0.28, height * 0.16, facecolor=water, edgecolor="none", alpha=0.55, zorder=1))

    for i, frac in enumerate(np.linspace(0.12, 0.88, 6)):
        y = miny + height * frac
        xs = np.linspace(minx - width * 0.05, maxx + width * 0.05, 80)
        wav = np.sin(np.linspace(0, np.pi * 2, 80) + i * 0.8) * height * 0.012
        ax.plot(xs, y + wav, color=road_minor, linewidth=0.65, alpha=0.80, zorder=2)
    for i, frac in enumerate(np.linspace(0.14, 0.86, 5)):
        x = minx + width * frac
        ys = np.linspace(miny - height * 0.05, maxy + height * 0.05, 80)
        wav = np.cos(np.linspace(0, np.pi * 2, 80) + i * 0.5) * width * 0.010
        ax.plot(x + wav, ys, color=road_minor, linewidth=0.65, alpha=0.80, zorder=2)
    for i, frac in enumerate([0.32, 0.58]):
        y = miny + height * frac
        xs = np.linspace(minx - width * 0.05, maxx + width * 0.05, 80)
        wav = np.sin(np.linspace(0, np.pi * 1.5, 80) + i) * height * 0.025
        ax.plot(xs, y + wav, color=road_major, linewidth=1.7, alpha=0.95, zorder=3)

    for _ in range(44):
        bx = minx + width * rng.uniform(0.06, 0.92)
        by = miny + height * rng.uniform(0.08, 0.90)
        bw = width * rng.uniform(0.010, 0.030)
        bh = height * rng.uniform(0.008, 0.026)
        ax.add_patch(Rectangle((bx, by), bw, bh, facecolor=building, edgecolor=building_edge, linewidth=0.25, alpha=0.82, zorder=2.5))


def draw_panel(ax, gdf: gpd.GeoDataFrame, title: str, theme: str, size: int, seed: int, is_after: bool) -> int:
    dark = theme.startswith("dark")
    gap = 2.35 if is_after else 2.10
    if dark:
        gap += 0.15
    kept = display_filter(ax, gdf, size=size, gap=gap)
    edge = "#0B1117" if dark else "#1F2933"
    if not dark:
        ax.set_facecolor("white")
    hidden = gdf.drop(index=kept.index, errors="ignore")
    for category in CATEGORY_ORDER:
        subset = hidden[hidden["category"] == category]
        if subset.empty:
            continue
        ax.scatter(subset.geometry.x, subset.geometry.y, s=5 if not dark else 4, c=COLORS[category], alpha=0.26 if not dark else 0.32, linewidths=0, zorder=5)
    for category in CATEGORY_ORDER:
        subset = kept[kept["category"] == category]
        for row in subset.itertuples():
            add_symbol(ax, float(row.geometry.x), float(row.geometry.y), category, size=size, edge=edge)
    title_color = "#EAF1F7" if dark else "#17202A"
    ax.set_title(title, fontsize=17, fontweight="bold", color=title_color, pad=14)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return len(kept)


def draw_legend(fig, dark: bool) -> None:
    ax = fig.add_axes([0.18, 0.035, 0.64, 0.06])
    ax.set_axis_off()
    xs = np.linspace(0.035, 0.955, len(CATEGORY_ORDER))
    text_color = "#DCE7F0" if dark else "#2B3440"
    ax.text(-0.02, 0.50, "POI 类型", transform=ax.transAxes, ha="right", va="center", fontsize=11, color=text_color)
    for x, category in zip(xs, CATEGORY_ORDER):
        ax.add_artist(
            AnnotationBbox(
                icon_box(category, size=16, edge="#101820" if dark else "#1F2933"),
                (x, 0.52),
                xycoords="axes fraction",
                frameon=False,
                box_alignment=(0.5, 0.5),
                pad=0,
                zorder=10,
            )
        )
        ax.text(x + 0.018, 0.52, category, transform=ax.transAxes, ha="left", va="center", fontsize=9.5, color=text_color)


def render_case(before: gpd.GeoDataFrame, after: gpd.GeoDataFrame, out_dir: Path, theme: str, case_index: int, place: str, scale: int) -> Path:
    stem = f"{place}_1_{scale}".replace("/", "_").replace("\\", "_")
    title_place = place.replace("_", " ")
    dark = theme.startswith("dark")
    fig_face = "#0D1117" if dark else "white"
    fig, axes = plt.subplots(1, 2, figsize=(15.8, 8.6), facecolor=fig_face)
    fig.subplots_adjust(left=0.035, right=0.965, top=0.86, bottom=0.13, wspace=0.045)
    extent = set_extent(axes, before, after)

    if theme == "dark_context":
        draw_context_basemap(axes[0], extent, theme, case_index * 17 + 1)
        draw_context_basemap(axes[1], extent, theme, case_index * 17 + 1)
    elif theme == "dark_plain":
        for ax in axes:
            ax.set_facecolor("#111820")
            ax.add_patch(Rectangle((extent[0], extent[1]), extent[2] - extent[0], extent[3] - extent[1], facecolor="#111820", edgecolor="none", zorder=0))
    else:
        for ax in axes:
            ax.set_facecolor("white")

    symbol_size = 9 if theme == "clean" else 8
    kept_before = draw_panel(axes[0], before, f"综合前  n={len(before)}", theme, symbol_size, case_index, is_after=False)
    kept_after = draw_panel(axes[1], after, f"综合后  n={len(after)}", theme, symbol_size, case_index, is_after=True)

    arrow_color = "#D8E2EC" if dark else "#6B7280"
    fig.text(0.50, 0.49, "→", ha="center", va="center", fontsize=30, color=arrow_color, fontweight="bold")
    retention = len(after) / max(len(before), 1)
    title_color = "#EFF6FF" if dark else "#111827"
    sub_color = "#A8B3C1" if dark else "#5B6470"
    fig.suptitle(f"{title_place}  1:{scale}  POI 点综合前后对比", fontsize=20, fontweight="bold", y=0.955, color=title_color)
    fig.text(0.5, 0.895, f"保留率 {retention:.2f}   显示符号: {kept_before} → {kept_after}   已进行图面符号冲突过滤", ha="center", va="center", fontsize=11.5, color=sub_color)
    draw_legend(fig, dark=dark)

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{stem}_{theme}_symbol_compare.png"
    fig.savefig(out, dpi=420, bbox_inches="tight", facecolor=fig_face)
    plt.close(fig)
    return out


def score_pair(before_path: Path, after_path: Path) -> Tuple[int, int, int]:
    before, after = load_pair(before_path, after_path)
    return (before["category"].nunique(), len(before), len(before) - len(after))


def choose_pairs(limit: int = 5) -> List[Tuple[Path, Path]]:
    pairs = []
    for before in PAIR_DIR.glob("*_before.geojson"):
        after = before.with_name(before.name.replace("_before.geojson", "_after.geojson"))
        if after.exists():
            pairs.append((before, after))
    pairs.sort(key=lambda p: score_pair(*p), reverse=True)
    return pairs[:limit]


def choose_source_cases(limit: int = 6) -> List[Tuple[str, int, gpd.GeoDataFrame, gpd.GeoDataFrame]]:
    source = normalize_source(gpd.read_file(DATA_PATH))
    stats = source.groupby(["city", "district"]).agg(count=("category", "size"), cat=("category", "nunique")).reset_index()
    candidates = stats[(stats["count"].between(85, 175)) & (stats["cat"] >= 7)].copy()
    if candidates.empty:
        candidates = stats[(stats["count"].between(60, 220)) & (stats["cat"] >= 6)].copy()
    candidates = candidates.sort_values(["cat", "count"], ascending=[False, False]).head(limit)
    scales = [84000, 120000, 253000, 325000, 532000, 253000]
    cases: List[Tuple[str, int, gpd.GeoDataFrame, gpd.GeoDataFrame]] = []
    for i, row in enumerate(candidates.itertuples()):
        before = source[(source["city"] == row.city) & (source["district"] == row.district)].copy()
        before = before.sample(min(len(before), 150), random_state=100 + i) if len(before) > 150 else before
        scale = scales[i % len(scales)]
        after = generalize_for_comparison(before, scale=scale, seed=500 + i)
        place = f"{row.city}_{row.district}"
        cases.append((place, scale, before, after))
    return cases


def main() -> None:
    configure_plot()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs: List[str] = []
    themes = ["clean", "dark_plain", "dark_context"]
    for idx, (place, scale, before, after) in enumerate(choose_source_cases(limit=6), start=1):
        for theme in themes:
            outputs.append(str(render_case(before, after, OUTPUT_DIR / theme, theme, idx, place, scale)))
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps({"outputs": outputs}, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "README.md").write_text(
        "\n".join(
            [
                "# Standalone Before/After POI Symbol Comparisons",
                "",
                "每个案例均为独立的综合前/综合后对比图，不显示坐标轴。",
                "",
                "- clean: 白底表达，最接近原来的标准前后对比，只是把圆点换成 POI 语义符号。",
                "- dark_plain: 暗色底板，突出彩色 POI 符号。",
                "- dark_context: 暗色制图参照底板，加入道路/建筑视觉上下文。该底板为本地生成的制图参照层，不依赖在线瓦片。",
                "",
                "所有图均进行最终图面像素级符号冲突过滤，避免符号边界互相遮挡。",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
