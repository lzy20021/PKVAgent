from __future__ import annotations

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
DATA_PATH = PROJECT_ROOT / "amap_poi_dataset_expanded" / "all_poi_expanded.geojson"
OUTPUT_DIR = PROJECT_ROOT / "output" / "classic_symbol_figures"

CATEGORY_ORDER = ["便利店", "学校", "停车场", "酒店", "银行", "医院", "餐饮", "交通设施"]
CATEGORY_LABEL_EN = {
    "便利店": "Convenience",
    "学校": "School",
    "停车场": "Parking",
    "酒店": "Hotel",
    "银行": "Bank",
    "医院": "Hospital",
    "餐饮": "Restaurant",
    "交通设施": "Transit",
}

# Colors are intentionally close to common web-map POI conventions:
# red = medical, blue = transit/parking, orange = food, teal = banking,
# purple = lodging, green = school, magenta = store.
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

QUERY_CATEGORY_MAP = {
    "便利店": "便利店",
    "学校": "学校",
    "停车场": "停车场",
    "酒店": "酒店",
    "银行": "银行",
    "医院": "医院",
    "餐饮": "餐饮",
    "餐厅": "餐饮",
    "咖啡厅": "餐饮",
    "公交站": "交通设施",
    "地铁站": "交通设施",
    "交通设施": "交通设施",
}


def choose_cjk_font() -> str:
    candidates = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC", "PingFang SC"]
    available = {font.name for font in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            return name
    return "DejaVu Sans"


def configure_plot() -> None:
    font_name = choose_cjk_font()
    plt.rcParams.update(
        {
            "font.family": font_name,
            "font.sans-serif": [font_name],
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def normalize_categories(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    out = gdf.copy()
    if "category" not in out.columns:
        if "query_category" not in out.columns:
            raise KeyError("Input data must contain either 'category' or 'query_category'.")
        out["category"] = out["query_category"]
    out["category"] = out["category"].astype(str).str.strip().map(QUERY_CATEGORY_MAP)
    out = out[out["category"].isin(CATEGORY_ORDER)].copy()
    out = out[out.geometry.notna() & ~out.geometry.is_empty].copy()
    if out.crs is None:
        out = out.set_crs("EPSG:4326")
    elif str(out.crs).upper() != "EPSG:4326":
        out = out.to_crs("EPSG:4326")
    return out


def icon_box(category: str, size: int = 22) -> DrawingArea:
    color = COLORS[category]
    da = DrawingArea(size, size, 0, 0)
    bg = FancyBboxPatch(
        (0.5, 0.5),
        size - 1,
        size - 1,
        boxstyle=f"round,pad=0,rounding_size={max(3, size * 0.18)}",
        facecolor=color,
        edgecolor="#1F2933",
        linewidth=max(0.7, size * 0.045),
    )
    da.add_artist(bg)
    w = "white"
    lw = max(1.05, size * 0.075)
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


def add_symbol(ax, x: float, y: float, category: str, size: int, xycoords: str = "data", zorder: int = 5) -> None:
    ab = AnnotationBbox(
        icon_box(category, size=size),
        (x, y),
        xycoords=xycoords,
        frameon=False,
        box_alignment=(0.5, 0.5),
        pad=0.0,
        annotation_clip=xycoords == "data",
        zorder=zorder,
    )
    ax.add_artist(ab)


def collision_free_points(ax, gdf: gpd.GeoDataFrame, size: int, gap_factor: float = 1.28) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf.copy()
    order_map = {category: rank for rank, category in enumerate(CATEGORY_ORDER)}
    work = gdf.copy()
    work["_symbol_order"] = work["category"].map(order_map).fillna(999).astype(int)
    work = work.sort_values(["_symbol_order", "category"]).copy()
    xy = ax.transData.transform(np.column_stack([work.geometry.x.to_numpy(), work.geometry.y.to_numpy()]))
    min_gap = float(size) * gap_factor
    keep_positions: List[Tuple[float, float]] = []
    keep_indices: List[int] = []
    for idx, (px, py) in zip(work.index.tolist(), xy):
        overlaps = any(abs(px - qx) < min_gap and abs(py - qy) < min_gap for qx, qy in keep_positions)
        if overlaps:
            continue
        keep_positions.append((float(px), float(py)))
        keep_indices.append(idx)
    return gdf.loc[keep_indices].copy()


def draw_symbol_points(
    ax,
    gdf: gpd.GeoDataFrame,
    size: int = 12,
    alpha_underlay: float = 0.12,
    avoid_overlap: bool = True,
    gap_factor: float = 1.28,
) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf.copy()
    draw_gdf = collision_free_points(ax, gdf, size=size, gap_factor=gap_factor) if avoid_overlap else gdf
    ax.scatter(gdf.geometry.x, gdf.geometry.y, s=max(2.5, size * 0.35), c="#808080", alpha=alpha_underlay, linewidths=0, zorder=2)
    for category in CATEGORY_ORDER:
        subset = draw_gdf[draw_gdf["category"] == category]
        for row in subset.itertuples():
            add_symbol(ax, float(row.geometry.x), float(row.geometry.y), category, size=size)
    return draw_gdf


def style_axis(ax, show_ticks: bool = True) -> None:
    ax.grid(True, color="#d8dde3", linestyle=":", linewidth=0.45, alpha=0.55, zorder=1)
    for spine in ax.spines.values():
        spine.set_linewidth(0.75)
        spine.set_edgecolor("#4b5563")
    if not show_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
    ax.tick_params(labelsize=8, length=2.5, width=0.5, colors="#4b5563")


def set_common_extent(axes: Iterable[plt.Axes], gdfs: Iterable[gpd.GeoDataFrame], pad_ratio: float = 0.11) -> None:
    bounds = [g.total_bounds for g in gdfs if not g.empty]
    if not bounds:
        return
    arr = np.asarray(bounds, dtype=float)
    minx, miny = arr[:, 0].min(), arr[:, 1].min()
    maxx, maxy = arr[:, 2].max(), arr[:, 3].max()
    padx = max((maxx - minx) * pad_ratio, 0.004)
    pady = max((maxy - miny) * pad_ratio, 0.004)
    for ax in axes:
        ax.set_xlim(minx - padx, maxx + padx)
        ax.set_ylim(miny - pady, maxy + pady)


def balanced_grid_generalize(gdf: gpd.GeoDataFrame, scale: int, seed: int = 42) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf.copy()
    ratio = 0.48 if scale <= 100000 else (0.36 if scale <= 300000 else 0.28)
    target = max(len(CATEGORY_ORDER) * 3, int(round(len(gdf) * ratio)))
    target = min(target, len(gdf))

    protected_parts = []
    remaining_parts = []
    for category in CATEGORY_ORDER:
        subset = gdf[gdf["category"] == category].copy()
        if subset.empty:
            continue
        protected = subset.head(min(3, len(subset)))
        protected_parts.append(protected)
        remaining_parts.append(subset.drop(index=protected.index))

    protected_gdf = gpd.GeoDataFrame(pd.concat(protected_parts, ignore_index=False), geometry="geometry", crs=gdf.crs)
    remaining = gpd.GeoDataFrame(pd.concat(remaining_parts, ignore_index=False), geometry="geometry", crs=gdf.crs) if remaining_parts else gdf.iloc[0:0].copy()
    slots = max(target - len(protected_gdf), 0)
    if slots <= 0 or remaining.empty:
        return protected_gdf.head(target).copy()

    b = gdf.total_bounds
    span = max(float(b[2] - b[0]), float(b[3] - b[1]), 1e-6)
    step = span / max(6.0, np.sqrt(slots) * (1.6 if scale <= 300000 else 1.35))
    work = remaining.copy()
    work["gx"] = np.floor((work.geometry.x - b[0]) / step).astype(int)
    work["gy"] = np.floor((work.geometry.y - b[1]) / step).astype(int)
    work["cell"] = work["gx"].astype(str) + "_" + work["gy"].astype(str)
    picked = work.sort_values(["category", "poi_uid" if "poi_uid" in work.columns else "category"]).groupby(["category", "cell"], as_index=False).head(1)
    if len(picked) > slots:
        picked = picked.sample(slots, random_state=seed)
    picked = picked.drop(columns=["gx", "gy", "cell"])
    return gpd.GeoDataFrame(pd.concat([protected_gdf, picked], ignore_index=False), geometry="geometry", crs=gdf.crs)


def thin_for_display(gdf: gpd.GeoDataFrame, target_count: int, seed: int = 42) -> gpd.GeoDataFrame:
    if len(gdf) <= target_count:
        return gdf.copy()
    parts = []
    per_cat = max(2, target_count // len(CATEGORY_ORDER))
    for category in CATEGORY_ORDER:
        subset = gdf[gdf["category"] == category]
        if subset.empty:
            continue
        take = min(len(subset), per_cat)
        parts.append(subset.sample(take, random_state=seed))
    out = gpd.GeoDataFrame(pd.concat(parts, ignore_index=False), geometry="geometry", crs=gdf.crs) if parts else gdf.iloc[0:0].copy()
    if len(out) < target_count:
        rest = gdf.drop(index=out.index, errors="ignore")
        if not rest.empty:
            out = gpd.GeoDataFrame(pd.concat([out, rest.sample(min(target_count - len(out), len(rest)), random_state=seed)]), geometry="geometry", crs=gdf.crs)
    return out


def pick_dense_region(source: gpd.GeoDataFrame) -> Tuple[str, str]:
    stats = source.groupby(["city", "district"]).agg(count=("poi_uid", "size"), cat=("category", "nunique")).reset_index()
    candidates = stats[(stats["count"].between(180, 520)) & (stats["cat"] >= 8)].copy()
    if candidates.empty:
        candidates = stats[stats["cat"] >= 7].copy()
    row = candidates.sort_values(["cat", "count"], ascending=[False, False]).iloc[0]
    return str(row["city"]), str(row["district"])


def choose_grid_regions(source: gpd.GeoDataFrame, n: int = 9) -> List[Tuple[str, str, int]]:
    stats = source.groupby(["city", "district"]).agg(count=("poi_uid", "size"), cat=("category", "nunique")).reset_index()
    candidates = stats[(stats["count"].between(90, 420)) & (stats["cat"] >= 7)].copy()
    if candidates.empty:
        candidates = stats.sort_values("count", ascending=False).head(n).copy()
    candidates = candidates.sort_values(["cat", "count"], ascending=[False, False]).head(n)
    scales = [84000, 253000, 532000] * 3
    return [(str(row.city), str(row.district), scales[i]) for i, row in enumerate(candidates.itertuples())]


def draw_manual_legend(fig, y: float = 0.055) -> None:
    ax = fig.add_axes([0.18, y - 0.018, 0.64, 0.055])
    ax.set_axis_off()
    xs = np.linspace(0.045, 0.955, len(CATEGORY_ORDER))
    for x, category in zip(xs, CATEGORY_ORDER):
        add_symbol(ax, x, 0.58, category, size=20, xycoords="axes fraction", zorder=10)
        ax.text(x + 0.020, 0.58, category, transform=ax.transAxes, va="center", ha="left", fontsize=9.5, color="#1f2933")
    ax.text(0.0, 0.58, "POI 类型", transform=ax.transAxes, va="center", ha="right", fontsize=10, color="#374151")


def draw_type1_figure(source: gpd.GeoDataFrame) -> Path:
    city, district = pick_dense_region(source)
    scale = 253000
    raw = source[(source["city"] == city) & (source["district"] == district)].copy()
    if len(raw) > 360:
        raw = thin_for_display(raw, 360, seed=7)
    generated = balanced_grid_generalize(raw, scale=scale, seed=13)
    raw_show = thin_for_display(raw, 115, seed=11)
    gen_show = thin_for_display(generated, 92, seed=17)

    fig, axes = plt.subplots(2, 3, figsize=(16.2, 9.6))
    fig.subplots_adjust(left=0.055, right=0.985, top=0.90, bottom=0.145, wspace=0.16, hspace=0.24)
    ax_a, ax_b, ax_c, ax_d, ax_e, ax_f = axes.ravel()
    for ax in axes.ravel()[:5]:
        style_axis(ax, show_ticks=True)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
    set_common_extent([ax_a, ax_b, ax_c, ax_e], [raw_show, gen_show])

    draw_symbol_points(ax_a, raw_show, size=11, gap_factor=1.40)
    ax_a.set_title("(a) 原始点分布", fontsize=11)

    draw_symbol_points(ax_b, gen_show, size=11, gap_factor=1.40)
    ax_b.set_title("(b) 综合后点分布", fontsize=11)

    ax_c.scatter(raw_show.geometry.x, raw_show.geometry.y, s=10, c="#b5bac1", alpha=0.22, linewidths=0, zorder=1)
    draw_symbol_points(ax_c, gen_show, size=11, alpha_underlay=0.04, gap_factor=1.40)
    ax_c.text(0.98, 0.02, "灰色: 原始点  彩色符号: 综合后", transform=ax_c.transAxes, ha="right", va="bottom", fontsize=8, color="#4b5563")
    ax_c.set_title("(c) 前后叠加对比", fontsize=11)

    b = raw.total_bounds
    minx, miny, maxx, maxy = [float(v) for v in b]
    qx0, qx1 = minx + (maxx - minx) * 0.32, minx + (maxx - minx) * 0.62
    qy0, qy1 = miny + (maxy - miny) * 0.30, miny + (maxy - miny) * 0.62
    raw_zoom = raw.cx[qx0:qx1, qy0:qy1]
    if raw_zoom.empty:
        raw_zoom = raw_show
    gen_zoom = generated.cx[qx0:qx1, qy0:qy1]
    if gen_zoom.empty:
        gen_zoom = gen_show
    raw_zoom = thin_for_display(raw_zoom, 42, seed=21)
    gen_zoom = thin_for_display(gen_zoom, 34, seed=23)
    set_common_extent([ax_d], [raw_zoom, gen_zoom], pad_ratio=0.18)
    ax_d.scatter(raw_zoom.geometry.x, raw_zoom.geometry.y, s=11, c="#b5bac1", alpha=0.25, linewidths=0, zorder=1)
    draw_symbol_points(ax_d, gen_zoom, size=12, alpha_underlay=0.03, gap_factor=1.45)
    ax_d.set_title("(d) 局部区域放大对比", fontsize=11)

    ax_e.scatter(raw_show.geometry.x, raw_show.geometry.y, s=9, c="#c6cbd1", alpha=0.18, linewidths=0, zorder=1)
    draw_symbol_points(ax_e, gen_show, size=11, alpha_underlay=0.03, gap_factor=1.40)
    ax_e.set_title("(e) 符号冲突消解示意", fontsize=11)
    ax_e.text(0.98, 0.03, f"raw={len(raw)}  gen={len(generated)}", transform=ax_e.transAxes, ha="right", va="bottom", fontsize=8, color="#8d1f1f")

    summary = (
        raw.groupby("category").size().rename("before").to_frame()
        .join(generated.groupby("category").size().rename("after"), how="left")
        .fillna(0)
        .reset_index()
    )
    summary["order"] = summary["category"].map(CATEGORY_ORDER.index)
    summary = summary.sort_values("order")
    summary["ratio"] = summary["after"] / summary["before"]
    bars = ax_f.bar(summary["category"], summary["ratio"], color=[COLORS[c] for c in summary["category"]], edgecolor="#263238", linewidth=0.6, zorder=3)
    for bar, value in zip(bars, summary["ratio"]):
        ax_f.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    ax_f.set_ylim(0, 1.12)
    ax_f.set_ylabel("Retention ratio")
    ax_f.set_xlabel("POI 类型")
    ax_f.tick_params(axis="x", rotation=35, labelsize=8)
    ax_f.set_title("(f) 类别保留率统计", fontsize=11)
    style_axis(ax_f, show_ticks=True)

    draw_manual_legend(fig, y=0.058)
    fig.suptitle(f"顾及经典符号语义的 POI 点综合结果（{city}{district}，1:{scale}）", fontsize=14, fontweight="bold", y=0.965)
    out_base = OUTPUT_DIR / "figure_type1_publication_panel_classic_symbols"
    for suffix, kwargs in [(".png", {"dpi": 360}), (".pdf", {}), (".svg", {})]:
        fig.savefig(out_base.with_suffix(suffix), bbox_inches="tight", **kwargs)
    plt.close(fig)
    return out_base.with_suffix(".png")


def draw_type2_grid(source: gpd.GeoDataFrame) -> Path:
    regions = choose_grid_regions(source, n=9)
    fig, axes = plt.subplots(3, 6, figsize=(19.2, 10.8))
    fig.subplots_adjust(left=0.03, right=0.99, top=0.885, bottom=0.115, wspace=0.07, hspace=0.32)
    for idx, (city, district, scale) in enumerate(regions):
        row = idx // 3
        col_pair = (idx % 3) * 2
        ax_before = axes[row, col_pair]
        ax_after = axes[row, col_pair + 1]
        raw = source[(source["city"] == city) & (source["district"] == district)].copy()
        raw_show = thin_for_display(raw, 50, seed=idx + 5)
        gen = balanced_grid_generalize(raw, scale=scale, seed=idx + 31)
        gen_show = thin_for_display(gen, 42, seed=idx + 41)
        set_common_extent([ax_before, ax_after], [raw_show, gen_show], pad_ratio=0.16)
        for ax in (ax_before, ax_after):
            style_axis(ax, show_ticks=False)
        draw_symbol_points(ax_before, raw_show, size=8, alpha_underlay=0.06, gap_factor=1.48)
        draw_symbol_points(ax_after, gen_show, size=8, alpha_underlay=0.05, gap_factor=1.48)
        ax_before.set_title(f"{city}{district} 1:{scale}\n综合前", fontsize=9.2, color="#111827")
        ax_after.set_title("综合后", fontsize=9.2, color="#111827")
        ax_before.text(1.03, 0.50, "→", transform=ax_before.transAxes, ha="left", va="center", fontsize=18, color="#8f969e", fontweight="bold")

    draw_manual_legend(fig, y=0.047)
    fig.suptitle("多区域经典符号点综合前后对比（八类 POI）", fontsize=15, fontweight="bold", y=0.975)
    out_base = OUTPUT_DIR / "figure_type2_multiregion_grid_classic_symbols"
    for suffix, kwargs in [(".png", {"dpi": 320}), (".pdf", {}), (".svg", {})]:
        fig.savefig(out_base.with_suffix(suffix), bbox_inches="tight", **kwargs)
    plt.close(fig)
    return out_base.with_suffix(".png")


def write_symbol_note(path1: Path, path2: Path) -> Path:
    readme = OUTPUT_DIR / "README_classic_symbols.md"
    readme.write_text(
        "\n".join(
            [
                "# Classic POI Symbol Outputs",
                "",
                "本次重新输出使用八类固定语义化地图符号，不再使用圆点、方块、三角形等抽象散点符号。",
                "",
                "| POI 类别 | 符号语义 | 颜色 |",
                "| --- | --- | --- |",
                "| 便利店 | 店铺/24h 商店 | 洋红 |",
                "| 学校 | 学位帽 | 绿色 |",
                "| 停车场 | P | 蓝色 |",
                "| 酒店 | 床 | 紫色 |",
                "| 银行 | 柱廊银行 | 青绿色 |",
                "| 医院 | 医疗十字 | 红色 |",
                "| 餐饮 | 刀叉 | 橙色 |",
                "| 交通设施 | 公交车 | 蓝灰色 |",
                "",
                "类别归并规则：公交站、地铁站归并为交通设施；餐厅、咖啡厅、餐饮归并为餐饮。",
                "",
                "参考符号来源：",
                "- Google Material Symbols / Material Design Icons: https://fonts.google.com/icons",
                "- Mapbox Maki: https://docs.mapbox.com/help/glossary/maki/ 和 https://github.com/mapbox/maki",
                "- OpenStreetMap map features: https://wiki.openstreetmap.org/wiki/Map_features",
                "",
                f"- 图1（六子图综合诊断图）：{path1}",
                f"- 图2（多区域综合前后对比图）：{path2}",
                "",
                "输出同时包含 PNG、PDF、SVG 三种格式，位于同一目录。",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return readme


def main() -> None:
    configure_plot()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    source = normalize_categories(gpd.read_file(DATA_PATH))
    keep_cols = [c for c in ["poi_uid", "name", "type", "category", "city", "district", "geometry"] if c in source.columns]
    source = source[keep_cols].copy()
    path1 = draw_type1_figure(source)
    path2 = draw_type2_grid(source)
    readme = write_symbol_note(path1, path2)
    print(path1)
    print(path2)
    print(readme)


if __name__ == "__main__":
    main()
