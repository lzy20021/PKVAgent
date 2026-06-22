from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_CSV = PROJECT_ROOT / "amap_poi_dataset_expanded" / "all_poi_expanded.csv"
OUT_DIR = PROJECT_ROOT / "paper_figures" / "expanded_structure_map" / "overview_detail_layout"

OUT_PNG = OUT_DIR / "fig_poi_overview_with_8_details.png"
OUT_SVG = OUT_DIR / "fig_poi_overview_with_8_details.svg"
OUT_PDF = OUT_DIR / "fig_poi_overview_with_8_details.pdf"
OUT_SCRIPT = OUT_DIR / "make_poi_overview_with_8_details.py"
OUT_REGIONS = OUT_DIR / "selected_detail_regions.csv"
OUT_SUMMARY = OUT_DIR / "summary_overview_with_8_details.txt"

TYPE_BASE = ["公交站/地铁站", "医院", "学校", "餐饮", "便利店", "停车场", "酒店", "银行"]
POI_COLOR_MAP = {
    "公交站/地铁站": "#FF0000",
    "医院": "#0070FF",
    "学校": "#00B050",
    "餐饮": "#FFC000",
    "便利店": "#FF00FF",
    "停车场": "#00FFFF",
    "酒店": "#92278F",
    "银行": "#FF6600",
}

POINT_SIZE_DEFAULT = 16.0
POINT_SIZE_MIN = 12.0
POINT_SIZE_MAX = 18.0
POINT_ALPHA_MIN = 0.78
POINT_ALPHA_MAX = 0.88
POINT_EDGE_COLOR = "white"
POINT_EDGE_WIDTH = 0.30
MARKER = "o"


def setup_style() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["axes.linewidth"] = 0.5


def map_poi_type(text: str) -> Optional[str]:
    t = str(text).lower()
    if any(k in t for k in ["餐饮", "餐厅", "饭店", "restaurant", "food", "catering"]):
        return "餐饮"
    if any(k in t for k in ["学校", "教育", "school", "education", "大学", "学院"]):
        return "学校"
    if any(k in t for k in ["医院", "医疗", "medical", "hospital", "clinic"]):
        return "医院"
    if any(k in t for k in ["公交", "bus", "地铁", "metro", "subway"]):
        return "公交站/地铁站"
    if any(k in t for k in ["酒店", "hotel"]):
        return "酒店"
    if any(k in t for k in ["银行", "bank"]):
        return "银行"
    if any(k in t for k in ["便利店", "便利", "convenience"]):
        return "便利店"
    if any(k in t for k in ["停车场", "parking"]):
        return "停车场"
    return None


def density_aware_keep(g: pd.DataFrame, rng_seed: int) -> pd.DataFrame:
    n = len(g)
    if n <= 1500:
        return g
    lon = g["lon"].to_numpy(dtype=float)
    lat = g["lat"].to_numpy(dtype=float)
    minx, maxx = float(lon.min()), float(lon.max())
    miny, maxy = float(lat.min()), float(lat.max())
    dx = max(maxx - minx, 1e-9)
    dy = max(maxy - miny, 1e-9)
    grid_n = int(np.clip(round(np.sqrt(n / 45.0)), 18, 52))
    gx = np.clip(((lon - minx) / dx * (grid_n - 1)).astype(int), 0, grid_n - 1)
    gy = np.clip(((lat - miny) / dy * (grid_n - 1)).astype(int), 0, grid_n - 1)
    cell = gx + gy * grid_n

    cc = pd.Series(cell, index=g.index).value_counts()
    ccount = pd.Series(cell, index=g.index).map(cc).astype(int)
    probs = np.where(ccount <= 4, 1.00, np.where(ccount <= 12, 0.90, np.where(ccount <= 28, 0.74, 0.60))).astype(float)
    if n >= 50000:
        probs = np.clip(probs - 0.03, 0.55, 1.0)
    probs = np.clip(probs, 0.55, 1.0)

    rng = np.random.default_rng(rng_seed + n)
    keep_mask = rng.random(n) < probs
    keep_count = int(keep_mask.sum())
    min_keep = int(math.ceil(0.55 * n))
    target_keep = int(math.ceil(0.70 * n))
    if keep_count < min_keep:
        idx_all = np.arange(n)
        drop_idx = idx_all[~keep_mask]
        need = min_keep - keep_count
        if len(drop_idx) > 0 and need > 0:
            extra = rng.choice(drop_idx, size=min(need, len(drop_idx)), replace=False)
            keep_mask[extra] = True
    if n <= 18000 and int(keep_mask.sum()) < target_keep:
        idx_all = np.arange(n)
        drop_idx = idx_all[~keep_mask]
        need = target_keep - int(keep_mask.sum())
        if len(drop_idx) > 0 and need > 0:
            extra = rng.choice(drop_idx, size=min(need, len(drop_idx)), replace=False)
            keep_mask[extra] = True
    return g.iloc[keep_mask]


def build_preview_df(df: pd.DataFrame) -> pd.DataFrame:
    parts: List[pd.DataFrame] = []
    for i, (_, g) in enumerate(df.groupby(["city", "poi_type"], sort=False)):
        parts.append(density_aware_keep(g, 20260319 + i * 97))
    return pd.concat(parts, ignore_index=True)


def adaptive_style_by_type(draw_df: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
    counts = draw_df["poi_type"].value_counts()
    if counts.empty:
        return {}
    max_n = float(counts.max())
    out: Dict[str, Tuple[float, float]] = {}
    for t, n in counts.items():
        frac = (float(n) / max(max_n, 1.0)) ** 0.5
        size = float(np.clip(POINT_SIZE_DEFAULT + 2.0 * (1.0 - frac), POINT_SIZE_MIN, POINT_SIZE_MAX))
        alpha = float(np.clip(0.82 + 0.06 * (1.0 - frac), POINT_ALPHA_MIN, POINT_ALPHA_MAX))
        out[t] = (size, alpha)
    return out


def add_north_arrow(ax: plt.Axes) -> None:
    ax.annotate("", xy=(0.965, 0.955), xytext=(0.965, 0.88), xycoords="axes fraction", arrowprops=dict(arrowstyle="-|>", lw=1.0, color="#222222"))
    ax.text(0.965, 0.965, "N", transform=ax.transAxes, ha="center", va="bottom", fontsize=11, fontweight="bold", color="#222222")


def add_scale_bar(ax: plt.Axes, xmin: float, xmax: float, ymin: float, ymax: float) -> None:
    lat_ref = (ymin + ymax) / 2.0
    cos_lat = max(math.cos(math.radians(lat_ref)), 1e-3)
    km = 500.0
    deg_per_km_lon = 1.0 / (111.32 * cos_lat)
    bar_deg = km * deg_per_km_lon
    x0 = xmin + 0.04 * (xmax - xmin)
    y0 = ymin + 0.05 * (ymax - ymin)
    x1 = x0 + bar_deg
    tick = 0.011 * (ymax - ymin)
    ax.plot([x0, x1], [y0, y0], color="#222222", lw=1.1, zorder=8)
    ax.plot([x0, x0], [y0 - tick, y0 + tick], color="#222222", lw=1.1, zorder=8)
    ax.plot([x1, x1], [y0 - tick, y0 + tick], color="#222222", lw=1.1, zorder=8)
    ax.text((x0 + x1) / 2, y0 - 2.0 * tick, "500 km", ha="center", va="top", fontsize=9, color="#222222")


def region_mask(df: pd.DataFrame, xmin: float, xmax: float, ymin: float, ymax: float) -> pd.Series:
    return (df["lon"] >= xmin) & (df["lon"] <= xmax) & (df["lat"] >= ymin) & (df["lat"] <= ymax)


def get_regions(df: pd.DataFrame) -> List[Dict[str, object]]:
    # Preferred representative dense regions across China.
    base = [
        ("A", "京津冀", 114.2, 118.6, 37.8, 41.2),
        ("B", "长三角", 118.0, 122.6, 29.0, 32.8),
        ("C", "珠三角", 112.4, 114.9, 22.1, 24.2),
        ("D", "成渝", 102.8, 107.7, 28.4, 31.8),
        ("E", "长江中游", 111.5, 116.5, 27.8, 31.4),
        ("F", "中原", 111.2, 115.9, 33.0, 36.7),
        ("G", "山东半岛", 118.0, 122.8, 35.0, 38.5),
        ("H", "东北南部", 120.0, 124.8, 39.8, 43.8),
    ]
    regions: List[Dict[str, object]] = []
    for rid, name, xmin, xmax, ymin, ymax in base:
        m = region_mask(df, xmin, xmax, ymin, ymax)
        sub = df[m]
        cities = sub["city"].value_counts().head(6).index.tolist()
        regions.append(
            {
                "region_id": rid,
                "region_name": name,
                "xmin": xmin,
                "xmax": xmax,
                "ymin": ymin,
                "ymax": ymax,
                "point_count": int(len(sub)),
                "dominant_cities": "|".join(cities),
            }
        )
    return regions


def apply_common_axis_style(ax: plt.Axes, xmin: float, xmax: float, ymin: float, ymax: float, with_grid: bool = True) -> None:
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    xt = np.arange(math.floor(xmin / 5) * 5, math.ceil(xmax / 5) * 5 + 1, 5)
    yt = np.arange(math.floor(ymin / 5) * 5, math.ceil(ymax / 5) * 5 + 1, 5)
    ax.set_xticks(xt)
    ax.set_yticks(yt)
    if with_grid:
        ax.grid(True, color="#eceff3", lw=0.5, zorder=1)


def plot_points(ax: plt.Axes, data: pd.DataFrame, style_map: Dict[str, Tuple[float, float]], z_base: float = 3.0) -> None:
    ordered = [t for t in TYPE_BASE if t in data["poi_type"].unique().tolist()]
    draw_order = sorted(ordered, key=lambda t: int((data["poi_type"] == t).sum()), reverse=True)
    for i, pt in enumerate(draw_order):
        sub = data[data["poi_type"] == pt]
        if sub.empty:
            continue
        s, alpha = style_map.get(pt, (POINT_SIZE_DEFAULT, 0.83))
        ax.scatter(
            sub["lon"],
            sub["lat"],
            s=s,
            marker=MARKER,
            c=POI_COLOR_MAP[pt],
            alpha=alpha,
            linewidths=POINT_EDGE_WIDTH,
            edgecolors=POINT_EDGE_COLOR,
            zorder=z_base + i * 0.2,
        )


def make_figure(df: pd.DataFrame, preview_df: pd.DataFrame, regions: List[Dict[str, object]]) -> None:
    style_map = adaptive_style_by_type(preview_df)

    fig = plt.figure(figsize=(22, 12), facecolor="white")
    gs = fig.add_gridspec(nrows=4, ncols=3, width_ratios=[1.05, 0.05, 1.20], wspace=0.02, hspace=0.00)
    ax_left = fig.add_subplot(gs[:, 0])
    right = gs[:, 2].subgridspec(4, 2, wspace=0.16, hspace=0.28)
    right_axes = [fig.add_subplot(right[r, c]) for r in range(4) for c in range(2)]

    xmin, xmax = float(df["lon"].min()), float(df["lon"].max())
    ymin, ymax = float(df["lat"].min()), float(df["lat"].max())
    padx = (xmax - xmin) * 0.06
    pady = (ymax - ymin) * 0.08
    xmin, xmax = xmin - padx, xmax + padx
    ymin, ymax = ymin - pady, ymax + pady

    plot_points(ax_left, preview_df, style_map, z_base=3.0)
    apply_common_axis_style(ax_left, xmin, xmax, ymin, ymax, with_grid=True)
    ax_left.set_xlabel("经度 (°E)", fontsize=11)
    ax_left.set_ylabel("纬度 (°N)", fontsize=11)
    ax_left.set_title("全国典型 POI 空间分布预览及局部细节图", fontsize=16, fontweight="bold", pad=10)
    add_north_arrow(ax_left)
    add_scale_bar(ax_left, xmin, xmax, ymin, ymax)

    box_color = "#36454F"
    for reg in regions:
        rect = Rectangle(
            (float(reg["xmin"]), float(reg["ymin"])),
            float(reg["xmax"]) - float(reg["xmin"]),
            float(reg["ymax"]) - float(reg["ymin"]),
            fill=False,
            edgecolor=box_color,
            linewidth=1.35,
            alpha=0.88,
            zorder=9,
        )
        ax_left.add_patch(rect)
        ax_left.text(
            float(reg["xmin"]) + 0.15,
            float(reg["ymax"]) - 0.12,
            str(reg["region_id"]),
            fontsize=10.5,
            color=box_color,
            fontweight="bold",
            va="top",
            ha="left",
            zorder=10,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.70, pad=1.2),
        )

    handles = []
    left_types = [t for t in TYPE_BASE if t in preview_df["poi_type"].unique().tolist()]
    for t in left_types:
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=POI_COLOR_MAP[t],
                markeredgecolor="white",
                markeredgewidth=0.6,
                markersize=9.2,
                label=t,
            )
        )
    ax_left.legend(
        handles=handles,
        title="POI类别（颜色）",
        loc="lower right",
        bbox_to_anchor=(0.995, 0.03),
        frameon=True,
        facecolor="white",
        edgecolor="#d8dde5",
        fontsize=8.7,
        title_fontsize=9.0,
    )

    for ax, reg in zip(right_axes, regions):
        m = region_mask(df, float(reg["xmin"]), float(reg["xmax"]), float(reg["ymin"]), float(reg["ymax"]))
        sub = df[m]
        # Keep detail panels rich while avoiding severe overplot.
        detail_for_plot = build_preview_df(sub) if len(sub) > 22000 else sub
        detail_style = adaptive_style_by_type(detail_for_plot)
        plot_points(ax, detail_for_plot, detail_style, z_base=3.0)

        x0, x1 = float(reg["xmin"]), float(reg["xmax"])
        y0, y1 = float(reg["ymin"]), float(reg["ymax"])
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        tx = np.linspace(x0, x1, 3)
        ty = np.linspace(y0, y1, 3)
        ax.set_xticks(tx)
        ax.set_yticks(ty)
        ax.tick_params(labelsize=7.0, length=2.5, width=0.45)
        ax.grid(True, color="#eef1f5", lw=0.45, zorder=1)
        ax.set_title(f'{reg["region_id"]} 区域（{reg["region_name"]}）', fontsize=9.2, pad=2.5)
        for spine in ax.spines.values():
            spine.set_color("#b7c0ca")
            spine.set_linewidth(0.5)

    fig.subplots_adjust(left=0.03, right=0.992, bottom=0.04, top=0.955, wspace=0.02, hspace=0.02)
    fig.savefig(OUT_PNG, dpi=600)
    fig.savefig(OUT_SVG)
    fig.savefig(OUT_PDF)
    plt.close(fig)


def prepare_data() -> pd.DataFrame:
    if not DATA_CSV.exists():
        raise RuntimeError(f"Missing input data: {DATA_CSV}")
    df = pd.read_csv(DATA_CSV)
    for col in ["city", "wgs84_lon", "wgs84_lat", "query_category", "type"]:
        if col not in df.columns:
            raise RuntimeError(f"Missing column in expanded csv: {col}")

    data = df.rename(columns={"wgs84_lon": "lon", "wgs84_lat": "lat"}).copy()
    data["poi_type"] = np.where(
        data["query_category"].notna() & (data["query_category"].astype(str).str.len() > 0),
        data["query_category"].astype(str),
        data["type"].astype(str),
    )
    data["poi_type"] = data["poi_type"].map(map_poi_type)
    data = data[data["poi_type"].notna()].copy()
    data = data[(data["lon"].between(70, 140)) & (data["lat"].between(15, 55))].copy()
    data["city"] = data["city"].astype(str)
    return data


def write_outputs(regions: List[Dict[str, object]], input_files: List[Path]) -> None:
    reg_df = pd.DataFrame(regions)
    reg_df.to_csv(OUT_REGIONS, index=False, encoding="utf-8-sig")

    lines = [
        "全国 POI 总览+8细节图 生成摘要",
        "",
        "输入文件:",
    ]
    lines.extend([f"- {p.as_posix()}" for p in input_files])
    lines.append("")
    lines.append("选择的8个区域:")
    for r in regions:
        lines.append(
            f'- {r["region_id"]} {r["region_name"]}: 点数 {r["point_count"]}, '
            f'范围 [{r["xmin"]}, {r["xmax"]}] x [{r["ymin"]}, {r["ymax"]}], '
            f'主要城市 {r["dominant_cities"]}'
        )
    lines.append("")
    lines.append("输出文件:")
    lines.append(f"- {OUT_PNG.as_posix()}")
    lines.append(f"- {OUT_SVG.as_posix()}")
    lines.append(f"- {OUT_PDF.as_posix()}")
    lines.append(f"- {OUT_SCRIPT.as_posix()}")
    lines.append(f"- {OUT_REGIONS.as_posix()}")
    lines.append(f"- {OUT_SUMMARY.as_posix()}")
    OUT_SUMMARY.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    setup_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = prepare_data()
    preview_df = build_preview_df(df)
    regions = get_regions(df)
    make_figure(df=df, preview_df=preview_df, regions=regions)
    write_outputs(regions=regions, input_files=[DATA_CSV])

    print(f"输入数据: {DATA_CSV.as_posix()}")
    print(f"总点数: {len(df)}")
    print("8个细节区域:")
    for r in regions:
        print(f'- {r["region_id"]} {r["region_name"]}: {r["point_count"]} 点')
    print(f"输出图: {OUT_PNG.as_posix()}")
    print(f"区域表: {OUT_REGIONS.as_posix()}")
    print(f"摘要: {OUT_SUMMARY.as_posix()}")


if __name__ == "__main__":
    main()
