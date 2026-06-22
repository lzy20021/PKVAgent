from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IN_CSV = PROJECT_ROOT / "paper_figures" / "merged_poi_with_taiwan" / "all_poi_merged.csv"
OUT_DIR = PROJECT_ROOT / "paper_figures" / "merged_poi_with_taiwan"
OUT_PREVIEW = OUT_DIR / "fig_china_poi_preview_with_taiwan.png"
OUT_COMBO = OUT_DIR / "fig_overview_detail_with_taiwan.png"
OUT_SUMMARY = OUT_DIR / "summary_with_taiwan.txt"

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


def setup_style() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["axes.linewidth"] = 0.5


def density_aware_keep(g: pd.DataFrame, seed: int) -> pd.DataFrame:
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
    probs = np.clip(probs, 0.55, 1.0)
    rng = np.random.default_rng(seed + n)
    mask = rng.random(n) < probs
    return g.iloc[mask]


def build_preview_df(df: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for i, (_, g) in enumerate(df.groupby(["city", "poi_type"], sort=False)):
        parts.append(density_aware_keep(g, 20260319 + i * 97))
    return pd.concat(parts, ignore_index=True)


def adaptive_style(df: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
    c = df["poi_type"].value_counts()
    if c.empty:
        return {}
    max_n = float(c.max())
    m: Dict[str, Tuple[float, float]] = {}
    for t, n in c.items():
        frac = (float(n) / max_n) ** 0.5
        size = float(np.clip(POINT_SIZE_DEFAULT + 2.0 * (1.0 - frac), POINT_SIZE_MIN, POINT_SIZE_MAX))
        alpha = float(np.clip(0.82 + 0.06 * (1.0 - frac), POINT_ALPHA_MIN, POINT_ALPHA_MAX))
        m[t] = (size, alpha)
    return m


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


def plot_points(ax: plt.Axes, df: pd.DataFrame, style_map: Dict[str, Tuple[float, float]]) -> None:
    ordered = [t for t in TYPE_BASE if t in df["poi_type"].unique().tolist()]
    draw_order = sorted(ordered, key=lambda t: int((df["poi_type"] == t).sum()), reverse=True)
    for i, t in enumerate(draw_order):
        sub = df[df["poi_type"] == t]
        s, a = style_map.get(t, (POINT_SIZE_DEFAULT, 0.83))
        ax.scatter(
            sub["lon"],
            sub["lat"],
            s=s,
            marker="o",
            c=POI_COLOR_MAP[t],
            alpha=a,
            linewidths=POINT_EDGE_WIDTH,
            edgecolors=POINT_EDGE_COLOR,
            zorder=3 + i * 0.2,
        )


def make_preview(df: pd.DataFrame, draw_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(13.5, 9.6), facecolor="white")
    style_map = adaptive_style(draw_df)
    plot_points(ax, draw_df, style_map)

    xmin, xmax = float(df["lon"].min()), float(df["lon"].max())
    ymin, ymax = float(df["lat"].min()), float(df["lat"].max())
    padx = (xmax - xmin) * 0.06
    pady = (ymax - ymin) * 0.08
    xmin, xmax = xmin - padx, xmax + padx
    ymin, ymax = ymin - pady, ymax + pady
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    xt = np.arange(math.floor(xmin / 5) * 5, math.ceil(xmax / 5) * 5 + 1, 5)
    yt = np.arange(math.floor(ymin / 5) * 5, math.ceil(ymax / 5) * 5 + 1, 5)
    ax.set_xticks(xt)
    ax.set_yticks(yt)
    ax.grid(True, color="#eceff3", lw=0.5, zorder=1)
    ax.set_xlabel("经度 (°E)", fontsize=11)
    ax.set_ylabel("纬度 (°N)", fontsize=11)
    ax.set_title("全国城市典型 POI 空间结构分布图（8类）", fontsize=16, fontweight="bold", pad=10)
    add_north_arrow(ax)
    add_scale_bar(ax, xmin, xmax, ymin, ymax)

    handles = []
    for k in TYPE_BASE:
        if k not in draw_df["poi_type"].unique():
            continue
        handles.append(
            Line2D([0], [0], marker="o", color="none", markerfacecolor=POI_COLOR_MAP[k], markeredgecolor="white", markeredgewidth=0.6, markersize=9.4, label=k)
        )
    ax.legend(handles=handles, title="POI类别（颜色）", loc="lower right", bbox_to_anchor=(0.995, 0.03), frameon=True, facecolor="white", edgecolor="#d8dde5", fontsize=8.8, title_fontsize=9)

    fig.tight_layout(rect=[0.02, 0.02, 0.985, 0.97])
    fig.savefig(OUT_PREVIEW, dpi=600)
    plt.close(fig)


def region_mask(df: pd.DataFrame, x0: float, x1: float, y0: float, y1: float) -> pd.Series:
    return (df["lon"] >= x0) & (df["lon"] <= x1) & (df["lat"] >= y0) & (df["lat"] <= y1)


def make_combo(df: pd.DataFrame, draw_df: pd.DataFrame) -> None:
    regions = [
        ("A", "台湾", 119.4, 122.3, 21.7, 25.6),
        ("B", "京津冀", 114.2, 118.6, 37.8, 41.2),
        ("C", "长三角", 118.0, 122.6, 29.0, 32.8),
        ("D", "珠三角", 112.4, 114.9, 22.1, 24.2),
        ("E", "成渝", 102.8, 107.7, 28.4, 31.8),
        ("F", "长江中游", 111.5, 116.5, 27.8, 31.4),
        ("G", "中原", 111.2, 115.9, 33.0, 36.7),
        ("H", "山东半岛", 118.0, 122.8, 35.0, 38.5),
    ]

    fig = plt.figure(figsize=(22, 12), facecolor="white")
    gs = fig.add_gridspec(nrows=4, ncols=3, width_ratios=[1.05, 0.05, 1.20], wspace=0.02, hspace=0.00)
    ax_left = fig.add_subplot(gs[:, 0])
    right = gs[:, 2].subgridspec(4, 2, wspace=0.16, hspace=0.28)
    right_axes = [fig.add_subplot(right[r, c]) for r in range(4) for c in range(2)]

    style_map = adaptive_style(draw_df)
    plot_points(ax_left, draw_df, style_map)

    xmin, xmax = float(df["lon"].min()), float(df["lon"].max())
    ymin, ymax = float(df["lat"].min()), float(df["lat"].max())
    padx = (xmax - xmin) * 0.06
    pady = (ymax - ymin) * 0.08
    xmin, xmax = xmin - padx, xmax + padx
    ymin, ymax = ymin - pady, ymax + pady
    ax_left.set_xlim(xmin, xmax)
    ax_left.set_ylim(ymin, ymax)
    xt = np.arange(math.floor(xmin / 5) * 5, math.ceil(xmax / 5) * 5 + 1, 5)
    yt = np.arange(math.floor(ymin / 5) * 5, math.ceil(ymax / 5) * 5 + 1, 5)
    ax_left.set_xticks(xt)
    ax_left.set_yticks(yt)
    ax_left.grid(True, color="#eceff3", lw=0.5, zorder=1)
    ax_left.set_xlabel("经度 (°E)", fontsize=11)
    ax_left.set_ylabel("纬度 (°N)", fontsize=11)
    ax_left.set_title("全国典型 POI 空间分布预览及局部细节图", fontsize=16, fontweight="bold", pad=10)
    add_north_arrow(ax_left)
    add_scale_bar(ax_left, xmin, xmax, ymin, ymax)

    for rid, name, x0, x1, y0, y1 in regions:
        rect = Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="#36454F", linewidth=1.35, alpha=0.88, zorder=9)
        ax_left.add_patch(rect)
        ax_left.text(x0 + 0.12, y1 - 0.10, rid, fontsize=10.5, color="#36454F", fontweight="bold", va="top", ha="left", zorder=10, bbox=dict(facecolor="white", edgecolor="none", alpha=0.70, pad=1.2))

    handles = []
    for k in TYPE_BASE:
        if k not in draw_df["poi_type"].unique():
            continue
        handles.append(
            Line2D([0], [0], marker="o", color="none", markerfacecolor=POI_COLOR_MAP[k], markeredgecolor="white", markeredgewidth=0.6, markersize=9.2, label=k)
        )
    ax_left.legend(handles=handles, title="POI类别（颜色）", loc="lower right", bbox_to_anchor=(0.995, 0.03), frameon=True, facecolor="white", edgecolor="#d8dde5", fontsize=8.7, title_fontsize=9.0)

    for ax, reg in zip(right_axes, regions):
        rid, name, x0, x1, y0, y1 = reg
        sub = df[region_mask(df, x0, x1, y0, y1)].copy()
        sub_draw = build_preview_df(sub) if len(sub) > 22000 else sub
        sub_style = adaptive_style(sub_draw)
        plot_points(ax, sub_draw, sub_style)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_xticks(np.linspace(x0, x1, 3))
        ax.set_yticks(np.linspace(y0, y1, 3))
        ax.tick_params(labelsize=7.0, length=2.5, width=0.45)
        ax.grid(True, color="#eef1f5", lw=0.45, zorder=1)
        ax.set_title(f"{rid} 区域（{name}）", fontsize=9.2, pad=2.5)
        for s in ax.spines.values():
            s.set_color("#b7c0ca")
            s.set_linewidth(0.5)

    fig.subplots_adjust(left=0.03, right=0.992, bottom=0.04, top=0.955, wspace=0.02, hspace=0.02)
    fig.savefig(OUT_COMBO, dpi=600)
    plt.close(fig)


def append_summary(df: pd.DataFrame) -> None:
    old = ""
    if OUT_SUMMARY.exists():
        old = OUT_SUMMARY.read_text(encoding="utf-8")
    lines = [
        "",
        "图件输出补充:",
        f"- 合并数据输入: {IN_CSV.as_posix()}",
        f"- 全国预览图: {OUT_PREVIEW.as_posix()}",
        f"- 总览+8细节图: {OUT_COMBO.as_posix()}",
        f"- 合并后总POI数量(复核): {len(df)}",
    ]
    OUT_SUMMARY.write_text(old + "\n" + "\n".join(lines), encoding="utf-8")


def main() -> None:
    setup_style()
    if not IN_CSV.exists():
        raise RuntimeError(f"Missing merged input: {IN_CSV}")
    df = pd.read_csv(IN_CSV)
    df = df[df["poi_type"].isin(TYPE_BASE)].copy()
    df = df[(df["lon"].between(70, 140)) & (df["lat"].between(15, 55))].copy()
    draw_df = build_preview_df(df)

    make_preview(df, draw_df)
    make_combo(df, draw_df)
    append_summary(df)
    print(f"预览图: {OUT_PREVIEW.as_posix()}")
    print(f"组合图: {OUT_COMBO.as_posix()}")
    print(f"摘要: {OUT_SUMMARY.as_posix()}")


if __name__ == "__main__":
    main()
