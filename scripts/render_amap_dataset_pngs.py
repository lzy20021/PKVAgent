from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
import pandas as pd
from matplotlib.patches import FancyArrowPatch, Rectangle
from pyproj import CRS, Transformer

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


SPLIT_COLORS = {"train": "#2E6BD2", "val": "#E38B2C", "test": "#28A25D"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render PNG maps from amap_poi_dataset sampled CSV.")
    p.add_argument("--input-csv", default="amap_poi_dataset/sampled/all_scales_sampled.csv")
    p.add_argument("--output-dir", default="amap_poi_dataset/png_maps")
    p.add_argument("--max-cities", type=int, default=0, help="0 means all cities")
    p.add_argument("--dpi", type=int, default=320)
    return p.parse_args()


def slugify(value: str) -> str:
    value = value.strip().replace(" ", "_")
    out = []
    for ch in value:
        if ch.isascii() and (ch.isalnum() or ch in {"_", "-"}):
            out.append(ch.lower())
        elif ch in {"_", "-"}:
            out.append(ch)
        else:
            out.append(f"u{ord(ch):x}")
    return "".join(out)


def nice_lonlat_step(span: float) -> float:
    if span > 12:
        return 2.0
    if span > 6:
        return 1.0
    if span > 3:
        return 0.5
    if span > 1.5:
        return 0.2
    if span > 0.8:
        return 0.1
    return 0.05


def frange(start: float, end: float, step: float) -> List[float]:
    vals: List[float] = []
    x = math.ceil(start / step) * step
    while x <= end + 1e-12:
        vals.append(round(x, 8))
        x += step
    return vals


def draw_north_arrow(fig: Any) -> None:
    x = 0.955
    y0 = 0.79
    y1 = 0.88
    fig.text(x, y1 + 0.02, "N", ha="center", va="bottom", fontsize=14, fontweight="bold", color="#000000")
    arr = FancyArrowPatch((x, y0), (x, y1), transform=fig.transFigure, arrowstyle="-|>", mutation_scale=18, lw=1.8, color="#000000")
    fig.add_artist(arr)


def draw_scale_bar(ax: Any, minx: float, miny: float, maxx: float, maxy: float) -> None:
    width = maxx - minx
    height = maxy - miny
    mean_lat = (miny + maxy) / 2.0
    meter_per_deg_lon = max(1.0, 111320.0 * math.cos(math.radians(mean_lat)))
    width_m = width * meter_per_deg_lon
    target = width_m * 0.22
    cands = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]
    bar_len_m = cands[0]
    for c in cands:
        if c <= target:
            bar_len_m = c
        else:
            break
    bar_len = bar_len_m / meter_per_deg_lon
    seg = bar_len / 4.0
    x0 = minx + width * 0.06
    y0 = miny + height * 0.07
    h = height * 0.015
    for i in range(4):
        xi = x0 + i * seg
        color = "#000000" if i % 2 == 0 else "#FFFFFF"
        ax.add_patch(Rectangle((xi, y0), seg, h, facecolor=color, edgecolor="#000000", linewidth=0.8, zorder=8))
    ax.add_patch(Rectangle((x0, y0), bar_len, h, fill=False, edgecolor="#000000", linewidth=0.8, zorder=8))
    unit_label = f"{int(bar_len_m/1000)} km" if bar_len_m >= 1000 else f"{int(bar_len_m)} m"
    mid_label = f"{bar_len_m/2000:.1f} km" if bar_len_m >= 2000 else f"{int(bar_len_m/2)} m"
    ax.text(x0, y0 - h * 0.7, "0", ha="center", va="top", fontsize=8, color="#111111", zorder=9)
    ax.text(x0 + bar_len / 2, y0 - h * 0.7, mid_label, ha="center", va="top", fontsize=8, color="#111111", zorder=9)
    ax.text(x0 + bar_len, y0 - h * 0.7, unit_label, ha="center", va="top", fontsize=8, color="#111111", zorder=9)
    ax.text(x0 + bar_len / 2, y0 + h * 1.7, "Scale Bar", ha="center", va="bottom", fontsize=8, color="#111111", zorder=9)


def draw_graticule(ax: Any, crs: CRS, minx: float, miny: float, maxx: float, maxy: float) -> None:
    to_ll = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    to_xy = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    corners_x = [minx, maxx, maxx, minx]
    corners_y = [miny, miny, maxy, maxy]
    lons, lats = to_ll.transform(corners_x, corners_y)
    lon_min, lon_max = float(min(lons)), float(max(lons))
    lat_min, lat_max = float(min(lats)), float(max(lats))
    lon_step = nice_lonlat_step(lon_max - lon_min)
    lat_step = nice_lonlat_step(lat_max - lat_min)
    lon_lines = frange(lon_min, lon_max, lon_step)
    lat_lines = frange(lat_min, lat_max, lat_step)
    width = maxx - minx
    height = maxy - miny
    x_in = minx + width * 0.01
    y_in = miny + height * 0.015

    for lon in lon_lines:
        lat_samples = np.linspace(lat_min, lat_max, 64)
        lon_samples = np.full_like(lat_samples, lon)
        x, y = to_xy.transform(lon_samples, lat_samples)
        ax.plot(x, y, color="#BEBEBE", linewidth=0.55, linestyle="--", alpha=0.8, zorder=1)
        tx, _ = to_xy.transform(lon, lat_min)
        if minx <= tx <= maxx:
            ax.text(tx, y_in, f"{lon:.2f}°E", ha="center", va="bottom", fontsize=7.2, color="#5A5A5A", zorder=2)

    for lat in lat_lines:
        lon_samples = np.linspace(lon_min, lon_max, 64)
        lat_samples = np.full_like(lon_samples, lat)
        x, y = to_xy.transform(lon_samples, lat_samples)
        ax.plot(x, y, color="#BEBEBE", linewidth=0.55, linestyle="--", alpha=0.8, zorder=1)
        _, ty = to_xy.transform(lon_min, lat)
        if miny <= ty <= maxy:
            ax.text(x_in, ty, f"{lat:.2f}°N", ha="left", va="center", fontsize=7.2, color="#5A5A5A", zorder=2)


def render_one(group_df: pd.DataFrame, out_png: Path, dpi: int) -> Dict[str, Any]:
    group_df = group_df[
        group_df["wgs84_lon"].between(70.0, 140.0, inclusive="both")
        & group_df["wgs84_lat"].between(3.0, 55.0, inclusive="both")
    ].copy()
    if group_df.empty:
        return {"city": "", "query_category": "", "target_scale": 0, "count": 0, "png_path": ""}

    city = str(group_df["city"].iloc[0])
    category = str(group_df["query_category"].iloc[0])
    scale = int(group_df["target_scale"].iloc[0])

    xs = group_df["wgs84_lon"].to_numpy()
    ys = group_df["wgs84_lat"].to_numpy()
    minx, maxx = float(xs.min()), float(xs.max())
    miny, maxy = float(ys.min()), float(ys.max())
    dx = max(1e-8, maxx - minx)
    dy = max(1e-8, maxy - miny)
    if dx > dy * 8:
        cy = (miny + maxy) / 2.0
        dy = dx / 8.0
        miny, maxy = cy - dy / 2.0, cy + dy / 2.0
    elif dy > dx * 8:
        cx = (minx + maxx) / 2.0
        dx = dy / 8.0
        minx, maxx = cx - dx / 2.0, cx + dx / 2.0
    pad_x = dx * 0.12
    pad_y = dy * 0.12
    ex0, ex1 = minx - pad_x, maxx + pad_x
    ey0, ey1 = miny - pad_y, maxy + pad_y

    fig, ax = plt.subplots(1, 1, figsize=(9.8, 7.2), facecolor="#FFFFFF")
    fig.subplots_adjust(left=0.08, right=0.92, top=0.88, bottom=0.08)

    crs_obj = CRS.from_epsg(4326)
    draw_graticule(ax, crs_obj, ex0, ey0, ex1, ey1)

    for split, sub in group_df.groupby("split"):
        color = SPLIT_COLORS.get(str(split), "#3A73D8")
        ax.scatter(sub["wgs84_lon"], sub["wgs84_lat"], s=16, c=color, alpha=0.88, linewidths=0.2, edgecolors="#2A2A2A", label=f"{split} ({len(sub)})", zorder=4)

    draw_scale_bar(ax, ex0, ey0, ex1, ey1)
    ax.set_xlim(ex0, ex1)
    ax.set_ylim(ey0, ey1)
    ax.set_aspect("auto")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_color("#000000")
        sp.set_linewidth(1.1)
    ax.legend(loc="upper right", fontsize=8, frameon=True)

    title = f"{city} | {category} | Scale 1:{scale}"
    fig.suptitle(title, fontsize=16, fontweight="bold", color="#000000", y=0.945)
    draw_north_arrow(fig)
    fig.add_artist(Rectangle((0.01, 0.01), 0.98, 0.98, transform=fig.transFigure, fill=False, edgecolor="#000000", linewidth=1.4))

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)
    return {"city": city, "query_category": category, "target_scale": scale, "count": int(len(group_df)), "png_path": str(out_png)}


def main() -> int:
    args = parse_args()
    in_csv = (PROJECT_ROOT / args.input_csv).resolve()
    out_dir = (PROJECT_ROOT / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_csv)
    needed = {"city", "query_category", "target_scale", "split", "wgs84_lon", "wgs84_lat"}
    miss = needed - set(df.columns)
    if miss:
        raise RuntimeError(f"Missing columns: {sorted(miss)}")

    if args.max_cities > 0:
        keep_cities = sorted(df["city"].dropna().unique().tolist())[: args.max_cities]
        df = df[df["city"].isin(keep_cities)].copy()

    rows: List[Dict[str, Any]] = []
    groups = df.groupby(["target_scale", "city", "query_category"], dropna=False)
    total = len(groups)
    for i, ((scale, city, category), grp) in enumerate(groups, start=1):
        city_slug = slugify(str(city))
        cat_slug = slugify(str(category))
        out_png = out_dir / f"scale_1_{int(scale)}" / city_slug / f"{cat_slug}.png"
        rec = render_one(grp, out_png, dpi=args.dpi)
        if rec["count"] > 0:
            rows.append(rec)
        if i % 100 == 0 or i == total:
            print(f"[progress] {i}/{total}")

    manifest = pd.DataFrame(rows)
    manifest.to_csv(out_dir / "png_manifest.csv", index=False, encoding="utf-8-sig")
    (out_dir / "png_manifest.json").write_text(json.dumps({"rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(str(out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
