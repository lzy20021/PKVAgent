from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import geopandas as gpd
import matplotlib
from pyproj import CRS, Transformer
from matplotlib.patches import FancyArrowPatch, Rectangle

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


SOURCE_SUMMARIES = {
    135000: PROJECT_ROOT / "tests/output/agent_point_generalization_batch_v31/_batch_summary.json",
    220000: PROJECT_ROOT / "tests/output/agent_point_generalization_batch_v31/_batch_summary.json",
    325000: PROJECT_ROOT / "tests/output/agent_point_generalization_batch_v32/_batch_summary.json",
}

OUTPUT_ROOT = PROJECT_ROOT / "tests/output/point_generalization_standard_grid_pngs"
ALL_DIR = OUTPUT_ROOT / "all"
SELECTED_DIR = OUTPUT_ROOT / "selected"


def load_rows(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload.get("rows", []))


def build_hybrid_rows() -> List[Dict[str, Any]]:
    loaded = {scale: load_rows(path) for scale, path in SOURCE_SUMMARIES.items()}
    rows: List[Dict[str, Any]] = []
    for scale, entries in loaded.items():
        rows.extend([row for row in entries if int(row["target_scale"]) == scale])
    rows.sort(key=lambda row: (int(row["target_scale"]), str(row["city"]), str(row["region_name"]), str(row["keyword"])))
    return rows


def marker_size(count: int, is_after: bool) -> float:
    if count <= 16:
        return 70.0 if is_after else 60.0
    if count <= 28:
        return 62.0 if is_after else 52.0
    if count <= 45:
        return 54.0 if is_after else 44.0
    return 46.0 if is_after else 38.0


def projected_center_text(gdf: gpd.GeoDataFrame) -> str:
    minx, miny, maxx, maxy = [float(v) for v in gdf.total_bounds]
    cx = (minx + maxx) / 2.0
    cy = (miny + maxy) / 2.0
    crs_label = str(gdf.crs) if gdf.crs is not None else "unknown CRS"
    return f"Center (projected): X={cx:.1f}, Y={cy:.1f} | {crs_label}"


def ensure_projected_pair(before_gdf: gpd.GeoDataFrame, after_gdf: gpd.GeoDataFrame) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    if before_gdf.crs is None:
        before_gdf = before_gdf.set_crs("EPSG:4326")
    if after_gdf.crs is None:
        after_gdf = after_gdf.set_crs(before_gdf.crs)
    if after_gdf.crs != before_gdf.crs:
        after_gdf = after_gdf.to_crs(before_gdf.crs)

    crs = CRS.from_user_input(before_gdf.crs)
    if crs.is_geographic:
        try:
            target = before_gdf.estimate_utm_crs()
        except Exception:
            target = "EPSG:3857"
        before_gdf = before_gdf.to_crs(target)
        after_gdf = after_gdf.to_crs(target)
    return before_gdf, after_gdf


def draw_standard_north_arrow(fig: Any) -> None:
    x = 0.962
    y0 = 0.845
    y1 = 0.915
    fig.text(x, y1 + 0.018, "N", ha="center", va="bottom", fontsize=15, fontweight="bold", color="#000000")
    arrow = FancyArrowPatch(
        (x, y0),
        (x, y1),
        transform=fig.transFigure,
        arrowstyle="-|>",
        mutation_scale=18,
        lw=1.9,
        color="#000000",
    )
    fig.add_artist(arrow)


def style_axis_frame(ax: Any) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)
        spine.set_color("#000000")


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
    out: List[float] = []
    x = math.ceil(start / step) * step
    while x <= end + 1e-12:
        out.append(round(x, 8))
        x += step
    return out


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
    x_in = minx + width * 0.012
    y_in = miny + height * 0.018

    for lon in lon_lines:
        lat_samples = np.linspace(lat_min, lat_max, 64)
        lon_samples = np.full_like(lat_samples, lon)
        x, y = to_xy.transform(lon_samples, lat_samples)
        ax.plot(x, y, color="#B5B5B5", linewidth=0.55, linestyle="--", alpha=0.8, zorder=1)

        tx, ty = to_xy.transform(lon, lat_min)
        if minx <= tx <= maxx:
            ax.text(tx, y_in, f"{lon:.2f}°E", ha="center", va="bottom", fontsize=7.6, color="#555555", clip_on=True, zorder=5)

    for lat in lat_lines:
        lon_samples = np.linspace(lon_min, lon_max, 64)
        lat_samples = np.full_like(lon_samples, lat)
        x, y = to_xy.transform(lon_samples, lat_samples)
        ax.plot(x, y, color="#B5B5B5", linewidth=0.55, linestyle="--", alpha=0.8, zorder=1)

        tx, ty = to_xy.transform(lon_min, lat)
        if miny <= ty <= maxy:
            ax.text(x_in, ty, f"{lat:.2f}°N", ha="left", va="center", fontsize=7.6, color="#555555", clip_on=True, zorder=5)


def choose_scale_bar_length(target_len: float) -> float:
    candidates = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]
    chosen = candidates[0]
    for c in candidates:
        if c <= target_len:
            chosen = c
        else:
            break
    return float(chosen)


def draw_scale_bar(ax: Any, minx: float, miny: float, maxx: float, maxy: float) -> None:
    width = maxx - minx
    height = maxy - miny
    bar_len = choose_scale_bar_length(width * 0.22)
    seg = bar_len / 4.0
    x0 = minx + width * 0.06
    y0 = miny + height * 0.05
    h = height * 0.012

    for i in range(4):
        xi = x0 + i * seg
        color = "#000000" if i % 2 == 0 else "#FFFFFF"
        ax.add_patch(Rectangle((xi, y0), seg, h, facecolor=color, edgecolor="#000000", linewidth=0.8, zorder=6))
    ax.add_patch(Rectangle((x0, y0), bar_len, h, fill=False, edgecolor="#000000", linewidth=0.8, zorder=6))

    label_end = f"{int(bar_len/1000)} km" if bar_len >= 1000 else f"{int(bar_len)} m"
    ax.text(x0, y0 - h * 0.9, "0", ha="center", va="top", fontsize=8.2, color="#111111", zorder=7)
    ax.text(x0 + bar_len / 2, y0 - h * 0.9, label_end if bar_len < 2000 else f"{bar_len/2000:.1f} km", ha="center", va="top", fontsize=8.2, color="#111111", zorder=7)
    ax.text(x0 + bar_len, y0 - h * 0.9, label_end, ha="center", va="top", fontsize=8.2, color="#111111", zorder=7)
    ax.text(x0 + bar_len / 2, y0 + h * 1.8, "Scale Bar", ha="center", va="bottom", fontsize=8.0, color="#111111", zorder=7)


def render_compare(before_path: Path, after_path: Path, output_path: Path, title: str) -> None:
    before_gdf = gpd.read_file(before_path)
    after_gdf = gpd.read_file(after_path)
    before_gdf, after_gdf = ensure_projected_pair(before_gdf, after_gdf)

    b0 = before_gdf.total_bounds
    b1 = after_gdf.total_bounds
    minx = float(min(b0[0], b1[0]))
    miny = float(min(b0[1], b1[1]))
    maxx = float(max(b0[2], b1[2]))
    maxy = float(max(b0[3], b1[3]))
    dx = max(1e-9, maxx - minx)
    dy = max(1e-9, maxy - miny)
    pad_x = dx * 0.10
    pad_y = dy * 0.10
    ex0, ey0, ex1, ey1 = minx - pad_x, miny - pad_y, maxx + pad_x, maxy + pad_y

    fig, axes = plt.subplots(1, 2, figsize=(15.8, 9.0), facecolor="#FFFFFF")
    fig.subplots_adjust(wspace=0.05, left=0.05, right=0.95, top=0.88, bottom=0.08)

    crs_obj = CRS.from_user_input(before_gdf.crs)
    panels: List[Tuple[Any, gpd.GeoDataFrame, str, str, str, float]] = [
        (axes[0], before_gdf, "Ungeneralized", "#2A5CAA", "#1E3E70", marker_size(len(before_gdf), is_after=False)),
        (axes[1], after_gdf, "Generalized", "#2C9C59", "#1F6C3D", marker_size(len(after_gdf), is_after=True)),
    ]

    for ax, gdf, subtitle, color, edge, size in panels:
        ax.set_facecolor("#FFFFFF")
        draw_graticule(ax, crs_obj, ex0, ey0, ex1, ey1)
        gdf.plot(ax=ax, color=color, markersize=size, edgecolor=edge, linewidth=0.25, alpha=0.95, zorder=4)
        draw_scale_bar(ax, ex0, ey0, ex1, ey1)
        ax.set_xlim(ex0, ex1)
        ax.set_ylim(ey0, ey1)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"{subtitle} ({len(gdf)})", fontsize=15, fontweight="bold", color="#111111", pad=10)
        style_axis_frame(ax)

    draw_standard_north_arrow(fig)
    fig.add_artist(
        Rectangle(
            (0.01, 0.01),
            0.98,
            0.98,
            transform=fig.transFigure,
            fill=False,
            edgecolor="#000000",
            linewidth=1.4,
        )
    )
    fig.text(0.03, 0.024, projected_center_text(before_gdf), ha="left", va="bottom", fontsize=10.0, color="#222222")
    fig.suptitle(title, fontsize=18, fontweight="bold", color="#000000", y=0.955)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=600, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def select_rows(rows: List[Dict[str, Any]], per_scale: int = 12) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    for scale in sorted({int(row["target_scale"]) for row in rows}):
        scale_rows = [row for row in rows if int(row["target_scale"]) == scale]
        scale_rows.sort(
            key=lambda row: (
                0 if str(row.get("completion_status", "")).startswith("completed") else 1,
                str(row["city"]),
                str(row["region_name"]),
                str(row["keyword"]),
            )
        )
        used_city = set()
        used_keyword = set()
        picked: List[Dict[str, Any]] = []
        for row in scale_rows:
            city = str(row["city"])
            keyword = str(row["keyword"])
            if city in used_city and keyword in used_keyword:
                continue
            picked.append(row)
            used_city.add(city)
            used_keyword.add(keyword)
            if len(picked) >= per_scale:
                break
        if len(picked) < per_scale:
            picked_ids = {(str(row["group_id"]), int(row["target_scale"])) for row in picked}
            for row in scale_rows:
                key = (str(row["group_id"]), int(row["target_scale"]))
                if key in picked_ids:
                    continue
                picked.append(row)
                if len(picked) >= per_scale:
                    break
        selected.extend(picked)
    return selected


def write_manifest(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    (path.with_suffix(".json")).write_text(
        json.dumps({"rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with path.with_suffix(".csv").open("w", encoding="utf-8-sig", newline="") as f:
        fieldnames = list(rows[0].keys()) if rows else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def main() -> int:
    rows = build_hybrid_rows()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    manifest_all: List[Dict[str, Any]] = []
    for row in rows:
        scale = int(row["target_scale"])
        group_id = str(row["group_id"])
        out_path = ALL_DIR / f"scale_1_{scale}" / f"{group_id}_standard_grid_compare.png"
        title = f"Study Area: {row['city']} {row['region_name']} {row['keyword']} | Scale 1:{scale}"
        render_compare(Path(row["before_geojson"]), Path(row["after_geojson"]), out_path, title)
        manifest_all.append(
            {
                "group_id": group_id,
                "city": row["city"],
                "region_name": row["region_name"],
                "keyword": row["keyword"],
                "target_scale": scale,
                "before_count": int(row["before_count"]),
                "after_count": int(row["after_count"]),
                "completion_status": row["completion_status"],
                "standard_grid_compare_png": str(out_path),
                "source_after_geojson": row["after_geojson"],
                "source_version": "v32" if scale == 325000 else "v31",
            }
        )

    selected_rows = select_rows(rows, per_scale=12)
    manifest_selected: List[Dict[str, Any]] = []
    for row in selected_rows:
        scale = int(row["target_scale"])
        group_id = str(row["group_id"])
        out_path = SELECTED_DIR / f"scale_1_{scale}" / f"{group_id}_standard_grid_compare.png"
        title = f"Study Area: {row['city']} {row['region_name']} {row['keyword']} | Scale 1:{scale}"
        render_compare(Path(row["before_geojson"]), Path(row["after_geojson"]), out_path, title)
        manifest_selected.append(
            {
                "group_id": group_id,
                "city": row["city"],
                "region_name": row["region_name"],
                "keyword": row["keyword"],
                "target_scale": scale,
                "before_count": int(row["before_count"]),
                "after_count": int(row["after_count"]),
                "completion_status": row["completion_status"],
                "standard_grid_compare_png": str(out_path),
                "source_after_geojson": row["after_geojson"],
                "source_version": "v32" if scale == 325000 else "v31",
            }
        )

    write_manifest(OUTPUT_ROOT / "all_manifest", manifest_all)
    write_manifest(OUTPUT_ROOT / "selected_manifest", manifest_selected)
    readme = [
        "# Standard Point Comparison PNGs (Scale Bar + Graticule)",
        "",
        "- 135000 / 220000 use v31",
        "- 325000 use v32",
        "- No minimum bounding rectangle",
        "- Includes north arrow, map frame, scale bar, and graticule labels",
    ]
    (OUTPUT_ROOT / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    print(OUTPUT_ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
