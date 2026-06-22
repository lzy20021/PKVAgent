from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shapely.geometry import Polygon

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


SOURCE_SUMMARIES = {
    135000: PROJECT_ROOT / "tests/output/agent_point_generalization_batch_v31/_batch_summary.json",
    220000: PROJECT_ROOT / "tests/output/agent_point_generalization_batch_v31/_batch_summary.json",
    325000: PROJECT_ROOT / "tests/output/agent_point_generalization_batch_v32/_batch_summary.json",
}

OUTPUT_ROOT = PROJECT_ROOT / "tests/output/point_generalization_publication_pngs"
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
        return 88.0 if is_after else 70.0
    if count <= 28:
        return 74.0 if is_after else 58.0
    if count <= 45:
        return 62.0 if is_after else 48.0
    return 50.0 if is_after else 40.0


def projected_center_text(gdf: gpd.GeoDataFrame) -> str:
    metric = gdf.copy()
    if metric.crs is not None and getattr(metric.crs, "is_geographic", False):
        try:
            metric = metric.to_crs(metric.estimate_utm_crs())
        except Exception:
            metric = metric.to_crs("EPSG:3857")
    minx, miny, maxx, maxy = [float(v) for v in metric.total_bounds]
    cx = (minx + maxx) / 2.0
    cy = (miny + maxy) / 2.0
    crs_label = str(metric.crs) if metric.crs is not None else "unknown CRS"
    return f"Center (projected): X={cx:.1f}, Y={cy:.1f} | {crs_label}"


def minimum_bounding_rectangle(gdf: gpd.GeoDataFrame):
    geom = gdf.geometry.dropna()
    if geom.empty:
        return None
    merged = geom.union_all() if hasattr(geom, "union_all") else geom.unary_union
    if merged.is_empty:
        return None
    hull = merged.convex_hull
    rect = hull.minimum_rotated_rectangle
    if rect.is_empty:
        return None
    if rect.geom_type != "Polygon":
        rect = hull.envelope
    if rect.is_empty or rect.geom_type != "Polygon":
        return None
    return rect


def plot_rectangle(ax: Any, rect: Polygon, color: str) -> None:
    x, y = rect.exterior.xy
    ax.plot(x, y, color=color, linewidth=2.0, alpha=0.95, zorder=4)


def render_compare(before_path: Path, after_path: Path, output_path: Path, title: str) -> None:
    before_gdf = gpd.read_file(before_path)
    after_gdf = gpd.read_file(after_path)

    before_rect = minimum_bounding_rectangle(before_gdf)
    after_rect = minimum_bounding_rectangle(after_gdf)

    bounds_candidates = [before_gdf.total_bounds, after_gdf.total_bounds]
    if before_rect is not None:
        bounds_candidates.append(before_rect.bounds)
    if after_rect is not None:
        bounds_candidates.append(after_rect.bounds)
    minx = float(min(b[0] for b in bounds_candidates))
    miny = float(min(b[1] for b in bounds_candidates))
    maxx = float(max(b[2] for b in bounds_candidates))
    maxy = float(max(b[3] for b in bounds_candidates))
    dx = max(1e-9, maxx - minx)
    dy = max(1e-9, maxy - miny)
    pad_x = dx * 0.10
    pad_y = dy * 0.10

    fig, axes = plt.subplots(1, 2, figsize=(15, 8.6), facecolor="#ffffff")
    fig.subplots_adjust(wspace=0.03, left=0.02, right=0.98, top=0.88, bottom=0.05)

    panels: List[Tuple[Any, gpd.GeoDataFrame, str, str, str, str, float, Any]] = [
        (
            axes[0],
            before_gdf,
            "Ungeneralized",
            "#2C5BEA",
            "#1C3FB0",
            "#0A1F6F",
            marker_size(len(before_gdf), is_after=False),
            before_rect,
        ),
        (
            axes[1],
            after_gdf,
            "Generalized",
            "#20C34A",
            "#118533",
            "#0B5B22",
            marker_size(len(after_gdf), is_after=True),
            after_rect,
        ),
    ]

    for ax, gdf, subtitle, color, edge, rect_color, size, rect in panels:
        ax.set_facecolor("#F5F7FA")
        gdf.plot(ax=ax, color=color, markersize=size, edgecolor=edge, linewidth=0.25, alpha=0.95)
        if rect is not None:
            plot_rectangle(ax, rect, rect_color)
        ax.set_xlim(minx - pad_x, maxx + pad_x)
        ax.set_ylim(miny - pad_y, maxy + pad_y)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"{subtitle} ({len(gdf)})", fontsize=15, fontweight="bold", color="#222222", pad=12)
        ax.set_axis_off()

    fig.text(
        0.03,
        0.025,
        projected_center_text(before_gdf),
        ha="left",
        va="bottom",
        fontsize=10.5,
        color="#333333",
    )
    fig.suptitle(title, fontsize=18, fontweight="bold", color="#111111", y=0.955)
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
            picked_ids = {
                (str(row["group_id"]), int(row["target_scale"]))
                for row in picked
            }
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
        out_path = ALL_DIR / f"scale_1_{scale}" / f"{group_id}_publication_compare.png"
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
                "publication_compare_png": str(out_path),
                "source_after_geojson": row["after_geojson"],
                "source_version": "v32" if scale == 325000 else "v31",
            }
        )

    selected_rows = select_rows(rows, per_scale=12)
    manifest_selected: List[Dict[str, Any]] = []
    for row in selected_rows:
        scale = int(row["target_scale"])
        group_id = str(row["group_id"])
        out_path = SELECTED_DIR / f"scale_1_{scale}" / f"{group_id}_publication_compare.png"
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
                "publication_compare_png": str(out_path),
                "source_after_geojson": row["after_geojson"],
                "source_version": "v32" if scale == 325000 else "v31",
            }
        )

    write_manifest(OUTPUT_ROOT / "all_manifest", manifest_all)
    write_manifest(OUTPUT_ROOT / "selected_manifest", manifest_selected)
    readme = [
        "# Publication Point Comparison PNGs",
        "",
        "- 135000 / 220000 使用 v3.1",
        "- 325000 使用 v3.2",
        "- `all/` 包含 300 张亮色大点版对比图",
        "- `selected/` 包含 36 张精选图，每个比例尺 12 张",
        "",
        "配色说明：",
        "- 左图未综合：亮青色",
        "- 右图综合后：亮玫红",
        "- 为 PPT / 论文展示增大了点符号尺寸",
    ]
    (OUTPUT_ROOT / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    print(OUTPUT_ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
