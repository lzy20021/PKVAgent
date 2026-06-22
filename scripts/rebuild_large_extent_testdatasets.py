"""
Rebuild TestDatasets scale samples from Datasets/datasets0324.shp with larger extents.

Outputs:
  TestDatasets/Scale_1_86000/sample_01.geojson ... sample_50.geojson
  TestDatasets/Scale_1_324000/sample_01.geojson ... sample_50.geojson
  TestDatasets/Scale_1_625000/sample_01.geojson ... sample_50.geojson
  TestDatasets/rebuild_large_extent_manifest.json
"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ui.point_symbol_core import poi_symbol_for_category


@dataclass(frozen=True)
class ScaleConfig:
    scale: int
    dir_name: str
    width_m: float
    height_m: float
    grid_m: float
    min_count: int


CONFIGS: List[ScaleConfig] = [
    ScaleConfig(scale=86000, dir_name="Scale_1_86000", width_m=22000.0, height_m=18000.0, grid_m=10000.0, min_count=90),
    ScaleConfig(scale=324000, dir_name="Scale_1_324000", width_m=95000.0, height_m=76000.0, grid_m=42000.0, min_count=350),
    ScaleConfig(scale=625000, dir_name="Scale_1_625000", width_m=190000.0, height_m=150000.0, grid_m=80000.0, min_count=550),
]


def _normalize_columns(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    out = gdf.copy()
    if "query_category" not in out.columns:
        if "query_cate" in out.columns:
            out["query_category"] = out["query_cate"]
        elif "type" in out.columns:
            out["query_category"] = out["type"]
        else:
            out["query_category"] = ""
    if "city" not in out.columns and "cityname" in out.columns:
        out["city"] = out["cityname"]
    if "district" not in out.columns and "adname" in out.columns:
        out["district"] = out["adname"]
    if "point_symbol" not in out.columns:
        out["point_symbol"] = out["query_category"].map(poi_symbol_for_category)
    else:
        missing = out["point_symbol"].isna() | (out["point_symbol"].astype(str).str.strip() == "")
        out.loc[missing, "point_symbol"] = out.loc[missing, "query_category"].map(poi_symbol_for_category)
    return out


def load_source(path: Path) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")
    gdf = gdf[~gdf.geometry.isna() & ~gdf.geometry.is_empty].copy()
    gdf = _normalize_columns(gdf)
    keep_cols = [
        c
        for c in [
            "poi_uid",
            "name",
            "type",
            "typecode",
            "address",
            "pname",
            "cityname",
            "adname",
            "query_category",
            "city",
            "district",
            "gcj_lon",
            "gcj_lat",
            "wgs84_lon",
            "wgs84_lat",
            "source",
            "point_symbol",
            "geometry",
        ]
        if c in gdf.columns
    ]
    return gdf[keep_cols].copy()


def build_candidates(metric: gpd.GeoDataFrame, cfg: ScaleConfig) -> pd.DataFrame:
    pts = metric.geometry
    work = pd.DataFrame(
        {
            "mx": pts.x,
            "my": pts.y,
            "cityname": metric["cityname"].astype(str) if "cityname" in metric.columns else "",
        },
        index=metric.index,
    )
    work["gx"] = (work["mx"] / cfg.grid_m).round().astype(int)
    work["gy"] = (work["my"] / cfg.grid_m).round().astype(int)
    cell_counts = work.groupby(["gx", "gy"], as_index=False).size().rename(columns={"size": "seed_count"})
    cell_counts["cx"] = cell_counts["gx"] * cfg.grid_m
    cell_counts["cy"] = cell_counts["gy"] * cfg.grid_m

    rows: List[Dict[str, Any]] = []
    for row in cell_counts.sort_values("seed_count", ascending=False).itertuples(index=False):
        bbox = box(
            row.cx - cfg.width_m / 2.0,
            row.cy - cfg.height_m / 2.0,
            row.cx + cfg.width_m / 2.0,
            row.cy + cfg.height_m / 2.0,
        )
        subset = metric[metric.intersects(bbox)]
        count = int(len(subset))
        if count < cfg.min_count:
            continue
        city_text = ""
        if "cityname" in subset.columns:
            city_mode = subset["cityname"].mode(dropna=True)
            city_text = str(city_mode.iloc[0]) if len(city_mode) else ""
        rows.append(
            {
                "cx": float(row.cx),
                "cy": float(row.cy),
                "seed_count": int(row.seed_count),
                "feature_count": count,
                "cityname": city_text,
            }
        )
    return pd.DataFrame(rows)


def pick_windows(metric: gpd.GeoDataFrame, cfg: ScaleConfig, sample_count: int) -> List[Dict[str, Any]]:
    candidates = build_candidates(metric, cfg)
    if candidates.empty:
        raise RuntimeError(f"{cfg.dir_name}: no candidate windows passed min_count={cfg.min_count}")

    chosen: List[Dict[str, Any]] = []
    min_center_dist = math.hypot(cfg.width_m, cfg.height_m) * 0.52
    for row in candidates.sort_values(["feature_count", "seed_count"], ascending=False).itertuples(index=False):
        if any(math.hypot(row.cx - it["cx"], row.cy - it["cy"]) < min_center_dist for it in chosen):
            continue
        chosen.append({"cx": float(row.cx), "cy": float(row.cy), "feature_count": int(row.feature_count), "cityname": row.cityname})
        if len(chosen) >= sample_count:
            break

    if len(chosen) < sample_count:
        for row in candidates.sort_values(["feature_count", "seed_count"], ascending=False).itertuples(index=False):
            same = any(abs(row.cx - it["cx"]) < 1.0 and abs(row.cy - it["cy"]) < 1.0 for it in chosen)
            if same:
                continue
            chosen.append({"cx": float(row.cx), "cy": float(row.cy), "feature_count": int(row.feature_count), "cityname": row.cityname})
            if len(chosen) >= sample_count:
                break

    if len(chosen) < sample_count:
        raise RuntimeError(f"{cfg.dir_name}: only found {len(chosen)} windows, expected {sample_count}")
    return chosen[:sample_count]


def export_samples(source_wgs84: gpd.GeoDataFrame, metric: gpd.GeoDataFrame, cfg: ScaleConfig, out_dir: Path, sample_count: int) -> List[Dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    windows = pick_windows(metric, cfg, sample_count)
    summaries: List[Dict[str, Any]] = []
    for i, win in enumerate(windows, start=1):
        bbox = box(
            win["cx"] - cfg.width_m / 2.0,
            win["cy"] - cfg.height_m / 2.0,
            win["cx"] + cfg.width_m / 2.0,
            win["cy"] + cfg.height_m / 2.0,
        )
        subset_metric = metric[metric.intersects(bbox)].copy()
        subset = source_wgs84.loc[subset_metric.index].copy()
        subset = subset.reset_index(drop=True)
        sample_path = out_dir / f"sample_{i:02d}.geojson"
        subset.to_file(sample_path, driver="GeoJSON")
        bounds = subset.total_bounds.tolist()
        summaries.append(
            {
                "sample": f"sample_{i:02d}.geojson",
                "count": int(len(subset)),
                "cityname": str(win["cityname"]),
                "bbox_wgs84": [round(float(v), 6) for v in bounds],
                "span_lon": round(float(bounds[2] - bounds[0]), 6),
                "span_lat": round(float(bounds[3] - bounds[1]), 6),
                "window_m": {"width": cfg.width_m, "height": cfg.height_m},
            }
        )
    return summaries


def main() -> None:
    src_path = ROOT / "Datasets" / "datasets0324.shp"
    out_root = ROOT / "TestDatasets"
    sample_count = 50

    source = load_source(src_path)
    metric = source.to_crs("EPSG:3857")

    manifest: Dict[str, Any] = {
        "source": str(src_path),
        "source_crs": "EPSG:4326",
        "metric_crs": "EPSG:3857",
        "sample_count_per_scale": sample_count,
        "scales": {},
    }

    for cfg in CONFIGS:
        summaries = export_samples(source, metric, cfg, out_root / cfg.dir_name, sample_count)
        manifest["scales"][cfg.dir_name] = {
            "scale": cfg.scale,
            "window_m": {"width": cfg.width_m, "height": cfg.height_m},
            "grid_m": cfg.grid_m,
            "min_count": cfg.min_count,
            "samples": summaries,
        }

    manifest_path = out_root / "rebuild_large_extent_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
