from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import geopandas as gpd
import matplotlib
import pandas as pd
import requests
from shapely.geometry import Point, Polygon, box

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.io.amap_poi import gcj02_to_wgs84, slugify

matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


AMAP_DISTRICT_URL = "https://restapi.amap.com/v3/config/district"
AMAP_POI_POLYGON_URL = "https://restapi.amap.com/v5/place/polygon"
AMAP_GEOCODE_URL = "https://restapi.amap.com/v3/geocode/geo"
POI_CATEGORIES = ["便利店", "停车场", "咖啡厅", "酒店", "餐厅", "医院", "学校", "地铁站", "公交站", "银行"]
TARGET_SCALES = [135000, 220000, 325000]
SCALE_SAMPLE_RATIO = {135000: 0.52, 220000: 0.36, 325000: 0.24}


CITY_TIERS: Dict[str, List[str]] = {
    "超一线": ["北京", "上海", "广州", "深圳"],
    "新一线": [
        "成都",
        "重庆",
        "杭州",
        "武汉",
        "西安",
        "苏州",
        "郑州",
        "南京",
        "天津",
        "长沙",
        "东莞",
        "宁波",
        "佛山",
        "合肥",
        "青岛",
    ],
    "二线": [
        "沈阳",
        "大连",
        "厦门",
        "福州",
        "济南",
        "昆明",
        "南宁",
        "哈尔滨",
        "长春",
        "南昌",
        "贵阳",
        "太原",
        "石家庄",
        "乌鲁木齐",
        "兰州",
        "海口",
        "珠海",
        "无锡",
        "温州",
        "徐州",
    ],
    "三线": [
        "常州",
        "嘉兴",
        "绍兴",
        "金华",
        "台州",
        "泉州",
        "惠州",
        "中山",
        "汕头",
        "烟台",
        "潍坊",
        "临沂",
        "洛阳",
        "南通",
        "盐城",
        "唐山",
        "保定",
        "廊坊",
        "宜昌",
        "襄阳",
        "芜湖",
        "赣州",
        "九江",
        "柳州",
        "桂林",
    ],
}


@dataclass
class CityRecord:
    city: str
    tier: str


@dataclass
class DistrictRecord:
    city: str
    tier: str
    district_name: str
    adcode: str
    level: str
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float
    source: str

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        return (self.min_lon, self.min_lat, self.max_lon, self.max_lat)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build large-scale AMap POI dataset for point generalization.")
    parser.add_argument("--api-key", required=True, help="AMap Web Service API key")
    parser.add_argument("--output-dir", default="amap_poi_dataset", help="Output dataset directory")
    parser.add_argument("--min-cities", type=int, default=50, help="Minimum selected cities")
    parser.add_argument("--districts-per-city", type=int, default=10, help="Minimum districts per city")
    parser.add_argument("--max-pages", type=int, default=2, help="Max pages per city-district-category query")
    parser.add_argument("--page-size", type=int, default=25, help="AMap page size")
    parser.add_argument("--sleep-seconds", type=float, default=0.08, help="Sleep between API calls")
    parser.add_argument("--seed", type=int, default=20260316, help="Random seed")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    parser.add_argument("--min-poi-per-group", type=int, default=10, help="Expected minimum POI per group")
    return parser.parse_args()


class AMapClient:
    def __init__(self, api_key: str, page_size: int, sleep_seconds: float):
        self.api_key = api_key
        self.page_size = page_size
        self.sleep_seconds = sleep_seconds
        self.session = requests.Session()
        self.request_count = 0

    def _get(self, url: str, params: Dict[str, Any]) -> Dict[str, Any]:
        params = dict(params)
        params["key"] = self.api_key
        resp = self.session.get(url, params=params, timeout=45)
        self.request_count += 1
        resp.raise_for_status()
        payload = resp.json()
        if str(payload.get("status", "0")) != "1":
            raise RuntimeError(f"AMap API error: {payload.get('info', 'unknown')}")
        if self.sleep_seconds > 0:
            time.sleep(self.sleep_seconds)
        return payload

    def fetch_city_tree(self, city_name: str) -> Dict[str, Any]:
        return self._get(
            AMAP_DISTRICT_URL,
            {"keywords": city_name, "subdistrict": 3, "extensions": "all", "filter": "", "offset": 1, "page": 1},
        )

    def fetch_pois_by_bbox(self, keyword: str, bbox: Tuple[float, float, float, float], max_pages: int) -> List[Dict[str, Any]]:
        polygon = f"{bbox[0]},{bbox[1]}|{bbox[2]},{bbox[3]}"
        dedup: Dict[str, Dict[str, Any]] = {}
        for page_num in range(1, max_pages + 1):
            payload = self._get(
                AMAP_POI_POLYGON_URL,
                {"keywords": keyword, "polygon": polygon, "page_num": page_num, "page_size": self.page_size},
            )
            pois = payload.get("pois") or []
            if not pois:
                break
            for poi in pois:
                pid = poi.get("id") or f"{poi.get('name','unknown')}@{poi.get('location','')}"
                dedup[pid] = poi
        return list(dedup.values())

    def fetch_geocode(self, city_name: str) -> Optional[Tuple[float, float]]:
        payload = self._get(AMAP_GEOCODE_URL, {"address": city_name, "city": city_name})
        geocodes = payload.get("geocodes") or []
        if not geocodes:
            return None
        location = str(geocodes[0].get("location", ""))
        if "," not in location:
            return None
        lon, lat = location.split(",", 1)
        return (float(lon), float(lat))


def parse_polyline_bbox(polyline: str) -> Optional[Tuple[float, float, float, float]]:
    if not polyline:
        return None
    min_lon, min_lat = 180.0, 90.0
    max_lon, max_lat = -180.0, -90.0
    found = False
    for ring in polyline.split("|"):
        for pair in ring.split(";"):
            if "," not in pair:
                continue
            try:
                lon, lat = pair.split(",", 1)
                lon_f = float(lon)
                lat_f = float(lat)
            except Exception:
                continue
            min_lon = min(min_lon, lon_f)
            min_lat = min(min_lat, lat_f)
            max_lon = max(max_lon, lon_f)
            max_lat = max(max_lat, lat_f)
            found = True
    if not found:
        return None
    if min_lon >= max_lon or min_lat >= max_lat:
        return None
    return (min_lon, min_lat, max_lon, max_lat)


def center_to_bbox(center: str, half_size_deg: float = 0.03) -> Optional[Tuple[float, float, float, float]]:
    if not center or "," not in center:
        return None
    try:
        lon, lat = center.split(",", 1)
        lon_f = float(lon)
        lat_f = float(lat)
    except Exception:
        return None
    return (lon_f - half_size_deg, lat_f - half_size_deg, lon_f + half_size_deg, lat_f + half_size_deg)


def select_cities(min_cities: int) -> List[CityRecord]:
    tier_order = ["超一线", "新一线", "二线", "三线"]
    quotas = {"超一线": 4, "新一线": 15, "二线": 16, "三线": 15}
    target = max(50, min_cities)
    selected: List[CityRecord] = []
    for tier in tier_order:
        for city in CITY_TIERS[tier][: quotas[tier]]:
            selected.append(CityRecord(city=city, tier=tier))
    if len(selected) < target:
        for tier in tier_order:
            already = {r.city for r in selected}
            for city in CITY_TIERS[tier]:
                if city in already:
                    continue
                selected.append(CityRecord(city=city, tier=tier))
                if len(selected) >= target:
                    break
            if len(selected) >= target:
                break
    return selected[:target]


def flatten_districts(root: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    def walk(node: Dict[str, Any]) -> None:
        children = node.get("districts") or []
        for child in children:
            out.append(child)
            walk(child)

    walk(root)
    return out


def pick_districts_for_city(city: CityRecord, payload: Dict[str, Any], min_count: int) -> List[DistrictRecord]:
    districts_top = payload.get("districts") or []
    if not districts_top:
        return []
    root = districts_top[0]
    all_children = flatten_districts(root)
    candidates = [d for d in all_children if d.get("level") in {"district", "street", "township"}]
    records: List[DistrictRecord] = []
    used_name = set()
    for d in candidates:
        name = str(d.get("name", "")).strip()
        if not name or name in used_name:
            continue
        bbox = parse_polyline_bbox(str(d.get("polyline", ""))) or center_to_bbox(str(d.get("center", "")))
        if bbox is None:
            continue
        used_name.add(name)
        records.append(
            DistrictRecord(
                city=city.city,
                tier=city.tier,
                district_name=name,
                adcode=str(d.get("adcode", "")),
                level=str(d.get("level", "")),
                min_lon=bbox[0],
                min_lat=bbox[1],
                max_lon=bbox[2],
                max_lat=bbox[3],
                source="admin",
            )
        )
    records.sort(key=lambda r: (0 if r.level == "district" else 1, r.district_name))
    if len(records) >= min_count:
        return records[:min_count]

    root_bbox = parse_polyline_bbox(str(root.get("polyline", ""))) or center_to_bbox(str(root.get("center", "")), 0.12)
    if root_bbox is not None:
        need = min_count - len(records)
        min_lon, min_lat, max_lon, max_lat = root_bbox
        cols = int(math.ceil(math.sqrt(need)))
        rows = int(math.ceil(need / cols))
        dx = (max_lon - min_lon) / max(cols, 1)
        dy = (max_lat - min_lat) / max(rows, 1)
        idx = 0
        for r in range(rows):
            for c in range(cols):
                if idx >= need:
                    break
                x0 = min_lon + c * dx
                y0 = min_lat + r * dy
                x1 = x0 + dx
                y1 = y0 + dy
                records.append(
                    DistrictRecord(
                        city=city.city,
                        tier=city.tier,
                        district_name=f"网格城区{idx+1:02d}",
                        adcode=str(root.get("adcode", "")),
                        level="grid",
                        min_lon=x0,
                        min_lat=y0,
                        max_lon=x1,
                        max_lat=y1,
                        source="grid_fill",
                    )
                )
                idx += 1
    return records[:min_count]


def build_grid_districts(city: CityRecord, center_lon: float, center_lat: float, min_count: int) -> List[DistrictRecord]:
    span_lon = 0.30
    span_lat = 0.24
    min_lon = center_lon - span_lon / 2
    max_lon = center_lon + span_lon / 2
    min_lat = center_lat - span_lat / 2
    max_lat = center_lat + span_lat / 2
    cols = int(math.ceil(math.sqrt(min_count)))
    rows = int(math.ceil(min_count / cols))
    dx = (max_lon - min_lon) / max(cols, 1)
    dy = (max_lat - min_lat) / max(rows, 1)
    records: List[DistrictRecord] = []
    idx = 0
    for r in range(rows):
        for c in range(cols):
            if idx >= min_count:
                break
            x0 = min_lon + c * dx
            y0 = min_lat + r * dy
            x1 = x0 + dx
            y1 = y0 + dy
            records.append(
                DistrictRecord(
                    city=city.city,
                    tier=city.tier,
                    district_name=f"网格城区{idx+1:02d}",
                    adcode="",
                    level="grid",
                    min_lon=x0,
                    min_lat=y0,
                    max_lon=x1,
                    max_lat=y1,
                    source="city_grid_fallback",
                )
            )
            idx += 1
    return records


def poi_to_row(
    city: CityRecord,
    district: DistrictRecord,
    category: str,
    poi: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    location = str(poi.get("location", ""))
    if "," not in location:
        return None
    try:
        gcj_lon, gcj_lat = [float(v) for v in location.split(",", 1)]
    except Exception:
        return None
    wgs_lon, wgs_lat = gcj02_to_wgs84(gcj_lon, gcj_lat)
    return {
        "poi_uid": poi.get("id") or "",
        "name": poi.get("name") or "",
        "type": poi.get("type") or "",
        "typecode": poi.get("typecode") or "",
        "address": poi.get("address") or "",
        "pname": poi.get("pname") or "",
        "cityname": poi.get("cityname") or "",
        "adname": poi.get("adname") or "",
        "business_area": poi.get("business_area") or "",
        "tel": poi.get("tel") or "",
        "rating": poi.get("rating") or "",
        "cost": poi.get("cost") or "",
        "query_category": category,
        "tier": city.tier,
        "city": city.city,
        "district": district.district_name,
        "district_level": district.level,
        "district_source": district.source,
        "district_adcode": district.adcode,
        "bbox_min_lon": district.min_lon,
        "bbox_min_lat": district.min_lat,
        "bbox_max_lon": district.max_lon,
        "bbox_max_lat": district.max_lat,
        "gcj_lon": gcj_lon,
        "gcj_lat": gcj_lat,
        "wgs84_lon": wgs_lon,
        "wgs84_lat": wgs_lat,
    }


def clean_dedup_rows(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["poi_uid"] = df["poi_uid"].fillna("")
    df["coord_key"] = df["wgs84_lon"].round(6).astype(str) + "," + df["wgs84_lat"].round(6).astype(str)
    df["dedup_key"] = df["poi_uid"]
    empty_uid = df["dedup_key"] == ""
    df.loc[empty_uid, "dedup_key"] = (
        df.loc[empty_uid, "name"].astype(str)
        + "|"
        + df.loc[empty_uid, "query_category"].astype(str)
        + "|"
        + df.loc[empty_uid, "city"].astype(str)
        + "|"
        + df.loc[empty_uid, "coord_key"].astype(str)
    )
    df = df.drop_duplicates(subset=["dedup_key"]).copy()
    df = df.drop(columns=["coord_key", "dedup_key"])
    return df


def sample_by_scale(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    sampled_parts: List[pd.DataFrame] = []
    rng = random.Random(seed)
    for scale in TARGET_SCALES:
        ratio = SCALE_SAMPLE_RATIO[scale]
        parts: List[pd.DataFrame] = []
        for _, grp in df.groupby(["tier", "city", "district", "query_category"], dropna=False):
            k = max(1, int(round(len(grp) * ratio)))
            k = min(k, len(grp))
            idx = list(grp.index)
            rng.shuffle(idx)
            pick = grp.loc[idx[:k]].copy()
            pick["target_scale"] = scale
            parts.append(pick)
        sampled_parts.append(pd.concat(parts, ignore_index=True))
    return pd.concat(sampled_parts, ignore_index=True)


def split_dataset(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    if df.empty:
        return df
    out_parts: List[pd.DataFrame] = []
    rng = random.Random(seed)
    for _, grp in df.groupby(["target_scale", "query_category"], dropna=False):
        n = len(grp)
        idx = list(grp.index)
        rng.shuffle(idx)
        n_train = int(round(n * 0.7))
        n_val = int(round(n * 0.15))
        n_test = n - n_train - n_val
        train_idx = idx[:n_train]
        val_idx = idx[n_train : n_train + n_val]
        test_idx = idx[n_train + n_val : n_train + n_val + n_test]
        train = grp.loc[train_idx].copy()
        val = grp.loc[val_idx].copy()
        test = grp.loc[test_idx].copy()
        train["split"] = "train"
        val["split"] = "val"
        test["split"] = "test"
        out_parts.extend([train, val, test])
    return pd.concat(out_parts, ignore_index=True)


def to_geojson(df: pd.DataFrame, path: Path) -> None:
    gdf = gpd.GeoDataFrame(df.copy(), geometry=gpd.points_from_xy(df["wgs84_lon"], df["wgs84_lat"]), crs="EPSG:4326")
    path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(path, driver="GeoJSON", encoding="utf-8")


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)


def plot_stats(category_stats: pd.DataFrame, city_stats: pd.DataFrame, tier_stats: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 4.8))
    cat_sorted = category_stats.sort_values("count", ascending=False)
    plt.bar(cat_sorted["query_category"], cat_sorted["count"], color="#4A7BD0")
    plt.xticks(rotation=25)
    plt.ylabel("Sample count")
    plt.title("POI Sample Count by Category")
    plt.tight_layout()
    plt.savefig(out_dir / "category_counts.png", dpi=220)
    plt.close()

    plt.figure(figsize=(11, 5.2))
    top_city = city_stats.sort_values("count", ascending=False).head(25)
    plt.bar(top_city["city"], top_city["count"], color="#3DAA70")
    plt.xticks(rotation=55, ha="right")
    plt.ylabel("Sample count")
    plt.title("City Coverage (Top 25 by count)")
    plt.tight_layout()
    plt.savefig(out_dir / "city_coverage_top25.png", dpi=220)
    plt.close()

    plt.figure(figsize=(6.5, 4.2))
    plt.bar(tier_stats["tier"], tier_stats["city_count"], color="#9C6BCF")
    plt.ylabel("City count")
    plt.title("City Count by Tier")
    plt.tight_layout()
    plt.savefig(out_dir / "tier_city_counts.png", dpi=220)
    plt.close()


def plot_qc_maps(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for scale in TARGET_SCALES:
        sub = df[df["target_scale"] == scale]
        if sub.empty:
            continue
        plt.figure(figsize=(10.5, 6.5))
        for category, grp in sub.groupby("query_category"):
            plt.scatter(grp["wgs84_lon"], grp["wgs84_lat"], s=4, alpha=0.5, label=category)
        plt.xlabel("Longitude (WGS84)")
        plt.ylabel("Latitude (WGS84)")
        plt.title(f"QC Scatter Map by Category | 1:{scale}")
        plt.legend(loc="upper right", ncol=2, fontsize=7, markerscale=2)
        plt.tight_layout()
        plt.savefig(out_dir / f"qc_scatter_scale_1_{scale}.png", dpi=220)
        plt.close()


def district_records_to_geojson(records: Sequence[DistrictRecord], path: Path) -> None:
    rows: List[Dict[str, Any]] = []
    geoms: List[Polygon] = []
    for r in records:
        rows.append(
            {
                "city": r.city,
                "tier": r.tier,
                "district": r.district_name,
                "adcode": r.adcode,
                "level": r.level,
                "source": r.source,
            }
        )
        geoms.append(box(r.min_lon, r.min_lat, r.max_lon, r.max_lat))
    gdf = gpd.GeoDataFrame(rows, geometry=geoms, crs="EPSG:4326")
    path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(path, driver="GeoJSON", encoding="utf-8")


def ensure_output_layout(root: Path, overwrite: bool) -> None:
    if root.exists() and overwrite:
        for child in root.glob("**/*"):
            if child.is_file():
                child.unlink()
    root.mkdir(parents=True, exist_ok=True)
    (root / "raw").mkdir(parents=True, exist_ok=True)
    (root / "sampled").mkdir(parents=True, exist_ok=True)
    (root / "splits").mkdir(parents=True, exist_ok=True)
    (root / "reports").mkdir(parents=True, exist_ok=True)
    (root / "qc").mkdir(parents=True, exist_ok=True)


def main() -> int:
    args = parse_args()
    random.seed(args.seed)

    out_root = (PROJECT_ROOT / args.output_dir).resolve()
    ensure_output_layout(out_root, overwrite=args.overwrite)

    client = AMapClient(api_key=args.api_key, page_size=args.page_size, sleep_seconds=args.sleep_seconds)
    selected_cities = select_cities(args.min_cities)

    city_rows = [{"city": c.city, "tier": c.tier} for c in selected_cities]
    pd.DataFrame(city_rows).to_csv(out_root / "reports" / "selected_cities.csv", index=False, encoding="utf-8-sig")

    districts_all: List[DistrictRecord] = []
    district_fetch_errors: List[Dict[str, str]] = []
    for city in selected_cities:
        try:
            payload = client.fetch_city_tree(city.city)
            records = pick_districts_for_city(city, payload, args.districts_per_city)
            if len(records) < args.districts_per_city:
                geo = client.fetch_geocode(city.city)
                if geo is not None:
                    extra = build_grid_districts(city, geo[0], geo[1], args.districts_per_city - len(records))
                    records.extend(extra)
            if len(records) < args.districts_per_city:
                district_fetch_errors.append({"city": city.city, "error": f"district_count={len(records)} (after fallback)"})
            districts_all.extend(records)
        except Exception as exc:
            try:
                geo = client.fetch_geocode(city.city)
                if geo is not None:
                    records = build_grid_districts(city, geo[0], geo[1], args.districts_per_city)
                    districts_all.extend(records)
                    district_fetch_errors.append({"city": city.city, "error": f"district_api_failed; used_grid_fallback: {exc}"})
                    continue
            except Exception:
                pass
            district_fetch_errors.append({"city": city.city, "error": str(exc)})

    district_df = pd.DataFrame(
        [
            {
                "city": d.city,
                "tier": d.tier,
                "district": d.district_name,
                "adcode": d.adcode,
                "level": d.level,
                "source": d.source,
                "min_lon": d.min_lon,
                "min_lat": d.min_lat,
                "max_lon": d.max_lon,
                "max_lat": d.max_lat,
            }
            for d in districts_all
        ]
    )
    save_csv(district_df, out_root / "reports" / "city_districts.csv")
    district_records_to_geojson(districts_all, out_root / "reports" / "city_districts.geojson")
    if district_fetch_errors:
        save_csv(pd.DataFrame(district_fetch_errors), out_root / "reports" / "district_fetch_errors.csv")

    city_to_districts: Dict[str, List[DistrictRecord]] = {}
    for d in districts_all:
        city_to_districts.setdefault(d.city, []).append(d)

    poi_rows: List[Dict[str, Any]] = []
    group_logs: List[Dict[str, Any]] = []
    total_groups = len(selected_cities) * args.districts_per_city * len(POI_CATEGORIES)
    current = 0

    for city in selected_cities:
        districts = city_to_districts.get(city.city, [])[: args.districts_per_city]
        for district in districts:
            for category in POI_CATEGORIES:
                current += 1
                try:
                    pois = client.fetch_pois_by_bbox(category, district.bbox, max_pages=args.max_pages)
                    before = len(pois)
                    row_count = 0
                    for poi in pois:
                        row = poi_to_row(city, district, category, poi)
                        if row is not None:
                            poi_rows.append(row)
                            row_count += 1
                    status = "ok" if row_count >= args.min_poi_per_group else "underfilled"
                    group_logs.append(
                        {
                            "city": city.city,
                            "tier": city.tier,
                            "district": district.district_name,
                            "category": category,
                            "raw_poi_count": before,
                            "valid_poi_count": row_count,
                            "status": status,
                        }
                    )
                except Exception as exc:
                    group_logs.append(
                        {
                            "city": city.city,
                            "tier": city.tier,
                            "district": district.district_name,
                            "category": category,
                            "raw_poi_count": 0,
                            "valid_poi_count": 0,
                            "status": "error",
                            "error": str(exc),
                        }
                    )
                if current % 200 == 0:
                    print(f"[progress] groups {current}/{total_groups} api_requests={client.request_count}")

    raw_df = clean_dedup_rows(poi_rows)
    save_csv(pd.DataFrame(group_logs), out_root / "reports" / "group_fetch_log.csv")
    save_csv(raw_df, out_root / "raw" / "all_poi.csv")
    to_geojson(raw_df, out_root / "raw" / "all_poi.geojson")

    sampled_df = sample_by_scale(raw_df, seed=args.seed)
    sampled_df = split_dataset(sampled_df, seed=args.seed)
    save_csv(sampled_df, out_root / "sampled" / "all_scales_sampled.csv")
    to_geojson(sampled_df, out_root / "sampled" / "all_scales_sampled.geojson")

    for split_name in ["train", "val", "test"]:
        split_df = sampled_df[sampled_df["split"] == split_name].copy()
        save_csv(split_df, out_root / "splits" / f"{split_name}.csv")
        to_geojson(split_df, out_root / "splits" / f"{split_name}.geojson")

    category_stats = sampled_df.groupby("query_category", as_index=False).size().rename(columns={"size": "count"})
    city_stats = sampled_df.groupby(["tier", "city"], as_index=False).size().rename(columns={"size": "count"})
    tier_stats = city_stats.groupby("tier", as_index=False).agg(city_count=("city", "nunique"), sample_count=("count", "sum"))
    split_stats = sampled_df.groupby(["target_scale", "split"], as_index=False).size().rename(columns={"size": "count"})

    save_csv(category_stats, out_root / "reports" / "category_counts.csv")
    save_csv(city_stats, out_root / "reports" / "city_coverage_counts.csv")
    save_csv(tier_stats, out_root / "reports" / "tier_coverage_counts.csv")
    save_csv(split_stats, out_root / "reports" / "split_counts_by_scale.csv")

    plot_stats(category_stats, city_stats, tier_stats, out_root / "reports")
    plot_qc_maps(sampled_df, out_root / "qc")

    summary = {
        "selected_city_count": len(selected_cities),
        "districts_target_per_city": args.districts_per_city,
        "district_records_count": len(districts_all),
        "total_groups_planned": total_groups,
        "group_logs_count": len(group_logs),
        "raw_rows_after_dedup": int(len(raw_df)),
        "sampled_rows_all_scales": int(len(sampled_df)),
        "api_requests": client.request_count,
        "target_scales": TARGET_SCALES,
        "categories": POI_CATEGORIES,
    }
    (out_root / "reports" / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(str(out_root))
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
