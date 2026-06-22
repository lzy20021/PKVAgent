"""
高德 POI 批量抓取与 GeoJSON 导出。

默认将高德返回的 GCJ-02 坐标转换为 WGS84，再写出标准 GeoJSON，
避免在 OSM/Leaflet 等常见底图上产生系统性偏移。
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import requests


AMAP_POLYGON_SEARCH_URL = "https://restapi.amap.com/v5/place/polygon"
DEFAULT_PAGE_SIZE = 25
DEFAULT_MIN_POINTS = 100
DEFAULT_MAX_POINTS = 1000
DEFAULT_SLEEP_SECONDS = 0.2


@dataclass(frozen=True)
class RegionSpec:
    city: str
    region_name: str
    bbox: Tuple[float, float, float, float]

    @property
    def slug(self) -> str:
        return slugify(f"{self.city}_{self.region_name}")


@dataclass(frozen=True)
class CategorySpec:
    keyword: str
    theme: str

    @property
    def slug(self) -> str:
        return slugify(self.keyword)


@dataclass(frozen=True)
class GroupSpec:
    group_id: str
    city: str
    region_name: str
    keyword: str
    theme: str
    bbox: Tuple[float, float, float, float]

    @property
    def output_name(self) -> str:
        return f"{self.group_id}.geojson"


@dataclass
class GroupResult:
    group_id: str
    output_path: str
    feature_count: int
    request_count: int
    pages_fetched: int
    status: str
    source_crs: str
    target_crs: str
    city: str
    region_name: str
    keyword: str
    bbox: Tuple[float, float, float, float]
    note: str = ""


def slugify(value: str) -> str:
    value = value.strip().replace(" ", "_")
    safe_chars = []
    for char in value:
        if char.isascii() and (char.isalnum() or char in {"_", "-"}):
            safe_chars.append(char.lower())
        elif char in {"-", "_"}:
            safe_chars.append(char)
        else:
            safe_chars.append(f"u{ord(char):x}")
    return "".join(safe_chars)


def build_default_groups() -> List[GroupSpec]:
    regions: Sequence[RegionSpec] = [
        RegionSpec("北京", "朝阳CBD", (116.438, 39.891, 116.507, 39.938)),
        RegionSpec("北京", "中关村", (116.292, 39.957, 116.352, 40.003)),
        RegionSpec("上海", "陆家嘴", (121.490, 31.219, 121.543, 31.259)),
        RegionSpec("上海", "徐家汇", (121.423, 31.178, 121.470, 31.218)),
        RegionSpec("广州", "珠江新城", (113.299, 23.113, 113.351, 23.145)),
        RegionSpec("广州", "北京路", (113.255, 23.112, 113.292, 23.145)),
        RegionSpec("深圳", "福田中心区", (114.035, 22.520, 114.087, 22.554)),
        RegionSpec("深圳", "南山科技园", (113.922, 22.521, 113.977, 22.556)),
        RegionSpec("杭州", "武林广场", (120.149, 30.260, 120.191, 30.291)),
        RegionSpec("杭州", "滨江区", (120.180, 30.180, 120.245, 30.227)),
        RegionSpec("南京", "新街口", (118.768, 32.025, 118.804, 32.055)),
        RegionSpec("南京", "鼓楼", (118.756, 32.055, 118.793, 32.086)),
        RegionSpec("武汉", "江汉路", (114.281, 30.570, 114.323, 30.603)),
        RegionSpec("武汉", "光谷", (114.376, 30.492, 114.430, 30.531)),
        RegionSpec("成都", "春熙路", (104.067, 30.651, 104.101, 30.676)),
        RegionSpec("成都", "高新区", (104.038, 30.542, 104.094, 30.587)),
        RegionSpec("重庆", "解放碑", (106.560, 29.550, 106.590, 29.572)),
        RegionSpec("重庆", "观音桥", (106.520, 29.570, 106.554, 29.601)),
        RegionSpec("西安", "小寨", (108.927, 34.214, 108.973, 34.247)),
        RegionSpec("西安", "钟楼", (108.930, 34.250, 108.966, 34.278)),
    ]
    categories: Sequence[CategorySpec] = [
        CategorySpec("餐厅", "超高密商业点"),
        CategorySpec("便利店", "社区服务点"),
        CategorySpec("咖啡厅", "品牌连锁点"),
        CategorySpec("酒店", "住宿服务点"),
        CategorySpec("停车场", "交通配套点"),
    ]

    groups: List[GroupSpec] = []
    for region in regions:
        for category in categories:
            group_id = f"{region.slug}__{category.slug}"
            groups.append(
                GroupSpec(
                    group_id=group_id,
                    city=region.city,
                    region_name=region.region_name,
                    keyword=category.keyword,
                    theme=category.theme,
                    bbox=region.bbox,
                )
            )
    return groups


def bbox_to_polygon_param(bbox: Tuple[float, float, float, float]) -> str:
    min_lon, min_lat, max_lon, max_lat = bbox
    return f"{min_lon},{min_lat}|{max_lon},{max_lat}"


def out_of_china(lon: float, lat: float) -> bool:
    return not (73.66 < lon < 135.05 and 3.86 < lat < 53.55)


def _transform_lat(lon: float, lat: float) -> float:
    result = (
        -100.0
        + 2.0 * lon
        + 3.0 * lat
        + 0.2 * lat * lat
        + 0.1 * lon * lat
        + 0.2 * math.sqrt(abs(lon))
    )
    result += (20.0 * math.sin(6.0 * lon * math.pi) + 20.0 * math.sin(2.0 * lon * math.pi)) * 2.0 / 3.0
    result += (20.0 * math.sin(lat * math.pi) + 40.0 * math.sin(lat / 3.0 * math.pi)) * 2.0 / 3.0
    result += (160.0 * math.sin(lat / 12.0 * math.pi) + 320.0 * math.sin(lat * math.pi / 30.0)) * 2.0 / 3.0
    return result


def _transform_lon(lon: float, lat: float) -> float:
    result = (
        300.0
        + lon
        + 2.0 * lat
        + 0.1 * lon * lon
        + 0.1 * lon * lat
        + 0.1 * math.sqrt(abs(lon))
    )
    result += (20.0 * math.sin(6.0 * lon * math.pi) + 20.0 * math.sin(2.0 * lon * math.pi)) * 2.0 / 3.0
    result += (20.0 * math.sin(lon * math.pi) + 40.0 * math.sin(lon / 3.0 * math.pi)) * 2.0 / 3.0
    result += (150.0 * math.sin(lon / 12.0 * math.pi) + 300.0 * math.sin(lon / 30.0 * math.pi)) * 2.0 / 3.0
    return result


def gcj02_to_wgs84(lon: float, lat: float) -> Tuple[float, float]:
    if out_of_china(lon, lat):
        return lon, lat
    a = 6378245.0
    ee = 0.00669342162296594323
    d_lat = _transform_lat(lon - 105.0, lat - 35.0)
    d_lon = _transform_lon(lon - 105.0, lat - 35.0)
    rad_lat = lat / 180.0 * math.pi
    magic = math.sin(rad_lat)
    magic = 1 - ee * magic * magic
    sqrt_magic = math.sqrt(magic)
    d_lat = (d_lat * 180.0) / ((a * (1 - ee)) / (magic * sqrt_magic) * math.pi)
    d_lon = (d_lon * 180.0) / (a / sqrt_magic * math.cos(rad_lat) * math.pi)
    mg_lat = lat + d_lat
    mg_lon = lon + d_lon
    return lon * 2 - mg_lon, lat * 2 - mg_lat


def make_feature(group: GroupSpec, poi: Dict[str, Any], target_crs: str) -> Optional[Dict[str, Any]]:
    location = poi.get("location") or ""
    if "," not in location:
        return None

    lon_text, lat_text = location.split(",", 1)
    gcj_lon = float(lon_text)
    gcj_lat = float(lat_text)
    if target_crs.upper() == "WGS84":
        lon, lat = gcj02_to_wgs84(gcj_lon, gcj_lat)
    else:
        lon, lat = gcj_lon, gcj_lat

    properties = {
        "group_id": group.group_id,
        "city": group.city,
        "region_name": group.region_name,
        "keyword": group.keyword,
        "theme": group.theme,
        "bbox": list(group.bbox),
        "source_crs": "GCJ-02",
        "target_crs": target_crs.upper(),
        "amap_id": poi.get("id"),
        "name": poi.get("name"),
        "type": poi.get("type"),
        "typecode": poi.get("typecode"),
        "address": poi.get("address"),
        "pname": poi.get("pname"),
        "cityname": poi.get("cityname"),
        "adname": poi.get("adname"),
        "business_area": poi.get("business_area"),
        "tel": poi.get("tel"),
        "rating": poi.get("rating"),
        "cost": poi.get("cost"),
    }
    return {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [lon, lat]},
        "properties": properties,
    }


def build_feature_collection(
    group: GroupSpec,
    pois: Iterable[Dict[str, Any]],
    target_crs: str = "WGS84",
) -> Dict[str, Any]:
    features: List[Dict[str, Any]] = []
    for poi in pois:
        feature = make_feature(group, poi, target_crs=target_crs)
        if feature is not None:
            features.append(feature)

    return {
        "type": "FeatureCollection",
        "name": group.group_id,
        "metadata": {
            "group_id": group.group_id,
            "city": group.city,
            "region_name": group.region_name,
            "keyword": group.keyword,
            "theme": group.theme,
            "bbox": list(group.bbox),
            "source_crs": "GCJ-02",
            "target_crs": target_crs.upper(),
            "feature_count": len(features),
        },
        "features": features,
    }


def write_geojson(collection: Dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(collection, ensure_ascii=False, indent=2), encoding="utf-8")


class AmapPOIClient:
    def __init__(
        self,
        api_key: str,
        *,
        page_size: int = DEFAULT_PAGE_SIZE,
        sleep_seconds: float = DEFAULT_SLEEP_SECONDS,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.api_key = api_key
        self.page_size = page_size
        self.sleep_seconds = sleep_seconds
        self.session = session or requests.Session()

    def fetch_group_pois(
        self,
        group: GroupSpec,
        *,
        min_points: int = DEFAULT_MIN_POINTS,
        max_points: int = DEFAULT_MAX_POINTS,
        max_pages: int = 40,
    ) -> Tuple[List[Dict[str, Any]], int, int]:
        dedup: Dict[str, Dict[str, Any]] = {}
        pages_fetched = 0
        request_count = 0

        for page_num in range(1, max_pages + 1):
            params = {
                "key": self.api_key,
                "keywords": group.keyword,
                "polygon": bbox_to_polygon_param(group.bbox),
                "page_size": self.page_size,
                "page_num": page_num,
            }
            response = self.session.get(AMAP_POLYGON_SEARCH_URL, params=params, timeout=30)
            request_count += 1
            response.raise_for_status()
            payload = response.json()
            info = str(payload.get("info", ""))
            if str(payload.get("status")) != "1":
                raise RuntimeError(f"{group.group_id}: AMap API error: {info}")

            pois = payload.get("pois") or []
            pages_fetched = page_num
            for poi in pois:
                poi_id = poi.get("id") or f"{poi.get('name','unknown')}@{poi.get('location','')}"
                dedup[poi_id] = poi

            if not pois:
                break
            if len(dedup) >= max_points:
                break

            if self.sleep_seconds > 0:
                time.sleep(self.sleep_seconds)

        pois_list = list(dedup.values())[:max_points]
        return pois_list, pages_fetched, request_count


def export_groups(
    api_key: str,
    output_dir: Path,
    *,
    groups: Optional[Sequence[GroupSpec]] = None,
    target_crs: str = "WGS84",
    min_points: int = DEFAULT_MIN_POINTS,
    max_points: int = DEFAULT_MAX_POINTS,
    max_pages: int = 40,
    overwrite: bool = False,
    sleep_seconds: float = DEFAULT_SLEEP_SECONDS,
) -> List[GroupResult]:
    output_dir.mkdir(parents=True, exist_ok=True)
    group_specs = list(groups or build_default_groups())
    client = AmapPOIClient(api_key, sleep_seconds=sleep_seconds)
    results: List[GroupResult] = []

    for group in group_specs:
        output_path = output_dir / group.output_name
        if output_path.exists() and not overwrite:
            existing = json.loads(output_path.read_text(encoding="utf-8"))
            feature_count = len(existing.get("features", []))
            results.append(
                GroupResult(
                    group_id=group.group_id,
                    output_path=str(output_path),
                    feature_count=feature_count,
                    request_count=0,
                    pages_fetched=0,
                    status="skipped_existing",
                    source_crs="GCJ-02",
                    target_crs=target_crs.upper(),
                    city=group.city,
                    region_name=group.region_name,
                    keyword=group.keyword,
                    bbox=group.bbox,
                    note="output already exists",
                )
            )
            continue

        try:
            pois, pages_fetched, request_count = client.fetch_group_pois(
                group,
                min_points=min_points,
                max_points=max_points,
                max_pages=max_pages,
            )
            collection = build_feature_collection(group, pois, target_crs=target_crs)
            write_geojson(collection, output_path)

            feature_count = len(collection["features"])
            if feature_count < min_points:
                status = "underfilled"
                note = f"fewer than requested minimum {min_points}"
            else:
                status = "ok"
                note = ""
            results.append(
                GroupResult(
                    group_id=group.group_id,
                    output_path=str(output_path),
                    feature_count=feature_count,
                    request_count=request_count,
                    pages_fetched=pages_fetched,
                    status=status,
                    source_crs="GCJ-02",
                    target_crs=target_crs.upper(),
                    city=group.city,
                    region_name=group.region_name,
                    keyword=group.keyword,
                    bbox=group.bbox,
                    note=note,
                )
            )
        except Exception as exc:
            results.append(
                GroupResult(
                    group_id=group.group_id,
                    output_path=str(output_path),
                    feature_count=0,
                    request_count=0,
                    pages_fetched=0,
                    status="error",
                    source_crs="GCJ-02",
                    target_crs=target_crs.upper(),
                    city=group.city,
                    region_name=group.region_name,
                    keyword=group.keyword,
                    bbox=group.bbox,
                    note=str(exc),
                )
            )

    return results


def write_summary(results: Sequence[GroupResult], output_path: Path) -> None:
    payload = {
        "group_count": len(results),
        "ok_count": sum(1 for item in results if item.status == "ok"),
        "underfilled_count": sum(1 for item in results if item.status == "underfilled"),
        "error_count": sum(1 for item in results if item.status == "error"),
        "skipped_count": sum(1 for item in results if item.status == "skipped_existing"),
        "results": [asdict(item) for item in results],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
