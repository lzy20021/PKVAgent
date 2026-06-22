from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "taiwan_poi_osm"
OUT_CSV = OUT_DIR / "taiwan_poi.csv"
OUT_GEOJSON = OUT_DIR / "taiwan_poi.geojson"
OUT_META = OUT_DIR / "taiwan_poi_fetch_meta.json"

OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]

# Taiwan admin area; fallback bbox covers main island + major offshore islands.
AREA_STMT = 'area["ISO3166-1"="TW"][admin_level=2]->.tw;'
BBOX = "(21.0,118.0,26.6,123.5)"

TAG_RULES: List[Tuple[str, str, List[str]]] = [
    ("餐饮", '[amenity~"^(restaurant|cafe|fast_food|food_court)$"]', ["amenity=restaurant/cafe/fast_food/food_court"]),
    ("学校", '[amenity~"^(school|university|college|kindergarten)$"]', ["amenity=school/university/college/kindergarten"]),
    ("医院", '[amenity~"^(hospital|clinic)$"]', ["amenity=hospital/clinic"]),
    ("公交站/地铁站", '[highway="bus_stop"]', ["highway=bus_stop"]),
    ("公交站/地铁站", '[railway="station"]', ["railway=station"]),
    ("公交站/地铁站", '[station="subway"]', ["station=subway"]),
    ("便利店", '[shop="convenience"]', ["shop=convenience"]),
    ("停车场", '[amenity="parking"]', ["amenity=parking"]),
    ("酒店", '[tourism~"^(hotel|motel|guest_house|hostel)$"]', ["tourism=hotel/motel/guest_house/hostel"]),
    ("银行", '[amenity~"^(bank|atm)$"]', ["amenity=bank/atm"]),
]


def overpass_query(filter_expr: str) -> str:
    return f"""
[out:json][timeout:240];
(
  {AREA_STMT}
  node(area.tw){filter_expr};
  way(area.tw){filter_expr};
  relation(area.tw){filter_expr};
);
out tags center;
"""


def request_overpass(query: str, retries: int = 4) -> dict:
    last_error = None
    for endpoint in OVERPASS_ENDPOINTS:
        for i in range(retries):
            try:
                r = requests.post(endpoint, data={"data": query}, timeout=180)
                if r.status_code == 429:
                    time.sleep(4 + i * 2)
                    continue
                r.raise_for_status()
                data = r.json()
                if "elements" not in data:
                    raise RuntimeError(f"Overpass response missing elements at {endpoint}")
                return data
            except Exception as e:  # noqa: BLE001
                last_error = e
                time.sleep(2 + i * 2)
    raise RuntimeError(f"Overpass request failed after retries: {last_error}")


def extract_lon_lat(ele: dict) -> Tuple[float, float]:
    if "lon" in ele and "lat" in ele:
        return float(ele["lon"]), float(ele["lat"])
    c = ele.get("center", {})
    if "lon" in c and "lat" in c:
        return float(c["lon"]), float(c["lat"])
    raise ValueError("No lon/lat found")


def fetch_taiwan_poi() -> pd.DataFrame:
    rows: List[dict] = []
    for idx, (poi_type, filt, _) in enumerate(TAG_RULES):
        q = overpass_query(filt)
        payload = request_overpass(q)
        elements = payload.get("elements", [])
        for ele in elements:
            try:
                lon, lat = extract_lon_lat(ele)
            except Exception:  # noqa: BLE001
                continue
            tags = ele.get("tags", {})
            city = (
                tags.get("addr:city")
                or tags.get("is_in:city")
                or tags.get("addr:county")
                or tags.get("addr:state")
                or "台湾"
            )
            district = (
                tags.get("addr:district")
                or tags.get("addr:suburb")
                or tags.get("addr:town")
                or tags.get("addr:village")
                or ""
            )
            rows.append(
                {
                    "osm_type": ele.get("type", ""),
                    "osm_id": str(ele.get("id", "")),
                    "name": tags.get("name", ""),
                    "poi_type": poi_type,
                    "lon": lon,
                    "lat": lat,
                    "city": str(city),
                    "district": str(district),
                    "tags_json": json.dumps(tags, ensure_ascii=False),
                    "source": "osm_overpass_taiwan",
                    "rule_index": idx,
                }
            )
        time.sleep(1.2)

    if not rows:
        return pd.DataFrame(
            columns=["osm_type", "osm_id", "name", "poi_type", "lon", "lat", "city", "district", "tags_json", "source"]
        )

    df = pd.DataFrame(rows)
    df = df[(df["lon"].between(118, 124)) & (df["lat"].between(20, 27))].copy()
    df["osm_key"] = df["osm_type"].astype(str) + ":" + df["osm_id"].astype(str)
    df = df.sort_values("rule_index").drop_duplicates(subset=["osm_key"], keep="first")
    df["xy_key"] = (
        df["poi_type"].astype(str)
        + "|"
        + df["lon"].round(6).astype(str)
        + "|"
        + df["lat"].round(6).astype(str)
    )
    df = df.drop_duplicates(subset=["xy_key"], keep="first").copy()
    df = df.drop(columns=["rule_index", "osm_key", "xy_key"])
    return df.reset_index(drop=True)


def to_geojson(df: pd.DataFrame, out_path: Path) -> None:
    features = []
    for r in df.itertuples(index=False):
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "osm_type": r.osm_type,
                    "osm_id": r.osm_id,
                    "name": r.name,
                    "poi_type": r.poi_type,
                    "city": r.city,
                    "district": r.district,
                    "source": r.source,
                },
                "geometry": {"type": "Point", "coordinates": [float(r.lon), float(r.lat)]},
            }
        )
    gj = {"type": "FeatureCollection", "name": "taiwan_poi_osm", "crs": {"type": "name", "properties": {"name": "EPSG:4326"}}, "features": features}
    out_path.write_text(json.dumps(gj, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = fetch_taiwan_poi()
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    to_geojson(df, OUT_GEOJSON)

    tag_desc = {
        "餐饮": "amenity=restaurant|cafe|fast_food|food_court",
        "学校": "amenity=school|university|college|kindergarten",
        "医院": "amenity=hospital|clinic",
        "公交站/地铁站": "highway=bus_stop OR railway=station OR station=subway",
        "便利店": "shop=convenience",
        "停车场": "amenity=parking",
        "酒店": "tourism=hotel|motel|guest_house|hostel",
        "银行": "amenity=bank|atm",
    }
    meta = {
        "total_poi": int(len(df)),
        "poi_counts": df["poi_type"].value_counts().to_dict(),
        "used_osm_tags": tag_desc,
        "overpass_endpoints": OVERPASS_ENDPOINTS,
        "out_csv": OUT_CSV.as_posix(),
        "out_geojson": OUT_GEOJSON.as_posix(),
    }
    OUT_META.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"台湾POI总量: {len(df)}")
    print("分类计数:")
    if len(df):
        print(df["poi_type"].value_counts().to_string())
    print(f"输出CSV: {OUT_CSV.as_posix()}")
    print(f"输出GeoJSON: {OUT_GEOJSON.as_posix()}")
    print(f"元数据: {OUT_META.as_posix()}")


if __name__ == "__main__":
    main()
