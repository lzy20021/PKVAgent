from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DATA_DIR = PROJECT_ROOT / "amap_poi_dataset_expanded"
OUT_FIG_DIR = PROJECT_ROOT / "paper_figures" / "expanded_structure_map"

DISTRICT_URL = "https://restapi.amap.com/v3/config/district"
POI_TEXT_URL = "https://restapi.amap.com/v3/place/text"

PRIMARY_CATS = ["餐饮", "学校", "医院", "公交站", "地铁站"]
EXTRA_CATS = ["酒店", "银行", "便利店", "停车场"]
ALL_CATS = PRIMARY_CATS + EXTRA_CATS

MUNICIPALITIES = {"北京", "上海", "天津", "重庆"}
PROVINCIAL_CAPITALS = {
    "石家庄", "太原", "呼和浩特", "沈阳", "长春", "哈尔滨", "南京", "杭州", "合肥", "福州", "南昌", "济南",
    "郑州", "武汉", "长沙", "广州", "南宁", "海口", "成都", "贵阳", "昆明", "拉萨", "西安", "兰州", "西宁", "银川", "乌鲁木齐"
}
PLAN_CITIES = {"大连", "青岛", "宁波", "厦门", "深圳"}


@dataclass
class CityInfo:
    city: str
    adcode: str
    province: str
    level: str


class AMapClient:
    def __init__(self, key: str, sleep: float = 0.08, retries: int = 3):
        self.key = key
        self.sleep = sleep
        self.retries = retries
        self.session = requests.Session()
        self.request_count = 0

    def get(self, url: str, params: Dict[str, Any]) -> Dict[str, Any]:
        payload = dict(params)
        payload["key"] = self.key
        last_err: Optional[Exception] = None
        for i in range(self.retries):
            try:
                resp = self.session.get(url, params=payload, timeout=40)
                self.request_count += 1
                resp.raise_for_status()
                data = resp.json()
                if str(data.get("status", "0")) != "1":
                    raise RuntimeError(f"AMap error: {data.get('info', 'unknown')} ({data.get('infocode', '')})")
                if self.sleep > 0:
                    time.sleep(self.sleep)
                return data
            except Exception as e:
                last_err = e
                time.sleep((i + 1) * 0.6)
        assert last_err is not None
        raise last_err

    def fetch_country_provinces(self) -> List[Dict[str, Any]]:
        data = self.get(DISTRICT_URL, {"keywords": "中国", "subdistrict": 1, "extensions": "base"})
        districts = data.get("districts") or []
        if not districts:
            return []
        return districts[0].get("districts") or []

    def fetch_cities_in_province(self, province_name: str) -> List[CityInfo]:
        data = self.get(DISTRICT_URL, {"keywords": province_name, "subdistrict": 2, "extensions": "base"})
        roots = data.get("districts") or []
        if not roots:
            return []
        root = roots[0]
        out: List[CityInfo] = []

        def walk(nodes: Iterable[Dict[str, Any]], province: str) -> None:
            for n in nodes:
                name = str(n.get("name", "")).strip()
                adcode = str(n.get("adcode", "")).strip()
                level = str(n.get("level", "")).strip()
                children = n.get("districts") or []
                if level in {"city", "district"}:
                    # municipalities may return district level directly
                    if name and adcode:
                        out.append(CityInfo(city=name, adcode=adcode, province=province, level=level))
                if children:
                    walk(children, province)

        walk(root.get("districts") or [], root.get("name", province_name))

        # fallback for municipality root
        if root.get("name") in MUNICIPALITIES and root.get("adcode"):
            out.append(CityInfo(city=str(root["name"]), adcode=str(root["adcode"]), province=str(root["name"]), level="city"))

        # dedup by city name
        uniq: Dict[str, CityInfo] = {}
        for c in out:
            if c.city not in uniq:
                uniq[c.city] = c
        return list(uniq.values())

    def fetch_district_names(self, city_name: str) -> List[str]:
        try:
            data = self.get(DISTRICT_URL, {"keywords": city_name, "subdistrict": 1, "extensions": "base"})
            roots = data.get("districts") or []
            if not roots:
                return []
            root = roots[0]
            nodes = root.get("districts") or []
            names = [str(n.get("name", "")).strip() for n in nodes if str(n.get("name", "")).strip()]
            return names[:5]
        except Exception:
            return []

    def search_poi_text(self, keyword: str, city: str, page: int = 1, offset: int = 25) -> List[Dict[str, Any]]:
        data = self.get(
            POI_TEXT_URL,
            {
                "keywords": keyword,
                "city": city,
                "citylimit": "true",
                "offset": offset,
                "page": page,
                "extensions": "base",
            },
        )
        return data.get("pois") or []


def load_existing_base() -> pd.DataFrame:
    expanded = OUT_DATA_DIR / "all_poi_expanded.csv"
    primary = PROJECT_ROOT / "amap_poi_dataset" / "raw" / "all_poi.csv"

    src = expanded if expanded.exists() else primary
    if not src.exists():
        return pd.DataFrame()
    df = pd.read_csv(src)

    # normalize to target schema
    cols = {c.lower(): c for c in df.columns}
    def col(name_list: List[str]) -> Optional[str]:
        for n in name_list:
            if n.lower() in cols:
                return cols[n.lower()]
        return None

    uid = col(["poi_uid", "id"])
    name = col(["name"])
    type_col = col(["type"])
    typecode = col(["typecode"])
    address = col(["address"])
    pname = col(["pname", "province"])
    cityname = col(["cityname", "city"])
    adname = col(["adname", "district"])
    query_category = col(["query_category", "category"])
    city = col(["city"])
    district = col(["district"])
    gcj_lon = col(["gcj_lon", "lon", "lng", "longitude"])
    gcj_lat = col(["gcj_lat", "lat", "latitude"])
    wgs_lon = col(["wgs84_lon"])
    wgs_lat = col(["wgs84_lat"])

    out = pd.DataFrame()
    out["poi_uid"] = df[uid].astype(str) if uid else ""
    out["name"] = df[name].astype(str) if name else ""
    out["type"] = df[type_col].astype(str) if type_col else ""
    out["typecode"] = df[typecode].astype(str) if typecode else ""
    out["address"] = df[address].astype(str) if address else ""
    out["pname"] = df[pname].astype(str) if pname else ""
    out["cityname"] = df[cityname].astype(str) if cityname else ""
    out["adname"] = df[adname].astype(str) if adname else ""
    out["query_category"] = df[query_category].astype(str) if query_category else ""
    out["city"] = df[city].astype(str) if city else out["cityname"]
    out["district"] = df[district].astype(str) if district else out["adname"]
    out["gcj_lon"] = pd.to_numeric(df[gcj_lon], errors="coerce") if gcj_lon else np.nan
    out["gcj_lat"] = pd.to_numeric(df[gcj_lat], errors="coerce") if gcj_lat else np.nan
    out["wgs84_lon"] = pd.to_numeric(df[wgs_lon], errors="coerce") if wgs_lon else out["gcj_lon"]
    out["wgs84_lat"] = pd.to_numeric(df[wgs_lat], errors="coerce") if wgs_lat else out["gcj_lat"]
    out["source"] = "existing"
    out = out.dropna(subset=["wgs84_lon", "wgs84_lat"])
    out = out[(out["wgs84_lon"].between(-180, 180)) & (out["wgs84_lat"].between(-90, 90))]
    return out


def gcj_to_wgs84(lng: float, lat: float) -> Tuple[float, float]:
    # imported lazily to keep script standalone in this folder
    import sys
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from src.io.amap_poi import gcj02_to_wgs84
    return gcj02_to_wgs84(lng, lat)


def parse_location(loc: str) -> Optional[Tuple[float, float]]:
    if not loc or "," not in loc:
        return None
    try:
        lon, lat = loc.split(",", 1)
        return float(lon), float(lat)
    except Exception:
        return None


def collect_target_cities(client: AMapClient, min_cities: int = 120) -> List[CityInfo]:
    # existing 50 cities as base
    existing_city_path = PROJECT_ROOT / "amap_poi_dataset" / "reports" / "selected_cities.csv"
    existing = []
    if existing_city_path.exists():
        d = pd.read_csv(existing_city_path)
        ccol = None
        for c in d.columns:
            if c.lower() in {"city", "城市"}:
                ccol = c
                break
        if ccol:
            existing = d[ccol].astype(str).str.strip().tolist()

    provinces = client.fetch_country_provinces()
    all_cities: Dict[str, CityInfo] = {}
    for p in provinces:
        pname = str(p.get("name", "")).strip()
        if not pname:
            continue
        for c in client.fetch_cities_in_province(pname):
            if c.city and c.city not in all_cities:
                all_cities[c.city] = c

    city_list = list(all_cities.values())

    def pri(c: CityInfo) -> Tuple[int, str]:
        score = 0
        if c.city in existing:
            score += 100
        if c.city in MUNICIPALITIES:
            score += 95
        if c.city in PROVINCIAL_CAPITALS:
            score += 85
        if c.city in PLAN_CITIES:
            score += 80
        if c.level == "city":
            score += 20
        return (-score, c.city)

    city_list.sort(key=pri)

    picked: List[CityInfo] = []
    seen = set()
    for c in city_list:
        if c.city in seen:
            continue
        picked.append(c)
        seen.add(c.city)
        if len(picked) >= max(min_cities, 130):
            break

    if len(picked) < min_cities:
        for c in city_list:
            if c.city not in seen:
                picked.append(c)
                seen.add(c.city)
                if len(picked) >= min_cities:
                    break

    return picked


def fetch_city_pois(client: AMapClient, city: CityInfo, max_pages_city: int = 8, max_pages_dist: int = 3) -> List[Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    districts = client.fetch_district_names(city.city)

    for cat in ALL_CATS:
        # direct city query
        for p in range(1, max_pages_city + 1):
            pois = client.search_poi_text(keyword=cat, city=city.city, page=p)
            if not pois:
                break
            for poi in pois:
                pid = str(poi.get("id", "")).strip()
                if not pid:
                    continue
                loc = parse_location(str(poi.get("location", "")))
                if loc is None:
                    continue
                glon, glat = loc
                wlon, wlat = gcj_to_wgs84(glon, glat)
                rows[pid] = {
                    "poi_uid": pid,
                    "name": str(poi.get("name", "")),
                    "type": str(poi.get("type", "")),
                    "typecode": str(poi.get("typecode", "")),
                    "address": str(poi.get("address", "")),
                    "pname": str(poi.get("pname", "")),
                    "cityname": str(poi.get("cityname", city.city)),
                    "adname": str(poi.get("adname", "")),
                    "query_category": cat,
                    "city": city.city,
                    "district": str(poi.get("adname", "")),
                    "gcj_lon": glon,
                    "gcj_lat": glat,
                    "wgs84_lon": wlon,
                    "wgs84_lat": wlat,
                    "source": "amap_api_incremental",
                }

        # district hotspot query to enrich typical areas
        for d in districts[:4]:
            kw = f"{d}{cat}"
            for p in range(1, max_pages_dist + 1):
                pois = client.search_poi_text(keyword=kw, city=city.city, page=p)
                if not pois:
                    break
                for poi in pois:
                    pid = str(poi.get("id", "")).strip()
                    if not pid:
                        continue
                    loc = parse_location(str(poi.get("location", "")))
                    if loc is None:
                        continue
                    glon, glat = loc
                    wlon, wlat = gcj_to_wgs84(glon, glat)
                    rows[pid] = {
                        "poi_uid": pid,
                        "name": str(poi.get("name", "")),
                        "type": str(poi.get("type", "")),
                        "typecode": str(poi.get("typecode", "")),
                        "address": str(poi.get("address", "")),
                        "pname": str(poi.get("pname", "")),
                        "cityname": str(poi.get("cityname", city.city)),
                        "adname": str(poi.get("adname", d)),
                        "query_category": cat,
                        "city": city.city,
                        "district": str(poi.get("adname", d)),
                        "gcj_lon": glon,
                        "gcj_lat": glat,
                        "wgs84_lon": wlon,
                        "wgs84_lat": wlat,
                        "source": "amap_api_incremental",
                    }

    return list(rows.values())


def save_geojson(df: pd.DataFrame, path: Path) -> None:
    features = []
    keep_cols = [
        "poi_uid", "name", "type", "typecode", "address", "pname", "cityname", "adname", "query_category", "city", "district", "source"
    ]
    for _, r in df.iterrows():
        props = {k: (None if pd.isna(r.get(k)) else r.get(k)) for k in keep_cols}
        features.append(
            {
                "type": "Feature",
                "properties": props,
                "geometry": {"type": "Point", "coordinates": [float(r["wgs84_lon"]), float(r["wgs84_lat"])]},
            }
        )
    payload = {"type": "FeatureCollection", "features": features}
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def save_checkpoint(merged: pd.DataFrame, city_stats: List[Dict[str, Any]], request_count: int) -> None:
    csv_path = OUT_DATA_DIR / "all_poi_expanded.csv"
    geojson_path = OUT_DATA_DIR / "all_poi_expanded.geojson"
    city_list_path = OUT_FIG_DIR / "expanded_city_list.csv"
    fetch_summary_path = OUT_FIG_DIR / "fetch_summary.json"

    merged.to_csv(csv_path, index=False, encoding="utf-8-sig")
    save_geojson(merged, geojson_path)
    pd.DataFrame(city_stats).to_csv(city_list_path, index=False, encoding="utf-8-sig")
    fetch_summary_path.write_text(
        json.dumps(
            {
                "checkpoint": True,
                "city_rows": len(city_stats),
                "poi_rows": int(len(merged)),
                "request_count": int(request_count),
                "output_csv": csv_path.as_posix(),
                "output_geojson": geojson_path.as_posix(),
                "expanded_city_list": city_list_path.as_posix(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Incrementally fetch and expand AMap POI dataset to >=120 cities")
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--min-cities", type=int, default=120)
    parser.add_argument("--sleep", type=float, default=0.08)
    parser.add_argument("--max-pages-city", type=int, default=6)
    parser.add_argument("--max-pages-dist", type=int, default=3)
    parser.add_argument("--city-cap", type=int, default=130)
    parser.add_argument("--seed", type=int, default=20260318)
    args = parser.parse_args()

    random.seed(args.seed)
    OUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)

    client = AMapClient(key=args.api_key, sleep=args.sleep)
    targets = collect_target_cities(client, min_cities=args.min_cities)
    targets = targets[: max(args.min_cities, args.city_cap)]

    existing = load_existing_base()
    existing_ids = set(existing["poi_uid"].astype(str)) if not existing.empty else set()
    existing_city_counts = existing["city"].astype(str).value_counts().to_dict() if not existing.empty else {}

    all_new_rows: List[Dict[str, Any]] = []
    city_stats = []

    checkpoint_every = 8
    for i, c in enumerate(targets, 1):
        if int(existing_city_counts.get(c.city, 0)) >= 900:
            city_stats.append(
                {
                    "city": c.city,
                    "province": c.province,
                    "adcode": c.adcode,
                    "level": c.level,
                    "fetched": 0,
                    "added": 0,
                    "status": "skipped_existing_enough",
                }
            )
            continue
        try:
            rows = fetch_city_pois(client, c, max_pages_city=args.max_pages_city, max_pages_dist=args.max_pages_dist)
            before = len(rows)
            rows = [r for r in rows if r["poi_uid"] not in existing_ids]
            for r in rows:
                existing_ids.add(r["poi_uid"])
                existing_city_counts[r["city"]] = int(existing_city_counts.get(r["city"], 0)) + 1
            all_new_rows.extend(rows)
            city_stats.append({"city": c.city, "province": c.province, "adcode": c.adcode, "level": c.level, "fetched": before, "added": len(rows), "status": "ok"})
            print(f"[{i}/{len(targets)}] {c.city}: fetched={before}, added={len(rows)}")
        except Exception as e:
            city_stats.append({"city": c.city, "province": c.province, "adcode": c.adcode, "level": c.level, "fetched": 0, "added": 0, "status": f"error: {e}"})
            print(f"[{i}/{len(targets)}] {c.city}: error={e}")

        if i % checkpoint_every == 0:
            tmp_new = pd.DataFrame(all_new_rows)
            merged_ck = pd.concat([existing, tmp_new], ignore_index=True) if not existing.empty else tmp_new
            merged_ck = merged_ck.drop_duplicates(subset=["poi_uid"], keep="first").dropna(subset=["wgs84_lon", "wgs84_lat"])
            save_checkpoint(merged_ck, city_stats, client.request_count)
            print(f"checkpoint saved at city {i}, poi_rows={len(merged_ck)}")

    new_df = pd.DataFrame(all_new_rows)
    if existing.empty:
        merged = new_df
    else:
        merged = pd.concat([existing, new_df], ignore_index=True)

    merged = merged.drop_duplicates(subset=["poi_uid"], keep="first").copy()
    merged = merged.dropna(subset=["wgs84_lon", "wgs84_lat"])

    csv_path = OUT_DATA_DIR / "all_poi_expanded.csv"
    geojson_path = OUT_DATA_DIR / "all_poi_expanded.geojson"
    merged.to_csv(csv_path, index=False, encoding="utf-8-sig")
    save_geojson(merged, geojson_path)

    city_df = pd.DataFrame(city_stats)
    city_df = city_df.sort_values(["status", "added", "fetched"], ascending=[True, False, False])
    city_list_path = OUT_FIG_DIR / "expanded_city_list.csv"
    city_df.to_csv(city_list_path, index=False, encoding="utf-8-sig")

    summary = {
        "target_city_count": len(targets),
        "successful_city_count": int((city_df["status"] == "ok").sum()) if not city_df.empty else 0,
        "total_requests": client.request_count,
        "new_rows": int(len(new_df)),
        "merged_rows": int(len(merged)),
        "api_endpoints": [DISTRICT_URL, POI_TEXT_URL],
        "output_csv": csv_path.as_posix(),
        "output_geojson": geojson_path.as_posix(),
        "expanded_city_list": city_list_path.as_posix(),
    }
    (OUT_FIG_DIR / "fetch_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
