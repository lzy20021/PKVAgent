from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MAINLAND_CSV = PROJECT_ROOT / "amap_poi_dataset_expanded" / "all_poi_expanded.csv"
TAIWAN_CSV = PROJECT_ROOT / "taiwan_poi_osm" / "taiwan_poi.csv"
TAIWAN_META = PROJECT_ROOT / "taiwan_poi_osm" / "taiwan_poi_fetch_meta.json"

OUT_DIR = PROJECT_ROOT / "paper_figures" / "merged_poi_with_taiwan"
OUT_CSV = OUT_DIR / "all_poi_merged.csv"
OUT_GEOJSON = OUT_DIR / "all_poi_merged.geojson"
OUT_SUMMARY = OUT_DIR / "summary_with_taiwan.txt"


def map_poi_type(text: str) -> Optional[str]:
    t = str(text).lower()
    if any(k in t for k in ["餐饮", "餐厅", "饭店", "restaurant", "food", "catering", "cafe", "fast_food"]):
        return "餐饮"
    if any(k in t for k in ["学校", "教育", "school", "education", "大学", "学院", "university", "college", "kindergarten"]):
        return "学校"
    if any(k in t for k in ["医院", "医疗", "medical", "hospital", "clinic"]):
        return "医院"
    if any(k in t for k in ["公交", "bus", "地铁", "metro", "subway", "station"]):
        return "公交站/地铁站"
    if any(k in t for k in ["酒店", "hotel", "motel", "guest_house", "hostel"]):
        return "酒店"
    if any(k in t for k in ["银行", "bank", "atm"]):
        return "银行"
    if any(k in t for k in ["便利店", "便利", "convenience"]):
        return "便利店"
    if any(k in t for k in ["停车场", "parking"]):
        return "停车场"
    return None


def to_geojson(df: pd.DataFrame, out_path: Path) -> None:
    feats = []
    for r in df.itertuples(index=False):
        feats.append(
            {
                "type": "Feature",
                "properties": {
                    "poi_id": r.poi_id,
                    "name": r.name,
                    "poi_type": r.poi_type,
                    "city": r.city,
                    "district": r.district,
                    "region": r.region,
                    "source": r.source,
                    "raw_category": r.raw_category,
                    "osm_type": r.osm_type,
                    "osm_id": r.osm_id,
                },
                "geometry": {"type": "Point", "coordinates": [float(r.lon), float(r.lat)]},
            }
        )
    gj = {"type": "FeatureCollection", "name": "all_poi_merged", "crs": {"type": "name", "properties": {"name": "EPSG:4326"}}, "features": feats}
    out_path.write_text(json.dumps(gj, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not MAINLAND_CSV.exists():
        raise RuntimeError(f"Missing mainland data: {MAINLAND_CSV}")
    if not TAIWAN_CSV.exists():
        raise RuntimeError(f"Missing taiwan data: {TAIWAN_CSV}")

    m = pd.read_csv(MAINLAND_CSV)
    tw = pd.read_csv(TAIWAN_CSV)

    m = m.rename(columns={"wgs84_lon": "lon", "wgs84_lat": "lat"}).copy()
    m["raw_category"] = np.where(
        m["query_category"].notna() & (m["query_category"].astype(str).str.len() > 0),
        m["query_category"].astype(str),
        m["type"].astype(str),
    )
    m["poi_type"] = m["raw_category"].map(map_poi_type)
    m = m[m["poi_type"].notna()].copy()
    m["poi_id"] = m["poi_uid"].astype(str)
    m["name"] = m["name"].fillna("").astype(str)
    m["city"] = m["city"].fillna("").astype(str)
    m["district"] = m["district"].fillna("").astype(str)
    m["region"] = "中国大陆"
    m["source"] = m["source"].fillna("amap").astype(str)
    m["osm_type"] = ""
    m["osm_id"] = ""
    m["tags_json"] = ""

    tw["poi_type"] = tw["poi_type"].map(map_poi_type)
    tw = tw[tw["poi_type"].notna()].copy()
    tw["poi_id"] = "osm_" + tw["osm_type"].astype(str) + "_" + tw["osm_id"].astype(str)
    tw["name"] = tw["name"].fillna("").astype(str)
    tw["city"] = tw["city"].fillna("台湾").astype(str)
    tw["district"] = tw["district"].fillna("").astype(str)
    tw["raw_category"] = tw["poi_type"]
    tw["region"] = "中国台湾"
    tw["source"] = tw["source"].fillna("osm_overpass_taiwan").astype(str)

    keep_cols = [
        "poi_id",
        "name",
        "poi_type",
        "lon",
        "lat",
        "city",
        "district",
        "region",
        "source",
        "raw_category",
        "osm_type",
        "osm_id",
        "tags_json",
    ]
    mm = m[keep_cols].copy()
    tt = tw[keep_cols].copy()
    all_df = pd.concat([mm, tt], ignore_index=True)
    all_df = all_df[(all_df["lon"].between(70, 140)) & (all_df["lat"].between(15, 55))].copy()

    # De-dup: keep unique IDs first, then remove coordinate duplicates at 1e-6 per category.
    all_df = all_df.drop_duplicates(subset=["poi_id"], keep="first")
    all_df["xy_key"] = (
        all_df["poi_type"].astype(str)
        + "|"
        + all_df["lon"].round(6).astype(str)
        + "|"
        + all_df["lat"].round(6).astype(str)
    )
    all_df = all_df.drop_duplicates(subset=["xy_key"], keep="first").drop(columns=["xy_key"])
    all_df = all_df.reset_index(drop=True)

    all_df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    to_geojson(all_df, OUT_GEOJSON)

    tw_cnt = int((all_df["region"] == "中国台湾").sum())
    tw_type_cnt = all_df.loc[all_df["region"] == "中国台湾", "poi_type"].value_counts().to_dict()
    tag_used = {}
    if TAIWAN_META.exists():
        try:
            tag_used = json.loads(TAIWAN_META.read_text(encoding="utf-8")).get("used_osm_tags", {})
        except Exception:  # noqa: BLE001
            tag_used = {}

    lines = [
        "台湾补齐后POI合并摘要",
        "",
        f"1. 台湾共抓取POI数量: {tw_cnt}",
        "2. 台湾各类别数量:",
    ]
    for k, v in tw_type_cnt.items():
        lines.append(f"- {k}: {v}")
    lines.append("3. 使用的OSM标签:")
    if tag_used:
        for k, v in tag_used.items():
            lines.append(f"- {k}: {v}")
    else:
        lines.append("- 见脚本 TAG_RULES（Overpass 标签规则）")
    lines.extend(
        [
            f"4. 合并后总POI数量: {len(all_df)}",
            "5. 输出文件路径:",
            f"- {OUT_CSV.as_posix()}",
            f"- {OUT_GEOJSON.as_posix()}",
        ]
    )
    OUT_SUMMARY.write_text("\n".join(lines), encoding="utf-8")

    print(f"台湾POI数量: {tw_cnt}")
    print("台湾分类数量:")
    for k, v in tw_type_cnt.items():
        print(f"{k}: {v}")
    print(f"合并总量: {len(all_df)}")
    print(f"输出CSV: {OUT_CSV.as_posix()}")
    print(f"输出GeoJSON: {OUT_GEOJSON.as_posix()}")
    print(f"摘要: {OUT_SUMMARY.as_posix()}")


if __name__ == "__main__":
    main()
