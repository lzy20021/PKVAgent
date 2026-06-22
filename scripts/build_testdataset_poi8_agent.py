"""
构建 TestDatasets：单城市内 8 类 POI 各约 15 条，共约 120 点，含 point_symbol / demo_type 字段，
供「当前图层 + 目标比例尺」智能体综合测试使用。

默认读取: amap_poi_dataset_expanded/all_poi_expanded.csv
输出: TestDatasets/poi_8types_<city>/points.geojson + manifest.json

用法:
  python scripts/build_testdataset_poi8_agent.py
  python scripts/build_testdataset_poi8_agent.py --csv E:\\path\\to\\all_poi_expanded.csv
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Dict, List, Set, Tuple

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# (demo_type, point_symbol, query_category 集合)
TYPE_SPECS: List[Tuple[str, str, Set[str]]] = [
    ("便利店", "circle", {"便利店"}),
    ("学校", "square", {"学校"}),
    ("停车场", "triangle", {"停车场"}),
    ("酒店", "diamond", {"酒店"}),
    ("银行", "cross", {"银行"}),
    ("医院", "star", {"医院"}),
    ("餐饮", "plus", {"餐饮", "餐厅", "咖啡厅"}),
    ("交通设施", "hexagon", {"公交站", "地铁站"}),
]

PER_TYPE_DEFAULT = 15
TOTAL_TARGET = PER_TYPE_DEFAULT * len(TYPE_SPECS)  # 120

# 多区域人工测试推荐城市（均在 all_poi_expanded 中校验过可同时覆盖 8 类）
DEFAULT_MULTI_CITIES: List[str] = [
    "广州市",
    "北京市",
    "上海市",
    "成都市",
    "武汉市",
    "深圳市",
    "南京市",
    "杭州市",
    "重庆市",
    "太原市",
    "兰州市",
    "威海市",
    "北海市",
    "合肥市",
    "昆明市",
    "南宁市",
    "沈阳市",
    "哈尔滨市",
    "青岛市",
    "厦门市",
]


def _city_has_all_types(df, city: str) -> bool:
    sub = df[df["cityname"] == city]
    if sub.empty:
        return False
    for _, _, cats in TYPE_SPECS:
        if not sub["query_category"].isin(cats).any():
            return False
    return True


def pick_city(df) -> str:
    cities = [str(c) for c in df["cityname"].dropna().unique().tolist()]
    random.seed(42)
    random.shuffle(cities)
    for c in cities:
        if _city_has_all_types(df, c):
            return c
    raise RuntimeError("数据集中找不到同时包含 8 类 POI 的城市，请检查 CSV。")


def build_one_city(
    df,
    city: str,
    *,
    per_type: int,
    out_root: str,
) -> str:
    """写入单城市数据集，返回 geojson 路径。"""
    if not _city_has_all_types(df, city):
        raise ValueError(f"城市 [{city}] 无法覆盖全部 8 类 POI")

    city_df = df[df["cityname"] == city].copy()
    features = []
    for ti, (demo_type, symbol, cats) in enumerate(TYPE_SPECS):
        pool = city_df[city_df["query_category"].isin(cats)]
        if pool.empty:
            raise RuntimeError(f"城市 [{city}] 缺少类别 {demo_type} (query_category in {cats})")
        n = min(per_type, len(pool))
        part = pool.sample(n=n, random_state=42 + ti)
        for _, r in part.iterrows():
            features.append(
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [float(r["wgs84_lon"]), float(r["wgs84_lat"])],
                    },
                    "properties": {
                        "name": str(r.get("name", ""))[:80],
                        "cityname": city,
                        "query_category": str(r.get("query_category", "")),
                        "demo_type": demo_type,
                        "point_symbol": symbol,
                        "typecode": str(r.get("typecode", "")),
                    },
                }
            )

    safe_city = "".join(c if c.isalnum() else "_" for c in city)[:24].strip("_") or "city"
    out_dir = os.path.join(out_root, f"poi_8types_{safe_city}")
    os.makedirs(out_dir, exist_ok=True)
    geo_path = os.path.join(out_dir, "points.geojson")

    import geopandas as gpd

    gdf_out = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
    gdf_out.to_file(geo_path, driver="GeoJSON")

    manifest = {
        "city": city,
        "feature_count": len(features),
        "per_type": per_type,
        "types": [x[0] for x in TYPE_SPECS],
        "geojson": geo_path,
        "symbol_by_demo_type": {x[0]: x[1] for x in TYPE_SPECS},
        "note": "在 MapGenAI 中加载 points.geojson 后，对智能体说：请你对我的当前图层做综合，目标比例尺是1:250000（比例尺可改）。",
        "rebuild_script": "python scripts/build_testdataset_poi8_agent.py",
        "agent_e2e_test": "python scripts/run_agent_test_poi8_current_layer.py --scale 250000",
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"城市: {city}")
    print(f"要素数: {len(features)} (每类最多 {per_type})")
    print(f"已写入: {geo_path}")
    return geo_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv",
        default=os.path.join(_ROOT, "amap_poi_dataset_expanded", "all_poi_expanded.csv"),
        help="expanded POI CSV",
    )
    ap.add_argument("--per-type", type=int, default=PER_TYPE_DEFAULT, help="每类采样数量")
    ap.add_argument("--city", default="", help="指定单个城市名；与 --all-regions / --cities 互斥")
    ap.add_argument(
        "--cities",
        default="",
        help="逗号或空格分隔的多城市，例如：广州市,北京市,上海市",
    )
    ap.add_argument(
        "--all-regions",
        action="store_true",
        help=f"一键生成默认多区域数据集（当前 {len(DEFAULT_MULTI_CITIES)} 个城市）",
    )
    ap.add_argument(
        "--out-root",
        default=os.path.join(_ROOT, "TestDatasets"),
        help="TestDatasets 根目录",
    )
    args = ap.parse_args()

    import pandas as pd

    if not os.path.isfile(args.csv):
        raise SystemExit(f"找不到 CSV: {args.csv}")

    df = pd.read_csv(args.csv)
    need = {"cityname", "query_category", "wgs84_lon", "wgs84_lat", "name"}
    miss = need - set(df.columns)
    if miss:
        raise SystemExit(f"CSV 缺少列: {miss}")

    modes = sum(
        1
        for x in (
            bool(args.city.strip()),
            bool(args.cities.strip()),
            bool(args.all_regions),
        )
        if x
    )
    if modes > 1:
        raise SystemExit("请只使用以下之一：--city、--cities 或 --all-regions")

    if args.all_regions:
        city_list = list(DEFAULT_MULTI_CITIES)
    elif args.cities.strip():
        raw = args.cities.replace(",", " ").split()
        city_list = [c.strip() for c in raw if c.strip()]
    elif args.city.strip():
        city_list = [args.city.strip()]
    else:
        city_list = [pick_city(df)]

    errors: List[str] = []
    written: List[str] = []
    for city in city_list:
        try:
            path = build_one_city(df, city, per_type=args.per_type, out_root=args.out_root)
            written.append(path)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{city}: {exc}")

    if errors:
        print("\n以下城市跳过或失败：")
        for e in errors:
            print(" ", e)
    if written:
        print(f"\n共生成 {len(written)} 个数据集目录（各含 points.geojson）。")


if __name__ == "__main__":
    main()
