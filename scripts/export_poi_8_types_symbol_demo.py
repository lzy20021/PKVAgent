"""
从 all_poi_expanded.csv 抽取 8 类 POI 样本，导出为 8 个 GeoJSON，
每类对应一种点符号（与界面「点符号」选项一致），便于在 MapGenAI 中加载测试。

八类与 query_category 对应关系：
  便利店 / 学校 / 停车场 / 酒店 / 银行 / 医院 —— CSV 中已有
  餐饮 —— 合并 餐饮、餐厅、咖啡厅
  交通设施 —— 合并 公交站、地铁站

用法:
  python scripts/export_poi_8_types_symbol_demo.py
  python scripts/export_poi_8_types_symbol_demo.py --csv E:\\path\\to\\all_poi_expanded.csv --out E:\\out_dir --per-type 400
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# 保证可从项目根目录运行
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description="导出八类 POI GeoJSON（每类一种符号）")
    parser.add_argument(
        "--csv",
        default=os.path.join(_ROOT, "amap_poi_dataset_expanded", "all_poi_expanded.csv"),
        help="POI CSV 路径",
    )
    parser.add_argument(
        "--out",
        default=os.path.join(_ROOT, "amap_poi_dataset_expanded", "poi_8_types_symbol_demo"),
        help="输出目录",
    )
    parser.add_argument("--per-type", type=int, default=400, help="每类最多导出点数（均匀抽样）")
    args = parser.parse_args()

    import pandas as pd

    if not os.path.isfile(args.csv):
        raise SystemExit(f"找不到 CSV: {args.csv}")

    df = pd.read_csv(args.csv)
    if "query_category" not in df.columns or "wgs84_lon" not in df.columns or "wgs84_lat" not in df.columns:
        raise SystemExit("CSV 需包含列: query_category, wgs84_lon, wgs84_lat")

    # (输出文件名前缀, 符号名, 匹配的 query_category 集合)
    specs = [
        ("poi_便利店", "circle", {"便利店"}),
        ("poi_学校", "square", {"学校"}),
        ("poi_停车场", "triangle", {"停车场"}),
        ("poi_酒店", "diamond", {"酒店"}),
        ("poi_银行", "cross", {"银行"}),
        ("poi_医院", "star", {"医院"}),
        ("poi_餐饮", "plus", {"餐饮", "餐厅", "咖啡厅"}),
        ("poi_交通设施", "hexagon", {"公交站", "地铁站"}),
    ]

    os.makedirs(args.out, exist_ok=True)
    manifest = []

    for base_name, symbol, cats in specs:
        sub = df[df["query_category"].isin(cats)]
        if sub.empty:
            print(f"[跳过] {base_name}: 无数据 (categories={cats})")
            continue
        n = min(args.per_type, len(sub))
        if len(sub) > n:
            sub = sub.sample(n, random_state=42)
        features = []
        for _, row in sub.iterrows():
            lon, lat = float(row["wgs84_lon"]), float(row["wgs84_lat"])
            props = {
                "poi_uid": row.get("poi_uid", ""),
                "name": row.get("name", ""),
                "query_category": row.get("query_category", ""),
                "demo_group": base_name.replace("poi_", ""),
                "point_symbol": symbol,
            }
            features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [lon, lat]},
                    "properties": props,
                }
            )
        fc = {"type": "FeatureCollection", "features": features}
        path = os.path.join(args.out, f"{base_name}.geojson")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(fc, f, ensure_ascii=False)
        manifest.append(
            {
                "file": path,
                "symbol": symbol,
                "count": len(features),
                "categories": sorted(cats),
            }
        )
        print(f"写入 {path}  ({len(features)} 点, 符号={symbol})")

    meta_path = os.path.join(args.out, "manifest.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"\n说明: 在 MapGenAI 中依次加载上述 GeoJSON；在「修改样式」中为点图层选择对应「点符号」。")
    print(f"符号对照见 manifest.json（属性里 point_symbol 仅作标注，渲染以图层样式为准）。")


if __name__ == "__main__":
    main()
