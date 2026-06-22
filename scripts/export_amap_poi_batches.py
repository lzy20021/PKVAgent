from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.io.amap_poi import build_default_groups, export_groups, write_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="批量抓取高德 POI，并导出为 GeoJSON")
    parser.add_argument("--api-key", help="高德 Web 服务 API Key；未传时读取 AMAP_API_KEY")
    parser.add_argument(
        "--output-dir",
        default="tests/data/amap_poi_batches",
        help="GeoJSON 输出目录",
    )
    parser.add_argument("--summary-path", default="tests/data/amap_poi_batches/_summary.json")
    parser.add_argument("--limit-groups", type=int, default=100, help="最多抓取多少组")
    parser.add_argument("--min-points", type=int, default=100)
    parser.add_argument("--max-points", type=int, default=1000)
    parser.add_argument("--max-pages", type=int, default=40)
    parser.add_argument("--sleep-seconds", type=float, default=0.2)
    parser.add_argument("--target-crs", choices=["WGS84", "GCJ-02"], default="WGS84")
    parser.add_argument(
        "--only-group-ids",
        nargs="+",
        default=None,
        help="仅导出指定 group_id，可传多个",
    )
    parser.add_argument(
        "--expand-bbox",
        type=float,
        default=0.0,
        help="对每个分组 bbox 四周统一扩展的经纬度值，例如 0.01",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def expand_bbox(bbox: tuple[float, float, float, float], delta: float) -> tuple[float, float, float, float]:
    min_lon, min_lat, max_lon, max_lat = bbox
    return (min_lon - delta, min_lat - delta, max_lon + delta, max_lat + delta)


def main() -> int:
    args = parse_args()
    api_key = args.api_key or os.getenv("AMAP_API_KEY")
    if not api_key:
        raise SystemExit("缺少 API key：请使用 --api-key 或环境变量 AMAP_API_KEY")

    groups = build_default_groups()
    if args.only_group_ids:
        group_ids = set(args.only_group_ids)
        groups = [group for group in groups if group.group_id in group_ids]
    if args.expand_bbox > 0:
        groups = [
            type(group)(
                group_id=group.group_id,
                city=group.city,
                region_name=group.region_name,
                keyword=group.keyword,
                theme=group.theme,
                bbox=expand_bbox(group.bbox, args.expand_bbox),
            )
            for group in groups
        ]
    groups = groups[: args.limit_groups]
    results = export_groups(
        api_key=api_key,
        output_dir=Path(args.output_dir),
        groups=groups,
        target_crs=args.target_crs,
        min_points=args.min_points,
        max_points=args.max_points,
        max_pages=args.max_pages,
        overwrite=args.overwrite,
        sleep_seconds=args.sleep_seconds,
    )
    write_summary(results, Path(args.summary_path))

    ok_count = sum(1 for item in results if item.status == "ok")
    underfilled_count = sum(1 for item in results if item.status == "underfilled")
    error_count = sum(1 for item in results if item.status == "error")
    skipped_count = sum(1 for item in results if item.status == "skipped_existing")
    print(
        f"groups={len(results)} ok={ok_count} underfilled={underfilled_count} error={error_count} "
        f"skipped={skipped_count} summary={args.summary_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
