from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import geopandas as gpd
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="比较两版点综合批处理结果并输出评价报告")
    parser.add_argument("--baseline-summary", required=True)
    parser.add_argument("--candidate-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def load_rows(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload.get("rows", []))


def compute_region_metrics(before_path: Path, after_path: Path) -> Dict[str, float]:
    before = gpd.read_file(before_path)
    after = gpd.read_file(after_path)
    if len(before) == 0 or len(after) == 0:
        return {
            "region_coverage_ratio": 0.0,
            "sparse_region_coverage_ratio": 0.0,
            "dense_region_singleton_ratio": 1.0,
        }

    if str(before.crs) != str(after.crs):
        after = after.to_crs(before.crs)
    if getattr(before.crs, "is_geographic", False):
        before = before.to_crs("EPSG:3857")
        after = after.to_crs("EPSG:3857")

    minx, miny, maxx, maxy = [float(v) for v in before.total_bounds]
    width = max(1e-9, maxx - minx)
    height = max(1e-9, maxy - miny)
    region_bins = max(3, min(10, int(round((max(6, len(after)) ** 0.5) * 1.15))))

    def region_series(gdf: gpd.GeoDataFrame) -> pd.Series:
        rx = ((gdf.geometry.x - minx) / width * region_bins).astype(int).clip(lower=0, upper=region_bins)
        ry = ((gdf.geometry.y - miny) / height * region_bins).astype(int).clip(lower=0, upper=region_bins)
        return pd.Series([f"{int(x)}_{int(y)}" for x, y in zip(rx.tolist(), ry.tolist())])

    before_regions = region_series(before)
    after_regions = region_series(after)
    before_freq = before_regions.value_counts(dropna=False)
    after_freq = after_regions.value_counts(dropna=False)

    before_set = set(str(v) for v in before_freq.index.tolist())
    after_set = set(str(v) for v in after_freq.index.tolist())
    region_coverage_ratio = float(len(before_set & after_set)) / float(max(1, len(before_set)))

    sparse_keys = [str(k) for k, v in before_freq.items() if int(v) <= 2]
    sparse_hits = [key for key in sparse_keys if key in after_set]
    sparse_region_coverage_ratio = float(len(sparse_hits)) / float(max(1, len(sparse_keys)))

    dense_threshold = max(8, int(round(float(before_freq.quantile(0.75))))) if len(before_freq) > 0 else 8
    dense_keys = [str(k) for k, v in before_freq.items() if int(v) >= dense_threshold]
    dense_singletons = sum(1 for key in dense_keys if int(after_freq.get(key, 0)) <= 1)
    dense_region_singleton_ratio = float(dense_singletons) / float(max(1, len(dense_keys)))

    return {
        "region_coverage_ratio": round(region_coverage_ratio, 4),
        "sparse_region_coverage_ratio": round(sparse_region_coverage_ratio, 4),
        "dense_region_singleton_ratio": round(dense_region_singleton_ratio, 4),
    }


def avg(values: Iterable[float]) -> float:
    values = list(values)
    return round(sum(values) / float(max(1, len(values))), 4)


def build_report(
    baseline_rows: List[Dict[str, Any]],
    candidate_rows: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    baseline_map = {(row["group_id"], int(row["target_scale"])): row for row in baseline_rows}
    candidate_map = {(row["group_id"], int(row["target_scale"])): row for row in candidate_rows}

    detail_rows: List[Dict[str, Any]] = []
    for key, cand in candidate_map.items():
        base = baseline_map[key]
        base_metrics = compute_region_metrics(Path(base["before_geojson"]), Path(base["after_geojson"]))
        cand_metrics = compute_region_metrics(Path(cand["before_geojson"]), Path(cand["after_geojson"]))
        detail_rows.append(
            {
                "group_id": cand["group_id"],
                "city": cand["city"],
                "region_name": cand["region_name"],
                "keyword": cand["keyword"],
                "target_scale": int(cand["target_scale"]),
                "baseline_after_count": int(base["after_count"]),
                "candidate_after_count": int(cand["after_count"]),
                "delta_after_count": int(cand["after_count"]) - int(base["after_count"]),
                "baseline_status": base["completion_status"],
                "candidate_status": cand["completion_status"],
                "baseline_region_coverage_ratio": base_metrics["region_coverage_ratio"],
                "candidate_region_coverage_ratio": cand_metrics["region_coverage_ratio"],
                "delta_region_coverage_ratio": round(
                    cand_metrics["region_coverage_ratio"] - base_metrics["region_coverage_ratio"], 4
                ),
                "baseline_sparse_region_coverage_ratio": base_metrics["sparse_region_coverage_ratio"],
                "candidate_sparse_region_coverage_ratio": cand_metrics["sparse_region_coverage_ratio"],
                "delta_sparse_region_coverage_ratio": round(
                    cand_metrics["sparse_region_coverage_ratio"] - base_metrics["sparse_region_coverage_ratio"], 4
                ),
                "baseline_dense_region_singleton_ratio": base_metrics["dense_region_singleton_ratio"],
                "candidate_dense_region_singleton_ratio": cand_metrics["dense_region_singleton_ratio"],
                "delta_dense_region_singleton_ratio": round(
                    cand_metrics["dense_region_singleton_ratio"] - base_metrics["dense_region_singleton_ratio"], 4
                ),
            }
        )

    scale_summary: Dict[int, Dict[str, Any]] = {}
    for scale in sorted({int(row["target_scale"]) for row in detail_rows}):
        items = [row for row in detail_rows if int(row["target_scale"]) == scale]
        scale_summary[scale] = {
            "count": len(items),
            "baseline_after_avg": avg(row["baseline_after_count"] for row in items),
            "candidate_after_avg": avg(row["candidate_after_count"] for row in items),
            "after_delta_avg": avg(row["delta_after_count"] for row in items),
            "baseline_region_coverage_avg": avg(row["baseline_region_coverage_ratio"] for row in items),
            "candidate_region_coverage_avg": avg(row["candidate_region_coverage_ratio"] for row in items),
            "baseline_sparse_region_coverage_avg": avg(
                row["baseline_sparse_region_coverage_ratio"] for row in items
            ),
            "candidate_sparse_region_coverage_avg": avg(
                row["candidate_sparse_region_coverage_ratio"] for row in items
            ),
            "baseline_dense_singleton_avg": avg(
                row["baseline_dense_region_singleton_ratio"] for row in items
            ),
            "candidate_dense_singleton_avg": avg(
                row["candidate_dense_region_singleton_ratio"] for row in items
            ),
        }

    overall = {
        "run_count": len(detail_rows),
        "baseline_status_counts": dict(Counter(row["baseline_status"] for row in detail_rows)),
        "candidate_status_counts": dict(Counter(row["candidate_status"] for row in detail_rows)),
        "improved_region_coverage_runs": sum(
            1 for row in detail_rows if row["delta_region_coverage_ratio"] > 0
        ),
        "improved_sparse_region_runs": sum(
            1 for row in detail_rows if row["delta_sparse_region_coverage_ratio"] > 0
        ),
        "reduced_dense_singleton_runs": sum(
            1 for row in detail_rows if row["delta_dense_region_singleton_ratio"] < 0
        ),
        "scale_summary": scale_summary,
    }
    return detail_rows, overall


def write_outputs(output_dir: Path, detail_rows: List[Dict[str, Any]], overall: Dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "evaluation.json").write_text(
        json.dumps({"overall": overall, "rows": detail_rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if detail_rows:
        fieldnames = list(detail_rows[0].keys())
        with (output_dir / "evaluation.csv").open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(detail_rows)

    md = [
        "# Point Generalization V1 vs V2 Evaluation",
        "",
        f"- run_count: {overall['run_count']}",
        f"- baseline_status_counts: {overall['baseline_status_counts']}",
        f"- candidate_status_counts: {overall['candidate_status_counts']}",
        f"- improved_region_coverage_runs: {overall['improved_region_coverage_runs']}",
        f"- improved_sparse_region_runs: {overall['improved_sparse_region_runs']}",
        f"- reduced_dense_singleton_runs: {overall['reduced_dense_singleton_runs']}",
        "",
        "| 比例尺 | 样本数 | v1平均点数 | v2平均点数 | 平均增量 | v1区域覆盖 | v2区域覆盖 | v1稀疏区覆盖 | v2稀疏区覆盖 | v1密集区单点塌缩 | v2密集区单点塌缩 |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for scale, stats in sorted(overall["scale_summary"].items()):
        md.append(
            f"| 1:{scale} | {stats['count']} | {stats['baseline_after_avg']} | {stats['candidate_after_avg']} | "
            f"{stats['after_delta_avg']} | {stats['baseline_region_coverage_avg']} | {stats['candidate_region_coverage_avg']} | "
            f"{stats['baseline_sparse_region_coverage_avg']} | {stats['candidate_sparse_region_coverage_avg']} | "
            f"{stats['baseline_dense_singleton_avg']} | {stats['candidate_dense_singleton_avg']} |"
        )
    (output_dir / "evaluation.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    baseline_rows = load_rows(Path(args.baseline_summary))
    candidate_rows = load_rows(Path(args.candidate_summary))
    detail_rows, overall = build_report(baseline_rows, candidate_rows)
    write_outputs(Path(args.output_dir), detail_rows, overall)
    print(Path(args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
