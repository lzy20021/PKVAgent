from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.agent.core import MapGenAgent
from src.agent.providers import BaseLLMProvider, LLMResponse


DEFAULT_SCALES = [135000, 220000, 325000]


class NullProvider(BaseLLMProvider):
    def __init__(self) -> None:
        super().__init__(api_key="", model="null")

    def chat(self, messages, tools=None) -> LLMResponse:  # type: ignore[override]
        return LLMResponse("")

    def stream_chat(self, messages, tools=None):  # type: ignore[override]
        if False:
            yield ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MapGenAI point generalization in batch and export before/after artifacts."
    )
    parser.add_argument(
        "--summary-path",
        default="tests/data/amap_poi_batches/_summary.json",
        help="Input summary json path.",
    )
    parser.add_argument(
        "--output-root",
        default="tests/output/agent_point_generalization_batch",
        help="Output directory for batch results.",
    )
    parser.add_argument(
        "--limit-groups",
        type=int,
        default=100,
        help="Max number of groups to process.",
    )
    parser.add_argument(
        "--source-scale",
        type=int,
        default=10000,
        help="Source scale denominator.",
    )
    parser.add_argument(
        "--scales",
        default=",".join(str(v) for v in DEFAULT_SCALES),
        help="target scales, comma separated (example: 120000,220000,532000)",
    )
    return parser.parse_args()

def parse_scales(scales_text: str) -> List[int]:
    values: List[int] = []
    for part in str(scales_text).split(","):
        token = part.strip()
        if not token:
            continue
        value = int(token)
        if value <= 0:
            raise ValueError(f"invalid scale: {value}")
        values.append(value)
    if not values:
        raise ValueError("no valid scales were provided")
    return values

def load_group_rows(summary_path: Path, limit_groups: int) -> List[Dict[str, Any]]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = list(payload.get("results", []))
    rows = [row for row in rows if str(row.get("output_path", "")).lower().endswith(".geojson")]
    return rows[:limit_groups]


def render_comparison(before_path: Path, after_path: Path, compare_png: Path, title: str) -> None:
    before_gdf = gpd.read_file(before_path)
    after_gdf = gpd.read_file(after_path)

    bounds = before_gdf.total_bounds
    minx, miny, maxx, maxy = [float(v) for v in bounds]
    dx = max(1e-9, maxx - minx)
    dy = max(1e-9, maxy - miny)
    pad_x = dx * 0.05
    pad_y = dy * 0.05

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, gdf, subtitle, color in [
        (axes[0], before_gdf, f"Before ({len(before_gdf)})", "#1f77b4"),
        (axes[1], after_gdf, f"After ({len(after_gdf)})", "#d81b60"),
    ]:
        gdf.plot(ax=ax, color=color, markersize=10)
        ax.set_xlim(minx - pad_x, maxx + pad_x)
        ax.set_ylim(miny - pad_y, maxy + pad_y)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(subtitle)
        ax.set_axis_off()

    fig.suptitle(title)
    compare_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(compare_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def run_one(
    agent: MapGenAgent,
    source_row: Dict[str, Any],
    source_scale: int,
    target_scale: int,
    output_root: Path,
) -> Dict[str, Any]:
    group_id = str(source_row["group_id"])
    city = str(source_row.get("city", ""))
    region_name = str(source_row.get("region_name", ""))
    keyword = str(source_row.get("keyword", ""))
    source_geojson = PROJECT_ROOT / Path(str(source_row["output_path"]))

    scale_dir = output_root / f"scale_1_{target_scale}" / group_id
    scale_dir.mkdir(parents=True, exist_ok=True)

    before_geojson = scale_dir / f"{group_id}_before.geojson"
    after_geojson = scale_dir / f"{group_id}_after.geojson"
    after_png = scale_dir / f"{group_id}_after.png"
    compare_png = scale_dir / f"{group_id}_compare.png"
    log_txt = scale_dir / f"{group_id}_agent_log.txt"

    shutil.copy2(source_geojson, before_geojson)

    task = {
        "parameters": {
            "input_file": str(before_geojson),
            "target_scale": target_scale,
            "source_scale": source_scale,
            "feature_count": int(source_row.get("feature_count", 0) or 0),
            "geometry_type": "point",
            "layer_name": group_id,
            "output_formats": ["geojson", "png"],
            "output_geojson": str(after_geojson),
            "output_png": str(after_png),
        }
    }

    started = time.time()
    reply = agent._run_scale_generalization(task) or ""
    elapsed = time.time() - started
    log_txt.write_text(reply, encoding="utf-8")

    if not after_geojson.exists():
        raise RuntimeError(f"after geojson not generated: {after_geojson}")

    render_comparison(
        before_path=before_geojson,
        after_path=after_geojson,
        compare_png=compare_png,
        title=f"{group_id} 1:{target_scale}",
    )

    before_count = len(gpd.read_file(before_geojson))
    after_count = len(gpd.read_file(after_geojson))
    completion_status = "unknown"
    for line in reply.splitlines():
        if line.lower().startswith("completion status:"):
            completion_status = line.split(":", 1)[1].strip()
            break

    return {
        "group_id": group_id,
        "city": city,
        "region_name": region_name,
        "keyword": keyword,
        "target_scale": target_scale,
        "before_count": before_count,
        "after_count": after_count,
        "completion_status": completion_status,
        "elapsed_seconds": round(elapsed, 3),
        "before_geojson": str(before_geojson),
        "after_geojson": str(after_geojson),
        "after_png": str(after_png),
        "compare_png": str(compare_png),
        "agent_log": str(log_txt),
    }


def write_summary(output_root: Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    summary_json = output_root / "_batch_summary.json"
    summary_csv = output_root / "_batch_summary.csv"
    summary_md = output_root / "_batch_summary.md"

    payload = {
        "run_count": len(rows),
        "group_count": len({row["group_id"] for row in rows}),
        "scales": sorted({row["target_scale"] for row in rows}),
        "rows": rows,
    }
    summary_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    fieldnames = [
        "group_id",
        "city",
        "region_name",
        "keyword",
        "target_scale",
        "before_count",
        "after_count",
        "completion_status",
        "elapsed_seconds",
        "before_geojson",
        "after_geojson",
        "after_png",
        "compare_png",
        "agent_log",
    ]
    with summary_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Agent Point Generalization Batch Summary",
        "",
        f"- run_count: {len(rows)}",
        f"- group_count: {len({row['group_id'] for row in rows})}",
        f"- scales: {sorted({row['target_scale'] for row in rows})}",
        "",
        "| group_id | 鍩庡競 | 鍖哄煙 | 绫诲瀷 | 鐩爣姣斾緥灏?| 缁煎悎鍓?| 缁煎悎鍚?| 鐘舵€?| 瀵规瘮鍥?|",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['group_id']} | {row['city']} | {row['region_name']} | {row['keyword']} | "
            f"1:{row['target_scale']} | {row['before_count']} | {row['after_count']} | "
            f"{row['completion_status']} | {Path(row['compare_png']).name} |"
        )
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    scales = parse_scales(args.scales)
    summary_path = PROJECT_ROOT / Path(args.summary_path)
    output_root = PROJECT_ROOT / Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    group_rows = load_group_rows(summary_path, args.limit_groups)
    agent = MapGenAgent(provider=NullProvider())
    agent.set_workspace(str(output_root))

    results: List[Dict[str, Any]] = []
    total = len(group_rows) * len(scales)
    current = 0
    for row in group_rows:
        for target_scale in scales:
            current += 1
            print(f"[{current}/{total}] {row['group_id']} -> 1:{target_scale}")
            result = run_one(
                agent=agent,
                source_row=row,
                source_scale=args.source_scale,
                target_scale=target_scale,
                output_root=output_root,
            )
            results.append(result)

    write_summary(output_root, results)
    print(output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

