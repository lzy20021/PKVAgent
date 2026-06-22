from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MPLBACKEND", "Agg")

import geopandas as gpd

from scripts.run_point_generalization_skill_check import (
    add_qualification_flags,
    evaluate,
    normalize_input,
    parse_scales,
    register_local_vlm_callback,
    run_agent,
    verdict,
)


DEFAULT_SCALES = [86000, 324000, 625000]
DEFAULT_DATASET_ROOT = "TestDatasets"
DEFAULT_OUTPUT_ROOT = "tests/output/testdatasets_agent_layout_batch"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run real MapGenAgent point generalization on TestDatasets and validate exported layouts."
    )
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--scales", default=",".join(str(s) for s in DEFAULT_SCALES))
    parser.add_argument("--limit-per-scale", type=int, default=20)
    parser.add_argument("--source-scale", type=int, default=10000)
    parser.add_argument("--max-features", type=int, default=900)
    parser.add_argument(
        "--require-real-vlm",
        action="store_true",
        default=True,
        help="Use the registered local VLM callback path for every agent run.",
    )
    return parser.parse_args()


def sample_inputs(dataset_root: Path, scale: int, limit: int) -> List[Path]:
    scale_dir = dataset_root / f"Scale_1_{scale}"
    files = sorted(scale_dir.glob("sample_??.geojson"))
    if limit > 0:
        files = files[:limit]
    return files


def read_image_size(path: Path) -> tuple[int, int]:
    try:
        from PIL import Image

        with Image.open(path) as img:
            return int(img.width), int(img.height)
    except Exception:
        try:
            import matplotlib.image as mpimg

            arr = mpimg.imread(path)
            return int(arr.shape[1]), int(arr.shape[0])
        except Exception:
            return 0, 0


def layout_quality(output_png: Path, output_pdf: Path) -> Dict[str, Any]:
    png_exists = output_png.is_file() and output_png.stat().st_size > 0
    pdf_exists = output_pdf.is_file() and output_pdf.stat().st_size > 0
    width, height = read_image_size(output_png) if png_exists else (0, 0)
    min_side = min(width, height)
    max_side = max(width, height)
    resolution_ok = int(min_side >= 2400 and max_side >= 3300)
    return {
        "layout_png_exists": int(png_exists),
        "layout_pdf_exists": int(pdf_exists),
        "layout_png_width": width,
        "layout_png_height": height,
        "layout_resolution_qualified": resolution_ok,
        "layout_export_qualified": int(png_exists and pdf_exists and resolution_ok),
    }


def no_24_artifact(after_path: Path) -> int:
    try:
        gdf = gpd.read_file(after_path)
    except Exception:
        return 0
    suspicious_fields = [
        "point_symbol",
        "point_symbol_label",
        "poi_category",
        "query_category",
        "category",
        "type",
    ]
    for field in suspicious_fields:
        if field in gdf.columns:
            values = set(gdf[field].dropna().astype(str).str.strip().tolist())
            if "24" in values or "$24$" in values:
                return 0
    return 1


def add_strict_flags(row: Dict[str, Any]) -> None:
    target_scale = int(row.get("target_scale", 0) or 0)
    retention = float(row.get("retention_rate", 0.0) or 0.0)
    min_retention = 0.07 if target_scale >= 500000 else (0.12 if target_scale >= 200000 else 0.25)
    max_retention = 0.70 if target_scale < 200000 else (0.34 if target_scale < 500000 else 0.18)
    row["retention_qualified"] = int(min_retention <= retention <= max_retention)
    row["category_coverage_qualified"] = int(float(row.get("category_coverage_ratio", 0) or 0) >= 1.0)
    row["region_coverage_qualified"] = int(float(row.get("region_coverage_ratio", 0) or 0) >= 0.72)
    row["symbol_qualified"] = int(
        int(row.get("recognized_symbol_count", 0) or 0)
        >= min(
            int(row.get("expected_symbol_count", 0) or 0),
            max(1, len(row.get("after_categories", {}) or {})),
        )
    )
    row["conflict_qualified"] = int(int(row.get("conflict_count", 0) or 0) == 0)
    row["reduction_qualified"] = int(int(row.get("output_count", 0) or 0) < int(row.get("input_count", 0) or 0))
    strict_fields = [
        "retention_qualified",
        "category_coverage_qualified",
        "region_coverage_qualified",
        "symbol_qualified",
        "conflict_qualified",
        "layout_export_qualified",
        "layout_resolution_qualified",
        "no_24_qualified",
        "reduction_qualified",
    ]
    row["all_qualified"] = int(all(int(row.get(field, 0) or 0) == 1 for field in strict_fields))
    row["verdict"] = "PASS" if row["all_qualified"] else "FAIL"


def write_summary(output_root: Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / "testdatasets_agent_layout_summary.json"
    csv_path = output_root / "testdatasets_agent_layout_summary.csv"
    md_path = output_root / "testdatasets_agent_layout_summary.md"
    json_path.write_text(
        json.dumps(
            {
                "run_count": len(rows),
                "pass_count": sum(int(r.get("all_qualified", 0) or 0) for r in rows),
                "scales": sorted({int(r["target_scale"]) for r in rows}),
                "rows": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    fieldnames = [
        "sample_id",
        "target_scale",
        "input_count",
        "output_count",
        "retention_rate",
        "category_coverage_ratio",
        "region_coverage_ratio",
        "conflict_count",
        "recognized_symbol_count",
        "expected_symbol_count",
        "layout_png_width",
        "layout_png_height",
        "retention_qualified",
        "category_coverage_qualified",
        "region_coverage_qualified",
        "symbol_qualified",
        "conflict_qualified",
        "layout_export_qualified",
        "layout_resolution_qualified",
        "no_24_qualified",
        "reduction_qualified",
        "all_qualified",
        "verdict",
        "raw_input",
        "normalized_input",
        "output_geojson",
        "output_png",
        "output_pdf",
        "agent_log",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    lines = [
        "# TestDatasets Agent Layout Batch",
        "",
        f"- run_count: {len(rows)}",
        f"- pass_count: {sum(int(r.get('all_qualified', 0) or 0) for r in rows)}",
        "",
        "| sample | scale | before | after | retention | class | region | conflicts | layout | no24 | all |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['sample_id']} | 1:{row['target_scale']} | {row['input_count']} | {row['output_count']} | "
            f"{row['retention_rate']} | {row['category_coverage_qualified']} | {row['region_coverage_qualified']} | "
            f"{row['conflict_count']} | {row['layout_export_qualified']} | {row['no_24_qualified']} | {row['all_qualified']} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    dataset_root = (PROJECT_ROOT / args.dataset_root).resolve()
    output_root = (PROJECT_ROOT / args.output_root).resolve()
    scales = parse_scales(args.scales)
    register_local_vlm_callback()
    rows: List[Dict[str, Any]] = []
    for scale in scales:
        files = sample_inputs(dataset_root, scale, args.limit_per_scale)
        if not files:
            raise FileNotFoundError(dataset_root / f"Scale_1_{scale}")
        for input_path in files:
            sample_id = input_path.stem
            sample_root = output_root / f"Scale_1_{scale}" / sample_id
            sample_root.mkdir(parents=True, exist_ok=True)
            print(f"[agent] {input_path} -> 1:{scale}")
            normalized = normalize_input(input_path, sample_root, args.max_features)
            run = run_agent(
                input_geojson=normalized,
                output_root=sample_root,
                source_scale=args.source_scale,
                target_scale=scale,
                require_real_vlm=bool(args.require_real_vlm),
            )
            after_path = Path(run["output_geojson"])
            row = evaluate(normalized, after_path, scale)
            row.update(run)
            row["target_scale"] = scale
            row["sample_id"] = sample_id
            row["raw_input"] = str(input_path)
            row["normalized_input"] = str(normalized)
            output_png = Path(run.get("output_png") or sample_root / f"scale_1_{scale}" / f"poi8_gen_1_{scale}.png")
            output_pdf = output_png.with_suffix(".pdf")
            row["output_png"] = str(output_png)
            row["output_pdf"] = str(output_pdf)
            row.update(layout_quality(output_png, output_pdf))
            row["no_24_qualified"] = no_24_artifact(after_path)
            add_strict_flags(row)
            rows.append(row)
            write_summary(output_root, rows)
            print(
                f"[{row['verdict']}] {sample_id} 1:{scale}: "
                f"{row['input_count']}->{row['output_count']} conflicts={row['conflict_count']} "
                f"layout={row['layout_export_qualified']} all={row['all_qualified']}"
            )
    write_summary(output_root, rows)
    print(output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
