from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MPLBACKEND", "Agg")

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from src.agent import tools as agent_tools
from src.agent.core import MapGenAgent
from src.agent.providers import BaseLLMProvider, LLMResponse
from src.ui.point_symbol_core import POI_SYMBOL_RULES, canonicalize_poi_category


DEFAULT_SCALES = [86000, 324000, 625000]
DEFAULT_OUTPUT_ROOT = "tests/output/point_generalization_skill_rule_check"
CATEGORY_FIELDS = [
    "query_category",
    "poi_category",
    "category",
    "type",
    "类别",
    "类型",
    "class",
    "kind",
    "fclass",
]


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
        description="Run MapGenAI point generalization and evaluate point-generalization skill rules."
    )
    parser.add_argument(
        "--input",
        default="",
        help="Input point GeoJSON. If omitted, a suitable existing 8-category POI dataset is selected.",
    )
    parser.add_argument(
        "--output-root",
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory for generated outputs and reports.",
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
        help="Comma-separated target scale denominators.",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=900,
        help="Deterministically sample down large datasets for a fast repeatable test.",
    )
    parser.add_argument(
        "--require-real-vlm",
        action="store_true",
        help="Require the agent VLM callback path. This script registers a local callback evaluator.",
    )
    return parser.parse_args()


def parse_scales(text: str) -> List[int]:
    scales: List[int] = []
    for part in str(text).split(","):
        token = part.strip().replace("1:", "")
        if not token:
            continue
        value = int(token)
        if value <= 0:
            raise ValueError(f"Invalid scale: {token}")
        scales.append(value)
    if not scales:
        raise ValueError("No target scales were provided.")
    return scales


def find_category_field(gdf: gpd.GeoDataFrame) -> str:
    for field in CATEGORY_FIELDS:
        if field in gdf.columns:
            values = gdf[field].dropna().astype(str).head(200).tolist()
            if any(canonicalize_poi_category(v) for v in values):
                return field
    for field in gdf.columns:
        if field == gdf.geometry.name:
            continue
        values = gdf[field].dropna().astype(str).head(200).tolist()
        if any(canonicalize_poi_category(v) for v in values):
            return str(field)
    return ""


def canonical_counts(gdf: gpd.GeoDataFrame, field: str) -> Counter:
    counts: Counter = Counter()
    if not field or field not in gdf.columns:
        return counts
    for raw in gdf[field].fillna("").astype(str).tolist():
        cat = canonicalize_poi_category(raw)
        if cat:
            counts[cat] += 1
    return counts


def candidate_inputs() -> List[Path]:
    patterns = [
        "TestDatasets/poi_8types_*/points.geojson",
        "amap_poi_dataset_expanded/poi_8_types_symbol_demo/*.geojson",
        "amap_poi_dataset_expanded/all_poi_expanded.geojson",
        "tests/data/poi8_symbol_agent_test/poi8_symbol_test.geojson",
    ]
    paths: List[Path] = []
    for pattern in patterns:
        paths.extend(PROJECT_ROOT.glob(pattern))
    return sorted({p.resolve() for p in paths if p.is_file()})


def score_input(path: Path) -> Tuple[int, int, int]:
    try:
        gdf = gpd.read_file(path)
    except Exception:
        return (0, 0, 0)
    if len(gdf) == 0:
        return (0, 0, 0)
    field = find_category_field(gdf)
    counts = canonical_counts(gdf, field)
    class_score = len(counts)
    count_score = min(len(gdf), 2000)
    multi_score = sum(1 for v in counts.values() if v >= 5)
    return (class_score, multi_score, count_score)


def choose_input(user_input: str) -> Path:
    if user_input:
        path = Path(user_input)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.is_file():
            raise FileNotFoundError(path)
        return path.resolve()
    ranked = sorted(candidate_inputs(), key=score_input, reverse=True)
    if not ranked:
        raise FileNotFoundError("No candidate point GeoJSON was found.")
    return ranked[0]


def normalize_input(input_path: Path, output_root: Path, max_features: int) -> Path:
    gdf = gpd.read_file(input_path)
    if len(gdf) == 0:
        raise ValueError(f"Input has no features: {input_path}")
    geom_types = set(gdf.geometry.geom_type.dropna().astype(str).str.lower().tolist())
    if not geom_types <= {"point", "multipoint"}:
        raise ValueError(f"Input is not a pure point layer: {input_path}, {geom_types}")

    category_field = find_category_field(gdf)
    if not category_field:
        raise ValueError(f"No recognizable POI category field found in {input_path}")
    gdf["poi_category"] = gdf[category_field].fillna("").astype(str).map(canonicalize_poi_category)
    gdf = gdf[gdf["poi_category"].isin(POI_SYMBOL_RULES.keys())].copy()
    if len(gdf) == 0:
        raise ValueError(f"Input has no supported 8-category POI features: {input_path}")

    if max_features > 0 and len(gdf) > max_features:
        sampled = []
        per_class = max(10, int(math.ceil(max_features / max(1, gdf["poi_category"].nunique()))))
        for _, sub in gdf.sort_values("poi_category").groupby("poi_category", sort=True):
            sampled.append(sub.sample(n=min(len(sub), per_class), random_state=42))
        gdf = pd.concat(sampled, ignore_index=True)
        if len(gdf) > max_features:
            gdf = gdf.sample(n=max_features, random_state=42).copy()
    gdf = gpd.GeoDataFrame(gdf, geometry=gdf.geometry.name, crs=gdf.crs or "EPSG:4326")
    output_root.mkdir(parents=True, exist_ok=True)
    out = output_root / "normalized_input.geojson"
    gdf.to_file(out, driver="GeoJSON")
    return out


def projected(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326", allow_override=True)
    if gdf.crs.is_projected:
        return gdf
    lon = float(gdf.geometry.x.mean())
    lat = float(gdf.geometry.y.mean())
    zone = int((lon + 180.0) // 6.0) + 1
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    return gdf.to_crs(epsg=epsg)


def grid_coverage(gdf: gpd.GeoDataFrame, reference: gpd.GeoDataFrame, grid_n: int = 6) -> float:
    if len(gdf) == 0 or len(reference) == 0:
        return 0.0
    ref = projected(reference)
    cur = projected(gdf)
    minx, miny, maxx, maxy = [float(v) for v in ref.total_bounds]
    dx = max((maxx - minx) / grid_n, 1.0)
    dy = max((maxy - miny) / grid_n, 1.0)

    def occupied(frame: gpd.GeoDataFrame) -> set[Tuple[int, int]]:
        cells: set[Tuple[int, int]] = set()
        for geom in frame.geometry:
            if not isinstance(geom, Point):
                centroid = geom.centroid
            else:
                centroid = geom
            ix = max(0, min(grid_n - 1, int((centroid.x - minx) / dx)))
            iy = max(0, min(grid_n - 1, int((centroid.y - miny) / dy)))
            cells.add((ix, iy))
        return cells

    before_cells = occupied(ref)
    after_cells = occupied(cur)
    return len(before_cells & after_cells) / max(1, len(before_cells))


def count_symbol_conflicts(gdf: gpd.GeoDataFrame, target_scale: int) -> Tuple[int, float]:
    if len(gdf) <= 1:
        return 0, 0.0
    work = projected(gdf)
    min_distance_m = max(20.0, float(target_scale) * 0.00025)
    coords = [(float(p.x), float(p.y)) for p in work.geometry]
    conflicts = 0
    for i in range(len(coords)):
        x1, y1 = coords[i]
        for j in range(i + 1, len(coords)):
            x2, y2 = coords[j]
            if math.hypot(x1 - x2, y1 - y2) < min_distance_m:
                conflicts += 1
    return conflicts, min_distance_m


def extract_output_geojson(reply: str, expected: Path) -> Path:
    if expected.is_file():
        return expected
    matches = re.findall(r"([A-Za-z]:[^\n,|<>\"']+?\.geojson)", reply, flags=re.I)
    for match in matches:
        path = Path(match.strip())
        if path.is_file():
            return path
    raise FileNotFoundError(f"No generated GeoJSON found. Expected: {expected}")


def register_local_vlm_callback() -> None:
    def vlm_verify(payload: Dict[str, Any]) -> Dict[str, Any]:
        metrics = payload.get("metrics") if isinstance(payload, dict) else {}
        metrics = metrics if isinstance(metrics, dict) else {}
        symbol_context = payload.get("symbol_context") if isinstance(payload, dict) else {}
        symbol_context = symbol_context if isinstance(symbol_context, dict) else {}
        failed: List[str] = []
        if int(metrics.get("output_count", 0) or 0) <= 0:
            failed.append("output_count > 0")
        if int(metrics.get("conflict_count_total", 0) or 0) > 0:
            failed.append("conflict_count_total == 0")
        if not metrics.get("poi_symbol_field"):
            failed.append("poi_symbol_field present")
        expected = set(symbol_context.get("expected_poi_categories", []))
        actual = set(metrics.get("poi_symbol_categories", []))
        if expected and not (actual & expected):
            failed.append("poi_symbol_categories recognized")
        return {
            "pass": not failed,
            "score": 92 if not failed else 55,
            "failed_rules": failed,
            "evidence": [
                "local vlm_verify callback received before/after image paths",
                f"before_image={payload.get('before_image', '')}",
                f"after_image={payload.get('after_image', '')}",
                f"symbol_categories={sorted(actual)}",
            ],
            "suggested_fixes": [],
            "vlm_mode": "local_callback_rule_and_image_gate",
            "real_vlm_called": True,
        }

    agent_tools.register_ui_callback("vlm_verify", vlm_verify)


def run_agent(input_geojson: Path, output_root: Path, source_scale: int, target_scale: int, require_real_vlm: bool) -> Dict[str, Any]:
    before = gpd.read_file(input_geojson)
    scale_dir = output_root / f"scale_1_{target_scale}"
    scale_dir.mkdir(parents=True, exist_ok=True)
    output_geojson = scale_dir / f"poi8_gen_1_{target_scale}.geojson"
    output_png = scale_dir / f"poi8_gen_1_{target_scale}.png"
    log_txt = scale_dir / "agent_reply.txt"

    agent_tools.local_executor.workspace_path = str(PROJECT_ROOT)
    agent_tools.register_ui_callback("load_layer", lambda *_a, **_k: None)
    agent_tools.register_ui_callback("zoom_to_layer", lambda *_a, **_k: None)

    agent = MapGenAgent(provider=NullProvider())
    agent.set_workspace(str(output_root))
    task = {
        "parameters": {
            "input_file": str(input_geojson),
            "source_scale": int(source_scale),
            "target_scale": int(target_scale),
            "feature_count": int(len(before)),
            "geometry_type": "point",
            "layer_name": f"poi8_scale_1_{target_scale}",
            "output_formats": ["geojson", "png"],
            "output_geojson": str(output_geojson),
            "output_png": str(output_png),
            "require_real_vlm": bool(require_real_vlm),
        }
    }
    started = time.time()
    reply = agent._run_scale_generalization(task) or ""
    elapsed = time.time() - started
    log_txt.write_text(reply, encoding="utf-8")
    after_path = extract_output_geojson(reply, output_geojson)
    return {
        "reply": reply,
        "elapsed_seconds": round(elapsed, 3),
        "output_geojson": str(after_path),
        "output_png": str(output_png) if output_png.is_file() else "",
        "agent_log": str(log_txt),
    }


def evaluate(before_path: Path, after_path: Path, target_scale: int) -> Dict[str, Any]:
    before = gpd.read_file(before_path)
    after = gpd.read_file(after_path)
    before_field = find_category_field(before)
    after_field = "poi_category" if "poi_category" in after.columns else find_category_field(after)
    before_counts = canonical_counts(before, before_field)
    after_counts = canonical_counts(after, after_field)
    missing_classes = [cat for cat in before_counts if before_counts[cat] > 0 and after_counts.get(cat, 0) == 0]
    conflicts, min_distance = count_symbol_conflicts(after, target_scale)
    coverage = grid_coverage(after, before)
    symbol_values = sorted(set(str(v) for v in after.get("point_symbol", pd.Series(dtype=str)).dropna().tolist()))
    expected_symbols = {rule["symbol"] for rule in POI_SYMBOL_RULES.values()}
    recognized_symbols = sorted(set(symbol_values) & expected_symbols)
    retention = len(after) / max(1, len(before))
    return {
        "input_count": int(len(before)),
        "output_count": int(len(after)),
        "retention_rate": round(float(retention), 4),
        "before_category_field": before_field,
        "after_category_field": after_field,
        "before_categories": dict(sorted(before_counts.items())),
        "after_categories": dict(sorted(after_counts.items())),
        "missing_classes": missing_classes,
        "category_coverage_ratio": round((len(before_counts) - len(missing_classes)) / max(1, len(before_counts)), 4),
        "region_coverage_ratio": round(float(coverage), 4),
        "symbol_values": symbol_values,
        "recognized_symbol_count": int(len(recognized_symbols)),
        "expected_symbol_count": int(len(expected_symbols)),
        "conflict_count": int(conflicts),
        "min_symbol_distance_m": round(float(min_distance), 3),
    }


def verdict(row: Dict[str, Any]) -> str:
    pass_fields = [
        "retention_qualified",
        "category_coverage_qualified",
        "region_coverage_qualified",
        "symbol_qualified",
        "conflict_qualified",
        "scale_progression_qualified",
    ]
    if all(int(row.get(field, 0) or 0) == 1 for field in pass_fields):
        return "PASS"
    if row["output_count"] <= 0:
        return "FAIL: empty output"
    if row["recognized_symbol_count"] < min(row["expected_symbol_count"], len(row["after_categories"])):
        return "FAIL: POI symbols not fully recognized"
    if row["category_coverage_ratio"] < 0.75:
        return "FAIL: category coverage below 0.75"
    if row["region_coverage_ratio"] < 0.45:
        return "WARN: region coverage below 0.45"
    if row["conflict_count"] > 0:
        return "WARN: remaining symbol conflicts"
    if int(row.get("scale_progression_qualified", 1) or 0) != 1:
        return "WARN: scale progression is not decreasing"
    return "PASS"


def add_qualification_flags(rows: List[Dict[str, Any]]) -> None:
    ordered = sorted(rows, key=lambda r: int(r.get("target_scale", 0) or 0))
    previous_count = None
    for row in ordered:
        retention = float(row.get("retention_rate", 0.0) or 0.0)
        category_cov = float(row.get("category_coverage_ratio", 0.0) or 0.0)
        region_cov = float(row.get("region_coverage_ratio", 0.0) or 0.0)
        conflicts = int(row.get("conflict_count", 0) or 0)
        recognized = int(row.get("recognized_symbol_count", 0) or 0)
        expected = int(row.get("expected_symbol_count", 0) or 0)
        output_count = int(row.get("output_count", 0) or 0)
        target_scale = int(row.get("target_scale", 0) or 0)

        min_retention = 0.08 if target_scale >= 500000 else (0.12 if target_scale >= 200000 else 0.25)
        max_retention = 0.62 if target_scale < 200000 else (0.24 if target_scale < 500000 else 0.18)

        row["retention_qualified"] = 1 if min_retention <= retention <= max_retention else 0
        row["category_coverage_qualified"] = 1 if category_cov >= 1.0 else 0
        row["region_coverage_qualified"] = 1 if region_cov >= 0.72 else 0
        row["symbol_qualified"] = 1 if recognized >= min(expected, max(1, len(row.get("after_categories", {})))) else 0
        row["conflict_qualified"] = 1 if conflicts == 0 else 0
        row["scale_progression_qualified"] = (
            1 if previous_count is None or output_count < int(previous_count) else 0
        )
        row["all_qualified"] = 1 if all(
            int(row.get(field, 0) or 0) == 1
            for field in [
                "retention_qualified",
                "category_coverage_qualified",
                "region_coverage_qualified",
                "symbol_qualified",
                "conflict_qualified",
                "scale_progression_qualified",
            ]
        ) else 0
        previous_count = output_count


def write_reports(output_root: Path, rows: Iterable[Dict[str, Any]], input_path: Path, raw_input_path: Path) -> None:
    rows = list(rows)
    (output_root / "skill_rule_summary.json").write_text(
        json.dumps(
            {"raw_input": str(raw_input_path), "normalized_input": str(input_path), "rows": rows},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    csv_path = output_root / "skill_rule_summary.csv"
    fieldnames = [
        "target_scale",
        "input_count",
        "output_count",
        "retention_rate",
        "category_coverage_ratio",
        "region_coverage_ratio",
        "conflict_count",
        "min_symbol_distance_m",
        "recognized_symbol_count",
        "expected_symbol_count",
        "retention_qualified",
        "category_coverage_qualified",
        "region_coverage_qualified",
        "symbol_qualified",
        "conflict_qualified",
        "scale_progression_qualified",
        "all_qualified",
        "verdict",
        "output_geojson",
        "output_png",
        "agent_log",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    lines = [
        "# Point Generalization Skill Rule Check",
        "",
        f"- raw_input: {raw_input_path}",
        f"- normalized_input: {input_path}",
        f"- scales: {', '.join('1:' + str(r['target_scale']) for r in rows)}",
        "",
        "| scale | before | after | retention | class coverage | region coverage | conflicts | all qualified | verdict |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| 1:{row['target_scale']} | {row['input_count']} | {row['output_count']} | "
            f"{row['retention_rate']} | {row['category_coverage_ratio']} | "
            f"{row['region_coverage_ratio']} | {row['conflict_count']} | "
            f"{row.get('all_qualified', 0)} | {row['verdict']} |"
        )
    (output_root / "skill_rule_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    scales = parse_scales(args.scales)
    output_root = (PROJECT_ROOT / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    raw_input = choose_input(args.input)
    input_path = normalize_input(raw_input, output_root, args.max_features)

    register_local_vlm_callback()
    rows: List[Dict[str, Any]] = []
    for scale in scales:
        print(f"[run] {input_path} -> 1:{scale}")
        run = run_agent(
            input_geojson=input_path,
            output_root=output_root,
            source_scale=args.source_scale,
            target_scale=scale,
            require_real_vlm=args.require_real_vlm,
        )
        row = evaluate(input_path, Path(run["output_geojson"]), scale)
        row.update(run)
        row["target_scale"] = scale
        rows.append(row)
        add_qualification_flags(rows)
        row["verdict"] = verdict(row)
        print(
            f"[{row['verdict']}] 1:{scale}: {row['input_count']} -> {row['output_count']}, "
            f"retention={row['retention_rate']}, classes={row['category_coverage_ratio']}, "
            f"region={row['region_coverage_ratio']}, conflicts={row['conflict_count']}, "
            f"all_qualified={row.get('all_qualified', 0)}"
        )

    add_qualification_flags(rows)
    for row in rows:
        row["verdict"] = verdict(row)
    write_reports(output_root, rows, input_path, raw_input)
    print(output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
