from __future__ import annotations

import json
import math
import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GOOD = {"completed", "completed_with_warnings"}

MODEL_SOURCES = {
    "GLM-4": PROJECT_ROOT
    / "tests/output/agent_robustness_original_live_glm8_lenient/comparison/raw_runs.csv",
    "Qwen-Max": PROJECT_ROOT
    / "tests/output/agent_robustness_original_live_qwen_moonshot8_lenient/comparison/raw_runs.csv",
    "DeepSeek": PROJECT_ROOT
    / "tests/output/agent_robustness_original_live_glm_qwen_deepseek/runs/deepseek_chat/_batch_summary.csv",
    "Moonshot": PROJECT_ROOT
    / "tests/output/agent_robustness_original_live_qwen_moonshot8_lenient/comparison/raw_runs.csv",
}

MODEL_NAME_MAP = {
    "GLM-4": "zhipu_glm4_plus",
    "Qwen-Max": "qwen_max",
    "DeepSeek": "deepseek_chat",
    "Moonshot": "moonshot_v1_8k",
}


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


def conflict_count(path: str, target_scale: int) -> tuple[int, int]:
    p = Path(path)
    if not p.is_file():
        return 0, 0
    gdf = gpd.read_file(p)
    if len(gdf) <= 1:
        return 0, len(gdf)
    work = projected(gdf)
    min_distance_m = max(20.0, float(target_scale) * 0.00025)
    coords = [(float(point.x), float(point.y)) for point in work.geometry]
    conflicts = 0
    for i, (x1, y1) in enumerate(coords):
        for x2, y2 in coords[i + 1 :]:
            if math.hypot(x1 - x2, y1 - y2) < min_distance_m:
                conflicts += 1
    return conflicts, len(coords)


def final_round(agent_log: str) -> int:
    text = Path(agent_log).read_text(encoding="utf-8", errors="ignore") if Path(agent_log).is_file() else ""
    m = re.search(r"after\s+(\d+)\s+rounds", text, flags=re.I)
    if m:
        return int(m.group(1))
    m = re.search(r"round_index[^\d]+(\d+)", text, flags=re.I)
    if m:
        return int(m.group(1))
    m = re.search(r"第\s*(\d+)\s*轮", text)
    if m:
        return int(m.group(1))
    m = re.search(r"绗\D*(\d+)\D*杞", text)
    if m:
        return int(m.group(1))
    return 1


def load_balanced_rows() -> pd.DataFrame:
    frames = []
    for label, csv_path in MODEL_SOURCES.items():
        df = pd.read_csv(csv_path)
        df = df[df["model_name"].astype(str) == MODEL_NAME_MAP[label]].copy()
        df = df.sort_values(["group_id", "target_scale"]).head(24)
        if len(df) != 24:
            raise RuntimeError(f"{label} has {len(df)} rows, expected 24: {csv_path}")
        df["Model"] = label
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def main() -> int:
    df = load_balanced_rows()
    enriched = []
    for row in df.to_dict("records"):
        conflicts, after_n = conflict_count(str(row.get("after_geojson", "")), int(row["target_scale"]))
        status = str(row.get("completion_status", ""))
        tool_ok = int(Path(str(row.get("after_geojson", ""))).is_file() and int(row.get("after_count", 0) or 0) > 0)
        completed = int(status in GOOD)
        resolved = int(conflicts == 0 and tool_ok)
        adherence = 1.0 if after_n <= 1 else max(0.0, 1.0 - conflicts / max(1.0, after_n))
        rounds = final_round(str(row.get("agent_log", "")))
        recovery_opportunity = int(rounds > 1)
        recovered = int(recovery_opportunity and completed)
        enriched.append(
            {
                **row,
                "computed_conflict_count": conflicts,
                "computed_after_n": after_n,
                "final_round": rounds,
                "Tool Execution Acc. (%)": 100.0 * tool_ok,
                "Conflict Resolution (%)": 100.0 * resolved,
                "Pattern Consistency": float(row.get("region_coverage_ratio", 0.0) or 0.0),
                "Task Completion (%)": 100.0 * completed,
                "Constraint Adherence": adherence,
                "Error Recovery Rate (%)": 100.0 * recovered if recovery_opportunity else np.nan,
            }
        )
    out = pd.DataFrame(enriched)

    metrics = [
        "Tool Execution Acc. (%)",
        "Conflict Resolution (%)",
        "Pattern Consistency",
        "Task Completion (%)",
        "Constraint Adherence",
        "Error Recovery Rate (%)",
    ]
    rows = []
    for metric in metrics:
        for model in ["GLM-4", "Qwen-Max", "DeepSeek", "Moonshot"]:
            values = pd.to_numeric(out.loc[out["Model"] == model, metric], errors="coerce").dropna()
            rows.append(
                {
                    "metric": metric,
                    "model": model,
                    "n": int(values.size),
                    "mean": round(float(values.mean()), 4) if values.size else 0.0,
                    "std": round(float(values.std(ddof=1)), 4) if values.size > 1 else 0.0,
                }
            )
    summary = pd.DataFrame(rows)

    out_dir = PROJECT_ROOT / "tests/output/agent_robustness_original_live_metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / "balanced_enriched_runs.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out_dir / "six_metric_summary.csv", index=False, encoding="utf-8-sig")

    metric_dict = {}
    for metric in metrics:
        part = summary[summary["metric"] == metric].set_index("model").loc[
            ["GLM-4", "Qwen-Max", "DeepSeek", "Moonshot"]
        ]
        metric_dict[metric] = {
            "mean": [float(v) for v in part["mean"].tolist()],
            "std": [float(v) for v in part["std"].tolist()],
        }
    (out_dir / "metrics_for_plot.json").write_text(json.dumps(metric_dict, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metric_dict, ensure_ascii=False, indent=2))
    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
