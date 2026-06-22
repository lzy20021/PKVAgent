from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.agent.core import MapGenAgent
from src.agent.providers import BaseLLMProvider, LLMResponse, Message, create_provider

GOOD = {"completed", "completed_with_warnings"}
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
    p = argparse.ArgumentParser("Benchmark MapGenAgent models and export charts + Excel.")
    p.add_argument("--mode", choices=["run", "analyze", "all"], default="all")
    p.add_argument("--summary-path", default="tests/data/amap_poi_batches/_summary.json")
    p.add_argument("--models-file", default="scripts/model_benchmark_models.cn.json")
    p.add_argument("--output-root", default="tests/output/agent_model_benchmark_cn")
    p.add_argument("--limit-groups", type=int, default=20)
    p.add_argument("--source-scale", type=int, default=10000)
    p.add_argument("--scales", default=",".join(str(v) for v in DEFAULT_SCALES))
    p.add_argument("--only-models", default="")
    p.add_argument("--max-models", type=int, default=0)
    return p.parse_args()


def parse_scales(text: str) -> List[int]:
    vals = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not vals:
        raise ValueError("no scales")
    return vals


def parse_filter(text: str) -> Optional[set[str]]:
    s = {x.strip() for x in str(text or "").split(",") if x.strip()}
    return s or None


def load_models(path: Path, name_filter: Optional[set[str]], max_models: int) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("models", payload) if isinstance(payload, dict) else payload
    out = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        if not r.get("name") or not r.get("provider") or not r.get("model"):
            continue
        if name_filter and str(r["name"]) not in name_filter:
            continue
        out.append(dict(r))
    return out[:max_models] if max_models > 0 else out


def api_key(cfg: Dict[str, Any]) -> str:
    k = str(cfg.get("api_key", "")).strip()
    if k:
        return k
    env_name = str(cfg.get("api_key_env", "")).strip()
    return str(os.getenv(env_name, "")).strip() if env_name else ""


def build_provider(cfg: Dict[str, Any]) -> BaseLLMProvider:
    provider = str(cfg.get("provider", "")).lower()
    if provider == "null":
        return NullProvider()
    key = api_key(cfg)
    if not key:
        raise ValueError(f"missing api key for {cfg.get('name')}")
    return create_provider(
        provider=provider,
        api_key=key,
        model=str(cfg.get("model")),
        base_url=(str(cfg.get("base_url", "")).strip() or None),
        temperature=float(cfg.get("temperature", 0.2)),
        max_tokens=int(cfg.get("max_tokens", 1024)),
    )


def safe_json_parse(text: str) -> Optional[Dict[str, Any]]:
    raw = str(text or "").strip()
    if not raw:
        return None
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw.replace("json", "", 1).strip()
    try:
        payload = json.loads(raw)
        return payload if isinstance(payload, dict) else None
    except Exception:
        p0 = raw.find("{")
        p1 = raw.rfind("}")
        if p0 >= 0 and p1 > p0:
            try:
                payload = json.loads(raw[p0 : p1 + 1])
                return payload if isinstance(payload, dict) else None
            except Exception:
                return None
    return None


def probe_provider(cfg: Dict[str, Any], provider: BaseLLMProvider) -> Dict[str, Any]:
    if str(cfg.get("provider", "")).lower() == "null":
        return {
            "model_name": str(cfg.get("name", "")),
            "provider": str(cfg.get("provider", "")),
            "llm_model": str(cfg.get("model", "")),
            "probe_status": "skipped",
            "probe_ok": 0,
            "probe_latency_seconds": 0.0,
            "probe_error": "null provider",
            "probe_response_excerpt": "",
        }

    messages = [
        Message("system", "Return compact JSON only."),
        Message("user", '{"task":"agent_live_probe","reply":"ok","format":"json"}'),
    ]
    t0 = time.time()
    try:
        rsp = provider.chat(messages)
        elapsed = round(time.time() - t0, 3)
        content = str(getattr(rsp, "content", "") or "").strip()
        payload = safe_json_parse(content)
        ok_tokens = {
            str(payload.get("reply", "")).lower(),
            str(payload.get("status", "")).lower(),
            str(payload.get("ok", "")).lower(),
        } if payload else set()
        ok = int(bool(payload) and bool(ok_tokens & {"ok", "true", "1"}))
        return {
            "model_name": str(cfg.get("name", "")),
            "provider": str(cfg.get("provider", "")),
            "llm_model": str(cfg.get("model", "")),
            "probe_status": "ok" if ok else "invalid_response",
            "probe_ok": ok,
            "probe_latency_seconds": elapsed,
            "probe_error": "" if ok else "response_not_valid_json",
            "probe_response_excerpt": content[:240],
        }
    except Exception as exc:
        return {
            "model_name": str(cfg.get("name", "")),
            "provider": str(cfg.get("provider", "")),
            "llm_model": str(cfg.get("model", "")),
            "probe_status": "error",
            "probe_ok": 0,
            "probe_latency_seconds": round(time.time() - t0, 3),
            "probe_error": f"{type(exc).__name__}: {exc}",
            "probe_response_excerpt": "",
        }


def load_groups(path: Path, n: int) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = [r for r in payload.get("results", []) if str(r.get("output_path", "")).endswith(".geojson")]
    return rows[:n]


def completion_status(reply: str) -> str:
    for line in str(reply or "").splitlines():
        t = line.strip().lower()
        if t.startswith("completion status:"):
            return t.split(":", 1)[1].strip()
    return "failed"


def region_metric(before_path: Path, after_path: Path) -> float:
    before = gpd.read_file(before_path)
    after = gpd.read_file(after_path)
    if len(before) == 0 or len(after) == 0:
        return 0.0
    if str(before.crs) != str(after.crs):
        after = after.to_crs(before.crs)
    minx, miny, maxx, maxy = [float(v) for v in before.total_bounds]
    w, h = max(1e-9, maxx - minx), max(1e-9, maxy - miny)
    bins = 8
    b = set(((before.geometry.x - minx) / w * bins).astype(int).clip(0, bins).astype(str) + "_" + ((before.geometry.y - miny) / h * bins).astype(int).clip(0, bins).astype(str))
    a = set(((after.geometry.x - minx) / w * bins).astype(int).clip(0, bins).astype(str) + "_" + ((after.geometry.y - miny) / h * bins).astype(int).clip(0, bins).astype(str))
    return float(len(a & b)) / float(max(1, len(b)))


def render_compare(before_path: Path, after_path: Path, out_png: Path, title: str) -> None:
    before = gpd.read_file(before_path)
    after = gpd.read_file(after_path)
    if len(before) == 0:
        return
    minx, miny, maxx, maxy = [float(v) for v in before.total_bounds]
    dx, dy = max(1e-9, maxx - minx), max(1e-9, maxy - miny)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4))
    for ax, g, t, c in [(axes[0], before, f"before({len(before)})", "#1f77b4"), (axes[1], after, f"after({len(after)})", "#d81b60")]:
        if len(g):
            g.plot(ax=ax, color=c, markersize=10)
        ax.set_xlim(minx - dx * 0.05, maxx + dx * 0.05)
        ax.set_ylim(miny - dy * 0.05, maxy + dy * 0.05)
        ax.set_aspect("equal", adjustable="box")
        ax.set_axis_off()
        ax.set_title(t)
    fig.suptitle(title)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def run_one(
    agent: MapGenAgent,
    cfg: Dict[str, Any],
    src: Dict[str, Any],
    source_scale: int,
    target_scale: int,
    root: Path,
    probe: Dict[str, Any],
) -> Dict[str, Any]:
    model_name = str(cfg["name"])
    group_id = str(src["group_id"])
    src_geo = PROJECT_ROOT / Path(str(src["output_path"]))
    out_dir = root / "runs" / model_name / f"scale_1_{target_scale}" / group_id
    out_dir.mkdir(parents=True, exist_ok=True)
    before_path = out_dir / f"{group_id}_before.geojson"
    after_path = out_dir / f"{group_id}_after.geojson"
    after_png = out_dir / f"{group_id}_after.png"
    comp_png = out_dir / f"{group_id}_compare.png"
    log_path = out_dir / f"{group_id}_agent_log.txt"
    shutil.copy2(src_geo, before_path)

    before_count = len(gpd.read_file(before_path)) if before_path.exists() else 0
    after_count = 0
    elapsed = 0.0
    err = ""
    reply = ""
    status = "failed"

    if int(probe.get("probe_ok", 0) or 0):
        task = {
            "parameters": {
                "input_file": str(before_path),
                "target_scale": int(target_scale),
                "source_scale": int(source_scale),
                "feature_count": int(src.get("feature_count", 0) or 0),
                "geometry_type": "point",
                "layer_name": group_id,
                "output_formats": ["geojson", "png"],
                "output_geojson": str(after_path),
                "output_png": str(after_png),
            }
        }
        t0 = time.time()
        try:
            reply = agent._run_scale_generalization(task) or ""
            status = completion_status(reply)
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            status = "failed"
        elapsed = round(time.time() - t0, 3)
        after_count = len(gpd.read_file(after_path)) if after_path.exists() else 0
        if after_path.exists():
            render_compare(before_path, after_path, comp_png, f"{model_name} | {group_id} | 1:{target_scale}")
    else:
        err = str(probe.get("probe_error", "") or "live llm probe failed")
        status = "llm_unreachable"
        reply = (
            "Live LLM probe failed before agent benchmark.\n"
            "Completion status: llm_unreachable\n"
            "Output files: none\n"
            "Verified: provider chat probe could not reach a live model endpoint."
        )

    log_path.write_text(reply if reply else err, encoding="utf-8")
    retention_ratio = (float(after_count) / float(before_count)) if before_count > 0 else 0.0
    count_delta = int(after_count) - int(before_count)
    region_coverage_ratio = round(region_metric(before_path, after_path), 4) if after_path.exists() else 0.0

    return {
        "model_name": model_name,
        "provider": str(cfg.get("provider", "")),
        "llm_model": str(cfg.get("model", "")),
        "vlm_model": str(cfg.get("vlm_model", "")),
        "llm_probe_ok": int(probe.get("probe_ok", 0) or 0),
        "llm_probe_status": str(probe.get("probe_status", "")),
        "llm_probe_latency_seconds": float(probe.get("probe_latency_seconds", 0.0) or 0.0),
        "llm_probe_error": str(probe.get("probe_error", "")),
        "group_id": group_id,
        "city": str(src.get("city", "")),
        "region_name": str(src.get("region_name", "")),
        "keyword": str(src.get("keyword", "")),
        "target_scale": int(target_scale),
        "before_count": int(before_count),
        "after_count": int(after_count),
        "completion_status": status,
        "elapsed_seconds": elapsed,
        "retention_ratio": round(retention_ratio, 4),
        "reduction_ratio": round(1.0 - retention_ratio, 4),
        "count_delta": int(count_delta),
        "count_change_abs": int(abs(count_delta)),
        "region_coverage_ratio": region_coverage_ratio,
        "error": err,
        "before_geojson": str(before_path),
        "after_geojson": str(after_path),
        "after_png": str(after_png),
        "compare_png": str(comp_png),
        "agent_log": str(log_path),
    }


def write_rows(rows: List[Dict[str, Any]], out_csv: Path, out_json: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({"run_count": len(rows), "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    if not rows:
        return
    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def run_phase(args: argparse.Namespace, models: List[Dict[str, Any]], scales: List[int], output_root: Path) -> None:
    groups = load_groups(PROJECT_ROOT / Path(args.summary_path), args.limit_groups)
    all_rows: List[Dict[str, Any]] = []
    probe_rows: List[Dict[str, Any]] = []
    for i, cfg in enumerate(models, start=1):
        print(f"[model {i}/{len(models)}] {cfg['name']}")
        provider = build_provider(cfg)
        probe = probe_provider(cfg, provider)
        probe_rows.append(dict(probe))
        print(f"  probe={probe['probe_status']} latency={probe['probe_latency_seconds']}s")
        agent = MapGenAgent(provider)
        agent.set_workspace(str(output_root / "runs" / str(cfg["name"])))
        one_rows: List[Dict[str, Any]] = []
        total, cur = len(groups) * len(scales), 0
        for g in groups:
            for s in scales:
                cur += 1
                print(f"  [{cur}/{total}] {g['group_id']} -> 1:{s}")
                row = run_one(agent, cfg, g, args.source_scale, s, output_root, probe)
                one_rows.append(row)
                all_rows.append(row)
        base = output_root / "runs" / str(cfg["name"]) / "_batch_summary"
        write_rows(one_rows, base.with_suffix(".csv"), base.with_suffix(".json"))
    write_rows(all_rows, output_root / "comparison" / "raw_runs.csv", output_root / "comparison" / "raw_runs.json")
    write_rows(probe_rows, output_root / "comparison" / "probe_results.csv", output_root / "comparison" / "probe_results.json")


def analyze_phase(output_root: Path, name_filter: Optional[set[str]]) -> Dict[str, Any]:
    raw_csv = output_root / "comparison" / "raw_runs.csv"
    if raw_csv.exists():
        df = pd.read_csv(raw_csv)
    else:
        rows = [pd.read_csv(p) for p in (output_root / "runs").glob("*/_batch_summary.csv")]
        df = pd.concat(rows, axis=0, ignore_index=True) if rows else pd.DataFrame()
    if len(df) == 0:
        return {"run_count": 0, "model_count": 0, "message": "no data"}
    if name_filter:
        df = df[df["model_name"].astype(str).isin(name_filter)].copy()

    probe_csv = output_root / "comparison" / "probe_results.csv"
    probe_df = pd.read_csv(probe_csv) if probe_csv.exists() else pd.DataFrame()
    if len(probe_df) and name_filter:
        probe_df = probe_df[probe_df["model_name"].astype(str).isin(name_filter)].copy()

    df["usable"] = df["completion_status"].isin(GOOD).astype(int)
    df["completed"] = (df["completion_status"] == "completed").astype(int)
    df["warning"] = (df["completion_status"] == "completed_with_warnings").astype(int)
    df["failed"] = (~df["completion_status"].isin(list(GOOD))).astype(int)
    if "retention_ratio" not in df.columns:
        df["retention_ratio"] = np.where(df["before_count"] > 0, df["after_count"] / df["before_count"], 0.0)
    if "reduction_ratio" not in df.columns:
        df["reduction_ratio"] = 1.0 - df["retention_ratio"]
    if "count_delta" not in df.columns:
        df["count_delta"] = df["after_count"] - df["before_count"]
    if "count_change_abs" not in df.columns:
        df["count_change_abs"] = (df["after_count"] - df["before_count"]).abs()

    model_df = df.groupby(["model_name", "provider", "llm_model", "vlm_model"], dropna=False).agg(
        run_count=("model_name", "size"),
        live_llm_rate=("llm_probe_ok", "mean"),
        usable_completion_rate=("usable", "mean"),
        completed_rate=("completed", "mean"),
        warning_rate=("warning", "mean"),
        failed_rate=("failed", "mean"),
        avg_runtime_seconds=("elapsed_seconds", "mean"),
        median_runtime_seconds=("elapsed_seconds", "median"),
        p95_runtime_seconds=("elapsed_seconds", lambda x: float(np.percentile(pd.to_numeric(x, errors="coerce").fillna(0.0), 95))),
        avg_retention_ratio=("retention_ratio", "mean"),
        retention_std=("retention_ratio", lambda x: float(pd.to_numeric(x, errors="coerce").fillna(0.0).std())),
        avg_reduction_ratio=("reduction_ratio", "mean"),
        avg_region_coverage_ratio=("region_coverage_ratio", "mean"),
        avg_count_delta=("count_delta", "mean"),
        avg_count_change_abs=("count_change_abs", "mean"),
        avg_probe_latency_seconds=("llm_probe_latency_seconds", "mean"),
    ).reset_index()

    runtime_max = max(1.0, float(model_df["avg_runtime_seconds"].max() or 1.0))
    probe_max = max(1.0, float(model_df["avg_probe_latency_seconds"].max() or 1.0))
    model_df["runtime_efficiency_score"] = 1.0 - (model_df["avg_runtime_seconds"] / runtime_max).clip(0, 1)
    model_df["probe_efficiency_score"] = 1.0 - (model_df["avg_probe_latency_seconds"] / probe_max).clip(0, 1)
    model_df["retention_balance_score"] = (1.0 - (model_df["avg_retention_ratio"] - 0.3).abs().clip(0, 1)).clip(0, 1)
    model_df["stability_score"] = 1.0 - model_df["retention_std"].fillna(0.0).clip(0, 1)
    model_df["composite_score"] = np.where(
        model_df["live_llm_rate"] > 0,
        (
            0.25 * model_df["live_llm_rate"]
            + 0.20 * model_df["usable_completion_rate"]
            + 0.15 * model_df["completed_rate"]
            + 0.15 * model_df["avg_region_coverage_ratio"].clip(0, 1)
            + 0.10 * model_df["retention_balance_score"]
            + 0.10 * model_df["runtime_efficiency_score"]
            + 0.05 * model_df["stability_score"]
        ).round(4),
        0.0,
    )
    model_df = model_df.sort_values(["composite_score", "usable_completion_rate", "avg_region_coverage_ratio"], ascending=[False, False, False])

    scale_df = df.groupby(["model_name", "target_scale"], dropna=False).agg(
        run_count=("model_name", "size"),
        live_llm_rate=("llm_probe_ok", "mean"),
        usable_completion_rate=("usable", "mean"),
        completed_rate=("completed", "mean"),
        avg_runtime_seconds=("elapsed_seconds", "mean"),
        avg_retention_ratio=("retention_ratio", "mean"),
        avg_region_coverage_ratio=("region_coverage_ratio", "mean"),
    ).reset_index()
    city_df = df.groupby(["model_name", "city"], dropna=False).agg(
        run_count=("model_name", "size"),
        usable_completion_rate=("usable", "mean"),
        avg_region_coverage_ratio=("region_coverage_ratio", "mean"),
    ).reset_index()
    keyword_df = df.groupby(["model_name", "keyword"], dropna=False).agg(
        run_count=("model_name", "size"),
        usable_completion_rate=("usable", "mean"),
        avg_region_coverage_ratio=("region_coverage_ratio", "mean"),
    ).reset_index()

    comp = output_root / "comparison"
    comp.mkdir(parents=True, exist_ok=True)
    df.to_csv(comp / "raw_runs.csv", index=False, encoding="utf-8-sig")
    model_df.to_csv(comp / "model_metrics.csv", index=False, encoding="utf-8-sig")
    scale_df.to_csv(comp / "model_scale_metrics.csv", index=False, encoding="utf-8-sig")
    city_df.to_csv(comp / "model_city_metrics.csv", index=False, encoding="utf-8-sig")
    keyword_df.to_csv(comp / "model_keyword_metrics.csv", index=False, encoding="utf-8-sig")
    if len(probe_df):
        probe_df.to_csv(comp / "probe_results.csv", index=False, encoding="utf-8-sig")

    figs = output_root / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(model_df))
    labels = model_df["model_name"].tolist()

    fig, ax1 = plt.subplots(figsize=(12, 6.2))
    ax1.bar(x, model_df["usable_completion_rate"], color="#3B82F6", alpha=0.85, label="usable_completion_rate")
    ax1.set_ylim(0, 1.05)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=25, ha="right")
    ax1.set_ylabel("Completion Rate")
    ax2 = ax1.twinx()
    ax2.plot(x, model_df["avg_runtime_seconds"], color="#EF4444", marker="o", linewidth=2, label="avg_runtime_seconds")
    ax2.set_ylabel("Runtime(s)")
    ax1.set_title("Completion vs Runtime")
    ax1.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figs / "fig_completion_vs_runtime.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12.5, 6.2))
    ax.bar(x - 0.18, model_df["composite_score"], width=0.36, color="#2563EB", alpha=0.9, label="composite_score")
    ax.bar(x + 0.18, model_df["live_llm_rate"], width=0.36, color="#F59E0B", alpha=0.85, label="live_llm_rate")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Composite Score and Live LLM Availability")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figs / "fig_composite_bar.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    piv = model_df[
        [
            "model_name",
            "live_llm_rate",
            "usable_completion_rate",
            "completed_rate",
            "warning_rate",
            "avg_retention_ratio",
            "avg_region_coverage_ratio",
            "composite_score",
        ]
    ].set_index("model_name")
    heat = piv.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(10.6, 5.8))
    im = ax.imshow(heat, cmap="YlGnBu", aspect="auto")
    ax.set_yticks(np.arange(len(piv.index)))
    ax.set_yticklabels(piv.index.tolist())
    ax.set_xticks(np.arange(len(piv.columns)))
    ax.set_xticklabels(piv.columns.tolist(), rotation=15, ha="right")
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    ax.set_title("Metrics Heatmap")
    fig.tight_layout()
    fig.savefig(figs / "fig_quality_heatmap.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    radar_cols = [
        "live_llm_rate",
        "usable_completion_rate",
        "completed_rate",
        "avg_region_coverage_ratio",
        "retention_balance_score",
        "runtime_efficiency_score",
        "stability_score",
    ]
    radar_labels = ["LiveLLM", "Usable", "Completed", "Coverage", "Retention", "Runtime", "Stability"]
    radar_df = model_df.head(12).copy()
    angles = np.linspace(0, 2 * math.pi, len(radar_cols), endpoint=False).tolist()
    angles += angles[:1]
    fig = plt.figure(figsize=(10.2, 8.4))
    ax = plt.subplot(111, polar=True)
    for _, row in radar_df.iterrows():
        values = [float(row[c]) for c in radar_cols]
        values += values[:1]
        ax.plot(angles, values, linewidth=1.6, label=str(row["model_name"]))
        ax.fill(angles, values, alpha=0.05)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar_labels)
    ax.set_ylim(0, 1.0)
    ax.set_title("Model Radar Comparison", pad=22)
    ax.grid(alpha=0.3)
    if len(radar_df):
        ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.12))
    fig.tight_layout()
    fig.savefig(figs / "fig_model_radar.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for name, g in scale_df.groupby("model_name"):
        gg = g.sort_values("target_scale")
        axes[0].plot(gg["target_scale"], gg["usable_completion_rate"], marker="o", linewidth=1.8, label=name)
        axes[1].plot(gg["target_scale"], gg["avg_region_coverage_ratio"], marker="o", linewidth=1.8, label=name)
    axes[0].set_title("Scale vs Usable Completion")
    axes[1].set_title("Scale vs Region Coverage")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.set_xlabel("Target Scale")
    axes[0].set_ylim(0, 1.05)
    axes[1].set_ylim(0, 1.05)
    handles, labels2 = axes[1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels2, loc="lower center", ncol=min(4, len(labels2)), bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout()
    fig.savefig(figs / "fig_scale_trends.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    excel_path = comp / "benchmark_results.xlsx"
    excel_ok = True
    excel_msg = str(excel_path)
    try:
        with pd.ExcelWriter(excel_path) as writer:
            df.to_excel(writer, sheet_name="raw_runs", index=False)
            model_df.to_excel(writer, sheet_name="model_metrics", index=False)
            scale_df.to_excel(writer, sheet_name="model_scale_metrics", index=False)
            city_df.to_excel(writer, sheet_name="model_city_metrics", index=False)
            keyword_df.to_excel(writer, sheet_name="model_keyword_metrics", index=False)
            if len(probe_df):
                probe_df.to_excel(writer, sheet_name="probe_results", index=False)
    except Exception as exc:
        excel_ok = False
        excel_msg = f"excel export failed: {exc}"

    summary = {
        "run_count": int(len(df)),
        "model_count": int(model_df["model_name"].nunique()),
        "live_llm_ok_models": int(model_df["live_llm_rate"].gt(0).sum()),
        "excel_ok": excel_ok,
        "excel_message": excel_msg,
        "output_root": str(output_root),
    }
    (comp / "analysis_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    args = parse_args()
    scales = parse_scales(args.scales)
    name_filter = parse_filter(args.only_models)
    output_root = PROJECT_ROOT / Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    models = load_models(PROJECT_ROOT / Path(args.models_file), name_filter, args.max_models)
    if args.mode in {"run", "all"}:
        run_phase(args, models, scales, output_root)
    if args.mode in {"analyze", "all"}:
        print(json.dumps(analyze_phase(output_root, name_filter), ensure_ascii=False, indent=2))
    print(output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
