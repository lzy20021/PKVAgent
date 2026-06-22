from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
from openai import OpenAI

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "tests" / "output" / "model_comparison_experiments"
RAW_DIR = OUTPUT_ROOT / "raw"
FIG_DIR = OUTPUT_ROOT / "figures"
EXCEL_DIR = OUTPUT_ROOT / "excels"
SUMMARY_PATH = PROJECT_ROOT / "tests" / "data" / "amap_poi_batches" / "_summary.json"


@dataclass
class VendorSpec:
    name: str
    key_env_candidates: List[str]
    base_urls: List[str]
    preferred_llm: List[str]
    preferred_vlm: List[str]


VENDORS: Dict[str, VendorSpec] = {
    "zai_or_zhipu": VendorSpec(
        name="zai_or_zhipu",
        key_env_candidates=["ZAI_API_KEY", "ZHIPUAI_API_KEY"],
        base_urls=["https://api.z.ai/api/paas/v4", "https://open.bigmodel.cn/api/paas/v4"],
        preferred_llm=["glm-5", "glm-4-7", "glm-4.7", "glm-4-plus", "glm-4-flash"],
        preferred_vlm=["glm-4.6v", "glm-4v-plus", "glm-4v", "glm-4v-flash"],
    ),
    "moonshot": VendorSpec(
        name="moonshot",
        key_env_candidates=["MOONSHOT_API_KEY"],
        base_urls=["https://api.moonshot.ai/v1", "https://api.moonshot.cn/v1"],
        preferred_llm=["kimi-k2.5", "kimi-k2-0905-preview", "kimi-k2-turbo-preview", "moonshot-v1-8k"],
        preferred_vlm=["moonshot-v1-vision-preview"],
    ),
    "deepseek": VendorSpec(
        name="deepseek",
        key_env_candidates=["DEEPSEEK_API_KEY"],
        base_urls=["https://api.deepseek.com/v1", "https://api.deepseek.com"],
        preferred_llm=["deepseek-chat", "deepseek-reasoner"],
        preferred_vlm=["deepseek-vl2"],
    ),
    "dashscope": VendorSpec(
        name="dashscope",
        key_env_candidates=["DASHSCOPE_API_KEY"],
        base_urls=["https://dashscope.aliyuncs.com/compatible-mode/v1"],
        preferred_llm=["qwen-max", "qwen-plus", "qwen-turbo"],
        preferred_vlm=["qwen-vl-max", "qwen-vl-plus", "qvq-max"],
    ),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run cross-vendor model comparison with real map data.")
    p.add_argument("--summary-path", default=str(SUMMARY_PATH))
    p.add_argument("--limit-groups", type=int, default=4)
    p.add_argument("--max-points-per-group", type=int, default=80)
    p.add_argument("--runs-per-task", type=int, default=2)
    p.add_argument("--timeout-seconds", type=int, default=90)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def ensure_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    EXCEL_DIR.mkdir(parents=True, exist_ok=True)


def env_status(env_name: str) -> Tuple[bool, int]:
    v = os.getenv(env_name)
    if not v:
        return False, 0
    return True, len(v)


def pick_api_key(spec: VendorSpec) -> Tuple[Optional[str], Optional[str]]:
    for env_name in spec.key_env_candidates:
        key = os.getenv(env_name, "").strip()
        if key:
            return key, env_name
    return None, None


def try_list_models(key: str, base_url: str, timeout_s: int) -> Tuple[List[str], str]:
    try:
        client = OpenAI(api_key=key, base_url=base_url, timeout=timeout_s)
        models = client.models.list()
        ids = sorted({getattr(m, "id", "") for m in models if getattr(m, "id", "")})
        return ids, ""
    except Exception as exc:
        return [], f"{type(exc).__name__}: {exc}"


def choose_best_model(preferred: Iterable[str], available: List[str]) -> Tuple[Optional[str], Optional[str]]:
    if not available:
        return None, None
    av_set = set(available)
    for p in preferred:
        if p in av_set:
            return p, p
    for p in preferred:
        p_norm = p.lower().replace(".", "-").replace("_", "-")
        for a in available:
            a_norm = a.lower().replace(".", "-").replace("_", "-")
            if p_norm in a_norm or a_norm in p_norm:
                return a, p
    return available[0], None


def load_groups(summary_path: Path, limit_groups: int) -> List[Dict[str, Any]]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = payload.get("results", [])
    out: List[Dict[str, Any]] = []
    for r in rows:
        path = PROJECT_ROOT / str(r.get("output_path", ""))
        if not path.exists() or not str(path).endswith(".geojson"):
            continue
        out.append(dict(r))
        if len(out) >= limit_groups:
            break
    return out


def select_points_for_task(gdf: gpd.GeoDataFrame, limit_n: int, seed: int) -> gpd.GeoDataFrame:
    if len(gdf) <= limit_n:
        return gdf.copy()
    rs = np.random.RandomState(seed)
    idx = rs.choice(np.arange(len(gdf)), size=limit_n, replace=False)
    return gdf.iloc[idx].copy().reset_index(drop=True)


def _dist_m(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    lon1, lat1 = a
    lon2, lat2 = b
    x = (lon2 - lon1) * 111320 * math.cos(math.radians((lat1 + lat2) / 2))
    y = (lat2 - lat1) * 110540
    return math.sqrt(x * x + y * y)


def build_ground_truth(gdf: gpd.GeoDataFrame) -> Dict[str, Any]:
    rows = gdf.copy()
    rows["id"] = rows["amap_id"].astype(str)
    coords = np.array([(geom.x, geom.y) for geom in rows.geometry], dtype=float)
    center = coords.mean(axis=0)
    d = np.linalg.norm(coords - center, axis=1)
    center_score = 1.0 - d / d.max() if d.max() > 0 else np.ones_like(d)
    name_len = rows["name"].astype(str).str.len().to_numpy(dtype=float)
    if name_len.max() > 0:
        name_len = name_len / name_len.max()
    rows["score"] = 0.7 * center_score + 0.3 * name_len
    rows = rows.sort_values("score", ascending=False).reset_index(drop=True)
    keep_n = max(10, int(len(rows) * 0.35))

    minx, miny, maxx, maxy = [float(v) for v in rows.total_bounds]
    min_dist_m = max(30.0, _dist_m((minx, miny), (maxx, maxy)) * 0.03)
    selected_ids: List[str] = []
    selected_xy: List[Tuple[float, float]] = []
    for _, r in rows.iterrows():
        pid = str(r["id"])
        xy = (float(r.geometry.x), float(r.geometry.y))
        if all(_dist_m(xy, sxy) >= min_dist_m for sxy in selected_xy):
            selected_ids.append(pid)
            selected_xy.append(xy)
        if len(selected_ids) >= keep_n:
            break
    if len(selected_ids) < keep_n:
        for _, r in rows.iterrows():
            pid = str(r["id"])
            if pid not in selected_ids:
                selected_ids.append(pid)
            if len(selected_ids) >= keep_n:
                break
    return {
        "all_ids": rows["id"].astype(str).tolist(),
        "target_ids": selected_ids,
        "key_ids": rows.head(min(12, len(rows)))["id"].astype(str).tolist(),
        "target_keep_n": keep_n,
        "min_dist_m": min_dist_m,
    }


def quadrant_hist(gdf: gpd.GeoDataFrame, ids: List[str]) -> Dict[str, float]:
    part = gdf[gdf["amap_id"].astype(str).isin(ids)]
    if len(part) == 0:
        return {"NE": 0.0, "NW": 0.0, "SE": 0.0, "SW": 0.0}
    cx, cy = float(part.geometry.x.mean()), float(part.geometry.y.mean())
    counts = {"NE": 0, "NW": 0, "SE": 0, "SW": 0}
    for p in part.geometry:
        if p.x >= cx and p.y >= cy:
            counts["NE"] += 1
        elif p.x < cx and p.y >= cy:
            counts["NW"] += 1
        elif p.x >= cx and p.y < cy:
            counts["SE"] += 1
        else:
            counts["SW"] += 1
    n = float(sum(counts.values()))
    return {k: v / n for k, v in counts.items()}


def build_task_prompt(group_row: Dict[str, Any], gdf: gpd.GeoDataFrame, gt: Dict[str, Any]) -> str:
    records = []
    for _, r in gdf.iterrows():
        records.append(
            {
                "id": str(r["amap_id"]),
                "name": str(r.get("name", "")),
                "type": str(r.get("type", "")),
                "lon": round(float(r.geometry.x), 6),
                "lat": round(float(r.geometry.y), 6),
            }
        )
    payload = {
        "task": "point_generalization",
        "goal": "在小比例尺下保留关键 POI，降低拥挤并保持空间均衡",
        "rules": [
            f"从输入点中选择约 {gt['target_keep_n']} 个点（允许偏差 ±15%）",
            f"尽量避免选择距离小于 {round(gt['min_dist_m'], 1)} 米的点对",
            "优先保留中心区和语义信息更丰富的点",
            "输出必须是 JSON，不要输出其他文字",
        ],
        "required_json_schema": {
            "selected_ids": ["string"],
            "rule_checklist": ["string"],
            "conflicts_resolved": [["id_a", "id_b"]],
            "compression_ratio": "number",
            "summary": "string",
        },
        "city": group_row.get("city"),
        "region_name": group_row.get("region_name"),
        "keyword": group_row.get("keyword"),
        "points": records,
    }
    return json.dumps(payload, ensure_ascii=False)


def parse_json_text(text: str) -> Optional[Dict[str, Any]]:
    t = (text or "").strip()
    if not t:
        return None
    if t.startswith("```"):
        m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", t, flags=re.S)
        if m:
            t = m.group(1)
    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else None
    except Exception:
        m = re.search(r"\{.*\}", t, flags=re.S)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None


def call_llm(client: OpenAI, model: str, prompt: str, timeout_s: int, tools=None, image_data_url=None) -> Dict[str, Any]:
    if image_data_url:
        messages = [
            {"role": "system", "content": "你是地图综合评测助手。"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            },
        ]
    else:
        messages = [{"role": "system", "content": "你是地图综合评测助手。"}, {"role": "user", "content": prompt}]
    t0 = time.perf_counter()
    rsp = client.chat.completions.create(model=model, messages=messages, temperature=0.2, timeout=timeout_s, tools=tools)
    elapsed = time.perf_counter() - t0
    msg = rsp.choices[0].message
    tool_calls = getattr(msg, "tool_calls", None) or []
    return {
        "elapsed": elapsed,
        "content": msg.content or "",
        "tool_calls": [{"name": getattr(getattr(tc, "function", None), "name", "")} for tc in tool_calls],
    }


def confusion_metrics(all_ids: List[str], pred_ids: List[str], gt_ids: List[str]) -> Dict[str, float]:
    u, p, t = set(all_ids), set(pred_ids), set(gt_ids)
    tp, fp, fn = len(p & t), len(p - t), len(t - p)
    tn = len(u) - tp - fp - fn
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    acc = (tp + tn) / len(u) if u else 0.0
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


def min_distance_violation_ratio(gdf: gpd.GeoDataFrame, pred_ids: List[str], min_dist_m: float) -> float:
    part = gdf[gdf["amap_id"].astype(str).isin(pred_ids)]
    if len(part) < 2:
        return 0.0
    coords = [(float(p.x), float(p.y)) for p in part.geometry]
    pairs = bad = 0
    for i, j in combinations(range(len(coords)), 2):
        pairs += 1
        if _dist_m(coords[i], coords[j]) < min_dist_m:
            bad += 1
    return bad / pairs if pairs else 0.0


def make_density_figure(gdf: gpd.GeoDataFrame, out_png: Path) -> str:
    minx, miny, maxx, maxy = [float(v) for v in gdf.total_bounds]
    cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
    fig, ax = plt.subplots(figsize=(5.4, 5.4))
    ax.scatter(gdf.geometry.x, gdf.geometry.y, s=12, alpha=0.85, c="#1d4ed8")
    ax.axvline(cx, color="#64748b", linestyle="--", linewidth=1)
    ax.axhline(cy, color="#64748b", linestyle="--", linewidth=1)
    ax.set_title("POI Quadrant Density")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)
    import base64

    return f"data:image/png;base64,{base64.b64encode(out_png.read_bytes()).decode('ascii')}"


def build_row_metrics(model_name, vendor, task_id, run_idx, elapsed, parsed, raw_content, tool_calls, gdf, gt, fallback_from):
    all_ids = [str(v) for v in gt["all_ids"]]
    target_ids = [str(v) for v in gt["target_ids"]]
    selected_ids = [str(x) for x in (parsed or {}).get("selected_ids", []) if str(x) in set(all_ids)]
    format_ok = int(parsed is not None and isinstance(parsed.get("selected_ids"), list)) if parsed else 0
    m = confusion_metrics(all_ids, selected_ids, target_ids)
    violation_ratio = min_distance_violation_ratio(gdf, selected_ids, float(gt["min_dist_m"]))
    target_n = int(gt["target_keep_n"])
    n_ok = abs(len(selected_ids) - target_n) <= max(2, int(target_n * 0.15))
    dist_ok = violation_ratio <= 0.08
    pred_hist = quadrant_hist(gdf, selected_ids)
    gt_hist = quadrant_hist(gdf, target_ids)
    spatial_balance = 1.0 - 0.5 * sum(abs(pred_hist[k] - gt_hist[k]) for k in ["NE", "NW", "SE", "SW"])
    return {
        "vendor": vendor,
        "model_name": model_name,
        "fallback_from": fallback_from or "",
        "task_id": task_id,
        "run_idx": run_idx,
        "response_seconds": round(float(elapsed), 4),
        "task_completion": int(parsed is not None),
        "instruction_following": int(format_ok and len(selected_ids) > 0),
        "output_format_correct": format_ok,
        "tool_call_success": int(any(tc.get("name") for tc in tool_calls)),
        "error_rate": int(parsed is None),
        "stability_placeholder": np.nan,
        "rule_satisfaction_rate": round((int(n_ok) + int(dist_ok)) / 2.0, 6),
        "key_point_retention_rate": round(len(set(selected_ids) & set(gt["key_ids"])) / max(1, len(gt["key_ids"])), 6),
        "conflict_resolution_rate": round(max(0.0, 1.0 - violation_ratio), 6),
        "spatial_distribution_balance": round(spatial_balance, 6),
        "compression_ratio": round(len(all_ids) / max(1, len(selected_ids)) if selected_ids else 0.0, 6),
        "accuracy": round(float(m["accuracy"]), 6),
        "precision": round(float(m["precision"]), 6),
        "recall": round(float(m["recall"]), 6),
        "f1": round(float(m["f1"]), 6),
        "selected_count": len(selected_ids),
        "target_count": target_n,
        "raw_response": raw_content[:4000],
        "parsed_json": json.dumps(parsed, ensure_ascii=False) if parsed else "",
    }


def aggregate_stability(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df.empty:
        return raw_df
    df = raw_df.copy()
    for (model_name, task_id), sub in df.groupby(["model_name", "task_id"]):
        sets = []
        for txt in sub["parsed_json"].tolist():
            if not txt:
                continue
            try:
                obj = json.loads(txt)
                sets.append(set(str(x) for x in obj.get("selected_ids", [])))
            except Exception:
                pass
        stability = np.nan
        if len(sets) >= 2:
            sims = [len(sets[i] & sets[j]) / max(1, len(sets[i] | sets[j])) for i, j in combinations(range(len(sets)), 2)]
            stability = float(np.mean(sims)) if sims else np.nan
        idx = (df["model_name"] == model_name) & (df["task_id"] == task_id)
        df.loc[idx, "stability_placeholder"] = stability
    return df


def run() -> int:
    args = parse_args()
    np.random.seed(args.seed)
    ensure_dirs()

    env_rows = [{"env": k, "present": env_status(k)[0], "length": env_status(k)[1]} for k in ["DEEPSEEK_API_KEY", "DASHSCOPE_API_KEY", "MOONSHOT_API_KEY", "ZAI_API_KEY", "ZHIPUAI_API_KEY"]]
    pd.DataFrame(env_rows).to_csv(RAW_DIR / "env_key_status.csv", index=False, encoding="utf-8-sig")

    groups = load_groups(Path(args.summary_path), args.limit_groups)
    if not groups:
        (OUTPUT_ROOT / "summary.txt").write_text("No input groups found.", encoding="utf-8")
        return 2

    discovery_rows, clients = [], {}
    for vn, spec in VENDORS.items():
        key, key_env = pick_api_key(spec)
        if not key:
            discovery_rows.append({"vendor": vn, "key_env": ",".join(spec.key_env_candidates), "key_present": 0, "base_url": "", "model_count": 0, "error": "missing_api_key"})
            continue
        best_models, best_url, best_err = [], "", ""
        for url in spec.base_urls:
            ids, err = try_list_models(key, url, args.timeout_seconds)
            discovery_rows.append({"vendor": vn, "key_env": key_env, "key_present": 1, "base_url": url, "model_count": len(ids), "error": err})
            if ids and len(ids) > len(best_models):
                best_models, best_url = ids, url
            if not best_err:
                best_err = err
        if best_url:
            clients[vn] = {"models": best_models, "client": OpenAI(api_key=key, base_url=best_url)}
        else:
            clients[vn] = {"models": [], "client": OpenAI(api_key=key, base_url=spec.base_urls[0]), "error": best_err}
    disc_df = pd.DataFrame(discovery_rows)
    disc_df.to_csv(RAW_DIR / "model_discovery.csv", index=False, encoding="utf-8-sig")

    eval_rows, raw_exceptions = [], []
    for group_idx, group in enumerate(groups):
        gdf = gpd.read_file(PROJECT_ROOT / str(group["output_path"]))
        gdf = select_points_for_task(gdf, args.max_points_per_group, args.seed + group_idx).reset_index(drop=True)
        gt = build_ground_truth(gdf)
        task_prompt = build_task_prompt(group, gdf, gt)
        task_id = str(group.get("group_id", f"group_{group_idx}"))
        vlm_png = RAW_DIR / f"{task_id}_vlm.png"
        data_url = make_density_figure(gdf, vlm_png)
        densest = max(quadrant_hist(gdf, gt["all_ids"]).items(), key=lambda kv: kv[1])[0]
        vlm_prompt = '观察图像中的四象限散点密度，输出JSON: {"densest_quadrant":"NE|NW|SE|SW","reason":"..."}'

        for vn, spec in VENDORS.items():
            cctx = clients.get(vn)
            if not cctx:
                continue
            llm_model, fallback_from = choose_best_model(spec.preferred_llm, cctx.get("models", []))
            if not llm_model:
                raw_exceptions.append({"vendor": vn, "task_id": task_id, "error": "no_llm_model_after_discovery"})
                continue
            for run_idx in range(1, args.runs_per_task + 1):
                try:
                    rsp = call_llm(cctx["client"], llm_model, task_prompt, args.timeout_seconds)
                    eval_rows.append(build_row_metrics(llm_model, vn, task_id, run_idx, float(rsp["elapsed"]), parse_json_text(rsp["content"]), str(rsp["content"]), list(rsp["tool_calls"]), gdf, gt, fallback_from))
                except Exception as exc:
                    raw_exceptions.append({"vendor": vn, "task_id": task_id, "model": llm_model, "run_idx": run_idx, "error": f"{type(exc).__name__}: {exc}"})

            tools = [{"type": "function", "function": {"name": "sum_numbers", "description": "求和两个整数", "parameters": {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}, "required": ["a", "b"]}}}]
            try:
                tr = call_llm(cctx["client"], llm_model, '请直接调用工具 sum_numbers 计算 17 + 25，并输出 JSON {"tool_result":42}', args.timeout_seconds, tools=tools)
                eval_rows.append({"vendor": vn, "model_name": llm_model, "fallback_from": fallback_from or "", "task_id": f"{task_id}__tool", "run_idx": 1, "response_seconds": round(float(tr["elapsed"]), 4), "task_completion": 1, "instruction_following": 1, "output_format_correct": int(parse_json_text(tr["content"]) is not None), "tool_call_success": int(any(tc.get("name") == "sum_numbers" for tc in tr["tool_calls"])), "error_rate": 0, "stability_placeholder": np.nan, "rule_satisfaction_rate": np.nan, "key_point_retention_rate": np.nan, "conflict_resolution_rate": np.nan, "spatial_distribution_balance": np.nan, "compression_ratio": np.nan, "accuracy": np.nan, "precision": np.nan, "recall": np.nan, "f1": np.nan, "selected_count": np.nan, "target_count": np.nan, "raw_response": str(tr["content"])[:4000], "parsed_json": json.dumps(parse_json_text(tr["content"]) or {}, ensure_ascii=False)})
            except Exception as exc:
                raw_exceptions.append({"vendor": vn, "task_id": f"{task_id}__tool", "model": llm_model, "error": f"{type(exc).__name__}: {exc}"})

            vlm_model, _ = choose_best_model(spec.preferred_vlm, cctx.get("models", []))
            if vlm_model:
                try:
                    vr = call_llm(cctx["client"], vlm_model, vlm_prompt, args.timeout_seconds, image_data_url=data_url)
                    pobj = parse_json_text(vr["content"]) or {}
                    pred_q = str(pobj.get("densest_quadrant", "")).upper().strip()
                    eval_rows.append({"vendor": vn, "model_name": vlm_model, "fallback_from": "", "task_id": f"{task_id}__vlm", "run_idx": 1, "response_seconds": round(float(vr["elapsed"]), 4), "task_completion": 1, "instruction_following": int(bool(pred_q)), "output_format_correct": int(bool(pobj)), "tool_call_success": 0, "error_rate": 0, "stability_placeholder": np.nan, "rule_satisfaction_rate": float(pred_q == densest), "key_point_retention_rate": np.nan, "conflict_resolution_rate": np.nan, "spatial_distribution_balance": np.nan, "compression_ratio": np.nan, "accuracy": float(pred_q == densest), "precision": np.nan, "recall": np.nan, "f1": np.nan, "selected_count": np.nan, "target_count": np.nan, "raw_response": str(vr["content"])[:4000], "parsed_json": json.dumps(pobj, ensure_ascii=False)})
                except Exception as exc:
                    raw_exceptions.append({"vendor": vn, "task_id": f"{task_id}__vlm", "model": vlm_model, "error": f"{type(exc).__name__}: {exc}"})

    raw_df, err_df = pd.DataFrame(eval_rows), pd.DataFrame(raw_exceptions)
    raw_df = aggregate_stability(raw_df) if not raw_df.empty else raw_df
    raw_df.to_csv(RAW_DIR / "raw_results.csv", index=False, encoding="utf-8-sig")
    (RAW_DIR / "raw_results.json").write_text(raw_df.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")
    err_df.to_csv(RAW_DIR / "errors.csv", index=False, encoding="utf-8-sig")

    if raw_df.empty:
        model_df = pd.DataFrame(columns=["vendor", "model_name", "n", "task_completion", "instruction_following", "output_format_correct", "tool_call_success", "response_seconds", "stability_placeholder", "error_rate", "rule_satisfaction_rate", "key_point_retention_rate", "conflict_resolution_rate", "spatial_distribution_balance", "compression_ratio", "accuracy", "precision", "recall", "f1"])
    else:
        model_df = raw_df.groupby(["vendor", "model_name"], dropna=False).agg(
            n=("model_name", "size"),
            task_completion=("task_completion", "mean"),
            instruction_following=("instruction_following", "mean"),
            output_format_correct=("output_format_correct", "mean"),
            tool_call_success=("tool_call_success", "mean"),
            response_seconds=("response_seconds", "mean"),
            stability_placeholder=("stability_placeholder", "mean"),
            error_rate=("error_rate", "mean"),
            rule_satisfaction_rate=("rule_satisfaction_rate", "mean"),
            key_point_retention_rate=("key_point_retention_rate", "mean"),
            conflict_resolution_rate=("conflict_resolution_rate", "mean"),
            spatial_distribution_balance=("spatial_distribution_balance", "mean"),
            compression_ratio=("compression_ratio", "mean"),
            accuracy=("accuracy", "mean"),
            precision=("precision", "mean"),
            recall=("recall", "mean"),
            f1=("f1", "mean"),
        ).reset_index()
    model_df.to_csv(RAW_DIR / "model_metrics.csv", index=False, encoding="utf-8-sig")

    if not model_df.empty:
        plot_df = model_df.copy()
        x, w = np.arange(len(plot_df)), 0.15
        bar_metrics = ["task_completion", "instruction_following", "output_format_correct", "tool_call_success", "f1"]
        fig, ax = plt.subplots(figsize=(12, 6))
        for i, m in enumerate(bar_metrics):
            ax.bar(x + i * w, plot_df[m].fillna(0).to_numpy(), width=w, label=m)
        ax.set_xticks(x + w * (len(bar_metrics) - 1) / 2)
        ax.set_xticklabels(plot_df["model_name"].tolist(), rotation=25, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_title("Core Quality Metrics by Model")
        ax.legend()
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "fig_core_quality_metrics.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

        fig, ax1 = plt.subplots(figsize=(11.8, 5.8))
        ax1.bar(plot_df["model_name"], plot_df["response_seconds"].fillna(0), color="#f59e0b", alpha=0.8)
        ax1.set_ylabel("Average Response Time (s)")
        ax1.tick_params(axis="x", rotation=25)
        ax2 = ax1.twinx()
        ax2.plot(plot_df["model_name"], plot_df["error_rate"].fillna(0), marker="o", color="#dc2626", linewidth=2)
        ax2.set_ylim(0, 1.05)
        ax2.set_ylabel("Error Rate")
        ax1.set_title("Speed vs Error Rate")
        ax1.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "fig_speed_vs_error.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

        heat_cols = ["rule_satisfaction_rate", "key_point_retention_rate", "conflict_resolution_rate", "spatial_distribution_balance", "accuracy", "precision", "recall", "f1"]
        hdf = plot_df.set_index("model_name")[heat_cols].fillna(0.0)
        fig, ax = plt.subplots(figsize=(11.5, 5.6))
        im = ax.imshow(hdf.to_numpy(dtype=float), aspect="auto", cmap="YlGnBu", vmin=0.0, vmax=1.0)
        ax.set_yticks(np.arange(len(hdf.index)))
        ax.set_yticklabels(hdf.index.tolist())
        ax.set_xticks(np.arange(len(hdf.columns)))
        ax.set_xticklabels(hdf.columns.tolist(), rotation=25, ha="right")
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        ax.set_title("Spatial + Classification Metrics Heatmap")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "fig_spatial_classification_heatmap.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

    excel_path = EXCEL_DIR / "model_comparison_results.xlsx"
    try:
        with pd.ExcelWriter(excel_path) as writer:
            pd.DataFrame(env_rows).to_excel(writer, sheet_name="env_keys", index=False)
            disc_df.to_excel(writer, sheet_name="model_discovery", index=False)
            raw_df.to_excel(writer, sheet_name="raw_results", index=False)
            model_df.to_excel(writer, sheet_name="model_metrics", index=False)
            err_df.to_excel(writer, sheet_name="errors", index=False)
        excel_msg = f"excel_ok:{excel_path}"
    except Exception as exc:
        excel_msg = f"excel_failed:{type(exc).__name__}:{exc}"

    summary_lines = [
        "Model Comparison Experiment Summary",
        f"output_root: {OUTPUT_ROOT}",
        f"groups_used: {len(groups)}",
        f"raw_rows: {len(raw_df)}",
        f"errors: {len(err_df)}",
        excel_msg,
        "",
        "Env key status:",
    ]
    for r in env_rows:
        summary_lines.append(f"  - {r['env']}: {'SET' if r['present'] else 'EMPTY'} (len={r['length']})")
    summary_lines.append("")
    summary_lines.append("Vendor discovery:")
    if disc_df.empty:
        summary_lines.append("  - no vendor discovery rows")
    else:
        for _, r in disc_df.iterrows():
            summary_lines.append(f"  - {r['vendor']} | base={r['base_url']} | key={r['key_present']} | models={r['model_count']} | error={r['error']}")
    (OUTPUT_ROOT / "summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")
    print("\n".join(summary_lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
