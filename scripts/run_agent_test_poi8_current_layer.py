"""
端到端测试：模拟「当前图层」为 TestDatasets 中的 points.geojson，
向智能体发送自然语言「请你对我的当前图层做综合，目标比例尺是1:xxxxxx」，
走完整 _run_scale_generalization 流水线（无真实 LLM/VLM，使用 NullProvider + 启发式验收）。

用法:
  python scripts/build_testdataset_poi8_agent.py
  python scripts/run_agent_test_poi8_current_layer.py
  python scripts/run_agent_test_poi8_current_layer.py --scale 220000
  python scripts/run_agent_test_poi8_current_layer.py --geojson E:\\path\\points.geojson
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ.setdefault("MPLBACKEND", "Agg")


def _find_default_geojson() -> Path | None:
    root = _ROOT / "TestDatasets"
    if not root.is_dir():
        return None
    candidates = sorted(root.glob("poi_8types_*/points.geojson"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _symbol_report(gdf) -> str:
    if gdf is None or len(gdf) == 0:
        return "empty"
    if "point_symbol" in gdf.columns:
        vc = gdf["point_symbol"].astype(str).value_counts()
        return ", ".join(f"{k}={int(v)}" for k, v in vc.items())
    return "no point_symbol column"


def _plot_compare(before_path: Path, after_path: Path, out_png: Path) -> None:
    import matplotlib.pyplot as plt
    import geopandas as gpd

    g0 = gpd.read_file(before_path)
    g1 = gpd.read_file(after_path)
    if g0.crs is not None and g1.crs is not None and str(g0.crs) != str(g1.crs):
        g1 = g1.to_crs(g0.crs)

    # markersize 过大时即使中心距已满足综合规则，仍会在图上“假重叠”
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=160)
    _ms = 14
    titles = [f"Before (n={len(g0)})", f"After (n={len(g1)})"]
    marker_map = {
        "circle": "o",
        "square": "s",
        "triangle": "^",
        "diamond": "D",
        "cross": "x",
        "star": "*",
        "plus": "P",
        "hexagon": "h",
    }
    colors = ["#c62828", "#1565c0", "#2e7d32", "#6a1b9a", "#ef6c00", "#00838f", "#4527a0", "#6d4c41"]

    def _draw(ax, gdf, title):
        ax.set_title(title)
        ax.set_axis_off()
        if "point_symbol" in gdf.columns:
            for i, (sym, sub) in enumerate(gdf.groupby("point_symbol")):
                mk = marker_map.get(str(sym).lower().strip(), "o")
                c = colors[i % len(colors)]
                sub.plot(ax=ax, color=c, markersize=_ms, marker=mk, linestyle="None", linewidth=0.35)
        else:
            gdf.plot(ax=ax, color="#d81b60", markersize=_ms, linewidth=0.35)

    _draw(axes[0], g0, titles[0])
    _draw(axes[1], g1, titles[1])
    b = g0.total_bounds
    pad = max(b[2] - b[0], b[3] - b[1]) * 0.05
    for ax in axes:
        ax.set_xlim(b[0] - pad, b[2] + pad)
        ax.set_ylim(b[1] - pad, b[3] + pad)
        ax.set_aspect("equal", adjustable="box")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--geojson", default="", help="测试用 points.geojson；默认取 TestDatasets/poi_8types_*/points.geojson 最新")
    ap.add_argument("--scale", type=int, default=250000, help="目标比例尺分母，如 250000 表示 1:250000")
    args = ap.parse_args()

    geo_path = Path(args.geojson) if args.geojson.strip() else _find_default_geojson()
    if geo_path is None or not geo_path.is_file():
        print("未找到 GeoJSON。请先运行: python scripts/build_testdataset_poi8_agent.py")
        return 1

    import geopandas as gpd

    gdf_in = gpd.read_file(geo_path)
    n_in = len(gdf_in)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = geo_path.parent / "agent_runs" / f"run_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    from src.agent.core import MapGenAgent
    from src.agent.providers import BaseLLMProvider, LLMResponse
    from src.agent import tools as agent_tools

    class NullProvider(BaseLLMProvider):
        def __init__(self) -> None:
            super().__init__(api_key="", model="null")

        def chat(self, messages, tools=None) -> LLMResponse:  # type: ignore[override]
            return LLMResponse("")

        def stream_chat(self, messages, tools=None):  # type: ignore[override]
            if False:
                yield ""

    def get_active_layer_info() -> Dict[str, Any]:
        gt = str(gdf_in.geometry.geom_type.iloc[0]) if n_in else "Point"
        return {
            "path": str(geo_path.resolve()),
            "name": "TestPOI8",
            "geom_type": gt,
            "feature_count": int(n_in),
        }

    # 无 UI：避免回调缺省导致异常
    agent_tools.ui_callbacks["get_active_layer_info"] = get_active_layer_info
    agent_tools.ui_callbacks["capture_map_screenshot"] = lambda _p: ""
    agent_tools.ui_callbacks["load_layer"] = lambda *_a, **_k: None
    agent_tools.ui_callbacks["zoom_to_layer"] = lambda *_a, **_k: None
    agent_tools.local_executor.workspace_path = str(_ROOT)

    user_msg = f"请你对我的当前图层做综合，目标比例尺是1:{args.scale}"

    agent = MapGenAgent(NullProvider())
    reply = agent.chat(user_msg)

    report_lines: List[str] = [
        f"时间: {run_id}",
        f"用户语句: {user_msg}",
        f"输入: {geo_path.resolve()}",
        f"输入要素: {n_in}",
        f"输入符号分布: {_symbol_report(gdf_in)}",
        "",
        "---------- 智能体回复 ----------",
        reply,
        "",
    ]

    # 解析输出 GeoJSON（Output files 行里多为「路径, 路径」，需兼容中文路径与全角逗号）
    out_geo = None
    fm = re.search(r"Output files:\s*([^\n]+)", reply)
    if fm:
        segment = fm.group(1)
        for sep in (",", "，"):
            segment = segment.replace(sep, ",")
        for token in segment.split(","):
            tkn = token.strip().strip('"')
            if not tkn.lower().endswith(".geojson"):
                continue
            p = Path(tkn)
            if p.is_file():
                out_geo = p
                break
    if out_geo is None:
        for m in re.finditer(r"([A-Za-z]:[^\s,，\n|<>\"]+\.geojson)", reply, flags=re.I):
            p = Path(m.group(1).strip())
            if p.is_file():
                out_geo = p
                break
    if out_geo is None:
        for m in re.finditer(r"([\w\d\u4e00-\u9fff\\/.-]+_gen_1_\d+(?:_r\d+)?\.geojson)", reply, flags=re.I):
            p = Path(m.group(1))
            if p.is_file():
                out_geo = p
                break
    if out_geo is None:
        stem = geo_path.stem
        parent = geo_path.parent
        matches = sorted(parent.glob(f"{stem}_gen_1_{args.scale}*.geojson"), key=lambda x: x.stat().st_mtime, reverse=True)
        if matches:
            out_geo = matches[0]

    if out_geo and out_geo.is_file():
        gdf_out = gpd.read_file(out_geo)
        report_lines.extend(
            [
                f"输出 GeoJSON: {out_geo.resolve()}",
                f"输出要素: {len(gdf_out)}",
                f"输出符号分布: {_symbol_report(gdf_out)}",
            ]
        )
        compare_png = out_dir / f"compare_before_after_1_{args.scale}.png"
        try:
            _plot_compare(geo_path, out_geo, compare_png)
            report_lines.append(f"对比图: {compare_png.resolve()}")
        except Exception as exc:
            report_lines.append(f"对比图生成失败: {exc}")
    else:
        report_lines.append("未能从回复或默认路径定位输出 GeoJSON。")

    report_path = out_dir / "TEST_REPORT.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    summary = {
        "run_id": run_id,
        "user_message": user_msg,
        "input_geojson": str(geo_path.resolve()),
        "input_count": n_in,
        "target_scale": args.scale,
        "output_geojson": str(out_geo.resolve()) if out_geo and out_geo.is_file() else "",
        "report_txt": str(report_path.resolve()),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n".join(report_lines))
    print(f"\n[已写入] {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
