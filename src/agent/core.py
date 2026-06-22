"""
MapGenAI agent core.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
import os
import math
import re
from typing import Any, Callable, Dict, List, Optional

from .code_generator import CodeGenerator
from .generalization_planner import GeneralizationPlanner, PlanningRequest
from .knowledge_base import AlgorithmKnowledgeBase
from .providers import BaseLLMProvider, Message, create_provider
from .quality_checker import QualityChecker
from .result_manager import ResultManager
from .task_classifier import TaskClassifier
from .tools import tool_registry, ui_callbacks
from .vlm_verifier import VLMVerifier, VerifyInput
from src.ui.point_symbol_core import POI_SYMBOL_RULES

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are MapGenAI, a GIS coding assistant.
Use tools to solve tasks and keep responses concise.
Prefer run_python_script/run_qgis_script for executable GIS work.
For map generalization tasks, follow skill `map-generalization`:
- define source/target scale and readability constraints
- choose operators by geometry type (selection/simplify/smooth/aggregate/displacement/typification)
- run iterative quality checks and preserve topology/network continuity
Always end with:
- Completion status (completed / completed_with_warnings / failed)
- Output files (full paths when available)
- Short note of what was verified
"""


@dataclass
class AgentContext:
    workspace_path: str = ""
    current_layers: List[str] = field(default_factory=list)
    history: List[Dict[str, Any]] = field(default_factory=list)


class MapGenAgent:
    def __init__(self, provider: BaseLLMProvider):
        self.provider = provider
        self.context = AgentContext()
        self.messages: List[Message] = [Message("system", SYSTEM_PROMPT)]
        self.event_callback: Optional[Callable[[str, Any], None]] = None
        self.max_auto_fix_attempts = 1

        self.task_classifier = TaskClassifier(provider)
        self.knowledge_base = AlgorithmKnowledgeBase()
        self.code_generator = CodeGenerator(provider, self.knowledge_base)
        self.quality_checker = QualityChecker()
        self.vlm_verifier = VLMVerifier(callback=ui_callbacks.get("vlm_verify"))
        self.result_manager = ResultManager(
            {
                "load_layer": lambda path, layer_name=None: ui_callbacks.get("load_layer", lambda *_: None)(
                    path, layer_name
                )
            }
        )

    def set_event_callback(self, callback: Callable[[str, Any], None]):
        self.event_callback = callback

    def _emit(self, event_type: str, data: Any):
        if self.event_callback:
            self.event_callback(event_type, data)

    def set_workspace(self, path: str):
        self.context.workspace_path = path

    def add_layer(self, layer_name: str):
        if layer_name not in self.context.current_layers:
            self.context.current_layers.append(layer_name)

    def remove_layer(self, layer_name: str):
        if layer_name in self.context.current_layers:
            self.context.current_layers.remove(layer_name)

    def get_context_info(self) -> str:
        info = []
        if self.context.workspace_path:
            info.append(f"workspace: {self.context.workspace_path}")
        if self.context.current_layers:
            info.append(f"layers: {', '.join(self.context.current_layers)}")
        return "\n".join(info) if info else "no active context"

    def _extract_request_hints(self, user_message: str) -> Dict[str, Any]:
        msg = str(user_message or "")
        files = re.findall(r"([A-Za-z0-9_\-./\\]+\.(?:shp|geojson|gpkg))", msg, flags=re.I)
        scale_match = re.search(r"1\s*[:：]\s*(\d{4,9})", msg, flags=re.I)
        return {
            "input_file": files[0] if files else "",
            "target_scale": int(scale_match.group(1)) if scale_match else 0,
            "has_current_layer": bool(re.search(r"当前图层|current layer|this layer", msg, flags=re.I)),
        }

    @staticmethod
    def _parse_scale_denominator(value: Any, default: int = 0) -> int:
        if value is None:
            return default
        if isinstance(value, bool):
            return default
        if isinstance(value, (int, float)):
            return int(value)
        text = str(value).strip()
        if not text:
            return default
        text = text.replace(",", "").replace("，", "")
        m = re.search(r"1\s*[:：]\s*(\d{2,9})", text)
        if m:
            return int(m.group(1))
        m = re.search(r"(\d{2,9})", text)
        if m:
            return int(m.group(1))
        return default

    @staticmethod
    def _resolve_existing_vector_path(path: str) -> str:
        raw = str(path or "").strip().strip("\"'")
        if not raw:
            return ""
        expanded = os.path.abspath(os.path.expanduser(raw))
        if os.path.exists(expanded):
            return expanded
        root, ext = os.path.splitext(expanded)
        candidates = []
        for candidate_ext in (".geojson", ".json", ".gpkg", ".shp"):
            if candidate_ext.lower() == ext.lower():
                continue
            candidates.append(root + candidate_ext)
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
        return expanded

    def _build_dynamic_todo(self, user_message: str, task_type: str = "unknown", params: Optional[Dict[str, Any]] = None) -> List[str]:
        hints = self._extract_request_hints(user_message)
        merged = {}
        merged.update(hints)
        if isinstance(params, dict):
            merged.update({k: v for k, v in params.items() if v is not None})

        input_hint = str(merged.get("input_file") or "").strip()
        if not input_hint and merged.get("use_current_layer", False):
            input_hint = "当前图层"
        if not input_hint and merged.get("has_current_layer", False):
            input_hint = "当前图层"
        input_hint = input_hint or "待解析数据源"

        scale_hint = self._parse_scale_denominator(merged.get("target_scale", 0), 0)
        scale_text = f"1:{scale_hint}" if scale_hint > 0 else "未指定比例尺"

        operator_hint = {
            "scale_generalization": "按几何类型组合 selection/simplify/smooth/aggregate/displacement",
            "line_simplification": "线要素简化与平滑参数",
            "polygon_generalization": "面要素合并/简化与拓扑保持",
            "spatial_analysis": "空间分析算子与参数（buffer/clip/overlay）",
            "coordinate_transform": "目标坐标系与重投影参数",
            "topology_repair": "拓扑修复规则与容差",
            "unknown": "根据上下文选择最小必要算子",
        }.get(task_type, "根据上下文选择最小必要算子")

        return [
            f"解析任务与数据源：{input_hint}，目标比例尺 {scale_text}",
            f"确定处理策略：任务类型 `{task_type}`，策略 {operator_hint}",
            "执行工具并校验结果：检查日志、质量指标与拓扑",
            "汇总交付：输出文件路径 + 完成状态说明",
        ]

    def _emit_dynamic_plan(self, user_message: str, task_type: str = "unknown", params: Optional[Dict[str, Any]] = None) -> None:
        plan_items = self._build_dynamic_todo(user_message, task_type=task_type, params=params)
        total = len(plan_items)
        for idx, item in enumerate(plan_items, start=1):
            self._emit("thinking", f"Plan {idx}/{total}: {item}")

    def chat(self, user_message: str) -> str:
        pipeline_reply = self._try_planned_pipeline(user_message)
        if pipeline_reply:
            self.messages.append(Message("assistant", pipeline_reply))
            self._emit("response", pipeline_reply)
            return pipeline_reply

        context_info = self.get_context_info()
        full_message = f"{user_message}\n\n[context]\n{context_info}"
        self.messages.append(Message("user", full_message))

        max_turns = 8
        current_turn = 0
        self._emit_dynamic_plan(user_message=user_message)
        while current_turn < max_turns:
            current_turn += 1
            tools = tool_registry.list_tools()
            self._emit("thinking", f"Todo 1/4: 第 {current_turn} 轮推理，复核上下文与约束")
            self._emit("thinking", "Todo 2/4: 生成本轮工具调用参数")
            self._emit("thinking", "Todo 3/4: 执行工具并分析返回")
            self._emit("thinking", "Todo 4/4: 更新输出文件与完成状态")

            try:
                response = self.provider.chat(self.messages, tools)
            except Exception as exc:
                error_msg = f"LLM call failed: {exc}"
                self._emit("error", error_msg)
                return error_msg

            if not response.tool_calls:
                reply = response.content or ""
                self.messages.append(Message("assistant", reply))
                self._emit("response", reply)
                return reply

            self.messages.append(Message("assistant", content=response.content or "", tool_calls=response.tool_calls))
            if response.content:
                self._emit("thought", response.content)

            tool_results = []
            for tool_call in response.tool_calls:
                self._emit("tool_start", {"name": tool_call.name, "args": tool_call.arguments})
                result = tool_registry.execute(tool_call.name, tool_call.arguments)
                self._emit("tool_end", {"name": tool_call.name, "result": result})

                tool_results.append(
                    {
                        "tool_call_id": getattr(tool_call, "id", None),
                        "name": tool_call.name,
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )

                if tool_call.name in {"run_python_script", "run_qgis_script"}:
                    auto_fix = self._auto_fix_and_retry(tool_call, result, tools)
                    if auto_fix:
                        tool_results.append(auto_fix)

            result_text = "Tool results:\n" + json.dumps(tool_results, ensure_ascii=False, indent=2)
            has_error = any(
                isinstance(item, dict)
                and isinstance(item.get("content"), str)
                and ('"status": "error"' in item["content"] or '"error"' in item["content"])
                for item in tool_results
            )
            if has_error:
                result_text += "\n\nIf failed, analyze traceback first, then fix and retry."
            self.messages.append(Message("user", result_text))

        return "Reached max turns, task not completed."

    def _try_planned_pipeline(self, user_message: str) -> Optional[str]:
        try:
            self._emit("thinking", "Todo 1/4: classify task and resolve data source")
            task = self.task_classifier.classify(user_message)
            task_type = task.get("task_type", "unknown")
            confidence = float(task.get("confidence", 0.0) or 0.0)
            params = task.get("parameters", {}) or {}
            self._emit_dynamic_plan(user_message=user_message, task_type=task_type, params=params)
            self._emit(
                "thought",
                f"Task analysis: type={task_type}, confidence={confidence:.2f}, parameters={json.dumps(params, ensure_ascii=False)}",
            )

            if task_type == "unknown" or confidence < 0.5:
                return None

            if task_type == "scale_generalization":
                return self._run_scale_generalization(task)

            if "input_file" not in params:
                return None

            if "output_file" not in params:
                params["output_file"] = re.sub(
                    r"\.(shp|geojson|gpkg)$",
                    r"_result.\1",
                    str(params["input_file"]),
                    flags=re.I,
                )
            task["parameters"] = params

            self._emit("thinking", f"Todo 2/4: generate code for `{task_type}`")
            code = self.code_generator.generate(task)

            self._emit("thinking", "Todo 3/4: quality check and execute")
            quality = self.quality_checker.check(code)
            if not quality.get("valid", False):
                issues = "; ".join(quality.get("issues", []))
                return (
                    f"Task `{task_type}` failed quality check: {issues}\n"
                    "Completion status: failed\n"
                    "Output files: none"
                )

            warnings = quality.get("warnings", [])
            if warnings:
                self._emit("thought", "Warnings: " + "; ".join(warnings))

            self._emit("tool_start", {"name": "run_python_script", "args": {"code": code}})
            result = tool_registry.execute("run_python_script", {"code": code})
            self._emit("tool_end", {"name": "run_python_script", "result": result})

            handled = self.result_manager.handle_result(result)
            if handled.get("status") != "success":
                return (
                    f"Task `{task_type}` failed.\n"
                    f"Error: {handled.get('message', '')}\n"
                    f"Suggestion: {handled.get('suggestion', '')}"
                )

            output_files = handled.get("output_files", [])
            files_text = ", ".join(output_files) if output_files else "none"
            stdout = (handled.get("stdout") or "").strip()
            self._emit("thinking", "Todo 4/4: summarize outputs and completion status")
            msg = (
                f"Planned pipeline succeeded for `{task_type}`.\n"
                "Completion status: completed\n"
                f"Output files: {files_text}\n"
                "Verified: execution finished and output files were parsed."
            )
            if stdout:
                msg += f"\nExecution log: {stdout[:300]}"
            return msg
        except Exception:
            logger.exception("Planned pipeline failed")
            return None

    def _get_active_layer_info(self) -> Dict[str, Any]:
        cb = ui_callbacks.get("get_active_layer_info")
        if not callable(cb):
            return {}
        try:
            info = cb()
            return info if isinstance(info, dict) else {}
        except Exception:
            return {}

    def _resolve_generalization_input(self, params: Dict[str, Any]) -> Dict[str, Any]:
        resolved = dict(params or {})
        resolved["target_scale"] = self._parse_scale_denominator(resolved.get("target_scale"), 0)
        resolved["source_scale"] = self._parse_scale_denominator(resolved.get("source_scale"), 10000)
        input_file = self._resolve_existing_vector_path(str(resolved.get("input_file", "")).strip())
        if input_file:
            resolved["input_file"] = input_file
        if input_file and os.path.exists(input_file):
            self._enrich_generalization_input_meta(resolved)
            return resolved

        active = self._get_active_layer_info()
        active_path = self._resolve_existing_vector_path(str(active.get("path", "")).strip())
        if active_path and os.path.exists(active_path):
            resolved["input_file"] = active_path
            resolved.setdefault("layer_name", str(active.get("name", "")).strip() or "generalized_layer")
            resolved.setdefault("geometry_type", str(active.get("geom_type", "")).strip().lower())
            resolved.setdefault("feature_count", int(active.get("feature_count", 0) or 0))
            self._enrich_generalization_input_meta(resolved)
        return resolved

    def _enrich_generalization_input_meta(self, resolved: Dict[str, Any]) -> None:
        input_file = self._resolve_existing_vector_path(str(resolved.get("input_file", "")).strip())
        if input_file:
            resolved["input_file"] = input_file
        if not input_file or not os.path.exists(input_file):
            return
        need_geom = not str(resolved.get("geometry_type", "")).strip()
        need_count = int(resolved.get("feature_count", 0) or 0) <= 0
        if not (need_geom or need_count):
            return
        try:
            import geopandas as gpd  # type: ignore

            gdf = gpd.read_file(input_file)
            if need_count:
                resolved["feature_count"] = int(len(gdf))
            if need_geom and len(gdf) > 0:
                gt = str(gdf.geometry.geom_type.iloc[0]).lower()
                if gt.startswith("multi"):
                    gt = gt.replace("multi", "", 1)
                resolved["geometry_type"] = gt
        except Exception:
            # Keep pipeline resilient; planner will fallback to default if metadata is unavailable.
            return

    def _capture_map_screenshot(self, path: str) -> str:
        cb = ui_callbacks.get("capture_map_screenshot")
        if not callable(cb):
            return ""
        try:
            out = cb(path)
            if isinstance(out, str) and out.strip():
                return out.strip()
            return path if os.path.exists(path) else ""
        except Exception:
            return ""

    def _extract_metrics_from_stdout(self, stdout: str) -> Dict[str, Any]:
        text = stdout or ""
        m = re.search(r"metrics:\s*(\{.*\})", text, flags=re.S)
        if not m:
            return {}
        try:
            return json.loads(m.group(1))
        except Exception:
            return {}

    def _extract_projected_crs_from_stdout(self, stdout: str) -> str:
        text = stdout or ""
        m = re.search(r"crs_auto_projected:\s*([^\r\n]+)", text, flags=re.I)
        return m.group(1).strip() if m else ""

    def _infer_point_semantic_profile(self, input_file: str) -> Dict[str, Any]:
        profile: Dict[str, Any] = {
            "semantic_type": "generic_point",
            "semantic_evidence": [],
            "structure_type": "unknown",
            "structure_evidence": [],
            "data_label": "",
            "distribution_summary": "",
        }
        if not input_file or not os.path.exists(input_file):
            return profile
        try:
            import geopandas as gpd  # type: ignore
            import pandas as pd  # type: ignore

            gdf = gpd.read_file(input_file)
            if len(gdf) == 0:
                return profile

            cols = [str(c) for c in gdf.columns]
            low_cols = [c.lower() for c in cols]
            text_blob = " ".join(low_cols)
            value_blob = ""
            try:
                sample_parts = []
                preferred_cols = [c for c in cols if str(c).lower() in {"name", "type", "typecode", "class", "category", "poi", "kind", "名称", "类别", "类型"}]
                if not preferred_cols:
                    preferred_cols = cols[: min(6, len(cols))]
                for c in preferred_cols:
                    series = gdf[c].dropna().astype(str).head(40).tolist()
                    if series:
                        sample_parts.extend(series)
                value_blob = " ".join(sample_parts).lower()
            except Exception:
                value_blob = ""

            if any(k in value_blob for k in ["加油站", "fuel", "gas station", "petrol", "charging station", "充电站"]):
                profile["semantic_type"] = "commercial_poi_points"
                profile["data_label"] = "加油站/能源补给点"
                profile["semantic_evidence"].append("sample values contain fuel or charging keywords")
            elif any(k in value_blob for k in ["餐厅", "饭店", "restaurant", "food", "cafe", "咖啡"]):
                profile["semantic_type"] = "commercial_poi_points"
                profile["data_label"] = "餐饮服务点"
                profile["semantic_evidence"].append("sample values contain catering keywords")
            elif any(k in value_blob for k in ["医院", "school", "学校", "clinic", "hospital"]):
                profile["semantic_type"] = "public_service_points"
                profile["data_label"] = "公共服务设施点"
                profile["semantic_evidence"].append("sample values contain public service keywords")
            if any(k in text_blob for k in ["line", "route", "station", "metro", "subway", "rail", "pointname"]):
                profile["semantic_type"] = "transport_network_points"
                profile["semantic_evidence"].append("detected transport-like fields")
            elif any(k in text_blob for k in ["hospital", "school", "clinic", "service", "facility"]):
                profile["semantic_type"] = "public_service_points"
                profile["semantic_evidence"].append("detected public-service fields")
            elif any(k in text_blob for k in ["sensor", "monitor", "alarm", "risk", "event"]):
                profile["semantic_type"] = "monitoring_points"
                profile["semantic_evidence"].append("detected monitoring-like fields")
            elif any(k in text_blob for k in ["shop", "poi", "store", "mall", "business", "food"]):
                profile["semantic_type"] = "commercial_poi_points"
                profile["semantic_evidence"].append("detected POI/business-like fields")
            else:
                profile["semantic_evidence"].append("no strong semantic keyword; use generic profile")

            metric = gdf.copy()
            if metric.crs is None:
                return profile
            if getattr(metric.crs, "is_geographic", False):
                try:
                    metric = metric.to_crs(metric.estimate_utm_crs())
                except Exception:
                    metric = metric.to_crs("EPSG:3857")
            b = metric.total_bounds
            width = max(1e-9, float(b[2] - b[0]))
            height = max(1e-9, float(b[3] - b[1]))
            aspect = max(width / height, height / width)

            # Nearest-neighbor coefficient of variation as a lightweight structure cue.
            rep = metric.geometry.representative_point()
            dists = []
            if len(rep) > 2:
                sidx = rep.sindex
                for i, g in enumerate(rep):
                    cand = [j for j in sidx.intersection(g.buffer(max(width, height) * 0.02).bounds) if j != i]
                    best = None
                    for j in cand:
                        d = g.distance(rep.iloc[j])
                        if best is None or d < best:
                            best = d
                    if best is not None:
                        dists.append(float(best))
            if dists:
                mean_d = sum(dists) / len(dists)
                std_d = (sum((x - mean_d) ** 2 for x in dists) / len(dists)) ** 0.5
                cv = std_d / max(1e-9, mean_d)
            else:
                cv = 0.0

            if aspect >= 3.0:
                profile["structure_type"] = "corridor"
                profile["structure_evidence"].append(f"elongated extent aspect={aspect:.2f}")
                profile["distribution_summary"] = "点位沿狭长方向展开，带有明显走廊状分布特征"
            elif cv >= 0.9:
                profile["structure_type"] = "clustered"
                profile["structure_evidence"].append(f"high nearest-neighbor CV={cv:.2f}")
                profile["distribution_summary"] = "点位集聚明显，局部热点较强"
            elif cv <= 0.35:
                profile["structure_type"] = "uniform"
                profile["structure_evidence"].append(f"low nearest-neighbor CV={cv:.2f}")
                profile["distribution_summary"] = "点位分布较均匀，适合做均衡保留"
            else:
                profile["structure_type"] = "mixed"
                profile["structure_evidence"].append(f"moderate nearest-neighbor CV={cv:.2f}")
                profile["distribution_summary"] = "点位既有局部聚集，也存在离散分布，属于混合型格局"

            # Coarse occupancy diagnostics for center-periphery / multi-core patterns.
            try:
                minx, miny, maxx, maxy = [float(v) for v in metric.total_bounds]
                dx = max(1e-9, maxx - minx)
                dy = max(1e-9, maxy - miny)
                gx = ((metric.geometry.x - minx) / dx * 4).astype(int).clip(lower=0, upper=3)
                gy = ((metric.geometry.y - miny) / dy * 4).astype(int).clip(lower=0, upper=3)
                cell = pd.Series([f"{int(x)}_{int(y)}" for x, y in zip(gx.tolist(), gy.tolist())])
                freq = cell.value_counts(dropna=False)
                shares = (freq / float(max(1, len(metric)))).tolist()
                top1 = float(shares[0]) if shares else 0.0
                top2 = float(sum(shares[:2])) if shares else 0.0
                center_mask = ((metric.geometry.x >= minx + dx * 0.35) & (metric.geometry.x <= minx + dx * 0.65) &
                               (metric.geometry.y >= miny + dy * 0.35) & (metric.geometry.y <= miny + dy * 0.65))
                center_share = float(center_mask.sum()) / float(max(1, len(metric)))
                if profile["structure_type"] != "corridor":
                    if top2 >= 0.52 and top1 < 0.40:
                        profile["structure_type"] = "multi-core"
                        profile["structure_evidence"].append(f"grid hot cells top2 share={top2:.2f}")
                        profile["distribution_summary"] = "热点不止一个，属于多核心分布"
                    elif center_share >= 0.38 and top1 >= 0.20:
                        profile["structure_type"] = "center-periphery"
                        profile["structure_evidence"].append(f"center share={center_share:.2f}")
                        profile["distribution_summary"] = "中心区域更密，外围更疏，呈中心-边缘分布"
            except Exception:
                pass

            # Keep top candidate feature fields for explainability.
            rank_fields = []
            for c in cols:
                lc = c.lower()
                if any(k in lc for k in ["importance", "priority", "weight", "level", "rank", "score", "line", "type", "class", "name"]):
                    rank_fields.append(c)
            profile["candidate_fields"] = rank_fields[:8]
            if not profile.get("data_label"):
                if profile["semantic_type"] == "transport_network_points":
                    profile["data_label"] = "交通网络相关点"
                elif profile["semantic_type"] == "public_service_points":
                    profile["data_label"] = "公共服务设施点"
                elif profile["semantic_type"] == "monitoring_points":
                    profile["data_label"] = "监测事件点"
                elif profile["semantic_type"] == "commercial_poi_points":
                    profile["data_label"] = "商业兴趣点"
                else:
                    profile["data_label"] = "通用点要素"
            return profile
        except Exception:
            return profile

    def _semantic_type_label(self, semantic_type: str, data_label: str = "") -> str:
        if data_label:
            return data_label
        mapping = {
            "generic_point": "通用点要素",
            "transport_network_points": "交通网络相关点",
            "public_service_points": "公共服务设施点",
            "monitoring_points": "监测事件点",
            "commercial_poi_points": "商业兴趣点",
        }
        return mapping.get(str(semantic_type or ""), str(semantic_type or "未知点类型"))

    def _structure_type_label(self, structure_type: str) -> str:
        mapping = {
            "unknown": "未知结构",
            "corridor": "走廊型",
            "clustered": "集聚型",
            "uniform": "均匀型",
            "mixed": "混合型",
            "multi-core": "多核心型",
            "center-periphery": "中心-边缘型",
        }
        return mapping.get(str(structure_type or ""), str(structure_type or "未知结构"))

    def _build_point_data_specific_reasoning(self, profile: Dict[str, Any], target_scale: int) -> List[str]:
        semantic_label = self._semantic_type_label(str(profile.get("semantic_type", "")), str(profile.get("data_label", "")))
        structure_label = self._structure_type_label(str(profile.get("structure_type", "")))
        distribution = str(profile.get("distribution_summary", "")).strip()
        lines = [
            f"当前数据识别为“{semantic_label}”，目标比例尺为 1:{target_scale}。",
            f"从空间分布看，该数据属于“{structure_label}”格局。{distribution}",
        ]
        if "加油站" in semantic_label or "能源补给" in semantic_label:
            lines.append("这类点位既要保留主干道路、城区热点和稀疏区域代表点，也要避免过密区域符号冲突。")
        elif "商业" in semantic_label or "餐饮" in semantic_label:
            lines.append("这类 POI 需要兼顾热点代表性、区域覆盖和类别均衡，不能只保留中心区高密度点。")
        elif "公共服务" in semantic_label:
            lines.append("这类设施点更强调区域可达性与覆盖完整性，不能让边缘片区完全空白。")
        else:
            lines.append("这类点要素会优先保证空间覆盖，再兼顾局部热点和读图清晰度。")
        return lines

    def _build_operator_reasoning(self, profile: Dict[str, Any], plan: Dict[str, Any]) -> str:
        semantic_type = str(profile.get("semantic_type", "generic_point"))
        structure_type = str(profile.get("structure_type", "unknown"))
        op_names = [str(op.get("name", "")) for op in plan.get("operators", []) if isinstance(op, dict)]
        reasons = [
            f"数据类型={self._semantic_type_label(semantic_type, str(profile.get('data_label', '')))}",
            f"空间结构={self._structure_type_label(structure_type)}",
            f"候选算子={op_names}",
        ]
        data_label = str(profile.get("data_label", ""))
        if semantic_type == "transport_network_points":
            reasons.append("优先保留网络骨架、关键节点和空间连续性")
        elif semantic_type == "monitoring_points":
            reasons.append("优先保证覆盖完整性，避免局部区域被过度删减")
        elif semantic_type == "commercial_poi_points" and ("加油站" in data_label or "能源补给" in data_label):
            reasons.append("优先保留主路沿线点、区域热点和稀疏区域代表点")
        elif semantic_type == "commercial_poi_points":
            reasons.append("优先兼顾区域覆盖、热点代表性和类别均衡")
        else:
            reasons.append("优先兼顾空间覆盖、局部热点和稀疏区域保护")
        return "; ".join(reasons)

    def _format_user_visible_decision_summary(
        self,
        *,
        target_scale: int,
        profile: Dict[str, Any],
        plan: Dict[str, Any],
        qa_metrics: Dict[str, Any],
        verify: Dict[str, Any],
        round_index: int,
        basis_doc_path: str,
    ) -> str:
        """对话/报告里可直接阅读的决策说明（不替代 basis 文档，但不再用一句「详见文档」敷衍）。"""
        lines: List[str] = []
        sem = self._semantic_type_label(str(profile.get("semantic_type", "")), str(profile.get("data_label", "")))
        struct = self._structure_type_label(str(profile.get("structure_type", "")))
        lines.append(f"【决策摘要 | 目标 1:{target_scale}】将数据判为「{sem}」，空间格局「{struct}」。")

        sev = profile.get("semantic_evidence") or []
        stv = profile.get("structure_evidence") or []
        bits: List[str] = []
        if isinstance(sev, (list, tuple)) and sev:
            bits.append("语义：" + "；".join(str(x) for x in list(sev)[:4]))
        if isinstance(stv, (list, tuple)) and stv:
            bits.append("结构：" + "；".join(str(x) for x in list(stv)[:4]))
        if bits:
            lines.append("【依据】 " + " | ".join(bits))

        operators = plan.get("operators", []) if isinstance(plan, dict) else []
        op_bits: List[str] = []
        for op in operators:
            if isinstance(op, dict):
                nm = str(op.get("name", "") or "?")
                pr = op.get("params", {})
                ps = ""
                if isinstance(pr, dict) and pr:
                    ps = json.dumps(pr, ensure_ascii=False)
                    if len(ps) > 120:
                        ps = ps[:117] + "..."
                op_bits.append(f"{nm}{(' ' + ps) if ps else ''}")
        if op_bits:
            lines.append("【执行链】 " + " → ".join(op_bits))

        stop = plan.get("stop_conditions", {}) if isinstance(plan, dict) else {}
        if isinstance(stop, dict) and stop:
            ss = json.dumps(stop, ensure_ascii=False)
            if len(ss) > 160:
                ss = ss[:157] + "..."
            lines.append(f"【停轮阈值】 {ss}")

        ic = qa_metrics.get("input_count", "—")
        oc = qa_metrics.get("output_count", "—")
        rr_raw = qa_metrics.get("retention_rate", "—")
        try:
            rrv = float(rr_raw)
            rr = f"{rrv:.1%}" if 0.0 <= rrv <= 1.0 else str(rr_raw)
        except (TypeError, ValueError):
            rr = str(rr_raw)
        ph = qa_metrics.get("point_hotspot_coverage_ratio")
        prcv = qa_metrics.get("point_region_coverage_ratio")
        cc = qa_metrics.get("conflict_count_total", "—")
        phs = f"{float(ph):.3f}" if isinstance(ph, (int, float)) else "—"
        prs = f"{float(prcv):.3f}" if isinstance(prcv, (int, float)) else "—"
        lines.append(
            f"【本轮指标 | 第 {round_index} 轮】输入 {ic} → 输出 {oc}，保留率 {rr}；"
            f"热点格覆盖 {phs}，区域覆盖 {prs}；符号冲突 {cc}。"
        )

        passed = verify.get("pass")
        lines.append(f"【验收】VLM {'通过' if passed else '未通过'}，score={verify.get('score', '—')}")
        failed = verify.get("failed_rules") or []
        if failed:
            lines.append(f"【未满足项】{failed}")
        ev = verify.get("evidence") or []
        if isinstance(ev, (list, tuple)) and ev:
            lines.append("【验收说明】" + " ".join(str(x) for x in list(ev)[:3]))

        if basis_doc_path:
            lines.append(f"【附录】算子参数与校验字段全文见同目录：{os.path.basename(basis_doc_path)}")

        return "\n".join(lines)

    def _write_selection_basis_doc(
        self,
        base_path: str,
        input_file: str,
        target_scale: int,
        profile: Dict[str, Any],
        plan: Dict[str, Any],
        verify: Dict[str, Any],
        output_files: List[str],
        projected_crs: str = "",
    ) -> str:
        try:
            out_md = f"{base_path}_selection_basis.md"
            operators = plan.get("operators", [])
            op_lines = []
            for op in operators:
                if not isinstance(op, dict):
                    continue
                op_lines.append(f"- `{op.get('name')}` params={json.dumps(op.get('params', {}), ensure_ascii=False)}")

            source_crs = "undefined"
            feature_count = "unknown"
            geometry_type = "unknown"
            try:
                import geopandas as gpd  # type: ignore

                src_gdf = gpd.read_file(input_file)
                source_crs = str(src_gdf.crs) if src_gdf.crs else "undefined"
                feature_count = str(int(len(src_gdf)))
                if len(src_gdf) > 0:
                    geometry_type = str(src_gdf.geometry.geom_type.iloc[0])
            except Exception:
                pass

            md = [
                "# Point Generalization Selection Basis",
                "",
                "## Task",
                f"- input: `{input_file}`",
                f"- target_scale: `1:{target_scale}`",
                f"- feature_count: `{feature_count}`",
                f"- geometry_type: `{geometry_type}`",
                f"- source_crs: `{source_crs}`",
                f"- projected_crs: `{projected_crs or 'N/A'}`",
                "",
                "## Skill-Driven Analysis",
                "- skill reference: `skills/point_generalization_skill.md`",
                f"- semantic_type: `{profile.get('semantic_type')}`",
                f"- structure_type: `{profile.get('structure_type')}`",
                f"- semantic_evidence: {profile.get('semantic_evidence')}",
                f"- structure_evidence: {profile.get('structure_evidence')}",
                f"- candidate_fields: {profile.get('candidate_fields', [])}",
                f"- poi_symbol_rules: {list(POI_SYMBOL_RULES.keys())}",
                "",
                "## Operator Decision",
                *op_lines,
                "",
                "## Algorithm Snippet",
                "```python",
                "score = semantic_score + structure_score + balance_score + sparse_region_bonus",
                "selected = grouped_anchor_keep + region_quota_keep + hotspot_representative_keep + axis_bin_continuity_keep + grid_best_keep + top_score_fill",
                "selected = resolve_conflicts_by_min_separation(selected)",
                "```",
                "",
                "## Validation (VLM + LLM)",
                f"- pass: {verify.get('pass')}",
                f"- score: {verify.get('score')}",
                f"- vlm_mode: {verify.get('vlm_mode', 'unknown')}",
                f"- real_vlm_called: {verify.get('real_vlm_called', False)}",
                f"- failed_rules: {verify.get('failed_rules', [])}",
                f"- evidence: {verify.get('evidence', [])}",
                "",
                "## Outputs",
                *[f"- `{p}`" for p in (output_files or [])],
                "",
            ]
            with open(out_md, "w", encoding="utf-8") as f:
                f.write("\n".join(md))
            return out_md
        except Exception:
            return ""

    def _build_generalization_code(
        self,
        plan: Dict[str, Any],
        input_file: str,
        output_geojson: str,
        output_png: str,
        output_layer_name: str,
        include_geojson: bool,
        include_png: bool,
    ) -> str:
        plan_json = json.dumps(plan, ensure_ascii=False)
        poi_symbol_rules_json = json.dumps(
            {
                "便利店": {"symbol": "convenience", "color": "#d23b7d", "label": ""},
                "学校": {"symbol": "school", "color": "#2e7d32", "label": "S"},
                "停车场": {"symbol": "parking", "color": "#1976d2", "label": "P"},
                "酒店": {"symbol": "hotel", "color": "#7b4fa1", "label": "H"},
                "银行": {"symbol": "bank", "color": "#00897b", "label": "B"},
                "医院": {"symbol": "hospital", "color": "#d32f2f", "label": "+"},
                "餐饮": {"symbol": "restaurant", "color": "#ef8a24", "label": "R"},
                "交通设施": {"symbol": "transit", "color": "#4e79a7", "label": "T"},
            },
            ensure_ascii=False,
        )
        poi_aliases_json = json.dumps(
            {
                "便利店": "便利店", "便利": "便利店", "超市": "便利店", "商店": "便利店",
                "学校": "学校", "大学": "学校", "中学": "学校", "小学": "学校", "幼儿园": "学校",
                "停车场": "停车场", "停车": "停车场",
                "酒店": "酒店", "宾馆": "酒店", "住宿": "酒店",
                "银行": "银行", "ATM": "银行", "atm": "银行",
                "医院": "医院", "诊所": "医院", "医疗": "医院",
                "餐饮": "餐饮", "餐厅": "餐饮", "饭店": "餐饮", "美食": "餐饮",
                "交通设施": "交通设施", "交通": "交通设施", "公交站": "交通设施", "公交": "交通设施",
                "地铁站": "交通设施", "地铁": "交通设施", "客运站": "交通设施", "火车站": "交通设施",
                "convenience": "便利店", "shop": "便利店", "store": "便利店",
                "school": "学校", "education": "学校",
                "parking": "停车场", "park": "停车场",
                "hotel": "酒店", "lodging": "酒店",
                "bank": "银行", "hospital": "医院", "medical": "医院",
                "restaurant": "餐饮", "food": "餐饮", "catering": "餐饮",
                "transport": "交通设施", "transit": "交通设施", "bus": "交通设施", "traffic": "交通设施",
            },
            ensure_ascii=False,
        )
        return f"""import json
import os
import math
import geopandas as gpd
import pandas as pd

try:
    from shapely.ops import unary_union, linemerge
    from shapely.affinity import translate
    from shapely.geometry import Point
except Exception:
    # Keep point-only workflows runnable even when optional shapely ops are unavailable.
    unary_union = None
    linemerge = None
    translate = None
    Point = None

plan = json.loads({plan_json!r})
input_file = r\"{input_file}\"
output_geojson = r\"{output_geojson}\"
output_png = r\"{output_png}\"
output_layer_name = {output_layer_name!r}
include_geojson = {bool(include_geojson)}
include_png = {bool(include_png)}

POI_SYMBOL_RULES = json.loads({poi_symbol_rules_json!r})
POI_CATEGORY_ALIASES = json.loads({poi_aliases_json!r})
POI_CATEGORY_FIELDS = [
    "query_category", "poi_category", "category", "type", "类别", "类型", "class", "kind", "fclass"
]

def _canonical_poi_category(value):
    text = str(value or "").strip()
    if not text or text.lower() == "nan":
        return ""
    if text in POI_CATEGORY_ALIASES:
        return POI_CATEGORY_ALIASES[text]
    low = text.lower()
    if low in POI_CATEGORY_ALIASES:
        return POI_CATEGORY_ALIASES[low]
    for key, category in POI_CATEGORY_ALIASES.items():
        if key and key in text:
            return category
    return text

def _detect_poi_category_field(df):
    for field in POI_CATEGORY_FIELDS:
        if field in df.columns and int(df[field].nunique(dropna=True)) >= 1:
            return field
    return ""

def _apply_poi_symbols(df):
    if df is None or len(df) == 0:
        return df, ""
    field = _detect_poi_category_field(df)
    if not field:
        if "point_symbol" not in df.columns:
            df["point_symbol"] = "circle"
        return df, ""
    df["poi_category"] = df[field].map(_canonical_poi_category)
    df["point_symbol"] = df["poi_category"].map(lambda c: POI_SYMBOL_RULES.get(str(c), {{}}).get("symbol", "circle"))
    df["point_symbol_color"] = df["poi_category"].map(lambda c: POI_SYMBOL_RULES.get(str(c), {{}}).get("color", "#5f6b7a"))
    df["point_symbol_label"] = df["poi_category"].map(lambda c: POI_SYMBOL_RULES.get(str(c), {{}}).get("label", ""))
    return df, field

gdf = gpd.read_file(input_file)
if gdf.empty:
    raise ValueError("Input layer is empty")
if gdf.crs is None:
    raise ValueError("Input layer CRS is undefined")

original_crs = gdf.crs
work = gdf.copy()
if getattr(work.crs, "is_geographic", False):
    # Auto-pick a suitable projected CRS for metric operators (prefer local UTM).
    target_metric_crs = None
    try:
        target_metric_crs = work.estimate_utm_crs()
    except Exception:
        target_metric_crs = None

    if target_metric_crs is None:
        try:
            ref = work.to_crs("EPSG:4326") if str(work.crs) != "EPSG:4326" else work
            c = ref.geometry.unary_union.centroid
            lon = float(c.x)
            lat = float(c.y)
            zone = int((lon + 180.0) // 6.0) + 1
            epsg = 32600 + zone if lat >= 0 else 32700 + zone
            target_metric_crs = f"EPSG:{{epsg}}"
        except Exception:
            target_metric_crs = "EPSG:3857"

    work = work.to_crs(target_metric_crs)
    print("crs_auto_projected:", str(target_metric_crs))
elif not getattr(work.crs, "is_projected", False):
    # Last-resort fallback for uncommon CRS objects.
    work = work.to_crs("EPSG:3857")
    print("crs_auto_projected:", "EPSG:3857")

input_metric_base = work.copy()
geom_type = str(plan.get("geometry_type", "line")).lower()
poi_symbol_field = ""
if geom_type == "point":
    work, poi_symbol_field = _apply_poi_symbols(work)
    input_metric_base = work.copy()
for op in plan.get("operators", []):
    name = op.get("name")
    p = op.get("params", {{}})
    if name == "selection":
        if geom_type == "polygon":
            min_area = float(p.get("min_area_m2", 0))
            work = work[work.geometry.area >= min_area].copy()
        elif geom_type == "line":
            min_seg = float(p.get("min_segment_m", 0))
            work = work[work.geometry.length >= min_seg].copy()
        elif geom_type == "point":
            keep_ratio = float(p.get("keep_ratio", 1.0))
            keep_ratio = max(0.05, min(1.0, keep_ratio))
            n = max(10, int(len(work) * keep_ratio))
            # Generic point selection: semantic importance + structure + spatial balance.
            if n < len(work):
                cand = work.reset_index(drop=True).copy()
                cand["_score"] = 0.0
                structure_hint = str(p.get("structure_type", plan.get("point_strategy", {{}}).get("structure_type", "mixed"))).lower()
                semantic_hint = str(p.get("semantic_type", plan.get("point_strategy", {{}}).get("semantic_type", "generic_point"))).lower()

                # 1) Field-semantic importance from numeric/hierarchy-like attributes.
                importance_keys = [
                    "importance", "priority", "weight", "score", "level", "grade", "rank",
                    "flow", "volume", "pop", "capacity", "service", "centrality",
                ]
                numeric_cols = []
                for col in cand.columns:
                    if col == "geometry":
                        continue
                    low = str(col).lower()
                    series_num = None
                    try:
                        series_num = pd.to_numeric(cand[col], errors="coerce")
                    except Exception:
                        series_num = None
                    if series_num is not None and series_num.notna().sum() > 0:
                        numeric_cols.append(col)
                        s = series_num.fillna(series_num.median() if series_num.notna().sum() > 0 else 0.0)
                        vmin = float(s.min())
                        vmax = float(s.max())
                        if vmax > vmin:
                            norm = (s - vmin) / (vmax - vmin)
                        else:
                            norm = s * 0.0
                        w = 1.0
                        if any(k in low for k in importance_keys):
                            w = 2.0
                        cand["_score"] += norm * w

                # 2) Category diversity bonus to avoid single-type collapse.
                category_keys = ["type", "class", "kind", "category", "subtype", "tag", "group"]
                for col in cand.columns:
                    if col == "geometry":
                        continue
                    low = str(col).lower()
                    if not any(k in low for k in category_keys):
                        continue
                    vals = cand[col].astype(str)
                    freq = vals.value_counts(dropna=False)
                    inv = vals.map(lambda x: 1.0 / float(freq.get(x, 1)))
                    vmax = float(inv.max()) if len(inv) > 0 else 0.0
                    if vmax > 0:
                        cand["_score"] += inv / vmax

                # 3) Group-structure preservation (works for line/route/type clusters).
                group_keys = ["line", "route", "corridor", "group", "category", "class", "type", "system"]
                group_col = ""
                for col in cand.columns:
                    if col == "geometry":
                        continue
                    low = str(col).lower()
                    if any(k in low for k in group_keys):
                        group_col = col
                        break

                keep_idx = set()
                protected_idx = set()
                # 3.0) Contour-preservation anchors: keep boundary control points in 8 directions.
                try:
                    pts = cand.geometry
                    if len(pts) > 0:
                        xs = pts.x
                        ys = pts.y
                        for idx in [
                            int(xs.idxmin()), int(xs.idxmax()),
                            int(ys.idxmin()), int(ys.idxmax()),
                            int((xs + ys).idxmin()), int((xs + ys).idxmax()),
                            int((xs - ys).idxmin()), int((xs - ys).idxmax()),
                        ]:
                            keep_idx.add(idx)
                            protected_idx.add(idx)
                except Exception:
                    pass

                if group_col:
                    grouped = list(cand.groupby(group_col))
                    gnum = max(1, len(grouped))
                    base_k = max(1, int(n / gnum * 0.5))
                    for _, grp in grouped:
                        g = grp.copy()
                        if len(g) <= base_k + 2:
                            for i in g.index.tolist():
                                keep_idx.add(int(i))
                            continue
                        # Keep top semantic points in each group.
                        top = g.sort_values("_score", ascending=False).head(base_k)
                        for i in top.index.tolist():
                            keep_idx.add(int(i))
                        # Keep farthest pair as structural anchors.
                        pts = [(int(i), (float(geom.x), float(geom.y))) for i, geom in zip(g.index.tolist(), g.geometry.tolist())]
                        p0_idx, p0 = pts[0]
                        far_idx = p0_idx
                        far_d = -1.0
                        for j_idx, j_pt in pts:
                            d = (j_pt[0] - p0[0]) ** 2 + (j_pt[1] - p0[1]) ** 2
                            if d > far_d:
                                far_d = d
                                far_idx = j_idx
                        keep_idx.add(int(far_idx))
                        p1 = dict(pts)[far_idx]
                        far2_idx = far_idx
                        far2_d = -1.0
                        for j_idx, j_pt in pts:
                            d = (j_pt[0] - p1[0]) ** 2 + (j_pt[1] - p1[1]) ** 2
                            if d > far2_d:
                                far2_d = d
                                far2_idx = j_idx
                        keep_idx.add(int(far2_idx))

                # 3.5) Region-coverage pass: keep one rep per occupied region first,
                # and give dense regions more than one rep so they do not collapse into a singleton.
                # The quota grid is aligned to the evaluation metric grid so optimization is
                # directly targeted at reducing dense_region_singleton_ratio.
                try:
                    if len(cand) > 1:
                        bb = cand.total_bounds
                        minx, miny, maxx, maxy = [float(x) for x in bb]
                        width = max(1e-9, maxx - minx)
                        height = max(1e-9, maxy - miny)
                        scale_den = float(plan.get("target_scale", 0) or 0)
                        region_bins = max(3, min(10, int(round((max(6, n) ** 0.5) * 1.15))))
                        if scale_den >= 300000:
                            region_bins = max(region_bins, 5)
                        elif scale_den >= 200000:
                            region_bins = max(region_bins, 4)
                        cand["_rx"] = ((cand.geometry.x - minx) / width * region_bins).astype(int).clip(lower=0, upper=region_bins)
                        cand["_ry"] = ((cand.geometry.y - miny) / height * region_bins).astype(int).clip(lower=0, upper=region_bins)
                        cand["_rkey"] = cand["_rx"].astype(str) + "_" + cand["_ry"].astype(str)
                        region_freq = cand["_rkey"].value_counts(dropna=False)
                        dense_threshold_eval = max(8, int(round(float(region_freq.quantile(0.75))))) if len(region_freq) > 0 else 8
                        region_groups = []
                        dense_region_key_set = set()
                        for key, grp in cand.groupby("_rkey"):
                            cnt = int(len(grp))
                            top_score = float(grp["_score"].max()) if len(grp) > 0 else 0.0
                            desired_quota = 1
                            if cnt >= 4:
                                desired_quota = 2
                            if cnt >= 8:
                                desired_quota = 3
                            if cnt >= 14:
                                desired_quota = 4
                            if cnt >= 20:
                                desired_quota = 5
                            if cnt >= 28:
                                desired_quota = max(desired_quota, int(round(cnt * 0.28)))
                            if scale_den >= 300000 and cnt >= 8:
                                desired_quota = max(desired_quota, 3)
                            elif scale_den >= 200000 and cnt >= 6:
                                desired_quota = max(desired_quota, 3)
                            if cnt >= dense_threshold_eval:
                                desired_quota = max(desired_quota, 2)
                            if cnt >= max(dense_threshold_eval + 4, 12):
                                desired_quota = max(desired_quota, 3)
                            if cnt >= dense_threshold_eval:
                                dense_region_key_set.add(str(key))
                            region_groups.append(
                                {{
                                    "cnt": cnt,
                                    "top_score": top_score,
                                    "key": str(key),
                                    "grp": grp.copy(),
                                    "desired_quota": desired_quota,
                                }}
                            )

                        region_count = int(len(region_groups))
                        sparse_region_count = int(sum(1 for item in region_groups if int(item["cnt"]) <= 2))
                        dense_region_count = int(sum(1 for item in region_groups if int(item["cnt"]) >= dense_threshold_eval))
                        target_region_floor = int(math.ceil(region_count * 0.72)) if region_count > 0 else 0
                        target_sparse_floor = int(math.ceil(sparse_region_count * 0.85)) if sparse_region_count > 0 else 0
                        dense_min_keep_floor = int(
                            sum(
                                min(int(item["cnt"]), max(1, min(int(item["desired_quota"]), 3)))
                                for item in region_groups
                                if int(item["cnt"]) >= dense_threshold_eval
                            )
                        )
                        dense_bonus = int(max(0, math.ceil(dense_region_count * 1.00)))
                        n = min(
                            len(cand),
                            max(
                                int(n),
                                int(len(keep_idx)),
                                target_region_floor,
                                target_sparse_floor + dense_bonus,
                                dense_min_keep_floor,
                            ),
                        )

                        # Pass A: every occupied region gets one representative if possible.
                        region_groups.sort(
                            key=lambda item: (
                                0 if int(item["cnt"]) <= 2 else 1,
                                0 if int(item["cnt"]) <= 4 else 1,
                                -int(item["cnt"]),
                                -float(item["top_score"]),
                            )
                        )
                        for item in region_groups:
                            if len(keep_idx) >= n:
                                break
                            grp = item["grp"]
                            picks = grp.sort_values("_score", ascending=False).index.tolist()
                            if not picks:
                                continue
                            pick0 = int(picks[0])
                            keep_idx.add(pick0)
                            protected_idx.add(pick0)

                        # Pass B: dense regions get extra representatives using a finer sub-grid,
                        # so a 6-12 point local region does not collapse to a singleton.
                        extra_candidates = []
                        region_quota_map = {{str(item["key"]): int(max(1, min(int(item["desired_quota"]), int(item["cnt"])))) for item in region_groups}}
                        for item in region_groups:
                            grp = item["grp"].copy()
                            cnt = int(item["cnt"])
                            quota = int(region_quota_map.get(str(item["key"]), 1))
                            current_kept = [int(i) for i in grp.index.tolist() if int(i) in keep_idx]
                            extra_needed = max(0, quota - len(current_kept))
                            if extra_needed <= 0:
                                continue

                            sub_bins = 2
                            if cnt >= 14:
                                sub_bins = 3
                            if cnt >= 28:
                                sub_bins = 4
                            gminx, gminy, gmaxx, gmaxy = [float(v) for v in grp.total_bounds]
                            gwidth = max(1e-9, gmaxx - gminx)
                            gheight = max(1e-9, gmaxy - gminy)
                            grp["_sx"] = ((grp.geometry.x - gminx) / gwidth * sub_bins).astype(int).clip(lower=0, upper=sub_bins)
                            grp["_sy"] = ((grp.geometry.y - gminy) / gheight * sub_bins).astype(int).clip(lower=0, upper=sub_bins)
                            grp["_skey"] = grp["_sx"].astype(str) + "_" + grp["_sy"].astype(str)

                            # One candidate per occupied sub-cell first.
                            sub_first = grp.sort_values("_score", ascending=False).drop_duplicates(subset=["_skey"])
                            for idx in sub_first.index.tolist():
                                idx_i = int(idx)
                                if idx_i in keep_idx:
                                    continue
                                priority = float(grp.loc[idx, "_score"]) + float(cnt) * 0.03
                                extra_candidates.append((priority, str(item["key"]), idx_i))

                            # Fallback: add far/high-score points from the same region.
                            rest = grp.drop(index=[i for i in grp.index.tolist() if int(i) in keep_idx], errors="ignore")
                            rest = rest.sort_values("_score", ascending=False)
                            for idx in rest.index.tolist():
                                idx_i = int(idx)
                                if idx_i in keep_idx:
                                    continue
                                priority = float(rest.loc[idx, "_score"]) + float(cnt) * 0.015
                                extra_candidates.append((priority, str(item["key"]), idx_i))

                        seen_extra = set()
                        extra_candidates = sorted(extra_candidates, key=lambda item: item[0], reverse=True)
                        for _, region_key, idx in extra_candidates:
                            if len(keep_idx) >= n:
                                break
                            if idx in seen_extra or idx in keep_idx:
                                continue
                            region_grp = cand[cand["_rkey"] == region_key]
                            current_region_kept = sum(1 for i in region_grp.index.tolist() if int(i) in keep_idx)
                            region_quota = int(region_quota_map.get(region_key, 1))
                            if current_region_kept >= region_quota:
                                continue
                            keep_idx.add(int(idx))
                            if current_region_kept < min(region_quota, 3):
                                protected_idx.add(int(idx))
                            seen_extra.add(int(idx))
                except Exception:
                    pass

                # 3.6) Evaluation-aligned dense-region anti-singleton pass:
                # dense occupied cells on the same grid used by the metric should keep at least two reps.
                try:
                    if len(cand) > 6:
                        eval_dense_keys = [str(item["key"]) for item in region_groups if int(item["cnt"]) >= dense_threshold_eval]
                        for key in eval_dense_keys:
                            if len(keep_idx) >= n:
                                break
                            grp = cand[cand["_rkey"] == key].copy()
                            if len(grp) < 2:
                                continue
                            kept_here = [int(i) for i in grp.index.tolist() if int(i) in keep_idx]
                            needed = max(0, 2 - len(kept_here))
                            if needed <= 0:
                                continue
                            picks = grp.sort_values("_score", ascending=False).index.tolist()
                            for idx in picks:
                                idx_i = int(idx)
                                if idx_i in keep_idx:
                                    continue
                                keep_idx.add(idx_i)
                                protected_idx.add(idx_i)
                                needed -= 1
                                if needed <= 0 or len(keep_idx) >= n:
                                    break
                except Exception:
                    pass

                # 3.65) Fine dense-region pass:
                # inside already-dense metric regions, use a finer sub-grid to avoid over-collapsing local sub-clusters.
                try:
                    if len(cand) > 6:
                        bb = cand.total_bounds
                        minx, miny, maxx, maxy = [float(x) for x in bb]
                        width = max(1e-9, maxx - minx)
                        height = max(1e-9, maxy - miny)
                        fine_bins = max(5, min(12, int(round((len(cand) ** 0.5) * 1.15))))
                        cand["_frx"] = ((cand.geometry.x - minx) / width * fine_bins).astype(int).clip(lower=0, upper=fine_bins)
                        cand["_fry"] = ((cand.geometry.y - miny) / height * fine_bins).astype(int).clip(lower=0, upper=fine_bins)
                        cand["_frkey"] = cand["_frx"].astype(str) + "_" + cand["_fry"].astype(str)
                        fine_freq = cand["_frkey"].value_counts(dropna=False)
                        fine_dense_threshold = max(4, int(round(float(fine_freq.quantile(0.75))))) if len(fine_freq) > 0 else 4
                        fine_dense_keys = [str(k) for k, v in fine_freq.items() if int(v) >= fine_dense_threshold]
                        for key in fine_dense_keys:
                            if len(keep_idx) >= n:
                                break
                            grp = cand[cand["_frkey"] == key].copy()
                            if len(grp) < 2:
                                continue
                            kept_here = [int(i) for i in grp.index.tolist() if int(i) in keep_idx]
                            needed = max(0, 2 - len(kept_here))
                            if needed <= 0:
                                continue
                            picks = grp.sort_values("_score", ascending=False).index.tolist()
                            for idx in picks:
                                idx_i = int(idx)
                                if idx_i in keep_idx:
                                    continue
                                keep_idx.add(idx_i)
                                protected_idx.add(idx_i)
                                needed -= 1
                                if needed <= 0 or len(keep_idx) >= n:
                                    break
                except Exception:
                    pass

                # 4) Dense-hotspot representative pass (prevents wiping dense clusters).
                try:
                    if len(cand) > 8:
                        bb = cand.total_bounds
                        minx, miny, maxx, maxy = [float(x) for x in bb]
                        width = max(1e-9, maxx - minx)
                        height = max(1e-9, maxy - miny)
                        hot_bins = max(6, min(20, int(round((len(cand) ** 0.5) * 0.8))))
                        cand["_hx"] = ((cand.geometry.x - minx) / width * hot_bins).astype(int).clip(lower=0, upper=hot_bins)
                        cand["_hy"] = ((cand.geometry.y - miny) / height * hot_bins).astype(int).clip(lower=0, upper=hot_bins)
                        cand["_hkey"] = cand["_hx"].astype(str) + "_" + cand["_hy"].astype(str)
                        hfreq = cand["_hkey"].value_counts(dropna=False)
                        if len(hfreq) > 0:
                            q90 = float(hfreq.quantile(0.90))
                            min_hot = max(3.0, q90)
                            hotspot_cells_all = [str(k) for k, v in hfreq.items() if float(v) >= min_hot]
                            hotspot_cells = hotspot_cells_all[: max(6, min(18, len(hotspot_cells_all)))]
                            if structure_hint == "clustered" or semantic_hint == "commercial_poi_points":
                                hotspot_quota = max(4, int(n * 0.28))
                            else:
                                hotspot_quota = max(3, int(n * 0.18))
                            hotspot_quota = max(0, min(n - len(keep_idx), hotspot_quota))
                            # Keep one high-score point per hotspot cell first.
                            for key in hotspot_cells:
                                if len(keep_idx) >= n or hotspot_quota <= 0:
                                    break
                                cell_grp = cand[cand["_hkey"] == key].copy()
                                if len(cell_grp) == 0:
                                    continue
                                pick_idx = int(cell_grp.sort_values("_score", ascending=False).index[0])
                                keep_idx.add(pick_idx)
                                hotspot_quota -= 1

                            # For largest hotspot, preserve local contour anchors (x/y extremes).
                            if hotspot_cells:
                                biggest_key = str(hfreq.index[0])
                                big_grp = cand[cand["_hkey"] == biggest_key].copy()
                                if len(big_grp) > 4:
                                    xs = big_grp.geometry.x
                                    ys = big_grp.geometry.y
                                    local_anchor_idx = {{
                                        int(xs.idxmin()), int(xs.idxmax()),
                                        int(ys.idxmin()), int(ys.idxmax()),
                                    }}
                                    for i in local_anchor_idx:
                                        if len(keep_idx) >= n:
                                            break
                                        keep_idx.add(int(i))
                except Exception:
                    pass

                # 5) Longitudinal continuity pass (prevents "middle collapse" in corridor-like distributions).
                try:
                    if len(cand) > 4:
                        mx = float(cand.geometry.x.mean())
                        my = float(cand.geometry.y.mean())
                        dx = cand.geometry.x - mx
                        dy = cand.geometry.y - my
                        sxx = float((dx * dx).mean())
                        syy = float((dy * dy).mean())
                        sxy = float((dx * dy).mean())
                        theta = 0.5 * math.atan2(2.0 * sxy, (sxx - syy))
                        ux = math.cos(theta)
                        uy = math.sin(theta)
                        cand["_axis_t"] = dx * ux + dy * uy
                        tmin = float(cand["_axis_t"].min())
                        tmax = float(cand["_axis_t"].max())
                        if tmax > tmin:
                            base_bins = max(8, min(28, int(round((max(6, n) ** 0.5) * 3.0))))
                            if structure_hint == "corridor":
                                axis_bins = min(36, max(10, int(base_bins * 1.25)))
                            elif structure_hint == "clustered":
                                axis_bins = max(6, int(base_bins * 0.75))
                            else:
                                axis_bins = base_bins
                            cand["_axis_bin"] = ((cand["_axis_t"] - tmin) / (tmax - tmin + 1e-9) * axis_bins).astype(int)
                            cand["_axis_bin"] = cand["_axis_bin"].clip(lower=0, upper=axis_bins)
                            occupied_bins = sorted([int(x) for x in cand["_axis_bin"].dropna().unique().tolist()])

                            # Continuity quota: transport/corridor keeps more middle sections.
                            if structure_hint == "corridor" or semantic_hint == "transport_network_points":
                                axis_quota = max(6, int(n * 0.55))
                            elif structure_hint == "uniform":
                                axis_quota = max(5, int(n * 0.45))
                            else:
                                axis_quota = max(4, int(n * 0.35))
                            axis_quota = max(0, min(n - len(keep_idx), axis_quota, len(occupied_bins)))

                            if axis_quota > 0 and occupied_bins:
                                chosen_pos = [int(round(i * (len(occupied_bins) - 1) / float(max(1, axis_quota - 1)))) for i in range(axis_quota)]
                                chosen_bins = sorted(set(occupied_bins[pos] for pos in chosen_pos))
                                for b0 in chosen_bins:
                                    if len(keep_idx) >= n:
                                        break
                                    grp = cand[cand["_axis_bin"] == b0].copy()
                                    if len(grp) == 0:
                                        continue
                                    bt = float(grp["_axis_t"].mean())
                                    span = max(1e-9, tmax - tmin)
                                    grp["_axis_rank"] = grp["_score"] - (grp["_axis_t"] - bt).abs() / span
                                    pick_idx = int(grp.sort_values("_axis_rank", ascending=False).index[0])
                                    keep_idx.add(pick_idx)
                except Exception:
                    pass

                # 6) Spatial-balance pass (one best point per grid).
                bb = cand.total_bounds
                minx, miny, maxx, maxy = [float(x) for x in bb]
                min_sep = float(plan.get("derived_constraints", {{}}).get("min_separation_m", 200.0))
                width = max(1e-9, maxx - minx)
                height = max(1e-9, maxy - miny)
                target_cells = max(4, int((n ** 0.5) * 2))
                cell = max(min_sep, min(width, height) / float(target_cells))
                cand["_gx"] = ((cand.geometry.x - minx) / cell).astype(int)
                cand["_gy"] = ((cand.geometry.y - miny) / cell).astype(int)

                first = cand.sort_values(["_gx", "_gy", "_score"], ascending=[True, True, False]).drop_duplicates(subset=["_gx", "_gy"])
                for i in first.index.tolist():
                    if len(keep_idx) >= n:
                        break
                    keep_idx.add(int(i))

                # 7) Fill remaining slots by global score.
                if len(keep_idx) < n:
                    rest = cand.drop(index=list(keep_idx)).sort_values("_score", ascending=False)
                    for i in rest.index.tolist():
                        keep_idx.add(int(i))
                        if len(keep_idx) >= n:
                            break

                cand = cand.loc[sorted(list(keep_idx))].copy()
                cand["__sel_score"] = cand["_score"]
                cand["__protected_keep"] = cand.index.map(lambda i: 1 if int(i) in protected_idx else 0)
                if "_rkey" in cand.columns:
                    cand["__metric_rkey"] = cand["_rkey"].astype(str)
                    cand["__metric_dense_region"] = cand["__metric_rkey"].map(
                        lambda k: 1 if str(k) in dense_region_key_set else 0
                    )
                    cand["__metric_region_quota"] = cand["__metric_rkey"].map(
                        lambda k: int(max(1, min(int(region_quota_map.get(str(k), 1)), 3)))
                    )
                    cand["__metric_region_bins"] = int(region_bins) if "region_bins" in locals() else 0
                    cand["__metric_minx"] = float(minx) if "minx" in locals() else 0.0
                    cand["__metric_miny"] = float(miny) if "miny" in locals() else 0.0
                    cand["__metric_width"] = float(width) if "width" in locals() else 0.0
                    cand["__metric_height"] = float(height) if "height" in locals() else 0.0
                    cand["__orig_x"] = cand.geometry.x
                    cand["__orig_y"] = cand.geometry.y
                work = cand.drop(
                    columns=[
                        c
                        for c in [
                            "_score", "_gx", "_gy", "_axis_t", "_axis_bin", "_axis_rank",
                            "_hx", "_hy", "_hkey", "_rx", "_ry", "_rkey", "_frx", "_fry", "_frkey",
                        ]
                        if c in cand.columns
                    ]
                ).copy()
            else:
                work = work.copy()
    elif name == "simplify":
        tol = float(p.get("tolerance_m", 0))
        preserve = bool(p.get("preserve_topology", True))
        work["geometry"] = work.geometry.simplify(tol, preserve_topology=preserve)
    elif name == "aggregate" and geom_type == "polygon":
        dist = float(p.get("distance_m", 0))
        if dist > 0:
            if unary_union is None:
                raise ImportError("shapely.ops.unary_union is unavailable for aggregate operator")
            merged = unary_union(work.geometry.buffer(dist / 2.0))
            if getattr(merged, "geom_type", "") == "Polygon":
                geoms = [merged]
            else:
                geoms = list(getattr(merged, "geoms", []))
            work = gpd.GeoDataFrame(geometry=geoms, crs=work.crs)
            work["geometry"] = work.geometry.buffer(-dist / 2.0)
    elif name == "merge" and geom_type == "line":
        if unary_union is None or linemerge is None:
            raise ImportError("shapely.ops merge functions are unavailable for merge operator")
        unioned = unary_union(work.geometry)
        merged = linemerge(unioned)
        geoms = list(merged.geoms) if hasattr(merged, "geoms") else [merged]
        work = gpd.GeoDataFrame(geometry=geoms, crs=work.crs)
    elif name == "smooth" and geom_type in ("line", "polygon"):
        scale_hint = float(plan.get("derived_constraints", {{}}).get("min_length_m", 1.0))
        work["geometry"] = work.geometry.simplify(max(0.1, scale_hint * 0.1), preserve_topology=True)
    elif name == "displacement":
        min_sep = float(p.get("min_sep_m", plan.get("derived_constraints", {{}}).get("min_separation_m", 0.0)))
        max_shift = float(p.get("max_shift_m", 0.0))
        strict_no_overlap = bool(p.get("strict_no_overlap", False))
        strict_pixel_gap = float(p.get("strict_pixel_gap", 0.0) or 0.0)
        strict_density_factor = float(p.get("strict_density_factor", 1.0) or 1.0)
        if geom_type == "point" and min_sep > 0 and len(work) > 1:
            work = work.reset_index(drop=True).copy()
            protected_flags = None
            if "__protected_keep" in work.columns:
                try:
                    protected_flags = [bool(int(v) > 0) for v in work["__protected_keep"].fillna(0).tolist()]
                except Exception:
                    protected_flags = None
            if translate is not None:
                rep = work.geometry.reset_index(drop=True).copy()
                rounds = 6
                for _ in range(rounds):
                    moved = False
                    sidx = rep.sindex
                    for i, g in enumerate(rep):
                        if g is None or g.is_empty:
                            continue
                        cand = list(sidx.intersection(g.buffer(min_sep).bounds))
                        for j in cand:
                            if j <= i:
                                continue
                            gj = rep.iloc[j]
                            d = g.distance(gj)
                            if d < min_sep:
                                # d==0（重合）时原先被 0<d 漏掉，导致综合后符号仍叠在一起
                                need = float(min_sep - d) if d > 1e-12 else float(min_sep)
                                shift_limit = max_shift if max_shift > 0 else min_sep
                                is_pi = bool(protected_flags[i]) if protected_flags is not None and i < len(protected_flags) else False
                                is_pj = bool(protected_flags[j]) if protected_flags is not None and j < len(protected_flags) else False
                                dx = float(g.x - gj.x)
                                dy = float(g.y - gj.y)
                                norm = (dx * dx + dy * dy) ** 0.5
                                if norm < 1e-9:
                                    ang = ((i * 137 + j * 211) % 6283) / 1000.0
                                    ux, uy = math.cos(ang), math.sin(ang)
                                else:
                                    ux, uy = dx / norm, dy / norm
                                if is_pi and not is_pj:
                                    shift_j = min(shift_limit, need)
                                    rep.iat[j] = translate(gj, xoff=-ux * shift_j, yoff=-uy * shift_j)
                                    moved = bool(shift_j > 0)
                                elif is_pj and not is_pi:
                                    shift_i = min(shift_limit, need)
                                    rep.iat[i] = translate(g, xoff=ux * shift_i, yoff=uy * shift_i)
                                    moved = bool(shift_i > 0)
                                elif is_pi and is_pj:
                                    # Dense-region representatives act like anchors: allow closer spacing
                                    # and avoid pushing both points out of their source cell.
                                    anchor_need = max(0.0, need - min_sep * 0.12)
                                    shift = min(shift_limit, anchor_need / 2.0)
                                    if shift > 0:
                                        rep.iat[i] = translate(g, xoff=ux * shift, yoff=uy * shift)
                                        rep.iat[j] = translate(gj, xoff=-ux * shift, yoff=-uy * shift)
                                        moved = True
                                else:
                                    shift = min(shift_limit, need / 2.0)
                                    rep.iat[i] = translate(g, xoff=ux * shift, yoff=uy * shift)
                                    rep.iat[j] = translate(gj, xoff=-ux * shift, yoff=-uy * shift)
                                    moved = True
                    if not moved:
                        break
                work["geometry"] = rep
                if Point is not None and "__metric_dense_region" in work.columns and "__metric_rkey" in work.columns:
                    try:
                        for i in range(len(work)):
                            if int(work.loc[i, "__metric_dense_region"]) <= 0:
                                continue
                            region_bins = int(work.loc[i, "__metric_region_bins"]) if "__metric_region_bins" in work.columns else 0
                            if region_bins <= 0:
                                continue
                            minx0 = float(work.loc[i, "__metric_minx"])
                            miny0 = float(work.loc[i, "__metric_miny"])
                            width0 = max(1e-9, float(work.loc[i, "__metric_width"]))
                            height0 = max(1e-9, float(work.loc[i, "__metric_height"]))
                            key = str(work.loc[i, "__metric_rkey"])
                            try:
                                sx, sy = [int(v) for v in key.split("_", 1)]
                            except Exception:
                                continue
                            x0 = minx0 + width0 * float(sx) / float(region_bins)
                            x1 = minx0 + width0 * float(sx + 1) / float(region_bins)
                            y0 = miny0 + height0 * float(sy) / float(region_bins)
                            y1 = miny0 + height0 * float(sy + 1) / float(region_bins)
                            g = work.geometry.iloc[i]
                            if g is None or g.is_empty:
                                continue
                            gx = float(g.x)
                            gy = float(g.y)
                            if (gx < x0) or (gx > x1) or (gy < y0) or (gy > y1):
                                work.iat[i, work.columns.get_loc("geometry")] = Point(
                                    float(work.loc[i, "__orig_x"]),
                                    float(work.loc[i, "__orig_y"]),
                                )
                    except Exception:
                        pass

            if strict_no_overlap and len(work) > 1:
                pool = work.copy()
                if "__sel_score" not in pool.columns:
                    pool["__sel_score"] = 0.0
                if "__protected_keep" not in pool.columns:
                    pool["__protected_keep"] = 0
                if "__metric_region_quota" not in pool.columns:
                    pool["__metric_region_quota"] = 1
                if "__metric_rkey" not in pool.columns:
                    pool["__metric_rkey"] = ""
                # Pixel-space hotspot thinning: keep around 1/3 in each dense cell by default.
                b = pool.total_bounds
                minx, miny, maxx, maxy = [float(v) for v in b]
                width = max(1e-9, maxx - minx)
                height = max(1e-9, maxy - miny)
                render_w = 2200.0
                render_h = 1760.0
                pool["__px"] = (pool.geometry.x - minx) / width * render_w
                pool["__py"] = (pool.geometry.y - miny) / height * render_h
                grid_px = max(20.0, strict_pixel_gap * 2.0 if strict_pixel_gap > 0 else 24.0)
                pool["__cgx"] = (pool["__px"] / grid_px).astype(int)
                pool["__cgy"] = (pool["__py"] / grid_px).astype(int)
                pool["__ckey"] = pool["__cgx"].astype(str) + "_" + pool["__cgy"].astype(str)
                cell_freq = pool["__ckey"].value_counts(dropna=False)
                cell_quota = {{}}
                for key, cnt in cell_freq.items():
                    cnt_i = int(cnt)
                    q = cnt_i if strict_density_factor >= 0.999 else max(1, int(round(cnt_i * strict_density_factor)))
                    cell_quota[str(key)] = q

                order = pool.sort_values(
                    ["__metric_region_quota", "__protected_keep", "__sel_score"],
                    ascending=[False, False, False],
                ).index.tolist()
                kept = []
                kept_geoms = []
                kept_px = []
                kept_protected = []
                used_cell = {{}}
                used_region = {{}}
                for idx in order:
                    g = pool.geometry.loc[idx]
                    if g is None or g.is_empty:
                        continue
                    ckey = str(pool.loc[idx, "__ckey"])
                    is_protected = int(pool.loc[idx, "__protected_keep"]) > 0
                    region_key = str(pool.loc[idx, "__metric_rkey"])
                    region_quota = max(1, int(pool.loc[idx, "__metric_region_quota"]) if "__metric_region_quota" in pool.columns else 1)
                    region_used = int(used_region.get(region_key, 0))
                    region_need = region_key and (region_used < region_quota)
                    used = int(used_cell.get(ckey, 0))
                    quota_limit = int(cell_quota.get(ckey, 1)) + (2 if is_protected else 0)
                    if region_need:
                        quota_limit = max(quota_limit, region_quota)
                    if used >= quota_limit:
                        continue
                    ok = True
                    for h, h_protected in zip(kept_geoms, kept_protected):
                        # 地面 min_sep 在小范围内对应像素很少；系数须接近 1 才与屏上符号尺寸一致
                        sep_factor = 0.92 if region_need else (0.96 if (is_protected or h_protected) else 1.0)
                        if g.distance(h) < (min_sep * sep_factor):
                            ok = False
                            break
                    if ok and strict_pixel_gap > 0:
                        px = float(pool.loc[idx, "__px"])
                        py = float(pool.loc[idx, "__py"])
                        px_gap = strict_pixel_gap * (0.92 if region_need else (0.96 if is_protected else 1.0))
                        for (qx, qy), h_protected in zip(kept_px, kept_protected):
                            gap_factor = 0.92 if region_need else (0.96 if (is_protected or h_protected) else 1.0)
                            if ((px - qx) ** 2 + (py - qy) ** 2) ** 0.5 < (px_gap * gap_factor):
                                ok = False
                                break
                    if ok:
                        kept.append(idx)
                        kept_geoms.append(g)
                        kept_px.append((float(pool.loc[idx, "__px"]), float(pool.loc[idx, "__py"])))
                        kept_protected.append(bool(is_protected))
                        used_cell[ckey] = used + 1
                        if region_key:
                            used_region[region_key] = region_used + 1
                if kept:
                    work = pool.loc[kept].copy().reset_index(drop=True)
                work = work.drop(columns=[c for c in ["__px", "__py", "__cgx", "__cgy", "__ckey"] if c in work.columns])
    elif name == "typification":
        if geom_type == "point" and len(work) > 1:
            before_n = int(len(work))
            min_sep = float(plan.get("derived_constraints", {{}}).get("min_separation_m", 0.0))
            min_sep = max(0.5, min_sep)
            pattern = str(p.get("pattern", "spatial_even")).lower()
            b = work.total_bounds
            minx, miny, maxx, maxy = [float(x) for x in b]
            width = max(1e-6, maxx - minx)
            height = max(1e-6, maxy - miny)
            cell_factor = 1.0
            if pattern == "axis_balanced":
                cell_factor = 0.9
            elif pattern in ("cluster_representative", "multi_core_balanced"):
                cell_factor = 1.2
            cell = max(min_sep, min(width, height) / max(10.0, (len(work) ** 0.5)) * cell_factor)
            tmp = work.copy()
            tmp["_gx"] = ((tmp.geometry.x - minx) / cell).astype(int)
            tmp["_gy"] = ((tmp.geometry.y - miny) / cell).astype(int)
            if "_score" not in tmp.columns:
                tmp["_score"] = 0.0
            kept = tmp.sort_values(["_gx", "_gy", "_score"], ascending=[True, True, False]).drop_duplicates(subset=["_gx", "_gy"])
            if "__protected_keep" in tmp.columns:
                protected = tmp[tmp["__protected_keep"] > 0].copy()
                if len(protected) > 0:
                    kept = gpd.GeoDataFrame(pd.concat([kept, protected], axis=0), crs=tmp.crs)
                    kept = kept[~kept.index.duplicated(keep="first")]
            # Keep contour anchors after typification to avoid shape collapse.
            try:
                xs = tmp.geometry.x
                ys = tmp.geometry.y
                anchor_idx = {{
                    int(xs.idxmin()), int(xs.idxmax()),
                    int(ys.idxmin()), int(ys.idxmax()),
                    int((xs + ys).idxmin()), int((xs + ys).idxmax()),
                    int((xs - ys).idxmin()), int((xs - ys).idxmax()),
                }}
                missed = [i for i in anchor_idx if i not in kept.index]
                if missed:
                    kept = gpd.GeoDataFrame(pd.concat([kept, tmp.loc[missed]], axis=0), crs=tmp.crs)
            except Exception:
                pass
            min_keep_after = max(12, int(before_n * 0.85))
            if len(kept) < min_keep_after:
                extra = tmp.drop(index=kept.index).copy()
                for col in extra.columns:
                    if col == "geometry":
                        continue
                    low = str(col).lower()
                    try:
                        s = pd.to_numeric(extra[col], errors="coerce")
                    except Exception:
                        s = None
                    if s is not None and s.notna().sum() > 0:
                        ss = s.fillna(s.median() if s.notna().sum() > 0 else 0.0)
                        vmin = float(ss.min())
                        vmax = float(ss.max())
                        if vmax > vmin:
                            extra["_score"] += (ss - vmin) / (vmax - vmin)
                    if any(k in low for k in ["type", "class", "kind", "category", "group", "tag"]):
                        vals = extra[col].astype(str)
                        freq = vals.value_counts(dropna=False)
                        inv = vals.map(lambda x: 1.0 / float(freq.get(x, 1)))
                        vmax = float(inv.max()) if len(inv) > 0 else 0.0
                        if vmax > 0:
                            extra["_score"] += inv / vmax
                need = int(min_keep_after - len(kept))
                extra_top = extra.sort_values("_score", ascending=False).head(max(0, need))
                if len(extra_top) > 0:
                    kept = gpd.GeoDataFrame(
                        pd.concat([kept, extra_top], axis=0, ignore_index=False),
                        crs=tmp.crs,
                    )
            if "_score" in kept.columns:
                kept["__sel_score"] = kept["_score"]
            work = kept.drop(columns=[c for c in ["_gx", "_gy", "_score"] if c in kept.columns]).copy()

if geom_type == "point" and len(work) > 0 and len(input_metric_base) > 0:
    try:
        minx, miny, maxx, maxy = [float(v) for v in input_metric_base.total_bounds]
        width = max(1e-9, maxx - minx)
        height = max(1e-9, maxy - miny)
        scale_den = int(plan.get("target_scale", 0) or 0)
        region_bins = max(3, min(10, int(round((max(6, len(work)) ** 0.5) * 1.15))))
        if scale_den <= 220000:
            region_bins = max(region_bins, 6)

        def _region_series(_gdf):
            _rx = ((_gdf.geometry.x - minx) / width * region_bins).astype(int).clip(lower=0, upper=region_bins)
            _ry = ((_gdf.geometry.y - miny) / height * region_bins).astype(int).clip(lower=0, upper=region_bins)
            return pd.Series([f"{{int(x)}}_{{int(y)}}" for x, y in zip(_rx.tolist(), _ry.tolist())], index=_gdf.index)

        def _pick_region_rep_idx(_key, _source_gdf):
            _grp = _source_gdf.loc[in_regions[in_regions == _key].index].copy()
            if len(_grp) == 0:
                return None
            _sx, _sy = [int(v) for v in str(_key).split("_", 1)]
            _cx = minx + width * (float(_sx) + 0.5) / float(region_bins)
            _cy = miny + height * (float(_sy) + 0.5) / float(region_bins)
            _grp["__repair_rank"] = ((_grp.geometry.x - _cx) ** 2 + (_grp.geometry.y - _cy) ** 2) ** 0.5
            _grp = _grp.sort_values("__repair_rank", ascending=True)
            for _, _row in _grp.iterrows():
                _geom = _row.geometry
                if _geom is None or _geom.is_empty:
                    continue
                _xy = (round(float(_geom.x), 6), round(float(_geom.y), 6))
                if _xy in existing_xy:
                    continue
                existing_xy.add(_xy)
                return _row.name
            return None

        in_regions = _region_series(input_metric_base)
        out_regions = _region_series(work)
        in_freq = in_regions.value_counts(dropna=False)
        out_freq = out_regions.value_counts(dropna=False)
        dense_threshold = max(8, int(round(float(in_freq.quantile(0.75))))) if len(in_freq) > 0 else 8
        dense_keys = [str(k) for k, v in in_freq.items() if int(v) >= dense_threshold]
        addition_indices = []
        existing_xy = set((round(float(g.x), 6), round(float(g.y), 6)) for g in work.geometry if g is not None and not g.is_empty)
        for key in dense_keys:
            dense_target = 2
            if scale_den >= 300000 and int(in_freq.get(key, 0)) >= max(dense_threshold + 4, 12):
                dense_target = 3
            need = max(0, dense_target - int(out_freq.get(key, 0)))
            if need <= 0:
                continue
            for _ in range(need):
                idx = _pick_region_rep_idx(key, input_metric_base)
                if idx is None:
                    break
                addition_indices.append(idx)
        if addition_indices:
            repair_gdf = input_metric_base.loc[addition_indices].copy()
            work = gpd.GeoDataFrame(pd.concat([work, repair_gdf], axis=0, ignore_index=True), crs=work.crs)
            out_regions = _region_series(work)
            out_freq = out_regions.value_counts(dropna=False)

        if scale_den <= 220000:
            sparse_keys = [str(k) for k, v in in_freq.items() if int(v) <= 2]
            out_region_set = set(str(v) for v in out_freq.index.tolist())
            sparse_missing = [key for key in sparse_keys if key not in out_region_set]
            sparse_repair_limit = 12 if scale_den <= 135000 else 8
            sparse_addition_indices = []
            for key in sparse_missing[:sparse_repair_limit]:
                idx = _pick_region_rep_idx(key, input_metric_base)
                if idx is not None:
                    sparse_addition_indices.append(idx)
            if sparse_addition_indices:
                sparse_gdf = input_metric_base.loc[sparse_addition_indices].copy()
                work = gpd.GeoDataFrame(pd.concat([work, sparse_gdf], axis=0, ignore_index=True), crs=work.crs)
                out_regions = _region_series(work)
                out_freq = out_regions.value_counts(dropna=False)
                out_region_set = set(str(v) for v in out_freq.index.tolist())

            target_region_cov = 0.76 if scale_den <= 135000 else 0.72
            in_region_set = set(str(v) for v in in_freq.index.tolist())
            covered = len(in_region_set & out_region_set)
            target_cover_count = int(math.ceil(len(in_region_set) * target_region_cov))
            region_need = max(0, target_cover_count - covered)
            if region_need > 0:
                missing_regions = [str(k) for k, v in sorted(in_freq.items(), key=lambda item: (int(item[1]), str(item[0])))]
                missing_regions = [key for key in missing_regions if key not in out_region_set]
                region_addition_indices = []
                region_repair_limit = 10 if scale_den <= 135000 else 6
                for key in missing_regions[: min(region_need, region_repair_limit)]:
                    idx = _pick_region_rep_idx(key, input_metric_base)
                    if idx is not None:
                        region_addition_indices.append(idx)
                if region_addition_indices:
                    region_gdf = input_metric_base.loc[region_addition_indices].copy()
                    work = gpd.GeoDataFrame(pd.concat([work, region_gdf], axis=0, ignore_index=True), crs=work.crs)
    except Exception:
        pass

work = work[work.geometry.notna() & (~work.geometry.is_empty)].copy()
if geom_type == "point" and len(work) > 0 and len(input_metric_base) > 0 and int(plan.get("target_scale", 0) or 0) >= 300000:
    try:
        minx, miny, maxx, maxy = [float(v) for v in input_metric_base.total_bounds]
        width = max(1e-9, maxx - minx)
        height = max(1e-9, maxy - miny)
        region_bins = max(6, min(10, int(round((max(6, len(work)) ** 0.5) * 1.15))))

        def _region_series_325(_gdf):
            _rx = ((_gdf.geometry.x - minx) / width * region_bins).astype(int).clip(lower=0, upper=region_bins)
            _ry = ((_gdf.geometry.y - miny) / height * region_bins).astype(int).clip(lower=0, upper=region_bins)
            return pd.Series([f"{{int(x)}}_{{int(y)}}" for x, y in zip(_rx.tolist(), _ry.tolist())], index=_gdf.index)

        in_regions_325 = _region_series_325(input_metric_base)
        out_regions_325 = _region_series_325(work)
        in_freq_325 = in_regions_325.value_counts(dropna=False)
        out_freq_325 = out_regions_325.value_counts(dropna=False)
        dense_threshold_325 = max(8, int(round(float(in_freq_325.quantile(0.75))))) if len(in_freq_325) > 0 else 8
        dense_keys_325 = [str(k) for k, v in in_freq_325.items() if int(v) >= dense_threshold_325]
        existing_xy_325 = set(
            (round(float(g.x), 6), round(float(g.y), 6))
            for g in work.geometry
            if g is not None and not g.is_empty
        )
        add_idx_325 = []
        for key in dense_keys_325:
            grp = input_metric_base.loc[in_regions_325[in_regions_325 == key].index].copy()
            if len(grp) == 0:
                continue
            kept_grp = work.loc[out_regions_325[out_regions_325 == key].index].copy()
            target_keep = 2
            if int(len(grp)) >= max(dense_threshold_325 + 4, 12):
                target_keep = 3
            need = max(0, target_keep - int(len(kept_grp)))
            if need <= 0:
                continue

            gminx, gminy, gmaxx, gmaxy = [float(v) for v in grp.total_bounds]
            gwidth = max(1e-9, gmaxx - gminx)
            gheight = max(1e-9, gmaxy - gminy)
            sub_bins = 3 if len(grp) < 12 else 4
            grp["_sx325"] = ((grp.geometry.x - gminx) / gwidth * sub_bins).astype(int).clip(lower=0, upper=sub_bins)
            grp["_sy325"] = ((grp.geometry.y - gminy) / gheight * sub_bins).astype(int).clip(lower=0, upper=sub_bins)
            grp["_skey325"] = grp["_sx325"].astype(str) + "_" + grp["_sy325"].astype(str)
            occupied_sub = set()
            if len(kept_grp) > 0:
                ksx = ((kept_grp.geometry.x - gminx) / gwidth * sub_bins).astype(int).clip(lower=0, upper=sub_bins)
                ksy = ((kept_grp.geometry.y - gminy) / gheight * sub_bins).astype(int).clip(lower=0, upper=sub_bins)
                occupied_sub = set(f"{{int(x)}}_{{int(y)}}" for x, y in zip(ksx.tolist(), ksy.tolist()))

            if len(kept_grp) > 0:
                def _dist_to_kept(_geom):
                    return min(float(_geom.distance(g)) for g in kept_grp.geometry if g is not None and not g.is_empty)
                grp["_repair325"] = grp.geometry.map(_dist_to_kept)
            else:
                cx = float(grp.geometry.x.mean())
                cy = float(grp.geometry.y.mean())
                grp["_repair325"] = ((grp.geometry.x - cx) ** 2 + (grp.geometry.y - cy) ** 2) ** 0.5

            grp = grp.sort_values(["_skey325", "_repair325"], ascending=[True, False])
            picked_sub = set()
            for idx, row in grp.iterrows():
                geom = row.geometry
                if geom is None or geom.is_empty:
                    continue
                xy = (round(float(geom.x), 6), round(float(geom.y), 6))
                if xy in existing_xy_325:
                    continue
                skey = str(row["_skey325"])
                if skey in occupied_sub or skey in picked_sub:
                    continue
                add_idx_325.append(idx)
                existing_xy_325.add(xy)
                picked_sub.add(skey)
                need -= 1
                if need <= 0:
                    break
            if need > 0:
                grp = grp.sort_values("_repair325", ascending=False)
                for idx, row in grp.iterrows():
                    geom = row.geometry
                    if geom is None or geom.is_empty:
                        continue
                    xy = (round(float(geom.x), 6), round(float(geom.y), 6))
                    if xy in existing_xy_325:
                        continue
                    add_idx_325.append(idx)
                    existing_xy_325.add(xy)
                    need -= 1
                    if need <= 0:
                        break
        if add_idx_325:
            repair_325 = input_metric_base.loc[add_idx_325].copy()
            work = gpd.GeoDataFrame(pd.concat([work, repair_325], axis=0, ignore_index=True), crs=work.crs)
    except Exception:
        pass

if geom_type == "point" and len(work) > 0 and len(input_metric_base) > 0 and "poi_category" in input_metric_base.columns:
    try:
        input_cats = [
            str(x)
            for x in input_metric_base["poi_category"].dropna().astype(str).unique().tolist()
            if str(x) and str(x) != "nan"
        ]
        out_cats = set()
        if "poi_category" in work.columns:
            out_cats = set(str(x) for x in work["poi_category"].dropna().astype(str).tolist())
        missing_cats = [cat for cat in input_cats if cat not in out_cats]
        if missing_cats:
            existing_xy = set(
                (round(float(g.x), 6), round(float(g.y), 6))
                for g in work.geometry
                if g is not None and not g.is_empty
            )
            add_rows = []
            for cat in missing_cats:
                cand = input_metric_base[input_metric_base["poi_category"].astype(str) == cat].copy()
                if len(cand) == 0:
                    continue
                if len(work) > 0:
                    def _dist_to_work(_geom):
                        vals = [float(_geom.distance(g)) for g in work.geometry if g is not None and not g.is_empty]
                        return min(vals) if vals else 0.0
                    cand["__cat_repair_dist"] = cand.geometry.map(_dist_to_work)
                else:
                    cand["__cat_repair_dist"] = 0.0
                if "__sel_score" not in cand.columns:
                    cand["__sel_score"] = 0.0
                cand = cand.sort_values(["__cat_repair_dist", "__sel_score"], ascending=[False, False])
                for idx, row in cand.iterrows():
                    geom = row.geometry
                    if geom is None or geom.is_empty:
                        continue
                    xy = (round(float(geom.x), 6), round(float(geom.y), 6))
                    if xy in existing_xy:
                        continue
                    add_rows.append(row.drop(labels=[c for c in ["__cat_repair_dist"] if c in row.index]))
                    existing_xy.add(xy)
                    break
            if add_rows:
                repair_cats = gpd.GeoDataFrame(add_rows, geometry="geometry", crs=work.crs)
                work = gpd.GeoDataFrame(pd.concat([work, repair_cats], axis=0, ignore_index=True), crs=work.crs)
    except Exception:
        pass

if geom_type == "point" and len(work) > 1 and translate is not None:
    try:
        scale_den = int(plan.get("target_scale", 0) or 0)
        final_sep = max(
            float(plan.get("derived_constraints", {{}}).get("min_separation_m", 0.0) or 0.0),
            float(scale_den) * 0.00025,
            20.0,
        )
        rep = work.geometry.reset_index(drop=True).copy()
        for _round in range(10):
            moved = False
            sidx = rep.sindex
            for i, g in enumerate(rep):
                if g is None or g.is_empty:
                    continue
                for j in list(sidx.intersection(g.buffer(final_sep).bounds)):
                    if j <= i:
                        continue
                    h = rep.iloc[j]
                    if h is None or h.is_empty:
                        continue
                    d = float(g.distance(h))
                    if d >= final_sep:
                        continue
                    need = (final_sep - d) + max(0.25, final_sep * 0.015)
                    dx = float(g.x - h.x)
                    dy = float(g.y - h.y)
                    norm = (dx * dx + dy * dy) ** 0.5
                    if norm < 1e-9:
                        ang = ((i * 137 + j * 211 + _round * 47) % 6283) / 1000.0
                        ux, uy = math.cos(ang), math.sin(ang)
                    else:
                        ux, uy = dx / norm, dy / norm
                    shift = need / 2.0
                    rep.iat[i] = translate(g, xoff=ux * shift, yoff=uy * shift)
                    rep.iat[j] = translate(h, xoff=-ux * shift, yoff=-uy * shift)
                    moved = True
            if not moved:
                break
        work = work.copy().reset_index(drop=True)
        work["geometry"] = rep
    except Exception:
        pass

if "__sel_score" in work.columns:
    work = work.drop(columns=["__sel_score"])
if "__protected_keep" in work.columns:
    work = work.drop(
        columns=[
            c
            for c in [
                "__protected_keep",
                "__metric_rkey",
                "__metric_dense_region",
                "__metric_region_quota",
                "__metric_region_bins",
                "__metric_minx",
                "__metric_miny",
                "__metric_width",
                "__metric_height",
                "__orig_x",
                "__orig_y",
            ]
            if c in work.columns
        ]
    )
if len(work) == 0:
    raise ValueError("All features were filtered out; adjust constraints.")

metric_work = work.copy()
export_work = work.copy()
if geom_type == "point":
    export_work, poi_symbol_field = _apply_poi_symbols(export_work)
    metric_work, _ = _apply_poi_symbols(metric_work)
if str(original_crs) != str(export_work.crs):
    export_work = export_work.to_crs(original_crs)

os.makedirs(os.path.dirname(output_geojson) or ".", exist_ok=True)
if include_geojson:
    export_work.to_file(output_geojson, driver="GeoJSON")
    show_layer(output_geojson, output_layer_name)

if include_png:
    try:
        import matplotlib.pyplot as plt
        output_pdf = os.path.splitext(output_png)[0] + ".pdf"
        _page_mm = {{
            "A4": (210.0, 297.0),
            "A3": (297.0, 420.0),
            "A2": (420.0, 594.0),
        }}
        _bounds = export_work.total_bounds
        _dx = max(1e-12, float(_bounds[2]) - float(_bounds[0]))
        _dy = max(1e-12, float(_bounds[3]) - float(_bounds[1]))
        _orientation = "landscape" if _dx >= _dy else "portrait"
        _page_name = "A4"
        _count = int(len(export_work))
        _scale_den = int(plan.get("target_scale", 0) or 0)
        if _count >= 110 or _scale_den >= 500000:
            _page_name = "A3"
        if _count >= 220:
            _page_name = "A2"
        _pw_mm, _ph_mm = _page_mm[_page_name]
        if _orientation == "landscape":
            _pw_mm, _ph_mm = max(_pw_mm, _ph_mm), min(_pw_mm, _ph_mm)
        else:
            _pw_mm, _ph_mm = min(_pw_mm, _ph_mm), max(_pw_mm, _ph_mm)
        fig, ax = plt.subplots(figsize=(_pw_mm / 25.4, _ph_mm / 25.4))
        fig.patch.set_facecolor("white")
        fig.subplots_adjust(left=0.045, right=0.985, top=0.91, bottom=0.075)
        if geom_type == "point":
            point_marker_size = 3.8 if _scale_den >= 500000 else (4.6 if _scale_den >= 200000 else 5.8)
            _marker_map = {{
                "circle": "o",
                "square": "s",
                "triangle": "^",
                "diamond": "D",
                "cross": "x",
                "star": "*",
                "plus": "P",
                "hexagon": "h",
                "convenience": "s",
                "school": "$S$",
                "parking": "$P$",
                "hotel": "$H$",
                "bank": "$B$",
                "hospital": "$+$",
                "restaurant": "$R$",
                "transit": "$T$",
            }}
            _symbol_colors = {{
                "convenience": "#d23b7d",
                "school": "#2e7d32",
                "parking": "#1976d2",
                "hotel": "#7b4fa1",
                "bank": "#00897b",
                "hospital": "#d32f2f",
                "restaurant": "#ef8a24",
                "transit": "#4e79a7",
            }}
            _colors = ["#c62828", "#1565c0", "#2e7d32", "#6a1b9a", "#ef6c00", "#00838f", "#4527a0", "#6d4c41"]
            if "point_symbol" in export_work.columns:
                for _i, (_sym, _sub) in enumerate(export_work.groupby("point_symbol")):
                    _mk = _marker_map.get(str(_sym).lower().strip(), "o")
                    _c = _symbol_colors.get(str(_sym).lower().strip())
                    if not _c and "point_symbol_color" in _sub.columns and len(_sub) > 0:
                        _c = str(_sub["point_symbol_color"].iloc[0])
                    _c = _c or _colors[_i % len(_colors)]
                    _sub.plot(ax=ax, color=_c, markersize=point_marker_size * 2.2, marker=_mk, linestyle="None")
            else:
                export_work.plot(ax=ax, color="#d81b60", markersize=point_marker_size)
        elif geom_type == "line":
            export_work.plot(ax=ax, color="#1976d2", linewidth=1.2)
        else:
            export_work.plot(ax=ax, facecolor="#66bb6a", edgecolor="#2e7d32", linewidth=0.6)
        _pad_x = max(_dx * 0.18, 1e-6)
        _pad_y = max(_dy * 0.18, 1e-6)
        ax.set_xlim(float(_bounds[0]) - _pad_x, float(_bounds[2]) + _pad_x)
        ax.set_ylim(float(_bounds[1]) - _pad_y, float(_bounds[3]) + _pad_y)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"Generalized 1:{{plan.get('target_scale')}} - {{_page_name}} {{_orientation}} adaptive layout")
        ax.set_axis_off()
        for _spine in ax.spines.values():
            _spine.set_visible(True)
            _spine.set_linewidth(0.8)
            _spine.set_edgecolor("#333333")
        fig.text(0.985, 0.03, f"{{_page_name}} {{_orientation}} | adaptive result extent | target scale 1:{{plan.get('target_scale')}}", ha="right", va="bottom", fontsize=8, color="#555555")
        fig.savefig(output_png, dpi=360, facecolor="white")
        fig.savefig(output_pdf, facecolor="white")
        print("layout_png:", output_png)
        print("layout_pdf:", output_pdf)
        plt.close(fig)
    except Exception as _img_err:
        # Do not block generalization when image dependency is missing.
        print("warn:image_export_skipped", str(_img_err))

metrics = {{
    "input_count": int(len(gdf)),
    "output_count": int(len(export_work)),
    "target_scale": int(plan.get("target_scale", 0)),
    "geometry_type": geom_type,
    "topology_error_count": int((~export_work.geometry.is_valid).sum()),
}}
metrics["retention_rate"] = float(metrics["output_count"]) / float(metrics["input_count"]) if metrics["input_count"] > 0 else 1.0
if geom_type == "point":
    metrics["poi_symbol_field"] = str(poi_symbol_field or "")
    metrics["poi_symbol_rule_count"] = int(len(POI_SYMBOL_RULES))
    metrics["poi_symbol_categories"] = sorted([str(x) for x in export_work.get("poi_category", pd.Series(dtype=str)).dropna().unique().tolist()])

metric = metric_work
if getattr(metric.crs, "is_geographic", False):
    metric = metric.to_crs("EPSG:3857")
rep = metric.geometry.representative_point()
min_sep = float(plan.get("derived_constraints", {{}}).get("min_separation_m", 0.0))
conflicts = 0
min_dist = None
if min_sep > 0 and len(rep) > 1:
    sidx = rep.sindex
    for i, g in enumerate(rep):
        if g is None or g.is_empty:
            continue
        cand = list(sidx.intersection(g.buffer(min_sep).bounds))
        for j in cand:
            if j <= i:
                continue
            d = g.distance(rep.iloc[j])
            if min_dist is None or d < min_dist:
                min_dist = d
            if d < min_sep:
                conflicts += 1
metrics["conflict_count_total"] = int(conflicts)
metrics["min_symbol_separation_m"] = float(min_dist) if min_dist is not None else float(min_sep)
b = metric.total_bounds
area_km2 = max(1e-6, abs((b[2]-b[0]) * (b[3]-b[1])) / 1_000_000.0)
metrics["conflict_density_per_km2"] = float(metrics["conflict_count_total"]) / float(area_km2)

if geom_type == "point":
    try:
        input_metric = gdf.copy()
        output_metric = work.copy()
        if str(input_metric.crs) != str(output_metric.crs):
            input_metric = input_metric.to_crs(output_metric.crs)
        if getattr(input_metric.crs, "is_geographic", False):
            input_metric = input_metric.to_crs("EPSG:3857")
            output_metric = output_metric.to_crs("EPSG:3857")

        if len(input_metric) > 4 and len(output_metric) > 0:
            in_x = input_metric.geometry.x
            in_y = input_metric.geometry.y
            mx = float(in_x.mean())
            my = float(in_y.mean())
            dx = in_x - mx
            dy = in_y - my
            sxx = float((dx * dx).mean())
            syy = float((dy * dy).mean())
            sxy = float((dx * dy).mean())
            theta = 0.5 * math.atan2(2.0 * sxy, (sxx - syy))
            ux = math.cos(theta)
            uy = math.sin(theta)
            in_t = (in_x - mx) * ux + (in_y - my) * uy
            out_t = (output_metric.geometry.x - mx) * ux + (output_metric.geometry.y - my) * uy
            tmin = float(in_t.min())
            tmax = float(in_t.max())
            if tmax > tmin:
                axis_bins = max(8, min(32, int(round((len(input_metric) ** 0.5) * 2.5))))
                in_bin = ((in_t - tmin) / (tmax - tmin + 1e-9) * axis_bins).astype(int).clip(lower=0, upper=axis_bins)
                out_bin = ((out_t - tmin) / (tmax - tmin + 1e-9) * axis_bins).astype(int).clip(lower=0, upper=axis_bins)
                in_set = set(int(v) for v in in_bin.dropna().unique().tolist())
                out_set = set(int(v) for v in out_bin.dropna().unique().tolist())
                metrics["point_axis_coverage_ratio"] = float(len(in_set & out_set)) / float(max(1, len(in_set)))

                run = 0
                longest = 0
                for bi in range(axis_bins + 1):
                    if bi in in_set and bi not in out_set:
                        run += 1
                        longest = max(longest, run)
                    elif bi in in_set:
                        run = 0
                metrics["point_middle_gap_ratio"] = float(longest) / float(max(1, len(in_set)))

            # Dense-hotspot retention metrics (grid-based proxy for VLM visual judgement).
            hot_bins = max(6, min(20, int(round((len(input_metric) ** 0.5) * 0.8))))
            in_bb = input_metric.total_bounds
            minx, miny, maxx, maxy = [float(v) for v in in_bb]
            width = max(1e-9, maxx - minx)
            height = max(1e-9, maxy - miny)
            in_hx = ((input_metric.geometry.x - minx) / width * hot_bins).astype(int).clip(lower=0, upper=hot_bins)
            in_hy = ((input_metric.geometry.y - miny) / height * hot_bins).astype(int).clip(lower=0, upper=hot_bins)
            out_hx = ((output_metric.geometry.x - minx) / width * hot_bins).astype(int).clip(lower=0, upper=hot_bins)
            out_hy = ((output_metric.geometry.y - miny) / height * hot_bins).astype(int).clip(lower=0, upper=hot_bins)
            in_keys = pd.Series([f"{{int(x)}}_{{int(y)}}" for x, y in zip(in_hx.tolist(), in_hy.tolist())])
            out_keys = pd.Series([f"{{int(x)}}_{{int(y)}}" for x, y in zip(out_hx.tolist(), out_hy.tolist())])
            in_freq = in_keys.value_counts(dropna=False)
            hotspot_cells = []
            if len(in_freq) > 0:
                q90 = float(in_freq.quantile(0.90))
                min_hot = max(3.0, q90)
                hotspot_cells_all = [str(k) for k, v in in_freq.items() if float(v) >= min_hot]
                hotspot_cells = hotspot_cells_all[: max(6, min(18, len(hotspot_cells_all)))]
            out_set_hot = set(str(v) for v in out_keys.tolist())
            covered_hot = [k for k in hotspot_cells if k in out_set_hot]
            metrics["point_hotspot_coverage_ratio"] = float(len(covered_hot)) / float(max(1, len(hotspot_cells)))
            if hotspot_cells:
                metrics["point_hotspot_rep_count"] = int(sum(1 for k in out_keys.tolist() if str(k) in set(hotspot_cells)))
            else:
                metrics["point_hotspot_rep_count"] = int(len(output_metric))

            # Region-coverage metrics: preserve region existence before pure thinning.
            region_bins = max(3, min(10, int(round((max(6, len(output_metric)) ** 0.5) * 1.15))))
            in_rx = ((input_metric.geometry.x - minx) / width * region_bins).astype(int).clip(lower=0, upper=region_bins)
            in_ry = ((input_metric.geometry.y - miny) / height * region_bins).astype(int).clip(lower=0, upper=region_bins)
            out_rx = ((output_metric.geometry.x - minx) / width * region_bins).astype(int).clip(lower=0, upper=region_bins)
            out_ry = ((output_metric.geometry.y - miny) / height * region_bins).astype(int).clip(lower=0, upper=region_bins)
            in_regions = pd.Series([f"{{int(x)}}_{{int(y)}}" for x, y in zip(in_rx.tolist(), in_ry.tolist())])
            out_regions = pd.Series([f"{{int(x)}}_{{int(y)}}" for x, y in zip(out_rx.tolist(), out_ry.tolist())])
            in_region_freq = in_regions.value_counts(dropna=False)
            out_region_freq = out_regions.value_counts(dropna=False)
            in_region_set = set(str(v) for v in in_region_freq.index.tolist())
            out_region_set = set(str(v) for v in out_region_freq.index.tolist())
            metrics["point_input_region_count"] = int(len(in_region_set))
            metrics["point_output_region_count"] = int(len(out_region_set))
            metrics["point_region_coverage_ratio"] = float(len(in_region_set & out_region_set)) / float(max(1, len(in_region_set)))

            sparse_region_keys = [str(k) for k, v in in_region_freq.items() if int(v) <= 2]
            sparse_region_hits = [k for k in sparse_region_keys if k in out_region_set]
            metrics["point_sparse_region_count"] = int(len(sparse_region_keys))
            metrics["point_sparse_region_loss_count"] = int(max(0, len(sparse_region_keys) - len(sparse_region_hits)))
            metrics["point_sparse_region_coverage_ratio"] = float(len(sparse_region_hits)) / float(max(1, len(sparse_region_keys)))

            dense_threshold = max(8, int(round(float(in_region_freq.quantile(0.75))))) if len(in_region_freq) > 0 else 8
            dense_region_keys = [str(k) for k, v in in_region_freq.items() if int(v) >= dense_threshold]
            dense_singletons = 0
            for key in dense_region_keys:
                if int(out_region_freq.get(key, 0)) <= 1:
                    dense_singletons += 1
            metrics["point_dense_region_count"] = int(len(dense_region_keys))
            metrics["point_dense_region_singleton_ratio"] = float(dense_singletons) / float(max(1, len(dense_region_keys)))
    except Exception as _point_eval_err:
        print("warn:point_structure_metrics_skipped", str(_point_eval_err))

if include_geojson:
    print("output:", output_geojson)
if include_png and os.path.exists(output_png):
    print("output:", output_png)
print("metrics:", json.dumps(metrics, ensure_ascii=False))
"""

    def _run_scale_generalization(self, task: Dict[str, Any]) -> Optional[str]:
        run_started = os.times()
        params = self._resolve_generalization_input(task.get("parameters", {}) or {})
        target_scale = self._parse_scale_denominator(params.get("target_scale", 0), 0)
        params["target_scale"] = target_scale
        if target_scale <= 0:
            return "Please provide target scale like `1:35000`."

        input_file = self._resolve_existing_vector_path(str(params.get("input_file", "")).strip())
        if input_file:
            params["input_file"] = input_file
        if not input_file or not os.path.exists(input_file):
            return "No valid input layer found. Select current layer first or provide file path."

        source_scale = self._parse_scale_denominator(params.get("source_scale", 10000), 10000)
        params["source_scale"] = source_scale
        feature_count = int(params.get("feature_count", 0) or 0)
        geometry_type = str(params.get("geometry_type", "")).lower()
        if geometry_type.startswith("multi"):
            geometry_type = geometry_type.replace("multi", "", 1)
        if geometry_type not in {"point", "line", "polygon"}:
            geometry_type = "polygon" if "polygon" in geometry_type else ("point" if "point" in geometry_type else "line")

        output_formats = params.get("output_formats") or ["geojson", "png"]
        if not isinstance(output_formats, list):
            output_formats = ["geojson", "png"]
        output_formats = [str(x).lower() for x in output_formats]
        if "geojson" not in output_formats and "png" not in output_formats:
            output_formats = ["geojson", "png"]

        base = os.path.splitext(input_file)[0] + f"_gen_1_{target_scale}"
        base_layer_name = str(params.get("layer_name") or os.path.basename(base))
        before_img = self._capture_map_screenshot(f"{base}_before_map.png")
        self._emit("thinking", "Todo 1/6: 读取当前数据，识别语义类型与空间分布特征")
        profile = self._infer_point_semantic_profile(input_file)
        for line in self._build_point_data_specific_reasoning(profile, target_scale):
            self._emit("thought", line)
        evidence = list(profile.get("semantic_evidence", [])) + list(profile.get("structure_evidence", []))
        if evidence:
            self._emit("thought", f"判定依据：{evidence}")
        self._emit("thought", "接下来会结合技能模板、当前分布特征和目标比例尺，选择更合适的综合算子。")
        self._emit("thinking", "Todo 2/6: 根据当前数据特征选择候选算子与参数")

        planner = GeneralizationPlanner()
        retention_min = planner.rules.get("retention_min", {}) if isinstance(planner.rules, dict) else {}
        if geometry_type == "point":
            if target_scale >= 500000:
                point_retention_min = 0.08
            elif target_scale >= 200000:
                point_retention_min = 0.12
            else:
                point_retention_min = 0.25
        else:
            point_retention_min = float(
                (planner.rules.get("thresholds", {}).get("point_min_keep_ratio", 0.15) if isinstance(planner.rules, dict) else 0.15)
                or 0.15
            )
        point_scale_floor_cfg = (
            planner.rules.get("thresholds", {}).get("point_scale_min_keep_ratio", {})
            if isinstance(planner.rules, dict)
            else {}
        )
        soft_max_rounds = int(planner.rules.get("thresholds", {}).get("max_rounds", 4) or 4)
        soft_max_rounds = max(1, min(soft_max_rounds, 4))
        hard_max_rounds = soft_max_rounds
        require_real_vlm = bool(
            str(os.environ.get("MAPGENAI_REQUIRE_REAL_VLM", "")).strip() in {"1", "true", "TRUE", "yes"}
            or bool(params.get("require_real_vlm", False))
        )

        current_input = input_file
        qa_metrics: Dict[str, Any] = {}
        last_fail_reason = ""
        final_outputs: List[str] = []
        final_verify: Dict[str, Any] = {}
        stable_outputs: List[str] = []
        stable_round = 0
        stable_metrics: Dict[str, Any] = {}
        stable_hotspot_cov = -1.0
        stable_verify_score = -1.0

        for round_idx in range(1, hard_max_rounds + 1):
            self._emit("thinking", f"Todo 2/4: plan iteration {round_idx}/{hard_max_rounds}")
            plan = planner.plan(
                PlanningRequest(
                    source_scale=source_scale,
                    target_scale=target_scale,
                    layer_stats={
                        "geometry_type": geometry_type,
                        "feature_count": feature_count,
                        "semantic_type": profile.get("semantic_type"),
                        "structure_type": profile.get("structure_type"),
                    },
                    qa_metrics=qa_metrics,
                    round_index=round_idx,
                    last_fail_reason=last_fail_reason,
                )
            )
            if geometry_type == "point":
                plan["point_strategy"] = {
                    "semantic_type": str(profile.get("semantic_type", "generic_point")),
                    "structure_type": str(profile.get("structure_type", "mixed")),
                    "semantic_evidence": list(profile.get("semantic_evidence", [])),
                    "structure_evidence": list(profile.get("structure_evidence", [])),
                    "skills": [
                        "skills/point_generalization_skill.md",
                        "map-generalization",
                    ],
                }
            if round_idx == 1:
                reason_text = self._build_operator_reasoning(profile, plan)
                self._emit("thought", f"算子决策：{reason_text}")
                self._emit("thinking", "Todo 3/6: 执行算子并生成当前轮结果")

            geojson_out = str(params.get("output_geojson") or f"{base}_r{round_idx}.geojson")
            png_out = str(params.get("output_png") or f"{base}_r{round_idx}.png")
            layer_name = f"{base_layer_name}_r{round_idx}"
            code = self._build_generalization_code(
                plan=plan,
                input_file=current_input,
                output_geojson=geojson_out,
                output_png=png_out,
                output_layer_name=layer_name,
                include_geojson="geojson" in output_formats,
                include_png="png" in output_formats,
            )

            self._emit("thinking", f"Todo 3/4: execute iteration {round_idx}/{hard_max_rounds}")
            self._emit("tool_start", {"name": "run_python_script", "args": {"code": code}})
            result = tool_registry.execute("run_python_script", {"code": code})
            self._emit("tool_end", {"name": "run_python_script", "result": result})

            handled = self.result_manager.handle_result(result)
            if handled.get("status") != "success":
                return (
                    "Scale generalization failed.\n"
                    f"Error: {handled.get('message', '')}\n"
                    f"Suggestion: {handled.get('suggestion', '')}"
                )

            stdout_text = str(result.get("stdout", "") or "")
            qa_metrics = self._extract_metrics_from_stdout(stdout_text)
            projected_crs = self._extract_projected_crs_from_stdout(stdout_text)
            if projected_crs:
                qa_metrics["projected_crs"] = projected_crs
            self._emit(
                "thought",
                f"第 {round_idx} 轮结果：输入 {qa_metrics.get('input_count')} 个，输出 {qa_metrics.get('output_count')} 个，"
                f"保留率 {qa_metrics.get('retention_rate')}，冲突数 {qa_metrics.get('conflict_count_total')}。",
            )
            after_img = png_out if "png" in output_formats and os.path.exists(png_out) else ""
            if not after_img:
                after_img = self._capture_map_screenshot(f"{base}_r{round_idx}_map.png")

            self.vlm_verifier.callback = ui_callbacks.get("vlm_verify")
            symbol_context = {
                "expected_poi_categories": list(POI_SYMBOL_RULES.keys()),
                "poi_symbol_rules": POI_SYMBOL_RULES,
                "poi_symbol_field": str(qa_metrics.get("poi_symbol_field") or ""),
                "poi_symbol_categories": qa_metrics.get("poi_symbol_categories", []),
                "check_items": [
                    "check whether the 8 POI categories use the expected classic pictogram symbols",
                    "check whether symbols are too large or overlap",
                    "compare before/after structure and identify vanished sparse/hotspot regions",
                ],
            }
            verify = self.vlm_verifier.verify(
                VerifyInput(
                    target_scale=target_scale,
                    metrics=qa_metrics,
                    stop_conditions=plan.get("stop_conditions", {}),
                    retention_min=(
                        {"point": point_retention_min}
                        if geometry_type == "point"
                        else (retention_min if isinstance(retention_min, dict) else {})
                    ),
                    before_image=before_img,
                    after_image=after_img,
                    last_fail_reason=last_fail_reason,
                    symbol_context=symbol_context,
                    require_real_vlm=require_real_vlm,
                )
            )
            self._emit("thinking", "Todo 4/6: 结合视觉分布和规则阈值验收当前轮结果")
            self._emit(
                "thought",
                f"验收结果：pass={verify.get('pass')}, score={verify.get('score')}, failed_rules={verify.get('failed_rules', [])}",
            )
            final_verify = verify
            final_outputs = handled.get("output_files", [])
            out_count = int(qa_metrics.get("output_count", 0) or 0)
            input_count = int(qa_metrics.get("input_count", feature_count) or feature_count or 0)
            structure_floor = {
                "corridor": 0.24,
                "uniform": 0.20,
                "clustered": 0.18,
                "center-periphery": 0.20,
                "multi-core": 0.22,
                "mixed": 0.18,
            }.get(str(profile.get("structure_type", "mixed")), 0.18)
            semantic_floor = 0.22 if str(profile.get("semantic_type", "")) in {"transport_network_points", "commercial_poi_points"} else 0.0
            scale_floor = 0.0
            if isinstance(point_scale_floor_cfg, dict) and point_scale_floor_cfg:
                try:
                    scale_floor = float(planner._interpolate(point_scale_floor_cfg, target_scale))
                except Exception:
                    scale_floor = 0.0
            min_keep_ratio = point_retention_min
            strict_no_overlap_scale = bool(geometry_type == "point" and target_scale >= 200000)
            if strict_no_overlap_scale:
                min_point_keep = max(10, int(input_count * min_keep_ratio)) if input_count > 0 else 10
            else:
                min_point_keep = max(18, int(input_count * min_keep_ratio)) if input_count > 0 else 18
            max_keep_ratio = 1.0
            if geometry_type == "point":
                if target_scale >= 500000:
                    max_keep_ratio = 0.18
                elif target_scale >= 200000:
                    max_keep_ratio = 0.34
                else:
                    max_keep_ratio = 0.70
            max_point_keep = int(math.ceil(input_count * max_keep_ratio)) if input_count > 0 else 0

            if geometry_type == "point" and min_point_keep <= out_count <= max_point_keep:
                current_hotspot_cov = float(qa_metrics.get("point_hotspot_coverage_ratio", -1.0) or -1.0)
                current_verify_score = float(verify.get("score", 0) or 0.0)
                better = (
                    (current_hotspot_cov > stable_hotspot_cov + 1e-9)
                    or (
                        abs(current_hotspot_cov - stable_hotspot_cov) <= 1e-9
                        and current_verify_score > stable_verify_score + 1e-9
                    )
                    or (
                        abs(current_hotspot_cov - stable_hotspot_cov) <= 1e-9
                        and abs(current_verify_score - stable_verify_score) <= 1e-9
                        and out_count > int(stable_metrics.get("output_count", 0) or 0)
                    )
                )
                if better:
                    stable_outputs = list(final_outputs)
                    stable_round = round_idx
                    stable_metrics = dict(qa_metrics)
                    stable_hotspot_cov = current_hotspot_cov
                    stable_verify_score = current_verify_score

            if verify.get("pass"):
                if geometry_type == "point" and out_count < min_point_keep:
                    self._emit(
                        "thought",
                        f"VLM 审核通过，但保留点数 {out_count} 低于硬阈值 {min_point_keep}；继续自动调参避免过度综合。",
                    )
                    last_fail_reason = "retention_rate >= class_min"
                    before_img = after_img or before_img
                    continue
                if geometry_type == "point" and out_count > max_point_keep:
                    self._emit(
                        "thought",
                        f"VLM 审核通过，但保留点数 {out_count} 高于上限 {max_point_keep}；继续自动调参压缩密集区。",
                    )
                    last_fail_reason = "point_rgb_hotspot_overdraw"
                    before_img = after_img or before_img
                    continue
                if callable(ui_callbacks.get("zoom_to_layer")):
                    try:
                        ui_callbacks["zoom_to_layer"](layer_name)
                    except Exception:
                        pass
                score = int(verify.get("score", 0) or 0)
                files_text = ", ".join(final_outputs) if final_outputs else ", ".join(
                    [
                        x
                        for x in [
                            geojson_out if "geojson" in output_formats else "",
                            png_out if "png" in output_formats else "",
                        ]
                        if x
                    ]
                )
                basis_doc = self._write_selection_basis_doc(
                    base_path=base,
                    input_file=input_file,
                    target_scale=target_scale,
                    profile=profile,
                    plan=plan,
                    verify=verify,
                    output_files=final_outputs,
                    projected_crs=str(qa_metrics.get("projected_crs") or ""),
                )
                if basis_doc:
                    files_text = f"{files_text}, {basis_doc}" if files_text else basis_doc
                cpu_elapsed = (os.times().elapsed - run_started.elapsed) if hasattr(run_started, "elapsed") else 0.0
                decision_txt = self._format_user_visible_decision_summary(
                    target_scale=target_scale,
                    profile=profile,
                    plan=plan,
                    qa_metrics=qa_metrics,
                    verify=verify,
                    round_index=round_idx,
                    basis_doc_path=basis_doc or "",
                )
                return (
                    f"Generalization completed at 1:{target_scale}.\n"
                    f"VLM verify: pass (score={score}, mode={verify.get('vlm_mode', 'unknown')}, real={verify.get('real_vlm_called', False)})\n"
                    "Completion status: completed\n"
                    f"Output files: {files_text}\n"
                    f"{decision_txt}\n"
                    "Verified: VLM acceptance checks passed.\n"
                    f"Runtime: {cpu_elapsed:.2f}s\n"
                    f"Projected CRS used: {qa_metrics.get('projected_crs') or 'N/A'}\n"
                    + ("GeoJSON auto-loaded to map." if "geojson" in output_formats else "Export finished.")
                )

            if geometry_type == "point" and round_idx > 1 and out_count < min_point_keep and stable_outputs:
                kept = int(stable_metrics.get("output_count", 0) or 0)
                self._emit(
                    "thought",
                    f"当前轮保留点数不足，暂不作为最终结果；继续自动调参，参考稳定轮次 {stable_round}（保留 {kept} 个点）。",
                )
                last_fail_reason = "retention_rate >= class_min; point_region_coverage_ratio >= min"
                before_img = after_img or before_img
                continue
                files_text = ", ".join(stable_outputs)
                kept = int(stable_metrics.get("output_count", 0) or 0)
                basis_doc = self._write_selection_basis_doc(
                    base_path=base,
                    input_file=input_file,
                    target_scale=target_scale,
                    profile=profile,
                    plan=plan,
                    verify=verify,
                    output_files=stable_outputs,
                    projected_crs=str(projected_crs or stable_metrics.get("projected_crs", "")),
                )
                if basis_doc:
                    files_text = f"{files_text}, {basis_doc}" if files_text else basis_doc
                cpu_elapsed = (os.times().elapsed - run_started.elapsed) if hasattr(run_started, "elapsed") else 0.0
                decision_txt = self._format_user_visible_decision_summary(
                    target_scale=target_scale,
                    profile=profile,
                    plan=plan,
                    qa_metrics=stable_metrics if stable_metrics else qa_metrics,
                    verify=verify,
                    round_index=int(stable_round or round_idx),
                    basis_doc_path=basis_doc or "",
                )
                return (
                    f"Generalization completed with guarded stop at 1:{target_scale}.\n"
                    f"Completion status: failed\n"
                    f"Output files: {files_text}\n"
                    f"Stability note: fallback to round {stable_round} to avoid over-generalization.\n"
                    f"VLM verify still failed (score={int(verify.get('score', 0) or 0)}, mode={verify.get('vlm_mode', 'unknown')}, real={verify.get('real_vlm_called', False)}).\n"
                    f"Verified: point retention guard triggered (kept {kept} features, "
                    f"avoid over-generalization below {min_point_keep}).\n"
                    f"{decision_txt}\n"
                    f"Runtime: {cpu_elapsed:.2f}s\n"
                    f"Projected CRS used: {projected_crs or stable_metrics.get('projected_crs', 'N/A')}"
                )

            # Point layers should re-run from original input each round to avoid compounding over-thinning.
            if geometry_type != "point" and "geojson" in output_formats and os.path.exists(geojson_out):
                current_input = geojson_out
            failed_rules = verify.get("failed_rules", [])
            if isinstance(failed_rules, list) and failed_rules:
                last_fail_reason = "; ".join(str(x) for x in failed_rules)[:200]
            before_img = after_img or before_img

        fail_score = int(final_verify.get("score", 0) or 0)
        fail_rules = final_verify.get("failed_rules", [])
        if stable_outputs:
            final_outputs = stable_outputs
        basis_doc = self._write_selection_basis_doc(
            base_path=base,
            input_file=input_file,
            target_scale=target_scale,
            profile=profile,
            plan=plan if isinstance(plan, dict) else {},
            verify=final_verify,
            output_files=final_outputs,
            projected_crs=str(stable_metrics.get("projected_crs") or qa_metrics.get("projected_crs") or ""),
        )
        files_text = ", ".join(final_outputs)
        if basis_doc:
            files_text = f"{files_text}, {basis_doc}" if files_text else basis_doc
        cpu_elapsed = (os.times().elapsed - run_started.elapsed) if hasattr(run_started, "elapsed") else 0.0
        decision_txt = self._format_user_visible_decision_summary(
            target_scale=target_scale,
            profile=profile,
            plan=plan if isinstance(plan, dict) else {},
            qa_metrics=stable_metrics if stable_metrics else qa_metrics,
            verify=final_verify,
            round_index=hard_max_rounds,
            basis_doc_path=basis_doc or "",
        )
        return (
            f"Generalization failed to pass verification at 1:{target_scale} after {hard_max_rounds} rounds.\n"
            f"VLM verify still failed (score={fail_score}, mode={final_verify.get('vlm_mode', 'unknown')}, real={final_verify.get('real_vlm_called', False)}).\n"
            "Completion status: failed\n"
            f"Failed rules: {fail_rules}\n"
            f"Output files: {files_text}\n"
            f"{decision_txt}\n"
            "Verified: outputs generated but acceptance thresholds not fully met.\n"
            f"Runtime: {cpu_elapsed:.2f}s\n"
            f"Projected CRS used: {stable_metrics.get('projected_crs') or qa_metrics.get('projected_crs') or 'N/A'}"
        )

    def _extract_code_from_text(self, text: str) -> str:
        if not text:
            return ""
        m = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.S | re.I)
        return m.group(1).strip() if m else ""

    def _auto_fix_and_retry(
        self,
        tool_call: Any,
        result: Dict[str, Any],
        tools: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(result, dict) or result.get("status") != "error":
            return None

        traceback_text = result.get("traceback") or result.get("stderr") or result.get("message", "")
        original_code = (tool_call.arguments or {}).get("code", "")
        if not traceback_text or not original_code:
            return None

        fix_prompt = (
            "Code execution failed. Return only fixed code or invoke same tool again.\n"
            f"Tool: {tool_call.name}\n"
            f"Original code:\n```python\n{original_code}\n```\n"
            f"Error:\n```\n{traceback_text}\n```"
        )

        for _ in range(self.max_auto_fix_attempts):
            self._emit("thinking", "Auto-fix retry...")
            try:
                fix_response = self.provider.chat(self.messages + [Message("user", fix_prompt)], tools)
            except Exception as exc:
                return {
                    "name": f"{tool_call.name}_auto_fix",
                    "content": json.dumps({"status": "error", "message": f"auto-fix failed: {exc}"}, ensure_ascii=False),
                }

            retry_call = None
            if fix_response.tool_calls:
                for tc in fix_response.tool_calls:
                    if tc.name == tool_call.name:
                        retry_call = tc
                        break

            if retry_call is None:
                code = self._extract_code_from_text(fix_response.content or "")
                if not code:
                    return {
                        "name": f"{tool_call.name}_auto_fix",
                        "content": json.dumps({"status": "error", "message": "auto-fix did not return code"}, ensure_ascii=False),
                    }

                class _RetryCall:
                    name = tool_call.name
                    arguments = {"code": code}

                retry_call = _RetryCall()

            self._emit("tool_start", {"name": f"{retry_call.name}_auto_fix", "args": retry_call.arguments})
            retry_result = tool_registry.execute(retry_call.name, retry_call.arguments)
            self._emit("tool_end", {"name": f"{retry_call.name}_auto_fix", "result": retry_result})

            return {
                "name": f"{retry_call.name}_auto_fix",
                "content": json.dumps(
                    {
                        "status": retry_result.get("status"),
                        "fixed_code": (retry_call.arguments or {}).get("code", ""),
                        "result": retry_result,
                    },
                    ensure_ascii=False,
                ),
            }

        return None

    def stream_chat(self, user_message: str):
        yield self.chat(user_message)

    def clear_history(self):
        self.messages = [Message("system", SYSTEM_PROMPT)]

    def get_available_tools(self) -> List[str]:
        return list(tool_registry.tools.keys())


def create_agent(config: Dict[str, Any]) -> MapGenAgent:
    ai_config = config.get("ai", {})
    provider = create_provider(
        provider=ai_config.get("provider", "zhipu"),
        api_key=ai_config.get("api_key", ""),
        model=ai_config.get("model"),
        temperature=ai_config.get("temperature", 0.7),
        max_tokens=ai_config.get("max_tokens", 4096),
        base_url=ai_config.get("base_url"),
    )
    return MapGenAgent(provider)
