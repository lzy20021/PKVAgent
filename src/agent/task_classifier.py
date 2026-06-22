import json
import re
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from .providers import Message


class TaskType(str, Enum):
    SCALE_GENERALIZATION = "scale_generalization"
    LINE_SIMPLIFICATION = "line_simplification"
    POLYGON_GENERALIZATION = "polygon_generalization"
    SPATIAL_ANALYSIS = "spatial_analysis"
    COORDINATE_TRANSFORM = "coordinate_transform"
    TOPOLOGY_REPAIR = "topology_repair"
    UNKNOWN = "unknown"


class TaskClassifier:
    """Hybrid GIS task classifier: keyword rules first, LLM fallback second."""

    def __init__(self, llm_provider: Any = None):
        self.llm = llm_provider
        self.task_patterns = self._load_patterns()
        self.parameter_patterns = self._load_parameter_patterns()
        self.unit_conversion = {
            "m": 1.0,
            "meter": 1.0,
            "meters": 1.0,
            "\u7c73": 1.0,
            "km": 1000.0,
            "kilometer": 1000.0,
            "kilometers": 1000.0,
            "\u5343\u7c73": 1000.0,
        }

    def _load_patterns(self) -> Dict[str, Dict[str, List[str]]]:
        return {
            TaskType.SCALE_GENERALIZATION.value: {
                "keywords": [
                    "\u7efc\u5408",
                    "\u5730\u56fe\u7efc\u5408",
                    "\u5236\u56fe\u7efc\u5408",
                    "\u6bd4\u4f8b\u5c3a",
                    "\u4efb\u610f\u6bd4\u4f8b\u5c3a",
                    "1:",
                    "scale",
                    "generalization",
                ],
                "algorithms": ["rule-based-generalization"],
                "patterns": [
                    r"1\s*[:\uff1a]\s*\d{4,9}",
                    r"(\u5730\u56fe)?\u7efc\u5408.*1\s*[:\uff1a]\s*\d{4,9}",
                    r"\u5f53\u524d\u56fe\u5c42.*\u7efc\u5408",
                ],
            },
            TaskType.LINE_SIMPLIFICATION.value: {
                "keywords": ["\u7b80\u5316", "\u5e73\u6ed1", "\u7ebf", "\u9053\u8def", "\u6cb3\u6d41", "simplify", "smooth"],
                "algorithms": ["douglas-peucker", "douglas", "visvalingam", "chaikin"],
                "patterns": [r"\u7b80\u5316.*\u7ebf", r"\u9053\u8def.*\u7b80\u5316", r"douglas[-_ ]?peucker", r"\u7ebf.*\u5e73\u6ed1"],
            },
            TaskType.POLYGON_GENERALIZATION.value: {
                "keywords": ["\u5408\u5e76", "\u9762", "\u5efa\u7b51", "\u5730\u5757", "dissolve"],
                "algorithms": ["dissolve", "simplify", "eliminate-holes"],
                "patterns": [r"\u5408\u5e76.*\u9762", r"\u5efa\u7b51.*\u5408\u5e76", r"dissolve", r"\u6d88\u9664.*\u788e\u7247"],
            },
            TaskType.SPATIAL_ANALYSIS.value: {
                "keywords": ["\u7f13\u51b2\u533a", "\u53e0\u52a0", "\u88c1\u526a", "\u7a7a\u95f4\u8fde\u63a5", "buffer", "clip", "overlay", "join"],
                "algorithms": ["buffer", "clip", "overlay", "spatial-join"],
                "patterns": [r"\u7f13\u51b2\u533a", r"buffer", r"\u88c1\u526a", r"\u53e0\u52a0.*\u5206\u6790", r"\u7a7a\u95f4.*\u8fde\u63a5"],
            },
            TaskType.COORDINATE_TRANSFORM.value: {
                "keywords": ["\u8f6c\u6362", "\u5750\u6807\u7cfb", "\u6295\u5f71", "epsg", "wgs84", "utm", "reproject"],
                "algorithms": ["reproject"],
                "patterns": [r"\u8f6c\u6362.*\u5750\u6807", r"\u5750\u6807.*\u8f6c\u6362", r"epsg[:=]?\s*\d+", r"wgs84.*utm"],
            },
            TaskType.TOPOLOGY_REPAIR.value: {
                "keywords": ["\u62d3\u6251", "\u4fee\u590d", "\u81ea\u76f8\u4ea4", "\u91cd\u590d", "\u65e0\u6548\u51e0\u4f55", "topology"],
                "algorithms": ["fix-geometry", "remove-duplicates"],
                "patterns": [r"\u62d3\u6251.*\u4fee\u590d", r"\u4fee\u590d.*\u62d3\u6251", r"\u81ea\u76f8\u4ea4", r"\u65e0\u6548.*\u51e0\u4f55"],
            },
        }

    def _load_parameter_patterns(self) -> Dict[str, str]:
        return {
            "target_scale": r"1\s*[:\uff1a]\s*(\d{2,9})",
            "scale_pair": r"(?:\u4ece|from)\s*1\s*[:\uff1a]\s*(\d{2,9}).*?(?:\u5230|to|\-|~|->)\s*1\s*[:\uff1a]\s*(\d{2,9})",
            "source_scale": r"(?:\u6e90\u6bd4\u4f8b\u5c3a|source\s*scale)\s*[:\uff1a=]?\s*1?\s*[:\uff1a]?\s*(\d{2,9})",
            "tolerance": r"\u5bb9\u5dee[\uff1a:=]?\s*(\d+(?:\.\d+)?)\s*(\u7c73|\u5343\u7c73|m|km)?",
            "distance": r"(\u8ddd\u79bb|\u7f13\u51b2\u533a?)[\uff1a:=]?\s*(\d+(?:\.\d+)?)\s*(\u7c73|\u5343\u7c73|m|km)?",
            "input_file": r"([A-Za-z0-9_\-./\\]+\.(?:shp|geojson|gpkg))",
            "field": r"\u6309\s*([A-Za-z0-9_]+)\s*\u5b57\u6bb5",
            "crs": r"EPSG[\uff1a:=]?\s*(\d+)",
            "preserve_topology": r"(\u4fdd\u6301|\u4e0d\u4fdd\u6301)\u62d3\u6251",
        }

    def classify(self, user_input: str) -> Dict[str, Any]:
        task_type, confidence = self._keyword_match(user_input)
        parameters = self._extract_parameters(user_input, task_type)
        algorithm = self._detect_algorithm(user_input, task_type)
        if algorithm:
            parameters["algorithm"] = algorithm

        if confidence <= 0.8:
            llm_result = self._llm_classify(user_input)
            if llm_result.get("task_type") and llm_result["task_type"] != TaskType.UNKNOWN.value:
                task_type = llm_result["task_type"]
                confidence = max(confidence, float(llm_result.get("confidence", 0.75)))
                llm_params = llm_result.get("parameters", {})
                if isinstance(llm_params, dict):
                    parameters = {**parameters, **llm_params}
                if llm_result.get("algorithm") and "algorithm" not in parameters:
                    parameters["algorithm"] = llm_result["algorithm"]
                method = "llm_classify"
            else:
                method = "keyword_match"
        else:
            method = "keyword_match"

        if "input_file" in parameters and "output_file" not in parameters:
            stem = re.sub(r"\.(shp|geojson|gpkg)$", "", str(parameters["input_file"]), flags=re.I)
            parameters["output_file"] = f"{stem}_result.geojson"
        parameters.setdefault("layer_name", "result_layer")

        return {
            "task_type": task_type,
            "algorithm": parameters.get("algorithm", ""),
            "parameters": parameters,
            "confidence": round(float(confidence), 3),
            "method": method,
            "raw_input": user_input,
            "description": user_input,
            "keywords": self._extract_keywords(user_input),
        }

    def _keyword_match(self, user_input: str) -> Tuple[str, float]:
        text = user_input.lower()
        best_type = TaskType.UNKNOWN.value
        best_score = 0.0

        for task_type, bucket in self.task_patterns.items():
            score = 0.0
            for kw in bucket.get("keywords", []):
                if kw.lower() in text:
                    score += 0.2
            for algo in bucket.get("algorithms", []):
                if algo.lower() in text:
                    score += 0.3
            for pattern in bucket.get("patterns", []):
                if re.search(pattern, user_input, flags=re.I):
                    score += 0.5
            if score > best_score:
                best_score = score
                best_type = task_type

        return best_type, min(best_score, 1.0)

    def _extract_parameters(self, user_input: str, task_type: str) -> Dict[str, Any]:
        text = user_input
        params: Dict[str, Any] = {}

        m_pair = re.search(self.parameter_patterns["scale_pair"], text, flags=re.I)
        if m_pair:
            params["source_scale"] = int(m_pair.group(1))
            params["target_scale"] = int(m_pair.group(2))

        m_scale = re.search(self.parameter_patterns["target_scale"], text, flags=re.I)
        if m_scale and "target_scale" not in params:
            params["target_scale"] = int(m_scale.group(1))
        m_src = re.search(self.parameter_patterns["source_scale"], text, flags=re.I)
        if m_src:
            params["source_scale"] = int(m_src.group(1))
        if re.search(r"\u5f53\u524d\u56fe\u5c42|\u5f53\u524d\u5c42|current layer|this layer", text, flags=re.I):
            params["use_current_layer"] = True
        if re.search(r"\u53ea.*geojson|only\s+geojson", text, flags=re.I):
            params["output_formats"] = ["geojson"]
        elif re.search(r"\u53ea.*\u56fe\u7247|only\s+image|only\s+png", text, flags=re.I):
            params["output_formats"] = ["png"]

        files = re.findall(self.parameter_patterns["input_file"], text, flags=re.I)
        if files:
            params["input_file"] = files[0]
            if len(files) > 1:
                params["output_file"] = files[1]

        m_tol = re.search(self.parameter_patterns["tolerance"], text, flags=re.I)
        if m_tol:
            value = float(m_tol.group(1))
            unit = (m_tol.group(2) or "m").lower()
            factor = self.unit_conversion.get(unit, 1.0)
            params["tolerance"] = value * factor
            params["unit"] = "meter"

        m_dist = re.search(self.parameter_patterns["distance"], text, flags=re.I)
        if m_dist:
            value = float(m_dist.group(2))
            unit = (m_dist.group(3) or "m").lower()
            factor = self.unit_conversion.get(unit, 1.0)
            params["distance"] = value * factor
            params["unit"] = "meter"

        m_field = re.search(self.parameter_patterns["field"], text, flags=re.I)
        if m_field:
            params["field"] = m_field.group(1)
            params["by_field"] = m_field.group(1)

        m_crs = re.findall(self.parameter_patterns["crs"], text, flags=re.I)
        if m_crs:
            params["target_crs"] = f"EPSG:{m_crs[-1]}"

        m_topo = re.search(self.parameter_patterns["preserve_topology"], text, flags=re.I)
        if m_topo:
            params["preserve_topology"] = m_topo.group(1) == "\u4fdd\u6301"

        if task_type == TaskType.SPATIAL_ANALYSIS.value and "distance" in params and "algorithm" not in params:
            if re.search(r"\u7f13\u51b2|buffer", text, flags=re.I):
                params["algorithm"] = "buffer"

        return params

    def _detect_algorithm(self, user_input: str, task_type: str) -> str:
        text = user_input.lower()
        algo_map = {
            r"generalization|\u5730\u56fe\u7efc\u5408|\u5236\u56fe\u7efc\u5408": "rule-based-generalization",
            r"douglas[-_ ]?peucker|\u9053\u683c\u62c9\u65af": "douglas-peucker",
            r"visvalingam": "visvalingam",
            r"chaikin|\u5e73\u6ed1": "chaikin",
            r"dissolve|\u5408\u5e76": "dissolve",
            r"eliminate[-_ ]?holes|\u6d88\u9664.*\u5b54\u6d1e": "eliminate-holes",
            r"buffer|\u7f13\u51b2": "buffer",
            r"clip|\u88c1\u526a": "clip",
            r"overlay|\u53e0\u52a0": "overlay",
            r"spatial[-_ ]?join|\u7a7a\u95f4.*\u8fde\u63a5": "spatial-join",
            r"reproject|epsg|\u5750\u6807.*\u8f6c\u6362": "reproject",
            r"fix[-_ ]?geometry|\u4fee\u590d.*\u62d3\u6251|\u65e0\u6548.*\u51e0\u4f55": "fix-geometry",
            r"remove[-_ ]?duplicates|\u53bb\u91cd": "remove-duplicates",
        }
        for pattern, algorithm in algo_map.items():
            if re.search(pattern, text, flags=re.I):
                return algorithm

        default_map = {
            TaskType.SCALE_GENERALIZATION.value: "rule-based-generalization",
            TaskType.LINE_SIMPLIFICATION.value: "douglas-peucker",
            TaskType.POLYGON_GENERALIZATION.value: "dissolve",
            TaskType.SPATIAL_ANALYSIS.value: "buffer",
            TaskType.COORDINATE_TRANSFORM.value: "reproject",
            TaskType.TOPOLOGY_REPAIR.value: "fix-geometry",
        }
        return default_map.get(task_type, "")

    def _llm_classify(self, user_input: str) -> Dict[str, Any]:
        if not self.llm:
            return {"task_type": TaskType.UNKNOWN.value, "algorithm": "", "parameters": {}, "confidence": 0.0}

        prompt = (
            "You are a GIS task classifier. Return JSON only.\\n"
            "Input:\\n"
            f"{user_input}\\n\\n"
            "Task types: scale_generalization, line_simplification, polygon_generalization, spatial_analysis, "
            "coordinate_transform, topology_repair, unknown.\\n"
            "Output JSON schema:\\n"
            "{\"task_type\":\"...\",\"algorithm\":\"...\",\"parameters\":{},\"confidence\":0.0}\\n"
        )
        try:
            resp = self.llm.chat([Message("user", prompt)])
            content = (resp.content or "").strip()
            return self._parse_llm_json(content)
        except Exception:
            return {"task_type": TaskType.UNKNOWN.value, "algorithm": "", "parameters": {}, "confidence": 0.0}

    def _parse_llm_json(self, text: str) -> Dict[str, Any]:
        block = text
        m = re.search(r"\{.*\}", text, flags=re.S)
        if m:
            block = m.group(0)
        try:
            payload = json.loads(block)
        except Exception:
            return {"task_type": TaskType.UNKNOWN.value, "algorithm": "", "parameters": {}, "confidence": 0.0}

        task_type = str(payload.get("task_type", TaskType.UNKNOWN.value))
        valid_types = {t.value for t in TaskType}
        if task_type not in valid_types:
            task_type = TaskType.UNKNOWN.value

        confidence = payload.get("confidence", 0.0)
        try:
            confidence = float(confidence)
        except Exception:
            confidence = 0.0

        parameters = payload.get("parameters", {})
        if not isinstance(parameters, dict):
            parameters = {}

        return {
            "task_type": task_type,
            "algorithm": str(payload.get("algorithm", "")),
            "parameters": parameters,
            "confidence": max(0.0, min(confidence, 1.0)),
        }

    def _extract_keywords(self, user_input: str) -> List[str]:
        tokens = re.findall(r"[A-Za-z_]+|[\u4e00-\u9fff]{2,}", user_input.lower())
        return [t for t in tokens if len(t) > 1][:12]
