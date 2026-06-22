from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class PlanningRequest:
    source_scale: int
    target_scale: int
    layer_stats: Dict[str, Any]
    qa_metrics: Dict[str, Any]
    round_index: int = 1
    last_fail_reason: str = ""


class GeneralizationPlanner:
    """
    Rule-based planner for arbitrary-scale map generalization.

    This is intentionally deterministic so it can be called by an LLM agent
    and audited by a VLM/verifier pipeline.
    """

    def __init__(self, rules_path: str = "knowledge_base/generalization_rules.yaml"):
        self.rules_path = Path(rules_path)
        self.rules: Dict[str, Any] = self._load_rules(self.rules_path)

    def _load_rules(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"Rules file not found: {path}")
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Rules config must be a mapping")
        return payload

    def _interpolate(self, anchors_map: Dict[Any, Any], target_scale: int) -> float:
        if not anchors_map:
            return 0.0
        pairs = sorted((int(k), float(v)) for k, v in anchors_map.items())
        if len(pairs) == 1:
            return pairs[0][1]
        if target_scale <= pairs[0][0]:
            (s0, v0), (s1, v1) = pairs[0], pairs[1]
            slope = (v1 - v0) / float(s1 - s0)
            return v0 + slope * float(target_scale - s0)
        if target_scale >= pairs[-1][0]:
            (s0, v0), (s1, v1) = pairs[-2], pairs[-1]
            slope = (v1 - v0) / float(s1 - s0)
            return v1 + slope * float(target_scale - s1)
        for (s0, v0), (s1, v1) in zip(pairs, pairs[1:]):
            if s0 <= target_scale <= s1:
                ratio = (target_scale - s0) / float(s1 - s0)
                return v0 + (v1 - v0) * ratio
        return pairs[-1][1]

    def _map_mm_to_ground(self, mm_value: float, target_scale: int) -> float:
        return mm_value * target_scale / 1000.0

    def _get_geometry_type(self, layer_stats: Dict[str, Any]) -> str:
        geom = str(layer_stats.get("geometry_type", "line")).strip().lower()
        if geom in {"point", "line", "polygon"}:
            return geom
        if "poly" in geom:
            return "polygon"
        if "point" in geom:
            return "point"
        return "line"

    def _get_density_level(self, layer_stats: Dict[str, Any], qa_metrics: Dict[str, Any]) -> str:
        explicit = str(layer_stats.get("density_level", "")).strip().lower()
        if explicit in {"low", "medium", "high"}:
            return explicit
        conflict_density = float(qa_metrics.get("conflict_density_per_km2", 0.0) or 0.0)
        if conflict_density >= 30:
            return "high"
        if conflict_density >= 10:
            return "medium"
        return "low"

    def _derive_constraints(self, target_scale: int) -> Dict[str, float]:
        symbol_mm = self.rules.get("scale", {}).get("symbol_mm", {})
        min_len_mm = float(symbol_mm.get("min_length", 0.35))
        min_area_mm2 = float(symbol_mm.get("min_area", 1.0))
        min_sep_mm = float(symbol_mm.get("min_separation", 0.30))
        return {
            "min_length_m": round(self._map_mm_to_ground(min_len_mm, target_scale), 3),
            "min_area_m2": round(min_area_mm2 * (target_scale / 1000.0) ** 2, 3),
            "min_separation_m": round(self._map_mm_to_ground(min_sep_mm, target_scale), 3),
        }

    def _build_operators(
        self,
        geometry_type: str,
        density_level: str,
        target_scale: int,
        constraints: Dict[str, float],
        layer_stats: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        params_cfg = self.rules.get("parameters", {})
        density_scale = float(self.rules.get("density_multipliers", {}).get(density_level, 1.0))
        operators: List[Dict[str, Any]] = []

        # 1) selection / elimination
        if geometry_type == "polygon":
            min_area_cfg = params_cfg.get("polygon", {}).get("min_area_m2", {})
            min_area = self._interpolate(min_area_cfg, target_scale) * density_scale
            operators.append({"name": "selection", "params": {"min_area_m2": round(min_area, 3)}})
        elif geometry_type == "point":
            structure_type = str(layer_stats.get("structure_type", "mixed")).strip().lower()
            semantic_type = str(layer_stats.get("semantic_type", "generic_point")).strip().lower()
            keep_cfg = params_cfg.get("point", {}).get("typify_keep_ratio", {})
            keep_ratio = self._interpolate(keep_cfg, target_scale) / density_scale
            min_keep_ratio = float(self.rules.get("thresholds", {}).get("point_min_keep_ratio", 0.1) or 0.1)
            structure_floor_cfg = self.rules.get("thresholds", {}).get("point_structure_min_keep_ratio", {})
            structure_floor = float(structure_floor_cfg.get(structure_type, 0.0) or 0.0)
            semantic_floor = 0.0
            if semantic_type == "transport_network_points":
                semantic_floor = float(
                    self.rules.get("thresholds", {}).get("point_transport_min_keep_ratio", 0.22) or 0.22
                )
            elif semantic_type == "commercial_poi_points":
                semantic_floor = float(
                    self.rules.get("thresholds", {}).get("point_commercial_min_keep_ratio", 0.20) or 0.20
                )
            scale_floor_cfg = self.rules.get("thresholds", {}).get("point_scale_min_keep_ratio", {})
            scale_floor = self._interpolate(scale_floor_cfg, target_scale) if isinstance(scale_floor_cfg, dict) else 0.0
            keep_ratio = max(min_keep_ratio, structure_floor, semantic_floor, scale_floor, min(1.0, keep_ratio))
            operators.append(
                {
                    "name": "selection",
                    "params": {
                        "keep_ratio": round(keep_ratio, 3),
                        "structure_type": structure_type,
                        "semantic_type": semantic_type,
                    },
                }
            )
        else:
            operators.append(
                {
                    "name": "selection",
                    "params": {"min_segment_m": round(constraints["min_length_m"] * density_scale, 3)},
                }
            )

        # 2) simplify
        simplify_key = params_cfg.get(geometry_type, {}).get("simplify_tolerance_m", {})
        simplify_tol = self._interpolate(simplify_key, target_scale) * density_scale
        operators.append(
            {
                "name": "simplify",
                "params": {
                    "tolerance_m": round(simplify_tol, 3),
                    "preserve_topology": True,
                },
            }
        )

        # 3) aggregate/merge where applicable
        if geometry_type == "polygon":
            operators.append(
                {
                    "name": "aggregate",
                    "params": {"distance_m": round(constraints["min_separation_m"] * 0.8, 3)},
                }
            )
        elif geometry_type == "line":
            operators.append(
                {
                    "name": "merge",
                    "params": {"max_gap_m": round(constraints["min_separation_m"] * 0.6, 3)},
                }
            )
        else:
            structure_type = str(layer_stats.get("structure_type", "mixed")).strip().lower()
            pattern = {
                "corridor": "axis_balanced",
                "uniform": "spatial_even",
                "clustered": "cluster_representative",
                "center-periphery": "center_periphery_balanced",
                "multi-core": "multi_core_balanced",
            }.get(structure_type, "spatial_even")
            operators.append({"name": "typification", "params": {"pattern": pattern}})

        # 4) displacement
        max_shift_cfg = params_cfg.get(geometry_type, {}).get("displacement_max_shift_m", {})
        if max_shift_cfg:
            max_shift = self._interpolate(max_shift_cfg, target_scale) * density_scale
        else:
            max_shift = constraints["min_separation_m"] * 1.8
        operators.append(
            {
                "name": "displacement",
                "params": {
                    "min_sep_m": constraints["min_separation_m"],
                    "max_shift_m": round(max_shift, 3),
                    "strict_no_overlap": bool(geometry_type == "point" and target_scale >= 80000),
                    # 参考视口约 2200px：需明显大于符号占位（否则仅满足地面 0.3mm 仍会在屏上叠标）
                    "strict_pixel_gap": (
                        32
                        if (geometry_type == "point" and target_scale >= 200000)
                        else (16 if (geometry_type == "point" and target_scale >= 80000) else 0)
                    ),
                    "strict_density_factor": (
                        0.55
                        if (geometry_type == "point" and target_scale >= 300000)
                        else (0.50 if (geometry_type == "point" and target_scale >= 200000) else 1.0)
                    ),
                },
            }
        )

        # 5) smooth (final polishing, mainly for line/polygon)
        if geometry_type in {"line", "polygon"}:
            operators.append({"name": "smooth", "params": {"method": "chaikin", "iterations": 1}})

        return operators

    def plan(self, req: PlanningRequest) -> Dict[str, Any]:
        feature_count = int(req.layer_stats.get("feature_count", 0) or 0)
        geometry_type = self._get_geometry_type(req.layer_stats)
        density_level = self._get_density_level(req.layer_stats, req.qa_metrics)
        constraints = self._derive_constraints(req.target_scale)
        operators = self._build_operators(
            geometry_type,
            density_level,
            req.target_scale,
            constraints,
            req.layer_stats,
        )
        if geometry_type == "point" and req.round_index > 1:
            fail_text = str(req.last_fail_reason or "")
            conflict_fail = "conflict_count_total" in fail_text
            rgb_overdraw_fail = "point_rgb_hotspot_overdraw" in fail_text
            coverage_fail = any(
                marker in fail_text
                for marker in [
                    "retention_rate",
                    "point_axis_coverage_ratio",
                    "point_hotspot_coverage_ratio",
                    "point_hotspot_rep_count",
                    "point_region_coverage_ratio",
                    "point_sparse_region_coverage_ratio",
                    "point_dense_region_singleton_ratio",
                ]
            )
            if conflict_fail or rgb_overdraw_fail:
                for op in operators:
                    if op.get("name") != "selection":
                        continue
                    params = op.get("params", {}) or {}
                    base_keep = float(params.get("keep_ratio", 1.0) or 1.0)
                    decay_step = 0.12
                    if rgb_overdraw_fail:
                        decay_step = 0.16
                    decay = min(0.75, decay_step * float(req.round_index - 1))
                    params["keep_ratio"] = round(max(0.02, base_keep - decay), 3)
                    break
                for op in operators:
                    if op.get("name") != "displacement":
                        continue
                    params = op.get("params", {}) or {}
                    px = float(params.get("strict_pixel_gap", 0.0) or 0.0)
                    if px > 0:
                        params["strict_pixel_gap"] = round(min(32.0, px + 4.0 * float(req.round_index - 1)), 2)
                    df = float(params.get("strict_density_factor", 1.0) or 1.0)
                    if rgb_overdraw_fail and df < 0.999:
                        params["strict_density_factor"] = round(max(0.08, df * 0.60), 3)
                    break
            elif coverage_fail:
                for op in operators:
                    if op.get("name") != "selection":
                        continue
                    params = op.get("params", {}) or {}
                    base_keep = float(params.get("keep_ratio", 0.2) or 0.2)
                    growth = min(0.35, 0.05 * float(req.round_index - 1))
                    params["keep_ratio"] = round(min(0.70, base_keep + growth), 3)
                    break
                for op in operators:
                    if op.get("name") != "displacement":
                        continue
                    params = op.get("params", {}) or {}
                    px = float(params.get("strict_pixel_gap", 0.0) or 0.0)
                    if px > 0:
                        params["strict_pixel_gap"] = round(max(4.0, px - 2.0 * float(req.round_index - 1)), 2)
                    df = float(params.get("strict_density_factor", 1.0) or 1.0)
                    if df < 0.999:
                        params["strict_density_factor"] = round(min(0.95, df + 0.10 * float(req.round_index - 1)), 3)
                    break

        thresholds = self.rules.get("thresholds", {})
        conflict_floor = int(thresholds.get("conflict_count_floor", 5))
        conflict_ratio = float(thresholds.get("conflict_count_ratio", 0.005))
        target_conflict = max(conflict_floor, int(round(feature_count * conflict_ratio)))
        if geometry_type == "point" and req.target_scale >= 200000:
            target_conflict = 0
        target_vlm_score = int(thresholds.get("vlm_pass_score", 80))
        max_rounds = int(thresholds.get("max_rounds", 4))
        max_middle_gap_ratio = float(thresholds.get("point_max_middle_gap_ratio", 0.35) or 0.35)
        min_axis_coverage_ratio = float(thresholds.get("point_min_axis_coverage_ratio", 0.6) or 0.6)
        min_hotspot_coverage_ratio = float(thresholds.get("point_min_hotspot_coverage_ratio", 0.5) or 0.5)
        min_hotspot_rep_count = int(thresholds.get("point_min_hotspot_rep_count", 3) or 3)
        point_rgb_hotspot_overdraw_max = float(thresholds.get("point_rgb_hotspot_overdraw_max", 0.08) or 0.08)
        point_region_min_coverage_ratio = float(thresholds.get("point_region_min_coverage_ratio", 0.72) or 0.72)
        point_sparse_region_min_coverage_ratio = float(
            thresholds.get("point_sparse_region_min_coverage_ratio", 0.85) or 0.85
        )
        point_dense_region_max_singleton_ratio = float(
            thresholds.get("point_dense_region_max_singleton_ratio", 0.35) or 0.35
        )

        current_conflict = int(req.qa_metrics.get("conflict_count_total", target_conflict * 2) or 0)
        expected_improve = max(0, current_conflict - target_conflict)

        rollback_if = [
            "topology_error_count > 0",
            "retention_rate_by_class.primary_road < 0.98",
        ]

        plan = {
            "round": req.round_index,
            "target_scale": req.target_scale,
            "source_scale": req.source_scale,
            "geometry_type": geometry_type,
            "density_level": density_level,
            "derived_constraints": constraints,
            "operators": operators,
            "expected_delta": {
                "conflict_count_total": -expected_improve,
                "topology_error_count": 0,
            },
            "stop_conditions": {
                "target_conflict_count": target_conflict,
                "target_vlm_score": target_vlm_score,
                "max_rounds": max_rounds,
                "point_max_middle_gap_ratio": max_middle_gap_ratio,
                "point_min_axis_coverage_ratio": min_axis_coverage_ratio,
                "point_min_hotspot_coverage_ratio": min_hotspot_coverage_ratio,
                "point_min_hotspot_rep_count": min_hotspot_rep_count,
                "point_rgb_hotspot_overdraw_max": point_rgb_hotspot_overdraw_max,
                "point_region_min_coverage_ratio": point_region_min_coverage_ratio,
                "point_sparse_region_min_coverage_ratio": point_sparse_region_min_coverage_ratio,
                "point_dense_region_max_singleton_ratio": point_dense_region_max_singleton_ratio,
            },
            "rollback_if": rollback_if,
            "last_fail_reason": req.last_fail_reason,
        }
        return plan

    def plan_from_dict(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        req = PlanningRequest(
            source_scale=self._parse_scale_denominator(payload.get("source_scale", 10000), 10000),
            target_scale=self._parse_scale_denominator(payload.get("target_scale", 25000), 25000),
            layer_stats=dict(payload.get("layer_stats", {}) or {}),
            qa_metrics=dict(payload.get("qa_metrics", {}) or {}),
            round_index=int(payload.get("round", 1)),
            last_fail_reason=str(payload.get("last_fail_reason", "")),
        )
        return self.plan(req)

    @staticmethod
    def _parse_scale_denominator(value: Any, default: int) -> int:
        if value is None or isinstance(value, bool):
            return default
        if isinstance(value, (int, float)):
            return int(value)
        text = str(value).strip().replace(",", "").replace("，", "")
        m = re.search(r"1\s*[:：]\s*(\d{2,9})", text)
        if m:
            return int(m.group(1))
        m = re.search(r"(\d{2,9})", text)
        if m:
            return int(m.group(1))
        return default
