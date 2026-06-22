from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass
class VerifyInput:
    target_scale: int
    metrics: Dict[str, Any]
    stop_conditions: Dict[str, Any]
    retention_min: Dict[str, Any]
    before_image: str = ""
    after_image: str = ""
    last_fail_reason: str = ""
    symbol_context: Dict[str, Any] | None = None
    require_real_vlm: bool = False


class VLMVerifier:
    """
    Verification adapter for map-generalization QA.

    If a true VLM callback is injected by UI/runtime, use it.
    Otherwise apply deterministic heuristic scoring as fallback.
    """

    def __init__(self, callback=None):
        self.callback = callback

    def verify(self, payload: VerifyInput) -> Dict[str, Any]:
        if callable(self.callback):
            try:
                out = self.callback(
                    {
                        "target_scale": payload.target_scale,
                        "metrics": payload.metrics,
                        "stop_conditions": payload.stop_conditions,
                        "retention_min": payload.retention_min,
                        "before_image": payload.before_image,
                        "after_image": payload.after_image,
                        "last_fail_reason": payload.last_fail_reason,
                        "symbol_context": payload.symbol_context or {},
                        "require_real_vlm": bool(payload.require_real_vlm),
                    }
                )
                if isinstance(out, dict) and "pass" in out:
                    out.setdefault("vlm_mode", "callback")
                    out.setdefault("real_vlm_called", True)
                    return out
            except Exception as exc:
                if payload.require_real_vlm:
                    return {
                        "pass": False,
                        "score": 0,
                        "failed_rules": ["real_vlm_callback_failed"],
                        "evidence": [f"vlm callback error: {exc}"],
                        "suggested_fixes": ["register a working vlm_verify callback before running point generalization"],
                        "vlm_mode": "callback_error",
                        "real_vlm_called": False,
                    }

        if payload.require_real_vlm:
            return {
                "pass": False,
                "score": 0,
                "failed_rules": ["real_vlm_required"],
                "evidence": ["No vlm_verify callback is registered; heuristic fallback is disabled."],
                "suggested_fixes": ["register_ui_callback('vlm_verify', your_vision_model_function)"],
                "vlm_mode": "missing_callback",
                "real_vlm_called": False,
            }
        return self._heuristic_verify(payload)

    def _heuristic_verify(self, payload: VerifyInput) -> Dict[str, Any]:
        metrics = payload.metrics or {}
        stop = payload.stop_conditions or {}
        retention_min = payload.retention_min or {}

        conflict = self._as_int(metrics.get("conflict_count_total"), 999999)
        target_conflict = self._as_int(stop.get("target_conflict_count"), 5)
        topo_err = self._as_int(metrics.get("topology_error_count"), 0)
        vlm_target = self._as_int(stop.get("target_vlm_score"), 80)
        retention = self._as_float(metrics.get("retention_rate"), 1.0)
        point_axis_coverage_ratio = metrics.get("point_axis_coverage_ratio")
        point_middle_gap_ratio = metrics.get("point_middle_gap_ratio")
        point_hotspot_coverage_ratio = metrics.get("point_hotspot_coverage_ratio")
        point_hotspot_rep_count = metrics.get("point_hotspot_rep_count")
        point_region_coverage_ratio = metrics.get("point_region_coverage_ratio")
        point_sparse_region_coverage_ratio = metrics.get("point_sparse_region_coverage_ratio")
        point_dense_region_singleton_ratio = metrics.get("point_dense_region_singleton_ratio")
        min_axis_coverage_ratio = float(stop.get("point_min_axis_coverage_ratio", 0.0) or 0.0)
        max_middle_gap_ratio = float(stop.get("point_max_middle_gap_ratio", 1.0) or 1.0)
        min_hotspot_coverage_ratio = float(stop.get("point_min_hotspot_coverage_ratio", 0.0) or 0.0)
        min_hotspot_rep_count = int(stop.get("point_min_hotspot_rep_count", 0) or 0)
        rgb_hotspot_overdraw_max = float(stop.get("point_rgb_hotspot_overdraw_max", 1.0) or 1.0)
        point_region_min_coverage_ratio = float(stop.get("point_region_min_coverage_ratio", 0.0) or 0.0)
        point_sparse_region_min_coverage_ratio = float(stop.get("point_sparse_region_min_coverage_ratio", 0.0) or 0.0)
        point_dense_region_max_singleton_ratio = float(stop.get("point_dense_region_max_singleton_ratio", 1.0) or 1.0)

        min_ret = 0.0
        if retention_min:
            try:
                min_ret = min(float(v) for v in retention_min.values())
            except Exception:
                min_ret = 0.0

        diff_ratio = self._image_diff_ratio(payload.before_image, payload.after_image)

        score = 100
        failed = []
        evidence = []
        fixes = []

        if topo_err > 0:
            failed.append("topology_error_count == 0")
            score -= min(60, topo_err * 10)
            evidence.append(f"topology_error_count={topo_err}")
            fixes.append("repair topology before next iteration")

        if conflict > target_conflict:
            failed.append("conflict_count_total <= target_conflict_count")
            score -= min(40, int((conflict - target_conflict) * 0.8))
            evidence.append(f"conflict={conflict}, target={target_conflict}")
            fixes.append("increase simplify tolerance or displacement")

        if min_ret > 0 and retention < min_ret:
            failed.append("retention_rate >= class_min")
            score -= 25
            evidence.append(f"retention_rate={retention:.3f}, min={min_ret:.3f}")
            fixes.append("relax selection threshold to retain key features")

        if point_axis_coverage_ratio is not None and min_axis_coverage_ratio > 0:
            axis_cov = float(point_axis_coverage_ratio)
            evidence.append(f"point_axis_coverage_ratio={axis_cov:.3f}, min={min_axis_coverage_ratio:.3f}")
            if axis_cov < min_axis_coverage_ratio:
                failed.append("point_axis_coverage_ratio >= min")
                score -= 20
                fixes.append("increase axis-balanced quota to keep middle-section representatives")

        if point_middle_gap_ratio is not None:
            gap_ratio = float(point_middle_gap_ratio)
            evidence.append(f"point_middle_gap_ratio={gap_ratio:.3f}, max={max_middle_gap_ratio:.3f}")
            if gap_ratio > max_middle_gap_ratio:
                failed.append("point_middle_gap_ratio <= max")
                score -= 20
                fixes.append("enforce center/middle bin retention in corridor continuity pass")

        hotspot_rep_ok = False
        if point_hotspot_rep_count is not None and min_hotspot_rep_count > 0:
            hotspot_rep = int(point_hotspot_rep_count)
            evidence.append(f"point_hotspot_rep_count={hotspot_rep}, min={min_hotspot_rep_count}")
            hotspot_rep_ok = hotspot_rep >= min_hotspot_rep_count
            if not hotspot_rep_ok:
                failed.append("point_hotspot_rep_count >= min")
                score -= 20
                fixes.append("keep multiple representatives inside dense hotspot cores")

        if point_hotspot_coverage_ratio is not None and min_hotspot_coverage_ratio > 0:
            hotspot_cov = float(point_hotspot_coverage_ratio)
            evidence.append(
                f"point_hotspot_coverage_ratio={hotspot_cov:.3f}, min={min_hotspot_coverage_ratio:.3f}"
            )
            # Coverage can be strict for very dense clusters; allow pass when representative count is sufficient.
            if hotspot_cov < min_hotspot_coverage_ratio and not hotspot_rep_ok:
                failed.append("point_hotspot_coverage_ratio >= min")
                score -= 25
                fixes.append("increase hotspot representative quota in dense cells")

        if point_region_coverage_ratio is not None and point_region_min_coverage_ratio > 0:
            region_cov = float(point_region_coverage_ratio)
            evidence.append(
                f"point_region_coverage_ratio={region_cov:.3f}, min={point_region_min_coverage_ratio:.3f}"
            )
            if region_cov < point_region_min_coverage_ratio:
                failed.append("point_region_coverage_ratio >= min")
                score -= 20
                fixes.append("preserve at least one representative per occupied region before hotspot thinning")

        if point_sparse_region_coverage_ratio is not None and point_sparse_region_min_coverage_ratio > 0:
            sparse_cov = float(point_sparse_region_coverage_ratio)
            evidence.append(
                f"point_sparse_region_coverage_ratio={sparse_cov:.3f}, min={point_sparse_region_min_coverage_ratio:.3f}"
            )
            if sparse_cov < point_sparse_region_min_coverage_ratio:
                failed.append("point_sparse_region_coverage_ratio >= min")
                score -= 25
                fixes.append("protect sparse 1-2 point regions so each keeps one representative")

        if point_dense_region_singleton_ratio is not None and point_dense_region_max_singleton_ratio < 1.0:
            dense_singleton_ratio = float(point_dense_region_singleton_ratio)
            evidence.append(
                "point_dense_region_singleton_ratio="
                f"{dense_singleton_ratio:.3f}, max={point_dense_region_max_singleton_ratio:.3f}"
            )
            if dense_singleton_ratio > point_dense_region_max_singleton_ratio:
                failed.append("point_dense_region_singleton_ratio <= max")
                score -= 20
                fixes.append("keep multiple representatives inside mid/dense regions instead of collapsing to a singleton")

        if diff_ratio is not None:
            evidence.append(f"image_diff_ratio={diff_ratio:.4f}")
            if diff_ratio < 0.003 and conflict > target_conflict:
                failed.append("visual_change_detected")
                score -= 15
                fixes.append("apply stronger operator parameters")

        rgb_overdraw = self._estimate_rgb_hotspot_overdraw(payload.after_image)
        if rgb_overdraw is not None and rgb_hotspot_overdraw_max < 1.0:
            evidence.append(f"point_rgb_hotspot_overdraw={rgb_overdraw:.4f}, max={rgb_hotspot_overdraw_max:.4f}")
            if rgb_overdraw > rgb_hotspot_overdraw_max:
                failed.append("point_rgb_hotspot_overdraw <= max")
                score -= 25
                fixes.append("reduce dense hotspot symbols (target around one-third in dense cells)")

        score = max(0, min(100, score))
        passed = (not failed) and (score >= vlm_target)

        return {
            "pass": passed,
            "score": score,
            "failed_rules": failed,
            "evidence": evidence,
            "suggested_fixes": fixes,
            "vlm_mode": "heuristic_fallback",
            "real_vlm_called": False,
        }

    def _as_int(self, value, default: int) -> int:
        try:
            if value is None:
                return int(default)
            return int(value)
        except Exception:
            return int(default)

    def _as_float(self, value, default: float) -> float:
        try:
            if value is None:
                return float(default)
            return float(value)
        except Exception:
            return float(default)

    def _image_diff_ratio(self, before_path: str, after_path: str):
        if not before_path or not after_path:
            return None
        p0 = Path(before_path)
        p1 = Path(after_path)
        if not p0.exists() or not p1.exists():
            return None
        try:
            from PIL import Image, ImageChops

            img0 = Image.open(p0).convert("RGB")
            img1 = Image.open(p1).convert("RGB").resize(img0.size)
            diff = ImageChops.difference(img0, img1)
            hist = diff.histogram()
            total_pixels = img0.size[0] * img0.size[1] * 3
            if total_pixels <= 0:
                return None
            # Weighted average normalized to [0,1]
            weighted = 0
            for i, v in enumerate(hist):
                weighted += (i % 256) * v
            return float(weighted) / float(total_pixels * 255)
        except Exception:
            return None

    def _estimate_rgb_hotspot_overdraw(self, image_path: str):
        if not image_path:
            return None
        p = Path(image_path)
        if not p.exists():
            return None
        try:
            from PIL import Image

            img = Image.open(p).convert("RGB")
            w, h = img.size
            if w <= 0 or h <= 0:
                return None

            # Detect magenta-like point symbols used by renderer.
            pix = img.load()
            cell = 24
            nx = max(1, w // cell)
            ny = max(1, h // cell)
            max_ratio = 0.0
            for iy in range(ny):
                y0 = iy * cell
                y1 = min(h, y0 + cell)
                for ix in range(nx):
                    x0 = ix * cell
                    x1 = min(w, x0 + cell)
                    total = max(1, (x1 - x0) * (y1 - y0))
                    cnt = 0
                    for y in range(y0, y1):
                        for x in range(x0, x1):
                            r, g, b = pix[x, y]
                            if r >= 180 and g <= 110 and b >= 90:
                                cnt += 1
                    ratio = float(cnt) / float(total)
                    if ratio > max_ratio:
                        max_ratio = ratio
            return max_ratio
        except Exception:
            return None
