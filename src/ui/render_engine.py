from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from src.ui.point_symbol_core import normalize_point_symbol

_GEOS_LOCK = threading.Lock()

try:
    from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal
except ImportError:
    # Mock for benchmark environment without PySide6
    class QObject:  # type: ignore
        def __init__(self, parent=None): pass
    
    class QRunnable: pass  # type: ignore
    
    class QThreadPool:  # type: ignore
        @staticmethod
        def globalInstance(): return QThreadPool()
        def setMaxThreadCount(self, c): pass
        def start(self, w): w.run()
        
    class Signal:  # type: ignore
        def __init__(self, *args): pass
        def emit(self, *args): pass
        def connect(self, f): self._slot = f
        
    class _SignalInstance:
        def emit(self, *args): 
            if hasattr(self, '_slot'): self._slot(*args)
        def connect(self, f): self._slot = f
        
    def Signal(*args): return _SignalInstance()


@dataclass
class LayerRenderTask:
    name: str
    z_index: int
    feature_count: int
    visible: bool
    gdf: Any
    geom_type: str
    style: Dict[str, Any]


class LayerRenderScheduler:
    """Layer scheduling with stable and predictable priority."""

    def order(self, tasks: List[LayerRenderTask]) -> List[LayerRenderTask]:
        visible = [t for t in tasks if t.visible]
        visible.sort(
            key=lambda t: (
                t.z_index,  # keep visual stacking first
                0 if t.feature_count <= 5000 else 1,
                t.feature_count,
            )
        )
        return visible


class RenderProfiler:
    """Simple profiler for render stages."""

    def __init__(self) -> None:
        self._marks: Dict[str, float] = {}
        self._durations_ms: Dict[str, float] = {}

    def start(self, name: str) -> None:
        self._marks[name] = time.perf_counter()

    def end(self, name: str) -> None:
        started = self._marks.pop(name, None)
        if started is None:
            return
        self._durations_ms[name] = (time.perf_counter() - started) * 1000.0

    def report(self) -> Dict[str, float]:
        return dict(self._durations_ms)

    def reset(self) -> None:
        self._marks.clear()
        self._durations_ms.clear()


class MemoryMonitor:
    """Best-effort process memory snapshot."""

    def __init__(self) -> None:
        self.peak_mb = 0.0
        self.last_mb = 0.0

    def snapshot_mb(self) -> float:
        mb = 0.0
        try:
            import psutil  # type: ignore

            proc = psutil.Process()
            mb = proc.memory_info().rss / (1024 * 1024)
        except Exception:
            mb = 0.0
        self.last_mb = mb
        self.peak_mb = max(self.peak_mb, mb)
        return mb


class _WorkerSignals(QObject):
    finished = Signal(object)


class _RenderWorker(QRunnable):
    def __init__(
        self,
        payload: Dict[str, Any],
        cancel_check,
    ) -> None:
        super().__init__()
        self.payload = payload
        self.cancel_check = cancel_check
        self.signals = _WorkerSignals()

    @staticmethod
    def _calc_lod(feature_count: int, zoom_level: float, geom_type: str) -> Tuple[float, float]:
        full_render_threshold = 8000
        if "Point" in geom_type:
            full_render_threshold = 20000
        elif "LineString" in geom_type:
            full_render_threshold = 12000

        if feature_count <= full_render_threshold:
            sample_rate = 1.0
            if zoom_level < 50:
                simplify_factor = 2.5
            elif zoom_level < 200:
                simplify_factor = 1.5
            else:
                simplify_factor = 0.8
        elif feature_count > 50000:
            if zoom_level < 50:
                sample_rate = 0.01
                simplify_factor = 5.0
            elif zoom_level < 200:
                sample_rate = 0.05
                simplify_factor = 3.0
            else:
                sample_rate = 0.2
                simplify_factor = 2.0
        elif feature_count > 10000:
            if zoom_level < 50:
                sample_rate = 0.05
                simplify_factor = 4.0
            elif zoom_level < 200:
                sample_rate = 0.2
                simplify_factor = 2.5
            else:
                sample_rate = 0.5
                simplify_factor = 1.5
        elif feature_count > 2000:
            if zoom_level < 50:
                sample_rate = 0.2
                simplify_factor = 3.0
            elif zoom_level < 200:
                sample_rate = 0.5
                simplify_factor = 2.0
            else:
                sample_rate = 1.0
                simplify_factor = 1.0
        else:
            sample_rate = 1.0
            simplify_factor = 0.8
        return sample_rate, simplify_factor

    @staticmethod
    def _to_screen(x: float, y: float, xmin: float, ymax: float, x_scale: float, y_scale: float) -> Tuple[float, float]:
        return (x - xmin) * x_scale, (ymax - y) * y_scale

    def run(self) -> None:
        payload = self.payload
        request_id = payload.get("request_id", -1)
        stage = payload.get("stage", "full")
        try:
            extent = payload["extent"]
            width = payload["width"]
            height = payload["height"]
            quality = payload["quality"]
            layers = payload["layers"]
            cache_key = payload.get("cache_key")

            xmin, ymin, xmax, ymax = extent
            if width <= 0 or height <= 0 or xmax == xmin or ymax == ymin:
                self.signals.finished.emit({"request_id": request_id, "stage": stage, "canceled": True})
                return

            x_scale = width / (xmax - xmin)
            y_scale = height / (ymax - ymin)
            # Pre-calculate offset for faster transform: sx = (x - xmin) * x_scale = x * x_scale - xmin * x_scale
            x_offset = -xmin * x_scale
            # sy = (ymax - y) * y_scale = ymax * y_scale - y * y_scale
            y_offset = ymax * y_scale

            geo_per_pixel = (xmax - xmin) / width if width > 0 else 1.0
            zoom_level = 360 / (xmax - xmin) if (xmax - xmin) > 0 else 1.0

            render_layers: List[Dict[str, Any]] = []
            total_features = 0
            total_visible = 0
            total_rendered = 0

            for layer in layers:
                if self.cancel_check(request_id):
                    self.signals.finished.emit({"request_id": request_id, "stage": stage, "canceled": True})
                    return

                gdf = layer.gdf
                try:
                    # Safe BBox filtering using NumPy (avoids rtree/libspatialindex in thread)
                    # bounds is a DataFrame with minx, miny, maxx, maxy
                    bounds = gdf.bounds
                    mask = (bounds.minx < xmax) & (bounds.maxx > xmin) & \
                           (bounds.miny < ymax) & (bounds.maxy > ymin)
                    visible_gdf = gdf[mask]
                except Exception:
                    # Fallback to full dataset if bounds calculation fails
                    visible_gdf = gdf

                if visible_gdf.empty:
                    continue

                feature_count = len(visible_gdf)
                total_features += len(gdf)
                total_visible += feature_count

                sample_rate, simplify_factor = self._calc_lod(feature_count, zoom_level, layer.geom_type)
                sample_rate *= float(quality.get("sample_multiplier", 1.0))
                sample_rate = max(0.01, min(1.0, sample_rate))
                simplify_factor *= float(quality.get("simplify_multiplier", 1.0))
                simplify_tolerance = geo_per_pixel * simplify_factor

                if sample_rate < 1.0 and feature_count > 0:
                    sample_count = max(int(feature_count * sample_rate), 50)
                    step = max(1, feature_count // sample_count)
                    visible_gdf = visible_gdf.iloc[::step]
                    feature_count = len(visible_gdf)

                total_rendered += feature_count
                geom_type = layer.geom_type or ""

                # --- Optimized Processing (Safe Mode) ---
                # 1. Vectorized Simplification (Fast & Reliable)
                # --- Optimized Processing (Safe Mode) ---
                # 1. Vectorized Simplification (Fast & Reliable)
                # PERMANENTLY DISABLED: GEOS simplify causes Segfault in QThreadPool on Windows
                # regardless of locking. Stability priority > Performance.
                # try:
                #     if simplify_tolerance > 0 and "Point" not in geom_type:
                #         with _GEOS_LOCK:
                #             visible_gdf = visible_gdf.copy()
                #             visible_gdf.geometry = visible_gdf.geometry.simplify(simplify_tolerance, preserve_topology=False)
                #             visible_gdf = visible_gdf[~visible_gdf.geometry.is_empty]
                # except Exception:
                #     pass

                points: List[Tuple[float, ...]] = []
                lines: List[List[Tuple[float, float]]] = []
                polygons: List[List[Tuple[float, float]]] = []

                # 2. Optimized Iteration (Avoid function calls in loop)
                # Vertex decimation based on LOD (replaces risky GEOS simplify)
                vertex_step = max(1, int(simplify_factor))

                def _point_symbol_series(_gdf: Any, _style: Dict[str, Any]) -> Any:
                    import pandas as pd

                    default_sym = normalize_point_symbol(_style.get("point_symbol"))
                    field = str(_style.get("point_symbol_field") or "").strip()
                    mmap_raw = _style.get("point_symbol_map")
                    if not isinstance(mmap_raw, dict):
                        mmap_raw = {}
                    str_mmap = {str(k): normalize_point_symbol(str(v)) for k, v in mmap_raw.items()}
                    if field and field in _gdf.columns:
                        def _one(v: Any) -> str:
                            k = str(v) if pd.notna(v) else ""
                            if k in str_mmap:
                                return str_mmap[k]
                            return normalize_point_symbol(k) if k else default_sym

                        return _gdf[field].map(_one)
                    return pd.Series([default_sym] * len(_gdf), index=_gdf.index)

                try:
                    sym_series = _point_symbol_series(visible_gdf, layer.style or {})
                    for pos in range(len(visible_gdf)):
                        geom = visible_gdf.geometry.iloc[pos]
                        sym = sym_series.iloc[pos]
                        if geom is None or geom.is_empty:
                            continue

                        g_type = geom.geom_type

                        if g_type == "Point":
                            sx = geom.x * x_scale + x_offset
                            sy = y_offset - geom.y * y_scale
                            points.append((sx, sy, sym))

                        elif g_type == "MultiPoint":
                            for pt in geom.geoms:
                                sx = pt.x * x_scale + x_offset
                                sy = y_offset - pt.y * y_scale
                                points.append((sx, sy, sym))
                                
                        elif g_type == 'LineString':
                            # Apply vertex step and handle 2D/3D coords via .xy
                            try:
                                xs, ys = geom.xy
                                pts = [(x * x_scale + x_offset, y_offset - y * y_scale) for x, y in zip(xs[::vertex_step], ys[::vertex_step])]
                                if len(pts) > 1:
                                    lines.append(pts)
                            except Exception: pass
                                
                        elif g_type == 'MultiLineString':
                            for part in geom.geoms:
                                try:
                                    xs, ys = part.xy
                                    pts = [(x * x_scale + x_offset, y_offset - y * y_scale) for x, y in zip(xs[::vertex_step], ys[::vertex_step])]
                                    if len(pts) > 1:
                                        lines.append(pts)
                                except Exception: pass
                                    
                        elif g_type == 'Polygon':
                            # Exterior ring
                            try:
                                xs, ys = geom.exterior.xy
                                pts = [(x * x_scale + x_offset, y_offset - y * y_scale) for x, y in zip(xs[::vertex_step], ys[::vertex_step])]
                                if len(pts) > 2:
                                    polygons.append(pts)
                            except Exception: pass
                                
                        elif g_type == 'MultiPolygon':
                            for part in geom.geoms:
                                try:
                                    xs, ys = part.exterior.xy
                                    pts = [(x * x_scale + x_offset, y_offset - y * y_scale) for x, y in zip(xs[::vertex_step], ys[::vertex_step])]
                                    if len(pts) > 2:
                                        polygons.append(pts)
                                except Exception: pass
                                    
                except Exception as e:
                    # Log error but don't crash the thread (limit logs)
                    # Use a unique key per error type to avoid spamming the same error
                    error_msg = str(e)
                    if error_msg not in getattr(self, "_logged_errors", set()):
                         print(f"Error processing geometry in layer {layer.name}: {e}")
                         if not hasattr(self, "_logged_errors"):
                             self._logged_errors = set()
                         self._logged_errors.add(error_msg)

                render_layers.append(
                    {
                        "name": layer.name,
                        "geom_type": geom_type,
                        "style": layer.style,
                        "points": points,
                        "lines": lines,
                        "polygons": polygons,
                    }
                )

            out = {
                "request_id": request_id,
                "stage": stage,
                "cache_key": cache_key,
                "extent": extent,
                "width": width,
                "height": height,
                "layers": render_layers,
                "lod_stats": {
                    "total_features": total_features,
                    "visible_features": total_visible,
                    "rendered_features": total_rendered,
                    "sample_rate": 0.0 if total_visible == 0 else min(1.0, total_rendered / max(total_visible, 1)),
                },
                "canceled": False,
            }
            self.signals.finished.emit(out)
        except Exception as exc:
            self.signals.finished.emit(
                {
                    "request_id": request_id,
                    "stage": stage,
                    "canceled": False,
                    "error": str(exc),
                    "layers": [],
                    "lod_stats": {"total_features": 0, "visible_features": 0, "rendered_features": 0, "sample_rate": 0.0},
                }
            )


class AsyncRenderEngine(QObject):
    render_ready = Signal(object)

    def __init__(self, max_threads: int = 2, parent: QObject | None = None):
        super().__init__(parent)
        self._pool = QThreadPool.globalInstance()
        self._pool.setMaxThreadCount(max(1, max_threads))
        self._lock = threading.Lock()
        self._latest_request_id = 0

    def submit(self, payload: Dict[str, Any]) -> int:
        with self._lock:
            self._latest_request_id += 1
            rid = self._latest_request_id
        task_payload = dict(payload)
        task_payload["request_id"] = rid
        worker = _RenderWorker(task_payload, self._is_canceled)
        worker.signals.finished.connect(self.render_ready.emit)
        self._pool.start(worker)
        return rid

    def cancel_all(self) -> None:
        with self._lock:
            self._latest_request_id += 1

    def _is_canceled(self, request_id: int) -> bool:
        with self._lock:
            return request_id != self._latest_request_id
