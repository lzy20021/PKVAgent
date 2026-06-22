from typing import List, Tuple, Optional, Dict, Any
from PySide6.QtCore import QRectF, QPointF
from PySide6.QtGui import QImage, QPainter
from shapely.geometry import box, Point, LineString, Polygon
from shapely.strtree import STRtree
import numpy as np


class PerformanceOptimizer:

    def __init__(self):
        self.spatial_index: Optional[STRtree] = None
        self.cache: Dict[str, Any] = {}
        self.cache_size_limit = 100

    def build_spatial_index(self, geometries: List[Any]) -> STRtree:
        self.spatial_index = STRtree(geometries)
        return self.spatial_index

    def query_visible_features(self, bbox: Tuple[float, float, float, float]) -> List[Any]:
        if self.spatial_index is None:
            return []
        query_box = box(*bbox)
        return list(self.spatial_index.query(query_box))

    def simplify_geometry(self, geometry: Any, tolerance: float) -> Any:
        try:
            return geometry.simplify(tolerance, preserve_topology=True)
        except:
            return geometry

    def adaptive_simplify(self, geometry: Any, zoom_level: float,
                         base_tolerance: float = 0.0001) -> Any:
        tolerance = base_tolerance * (20 - zoom_level) / 10
        tolerance = max(0.00001, min(tolerance, 0.01))
        return self.simplify_geometry(geometry, tolerance)

    def clip_geometry(self, geometry: Any, bbox: Tuple[float, float, float, float]) -> Optional[Any]:
        try:
            clip_box = box(*bbox)
            clipped = geometry.intersection(clip_box)
            return clipped if not clipped.is_empty else None
        except:
            return geometry

    def batch_transform_coordinates(self, coords: np.ndarray,
                                   transform_func) -> np.ndarray:
        return np.apply_along_axis(transform_func, 1, coords)

    def cache_get(self, key: str) -> Optional[Any]:
        return self.cache.get(key)

    def cache_set(self, key: str, value: Any):
        if len(self.cache) >= self.cache_size_limit:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        self.cache[key] = value

    def cache_clear(self):
        self.cache.clear()

    def create_tile_buffer(self, width: int, height: int) -> QImage:
        return QImage(width, height, QImage.Format.Format_ARGB32_Premultiplied)

    def lod_sample_features(self, features: List[Any], zoom_level: float,
                           max_features: int = 10000) -> List[Any]:
        if len(features) <= max_features:
            return features

        if zoom_level < 5:
            sample_rate = 0.01
        elif zoom_level < 10:
            sample_rate = 0.1
        elif zoom_level < 15:
            sample_rate = 0.5
        else:
            sample_rate = 1.0

        target_count = int(len(features) * sample_rate)
        target_count = max(100, min(target_count, max_features))

        indices = np.linspace(0, len(features) - 1, target_count, dtype=int)
        return [features[i] for i in indices]

    def estimate_geometry_complexity(self, geometry: Any) -> int:
        if hasattr(geometry, 'coords'):
            return len(list(geometry.coords))
        elif hasattr(geometry, 'exterior'):
            return len(list(geometry.exterior.coords))
        elif hasattr(geometry, 'geoms'):
            return sum(self.estimate_geometry_complexity(g) for g in geometry.geoms)
        return 0

    def should_render_feature(self, geometry: Any, zoom_level: float,
                             min_pixel_size: float = 2.0) -> bool:
        bounds = geometry.bounds
        width = bounds[2] - bounds[0]
        height = bounds[3] - bounds[1]

        pixel_size = max(width, height) * (2 ** zoom_level)
        return pixel_size >= min_pixel_size
