from __future__ import annotations

import csv
import json
import math
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from PySide6.QtCore import QPointF, QRect, QRectF, QSizeF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QBrush, QMouseEvent, QPageSize, QPaintEvent, QPainter, QPdfWriter, QPen, QPixmap, QPolygonF, QResizeEvent, QShowEvent, QWheelEvent
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from src.ui.i18n import tr
from src.ui.point_symbol_core import infer_point_symbol_style_from_gdf
from src.ui.point_symbol_render import draw_point_symbol, normalize_point_symbol
from src.ui.render_engine import AsyncRenderEngine, LayerRenderScheduler, LayerRenderTask, MemoryMonitor, RenderProfiler, _RenderWorker


PAGE_SIZES_MM = {
    "A4": (210.0, 297.0),
    "A3": (297.0, 420.0),
    "A2": (420.0, 594.0),
    "LETTER": (215.9, 279.4),
}

PAGE_SIZE_ORDER = ("A4", "A3", "A2")


class MapWidget(QWidget):
    map_clicked = Signal(float, float)
    mouse_moved = Signal(float, float)
    zoom_changed = Signal(float)
    # 对比模式下人工干预（已更新右侧综合后图层数据）
    compare_intervention = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(100, 100)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent)

        self.layers: Dict[str, Dict[str, Any]] = {}
        self.layer_order: List[str] = []
        self._extent = [73.0, 18.0, 135.0, 54.0]
        self._home_extent = [73.0, 18.0, 135.0, 54.0]
        self._cache: QPixmap | None = None
        self._cache_dpr: float = 1.0
        self._needs_full_redraw = True
        self._pan_start: QPointF | None = None
        self._pan_button: Qt.MouseButton | None = None
        self._pan_offset_x = 0
        self._pan_offset_y = 0
        self._left_press_pos: QPointF | None = None
        self._pan_drag_threshold_px = 5.0
        self._screen_dpr_hooked = False

        # 综合前后对比：左右分屏，竖线可拖动（仅影响分界比例，不重算几何）
        self._compare_enabled = False
        self._compare_layer_before = ""
        self._compare_layer_after = ""
        self._compare_split = 0.5
        self._cache_before: QPixmap | None = None
        self._cache_after: QPixmap | None = None
        self._drag_compare_split = False
        self._compare_divider_hit_px = 10
        # 对比模式下左右各自范围与平移侧（避免两侧绑在同一张“底图”上一起动）
        self._compare_extent_before: List[float] = [73.0, 18.0, 135.0, 54.0]
        self._compare_extent_after: List[float] = [73.0, 18.0, 135.0, 54.0]
        self._compare_pan_side: str | None = None  # "before" | "after"
        self._compare_extent_dirty: bool = False  # 为 True 时下一帧按左右半幅宽高拟合数据范围并居中

        self._lod_stats = {"total_features": 0, "visible_features": 0, "rendered_features": 0, "sample_rate": 1.0}
        self._last_render_profile: Dict[str, float] = {}
        self._perf_history: List[Dict[str, Any]] = []

        self.default_styles = {
            "Point": {"color": "#E91E63", "size": 8},
            "MultiPoint": {"color": "#E91E63", "size": 8},
            "LineString": {"color": "#2196F3", "width": 2},
            "MultiLineString": {"color": "#2196F3", "width": 2},
            "Polygon": {"fill": "#4CAF50", "stroke": "#2E7D32", "fill_opacity": 0.5, "stroke_width": 1},
            "MultiPolygon": {"fill": "#4CAF50", "stroke": "#2E7D32", "fill_opacity": 0.5, "stroke_width": 1},
            "Mixed": {"color": "#2196F3", "width": 2, "size": 8, "fill": "#4CAF50", "stroke": "#2E7D32", "fill_opacity": 0.5, "stroke_width": 1},
        }

        self._render_scheduler = LayerRenderScheduler()
        self._render_profiler = RenderProfiler()
        self._memory_monitor = MemoryMonitor()
        self._async_engine = AsyncRenderEngine(max_threads=2, parent=self)
        self._async_engine.render_ready.connect(self._on_async_render_ready)
        self._render_inflight = False
        self._render_submit_ts = 0.0
        self._latest_request_id = 0
        self._current_stage = "full"

    def _effective_dpr(self) -> float:
        win = self.window()
        wh = win.windowHandle() if win is not None else None
        if wh is not None:
            return max(1.0, float(wh.devicePixelRatio()))
        return max(1.0, float(self.devicePixelRatio()))

    def _render_pixel_size(self) -> Tuple[int, int, float]:
        dpr = self._effective_dpr()
        w, h = max(1, self.width()), max(1, self.height())
        rw = max(1, int(round(w * dpr)))
        rh = max(1, int(round(h * dpr)))
        return rw, rh, dpr

    def _begin_pan(self, anchor: QPointF, button: Qt.MouseButton) -> None:
        self._pan_start = QPointF(anchor)
        self._pan_button = button
        self._left_press_pos = None
        self._async_engine.cancel_all()
        self._render_inflight = False
        if self._compare_ready():
            ax = float(anchor.x())
            self._compare_pan_side = "before" if ax < float(self._compare_divider_x()) else "after"
        else:
            self._compare_pan_side = None
        self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def _commit_pan(self) -> None:
        if self._pan_start is None:
            self._pan_button = None
            self._compare_pan_side = None
            self.setCursor(Qt.CursorShape.ArrowCursor)
            return
        w, h = max(self.width(), 1), max(self.height(), 1)
        if self._compare_ready() and self._compare_pan_side in ("before", "after"):
            div = float(self._compare_divider_x())
            wl = max(1.0, div)
            wr = max(1.0, float(w) - div)
            pw = wl if self._compare_pan_side == "before" else wr
            if self._compare_pan_side == "before":
                xmin, ymin, xmax, ymax = self._compare_extent_before
            else:
                xmin, ymin, xmax, ymax = self._compare_extent_after
            span_x = xmax - xmin
            span_y = ymax - ymin
            # 半幅宽度对应 span_x，不能用整屏 w（否则拖一下范围错位，综合后像“消失”）
            geo_dx = -self._pan_offset_x / pw * span_x
            geo_dy = self._pan_offset_y / float(h) * span_y
            new_ext = [
                xmin + geo_dx,
                ymin + geo_dy,
                xmax + geo_dx,
                ymax + geo_dy,
            ]
            if self._compare_pan_side == "before":
                self._compare_extent_before = new_ext
            else:
                self._compare_extent_after = new_ext
            self._extent = list(self._compare_extent_before)
        else:
            xmin, ymin, xmax, ymax = self._extent
            geo_dx = -self._pan_offset_x / float(w) * (xmax - xmin)
            geo_dy = self._pan_offset_y / float(h) * (ymax - ymin)
            self._extent = [xmin + geo_dx, ymin + geo_dy, xmax + geo_dx, ymax + geo_dy]

        self._pan_offset_x = 0
        self._pan_offset_y = 0
        self._pan_start = None
        self._pan_button = None
        self._compare_pan_side = None
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self._mark_dirty()

    def _pan_still_held(self, event: QMouseEvent) -> bool:
        if self._pan_button is None or self._pan_start is None:
            return False
        b = event.buttons()
        if self._pan_button == Qt.MouseButton.LeftButton:
            return bool(b & Qt.MouseButton.LeftButton)
        if self._pan_button == Qt.MouseButton.MiddleButton:
            return bool(b & Qt.MouseButton.MiddleButton)
        if self._pan_button == Qt.MouseButton.RightButton:
            return bool(b & Qt.MouseButton.RightButton)
        return False

    def _geo_from_screen(self, sx: float, sy: float, extent: List[float]) -> Tuple[float, float]:
        w, h = self.width(), self.height()
        xmin, ymin, xmax, ymax = extent
        if w <= 0 or h <= 0:
            return (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
        x = xmin + sx / w * (xmax - xmin)
        y = ymax - sy / h * (ymax - ymin)
        return x, y

    def _geo_from_screen_compare_panel(self, sx: float, sy: float, extent: List[float], panel: str) -> Tuple[float, float]:
        """对比模式下按左/右半幅宽度换算地理坐标（与渲染时半幅 span 一致）。"""
        w, h = max(1, self.width()), max(1, self.height())
        div = float(self._compare_divider_x())
        wl = max(1.0, div)
        wr = max(1.0, float(w) - div)
        xmin, ymin, xmax, ymax = extent
        span_x = xmax - xmin
        span_y = ymax - ymin
        if panel == "before":
            rel_x = float(sx) / wl
        else:
            rel_x = (float(sx) - div) / wr
        rel_x = max(0.0, min(1.0, rel_x))
        x = xmin + rel_x * span_x
        y = ymax - (float(sy) / float(h)) * span_y
        return x, y

    def _geo_to_screen_compare_panel(self, lon: float, lat: float, extent: List[float], panel: str) -> Tuple[float, float]:
        """与 _geo_from_screen_compare_panel 互逆，用于对比模式下点选命中（屏幕像素，左上为原点）。"""
        w, h = max(1, self.width()), max(1, self.height())
        div = float(self._compare_divider_x())
        wl = max(1.0, div)
        wr = max(1.0, float(w) - div)
        xmin, ymin, xmax, ymax = extent
        span_x = max(1e-12, xmax - xmin)
        span_y = max(1e-12, ymax - ymin)
        rel_x = (float(lon) - xmin) / span_x
        rel_y = (float(ymax) - float(lat)) / span_y
        rel_x = max(0.0, min(1.0, rel_x))
        rel_y = max(0.0, min(1.0, rel_y))
        if panel == "before":
            sx = rel_x * wl
        else:
            sx = div + rel_x * wr
        sy = rel_y * float(h)
        return sx, sy

    def _geo_to_screen(self, lon: float, lat: float, extent: List[float]) -> Tuple[float, float]:
        w, h = max(1, self.width()), max(1, self.height())
        xmin, ymin, xmax, ymax = extent
        span_x = max(1e-12, xmax - xmin)
        span_y = max(1e-12, ymax - ymin)
        sx = (float(lon) - xmin) / span_x * float(w)
        sy = (float(ymax) - float(lat)) / span_y * float(h)
        return sx, sy

    def _ensure_feature_identity(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        if gdf is None or getattr(gdf, "empty", True):
            return gdf
        if "__mapgen_feature_uid" in gdf.columns:
            ids = gdf["__mapgen_feature_uid"].astype(str).fillna("")
            if (ids != "").all():
                return gdf
        out = gdf.copy()
        feature_ids: List[str] = []
        for idx in range(len(out)):
            row = out.iloc[idx]
            uid = ""
            for col in ("__mapgen_feature_uid", "poi_uid", "feature_id", "uid", "id"):
                if col in out.columns:
                    val = row.get(col)
                    if pd.notna(val):
                        uid = str(val).strip()
                    if uid:
                        break
            if not uid:
                geom = row.geometry
                gx = gy = 0.0
                if geom is not None and not geom.is_empty:
                    if geom.geom_type == "Point":
                        gx, gy = float(geom.x), float(geom.y)
                    elif geom.geom_type == "MultiPoint" and len(getattr(geom, "geoms", [])) > 0:
                        g0 = geom.geoms[0]
                        gx, gy = float(g0.x), float(g0.y)
                name = str(row.get("name") or row.get("poi_name") or "").strip()
                category = str(row.get("poi_category") or row.get("query_category") or row.get("type") or "").strip()
                symbol = str(row.get("point_symbol") or "").strip()
                uid = f"auto:{idx}:{gx:.7f}:{gy:.7f}:{name}:{category}:{symbol}"
            feature_ids.append(uid)
        out["__mapgen_feature_uid"] = feature_ids
        return out

    def _point_symbol_series_for_gdf(self, gdf: gpd.GeoDataFrame, style: Dict[str, Any]) -> pd.Series:
        default_sym = normalize_point_symbol(style.get("point_symbol"))
        field = str(style.get("point_symbol_field") or "").strip()
        mmap_raw = style.get("point_symbol_map")
        if not isinstance(mmap_raw, dict):
            mmap_raw = {}
        str_mmap = {str(k): normalize_point_symbol(str(v)) for k, v in mmap_raw.items()}
        if field and field in gdf.columns:
            def _one(v: Any) -> str:
                k = str(v) if pd.notna(v) else ""
                if k in str_mmap:
                    return str_mmap[k]
                return normalize_point_symbol(k) if k else default_sym

            return gdf[field].map(_one)
        return pd.Series([default_sym] * len(gdf), index=gdf.index)

    def _point_radius_px_from_style(self, style: Dict[str, Any] | None) -> float:
        st = style or {}
        radius_val = st.get("radius")
        if radius_val is not None:
            try:
                return max(2.0, float(radius_val))
            except Exception:
                pass
        try:
            return max(2.0, float(st.get("size", 8)) / 2.0)
        except Exception:
            return 4.0

    def _refresh_point_symbol_style_for_layer(self, layer_name: str) -> None:
        entry = self.layers.get(layer_name)
        if not entry:
            return
        geom_type = str(entry.get("geom_type", "") or "")
        if geom_type not in ("Point", "MultiPoint"):
            return
        gdf = entry.get("gdf")
        if gdf is None or getattr(gdf, "empty", True):
            return
        current = dict(entry.get("style", {}) or {})
        auto = infer_point_symbol_style_from_gdf(gdf)
        if auto:
            current["point_symbol_field"] = auto["point_symbol_field"]
            current["point_symbol_map"] = auto.get("point_symbol_map") or {}
            if "point_symbol" in current:
                current.pop("point_symbol", None)
        entry["style"] = current

    def _materialize_point_symbols_for_layer(self, layer_name: str) -> None:
        entry = self.layers.get(layer_name)
        if not entry:
            return
        geom_type = str(entry.get("geom_type", "") or "")
        if geom_type not in ("Point", "MultiPoint"):
            return
        gdf = entry.get("gdf")
        if gdf is None or getattr(gdf, "empty", True):
            return
        style = dict(entry.get("style", {}) or {})
        try:
            sym_series = self._point_symbol_series_for_gdf(gdf, style)
        except Exception:
            return
        out = gdf.copy()
        out["point_symbol"] = sym_series.astype(str)
        entry["gdf"] = out
        style["point_symbol_field"] = "point_symbol"
        style["point_symbol_map"] = {sym: sym for sym in sorted(set(out["point_symbol"].astype(str).tolist())) if sym}
        style.pop("point_symbol", None)
        entry["style"] = style

    def _screen_conflict_priority(self, row: pd.Series) -> float:
        score = 0.0
        for col, weight in (
            ("__mapgen_user_pick", 1_000_000.0),
            ("__protected_keep", 100_000.0),
            ("guaranteed", 10_000.0),
            ("importance_score", 1_000.0),
            ("__sel_score", 1_000.0),
            ("score", 100.0),
        ):
            try:
                score += float(row.get(col, 0.0) or 0.0) * weight
            except Exception:
                continue
        return score

    def _iter_visible_point_screen_records(self, layer_name: str, panel: str) -> List[Dict[str, Any]]:
        entry = self.layers.get(layer_name)
        if not entry:
            return []
        gdf = entry.get("gdf")
        if gdf is None or getattr(gdf, "empty", True):
            return []
        gdf = self._ensure_feature_identity(gdf)
        self.layers[layer_name]["gdf"] = gdf
        geom_type = str(entry.get("geom_type", "") or "")
        if geom_type not in ("Point", "MultiPoint"):
            return []
        style = (entry.get("style", {}) or {}).copy()
        ext = (
            self._compare_extent_before
            if self._compare_ready() and panel == "before"
            else self._compare_extent_after
            if self._compare_ready() and panel == "after"
            else self._extent
        )
        radius = self._point_radius_px_from_style(style)
        sym_series = self._point_symbol_series_for_gdf(gdf, style)
        records: List[Dict[str, Any]] = []
        for pos in range(len(gdf)):
            geom = gdf.geometry.iloc[pos]
            if geom is None or geom.is_empty:
                continue
            parts = [geom] if geom.geom_type == "Point" else list(getattr(geom, "geoms", []))
            row = gdf.iloc[pos]
            uid = str(row.get("__mapgen_feature_uid") or "").strip()
            for part_idx, pt in enumerate(parts):
                sx, sy = (
                    self._geo_to_screen_compare_panel(float(pt.x), float(pt.y), ext, panel)
                    if self._compare_ready()
                    else self._geo_to_screen(float(pt.x), float(pt.y), ext)
                )
                if sx < -radius or sx > self.width() + radius or sy < -radius or sy > self.height() + radius:
                    continue
                records.append(
                    {
                        "iloc": pos,
                        "feature_uid": uid,
                        "sx": sx,
                        "sy": sy,
                        "radius_px": radius,
                        "symbol": str(sym_series.iloc[pos]),
                        "priority": self._screen_conflict_priority(row),
                        "draw_order": len(records),
                        "part_idx": part_idx,
                    }
                )
        return records

    @staticmethod
    def _screen_records_overlap(a: Dict[str, Any], b: Dict[str, Any], min_gap_px: float = 0.0) -> bool:
        dx = float(a["sx"]) - float(b["sx"])
        dy = float(a["sy"]) - float(b["sy"])
        dist = (dx * dx + dy * dy) ** 0.5
        threshold = float(a["radius_px"]) + float(b["radius_px"]) + float(min_gap_px)
        return dist < threshold

    def _pick_visible_point_record_at_screen(
        self,
        screen_x: float,
        screen_y: float,
        layer_name: str,
        panel: str,
        max_px: float | None = None,
    ) -> Dict[str, Any] | None:
        records = self._iter_visible_point_screen_records(layer_name, panel)
        if not records:
            return None
        dpr = self._effective_dpr()
        cap = (max_px if max_px is not None else 16.0) * dpr
        for rec in reversed(records):
            radius = max(float(rec["radius_px"]), cap * 0.35)
            dx = float(screen_x) - float(rec["sx"])
            dy = float(screen_y) - float(rec["sy"])
            if (dx * dx + dy * dy) ** 0.5 <= radius:
                return rec
        best: Dict[str, Any] | None = None
        best_d = cap + 1.0
        for rec in records:
            dx = float(screen_x) - float(rec["sx"])
            dy = float(screen_y) - float(rec["sy"])
            dist = (dx * dx + dy * dy) ** 0.5
            if dist < best_d:
                best_d = dist
                best = rec
        return best if best is not None and best_d <= cap else None

    def _nearest_point_iloc_at_screen(
        self,
        screen_x: float,
        screen_y: float,
        layer_name: str,
        panel: str,
        max_px: float | None = None,
    ) -> int | None:
        """在指定对比半幅内，找距点击位置最近的点要素 iloc 下标；无命中返回 None。"""
        rec = self._pick_visible_point_record_at_screen(screen_x, screen_y, layer_name, panel, max_px=max_px)
        return int(rec["iloc"]) if rec is not None else None

    def _compare_intervention_add_from_before(self, screen_x: float, screen_y: float) -> None:
        lb = self._compare_layer_before
        la = self._compare_layer_after
        if not lb or not la:
            return
        rec = self._pick_visible_point_record_at_screen(screen_x, screen_y, lb, "before")
        if rec is None:
            self.compare_intervention.emit("未命中综合前任何点（请放大后双击）")
            return
        self._materialize_point_symbols_for_layer(lb)
        self._materialize_point_symbols_for_layer(la)
        g_before = self._ensure_feature_identity(self.layers[lb]["gdf"])
        g_after = self._ensure_feature_identity(self.layers[la]["gdf"])
        self.layers[lb]["gdf"] = g_before
        self.layers[la]["gdf"] = g_after
        row = g_before.iloc[[int(rec["iloc"])]].copy()
        uid = str(row.iloc[0].get("__mapgen_feature_uid") or "").strip()
        if uid and "__mapgen_feature_uid" in g_after.columns:
            after_ids = set(g_after["__mapgen_feature_uid"].astype(str).tolist())
            if uid in after_ids:
                self.compare_intervention.emit("该点已在综合后图层中")
                return
        if False and len(g_after) > 0:
            try:
                dmin = float(g_after.geometry.distance(new_g).min())
            except Exception:
                dmin = 0.0
            if dmin < 1e-10:
                self.compare_intervention.emit("该点已在综合后图层中")
                return
        try:
            merged = gpd.GeoDataFrame(
                pd.concat([g_after.reset_index(drop=True), row], ignore_index=True),
                crs=g_after.crs,
            )
        except Exception:
            self.compare_intervention.emit("添加失败：无法合并要素")
            return
        if "__mapgen_user_pick" not in merged.columns:
            merged["__mapgen_user_pick"] = 0
        try:
            merged.loc[merged.index[-1], "__mapgen_user_pick"] = 1
        except Exception:
            pass
        merged = self._ensure_feature_identity(merged)
        self.layers[la]["gdf"] = merged
        self.layers[la]["feature_count"] = int(len(merged))
        self._materialize_point_symbols_for_layer(la)
        self._mark_dirty(reason="data")
        self.compare_intervention.emit(f"已从综合前补入 1 点至「{la}」（共 {len(merged)} 个点）")

    def _compare_intervention_remove_after(self, screen_x: float, screen_y: float) -> None:
        la = self._compare_layer_after
        if not la or la not in self.layers:
            return
        rec = self._pick_visible_point_record_at_screen(screen_x, screen_y, la, "after")
        if rec is None:
            self.compare_intervention.emit("未命中综合后任何点（请放大后双击删除）")
            return
        self._materialize_point_symbols_for_layer(la)
        g_after = self._ensure_feature_identity(self.layers[la]["gdf"])
        self.layers[la]["gdf"] = g_after
        uid = str(rec.get("feature_uid") or "").strip()
        if uid and "__mapgen_feature_uid" in g_after.columns:
            merged = g_after[g_after["__mapgen_feature_uid"].astype(str) != uid].copy().reset_index(drop=True)
        else:
            idx = int(rec["iloc"])
            keep_rows = [j for j in range(len(g_after)) if j != idx]
            merged = g_after.iloc[keep_rows].copy().reset_index(drop=True)
        self.layers[la]["gdf"] = merged
        self.layers[la]["feature_count"] = int(len(merged))
        self._materialize_point_symbols_for_layer(la)
        self._mark_dirty(reason="data")
        self.compare_intervention.emit(f"已从「{la}」删除 1 个点（剩余 {len(merged)} 个）")

    def resolve_visible_point_conflicts(self, layer_name: str, panel: str = "after", min_gap_px: float = 0.0) -> Dict[str, Any]:
        if not layer_name or layer_name not in self.layers:
            return {"layer_name": layer_name, "removed_count": 0, "conflict_count_before": 0, "conflict_count_after": 0}
        entry = self.layers[layer_name]
        gdf = entry.get("gdf")
        if gdf is None or getattr(gdf, "empty", True):
            return {"layer_name": layer_name, "removed_count": 0, "conflict_count_before": 0, "conflict_count_after": 0}
        gdf = self._ensure_feature_identity(gdf)
        self.layers[layer_name]["gdf"] = gdf
        records = self._iter_visible_point_screen_records(layer_name, panel)
        if len(records) <= 1:
            return {"layer_name": layer_name, "removed_count": 0, "conflict_count_before": 0, "conflict_count_after": 0}

        conflict_before = 0
        for i in range(len(records)):
            for j in range(i + 1, len(records)):
                if self._screen_records_overlap(records[i], records[j], min_gap_px=min_gap_px):
                    conflict_before += 1

        ordered = sorted(records, key=lambda rec: (-float(rec["priority"]), -int(rec["draw_order"])))
        kept: List[Dict[str, Any]] = []
        removed_uids: set[str] = set()
        visible_uids = {str(rec["feature_uid"]) for rec in records if str(rec["feature_uid"]).strip()}
        for rec in ordered:
            uid = str(rec.get("feature_uid") or "").strip()
            if uid and uid in removed_uids:
                continue
            if any(self._screen_records_overlap(rec, kept_rec, min_gap_px=min_gap_px) for kept_rec in kept):
                if uid:
                    removed_uids.add(uid)
                continue
            kept.append(rec)

        conflict_after = 0
        for i in range(len(kept)):
            for j in range(i + 1, len(kept)):
                if self._screen_records_overlap(kept[i], kept[j], min_gap_px=min_gap_px):
                    conflict_after += 1

        if not removed_uids:
            return {
                "layer_name": layer_name,
                "removed_count": 0,
                "conflict_count_before": conflict_before,
                "conflict_count_after": conflict_after,
            }

        keep_mask: List[bool] = []
        for idx in range(len(gdf)):
            uid = str(gdf.iloc[idx].get("__mapgen_feature_uid") or "").strip()
            keep_mask.append(not (uid and uid in visible_uids and uid in removed_uids))
        merged = gdf.loc[keep_mask].copy().reset_index(drop=True)
        self.layers[layer_name]["gdf"] = merged
        self.layers[layer_name]["feature_count"] = int(len(merged))
        self._refresh_point_symbol_style_for_layer(layer_name)
        self._mark_dirty(reason="data")
        return {
            "layer_name": layer_name,
            "removed_count": int(len(removed_uids)),
            "conflict_count_before": conflict_before,
            "conflict_count_after": conflict_after,
        }

    def screen_to_geo(self, sx: float, sy: float) -> Tuple[float, float]:
        if self._compare_ready():
            div = float(self._compare_divider_x())
            panel = "before" if float(sx) < div else "after"
            ext = self._compare_extent_before if panel == "before" else self._compare_extent_after
            return self._geo_from_screen_compare_panel(sx, sy, ext, panel)
        return self._geo_from_screen(sx, sy, self._extent)

    def _build_layer_tasks(self, names: List[str] | None = None) -> List[LayerRenderTask]:
        tasks: List[LayerRenderTask] = []
        target_names = list(names) if names is not None else list(self.layer_order)
        for z_idx, name in enumerate(target_names):
            layer = self.layers.get(name)
            if not layer or not layer.get("visible", True):
                continue
            tasks.append(
                LayerRenderTask(
                    name=name,
                    z_index=z_idx,
                    feature_count=int(layer.get("feature_count", 0)),
                    visible=True,
                    gdf=layer["gdf"],
                    geom_type=layer.get("geom_type", "") or "",
                    style=layer.get("style", {}) or {},
                )
            )
        return self._render_scheduler.order(tasks)

    def _build_layer_tasks_for_compare(self, layer_name: str) -> List[LayerRenderTask]:
        """单图层渲染（对比模式忽略图层列表中的可见性开关）。"""
        if layer_name not in self.layers:
            return []
        layer = self.layers[layer_name]
        if not bool(layer.get("visible", True)):
            return []
        z_idx = self.layer_order.index(layer_name) if layer_name in self.layer_order else 0
        tasks = [
            LayerRenderTask(
                name=layer_name,
                z_index=z_idx,
                feature_count=int(layer.get("feature_count", 0)),
                visible=True,
                gdf=layer["gdf"],
                geom_type=layer.get("geom_type", "") or "",
                style=layer.get("style", {}) or {},
            )
        ]
        return self._render_scheduler.order(tasks)

    def _compare_ready(self) -> bool:
        if not self._compare_enabled:
            return False
        b, a = self._compare_layer_before, self._compare_layer_after
        if not b or not a or b == a:
            return False
        return b in self.layers and a in self.layers

    def _compare_divider_x(self) -> int:
        w = max(1, self.width())
        return int(max(1, min(w - 1, round(w * self._compare_split))))

    def _hit_compare_divider(self, x: float) -> bool:
        if not self._compare_ready():
            return False
        return abs(float(x) - float(self._compare_divider_x())) <= float(self._compare_divider_hit_px)

    def set_compare_mode(self, enabled: bool, layer_before: str = "", layer_after: str = "") -> None:
        was_active = self._compare_ready()
        prev_b = self._compare_layer_before
        prev_a = self._compare_layer_after
        self._compare_enabled = bool(enabled)
        new_b = str(layer_before or "").strip()
        new_a = str(layer_after or "").strip()
        self._compare_layer_before = new_b
        self._compare_layer_after = new_a
        self._drag_compare_split = False
        self._compare_pan_side = None
        if self._compare_enabled:
            if (not was_active) or (new_b != prev_b) or (new_a != prev_a):
                self._compare_extent_dirty = True
        else:
            self._compare_extent_dirty = False
            self._cache_before = None
            self._cache_after = None
            if was_active:
                self._extent = list(self._compare_extent_before)
        self._mark_dirty()

    def refit_compare_panels(self) -> None:
        """两侧按各自图层范围重新拟合视图（保持对比开启）。"""
        if self._compare_ready():
            self._compare_extent_dirty = True
            self._mark_dirty()

    def compare_mode_active(self) -> bool:
        return self._compare_ready()

    def get_compare_config(self) -> Tuple[bool, str, str]:
        return (self._compare_enabled, self._compare_layer_before, self._compare_layer_after)

    def _compute_extent_for_layer_in_box(self, layer_name: str, logical_w: int, logical_h: int) -> List[float]:
        """按与半屏一致的宽高比包络图层数据并留白，使该层在对应半幅渲染时居中、少贴边。"""
        logical_w = max(1, int(logical_w))
        logical_h = max(1, int(logical_h))
        layer = self.layers.get(layer_name) or {}
        gdf = layer.get("gdf")
        if gdf is None or getattr(gdf, "empty", True):
            return list(self._extent)
        try:
            minx, miny, maxx, maxy = [float(v) for v in gdf.total_bounds]
        except Exception:
            return list(self._extent)
        cx = 0.5 * (minx + maxx)
        cy = 0.5 * (miny + maxy)
        dx = max(maxx - minx, 1e-9)
        dy = max(maxy - miny, 1e-9)
        want_aspect = float(logical_w) / float(logical_h)
        data_aspect = dx / dy
        pad = 0.14
        if data_aspect > want_aspect:
            dx_eff = dx * (1.0 + pad)
            dy_eff = dx_eff / want_aspect
        else:
            dy_eff = dy * (1.0 + pad)
            dx_eff = dy_eff * want_aspect
        return [cx - 0.5 * dx_eff, cy - 0.5 * dy_eff, cx + 0.5 * dx_eff, cy + 0.5 * dy_eff]

    def _fit_compare_extents_to_layer_bounds(self, wl: int, wr: int, h: int) -> None:
        self._compare_extent_before = self._compute_extent_for_layer_in_box(self._compare_layer_before, wl, h)
        self._compare_extent_after = self._compute_extent_for_layer_in_box(self._compare_layer_after, wr, h)

    def _stage_quality(self, stage: str) -> Dict[str, float]:
        if stage == "preview":
            return {"sample_multiplier": 0.35, "simplify_multiplier": 1.8}
        return {"sample_multiplier": 1.0, "simplify_multiplier": 1.0}

    def _request_async_render(self, stage: str) -> None:
        # Safety-first mode on Windows: render synchronously to avoid
        # occasional native crashes from async geometry rendering.
        self._current_stage = stage
        self._render_to_cache_sync()

    def _run_render_payload(
        self,
        tasks: List[LayerRenderTask],
        extent: Tuple[float, float, float, float] | None = None,
        pixel_w: int | None = None,
        pixel_h: int | None = None,
    ) -> Dict[str, Any] | None:
        dpr = self._effective_dpr()
        w0, h0 = max(1, self.width()), max(1, self.height())
        rw = max(1, int(round((pixel_w if pixel_w is not None else w0) * dpr)))
        rh = max(1, int(round((pixel_h if pixel_h is not None else h0) * dpr)))
        ext_tuple = extent if extent is not None else tuple(self._extent)
        payload = {
            "request_id": 1,
            "stage": "full",
            "extent": ext_tuple,
            "width": rw,
            "height": rh,
            "quality": self._stage_quality("full"),
            "layers": tasks,
            "cache_key": "",
        }
        result_holder: Dict[str, Any] = {}
        worker = _RenderWorker(payload, cancel_check=lambda rid: False)
        worker.signals.finished.connect(lambda out: result_holder.update({"result": out}))
        worker.run()
        result = result_holder.get("result")
        if isinstance(result, dict) and not result.get("error") and not result.get("canceled"):
            return result
        return None

    def _render_to_cache_sync(self) -> None:
        w, h = self.width(), self.height()
        if w <= 0 or h <= 0:
            return
        rw, rh, dpr = self._render_pixel_size()

        if self._compare_ready():
            tasks_l = self._build_layer_tasks_for_compare(self._compare_layer_before)
            tasks_r = self._build_layer_tasks_for_compare(self._compare_layer_after)
            if tasks_l and tasks_r:
                sx = self._compare_divider_x()
                wl = max(1, sx)
                wr = max(1, w - sx)
                if self._compare_extent_dirty:
                    self._fit_compare_extents_to_layer_bounds(wl, wr, h)
                    self._compare_extent_dirty = False
                    self._extent = list(self._compare_extent_before)
                self._render_profiler.reset()
                self._render_profiler.start("submit")
                r1 = self._run_render_payload(
                    tasks_l, tuple(self._compare_extent_before), pixel_w=wl, pixel_h=h
                )
                r2 = self._run_render_payload(
                    tasks_r, tuple(self._compare_extent_after), pixel_w=wr, pixel_h=h
                )
                self._render_profiler.end("submit")
                if r1 and r2:
                    self._render_profiler.start("rasterize")
                    self._cache_before = self._draw_render_plan(r1)
                    self._cache_before.setDevicePixelRatio(dpr)
                    self._cache_after = self._draw_render_plan(r2)
                    self._cache_after.setDevicePixelRatio(dpr)
                    self._cache = None
                    self._cache_dpr = dpr
                    self._render_profiler.end("rasterize")
                    self._last_render_profile = self._render_profiler.report()
                    self._lod_stats = r2.get("lod_stats", self._lod_stats)
                    self._needs_full_redraw = False
                    return
                self._cache_before = None
                self._cache_after = None
                self._compare_enabled = False

        payload = {
            "request_id": 1,
            "stage": "full",
            "extent": tuple(self._extent),
            "width": rw,
            "height": rh,
            "quality": self._stage_quality("full"),
            "layers": self._build_layer_tasks(),
            "cache_key": "",
        }
        result_holder: Dict[str, Any] = {}
        worker = _RenderWorker(payload, cancel_check=lambda rid: False)
        worker.signals.finished.connect(lambda out: result_holder.update({"result": out}))
        self._render_profiler.reset()
        self._render_profiler.start("submit")
        worker.run()
        self._render_profiler.end("submit")
        result = result_holder.get("result")
        if isinstance(result, dict) and not result.get("error") and not result.get("canceled"):
            self._render_profiler.start("rasterize")
            self._cache = self._draw_render_plan(result)
            self._cache.setDevicePixelRatio(dpr)
            self._cache_dpr = dpr
            self._cache_before = None
            self._cache_after = None
            self._render_profiler.end("rasterize")
            self._last_render_profile = self._render_profiler.report()
            self._lod_stats = result.get("lod_stats", self._lod_stats)
            self._needs_full_redraw = False

    def _draw_render_plan(self, plan: Dict[str, Any]) -> QPixmap:
        pixmap = QPixmap(int(plan.get("width", 0)), int(plan.get("height", 0)))
        pixmap.fill(QColor("white"))
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        for layer in plan.get("layers", []):
            layer_name = layer.get("name", "")
            style = ((self.layers.get(layer_name) or {}).get("style", {}) or layer.get("style", {}) or {})
            points = layer.get("points", []) or []
            lines = layer.get("lines", []) or []
            polygons = layer.get("polygons", []) or []
            if points:
                fill_color = QColor(style.get("fill", style.get("fillColor", "#E91E63")))
                stroke_color = QColor(style.get("stroke", style.get("color", "#C2185B")))
                stroke_width = float(style.get("stroke_width", style.get("weight", 1)))
                radius_val = style.get("radius", None)
                if radius_val is not None:
                    radius = float(radius_val)
                else:
                    radius = float(style.get("size", 8)) / 2.0
                fill_opacity = float(style.get("fill_opacity", style.get("fillOpacity", 0.8)))
                stroke_opacity = float(style.get("opacity", 1.0))
                fill_color.setAlphaF(max(0.0, min(1.0, fill_opacity)))
                stroke_color.setAlphaF(max(0.0, min(1.0, stroke_opacity)))
                painter.setPen(QPen(stroke_color, stroke_width))
                painter.setBrush(QBrush(fill_color))
                default_sym = normalize_point_symbol(style.get("point_symbol"))
                for pt in points:
                    if len(pt) >= 3:
                        sx, sy, sym = float(pt[0]), float(pt[1]), str(pt[2])
                    else:
                        sx, sy = float(pt[0]), float(pt[1])
                        sym = default_sym
                    draw_point_symbol(painter, sx, sy, radius, sym)
            if lines:
                pen = QPen(QColor(style.get("color", "#2196F3")), float(style.get("width", 2)))
                pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
                painter.setPen(pen)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                for line in lines:
                    painter.drawPolyline(QPolygonF([QPointF(float(x), float(y)) for x, y in line]))
            if polygons:
                fill = QColor(style.get("fill", "#4CAF50"))
                fill.setAlphaF(float(style.get("fill_opacity", 0.5)))
                painter.setPen(QPen(QColor(style.get("stroke", "#2E7D32")), float(style.get("stroke_width", 1))))
                painter.setBrush(QBrush(fill))
                for ring in polygons:
                    painter.drawPolygon(QPolygonF([QPointF(float(x), float(y)) for x, y in ring]))
        painter.end()
        return pixmap

    @staticmethod
    def _extent_for_aspect(extent: List[float] | Tuple[float, float, float, float], pixel_w: int, pixel_h: int) -> List[float]:
        xmin, ymin, xmax, ymax = [float(v) for v in extent]
        width = max(1e-12, xmax - xmin)
        height = max(1e-12, ymax - ymin)
        target_aspect = max(1e-9, float(pixel_w) / float(max(1, pixel_h)))
        current_aspect = width / height
        cx = (xmin + xmax) / 2.0
        cy = (ymin + ymax) / 2.0
        if current_aspect < target_aspect:
            width = height * target_aspect
        else:
            height = width / target_aspect
        return [cx - width / 2.0, cy - height / 2.0, cx + width / 2.0, cy + height / 2.0]

    def _visible_layer_bounds(self, names: List[str] | None = None) -> List[float] | None:
        target_names = names if names is not None else list(self.layer_order)
        bounds_list = []
        for name in target_names:
            entry = self.layers.get(name)
            if not entry or not bool(entry.get("visible", True)):
                continue
            gdf = entry.get("gdf")
            if gdf is None or getattr(gdf, "empty", True):
                continue
            try:
                b = [float(v) for v in gdf.total_bounds]
            except Exception:
                continue
            if any(math.isnan(v) or math.isinf(v) for v in b):
                continue
            bounds_list.append(b)
        if not bounds_list:
            return None
        return [
            min(b[0] for b in bounds_list),
            min(b[1] for b in bounds_list),
            max(b[2] for b in bounds_list),
            max(b[3] for b in bounds_list),
        ]

    def _expand_data_extent(self, extent: List[float], padding_ratio: float = 0.16) -> List[float]:
        xmin, ymin, xmax, ymax = [float(v) for v in extent]
        cx = (xmin + xmax) / 2.0
        cy = (ymin + ymax) / 2.0
        width = max(abs(xmax - xmin), 1e-6)
        height = max(abs(ymax - ymin), 1e-6)
        # Single points or very tight clusters need a real visual frame.
        min_span = max(width, height, 0.002)
        width = max(width, min_span * 0.25)
        height = max(height, min_span * 0.25)
        pad_x = width * padding_ratio
        pad_y = height * padding_ratio
        return [cx - width / 2.0 - pad_x, cy - height / 2.0 - pad_y, cx + width / 2.0 + pad_x, cy + height / 2.0 + pad_y]

    def _extent_ground_size_m(self, extent: List[float]) -> Tuple[float, float]:
        xmin, ymin, xmax, ymax = [float(v) for v in extent]
        center_lat = (ymin + ymax) / 2.0
        meters_per_deg_lon = 111320.0 * max(0.01, math.cos(math.radians(center_lat)))
        meters_per_deg_lat = 111320.0
        return abs(xmax - xmin) * meters_per_deg_lon, abs(ymax - ymin) * meters_per_deg_lat

    def _choose_adaptive_page(self, extent: List[float], margin_mm: float, title: str) -> Tuple[str, str]:
        width = max(1e-12, float(extent[2]) - float(extent[0]))
        height = max(1e-12, float(extent[3]) - float(extent[1]))
        orientation = "landscape" if width >= height else "portrait"
        scale = max(1, self._scale_denominator_for_extent(extent, max(self.width(), 1)))
        ground_w_m, ground_h_m = self._extent_ground_size_m(extent)
        required_w_mm = ground_w_m / float(scale) * 1000.0
        required_h_mm = ground_h_m / float(scale) * 1000.0
        title_mm = 10.0 if title else 0.0
        footer_mm = 7.0
        margin_mm = max(0.0, float(margin_mm))
        for page_name in PAGE_SIZE_ORDER:
            pw, ph = PAGE_SIZES_MM[page_name]
            if orientation == "landscape":
                pw, ph = max(pw, ph), min(pw, ph)
            else:
                pw, ph = min(pw, ph), max(pw, ph)
            map_w_mm = max(1.0, pw - margin_mm * 2.0)
            map_h_mm = max(1.0, ph - margin_mm * 2.0 - title_mm - footer_mm)
            if required_w_mm <= map_w_mm and required_h_mm <= map_h_mm:
                return page_name, orientation
        return PAGE_SIZE_ORDER[-1], orientation

    def _render_pixmap_for_page(self, pixel_w: int, pixel_h: int, fit_to_data: bool = False, layer_names: List[str] | None = None) -> QPixmap:
        pixel_w = max(1, int(pixel_w))
        pixel_h = max(1, int(pixel_h))
        if self._compare_ready() and not layer_names:
            sx = max(1, min(pixel_w - 1, int(round(pixel_w * self._compare_split))))
            left_w = max(1, sx)
            right_w = max(1, pixel_w - sx)
            tasks_l = self._build_layer_tasks_for_compare(self._compare_layer_before)
            tasks_r = self._build_layer_tasks_for_compare(self._compare_layer_after)
            left = QPixmap(left_w, pixel_h)
            left.fill(QColor("white"))
            right = QPixmap(right_w, pixel_h)
            right.fill(QColor("white"))
            if tasks_l:
                source_l = self._visible_layer_bounds([self._compare_layer_before]) if fit_to_data else None
                ext_l0 = self._expand_data_extent(source_l) if source_l else self._compare_extent_before
                ext_l = self._extent_for_aspect(ext_l0, left_w, pixel_h)
                plan_l = self._run_render_payload(tasks_l, tuple(ext_l), pixel_w=left_w, pixel_h=pixel_h)
                if plan_l:
                    left = self._draw_render_plan(plan_l)
            if tasks_r:
                source_r = self._visible_layer_bounds([self._compare_layer_after]) if fit_to_data else None
                ext_r0 = self._expand_data_extent(source_r) if source_r else self._compare_extent_after
                ext_r = self._extent_for_aspect(ext_r0, right_w, pixel_h)
                plan_r = self._run_render_payload(tasks_r, tuple(ext_r), pixel_w=right_w, pixel_h=pixel_h)
                if plan_r:
                    right = self._draw_render_plan(plan_r)
            out = QPixmap(pixel_w, pixel_h)
            out.fill(QColor("white"))
            painter = QPainter(out)
            painter.drawPixmap(0, 0, left)
            painter.drawPixmap(sx, 0, right)
            painter.setPen(QPen(QColor("#1565c0"), max(1, int(round(pixel_w * 0.0012)))))
            painter.drawLine(sx, 0, sx, pixel_h)
            painter.end()
            return out

        source = self._visible_layer_bounds(layer_names) if fit_to_data else None
        ext0 = self._expand_data_extent(source) if source else self._extent
        ext = self._extent_for_aspect(ext0, pixel_w, pixel_h)
        plan = self._run_render_payload(self._build_layer_tasks(layer_names), tuple(ext), pixel_w=pixel_w, pixel_h=pixel_h)
        if plan:
            return self._draw_render_plan(plan)
        blank = QPixmap(pixel_w, pixel_h)
        blank.fill(QColor("white"))
        return blank

    def export_layout(
        self,
        path: str,
        page_size: str = "A4",
        orientation: str = "landscape",
        dpi: int = 300,
        title: str = "",
        margin_mm: float = 12.0,
        fit_to_data: bool = True,
        layer_names: List[str] | None = None,
    ) -> str:
        page, page_w_mm, page_h_mm = self.render_layout_preview(
            page_size=page_size,
            orientation=orientation,
            dpi=dpi,
            title=title,
            margin_mm=margin_mm,
            fit_to_data=fit_to_data,
            layer_names=layer_names,
        )
        dpi = max(72, min(1200, int(dpi or 300)))

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        suffix = os.path.splitext(path)[1].lower()
        if suffix == ".pdf":
            fmt = str(page_size or "A4").upper()
            writer = QPdfWriter(path)
            writer.setResolution(dpi)
            writer.setPageSize(QPageSize(QSizeF(page_w_mm, page_h_mm), QPageSize.Unit.Millimeter, fmt))
            pdf_painter = QPainter(writer)
            pdf_painter.drawPixmap(QRect(0, 0, page.width(), page.height()), page)
            pdf_painter.end()
        else:
            page.save(path)
        return path

    def render_layout_preview(
        self,
        page_size: str = "A4",
        orientation: str = "landscape",
        dpi: int = 150,
        title: str = "",
        margin_mm: float = 12.0,
        fit_to_data: bool = True,
        layer_names: List[str] | None = None,
    ) -> Tuple[QPixmap, float, float]:
        data_extent = self._visible_layer_bounds(layer_names)
        expanded_extent = self._expand_data_extent(data_extent) if data_extent else None
        requested_page = str(page_size or "A4").upper()
        requested_orientation = str(orientation or "landscape").lower()
        if requested_page in {"AUTO", "自适应"} or requested_orientation in {"auto", "自适应"}:
            adaptive_page, adaptive_orientation = self._choose_adaptive_page(
                expanded_extent or list(self._extent),
                margin_mm=margin_mm,
                title=title,
            )
            if requested_page in {"AUTO", "自适应"}:
                requested_page = adaptive_page
            if requested_orientation in {"auto", "自适应"}:
                requested_orientation = adaptive_orientation

        fmt = requested_page
        page_w_mm, page_h_mm = PAGE_SIZES_MM.get(fmt, PAGE_SIZES_MM["A4"])
        if requested_orientation.startswith("land"):
            page_w_mm, page_h_mm = max(page_w_mm, page_h_mm), min(page_w_mm, page_h_mm)
        else:
            page_w_mm, page_h_mm = min(page_w_mm, page_h_mm), max(page_w_mm, page_h_mm)

        dpi = max(72, min(1200, int(dpi or 300)))
        px_per_mm = dpi / 25.4
        page_w_px = max(1, int(round(page_w_mm * px_per_mm)))
        page_h_px = max(1, int(round(page_h_mm * px_per_mm)))
        margin_px = max(0, int(round(float(margin_mm) * px_per_mm)))
        title_px = int(round(10.0 * px_per_mm)) if title else 0
        footer_px = int(round(7.0 * px_per_mm))
        map_x = margin_px
        map_y = margin_px + title_px
        map_w = max(1, page_w_px - margin_px * 2)
        map_h = max(1, page_h_px - margin_px * 2 - title_px - footer_px)

        page = QPixmap(page_w_px, page_h_px)
        page.fill(QColor("white"))
        painter = QPainter(page)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        if title:
            painter.setPen(QColor("#222222"))
            font = painter.font()
            font.setBold(True)
            font.setPointSizeF(11.0)
            painter.setFont(font)
            painter.drawText(QRectF(margin_px, margin_px * 0.35, map_w, title_px), Qt.AlignmentFlag.AlignCenter, title)
        map_pixmap = self._render_pixmap_for_page(map_w, map_h, fit_to_data=fit_to_data, layer_names=layer_names)
        painter.drawPixmap(map_x, map_y, map_pixmap)
        painter.setPen(QPen(QColor("#333333"), max(1, int(round(px_per_mm * 0.18)))))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(QRect(map_x, map_y, map_w, map_h))
        painter.setPen(QColor("#555555"))
        font = painter.font()
        font.setBold(False)
        font.setPointSizeF(7.5)
        painter.setFont(font)
        scale_extent = None
        if fit_to_data and expanded_extent:
            scale_extent = self._extent_for_aspect(expanded_extent, map_w, map_h)
        elif self._compare_ready():
            scale_extent = self._compare_extent_before
        else:
            scale_extent = self._extent
        scale_estimate = self._scale_denominator_for_extent(scale_extent, map_w)
        extent_label = "adaptive extent" if fit_to_data else "screen extent"
        scale_text = f"Scale estimate 1:{scale_estimate} | {fmt} {requested_orientation} | {extent_label} | {dpi} dpi"
        painter.drawText(QRectF(margin_px, page_h_px - margin_px - footer_px, map_w, footer_px), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, scale_text)
        painter.end()
        return page, page_w_mm, page_h_mm

    def _on_async_render_ready(self, result: Dict[str, Any]) -> None:
        if not result or result.get("request_id") != self._latest_request_id:
            return
        self._render_inflight = False
        if result.get("canceled") or result.get("error"):
            return
        if self._compare_ready():
            return
        self._render_profiler.start("rasterize")
        self._cache = self._draw_render_plan(result)
        self._render_profiler.end("rasterize")
        self._last_render_profile = self._render_profiler.report()
        self._lod_stats = result.get("lod_stats", self._lod_stats)
        self._memory_monitor.snapshot_mb()
        self._perf_history.append(
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "stage": self._current_stage,
                "submit_ms": float(self._last_render_profile.get("submit", 0.0) or 0.0),
                "rasterize_ms": float(self._last_render_profile.get("rasterize", 0.0) or 0.0),
                "memory_mb": float(self._memory_monitor.last_mb or 0.0),
                "memory_peak_mb": float(self._memory_monitor.peak_mb or 0.0),
                "total_features": int(self._lod_stats.get("total_features", 0) or 0),
                "visible_features": int(self._lod_stats.get("visible_features", 0) or 0),
                "rendered_features": int(self._lod_stats.get("rendered_features", 0) or 0),
                "sample_rate": float(self._lod_stats.get("sample_rate", 0.0) or 0.0),
            }
        )
        if len(self._perf_history) > 500:
            self._perf_history = self._perf_history[-500:]
        self._needs_full_redraw = False
        self._pan_offset_x = 0
        self._pan_offset_y = 0
        if result.get("stage") == "preview":
            self._request_async_render("full")
        self.update()

    def _mark_dirty(self, trigger_update: bool = True, reason: str = "view") -> None:
        self._async_engine.cancel_all()
        self._render_inflight = False
        self._needs_full_redraw = True
        if trigger_update:
            self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
            w, h = self.width(), self.height()
            ox, oy = self._pan_offset_x, self._pan_offset_y
            need_render = self._needs_full_redraw
            if self._compare_ready():
                need_render = need_render or self._cache_before is None or self._cache_after is None
            else:
                need_render = need_render or self._cache is None
            if need_render:
                self._render_to_cache_sync()

            if self._compare_ready() and self._cache_before is not None and self._cache_after is not None:
                ox_l = oy_l = ox_r = oy_r = 0
                if self._pan_start is not None and self._compare_pan_side == "before":
                    ox_l, oy_l = self._pan_offset_x, self._pan_offset_y
                elif self._pan_start is not None and self._compare_pan_side == "after":
                    ox_r, oy_r = self._pan_offset_x, self._pan_offset_y
                if ox_l or oy_l or ox_r or oy_r:
                    painter.fillRect(self.rect(), QColor("white"))
                sx = self._compare_divider_x()
                painter.setClipRect(QRect(0, 0, sx, h))
                painter.drawPixmap(ox_l, oy_l, self._cache_before)
                # 右侧缓存必须画在 [sx, w) 起点；若从 x=0 画，当 sx > w/2 时 pixmap 宽度 w-sx < sx，与裁剪区无交集 → 整侧空白
                painter.setClipRect(QRect(sx, 0, w - sx, h))
                painter.drawPixmap(sx + ox_r, oy_r, self._cache_after)
                painter.setClipping(False)
                painter.setPen(QPen(QColor("#1565c0"), 2))
                painter.drawLine(sx, 0, sx, h)
                painter.setPen(QPen(QColor("#0d47a1")))
                painter.drawText(6, 18, "综合前")
                painter.drawText(sx + 6, 18, "综合后")
            elif self._cache is not None:
                if ox or oy:
                    painter.fillRect(self.rect(), QColor("white"))
                    painter.drawPixmap(ox, oy, self._cache)
                else:
                    painter.drawPixmap(0, 0, self._cache)
            else:
                painter.fillRect(self.rect(), QColor("white"))
        finally:
            painter.end()

    def showEvent(self, event: QShowEvent) -> None:
        super().showEvent(event)
        win = self.window()
        wh = win.windowHandle() if win is not None else None
        if wh is not None and not self._screen_dpr_hooked:
            self._screen_dpr_hooked = True
            wh.screenChanged.connect(self._on_screen_changed)

    def _on_screen_changed(self, *args: Any) -> None:
        self._mark_dirty()

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        self._left_press_pos = None
        super().mouseDoubleClickEvent(event)
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if not self._compare_ready():
            return
        sx = float(event.position().x())
        if self._hit_compare_divider(sx):
            return
        if sx < float(self._compare_divider_x()):
            self._compare_intervention_add_from_before(sx, float(event.position().y()))
        else:
            self._compare_intervention_remove_after(sx, float(event.position().y()))
        event.accept()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            if self._compare_ready() and self._hit_compare_divider(event.position().x()):
                self._drag_compare_split = True
                self._left_press_pos = None
                return
            self._left_press_pos = QPointF(event.position())
        elif event.button() == Qt.MouseButton.MiddleButton:
            self._begin_pan(event.position(), Qt.MouseButton.MiddleButton)
        elif event.button() == Qt.MouseButton.RightButton:
            self._begin_pan(event.position(), Qt.MouseButton.RightButton)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            if self._drag_compare_split:
                self._drag_compare_split = False
                self.setCursor(Qt.CursorShape.ArrowCursor)
                self._mark_dirty()
                return
            if self._pan_button == Qt.MouseButton.LeftButton and self._pan_start is not None:
                self._commit_pan()
            elif self._left_press_pos is not None:
                px, py = self._left_press_pos.x(), self._left_press_pos.y()
                x, y = self.screen_to_geo(px, py)
                self.map_clicked.emit(x, y)
                self._left_press_pos = None
        elif event.button() == Qt.MouseButton.MiddleButton:
            if self._pan_button == Qt.MouseButton.MiddleButton and self._pan_start is not None:
                self._commit_pan()
        elif event.button() == Qt.MouseButton.RightButton:
            if self._pan_button == Qt.MouseButton.RightButton and self._pan_start is not None:
                self._commit_pan()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        pos = event.position()
        x, y = self.screen_to_geo(pos.x(), pos.y())
        self.mouse_moved.emit(x, y)

        if self._drag_compare_split and self._compare_ready():
            ww = max(1, self.width())
            self._compare_split = max(0.05, min(0.95, pos.x() / float(ww)))
            self.setCursor(Qt.CursorShape.SplitHCursor)
            self._mark_dirty()
            return

        if (
            self._compare_ready()
            and not self._drag_compare_split
            and self._pan_start is None
            and event.buttons() == Qt.MouseButton.NoButton
        ):
            if self._hit_compare_divider(pos.x()):
                self.setCursor(Qt.CursorShape.SplitHCursor)
            else:
                self.setCursor(Qt.CursorShape.ArrowCursor)

        if self._left_press_pos is not None and self._pan_start is None and (event.buttons() & Qt.MouseButton.LeftButton):
            dx = float(pos.x() - self._left_press_pos.x())
            dy = float(pos.y() - self._left_press_pos.y())
            if (dx * dx + dy * dy) ** 0.5 >= self._pan_drag_threshold_px:
                self._begin_pan(self._left_press_pos, Qt.MouseButton.LeftButton)

        if self._pan_start is not None and self._pan_still_held(event):
            self._pan_offset_x = int(pos.x() - self._pan_start.x())
            self._pan_offset_y = int(pos.y() - self._pan_start.y())
            self.update()

    def wheelEvent(self, event: QWheelEvent) -> None:
        pos = event.position()
        scale = 0.85 if event.angleDelta().y() > 0 else 1.18
        if self._compare_ready():
            sx = self._compare_divider_x()
            use_before = pos.x() < sx
            ext = self._compare_extent_before if use_before else self._compare_extent_after
            panel = "before" if use_before else "after"
            mx, my = self._geo_from_screen_compare_panel(pos.x(), pos.y(), ext, panel)
            xmin, ymin, xmax, ymax = ext
            new_ext = [
                mx - (mx - xmin) * scale,
                my - (my - ymin) * scale,
                mx + (xmax - mx) * scale,
                my + (ymax - my) * scale,
            ]
            if use_before:
                self._compare_extent_before = new_ext
            else:
                self._compare_extent_after = new_ext
            self._extent = list(self._compare_extent_before)
            ref = new_ext
        else:
            mx, my = self._geo_from_screen(pos.x(), pos.y(), self._extent)
            xmin, ymin, xmax, ymax = self._extent
            self._extent = [
                mx - (mx - xmin) * scale,
                my - (my - ymin) * scale,
                mx + (xmax - mx) * scale,
                my + (ymax - my) * scale,
            ]
            ref = self._extent
        self._mark_dirty()
        self.zoom_changed.emit(360.0 / max(ref[2] - ref[0], 1e-9))

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self._mark_dirty(trigger_update=False)

    def add_layer(self, name: str, geojson: Dict[str, Any], style: Dict[str, Any] | None = None) -> None:
        gdf = gpd.GeoDataFrame.from_features(geojson.get("features", []))
        if gdf.empty:
            return
        if gdf.crs is None:
            bounds = gdf.total_bounds
            if abs(bounds[0]) <= 180 and abs(bounds[1]) <= 90 and abs(bounds[2]) <= 180 and abs(bounds[3]) <= 90:
                gdf.set_crs("EPSG:4326", inplace=True)
            else:
                # Unknown/local projected coordinates: keep raw coordinates and avoid wrong auto-reprojection.
                # Internal renderer only needs consistent XY units.
                gdf.set_crs("EPSG:4326", inplace=True, allow_override=True)
        elif gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs("EPSG:4326")

        gdf = gdf.explode(index_parts=False)
        if "geometry" not in gdf.columns:
            return
        gdf = gdf[~gdf.geometry.isna()]
        gdf = gdf[~gdf.geometry.is_empty]
        if gdf.empty:
            return

        geom_types = sorted(set(gdf.geometry.geom_type.tolist()))
        geom_type = geom_types[0] if len(geom_types) == 1 else "Mixed"

        st = self.default_styles.get(geom_type, {}).copy()
        if style:
            is_point = geom_type in ("Point", "MultiPoint")
            is_polygon = "Polygon" in geom_type
            if "color" in style:
                if is_point or is_polygon:
                    st["stroke"] = style["color"]
                else:
                    st["color"] = style["color"]
            if "fillColor" in style:
                st["fill"] = style["fillColor"]
            if "fillOpacity" in style:
                st["fill_opacity"] = style["fillOpacity"]
            if "opacity" in style:
                st["opacity"] = style["opacity"]
            if "weight" in style or "linewidth" in style:
                w = style.get("weight", style.get("linewidth", 2))
                if is_point or is_polygon:
                    st["stroke_width"] = w
                else:
                    st["width"] = w
            if "radius" in style:
                st["radius"] = float(style["radius"])
                st["size"] = float(style["radius"]) * 2.0
            if "markersize" in style:
                st["size"] = style["markersize"] / 5
            if "point_symbol" in style:
                st["point_symbol"] = normalize_point_symbol(style.get("point_symbol"))
            if "point_symbol_field" in style:
                pf = style.get("point_symbol_field")
                if pf is not None and str(pf).strip():
                    st["point_symbol_field"] = str(pf).strip()
                else:
                    st.pop("point_symbol_field", None)
                    st.pop("point_symbol_map", None)
            if isinstance(style.get("point_symbol_map"), dict) and st.get("point_symbol_field"):
                st["point_symbol_map"] = {str(k): str(v) for k, v in style["point_symbol_map"].items()}

        if geom_type in ("Point", "MultiPoint") and (not style or "point_symbol_field" not in style):
            auto = infer_point_symbol_style_from_gdf(gdf)
            if auto:
                st.setdefault("point_symbol_field", auto["point_symbol_field"])
                st.setdefault("point_symbol_map", auto.get("point_symbol_map") or {})

        gdf = self._ensure_feature_identity(gdf)
        self.layers[name] = {"gdf": gdf, "style": st, "visible": True, "geom_type": geom_type, "feature_count": int(len(gdf))}
        if name not in self.layer_order:
            self.layer_order.append(name)
        self._mark_dirty(trigger_update=False, reason="data")
        self.zoom_to_layer(name)

    def remove_layer(self, name: str) -> None:
        if name in self.layers:
            del self.layers[name]
        if name in self.layer_order:
            self.layer_order.remove(name)
            
        if self._compare_enabled and name in (self._compare_layer_before, self._compare_layer_after):
            visible_layers = [l for l in self.layer_order if self.layers.get(l, {}).get("visible", False)]
            if len(visible_layers) < 2:
                self._compare_enabled = False
                self._compare_layer_before = ""
                self._compare_layer_after = ""
                self._cache_before = None
                self._cache_after = None
                self._drag_compare_split = False
                self._compare_pan_side = None
            else:
                b = self._compare_layer_before
                a = self._compare_layer_after
                changed = False
                
                if name == b:
                    candidates = [l for l in visible_layers if l != a]
                    if candidates:
                        self._compare_layer_before = candidates[0]
                        changed = True
                        
                if name == a or self._compare_layer_after == name:
                    candidates = [l for l in reversed(visible_layers) if l != self._compare_layer_before]
                    if candidates:
                        self._compare_layer_after = candidates[0]
                        changed = True
                        
                if changed:
                    self._compare_extent_dirty = True
                        
        self._mark_dirty(reason="data")

    def set_layer_visibility(self, name: str, visible: bool) -> None:
        if name in self.layers:
            self.layers[name]["visible"] = visible
            
            # Auto-reassign compare layers if needed
            if self._compare_enabled:
                visible_layers = [l for l in self.layer_order if self.layers.get(l, {}).get("visible", False)]
                if len(visible_layers) < 2:
                    self._compare_enabled = False
                    self._compare_layer_before = ""
                    self._compare_layer_after = ""
                else:
                    b = self._compare_layer_before
                    a = self._compare_layer_after
                    changed = False
                    
                    if not self.layers.get(b, {}).get("visible", False):
                        # b is hidden, find a new b that is not a
                        candidates = [l for l in visible_layers if l != a]
                        if candidates:
                            self._compare_layer_before = candidates[0]
                            changed = True
                            
                    if not self.layers.get(a, {}).get("visible", False):
                        # a is hidden, find a new a that is not b
                        candidates = [l for l in reversed(visible_layers) if l != self._compare_layer_before]
                        if candidates:
                            self._compare_layer_after = candidates[0]
                            changed = True

                    if changed:
                        self._compare_extent_dirty = True

            self._mark_dirty()

    def update_layer_style(self, name: str, style: Dict[str, Any]) -> None:
        if name not in self.layers:
            return
        current = self.layers[name].get("style", {}) or {}
        geom_type = str(self.layers[name].get("geom_type", "") or "")
        is_point = geom_type in ("Point", "MultiPoint")
        is_polygon = "Polygon" in geom_type
        style = style or {}

        if "fillColor" in style:
            current["fill"] = style["fillColor"]
        if "fillOpacity" in style:
            current["fill_opacity"] = style["fillOpacity"]
        if "opacity" in style:
            current["opacity"] = style["opacity"]
        if "color" in style:
            if is_point or is_polygon:
                current["stroke"] = style["color"]
            else:
                current["color"] = style["color"]
        if "weight" in style or "linewidth" in style:
            w = style.get("weight", style.get("linewidth", 2))
            if is_point or is_polygon:
                current["stroke_width"] = w
            else:
                current["width"] = w
        if "radius" in style:
            try:
                current["radius"] = float(style["radius"])
                current["size"] = float(style["radius"]) * 2.0
            except Exception:
                pass
        if "size" in style:
            current["size"] = style["size"]
        if "point_symbol" in style:
            current["point_symbol"] = normalize_point_symbol(style.get("point_symbol"))
        if "point_symbol_field" in style:
            pf = style.get("point_symbol_field")
            if pf is None or str(pf).strip() == "":
                current.pop("point_symbol_field", None)
                current.pop("point_symbol_map", None)
            else:
                current["point_symbol_field"] = str(pf).strip()
        if isinstance(style.get("point_symbol_map"), dict):
            current["point_symbol_map"] = {str(k): str(v) for k, v in style["point_symbol_map"].items()}
        self.layers[name]["style"] = current
        self._mark_dirty(reason="style")

    def zoom_to_layer(self, name: str) -> None:
        if name not in self.layers:
            return
        bounds = self.layers[name]["gdf"].total_bounds
        if np.any(np.isnan(bounds)) or np.any(np.isinf(bounds)):
            return
        dx = max((bounds[2] - bounds[0]) * 0.1, 0.5)
        dy = max((bounds[3] - bounds[1]) * 0.1, 0.5)
        new_ext = [bounds[0] - dx, bounds[1] - dy, bounds[2] + dx, bounds[3] + dy]
        self._extent = new_ext
        if self._compare_ready():
            self._compare_extent_before = new_ext[:]
            self._compare_extent_after = new_ext[:]
        self._mark_dirty(reason="data")

    def zoom_home(self) -> None:
        self._extent = self._home_extent.copy()
        if self._compare_ready():
            self._compare_extent_before = self._home_extent.copy()
            self._compare_extent_after = self._home_extent.copy()
        self._mark_dirty()

    def apply_toolbar_zoom(self, half_span_multiplier: float) -> None:
        """工具栏 +/-：非对比模式缩放主范围；对比模式下左右各自以当前中心缩放（倍数相同）。"""

        def zoom_one(ext: List[float]) -> List[float]:
            xmin, ymin, xmax, ymax = ext
            cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
            hw = (xmax - xmin) / 2 * half_span_multiplier
            hh = (ymax - ymin) / 2 * half_span_multiplier
            return [cx - hw, cy - hh, cx + hw, cy + hh]

        if self._compare_ready():
            self._compare_extent_before = zoom_one(self._compare_extent_before)
            self._compare_extent_after = zoom_one(self._compare_extent_after)
            self._extent = list(self._compare_extent_before)
        else:
            self._extent = zoom_one(self._extent)
        self._mark_dirty()

    def set_center(self, lat: float, lng: float, zoom: float | None = None) -> None:
        xmin, ymin, xmax, ymax = self._extent
        half_w = (xmax - xmin) / 2
        half_h = (ymax - ymin) / 2
        if zoom:
            half_w = 180 / zoom / 2
            half_h = half_w * self.height() / self.width() if self.width() > 0 else half_w
        new_ext = [lng - half_w, lat - half_h, lng + half_w, lat + half_h]
        self._extent = new_ext
        if self._compare_ready():
            self._compare_extent_before = new_ext[:]
            self._compare_extent_after = new_ext[:]
        self._mark_dirty()

    def move_layer_up(self, name: str) -> None:
        if name in self.layer_order:
            i = self.layer_order.index(name)
            if i > 0:
                self.layer_order[i], self.layer_order[i - 1] = self.layer_order[i - 1], self.layer_order[i]
                self._mark_dirty()

    def move_layer_down(self, name: str) -> None:
        if name in self.layer_order:
            i = self.layer_order.index(name)
            if i < len(self.layer_order) - 1:
                self.layer_order[i], self.layer_order[i + 1] = self.layer_order[i + 1], self.layer_order[i]
                self._mark_dirty()

    def refresh(self) -> None:
        self._mark_dirty()

    def get_layer_names(self) -> List[str]:
        return list(self.layer_order)

    def clear_layers(self) -> None:
        self.layers.clear()
        self.layer_order.clear()
        self._compare_enabled = False
        self._compare_layer_before = ""
        self._compare_layer_after = ""
        self._cache_before = None
        self._cache_after = None
        self._drag_compare_split = False
        self._compare_pan_side = None
        self._mark_dirty(reason="data")

    def get_render_diagnostics(self) -> Dict[str, Any]:
        return {
            "profile": dict(self._last_render_profile),
            "memory_mb": self._memory_monitor.last_mb,
            "memory_peak_mb": self._memory_monitor.peak_mb,
            "inflight": self._render_inflight,
            "stage": self._current_stage,
            "geometry_cache_size": 0,
            "page_cache_size": 0,
        }

    def get_scale_denominator(self) -> int:
        """Estimate map scale denominator from current extent and widget size."""
        w = max(self.width(), 1)
        xmin, ymin, xmax, ymax = (self._compare_extent_before if self._compare_ready() else self._extent)
        return self._scale_denominator_for_extent([xmin, ymin, xmax, ymax], w)

    def _scale_denominator_for_extent(self, extent: List[float], pixel_width: int) -> int:
        """Estimate scale denominator for a specific export/render extent."""
        w = max(int(pixel_width), 1)
        xmin, ymin, xmax, ymax = [float(v) for v in extent]
        lon_span = max(1e-9, abs(xmax - xmin))
        center_lat = (ymin + ymax) / 2.0
        meters_per_deg_lon = 111320.0 * max(0.01, math.cos(math.radians(center_lat)))
        meters_per_pixel = (lon_span * meters_per_deg_lon) / float(w)
        # Pixel size at 96 DPI.
        meters_per_pixel_on_map = 0.000264583
        scale = meters_per_pixel / meters_per_pixel_on_map
        return max(1, int(round(scale)))

    def set_scale_denominator(self, scale_denom: int) -> None:
        """Set extent by target scale denominator around current center."""
        target = max(1, int(scale_denom))
        w = max(self.width(), 1)
        h = max(self.height(), 1)
        meters_per_pixel_on_map = 0.000264583
        target_m_per_px = target * meters_per_pixel_on_map

        def apply_to_extent(ext: List[float]) -> List[float]:
            xmin, ymin, xmax, ymax = ext
            center_lon = (xmin + xmax) / 2.0
            center_lat = (ymin + ymax) / 2.0
            meters_per_deg_lon = 111320.0 * max(0.01, math.cos(math.radians(center_lat)))
            meters_per_deg_lat = 111320.0
            target_lon_span_deg = (target_m_per_px * w) / meters_per_deg_lon
            target_lat_span_deg = (target_m_per_px * h) / meters_per_deg_lat
            return [
                center_lon - target_lon_span_deg / 2.0,
                center_lat - target_lat_span_deg / 2.0,
                center_lon + target_lon_span_deg / 2.0,
                center_lat + target_lat_span_deg / 2.0,
            ]

        if self._compare_ready():
            self._compare_extent_before = apply_to_extent(self._compare_extent_before)
            self._compare_extent_after = apply_to_extent(self._compare_extent_after)
            self._extent = list(self._compare_extent_before)
        else:
            self._extent = apply_to_extent(self._extent)
        self._mark_dirty()

    def get_performance_report(self) -> Dict[str, Any]:
        return {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "count": len(self._perf_history),
            "summary": {},
            "history": list(self._perf_history),
        }

    def export_performance_report(self, path: str, fmt: str) -> None:
        report = self.get_performance_report()
        if fmt == "json":
            with open(path, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            return
        if fmt == "csv":
            fields = ["timestamp", "stage", "submit_ms", "rasterize_ms", "memory_mb", "memory_peak_mb", "total_features", "visible_features", "rendered_features", "sample_rate"]
            with open(path, "w", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                for row in report.get("history", []):
                    writer.writerow({k: row.get(k, "") for k in fields})
            return
        raise ValueError(f"Unsupported format: {fmt}")


class NativeMapCanvas(QFrame):
    map_clicked = Signal(float, float)
    layer_selected = Signal(str)
    zoom_changed = Signal(float)
    extent_changed = Signal(tuple)
    compare_intervention = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.Shape.NoFrame)
        self.setStyleSheet("background: white;")
        self.setMinimumSize(400, 300)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        toolbar = QWidget()
        toolbar.setStyleSheet("background: #fafafa; border-bottom: 1px solid #ddd;")
        toolbar.setFixedHeight(32)
        tb = QHBoxLayout(toolbar)
        tb.setContentsMargins(4, 2, 4, 2)
        tb.setSpacing(2)
        self.btn_home = QPushButton()
        self.btn_home.setFixedSize(40, 24)
        self.btn_home.clicked.connect(self._on_home)
        self.btn_zi = QPushButton("+")
        self.btn_zi.setFixedSize(28, 24)
        self.btn_zi.clicked.connect(self._on_zoom_in)
        self.btn_zo = QPushButton("-")
        self.btn_zo.setFixedSize(28, 24)
        self.btn_zo.clicked.connect(self._on_zoom_out)
        self.btn_scale = QPushButton()
        self.btn_scale.setFixedSize(64, 24)
        self.btn_scale.clicked.connect(self._open_scale_dialog)
        self.btn_compare = QPushButton()
        self.btn_compare.setFixedSize(56, 24)
        self.btn_compare.clicked.connect(self._open_compare_dialog)
        self.btn_export = QPushButton()
        self.btn_export.setFixedSize(56, 24)
        self.btn_export.clicked.connect(self._open_layout_export_dialog)
        for btn in [self.btn_home, self.btn_zi, self.btn_zo, self.btn_scale, self.btn_compare, self.btn_export]:
            btn.setStyleSheet("QPushButton { background: white; border: 1px solid #ccc; border-radius: 3px; } QPushButton:hover { background: #e0e0e0; }")
            tb.addWidget(btn)
        tb.addStretch()
        layout.addWidget(toolbar)
        self.map_widget = MapWidget()
        self.map_widget.map_clicked.connect(self.map_clicked.emit)
        self.map_widget.zoom_changed.connect(self.zoom_changed.emit)
        self.map_widget.compare_intervention.connect(self.compare_intervention.emit)
        self.map_widget.mouse_moved.connect(self._on_mouse_moved)
        layout.addWidget(self.map_widget, 1)
        self.status_bar = QLabel()
        self.status_bar.setStyleSheet("QLabel { background: #fafafa; border-top: 1px solid #ddd; padding: 4px 8px; font-size: 11px; color: #666; }")
        self.status_bar.setFixedHeight(24)
        layout.addWidget(self.status_bar)
        self.retranslate_ui()

    def retranslate_ui(self) -> None:
        self.btn_home.setText(tr("map.toolbar.fit"))
        self.btn_scale.setText(tr("map.toolbar.scale"))
        self.btn_compare.setText(tr("map.toolbar.compare"))
        self.btn_compare.setToolTip(tr("map.compare_tooltip"))
        self.btn_export.setText(tr("map.toolbar.export"))
        self.btn_export.setToolTip(tr("map.export_tooltip"))
        if not self.status_bar.text() or self.status_bar.text() == tr("map.status.default"):
            self.status_bar.setText(tr("map.status.default"))

    def _localize_dialog_buttons(self, btn_box: QDialogButtonBox) -> None:
        ok_btn = btn_box.button(QDialogButtonBox.StandardButton.Ok)
        cancel_btn = btn_box.button(QDialogButtonBox.StandardButton.Cancel)
        if ok_btn:
            ok_btn.setText(tr("common.ok"))
        if cancel_btn:
            cancel_btn.setText(tr("common.cancel"))

    def _on_mouse_moved(self, x: float, y: float) -> None:
        extent = self.map_widget._extent
        zoom = 360 / (extent[2] - extent[0]) if extent[2] != extent[0] else 1
        scale_denom = self.map_widget.get_scale_denominator()
        stats = self.map_widget._lod_stats
        diag = self.map_widget.get_render_diagnostics()
        profile = diag.get("profile", {}) if isinstance(diag, dict) else {}
        raster_ms = profile.get("rasterize", 0.0)
        dpr = self.map_widget._effective_dpr()
        if stats["sample_rate"] < 1.0:
            txt = f"{dpr:.1f}x | {tr('map.status.help')} | Coords: {x:.6f}, {y:.6f} | Zoom: {zoom:.1f}x | Scale: 1:{scale_denom} | LOD: {stats['rendered_features']}/{stats['visible_features']} ({stats['sample_rate']*100:.0f}%) | {raster_ms:.1f}ms"
        else:
            txt = f"{dpr:.1f}x | {tr('map.status.help')} | Coords: {x:.6f}, {y:.6f} | Zoom: {zoom:.1f}x | Scale: 1:{scale_denom} | {raster_ms:.1f}ms"
        self.status_bar.setText(txt)

    def _open_scale_dialog(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle(tr("map.scale_title"))
        dialog.setMinimumWidth(320)
        layout = QVBoxLayout(dialog)

        form = QFormLayout()
        scale_spin = QSpinBox(dialog)
        scale_spin.setRange(100, 100000000)
        scale_spin.setSingleStep(1000)
        scale_spin.setValue(self.map_widget.get_scale_denominator())
        form.addRow(tr("map.scale_target"), scale_spin)
        layout.addLayout(form)

        btn_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, parent=dialog)
        self._localize_dialog_buttons(btn_box)
        btn_box.accepted.connect(dialog.accept)
        btn_box.rejected.connect(dialog.reject)
        layout.addWidget(btn_box)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.map_widget.set_scale_denominator(scale_spin.value())

    def _open_compare_dialog(self) -> None:
        names = self.map_widget.get_layer_names()
        if len(names) < 2:
            QMessageBox.information(self, tr("map.toolbar.compare"), tr("map.compare_need_layers"))
            return
        en, lb, la = self.map_widget.get_compare_config()
        dialog = QDialog(self)
        dialog.setWindowTitle(tr("map.compare_title"))
        dialog.setMinimumWidth(360)
        v = QVBoxLayout(dialog)
        chk = QCheckBox(tr("map.compare_enable"))
        chk.setChecked(en and bool(lb) and bool(la))
        v.addWidget(chk)
        form = QFormLayout()
        cb_before = QComboBox(dialog)
        cb_after = QComboBox(dialog)
        for n in names:
            cb_before.addItem(n)
            cb_after.addItem(n)

        def _default_after_idx() -> int:
            for i, n in enumerate(names):
                low = n.lower()
                if "_gen_" in low or "gen_" in low:
                    return i
            return min(1, len(names) - 1)

        def _default_before_idx(after_i: int) -> int:
            if after_i != 0:
                return 0
            return min(1, len(names) - 1)

        ai = _default_after_idx()
        bi = _default_before_idx(ai)
        if lb in names:
            cb_before.setCurrentText(lb)
        else:
            cb_before.setCurrentIndex(bi)
        if la in names:
            cb_after.setCurrentText(la)
        else:
            cb_after.setCurrentIndex(ai)
        form.addRow(tr("map.compare_before"), cb_before)
        form.addRow(tr("map.compare_after"), cb_after)
        v.addLayout(form)
        hint = QLabel(tr("map.compare_hint"))
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #666; font-size: 11px;")
        v.addWidget(hint)
        btn_refit = QPushButton(tr("map.compare_refit"))
        btn_refit.setToolTip(tr("map.compare_refit_tooltip"))
        btn_refit.clicked.connect(self.map_widget.refit_compare_panels)
        v.addWidget(btn_refit)
        btn_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, parent=dialog)
        self._localize_dialog_buttons(btn_box)
        btn_box.accepted.connect(dialog.accept)
        btn_box.rejected.connect(dialog.reject)
        v.addWidget(btn_box)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        if chk.isChecked():
            b = cb_before.currentText().strip()
            a = cb_after.currentText().strip()
            if not b or not a or b == a:
                QMessageBox.warning(self, tr("map.toolbar.compare"), tr("map.compare_select_different"))
                return
            self.map_widget.set_compare_mode(True, b, a)
        else:
            self.map_widget.set_compare_mode(False, "", "")

    def _open_layout_export_dialog(self) -> None:
        if not self.map_widget.get_layer_names():
            QMessageBox.information(self, tr("map.layout_title"), tr("map.layout_need_layer"))
            return
        dialog = QDialog(self)
        dialog.setWindowTitle(tr("map.layout_title"))
        dialog.setMinimumWidth(360)
        layout = QVBoxLayout(dialog)
        form = QFormLayout()

        page_combo = QComboBox(dialog)
        page_combo.addItems(["Auto", "A4", "A3", "A2", "Letter"])
        page_combo.setCurrentText("Auto")
        form.addRow(tr("map.layout_page"), page_combo)

        orient_combo = QComboBox(dialog)
        orient_combo.addItems(["auto", "landscape", "portrait"])
        orient_combo.setCurrentText("auto")
        form.addRow(tr("map.layout_orientation"), orient_combo)

        dpi_spin = QSpinBox(dialog)
        dpi_spin.setRange(72, 1200)
        dpi_spin.setSingleStep(50)
        dpi_spin.setValue(300)
        form.addRow("DPI:", dpi_spin)

        margin_spin = QDoubleSpinBox(dialog)
        margin_spin.setRange(0.0, 50.0)
        margin_spin.setSingleStep(1.0)
        margin_spin.setValue(12.0)
        form.addRow(tr("map.layout_margin"), margin_spin)

        fit_check = QCheckBox(tr("map.layout_fit"))
        fit_check.setChecked(True)
        fit_check.setToolTip(tr("map.layout_fit_tooltip"))
        form.addRow("", fit_check)

        after_only_check = QCheckBox(tr("map.layout_after_only"))
        after_only_check.setChecked(self.map_widget.compare_mode_active())
        after_only_check.setEnabled(self.map_widget.compare_mode_active())
        after_only_check.setToolTip(tr("map.layout_after_only_tooltip"))
        form.addRow("", after_only_check)

        title_edit = QComboBox(dialog)
        title_edit.setEditable(True)
        title_edit.addItem(tr("map.layout_map_title"))
        title_edit.addItem("")
        form.addRow(tr("map.layout_title_label"), title_edit)
        layout.addLayout(form)

        hint = QLabel(tr("map.layout_hint"))
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #666; font-size: 11px;")
        layout.addWidget(hint)

        preview_label = QLabel(dialog)
        preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        preview_label.setStyleSheet("QLabel { background: #f5f5f5; border: 1px solid #ccc; }")
        preview_label.setMinimumSize(520, 340)
        preview_scroll = QScrollArea(dialog)
        preview_scroll.setWidgetResizable(True)
        preview_scroll.setMinimumHeight(360)
        preview_scroll.setWidget(preview_label)
        layout.addWidget(preview_scroll, 1)

        def export_layer_names() -> List[str] | None:
            if after_only_check.isChecked() and self.map_widget.compare_mode_active():
                _, _, after_name = self.map_widget.get_compare_config()
                if after_name:
                    return [after_name]
            return None

        def update_preview() -> None:
            try:
                pixmap, _, _ = self.map_widget.render_layout_preview(
                    page_size=page_combo.currentText(),
                    orientation=orient_combo.currentText(),
                    dpi=96,
                    title=title_edit.currentText(),
                    margin_mm=margin_spin.value(),
                    fit_to_data=fit_check.isChecked(),
                    layer_names=export_layer_names(),
                )
                target = preview_scroll.viewport().size()
                if target.width() <= 10 or target.height() <= 10:
                    target = preview_label.size()
                scaled = pixmap.scaled(
                    max(100, target.width() - 18),
                    max(100, target.height() - 18),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                preview_label.setPixmap(scaled)
                preview_label.setText("")
            except Exception as exc:
                preview_label.setPixmap(QPixmap())
                preview_label.setText(tr("map.layout_preview_failed", error=exc))

        preview_btn = QPushButton(tr("map.layout_refresh_preview"), dialog)
        preview_btn.clicked.connect(update_preview)
        layout.addWidget(preview_btn)
        page_combo.currentTextChanged.connect(lambda *_: update_preview())
        orient_combo.currentTextChanged.connect(lambda *_: update_preview())
        margin_spin.valueChanged.connect(lambda *_: update_preview())
        fit_check.stateChanged.connect(lambda *_: update_preview())
        after_only_check.stateChanged.connect(lambda *_: update_preview())
        title_edit.currentTextChanged.connect(lambda *_: update_preview())
        QTimer.singleShot(0, update_preview)

        btn_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, parent=dialog)
        self._localize_dialog_buttons(btn_box)
        btn_box.accepted.connect(dialog.accept)
        btn_box.rejected.connect(dialog.reject)
        layout.addWidget(btn_box)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        default_name = f"map_layout_{page_combo.currentText()}_{orient_combo.currentText()}.pdf"
        path, _ = QFileDialog.getSaveFileName(
            self,
            tr("map.layout_title"),
            os.path.join(os.getcwd(), default_name),
            tr("map.layout_filter"),
        )
        if not path:
            return
        try:
            self.export_layout(
                path,
                page_size=page_combo.currentText(),
                orientation=orient_combo.currentText(),
                dpi=dpi_spin.value(),
                title=title_edit.currentText(),
                margin_mm=margin_spin.value(),
                fit_to_data=fit_check.isChecked(),
                layer_names=export_layer_names(),
            )
            QMessageBox.information(self, tr("map.layout_title"), tr("map.layout_exported", path=path))
        except Exception as exc:
            QMessageBox.warning(self, tr("map.layout_export_failed"), str(exc))

    def _on_home(self) -> None:
        self.map_widget.zoom_home()

    def _on_zoom_in(self) -> None:
        self.map_widget.apply_toolbar_zoom(0.7)

    def _on_zoom_out(self) -> None:
        self.map_widget.apply_toolbar_zoom(1.4)

    def zoom_home(self) -> None: self._on_home()
    def zoom_in(self) -> None: self._on_zoom_in()
    def zoom_out(self) -> None: self._on_zoom_out()
    def add_layer(self, name: str, geojson: Dict, style: Dict = None): self.map_widget.add_layer(name, geojson, style)
    def remove_layer(self, name: str): self.map_widget.remove_layer(name)
    def set_layer_visibility(self, name: str, visible: bool): self.map_widget.set_layer_visibility(name, visible)
    def zoom_to_layer(self, name: str): self.map_widget.zoom_to_layer(name)
    def set_center(self, lat: float, lng: float, zoom: float = None): self.map_widget.set_center(lat, lng, zoom)
    def move_layer_up(self, name: str): self.map_widget.move_layer_up(name)
    def move_layer_down(self, name: str): self.map_widget.move_layer_down(name)
    def refresh(self): self.map_widget.refresh()
    def get_layer_names(self) -> List[str]: return self.map_widget.get_layer_names()
    def clear_layers(self): self.map_widget.clear_layers()
    def update_layer_style(self, name: str, style: dict): self.map_widget.update_layer_style(name, style)
    def get_performance_report(self) -> Dict[str, Any]: return self.map_widget.get_performance_report()
    def export_performance_report(self, path: str, fmt: str) -> None: self.map_widget.export_performance_report(path, fmt)
    def export_image(self, path: str, dpi: int = 150): self.map_widget.grab().save(path)
    def export_layout(self, path: str, page_size: str = "A4", orientation: str = "landscape", dpi: int = 300, title: str = "", margin_mm: float = 12.0, fit_to_data: bool = True, layer_names: List[str] | None = None) -> str:
        return self.map_widget.export_layout(path, page_size=page_size, orientation=orientation, dpi=dpi, title=title, margin_mm=margin_mm, fit_to_data=fit_to_data, layer_names=layer_names)
    def set_compare_mode(self, enabled: bool, layer_before: str = "", layer_after: str = "") -> None:
        self.map_widget.set_compare_mode(enabled, layer_before, layer_after)

    def refit_compare_panels(self) -> None:
        self.map_widget.refit_compare_panels()

    def get_compare_config(self) -> Tuple[bool, str, str]:
        return self.map_widget.get_compare_config()

    def get_layer_geojson(self, name: str) -> Dict[str, Any] | None:
        if name not in self.map_widget.layers:
            return None
        try:
            return json.loads(self.map_widget.layers[name]["gdf"].to_json())
        except Exception:
            return None

    def resolve_visible_point_conflicts(self, layer_name: str, panel: str = "after", min_gap_px: float = 0.0) -> Dict[str, Any]:
        return self.map_widget.resolve_visible_point_conflicts(layer_name, panel=panel, min_gap_px=min_gap_px)
