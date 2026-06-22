"""Native canvas point symbol drawing. Logic names live in point_symbol_core (no Qt)."""

from __future__ import annotations

import math
from typing import Iterable, Tuple

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPen, QPolygonF

from src.ui.point_symbol_core import KNOWN_POINT_SYMBOLS, normalize_point_symbol

__all__ = [
    "KNOWN_POINT_SYMBOLS",
    "normalize_point_symbol",
    "draw_point_symbol",
]


def _poly_from_points(points: Iterable[Tuple[float, float]]) -> QPolygonF:
    return QPolygonF([QPointF(float(x), float(y)) for x, y in points])


_POI_ICON_COLORS = {
    "convenience": "#d23b7d",
    "school": "#2e7d32",
    "parking": "#1976d2",
    "hotel": "#7b4fa1",
    "bank": "#00897b",
    "hospital": "#d32f2f",
    "restaurant": "#ef8a24",
    "transit": "#4e79a7",
}

_POI_ICON_LABELS = {
    "convenience": "",
    "school": "S",
    "parking": "P",
    "hotel": "H",
    "bank": "B",
    "hospital": "+",
    "restaurant": "R",
    "transit": "T",
}


def _draw_labeled_icon(painter: QPainter, cx: float, cy: float, radius: float, symbol: str) -> None:
    old_pen = QPen(painter.pen())
    old_brush = QBrush(painter.brush())
    old_font = QFont(painter.font())

    fill = QColor(_POI_ICON_COLORS.get(symbol, "#555555"))
    outline = QColor("#263238")
    painter.setPen(QPen(outline, max(1.0, radius * 0.16)))
    painter.setBrush(QBrush(fill))
    box = QRectF(cx - radius, cy - radius, 2.0 * radius, 2.0 * radius)
    painter.drawRoundedRect(box, radius * 0.22, radius * 0.22)

    painter.setPen(QPen(QColor("white"), max(1.0, radius * 0.18), Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
    label = _POI_ICON_LABELS.get(symbol, "")
    if symbol == "convenience":
        roof = _poly_from_points(
            [
                (cx - radius * 0.58, cy - radius * 0.10),
                (cx, cy - radius * 0.58),
                (cx + radius * 0.58, cy - radius * 0.10),
            ]
        )
        painter.drawPolyline(roof)
        painter.drawLine(QPointF(cx - radius * 0.46, cy - radius * 0.06), QPointF(cx - radius * 0.46, cy + radius * 0.48))
        painter.drawLine(QPointF(cx + radius * 0.46, cy - radius * 0.06), QPointF(cx + radius * 0.46, cy + radius * 0.48))
        painter.drawLine(QPointF(cx - radius * 0.50, cy + radius * 0.48), QPointF(cx + radius * 0.50, cy + radius * 0.48))
        painter.drawRect(QRectF(cx - radius * 0.15, cy + radius * 0.12, radius * 0.30, radius * 0.36))
    elif symbol == "hospital":
        d = radius * 0.52
        painter.drawLine(QPointF(cx - d, cy), QPointF(cx + d, cy))
        painter.drawLine(QPointF(cx, cy - d), QPointF(cx, cy + d))
    elif symbol == "restaurant":
        painter.drawLine(QPointF(cx - radius * 0.35, cy - radius * 0.48), QPointF(cx - radius * 0.35, cy + radius * 0.50))
        painter.drawLine(QPointF(cx - radius * 0.52, cy - radius * 0.40), QPointF(cx - radius * 0.18, cy - radius * 0.40))
        painter.drawLine(QPointF(cx + radius * 0.30, cy - radius * 0.50), QPointF(cx + radius * 0.30, cy + radius * 0.50))
        painter.drawLine(QPointF(cx + radius * 0.16, cy - radius * 0.50), QPointF(cx + radius * 0.44, cy - radius * 0.50))
    elif symbol == "bank":
        roof = _poly_from_points(
            [
                (cx - radius * 0.62, cy - radius * 0.25),
                (cx, cy - radius * 0.62),
                (cx + radius * 0.62, cy - radius * 0.25),
            ]
        )
        painter.drawPolygon(roof)
        for offset in (-0.38, 0.0, 0.38):
            painter.drawLine(QPointF(cx + radius * offset, cy - radius * 0.18), QPointF(cx + radius * offset, cy + radius * 0.46))
        painter.drawLine(QPointF(cx - radius * 0.58, cy + radius * 0.52), QPointF(cx + radius * 0.58, cy + radius * 0.52))
    else:
        font = QFont(old_font)
        font.setBold(True)
        font.setPointSizeF(max(6.0, radius * (0.88 if len(label) == 1 else 0.58)))
        painter.setFont(font)
        painter.drawText(box, Qt.AlignmentFlag.AlignCenter, label)

    painter.setPen(old_pen)
    painter.setBrush(old_brush)
    painter.setFont(old_font)


def draw_point_symbol(painter: QPainter, cx: float, cy: float, radius: float, symbol: str | None) -> None:
    """Draw a filled point marker with outline; radius matches circle semantic (half-extent)."""
    sym = normalize_point_symbol(symbol)
    r = max(0.5, float(radius))

    if sym in _POI_ICON_COLORS:
        _draw_labeled_icon(painter, cx, cy, r * 1.08, sym)
        return

    if sym == "circle":
        painter.drawEllipse(QPointF(cx, cy), r, r)
        return

    if sym == "square":
        painter.drawRect(cx - r, cy - r, 2.0 * r, 2.0 * r)
        return

    if sym == "triangle":
        h = 1.732 * r
        pts = [(cx, cy - 0.9 * r), (cx - h * 0.55, cy + 0.65 * r), (cx + h * 0.55, cy + 0.65 * r)]
        painter.drawPolygon(_poly_from_points(pts))
        return

    if sym == "diamond":
        pts = [(cx, cy - r * 1.05), (cx + r * 1.05, cy), (cx, cy + r * 1.05), (cx - r * 1.05, cy)]
        painter.drawPolygon(_poly_from_points(pts))
        return

    if sym == "cross":
        d = r * 0.85
        pen = painter.pen()
        lw = max(pen.widthF(), max(1.25, r * 0.28))
        painter.setPen(QPen(pen.color(), lw, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawLine(QPointF(cx - d, cy - d), QPointF(cx + d, cy + d))
        painter.drawLine(QPointF(cx - d, cy + d), QPointF(cx + d, cy - d))
        painter.setPen(pen)
        return

    if sym == "plus":
        d = r * 0.95
        pen = painter.pen()
        lw = max(pen.widthF(), max(1.25, r * 0.28))
        painter.setPen(QPen(pen.color(), lw, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawLine(QPointF(cx - d, cy), QPointF(cx + d, cy))
        painter.drawLine(QPointF(cx, cy - d), QPointF(cx, cy + d))
        painter.setPen(pen)
        return

    if sym == "star":
        pts = []
        for i in range(10):
            angle = math.pi / 2 + i * math.pi / 5
            rad = r * (1.0 if i % 2 == 0 else 0.42)
            pts.append((cx + rad * math.cos(angle), cy - rad * math.sin(angle)))
        painter.drawPolygon(_poly_from_points(pts))
        return

    if sym == "hexagon":
        pts = []
        for i in range(6):
            angle = math.pi / 6 + i * math.pi / 3
            pts.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
        painter.drawPolygon(_poly_from_points(pts))
        return

    painter.drawEllipse(QPointF(cx, cy), r, r)
