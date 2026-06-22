from typing import Dict, Any, Optional, Tuple, List
from PySide6.QtGui import QColor, QPen, QBrush, QLinearGradient, QRadialGradient
from PySide6.QtCore import Qt
import numpy as np


class StyleRenderer:

    @staticmethod
    def create_gradient_brush(color1: str, color2: str, gradient_type: str = 'linear') -> QBrush:
        if gradient_type == 'linear':
            gradient = QLinearGradient(0, 0, 100, 100)
        else:
            gradient = QRadialGradient(50, 50, 50)

        gradient.setColorAt(0, QColor(color1))
        gradient.setColorAt(1, QColor(color2))
        return QBrush(gradient)

    @staticmethod
    def create_pen(color: str, width: float, style: str = 'solid', cap: str = 'round', join: str = 'round') -> QPen:
        pen = QPen(QColor(color))
        pen.setWidthF(width)

        style_map = {
            'solid': Qt.PenStyle.SolidLine,
            'dash': Qt.PenStyle.DashLine,
            'dot': Qt.PenStyle.DotLine,
            'dashdot': Qt.PenStyle.DashDotLine
        }
        pen.setStyle(style_map.get(style, Qt.PenStyle.SolidLine))

        cap_map = {
            'round': Qt.PenCapStyle.RoundCap,
            'square': Qt.PenCapStyle.SquareCap,
            'flat': Qt.PenCapStyle.FlatCap
        }
        pen.setCapStyle(cap_map.get(cap, Qt.PenCapStyle.RoundCap))

        join_map = {
            'round': Qt.PenJoinStyle.RoundJoin,
            'bevel': Qt.PenJoinStyle.BevelJoin,
            'miter': Qt.PenJoinStyle.MiterJoin
        }
        pen.setJoinStyle(join_map.get(join, Qt.PenJoinStyle.RoundJoin))

        return pen

    @staticmethod
    def create_brush(color: str, opacity: float = 1.0, pattern: str = 'solid') -> QBrush:
        qcolor = QColor(color)
        qcolor.setAlphaF(opacity)

        pattern_map = {
            'solid': Qt.BrushStyle.SolidPattern,
            'dense1': Qt.BrushStyle.Dense1Pattern,
            'dense2': Qt.BrushStyle.Dense2Pattern,
            'dense3': Qt.BrushStyle.Dense3Pattern,
            'horizontal': Qt.BrushStyle.HorPattern,
            'vertical': Qt.BrushStyle.VerPattern,
            'cross': Qt.BrushStyle.CrossPattern,
            'diagonal': Qt.BrushStyle.BDiagPattern
        }

        brush = QBrush(qcolor)
        brush.setStyle(pattern_map.get(pattern, Qt.BrushStyle.SolidPattern))
        return brush

    @staticmethod
    def interpolate_color(color1: str, color2: str, ratio: float) -> str:
        c1 = QColor(color1)
        c2 = QColor(color2)

        r = int(c1.red() * (1 - ratio) + c2.red() * ratio)
        g = int(c1.green() * (1 - ratio) + c2.green() * ratio)
        b = int(c1.blue() * (1 - ratio) + c2.blue() * ratio)

        return f'#{r:02x}{g:02x}{b:02x}'

    @staticmethod
    def get_categorized_style(value: Any, categories: Dict[Any, Dict[str, Any]]) -> Dict[str, Any]:
        return categories.get(value, categories.get('default', {}))

    @staticmethod
    def get_graduated_style(value: float, breaks: List[float], colors: List[str]) -> str:
        for i, break_val in enumerate(breaks):
            if value <= break_val:
                return colors[i]
        return colors[-1]

    @staticmethod
    def apply_data_driven_style(feature_value: float, min_val: float, max_val: float,
                                min_size: float, max_size: float) -> float:
        if max_val == min_val:
            return min_size
        ratio = (feature_value - min_val) / (max_val - min_val)
        return min_size + ratio * (max_size - min_size)
