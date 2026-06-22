"""
线要素综合算法
"""
from typing import List, Tuple
import math


def douglas_peucker(coords: List[Tuple[float, float]], tolerance: float) -> List[Tuple[float, float]]:
    """
    Douglas-Peucker线简化算法

    Args:
        coords: 坐标点列表 [(x1,y1), (x2,y2), ...]
        tolerance: 容差值

    Returns:
        简化后的坐标点列表
    """
    if len(coords) <= 2:
        return coords

    # 找到距离首尾连线最远的点
    start, end = coords[0], coords[-1]
    max_dist = 0
    max_idx = 0

    for i in range(1, len(coords) - 1):
        dist = point_to_line_distance(coords[i], start, end)
        if dist > max_dist:
            max_dist = dist
            max_idx = i

    # 如果最大距离大于容差，递归处理
    if max_dist > tolerance:
        left = douglas_peucker(coords[:max_idx + 1], tolerance)
        right = douglas_peucker(coords[max_idx:], tolerance)
        return left[:-1] + right
    else:
        return [start, end]


def point_to_line_distance(point: Tuple[float, float],
                           line_start: Tuple[float, float],
                           line_end: Tuple[float, float]) -> float:
    """计算点到线段的距离"""
    x, y = point
    x1, y1 = line_start
    x2, y2 = line_end

    # 线段长度的平方
    line_len_sq = (x2 - x1) ** 2 + (y2 - y1) ** 2

    if line_len_sq == 0:
        return math.sqrt((x - x1) ** 2 + (y - y1) ** 2)

    # 投影参数
    t = max(0, min(1, ((x - x1) * (x2 - x1) + (y - y1) * (y2 - y1)) / line_len_sq))

    # 投影点
    proj_x = x1 + t * (x2 - x1)
    proj_y = y1 + t * (y2 - y1)

    return math.sqrt((x - proj_x) ** 2 + (y - proj_y) ** 2)


def visvalingam_whyatt(coords: List[Tuple[float, float]], threshold: float) -> List[Tuple[float, float]]:
    """
    Visvalingam-Whyatt线简化算法
    基于有效面积的简化方法

    Args:
        coords: 坐标点列表
        threshold: 面积阈值

    Returns:
        简化后的坐标点列表
    """
    if len(coords) <= 2:
        return coords

    # 计算每个点的有效面积
    points = list(coords)
    areas = [float('inf')] + [
        triangle_area(points[i-1], points[i], points[i+1])
        for i in range(1, len(points) - 1)
    ] + [float('inf')]

    while len(points) > 2:
        # 找到最小面积的点
        min_area = min(areas[1:-1])
        if min_area >= threshold:
            break

        min_idx = areas.index(min_area)

        # 移除该点
        points.pop(min_idx)
        areas.pop(min_idx)

        # 更新相邻点的面积
        if min_idx > 1 and min_idx < len(points):
            areas[min_idx - 1] = triangle_area(
                points[min_idx - 2], points[min_idx - 1], points[min_idx]
            ) if min_idx - 2 >= 0 and min_idx < len(points) else float('inf')

        if min_idx < len(points) - 1 and min_idx > 0:
            areas[min_idx] = triangle_area(
                points[min_idx - 1], points[min_idx], points[min_idx + 1]
            ) if min_idx + 1 < len(points) else float('inf')

    return points


def triangle_area(p1: Tuple[float, float], p2: Tuple[float, float],
                  p3: Tuple[float, float]) -> float:
    """计算三角形面积"""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    return abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2)


def smooth_bezier(coords: List[Tuple[float, float]], smoothness: float = 0.5,
                  num_points: int = 10) -> List[Tuple[float, float]]:
    """
    贝塞尔曲线平滑

    Args:
        coords: 原始坐标点
        smoothness: 平滑程度 (0-1)
        num_points: 每段插值点数

    Returns:
        平滑后的坐标点
    """
    if len(coords) < 3:
        return coords

    result = [coords[0]]

    for i in range(len(coords) - 1):
        p0 = coords[max(0, i - 1)]
        p1 = coords[i]
        p2 = coords[min(len(coords) - 1, i + 1)]
        p3 = coords[min(len(coords) - 1, i + 2)]

        # 计算控制点
        ctrl1 = (
            p1[0] + smoothness * (p2[0] - p0[0]) / 6,
            p1[1] + smoothness * (p2[1] - p0[1]) / 6
        )
        ctrl2 = (
            p2[0] - smoothness * (p3[0] - p1[0]) / 6,
            p2[1] - smoothness * (p3[1] - p1[1]) / 6
        )

        # 生成贝塞尔曲线点
        for t in range(1, num_points + 1):
            t_norm = t / num_points
            point = cubic_bezier(p1, ctrl1, ctrl2, p2, t_norm)
            result.append(point)

    return result


def cubic_bezier(p0: Tuple[float, float], p1: Tuple[float, float],
                 p2: Tuple[float, float], p3: Tuple[float, float],
                 t: float) -> Tuple[float, float]:
    """三次贝塞尔曲线插值"""
    t2 = t * t
    t3 = t2 * t
    mt = 1 - t
    mt2 = mt * mt
    mt3 = mt2 * mt

    x = mt3 * p0[0] + 3 * mt2 * t * p1[0] + 3 * mt * t2 * p2[0] + t3 * p3[0]
    y = mt3 * p0[1] + 3 * mt2 * t * p1[1] + 3 * mt * t2 * p2[1] + t3 * p3[1]

    return (x, y)
