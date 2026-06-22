"""
面要素综合算法
"""
from typing import List, Tuple, Dict, Any
import math


def simplify_polygon(coords: List[Tuple[float, float]], tolerance: float,
                     preserve_angles: bool = True) -> List[Tuple[float, float]]:
    """
    多边形简化

    Args:
        coords: 多边形坐标
        tolerance: 容差
        preserve_angles: 是否保持直角

    Returns:
        简化后的坐标
    """
    from .line import douglas_peucker

    # 先用DP算法简化
    simplified = douglas_peucker(coords, tolerance)

    if preserve_angles:
        # 直角化处理
        simplified = orthogonalize(simplified, angle_threshold=15)

    return simplified


def orthogonalize(coords: List[Tuple[float, float]],
                  angle_threshold: float = 15) -> List[Tuple[float, float]]:
    """
    直角化处理 - 将接近直角的角度调整为90度

    Args:
        coords: 坐标点列表
        angle_threshold: 角度阈值（度）

    Returns:
        直角化后的坐标
    """
    if len(coords) < 3:
        return coords

    result = list(coords)

    for i in range(1, len(result) - 1):
        p1 = result[i - 1]
        p2 = result[i]
        p3 = result[i + 1]

        angle = calculate_angle(p1, p2, p3)

        # 如果接近90度，调整为90度
        if abs(angle - 90) < angle_threshold:
            # 调整p2的位置使角度为90度
            result[i] = adjust_to_right_angle(p1, p2, p3)

    return result


def calculate_angle(p1: Tuple[float, float], p2: Tuple[float, float],
                    p3: Tuple[float, float]) -> float:
    """计算三点形成的角度（度）"""
    v1 = (p1[0] - p2[0], p1[1] - p2[1])
    v2 = (p3[0] - p2[0], p3[1] - p2[1])

    dot = v1[0] * v2[0] + v1[1] * v2[1]
    mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
    mag2 = math.sqrt(v2[0]**2 + v2[1]**2)

    if mag1 == 0 or mag2 == 0:
        return 0

    cos_angle = max(-1, min(1, dot / (mag1 * mag2)))
    return math.degrees(math.acos(cos_angle))


def adjust_to_right_angle(p1: Tuple[float, float], p2: Tuple[float, float],
                          p3: Tuple[float, float]) -> Tuple[float, float]:
    """调整中间点使角度为90度"""
    # 简化实现：保持p2在p1-p3连线的垂直方向上
    v1 = (p1[0] - p2[0], p1[1] - p2[1])
    len1 = math.sqrt(v1[0]**2 + v1[1]**2)

    if len1 == 0:
        return p2

    # 单位向量
    u1 = (v1[0] / len1, v1[1] / len1)

    # 垂直向量
    perp = (-u1[1], u1[0])

    # 计算p3到p1-p2线的投影
    v3 = (p3[0] - p2[0], p3[1] - p2[1])
    proj = v3[0] * perp[0] + v3[1] * perp[1]

    # 新的p2位置
    return p2


def merge_adjacent_polygons(polygons: List[List[Tuple[float, float]]],
                            distance_threshold: float) -> List[List[Tuple[float, float]]]:
    """
    合并相邻多边形

    Args:
        polygons: 多边形列表
        distance_threshold: 距离阈值

    Returns:
        合并后的多边形列表
    """
    if len(polygons) <= 1:
        return polygons

    # 构建邻接关系
    merged = []
    used = set()

    for i, poly1 in enumerate(polygons):
        if i in used:
            continue

        current_group = [poly1]
        used.add(i)

        for j, poly2 in enumerate(polygons):
            if j in used:
                continue

            if polygons_are_adjacent(poly1, poly2, distance_threshold):
                current_group.append(poly2)
                used.add(j)

        # 合并组内多边形
        if len(current_group) > 1:
            merged.append(union_polygons(current_group))
        else:
            merged.append(poly1)

    return merged


def polygons_are_adjacent(poly1: List[Tuple[float, float]],
                          poly2: List[Tuple[float, float]],
                          threshold: float) -> bool:
    """判断两个多边形是否相邻"""
    for p1 in poly1:
        for p2 in poly2:
            dist = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
            if dist < threshold:
                return True
    return False


def union_polygons(polygons: List[List[Tuple[float, float]]]) -> List[Tuple[float, float]]:
    """合并多个多边形（简化实现，返回凸包）"""
    all_points = []
    for poly in polygons:
        all_points.extend(poly)

    return convex_hull(all_points)


def convex_hull(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """计算凸包"""
    if len(points) < 3:
        return points

    # 按x坐标排序
    sorted_points = sorted(points, key=lambda p: (p[0], p[1]))

    # 构建下凸包
    lower = []
    for p in sorted_points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    # 构建上凸包
    upper = []
    for p in reversed(sorted_points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return lower[:-1] + upper[:-1]


def cross(o: Tuple[float, float], a: Tuple[float, float],
          b: Tuple[float, float]) -> float:
    """计算叉积"""
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def displace_polygon(polygon: List[Tuple[float, float]],
                     direction: Tuple[float, float],
                     distance: float) -> List[Tuple[float, float]]:
    """
    位移多边形

    Args:
        polygon: 多边形坐标
        direction: 位移方向（单位向量）
        distance: 位移距离

    Returns:
        位移后的坐标
    """
    dx = direction[0] * distance
    dy = direction[1] * distance

    return [(x + dx, y + dy) for x, y in polygon]


def calculate_displacement_vector(poly1: List[Tuple[float, float]],
                                  poly2: List[Tuple[float, float]],
                                  min_gap: float) -> Tuple[float, float]:
    """
    计算两个多边形之间的位移向量

    Args:
        poly1: 第一个多边形
        poly2: 第二个多边形
        min_gap: 最小间隙

    Returns:
        位移向量 (dx, dy)
    """
    # 计算质心
    c1 = polygon_centroid(poly1)
    c2 = polygon_centroid(poly2)

    # 方向向量
    dx = c2[0] - c1[0]
    dy = c2[1] - c1[1]
    dist = math.sqrt(dx**2 + dy**2)

    if dist == 0:
        return (min_gap, 0)

    # 单位向量
    ux, uy = dx / dist, dy / dist

    return (ux * min_gap, uy * min_gap)


def polygon_centroid(polygon: List[Tuple[float, float]]) -> Tuple[float, float]:
    """计算多边形质心"""
    if not polygon:
        return (0, 0)

    x_sum = sum(p[0] for p in polygon)
    y_sum = sum(p[1] for p in polygon)
    n = len(polygon)

    return (x_sum / n, y_sum / n)


def polygon_area(polygon: List[Tuple[float, float]]) -> float:
    """计算多边形面积（Shoelace公式）"""
    n = len(polygon)
    if n < 3:
        return 0

    area = 0
    for i in range(n):
        j = (i + 1) % n
        area += polygon[i][0] * polygon[j][1]
        area -= polygon[j][0] * polygon[i][1]

    return abs(area) / 2
