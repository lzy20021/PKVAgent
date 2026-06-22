"""
点要素综合算法
"""
from typing import List, Tuple, Dict, Any
import math
import random


def dbscan_cluster(points: List[Tuple[float, float]], eps: float,
                   min_samples: int) -> Dict[int, List[Tuple[float, float]]]:
    """
    DBSCAN聚类算法

    Args:
        points: 点坐标列表
        eps: 邻域半径
        min_samples: 最小样本数

    Returns:
        聚类结果 {cluster_id: [points]}
    """
    n = len(points)
    labels = [-1] * n  # -1表示噪声
    cluster_id = 0

    for i in range(n):
        if labels[i] != -1:
            continue

        neighbors = get_neighbors(points, i, eps)

        if len(neighbors) < min_samples:
            continue  # 噪声点

        # 扩展聚类
        labels[i] = cluster_id
        seed_set = list(neighbors)

        j = 0
        while j < len(seed_set):
            q = seed_set[j]
            if labels[q] == -1:
                labels[q] = cluster_id

            if labels[q] != -1 and labels[q] != cluster_id:
                j += 1
                continue

            labels[q] = cluster_id
            q_neighbors = get_neighbors(points, q, eps)

            if len(q_neighbors) >= min_samples:
                seed_set.extend([n for n in q_neighbors if n not in seed_set])

            j += 1

        cluster_id += 1

    # 整理结果
    clusters = {}
    for i, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(points[i])

    return clusters


def get_neighbors(points: List[Tuple[float, float]], idx: int,
                  eps: float) -> List[int]:
    """获取邻域内的点索引"""
    neighbors = []
    p = points[idx]

    for i, q in enumerate(points):
        if i != idx:
            dist = math.sqrt((p[0] - q[0])**2 + (p[1] - q[1])**2)
            if dist <= eps:
                neighbors.append(i)

    return neighbors


def select_points_random(points: List[Tuple[float, float]],
                         ratio: float) -> List[Tuple[float, float]]:
    """
    随机选取点

    Args:
        points: 点列表
        ratio: 保留比例

    Returns:
        选取后的点列表
    """
    n = int(len(points) * ratio)
    return random.sample(points, min(n, len(points)))


def select_points_grid(points: List[Tuple[float, float]], ratio: float,
                       grid_size: float = None) -> List[Tuple[float, float]]:
    """
    网格选取点 - 每个网格保留一个点

    Args:
        points: 点列表
        ratio: 保留比例
        grid_size: 网格大小

    Returns:
        选取后的点列表
    """
    if not points:
        return []

    # 计算边界
    min_x = min(p[0] for p in points)
    max_x = max(p[0] for p in points)
    min_y = min(p[1] for p in points)
    max_y = max(p[1] for p in points)

    # 计算网格大小
    if grid_size is None:
        area = (max_x - min_x) * (max_y - min_y)
        target_cells = len(points) * ratio
        grid_size = math.sqrt(area / target_cells) if target_cells > 0 else 1

    # 分配到网格
    grid = {}
    for p in points:
        cell_x = int((p[0] - min_x) / grid_size)
        cell_y = int((p[1] - min_y) / grid_size)
        key = (cell_x, cell_y)

        if key not in grid:
            grid[key] = p

    return list(grid.values())


def select_points_importance(points: List[Tuple[float, float]],
                             weights: List[float],
                             ratio: float) -> List[Tuple[float, float]]:
    """
    按重要性选取点

    Args:
        points: 点列表
        weights: 重要性权重
        ratio: 保留比例

    Returns:
        选取后的点列表
    """
    if len(weights) != len(points):
        return select_points_random(points, ratio)

    # 按权重排序
    indexed = list(zip(points, weights))
    indexed.sort(key=lambda x: x[1], reverse=True)

    n = int(len(points) * ratio)
    return [p for p, w in indexed[:n]]


def displace_points(points: List[Tuple[float, float]],
                    min_distance: float) -> List[Tuple[float, float]]:
    """
    点位移 - 解决点重叠

    Args:
        points: 点列表
        min_distance: 最小距离

    Returns:
        位移后的点列表
    """
    result = list(points)
    max_iterations = 100

    for _ in range(max_iterations):
        moved = False

        for i in range(len(result)):
            for j in range(i + 1, len(result)):
                p1, p2 = result[i], result[j]
                dist = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

                if dist < min_distance and dist > 0:
                    # 计算位移
                    dx = (p2[0] - p1[0]) / dist
                    dy = (p2[1] - p1[1]) / dist
                    move = (min_distance - dist) / 2

                    result[i] = (p1[0] - dx * move, p1[1] - dy * move)
                    result[j] = (p2[0] + dx * move, p2[1] + dy * move)
                    moved = True

        if not moved:
            break

    return result


def calculate_point_density(points: List[Tuple[float, float]],
                            radius: float) -> List[float]:
    """
    计算点密度

    Args:
        points: 点列表
        radius: 搜索半径

    Returns:
        每个点的密度值
    """
    densities = []

    for i, p in enumerate(points):
        count = 0
        for j, q in enumerate(points):
            if i != j:
                dist = math.sqrt((p[0] - q[0])**2 + (p[1] - q[1])**2)
                if dist <= radius:
                    count += 1
        densities.append(count)

    return densities


def cluster_centroid(points: List[Tuple[float, float]]) -> Tuple[float, float]:
    """计算点集质心"""
    if not points:
        return (0, 0)

    x = sum(p[0] for p in points) / len(points)
    y = sum(p[1] for p in points) / len(points)

    return (x, y)
