from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors


DEFAULT_COLORS = {
    "便利店": "#d23b7d",
    "学校": "#2e7d32",
    "停车场": "#1976d2",
    "酒店": "#7b4fa1",
    "银行": "#00897b",
    "医院": "#d32f2f",
    "餐饮": "#ef8a24",
    "交通设施": "#4e79a7",
    "其他": "#7f8c8d",
}

POI_CATEGORY_ALIASES = {
    "便利": "便利店",
    "超市": "便利店",
    "商店": "便利店",
    "大学": "学校",
    "中学": "学校",
    "小学": "学校",
    "幼儿园": "学校",
    "停车": "停车场",
    "宾馆": "酒店",
    "住宿": "酒店",
    "ATM": "银行",
    "atm": "银行",
    "诊所": "医院",
    "医疗": "医院",
    "餐厅": "餐饮",
    "饭店": "餐饮",
    "美食": "餐饮",
    "交通": "交通设施",
    "公交站": "交通设施",
    "公交": "交通设施",
    "地铁站": "交通设施",
    "地铁": "交通设施",
    "客运站": "交通设施",
    "火车站": "交通设施",
}


@dataclass
class SymbolRule:
    category: str
    symbol_size: float
    min_distance: float
    priority: float
    preserve_floor: int
    color: str
    marker: str = "o"

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def canonicalize_poi_category(value: object) -> str:
    text = str(value or "").strip()
    if not text or text.lower() == "nan":
        return "其他"
    if text in DEFAULT_COLORS:
        return text
    if text in POI_CATEGORY_ALIASES:
        return POI_CATEGORY_ALIASES[text]
    for key, category in POI_CATEGORY_ALIASES.items():
        if key and key in text:
            return category
    return text


def build_default_symbol_rules(categories: Sequence[str]) -> Dict[str, SymbolRule]:
    base = {
        "便利店": SymbolRule("便利店", 0.58, 0.48, 0.86, 1, DEFAULT_COLORS["便利店"], "convenience"),
        "学校": SymbolRule("学校", 0.62, 0.50, 1.15, 2, DEFAULT_COLORS["学校"], "school"),
        "停车场": SymbolRule("停车场", 0.58, 0.48, 0.88, 1, DEFAULT_COLORS["停车场"], "parking"),
        "酒店": SymbolRule("酒店", 0.60, 0.50, 0.95, 1, DEFAULT_COLORS["酒店"], "hotel"),
        "银行": SymbolRule("银行", 0.58, 0.50, 1.00, 1, DEFAULT_COLORS["银行"], "bank"),
        "医院": SymbolRule("医院", 0.64, 0.54, 1.22, 2, DEFAULT_COLORS["医院"], "hospital"),
        "餐饮": SymbolRule("餐饮", 0.60, 0.48, 0.90, 1, DEFAULT_COLORS["餐饮"], "restaurant"),
        "交通设施": SymbolRule("交通设施", 0.60, 0.50, 1.05, 1, DEFAULT_COLORS["交通设施"], "transit"),
        "其他": SymbolRule("其他", 0.92, 0.60, 0.82, 1, DEFAULT_COLORS["其他"], "o"),
    }
    fallback_key = list(base.keys())[-1]
    fallback_rule = base[fallback_key]
    rules = {}
    for category in categories:
        canonical = canonicalize_poi_category(category)
        rule = base.get(canonical, fallback_rule)
        rules[category] = SymbolRule(
            str(category),
            rule.symbol_size,
            rule.min_distance,
            rule.priority,
            rule.preserve_floor,
            rule.color,
            rule.marker,
        )
    rules.setdefault(fallback_key, fallback_rule)
    return rules
def build_skill_blueprint() -> Dict[str, object]:
    return {
        "name": "point_generalization_skills_dlm_dcm",
        "purpose": "DLM/DCM 双层驱动的多类别 POI 点综合技能编排",
        "layers": {
            "DLM": {
                "description": "识别点类别、重要性、聚类、线性、多核、稀疏密集差异与跨类别共现结构。",
                "skills": [
                    {
                        "name": "representative_poi_retain",
                        "purpose": "通过频次逆权重、中心性、局部代表性和名称代理指标构造重要性。",
                        "triggers": ["存在多类别 POI", "缺少显式重要性字段"],
                        "inputs": ["poi_uid", "name", "category", "geometry"],
                        "logic": [
                            "计算类别稀有度、kNN 中心性、局部锚点贡献和聚类代表性。",
                            "生成 importance_score，并标记 minority_category。"
                        ],
                        "outputs": ["importance_score", "minority_category", "cluster_id"],
                        "priority": 1,
                    },
                    {
                        "name": "structure_aware_selection",
                        "purpose": "按 clustered/linear/multi_core/sparse/dense 结构采用不同保留策略。",
                        "triggers": ["局部聚类明显", "存在线性走廊", "多中心城市结构"],
                        "inputs": ["projected geometry", "density", "cluster_id", "importance_score"],
                        "logic": [
                            "DBSCAN 识别聚类，PCA 识别线性结构，多核由聚类主中心判断。",
                            "团簇保留中心+轮廓锚点，线性保留分段骨架，稀疏区设覆盖底线。"
                        ],
                        "outputs": ["structure_type", "core_rank", "structure_bonus"],
                        "priority": 2,
                    },
                    {
                        "name": "category_balance_preserve",
                        "purpose": "在局部窗口内维持多类别共存并保护低频类别。",
                        "triggers": ["局部区域原本存在多类别", "低频类别存在被高频类别淹没风险"],
                        "inputs": ["grid_id", "category", "importance_score", "preserve_floor"],
                        "logic": [
                            "先为每个局部窗口按类别预留候选点。",
                            "对 minority 类别执行最低保留底线，避免类别塌缩。"
                        ],
                        "outputs": ["guaranteed_candidates", "local_diversity_target"],
                        "priority": 3,
                    },
                ],
            },
            "DCM": {
                "description": "把符号尺寸、最小间距、异类冲突、视觉均衡和局部协调显式纳入决策。",
                "skills": [
                    {
                        "name": "symbol_size_aware_generalization",
                        "purpose": "以符号真实占位而非裸点距离判断冲突。",
                        "triggers": ["目标比例尺变化", "类别符号尺寸不同"],
                        "inputs": ["symbol_size", "min_distance", "target_scale", "geometry"],
                        "logic": [
                            "把 mm 尺寸转换为地面米尺度，形成 symbol_radius_m。",
                            "冲突检测按符号半径和最小间距合成阈值。"
                        ],
                        "outputs": ["symbol_radius_m", "symbol_conflict_pairs"],
                        "priority": 4,
                    },
                    {
                        "name": "cross_category_conflict_resolution",
                        "purpose": "消解跨类别与同类别符号冲突，同时保留类别多样性。",
                        "triggers": ["存在符号重叠", "异类符号冲突明显"],
                        "inputs": ["selected candidates", "symbol rules", "diversity constraints"],
                        "logic": [
                            "基于得分、类别保障、局部多样性和结构角色做贪心选择。",
                            "对受保障但冲突的点尝试局部位移，而不是直接删掉弱势类别。"
                        ],
                        "outputs": ["selected features", "resolved_conflicts", "displacement_m"],
                        "priority": 5,
                    },
                    {
                        "name": "layout_aesthetic_optimization",
                        "purpose": "改善局部过密、视觉偏斜和大面积空白，输出论文级图面。",
                        "triggers": ["图面局部拥挤", "类别空间失衡", "存在大块空白"],
                        "inputs": ["selected features", "grid statistics", "symbol rules"],
                        "logic": [
                            "评价网格均衡度、类别覆盖度和局部冲突。",
                            "迭代微调位置并补充缺失类别代表点，直到达到平衡阈值。"
                        ],
                        "outputs": ["layout_balance_index", "final cartographic placement"],
                        "priority": 6,
                    },
                ],
            },
        },
    }


class PointGeneralizationEngine:
    def __init__(self, symbol_rules: Dict[str, SymbolRule]):
        self.symbol_rules = symbol_rules

    @staticmethod
    def normalize_category(value: object) -> str:
        text = str(value or "").strip()
        if not text or text.lower() == "nan":
            return "其他"
        return text

    @staticmethod
    def metric_radius_from_rule(rule: SymbolRule, target_scale: int) -> float:
        return (rule.symbol_size * target_scale / 1000.0) * 0.5

    @staticmethod
    def metric_gap_from_rule(rule: SymbolRule, target_scale: int) -> float:
        return rule.min_distance * target_scale / 1000.0

    @staticmethod
    def _score_name(name: object) -> float:
        text = str(name or "")
        keywords = ["广场", "中心", "总店", "医院", "大学", "地铁", "公园", "站", "大厦"]
        matched = sum(1 for item in keywords if item in text)
        return min(1.0, 0.25 + matched * 0.12)

    def analyze_dlm(self, gdf_metric: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        work = gdf_metric.copy()
        coords = np.column_stack([work.geometry.x.to_numpy(), work.geometry.y.to_numpy()])
        if len(work) == 0:
            return work

        n_neighbors = min(8, len(work))
        nn = NearestNeighbors(n_neighbors=n_neighbors)
        nn.fit(coords)
        distances, indices = nn.kneighbors(coords)
        kth = distances[:, -1] if distances.shape[1] > 1 else np.zeros(len(work))
        eps = float(np.quantile(kth, 0.75)) if len(kth) > 2 else 250.0
        eps = max(120.0, min(450.0, eps))
        clustering = DBSCAN(eps=eps, min_samples=min(4, len(work))).fit(coords)
        work["cluster_id"] = clustering.labels_

        radius = max(220.0, min(500.0, float(np.quantile(kth, 0.8)) if len(kth) > 2 else 300.0))
        density_nbrs = NearestNeighbors(radius=radius)
        density_nbrs.fit(coords)
        neighborhoods = density_nbrs.radius_neighbors(coords, return_distance=False)
        local_density = np.array([len(ids) for ids in neighborhoods], dtype=float)
        work["local_density"] = local_density
        dense_q = float(np.quantile(local_density, 0.7)) if len(local_density) > 2 else float(local_density.max())
        sparse_q = float(np.quantile(local_density, 0.3)) if len(local_density) > 2 else float(local_density.min())
        work["density_band"] = np.where(
            local_density >= dense_q,
            "dense",
            np.where(local_density <= sparse_q, "sparse", "medium"),
        )

        mean_dist = distances[:, 1:].mean(axis=1) if distances.shape[1] > 1 else np.full(len(work), 1.0)
        centrality = 1.0 / np.maximum(mean_dist, 1.0)
        work["centrality"] = centrality / centrality.max()

        cat_freq = work["category"].value_counts()
        rarity = work["category"].map(lambda c: 1.0 / math.sqrt(float(cat_freq.get(c, 1))))
        work["rarity_score"] = rarity / float(rarity.max())

        cluster_sizes = work["cluster_id"].value_counts().to_dict()
        global_centroid = np.array([coords[:, 0].mean(), coords[:, 1].mean()])
        work["representative_score"] = 0.0
        work["structure_type"] = "scattered"
        work["multi_core_flag"] = False
        core_count = 0
        cluster_centroids: Dict[int, np.ndarray] = {}
        for cluster_id, group in work.groupby("cluster_id"):
            if cluster_id == -1:
                continue
            group_coords = np.column_stack([group.geometry.x.to_numpy(), group.geometry.y.to_numpy()])
            centroid = np.array([group_coords[:, 0].mean(), group_coords[:, 1].mean()])
            cluster_centroids[int(cluster_id)] = centroid
            centered = group_coords - centroid
            if len(group_coords) >= 3:
                cov = np.cov(centered.T)
                eigvals = np.sort(np.real(np.linalg.eigvals(cov)))[::-1]
                ratio = float(eigvals[0] / max(eigvals[-1], 1.0))
            else:
                ratio = 1.0
            structure_type = "linear" if ratio >= 6.0 else "clustered"
            if np.linalg.norm(centroid - global_centroid) > 1200.0:
                core_count += 1
            dist_to_centroid = np.linalg.norm(group_coords - centroid, axis=1)
            rep = 1.0 - (dist_to_centroid / max(dist_to_centroid.max(), 1.0))
            work.loc[group.index, "representative_score"] = rep
            work.loc[group.index, "structure_type"] = structure_type

        work.loc[work["cluster_id"] == -1, "representative_score"] = 0.85
        work.loc[work["cluster_id"] == -1, "structure_type"] = np.where(
            work.loc[work["cluster_id"] == -1, "density_band"] == "sparse",
            "sparse",
            "scattered",
        )
        work["multi_core_flag"] = core_count >= 2

        centroid_points = np.array(list(cluster_centroids.values())) if cluster_centroids else np.empty((0, 2))
        if len(centroid_points) >= 2:
            centroid_dist = np.linalg.norm(centroid_points - centroid_points.mean(axis=0), axis=1)
            work["multi_core_flag"] = bool(np.count_nonzero(centroid_dist > 1000.0) >= 2)

        grid_size = max(300.0, min(800.0, radius * 1.25))
        work["grid_x"] = np.floor((work.geometry.x - work.geometry.x.min()) / grid_size).astype(int)
        work["grid_y"] = np.floor((work.geometry.y - work.geometry.y.min()) / grid_size).astype(int)
        work["grid_id"] = work["grid_x"].astype(str) + "_" + work["grid_y"].astype(str)
        grid_stats = work.groupby("grid_id")["category"].agg(lambda s: sorted(set(s))).to_dict()
        work["local_category_count"] = work["grid_id"].map(lambda gid: len(grid_stats.get(gid, [])))
        minority_threshold = float(cat_freq.quantile(0.4)) if len(cat_freq) > 1 else float(cat_freq.iloc[0])
        work["minority_category"] = work["category"].map(lambda c: cat_freq.get(c, 0) <= minority_threshold)

        name_score = work["name"].map(self._score_name)
        structure_bonus_map = {"clustered": 0.80, "linear": 0.95, "sparse": 1.00, "scattered": 0.75}
        structure_bonus = work["structure_type"].map(lambda v: structure_bonus_map.get(v, 0.75))
        dense_penalty = np.where(work["density_band"] == "dense", 0.65, np.where(work["density_band"] == "medium", 0.82, 1.0))

        work["importance_score"] = (
            0.28 * work["centrality"]
            + 0.20 * work["rarity_score"]
            + 0.18 * work["representative_score"]
            + 0.16 * name_score
            + 0.18 * structure_bonus
        ) * dense_penalty
        work["importance_score"] = work["importance_score"] / work["importance_score"].max()
        work["structure_bonus"] = structure_bonus
        return work

    def _build_quota(self, work: gpd.GeoDataFrame, target_scale: int) -> Dict[str, int]:
        raw_count = len(work)
        if target_scale <= 150000:
            ratio = 0.42
        elif target_scale <= 250000:
            ratio = 0.28
        else:
            ratio = 0.20
        target_keep = max(18, int(math.ceil(raw_count * ratio)))
        per_category = work["category"].value_counts().to_dict()
        quotas: Dict[str, int] = {}
        for category, count in per_category.items():
            rule = self.symbol_rules.get(category, self.symbol_rules["其他"])
            quota = max(rule.preserve_floor, int(round(target_keep * count / raw_count)))
            if count <= 4:
                quota = max(quota, 1)
            if work.loc[work["category"] == category, "minority_category"].any():
                quota = max(quota, rule.preserve_floor + 1)
            quotas[category] = min(count, quota)
        return quotas

    def _structure_candidates(self, work: gpd.GeoDataFrame) -> pd.Index:
        selected: List[int] = []
        for _, group in work.groupby("cluster_id"):
            ordered = group.sort_values(["importance_score", "representative_score"], ascending=False)
            if len(group) <= 3:
                selected.extend(ordered.index.tolist())
                continue
            if str(group["structure_type"].iloc[0]) == "linear":
                centroid = np.array([group.geometry.x.mean(), group.geometry.y.mean()])
                coords = np.column_stack([group.geometry.x.to_numpy(), group.geometry.y.to_numpy()]) - centroid
                _, _, vh = np.linalg.svd(coords, full_matrices=False)
                axis = vh[0]
                proj = coords @ axis
                bins = np.linspace(proj.min(), proj.max(), min(6, len(group)))
                group = group.assign(axis_proj=proj)
                for start, end in zip(bins[:-1], bins[1:]):
                    subset = group[(group["axis_proj"] >= start) & (group["axis_proj"] <= end)]
                    if not subset.empty:
                        selected.append(int(subset.sort_values("importance_score", ascending=False).index[0]))
            else:
                selected.append(int(ordered.index[0]))
                if len(group) >= 8:
                    selected.extend(ordered.index[1:3].tolist())
                xs = group.geometry.x
                ys = group.geometry.y
                anchor_ids = {
                    int(xs.idxmin()),
                    int(xs.idxmax()),
                    int(ys.idxmin()),
                    int(ys.idxmax()),
                }
                selected.extend(list(anchor_ids))
        return pd.Index(sorted(set(selected)))

    def _seed_selection(self, work: gpd.GeoDataFrame, target_scale: int) -> gpd.GeoDataFrame:
        quotas = self._build_quota(work, target_scale)
        chosen: List[int] = []
        local_targets = work.groupby("grid_id")["category"].agg(lambda s: len(set(s)))
        for grid_id, group in work.groupby("grid_id"):
            desired_diversity = min(int(local_targets.get(grid_id, 1)), 3)
            by_category = group.groupby("category")
            winners = []
            for category, cat_group in by_category:
                winners.append(cat_group.sort_values("importance_score", ascending=False).index[0])
            ordered = group.loc[sorted(set(winners))].sort_values(
                ["minority_category", "importance_score"],
                ascending=[False, False],
            )
            chosen.extend(ordered.index[:desired_diversity].tolist())

        chosen.extend(self._structure_candidates(work).tolist())

        for category, quota in quotas.items():
            cat_group = work[work["category"] == category].sort_values(
                ["minority_category", "importance_score", "representative_score"],
                ascending=[False, False, False],
            )
            chosen.extend(cat_group.index[:quota].tolist())

        seeds = work.loc[sorted(set(chosen))].copy()
        seeds["guaranteed"] = False
        for grid_id, group in work.groupby("grid_id"):
            if group["category"].nunique() > 1:
                winners = group.sort_values(["minority_category", "importance_score"], ascending=[False, False]).groupby("category").head(1)
                winner_idx = [idx for idx in winners.index if idx in seeds.index]
                if winner_idx:
                    seeds.loc[winner_idx, "guaranteed"] = True
        for category, rule in self.symbol_rules.items():
            cat_idx = seeds.index[seeds["category"] == category]
            if len(cat_idx) > 0:
                seeds.loc[cat_idx[:rule.preserve_floor], "guaranteed"] = True
        return seeds

    def _conflict_threshold(self, row_a: pd.Series, row_b: pd.Series, target_scale: int) -> float:
        rule_a = self.symbol_rules.get(str(row_a["category"]), self.symbol_rules["其他"])
        rule_b = self.symbol_rules.get(str(row_b["category"]), self.symbol_rules["其他"])
        radius = self.metric_radius_from_rule(rule_a, target_scale) + self.metric_radius_from_rule(rule_b, target_scale)
        gap = max(self.metric_gap_from_rule(rule_a, target_scale), self.metric_gap_from_rule(rule_b, target_scale))
        return radius + gap

    def count_symbol_conflicts(self, work: gpd.GeoDataFrame, target_scale: int) -> int:
        if len(work) <= 1:
            return 0
        coords = np.column_stack([work.geometry.x.to_numpy(), work.geometry.y.to_numpy()])
        nbrs = NearestNeighbors(radius=1200.0)
        nbrs.fit(coords)
        neighbors = nbrs.radius_neighbors(coords, return_distance=False)
        conflicts = 0
        for idx, ids in enumerate(neighbors):
            row_a = work.iloc[idx]
            for other in ids:
                if other <= idx:
                    continue
                row_b = work.iloc[other]
                if row_a.geometry.distance(row_b.geometry) < self._conflict_threshold(row_a, row_b, target_scale):
                    conflicts += 1
        return conflicts

    def _utility(self, row: pd.Series, selected: gpd.GeoDataFrame, raw_grid_stats: Dict[str, int]) -> float:
        grid_id = str(row["grid_id"])
        local_selected = selected[selected["grid_id"] == grid_id]
        local_categories = set(local_selected["category"])
        diversity_bonus = 0.20 if row["category"] not in local_categories else -0.05
        minority_bonus = 0.16 if bool(row["minority_category"]) else 0.0
        sparse_bonus = 0.12 if str(row["density_band"]) == "sparse" else 0.0
        vacancy_bonus = 0.08 if len(local_selected) < raw_grid_stats.get(grid_id, 1) else 0.0
        guaranteed_bonus = 0.25 if bool(row.get("guaranteed", False)) else 0.0
        return float(row["importance_score"]) + diversity_bonus + minority_bonus + sparse_bonus + vacancy_bonus + guaranteed_bonus

    def _attempt_displacement(
        self,
        candidate: pd.Series,
        selected: gpd.GeoDataFrame,
        target_scale: int,
    ) -> Tuple[Point, float]:
        origin = candidate.geometry
        max_shift = max(80.0, self.metric_gap_from_rule(self.symbol_rules.get(str(candidate["category"]), self.symbol_rules["其他"]), target_scale) * 0.8)
        angles = np.linspace(0.0, 2.0 * math.pi, 12, endpoint=False)
        distances = [max_shift * factor for factor in (0.35, 0.7, 1.0)]
        best_geom = origin
        best_cost = float("inf")
        for distance in distances:
            for angle in angles:
                trial = Point(origin.x + math.cos(angle) * distance, origin.y + math.sin(angle) * distance)
                valid = True
                overlap_penalty = 0.0
                for _, kept in selected.iterrows():
                    threshold = self._conflict_threshold(candidate, kept, target_scale)
                    dist = trial.distance(kept.geometry)
                    if dist < threshold:
                        valid = False
                        overlap_penalty += threshold - dist
                if valid:
                    return trial, distance
                if overlap_penalty < best_cost:
                    best_geom = trial
                    best_cost = overlap_penalty
        return best_geom, max_shift

    def optimize_dcm(self, work: gpd.GeoDataFrame, target_scale: int) -> gpd.GeoDataFrame:
        raw_grid_stats = work.groupby("grid_id")["category"].agg(lambda s: min(len(set(s)), 3)).to_dict()
        seeds = self._seed_selection(work, target_scale)
        ordered = seeds.assign(
            utility=seeds.apply(lambda row: self._utility(row, seeds.iloc[0:0], raw_grid_stats), axis=1)
        ).sort_values(["guaranteed", "utility"], ascending=[False, False])

        kept_rows: List[pd.Series] = []
        selected = seeds.iloc[0:0].copy()
        for _, row in ordered.iterrows():
            if selected.empty:
                kept_rows.append(row.copy())
                selected = gpd.GeoDataFrame(kept_rows, geometry="geometry", crs=work.crs)
                continue
            conflict_indices = []
            for idx, kept in selected.iterrows():
                if row.geometry.distance(kept.geometry) < self._conflict_threshold(row, kept, target_scale):
                    conflict_indices.append(idx)
            if not conflict_indices:
                kept_rows.append(row.copy())
                selected = gpd.GeoDataFrame(kept_rows, geometry="geometry", crs=work.crs)
                continue

            if bool(row.get("guaranteed", False)):
                displaced_geom, shift = self._attempt_displacement(row, selected, target_scale)
                test_row = row.copy()
                test_row.geometry = displaced_geom
                still_conflict = False
                for _, kept in selected.iterrows():
                    if test_row.geometry.distance(kept.geometry) < self._conflict_threshold(test_row, kept, target_scale):
                        still_conflict = True
                        break
                if not still_conflict:
                    test_row["displacement_m"] = float(shift)
                    kept_rows.append(test_row)
                    selected = gpd.GeoDataFrame(kept_rows, geometry="geometry", crs=work.crs)
                    continue

            row_utility = self._utility(row, selected, raw_grid_stats)
            replaceable = []
            for idx in conflict_indices:
                kept = selected.loc[idx]
                if bool(kept.get("guaranteed", False)):
                    continue
                replaceable.append((idx, self._utility(kept, selected.drop(index=idx), raw_grid_stats)))
            if replaceable:
                weakest_idx, weakest_utility = sorted(replaceable, key=lambda item: item[1])[0]
                if row_utility > weakest_utility + 0.06:
                    selected = selected.drop(index=weakest_idx)
                    kept_rows = [candidate for candidate in kept_rows if int(candidate.name) != int(weakest_idx)]
                    kept_rows.append(row.copy())
                    selected = gpd.GeoDataFrame(kept_rows, geometry="geometry", crs=work.crs)

        selected = gpd.GeoDataFrame(kept_rows, geometry="geometry", crs=work.crs)
        if len(selected) == 0:
            return selected

        # Layout polish: backfill missing categories in grids with low diversity if conflicts permit.
        existing_ids = set(selected["poi_uid"])
        for grid_id, group in work.groupby("grid_id"):
            original_categories = set(group["category"])
            selected_group = selected[selected["grid_id"] == grid_id]
            missing_categories = [cat for cat in original_categories if cat not in set(selected_group["category"])]
            for category in missing_categories:
                candidate_rows = group[(group["category"] == category) & (~group["poi_uid"].isin(existing_ids))]
                if candidate_rows.empty:
                    continue
                candidate = candidate_rows.sort_values(["minority_category", "importance_score"], ascending=[False, False]).iloc[0].copy()
                displaced_geom, shift = self._attempt_displacement(candidate, selected, target_scale)
                candidate.geometry = displaced_geom
                okay = True
                for _, kept in selected.iterrows():
                    if candidate.geometry.distance(kept.geometry) < self._conflict_threshold(candidate, kept, target_scale):
                        okay = False
                        break
                if okay:
                    candidate["displacement_m"] = float(shift)
                    selected = gpd.GeoDataFrame(
                        pd.concat([selected, gpd.GeoDataFrame([candidate], geometry="geometry", crs=work.crs)], ignore_index=True),
                        geometry="geometry",
                        crs=work.crs,
                    )
                    existing_ids.add(str(candidate["poi_uid"]))
        return selected

    def summarize_metrics(
        self,
        original: gpd.GeoDataFrame,
        selected: gpd.GeoDataFrame,
        target_scale: int,
        sample_id: str,
    ) -> Dict[str, object]:
        def avg_nearest_neighbor(gdf: gpd.GeoDataFrame) -> float:
            if len(gdf) <= 1:
                return 0.0
            coords = np.column_stack([gdf.geometry.x.to_numpy(), gdf.geometry.y.to_numpy()])
            nn = NearestNeighbors(n_neighbors=min(2, len(gdf)))
            nn.fit(coords)
            distances, _ = nn.kneighbors(coords)
            return float(distances[:, 1].mean()) if distances.shape[1] > 1 else 0.0

        orig_categories = set(original["category"])
        selected_categories = set(selected["category"])
        category_retentions = []
        minority_categories = set(original.loc[original["minority_category"], "category"])
        for category, count in original["category"].value_counts().items():
            retained = int((selected["category"] == category).sum())
            category_retentions.append(retained / max(count, 1))

        def local_diversity_ratio(source: gpd.GeoDataFrame, subset: gpd.GeoDataFrame) -> float:
            ratios = []
            for grid_id, group in source.groupby("grid_id"):
                before = len(set(group["category"]))
                after = len(set(subset[subset["grid_id"] == grid_id]["category"]))
                if before > 0:
                    ratios.append(after / before)
            return float(np.mean(ratios)) if ratios else 0.0

        def balance_index(gdf: gpd.GeoDataFrame) -> float:
            if len(gdf) == 0:
                return 0.0
            counts = gdf.groupby("grid_id").size().to_numpy(dtype=float)
            if len(counts) <= 1:
                return 1.0
            cv = float(np.std(counts) / max(np.mean(counts), 1e-6))
            return max(0.0, 1.0 - min(cv, 1.0))

        minority_survival = 0.0
        if minority_categories:
            minority_survival = float(
                np.mean([1.0 if category in selected_categories else 0.0 for category in minority_categories])
            )

        return {
            "sample_id": sample_id,
            "target_scale": target_scale,
            "raw_count": int(len(original)),
            "generalized_count": int(len(selected)),
            "retention_rate": round(len(selected) / max(len(original), 1), 4),
            "raw_category_count": int(len(orig_categories)),
            "generalized_category_count": int(len(selected_categories)),
            "category_coverage_rate": round(len(selected_categories) / max(len(orig_categories), 1), 4),
            "category_retention_ratio": round(float(np.mean(category_retentions)) if category_retentions else 0.0, 4),
            "local_category_diversity": round(local_diversity_ratio(original, selected), 4),
            "minority_category_survival_rate": round(minority_survival, 4),
            "avg_nearest_neighbor_before_m": round(avg_nearest_neighbor(original), 2),
            "avg_nearest_neighbor_after_m": round(avg_nearest_neighbor(selected), 2),
            "symbol_conflicts_before": int(self.count_symbol_conflicts(original, target_scale)),
            "symbol_conflicts_after": int(self.count_symbol_conflicts(selected, target_scale)),
            "layout_balance_index": round(balance_index(selected), 4),
        }

    @staticmethod
    def save_symbol_rules(symbol_rules: Dict[str, SymbolRule], output_path: Path) -> None:
        payload = [rule.to_dict() for _, rule in sorted(symbol_rules.items(), key=lambda item: item[0])]
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
