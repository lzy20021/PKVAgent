from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.skills.point_generalization_dlm_dcm import (
    PointGeneralizationEngine,
    build_default_symbol_rules,
    build_skill_blueprint,
)


DATASET_DIR = Path(r"E:\GradOpening\MapGenAI\amap_poi_dataset_expanded")
OUTPUT_ROOT = Path(r"E:\GradOpening\outputs")
FALLBACK_OUTPUT_ROOT = PROJECT_ROOT / 'output' / 'dlm_dcm_exports'
FIELD_CANDIDATES = {
    "poi_uid": ["poi_uid", "id", "uid"],
    "name": ["name", "名称"],
    "type": ["type", "类别", "类型"],
    "category": ["query_category", "query_cate", "category", "class", "kind"],
    "city": ["city", "cityname", "城市"],
    "district": ["district", "adname", "区县"],
    "wgs84_lon": ["wgs84_lon", "lon", "longitude", "x"],
    "wgs84_lat": ["wgs84_lat", "lat", "latitude", "y"],
}


def configure_plotting() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.facecolor": "#fbfaf5",
            "axes.facecolor": "#fbfaf5",
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "legend.frameon": True,
        }
    )


def make_output_dir() -> Path:
    roots = [OUTPUT_ROOT, FALLBACK_OUTPUT_ROOT]
    last_error = None
    for root in roots:
        try:
            root.mkdir(parents=True, exist_ok=True)
            output_dir = root / f"dlm_dcm_generalization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            output_dir.mkdir(parents=True, exist_ok=False)
            for name in ["figures", "case_study_samples", "geojson"]:
                (output_dir / name).mkdir(parents=True, exist_ok=True)
            return output_dir
        except Exception as exc:
            last_error = exc
            continue
    raise last_error if last_error else RuntimeError("unable to create output directory")


def build_logger(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger("dlm_dcm_generalization")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(output_dir / "run_log.txt", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))
    logger.addHandler(sh)
    return logger


def scan_dataset(dataset_dir: Path) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for path in sorted(dataset_dir.rglob("*")):
        if path.suffix.lower() not in {".csv", ".geojson", ".shp"}:
            continue
        row = {
            "file_path": str(path),
            "format": path.suffix.lower().lstrip("."),
            "size_bytes": path.stat().st_size,
            "feature_count": "",
            "field_mapping": "",
            "city_field": "",
            "category_field": "",
            "name_field": "",
            "x_field": "",
            "y_field": "",
        }
        try:
            if path.suffix.lower() == ".csv":
                df = pd.read_csv(path, nrows=200)
                columns = [str(col) for col in df.columns]
                row["feature_count"] = int(sum(1 for _ in open(path, "r", encoding="utf-8", errors="ignore")) - 1)
            else:
                gdf = gpd.read_file(path, rows=200)
                columns = [str(col) for col in gdf.columns]
                row["feature_count"] = int(len(gpd.read_file(path)))
            mapping = {}
            for target, candidates in FIELD_CANDIDATES.items():
                mapping[target] = next((col for col in columns if col in candidates), "")
            row["field_mapping"] = json.dumps(mapping, ensure_ascii=False)
            row["city_field"] = mapping["city"]
            row["category_field"] = mapping["category"]
            row["name_field"] = mapping["name"]
            row["x_field"] = mapping["wgs84_lon"]
            row["y_field"] = mapping["wgs84_lat"]
        except Exception as exc:
            row["field_mapping"] = f"error: {exc}"
        rows.append(row)
    return pd.DataFrame(rows)


def normalize_dataset(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    mapping: Dict[str, str] = {}
    for target, candidates in FIELD_CANDIDATES.items():
        mapping[target] = next((col for col in gdf.columns if col in candidates), "")
    work = gdf.copy()
    if work.crs is None:
        work = work.set_crs(4326)
    if work.crs.to_epsg() != 4326:
        work = work.to_crs(4326)
    work["poi_uid"] = work.get(mapping["poi_uid"], work.index.astype(str)).astype(str)
    work["name"] = work.get(mapping["name"], "").astype(str)
    work["type"] = work.get(mapping["type"], "").astype(str)
    work["category"] = work.get(mapping["category"], work["type"]).astype(str).replace({"nan": "其他", "": "其他"}).fillna("其他")
    work["city"] = work.get(mapping["city"], "").astype(str)
    work["district"] = work.get(mapping["district"], "").astype(str)
    work = work[work.geometry.notna() & ~work.geometry.is_empty].copy()
    work["sample_label"] = work["city"] + "_" + work["district"]
    return work[["poi_uid", "name", "type", "category", "city", "district", "sample_label", "geometry"]]


def select_samples(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    grouped = (
        gdf.groupby(["city", "district"])
        .agg(raw_count=("poi_uid", "size"), category_count=("category", "nunique"))
        .reset_index()
    )
    grouped = grouped[(grouped["raw_count"] >= 160) & (grouped["category_count"] >= 6) & (grouped["raw_count"] <= 320)].copy()
    chosen = []
    for _, group in grouped.sort_values(["category_count", "raw_count"], ascending=False).groupby("city"):
        chosen.append(group.iloc[0])
        if len(chosen) >= 4:
            break
    return pd.DataFrame(chosen).sort_values(["category_count", "raw_count"], ascending=False)


def save_skill_files(output_dir: Path) -> None:
    blueprint = build_skill_blueprint()
    Path("skills").mkdir(parents=True, exist_ok=True)
    (Path("skills") / "point_generalization_skills_dlm_dcm.json").write_text(
        json.dumps(blueprint, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report = [
        "# DLM / DCM Skills Design Report",
        "",
        "## DLM 层",
        "- `representative_poi_retain`: 以类别稀有度、中心性、局部代表性构造代理重要性。",
        "- `structure_aware_selection`: 用聚类和 PCA 判断 clustered / linear / sparse / multi-core。",
        "- `category_balance_preserve`: 在局部网格先保每类代表点，避免单一类别塌缩。",
        "",
        "## DCM 层",
        "- `symbol_size_aware_generalization`: 把符号尺寸和最小间距换算为地面冲突阈值。",
        "- `cross_category_conflict_resolution`: 在异类冲突中保留 minority 类别并允许位移。",
        "- `layout_aesthetic_optimization`: 对局部空白和偏斜做补点与轻量重排。",
        "",
        "## DLM -> DCM",
        "- DLM 输出结构标签、重要性和局部类别多样性目标。",
        "- DCM 基于这些变量完成符号保留、冲突消解与图面优化。",
    ]
    (output_dir / "skills_design_report.md").write_text("\n".join(report), encoding="utf-8")


def marker_area(size_mm: float) -> float:
    return (size_mm * 8.0) ** 2


def legend_handles(symbol_rules: Dict[str, object], categories: List[str]) -> List[Line2D]:
    handles = []
    for category in categories:
        rule = symbol_rules[category]
        handles.append(Line2D([0], [0], marker=rule.marker, color="white", label=category, markerfacecolor=rule.color, markeredgecolor="#22333b", markersize=max(5.0, rule.symbol_size * 4.0)))
    return handles


def plot_map(ax, gdf: gpd.GeoDataFrame, symbol_rules: Dict[str, object], title: str) -> None:
    categories = sorted(gdf["category"].unique().tolist())
    for category in categories:
        subset = gdf[gdf["category"] == category]
        rule = symbol_rules[category]
        ax.scatter(subset.geometry.x, subset.geometry.y, s=marker_area(rule.symbol_size), c=rule.color, marker=rule.marker, alpha=0.82, linewidths=0.4, edgecolors="#172026")
    ax.set_title(title, fontsize=12, pad=8)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(handles=legend_handles(symbol_rules, categories), title="POI 类别", fontsize=8, title_fontsize=9, ncol=2, loc="best")


def plot_category_retention(ax, before: gpd.GeoDataFrame, after: gpd.GeoDataFrame, symbol_rules: Dict[str, object]) -> None:
    summary = (
        before.groupby("category").size().rename("before").to_frame()
        .join(after.groupby("category").size().rename("after"), how="left")
        .fillna(0)
        .reset_index()
    )
    summary["ratio"] = summary["after"] / summary["before"]
    ax.bar(summary["category"], summary["ratio"], color=[symbol_rules[cat].color for cat in summary["category"]], edgecolor="#253237")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("保留率")
    ax.set_title("类别保留效果")
    ax.tick_params(axis="x", rotation=35)


def plot_conflicts(ax, metric_selected: gpd.GeoDataFrame, wgs_selected: gpd.GeoDataFrame, engine: PointGeneralizationEngine, target_scale: int, symbol_rules: Dict[str, object]) -> None:
    plot_map(ax, wgs_selected, symbol_rules, f"符号冲突示意 1:{target_scale}")
    metric_selected = metric_selected.reset_index(drop=True)
    wgs_selected = wgs_selected.reset_index(drop=True)
    for i in range(len(metric_selected)):
        for j in range(i + 1, len(metric_selected)):
            a = metric_selected.iloc[i]
            b = metric_selected.iloc[j]
            if a.geometry.distance(b.geometry) < engine._conflict_threshold(a, b, target_scale):
                ax.plot([wgs_selected.iloc[i].geometry.x, wgs_selected.iloc[j].geometry.x], [wgs_selected.iloc[i].geometry.y, wgs_selected.iloc[j].geometry.y], color="#d62828", linewidth=1.1, alpha=0.6)


def plot_zoom(ax, before: gpd.GeoDataFrame, after: gpd.GeoDataFrame, symbol_rules: Dict[str, object]) -> None:
    densest = before.groupby("grid_id").size().sort_values(ascending=False)
    plot_map(ax, after, symbol_rules, "局部放大图")
    if densest.empty:
        return
    bounds = before[before["grid_id"] == densest.index[0]].total_bounds
    pad_x = max((bounds[2] - bounds[0]) * 0.25, 0.005)
    pad_y = max((bounds[3] - bounds[1]) * 0.25, 0.005)
    ax.set_xlim(bounds[0] - pad_x, bounds[2] + pad_x)
    ax.set_ylim(bounds[1] - pad_y, bounds[3] + pad_y)


def run_case(sample_row: pd.Series, source: gpd.GeoDataFrame, engine: PointGeneralizationEngine, symbol_rules: Dict[str, object], output_dir: Path, logger: logging.Logger, scales: List[int]) -> Tuple[List[Dict[str, object]], List[gpd.GeoDataFrame]]:
    sample_id = f"{sample_row['city']}_{sample_row['district']}".replace("/", "_")
    sample = source[(source["city"] == sample_row["city"]) & (source["district"] == sample_row["district"])].copy()
    sample_metric = sample.to_crs(sample.estimate_utm_crs())
    sample_metric = engine.analyze_dlm(sample_metric)
    sample_wgs = sample_metric.to_crs(4326)
    case_dir = output_dir / "case_study_samples" / sample_id
    case_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []
    outputs: List[gpd.GeoDataFrame] = []
    logger.info("Sample %s: %s points, %s categories.", sample_id, len(sample_metric), sample_metric["category"].nunique())

    for scale in scales:
        selected_metric = engine.optimize_dcm(sample_metric, scale)
        selected_metric["sample_id"] = sample_id
        selected_metric["target_scale"] = scale
        selected = selected_metric.to_crs(4326)
        outputs.append(selected)
        metrics = engine.summarize_metrics(sample_metric, selected_metric, scale, sample_id)
        metrics["city"] = sample_row["city"]
        metrics["district"] = sample_row["district"]
        rows.append(metrics)
        selected.to_file(output_dir / "geojson" / f"{sample_id}_{scale}.geojson", driver="GeoJSON")

        fig, axes = plt.subplots(2, 3, figsize=(19, 12), constrained_layout=True)
        plot_map(axes[0, 0], sample_wgs, symbol_rules, f"原始点分布\n{sample_id}")
        plot_map(axes[0, 1], selected, symbol_rules, f"综合后点分布\n1:{scale}")
        plot_map(axes[0, 2], sample_wgs, symbol_rules, "前后对比：原始")
        plot_map(axes[1, 0], selected, symbol_rules, "前后对比：综合后")
        plot_conflicts(axes[1, 1], selected_metric, selected, engine, scale, symbol_rules)
        plot_category_retention(axes[1, 2], sample_wgs, selected, symbol_rules)
        fig.savefig(case_dir / f"{sample_id}_{scale}_comparison.png", dpi=220)
        plt.close(fig)

        fig2, ax2 = plt.subplots(1, 1, figsize=(8.5, 7.5), constrained_layout=True)
        plot_zoom(ax2, sample_wgs, selected, symbol_rules)
        fig2.savefig(case_dir / f"{sample_id}_{scale}_zoom.png", dpi=240)
        plt.close(fig2)
    return rows, outputs


def write_summary(output_dir: Path, inventory: pd.DataFrame, selected_samples: pd.DataFrame, metrics_df: pd.DataFrame) -> None:
    lines = [
        "# DLM / DCM POI Generalization Experiment",
        "",
        "## 实验目标",
        "- 构建面向符号表达优化的多类别 POI 智能点综合实验。",
        "- 将 DLM 的空间结构理解映射到 DCM 的符号组织与图面优化。",
        "",
        "## 数据扫描结果",
        f"- 可用样本文件数：{len(inventory)}",
        f"- 重跑样本数：{metrics_df['sample_id'].nunique()}",
        f"- 覆盖尺度数：{metrics_df['target_scale'].nunique()}",
        "- 抽样规则：类别数 >= 6 且点数 >= 160 的城市-区县组合，每个城市选择类别最丰富的一个代表区。",
        "",
        "## 技能升级点",
        "- 新增 DLM/DCM 双层技能蓝图和六个刚性技能模块。",
        "- 新流程显式考虑符号尺寸、局部类别多样性、异类冲突和图面均衡。",
        "",
        "## 核心算法调整点",
        "- DBSCAN + PCA 识别聚类、线性、多核、稀疏结构。",
        "- 局部网格多类别预留 + minority 类别最低保留保障。",
        "- 基于符号占位的冲突检测与位移式冲突消解。",
        "",
        "## DLM / DCM 体现",
        "- DLM 输出 `importance_score`、`structure_type`、`local_density`、`minority_category`、`grid_id`。",
        "- DCM 基于 `symbol_rules` 和 DLM 指标完成符号保留、位移和图面平衡。",
        "",
        "## 主要结果概述",
        f"- 平均保留率：{metrics_df['retention_rate'].mean():.3f}",
        f"- 平均类别覆盖率：{metrics_df['category_coverage_rate'].mean():.3f}",
        f"- 平均局部类别多样性：{metrics_df['local_category_diversity'].mean():.3f}",
        f"- 平均 minority 类别生存率：{metrics_df['minority_category_survival_rate'].mean():.3f}",
        f"- 平均符号冲突：{metrics_df['symbol_conflicts_before'].mean():.1f} -> {metrics_df['symbol_conflicts_after'].mean():.1f}",
        "",
        "## 运行样本",
    ]
    for _, row in selected_samples.iterrows():
        lines.append(f"- {row['city']} {row['district']}：原始点数 {int(row['raw_count'])}，类别数 {int(row['category_count'])}")
    (output_dir / "experiment_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    configure_plotting()
    output_dir = make_output_dir()
    logger = build_logger(output_dir)
    logger.info("Output directory: %s", output_dir)
    save_skill_files(output_dir)

    inventory = scan_dataset(DATASET_DIR)
    inventory.to_csv(output_dir / "dataset_inventory.csv", index=False, encoding="utf-8-sig")
    source = normalize_dataset(gpd.read_file(DATASET_DIR / "all_poi_expanded.geojson"))
    symbol_rules = build_default_symbol_rules(sorted(source["category"].unique().tolist()))
    PointGeneralizationEngine.save_symbol_rules(symbol_rules, output_dir / "symbol_rules.json")

    selected_samples = select_samples(source)
    scales = [120000, 300000]
    engine = PointGeneralizationEngine(symbol_rules)
    all_rows: List[Dict[str, object]] = []
    outputs: List[gpd.GeoDataFrame] = []
    for _, sample_row in selected_samples.iterrows():
        rows, frames = run_case(sample_row, source, engine, symbol_rules, output_dir, logger, scales)
        all_rows.extend(rows)
        outputs.extend(frames)

    metrics_df = pd.DataFrame(all_rows)
    metrics_df.to_csv(output_dir / "metrics_summary.csv", index=False, encoding="utf-8-sig")
    if outputs:
        final_gdf = gpd.GeoDataFrame(pd.concat(outputs, ignore_index=True), geometry="geometry", crs=4326)
        final_gdf.to_file(output_dir / "final_selected_samples.geojson", driver="GeoJSON")
    write_summary(output_dir, inventory, selected_samples, metrics_df)
    print(output_dir)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        FALLBACK_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        with (FALLBACK_OUTPUT_ROOT / "dlm_dcm_generalization_failed.log").open("a", encoding="utf-8") as fh:
            fh.write(f"{datetime.now().isoformat()} | {exc}\n")
        raise
