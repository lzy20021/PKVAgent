from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(r"E:\GradOpening\MapGenAI")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.refine_engine_zoom_maps as base
from src.skills.point_generalization_dlm_dcm import PointGeneralizationEngine

SOURCE_CSV = PROJECT_ROOT / "amap_poi_dataset_expanded" / "all_poi_expanded.csv"
SOURCE_GEOJSON = PROJECT_ROOT / "amap_poi_dataset_expanded" / "all_poi_expanded.geojson"
OUTPUT_ROOT = PROJECT_ROOT / "output" / "multi_region_clean_exports"
SUMMARY_PATH = OUTPUT_ROOT / "multi_region_export_summary.md"
TARGET_SCALES = [84000, 253000, 532000]
TOTAL_REGIONS = 60
CURRENT_REGION = ("\u4e09\u4e9a\u5e02", "\u5929\u6daf\u533a")
CATEGORY_MAP = {
    "\u516c\u4ea4\u7ad9": "\u4ea4\u901a\u8bbe\u65bd",
    "\u5730\u94c1\u7ad9": "\u4ea4\u901a\u8bbe\u65bd",
    "\u9910\u5385": "\u9910\u996e",
    "\u5496\u5561\u5385": "\u9910\u996e",
}
NEEDED_FILES = {
    "compare_before_after_clean.png", "compare_before_after_clean.pdf", "compare_before_after_clean.svg",
    "before_only_clean.png", "before_only_clean.pdf", "before_only_clean.svg",
    "after_only_clean.png", "after_only_clean.pdf", "after_only_clean.svg",
}


def compute_stats() -> pd.DataFrame:
    df = pd.read_csv(SOURCE_CSV, usecols=["cityname", "adname", "query_category"], encoding="utf-8")
    df["category"] = df["query_category"].map(CATEGORY_MAP).fillna(df["query_category"])
    df = df[df["category"].isin(base.CATEGORY_ORDER)]
    rows = []
    for (city, district), sub in df.groupby(["cityname", "adname"]):
        count = len(sub)
        if count < 25 or count > 650:
            continue
        vc = sub["category"].value_counts()
        probs = vc / count
        entropy = float(-(probs * np.log(probs)).sum())
        rows.append((city, district, count, int(vc.size), entropy))
    stats = pd.DataFrame(rows, columns=["city", "district", "count", "cat_count", "entropy"])
    stats["density_label"] = np.where(stats["count"] >= 180, "high", np.where(stats["count"] >= 70, "medium", "sparse"))
    stats["score"] = np.where(
        stats["density_label"] == "high",
        stats["entropy"] * 100 + stats["cat_count"] * 20 + np.minimum(stats["count"], 400) * 0.08,
        np.where(
            stats["density_label"] == "medium",
            stats["entropy"] * 100 + stats["cat_count"] * 20 + (180 - np.abs(stats["count"] - 120)) * 0.15,
            stats["entropy"] * 100 + stats["cat_count"] * 22 + (80 - stats["count"]) * 0.35,
        ),
    )
    return stats.sort_values(["density_label", "score"], ascending=[True, False]).reset_index(drop=True)



def select_regions(stats: pd.DataFrame) -> pd.DataFrame:
    quotas = {"high": 25, "medium": 20, "sparse": 15}
    chosen = []
    used_regions = set()
    used_cities = Counter()
    current = stats[(stats["city"] == CURRENT_REGION[0]) & (stats["district"] == CURRENT_REGION[1])]
    if not current.empty:
        row = current.iloc[0]
        chosen.append(row)
        used_regions.add((row["city"], row["district"]))
        used_cities[row["city"]] += 1
        quotas[row["density_label"]] -= 1
    for density in ["high", "medium", "sparse"]:
        subset = stats[(stats["density_label"] == density) & (stats["cat_count"] >= (7 if density != "sparse" else 6))].sort_values("score", ascending=False)
        for _, row in subset.iterrows():
            key = (row["city"], row["district"])
            if key in used_regions or used_cities[row["city"]] >= 1:
                continue
            chosen.append(row)
            used_regions.add(key)
            used_cities[row["city"]] += 1
            quotas[density] -= 1
            if quotas[density] <= 0:
                break
    if len(chosen) < TOTAL_REGIONS:
        leftovers = stats.sort_values(["score", "count"], ascending=[False, False])
        for _, row in leftovers.iterrows():
            key = (row["city"], row["district"])
            if key in used_regions or used_cities[row["city"]] >= 2:
                continue
            chosen.append(row)
            used_regions.add(key)
            used_cities[row["city"]] += 1
            if len(chosen) >= TOTAL_REGIONS:
                break
    return pd.DataFrame(chosen).drop_duplicates(subset=["city", "district"]).head(TOTAL_REGIONS).reset_index(drop=True)



def normalize_source(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    out = gdf.copy()
    out["cityname"] = out["cityname"].astype(str)
    out["adname"] = out["adname"].astype(str)
    out["category"] = out["query_category"].map(CATEGORY_MAP).fillna(out["query_category"])
    out = out[out["category"].isin(base.CATEGORY_ORDER)].copy()
    return out[["poi_uid", "name", "type", "category", "cityname", "adname", "geometry"]].copy()



def display_name(city: str, district: str) -> str:
    return city if city == district else city + district



def folder_name(city: str, district: str) -> str:
    return f"{city}_{district}".replace("/", "_").replace("\\", "_")



def trait_text(density_label: str, cat_count: int, count: int) -> str:
    density = {"high": "\u9ad8\u5bc6\u5ea6", "medium": "\u4e2d\u5bc6\u5ea6", "sparse": "\u76f8\u5bf9\u7a00\u758f"}[density_label]
    diversity = "\u591a\u7c7b\u522b\u5171\u5b58\u660e\u663e" if cat_count >= 8 else "\u7c7b\u522b\u7ec4\u5408\u8f83\u4e30\u5bcc"
    diff = "\u524d\u540e\u5dee\u5f02\u8f83\u660e\u663e" if (density_label == "high" or count >= 160) else "\u524d\u540e\u5dee\u5f02\u9002\u4e2d"
    return f"{density}\uff0c{diversity}\uff0c{diff}"



def scale_output_dir(region_dir: Path, scale: int) -> Path:
    return region_dir if scale == 84000 else region_dir / str(scale)



def ensure_bounds(raw_metric: gpd.GeoDataFrame, gen_metric: gpd.GeoDataFrame, bounds, symbol_rules, scale: int):
    bx0, by0, bx1, by1 = bounds
    pad_unit = max(base.symbol_half_extent(rule, scale, 1.0) for rule in symbol_rules.values()) * 2.5
    for _ in range(12):
        bounds = (bx0, by0, bx1, by1)
        bounds, raw_zoom, gen_zoom = base.expand_bounds_for_symbols(bounds, raw_metric, gen_metric, scale, symbol_rules)
        try:
            base.verify_bounds(bounds, raw_zoom, symbol_rules, scale, True)
            base.verify_bounds(bounds, gen_zoom, symbol_rules, scale, False)
            return bounds, raw_zoom, gen_zoom
        except RuntimeError:
            width = bx1 - bx0
            height = by1 - by0
            extra = max(width, height) * 0.08 + pad_unit
            bx0 -= extra
            by0 -= extra
            bx1 += extra
            by1 += extra
    raise RuntimeError(f"Unable to find unclipped bounds for region export at 1:{scale}")



def export_compare(out_dir: Path, title_text: str, raw_zoom: gpd.GeoDataFrame, gen_zoom: gpd.GeoDataFrame, bounds, symbol_rules, scale: int) -> None:
    minx, miny, maxx, maxy = bounds
    fig = base.plt.figure(figsize=(12.6, 6.5))
    gs = fig.add_gridspec(1, 2, left=0.075, right=0.965, top=0.87, bottom=0.11, wspace=0.16)
    ax_before = fig.add_subplot(gs[0, 0])
    ax_after = fig.add_subplot(gs[0, 1])
    for ax in (ax_before, ax_after):
        base.style_ax(ax)
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)
    base.render_map(ax_before, raw_zoom, symbol_rules, scale, raw_mode=True)
    base.render_map(ax_after, gen_zoom, symbol_rules, scale, raw_mode=False)
    ax_before.set_title("\u7efc\u5408\u524d", pad=8, fontproperties=base.CJK_FONT_PROP)
    ax_after.set_title("\u7efc\u5408\u540e", pad=8, fontproperties=base.CJK_FONT_PROP)
    title = fig.suptitle(title_text, fontsize=13.2, fontweight="bold", y=0.95, fontproperties=base.CJK_FONT_PROP)
    base.apply_text_font(title)
    save_base = out_dir / "compare_before_after_clean"
    fig.savefig(save_base.with_suffix(".png"), dpi=500, bbox_inches="tight")
    fig.savefig(save_base.with_suffix(".pdf"), dpi=500, bbox_inches="tight")
    svg_path = save_base.with_suffix(".svg")
    fig.savefig(svg_path, dpi=500, bbox_inches="tight")
    base.plt.close(fig)
    base.strip_svg_comments(svg_path)



def export_single(out_dir: Path, gdf: gpd.GeoDataFrame, bounds, symbol_rules, before: bool, scale: int) -> None:
    minx, miny, maxx, maxy = bounds
    fig, ax = base.plt.subplots(figsize=(6.45, 6.5))
    fig.subplots_adjust(left=0.12, right=0.96, top=0.90, bottom=0.11)
    base.style_ax(ax)
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    base.render_map(ax, gdf, symbol_rules, scale, raw_mode=before)
    ax.set_title("\u7efc\u5408\u524d" if before else "\u7efc\u5408\u540e", pad=8, fontproperties=base.CJK_FONT_PROP)
    save_base = out_dir / ("before_only_clean" if before else "after_only_clean")
    fig.savefig(save_base.with_suffix(".png"), dpi=500, bbox_inches="tight")
    fig.savefig(save_base.with_suffix(".pdf"), dpi=500, bbox_inches="tight")
    svg_path = save_base.with_suffix(".svg")
    fig.savefig(svg_path, dpi=500, bbox_inches="tight")
    base.plt.close(fig)
    base.strip_svg_comments(svg_path)



def export_region_scale(subset: gpd.GeoDataFrame, region_name: str, out_dir: Path, scale: int) -> None:
    existing_files = {f.name for f in out_dir.iterdir() if f.is_file()} if out_dir.exists() else set()
    if NEEDED_FILES.issubset(existing_files):
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    metric_crs = subset.estimate_utm_crs() or "EPSG:3857"
    raw_metric = subset.to_crs(metric_crs)
    categories = base.ordered_categories(raw_metric["category"].dropna().astype(str).unique().tolist())
    symbol_rules = base.build_symbol_rules(categories)
    other_key = chr(0x5176) + chr(0x4ED6)
    if other_key not in symbol_rules:
        first_rule = next(iter(symbol_rules.values()))
        symbol_rules[other_key] = base.SymbolRule(other_key, first_rule.symbol_size, first_rule.min_distance, first_rule.priority, 1, "#7f8c8d", "circle")
    engine = PointGeneralizationEngine(symbol_rules)
    raw_metric = engine.analyze_dlm(raw_metric)
    gen_metric = engine.optimize_dcm(raw_metric.copy(), scale)
    anchor = base.choose_zoom_bounds(raw_metric, gen_metric, engine, scale, symbol_rules)
    bounds, raw_zoom, gen_zoom = base.build_scale_bounds(anchor, raw_metric, gen_metric, scale, symbol_rules)
    bounds, raw_zoom, gen_zoom = ensure_bounds(raw_metric, gen_metric, bounds, symbol_rules, scale)
    title_text = f"{region_name}\u90e8\u5206poi\u7efc\u5408\u524d\u540e\u5bf9\u6bd4\u56fe"
    export_compare(out_dir, title_text, raw_zoom, gen_zoom, bounds, symbol_rules, scale)
    export_single(out_dir, raw_zoom, bounds, symbol_rules, True, scale)
    export_single(out_dir, gen_zoom, bounds, symbol_rules, False, scale)



def write_summary(selected: pd.DataFrame) -> None:
    lines = [
        '# Multi Region Export Summary',
        '',
        f'- \u4e00\u5171\u5bfc\u51fa\u533a\u57df\uff1a{len(selected)}',
        f'- \u65b0\u589e\u5bfc\u51fa\u533a\u57df\uff1a{max(len(selected) - 1, 0)}',
        f'- \u540c\u6b65\u6bd4\u4f8b\u5c3a\uff1a{", ".join(f"1:{s}" for s in TARGET_SCALES)}',
        f'- \u603b\u8f93\u51fa\u8def\u5f84\uff1a{OUTPUT_ROOT}',
        '',
        '## \u533a\u57df\u6e05\u5355',
    ]
    main_regions = []
    supp_regions = []
    for _, row in selected.iterrows():
        region_name = display_name(str(row['city']), str(row['district']))
        trait = trait_text(str(row['density_label']), int(row['cat_count']), int(row['count']))
        region_dir = OUTPUT_ROOT / folder_name(str(row['city']), str(row['district']))
        lines.append(f'- {region_name}\uff1a{trait}')
        lines.append(f'  \u8f93\u51fa\u8def\u5f84\uff1a{region_dir}')
        rec_main = str(row['density_label']) != 'sparse' and int(row['cat_count']) >= 7 and int(row['count']) >= 90
        (main_regions if rec_main else supp_regions).append(region_name)
    lines.append('')
    lines.append('## \u9002\u5408\u8bba\u6587\u6b63\u6587')
    for name in main_regions:
        lines.append(f'- {name}')
    lines.append('')
    lines.append('## \u9002\u5408\u8865\u5145\u6750\u6599')
    for name in supp_regions:
        lines.append(f'- {name}')
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8-sig")



def main() -> None:
    base.configure_plotting()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    source = normalize_source(gpd.read_file(SOURCE_GEOJSON))
    stats = compute_stats()
    selected = select_regions(stats)
    for i, row in selected.iterrows():
        city = str(row['city'])
        district = str(row['district'])
        region_name = display_name(city, district)
        region_dir = OUTPUT_ROOT / folder_name(city, district)
        subset = source[(source['cityname'] == city) & (source['adname'] == district)].copy()
        if len(subset) < 25:
            continue
        for scale in TARGET_SCALES:
            out_dir = scale_output_dir(region_dir, scale)
            export_region_scale(subset, region_name, out_dir, scale)
        print(f"[{i+1}/{len(selected)}] {region_name}")
    write_summary(selected)
    print(OUTPUT_ROOT)
    print(SUMMARY_PATH)
    print(len(selected))


if __name__ == '__main__':
    main()
