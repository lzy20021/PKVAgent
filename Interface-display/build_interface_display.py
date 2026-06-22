from __future__ import annotations

import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
DATA_PATH = PROJECT_ROOT / "amap_poi_dataset_expanded" / "all_poi_expanded.geojson"
STRUCTURE_PATH = PROJECT_ROOT / "paper_figures" / "expanded_structure_map" / "poi_structure_labels_8types.csv"
SEED = 20260409

STRUCTURE_ORDER = [
    "多簇集型",
    "单核心集聚型",
    "双核心/多核心型",
    "廊道带状型",
    "边缘偏置型",
    "稀疏离散型",
    "均匀分布型",
    "混合过渡型",
]

STRUCTURE_DISPLAY = {
    "多簇集型": "Multi-cluster",
    "单核心集聚型": "Single-core concentration",
    "双核心/多核心型": "Dual / multi core",
    "廊道带状型": "Corridor-belt",
    "边缘偏置型": "Edge-biased",
    "稀疏离散型": "Sparse-dispersed",
    "均匀分布型": "Uniform",
    "混合过渡型": "Hybrid transition",
}

CATEGORY_META = {
    "餐饮": {"shape": "diamond", "color": "#D97A5A", "stroke": "#8A4F38", "priority": 3, "size": 9},
    "酒店": {"shape": "square", "color": "#6488C4", "stroke": "#35537F", "priority": 4, "size": 10},
    "学校": {"shape": "triangle", "color": "#5FA06B", "stroke": "#356343", "priority": 5, "size": 11},
    "医院": {"shape": "cross", "color": "#C95E7C", "stroke": "#8A314B", "priority": 6, "size": 12},
    "银行": {"shape": "star", "color": "#B58A3A", "stroke": "#7A5B1D", "priority": 2, "size": 10},
    "交通设施": {"shape": "hex", "color": "#8F7AE6", "stroke": "#5D4AB8", "priority": 5, "size": 10},
}

REPRESENTATIVE_CATEGORIES = list(CATEGORY_META.keys())
PRIMARY_STRUCTURE = "单核心集聚型"
PRIMARY_CITY = "北京"


def normalize_city(name: str) -> str:
    text = str(name or "").strip()
    for suffix in ["城区", "市", "地区", "盟", "自治州", "自治县"]:
        if text.endswith(suffix) and len(text) > 2:
            text = text[: -len(suffix)]
            break
    return text


def classify_category(props):
    text = " ".join(str(props.get(key, "") or "") for key in ("query_category", "type", "name", "typecode", "address")).lower()
    if any(token in text for token in ["地铁", "公交", "metro", "subway", "bus", "停车场", "parking"]):
        return "交通设施"
    if any(token in text for token in ["医院", "医疗", "medical", "hospital", "clinic"]):
        return "医院"
    if any(token in text for token in ["学校", "教育", "school", "大学", "学院", "education"]):
        return "学校"
    if any(token in text for token in ["酒店", "hotel"]):
        return "酒店"
    if any(token in text for token in ["银行", "bank"]):
        return "银行"
    if any(token in text for token in ["餐厅", "餐饮", "咖啡", "restaurant", "food", "catering", "coffee"]):
        return "餐饮"
    return None


def load_structure_rows():
    rows = []
    with STRUCTURE_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            item = dict(row)
            item["city"] = normalize_city(item["city"])
            for key in ("point_count", "density", "anisotropy", "comp_count", "dominant_share", "edge_ratio", "center_ratio"):
                item[key] = float(item[key])
            rows.append(item)
    return rows


def load_features():
    with DATA_PATH.open("r", encoding="utf-8") as handle:
        collection = json.load(handle)
    features = []
    for feature in collection["features"]:
        props = feature.get("properties", {})
        geom = feature.get("geometry", {})
        coords = geom.get("coordinates") or []
        if geom.get("type") != "Point" or len(coords) < 2:
            continue
        category = classify_category(props)
        if category is None:
            continue
        city = normalize_city(props.get("cityname") or props.get("city") or props.get("pname") or "")
        if not city:
            continue
        features.append(
            {
                "id": props.get("poi_uid") or f"{city}-{len(features)}",
                "name": str(props.get("name") or ""),
                "city": city,
                "district": str(props.get("district") or props.get("adname") or ""),
                "lon": float(coords[0]),
                "lat": float(coords[1]),
                "category": category,
                "query_category": str(props.get("query_category") or ""),
            }
        )
    return features


def weighted_choice(rows, rng):
    weights = []
    for row in rows:
        point_score = max(1.0, row["point_count"])
        density_score = max(0.2, min(3.0, math.log10(max(10.0, row["density"])) - 0.6))
        anomaly_penalty = 0.15 if row["anisotropy"] > 50 else 1.0
        weights.append(point_score * density_score * anomaly_penalty)
    total = sum(weights)
    pick = rng.random() * total
    cursor = 0.0
    for row, weight in zip(rows, weights):
        cursor += weight
        if cursor >= pick:
            return row
    return rows[-1]


def choose_structure_samples(rows, features):
    city_counts = Counter(item["city"] for item in features)
    rng = random.Random(SEED)
    grouped = defaultdict(list)
    for row in rows:
        if city_counts[row["city"]] > 0:
            grouped[row["structure_type_8"]].append(row)

    selected = []
    for structure in STRUCTURE_ORDER:
        candidates = sorted(grouped.get(structure, []), key=lambda item: item["point_count"], reverse=True)
        if not candidates:
            continue
        pool = candidates if len(candidates) <= 3 else candidates[:3]
        if structure == PRIMARY_STRUCTURE:
            chosen = next((item for item in candidates if item["city"] == PRIMARY_CITY), candidates[0])
        else:
            chosen = weighted_choice(pool, rng)
        selected.append(
            {
                "structure_type_8": structure,
                "structure_display": STRUCTURE_DISPLAY[structure],
                "city": chosen["city"],
                "point_count": int(chosen["point_count"]),
                "density": round(float(chosen["density"]), 3),
                "anisotropy": round(float(chosen["anisotropy"]), 3),
                "selection_seed": SEED,
                "selection_pool": [item["city"] for item in pool],
                "mapping_note": "论文结构名称与 structure_type_8 字段直接一致，无需语义改写。",
            }
        )
    return selected


def get_focus_points(city_features):
    xs = [item["lon"] for item in city_features]
    ys = [item["lat"] for item in city_features]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    best = None
    best_score = None
    for grid in (9, 11, 13):
        cell_counts = Counter()
        for item in city_features:
            gx = min(grid - 1, max(0, int((item["lon"] - minx) / max(maxx - minx, 1e-9) * grid)))
            gy = min(grid - 1, max(0, int((item["lat"] - miny) / max(maxy - miny, 1e-9) * grid)))
            cell_counts[(gx, gy)] += 1
        cell = cell_counts.most_common(1)[0][0]
        for expand in (1, 2, 3):
            selected = []
            for item in city_features:
                gx = min(grid - 1, max(0, int((item["lon"] - minx) / max(maxx - minx, 1e-9) * grid)))
                gy = min(grid - 1, max(0, int((item["lat"] - miny) / max(maxy - miny, 1e-9) * grid)))
                if abs(gx - cell[0]) <= expand and abs(gy - cell[1]) <= expand:
                    selected.append(item)
            score = (abs(len(selected) - 220), expand, -len(selected))
            if best_score is None or score < best_score:
                best = selected
                best_score = score
            if 80 <= len(selected) <= 320:
                return selected, {
                    "minLon": min(item["lon"] for item in selected),
                    "maxLon": max(item["lon"] for item in selected),
                    "minLat": min(item["lat"] for item in selected),
                    "maxLat": max(item["lat"] for item in selected),
                }
    if not best:
        best = city_features
    bbox = {
        "minLon": min(item["lon"] for item in best),
        "maxLon": max(item["lon"] for item in best),
        "minLat": min(item["lat"] for item in best),
        "maxLat": max(item["lat"] for item in best),
    }
    return best, bbox


def with_screen_position(points, bbox):
    width = max(bbox["maxLon"] - bbox["minLon"], 1e-9)
    height = max(bbox["maxLat"] - bbox["minLat"], 1e-9)
    enriched = []
    for item in points:
        x = (item["lon"] - bbox["minLon"]) / width
        y = 1.0 - (item["lat"] - bbox["minLat"]) / height
        entry = dict(item)
        entry["x"] = round(x, 6)
        entry["y"] = round(y, 6)
        meta = CATEGORY_META[entry["category"]]
        entry["symbol"] = meta["shape"]
        entry["symbolColor"] = meta["color"]
        entry["symbolStroke"] = meta["stroke"]
        entry["symbolSize"] = meta["size"]
        entry["priority"] = meta["priority"]
        enriched.append(entry)
    return enriched


def generalize_points(points):
    sorted_points = sorted(points, key=lambda item: (-item["priority"], item["category"], item["y"], item["x"]))
    counts = Counter(item["category"] for item in points)
    quotas = {category: max(6, min(28, int(count * 0.18))) for category, count in counts.items()}
    kept = []
    occupied = defaultdict(set)

    def can_keep(item):
        cell = (int(item["x"] * 10), int(item["y"] * 8))
        category_set = occupied[cell]
        if len(category_set) >= 3 and item["category"] not in category_set:
            return False
        if item["category"] in ("餐饮", "银行"):
            min_dist = 0.028
        elif item["priority"] >= 5:
            min_dist = 0.045
        else:
            min_dist = 0.036
        for existing in kept:
            dx = item["x"] - existing["x"]
            dy = item["y"] - existing["y"]
            dist = math.sqrt(dx * dx + dy * dy)
            limit = min_dist if item["category"] == existing["category"] else min_dist * 0.72
            if dist < limit:
                return False
        return True

    per_category = defaultdict(int)
    for category in sorted(quotas, key=lambda name: (quotas[name], -CATEGORY_META[name]["priority"], name)):
        for item in [p for p in sorted_points if p["category"] == category]:
            if per_category[category] >= quotas[category]:
                break
            if not can_keep(item):
                continue
            kept.append(item)
            occupied[(int(item["x"] * 10), int(item["y"] * 8))].add(item["category"])
            per_category[category] += 1

    for item in sorted_points:
        if len(kept) >= 120:
            break
        if item in kept:
            continue
        if not can_keep(item):
            continue
        kept.append(item)
        occupied[(int(item["x"] * 10), int(item["y"] * 8))].add(item["category"])
    return kept


def feature_collection(before, after):
    features = []
    for state, points in (("before", before), ("after", after)):
        for item in points:
            features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [item["lon"], item["lat"]]},
                    "properties": {
                        "id": item["id"],
                        "name": item["name"],
                        "city": item["city"],
                        "district": item["district"],
                        "category": item["category"],
                        "query_category": item["query_category"],
                        "state": state,
                        "symbol": item["symbol"],
                    },
                }
            )
    return {"type": "FeatureCollection", "features": features}


def svg_symbol(shape, cx, cy, size, fill, stroke, opacity=1.0):
    stroke_width = max(1.1, size * 0.16)
    if shape == "square":
        half = size * 0.52
        return f'<rect x="{cx-half:.2f}" y="{cy-half:.2f}" width="{2*half:.2f}" height="{2*half:.2f}" rx="{size*0.18:.2f}" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width:.2f}" opacity="{opacity:.3f}"/>'
    if shape == "triangle":
        h = size * 0.62
        pts = [(cx, cy - h), (cx - h, cy + h * 0.85), (cx + h, cy + h * 0.85)]
        path = " ".join(f"{x:.2f},{y:.2f}" for x, y in pts)
        return f'<polygon points="{path}" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width:.2f}" opacity="{opacity:.3f}"/>'
    if shape == "diamond":
        h = size * 0.68
        pts = [(cx, cy - h), (cx + h, cy), (cx, cy + h), (cx - h, cy)]
        path = " ".join(f"{x:.2f},{y:.2f}" for x, y in pts)
        return f'<polygon points="{path}" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width:.2f}" opacity="{opacity:.3f}"/>'
    if shape == "cross":
        arm = size * 0.58
        inner = size * 0.22
        path = [
            (cx - inner, cy - arm), (cx + inner, cy - arm), (cx + inner, cy - inner),
            (cx + arm, cy - inner), (cx + arm, cy + inner), (cx + inner, cy + inner),
            (cx + inner, cy + arm), (cx - inner, cy + arm), (cx - inner, cy + inner),
            (cx - arm, cy + inner), (cx - arm, cy - inner), (cx - inner, cy - inner),
        ]
        pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in path)
        return f'<polygon points="{pts}" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width:.2f}" opacity="{opacity:.3f}"/>'
    if shape == "star":
        outer = size * 0.72
        inner = size * 0.32
        pts = []
        for index in range(10):
            angle = math.radians(-90 + index * 36)
            radius = outer if index % 2 == 0 else inner
            pts.append((cx + math.cos(angle) * radius, cy + math.sin(angle) * radius))
        path = " ".join(f"{x:.2f},{y:.2f}" for x, y in pts)
        return f'<polygon points="{path}" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width:.2f}" opacity="{opacity:.3f}"/>'
    if shape == "hex":
        radius = size * 0.62
        pts = []
        for index in range(6):
            angle = math.radians(30 + index * 60)
            pts.append((cx + math.cos(angle) * radius, cy + math.sin(angle) * radius))
        path = " ".join(f"{x:.2f},{y:.2f}" for x, y in pts)
        return f'<polygon points="{path}" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width:.2f}" opacity="{opacity:.3f}"/>'
    return f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{size*0.46:.2f}" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width:.2f}" opacity="{opacity:.3f}"/>'


def draw_map_points(points, box, opacity=0.96, size_scale=1.0):
    x0, y0, width, height = box
    parts = []
    for item in points:
        px = x0 + item["x"] * width
        py = y0 + item["y"] * height
        parts.append(
            svg_symbol(
                item["symbol"],
                px,
                py,
                item["symbolSize"] * size_scale,
                item["symbolColor"],
                item["symbolStroke"],
                opacity,
            )
        )
    return "\n".join(parts)


def generate_interface_svg(payload):
    before = payload["primarySample"]["before"]
    after = payload["primarySample"]["after"]
    selected_samples = payload["selectedSamples"]
    summary = payload["primarySample"]["summary"]
    counts_before = Counter(item["category"] for item in before)
    counts_after = Counter(item["category"] for item in after)
    primary_structure = next((item["structure_display"] for item in selected_samples if item["city"] == payload["primarySample"]["city"]), STRUCTURE_DISPLAY[payload["primarySample"]["structure"]])
    width, height = 1600, 960
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<defs>",
        '<linearGradient id="bg" x1="0" x2="1" y1="0" y2="1"><stop offset="0%" stop-color="#F8FAFD"/><stop offset="100%" stop-color="#EEF3F8"/></linearGradient>',
        '<filter id="shadow" x="-10%" y="-10%" width="130%" height="130%"><feDropShadow dx="0" dy="10" stdDeviation="14" flood-color="#AFC2D7" flood-opacity="0.16"/></filter>',
        "</defs>",
        '<rect width="100%" height="100%" fill="url(#bg)"/>',
        '<rect x="24" y="24" width="1552" height="912" rx="24" fill="#FCFDFE" stroke="#E2EAF3" filter="url(#shadow)"/>',
        '<rect x="24" y="24" width="1552" height="100" rx="24" fill="#FFFFFF" stroke="#E8EEF5"/>',
        '<text x="58" y="63" font-size="12" font-family="Segoe UI, Microsoft YaHei, sans-serif" font-weight="700" letter-spacing="1.5" fill="#5F7691">POINTGENAI INTERFACE</text>',
        '<text x="58" y="92" font-size="31" font-family="Segoe UI, Microsoft YaHei, sans-serif" font-weight="700" fill="#20456C">Symbol-aware Point Generalization Interface</text>',
        '<text x="58" y="116" font-size="13" font-family="Segoe UI, Microsoft YaHei, sans-serif" fill="#5F7388">Explainable agent workflow for POI point generalization with symbol-aware semantic readability.</text>',
        '<rect x="1368" y="56" width="170" height="30" rx="15" fill="#F2F7FC" stroke="#CBDBEA"/>',
        '<text x="1453" y="75" text-anchor="middle" font-size="12" font-family="Segoe UI, Microsoft YaHei, sans-serif" fill="#4F6782">SVG export ready</text>',
        '<rect x="40" y="140" width="232" height="752" rx="20" fill="#F8FBFE" stroke="#E2E9F2"/>',
        '<text x="60" y="176" font-size="18" font-family="Segoe UI, Microsoft YaHei, sans-serif" font-weight="700" fill="#24364D">Sample Selection</text>',
        '<rect x="56" y="200" width="200" height="128" rx="16" fill="#FFFFFF" stroke="#DFE7F1"/>',
        f'<text x="72" y="226" font-size="12" font-family="Segoe UI, Microsoft YaHei, sans-serif" fill="#6A7B8D">Primary sample</text>',
        f'<text x="72" y="256" font-size="32" font-family="Segoe UI, Microsoft YaHei, sans-serif" font-weight="700" fill="#203F62">{payload["primarySample"]["city"]}</text>',
        f'<text x="72" y="277" font-size="12" font-family="Segoe UI, Microsoft YaHei, sans-serif" fill="#5D6F85">{primary_structure}</text>',
        f'<text x="72" y="302" font-size="17" font-family="Segoe UI, Microsoft YaHei, sans-serif" font-weight="700" fill="#26598A">{summary["beforeCount"]} → {summary["afterCount"]}</text>',
        f'<text x="72" y="320" font-size="12" font-family="Segoe UI, Microsoft YaHei, sans-serif" fill="#5D6F85">Reduction {summary["reductionRate"]}%</text>',
    ]
    sample_y = 344
    for item in selected_samples:
        active = item["city"] == payload["primarySample"]["city"]
        parts.append(f'<rect x="56" y="{sample_y}" width="200" height="54" rx="13" fill="{"#EDF4FB" if active else "#FFFFFF"}" stroke="{"#ADC6DF" if active else "#E4EBF4"}"/>')
        parts.append(f'<text x="72" y="{sample_y+22}" font-size="14" font-family="Segoe UI, Microsoft YaHei, sans-serif" font-weight="700" fill="#22334D">{item["city"]}</text>')
        parts.append(f'<text x="72" y="{sample_y+40}" font-size="12" font-family="Segoe UI, Microsoft YaHei, sans-serif" fill="#6D7E92">{item["structure_display"]}</text>')
        sample_y += 68
    parts.extend(
        [
            '<rect x="300" y="140" width="744" height="752" rx="20" fill="#FFFFFF" stroke="#E2E9F2"/>',
            '<text x="328" y="174" font-size="18" font-family="Segoe UI, Microsoft YaHei, sans-serif" font-weight="700" fill="#24364D">Symbol-aware Generalization Result</text>',
            '<text x="328" y="196" font-size="12" font-family="Segoe UI, Microsoft YaHei, sans-serif" fill="#6D7E92">Before / After / Overlay comparison for explainable POI point generalization</text>',
            '<rect x="328" y="212" width="94" height="32" rx="16" fill="#DCEEFF" stroke="#B9D7F7"/>',
            '<text x="375" y="232" text-anchor="middle" font-size="12" font-family="Segoe UI, Microsoft YaHei, sans-serif" fill="#2B4F75">Overlay</text>',
            '<rect x="432" y="212" width="94" height="32" rx="16" fill="#F7F9FC" stroke="#DDE7F2"/>',
            '<text x="479" y="232" text-anchor="middle" font-size="12" font-family="Segoe UI, Microsoft YaHei, sans-serif" fill="#5E7087">Before</text>',
            '<rect x="536" y="212" width="94" height="32" rx="16" fill="#F7F9FC" stroke="#DDE7F2"/>',
            '<text x="583" y="232" text-anchor="middle" font-size="12" font-family="Segoe UI, Microsoft YaHei, sans-serif" fill="#5E7087">After</text>',
            '<rect x="328" y="252" width="688" height="474" rx="16" fill="#F8FBFE" stroke="#E1EAF3"/>',
            '<rect x="348" y="272" width="648" height="434" rx="14" fill="#FCFDFE" stroke="#DDE7F2"/>',
            '<rect x="356" y="280" width="632" height="418" rx="12" fill="none" stroke="#E4EDF6" stroke-dasharray="4 4"/>',
            '<rect x="454" y="336" width="430" height="258" rx="10" fill="none" stroke="#C8D8E7" stroke-width="1.4"/>',
            f'<rect x="366" y="288" width="{128}" height="24" rx="12" fill="#E7F0FA" stroke="#BED1E4"/>',
            f'<text x="376" y="304" font-size="11" font-family="Segoe UI, Microsoft YaHei, sans-serif" fill="#31567F">Sample city: {payload["primarySample"]["city"]}</text>',
            f'<rect x="498" y="288" width="{236}" height="24" rx="12" fill="#F3F7FC" stroke="#DAE4EE"/>',
            f'<text x="508" y="304" font-size="11" font-family="Segoe UI, Microsoft YaHei, sans-serif" fill="#5C6F85">Structure type: {primary_structure}</text>',
            f'<text x="366" y="680" font-size="11.5" font-family="Segoe UI, Microsoft YaHei, sans-serif" fill="#5A7290">light symbols = input points; solid symbols = generalized output</text>',
        ]
    )
    parts.append(draw_map_points(before, (356, 294, 632, 386), opacity=0.22, size_scale=0.72))
    parts.append(draw_map_points(after, (356, 294, 632, 386), opacity=0.98, size_scale=1.05))
    legend_x = 370
    offset = 0
    for category in REPRESENTATIVE_CATEGORIES:
        meta = CATEGORY_META[category]
        parts.append(svg_symbol(meta["shape"], legend_x + offset, 710, 8.2, meta["color"], meta["stroke"], 1.0))
        parts.append(f'<text x="{legend_x + offset + 14}" y="714" font-size="11.5" font-family="Segoe UI, Microsoft YaHei, sans-serif" fill="#58697E">{category}</text>')
        offset += 105
    parts.extend(
        [
            '<text x="336" y="742" font-size="12" font-family="Segoe UI, Microsoft YaHei, sans-serif" fill="#57708E">Category-specific symbols are preserved after generalization for semantic readability.</text>',
            '<rect x="328" y="756" width="214" height="122" rx="14" fill="#FFFFFF" stroke="#DFE7F1"/>',
            '<rect x="556" y="756" width="214" height="122" rx="14" fill="#FFFFFF" stroke="#DFE7F1"/>',
            '<rect x="784" y="756" width="214" height="122" rx="14" fill="#FFFFFF" stroke="#DFE7F1"/>',
            '<text x="346" y="778" font-size="11" font-family="Segoe UI, Microsoft YaHei, sans-serif" fill="#6D7E92" letter-spacing="0.8">INPUT SUBSET</text>',
            f'<text x="346" y="826" font-size="38" font-family="Segoe UI, Microsoft YaHei, sans-serif" font-weight="700" fill="#294F78">{summary["beforeCount"]}</text>',
            '<text x="346" y="846" font-size="12" font-family="Segoe UI, Microsoft YaHei, sans-serif" fill="#617489">focus-region POIs</text>',
            '<text x="574" y="778" font-size="11" font-family="Segoe UI, Microsoft YaHei, sans-serif" fill="#6D7E92" letter-spacing="0.8">GENERALIZED OUTPUT</text>',
            f'<text x="574" y="826" font-size="38" font-family="Segoe UI, Microsoft YaHei, sans-serif" font-weight="700" fill="#294F78">{summary["afterCount"]}</text>',
            '<text x="574" y="846" font-size="12" font-family="Segoe UI, Microsoft YaHei, sans-serif" fill="#617489">symbol-aware POIs</text>',
            '<text x="802" y="778" font-size="11" font-family="Segoe UI, Microsoft YaHei, sans-serif" fill="#6D7E92" letter-spacing="0.8">REDUCTION</text>',
            f'<text x="802" y="826" font-size="38" font-family="Segoe UI, Microsoft YaHei, sans-serif" font-weight="700" fill="#294F78">{summary["reductionRate"]}%</text>',
            '<text x="802" y="846" font-size="12" font-family="Segoe UI, Microsoft YaHei, sans-serif" fill="#617489">semantic categories preserved</text>',
            '<rect x="1060" y="140" width="500" height="752" rx="20" fill="#F8FBFE" stroke="#E2E9F2"/>',
            '<text x="1086" y="176" font-size="18" font-family="Segoe UI, Microsoft YaHei, sans-serif" font-weight="700" fill="#24364D">Agent Understanding &amp; Output</text>',
        ]
    )
    cards = [
        ("01  Data Input", f'{payload["primarySample"]["city"]} focus subset with {summary["beforeCount"]} POIs from 6 representative categories.'),
        ("02  Generalization Request", "Reduce overlap while preserving category semantics and symbol readability for publication figures."),
        ("03  Agent Understanding", "Single-core concentration requires both geometric thinning and stable category-symbol continuity."),
        ("04  Strategy / Skills / Rules", "Category-priority filtering + minimum spacing + fixed 6-symbol grammar (shape, size, stroke, opacity)."),
        ("05  Output", f'{summary["afterCount"]} outputs remain semantically traceable in before/after/overlay comparisons.'),
    ]
    y = 212
    for title, body in cards:
        parts.append(f'<line x1="1106" y1="{y+22}" x2="1106" y2="{y+102}" stroke="#D8E3EE" stroke-width="1.4"/>')
        parts.append(f'<circle cx="1106" cy="{y+22}" r="4.8" fill="#EDF4FB" stroke="#C8D8E7"/>')
        parts.append(f'<rect x="1122" y="{y}" width="418" height="106" rx="14" fill="#FFFFFF" stroke="#E2EAF3"/>')
        parts.append(f'<text x="1140" y="{y+30}" font-size="14" font-family="Segoe UI, Microsoft YaHei, sans-serif" font-weight="700" fill="#22334D">{title}</text>')
        parts.append(f'<text x="1140" y="{y+54}" font-size="12" font-family="Segoe UI, Microsoft YaHei, sans-serif" fill="#55667D">{body}</text>')
        y += 116
    parts.append('<text x="1086" y="798" font-size="12" font-family="Segoe UI, Microsoft YaHei, sans-serif" fill="#64788F">Category Retention (Input vs Output)</text>')
    y = 824
    max_count = max([*counts_before.values(), *counts_after.values(), 1])
    for category in REPRESENTATIVE_CATEGORIES:
        before_w = 132 * counts_before.get(category, 0) / max_count
        after_w = 132 * counts_after.get(category, 0) / max_count
        parts.append(f'<text x="1086" y="{y}" font-size="12" font-family="Segoe UI, Microsoft YaHei, sans-serif" fill="#5A6B80">{category}</text>')
        parts.append(f'<rect x="1146" y="{y-8}" width="132" height="8" rx="4" fill="#EDF2F7"/>')
        parts.append(f'<rect x="1146" y="{y-8}" width="{before_w:.2f}" height="8" rx="4" fill="#B9CBE0"/>')
        parts.append(f'<rect x="1286" y="{y-8}" width="132" height="8" rx="4" fill="#EDF2F7"/>')
        parts.append(f'<rect x="1286" y="{y-8}" width="{after_w:.2f}" height="8" rx="4" fill="#5F7FA2"/>')
        parts.append(f'<text x="1430" y="{y}" font-size="11" font-family="Segoe UI, Microsoft YaHei, sans-serif" fill="#5F7388">{counts_after.get(category, 0)}/{counts_before.get(category, 0)}</text>')
        y += 20
    parts.append("</svg>")
    return "\n".join(parts)


def write_selected_samples_csv(rows):
    path = ROOT / "selected_samples.csv"
    fieldnames = [
        "structure_type_8",
        "structure_display",
        "city",
        "point_count",
        "density",
        "anisotropy",
        "selection_seed",
        "selection_pool",
        "mapping_note",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            item = dict(row)
            item["selection_pool"] = "|".join(item["selection_pool"])
            writer.writerow(item)


def build_readme(payload):
    selected = payload["selectedSamples"]
    primary = payload["primarySample"]
    lines = [
        "# Interface-display",
        "",
        "## 数据读取情况",
        f"- 读取数据目录：`{DATA_PATH.parent}`",
        f"- 主数据文件：`{DATA_PATH.name}`，自动识别城市、类别、行政区和点坐标字段。",
        f"- 结构标签文件：`{STRUCTURE_PATH}`，用于匹配论文中的 8 种空间结构。",
        "",
        "## 随机抽取的代表样本",
        f"- 固定随机种子：`{SEED}`。",
        f"- 主展示样本：`{primary['city']}`，结构类型为 `{primary['structure']}`。",
    ]
    for row in selected:
        lines.append(f"- {row['structure_type_8']}：{row['city']}（候选池：{', '.join(row['selection_pool'])}）")
    lines.extend(
        [
            "",
            "## 八种结构匹配说明",
            "- 论文结构名称与 `structure_type_8` 字段直接对应，为 1:1 匹配。",
            "- 本次未出现必须重命名的结构标签，映射说明同步记录在 `selected_samples.json/csv` 中。",
            "",
            "## 页面打开方式",
            "- 直接打开 `index.html` 即可查看静态展示页面。",
            "- 页面使用 `demo_bundle.js` 内嵌数据，不依赖本地后端服务。",
            "",
            "## 如何导出 SVG",
            "- 页面主视图与预生成 SVG 均已对齐论文展示风格，可直接截图使用。",
            "- 本轮新增 `interface_display_v2.svg`（推荐），并保留 `interface_display.svg`（原命名兼容）。",
            "",
            "## 顾及符号表达的设计实现",
            "- 不再使用单一圆点，而是使用菱形、方形、三角形、十字、星形、六边形表达不同 POI 类别。",
            "- 颜色、描边和尺寸统一在柔和浅色体系内控制，兼顾类别区分与学术图面克制感。",
            "- 综合后结果采用类别保持 + 最小间距 + 网格约束策略，以突出图面可读性。",
            "",
            "## 主要文件",
            "- `index.html`：静态展示主页面。",
            "- `styles.css`：页面样式。",
            "- `app.js`：页面交互与 SVG 导出。",
            "- `demo_bundle.js`：页面直接读取的数据包。",
            "- `selected_samples.json / csv`：8 类结构样本清单。",
            "- `demo_points.json`：前端演示数据。",
            "- `demo_data.geojson`：前后对比 GeoJSON。",
            "- `interface_display.svg`：预生成 SVG（兼容原文件名）。",
            "- `interface_display_v2.svg`：论文展示版 SVG（推荐）。",
        ]
    )
    return "\n".join(lines) + "\n"


def main():
    structure_rows = load_structure_rows()
    features = load_features()
    selected_samples = choose_structure_samples(structure_rows, features)
    primary_city_features = [item for item in features if item["city"] == PRIMARY_CITY]
    focus_points, bbox = get_focus_points(primary_city_features)
    category_counts = Counter(item["category"] for item in focus_points)
    top_categories = [name for name, _ in category_counts.most_common() if name in REPRESENTATIVE_CATEGORIES][:6]
    focus_points = [item for item in focus_points if item["category"] in top_categories]
    before = with_screen_position(focus_points, bbox)
    after = generalize_points(before)
    summary = {
        "beforeCount": len(before),
        "afterCount": len(after),
        "reductionRate": round((1 - len(after) / max(1, len(before))) * 100, 1),
        "cityFeatureCount": len(primary_city_features),
        "categoryCountBefore": dict(Counter(item["category"] for item in before)),
        "categoryCountAfter": dict(Counter(item["category"] for item in after)),
        "focusBbox": bbox,
    }
    payload = {
        "meta": {
            "title": "Symbol-aware Point Generalization Interface",
            "subtitle": "Point Generalization Agent Interface for explainable POI workflows",
            "seed": SEED,
            "sourceDirectory": str(DATA_PATH.parent),
            "sourceFile": DATA_PATH.name,
        },
        "selectedSamples": selected_samples,
        "symbolSystem": [
            {"category": category, "shape": meta["shape"], "color": meta["color"], "stroke": meta["stroke"], "priority": meta["priority"]}
            for category, meta in CATEGORY_META.items()
        ],
        "primarySample": {
            "city": PRIMARY_CITY,
            "structure": PRIMARY_STRUCTURE,
            "before": before,
            "after": after,
            "summary": summary,
            "questions": ["输入的是哪些数据？", "综合要求是什么？", "智能体如何理解？", "最终输出结果是什么？"],
        },
    }
    (ROOT / "selected_samples.json").write_text(json.dumps(selected_samples, ensure_ascii=False, indent=2), encoding="utf-8")
    write_selected_samples_csv(selected_samples)
    (ROOT / "demo_points.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (ROOT / "demo_data.geojson").write_text(json.dumps(feature_collection(before, after), ensure_ascii=False, indent=2), encoding="utf-8")
    (ROOT / "demo_bundle.js").write_text("window.INTERFACE_DISPLAY_DATA = " + json.dumps(payload, ensure_ascii=False, indent=2) + ";\n", encoding="utf-8")
    svg_markup = generate_interface_svg(payload)
    (ROOT / "interface_display.svg").write_text(svg_markup, encoding="utf-8")
    (ROOT / "interface_display_v2.svg").write_text(svg_markup, encoding="utf-8")
    (ROOT / "README.md").write_text(build_readme(payload), encoding="utf-8")
    print("Generated files:")
    for name in ["selected_samples.json", "selected_samples.csv", "demo_points.json", "demo_data.geojson", "demo_bundle.js", "interface_display.svg", "interface_display_v2.svg", "README.md"]:
        print(f"- {ROOT / name}")


if __name__ == "__main__":
    main()
