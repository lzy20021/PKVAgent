"""Point symbol names and categorical mapping (no Qt — safe for render threads)."""

from __future__ import annotations

from typing import Any, Dict, List, Set

_POINT_SYMBOL_ALIASES = {
    "circle": "circle",
    "圆": "circle",
    "圆形": "circle",
    "square": "square",
    "方": "square",
    "方形": "square",
    "rectangle": "square",
    "triangle": "triangle",
    "三角": "triangle",
    "三角形": "triangle",
    "diamond": "diamond",
    "菱形": "diamond",
    "cross": "cross",
    "叉": "cross",
    "叉号": "cross",
    "x": "cross",
    "star": "star",
    "星": "star",
    "五角星": "star",
    "plus": "plus",
    "加号": "plus",
    "+": "plus",
    "hexagon": "hexagon",
    "六边形": "hexagon",
    "六边": "hexagon",
    "convenience": "convenience",
    "shop": "convenience",
    "store": "convenience",
    "school": "school",
    "education": "school",
    "parking": "parking",
    "park": "parking",
    "hotel": "hotel",
    "lodging": "hotel",
    "bank": "bank",
    "atm": "bank",
    "hospital": "hospital",
    "medical": "hospital",
    "restaurant": "restaurant",
    "food": "restaurant",
    "catering": "restaurant",
    "transit": "transit",
    "transport": "transit",
    "bus": "transit",
    "traffic": "transit",
    "交通设施": "transit",
    "交通": "transit",
    "公交站": "transit",
    "公交": "transit",
    "地铁站": "transit",
    "便利店": "convenience",
    "超市": "convenience",
    "学校": "school",
    "停车场": "parking",
    "停车": "parking",
    "酒店": "hotel",
    "宾馆": "hotel",
    "银行": "bank",
    "医院": "hospital",
    "诊所": "hospital",
    "餐饮": "restaurant",
    "餐厅": "restaurant",
    "饭店": "restaurant",
}

POI_SYMBOL_RULES: Dict[str, Dict[str, str]] = {
    "便利店": {"symbol": "convenience", "color": "#d23b7d", "label": ""},
    "学校": {"symbol": "school", "color": "#2e7d32", "label": "S"},
    "停车场": {"symbol": "parking", "color": "#1976d2", "label": "P"},
    "酒店": {"symbol": "hotel", "color": "#7b4fa1", "label": "H"},
    "银行": {"symbol": "bank", "color": "#00897b", "label": "B"},
    "医院": {"symbol": "hospital", "color": "#d32f2f", "label": "+"},
    "餐饮": {"symbol": "restaurant", "color": "#ef8a24", "label": "R"},
    "交通设施": {"symbol": "transit", "color": "#4e79a7", "label": "T"},
}

POI_CATEGORY_ALIASES: Dict[str, str] = {
    "便利店": "便利店",
    "便利": "便利店",
    "超市": "便利店",
    "商店": "便利店",
    "convenience": "便利店",
    "shop": "便利店",
    "store": "便利店",
    "学校": "学校",
    "大学": "学校",
    "中学": "学校",
    "小学": "学校",
    "幼儿园": "学校",
    "school": "学校",
    "education": "学校",
    "停车场": "停车场",
    "停车": "停车场",
    "parking": "停车场",
    "park": "停车场",
    "酒店": "酒店",
    "宾馆": "酒店",
    "住宿": "酒店",
    "hotel": "酒店",
    "lodging": "酒店",
    "银行": "银行",
    "ATM": "银行",
    "atm": "银行",
    "bank": "银行",
    "医院": "医院",
    "诊所": "医院",
    "医疗": "医院",
    "hospital": "医院",
    "medical": "医院",
    "餐饮": "餐饮",
    "餐厅": "餐饮",
    "饭店": "餐饮",
    "美食": "餐饮",
    "restaurant": "餐饮",
    "food": "餐饮",
    "catering": "餐饮",
    "交通设施": "交通设施",
    "交通": "交通设施",
    "公交站": "交通设施",
    "公交": "交通设施",
    "地铁站": "交通设施",
    "地铁": "交通设施",
    "客运站": "交通设施",
    "火车站": "交通设施",
    "transport": "交通设施",
    "transit": "交通设施",
    "bus": "交通设施",
    "traffic": "交通设施",
}

KNOWN_POINT_SYMBOLS = (
    "circle",
    "square",
    "triangle",
    "diamond",
    "cross",
    "star",
    "plus",
    "hexagon",
    "convenience",
    "school",
    "parking",
    "hotel",
    "bank",
    "hospital",
    "restaurant",
    "transit",
)

SYMBOL_SEQUENCE: List[str] = list(KNOWN_POINT_SYMBOLS)

# 优先匹配的字段名（精确）
PRIORITY_FIELDS: List[str] = [
    "point_symbol",
    "demo_type",
    "query_category",
    "poi_type",
    "fclass",
    "type",
    "category",
    "class",
    "kind",
]

_SKIP_SUBSTRING_COLS: Set[str] = {
    "geometry",
    "id",
    "fid",
    "x",
    "y",
    "lon",
    "lat",
    "lng",
    "longitude",
    "latitude",
    "index",
    "objectid",
    "osm_id",
}


def normalize_point_symbol(raw: str | None) -> str:
    if raw is None:
        return "circle"
    key = str(raw).strip().lower()
    if key in _POINT_SYMBOL_ALIASES:
        return _POINT_SYMBOL_ALIASES[key]
    s = str(raw).strip()
    if s in _POINT_SYMBOL_ALIASES:
        return _POINT_SYMBOL_ALIASES[s]
    if key in KNOWN_POINT_SYMBOLS:
        return key
    return "circle"


def canonicalize_poi_category(raw: Any) -> str:
    text = "" if raw is None else str(raw).strip()
    if not text or text.lower() == "nan":
        return ""
    if text in POI_CATEGORY_ALIASES:
        return POI_CATEGORY_ALIASES[text]
    low = text.lower()
    if low in POI_CATEGORY_ALIASES:
        return POI_CATEGORY_ALIASES[low]
    for key, category in POI_CATEGORY_ALIASES.items():
        if key and key in text:
            return category
    return text


def poi_symbol_for_category(raw: Any) -> str:
    category = canonicalize_poi_category(raw)
    rule = POI_SYMBOL_RULES.get(category)
    return normalize_point_symbol(rule["symbol"]) if rule else ""


def poi_color_for_category(raw: Any) -> str:
    category = canonicalize_poi_category(raw)
    rule = POI_SYMBOL_RULES.get(category)
    return rule["color"] if rule else ""


def assign_symbols_to_categories(categories: List[str]) -> Dict[str, str]:
    """Stable mapping: each distinct category string -> rotating canonical symbol."""
    cats = sorted({str(c) for c in categories if c is not None})
    out: Dict[str, str] = {}
    used = set()
    for c in cats:
        fixed = poi_symbol_for_category(c)
        if fixed:
            out[c] = fixed
            used.add(fixed)
    for i, c in enumerate(cats):
        if c in out:
            continue
        out[c] = SYMBOL_SEQUENCE[(i + len(used)) % len(SYMBOL_SEQUENCE)]
    return out


def _field_usable(gdf: Any, col: str) -> Dict[str, Any] | None:
    import pandas as pd

    if col not in gdf.columns:
        return None
    ser = gdf[col]
    nunique = int(ser.nunique(dropna=True))
    if nunique < 2 or nunique > 64:
        return None

    uniq_vals = sorted({str(v) if pd.notna(v) else "" for v in ser.unique().tolist()})
    if len(uniq_vals) < 2:
        return None

    if str(col).strip().lower() == "point_symbol":
        mmap = {k: normalize_point_symbol(k) for k in uniq_vals}
        return {"point_symbol_field": col, "point_symbol_map": mmap}

    mmap = assign_symbols_to_categories(uniq_vals)
    return {"point_symbol_field": col, "point_symbol_map": mmap}


def _name_heuristic_col(gdf: Any, col: str) -> bool:
    if col in _SKIP_SUBSTRING_COLS:
        return False
    sl = str(col).lower()
    if "geom" in sl and col != "geometry":
        return False
    if any(k in sl for k in ("type", "category", "class", "kind")):
        return True
    label = str(col)
    return "类型" in label or "种类" in label or "类别" in label


def infer_point_symbol_style_from_gdf(gdf: Any) -> Dict[str, Any]:
    """
    If point layer has a suitable categorical column (2–64 classes), return
    point_symbol_field + point_symbol_map for differentiated rendering.
    """
    if gdf is None or getattr(gdf, "empty", True) or "geometry" not in getattr(gdf, "columns", []):
        return {}

    for col in PRIORITY_FIELDS:
        r = _field_usable(gdf, col)
        if r:
            return r

    for col in gdf.columns:
        if col == "geometry" or col in PRIORITY_FIELDS:
            continue
        if not _name_heuristic_col(gdf, col):
            continue
        r = _field_usable(gdf, col)
        if r:
            return r

    return {}
