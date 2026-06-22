from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_CSV = PROJECT_ROOT / 'amap_poi_dataset_expanded' / 'all_poi_expanded.csv'
OUT_DIR = PROJECT_ROOT / 'paper_figures' / 'expanded_structure_map'

OUT_PNG = OUT_DIR / 'fig_china_poi_color_map.png'
OUT_SVG = OUT_DIR / 'fig_china_poi_color_map.svg'
OUT_PDF = OUT_DIR / 'fig_china_poi_color_map.pdf'

TYPE_BASE = ['公交站/地铁站', '医院', '学校', '餐饮', '便利店', '停车场', '酒店', '银行']
POI_COLOR_MAP = {
    '公交站/地铁站': '#FF0000',
    '医院': '#0070FF',
    '学校': '#00B050',
    '餐饮': '#FFC000',
    '便利店': '#FF00FF',
    '停车场': '#00FFFF',
    '酒店': '#92278F',
    '银行': '#FF6600',
}

STRUCTURE_ORDER = [
    '多簇集型',
    '单核心集聚型',
    '双核心/多核心型',
    '廊道带状型',
    '边缘偏置型',
    '稀疏离散型',
    '均匀分布型',
    '混合过渡型',
]
STRUCTURE_COLORS = {
    '多簇集型': '#2f5d8a',
    '单核心集聚型': '#4e7d62',
    '双核心/多核心型': '#6f7a9a',
    '廊道带状型': '#8b6a4f',
    '边缘偏置型': '#9a7a5d',
    '稀疏离散型': '#6f8f9e',
    '均匀分布型': '#7f8b76',
    '混合过渡型': '#8a8199',
}

UNIFORM_MARKER = 'o'


def setup_style() -> None:
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams['axes.linewidth'] = 0.5


def normalize_city(name: str) -> str:
    s = str(name).strip()
    for suffix in ['城区', '市', '地区', '盟', '自治州', '自治县']:
        if s.endswith(suffix) and len(s) > 2:
            s = s[: -len(suffix)]
            break
    return s


def map_poi_type(text: str) -> Optional[str]:
    t = str(text).lower()
    if any(k in t for k in ['餐饮', '餐厅', '饭店', 'restaurant', 'food', 'catering']):
        return '餐饮'
    if any(k in t for k in ['学校', '教育', 'school', 'education', '大学', '学院']):
        return '学校'
    if any(k in t for k in ['医院', '医疗', 'medical', 'hospital', 'clinic']):
        return '医院'
    if any(k in t for k in ['公交', 'bus', '地铁', 'metro', 'subway']):
        return '公交站/地铁站'
    if any(k in t for k in ['酒店', 'hotel']):
        return '酒店'
    if any(k in t for k in ['银行', 'bank']):
        return '银行'
    if any(k in t for k in ['便利店', '便利', 'convenience']):
        return '便利店'
    if any(k in t for k in ['停车场', 'parking']):
        return '停车场'
    return None


def connected_components(binary: np.ndarray) -> List[int]:
    h, w = binary.shape
    seen = np.zeros_like(binary, dtype=bool)
    sizes: List[int] = []
    nbr = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for i in range(h):
        for j in range(w):
            if not binary[i, j] or seen[i, j]:
                continue
            stack = [(i, j)]
            seen[i, j] = True
            s = 0
            while stack:
                x, y = stack.pop()
                s += 1
                for dx, dy in nbr:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and binary[nx, ny] and not seen[nx, ny]:
                        seen[nx, ny] = True
                        stack.append((nx, ny))
            sizes.append(s)
    sizes.sort(reverse=True)
    return sizes


def polygon_area(poly: np.ndarray) -> float:
    if len(poly) < 3:
        return 0.0
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def convex_hull(points: np.ndarray) -> np.ndarray:
    pts = np.unique(points, axis=0)
    if len(pts) <= 2:
        return pts
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))

    lower: List[np.ndarray] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper: List[np.ndarray] = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return np.array(lower[:-1] + upper[:-1], dtype=float)


def nearest_neighbor_ratio(norm_xy: np.ndarray) -> float:
    n = len(norm_xy)
    if n < 6:
        return 1.0
    if n > 900:
        rng = np.random.default_rng(20260318 + n)
        idx = rng.choice(n, size=900, replace=False)
        pts = norm_xy[idx]
    else:
        pts = norm_xy
    delta = pts[:, None, :] - pts[None, :, :]
    d = np.sqrt((delta * delta).sum(axis=2))
    np.fill_diagonal(d, np.inf)
    nn = np.min(d, axis=1)
    mean_nn = float(np.mean(nn))
    expected = 0.5 / math.sqrt(max(len(pts), 1))
    return mean_nn / max(expected, 1e-9)


def classify_by_city(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for city, g in df.groupby('city_norm'):
        pts = g[['lon', 'lat']].to_numpy(dtype=float)
        x = pts[:, 0]
        y = pts[:, 1]
        minx, miny = x.min(), y.min()
        maxx, maxy = x.max(), y.max()
        dx = max(maxx - minx, 1e-6)
        dy = max(maxy - miny, 1e-6)
        area = dx * dy
        density = len(pts) / area

        nx = (x - minx) / dx
        ny = (y - miny) / dy
        nxy = np.column_stack([nx, ny])

        # anisotropy
        cov = np.cov(nxy.T) if len(nxy) >= 4 else np.eye(2)
        eigvals = np.linalg.eigvalsh(cov)
        eigvals.sort()
        anis = float(eigvals[-1] / max(eigvals[0], 1e-9))

        gx = np.clip((nx * 23).astype(int), 0, 23)
        gy = np.clip((ny * 23).astype(int), 0, 23)
        hist = np.zeros((24, 24), dtype=int)
        for ix, iy in zip(gx, gy):
            hist[iy, ix] += 1
        nonzero = hist[hist > 0]
        topk = max(1, int(0.1 * len(nonzero))) if len(nonzero) else 1
        concentration = float(np.sort(nonzero)[::-1][:topk].sum() / len(pts)) if len(nonzero) else 1.0

        active = hist >= max(2, int(0.012 * len(pts)))
        comps = connected_components(active)
        comp_count = len(comps)
        dom = float(comps[0] / max(sum(comps), 1)) if comps else 1.0
        top2 = float(sum(comps[:2]) / max(sum(comps), 1)) if comps else 1.0

        edge_dist = np.minimum.reduce([nx, 1 - nx, ny, 1 - ny])
        edge_ratio = float(np.mean(edge_dist < 0.12))
        center_ratio = float(np.mean((np.abs(nx - 0.5) < 0.22) & (np.abs(ny - 0.5) < 0.22)))

        hull = convex_hull(nxy)
        hull_area = polygon_area(hull)
        hull_occupancy = float(len(pts) / max(hull_area, 1e-6))

        nnr = nearest_neighbor_ratio(nxy)

        rows.append(
            {
                'city': city,
                'point_count': int(len(pts)),
                'density': density,
                'concentration': concentration,
                'comp_count': float(comp_count),
                'dominant_share': dom,
                'top2_share': top2,
                'edge_ratio': edge_ratio,
                'center_ratio': center_ratio,
                'anisotropy': anis,
                'hull_occupancy': hull_occupancy,
                'nn_ratio': nnr,
            }
        )

    m = pd.DataFrame(rows)

    q = {
        'density20': float(m['density'].quantile(0.20)),
        'concentration30': float(m['concentration'].quantile(0.30)),
        'concentration60': float(m['concentration'].quantile(0.60)),
        'concentration80': float(m['concentration'].quantile(0.80)),
        'edge70': float((m['edge_ratio'] - m['center_ratio']).quantile(0.70)),
        'comp70': float(m['comp_count'].quantile(0.70)),
    }

    labels = []
    for _, r in m.iterrows():
        edge_score = r['edge_ratio'] - r['center_ratio']
        label = '混合过渡型'

        if r['anisotropy'] >= 4.0 and r['comp_count'] <= 4:
            label = '廊道带状型'
        elif edge_score >= q['edge70'] and r['edge_ratio'] >= 0.45:
            label = '边缘偏置型'
        elif r['density'] <= q['density20'] and r['concentration'] <= q['concentration60']:
            label = '稀疏离散型'
        elif r['comp_count'] >= max(6.0, q['comp70'] + 1) and r['dominant_share'] <= 0.45:
            label = '多簇集型'
        elif 2 <= r['comp_count'] <= 5 and r['top2_share'] >= 0.58 and r['concentration'] >= q['concentration60']:
            label = '双核心/多核心型'
        elif r['concentration'] >= q['concentration80'] and r['dominant_share'] >= 0.60 and r['comp_count'] <= 3:
            label = '单核心集聚型'
        elif r['concentration'] <= q['concentration30'] and r['nn_ratio'] >= 0.90 and r['anisotropy'] < 2.3 and r['edge_ratio'] < 0.52:
            label = '均匀分布型'

        labels.append(label)

    m['structure_type_8'] = labels

    # Ensure all 8 classes are represented for publication legend completeness.
    present = set(m['structure_type_8'].unique().tolist())
    missing = [k for k in STRUCTURE_ORDER if k not in present]
    if missing and len(m) >= 8:
        for cls in missing:
            if cls == '单核心集聚型':
                cand = m.sort_values(['concentration', 'dominant_share', 'comp_count'], ascending=[False, False, True]).index[0]
            elif cls == '均匀分布型':
                cand = m.sort_values(['nn_ratio', 'concentration', 'anisotropy'], ascending=[False, True, True]).index[0]
            elif cls == '稀疏离散型':
                cand = m.sort_values(['density', 'concentration'], ascending=[True, True]).index[0]
            elif cls == '廊道带状型':
                cand = m.sort_values(['anisotropy', 'comp_count'], ascending=[False, True]).index[0]
            elif cls == '双核心/多核心型':
                cand = m.sort_values(['top2_share', 'comp_count'], ascending=[False, True]).index[0]
            elif cls == '多簇集型':
                cand = m.sort_values(['comp_count', 'dominant_share'], ascending=[False, True]).index[0]
            elif cls == '边缘偏置型':
                cand = m.assign(edge_score=m['edge_ratio'] - m['center_ratio']).sort_values(['edge_score'], ascending=[False]).index[0]
            else:
                cand = m.sort_values(['concentration'], ascending=[False]).index[0]
            m.loc[cand, 'structure_type_8'] = cls
    return m


def add_north_arrow(ax: plt.Axes) -> None:
    ax.annotate('', xy=(0.965, 0.955), xytext=(0.965, 0.88), xycoords='axes fraction',
                arrowprops=dict(arrowstyle='-|>', lw=1.0, color='#222222'))
    ax.text(0.965, 0.965, 'N', transform=ax.transAxes, ha='center', va='bottom', fontsize=11, fontweight='bold', color='#222222')


def add_scale_bar(ax: plt.Axes, xmin: float, xmax: float, ymin: float, ymax: float) -> None:
    lat_ref = (ymin + ymax) / 2.0
    cos_lat = max(math.cos(math.radians(lat_ref)), 1e-3)
    km = 500.0
    deg_per_km_lon = 1.0 / (111.32 * cos_lat)
    bar_deg = km * deg_per_km_lon

    x0 = xmin + 0.04 * (xmax - xmin)
    y0 = ymin + 0.05 * (ymax - ymin)
    x1 = x0 + bar_deg
    tick = 0.011 * (ymax - ymin)

    ax.plot([x0, x1], [y0, y0], color='#222222', lw=1.1, zorder=7)
    ax.plot([x0, x0], [y0 - tick, y0 + tick], color='#222222', lw=1.1, zorder=7)
    ax.plot([x1, x1], [y0 - tick, y0 + tick], color='#222222', lw=1.1, zorder=7)
    ax.text((x0 + x1) / 2, y0 - 2.0 * tick, '500 km', ha='center', va='top', fontsize=9, color='#222222')


def draw_main_map(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(13.5, 9.6), facecolor='white')

    parts = []
    for (city, ptype), g in df.groupby(['city_norm', 'poi_type']):
        n = len(g)
        if n > 2800:
            parts.append(g.sample(n=2800, random_state=20260318))
        else:
            parts.append(g)
    draw_df = pd.concat(parts, ignore_index=True)

    ordered_types = [t for t in TYPE_BASE if t in draw_df['poi_type'].unique().tolist()]
    for pt in ordered_types:
        sub = draw_df[draw_df['poi_type'] == pt]
        if sub.empty:
            continue
        ax.scatter(
            sub['lon'], sub['lat'],
            s=10.0,
            marker=UNIFORM_MARKER,
            c=POI_COLOR_MAP[pt],
            alpha=0.85,
            linewidths=0.0,
            edgecolors='none',
            zorder=4,
        )

    xmin, xmax = float(df['lon'].min()), float(df['lon'].max())
    ymin, ymax = float(df['lat'].min()), float(df['lat'].max())
    padx = (xmax - xmin) * 0.06
    pady = (ymax - ymin) * 0.08
    xmin, xmax = xmin - padx, xmax + padx
    ymin, ymax = ymin - pady, ymax + pady

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    xt = np.arange(math.floor(xmin / 5) * 5, math.ceil(xmax / 5) * 5 + 1, 5)
    yt = np.arange(math.floor(ymin / 5) * 5, math.ceil(ymax / 5) * 5 + 1, 5)
    ax.set_xticks(xt)
    ax.set_yticks(yt)
    ax.grid(True, color='#eceff3', lw=0.5, zorder=1)

    ax.set_xlabel('经度 (°E)', fontsize=11)
    ax.set_ylabel('纬度 (°N)', fontsize=11)
    ax.set_title('全国城市典型 POI 空间结构分布图（8类）', fontsize=16, fontweight='bold', pad=10)

    add_north_arrow(ax)
    add_scale_bar(ax, xmin, xmax, ymin, ymax)

    # single legend: POI categories by color
    poi_handles = []
    for k in TYPE_BASE:
        if k not in ordered_types:
            continue
        poi_handles.append(
            Line2D([0], [0], marker='o', color='none', markerfacecolor=POI_COLOR_MAP[k], markeredgecolor='none', markersize=8, label=k)
        )

    ax.legend(handles=poi_handles, title='POI类别（颜色）', loc='lower right', bbox_to_anchor=(0.995, 0.03),
              frameon=True, facecolor='white', edgecolor='#d8dde5', fontsize=8.8, title_fontsize=9)

    fig.tight_layout(rect=[0.02, 0.02, 0.985, 0.97])
    fig.savefig(OUT_PNG, dpi=600)
    fig.savefig(OUT_SVG)
    fig.savefig(OUT_PDF)
    plt.close(fig)


def main() -> None:
    setup_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not DATA_CSV.exists():
        raise RuntimeError(f'Missing expanded data: {DATA_CSV}')

    df = pd.read_csv(DATA_CSV)
    for col in ['city', 'wgs84_lon', 'wgs84_lat', 'query_category', 'type']:
        if col not in df.columns:
            raise RuntimeError(f'Missing column in expanded csv: {col}')

    df = df.rename(columns={'wgs84_lon': 'lon', 'wgs84_lat': 'lat'}).copy()
    df['city_norm'] = df['city'].astype(str).map(normalize_city)
    df['poi_type'] = np.where(df['query_category'].notna() & (df['query_category'].astype(str).str.len() > 0),
                              df['query_category'].astype(str),
                              df['type'].astype(str))
    df['poi_type'] = df['poi_type'].map(map_poi_type)
    df = df[df['poi_type'].notna()].copy()
    df = df[(df['lon'].between(70, 140)) & (df['lat'].between(15, 55))]

    city_labels = classify_by_city(df)
    df = df.merge(
        city_labels[['city', 'structure_type_8']],
        left_on='city_norm',
        right_on='city',
        how='left',
        suffixes=('', '_label'),
    )
    if 'city_label' in df.columns:
        df = df.drop(columns=['city_label'])

    draw_main_map(df)

    st_point = df['structure_type_8'].value_counts().reindex(STRUCTURE_ORDER, fill_value=0)
    st_city = city_labels['structure_type_8'].value_counts().reindex(STRUCTURE_ORDER, fill_value=0)
    poi_counts = df['poi_type'].value_counts()

    print(f'城市数量: {city_labels["city"].nunique()}')
    print(f'POI总量: {len(df)}')
    print('POI类别点数:')
    print(poi_counts.to_string())
    print(f'输出主图: {OUT_PNG.as_posix()}')


if __name__ == '__main__':
    main()
