from __future__ import annotations

import base64
import html
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "TestDatasets" / "Scale_1_86000"
SYMBOL_DIR = ROOT / "output" / "arcgis_10_2_poi_symbols"
OUT_DIR = ROOT / "paper_ui_rework"


CATEGORIES = [
    ("hospital", "医院", "医疗服务"),
    ("school", "学校", "科教文化"),
    ("parking", "停车场", "停车设施"),
    ("hotel", "酒店", "住宿服务"),
    ("bank", "银行", "金融服务"),
    ("restaurant", "餐饮", "餐饮服务"),
    ("convenience_store", "便利店", "便民购物"),
    ("transit", "交通设施", "公共交通"),
]

SYMBOL_BY_CN = {
    "医院": "hospital",
    "学校": "school",
    "停车场": "parking",
    "酒店": "hotel",
    "银行": "bank",
    "餐饮": "restaurant",
    "便利店": "convenience_store",
    "交通设施": "transit",
    "公交站": "transit",
    "地铁站": "transit",
}

SAMPLES = [
    ("sample_40", "sample_40.geojson", "sample_40_gen_1_50000_r1.geojson", "台州市椒江区", "保留要素最多的综合结果"),
    ("sample_13", "sample_13.geojson", "sample_13_gen_1_50000_r1.geojson", "南京市城区样区", "密集城区多类别 POI"),
    ("sample_14", "sample_14.geojson", "sample_14_gen_1_50000_r1.geojson", "常州市城区样区", "多中心分布样区"),
    ("sample_35", "sample_35.geojson", "sample_35_gen_1_50000_r2.geojson", "长春市城区样区", "较高保留率综合结果"),
]


def read_geojson(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return [ft for ft in data.get("features", []) if ft.get("geometry", {}).get("type") == "Point"]


def feature_symbol(feature: dict) -> str:
    props = feature.get("properties") or {}
    raw = props.get("point_symbol") or props.get("symbol") or props.get("category") or props.get("query_category") or props.get("type")
    if raw in {"hospital", "school", "parking", "hotel", "bank", "restaurant", "convenience_store", "transit"}:
        return raw
    text = str(raw or "")
    for cn, sym in SYMBOL_BY_CN.items():
        if cn in text:
            return sym
    return "other"


def feature_category(feature: dict) -> str:
    sym = feature_symbol(feature)
    for item_sym, name, _ in CATEGORIES:
        if item_sym == sym:
            return name
    props = feature.get("properties") or {}
    return str(props.get("category") or props.get("query_category") or props.get("type") or "其他")


def icon_data_uri(symbol: str) -> str:
    path = SYMBOL_DIR / f"{symbol}.png"
    if not path.exists():
        path = SYMBOL_DIR / "hospital.png"
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{b64}"


ICON_URI = {sym: icon_data_uri(sym) for sym, _, _ in CATEGORIES}


def bounds(features: list[dict]) -> tuple[float, float, float, float]:
    xs = [ft["geometry"]["coordinates"][0] for ft in features]
    ys = [ft["geometry"]["coordinates"][1] for ft in features]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    padx = (maxx - minx) * 0.08 or 0.01
    pady = (maxy - miny) * 0.08 or 0.01
    return minx - padx, miny - pady, maxx + padx, maxy + pady


def point_position(feature: dict, b: tuple[float, float, float, float]) -> tuple[float, float]:
    minx, miny, maxx, maxy = b
    x, y = feature["geometry"]["coordinates"][:2]
    px = (x - minx) / (maxx - minx) * 100
    py = 100 - (y - miny) / (maxy - miny) * 100
    return px, py


def point_html(features: list[dict], b: tuple[float, float, float, float], max_points: int, dimmed: bool = False) -> str:
    ordered = sorted(features, key=lambda ft: (feature_symbol(ft) == "other", -(ft.get("properties") or {}).get("importance", 0)))
    chunks = []
    for ft in ordered[:max_points]:
        sym = feature_symbol(ft)
        if sym == "other":
            continue
        x, y = point_position(ft, b)
        name = html.escape(str((ft.get("properties") or {}).get("name", feature_category(ft))))
        cls = "poi dimmed" if dimmed else "poi"
        chunks.append(
            f'<img class="{cls}" src="{ICON_URI[sym]}" alt="{feature_category(ft)}" title="{name}" '
            f'style="left:{x:.3f}%;top:{y:.3f}%;">'
        )
    return "\n".join(chunks)


def point_svg(
    features: list[dict],
    b: tuple[float, float, float, float],
    x0: float,
    y0: float,
    width: float,
    height: float,
    max_points: int,
    dimmed: bool = False,
) -> str:
    ordered = sorted(features, key=lambda ft: (feature_symbol(ft) == "other", -(ft.get("properties") or {}).get("importance", 0)))
    size = 18 if dimmed else 23
    opacity = ".42" if dimmed else "1"
    chunks = []
    for ft in ordered[:max_points]:
        sym = feature_symbol(ft)
        if sym == "other":
            continue
        px, py = point_position(ft, b)
        x = x0 + px / 100 * width - size / 2
        y = y0 + py / 100 * height - size / 2
        chunks.append(f'<image href="{ICON_URI[sym]}" x="{x:.2f}" y="{y:.2f}" width="{size}" height="{size}" opacity="{opacity}"/>')
    return "\n".join(chunks)


def category_rows(before: list[dict], after: list[dict]) -> str:
    bc = Counter(feature_symbol(ft) for ft in before)
    ac = Counter(feature_symbol(ft) for ft in after)
    rows = []
    for sym, name, desc in CATEGORIES:
        rows.append(
            "<div class='cat-row'>"
            f"<img src='{ICON_URI[sym]}' alt='{name}'>"
            f"<div><strong>{name}</strong><span>{desc}</span></div>"
            f"<b>{ac[sym]}</b><em>/ {bc[sym]}</em>"
            "</div>"
        )
    return "\n".join(rows)


def sample_svg(sample_id: str, before_file: str, after_file: str, title: str, note: str) -> str:
    before = read_geojson(DATA_DIR / before_file)
    after = read_geojson(DATA_DIR / after_file)
    b = bounds(before + after)
    bc = Counter(feature_symbol(ft) for ft in before)
    ac = Counter(feature_symbol(ft) for ft in after)
    kept_rate = len(after) / len(before) * 100 if before else 0
    left_points = point_svg(before, b, 276, 147, 405, 603, 360, dimmed=True)
    right_points = point_svg(after, b, 698, 147, 405, 603, 320, dimmed=False)
    legend = []
    for i, (sym, name, desc) in enumerate(CATEGORIES):
        y = 520 + i * 36
        legend.append(
            f'<image href="{ICON_URI[sym]}" x="1162" y="{y}" width="24" height="24"/>'
            f'<text class="legend-name" x="1195" y="{y+15}">{name}</text>'
            f'<text class="legend-count" x="1292" y="{y+15}">{ac[sym]}</text>'
            f'<text class="legend-before" x="1330" y="{y+15}">/ {bc[sym]}</text>'
        )
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="1440" height="900" viewBox="0 0 1440 900">
<defs>
  <pattern id="grid" width="54" height="54" patternUnits="userSpaceOnUse"><path d="M54 0H0V54" fill="none" stroke="#d8e5ec" stroke-width="1"/></pattern>
  <filter id="shadow" x="-10%" y="-10%" width="120%" height="130%"><feDropShadow dx="0" dy="4" stdDeviation="6" flood-color="#1f2937" flood-opacity=".12"/></filter>
  <style>
    text{{font-family:"Microsoft YaHei","PingFang SC","Noto Sans CJK SC",Arial,sans-serif;fill:#17202a;letter-spacing:0}}
    .brand{{font-size:18px;font-weight:800;fill:#0e4f8a}} .menu{{font-size:13px;fill:#344054}}
    .head{{font-size:14px;font-weight:800}} .small{{font-size:11px;fill:#667085}}
    .panel{{fill:#fff;stroke:#d9e1ea;stroke-width:1}} .map{{fill:#edf5f4;stroke:#cbd6e2;stroke-width:1.2}}
    .map-title{{font-size:15px;font-weight:800;fill:#12344d}} .label{{font-size:12px;fill:#344054}}
    .legend-name{{font-size:12px;font-weight:700;fill:#1d2939}} .legend-count{{font-size:13px;font-weight:800;fill:#0e5e9f}} .legend-before{{font-size:11px;fill:#667085}}
  </style>
</defs>
<rect width="1440" height="900" fill="#f8fafc"/>
<rect x=".5" y=".5" width="1439" height="899" fill="none" stroke="#cfd8e3"/>
<rect x="0" y="0" width="1440" height="52" fill="#fff" stroke="#d9e1ea"/>
<text class="brand" x="22" y="33">MapGenAI 智能综合平台</text>
<text class="menu" x="232" y="32">文件</text><text class="menu" x="284" y="32">编辑</text><text class="menu" x="336" y="32">视图</text><text class="menu" x="388" y="32">图层</text><text class="menu" x="440" y="32">智能体</text><text class="menu" x="506" y="32">导出</text>
<rect x="0" y="52" width="1440" height="42" fill="#fbfcfe" stroke="#d9e1ea"/>
<rect x="14" y="59" width="54" height="28" rx="5" fill="#fff" stroke="#cfd8e3"/><text class="label" x="29" y="78">打开</text>
<rect x="76" y="59" width="54" height="28" rx="5" fill="#fff" stroke="#cfd8e3"/><text class="label" x="91" y="78">缩放</text>
<rect x="138" y="59" width="54" height="28" rx="5" fill="#fff" stroke="#cfd8e3"/><text class="label" x="153" y="78">选择</text>
<rect x="200" y="59" width="116" height="28" rx="5" fill="#edf6ff" stroke="#a8cdec"/><text class="label" x="218" y="78" fill="#0e5e9f">综合前后对比</text>
<rect class="panel" x="0" y="94" width="250" height="778"/>
<rect class="panel" x="1118" y="94" width="322" height="778"/>
<rect x="0" y="94" width="250" height="44" fill="#fbfcfe" stroke="#d9e1ea"/><text class="head" x="14" y="122">工作区</text><text class="small" x="175" y="122">{sample_id}</text>
<text class="label" x="18" y="168">数据源 {before_file}</text>
<rect x="10" y="183" width="230" height="34" rx="5" fill="#eaf4ff"/><text class="label" x="20" y="205" fill="#0e5e9f">综合结果 {after_file}</text>
<text class="label" x="18" y="246">选择依据 selection_basis.md</text>
<text class="label" x="18" y="286">导出图像 publication.svg</text>
<rect x="0" y="334" width="250" height="44" fill="#fbfcfe" stroke="#d9e1ea"/><text class="head" x="14" y="362">图层管理</text><text class="small" x="174" y="362">EPSG:4326</text>
<circle cx="24" cy="411" r="6" fill="#1d6fb8"/><text class="label" x="42" y="416">原始 POI 图层</text><text class="head" x="196" y="416">{len(before)}</text>
<circle cx="24" cy="452" r="6" fill="#2f8f83"/><text class="label" x="42" y="457">综合后 POI 图层</text><text class="head" x="196" y="457">{len(after)}</text>
<circle cx="24" cy="493" r="6" fill="#d97706"/><text class="label" x="42" y="498">八类专题符号</text><text class="head" x="202" y="498">8</text>

<rect x="266" y="147" width="405" height="603" rx="6" class="map"/>
<rect x="276" y="157" width="385" height="583" fill="url(#grid)" opacity=".9"/>
<path d="M345 414 C455 367 559 375 650 326" fill="none" stroke="#6d7e90" stroke-width="5" opacity=".26"/>
<path d="M500 270 C560 390 526 530 614 690" fill="none" stroke="#47759e" stroke-width="4" opacity=".2"/>
{left_points}
<rect x="284" y="159" width="230" height="48" rx="5" fill="#fff" fill-opacity=".95" stroke="#b4c1cf" filter="url(#shadow)"/>
<text class="map-title" x="296" y="181">综合前</text><text class="small" x="296" y="199">{title}，原始要素 {len(before)} 个</text>
<rect x="282" y="704" width="178" height="28" rx="4" fill="#fff" fill-opacity=".92" stroke="#cbd6e2"/><text class="small" x="294" y="723">1:50,000 目标尺度 · WGS84</text>

<rect x="698" y="147" width="405" height="603" rx="6" class="map"/>
<rect x="708" y="157" width="385" height="583" fill="url(#grid)" opacity=".9"/>
<path d="M777 414 C887 367 991 375 1082 326" fill="none" stroke="#6d7e90" stroke-width="5" opacity=".26"/>
<path d="M932 270 C992 390 958 530 1046 690" fill="none" stroke="#47759e" stroke-width="4" opacity=".2"/>
{right_points}
<rect x="716" y="159" width="245" height="48" rx="5" fill="#fff" fill-opacity=".95" stroke="#b4c1cf" filter="url(#shadow)"/>
<text class="map-title" x="728" y="181">综合后</text><text class="small" x="728" y="199">{note}，保留 {len(after)} 个</text>
<rect x="714" y="704" width="188" height="28" rx="4" fill="#fff" fill-opacity=".92" stroke="#cbd6e2"/><text class="small" x="726" y="723">符号保持平台原始类别样式</text>

<rect x="1118" y="94" width="322" height="44" fill="#fbfcfe" stroke="#d9e1ea"/><text class="head" x="1132" y="122">综合智能体</text><text class="small" x="1370" y="122">已完成</text>
<rect x="1132" y="156" width="282" height="56" rx="7" fill="#eef7ff" stroke="#b8d7ef"/><text class="label" x="1146" y="180">将当前 POI 图层综合至 1:50,000，</text><text class="label" x="1146" y="199">并保持八类符号可区分。</text>
<rect x="1132" y="232" width="282" height="164" rx="7" fill="#fff" stroke="#d9e1ea"/>
<text class="head" x="1148" y="262">智能体执行摘要</text>
<circle cx="1159" cy="294" r="11" fill="#e6f4ef"/><text class="small" x="1156" y="298" font-weight="800" fill="#1f7a68">1</text><text class="label" x="1178" y="299">读取 GeoJSON 与类别分布</text>
<circle cx="1159" cy="330" r="11" fill="#e6f4ef"/><text class="small" x="1156" y="334" font-weight="800" fill="#1f7a68">2</text><text class="label" x="1178" y="335">执行选取、位移与典型化</text>
<circle cx="1159" cy="366" r="11" fill="#e6f4ef"/><text class="small" x="1156" y="370" font-weight="800" fill="#1f7a68">3</text><text class="label" x="1178" y="371">复核综合后符号显示</text>
<rect x="1132" y="416" width="132" height="70" rx="6" fill="#fff" stroke="#d9e1ea"/><text x="1150" y="454" font-size="24" font-weight="800" fill="#0e5e9f">{len(after)}</text><text class="small" x="1150" y="474">保留要素</text>
<rect x="1282" y="416" width="132" height="70" rx="6" fill="#fff" stroke="#d9e1ea"/><text x="1300" y="454" font-size="24" font-weight="800" fill="#0e5e9f">{kept_rate:.1f}%</text><text class="small" x="1300" y="474">保留率</text>
<text class="head" x="1132" y="514">八类符号统计</text>
{"".join(legend)}
<rect x="0" y="872" width="1440" height="28" fill="#fff" stroke="#d9e1ea"/><text class="small" x="14" y="891">输入: {before_file} · 输出: {after_file}</text><text class="small" x="1244" y="891">MapGenAI · 论文图 SVG 重绘版</text>
</svg>"""


def sample_html(sample_id: str, before_file: str, after_file: str, title: str, note: str) -> str:
    before = read_geojson(DATA_DIR / before_file)
    after = read_geojson(DATA_DIR / after_file)
    b = bounds(before + after)
    city = html.escape(title)
    before_points = point_html(before, b, 360, dimmed=True)
    after_points = point_html(after, b, 320)
    rows = category_rows(before, after)
    kept_rate = len(after) / len(before) * 100 if before else 0
    after_counter = Counter(feature_symbol(ft) for ft in after)
    dominant = [name for sym, name, _ in CATEGORIES if after_counter[sym] > 0][:5]
    dominant_text = "、".join(dominant)
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MapGenAI 论文界面图 - {city}</title>
<style>
:root {{
  --ink:#17202a; --muted:#667085; --line:#d9e1ea; --soft:#f6f8fb; --panel:#ffffff;
  --accent:#1d6fb8; --accent-2:#2f8f83; --warn:#d97706;
  font-family:"Microsoft YaHei","PingFang SC","Noto Sans CJK SC",Arial,sans-serif;
}}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:#eef2f6; color:var(--ink); font-size:14px; }}
.canvas {{ width:1440px; height:900px; margin:0 auto; background:#f8fafc; display:grid; grid-template-rows:52px 42px 1fr 28px; border:1px solid #cfd8e3; }}
.topbar {{ display:flex; align-items:center; gap:22px; padding:0 22px; background:#fff; border-bottom:1px solid var(--line); }}
.brand {{ font-weight:800; color:#0e4f8a; font-size:18px; }}
.menu {{ color:#344054; font-size:13px; display:flex; gap:18px; }}
.toolbar {{ display:flex; align-items:center; gap:8px; padding:0 14px; background:#fbfcfe; border-bottom:1px solid var(--line); }}
.tool {{ height:28px; padding:0 10px; border:1px solid #cfd8e3; background:#fff; border-radius:5px; display:flex; align-items:center; gap:6px; font-size:12px; color:#344054; }}
.tool.active {{ color:#0e5e9f; background:#edf6ff; border-color:#a8cdec; font-weight:700; }}
.work {{ display:grid; grid-template-columns:250px 1fr 322px; min-height:0; }}
.left,.right {{ background:#fff; border-color:var(--line); min-height:0; }}
.left {{ border-right:1px solid var(--line); display:grid; grid-template-rows:44px 224px 44px 1fr; }}
.right {{ border-left:1px solid var(--line); display:grid; grid-template-rows:44px 1fr; }}
.panel-head {{ height:44px; padding:0 14px; display:flex; align-items:center; justify-content:space-between; border-bottom:1px solid var(--line); font-weight:700; background:#fbfcfe; }}
.files {{ padding:9px; display:flex; flex-direction:column; gap:5px; }}
.file {{ padding:8px 10px; border-radius:5px; display:flex; gap:8px; align-items:center; color:#344054; font-size:13px; }}
.file.selected {{ background:#eaf4ff; color:#0e5e9f; font-weight:700; }}
.layers {{ padding:12px 14px; display:flex; flex-direction:column; gap:12px; }}
.layer {{ display:grid; grid-template-columns:16px 1fr auto; gap:8px; align-items:center; font-size:12px; color:#344054; }}
.layer .swatch {{ width:12px; height:12px; border-radius:3px; background:#1d6fb8; }}
.main {{ position:relative; padding:16px; overflow:hidden; background:#eef3f7; }}
.map-grid {{ height:100%; display:grid; grid-template-columns:1fr 1fr; gap:14px; }}
.map {{ position:relative; overflow:hidden; border:1px solid #cbd6e2; background:
  linear-gradient(90deg, rgba(29,111,184,.065) 1px, transparent 1px),
  linear-gradient(0deg, rgba(29,111,184,.055) 1px, transparent 1px),
  radial-gradient(circle at 28% 24%, rgba(255,255,255,.85), transparent 30%),
  #edf5f4; background-size:54px 54px,54px 54px,100% 100%; border-radius:6px; }}
.map::before,.map::after {{ content:""; position:absolute; inset:auto; pointer-events:none; }}
.map::before {{ width:58%; height:18%; left:18%; top:40%; border-top:5px solid rgba(109,126,144,.26); transform:rotate(-17deg); border-radius:50%; }}
.map::after {{ width:46%; height:50%; left:42%; top:20%; border-left:4px solid rgba(71,117,158,.2); transform:rotate(23deg); border-radius:50%; }}
.map-title {{ position:absolute; top:12px; left:12px; z-index:4; background:rgba(255,255,255,.94); border:1px solid rgba(180,193,207,.8); border-radius:5px; padding:7px 10px; font-weight:800; color:#12344d; box-shadow:0 2px 8px rgba(31,41,55,.08); }}
.map-title span {{ display:block; color:#667085; font-size:11px; font-weight:500; margin-top:2px; }}
.scale {{ position:absolute; left:16px; bottom:16px; z-index:4; color:#344054; font-size:11px; background:rgba(255,255,255,.9); padding:6px 8px; border:1px solid #cbd6e2; border-radius:4px; }}
.poi {{ position:absolute; width:24px; height:24px; transform:translate(-50%,-50%); z-index:3; image-rendering:auto; filter:drop-shadow(0 1px 2px rgba(15,23,42,.3)); }}
.poi.dimmed {{ width:18px; height:18px; opacity:.42; filter:drop-shadow(0 1px 1px rgba(15,23,42,.18)); }}
.chat {{ padding:14px; overflow:hidden; display:flex; flex-direction:column; gap:12px; background:#f7f9fc; }}
.bubble {{ border:1px solid var(--line); background:#fff; border-radius:7px; padding:12px; line-height:1.55; }}
.bubble.user {{ background:#eef7ff; color:#0e4f8a; border-color:#b8d7ef; }}
.step {{ display:grid; grid-template-columns:22px 1fr; gap:8px; margin:8px 0; color:#344054; }}
.num {{ width:22px; height:22px; border-radius:50%; background:#e6f4ef; color:#1f7a68; display:flex; align-items:center; justify-content:center; font-weight:800; font-size:11px; }}
.metric-box {{ display:grid; grid-template-columns:1fr 1fr; gap:8px; }}
.metric {{ background:#fff; border:1px solid var(--line); border-radius:6px; padding:9px; }}
.metric b {{ display:block; font-size:20px; color:#0e5e9f; }}
.metric span {{ color:#667085; font-size:11px; }}
.cat-list {{ display:flex; flex-direction:column; gap:6px; margin-top:9px; }}
.cat-row {{ height:34px; display:grid; grid-template-columns:26px 1fr auto auto; gap:8px; align-items:center; border:1px solid #e3e9f0; background:#fff; border-radius:5px; padding:4px 7px; }}
.cat-row img {{ width:23px; height:23px; }}
.cat-row strong {{ display:block; font-size:12px; color:#1d2939; }}
.cat-row span,.cat-row em {{ color:#667085; font-style:normal; font-size:10px; }}
.cat-row b {{ color:#0e5e9f; font-size:13px; }}
.status {{ display:flex; align-items:center; justify-content:space-between; padding:0 14px; color:#667085; font-size:11px; background:#fff; border-top:1px solid var(--line); }}
</style>
</head>
<body>
<div class="canvas">
  <div class="topbar"><div class="brand">MapGenAI 智能综合平台</div><div class="menu"><span>文件</span><span>编辑</span><span>视图</span><span>图层</span><span>智能体</span><span>导出</span></div></div>
  <div class="toolbar"><div class="tool">打开</div><div class="tool">缩放</div><div class="tool">选择</div><div class="tool active">综合前后对比</div><div class="tool">符号检查</div><div class="tool">导出论文图</div></div>
  <div class="work">
    <aside class="left">
      <div class="panel-head"><span>工作区</span><span>{sample_id}</span></div>
      <div class="files">
        <div class="file selected">数据源 {before_file}</div>
        <div class="file">综合结果 {after_file}</div>
        <div class="file">选择依据 selection_basis.md</div>
        <div class="file">导出图像 publication.png</div>
      </div>
      <div class="panel-head"><span>图层管理</span><span>EPSG:4326</span></div>
      <div class="layers">
        <div class="layer"><span class="swatch"></span><span>原始 POI 图层</span><b>{len(before)}</b></div>
        <div class="layer"><span class="swatch" style="background:#2f8f83"></span><span>综合后 POI 图层</span><b>{len(after)}</b></div>
        <div class="layer"><span class="swatch" style="background:#d97706"></span><span>八类专题符号</span><b>8</b></div>
      </div>
    </aside>
    <main class="main">
      <div class="map-grid">
        <section class="map">
          <div class="map-title">综合前<span>{city}，原始要素 {len(before)} 个</span></div>
          {before_points}
          <div class="scale">1:50,000 目标尺度 · WGS84 经纬度</div>
        </section>
        <section class="map">
          <div class="map-title">综合后<span>{note}，保留 {len(after)} 个</span></div>
          {after_points}
          <div class="scale">符号保持平台原始类别样式</div>
        </section>
      </div>
    </main>
    <aside class="right">
      <div class="panel-head"><span>综合智能体</span><span>已完成</span></div>
      <div class="chat">
        <div class="bubble user">将当前 POI 图层综合至 1:50,000，并保持八类符号可区分。</div>
        <div class="bubble">
          <strong>智能体执行摘要</strong>
          <div class="step"><span class="num">1</span><span>读取样区 GeoJSON，识别坐标范围、类别分布和点密度。</span></div>
          <div class="step"><span class="num">2</span><span>依据目标尺度执行选取、去冗余、位移与典型化。</span></div>
          <div class="step"><span class="num">3</span><span>输出综合后图层，并用平台原有 POI 符号复核显示。</span></div>
        </div>
        <div class="metric-box">
          <div class="metric"><b>{len(after)}</b><span>保留要素</span></div>
          <div class="metric"><b>{kept_rate:.1f}%</b><span>保留率</span></div>
        </div>
        <div class="bubble"><strong>主要保留类别</strong><br>{dominant_text}<div class="cat-list">{rows}</div></div>
      </div>
    </aside>
  </div>
  <div class="status"><span>输入: {before_file} · 输出: {after_file}</span><span>MapGenAI · 论文图 UI 重绘版</span></div>
</div>
</body>
</html>"""


def agent_svg() -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="1440" height="900" viewBox="0 0 1440 900">
<defs>
  <filter id="shadow" x="-10%" y="-10%" width="120%" height="130%"><feDropShadow dx="0" dy="8" stdDeviation="10" flood-color="#1f2937" flood-opacity=".12"/></filter>
  <marker id="arrow" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="8" markerHeight="8" orient="auto"><path d="M0 0 L10 5 L0 10z" fill="#667085"/></marker>
  <style>
    text{{font-family:"Microsoft YaHei","PingFang SC",Arial,sans-serif;fill:#17202a;letter-spacing:0}}
    .title{{font-size:34px;font-weight:800}} .sub{{font-size:16px;fill:#667085}}
    .box{{fill:#fff;stroke:#d9e1ea;stroke-width:1.4;rx:8;filter:url(#shadow)}} .head{{font-size:20px;font-weight:800}}
    .small{{font-size:14px;fill:#667085}} .label{{font-size:15px;font-weight:700;fill:#344054}}
    .pill{{fill:#eef7ff;stroke:#b8d7ef;rx:8}} .green{{fill:#edf8f4;stroke:#b7ded3}} .orange{{fill:#fff6e9;stroke:#f3cf99}}
    .line{{stroke:#667085;stroke-width:2;fill:none;marker-end:url(#arrow)}} .thin{{stroke:#a6b3c2;stroke-width:1.4;fill:none}}
  </style>
</defs>
<rect width="1440" height="900" fill="#f6f8fb"/>
<text class="title" x="90" y="82">MapGenAI 平台智能体组成</text>
<text class="sub" x="90" y="112">由任务编排、数据剖析、规则检索、综合执行、视觉评估和成果导出智能体协同完成点要素综合。</text>

<rect class="box" x="90" y="185" width="250" height="120"/>
<text class="head" x="125" y="232">用户任务入口</text>
<text class="small" x="125" y="264">自然语言需求</text>
<text class="small" x="125" y="286">目标比例尺与图层</text>

<rect class="box" x="470" y="168" width="300" height="154"/>
<text class="head" x="510" y="220">编排智能体</text>
<text class="small" x="510" y="252">解析任务 · 制定工作流</text>
<text class="small" x="510" y="276">分配工具 · 汇总状态</text>
<rect class="pill" x="510" y="292" width="178" height="32"/><text class="label" x="532" y="314">LLM Reasoning Core</text>

<rect class="box" x="890" y="140" width="400" height="256"/>
<text class="head" x="930" y="188">专业智能体与工具集</text>
<rect class="pill" x="930" y="218" width="145" height="54"/><text class="label" x="954" y="251">数据剖析</text>
<rect class="pill" x="1100" y="218" width="145" height="54"/><text class="label" x="1124" y="251">规则检索</text>
<rect class="green" x="930" y="294" width="145" height="54"/><text class="label" x="954" y="327">综合执行</text>
<rect class="green" x="1100" y="294" width="145" height="54"/><text class="label" x="1124" y="327">质量评估</text>
<text class="small" x="930" y="376">密度、稀有性、代表性、冲突距离与符号完整性共同参与决策。</text>

<rect class="box" x="360" y="490" width="720" height="172"/>
<text class="head" x="405" y="540">GIS 数据与符号资源层</text>
<text class="small" x="405" y="574">GeoJSON 样区数据 · 选择依据 · 运行日志 · ArcGIS 10.2 POI 符号库</text>
{"".join(f'<image href="{ICON_URI[sym]}" x="{410+i*72}" y="600" width="42" height="42"/>' for i,(sym,_,_) in enumerate(CATEGORIES))}
{"".join(f'<text class="small" x="{410+i*72}" y="660">{name}</text>' for i,(_,name,_) in enumerate(CATEGORIES))}

<rect class="box" x="92" y="620" width="250" height="118"/>
<text class="head" x="126" y="666">综合前图层</text>
<text class="small" x="126" y="699">原始 POI · 高密度显示</text>

<rect class="box" x="1110" y="620" width="250" height="118"/>
<text class="head" x="1144" y="666">论文成果图</text>
<text class="small" x="1144" y="699">HTML / SVG · 清晰字体</text>

<path class="line" d="M340 245 H462"/>
<path class="line" d="M770 245 H882"/>
<path class="line" d="M620 324 V480"/>
<path class="line" d="M342 679 H352 C420 679 420 576 360 576"/>
<path class="line" d="M1080 576 C1130 576 1130 679 1100 679"/>
<path class="thin" d="M1090 396 C1090 470 980 500 920 490"/>
<text class="small" x="394" y="228">任务解析</text>
<text class="small" x="794" y="228">工具调用</text>
<text class="small" x="642" y="412">数据读写</text>
<text class="small" x="1148" y="592">可视化导出</text>
</svg>"""


def symbol_svg() -> str:
    cards = []
    for i, (sym, name, desc) in enumerate(CATEGORIES):
        x = 96 + (i % 4) * 312
        y = 190 + (i // 4) * 210
        cards.append(f"""
<rect x="{x}" y="{y}" width="250" height="150" rx="8" fill="#fff" stroke="#d9e1ea"/>
<image href="{ICON_URI[sym]}" x="{x+24}" y="{y+34}" width="72" height="72"/>
<text x="{x+116}" y="{y+62}" class="name">{name}</text>
<text x="{x+116}" y="{y+92}" class="desc">{desc}</text>
<text x="{x+116}" y="{y+120}" class="desc">平台原始符号文件</text>""")
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="1440" height="760" viewBox="0 0 1440 760">
<style>
text{{font-family:"Microsoft YaHei","PingFang SC",Arial,sans-serif;fill:#17202a;letter-spacing:0}}
.title{{font-size:34px;font-weight:800}} .sub{{font-size:16px;fill:#667085}} .name{{font-size:24px;font-weight:800}} .desc{{font-size:14px;fill:#667085}}
</style>
<rect width="1440" height="760" fill="#f6f8fb"/>
<text x="90" y="88" class="title">八类 POI 符号体系</text>
<text x="90" y="120" class="sub">符号来自平台现有 ArcGIS 10.2 POI 图标库，论文图仅调整 UI 承载形式，不改变类别符号。</text>
{"".join(cards)}
</svg>"""


def index_html(files: list[str]) -> str:
    links = "\n".join(f'<li><a href="{html.escape(name)}">{html.escape(name)}</a></li>' for name in files)
    return f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="UTF-8"><title>MapGenAI 论文 UI 图重绘</title>
<style>body{{font-family:"Microsoft YaHei","PingFang SC",Arial,sans-serif;background:#f6f8fb;color:#17202a;margin:40px}}a{{color:#0e5e9f}}li{{margin:10px 0}}</style></head>
<body><h1>MapGenAI 论文 UI 图重绘</h1><p>以下页面均为独立 HTML/SVG，可直接放入浏览器预览或用于论文截图。</p><ul>{links}</ul></body></html>"""


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    made: list[str] = []
    for sample_id, before_file, after_file, title, note in SAMPLES:
        out = OUT_DIR / f"{sample_id}_ui_zh.html"
        out.write_text(sample_html(sample_id, before_file, after_file, title, note), encoding="utf-8")
        made.append(out.name)
        svg_out = OUT_DIR / f"{sample_id}_ui_zh.svg"
        svg_out.write_text(sample_svg(sample_id, before_file, after_file, title, note), encoding="utf-8")
        made.append(svg_out.name)
    (OUT_DIR / "agent_platform_composition.svg").write_text(agent_svg(), encoding="utf-8")
    made.append("agent_platform_composition.svg")
    (OUT_DIR / "poi_symbol_system_8types.svg").write_text(symbol_svg(), encoding="utf-8")
    made.append("poi_symbol_system_8types.svg")
    (OUT_DIR / "index.html").write_text(index_html(made), encoding="utf-8")
    print(f"Wrote {len(made)} figure files to {OUT_DIR}")
    for name in made:
        print(name)


if __name__ == "__main__":
    main()
