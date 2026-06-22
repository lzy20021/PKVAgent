from pathlib import Path
import math
import io

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import requests
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "output" / "poi_raw_distribution_figure"
INPUT_CANDIDATES = [
    PROJECT_ROOT / "Datasets" / "datasets0324.shp",
    PROJECT_ROOT / "amap_poi_dataset_expanded" / "Datasets0324.shp",
    PROJECT_ROOT / "amap_poi_dataset_expanded" / "all_poi_expanded.geojson",
]

TITLE_MAIN = "Raw POI Distribution and Representative Local Spatial Patterns"
AMAP_KEY = "e89bb30fa66390ba31870478abd23bd1"
BASEMAP_ALPHA = 0.28
STORE_COLOR = "#DCE775"

CATEGORY_ORDER = [
    "\u4ea4\u901a\u8bbe\u65bd",
    "\u9910\u996e",
    "\u5b66\u6821",
    "\u533b\u9662",
    "\u94f6\u884c",
    "\u9152\u5e97",
    "\u505c\u8f66\u573a",
    "\u4fbf\u5229\u5e97",
]

CATEGORY_LABELS = {
    "\u4ea4\u901a\u8bbe\u65bd": "Transportation",
    "\u9910\u996e": "Catering",
    "\u5b66\u6821": "Schools",
    "\u533b\u9662": "Hospitals",
    "\u94f6\u884c": "Banks",
    "\u9152\u5e97": "Hotels",
    "\u505c\u8f66\u573a": "Parking",
    "\u4fbf\u5229\u5e97": "Convenience Stores",
}

CATEGORY_COLORS = {
    "\u4ea4\u901a\u8bbe\u65bd": "#2C7FB8",
    "\u9910\u996e": "#F3A35C",
    "\u5b66\u6821": "#66CCFF",
    "\u533b\u9662": "#4DD0E1",
    "\u94f6\u884c": "#8B80D8",
    "\u9152\u5e97": "#B48A78",
    "\u505c\u8f66\u573a": "#E89AD0",
    "\u4fbf\u5229\u5e97": STORE_COLOR,
}

PATTERN_WINDOWS = [
    {"label": "A", "title": "Dense Cluster", "center": (121.60, 38.90), "half_size": 0.025},
    {"label": "B", "title": "Linear Pattern", "center": (110.93, 36.69), "half_size": 0.030},
    {"label": "C", "title": "Sparse Pattern", "center": (115.56, 28.84), "half_size": 0.040},
    {"label": "D", "title": "Multi-core Pattern", "center": (110.32, 20.03), "half_size": 0.020},
]


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans"],
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.edgecolor": "#E4E8EC",
            "axes.labelcolor": "#727A82",
            "xtick.color": "#858E96",
            "ytick.color": "#858E96",
            "axes.titlecolor": "#2D3135",
            "axes.linewidth": 0.5,
            "grid.color": "#F4F4F4",
            "grid.linewidth": 0.45,
            "grid.linestyle": "-",
            "grid.alpha": 1.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def resolve_input_path() -> Path:
    for path in INPUT_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError("No POI dataset found in expected project locations.")


def load_data(path: Path) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    elif str(gdf.crs).upper() != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    gdf["type"] = gdf["type"].fillna("\u5176\u4ed6")
    gdf = gdf[gdf["type"].isin(CATEGORY_ORDER)].copy()
    return gdf


def style_axes(ax, show_labels=True, show_grid=True):
    ax.set_facecolor("white")
    ax.set_axisbelow(True)
    if show_grid:
        ax.xaxis.grid(True, color="#F4F4F4", linewidth=0.45)
        ax.yaxis.grid(True, color="#F4F4F4", linewidth=0.45)
    else:
        ax.xaxis.grid(False)
        ax.yaxis.grid(False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color("#E6EAEE")
    ax.tick_params(labelsize=9, width=0.45, length=2.8, colors="#8A939B")
    if not show_labels:
        ax.set_xticks([])
        ax.set_yticks([])


def add_north_arrow(ax):
    ax.annotate(
        "N",
        xy=(0.055, 0.935),
        xytext=(0.055, 0.875),
        xycoords="axes fraction",
        ha="center",
        va="center",
        fontsize=8.2,
        fontweight="normal",
        color="#737C84",
        arrowprops={"arrowstyle": "-|>", "lw": 0.65, "color": "#7A838B", "mutation_scale": 8},
        zorder=8,
    )


def scatter_by_category(ax, gdf, point_size, alpha, edge_width):
    for category in CATEGORY_ORDER:
        subset = gdf[gdf["type"] == category]
        if subset.empty:
            continue
        ax.scatter(
            subset.geometry.x,
            subset.geometry.y,
            s=13.6 if point_size == 13.0 else point_size,
            c=CATEGORY_COLORS[category],
            alpha=alpha,
            linewidths=edge_width,
            edgecolors="white",
            zorder=3,
            rasterized=len(subset) > 10000,
        )


def add_pattern_boxes(ax_main):
    for window in PATTERN_WINDOWS:
        cx, cy = window["center"]
        half = window["half_size"]
        ax_main.add_patch(
            mpatches.Rectangle((cx - half, cy - half), half * 2, half * 2, linewidth=0.65, edgecolor="#7A8289", facecolor="none", zorder=6)
        )
        offset = half * (1.55 if cy > 26 else -1.65)
        ax_main.text(
            cx,
            cy + offset,
            window["label"],
            ha="center",
            va="center",
            fontsize=9,
            fontweight="normal",
            color="#596067",
            bbox={"boxstyle": "round,pad=0.18,rounding_size=0.45", "facecolor": "white", "edgecolor": "#A0A7AE", "linewidth": 0.55},
            zorder=7,
        )


def plot_local_window(ax, gdf, window):
    cx, cy = window["center"]
    half = window["half_size"]
    local = gdf[gdf.geometry.x.between(cx - half, cx + half) & gdf.geometry.y.between(cy - half, cy + half)]
    scatter_by_category(ax, local, point_size=37, alpha=0.82, edge_width=0.34)
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    style_axes(ax, show_labels=False, show_grid=False)
    ax.set_title(f"{window['label']}  {window['title']}", fontsize=10.5, fontweight="normal", pad=7, loc="left")


def build_legend(fig):
    handles = [
        Line2D([0], [0], marker="o", linestyle="", label=CATEGORY_LABELS[c], markerfacecolor=CATEGORY_COLORS[c], markeredgecolor="white", markeredgewidth=0.45, markersize=8.8, alpha=0.86)
        for c in CATEGORY_ORDER
    ]
    legend = fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.035), ncol=4, frameon=False, title="POI Categories", fontsize=8.7, title_fontsize=8.9, columnspacing=1.05, handletextpad=0.38, borderaxespad=0.0)
    legend.get_title().set_color("#7A8289")


def lonlat_to_tile(lon, lat, z):
    lat = max(min(lat, 85.05112878), -85.05112878)
    x = (lon + 180.0) / 360.0 * (2 ** z)
    lat_rad = math.radians(lat)
    y = (1 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2 * (2 ** z)
    return x, y


def fetch_amap_basemap(bounds, zoom=4):
    minx, miny, maxx, maxy = bounds
    x0f, y1f = lonlat_to_tile(minx, miny, zoom)
    x1f, y0f = lonlat_to_tile(maxx, maxy, zoom)
    x0, x1 = int(math.floor(x0f)), int(math.floor(x1f))
    y0, y1 = int(math.floor(y0f)), int(math.floor(y1f))
    width = (x1 - x0 + 1) * 256
    height = (y1 - y0 + 1) * 256
    canvas = Image.new("RGBA", (width, height), (255, 255, 255, 0))
    session = requests.Session()
    for tx in range(x0, x1 + 1):
        for ty in range(y0, y1 + 1):
            url = f"https://wprd01.is.autonavi.com/appmaptile?style=7&x={tx}&y={ty}&z={zoom}&key={AMAP_KEY}"
            resp = session.get(url, timeout=15)
            resp.raise_for_status()
            tile = Image.open(io.BytesIO(resp.content)).convert("RGBA")
            canvas.paste(tile, ((tx - x0) * 256, (ty - y0) * 256))
    n = 2 ** zoom
    left = x0 / n * 360.0 - 180.0
    right = (x1 + 1) / n * 360.0 - 180.0
    top = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y0 / n))))
    bottom = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y1 + 1) / n))))
    return canvas, (left, right, bottom, top), "Amap XYZ raster tiles over HTTPS"


def draw_figure(gdf, with_basemap=True):
    fig = plt.figure(figsize=(17.2, 10.4), constrained_layout=False)
    fig.patch.set_facecolor("white")
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1.82, 1.0, 1.0], height_ratios=[1.0, 1.0], wspace=0.18, hspace=0.18)
    ax_main = fig.add_subplot(gs[:, 0])
    style_axes(ax_main, show_labels=True, show_grid=True)

    basemap_ok = False
    basemap_method = "Not used"
    basemap_error = ""
    projection_conversion = "No"

    bounds = gdf.total_bounds
    x_pad = (bounds[2] - bounds[0]) * 0.025
    y_pad = (bounds[3] - bounds[1]) * 0.03
    plot_bounds = (bounds[0] - x_pad, bounds[1] - y_pad, bounds[2] + x_pad, bounds[3] + y_pad)

    if with_basemap:
        try:
            basemap_img, basemap_extent, basemap_method = fetch_amap_basemap(plot_bounds, zoom=4)
            ax_main.imshow(basemap_img, extent=basemap_extent, origin="upper", alpha=BASEMAP_ALPHA, zorder=1)
            basemap_ok = True
        except Exception as exc:
            basemap_error = repr(exc)

    scatter_by_category(ax_main, gdf, point_size=13.0, alpha=0.76, edge_width=0.22)
    add_pattern_boxes(ax_main)
    add_north_arrow(ax_main)
    ax_main.set_xlim(plot_bounds[0], plot_bounds[2])
    ax_main.set_ylim(plot_bounds[1], plot_bounds[3])
    ax_main.set_xlabel("Longitude (\N{DEGREE SIGN}E)", fontsize=10)
    ax_main.set_ylabel("Latitude (\N{DEGREE SIGN}N)", fontsize=10)
    fig.suptitle(TITLE_MAIN, fontsize=13.6, fontweight="normal", y=0.963, color="#3A4045")

    detail_axes = [fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2])]
    for ax, window in zip(detail_axes, PATTERN_WINDOWS):
        plot_local_window(ax, gdf, window)

    build_legend(fig)
    fig.subplots_adjust(left=0.055, right=0.985, top=0.900, bottom=0.128)
    return fig, basemap_ok, basemap_method, basemap_error, projection_conversion


def write_summary(path, basemap_ok, method, error, projection_conversion):
    lines = [
        f"Convenience Stores final color: {STORE_COLOR}",
        f"Amap basemap connected successfully: {'Yes' if basemap_ok else 'No'}",
        f"Basemap integration method: {method if basemap_ok else 'Attempted Amap XYZ raster tile requests via requests/PIL fallback renderer'}",
        f"Basemap failure reason: {error if not basemap_ok else 'N/A'}",
        f"Basemap alpha: {BASEMAP_ALPHA}",
        f"Coordinate projection conversion applied: {projection_conversion}",
    ]
    if not basemap_ok:
        lines.append("Fallback outputs without basemap were generated after the tile requests failed.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_figure(fig, path):
    kwargs = dict(bbox_inches="tight", facecolor="white", edgecolor="none")
    if path.suffix.lower() == ".png":
        fig.savefig(path, dpi=600, **kwargs)
    else:
        fig.savefig(path, **kwargs)


def main():
    setup_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    input_path = resolve_input_path()
    gdf = load_data(input_path)

    fig_base, _, _, _, _ = draw_figure(gdf, with_basemap=False)
    backup_png = OUTPUT_DIR / "final_poi_raw_distribution_bright_storecolor.png"
    backup_pdf = OUTPUT_DIR / "final_poi_raw_distribution_bright_storecolor.pdf"
    save_figure(fig_base, backup_png)
    save_figure(fig_base, backup_pdf)
    plt.close(fig_base)

    fig_map, basemap_ok, method, error, projection_conversion = draw_figure(gdf, with_basemap=True)
    with_png = OUTPUT_DIR / "final_poi_raw_distribution_with_basemap.png"
    with_pdf = OUTPUT_DIR / "final_poi_raw_distribution_with_basemap.pdf"
    with_svg = OUTPUT_DIR / "final_poi_raw_distribution_with_basemap.svg"
    save_figure(fig_map, with_png)
    save_figure(fig_map, with_pdf)
    save_figure(fig_map, with_svg)
    plt.close(fig_map)

    script_copy = OUTPUT_DIR / "plot_rebuild_with_basemap.py"
    if Path(__file__).resolve() != script_copy.resolve():
        script_copy.write_text(Path(__file__).read_text(encoding="utf-8"), encoding="utf-8")

    summary_path = OUTPUT_DIR / "basemap_summary.txt"
    write_summary(summary_path, basemap_ok, method, error, projection_conversion)

    print(f"INPUT={input_path}")
    print(f"POINTS={len(gdf)}")
    print(f"OUTPUT={with_png}")
    print(f"OUTPUT={with_pdf}")
    print(f"OUTPUT={with_svg}")
    print(f"OUTPUT={backup_png}")
    print(f"OUTPUT={backup_pdf}")
    print(f"OUTPUT={script_copy}")
    print(f"OUTPUT={summary_path}")
    print(f"BASEMAP_OK={basemap_ok}")


if __name__ == "__main__":
    main()
