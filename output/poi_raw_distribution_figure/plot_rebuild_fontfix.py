from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib import font_manager as fm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "output" / "poi_raw_distribution_figure"
INPUT_CANDIDATES = [
    PROJECT_ROOT / "Datasets" / "datasets0324.shp",
    PROJECT_ROOT / "amap_poi_dataset_expanded" / "Datasets0324.shp",
    PROJECT_ROOT / "amap_poi_dataset_expanded" / "all_poi_expanded.geojson",
]

TITLE_MAIN = "Raw POI Distribution and Representative Local Spatial Patterns"
STORE_COLOR = "#D8F04A"
FONT_CANDIDATES = ["Microsoft YaHei", "????", "SimHei"]

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


def choose_font() -> str:
    installed = {f.name for f in fm.fontManager.ttflist}
    for name in FONT_CANDIDATES:
        if name in installed:
            return name
    return "DejaVu Sans"


FONT_NAME = choose_font()


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [FONT_NAME],
            "axes.unicode_minus": False,
            "text.antialiased": True,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "none",
            "axes.edgecolor": "#E4E8EC",
            "axes.labelcolor": "#6E767E",
            "xtick.color": "#848D95",
            "ytick.color": "#848D95",
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
        fontsize=8.6,
        fontweight="semibold",
        color="#6D757D",
        fontfamily=FONT_NAME,
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
            s=point_size,
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
            fontweight="semibold",
            fontfamily=FONT_NAME,
            color="#596067",
            bbox={"boxstyle": "round,pad=0.18,rounding_size=0.45", "facecolor": "white", "edgecolor": "#A0A7AE", "linewidth": 0.55},
            zorder=7,
        )


def style_local_title(ax, text):
    title = ax.set_title(text, fontsize=10.7, fontweight="semibold", pad=8, loc="left", fontfamily=FONT_NAME)
    title.set_position((0.0, 1.02))


def plot_local_window(ax, gdf, window):
    cx, cy = window["center"]
    half = window["half_size"]
    local = gdf[gdf.geometry.x.between(cx - half, cx + half) & gdf.geometry.y.between(cy - half, cy + half)]
    scatter_by_category(ax, local, point_size=37, alpha=0.82, edge_width=0.34)
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    style_axes(ax, show_labels=False, show_grid=False)
    style_local_title(ax, f"{window['label']}  {window['title']}")


def build_legend(fig):
    handles = [
        Line2D([0], [0], marker="o", linestyle="", label=CATEGORY_LABELS[c], markerfacecolor=CATEGORY_COLORS[c], markeredgecolor="white", markeredgewidth=0.45, markersize=9.0, alpha=0.88)
        for c in CATEGORY_ORDER
    ]
    legend = fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.035),
        ncol=4,
        frameon=False,
        title="POI Categories",
        fontsize=8.9,
        title_fontsize=9.2,
        columnspacing=1.08,
        handletextpad=0.38,
        borderaxespad=0.0,
        prop={"family": FONT_NAME, "weight": "normal", "size": 8.9},
    )
    legend.get_title().set_color("#707981")
    legend.get_title().set_fontfamily(FONT_NAME)
    legend.get_title().set_fontweight("semibold")


def write_summary(path: Path):
    lines = [
        f"Final font name: {FONT_NAME}",
        "Main title weight: semibold",
        "Local subplot title weight: semibold",
        "Axis label weight: semibold",
        "Legend title weight: semibold",
        f"Convenience Stores final color: {STORE_COLOR}",
        "Subplot title alignment adjustment: all four local subplot titles use the same axes title mechanism with loc='left', identical pad=8, and a unified title position of x=0.0, y=1.02.",
        "Export clarity settings: PNG saved at 900 dpi; PDF/SVG kept as vector outputs with pdf.fonttype=42 and svg.fonttype='none'.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_figure(fig, path: Path):
    kwargs = dict(bbox_inches="tight", facecolor="white", edgecolor="none")
    if path.suffix.lower() == ".png":
        fig.savefig(path, dpi=900, **kwargs)
    else:
        fig.savefig(path, **kwargs)


def main():
    setup_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    input_path = resolve_input_path()
    gdf = load_data(input_path)

    fig = plt.figure(figsize=(17.2, 10.4), constrained_layout=False)
    fig.patch.set_facecolor("white")
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1.82, 1.0, 1.0], height_ratios=[1.0, 1.0], wspace=0.18, hspace=0.18)

    ax_main = fig.add_subplot(gs[:, 0])
    style_axes(ax_main, show_labels=True, show_grid=True)
    scatter_by_category(ax_main, gdf, point_size=13.0, alpha=0.76, edge_width=0.22)
    add_pattern_boxes(ax_main)
    add_north_arrow(ax_main)

    bounds = gdf.total_bounds
    x_pad = (bounds[2] - bounds[0]) * 0.025
    y_pad = (bounds[3] - bounds[1]) * 0.03
    ax_main.set_xlim(bounds[0] - x_pad, bounds[2] + x_pad)
    ax_main.set_ylim(bounds[1] - y_pad, bounds[3] + y_pad)
    ax_main.set_xlabel("Longitude (\N{DEGREE SIGN}E)", fontsize=10, fontweight="semibold", fontfamily=FONT_NAME)
    ax_main.set_ylabel("Latitude (\N{DEGREE SIGN}N)", fontsize=10, fontweight="semibold", fontfamily=FONT_NAME)
    ax_main.set_title("")
    fig.suptitle(TITLE_MAIN, fontsize=13.5, fontweight="semibold", y=0.963, color="#3A4045", fontfamily=FONT_NAME)

    for label in ax_main.get_xticklabels() + ax_main.get_yticklabels():
        label.set_fontfamily(FONT_NAME)
        label.set_fontweight("normal")

    detail_axes = [
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[0, 2]),
        fig.add_subplot(gs[1, 1]),
        fig.add_subplot(gs[1, 2]),
    ]
    for ax, window in zip(detail_axes, PATTERN_WINDOWS):
        plot_local_window(ax, gdf, window)

    build_legend(fig)
    fig.subplots_adjust(left=0.055, right=0.985, top=0.900, bottom=0.128)

    out_png = OUTPUT_DIR / "final_poi_raw_distribution_fontfix.png"
    out_pdf = OUTPUT_DIR / "final_poi_raw_distribution_fontfix.pdf"
    out_svg = OUTPUT_DIR / "final_poi_raw_distribution_fontfix.svg"
    save_figure(fig, out_png)
    save_figure(fig, out_pdf)
    save_figure(fig, out_svg)
    plt.close(fig)

    summary_path = OUTPUT_DIR / "fontfix_summary.txt"
    write_summary(summary_path)

    print(f"INPUT={input_path}")
    print(f"POINTS={len(gdf)}")
    print(f"FONT={FONT_NAME}")
    print(f"OUTPUT={out_png}")
    print(f"OUTPUT={out_pdf}")
    print(f"OUTPUT={out_svg}")
    print(f"OUTPUT={summary_path}")


if __name__ == "__main__":
    main()
