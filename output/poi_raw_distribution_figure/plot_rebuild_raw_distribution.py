from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "output" / "poi_raw_distribution_figure"
INPUT_CANDIDATES = [
    PROJECT_ROOT / "Datasets" / "datasets0324.shp",
    PROJECT_ROOT / "amap_poi_dataset_expanded" / "Datasets0324.shp",
    PROJECT_ROOT / "amap_poi_dataset_expanded" / "all_poi_expanded.geojson",
]

TITLE_MAIN = "Raw POI Distribution and Representative Local Spatial Patterns"
TITLE_ALT = "Spatial Distribution of Original POI Data Across the Study Area"

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
    "\u4fbf\u5229\u5e97": "#C8CF66",
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

    if "type" not in gdf.columns:
        raise KeyError("Input dataset must contain a 'type' field.")

    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    gdf["type"] = gdf["type"].fillna("\u5176\u4ed6")
    gdf = gdf[gdf["type"].isin(CATEGORY_ORDER)].copy()
    gdf["type_en"] = gdf["type"].map(CATEGORY_LABELS)
    return gdf


def style_axes(ax, show_labels: bool = True, show_grid: bool = True) -> None:
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


def add_north_arrow(ax) -> None:
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
        arrowprops={
            "arrowstyle": "-|>",
            "lw": 0.65,
            "color": "#7A838B",
            "mutation_scale": 8,
            "shrinkA": 0,
            "shrinkB": 0,
        },
        zorder=8,
    )


def scatter_by_category(ax, gdf: gpd.GeoDataFrame, point_size: float, alpha: float, edge_width: float) -> None:
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


def add_pattern_boxes(ax_main) -> None:
    for window in PATTERN_WINDOWS:
        cx, cy = window["center"]
        half = window["half_size"]
        rect = mpatches.Rectangle(
            (cx - half, cy - half),
            half * 2,
            half * 2,
            linewidth=0.65,
            edgecolor="#7A8289",
            facecolor="none",
            zorder=6,
        )
        ax_main.add_patch(rect)
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
            bbox={
                "boxstyle": "round,pad=0.18,rounding_size=0.45",
                "facecolor": "white",
                "edgecolor": "#A0A7AE",
                "linewidth": 0.55,
            },
            zorder=7,
        )


def plot_local_window(ax, gdf: gpd.GeoDataFrame, window: dict) -> None:
    cx, cy = window["center"]
    half = window["half_size"]
    local = gdf[
        gdf.geometry.x.between(cx - half, cx + half) & gdf.geometry.y.between(cy - half, cy + half)
    ]
    scatter_by_category(ax, local, point_size=37, alpha=0.82, edge_width=0.34)
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    style_axes(ax, show_labels=False, show_grid=False)
    ax.set_title(
        f"{window['label']}  {window['title']}",
        fontsize=10.5,
        fontweight="normal",
        pad=7,
        loc="left",
    )


def build_legend(fig) -> None:
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            label=CATEGORY_LABELS[category],
            markerfacecolor=CATEGORY_COLORS[category],
            markeredgecolor="white",
            markeredgewidth=0.45,
            markersize=8.8,
            alpha=0.86,
        )
        for category in CATEGORY_ORDER
    ]
    legend = fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.035),
        ncol=4,
        frameon=False,
        title="POI Categories",
        fontsize=8.7,
        title_fontsize=8.9,
        columnspacing=1.05,
        handletextpad=0.38,
        borderaxespad=0.0,
    )
    legend.get_title().set_color("#7A8289")


def write_summary(input_path: Path, gdf: gpd.GeoDataFrame) -> Path:
    summary_path = OUTPUT_DIR / "raw_distribution_summary.txt"
    lines = [
        "Raw POI distribution figure rebuild summary",
        "",
        "1. This figure is a raw data presentation figure. No point generalization, sampling, deletion, aggregation, or merging was applied to the POI dataset.",
        "2. Only visualization-level optimization was performed.",
        "3. The optimization includes background whitening, lighter gridlines, slightly larger point symbols, controlled transparency, fine white point outlines, cleaner legend design, lighter title hierarchy, adjusted subplot layout, and thinner subplot/map borders.",
        "4. The purpose of these changes is to preserve the full sense of original data volume, spatial coverage, and category richness while improving publication quality and readability for a paper figure.",
        "",
        f"Primary title: {TITLE_MAIN}",
        f"Alternative title: {TITLE_ALT}",
        "",
        f"Input dataset: {input_path}",
        f"Total retained POI points: {len(gdf)}",
        f"CRS used for plotting: {gdf.crs}",
        "",
        "Output files:",
        f"- {OUTPUT_DIR / 'final_poi_raw_distribution.png'}",
        f"- {OUTPUT_DIR / 'final_poi_raw_distribution.pdf'}",
        f"- {OUTPUT_DIR / 'final_poi_raw_distribution.svg'}",
        f"- {OUTPUT_DIR / 'plot_rebuild_raw_distribution.py'}",
        f"- {summary_path}",
        "",
        "This revision further increases background brightness and point size to improve readability in print and on-screen viewing while preserving all original POI data.",
        "",
        "The color palette was further brightened by replacing the dominant dark green category with a lighter sky-blue tone, improving the overall visual brightness and reducing the heavy appearance of the figure.",
        "",
        "The Schools category was reassigned to a brighter light-blue tone, the Other category was removed from visualization, and the point symbols were further enlarged to improve clarity.",
    ]
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_path


def main() -> None:
    setup_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    input_path = resolve_input_path()
    gdf = load_data(input_path)

    fig = plt.figure(figsize=(17.2, 10.4), constrained_layout=False)
    gs = GridSpec(
        2,
        3,
        figure=fig,
        width_ratios=[1.82, 1.0, 1.0],
        height_ratios=[1.0, 1.0],
        wspace=0.18,
        hspace=0.18,
    )

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
    ax_main.set_xlabel("Longitude (\N{DEGREE SIGN}E)", fontsize=10)
    ax_main.set_ylabel("Latitude (\N{DEGREE SIGN}N)", fontsize=10)
    fig.suptitle(TITLE_MAIN, fontsize=13.6, fontweight="normal", y=0.963, color="#3A4045")

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

    save_kwargs = dict(bbox_inches="tight", facecolor="white", edgecolor="none")
    output_names = [
        "final_poi_raw_distribution.png",
        "final_poi_raw_distribution.pdf",
        "final_poi_raw_distribution.svg",
        "final_poi_raw_distribution_bright.png",
        "final_poi_raw_distribution_bright.pdf",
        "final_poi_raw_distribution_bright.svg",
        "final_poi_raw_distribution_brighter_palette.png",
        "final_poi_raw_distribution_brighter_palette.pdf",
        "final_poi_raw_distribution_brighter_palette.svg",
        "final_poi_raw_distribution_no_other.png",
        "final_poi_raw_distribution_no_other.pdf",
        "final_poi_raw_distribution_no_other.svg",
        "final_poi_raw_distribution_final.png",
        "final_poi_raw_distribution_final.pdf",
        "final_poi_raw_distribution_final.svg",
    ]
    for name in output_names:
        out = OUTPUT_DIR / name
        if out.suffix.lower() == ".png":
            fig.savefig(out, dpi=600, **save_kwargs)
        else:
            fig.savefig(out, **save_kwargs)

    ax_main.xaxis.grid(False)
    ax_main.yaxis.grid(False)
    fig.savefig(OUTPUT_DIR / "final_poi_raw_distribution_final_nogrid.png", dpi=600, **save_kwargs)
    fig.savefig(OUTPUT_DIR / "final_poi_raw_distribution_final_nogrid.pdf", **save_kwargs)
    plt.close(fig)

    write_summary(input_path, gdf)

    print(f"INPUT={input_path}")
    print(f"POINTS={len(gdf)}")
    for name in output_names:
        print(f"OUTPUT={OUTPUT_DIR / name}")
    print(f"OUTPUT={OUTPUT_DIR / 'final_poi_raw_distribution_final_nogrid.png'}")
    print(f"OUTPUT={OUTPUT_DIR / 'final_poi_raw_distribution_final_nogrid.pdf'}")


if __name__ == "__main__":
    main()
