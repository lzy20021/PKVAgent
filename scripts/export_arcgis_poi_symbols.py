from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(r"E:\GradOpening\MapGenAI")
OUT_DIR = PROJECT_ROOT / "output" / "arcgis_10_2_poi_symbols"

CATEGORIES = [
    ("便利店", "convenience_store", "#D84B8A"),
    ("学校", "school", "#2E7D32"),
    ("停车场", "parking", "#2F80ED"),
    ("酒店", "hotel", "#7B4FA3"),
    ("银行", "bank", "#00897B"),
    ("医院", "hospital", "#D32F2F"),
    ("餐饮", "restaurant", "#F28E2B"),
    ("交通设施", "transit", "#4E79A7"),
]


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        r"C:\Windows\Fonts\msyhbd.ttc" if bold else r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf",
    ]
    for item in candidates:
        path = Path(item)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def rounded(draw: ImageDraw.ImageDraw, xy, radius: int, fill, outline, width: int) -> None:
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def draw_icon(category: str, color: str, size: int) -> Image.Image:
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    pad = int(size * 0.07)
    rounded(draw, (pad, pad, size - pad, size - pad), int(size * 0.16), color, "#1F2933", max(2, int(size * 0.045)))
    w = "white"
    lw = max(3, int(size * 0.075))

    def rect(x0, y0, x1, y1, fill=w, outline=None, width=1):
        draw.rectangle((x0 * size, y0 * size, x1 * size, y1 * size), fill=fill, outline=outline, width=width)

    def line(points):
        draw.line([(x * size, y * size) for x, y in points], fill=w, width=lw, joint="curve")

    def poly(points, fill=w, outline=None):
        draw.polygon([(x * size, y * size) for x, y in points], fill=fill, outline=outline)

    if category == "便利店":
        line([(0.22, 0.42), (0.50, 0.24), (0.78, 0.42)])
        draw.rectangle((0.25 * size, 0.42 * size, 0.75 * size, 0.74 * size), outline=w, width=lw)
        rect(0.36, 0.53, 0.48, 0.73)
        rect(0.55, 0.53, 0.67, 0.73)
    elif category == "学校":
        poly([(0.14, 0.43), (0.50, 0.24), (0.86, 0.43), (0.50, 0.61)])
        rect(0.31, 0.56, 0.69, 0.65)
        rect(0.76, 0.43, 0.81, 0.72)
        draw.ellipse((0.755 * size, 0.70 * size, 0.825 * size, 0.77 * size), fill=w)
    elif category == "停车场":
        text = "P"
        fnt = font(int(size * 0.68), bold=True)
        bbox = draw.textbbox((0, 0), text, font=fnt)
        draw.text(((size - bbox[2] + bbox[0]) / 2, (size - bbox[3] + bbox[1]) / 2 - size * 0.02), text, fill=w, font=fnt)
    elif category == "酒店":
        rect(0.18, 0.52, 0.82, 0.70)
        rect(0.18, 0.35, 0.31, 0.70)
        rect(0.32, 0.40, 0.54, 0.51)
        rect(0.18, 0.71, 0.82, 0.78)
    elif category == "银行":
        poly([(0.15, 0.39), (0.50, 0.18), (0.85, 0.39)])
        for x in [0.27, 0.43, 0.59, 0.75]:
            rect(x - 0.035, 0.43, x + 0.035, 0.74)
        rect(0.17, 0.78, 0.83, 0.85)
    elif category == "医院":
        rect(0.43, 0.20, 0.57, 0.80)
        rect(0.20, 0.43, 0.80, 0.57)
    elif category == "餐饮":
        rect(0.30, 0.23, 0.36, 0.79)
        for x in [0.23, 0.30, 0.37]:
            rect(x, 0.23, x + 0.035, 0.48)
        rect(0.62, 0.25, 0.68, 0.80)
        poly([(0.58, 0.25), (0.73, 0.25), (0.68, 0.50), (0.62, 0.50)])
    elif category == "交通设施":
        rect(0.22, 0.31, 0.78, 0.67)
        rect(0.28, 0.37, 0.46, 0.51, fill=color)
        rect(0.54, 0.37, 0.72, 0.51, fill=color)
        draw.ellipse((0.28 * size, 0.68 * size, 0.39 * size, 0.79 * size), fill=w)
        draw.ellipse((0.61 * size, 0.68 * size, 0.72 * size, 0.79 * size), fill=w)
    return img


def make_contact_sheet(items) -> None:
    cell_w, cell_h = 260, 120
    sheet = Image.new("RGBA", (cell_w * 2, cell_h * 4), "white")
    draw = ImageDraw.Draw(sheet)
    fnt_cn = font(22, bold=True)
    fnt_en = font(16)
    for i, (cn, en, color) in enumerate(items):
        x = (i % 2) * cell_w
        y = (i // 2) * cell_h
        icon = draw_icon(cn, color, 72)
        sheet.alpha_composite(icon, (x + 18, y + 24))
        draw.text((x + 105, y + 30), cn, fill="#1F2933", font=fnt_cn)
        draw.text((x + 105, y + 62), en, fill="#5B6772", font=fnt_en)
    sheet.convert("RGB").save(OUT_DIR / "poi_symbol_contact_sheet.png", "PNG")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for cn, en, color in CATEGORIES:
        for size in [32, 48, 64, 96, 128, 256]:
            img = draw_icon(cn, color, size)
            img.save(OUT_DIR / f"{cn}_{en}_{size}px.png", "PNG")
            bmp = Image.new("RGB", img.size, "white")
            bmp.paste(img, mask=img.getchannel("A"))
            bmp = bmp.convert("P", palette=Image.Palette.ADAPTIVE, colors=256)
            bmp.save(OUT_DIR / f"{cn}_{en}_{size}px.bmp", "BMP")
        draw_icon(cn, color, 128).save(OUT_DIR / f"{en}.png", "PNG")
        bmp128 = Image.new("RGB", (128, 128), "white")
        img128 = draw_icon(cn, color, 128)
        bmp128.paste(img128, mask=img128.getchannel("A"))
        bmp128.convert("P", palette=Image.Palette.ADAPTIVE, colors=256).save(OUT_DIR / f"{en}.bmp", "BMP")
    make_contact_sheet(CATEGORIES)

    lines = [
        "# ArcGIS 10.2 POI Symbol Pack",
        "",
        "这些 PNG/BMP 是为 ArcGIS 10.2 Picture Marker Symbol 准备的符号源文件。",
        "注意：在“打开 Esri 样式 (*.style)”对话框里看不到 PNG/BMP 是正常的；那个对话框只能打开已经生成好的 .style。",
        "",
        "建议导入尺寸：",
        "- 1:84000 到 1:120000：使用 96px 或 128px PNG，ArcMap 符号大小设为 13-16 pt。",
        "- 1:253000：使用 64px 或 96px PNG，ArcMap 符号大小设为 10-13 pt。",
        "- 1:532000：使用 48px 或 64px PNG，ArcMap 符号大小设为 8-10 pt。",
        "",
        "ArcMap 10.2 生成 .style 的步骤：",
        "1. 打开 ArcMap，进入 Customize > Style Manager。",
        "2. 点击 Styles > Create New Style，保存为 MapGenAI_POI_Classic.style。",
        "3. 在左侧展开新建 style，点击 Marker Symbols。",
        "4. 在右侧空白处右键 New > Marker Symbol。",
        "5. Type 选择 Picture Marker Symbol，点击 Picture，选择本目录对应 PNG。",
        "6. Name 分别填写：便利店、学校、停车场、酒店、银行、医院、餐饮、交通设施。",
        "7. Category 建议填写 MapGenAI POI；Tags 可填写 poi, classic, amenity。",
        "8. 保存后，该 .style 即可在 ArcGIS 10.2 的 Style References 中引用。",
        "",
        "如果要自动生成 .style，请把整个目录复制到安装了 ArcGIS Desktop 10.2 的电脑上，运行 build_style_arcobjects_10_2.bat。",
        "",
        "符号对应：",
    ]
    for cn, en, _ in CATEGORIES:
        lines.append(f"- {cn}: {en}.png")
    (OUT_DIR / "ArcGIS10_2_STYLE_IMPORT_README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT_DIR)


if __name__ == "__main__":
    main()
