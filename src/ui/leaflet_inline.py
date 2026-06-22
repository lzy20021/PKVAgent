"""
生成内联Leaflet的HTML模板
将Leaflet CSS和JS内联到HTML中，避免外部文件加载问题
"""
import os

def generate_inline_leaflet_html():
    """读取Leaflet文件并生成内联HTML模板"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    css_path = os.path.join(base_dir, "resources", "leaflet.css")
    js_path = os.path.join(base_dir, "resources", "leaflet.js")

    with open(css_path, 'r', encoding='utf-8') as f:
        leaflet_css = f.read()

    with open(js_path, 'r', encoding='utf-8') as f:
        leaflet_js = f.read()

    return leaflet_css, leaflet_js


if __name__ == "__main__":
    css, js = generate_inline_leaflet_html()
    print(f"CSS length: {len(css)}")
    print(f"JS length: {len(js)}")
