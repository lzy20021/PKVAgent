"""
GIS渲染引擎Skills演示
展示如何使用StyleRenderer和PerformanceOptimizer
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.skills import StyleRenderer, PerformanceOptimizer
from shapely.geometry import Point, LineString, Polygon
import geopandas as gpd

def demo_style_renderer():
    print("=== 样式渲染Skill演示 ===\n")

    # 1. 创建渐变画刷
    gradient_brush = StyleRenderer.create_gradient_brush('#FF5722', '#FFC107', 'linear')
    print("✓ 创建线性渐变画刷")

    # 2. 创建自定义画笔
    pen = StyleRenderer.create_pen('#2196F3', 3.0, style='dash', cap='round')
    print("✓ 创建虚线画笔（圆角端点）")

    # 3. 创建带透明度的填充
    brush = StyleRenderer.create_brush('#4CAF50', opacity=0.6, pattern='cross')
    print("✓ 创建交叉图案填充（60%透明度）")

    # 4. 颜色插值
    color = StyleRenderer.interpolate_color('#FF0000', '#0000FF', 0.5)
    print(f"✓ 颜色插值（红→蓝 50%）: {color}")

    # 5. 分类样式
    categories = {
        'residential': {'fill': '#FFEB3B', 'stroke': '#F57C00'},
        'commercial': {'fill': '#E91E63', 'stroke': '#880E4F'},
        'default': {'fill': '#9E9E9E', 'stroke': '#424242'}
    }
    style = StyleRenderer.get_categorized_style('residential', categories)
    print(f"✓ 分类样式（residential）: {style}")

    # 6. 分级样式
    breaks = [10, 50, 100, 500]
    colors = ['#FFF9C4', '#FFF176', '#FFEB3B', '#FBC02D', '#F57F17']
    color = StyleRenderer.get_graduated_style(75, breaks, colors)
    print(f"✓ 分级样式（值=75）: {color}")

    # 7. 数据驱动样式
    size = StyleRenderer.apply_data_driven_style(75, 0, 100, 5, 20)
    print(f"✓ 数据驱动大小（值=75，范围0-100）: {size:.1f}px\n")

def demo_performance_optimizer():
    print("=== 性能优化Skill演示 ===\n")

    optimizer = PerformanceOptimizer()

    # 1. 创建测试数据
    geometries = [
        Point(116.4 + i*0.01, 39.9 + i*0.01)
        for i in range(1000)
    ]
    print(f"✓ 创建1000个点要素")

    # 2. 构建空间索引
    optimizer.build_spatial_index(geometries)
    print("✓ 构建R-tree空间索引")

    # 3. 视口查询
    bbox = (116.4, 39.9, 116.5, 40.0)
    visible = optimizer.query_visible_features(bbox)
    print(f"✓ 视口查询：{len(visible)}/{len(geometries)} 个可见要素")

    # 4. 几何简化
    line = LineString([(116.4+i*0.001, 39.9+i*0.001) for i in range(100)])
    simplified = optimizer.simplify_geometry(line, tolerance=0.005)
    print(f"✓ 几何简化：{len(line.coords)} → {len(simplified.coords)} 个顶点")

    # 5. 自适应简化
    adaptive = optimizer.adaptive_simplify(line, zoom_level=10)
    print(f"✓ 自适应简化（zoom=10）：{len(adaptive.coords)} 个顶点")

    # 6. 几何裁剪
    poly = Polygon([(116.35, 39.85), (116.55, 39.85), (116.55, 40.05), (116.35, 40.05)])
    clipped = optimizer.clip_geometry(poly, bbox)
    print(f"✓ 几何裁剪：面积 {poly.area:.6f} → {clipped.area:.6f}")

    # 7. LOD采样
    features = list(range(50000))
    sampled = optimizer.lod_sample_features(features, zoom_level=5)
    print(f"✓ LOD采样（zoom=5）：{len(features)} → {len(sampled)} 个要素")

    # 8. 复杂度估算
    complexity = optimizer.estimate_geometry_complexity(line)
    print(f"✓ 几何复杂度：{complexity} 个顶点")

    # 9. 渲染判断
    should_render = optimizer.should_render_feature(Point(116.4, 39.9), zoom_level=15)
    print(f"✓ 是否渲染（zoom=15）：{should_render}")

    # 10. 缓存操作
    optimizer.cache_set('test_key', {'data': 'cached_value'})
    cached = optimizer.cache_get('test_key')
    print(f"✓ 缓存读写：{cached}\n")

def demo_integration():
    print("=== 集成示例：样式化+优化渲染 ===\n")

    # 创建测试GeoDataFrame
    points = [Point(116.4 + i*0.01, 39.9 + i*0.01) for i in range(100)]
    gdf = gpd.GeoDataFrame({'population': range(100, 200)}, geometry=points)

    optimizer = PerformanceOptimizer()
    optimizer.build_spatial_index(gdf.geometry.tolist())

    # 视口查询
    bbox = (116.4, 39.9, 116.5, 40.0)
    visible = optimizer.query_visible_features(bbox)
    print(f"✓ 可见要素：{len(visible)}/{len(gdf)}")

    # 数据驱动样式
    min_pop = gdf['population'].min()
    max_pop = gdf['population'].max()

    for idx, row in gdf.head(5).iterrows():
        size = StyleRenderer.apply_data_driven_style(
            row['population'], min_pop, max_pop, 5, 20
        )
        color = StyleRenderer.get_graduated_style(
            row['population'],
            [120, 140, 160, 180],
            ['#FFF9C4', '#FFEB3B', '#FBC02D', '#F57F17', '#E65100']
        )
        print(f"  要素{idx}: 人口={row['population']}, 大小={size:.1f}px, 颜色={color}")

    print("\n✓ 完成集成演示")

if __name__ == '__main__':
    demo_style_renderer()
    demo_performance_optimizer()
    demo_integration()

    print("\n" + "="*50)
    print("Skills演示完成！")
    print("="*50)
