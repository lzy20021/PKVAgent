"""
矢量数据IO模块
支持GeoJSON、Shapefile、GeoPackage格式
"""
import os
import json
from typing import Dict, List, Any, Optional, Tuple


def read_geojson(file_path: str) -> Dict:
    """读取GeoJSON文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def write_geojson(geojson: Dict, file_path: str):
    """写入GeoJSON文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(geojson, f, ensure_ascii=False, indent=2)


def read_shapefile(file_path: str) -> Dict:
    """读取Shapefile文件"""
    try:
        import geopandas as gpd
        gdf = gpd.read_file(file_path)
        return json.loads(gdf.to_json())
    except ImportError:
        # 简化实现，不依赖geopandas
        return read_shapefile_simple(file_path)


def read_shapefile_simple(file_path: str) -> Dict:
    """简化的Shapefile读取（不依赖geopandas）"""
    try:
        import fiona
        features = []
        with fiona.open(file_path, 'r') as src:
            for feature in src:
                features.append({
                    'type': 'Feature',
                    'geometry': dict(feature['geometry']),
                    'properties': dict(feature['properties'])
                })

        return {
            'type': 'FeatureCollection',
            'features': features
        }
    except ImportError:
        raise ImportError("需要安装fiona库: pip install fiona")


def write_shapefile(geojson: Dict, file_path: str):
    """写入Shapefile文件"""
    try:
        import geopandas as gpd
        gdf = gpd.GeoDataFrame.from_features(geojson['features'])
        gdf.to_file(file_path, driver='ESRI Shapefile')
    except ImportError:
        raise ImportError("需要安装geopandas库: pip install geopandas")


def read_geopackage(file_path: str, layer: str = None) -> Dict:
    """读取GeoPackage文件"""
    try:
        import geopandas as gpd
        gdf = gpd.read_file(file_path, layer=layer)
        return json.loads(gdf.to_json())
    except ImportError:
        raise ImportError("需要安装geopandas库: pip install geopandas")


def write_geopackage(geojson: Dict, file_path: str, layer: str = 'layer'):
    """写入GeoPackage文件"""
    try:
        import geopandas as gpd
        gdf = gpd.GeoDataFrame.from_features(geojson['features'])
        gdf.to_file(file_path, driver='GPKG', layer=layer)
    except ImportError:
        raise ImportError("需要安装geopandas库: pip install geopandas")


def read_vector_file(file_path: str) -> Dict:
    """
    自动识别格式并读取矢量文件

    Args:
        file_path: 文件路径

    Returns:
        GeoJSON格式的数据
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext == '.geojson' or ext == '.json':
        return read_geojson(file_path)
    elif ext == '.shp':
        return read_shapefile(file_path)
    elif ext == '.gpkg':
        return read_geopackage(file_path)
    else:
        raise ValueError(f"不支持的文件格式: {ext}")


def write_vector_file(geojson: Dict, file_path: str):
    """
    自动识别格式并写入矢量文件

    Args:
        geojson: GeoJSON数据
        file_path: 文件路径
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext == '.geojson' or ext == '.json':
        write_geojson(geojson, file_path)
    elif ext == '.shp':
        write_shapefile(geojson, file_path)
    elif ext == '.gpkg':
        write_geopackage(geojson, file_path)
    else:
        raise ValueError(f"不支持的文件格式: {ext}")


def get_bounds(geojson: Dict) -> Tuple[float, float, float, float]:
    """
    获取GeoJSON的边界范围

    Returns:
        (min_x, min_y, max_x, max_y)
    """
    coords = []

    def extract_coords(geometry):
        if geometry['type'] == 'Point':
            coords.append(geometry['coordinates'])
        elif geometry['type'] in ['LineString', 'MultiPoint']:
            coords.extend(geometry['coordinates'])
        elif geometry['type'] in ['Polygon', 'MultiLineString']:
            for ring in geometry['coordinates']:
                coords.extend(ring)
        elif geometry['type'] == 'MultiPolygon':
            for polygon in geometry['coordinates']:
                for ring in polygon:
                    coords.extend(ring)

    for feature in geojson.get('features', []):
        if feature.get('geometry'):
            extract_coords(feature['geometry'])

    if not coords:
        return (0, 0, 0, 0)

    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]

    return (min(xs), min(ys), max(xs), max(ys))


def get_center(geojson: Dict) -> Tuple[float, float]:
    """获取GeoJSON的中心点"""
    bounds = get_bounds(geojson)
    return ((bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2)


def get_feature_count(geojson: Dict) -> int:
    """获取要素数量"""
    return len(geojson.get('features', []))


def get_geometry_type(geojson: Dict) -> str:
    """获取几何类型"""
    features = geojson.get('features', [])
    if not features:
        return 'Unknown'

    types = set()
    for f in features:
        if f.get('geometry'):
            types.add(f['geometry']['type'])

    if len(types) == 1:
        return types.pop()
    return 'Mixed'
