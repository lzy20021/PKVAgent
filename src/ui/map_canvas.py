"""
地图画布组件
基于Folium和WebEngine实现交互式地图
"""
import os
import json
import tempfile
from typing import Dict, List, Any, Optional, Tuple
from PySide6.QtWidgets import QWidget, QVBoxLayout
from PySide6.QtCore import Signal, QUrl
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWebChannel import QWebChannel


class MapCanvas(QWidget):
    """地图画布"""

    # 信号
    map_clicked = Signal(float, float)  # 点击坐标
    layer_selected = Signal(str)  # 选中图层
    zoom_changed = Signal(int)  # 缩放级别变化

    def __init__(self, parent=None):
        super().__init__(parent)
        self.layers: Dict[str, Dict] = {}
        self.center = [39.9042, 116.4074]
        self.zoom = 10
        self.temp_file = None

        self.setup_ui()
        self.init_map()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.web_view = QWebEngineView()
        layout.addWidget(self.web_view)

    def init_map(self):
        """初始化地图"""
        html = self.generate_map_html()
        self.load_html(html)

    def generate_map_html(self) -> str:
        """生成地图HTML"""
        layers_js = json.dumps(self.layers)

        html = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        html, body, #map {{
            width: 100%;
            height: 100%;
            margin: 0;
            padding: 0;
        }}
        .info-panel {{
            position: absolute;
            bottom: 10px;
            left: 10px;
            background: white;
            padding: 8px 12px;
            border-radius: 4px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.2);
            font-size: 12px;
            z-index: 1000;
        }}
    </style>
</head>
<body>
    <div id="map"></div>
    <div class="info-panel" id="info">
        坐标: -- | 缩放: {self.zoom}
    </div>
    <script>
        var map = L.map('map').setView([{self.center[0]}, {self.center[1]}], {self.zoom});

        // 底图
        L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            attribution: '&copy; OpenStreetMap contributors',
            maxZoom: 19
        }}).addTo(map);

        // 图层存储
        var layers = {{}};
        var layerData = {layers_js};

        // 加载已有图层
        for (var name in layerData) {{
            addGeoJSONLayer(name, layerData[name].geojson, layerData[name].style);
        }}

        // 添加GeoJSON图层
        function addGeoJSONLayer(name, geojson, style) {{
            if (layers[name]) {{
                map.removeLayer(layers[name]);
            }}

            var defaultStyle = {{
                color: '#3388ff',
                weight: 2,
                opacity: 0.8,
                fillColor: '#3388ff',
                fillOpacity: 0.4
            }};

            var layerStyle = Object.assign({{}}, defaultStyle, style || {{}});

            layers[name] = L.geoJSON(geojson, {{
                style: function(feature) {{
                    return layerStyle;
                }},
                pointToLayer: function(feature, latlng) {{
                    return L.circleMarker(latlng, {{
                        radius: 6,
                        fillColor: layerStyle.fillColor,
                        color: layerStyle.color,
                        weight: 1,
                        opacity: 1,
                        fillOpacity: 0.8
                    }});
                }},
                onEachFeature: function(feature, layer) {{
                    if (feature.properties) {{
                        var popup = '<b>' + name + '</b><br>';
                        for (var key in feature.properties) {{
                            popup += key + ': ' + feature.properties[key] + '<br>';
                        }}
                        layer.bindPopup(popup);
                    }}
                }}
            }}).addTo(map);

            return layers[name];
        }}

        // 移除图层
        function removeLayer(name) {{
            if (layers[name]) {{
                map.removeLayer(layers[name]);
                delete layers[name];
            }}
        }}

        // 缩放到图层
        function zoomToLayer(name) {{
            if (layers[name]) {{
                map.fitBounds(layers[name].getBounds());
            }}
        }}

        // 设置图层样式
        function setLayerStyle(name, style) {{
            if (layers[name]) {{
                layers[name].setStyle(style);
            }}
        }}

        // 更新信息面板
        function updateInfo(lat, lng, zoom) {{
            document.getElementById('info').innerHTML =
                '坐标: ' + lat.toFixed(6) + ', ' + lng.toFixed(6) + ' | 缩放: ' + zoom;
        }}

        // 地图事件
        map.on('mousemove', function(e) {{
            updateInfo(e.latlng.lat, e.latlng.lng, map.getZoom());
        }});

        map.on('click', function(e) {{
            console.log('MAP_CLICK:' + e.latlng.lat + ',' + e.latlng.lng);
        }});

        map.on('zoomend', function() {{
            console.log('ZOOM_CHANGE:' + map.getZoom());
        }});

        // 暴露给Python的接口
        window.mapAPI = {{
            addLayer: addGeoJSONLayer,
            removeLayer: removeLayer,
            zoomToLayer: zoomToLayer,
            setStyle: setLayerStyle,
            setView: function(lat, lng, zoom) {{
                map.setView([lat, lng], zoom);
            }},
            getCenter: function() {{
                var c = map.getCenter();
                return [c.lat, c.lng];
            }},
            getZoom: function() {{
                return map.getZoom();
            }}
        }};
    </script>
</body>
</html>'''
        return html

    def load_html(self, html: str):
        """加载HTML到WebView"""
        # 使用临时文件
        if self.temp_file:
            try:
                os.unlink(self.temp_file)
            except:
                pass

        fd, self.temp_file = tempfile.mkstemp(suffix='.html')
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(html)

        self.web_view.setUrl(QUrl.fromLocalFile(self.temp_file))

    def add_layer(self, name: str, geojson: Dict, style: Dict = None):
        """添加图层"""
        self.layers[name] = {
            'geojson': geojson,
            'style': style or {}
        }
        # 执行JavaScript添加图层
        js = f"window.mapAPI.addLayer('{name}', {json.dumps(geojson)}, {json.dumps(style or {})});"
        self.web_view.page().runJavaScript(js)

    def remove_layer(self, name: str):
        """移除图层"""
        if name in self.layers:
            del self.layers[name]
        js = f"window.mapAPI.removeLayer('{name}');"
        self.web_view.page().runJavaScript(js)

    def zoom_to_layer(self, name: str):
        """缩放到图层"""
        js = f"window.mapAPI.zoomToLayer('{name}');"
        self.web_view.page().runJavaScript(js)

    def set_layer_style(self, name: str, style: Dict):
        """设置图层样式"""
        if name in self.layers:
            self.layers[name]['style'] = style
        js = f"window.mapAPI.setStyle('{name}', {json.dumps(style)});"
        self.web_view.page().runJavaScript(js)

    def set_view(self, lat: float, lng: float, zoom: int = None):
        """设置地图视图"""
        self.center = [lat, lng]
        if zoom:
            self.zoom = zoom
        js = f"window.mapAPI.setView({lat}, {lng}, {zoom or self.zoom});"
        self.web_view.page().runJavaScript(js)

    def refresh(self):
        """刷新地图"""
        html = self.generate_map_html()
        self.load_html(html)

    def get_layer_names(self) -> List[str]:
        """获取所有图层名称"""
        return list(self.layers.keys())

    def clear_layers(self):
        """清除所有图层"""
        for name in list(self.layers.keys()):
            self.remove_layer(name)

    def __del__(self):
        """清理临时文件"""
        if self.temp_file and os.path.exists(self.temp_file):
            try:
                os.unlink(self.temp_file)
            except:
                pass
