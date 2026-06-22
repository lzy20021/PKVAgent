"""
MapGenAI 主窗口
地图综合智能体 - 主界面
"""
import sys
import os
import time
import re
import math
import yaml
import json
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QMenuBar, QMenu, QToolBar, QStatusBar, QTreeWidget,
    QTreeWidgetItem, QTextEdit, QLineEdit, QPushButton, QLabel,
    QFrame, QScrollArea, QDockWidget, QFileDialog, QMessageBox,
    QListWidget, QListWidgetItem, QTabWidget, QProgressBar, QInputDialog, QAbstractItemView,
    QDialog, QTableWidget, QTableWidgetItem, QColorDialog, QGridLayout, QFormLayout, QComboBox,
    QSizePolicy,
    QDoubleSpinBox, QSpinBox
)
from PySide6.QtCore import Qt, QSize, Signal, QThread, QTimer, QObject, QMetaObject, Slot, QPointF, QRectF
from PySide6.QtGui import QAction, QActionGroup, QIcon, QFont, QColor, QPalette, QPixmap, QTextCursor, QTextOption, QPainter, QPen, QBrush, QPainterPath, QPolygonF

# 导入Agent相关模块
from src.agent.tools import register_ui_callback
from src.ui.env_config_dialog import EnvConfigDialog
from src.ui.env_status_widget import EnvStatusWidget
from src.ui.i18n import LANGUAGE_LABELS, SUPPORTED_LANGUAGES, current_language, language_label, set_language, tr
from src.ui.point_symbol_core import POI_SYMBOL_RULES, assign_symbols_to_categories, infer_point_symbol_style_from_gdf
from src.ui.point_symbol_render import normalize_point_symbol


class ModelCheckThread(QThread):
    """模型检测线程"""
    finished = Signal(list)
    error = Signal(str)
    
    def __init__(self, provider, api_key):
        super().__init__()
        self.provider = provider
        self.api_key = api_key
        
    def run(self):
        try:
            from src.utils.model_detector import check_models
            models = check_models(self.provider, self.api_key)
            self.finished.emit(models)
        except Exception as e:
            self.error.emit(str(e))


class WorkspacePanel(QWidget):
    """工作区面板 - 显示文件夹内容"""
    file_double_clicked = Signal(str)
    panel_closed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.root_path = ""
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # 标题栏
        header = QFrame()
        header.setStyleSheet("background: #e0e0e0;")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(8, 6, 8, 6)

        self.title_label = QLabel()
        self.title_label.setStyleSheet("font-weight: bold;")
        header_layout.addWidget(self.title_label)
        header_layout.addStretch()

        self.refresh_btn = QPushButton()
        self.refresh_btn.setFixedSize(64, 24)
        self.refresh_btn.clicked.connect(self.refresh)
        header_layout.addWidget(self.refresh_btn)

        self.close_btn = QPushButton("✕")
        self.close_btn.setFixedSize(24, 24)
        self.close_btn.setStyleSheet("QPushButton{background:#fff;border:1px solid #ccc;border-radius:3px;}QPushButton:hover{background:#e0e0e0;}")
        self.close_btn.clicked.connect(self.panel_closed.emit)
        header_layout.addWidget(self.close_btn)

        layout.addWidget(header)

        # 文件树
        self.file_tree = QTreeWidget()
        self.file_tree.setHeaderHidden(True)
        self.file_tree.setStyleSheet("""
            QTreeWidget { border: none; background: #fafafa; }
            QTreeWidget::item { padding: 3px; }
            QTreeWidget::item:hover { background: #e3f2fd; }
            QTreeWidget::item:selected { background: #1976d2; color: white; }
        """)
        self.file_tree.itemDoubleClicked.connect(self.on_item_double_clicked)
        layout.addWidget(self.file_tree)
        self.retranslate_ui()

    def set_root(self, path):
        self.root_path = path
        self._update_title()
        self.refresh()

    def _update_title(self):
        if self.root_path:
            self.title_label.setText(tr("workspace.title_with_name", name=os.path.basename(self.root_path)))
        else:
            self.title_label.setText(tr("workspace.title"))

    def retranslate_ui(self):
        self._update_title()
        self.refresh_btn.setText(tr("workspace.refresh"))
        self.close_btn.setToolTip(tr("workspace.close_tooltip"))

    def refresh(self):
        self.file_tree.clear()
        if not self.root_path or not os.path.isdir(self.root_path):
            return
        root_item = QTreeWidgetItem(self.file_tree, [os.path.basename(self.root_path)])
        root_item.setData(0, Qt.ItemDataRole.UserRole, self.root_path)
        self._populate(root_item, self.root_path, 0)
        root_item.setExpanded(True)

    def _populate(self, parent, path, depth):
        if depth > 4:
            return
        try:
            items = sorted(os.listdir(path), key=lambda x: (not os.path.isdir(os.path.join(path, x)), x.lower()))
            for name in items:
                if name.startswith("."):
                    continue
                full_path = os.path.join(path, name)
                item = QTreeWidgetItem(parent, [name])
                item.setData(0, Qt.ItemDataRole.UserRole, full_path)
                if os.path.isdir(full_path):
                    self._populate(item, full_path, depth + 1)
        except PermissionError:
            pass

    def on_item_double_clicked(self, item, column):
        path = item.data(0, Qt.ItemDataRole.UserRole)
        if path and os.path.isfile(path):
            self.file_double_clicked.emit(path)


# ... MapCanvas, LayerPanel, AlgorithmPanel classes remain unchanged ...
# 为了节省篇幅，这里假设 MapCanvas, LayerPanel, AlgorithmPanel 代码保持不变
# 实际写入时必须包含完整代码，因此我将它们复制过来

class MapCanvas(QFrame):
    """地图渲染画布 - 基于Leaflet的交互式地图"""
    layer_clicked = Signal(str)
    map_ready = Signal()  # 地图加载完成信号

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Sunken)
        self.setMinimumSize(400, 300)

        self.layers = {}  # {name: {"geojson": data, "style": style, "visible": True}}
        self.layer_order = []  # 图层渲染顺序
        self.center = [39.9042, 116.4074]
        self.zoom = 10
        self.temp_file = None
        self._map_loaded = False
        self._pending_layers = []  # 待添加的图层队列
        self.web_view = None
        self._use_external_browser = False  # 是否使用外部浏览器

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        try:
            from PySide6.QtWebEngineWidgets import QWebEngineView
            self.web_view = QWebEngineView()
            self.web_view.loadFinished.connect(self._on_load_finished)
            layout.addWidget(self.web_view)
            self.init_map()
        except ImportError:
            self._setup_fallback_ui(layout, tr("map.fallback_install"))

    def _setup_fallback_ui(self, layout, message):
        """设置回退UI"""
        self.web_view = None
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        placeholder = QLabel(f"{tr('map.render_area')}\n\n{message}")
        placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        placeholder.setStyleSheet("color: #888; font-size: 14px; background: #f5f5f5;")
        container_layout.addWidget(placeholder)

        # 添加"在浏览器中打开"按钮
        self.open_browser_btn = QPushButton(tr("map.open_in_browser"))
        self.open_browser_btn.setStyleSheet("""
            QPushButton {
                background: #1976d2; color: white; border: none;
                padding: 10px 20px; border-radius: 5px; font-size: 14px;
            }
            QPushButton:hover { background: #1565c0; }
        """)
        self.open_browser_btn.clicked.connect(self._open_in_browser)
        self.open_browser_btn.setVisible(False)
        container_layout.addWidget(self.open_browser_btn)

        layout.addWidget(container)

    def _on_load_finished(self, ok):
        """页面加载完成回调"""
        if ok:
            self._map_loaded = True
            # 处理待添加的图层
            for name, geojson, style in self._pending_layers:
                self._add_layer_js(name, geojson, style)
            self._pending_layers.clear()
            self.map_ready.emit()
        else:
            # WebEngine加载失败，切换到外部浏览器模式
            print("WebEngine加载失败，切换到外部浏览器模式")
            self._use_external_browser = True
            self._setup_external_browser_mode()

    def _setup_external_browser_mode(self):
        """设置外部浏览器模式"""
        if hasattr(self, 'open_browser_btn'):
            self.open_browser_btn.setVisible(True)
        # 生成HTML文件
        self._generate_and_save_html()

    def _generate_and_save_html(self):
        """生成并保存HTML文件"""
        html = self._generate_map_html()
        if not self.temp_file:
            import tempfile
            fd, self.temp_file = tempfile.mkstemp(suffix='.html', prefix='mapgenai_')
            os.close(fd)
        with open(self.temp_file, 'w', encoding='utf-8') as f:
            f.write(html)

    def _open_in_browser(self):
        """在系统浏览器中打开地图"""
        self._generate_and_save_html()
        import webbrowser
        webbrowser.open(f'file:///{self.temp_file.replace(os.sep, "/")}')

    def init_map(self):
        """初始化地图"""
        if not self.web_view:
            return
        self._map_loaded = False
        html = self._generate_map_html()
        self._load_html(html)

    def _get_leaflet_inline(self):
        """获取内联的Leaflet CSS和JS"""
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        css_path = os.path.join(base_dir, "resources", "leaflet.css")
        js_path = os.path.join(base_dir, "resources", "leaflet.js")

        try:
            with open(css_path, 'r', encoding='utf-8') as f:
                leaflet_css = f.read()
            with open(js_path, 'r', encoding='utf-8') as f:
                leaflet_js = f.read()
            return leaflet_css, leaflet_js
        except:
            return None, None

    def _generate_map_html(self) -> str:
        """生成地图HTML"""
        import json

        # 尝试获取内联的Leaflet
        leaflet_css, leaflet_js = self._get_leaflet_inline()

        if leaflet_css and leaflet_js:
            # 使用内联方式
            css_tag = f'<style>{leaflet_css}</style>'
            js_tag = f'<script>{leaflet_js}</script>'
        else:
            # 回退到CDN
            css_tag = '<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>'
            js_tag = '<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>'

        # 按顺序生成图层JS
        layers_data = {}
        for name in self.layer_order:
            if name in self.layers and self.layers[name].get("visible", True):
                layers_data[name] = {
                    "geojson": self.layers[name]["geojson"],
                    "style": self.layers[name].get("style", {})
                }

        layers_js = json.dumps(layers_data, ensure_ascii=False)

        html = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    {css_tag}
    {js_tag}
    <style>
        html, body, #map {{ width: 100%; height: 100%; margin: 0; padding: 0; }}
        .info-panel {{
            position: absolute; bottom: 10px; left: 10px;
            background: white; padding: 8px 12px; border-radius: 4px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.2); font-size: 12px; z-index: 1000;
        }}
    </style>
</head>
<body>
    <div id="map"></div>
    <div class="info-panel" id="info">坐标: -- | 缩放: {self.zoom}</div>
    <script>
        console.log('Initializing map...');
        var map = L.map('map').setView([{self.center[0]}, {self.center[1]}], {self.zoom});
        console.log('Map created');

        L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            attribution: '&copy; OpenStreetMap', maxZoom: 19
        }}).addTo(map);
        console.log('Tile layer added');

        var layers = {{}};
        var layerData = {layers_js};

        function getDefaultStyle() {{
            return {{ color: '#3388ff', weight: 2, opacity: 0.8, fillColor: '#3388ff', fillOpacity: 0.4 }};
        }}

        function addGeoJSONLayer(name, geojson, style) {{
            console.log('Adding layer:', name, 'features:', geojson.features ? geojson.features.length : 0);
            try {{
                if (layers[name]) map.removeLayer(layers[name]);
                var layerStyle = Object.assign(getDefaultStyle(), style || {{}});
                layers[name] = L.geoJSON(geojson, {{
                    style: function(feature) {{ return layerStyle; }},
                    pointToLayer: function(feature, latlng) {{
                        return L.circleMarker(latlng, {{
                            radius: 6, fillColor: layerStyle.fillColor, color: layerStyle.color,
                            weight: 1, opacity: 1, fillOpacity: 0.8
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
                console.log('Layer added successfully:', name);
                return layers[name];
            }} catch(e) {{
                console.error('Error adding layer:', name, e);
                return null;
            }}
        }}

        function removeLayer(name) {{
            if (layers[name]) {{ map.removeLayer(layers[name]); delete layers[name]; }}
        }}

        function setLayerVisibility(name, visible) {{
            if (layers[name]) {{
                if (visible) map.addLayer(layers[name]);
                else map.removeLayer(layers[name]);
            }}
        }}

        function zoomToLayer(name) {{
            if (layers[name]) {{
                try {{
                    var bounds = layers[name].getBounds();
                    if (bounds.isValid()) {{
                        map.fitBounds(bounds, {{padding: [20, 20]}});
                    }}
                }} catch(e) {{
                    console.error('Error zooming to layer:', e);
                }}
            }}
        }}

        function setLayerStyle(name, style) {{
            if (layers[name]) layers[name].setStyle(style);
        }}

        // 加载已有图层
        console.log('Loading existing layers:', Object.keys(layerData));
        for (var name in layerData) {{
            var layer = addGeoJSONLayer(name, layerData[name].geojson, layerData[name].style);
            if (layer) {{
                try {{
                    var bounds = layer.getBounds();
                    if (bounds.isValid()) {{
                        map.fitBounds(bounds, {{padding: [20, 20]}});
                    }}
                }} catch(e) {{}}
            }}
        }}

        map.on('mousemove', function(e) {{
            document.getElementById('info').innerHTML =
                '坐标: ' + e.latlng.lat.toFixed(6) + ', ' + e.latlng.lng.toFixed(6) + ' | 缩放: ' + map.getZoom();
        }});

        window.mapAPI = {{
            addLayer: addGeoJSONLayer,
            removeLayer: removeLayer,
            zoomToLayer: zoomToLayer,
            setStyle: setLayerStyle,
            setVisibility: setLayerVisibility,
            setView: function(lat, lng, zoom) {{ map.setView([lat, lng], zoom); }},
            fitBounds: function(bounds) {{ map.fitBounds(bounds); }},
            isReady: function() {{ return true; }}
        }};

        console.log('Map initialization complete, mapAPI ready');
    </script>
</body>
</html>'''
        return html

    def _load_html(self, html: str):
        """加载HTML到WebView"""
        import tempfile
        from PySide6.QtCore import QUrl

        if self.temp_file:
            try: os.unlink(self.temp_file)
            except: pass

        fd, self.temp_file = tempfile.mkstemp(suffix='.html')
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(html)

        self.web_view.setUrl(QUrl.fromLocalFile(self.temp_file))

    def _add_layer_js(self, name: str, geojson: dict, style: dict):
        """通过JavaScript添加图层"""
        import json
        geojson_str = json.dumps(geojson, ensure_ascii=False)
        style_str = json.dumps(style or {})
        js = f'''
        (function() {{
            if (window.mapAPI && window.mapAPI.addLayer) {{
                var layer = window.mapAPI.addLayer('{name}', {geojson_str}, {style_str});
                if (layer) {{
                    try {{
                        var bounds = layer.getBounds();
                        if (bounds.isValid()) {{
                            map.fitBounds(bounds, {{padding: [20, 20]}});
                        }}
                    }} catch(e) {{}}
                }}
            }} else {{
                console.error('mapAPI not ready');
            }}
        }})();
        '''
        self.web_view.page().runJavaScript(js)

    def add_layer(self, name: str, geojson: dict, style: dict = None):
        """添加图层"""
        self.layers[name] = {"geojson": geojson, "style": style or {}, "visible": True}
        if name not in self.layer_order:
            self.layer_order.append(name)

        if self._use_external_browser:
            # 外部浏览器模式：更新HTML文件并显示按钮
            self._generate_and_save_html()
            if hasattr(self, 'open_browser_btn'):
                self.open_browser_btn.setVisible(True)
        elif self.web_view:
            if self._map_loaded:
                self._add_layer_js(name, geojson, style)
            else:
                # 页面还没加载完，加入队列
                self._pending_layers.append((name, geojson, style))

    def remove_layer(self, name: str):
        """移除图层"""
        if name in self.layers:
            del self.layers[name]
        if name in self.layer_order:
            self.layer_order.remove(name)

        if self.web_view and self._map_loaded:
            js = f"window.mapAPI && window.mapAPI.removeLayer('{name}');"
            self.web_view.page().runJavaScript(js)

    def set_layer_visibility(self, name: str, visible: bool):
        """设置图层可见性"""
        if name in self.layers:
            self.layers[name]["visible"] = visible

        if self.web_view and self._map_loaded:
            js = f"window.mapAPI && window.mapAPI.setVisibility('{name}', {'true' if visible else 'false'});"
            self.web_view.page().runJavaScript(js)

    def zoom_to_layer(self, name: str):
        """缩放到图层"""
        if self.web_view and self._map_loaded:
            js = f"window.mapAPI && window.mapAPI.zoomToLayer('{name}');"
            self.web_view.page().runJavaScript(js)

    def move_layer_up(self, name: str):
        """上移图层（渲染顺序）"""
        if name in self.layer_order:
            idx = self.layer_order.index(name)
            if idx > 0:
                self.layer_order[idx], self.layer_order[idx-1] = self.layer_order[idx-1], self.layer_order[idx]
                self.refresh()

    def move_layer_down(self, name: str):
        """下移图层（渲染顺序）"""
        if name in self.layer_order:
            idx = self.layer_order.index(name)
            if idx < len(self.layer_order) - 1:
                self.layer_order[idx], self.layer_order[idx+1] = self.layer_order[idx+1], self.layer_order[idx]
                self.refresh()

    def refresh(self):
        """刷新地图"""
        if self.web_view:
            html = self._generate_map_html()
            self._load_html(html)

    def set_center(self, lat: float, lng: float, zoom: int = None):
        """设置地图中心"""
        self.center = [lat, lng]
        if zoom:
            self.zoom = zoom
        if self.web_view:
            js = f"window.mapAPI && window.mapAPI.setView({lat}, {lng}, {zoom or self.zoom});"
            self.web_view.page().runJavaScript(js)
            
    def update_layer_style(self, name, style):
        """更新图层样式"""
        if name in self.layers:
            self.layers[name]['style'] = style
        if self.web_view and self._map_loaded:
            import json
            style_str = json.dumps(style)
            js = f"window.mapAPI && window.mapAPI.setStyle('{name}', {style_str});"
            self.web_view.page().runJavaScript(js)

    def __del__(self):
        if self.temp_file and os.path.exists(self.temp_file):
            try: os.unlink(self.temp_file)
            except: pass

class LayerPanel(QWidget):
    # ... LayerPanel code remains the same ...
    # (Copying LayerPanel code from previous version)
    layer_removed = Signal(str)
    layer_visibility_changed = Signal(str, bool)
    layer_moved_up = Signal(str)
    layer_moved_down = Signal(str)
    layer_zoom_requested = Signal(str)
    panel_closed = Signal()
    # 新增信号
    layer_renamed = Signal(str, str)  # old_name, new_name
    layer_export_requested = Signal(str, str)  # name, format
    open_attribute_table = Signal(str)  # name
    layer_properties_requested = Signal(str)  # name
    layer_style_changed = Signal(str, dict)  # name, style - 样式变更信号
    layer_crs_changed = Signal(str, object, str, str, int)  # name, geojson, crs, geom_type, feature_count

    def __init__(self, parent=None):
        super().__init__(parent)
        self.layers = {}  # 存储图层信息 {name: {"item": item, "path": path, "geojson": data, "crs": crs}}
        self.layer_order = []  # 图层顺序
        self._block_signals = False  # 防止信号循环
        self._copied_style = None  # 复制的样式
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        header = QFrame()
        header.setStyleSheet("background: #e0e0e0;")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(8, 6, 8, 6)

        self.title_label = QLabel()
        self.title_label.setStyleSheet("font-weight: bold;")
        header_layout.addWidget(self.title_label)
        header_layout.addStretch()

        self.close_btn = QPushButton("✕")
        self.close_btn.setFixedSize(24, 24)
        self.close_btn.setStyleSheet("QPushButton{background:#fff;border:1px solid #ccc;border-radius:3px;}QPushButton:hover{background:#e0e0e0;}")
        self.close_btn.clicked.connect(self.panel_closed.emit)
        header_layout.addWidget(self.close_btn)

        layout.addWidget(header)

        # 工具栏
        toolbar = QFrame()
        toolbar.setStyleSheet("background: #f0f0f0; border-bottom: 1px solid #ddd;")
        tb_layout = QHBoxLayout(toolbar)
        tb_layout.setContentsMargins(4, 4, 4, 4)
        tb_layout.setSpacing(2)

        self.btn_up = QPushButton("↑")
        self.btn_up.setFixedSize(28, 24)
        self.btn_up.setToolTip(tr("layer.move_up"))
        self.btn_up.clicked.connect(self.move_layer_up)

        self.btn_down = QPushButton("↓")
        self.btn_down.setFixedSize(28, 24)
        self.btn_down.setToolTip(tr("layer.move_down"))
        self.btn_down.clicked.connect(self.move_layer_down)

        self.btn_delete = QPushButton("✕")
        self.btn_delete.setFixedSize(28, 24)
        self.btn_delete.setToolTip(tr("layer.delete"))
        self.btn_delete.clicked.connect(self.delete_selected_layer)

        self.btn_zoom = QPushButton("⌖")
        self.btn_zoom.setFixedSize(28, 24)
        self.btn_zoom.setToolTip(tr("layer.zoom_to"))
        self.btn_zoom.clicked.connect(self.zoom_to_selected)

        for btn in [self.btn_up, self.btn_down, self.btn_delete, self.btn_zoom]:
            btn.setStyleSheet("QPushButton{background:#fff;border:1px solid #ccc;border-radius:3px;}QPushButton:hover{background:#e0e0e0;}")
            tb_layout.addWidget(btn)

        tb_layout.addStretch()
        layout.addWidget(toolbar)

        # 图层列表
        self.layer_tree = QTreeWidget()
        self.layer_tree.setHeaderHidden(True)
        self.layer_tree.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.layer_tree.setStyleSheet("""
            QTreeWidget { border: none; background: #fafafa; }
            QTreeWidget::item { padding: 4px; }
            QTreeWidget::item:hover { background: #e3f2fd; }
            QTreeWidget::item:selected { background: #1976d2; color: white; }
        """)
        self.layer_tree.itemChanged.connect(self.on_item_changed)
        # 启用右键菜单
        self.layer_tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.layer_tree.customContextMenuRequested.connect(self.show_context_menu)

        # 提示文字
        self.placeholder = QLabel()
        self.placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.placeholder.setStyleSheet("color: #999; padding: 30px;")

        layout.addWidget(self.placeholder)
        layout.addWidget(self.layer_tree)
        self.layer_tree.hide()
        self.retranslate_ui()

    def retranslate_ui(self):
        self.title_label.setText(tr("layer.title"))
        self.close_btn.setToolTip(tr("layer.close_tooltip"))
        self.btn_up.setToolTip(tr("layer.move_up"))
        self.btn_down.setToolTip(tr("layer.move_down"))
        self.btn_delete.setToolTip(tr("layer.delete"))
        self.btn_zoom.setToolTip(tr("layer.zoom_to"))
        self.placeholder.setText(tr("layer.drop_hint"))
        for name, layer_data in self.layers.items():
            self._update_layer_item_text(name, layer_data)

    def _format_layer_display_text(self, name, geom_type="", feature_count=0):
        return f"{name} ({geom_type}, {tr('layer.feature_count', count=feature_count)})"

    def _update_layer_item_text(self, name, layer_data):
        item = layer_data.get("item") if isinstance(layer_data, dict) else None
        if item is not None:
            item.setText(
                0,
                self._format_layer_display_text(
                    name,
                    layer_data.get("geom_type", ""),
                    layer_data.get("feature_count", 0),
                ),
            )

    def add_layer(self, name, geom_type="", feature_count=0, file_path="", geojson=None, crs="EPSG:4326"):
        """添加图层"""
        if name in self.layers:
            return  # 已存在

        if self.layer_tree.isHidden():
            self.placeholder.hide()
            self.layer_tree.show()

        self._block_signals = True
        display_text = self._format_layer_display_text(name, geom_type, feature_count)
        item = QTreeWidgetItem(self.layer_tree, [display_text])
        item.setData(0, Qt.ItemDataRole.UserRole, name)
        item.setCheckState(0, Qt.CheckState.Checked)
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        self._block_signals = False

        self.layers[name] = {
            "item": item,
            "path": file_path,
            "geojson": geojson,
            "geom_type": geom_type,
            "feature_count": feature_count,
            "crs": crs,
            "style": {}
        }
        self.layer_order.append(name)

    def update_layer_geojson(self, name: str, geojson: dict) -> None:
        """将图层面板中已存在图层的 GeoJSON 与要素数同步为地图中的编辑结果（如对比模式下人工增删点）。"""
        if not name or name not in self.layers or not isinstance(geojson, dict):
            return
        features = geojson.get("features") or []
        fc = len(features)
        layer_data = self.layers[name]
        layer_data["geojson"] = geojson
        layer_data["feature_count"] = fc
        item = layer_data.get("item")
        geom_type = layer_data.get("geom_type", "")
        if item is not None:
            item.setText(0, self._format_layer_display_text(name, geom_type, fc))

    # ... (Rest of LayerPanel methods omitted for brevity, assuming standard implementation as before) ...
    # 为了保证代码正确运行，我们需要完整保留LayerPanel的所有方法。
    # 这里简单起见，我只保留关键方法，如果需要完整运行，应该复制原来的所有方法。
    # 实际上，上面的LayerPanel代码已经包含了很多方法，我将补充缺失的 show_context_menu 等
    
    def show_context_menu(self, position):
        item = self.layer_tree.itemAt(position)
        if not item:
            return
        name = item.data(0, Qt.ItemDataRole.UserRole)
        if not name or name not in self.layers:
            return

        menu = QMenu(self)
        menu.addAction(tr("layer.context.open_attribute"), lambda: self._open_attribute_table(name))
        menu.addSeparator()

        menu.addAction(tr("layer.zoom_to"), lambda: self.layer_zoom_requested.emit(name))
        menu.addSeparator()

        menu.addAction(tr("layer.context.rename"), lambda: self._rename_layer(name))
        menu.addAction(tr("layer.context.edit_style"), lambda: self._edit_simple_style(name))
        menu.addAction(tr("layer.context.change_crs"), lambda: self._change_layer_crs(name))

        style_menu = menu.addMenu(tr("layer.context.style"))
        style_menu.addAction(tr("layer.context.copy_style"), lambda: self._copy_style(name))
        paste_action = style_menu.addAction(tr("layer.context.paste_style"), lambda: self._paste_style(name))
        paste_action.setEnabled(self._copied_style is not None)

        export_menu = menu.addMenu(tr("layer.context.export"))
        export_menu.addAction(tr("layer.context.save_geojson"), lambda: self._export_layer(name, "geojson"))
        export_menu.addAction(tr("layer.context.save_shp"), lambda: self._export_layer(name, "shp"))
        export_menu.addAction(tr("layer.context.save_gpkg"), lambda: self._export_layer(name, "gpkg"))

        menu.addSeparator()
        menu.addAction(tr("layer.context.properties"), lambda: self._show_layer_properties(name))
        menu.addSeparator()
        menu.addAction(tr("layer.context.remove"), lambda: self._confirm_remove_layer(name))
        menu.exec(self.layer_tree.viewport().mapToGlobal(position))

    def _load_crs_catalog(self):
        cached = getattr(self, "_crs_catalog_cache", None)
        if cached is not None:
            return cached

        catalog = {"Geographic": [], "Projected": [], "Other": []}
        try:
            from pyproj.database import query_crs_info

            infos = query_crs_info(auth_name="EPSG")
            for info in infos:
                code = f"{info.auth_name}:{info.code}"
                name = (info.name or code).strip()
                type_text = str(info.type).upper()
                if "GEOGRAPHIC" in type_text or "GEODETIC" in type_text:
                    group = "Geographic"
                elif "PROJECTED" in type_text:
                    group = "Projected"
                else:
                    group = "Other"
                catalog[group].append((name, code))
        except Exception:
            catalog["Geographic"] = [
                ("WGS 84", "EPSG:4326"),
                ("CGCS 2000", "EPSG:4490"),
                ("Xian 1980", "EPSG:4610"),
                ("Beijing 1954", "EPSG:4214"),
            ]
            catalog["Projected"] = [
                ("WGS 84 / Pseudo-Mercator", "EPSG:3857"),
                ("CGCS2000 / 3-degree Gauss-Kruger CM 105E", "EPSG:4547"),
                ("CGCS2000 / 3-degree Gauss-Kruger CM 111E", "EPSG:4549"),
            ]
            catalog["Other"] = []

        for key in catalog:
            catalog[key] = sorted(catalog[key], key=lambda x: x[0].lower())

        self._crs_catalog_cache = catalog
        return catalog

    def _pick_crs_from_catalog(self, title, initial_crs="EPSG:4326"):
        catalog = self._load_crs_catalog()
        init = (initial_crs or "").upper()

        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        dialog.setMinimumSize(680, 520)
        layout = QVBoxLayout(dialog)

        tip = QLabel(tr("layer.crs.pick_tip"))
        layout.addWidget(tip)

        search = QLineEdit()
        search.setPlaceholderText(tr("layer.crs.search_placeholder"))
        layout.addWidget(search)

        tree = QTreeWidget()
        tree.setColumnCount(2)
        tree.setHeaderLabels([tr("layer.crs.name"), tr("layer.crs.code")])
        tree.setRootIsDecorated(True)
        tree.setAlternatingRowColors(True)
        layout.addWidget(tree, 1)

        roots = {
            "Favorites": QTreeWidgetItem(tree, [tr("layer.crs.favorites"), ""]),
            "Geographic": QTreeWidgetItem(tree, [tr("layer.crs.geographic"), ""]),
            "Projected": QTreeWidgetItem(tree, [tr("layer.crs.projected"), ""]),
            "Other": QTreeWidgetItem(tree, [tr("layer.crs.other"), ""]),
        }
        for r in roots.values():
            r.setExpanded(False)

        by_code = {}
        for group in ("Geographic", "Projected", "Other"):
            for name, code in catalog.get(group, []):
                item = QTreeWidgetItem(roots[group], [name, code])
                item.setData(0, Qt.ItemDataRole.UserRole, code)
                by_code[code.upper()] = (name, group)

        favorites = [
            "EPSG:4326",
            "EPSG:4490",
            "EPSG:3857",
            "EPSG:4610",
            "EPSG:4214",
            "EPSG:4547",
            "EPSG:4549",
        ]
        for code in favorites:
            rec = by_code.get(code.upper())
            if rec:
                item = QTreeWidgetItem(roots["Favorites"], [rec[0], code.upper()])
                item.setData(0, Qt.ItemDataRole.UserRole, code.upper())
        roots["Favorites"].setExpanded(True)

        def apply_filter(text):
            key = (text or "").strip().lower()
            for i in range(tree.topLevelItemCount()):
                root = tree.topLevelItem(i)
                visible_children = 0
                for j in range(root.childCount()):
                    child = root.child(j)
                    name_text = (child.text(0) or "").lower()
                    code_text = (child.text(1) or "").lower()
                    visible = (not key) or (key in name_text) or (key in code_text)
                    child.setHidden(not visible)
                    if visible:
                        visible_children += 1
                root.setHidden(visible_children == 0)
                if key and visible_children > 0:
                    root.setExpanded(True)

        search.textChanged.connect(apply_filter)

        button_row = QHBoxLayout()
        button_row.addStretch(1)
        ok_btn = QPushButton(tr("common.ok"))
        cancel_btn = QPushButton(tr("common.cancel"))
        ok_btn.setFixedWidth(96)
        cancel_btn.setFixedWidth(96)
        button_row.addWidget(cancel_btn)
        button_row.addWidget(ok_btn)
        layout.addLayout(button_row)

        picked = {"code": None}

        def accept_pick():
            current = tree.currentItem()
            if not current:
                QMessageBox.warning(dialog, tr("common.warning"), tr("layer.crs.select_warning"))
                return
            code = current.data(0, Qt.ItemDataRole.UserRole)
            if not code:
                QMessageBox.warning(dialog, tr("common.warning"), tr("layer.crs.select_item_warning"))
                return
            picked["code"] = str(code)
            dialog.accept()

        ok_btn.clicked.connect(accept_pick)
        cancel_btn.clicked.connect(dialog.reject)
        tree.itemDoubleClicked.connect(lambda *_: accept_pick())

        if init:
            for i in range(tree.topLevelItemCount()):
                root = tree.topLevelItem(i)
                for j in range(root.childCount()):
                    child = root.child(j)
                    code = str(child.data(0, Qt.ItemDataRole.UserRole) or "").upper()
                    if code == init:
                        tree.setCurrentItem(child)
                        root.setExpanded(True)
                        break

        ok = dialog.exec() == QDialog.DialogCode.Accepted
        return picked["code"], ok

    def _change_layer_crs(self, name):
        if name not in self.layers:
            return

        layer_data = self.layers[name]
        geojson = layer_data.get("geojson")
        file_path = layer_data.get("path") or ""

        try:
            import geopandas as gpd

            gdf = None
            # Prefer reading original vector file so SHP/GPKG CRS metadata is preserved.
            if file_path and os.path.exists(file_path):
                ext = os.path.splitext(file_path)[1].lower()
                if ext in [".shp", ".gpkg", ".geojson", ".json"]:
                    try:
                        gdf = gpd.read_file(file_path)
                    except Exception:
                        gdf = None

            # Fallback to in-memory GeoJSON.
            if gdf is None:
                if not geojson or not isinstance(geojson, dict):
                    QMessageBox.warning(self, tr("common.warning"), tr("layer.crs.no_vector"))
                    return
                gdf = gpd.GeoDataFrame.from_features(geojson.get("features", []))
        except Exception as e:
            QMessageBox.critical(self, tr("layer.crs.convert_failed"), str(e))
            return

        if gdf.empty:
            QMessageBox.warning(self, tr("common.warning"), tr("layer.crs.empty_layer"))
            return

        current_crs = str(gdf.crs) if getattr(gdf, "crs", None) else layer_data.get("crs", "EPSG:4326")
        source_crs, ok = self._pick_crs_from_catalog(tr("layer.crs.source_title"), initial_crs=str(current_crs or "EPSG:4326"))
        if not ok:
            return
        if not source_crs:
            QMessageBox.warning(self, tr("common.warning"), tr("layer.crs.source_required"))
            return

        target_crs, ok = self._pick_crs_from_catalog(tr("layer.crs.target_title"), initial_crs="EPSG:4326")
        if not ok:
            return
        if not target_crs:
            QMessageBox.warning(self, tr("common.warning"), tr("layer.crs.target_required"))
            return

        try:
            from pyproj import CRS

            # Validate CRS strings early to provide clear feedback.
            CRS.from_user_input(source_crs)
            CRS.from_user_input(target_crs)

            if gdf.crs is None:
                gdf.set_crs(source_crs, inplace=True, allow_override=True)
            else:
                # Respect user-selected source CRS for inconsistent metadata cases.
                gdf.set_crs(source_crs, inplace=True, allow_override=True)

            if source_crs != target_crs:
                gdf = gdf.to_crs(target_crs)

            new_geojson = json.loads(gdf.to_json())
            geom_type = "Unknown"
            if len(gdf) > 0:
                geom_types = sorted(set(gdf.geometry.geom_type.tolist()))
                geom_type = geom_types[0] if len(geom_types) == 1 else "Mixed"
            feature_count = int(len(gdf))

            layer_data["geojson"] = new_geojson
            layer_data["crs"] = target_crs
            layer_data["geom_type"] = geom_type
            layer_data["feature_count"] = feature_count

            item = layer_data.get("item")
            if item:
                item.setText(0, self._format_layer_display_text(name, geom_type, feature_count))

            self.layer_crs_changed.emit(name, new_geojson, target_crs, geom_type, feature_count)
            QMessageBox.information(self, tr("common.complete"), tr("layer.crs.convert_complete", crs=target_crs))
        except Exception as e:
            QMessageBox.critical(self, tr("layer.crs.convert_failed"), str(e))

    def _confirm_remove_layer(self, name):
        reply = QMessageBox.question(
            self,
            tr("layer.confirm_title"),
            tr("layer.confirm_remove", name=name),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._remove_layer_with_confirm(name)

    def _show_layer_properties(self, name):
        if name not in self.layers:
            return
        layer_data = self.layers[name]
        info = tr(
            "layer.properties_info",
            name=name,
            path=layer_data.get("path", "N/A"),
            geom_type=layer_data.get("geom_type", "Unknown"),
            count=layer_data.get("feature_count", 0),
            crs=layer_data.get("crs", "EPSG:4326"),
        )
        QMessageBox.information(self, tr("layer.properties_title"), info)

    def _rename_layer(self, old_name):
        if old_name not in self.layers:
            return
        new_name, ok = QInputDialog.getText(self, tr("layer.rename_title"), tr("layer.new_name"), text=old_name)
        new_name = (new_name or "").strip()
        if not ok:
            return
        if not new_name:
            QMessageBox.warning(self, tr("common.warning"), tr("layer.name_empty"))
            return
        if new_name == old_name:
            return
        if new_name in self.layers:
            QMessageBox.warning(self, tr("common.warning"), tr("layer.name_exists"))
            return

        layer_data = self.layers.pop(old_name)
        item = layer_data.get("item")
        layer_data["item"] = item
        self.layers[new_name] = layer_data

        if old_name in self.layer_order:
            idx = self.layer_order.index(old_name)
            self.layer_order[idx] = new_name

        geom_type = layer_data.get("geom_type", "")
        feature_count = layer_data.get("feature_count", 0)
        display_text = self._format_layer_display_text(new_name, geom_type, feature_count)
        if item:
            item.setData(0, Qt.ItemDataRole.UserRole, new_name)
            self._block_signals = True
            item.setText(0, display_text)
            self._block_signals = False

        self.layer_renamed.emit(old_name, new_name)

    def _copy_style(self, name):
        if name not in self.layers:
            return
        style = self.layers[name].get("style") or {}
        self._copied_style = dict(style)

    def _paste_style(self, name):
        if name not in self.layers:
            return
        if self._copied_style is None:
            QMessageBox.warning(self, tr("common.warning"), tr("layer.no_style_to_paste"))
            return
        style = dict(self._copied_style)
        self.layers[name]["style"] = style
        self.layer_style_changed.emit(name, style)

    def _open_attribute_table(self, name):
        if name not in self.layers:
            return
        geojson = self.layers[name].get("geojson")
        if not geojson or not isinstance(geojson, dict):
            QMessageBox.warning(self, tr("common.warning"), tr("layer.no_attribute_data"))
            return

        features = geojson.get("features") or []
        props_list = []
        for f in features:
            props_list.append(f.get("properties") or {})

        columns = []
        col_set = set()
        for props in props_list[:200]:
            for k in props.keys():
                if k not in col_set:
                    col_set.add(k)
                    columns.append(k)

        dlg = QDialog(self)
        dlg.setWindowTitle(tr("layer.attribute_title", name=name))
        dlg.setMinimumSize(800, 500)
        layout = QVBoxLayout(dlg)

        table = QTableWidget()
        table.setColumnCount(len(columns))
        table.setHorizontalHeaderLabels(columns)

        row_limit = min(len(props_list), 2000)
        table.setRowCount(row_limit)
        for r in range(row_limit):
            props = props_list[r]
            for c, key in enumerate(columns):
                val = props.get(key, "")
                if val is None:
                    val = ""
                table.setItem(r, c, QTableWidgetItem(str(val)))

        table.resizeColumnsToContents()
        layout.addWidget(table)
        dlg.exec()

    def on_item_changed(self, item, column):
        if self._block_signals: return
        name = item.data(0, Qt.ItemDataRole.UserRole)
        if name:
            visible = item.checkState(0) == Qt.CheckState.Checked
            self.layer_visibility_changed.emit(name, visible)

    def get_selected_layer_name(self):
        item = self.layer_tree.currentItem()
        return item.data(0, Qt.ItemDataRole.UserRole) if item else None

    def select_all_layers(self):
        self.layer_tree.selectAll()

    def clear_selection(self):
        self.layer_tree.clearSelection()

    def show_selected_layer_properties(self):
        name = self.get_selected_layer_name()
        if not name or name not in self.layers:
            return
        
        layer_data = self.layers[name]
        info = tr(
            "layer.properties_info",
            name=name,
            path=layer_data.get("path", "N/A"),
            geom_type=layer_data.get("geom_type", "Unknown"),
            count=layer_data.get("feature_count", 0),
            crs=layer_data.get("crs", "EPSG:4326"),
        )
        
        QMessageBox.information(self, tr("layer.properties_title"), info)

    def edit_selected_layer_style(self):
        name = self.get_selected_layer_name()
        if not name:
            QMessageBox.warning(self, tr("common.warning"), tr("layer.select_first"))
            return
        self._edit_simple_style(name)

    def move_layer_up(self):
        name = self.get_selected_layer_name()
        if name:
            self.layer_moved_up.emit(name)
            # Update UI order
            item = self.layers[name]["item"]
            idx = self.layer_tree.indexOfTopLevelItem(item)
            if idx > 0:
                self.layer_tree.takeTopLevelItem(idx)
                self.layer_tree.insertTopLevelItem(idx - 1, item)
                self.layer_tree.setCurrentItem(item)

    def move_layer_down(self):
        name = self.get_selected_layer_name()
        if name:
            self.layer_moved_down.emit(name)
            # Update UI order
            item = self.layers[name]["item"]
            idx = self.layer_tree.indexOfTopLevelItem(item)
            if idx < self.layer_tree.topLevelItemCount() - 1:
                self.layer_tree.takeTopLevelItem(idx)
                self.layer_tree.insertTopLevelItem(idx + 1, item)
                self.layer_tree.setCurrentItem(item)

    def delete_selected_layer(self):
        name = self.get_selected_layer_name()
        if name:
            reply = QMessageBox.question(self, tr("layer.confirm_title"), tr("layer.confirm_delete", name=name), QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.Yes:
                self._remove_layer_with_confirm(name)

    def zoom_to_selected(self):
        name = self.get_selected_layer_name()
        if name:
            self.layer_zoom_requested.emit(name)

    def remove_layer(self, name):
        if name in self.layers:
            item = self.layers[name]["item"]
            idx = self.layer_tree.indexOfTopLevelItem(item)
            if idx >= 0:
                self.layer_tree.takeTopLevelItem(idx)
            del self.layers[name]
            if name in self.layer_order:
                self.layer_order.remove(name)
        if self.layer_tree.topLevelItemCount() == 0:
            self.layer_tree.hide()
            self.placeholder.show()

    def _remove_layer_with_confirm(self, name):
        self.remove_layer(name)
        self.layer_removed.emit(name)
        
    def _export_layer(self, name, format_type):
        if name not in self.layers:
            return
        geojson = self.layers[name].get("geojson")
        if not geojson:
            QMessageBox.warning(self, tr("layer.export_failed"), tr("layer.export_empty"))
            return

        format_type = (format_type or "").lower()
        if format_type == "geojson":
            default_name = f"{name}.geojson"
            filter_str = "GeoJSON (*.geojson)"
        elif format_type in ("shp", "shapefile"):
            default_name = f"{name}.shp"
            filter_str = "Shapefile (*.shp)"
        elif format_type in ("gpkg", "geopackage"):
            default_name = f"{name}.gpkg"
            filter_str = "GeoPackage (*.gpkg)"
        else:
            default_name = f"{name}.geojson"
            filter_str = "GeoJSON (*.geojson);;Shapefile (*.shp);;GeoPackage (*.gpkg)"

        save_path, _ = QFileDialog.getSaveFileName(self, tr("layer.export_title"), default_name, filter_str)
        if not save_path:
            return

        try:
            from src.io.vector import write_vector_file
            write_vector_file(geojson, save_path)
            QMessageBox.information(self, tr("layer.export_success"), tr("layer.export_success_message", path=save_path))
        except Exception as e:
            QMessageBox.critical(self, tr("layer.export_failed"), str(e))

    def _edit_simple_style(self, name):
        if name not in self.layers:
            return

        layer_data = self.layers[name]
        geom_type = (layer_data.get("geom_type") or "").lower()
        current_style = dict(layer_data.get("style") or {})

        def _qcolor_from_hex(h):
            try:
                c = QColor(h)
                if c.isValid():
                    return c
            except Exception:
                pass
            return QColor("#3388ff")

        fill_hex = current_style.get("fillColor") or current_style.get("color") or "#3388ff"
        stroke_hex = current_style.get("color") or "#3388ff"
        weight_val = float(current_style.get("weight") or current_style.get("linewidth") or 2)
        opacity_val = float(current_style.get("opacity") or 0.8)
        fill_opacity_val = float(current_style.get("fillOpacity") or 0.4)
        radius_val = int(current_style.get("radius") or 6)

        dlg = QDialog(self)
        dlg.setWindowTitle(tr("style.title", name=name))
        dlg.setMinimumWidth(520)
        layout = QVBoxLayout(dlg)

        form_group = QFrame()
        form_layout = QFormLayout(form_group)
        form_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        target_combo = QComboBox()
        target_combo.addItem(tr("style.fill_color"), "fill")
        target_combo.addItem(tr("style.stroke_color"), "stroke")
        form_layout.addRow(tr("style.apply_to"), target_combo)

        fill_preview = QPushButton(fill_hex)
        fill_preview.setFixedHeight(28)
        fill_preview.setStyleSheet(f"QPushButton{{text-align:left;padding-left:10px;background:{fill_hex};color:#fff;border:1px solid #ccc;border-radius:4px;}}")
        form_layout.addRow(tr("style.fill"), fill_preview)

        stroke_preview = QPushButton(stroke_hex)
        stroke_preview.setFixedHeight(28)
        stroke_preview.setStyleSheet(f"QPushButton{{text-align:left;padding-left:10px;background:{stroke_hex};color:#fff;border:1px solid #ccc;border-radius:4px;}}")
        form_layout.addRow(tr("style.stroke"), stroke_preview)

        weight_spin = QDoubleSpinBox()
        weight_spin.setRange(0.0, 20.0)
        weight_spin.setSingleStep(0.5)
        weight_spin.setValue(weight_val)
        form_layout.addRow(tr("style.weight"), weight_spin)

        opacity_spin = QDoubleSpinBox()
        opacity_spin.setRange(0.0, 1.0)
        opacity_spin.setSingleStep(0.05)
        opacity_spin.setValue(opacity_val)
        form_layout.addRow(tr("style.stroke_opacity"), opacity_spin)

        fill_opacity_spin = QDoubleSpinBox()
        fill_opacity_spin.setRange(0.0, 1.0)
        fill_opacity_spin.setSingleStep(0.05)
        fill_opacity_spin.setValue(fill_opacity_val)
        form_layout.addRow(tr("style.fill_opacity"), fill_opacity_spin)

        radius_spin = QSpinBox()
        radius_spin.setRange(1, 40)
        radius_spin.setValue(radius_val)
        symbol_combo = QComboBox()
        symbol_combo.setIconSize(QSize(22, 22))
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        poi_icon_dir = os.path.join(project_root, "output", "arcgis_10_2_poi_symbols")
        poi_symbol_icon_files = {
            "convenience": "convenience_store.png",
            "school": "school.png",
            "parking": "parking.png",
            "hotel": "hotel.png",
            "bank": "bank.png",
            "hospital": "hospital.png",
            "restaurant": "restaurant.png",
            "transit": "transit.png",
        }
        point_symbol_choices = [
            (tr("symbol.circle"), "circle"),
            (tr("symbol.square"), "square"),
            (tr("symbol.triangle"), "triangle"),
            (tr("symbol.diamond"), "diamond"),
            (tr("symbol.cross"), "cross"),
            (tr("symbol.star"), "star"),
            (tr("symbol.plus"), "plus"),
            (tr("symbol.hexagon"), "hexagon"),
            (tr("symbol.convenience"), "convenience"),
            (tr("symbol.school"), "school"),
            (tr("symbol.parking"), "parking"),
            (tr("symbol.hotel"), "hotel"),
            (tr("symbol.bank"), "bank"),
            (tr("symbol.hospital"), "hospital"),
            (tr("symbol.restaurant"), "restaurant"),
            (tr("symbol.transit"), "transit"),
        ]
        cur_sym = normalize_point_symbol(current_style.get("point_symbol"))
        for label, key in point_symbol_choices:
            icon_file = poi_symbol_icon_files.get(key)
            icon_path = os.path.join(poi_icon_dir, icon_file) if icon_file else ""
            if icon_path and os.path.exists(icon_path):
                symbol_combo.addItem(QIcon(icon_path), label, key)
            else:
                symbol_combo.addItem(label, key)
        for i in range(symbol_combo.count()):
            if symbol_combo.itemData(i) == cur_sym:
                symbol_combo.setCurrentIndex(i)
                break

        sym_label = QLabel(tr("style.point_symbol"))
        symbol_mode = QComboBox()
        symbol_mode.addItem(tr("style.unified"), "unified")
        symbol_mode.addItem(tr("style.by_field"), "by_field")
        cur_field = str(current_style.get("point_symbol_field") or "").strip()

        field_combo = QComboBox()
        field_combo.setMinimumWidth(200)
        geojson = layer_data.get("geojson") or {}
        prop_keys: list = []
        seen_k = set()
        for f in (geojson.get("features") or [])[:500]:
            for k in (f.get("properties") or {}).keys():
                if k not in seen_k and k != "geometry":
                    seen_k.add(k)
                    prop_keys.append(k)
        prop_keys.sort()
        for k in prop_keys:
            field_combo.addItem(k, k)
        if not cur_field:
            for preferred in ["query_category", "poi_category", "category", "类别", "类型", "type", "poi_type", "class", "kind"]:
                if preferred in prop_keys:
                    cur_field = preferred
                    break
        if cur_field:
            ix = field_combo.findData(cur_field)
            if ix >= 0:
                field_combo.setCurrentIndex(ix)
        symbol_mode.setCurrentIndex(1 if cur_field else 0)

        def _sync_symbol_mode_widgets():
            by_field = symbol_mode.currentData() == "by_field"
            field_combo.setEnabled(by_field)
            sym_label.setText(tr("style.default_symbol") if by_field else tr("style.point_symbol"))

        symbol_mode.currentIndexChanged.connect(lambda _i=0: _sync_symbol_mode_widgets())
        _sync_symbol_mode_widgets()

        if "point" in geom_type:
            form_layout.addRow(tr("style.point_size"), radius_spin)
            form_layout.addRow(tr("style.symbol_mode"), symbol_mode)
            form_layout.addRow(tr("style.category_field"), field_combo)
            form_layout.addRow(sym_label, symbol_combo)
            if os.path.isdir(poi_icon_dir):
                icon_hint = QLabel(tr("style.icon_hint", path=poi_icon_dir))
                icon_hint.setWordWrap(True)
                icon_hint.setStyleSheet("color:#666;font-size:11px;")
                form_layout.addRow("", icon_hint)

        layout.addWidget(form_group)

        palette_frame = QFrame()
        palette_layout = QVBoxLayout(palette_frame)
        palette_layout.setContentsMargins(0, 0, 0, 0)

        palette_title = QLabel(tr("style.palette"))
        palette_title.setStyleSheet("font-weight:bold;color:#333;margin:6px 0;")
        palette_layout.addWidget(palette_title)

        grid = QGridLayout()
        grid.setSpacing(6)

        colors = [
            "#000000","#404040","#808080","#C0C0C0","#FFFFFF","#FF0000","#00FF00","#0000FF",
            "#FFFF00","#00FFFF","#FF00FF","#800000","#008000","#000080","#808000","#008080",
            "#800080","#FFA500","#A52A2A","#8B4513","#2E8B57","#4682B4","#1E90FF","#6A5ACD",
            "#DC143C","#FF1493","#FF69B4","#FFD700","#ADFF2F","#32CD32","#00FA9A","#20B2AA",
            "#00BFFF","#4169E1","#7B68EE","#9370DB","#BA55D3","#D2691E","#CD5C5C","#F4A460",
        ]

        def _apply_hex(hex_color):
            nonlocal fill_hex, stroke_hex
            if target_combo.currentData() == "fill":
                fill_hex = hex_color
                fill_preview.setText(hex_color)
                fill_preview.setStyleSheet(f"QPushButton{{text-align:left;padding-left:10px;background:{hex_color};color:#fff;border:1px solid #ccc;border-radius:4px;}}")
            else:
                stroke_hex = hex_color
                stroke_preview.setText(hex_color)
                stroke_preview.setStyleSheet(f"QPushButton{{text-align:left;padding-left:10px;background:{hex_color};color:#fff;border:1px solid #ccc;border-radius:4px;}}")

        for idx, c in enumerate(colors):
            btn = QPushButton("")
            btn.setFixedSize(22, 22)
            btn.setStyleSheet(f"QPushButton{{background:{c};border:1px solid #bbb;border-radius:3px;}}QPushButton:hover{{border:1px solid #1976d2;}}")
            btn.clicked.connect(lambda _=False, cc=c: _apply_hex(cc))
            grid.addWidget(btn, idx // 8, idx % 8)

        palette_layout.addLayout(grid)

        more_btn = QPushButton(tr("style.more_colors"))
        more_btn.setFixedWidth(120)
        def _pick_more():
            initial = _qcolor_from_hex(fill_hex if target_combo.currentData() == "fill" else stroke_hex)
            picked = QColorDialog.getColor(initial, dlg, tr("style.pick_color"))
            if picked.isValid():
                _apply_hex(picked.name())
        more_btn.clicked.connect(_pick_more)
        palette_layout.addWidget(more_btn, alignment=Qt.AlignmentFlag.AlignLeft)

        layout.addWidget(palette_frame)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        ok_btn = QPushButton(tr("common.ok"))
        cancel_btn = QPushButton(tr("common.cancel"))
        ok_btn.setFixedWidth(90)
        cancel_btn.setFixedWidth(90)
        btn_row.addWidget(cancel_btn)
        btn_row.addWidget(ok_btn)
        layout.addLayout(btn_row)

        def _apply_and_close():
            style = {
                "color": stroke_hex,
                "weight": float(weight_spin.value()),
                "opacity": float(opacity_spin.value()),
                "fillColor": fill_hex,
                "fillOpacity": float(fill_opacity_spin.value()),
            }
            if "point" in geom_type:
                style["radius"] = int(radius_spin.value())
                sym = symbol_combo.currentData()
                if sym:
                    style["point_symbol"] = str(sym)
                if symbol_mode.currentData() == "by_field":
                    fld = field_combo.currentData()
                    if not fld:
                        QMessageBox.warning(dlg, tr("common.warning"), tr("style.field_required"))
                        return
                    vals: list = []
                    for f in (layer_data.get("geojson") or {}).get("features") or []:
                        p = f.get("properties") or {}
                        if fld in p:
                            v = p[fld]
                            vals.append("" if v is None else str(v))
                    uniq = sorted(set(vals))
                    if len(uniq) < 2:
                        QMessageBox.warning(dlg, tr("common.warning"), tr("style.field_too_few"))
                        return
                    if str(fld).strip().lower() == "point_symbol":
                        mmap = {vk: normalize_point_symbol(vk) for vk in uniq}
                    else:
                        mmap = assign_symbols_to_categories(uniq)
                    style["point_symbol_field"] = str(fld)
                    style["point_symbol_map"] = mmap
                else:
                    style["point_symbol_field"] = ""
                    style["point_symbol_map"] = {}
            layer_data["style"] = dict(layer_data.get("style") or {})
            layer_data["style"].update(style)
            self.layer_style_changed.emit(name, style)
            dlg.accept()

        ok_btn.clicked.connect(_apply_and_close)
        cancel_btn.clicked.connect(dlg.reject)

        fill_preview.clicked.connect(lambda: target_combo.setCurrentIndex(0))
        stroke_preview.clicked.connect(lambda: target_combo.setCurrentIndex(1))
        dlg.exec()

class AlgorithmPanel(QWidget):
    """算法工具面板"""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.title = QLabel()
        self.title.setStyleSheet("font-weight: bold; padding: 8px; background: #e0e0e0;")
        layout.addWidget(self.title)

        self.algo_tree = QTreeWidget()
        self.algo_tree.setHeaderHidden(True)
        self.algo_tree.setStyleSheet("QTreeWidget { border: none; background: #fafafa; }")

        layout.addWidget(self.algo_tree)
        self.retranslate_ui()

    def retranslate_ui(self):
        self.title.setText(tr("algorithm.title"))
        self.algo_tree.clear()
        categories = [
            tr("algorithm.line"),
            tr("algorithm.polygon"),
            tr("algorithm.point"),
            tr("algorithm.conflict"),
            tr("algorithm.carto"),
        ]
        for cat in categories:
            item = QTreeWidgetItem(self.algo_tree, [cat])
            item.setExpanded(True)
            QTreeWidgetItem(item, [tr("algorithm.use_ai")])


class AutoHeightTextEdit(QTextEdit):
    """Read-only text view that wraps aggressively and grows to content height."""

    def __init__(self, text="", min_height=32, max_height=1600, parent=None):
        super().__init__(parent)
        self._min_height = min_height
        self._max_height = max_height
        self.setReadOnly(True)
        self.setAcceptRichText(False)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        self.setWordWrapMode(QTextOption.WrapMode.WrapAnywhere)
        option = QTextOption()
        option.setWrapMode(QTextOption.WrapMode.WrapAnywhere)
        self.document().setDefaultTextOption(option)
        self.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse | Qt.TextInteractionFlag.TextSelectableByKeyboard
        )
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.document().setDocumentMargin(0)
        self.document().documentLayout().documentSizeChanged.connect(self.adjust_text_height)
        self.setPlainText(text)
        QTimer.singleShot(0, self.adjust_text_height)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.adjust_text_height()

    def _sync_document_width(self):
        width = max(80, self.viewport().width() - 2)
        if abs(self.document().textWidth() - width) > 1:
            self.document().setTextWidth(width)

    def adjust_text_height(self, *_args):
        self._sync_document_width()
        height = int(self.document().size().height()) + 14
        target = max(self._min_height, min(height, self._max_height))
        self.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
            if height > self._max_height
            else Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.setFixedHeight(target)
        self.updateGeometry()


class ChatBubble(QFrame):
    """现代化聊天气泡 - 类似主流AI IDE"""
    def __init__(self, text="", is_user=True, parent=None):
        super().__init__(parent)
        self.is_user = is_user
        self.setFrameStyle(QFrame.Shape.NoFrame)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        # 主布局
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(6, 5, 6, 5)
        main_layout.setSpacing(8)

        if is_user:
            main_layout.addStretch(2)

        # 消息容器
        self.bubble = QFrame()
        self.bubble.setObjectName("bubble")
        self.bubble.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        bubble_layout = QVBoxLayout(self.bubble)
        bubble_layout.setContentsMargins(13, 10, 13, 12)
        bubble_layout.setSpacing(7)

        # 角色标签
        self.role_label = QLabel()
        self.role_label.setStyleSheet(f"""
            font-size: 12px;
            font-weight: 700;
            color: {'#1d4ed8' if is_user else '#475569'};
            margin-bottom: 2px;
        """)
        bubble_layout.addWidget(self.role_label)

        # 消息内容
        self.content_label = AutoHeightTextEdit(text, min_height=40, max_height=1600)
        self.content_label.setPlainText(text)
        self.content_label.setStyleSheet("""
            font-family: 'Segoe UI', 'Microsoft YaHei UI', 'Microsoft YaHei', Arial, sans-serif;
            font-size: 14px;
            line-height: 1.58;
            color: #172033;
            background: transparent;
            border: none;
        """)
        bubble_layout.addWidget(self.content_label)

        # 气泡样式
        if is_user:
            self.bubble.setStyleSheet("""
                QFrame#bubble {
                    background: #EAF3FF;
                    border-radius: 10px;
                    border: 1px solid #B8D7FF;
                }
            """)
            self.bubble.setMaximumWidth(460)
        else:
            self.bubble.setStyleSheet("""
                QFrame#bubble {
                    background: #ffffff;
                    border-radius: 10px;
                    border: 1px solid #D7E2EE;
                }
            """)
            self.bubble.setMaximumWidth(680)

        main_layout.addWidget(self.bubble)

        if not is_user:
            main_layout.addStretch(1)

        self.retranslate_ui()
        self._adjust_text_height()

    def retranslate_ui(self):
        self.role_label.setText(tr("ai.you") if self.is_user else tr("ai.assistant"))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.refresh_layout()

    def refresh_layout(self):
        available = max(240, self.width() - 28)
        cap = 460 if self.is_user else 680
        self.bubble.setMaximumWidth(min(cap, available))
        self.content_label.adjust_text_height()

    def set_text(self, text):
        self.content_label.setPlainText(text)
        self._adjust_text_height()

    def append_text(self, text):
        if not text:
            return
        cursor = self.content_label.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(str(text))
        self.content_label.setTextCursor(cursor)
        self._adjust_text_height()

    def get_text(self):
        return self.content_label.toPlainText()

    def _adjust_text_height(self, *_args):
        self.content_label.adjust_text_height()


class ThinkingBlock(QFrame):
    """思考过程块 - 可折叠，实时显示思考内容"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.is_expanded = True  # 默认展开
        self.thinking_lines = []
        self.todo_items = []
        self.todo_state = {}
        self._set_todo_items(self._default_todo_labels())
        self._pulse_index = 0
        self.started_ts = time.perf_counter()
        self.workflow_lines = []
        self.analysis_lines = []
        self.decision_lines = []
        self.validation_lines = []
        self.code_snippets = []
        self.streaming_line = ""

        self.setFrameStyle(QFrame.Shape.NoFrame)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(6, 8, 6, 8)
        main_layout.setSpacing(10)

        # 容器
        self.container = QFrame()
        self.container.setObjectName("thinking_container")
        self.container.setStyleSheet("""
            QFrame#thinking_container {
                background: #ffffff;
                border: 1px solid #d7e2ee;
                border-radius: 8px;
            }
        """)

        container_layout = QVBoxLayout(self.container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(0)

        # 头部（可点击折叠）
        self.header = QFrame()
        self.header.setStyleSheet("""
            QFrame {
                background: #edf6ff;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                border-bottom: 1px solid #d6e5f4;
            }
        """)
        self.header.setCursor(Qt.CursorShape.PointingHandCursor)
        self.header.mousePressEvent = lambda e: self.toggle_expand()

        header_layout = QHBoxLayout(self.header)
        header_layout.setContentsMargins(14, 9, 14, 9)
        header_layout.setSpacing(8)

        # 思考图标和标题
        self.status_icon = QLabel("•")
        self.status_icon.setStyleSheet("font-size: 20px; color: #2563eb; font-weight: 700;")
        header_layout.addWidget(self.status_icon)

        self.title_label = QLabel(tr("ai.thinking"))
        self.title_label.setStyleSheet("font-size: 14px; color: #25405f; font-weight: 600;")
        header_layout.addWidget(self.title_label)

        header_layout.addStretch()

        self.toggle_icon = QLabel("▼")
        self.toggle_icon.setStyleSheet("font-size: 12px; color: #64748b;")
        header_layout.addWidget(self.toggle_icon)

        container_layout.addWidget(self.header)

        # 内容区域
        self.content_frame = QFrame()
        content_layout = QVBoxLayout(self.content_frame)
        content_layout.setContentsMargins(14, 10, 14, 12)

        self.content_label = AutoHeightTextEdit(min_height=132, max_height=520)
        self.content_label.setStyleSheet("""
            font-size: 13px;
            color: #334155;
            font-family: 'Segoe UI', 'Microsoft YaHei UI', 'Microsoft YaHei', Arial, sans-serif;
            line-height: 1.52;
            background: transparent;
            border: none;
        """)
        content_layout.addWidget(self.content_label)

        container_layout.addWidget(self.content_frame)

        main_layout.addWidget(self.container, 1)
        self._adjust_content_height()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.refresh_layout()

    def refresh_layout(self):
        self.content_label.adjust_text_height()

    def _is_english(self):
        return current_language() == "en_US"

    def _default_todo_labels(self):
        return [
            tr("ai.thinking.todo.data"),
            tr("ai.thinking.todo.operator"),
            tr("ai.thinking.todo.execute"),
            tr("ai.thinking.todo.output"),
        ]

    def _fmt(self, key, **kwargs):
        return tr(f"ai.thinking.{key}", **kwargs)

    def _fmt_tool_step(self, key, tool, target=""):
        text = self._fmt(key, tool=tool, target=target)
        if not str(target or "").strip():
            text = re.sub(r"\s*[（(](?:数据|输出|data|output):\s*[）)]", "", text, flags=re.I)
        return text

    def toggle_expand(self):
        self.is_expanded = not self.is_expanded
        self.content_frame.setVisible(self.is_expanded)
        self.toggle_icon.setText("▼" if self.is_expanded else "▶")

        # 更新header样式
        if self.is_expanded:
            self.header.setStyleSheet("""
                QFrame {
                    background: #edf6ff;
                    border-top-left-radius: 8px;
                    border-top-right-radius: 8px;
                    border-bottom: 1px solid #d6e5f4;
                }
            """)
        else:
            self.header.setStyleSheet("""
                QFrame {
                    background: #edf6ff;
                    border-radius: 8px;
                }
            """)

    def append_thinking(self, text):
        """追加思考内容"""
        normalized = self._normalize_text(str(text))
        self._update_todo_by_text(normalized)
        if normalized:
            self.thinking_lines.append(normalized)
            self._append_workflow_line(normalized)
            low = normalized.lower()
            if any(k in low for k in [
                "读取数据", "当前图层", "semantic", "structure", "语义类型", "空间结构", "依据技能",
                "candidate_fields", "data source", "current layer", "spatial pattern", "identified as",
            ]):
                self.analysis_lines.append(normalized)
                self.analysis_lines = self.analysis_lines[-4:]
            if any(k in low for k in [
                "算子决策", "算子", "operators=", "operator", "selection", "typification",
                "displacement", "simplify", "prioritize", "candidate operators",
            ]):
                self.decision_lines.append(normalized)
                self.decision_lines = self.decision_lines[-4:]
            if any(k in low for k in [
                "验收", "评分", "verify", "verification", "vlm", "llm", "completion status",
                "输出文件", "output files", "projected crs",
            ]):
                self.validation_lines.append(normalized)
                self.validation_lines = self.validation_lines[-4:]
        # 限制长度，避免过长
        self.thinking_lines = self.thinking_lines[-6:]
        self._refresh_content()

    def append_stream_fragment(self, text):
        frag = str(text or "")
        if not frag:
            return
        self.streaming_line += frag
        self._refresh_content()

    def finalize_stream_line(self):
        text = self.streaming_line
        self.streaming_line = ""
        if text.strip():
            self.append_thinking(text)
        else:
            self._refresh_content()

    def _adjust_content_height(self, *_args):
        self.content_label.adjust_text_height()

    def add_code_snippet(self, code_text):
        code = (code_text or "").strip()
        if not code:
            return
        lines = [ln.rstrip() for ln in code.splitlines() if ln.strip()]
        if not lines:
            return
        preview = "\n".join(lines[:6])
        self.code_snippets.append(preview)
        self.code_snippets = self.code_snippets[-2:]
        self._refresh_content()

    def pulse(self):
        hints = [
            self._fmt("pulse.data"),
            self._fmt("pulse.match"),
            self._fmt("pulse.select"),
            self._fmt("pulse.output"),
        ]
        hint = hints[self._pulse_index % len(hints)]
        self._pulse_index += 1
        if not self.thinking_lines or self.thinking_lines[-1] != hint:
            self.thinking_lines.append(hint)
            self.thinking_lines = self.thinking_lines[-6:]
            self._append_workflow_line(hint, keep_short=True)
            self._refresh_content()

    def add_workflow_step(self, text, elapsed_sec=None):
        msg = self._normalize_text(text)
        if not msg:
            return
        if elapsed_sec is not None:
            msg = self._fmt("elapsed_suffix", text=msg, seconds=f"{elapsed_sec:.2f}")
        self._append_workflow_line(msg)
        self._refresh_content()

    def _append_workflow_line(self, text, keep_short=False):
        if not text:
            return
        if keep_short and self.workflow_lines and self.workflow_lines[-1] == text:
            return
        self.workflow_lines.append(text)
        self.workflow_lines = self.workflow_lines[-8:]

    def _normalize_text(self, text):
        t = (text or "").strip()
        prefix = ""
        if t.startswith("💭"):
            prefix = "💭 "
            t = t[1:].strip()
        lower = t.lower()
        translated = None
        if lower in {"analyzing...", "analyzing task..."}:
            translated = self._fmt("analyzing")
        elif re.match(r"^todo\s*1\s*/\s*6", t, flags=re.I):
            translated = self._fmt("todo1_6")
        elif re.match(r"^todo\s*2\s*/\s*6", t, flags=re.I):
            translated = self._fmt("todo2_6")
        elif re.match(r"^todo\s*3\s*/\s*6", t, flags=re.I):
            translated = self._fmt("todo3_6")
        elif re.match(r"^todo\s*4\s*/\s*6", t, flags=re.I):
            translated = self._fmt("todo4_6")
        else:
            m = re.match(r"^todo\s*2\s*/\s*4\s*:\s*plan iteration\s*(\d+)\s*/\s*(\d+)", t, flags=re.I)
            if m:
                translated = self._fmt("plan_iteration", index=m.group(1), total=m.group(2))
            if translated is None:
                m = re.match(r"^todo\s*3\s*/\s*4\s*:\s*execute iteration\s*(\d+)\s*/\s*(\d+)", t, flags=re.I)
                if m:
                    translated = self._fmt("execute_iteration", index=m.group(1), total=m.group(2))
            if translated is None:
                m = re.match(r"^plan\s*(\d+)\s*/\s*(\d+)\s*:\s*(.+)$", t, flags=re.I)
                if m:
                    translated = self._translate_plan_label(m.group(3).strip(), int(m.group(1)), int(m.group(2)))
            if translated is None and t.startswith("判定依据："):
                detail = t.split("：", 1)[1] if "：" in t else t
                translated = self._fmt("evidence", detail=detail)
            if translated is None and t.startswith("算子决策："):
                detail = t.split("：", 1)[1] if "：" in t else t
                translated = self._translate_operator_reasoning(detail)
            if translated is None:
                m = re.match(r"^第\s*(\d+)\s*轮结果：输入\s*(\d+)\s*个，输出\s*(\d+)\s*个，保留率\s*([^，]+)，冲突数\s*(\d+)", t)
                if m:
                    translated = self._fmt(
                        "round_result",
                        index=m.group(1),
                        input=m.group(2),
                        output=m.group(3),
                        retention=m.group(4),
                        conflicts=m.group(5),
                    )
            if translated is None:
                m = re.match(r"^验收结果：pass=(True|False), score=([^,]+), failed_rules=(.+)$", t, flags=re.I)
                if m:
                    passed = self._fmt("passed") if m.group(1).lower() == "true" else self._fmt("not_passed")
                    translated = self._fmt(
                        "verification_result",
                        status=passed,
                        score=m.group(2),
                        rules=self._translate_failed_rules(m.group(3)),
                    )
            if translated is None:
                m = re.match(r"^调用工具[:：]\s*([^（(]+)(?:[（(]数据[:：]\s*(.*?)[）)])?$", t)
                if m:
                    translated = self._fmt_tool_step("tool_call", m.group(1).strip(), (m.group(2) or "").strip())
            if translated is None:
                m = re.match(r"^完成工具[:：]\s*([^（(]+)(?:[（(]输出[:：]\s*(.*?)[）)])?$", t)
                if m:
                    translated = self._fmt_tool_step("tool_done_step", m.group(1).strip(), (m.group(2) or "").strip())
            if translated is None:
                m = re.match(r"^任务完成[（(]总耗时\s*([0-9.]+)s[）)]$", t)
                if m:
                    translated = self._fmt("task_complete", seconds=m.group(1))
        if lower.startswith("warnings:"):
            translated = t
        return prefix + (translated if translated is not None else t)

    def _translate_plan_label(self, label, idx, total):
        low = label.lower()
        zh_label = label
        if "resolve data source" in low or "classify task" in low or "复核上下文" in label:
            zh_label = tr("ai.thinking.plan.data")
        elif "生成本轮工具调用参数" in label or "generate code" in low:
            zh_label = tr("ai.thinking.plan.code")
        elif "执行工具" in label or "quality check" in low or "execute" in low:
            zh_label = tr("ai.thinking.plan.execute")
        elif "输出文件" in label or "completion status" in low or "summarize outputs" in low:
            zh_label = tr("ai.thinking.plan.output")
        elif "selection/simplify" in low or "displacement" in low:
            zh_label = tr("ai.thinking.plan.operators")
        return self._fmt("plan_step", index=idx, total=total, label=zh_label)

    def _translate_operator_reasoning(self, detail):
        text = str(detail or "")
        semantic = ""
        structure = ""
        ops = ""
        m = re.search(r"(?:数据类型|data type)=([^;]+)", text, flags=re.I)
        if m:
            semantic = m.group(1).strip()
        m = re.search(r"(?:空间结构|spatial structure)=([^;]+)", text, flags=re.I)
        if m:
            structure = m.group(1).strip()
        m = re.search(r"(?:候选算子|candidate operators)=\[([^\]]*)\]", text, flags=re.I)
        if m:
            ops = m.group(1).replace("'", "").strip()
        reason = ""
        if "主路沿线点" in text:
            reason = self._fmt("reason.road")
        elif "热点代表性" in text:
            reason = self._fmt("reason.hotspot")
        elif "覆盖完整性" in text:
            reason = self._fmt("reason.coverage")
        elif "空间覆盖" in text:
            reason = self._fmt("reason.spatial")
        parts = []
        if semantic:
            semantic = self._localize_semantic_label(semantic)
            parts.append(self._fmt("operator.semantic", semantic=semantic))
        if structure:
            structure = self._localize_structure_label(structure)
            parts.append(self._fmt("operator.structure", structure=structure))
        if ops:
            parts.append(self._fmt("operator.ops", ops=ops))
        if reason:
            parts.append(reason)
        joiner = "; " if self._is_english() else "；"
        end = "." if self._is_english() else "。"
        clean_parts = [str(part).strip().rstrip("。.") for part in parts if str(part).strip()]
        return joiner.join(clean_parts) + (end if clean_parts else "")

    def _localize_semantic_label(self, value):
        text = str(value or "").strip().strip("“”\"")
        if not self._is_english():
            return text
        mapping = {
            "通用点要素": "generic point features",
            "交通网络相关点": "transport-network related points",
            "公共服务设施点": "public service facility points",
            "监测事件点": "monitoring/event points",
            "商业兴趣点": "commercial POI points",
            "加油站/能源补给点": "gas station / energy supply points",
            "加油站或能源补给点": "gas station / energy supply points",
            "餐饮服务点": "restaurant / catering POI points",
            "generic_point": "generic point features",
            "transport_network_points": "transport-network related points",
            "public_service_points": "public service facility points",
            "monitoring_points": "monitoring/event points",
            "commercial_poi_points": "commercial POI points",
        }
        return mapping.get(text, text)

    def _localize_structure_label(self, value):
        text = str(value or "").strip().strip("“”\"")
        if not self._is_english():
            return text
        mapping = {
            "未知结构": "unknown pattern",
            "走廊型": "corridor pattern",
            "集聚型": "clustered pattern",
            "均匀型": "uniform pattern",
            "混合型": "mixed pattern",
            "多核心型": "multi-core pattern",
            "中心-边缘型": "center-periphery pattern",
            "unknown": "unknown pattern",
            "corridor": "corridor pattern",
            "clustered": "clustered pattern",
            "uniform": "uniform pattern",
            "mixed": "mixed pattern",
            "multi-core": "multi-core pattern",
            "center-periphery": "center-periphery pattern",
        }
        return mapping.get(text, text)

    def _translate_failed_rules(self, raw):
        text = str(raw or "")
        mapping = {
            "conflict_count_total <= target_conflict_count": tr("ai.thinking.rule.conflict"),
            "visual_change_detected": tr("ai.thinking.rule.visual_change"),
            "real_vlm_required": tr("ai.thinking.rule.real_vlm_required"),
            "real_vlm_callback_failed": tr("ai.thinking.rule.real_vlm_callback_failed"),
            "point_axis_coverage_ratio >= min": tr("ai.thinking.rule.axis"),
            "point_middle_gap_ratio <= max": tr("ai.thinking.rule.middle_gap"),
            "point_hotspot_rep_count >= min": tr("ai.thinking.rule.hotspot_rep"),
            "point_hotspot_coverage_ratio >= min": tr("ai.thinking.rule.hotspot"),
            "point_region_coverage_ratio >= min": tr("ai.thinking.rule.region"),
            "point_sparse_region_coverage_ratio >= min": tr("ai.thinking.rule.sparse"),
            "point_dense_region_singleton_ratio <= max": tr("ai.thinking.rule.dense_singleton"),
            "point_rgb_hotspot_overdraw <= max": tr("ai.thinking.rule.rgb_overdraw"),
            "retention_rate >= class_min": tr("ai.thinking.rule.retention"),
        }
        for src, dst in mapping.items():
            text = text.replace(src, dst)
        if self._is_english():
            text = re.sub(r"^\s*\[\s*", "", text)
            text = re.sub(r"\s*\]\s*$", "", text)
            text = text.replace("'", "").replace('"', "")
        return text

    def _set_todo_items(self, labels):
        normalized = [str(x).strip() for x in (labels or []) if str(x).strip()]
        if not normalized:
            return
        old_state = dict(self.todo_state)
        self.todo_items = [(f"todo_{i+1}", label) for i, label in enumerate(normalized)]
        self.todo_state = {}
        for key, _ in self.todo_items:
            self.todo_state[key] = old_state.get(key, "todo")

    def _ensure_todo_slot(self, index, total, label):
        total = max(total, len(self.todo_items), 1)
        labels = [x[1] for x in self.todo_items]
        if len(labels) < total:
            labels.extend([self._fmt("todo.pending", index=i + 1) for i in range(len(labels), total)])
        if 1 <= index <= total:
            labels[index - 1] = self._normalize_todo_label(label)
        self._set_todo_items(labels)

    def _update_todo_status(self, index, status):
        if not (1 <= index <= len(self.todo_items)):
            return
        key = self.todo_items[index - 1][0]
        self.todo_state[key] = status

    def _update_todo_by_text(self, text):
        low = str(text or "").lower()

        plan_match = re.match(r"^(?:plan|计划)\s*(\d+)\s*/\s*(\d+)\s*:\s*(.+)$", str(text or ""), flags=re.I)
        if plan_match:
            idx = int(plan_match.group(1))
            total = int(plan_match.group(2))
            label = plan_match.group(3).strip()
            self._ensure_todo_slot(idx, total, label)
            return

        todo_match = re.match(r"^todo\s*(\d+)\s*/\s*(\d+)\s*:\s*(.+)$", str(text or ""), flags=re.I)
        if todo_match:
            idx = int(todo_match.group(1))
            total = int(todo_match.group(2))
            label = todo_match.group(3).strip()
            self._ensure_todo_slot(idx, total, label)
            for i in range(1, idx):
                self._update_todo_status(i, "done")
            self._update_todo_status(idx, "doing")
            return

        for i, (_, label) in enumerate(self.todo_items, start=1):
            if (
                "输出文件" in label
                or "output" in str(label).lower()
                or "completion" in str(label).lower()
            ) and ("output files" in low or "输出" in low or "completed" in low or "task complete" in low):
                self._update_todo_status(i, "done")

    def _normalize_todo_label(self, label):
        low = str(label or "").lower()
        if "resolve data source" in low or "classify task" in low or "读取当前数据" in label:
            return tr("ai.thinking.todo.data")
        if "generate code" in low or "工具调用参数" in label:
            return tr("ai.thinking.todo.code")
        if "quality check" in low or "execute" in low or "执行工具" in label:
            return tr("ai.thinking.todo.execute")
        if "output files" in low or "completion status" in low or "输出文件" in label:
            return tr("ai.thinking.todo.output")
        if "选择候选算子" in label or "selection/simplify" in low or "displacement" in low:
            return tr("ai.thinking.todo.operator")
        return str(label or "").strip()

    def _refresh_content(self):
        details = self.workflow_lines[-6:]
        elapsed_total = time.perf_counter() - self.started_ts
        analysis = self.analysis_lines[-3:]
        decisions = self.decision_lines[-3:]
        validations = self.validation_lines[-3:]
        if details:
            details_text = "\n".join([f"· {d}" for d in details])
            text = self._fmt("progress_with_elapsed", seconds=f"{elapsed_total:.1f}") + "\n" + details_text
        else:
            text = self._fmt("progress_with_elapsed", seconds=f"{elapsed_total:.1f}")

        if analysis:
            text += "\n\n" + self._fmt("section.analysis") + "\n" + "\n".join([f"· {x}" for x in analysis])
        if decisions:
            text += "\n\n" + self._fmt("section.decision") + "\n" + "\n".join([f"· {x}" for x in decisions])
        if validations:
            text += "\n\n" + self._fmt("section.validation") + "\n" + "\n".join([f"· {x}" for x in validations])
        if self.streaming_line.strip():
            text += "\n\n" + self._fmt("section.streaming") + "\n" + self.streaming_line
        self.content_label.setPlainText(text)

    def set_status(self, status, icon="⏳"):
        """设置状态"""
        self.status_icon.setText(icon)
        self.title_label.setText(status)

    def finish(self):
        """完成思考"""
        for key, _ in self.todo_items:
            self.todo_state[key] = "done"
        self._append_workflow_line(self._fmt("task_complete", seconds=f"{time.perf_counter() - self.started_ts:.2f}"))
        self._refresh_content()
        self.set_status(tr("ai.tool_done"), "✓")
        self.title_label.setStyleSheet("font-size: 14px; color: #15803d; font-weight: 600;")
        self.status_icon.setStyleSheet("font-size: 15px; color: #15803d; font-weight: 700;")


class ToolCallBlock(QFrame):
    """工具调用块 - 显示工具执行状态"""
    def __init__(self, tool_name, parent=None):
        super().__init__(parent)
        self.tool_name = tool_name
        self.elapsed_sec = None

        self.setFrameStyle(QFrame.Shape.NoFrame)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(8, 2, 8, 2)

        # 工具调用指示器
        self.indicator = QFrame()
        self.indicator.setObjectName("tool_indicator")
        self.indicator.setStyleSheet("""
            QFrame#tool_indicator {
                background: #FFFFFF;
                border: 1px solid #DDE8F4;
                border-radius: 9px;
            }
        """)
        self.indicator.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        indicator_layout = QHBoxLayout(self.indicator)
        indicator_layout.setContentsMargins(10, 6, 10, 6)
        indicator_layout.setSpacing(8)

        self.icon_label = QLabel("•")
        self.icon_label.setFixedWidth(12)
        self.icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.icon_label.setStyleSheet("font-size: 14px; color: #F59E0B; font-weight: 800;")
        indicator_layout.addWidget(self.icon_label)

        self.name_label = QLabel(tr("ai.tool_calling", tool=tool_name))
        self.name_label.setStyleSheet("""
            font-size: 12px;
            color: #174A7C;
            font-weight: 700;
            font-family: 'Segoe UI', 'Microsoft YaHei UI', Arial, sans-serif;
        """)
        self.name_label.setWordWrap(False)
        self.name_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        indicator_layout.addWidget(self.name_label, 1)

        self.status_label = QLabel(tr("ai.tool_running"))
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.status_label.setStyleSheet("""
            QLabel {
                background: #FFF7ED;
                color: #B45309;
                border: 1px solid #FED7AA;
                border-radius: 8px;
                padding: 2px 7px;
                font-size: 10px;
                font-weight: 800;
            }
        """)
        indicator_layout.addWidget(self.status_label)

        main_layout.addWidget(self.indicator, 1)

    def set_complete(self, success=True, elapsed_sec=None):
        self.elapsed_sec = elapsed_sec
        if success:
            self.icon_label.setText("✓")
            self.icon_label.setStyleSheet("font-size: 13px; color: #16A34A; font-weight: 800;")
            done_text = tr("ai.tool_done")
            if elapsed_sec is not None:
                done_text = f"{done_text} {elapsed_sec:.1f}s"
            self.status_label.setText(done_text)
            self.status_label.setStyleSheet("""
                QLabel {
                    background: #ECFDF3;
                    color: #15803D;
                    border: 1px solid #BBF7D0;
                    border-radius: 8px;
                    padding: 2px 7px;
                    font-size: 10px;
                    font-weight: 800;
                }
            """)
            self.indicator.setStyleSheet("""
                QFrame#tool_indicator {
                    background: #FFFFFF;
                    border: 1px solid #DDE8F4;
                    border-radius: 9px;
                }
            """)
        else:
            self.icon_label.setText("✗")
            self.icon_label.setStyleSheet("font-size: 13px; color: #DC2626; font-weight: 800;")
            self.status_label.setText(tr("ai.tool_failed"))
            self.status_label.setStyleSheet("""
                QLabel {
                    background: #FEF2F2;
                    color: #DC2626;
                    border: 1px solid #FECACA;
                    border-radius: 8px;
                    padding: 2px 7px;
                    font-size: 10px;
                    font-weight: 800;
                }
            """)

    def retranslate_ui(self):
        self.name_label.setText(tr("ai.tool_calling", tool=self.tool_name))


class GeneratedFileItem(QFrame):
    """Compact generated-file row with safe path wrapping."""

    def __init__(self, file_path, parent=None):
        super().__init__(parent)
        self.file_path = str(file_path or "").strip()
        self.setObjectName("generatedFileItem")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setStyleSheet("""
            QFrame#generatedFileItem {
                background: #FFFFFF;
                border: 1px solid #E2ECF7;
                border-radius: 9px;
            }
        """)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(9, 8, 9, 8)
        layout.setSpacing(8)

        icon = QLabel(self._file_icon())
        icon.setFixedSize(24, 24)
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon.setStyleSheet("""
            QLabel {
                background: #EFF6FF;
                color: #2563EB;
                border: 1px solid #D5E3F2;
                border-radius: 7px;
                font-size: 13px;
                font-weight: 800;
            }
        """)
        layout.addWidget(icon)

        text_stack = QVBoxLayout()
        text_stack.setContentsMargins(0, 0, 0, 0)
        text_stack.setSpacing(3)
        name = os.path.basename(self.file_path) or self.file_path
        self.name_label = QLabel(name)
        self.name_label.setStyleSheet("font-size: 12px; color: #0B1F33; font-weight: 800;")
        self.name_label.setWordWrap(True)
        self.name_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        text_stack.addWidget(self.name_label)

        self.path_label = AutoHeightTextEdit(self.file_path, min_height=18, max_height=58)
        self.path_label.setToolTip(self.file_path)
        self.path_label.setStyleSheet("""
            font-family: Consolas, 'Segoe UI', 'Microsoft YaHei UI', monospace;
            font-size: 10px;
            line-height: 1.35;
            color: #64748B;
            background: transparent;
            border: none;
        """)
        text_stack.addWidget(self.path_label)
        layout.addLayout(text_stack, 1)

    def _file_icon(self):
        ext = os.path.splitext(self.file_path)[1].lower()
        if ext in {".png", ".jpg", ".jpeg"}:
            return "IMG"
        if ext in {".geojson", ".json", ".shp", ".gpkg"}:
            return "GIS"
        if ext in {".md", ".txt", ".log"}:
            return "DOC"
        return "OUT"

    def refresh_layout(self):
        self.path_label.adjust_text_height()


class MapAssistantIcon(QWidget):
    """Small AI + map glyph for the assistant header."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(36, 36)

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect().adjusted(1, 1, -1, -1)
        painter.setPen(QPen(QColor(204, 226, 255, 170), 1))
        painter.setBrush(QBrush(QColor(255, 255, 255, 34)))
        painter.drawRoundedRect(rect, 9, 9)

        painter.setPen(QPen(QColor(219, 239, 255, 210), 1.5))
        path = QPainterPath()
        path.moveTo(8, 23)
        path.cubicTo(13, 16, 18, 27, 24, 18)
        path.cubicTo(27, 14, 30, 16, 32, 12)
        painter.drawPath(path)

        painter.setBrush(QBrush(QColor("#22C55E")))
        painter.setPen(Qt.PenStyle.NoPen)
        for x, y, r in [(10, 24, 2), (19, 21, 2), (27, 15, 2)]:
            painter.drawEllipse(x - r, y - r, r * 2, r * 2)

        painter.setBrush(QBrush(QColor(255, 255, 255, 230)))
        painter.drawEllipse(9, 8, 9, 9)
        painter.setPen(QPen(QColor("#1F5AA6"), 1.2))
        painter.drawLine(13, 11, 13, 15)
        painter.drawLine(11, 13, 15, 13)


class HeaderMapTexture(QWidget):
    """Subtle GIS grid overlay for the header."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(QColor(255, 255, 255, 26), 1))
        step = 24
        for x in range(-step, self.width() + step, step):
            painter.drawLine(x, 0, x + 28, self.height())
        for y in range(8, self.height(), 18):
            painter.drawLine(0, y, self.width(), y + 8)
        painter.setPen(QPen(QColor(157, 214, 255, 36), 1.2))
        path = QPainterPath()
        path.moveTo(self.width() * 0.48, self.height() * 0.75)
        path.cubicTo(self.width() * 0.62, self.height() * 0.2, self.width() * 0.78, self.height() * 0.88, self.width() * 1.02, self.height() * 0.22)
        painter.drawPath(path)


class MapGeneralizationGraphic(QWidget):
    """Lightweight SVG-like map generalization illustration."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(118)
        self.setMaximumHeight(146)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._phase = 0
        self._timer = QTimer(self)
        self._timer.setInterval(90)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    def _tick(self):
        self._phase = (self._phase + 1) % 80
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        pad = 10
        bg = self.rect().adjusted(3, 3, -3, -3)
        y_offset = math.sin(self._phase / 80 * math.tau) * 1.4
        bg.translate(0, int(y_offset))

        painter.setPen(QPen(QColor("#D5E3F2"), 1))
        painter.setBrush(QBrush(QColor(255, 255, 255, 216)))
        painter.drawRoundedRect(bg, 10, 10)

        painter.setClipRect(bg.adjusted(1, 1, -1, -1))
        painter.setPen(QPen(QColor(31, 90, 166, 22), 1))
        grid_step = 22
        for x in range(pad, w, grid_step):
            painter.drawLine(x, bg.top(), x - 18, bg.bottom())
        for y in range(bg.top() + 12, bg.bottom(), grid_step):
            painter.drawLine(bg.left(), y, bg.right(), y - 8)

        painter.setPen(QPen(QColor("#86A7C9"), 1.3))
        for offset in (0, 13, -14):
            path = QPainterPath()
            path.moveTo(bg.left() + 18, bg.center().y() + offset)
            path.cubicTo(w * 0.28, bg.top() + 20 + offset, w * 0.42, bg.bottom() - 18 + offset, w * 0.58, bg.center().y() - 8 + offset)
            path.cubicTo(w * 0.72, bg.top() + 22 + offset, w * 0.86, bg.bottom() - 26 + offset, bg.right() - 12, bg.center().y() + offset)
            painter.drawPath(path)

        left = QRectF(bg.left() + 16, bg.top() + 18, max(84, w * 0.28), bg.height() - 38)
        right = QRectF(bg.right() - max(86, w * 0.30) - 16, bg.top() + 18, max(86, w * 0.30), bg.height() - 38)
        for rect, color in [(left, QColor(238, 245, 251, 232)), (right, QColor(236, 253, 243, 225))]:
            painter.setPen(QPen(QColor("#D5E3F2"), 1))
            painter.setBrush(QBrush(color))
            painter.drawRoundedRect(rect, 7, 7)

        painter.setPen(QPen(QColor("#90AFCF"), 1))
        road = QPainterPath()
        road.moveTo(left.left() + 8, left.bottom() - 14)
        road.cubicTo(left.center().x() - 12, left.top() + 10, left.center().x() + 22, left.bottom() - 18, left.right() - 8, left.top() + 18)
        painter.drawPath(road)

        before_points = [(0.22, 0.33), (0.36, 0.27), (0.42, 0.48), (0.28, 0.62), (0.62, 0.36), (0.72, 0.55), (0.55, 0.70), (0.80, 0.28)]
        after_points = [(0.26, 0.36), (0.48, 0.48), (0.68, 0.34), (0.62, 0.66), (0.82, 0.54)]
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor("#2563EB")))
        for px, py in before_points:
            painter.drawEllipse(QPointF(left.left() + px * left.width(), left.top() + py * left.height()), 2.6, 2.6)
        painter.setBrush(QBrush(QColor("#22C55E")))
        for px, py in after_points:
            painter.drawEllipse(QPointF(right.left() + px * right.width(), right.top() + py * right.height()), 3.0, 3.0)

        painter.setClipping(False)
        painter.setPen(QPen(QColor("#1F5AA6"), 1.6))
        mid_y = bg.center().y()
        painter.drawLine(left.right() + 10, mid_y, right.left() - 18, mid_y)
        arrow = QPolygonF([
            QPointF(right.left() - 18, mid_y),
            QPointF(right.left() - 25, mid_y - 4),
            QPointF(right.left() - 25, mid_y + 4),
        ])
        painter.setBrush(QBrush(QColor("#1F5AA6")))
        painter.drawPolygon(arrow)

        painter.setPen(QPen(QColor("#5D6F86"), 1))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(QRectF(bg.center().x() - 24, mid_y - 13, 48, 26), 8, 8)
        painter.setPen(QPen(QColor("#F59E0B"), 1.5))
        painter.drawText(QRectF(bg.center().x() - 21, mid_y - 9, 42, 18), Qt.AlignmentFlag.AlignCenter, "1:n")


class WorkflowCommandCard(QFrame):
    command_requested = Signal(str)

    def __init__(self, icon_key: str, title_key: str, command_key: str, parent=None):
        super().__init__(parent)
        self.icon_key = icon_key
        self.title_key = title_key
        self.command_key = command_key
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMinimumHeight(58)
        self._hover = False
        self.setObjectName("workflowCard")
        self.setStyleSheet("""
            QFrame#workflowCard {
                background: rgba(255,255,255,0.92);
                border: 1px solid #D5E3F2;
                border-radius: 8px;
            }
            QFrame#workflowCard:hover {
                background: #FFFFFF;
                border: 1px solid #9DC6F5;
            }
        """)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(9)
        self.icon = WorkflowIcon(icon_key)
        layout.addWidget(self.icon)
        text_col = QVBoxLayout()
        text_col.setContentsMargins(0, 0, 0, 0)
        text_col.setSpacing(2)
        self.title_label = QLabel()
        self.title_label.setStyleSheet("font-size: 11px; font-weight: 700; color: #174A7C;")
        self.command_label = QLabel()
        self.command_label.setStyleSheet("font-size: 12px; color: #0B1F33;")
        self.command_label.setWordWrap(True)
        text_col.addWidget(self.title_label)
        text_col.addWidget(self.command_label)
        layout.addLayout(text_col, 1)
        self.retranslate_ui()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.command_requested.emit(self.command_label.text())
        super().mousePressEvent(event)

    def enterEvent(self, event):
        self.move(self.x(), self.y() - 1)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.move(self.x(), self.y() + 1)
        super().leaveEvent(event)

    def retranslate_ui(self):
        self.title_label.setText(tr(self.title_key))
        self.command_label.setText(tr(self.command_key))


class WorkflowIcon(QWidget):
    def __init__(self, icon_key: str, parent=None):
        super().__init__(parent)
        self.icon_key = icon_key
        self.setFixedSize(28, 28)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(QColor("#B8CDE5"), 1))
        painter.setBrush(QBrush(QColor("#EEF5FB")))
        painter.drawRoundedRect(self.rect().adjusted(1, 1, -1, -1), 7, 7)
        painter.setPen(QPen(QColor("#2563EB"), 1.6))
        if self.icon_key == "scale":
            painter.drawLine(7, 18, 21, 18)
            for x in (8, 13, 18):
                painter.drawLine(x, 15, x, 18)
            painter.drawLine(8, 10, 20, 10)
        elif self.icon_key == "operator":
            pts = [(8, 9), (20, 9), (14, 20)]
            for x, y in pts:
                painter.drawEllipse(x - 2, y - 2, 4, 4)
            painter.drawLine(10, 9, 18, 9)
            painter.drawLine(9, 11, 13, 18)
            painter.drawLine(19, 11, 15, 18)
        elif self.icon_key == "quality":
            path = QPainterPath()
            path.moveTo(14, 6)
            path.lineTo(21, 9)
            path.lineTo(19, 18)
            path.lineTo(14, 22)
            path.lineTo(9, 18)
            path.lineTo(7, 9)
            path.closeSubpath()
            painter.drawPath(path)
            painter.drawLine(11, 14, 13, 17)
            painter.drawLine(13, 17, 18, 11)
        else:
            painter.drawRect(9, 7, 10, 14)
            painter.drawLine(11, 12, 17, 12)
            painter.drawLine(11, 16, 17, 16)
            painter.drawLine(18, 18, 22, 18)
            painter.drawLine(20, 16, 22, 18)
            painter.drawLine(20, 20, 22, 18)


class VisualOverviewBanner(QFrame):
    """Hero banner for the map-generalization overview image."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._source_pixmap = QPixmap()
        self._fallback = MapGeneralizationGraphic()
        self._fallback.setMinimumHeight(118)
        self._fallback.setMaximumHeight(148)
        self.setFrameStyle(QFrame.Shape.NoFrame)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setObjectName("visualOverviewBanner")
        self.setStyleSheet("""
            QFrame#visualOverviewBanner {
                background: rgba(255,255,255,0.94);
                border: 1px solid #D5E3F2;
                border-radius: 12px;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 9, 10, 10)
        layout.setSpacing(7)

        title_row = QHBoxLayout()
        title_row.setContentsMargins(2, 0, 2, 0)
        title_row.setSpacing(6)
        self.title = QLabel()
        self.title.setStyleSheet("font-size: 11px; font-weight: 800; color: #174A7C;")
        title_row.addWidget(self.title)
        title_row.addStretch()
        self.badge = QLabel("Input -> AI -> Output")
        self.badge.setStyleSheet("""
            QLabel {
                background: #F1F7FE;
                color: #5D6F86;
                border: 1px solid #D5E3F2;
                border-radius: 8px;
                padding: 2px 7px;
                font-size: 9px;
                font-weight: 700;
            }
        """)
        title_row.addWidget(self.badge)
        layout.addLayout(title_row)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumHeight(112)
        self.image_label.setMaximumHeight(168)
        self.image_label.setStyleSheet("""
            QLabel {
                background: #F8FBFE;
                border: 1px solid #E2ECF7;
                border-radius: 10px;
            }
        """)
        layout.addWidget(self.image_label)
        layout.addWidget(self._fallback)
        self._load_pixmap()
        self.retranslate_ui()

    @staticmethod
    def asset_path():
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        return os.path.join(root, "resources", "ai_assistant", "map_generalization_overview.png")

    def _load_pixmap(self):
        path = self.asset_path()
        if os.path.exists(path):
            self._source_pixmap = QPixmap(path)
        self._sync_visual()

    def _sync_visual(self):
        has_image = not self._source_pixmap.isNull()
        self.image_label.setVisible(has_image)
        self._fallback.setVisible(not has_image)
        if not has_image:
            return
        target = self.image_label.contentsRect().size()
        if target.width() <= 8 or target.height() <= 8:
            return
        scaled = self._source_pixmap.scaled(
            target,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.refresh_layout()

    def refresh_layout(self):
        width = max(260, self.width() - 28)
        height = max(104, min(168, int(width * 0.26)))
        self.image_label.setFixedHeight(height)
        self._fallback.setFixedHeight(height)
        self._sync_visual()

    def retranslate_ui(self):
        self.title.setText(tr("ai.visual_overview"))


class WelcomeMessage(QFrame):
    """欢迎消息 - 居中显示"""
    command_requested = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.Shape.NoFrame)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 14, 10, 14)
        layout.setSpacing(10)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        hero = QFrame()
        hero.setObjectName("assistantHero")
        hero.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        hero.setStyleSheet("""
            QFrame#assistantHero {
                background: rgba(255,255,255,0.88);
                border: 1px solid #D5E3F2;
                border-radius: 10px;
            }
        """)
        hero_layout = QVBoxLayout(hero)
        hero_layout.setContentsMargins(14, 13, 14, 13)
        hero_layout.setSpacing(8)

        self.mark = QLabel("MapGenAI")
        self.mark.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.mark.setStyleSheet("""
            QLabel {
                color: #174A7C;
                background: #EEF5FB;
                border: 1px solid #D5E3F2;
                border-radius: 10px;
                padding: 3px 10px;
                font-size: 11px;
                font-weight: 800;
                letter-spacing: 0px;
            }
        """)
        hero_layout.addWidget(self.mark, alignment=Qt.AlignmentFlag.AlignCenter)

        # 标题
        self.title = QLabel()
        self.title.setStyleSheet("font-size: 17px; font-weight: 800; color: #0B1F33; margin-top: 1px;")
        self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hero_layout.addWidget(self.title)

        # 描述
        self.desc = QLabel()
        self.desc.setStyleSheet("font-size: 12px; color: #5D6F86; line-height: 1.45;")
        self.desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.desc.setWordWrap(True)
        hero_layout.addWidget(self.desc)

        self.capability_tag = QLabel()
        self.capability_tag.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.capability_tag.setStyleSheet("""
            QLabel {
                color: #0F3D68;
                background: #F6FAFD;
                border: 1px solid #D5E3F2;
                border-radius: 9px;
                padding: 4px 8px;
                font-size: 10px;
                font-weight: 700;
            }
        """)
        hero_layout.addWidget(self.capability_tag, alignment=Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(hero)

        self.visual_banner = VisualOverviewBanner()
        layout.addWidget(self.visual_banner)

        # 示例提示
        examples_frame = QFrame()
        examples_frame.setObjectName("workflowPanel")
        examples_frame.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        examples_frame.setStyleSheet("""
            QFrame#workflowPanel {
                background: rgba(255,255,255,0.88);
                border: 1px solid #d7e2ee;
                border-radius: 10px;
            }
        """)
        examples_layout = QVBoxLayout(examples_frame)
        examples_layout.setContentsMargins(12, 11, 12, 12)
        examples_layout.setSpacing(8)

        self.examples_title = QLabel()
        self.examples_title.setStyleSheet("font-size: 12px; color: #174A7C; font-weight: 800; margin-bottom: 2px;")
        examples_layout.addWidget(self.examples_title)

        self.workflow_cards = []
        card_specs = [
            ("scale", "ai.workflow.scale_title", "ai.workflow.scale_command"),
            ("operator", "ai.workflow.operator_title", "ai.workflow.operator_command"),
            ("quality", "ai.workflow.quality_title", "ai.workflow.quality_command"),
            ("export", "ai.workflow.export_title", "ai.workflow.export_command"),
        ]
        for icon_key, title_key, command_key in card_specs:
            card = WorkflowCommandCard(icon_key, title_key, command_key)
            card.command_requested.connect(self.command_requested.emit)
            examples_layout.addWidget(card)
            self.workflow_cards.append(card)

        layout.addWidget(examples_frame)

        flow_frame = QFrame()
        flow_frame.setObjectName("agentFlow")
        flow_frame.setStyleSheet("""
            QFrame#agentFlow {
                background: rgba(246,250,253,0.74);
                border: 1px solid #D5E3F2;
                border-radius: 10px;
            }
        """)
        flow_layout = QVBoxLayout(flow_frame)
        flow_layout.setContentsMargins(10, 8, 10, 8)
        flow_layout.setSpacing(6)
        self.flow_labels = []
        flow_row = QHBoxLayout()
        flow_row.setContentsMargins(0, 0, 0, 0)
        flow_row.setSpacing(4)
        for idx, key in enumerate(["ai.flow.input", "ai.flow.scale", "ai.flow.operator", "ai.flow.skill", "ai.flow.quality", "ai.flow.output"]):
            chip = QLabel()
            chip.setAlignment(Qt.AlignmentFlag.AlignCenter)
            chip.setWordWrap(True)
            chip.setStyleSheet("""
                QLabel {
                    background: #FFFFFF;
                    border: 1px solid #D5E3F2;
                    border-radius: 8px;
                    color: #5D6F86;
                    padding: 3px 5px;
                    font-size: 9px;
                    font-weight: 700;
                }
            """)
            flow_row.addWidget(chip, 1)
            self.flow_labels.append((chip, key))
            if idx < 5:
                arrow = QLabel("›")
                arrow.setAlignment(Qt.AlignmentFlag.AlignCenter)
                arrow.setStyleSheet("color: #7A93AE; font-size: 12px; font-weight: 700;")
                flow_row.addWidget(arrow)
        flow_layout.addLayout(flow_row)
        layout.addWidget(flow_frame)
        self.retranslate_ui()

    def retranslate_ui(self):
        self.title.setText(tr("ai.welcome_title"))
        self.desc.setText(tr("ai.welcome_desc"))
        self.capability_tag.setText(tr("ai.capability_tags"))
        self.examples_title.setText(tr("ai.common_commands"))
        for card in self.workflow_cards:
            card.retranslate_ui()
        for chip, key in self.flow_labels:
            chip.setText(tr(key))
        self.visual_banner.retranslate_ui()

    def refresh_layout(self):
        self.desc.setMaximumWidth(max(260, self.width() - 48))
        self.visual_banner.refresh_layout()


# 向后兼容别名
ChatMessage = ChatBubble


from PySide6.QtWidgets import QSizePolicy
from PySide6.QtGui import QTextCursor


class AIWorkerThread(QThread):
    """AI响应工作线程 - 升级版，支持Agent事件"""
    event_signal = Signal(str, object)  # type, data
    finished_signal = Signal()
    error_signal = Signal(str)

    def __init__(self, agent, user_message, parent=None):
        super().__init__(parent)
        self.agent = agent
        self.user_message = user_message

    def run(self):
        try:
            # 定义回调函数，桥接Agent事件到Qt信号
            def on_event(event_type, data):
                self.event_signal.emit(event_type, data)
            
            # 设置回调
            self.agent.set_event_callback(on_event)
            
            # 执行对话 (阻塞调用，但内部会触发回调)
            self.agent.chat(self.user_message)
            
            self.finished_signal.emit()
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error_signal.emit(str(e))


class _AgentUiBridge(QObject):
    """Marshal agent UI callbacks to the main UI thread."""
    load_layer_signal = Signal(str, object)
    zoom_signal = Signal(str)
    style_signal = Signal(str, object)

    def __init__(self, main_window):
        super().__init__(main_window)
        self.main_window = main_window
        self.load_layer_signal.connect(self._on_load_layer, Qt.ConnectionType.QueuedConnection)
        self.zoom_signal.connect(self._on_zoom, Qt.ConnectionType.QueuedConnection)
        self.style_signal.connect(self._on_style, Qt.ConnectionType.QueuedConnection)

    def post_load_layer(self, file_path, layer_name=None):
        self.load_layer_signal.emit(str(file_path), layer_name)

    def post_zoom(self, layer_name):
        self.zoom_signal.emit(str(layer_name))

    def post_style(self, layer_name, style):
        self.style_signal.emit(str(layer_name), style if isinstance(style, dict) else {})

    def _on_load_layer(self, file_path, layer_name):
        self.main_window.load_vector_layer(file_path, layer_name)

    def _on_zoom(self, layer_name):
        self.main_window.map_canvas.zoom_to_layer(layer_name)

    def _on_style(self, layer_name, style):
        self.main_window.map_canvas.update_layer_style(layer_name, style)


class AIAssistantPanel(QWidget):
    """AI助手面板 - 现代化设计，类似主流AI IDE"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.agent = None
        self.messages = []
        self.current_ai_bubble = None
        self.current_thinking_block = None
        self.current_tool_blocks = {}
        self.worker = None
        self.generated_files = []
        self.has_messages = False  # 是否有对话消息
        self._tool_started_at = {}
        self._pending_response_text = ""
        self._response_cursor = 0
        self._pending_files_after_stream = []
        self._delayed_final_response = None
        self._thinking_queue = []
        self._thinking_active_text = ""
        self._thinking_active_cursor = 0
        self.thinking_heartbeat = QTimer(self)
        self.thinking_heartbeat.setInterval(900)
        self.thinking_heartbeat.timeout.connect(self._on_thinking_heartbeat)
        self.thinking_stream_timer = QTimer(self)
        self.thinking_stream_timer.setInterval(28)
        self.thinking_stream_timer.timeout.connect(self._on_thinking_stream_tick)
        self.response_stream_timer = QTimer(self)
        self.response_stream_timer.setInterval(24)
        self.response_stream_timer.timeout.connect(self._on_response_stream_tick)
        self.setup_ui()
        self.load_agent()

    def _ensure_thinking_block(self):
        if not self.current_thinking_block:
            self.current_thinking_block = ThinkingBlock()
            self.chat_layout.addWidget(self.current_thinking_block)
        if not self.thinking_heartbeat.isActive():
            self.thinking_heartbeat.start()

    def _stop_thinking_block(self):
        if hasattr(self, "thinking_heartbeat") and self.thinking_heartbeat.isActive():
            self.thinking_heartbeat.stop()
        if hasattr(self, "thinking_stream_timer") and self.thinking_stream_timer.isActive():
            self.thinking_stream_timer.stop()

    def _queue_thinking_text(self, text):
        msg = str(text or "").strip()
        if not msg:
            return
        self._ensure_thinking_block()
        msg = self.current_thinking_block._normalize_text(msg)
        self._thinking_queue.append(msg)
        if not self.thinking_stream_timer.isActive():
            self.thinking_stream_timer.start()

    def _flush_thinking_queue(self):
        if self.current_thinking_block and self._thinking_active_text:
            remaining = self._thinking_active_text[self._thinking_active_cursor:]
            if remaining:
                self.current_thinking_block.append_stream_fragment(remaining)
            self.current_thinking_block.finalize_stream_line()
        self._thinking_active_text = ""
        self._thinking_active_cursor = 0
        while self._thinking_queue and self.current_thinking_block:
            self.current_thinking_block.append_thinking(self._thinking_queue.pop(0))
        if self.thinking_stream_timer.isActive():
            self.thinking_stream_timer.stop()
        self._start_delayed_response_if_ready()

    def _on_thinking_stream_tick(self):
        if not self.current_thinking_block:
            self.thinking_stream_timer.stop()
            self._start_delayed_response_if_ready()
            return
        if not self._thinking_active_text:
            if not self._thinking_queue:
                self.thinking_stream_timer.stop()
                self._start_delayed_response_if_ready()
                return
            self._thinking_active_text = self._thinking_queue.pop(0)
            self._thinking_active_cursor = 0
        step = 2
        nxt = min(len(self._thinking_active_text), self._thinking_active_cursor + step)
        frag = self._thinking_active_text[self._thinking_active_cursor:nxt]
        self._thinking_active_cursor = nxt
        self.current_thinking_block.append_stream_fragment(frag)
        if self._thinking_active_cursor >= len(self._thinking_active_text):
            self.current_thinking_block.finalize_stream_line()
            self._thinking_active_text = ""
            self._thinking_active_cursor = 0
            self._start_delayed_response_if_ready()
        self.scroll_to_bottom()

    def _start_delayed_response_if_ready(self):
        if not self._delayed_final_response:
            return
        if self._thinking_queue or self._thinking_active_text:
            return
        self._remove_layout_spacers()
        raw_response, summary_widget = self._delayed_final_response
        self._delayed_final_response = None
        if self.current_thinking_block:
            self.current_thinking_block.finish()
            self._stop_thinking_block()
            self.current_thinking_block = None
        if summary_widget:
            self.chat_layout.addWidget(summary_widget)
        self._start_stream_response(self._build_display_response(raw_response))
        self.current_tool_blocks = {}
        self._ensure_bottom_stretch()
        self.scroll_to_bottom()

    def _on_thinking_heartbeat(self):
        if self.current_thinking_block and not self._thinking_queue and not self._thinking_active_text:
            self.current_thinking_block.pulse()
            self.scroll_to_bottom()

    def load_agent(self):
        """加载AI Agent"""
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "config.yaml")

        try:
            if os.path.exists(config_path):
                with open(config_path, "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f) or {}

                    if config.get("ai", {}).get("api_key"):
                        from src.agent.core import create_agent
                        self.agent = create_agent(config)
                        self.update_status()
        except Exception as e:
            print(f"加载AI配置失败: {e}")

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.setStyleSheet("""
            AIAssistantPanel {
                background: #EEF5FB;
                color: #0B1F33;
            }
        """)

        # 标题栏
        header = QFrame()
        header.setObjectName("assistantHeader")
        header.setStyleSheet("""
            QFrame#assistantHeader {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #0F3D68, stop:0.48 #174A7C, stop:1 #1F5AA6);
                border-bottom: 1px solid rgba(13,47,82,0.72);
            }
        """)
        header.setMinimumHeight(72)
        header_texture = HeaderMapTexture(header)
        header_texture.setGeometry(header.rect())
        header_texture.lower()
        header.resizeEvent = lambda event, tex=header_texture: (tex.setGeometry(header.rect()), QFrame.resizeEvent(header, event))

        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(16, 12, 14, 12)
        header_layout.setSpacing(11)

        self.header_icon = MapAssistantIcon()
        header_layout.addWidget(self.header_icon)

        title_stack = QVBoxLayout()
        title_stack.setContentsMargins(0, 0, 0, 0)
        title_stack.setSpacing(1)
        self.title_label = QLabel()
        self.title_label.setStyleSheet("color: white; font-weight: 800; font-size: 16px; letter-spacing: 0px;")
        self.subtitle_label = QLabel()
        self.subtitle_label.setStyleSheet("color: rgba(219,239,255,0.80); font-size: 10px; font-weight: 600;")
        title_stack.addWidget(self.title_label)
        title_stack.addWidget(self.subtitle_label)
        header_layout.addLayout(title_stack, 1)

        header_layout.addStretch()

        self.status_pill = QFrame()
        self.status_pill.setObjectName("statusPill")
        self.status_pill.setStyleSheet("""
            QFrame#statusPill {
                background: rgba(255,255,255,0.14);
                border: 1px solid rgba(255,255,255,0.24);
                border-radius: 12px;
            }
        """)
        status_layout = QHBoxLayout(self.status_pill)
        status_layout.setContentsMargins(8, 4, 9, 4)
        status_layout.setSpacing(6)
        self.status_indicator = QFrame()
        self.status_indicator.setFixedSize(8, 8)
        status_layout.addWidget(self.status_indicator)

        self.status_label = QLabel()
        self.status_label.setStyleSheet("color: rgba(255,255,255,0.94); font-size: 11px; font-weight: 700;")
        status_layout.addWidget(self.status_label)
        header_layout.addWidget(self.status_pill)

        # 清空按钮
        self.clear_btn = QPushButton()
        self.clear_btn.setStyleSheet("""
            QPushButton {
                background: rgba(255,255,255,0.10);
                color: white;
                border: 1px solid rgba(255,255,255,0.28);
                border-radius: 8px;
                padding: 6px 10px;
                font-size: 11px;
                font-weight: 700;
            }
            QPushButton:hover {
                background: rgba(255,255,255,0.18);
                border: 1px solid rgba(255,255,255,0.42);
            }
        """)
        self.clear_btn.clicked.connect(self.clear_chat)
        header_layout.addWidget(self.clear_btn)

        layout.addWidget(header)

        # 聊天区域
        self.chat_scroll = QScrollArea()
        self.chat_scroll.setWidgetResizable(True)
        self.chat_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.chat_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.chat_scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #EEF5FB, stop:1 #F6FAFD);
            }
            QScrollBar:vertical {
                background: transparent;
                width: 12px;
                margin: 6px 3px 6px 2px;
            }
            QScrollBar::handle:vertical {
                background: #bdc9d8;
                border-radius: 4px;
                min-height: 36px;
            }
            QScrollBar::handle:vertical:hover {
                background: #94a3b8;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background: transparent;
            }
        """)

        self.chat_container = QWidget()
        self.chat_container.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #EEF5FB, stop:1 #F6FAFD);
            }
        """)
        self.chat_layout = QVBoxLayout(self.chat_container)
        self.chat_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.chat_layout.setSpacing(7)
        self.chat_layout.setContentsMargins(10, 12, 16, 12)

        # 欢迎消息
        self.welcome_widget = WelcomeMessage()
        self.welcome_widget.command_requested.connect(self._submit_suggested_command)
        self.chat_layout.addWidget(self.welcome_widget)

        # 底部弹性空间
        self.chat_layout.addStretch()

        self.chat_scroll.setWidget(self.chat_container)
        layout.addWidget(self.chat_scroll, 1)

        # 输入区域
        input_frame = QFrame()
        input_frame.setObjectName("assistantInputArea")
        input_frame.setStyleSheet("""
            QFrame#assistantInputArea {
                background: rgba(246,250,253,0.96);
                border-top: 1px solid #D5E3F2;
            }
        """)
        input_layout = QVBoxLayout(input_frame)
        input_layout.setContentsMargins(12, 10, 12, 12)
        input_layout.setSpacing(8)

        # 输入框容器
        input_container = QFrame()
        input_container.setObjectName("inputContainer")
        input_container.setStyleSheet("""
            QFrame#inputContainer {
                background: #FFFFFF;
                border: 1px solid #D5E3F2;
                border-radius: 12px;
            }
            QFrame#inputContainer:focus-within {
                border-color: #2563eb;
                background: white;
            }
        """)
        input_container_layout = QHBoxLayout(input_container)
        input_container_layout.setContentsMargins(13, 10, 10, 10)
        input_container_layout.setSpacing(10)

        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText(tr("ai.input_placeholder"))
        self.input_field.setStyleSheet("""
            QLineEdit {
                border: none;
                background: transparent;
                font-size: 14px;
                color: #0B1F33;
                min-height: 26px;
            }
            QLineEdit::placeholder {
                color: #7A93AE;
            }
        """)
        self.input_field.returnPressed.connect(self.send_message)

        self.send_btn = QPushButton("→")
        self.send_btn.setFixedSize(36, 36)
        self.send_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #2F6DF6, stop:1 #174A7C);
                color: white;
                border: none;
                border-radius: 18px;
                font-size: 18px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #3B82F6, stop:1 #1F5AA6);
            }
            QPushButton:pressed {
                background: #1e40af;
            }
            QPushButton:disabled {
                background: #bdbdbd;
            }
        """)
        self.send_btn.clicked.connect(self.send_message)

        input_container_layout.addWidget(self.input_field, 1)
        input_container_layout.addWidget(self.send_btn)

        input_layout.addWidget(input_container)

        layout.addWidget(input_frame)
        self.retranslate_ui()

    def retranslate_ui(self):
        self.title_label.setText(tr("ai.panel_title"))
        self.subtitle_label.setText(tr("ai.copilot_subtitle"))
        self.clear_btn.setText(tr("ai.clear"))
        self.input_field.setPlaceholderText(tr("ai.input_placeholder"))
        if self.welcome_widget:
            self.welcome_widget.retranslate_ui()
        self.update_status()
        self.refresh_chat_layout()

    def _submit_suggested_command(self, command):
        command = str(command or "").strip()
        if not command or not self.input_field.isEnabled():
            return
        self.input_field.setText(command)
        self.send_message()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.refresh_chat_layout()

    def _remove_layout_spacers(self):
        if not hasattr(self, "chat_layout"):
            return
        for i in range(self.chat_layout.count() - 1, -1, -1):
            item = self.chat_layout.itemAt(i)
            if item and item.spacerItem():
                self.chat_layout.takeAt(i)

    def _ensure_bottom_stretch(self):
        if not hasattr(self, "chat_layout"):
            return
        last = self.chat_layout.itemAt(self.chat_layout.count() - 1)
        if last and last.spacerItem():
            return
        self.chat_layout.addStretch()

    def _wrap_timeline_widget(self, inner_widget):
        wrapper = QFrame()
        wrapper.setFrameStyle(QFrame.Shape.NoFrame)
        wrapper.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        wrapper_layout = QHBoxLayout(wrapper)
        wrapper_layout.setContentsMargins(6, 4, 6, 4)
        wrapper_layout.setSpacing(0)
        wrapper_layout.addWidget(inner_widget, 1)
        return wrapper

    def refresh_chat_layout(self):
        if not hasattr(self, "chat_layout"):
            return
        for i in range(self.chat_layout.count()):
            item = self.chat_layout.itemAt(i)
            widget = item.widget() if item else None
            if widget and hasattr(widget, "refresh_layout"):
                widget.refresh_layout()

    def update_status(self):
        """更新连接状态"""
        if self.agent:
            self.status_label.setText(tr("ai.ready"))
            self.status_indicator.setStyleSheet("""
                QFrame {
                    background: #22c55e;
                    border: 1px solid rgba(255,255,255,0.86);
                    border-radius: 4px;
                }
            """)
        else:
            self.status_label.setText(tr("ai.not_configured"))
            self.status_indicator.setStyleSheet("""
                QFrame {
                    background: #F59E0B;
                    border: 1px solid rgba(255,255,255,0.70);
                    border-radius: 4px;
                }
            """)

    def clear_chat(self):
        """清空聊天记录"""
        # 移除所有消息
        while self.chat_layout.count() > 0:
            item = self.chat_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # 重新添加欢迎消息
        self.welcome_widget = WelcomeMessage()
        self.chat_layout.addWidget(self.welcome_widget)
        self.chat_layout.addStretch()

        self.has_messages = False
        self.messages = []
        self.current_ai_bubble = None
        self.current_thinking_block = None
        self.current_tool_blocks = {}

    def send_message(self):
        text = self.input_field.text().strip()
        if not text:
            return

        # 隐藏欢迎消息
        if not self.has_messages:
            self.welcome_widget.hide()
            self.has_messages = True

        # 移除底部stretch
        for i in range(self.chat_layout.count() - 1, -1, -1):
            item = self.chat_layout.itemAt(i)
            if item.spacerItem():
                self.chat_layout.removeItem(item)
                break

        # 添加用户消息
        user_bubble = ChatBubble(text, is_user=True)
        self.chat_layout.addWidget(user_bubble)
        self.messages.append({"role": "user", "content": text})

        self.input_field.clear()

        # 禁用输入
        self.input_field.setEnabled(False)
        self.send_btn.setEnabled(False)

        # 添加底部stretch
        self.chat_layout.addStretch()

        if self.agent:
            # 启动工作线程
            self.worker = AIWorkerThread(self.agent, text)
            self.worker.event_signal.connect(self.on_event_received, Qt.ConnectionType.QueuedConnection)
            self.worker.finished_signal.connect(self.on_response_finished, Qt.ConnectionType.QueuedConnection)
            self.worker.error_signal.connect(self.on_response_error, Qt.ConnectionType.QueuedConnection)
            self.worker.start()
        else:
            self.add_fallback_response(text)

        self.scroll_to_bottom()

    def on_event_received(self, event_type, data):
        """处理Agent事件"""

        # 移除底部stretch以便添加新内容
        for i in range(self.chat_layout.count() - 1, -1, -1):
            item = self.chat_layout.itemAt(i)
            if item.spacerItem():
                self.chat_layout.removeItem(item)
                break

        if event_type == "thinking":
            # 显示思考状态
            self._queue_thinking_text(str(data))

        elif event_type == "thought":
            # 思维链内容
            self._queue_thinking_text(f"💭 {data}")

        elif event_type == "tool_start":
            # 显示工具调用
            tool_name = data.get("name", "Unknown Tool")
            args = data.get("args", {}) if isinstance(data, dict) else {}
            tool_block = ToolCallBlock(tool_name)
            self.current_tool_blocks[tool_name] = tool_block
            self._tool_started_at[tool_name] = time.perf_counter()
            self.chat_layout.addWidget(tool_block)

            if self.current_thinking_block:
                target_text = self._summarize_tool_target(args)
                step = self.current_thinking_block._fmt_tool_step("tool_call", tool_name, target_text)
                self.current_thinking_block.add_workflow_step(step)
                code_text = args.get("code") if isinstance(args, dict) else ""
                if isinstance(code_text, str) and code_text.strip():
                    self.current_thinking_block.add_code_snippet(code_text)

        elif event_type == "tool_end":
            # 工具执行完成
            tool_name = data.get("name", "")
            result = data.get("result", {})

            if tool_name in self.current_tool_blocks:
                started_for_block = self._tool_started_at.get(tool_name)
                elapsed_for_block = (time.perf_counter() - started_for_block) if started_for_block else None
                self.current_tool_blocks[tool_name].set_complete(True, elapsed_for_block)

            if self.current_thinking_block:
                started = self._tool_started_at.pop(tool_name, None)
                elapsed = (time.perf_counter() - started) if started else None
                out_text = self._summarize_result_target(result)
                step = self.current_thinking_block._fmt_tool_step("tool_done_step", tool_name, out_text)
                self.current_thinking_block.add_workflow_step(step, elapsed_sec=elapsed)

            # 检测文件生成
            if isinstance(result, dict):
                for file_path in self._extract_output_files_from_result(result):
                    if file_path not in self.generated_files:
                        self.generated_files.append(file_path)

        elif event_type == "response":
            # 最终回复
            raw_response = str(data)
            summary_widget = self._create_result_summary_widget(raw_response)
            self._delayed_final_response = (raw_response, summary_widget)
            self._start_delayed_response_if_ready()

        # 重新添加底部stretch
        self.chat_layout.addStretch()
        self.scroll_to_bottom()

    def _start_stream_response(self, text):
        self._pending_response_text = str(text or "")
        self._response_cursor = 0
        self._pending_files_after_stream = list(self.generated_files)
        self.generated_files = []

        # 若上一次流式仍在进行，先结束
        if self.response_stream_timer.isActive():
            self.response_stream_timer.stop()

        self.current_ai_bubble = ChatBubble("", is_user=False)
        self.chat_layout.addWidget(self.current_ai_bubble)
        self.response_stream_timer.start()

    def _on_response_stream_tick(self):
        if not self.current_ai_bubble:
            self.response_stream_timer.stop()
            return
        if self._response_cursor >= len(self._pending_response_text):
            self.response_stream_timer.stop()
            full_text = self._pending_response_text
            self.messages.append({"role": "assistant", "content": full_text})

            # 流式结束后再显示文件列表
            if self._pending_files_after_stream:
                self._remove_layout_spacers()
                files_widget = self._create_files_widget(self._pending_files_after_stream)
                self.chat_layout.addWidget(files_widget)
                self._pending_files_after_stream = []
            self._ensure_bottom_stretch()
            self.scroll_to_bottom()
            return

        step = 4
        nxt = min(len(self._pending_response_text), self._response_cursor + step)
        frag = self._pending_response_text[self._response_cursor:nxt]
        self._response_cursor = nxt
        self.current_ai_bubble.append_text(frag)
        self.scroll_to_bottom()

    def _summarize_tool_target(self, args):
        if not isinstance(args, dict):
            return ""
        candidates = []
        for k in ["input_file", "path", "file_path", "source", "layer_name"]:
            v = args.get(k)
            if isinstance(v, str) and v.strip():
                candidates.append(v.strip())
        if not candidates and isinstance(args.get("code"), str):
            code = args.get("code", "")
            m = re.search(r"['\"]([^'\"]+\.(?:shp|geojson|gpkg|json))['\"]", code, flags=re.I)
            if m:
                candidates.append(m.group(1))
        if not candidates:
            return ""
        name = os.path.basename(candidates[0])
        return name or candidates[0]

    def _summarize_result_target(self, result):
        if not isinstance(result, dict):
            return ""
        for key in ["output_path", "file_path", "path"]:
            v = result.get(key)
            if isinstance(v, str) and v.strip():
                return os.path.basename(v.strip()) or v.strip()
        files = result.get("files")
        if isinstance(files, list) and files:
            first = files[0]
            if isinstance(first, str):
                return os.path.basename(first) or first
        stdout = result.get("stdout")
        if isinstance(stdout, str):
            m = re.search(r"([\\w\\-\\u4e00-\\u9fff\\.]+\\.(?:shp|geojson|gpkg|json))", stdout, flags=re.I)
            if m:
                return m.group(1)
        return ""

    def _extract_output_files_from_result(self, result):
        files = []
        if not isinstance(result, dict):
            return files
        for key in ["output_path", "file_path", "path"]:
            v = result.get(key)
            if isinstance(v, str) and v.strip():
                files.append(v.strip())
        listed = result.get("files")
        if isinstance(listed, list):
            for item in listed:
                if isinstance(item, str) and item.strip():
                    files.append(item.strip())
        events = result.get("events")
        if isinstance(events, list):
            for event in events:
                if isinstance(event, dict) and event.get("type") == "show_layer":
                    p = event.get("path")
                    if isinstance(p, str) and p.strip():
                        files.append(p.strip())
        stdout = result.get("stdout")
        if isinstance(stdout, str):
            patterns = [
                r"output\s*[:=]\s*([^\s]+?\.(?:shp|geojson|gpkg|png|jpg|jpeg))",
                r"saved\s*[:=]\s*([^\s]+?\.(?:shp|geojson|gpkg|png|jpg|jpeg))",
                r"result\s*[:=]\s*([^\s]+?\.(?:shp|geojson|gpkg|png|jpg|jpeg))",
            ]
            for pattern in patterns:
                files.extend(re.findall(pattern, stdout, flags=re.I))
        deduped = []
        seen = set()
        for p in files:
            if p not in seen:
                deduped.append(p)
                seen.add(p)
        return deduped

    def _create_files_widget(self, files):
        """创建文件列表组件"""
        widget = QFrame()
        widget.setObjectName("generatedFilesCard")
        widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        widget.setStyleSheet("""
            QFrame#generatedFilesCard {
                background: #F8FBFE;
                border: 1px solid #D5E3F2;
                border-radius: 12px;
            }
        """)
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(12, 10, 12, 12)
        layout.setSpacing(8)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(8)
        title = QLabel(tr("ai.generated_files"))
        title.setStyleSheet("font-weight: 800; color: #174A7C; font-size: 12px;")
        header.addWidget(title)
        header.addStretch()
        count = QLabel(str(len(files)))
        count.setStyleSheet("""
            QLabel {
                background: #EAF3FF;
                color: #2563EB;
                border: 1px solid #C7DDF7;
                border-radius: 8px;
                padding: 2px 7px;
                font-size: 10px;
                font-weight: 800;
            }
        """)
        header.addWidget(count)
        layout.addLayout(header)

        list_host = QWidget()
        list_layout = QVBoxLayout(list_host)
        list_layout.setContentsMargins(0, 0, 0, 0)
        list_layout.setSpacing(6)
        for file_path in files:
            list_layout.addWidget(GeneratedFileItem(file_path))

        if len(files) > 3:
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.Shape.NoFrame)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            scroll.setMaximumHeight(260)
            scroll.setStyleSheet("""
                QScrollArea { border: none; background: transparent; }
                QScrollBar:vertical { background: transparent; width: 8px; }
                QScrollBar::handle:vertical { background: #CBD5E1; border-radius: 4px; min-height: 28px; }
                QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
                QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }
            """)
            scroll.setWidget(list_host)
            layout.addWidget(scroll)
        else:
            layout.addWidget(list_host)

        return self._wrap_timeline_widget(widget)

    def _create_result_summary_widget(self, response_text):
        summary = self._parse_response_summary(response_text)
        if not summary:
            return None
        widget = QFrame()
        widget.setObjectName("resultSummaryCard")
        widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        widget.setStyleSheet("""
            QFrame#resultSummaryCard {
                background: #FFFFFF;
                border: 1px solid #D5E3F2;
                border-radius: 12px;
            }
        """)
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(12, 11, 12, 12)
        layout.setSpacing(8)

        title = QLabel(tr("ai.result_summary"))
        title.setStyleSheet("font-weight: 800; color: #174A7C; font-size: 12px;")
        layout.addWidget(title)

        basis_info = self._read_basis_doc_details(summary.get("basis_doc", ""))
        overview_lines = []
        if summary.get("scale"):
            overview_lines.append(f"目标比例尺: {summary['scale']}")
        overview_lines.append("完成状态: 已完成")
        output_count = self._read_output_feature_count(summary.get("files") or [])
        if output_count is not None:
            overview_lines.append(f"输出要素: {output_count} 个")
        if summary.get("runtime"):
            overview_lines.append(f"处理耗时: {summary['runtime']}")
        self._add_summary_section(layout, tr("ai.summary_overview"), overview_lines, "#0d47a1")

        data_lines = []
        if basis_info.get("input"):
            data_lines.append(f"数据源: {basis_info['input']}")
        if basis_info.get("feature_count") or basis_info.get("geometry_type"):
            data_lines.append(
                f"输入数据: {self._humanize_geometry_type(basis_info.get('geometry_type', '未知类型'))}要素，"
                f"{basis_info.get('feature_count', '未知')} 个要素"
            )
        crs_note = self._build_crs_note(basis_info.get("source_crs", ""), summary.get("crs") or basis_info.get("projected_crs", ""))
        if crs_note:
            data_lines.append(crs_note)
        self._add_summary_section(layout, tr("ai.summary_data"), data_lines, "#1e3a5f")

        method_lines = []
        if basis_info.get("semantic_type"):
            method_lines.append(f"语义识别: {self._humanize_semantic_type(basis_info['semantic_type'])}")
        if basis_info.get("structure_type"):
            method_lines.append(f"空间结构: {self._humanize_structure_type(basis_info['structure_type'])}")
        if summary.get("basis"):
            method_lines.append(f"综合策略: {summary['basis']}")
        if summary.get("failed_rules"):
            method_lines.append(f"当前未满足项: {self._translate_failed_rules(summary['failed_rules'])}")
        self._add_summary_section(layout, tr("ai.summary_method"), method_lines, "#1e3a5f")

        operator_lines = []
        for item in basis_info.get("operators", [])[:5]:
            operator_lines.append(self._format_operator_label(item))
        self._add_summary_section(layout, tr("ai.summary_operators"), operator_lines, "#1e3a5f")

        basis_doc = summary.get("basis_doc", "")
        if basis_doc and os.path.exists(basis_doc):
            brief = self._read_basis_doc_brief(basis_doc)
            self._add_summary_section(layout, tr("ai.summary_notes"), brief, "#1e3a5f")
        return self._wrap_timeline_widget(widget)

    def _parse_response_summary(self, text):
        t = str(text or "")
        if "Completion status" not in t and "Output files" not in t and "Generalization" not in t:
            return {}
        out = {}
        m_scale = re.search(r"(?:Generalization .*? at|目标比例尺[:：]?)\\s*(1:\\d+)", t, flags=re.I)
        if m_scale:
            out["scale"] = m_scale.group(1)
        m_iteration = re.search(r"Iteration:\s*([^\r\n]+)", t, flags=re.I)
        if m_iteration:
            out["iteration"] = m_iteration.group(1).strip()
        m_warning_iter = re.search(r"after\s*(\d+)\s*iterations?", t, flags=re.I)
        if m_warning_iter and not out.get("iteration"):
            out["iteration"] = f"共 {m_warning_iter.group(1)} 轮"
        m_status = re.search(r"Completion status:\\s*([^\\r\\n]+)", t, flags=re.I)
        if m_status:
            out["status"] = m_status.group(1).strip()
        m_verify = re.search(r"VLM verify:\\s*([^\\r\\n]+)", t, flags=re.I)
        if m_verify:
            out["verify"] = m_verify.group(1).strip()
        m_verify_fail = re.search(r"VLM verify still failed\s*\(score=([^)]+)\)", t, flags=re.I)
        if m_verify_fail:
            out["verify"] = f"未通过（评分 {m_verify_fail.group(1).strip()}）"
        m_crs = re.search(r"Projected CRS used:\\s*([^\\r\\n]+)", t, flags=re.I)
        if m_crs:
            out["crs"] = m_crs.group(1).strip()
        m_basis = re.search(r"Selection basis:\s*([^\r\n]+)", t, flags=re.I)
        if m_basis:
            out["basis"] = m_basis.group(1).strip()
        if not out.get("basis"):
            m_dec = re.search(r"【决策摘要[^】]*】[^\r\n]*", t)
            if m_dec:
                out["basis"] = m_dec.group(0).strip()
        m_failed = re.search(r"Failed rules:\\s*([^\\r\\n]+)", t, flags=re.I)
        if m_failed:
            out["failed_rules"] = m_failed.group(1).strip()
        m_runtime = re.search(r"Runtime:\\s*([^\\r\\n]+)", t, flags=re.I)
        if m_runtime:
            out["runtime"] = m_runtime.group(1).strip()
        m_files = re.search(r"Output files:\\s*([^\\r\\n]+)", t, flags=re.I)
        if m_files:
            files = [x.strip() for x in m_files.group(1).split(",") if x.strip()]
            out["files"] = files
            for p in files:
                if p.lower().endswith("_selection_basis.md"):
                    out["basis_doc"] = p
                    break
        return out

    def _read_basis_doc_brief(self, path):
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                lines = [ln.strip() for ln in f.readlines() if ln.strip()]
            picks = []
            for ln in lines:
                if ln.startswith("- semantic_type:") or ln.startswith("- structure_type:"):
                    item = ln[2:].strip()
                    if item.startswith("semantic_type:"):
                        value = item.split(":", 1)[1].strip().strip("`")
                        picks.append(f"语义类型: {self._humanize_semantic_type(value)}")
                    elif item.startswith("structure_type:"):
                        value = item.split(":", 1)[1].strip().strip("`")
                        picks.append(f"空间结构: {self._humanize_structure_type(value)}")
                if ln.startswith("- source_crs:") or ln.startswith("- projected_crs:"):
                    picks.append(ln[2:].strip())
                if ln.startswith("- `"):
                    picks.append("算子 " + self._format_operator_label(ln[2:].strip()))
                if len(picks) >= 4:
                    break
            return picks
        except Exception:
            return []

    def _add_summary_section(self, layout, title, lines, color):
        clean_lines = [str(x).strip() for x in (lines or []) if str(x).strip()]
        if not clean_lines:
            return
        section = QFrame()
        section.setObjectName("summarySection")
        section.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        section.setStyleSheet("""
            QFrame#summarySection {
                background: #F8FBFE;
                border: 1px solid #E2ECF7;
                border-radius: 9px;
            }
        """)
        section_layout = QVBoxLayout(section)
        section_layout.setContentsMargins(10, 8, 10, 9)
        section_layout.setSpacing(5)
        head = QLabel(title)
        head.setStyleSheet("font-weight: 800; color: #2563EB; font-size: 11px;")
        section_layout.addWidget(head)
        for item in clean_lines:
            lb = AutoHeightTextEdit(f"• {item}", min_height=20, max_height=120)
            lb.setStyleSheet(f"""
                font-family: 'Segoe UI', 'Microsoft YaHei UI', 'Microsoft YaHei', Arial, sans-serif;
                color: {color};
                font-size: 11px;
                line-height: 1.45;
                background: transparent;
                border: none;
            """)
            section_layout.addWidget(lb)
        layout.addWidget(section)

    def _read_basis_doc_details(self, path):
        details = {}
        if not path or not os.path.exists(path):
            return details
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                lines = [ln.strip() for ln in f.readlines() if ln.strip()]
            operators = []
            for ln in lines:
                if ln.startswith("- input:"):
                    details["input"] = self._strip_md_code(ln.split(":", 1)[1].strip())
                elif ln.startswith("- feature_count:"):
                    details["feature_count"] = self._strip_md_code(ln.split(":", 1)[1].strip())
                elif ln.startswith("- geometry_type:"):
                    details["geometry_type"] = self._strip_md_code(ln.split(":", 1)[1].strip())
                elif ln.startswith("- source_crs:"):
                    details["source_crs"] = self._strip_md_code(ln.split(":", 1)[1].strip())
                elif ln.startswith("- projected_crs:"):
                    details["projected_crs"] = self._strip_md_code(ln.split(":", 1)[1].strip())
                elif ln.startswith("- semantic_type:"):
                    details["semantic_type"] = self._strip_md_code(ln.split(":", 1)[1].strip())
                elif ln.startswith("- structure_type:"):
                    details["structure_type"] = self._strip_md_code(ln.split(":", 1)[1].strip())
                elif ln.startswith("- `"):
                    operators.append(ln[2:].strip())
            if operators:
                details["operators"] = operators
        except Exception:
            return {}
        return details

    def _strip_md_code(self, text):
        return str(text or "").strip().strip("`")

    def _humanize_semantic_type(self, value):
        text = str(value or "").strip()
        mapping = {
            "generic_point": "通用点要素",
            "transport_network_points": "交通网络相关点",
            "public_service_points": "公共服务设施点",
            "monitoring_points": "监测事件点",
            "commercial_poi_points": "商业兴趣点",
            "加油站/能源补给点": "加油站或能源补给点",
            "餐饮服务点": "餐饮服务点",
            "公共服务设施点": "公共服务设施点",
            "交通网络相关点": "交通网络相关点",
            "监测事件点": "监测事件点",
            "商业兴趣点": "商业兴趣点",
            "通用点要素": "通用点要素",
        }
        return mapping.get(text, text or "未知点类型")

    def _humanize_structure_type(self, value):
        text = str(value or "").strip()
        mapping = {
            "unknown": "未知结构",
            "corridor": "走廊型分布",
            "clustered": "集聚型分布",
            "uniform": "均匀型分布",
            "mixed": "混合型分布",
            "multi-core": "多核心分布",
            "center-periphery": "中心-边缘分布",
            "走廊型": "走廊型分布",
            "集聚型": "集聚型分布",
            "均匀型": "均匀型分布",
            "混合型": "混合型分布",
            "多核心型": "多核心分布",
            "中心-边缘型": "中心-边缘分布",
        }
        return mapping.get(text, text or "未知结构")

    def _humanize_geometry_type(self, value):
        text = str(value or "").strip().lower()
        mapping = {
            "point": "点",
            "multipoint": "点",
            "line": "线",
            "linestring": "线",
            "multilinestring": "线",
            "polygon": "面",
            "multipolygon": "面",
        }
        return mapping.get(text, str(value or "未知类型"))

    def _humanize_operator_name(self, value):
        text = str(value or "").strip()
        mapping = {
            "selection": "选择保留",
            "simplify": "简化",
            "typification": "类型化保留",
            "displacement": "位移避让",
            "aggregation": "聚合",
            "aggregate": "聚合",
            "merge": "合并",
            "smooth": "平滑",
            "buffer": "缓冲",
        }
        return mapping.get(text, text)

    def _localize_status(self, status):
        value = str(status or "").strip()
        mapping = {
            "completed": "已完成",
            "completed_with_warnings": "已完成，但仍有待优化项",
            "failed": "未通过验收",
        }
        return mapping.get(value, value)

    def _localize_verify(self, verify):
        text = str(verify or "").strip()
        text = text.replace("pass", "通过").replace("failed", "未通过").replace("score", "评分")
        text = text.replace("VLM verify", "验收检查")
        text = text.replace("still failed", "仍未通过")
        return text

    def _build_crs_note(self, source_crs, projected_crs):
        src = str(source_crs or "").strip()
        dst = str(projected_crs or "").strip()
        if not src and not dst:
            return ""
        if src.lower() in {"undefined", "none", "n/a"}:
            if dst and dst.upper() != "N/A":
                return f"坐标系处理: 原始数据未记录坐标系，结果阶段使用 {dst}"
            return "坐标系处理: 原始数据未记录坐标系"
        if dst and dst.upper() != "N/A" and src != dst:
            return f"坐标系处理: 数据源原始坐标系为 {src}，计算时自动转换为 {dst}"
        return f"坐标系处理: 数据源坐标系为 {src}"

    def _describe_output_files(self, files):
        visible_files = [p for p in (files or []) if p and not p.lower().endswith("_selection_basis.md")]
        if not visible_files:
            return ""

        geo_files = [p for p in visible_files if p.lower().endswith(".geojson")]
        png_files = [p for p in visible_files if p.lower().endswith(".png")]
        other_files = [p for p in visible_files if p not in geo_files and p not in png_files]

        clauses = []
        if geo_files:
            geo_names = "、".join(os.path.basename(p) for p in geo_files[:2])
            if len(geo_files) > 2:
                geo_names += " 等文件"
            clauses.append(f"空间数据成果已输出为 {len(geo_files)} 个 GeoJSON 文件，如 {geo_names}")
        if png_files:
            png_names = "、".join(os.path.basename(p) for p in png_files[:2])
            if len(png_files) > 2:
                png_names += " 等文件"
            clauses.append(f"同时生成 {len(png_files)} 张结果预览图，如 {png_names}")
        if other_files:
            clauses.append(f"另有 {len(other_files)} 个辅助结果文件可供复核")

        if not clauses:
            return ""
        return "；".join(clauses) + "。"

    def _read_output_feature_count(self, files):
        for path in (files or []):
            file_path = str(path or "").strip()
            if not file_path.lower().endswith(".geojson") or not os.path.exists(file_path):
                continue
            try:
                with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                    payload = json.load(f)
                features = payload.get("features", [])
                if isinstance(features, list):
                    return len(features)
            except Exception:
                continue
        return None

    def _build_strategy_summary(self, basis_info):
        parts = []
        semantic_type = basis_info.get("semantic_type")
        structure_type = basis_info.get("structure_type")
        operators = basis_info.get("operators", [])
        if semantic_type:
            parts.append(f"针对{self._humanize_semantic_type(semantic_type)}")
        if structure_type:
            parts.append(f"结合{self._humanize_structure_type(structure_type)}特征")
        if operators:
            op_names = "、".join(self._format_operator_label(x) for x in operators[:4])
            parts.append(f"采用 {op_names} 的组合策略")
        if not parts:
            return "本次处理采用了与当前数据特征相匹配的综合策略。"
        return "，".join(parts) + "。"

    def _build_display_response(self, raw_text):
        summary = self._parse_response_summary(raw_text)
        if not summary:
            return raw_text
        parts = []
        basis_info = self._read_basis_doc_details(summary.get("basis_doc", ""))
        files = summary.get("files") or []
        output_count = self._read_output_feature_count(files)

        intro = "本次处理结果为已完成。"
        if summary.get("scale"):
            intro += f"\n本次面向 {summary['scale']} 的目标比例尺完成了综合处理。"
        if output_count is not None:
            intro += f"\n最终输出结果中保留了 {output_count} 个要素。"
        parts.append(intro)

        strategy_text = self._build_strategy_summary(basis_info)
        if strategy_text:
            parts.append("处理说明\n" + strategy_text)

        if files:
            output_sentence = self._describe_output_files(files)
            out_lines = []
            if output_sentence:
                out_lines.append(output_sentence)
            out_lines.append("下方文件列表给出了各成果文件的完整路径，可直接打开或继续用于后续处理。")
            parts.append("输出结果\n" + "\n".join([f"• {x}" for x in out_lines]))
        return "\n".join(parts)

    def _format_operator_label(self, text):
        item = str(text or "").strip()
        m = re.search(r"`([^`]+)`", item)
        name = m.group(1) if m else item
        params = ""
        m_params = re.search(r"params=(.+)$", item)
        if m_params:
            params = m_params.group(1).strip()
        name = self._humanize_operator_name(name)
        if params:
            return f"{name}，参数 {params}"
        return name

    def on_response_finished(self):
        """响应完成"""
        if self.current_thinking_block:
            self._flush_thinking_queue()
            if self.current_thinking_block:
                self.current_thinking_block.finish()
        self._stop_thinking_block()
        self.current_thinking_block = None
        self.current_tool_blocks = {}

        self.input_field.setEnabled(True)
        self.send_btn.setEnabled(True)
        self.input_field.setFocus()

    def on_response_error(self, error):
        """响应错误"""
        # 移除底部stretch
        for i in range(self.chat_layout.count() - 1, -1, -1):
            item = self.chat_layout.itemAt(i)
            if item.spacerItem():
                self.chat_layout.removeItem(item)
                break

        if self.current_thinking_block:
            self._flush_thinking_queue()
            self.current_thinking_block.set_status(tr("ai.error_status"), "✗")
        self._stop_thinking_block()

        # 显示错误消息
        error_bubble = ChatBubble(tr("ai.error_message", error=error), is_user=False)
        self.chat_layout.addWidget(error_bubble)

        self.chat_layout.addStretch()

        self.input_field.setEnabled(True)
        self.send_btn.setEnabled(True)
        self.scroll_to_bottom()

    def add_fallback_response(self, user_text):
        """无AI配置时的回退响应"""
        # 移除底部stretch
        for i in range(self.chat_layout.count() - 1, -1, -1):
            item = self.chat_layout.itemAt(i)
            if item.spacerItem():
                self.chat_layout.removeItem(item)
                break

        response = tr("ai.fallback")
        ai_bubble = ChatBubble(response, is_user=False)
        self.chat_layout.addWidget(ai_bubble)

        self.chat_layout.addStretch()

        self.input_field.setEnabled(True)
        self.send_btn.setEnabled(True)
        self.scroll_to_bottom()

    def scroll_to_bottom(self):
        QApplication.processEvents()
        QTimer.singleShot(50, lambda: self.chat_scroll.verticalScrollBar().setValue(
            self.chat_scroll.verticalScrollBar().maximum()
        ))

    def reload_provider(self):
        self.agent = None
        self.load_agent()


class MainWindow(QMainWindow):
    """主窗口"""
    def __init__(self):
        super().__init__()
        self._config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "config.yaml",
        )
        self._runtime_config = self._load_runtime_config()
        self._language = set_language(self._runtime_config.get("language"))
        self._active_layer_snapshot = {}
        self._layout_export_request = {}
        self._layout_export_result = ""
        self.setWindowTitle(tr("app.title"))
        self.setMinimumSize(1200, 800)
        self.setup_ui()
        self.setup_menu()
        self.setup_toolbar()
        self.setup_statusbar()
        if not self._runtime_config.get("python_env", {}).get("validated"):
            QTimer.singleShot(100, self.open_env_config_dialog)
        self._agent_ui_bridge = _AgentUiBridge(self)
        
        # 注册UI回调给Agent工具
        self.register_agent_callbacks()

    def setup_toolbar(self):
        self.toolbar = QToolBar()
        self.toolbar.setIconSize(QSize(20, 20))
        self.addToolBar(self.toolbar)
        
        # 打开
        self.open_act = QAction(self)
        self.open_act.setShortcut("Ctrl+O")
        self.open_act.triggered.connect(self.import_data)
        self.toolbar.addAction(self.open_act)
        
        self.toolbar.addSeparator()
        
        # 缩放
        self.zoom_in_act = QAction(self)
        self.zoom_in_act.triggered.connect(self.zoom_in)
        self.toolbar.addAction(self.zoom_in_act)
        
        self.zoom_out_act = QAction(self)
        self.zoom_out_act.triggered.connect(self.zoom_out)
        self.toolbar.addAction(self.zoom_out_act)
        
        self.zoom_home_act = QAction(self)
        self.zoom_home_act.triggered.connect(self.zoom_home)
        self.toolbar.addAction(self.zoom_home_act)
        
        self.toolbar.addSeparator()
        
        # AI助手
        self.ai_act = QAction(self)
        self.ai_act.triggered.connect(self.show_ai_panel)
        self.toolbar.addAction(self.ai_act)
        self.retranslate_toolbar()

    def retranslate_toolbar(self):
        self.open_act.setText(tr("action.open"))
        self.zoom_in_act.setText(tr("action.zoom_in"))
        self.zoom_out_act.setText(tr("action.zoom_out"))
        self.zoom_home_act.setText(tr("action.zoom_home"))
        self.ai_act.setText(tr("action.show_ai"))

    def setup_statusbar(self):
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage(tr("common.ready"))
        self.env_status_widget = EnvStatusWidget(self)
        self.env_status_widget.config_requested.connect(self.open_env_config_dialog)
        self.statusBar().addPermanentWidget(self.env_status_widget)
        self.refresh_env_status_widget(self._runtime_config)

    def _load_runtime_config(self):
        if not os.path.exists(self._config_path):
            return {}
        try:
            with open(self._config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception:
            return {}

    def refresh_env_status_widget(self, config=None):
        if config is None:
            config = self._load_runtime_config()
        if not hasattr(self, "env_status_widget"):
            return
        py_status = config.get("python_env", {}) if isinstance(config, dict) else {}
        qgis_status = config.get("qgis_env", {}) if isinstance(config, dict) else {}
        self.env_status_widget.update_status(py_status, qgis_status)

    def open_env_config_dialog(self):
        self._runtime_config = self._load_runtime_config()
        dialog = EnvConfigDialog(self, config=self._runtime_config, config_path=self._config_path)
        dialog.env_config_changed.connect(self.on_env_config_changed)
        dialog.exec()

    def on_env_config_changed(self, config):
        self._runtime_config = config or {}
        self._runtime_config["language"] = self._language
        self.refresh_env_status_widget(self._runtime_config)

    def _save_runtime_config(self):
        with open(self._config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self._runtime_config, f, allow_unicode=True, sort_keys=False)

    def set_app_language(self, language):
        normalized = set_language(language)
        self._language = normalized
        self._runtime_config = self._load_runtime_config()
        self._runtime_config["language"] = normalized
        self._save_runtime_config()
        self.retranslate_ui()

    def retranslate_ui(self):
        self.setWindowTitle(tr("app.title"))
        self._layer_tab_label = tr("menu.layer")
        self.retranslate_menu()
        self.retranslate_toolbar()
        if hasattr(self, "workspace_panel"):
            self.workspace_panel.retranslate_ui()
        if hasattr(self, "layer_panel"):
            self.layer_panel.retranslate_ui()
        if hasattr(self, "left_tabs"):
            index = self.left_tabs.indexOf(self.layer_panel)
            if index != -1:
                self.left_tabs.setTabText(index, tr("menu.layer"))
        if hasattr(self, "map_canvas") and hasattr(self.map_canvas, "retranslate_ui"):
            self.map_canvas.retranslate_ui()
        if hasattr(self, "ai_panel"):
            self.ai_panel.retranslate_ui()
        if hasattr(self, "env_status_widget"):
            self.env_status_widget.retranslate_ui()
        self.statusBar().showMessage(tr("common.ready"))

    def register_agent_callbacks(self):
        """注册Agent工具回调"""
        register_ui_callback("load_layer", self._agent_ui_bridge.post_load_layer)
        register_ui_callback("zoom_to_layer", self._agent_ui_bridge.post_zoom)
        register_ui_callback("set_layer_style", self._agent_ui_bridge.post_style)
        register_ui_callback("get_active_layer_info", self.get_active_layer_info)
        register_ui_callback("capture_map_screenshot", self.capture_map_screenshot)
        register_ui_callback("export_map_layout", self.export_map_layout_for_agent)
        
        # 将工作区路径同步给Agent (如果有)
        # 注意：Agent是在Panel中初始化的，这里只是注册全局工具回调
        # 如果需要更紧密的集成，可以将Agent提升到MainWindow级别

    @Slot()
    def _refresh_active_layer_snapshot(self) -> None:
        """在主线程根据图层面板当前选中项更新快照（供 Agent 在后台线程安全读取）。"""
        name = self.layer_panel.get_selected_layer_name()
        if not name and self.layer_panel.layer_order:
            name = self.layer_panel.layer_order[-1]
        if not name:
            self._active_layer_snapshot = {}
            return
        layer_data = self.layer_panel.layers.get(name, {})
        if not isinstance(layer_data, dict):
            self._active_layer_snapshot = {}
            return
        self._active_layer_snapshot = {
            "name": name,
            "path": layer_data.get("path", ""),
            "geom_type": layer_data.get("geom_type", ""),
            "feature_count": layer_data.get("feature_count", 0),
            "crs": layer_data.get("crs", ""),
        }

    @Slot()
    def _export_layout_snapshot(self) -> None:
        req = dict(getattr(self, "_layout_export_request", {}) or {})
        path = str(req.get("path") or "")
        try:
            if not path:
                path = os.path.join(os.getcwd(), f"map_layout_{int(time.time())}.png")
            self.map_canvas.export_layout(
                path,
                page_size=str(req.get("page_size") or "A4"),
                orientation=str(req.get("orientation") or "landscape"),
                dpi=int(req.get("dpi") or 220),
                title=str(req.get("title") or "POI 点综合验收图幅"),
                margin_mm=float(req.get("margin_mm") or 12.0),
            )
            self._layout_export_result = path
        except Exception:
            self._layout_export_result = ""

    def _try_auto_compare_after_generalized_load(self, file_path: str, loaded_layer_name: str) -> None:
        """综合结果 GeoJSON（* _gen_1_<scale>_r<n>.geojson）加载后自动开对比：左综合前、右综合后并拟合视图。"""
        base = os.path.basename(file_path)
        m = re.match(r"^(.+?)_gen_1_\d+_r\d+\.geojson$", base, flags=re.I)
        if not m:
            return
        stem = m.group(1)
        layers = self.layer_panel.layers
        before: str | None = None
        if stem in layers:
            before = stem
        elif "points" in layers and loaded_layer_name != "points":
            before = "points"
        else:
            for ln in self.layer_panel.layer_order:
                if ln == loaded_layer_name:
                    continue
                if "_gen_1_" in ln:
                    continue
                gt = (layers.get(ln) or {}).get("geom_type", "")
                if gt in ("Point", "MultiPoint"):
                    before = ln
                    break
        if before:
            self.map_canvas.set_compare_mode(True, before, loaded_layer_name)
            QTimer.singleShot(0, lambda ln=loaded_layer_name: self._auto_resolve_compare_after_visible_conflicts(ln))

    def get_active_layer_info(self):
        """给 Agent 返回当前活动图层元数据。"""
        if QThread.currentThread() is not self.thread():
            try:
                QMetaObject.invokeMethod(
                    self,
                    "_refresh_active_layer_snapshot",
                    Qt.ConnectionType.BlockingQueuedConnection,
                )
            except Exception:
                pass
            return dict(self._active_layer_snapshot)

        self._refresh_active_layer_snapshot()
        return dict(self._active_layer_snapshot)

    def capture_map_screenshot(self, path=None):
        """给 Agent 导出固定图幅地图图像，而不是不稳定的屏幕截图。"""
        if QThread.currentThread() is not self.thread():
            if not path:
                path = os.path.join(os.getcwd(), f"map_capture_{int(time.time())}.png")
            root, ext = os.path.splitext(str(path))
            if ext.lower() not in {".png", ".pdf"}:
                path = root + ".png"
            self._layout_export_request = {
                "path": str(path),
                "page_size": "A4",
                "orientation": "landscape",
                "dpi": 220,
                "title": "POI 点综合验收图幅",
                "margin_mm": 12.0,
            }
            self._layout_export_result = ""
            try:
                QMetaObject.invokeMethod(self, "_export_layout_snapshot", Qt.ConnectionType.BlockingQueuedConnection)
            except Exception:
                return ""
            return str(self._layout_export_result or "")
        try:
            if not path:
                import time

                path = os.path.join(os.getcwd(), f"map_capture_{int(time.time())}.png")
            root, ext = os.path.splitext(path)
            if ext.lower() not in {".png", ".pdf"}:
                path = root + ".png"
            self.map_canvas.export_layout(path, page_size="A4", orientation="landscape", dpi=220, title="POI 点综合验收图幅")
            return path
        except Exception:
            return ""

    def export_map_layout_for_agent(self, path=None, page_size="A4", orientation="landscape", dpi=300, title="POI 点综合验收图幅"):
        """Agent/VLM 使用的固定图幅导出回调。"""
        if QThread.currentThread() is not self.thread():
            if not path:
                path = os.path.join(os.getcwd(), f"map_layout_{int(time.time())}.pdf")
            self._layout_export_request = {
                "path": str(path),
                "page_size": str(page_size or "A4"),
                "orientation": str(orientation or "landscape"),
                "dpi": int(dpi or 300),
                "title": str(title or ""),
                "margin_mm": 12.0,
            }
            self._layout_export_result = ""
            try:
                QMetaObject.invokeMethod(self, "_export_layout_snapshot", Qt.ConnectionType.BlockingQueuedConnection)
            except Exception:
                return ""
            return str(self._layout_export_result or "")
        try:
            if not path:
                import time

                path = os.path.join(os.getcwd(), f"map_layout_{int(time.time())}.pdf")
            self.map_canvas.export_layout(
                path,
                page_size=str(page_size or "A4"),
                orientation=str(orientation or "landscape"),
                dpi=int(dpi or 300),
                title=str(title or ""),
            )
            return path
        except Exception:
            return ""

    def setup_ui(self):
        # ... (Same setup_ui as before) ...
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self._workspace_splitter_sizes = None
        self._layer_tab_label = tr("menu.layer")

        # 左侧面板
        left_panel = QWidget()
        self.left_panel = left_panel
        self._left_panel_min_width = 250
        self._left_panel_max_width = 350
        self._main_splitter_sizes_before_left_collapse = None

        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)

        self.workspace_panel = WorkspacePanel()
        self.workspace_panel.file_double_clicked.connect(self.on_file_double_clicked)
        self.workspace_panel.panel_closed.connect(self.hide_workspace_panel)

        self.left_tabs = QTabWidget()
        self.left_tabs.setStyleSheet("""
            QTabWidget::pane { border: none; }
            QTabBar::tab { padding: 6px 14px; background: #e0e0e0; }
            QTabBar::tab:selected { background: #1976d2; color: white; }
        """)

        self.layer_panel = LayerPanel()
        self.left_tabs.addTab(self.layer_panel, self._layer_tab_label)
        self.layer_panel.panel_closed.connect(self.hide_layer_panel)

        self.left_splitter = QSplitter(Qt.Orientation.Vertical)
        self.left_splitter.addWidget(self.workspace_panel)
        self.left_splitter.addWidget(self.left_tabs)
        self.left_splitter.setSizes([300, 400])

        left_layout.addWidget(self.left_splitter)
        left_panel.setMinimumWidth(self._left_panel_min_width)
        left_panel.setMaximumWidth(self._left_panel_max_width)

        # 中间地图区域
        from src.ui.native_map_canvas import NativeMapCanvas
        self.map_canvas = NativeMapCanvas()

        # 右侧AI助手
        self.ai_panel = AIAssistantPanel()
        self.ai_panel.setMinimumWidth(360)
        self.ai_panel.setMaximumWidth(460)

        self.main_splitter.addWidget(left_panel)
        self.main_splitter.addWidget(self.map_canvas)
        self.main_splitter.addWidget(self.ai_panel)
        self.main_splitter.setSizes([280, 560, 420])

        main_layout.addWidget(self.main_splitter)

        # 连接图层面板信号到地图画布
        self.layer_panel.layer_removed.connect(self.on_layer_removed)
        self.layer_panel.layer_visibility_changed.connect(self.on_layer_visibility_changed)
        self.layer_panel.layer_moved_up.connect(self.on_layer_moved_up)
        self.layer_panel.layer_moved_down.connect(self.on_layer_moved_down)
        self.layer_panel.layer_zoom_requested.connect(self.on_layer_zoom_requested)
        self.layer_panel.layer_style_changed.connect(self.on_layer_style_changed)
        self.layer_panel.layer_crs_changed.connect(self.on_layer_crs_changed)
        self.layer_panel.layer_tree.itemSelectionChanged.connect(self._refresh_active_layer_snapshot)
        self.map_canvas.compare_intervention.connect(self._on_compare_intervention)

    @Slot()
    def _persist_layer_geojson_if_possible(self, name: str, geojson: dict) -> None:
        if not name or not isinstance(geojson, dict):
            return
        layer_info = self.layer_panel.layers.get(name, {})
        if not isinstance(layer_info, dict):
            return
        path = str(layer_info.get("path") or "").strip()
        if not path:
            return
        ext = os.path.splitext(path)[1].lower()
        if ext not in {".geojson", ".json"}:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(geojson, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    @Slot()
    def _sync_compare_after_layer(self) -> None:
        try:
            _, _, after_name = self.map_canvas.get_compare_config()
            if not after_name:
                return
            gj = self.map_canvas.get_layer_geojson(after_name)
            if gj is None:
                return
            self.layer_panel.update_layer_geojson(after_name, gj)
            self._persist_layer_geojson_if_possible(after_name, gj)
        except Exception:
            pass

    @Slot()
    def _auto_resolve_compare_after_visible_conflicts(self, expected_after_name: str = "") -> None:
        try:
            enabled, _, after_name = self.map_canvas.get_compare_config()
            if not enabled or not after_name:
                return
            if expected_after_name and after_name != expected_after_name:
                return
            result = self.map_canvas.resolve_visible_point_conflicts(after_name, panel="after")
            if int(result.get("removed_count", 0) or 0) > 0:
                self._sync_compare_after_layer()
                self.statusBar().showMessage(
                    f"已按当前屏幕视图实时去除「{after_name}」中的 {int(result.get('removed_count', 0) or 0)} 个遮挡点"
                )
        except Exception:
            pass

    @Slot()
    def _on_compare_intervention(self, message: str) -> None:
        self.statusBar().showMessage(message)
        self._sync_compare_after_layer()
        self._refresh_active_layer_snapshot()

    # ... (Rest of MainWindow methods: load_vector_layer, export_data, etc. are same as before) ...
    # 为了完整性，这里必须包含这些方法，但我会保留它们不变。
    
    def on_layer_removed(self, name):
        self.map_canvas.remove_layer(name)
        if getattr(self.ai_panel, "agent", None):
            self.ai_panel.agent.remove_layer(name)
        self._refresh_active_layer_snapshot()
        self.statusBar().showMessage(tr("status.layer_deleted", name=name))

    def on_layer_visibility_changed(self, name, visible):
        self.map_canvas.set_layer_visibility(name, visible)

    def on_layer_style_changed(self, name, style):
        self.map_canvas.update_layer_style(name, style)
        self.statusBar().showMessage(tr("status.layer_style_updated", name=name))

    def on_layer_crs_changed(self, name, geojson, target_crs, geom_type, feature_count):
        style = {}
        layer_info = self.layer_panel.layers.get(name, {})
        if isinstance(layer_info, dict):
            style = dict(layer_info.get("style") or {})

        if not style:
            if geom_type in ["Point", "MultiPoint"]:
                style = {"color": "#e91e63", "fillColor": "#e91e63", "markersize": 60}
            elif geom_type in ["LineString", "MultiLineString"]:
                style = {"color": "#2196f3", "linewidth": 2}
            elif geom_type in ["Polygon", "MultiPolygon"]:
                style = {"color": "#4caf50", "fillColor": "#4caf50", "fillOpacity": 0.5}

        self.map_canvas.add_layer(name, geojson, style)

        item = layer_info.get("item") if isinstance(layer_info, dict) else None
        if item:
            visible = item.checkState(0) == Qt.CheckState.Checked
            self.map_canvas.set_layer_visibility(name, visible)

        self.statusBar().showMessage(
            tr("status.layer_crs_changed", name=name, crs=target_crs, geom_type=geom_type, count=feature_count)
        )

    def on_layer_moved_up(self, name):
        self.map_canvas.move_layer_up(name)

    def on_layer_moved_down(self, name):
        self.map_canvas.move_layer_down(name)

    def on_layer_zoom_requested(self, name):
        self.map_canvas.zoom_to_layer(name)

    def on_file_double_clicked(self, path):
        ext = os.path.splitext(path)[1].lower()
        if ext in [".geojson", ".json", ".shp", ".gpkg"]:
            self.load_vector_layer(path)
        else:
            self.statusBar().showMessage(tr("status.file_opened", path=path))

    def _resolve_existing_vector_path(self, file_path):
        path = os.path.abspath(os.path.expanduser(str(file_path or "").strip().strip("\"'")))
        if os.path.exists(path):
            return path
        root, ext = os.path.splitext(path)
        for candidate_ext in (".geojson", ".json", ".gpkg", ".shp"):
            if candidate_ext.lower() == ext.lower():
                continue
            candidate = root + candidate_ext
            if os.path.exists(candidate):
                return candidate
        return path

    def load_vector_layer(self, file_path, layer_name=None): # Updated signature
        try:
            from src.io import read_vector_file, get_geometry_type, get_feature_count, get_center

            original_path = str(file_path or "")
            file_path = self._resolve_existing_vector_path(original_path)
            if not os.path.exists(file_path):
                raise FileNotFoundError(tr("layer.file_not_found", path=original_path or file_path))

            geojson = read_vector_file(file_path)
            name = layer_name or os.path.splitext(os.path.basename(file_path))[0]
            geom_type = get_geometry_type(geojson)
            feature_count = get_feature_count(geojson)
            crs_text = "EPSG:4326"
            gdf_meta = None
            try:
                import geopandas as gpd

                gdf_meta = gpd.read_file(file_path)
                if getattr(gdf_meta, "crs", None):
                    crs_text = str(gdf_meta.crs)
                else:
                    crs_text = "Unknown"
            except Exception:
                gdf_meta = None

            style = {}
            if geom_type in ["Point", "MultiPoint"]:
                style = {"color": "#e91e63", "fillColor": "#e91e63", "markersize": 60}
                feats = geojson.get("features") or []
                if feats:
                    ps0 = (feats[0].get("properties") or {}).get("point_symbol")
                    if ps0:
                        style["point_symbol"] = normalize_point_symbol(str(ps0))
                if gdf_meta is not None and not getattr(gdf_meta, "empty", True):
                    style.update(infer_point_symbol_style_from_gdf(gdf_meta))
            elif geom_type in ["LineString", "MultiLineString"]:
                style = {"color": "#2196f3", "linewidth": 2}
            elif geom_type in ["Polygon", "MultiPolygon"]:
                style = {"color": "#4caf50", "fillColor": "#4caf50", "fillOpacity": 0.5}

            self.layer_panel.add_layer(name, geom_type, feature_count, file_path, geojson, crs=crs_text)
            self.map_canvas.add_layer(name, geojson, style)
            if name in self.layer_panel.layers:
                self.layer_panel.layers[name]["style"] = dict(style)
            if getattr(self.ai_panel, "agent", None):
                self.ai_panel.agent.add_layer(name)
            self.layer_panel.layer_tree.setCurrentItem(self.layer_panel.layers[name]["item"])
            self._refresh_active_layer_snapshot()
            self._try_auto_compare_after_generalized_load(str(file_path), name)
            self.statusBar().showMessage(tr("status.layer_loaded", name=name, geom_type=geom_type, count=feature_count))
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.warning(self, tr("layer.load_failed"), tr("layer.load_failed_message", error=e))

    def hide_workspace_panel(self):
        if not self.workspace_panel.isVisible(): return
        self._workspace_splitter_sizes = self.left_splitter.sizes()
        self.workspace_panel.setVisible(False)
        sizes = self.left_splitter.sizes()
        if len(sizes) == 2: self.left_splitter.setSizes([0, sum(sizes)])
        self._update_left_panel_layout()

    def show_workspace_panel(self):
        if self.workspace_panel.isVisible(): return
        self.workspace_panel.setVisible(True)
        if self._workspace_splitter_sizes: self.left_splitter.setSizes(self._workspace_splitter_sizes)
        else: self.left_splitter.setSizes([300, 400])
        self._update_left_panel_layout()

    def hide_layer_panel(self):
        index = self.left_tabs.indexOf(self.layer_panel)
        if index != -1: self.left_tabs.removeTab(index)
        self._update_left_panel_layout()

    def show_layer_panel(self):
        if self.left_tabs.indexOf(self.layer_panel) == -1:
            self.left_tabs.insertTab(0, self.layer_panel, self._layer_tab_label)
        self.left_tabs.setCurrentWidget(self.layer_panel)
        self._update_left_panel_layout()

    def _update_left_panel_layout(self):
        workspace_visible = self.workspace_panel.isVisible()
        layer_visible = self.left_tabs.indexOf(self.layer_panel) != -1

        if not workspace_visible and not layer_visible:
            if self.left_panel.isVisible():
                self._main_splitter_sizes_before_left_collapse = self.main_splitter.sizes()
            self.left_panel.setMinimumWidth(0)
            self.left_panel.setMaximumWidth(0)
            self.left_panel.setVisible(False)
            sizes = self.main_splitter.sizes()
            if len(sizes) == 3:
                self.main_splitter.setSizes([0, sizes[0] + sizes[1], sizes[2]])
        else:
            if not self.left_panel.isVisible():
                self.left_panel.setVisible(True)
            self.left_panel.setMinimumWidth(self._left_panel_min_width)
            self.left_panel.setMaximumWidth(self._left_panel_max_width)
            if self._main_splitter_sizes_before_left_collapse:
                self.main_splitter.setSizes(self._main_splitter_sizes_before_left_collapse)
                self._main_splitter_sizes_before_left_collapse = None

    def show_ai_panel(self):
        if not self.ai_panel.isVisible(): self.ai_panel.setVisible(True)

    def zoom_in(self): self.map_canvas.zoom_in()
    def zoom_out(self): self.map_canvas.zoom_out()
    def zoom_home(self): self.map_canvas.zoom_home()
    def select_all_layers(self): self.layer_panel.select_all_layers()
    def clear_layer_selection(self): self.layer_panel.clear_selection()
    def undo_action(self): self.statusBar().showMessage(tr("status.no_undo"))
    def redo_action(self): self.statusBar().showMessage(tr("status.no_redo"))
    def show_layer_properties(self): self.layer_panel.show_selected_layer_properties()
    def edit_layer_style(self): self.layer_panel.edit_selected_layer_style()

    def setup_menu(self):
        menubar = self.menuBar()
        menubar.clear()
        
        # 文件菜单
        self.file_menu = menubar.addMenu("")
        self.open_folder_action = QAction(self)
        self.open_folder_action.triggered.connect(self.open_folder)
        self.file_menu.addAction(self.open_folder_action)
        self.import_data_action = QAction(self)
        self.import_data_action.triggered.connect(self.import_data)
        self.file_menu.addAction(self.import_data_action)
        self.file_menu.addSeparator()
        self.exit_action = QAction(self)
        self.exit_action.triggered.connect(self.close)
        self.file_menu.addAction(self.exit_action)
        
        # 编辑菜单
        self.edit_menu = menubar.addMenu("")
        self.select_all_layers_action = QAction(self)
        self.select_all_layers_action.triggered.connect(self.select_all_layers)
        self.edit_menu.addAction(self.select_all_layers_action)
        self.clear_selection_action = QAction(self)
        self.clear_selection_action.triggered.connect(self.clear_layer_selection)
        self.edit_menu.addAction(self.clear_selection_action)
        
        # 视图菜单
        self.view_menu = menubar.addMenu("")
        self.show_workspace_action = QAction(self)
        self.show_workspace_action.triggered.connect(self.show_workspace_panel)
        self.view_menu.addAction(self.show_workspace_action)
        self.show_layer_panel_action = QAction(self)
        self.show_layer_panel_action.triggered.connect(self.show_layer_panel)
        self.view_menu.addAction(self.show_layer_panel_action)
        self.show_ai_action = QAction(self)
        self.show_ai_action.triggered.connect(self.show_ai_panel)
        self.view_menu.addAction(self.show_ai_action)
        self.view_menu.addSeparator()
        self.menu_zoom_in_action = QAction(self)
        self.menu_zoom_in_action.triggered.connect(self.zoom_in)
        self.view_menu.addAction(self.menu_zoom_in_action)
        self.menu_zoom_out_action = QAction(self)
        self.menu_zoom_out_action.triggered.connect(self.zoom_out)
        self.view_menu.addAction(self.menu_zoom_out_action)
        self.menu_zoom_home_action = QAction(self)
        self.menu_zoom_home_action.triggered.connect(self.zoom_home)
        self.view_menu.addAction(self.menu_zoom_home_action)
        
        # 图层菜单
        self.layer_menu = menubar.addMenu("")
        self.layer_properties_action = QAction(self)
        self.layer_properties_action.triggered.connect(self.show_layer_properties)
        self.layer_menu.addAction(self.layer_properties_action)
        self.edit_style_action = QAction(self)
        self.edit_style_action.triggered.connect(self.edit_layer_style)
        self.layer_menu.addAction(self.edit_style_action)
        self.layer_menu.addSeparator()
        self.export_data_action = QAction(self)
        self.export_data_action.triggered.connect(self.export_data)
        self.layer_menu.addAction(self.export_data_action)
        
        # 配置菜单
        self.config_menu = menubar.addMenu("")
        self.env_config_action = QAction(self)
        self.env_config_action.triggered.connect(self.open_env_config_dialog)
        self.config_menu.addAction(self.env_config_action)
        self.ai_settings_action = QAction(self)
        self.ai_settings_action.triggered.connect(self.ai_settings)
        self.config_menu.addAction(self.ai_settings_action)
        self.language_menu = self.config_menu.addMenu("")
        self.language_group = QActionGroup(self)
        self.language_group.setExclusive(True)
        self.language_actions = {}
        for language in SUPPORTED_LANGUAGES:
            action = QAction(LANGUAGE_LABELS.get(language, language), self)
            action.setCheckable(True)
            action.setData(language)
            action.triggered.connect(lambda _checked=False, lang=language: self.set_app_language(lang))
            self.language_group.addAction(action)
            self.language_menu.addAction(action)
            self.language_actions[language] = action
        
        # 帮助菜单
        self.help_menu = menubar.addMenu("")
        self.about_action = QAction(self)
        self.about_action.triggered.connect(self.show_about)
        self.help_menu.addAction(self.about_action)
        self.retranslate_menu()

    def retranslate_menu(self):
        self.file_menu.setTitle(tr("menu.file"))
        self.open_folder_action.setText(tr("action.open_folder"))
        self.import_data_action.setText(tr("action.import_data"))
        self.exit_action.setText(tr("action.exit"))
        self.edit_menu.setTitle(tr("menu.edit"))
        self.select_all_layers_action.setText(tr("action.select_all_layers"))
        self.clear_selection_action.setText(tr("action.clear_selection"))
        self.view_menu.setTitle(tr("menu.view"))
        self.show_workspace_action.setText(tr("action.show_workspace"))
        self.show_layer_panel_action.setText(tr("action.show_layer_panel"))
        self.show_ai_action.setText(tr("action.show_ai"))
        self.menu_zoom_in_action.setText(tr("action.zoom_in"))
        self.menu_zoom_out_action.setText(tr("action.zoom_out"))
        self.menu_zoom_home_action.setText(tr("action.zoom_home"))
        self.layer_menu.setTitle(tr("menu.layer"))
        self.layer_properties_action.setText(tr("action.layer_properties"))
        self.edit_style_action.setText(tr("action.edit_style"))
        self.export_data_action.setText(tr("action.export_data"))
        self.config_menu.setTitle(tr("menu.config"))
        self.env_config_action.setText(tr("action.env_config"))
        self.ai_settings_action.setText(tr("action.ai_settings"))
        self.language_menu.setTitle(f"{tr('menu.language')}: {language_label(self._language)}")
        for language, action in self.language_actions.items():
            action.setText(LANGUAGE_LABELS.get(language, language))
            action.setChecked(language == self._language)
        self.help_menu.setTitle(tr("menu.help"))
        self.about_action.setText(tr("action.about"))

    def open_folder(self):
        path = QFileDialog.getExistingDirectory(self, tr("dialog.choose_workspace"))
        if path:
            self.workspace_panel.set_root(path)
            self.statusBar().showMessage(tr("status.workspace_opened", path=path))
            # Update agent workspace if possible
            if self.ai_panel.agent:
                self.ai_panel.agent.set_workspace(path)

    def import_data(self):
        file_path, _ = QFileDialog.getOpenFileName(self, tr("dialog.import_data"), "", tr("dialog.vector_filter"))
        if file_path: self.load_vector_layer(file_path)

    def export_data(self):
        name = self.layer_panel.get_selected_layer_name()
        if not name: return
        self.layer_panel._export_layer(name, "geojson") # Simplified

    def ai_settings(self):
        from PySide6.QtWidgets import QDialog, QFormLayout, QComboBox, QGroupBox, QFileDialog, QHBoxLayout, QPushButton, QLineEdit, QMessageBox
        
        # 读取现有配置
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "config.yaml")
        current_config = {}
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    current_config = yaml.safe_load(f) or {}
            except:
                pass

        dialog = QDialog(self)
        dialog.setWindowTitle(tr("ai.settings_title"))
        dialog.setMinimumWidth(500)
        layout = QVBoxLayout(dialog)
        
        # AI设置
        ai_group = QGroupBox(tr("ai.model_settings"))
        form = QFormLayout(ai_group)
        
        provider_combo = QComboBox()
        provider_combo.addItems(["zhipu", "openai", "anthropic", "qwen"])
        provider_combo.setCurrentText(current_config.get("ai", {}).get("provider", "zhipu"))
        form.addRow(tr("ai.provider"), provider_combo)
        
        api_key_edit = QLineEdit()
        api_key_edit.setText(current_config.get("ai", {}).get("api_key", ""))
        api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        form.addRow(tr("ai.api_key"), api_key_edit)
        
        # 模型选择与检测
        model_layout = QHBoxLayout()
        model_combo = QComboBox()
        model_combo.setEditable(True)
        model_combo.setCurrentText(current_config.get("ai", {}).get("model", ""))
        model_combo.setPlaceholderText(tr("ai.model_placeholder"))
        model_layout.addWidget(model_combo)

        vlm_combo = QComboBox()
        vlm_combo.setEditable(True)
        vlm_combo.setCurrentText(current_config.get("ai", {}).get("vlm_model", ""))
        vlm_combo.setPlaceholderText(tr("ai.vlm_placeholder"))
        
        check_btn = QPushButton(tr("common.detect"))
        check_btn.setToolTip(tr("ai.check_tooltip"))
        check_btn.clicked.connect(
            lambda: self.check_api_models(
                provider_combo.currentText(),
                api_key_edit.text(),
                model_combo,
                check_btn,
                vlm_combo,
            )
        )
        model_layout.addWidget(check_btn)
        
        form.addRow(tr("ai.model_name"), model_layout)
        form.addRow(tr("ai.vlm_model"), vlm_combo)
        
        layout.addWidget(ai_group)
        
        # 环境设置
        env_group = QGroupBox(tr("ai.env_settings"))
        env_form = QFormLayout(env_group)
        
        # QGIS路径
        qgis_path_edit = QLineEdit()
        qgis_path_edit.setText(current_config.get("qgis_path", ""))
        qgis_path_edit.setPlaceholderText(tr("ai.qgis_placeholder"))
        
        qgis_btn = QPushButton(tr("common.browse"))
        qgis_btn.clicked.connect(lambda: qgis_path_edit.setText(QFileDialog.getExistingDirectory(dialog, tr("dialog.choose_qgis"))))
        
        qgis_layout = QHBoxLayout()
        qgis_layout.addWidget(qgis_path_edit)
        qgis_layout.addWidget(qgis_btn)
        env_form.addRow(tr("ai.qgis_path"), qgis_layout)
        
        # Python路径
        python_path_edit = QLineEdit()
        python_path_edit.setText(current_config.get("python_path", ""))
        python_path_edit.setPlaceholderText(tr("ai.python_placeholder"))
        
        python_btn = QPushButton(tr("common.browse"))
        python_btn.clicked.connect(lambda: python_path_edit.setText(QFileDialog.getOpenFileName(dialog, tr("dialog.choose_python"), "", tr("dialog.python_filter"))[0]))
        
        python_layout = QHBoxLayout()
        python_layout.addWidget(python_path_edit)
        python_layout.addWidget(python_btn)
        env_form.addRow(tr("ai.python_path"), python_layout)
        
        layout.addWidget(env_group)
        
        # 按钮
        btn_box = QGroupBox()
        btn_box.setStyleSheet("border:none;")
        btn_layout = QHBoxLayout(btn_box)
        btn_layout.addStretch()
        
        save_btn = QPushButton(tr("ai.save_config"))
        save_btn.setStyleSheet("background: #1976d2; color: white; padding: 6px 20px; font-weight: bold;")
        save_btn.clicked.connect(lambda: self.save_ai_config(
            dialog, 
            provider_combo.currentText(), 
            api_key_edit.text(), 
            model_combo.currentText(),
            vlm_combo.currentText(),
            qgis_path_edit.text(),
            python_path_edit.text()
        ))
        
        cancel_btn = QPushButton(tr("common.cancel"))
        cancel_btn.clicked.connect(dialog.reject)
        
        btn_layout.addWidget(cancel_btn)
        btn_layout.addWidget(save_btn)
        layout.addWidget(btn_box)
        
        dialog.exec()

    def check_api_models(self, provider, api_key, combo, btn, vlm_combo=None):
        if not api_key:
            QMessageBox.warning(self, tr("common.warning"), tr("ai.api_key_required"))
            return
            
        btn.setEnabled(False)
        btn.setText(tr("ai.checking"))
        
        self.check_thread = ModelCheckThread(provider, api_key)
        self.check_thread.finished.connect(lambda models: self.on_check_finished(models, combo, btn, vlm_combo, provider, api_key))
        self.check_thread.error.connect(lambda err: self.on_check_error(err, btn))
        self.check_thread.start()

    def on_check_finished(self, models, combo, btn, vlm_combo=None, provider="zhipu", api_key=""):
        btn.setEnabled(True)
        btn.setText(tr("common.detect"))
        if models:
            combo.clear()
            combo.addItems(models)
            if vlm_combo is not None:
                vlm_combo.clear()
                candidates = self._build_vlm_candidates(provider, models)
                usable = self._filter_usable_vlm_models(provider, api_key, candidates)
                final_vlm = usable if usable else candidates
                vlm_combo.addItems(final_vlm)
                preferred = final_vlm[0] if final_vlm else ""
                if preferred:
                    vlm_combo.setCurrentText(preferred)
            QMessageBox.information(self, tr("common.success"), tr("ai.check_success", count=len(models)))
            combo.showPopup()
        else:
            QMessageBox.warning(self, tr("common.warning"), tr("ai.check_empty"))

    def _build_vlm_candidates(self, provider, models):
        seen = set()
        out = []

        def add(x):
            if not x:
                return
            x = str(x).strip()
            if not x or x in seen:
                return
            seen.add(x)
            out.append(x)

        for m in models or []:
            lowered = str(m).lower()
            if "4v" in lowered or "6v" in lowered or "vision" in lowered or "vlm" in lowered:
                add(m)

        if provider == "zhipu":
            for m in ["glm-4v", "glm-4v-plus", "glm-4.6v"]:
                add(m)

        if not out:
            for m in models or []:
                add(m)
        return out

    def _filter_usable_vlm_models(self, provider, api_key, candidates):
        if not candidates:
            return []
        if not api_key:
            return list(candidates)
        if provider != "zhipu":
            return list(candidates)

        usable = []
        for model in candidates:
            ok = self._verify_zhipu_model_callable(api_key, model)
            if ok:
                usable.append(model)
        return usable

    def _verify_zhipu_model_callable(self, api_key, model):
        # Prefer official SDK if available (more reliable for model routing).
        try:
            from zhipuai import ZhipuAI

            client = ZhipuAI(api_key=api_key)
            client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=8,
            )
            return True
        except Exception:
            pass

        try:
            import requests
            url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 8,
                "temperature": 0.0,
            }
            resp = requests.post(url, headers=headers, json=payload, timeout=12)
            return resp.status_code == 200
        except Exception:
            return False

    def on_check_error(self, err, btn):
        btn.setEnabled(True)
        btn.setText(tr("common.detect"))
        QMessageBox.warning(self, tr("ai.check_failed"), tr("ai.check_failed_message", error=err))

    def save_ai_config(self, dialog, provider, key, model, vlm_model, qgis_path, python_path):
        config = self._load_runtime_config()
        config["ai"] = {
            "provider": provider,
            "api_key": key,
            "model": model,
            "vlm_model": vlm_model,
        }
        config["qgis_path"] = qgis_path
        config["python_path"] = python_path
        config["language"] = self._language

        config_path = self._config_path
        try:
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)
            
            # 重新加载Agent
            if hasattr(self, 'ai_panel'):
                self.ai_panel.reload_provider()
            
            # 更新全局工具配置
            try:
                from src.agent.tools import configure_tools
                configure_tools(config)
            except Exception as e:
                print(f"配置工具失败: {e}")

            self._runtime_config = config
            self.refresh_env_status_widget(self._runtime_config)
            
            QMessageBox.information(self, tr("common.success"), tr("ai.config_saved"))
            dialog.accept()
        except Exception as e:
            QMessageBox.critical(self, tr("common.error"), tr("ai.config_save_failed", error=e))

    def show_about(self):
        QMessageBox.about(self, tr("about.title"), tr("about.message"))

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    font = QFont("Microsoft YaHei", 9)
    app.setFont(font)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
