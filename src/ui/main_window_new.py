"""
MapGenAI 主窗口
地图综合智能体 - 主界面
"""
import sys
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QMenuBar, QMenu, QToolBar, QStatusBar, QTreeWidget,
    QTreeWidgetItem, QTextEdit, QLineEdit, QPushButton, QLabel,
    QFrame, QScrollArea, QDockWidget, QFileDialog, QMessageBox,
    QListWidget, QListWidgetItem, QTabWidget, QProgressBar
)
from PySide6.QtCore import Qt, QSize, Signal, QThread
from PySide6.QtGui import QAction, QIcon, QFont, QColor, QPalette, QPixmap


class MapCanvas(QFrame):
    """地图渲染画布"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Sunken)
        self.setMinimumSize(400, 300)
        self.setStyleSheet("""
            QFrame {
                background-color: #f5f5f5;
                border: 1px solid #ccc;
                border-radius: 4px;
            }
        """)

        layout = QVBoxLayout(self)
        self.placeholder = QLabel("地图渲染区域\n\n拖拽或导入地图数据开始使用")
        self.placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.placeholder.setStyleSheet("color: #888; font-size: 16px;")
        layout.addWidget(self.placeholder)


class LayerPanel(QWidget):
    """图层管理面板"""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        title = QLabel("图层管理")
        title.setStyleSheet("font-weight: bold; padding: 8px; background: #e0e0e0;")
        layout.addWidget(title)

        self.layer_tree = QTreeWidget()
        self.layer_tree.setHeaderHidden(True)
        self.layer_tree.setStyleSheet("""
            QTreeWidget {
                border: none;
                background: #fafafa;
            }
            QTreeWidget::item {
                padding: 4px;
            }
            QTreeWidget::item:hover {
                background: #e3f2fd;
            }
            QTreeWidget::item:selected {
                background: #1976d2;
                color: white;
            }
        """)

        root = QTreeWidgetItem(self.layer_tree, ["项目图层"])
        root.setExpanded(True)
        QTreeWidgetItem(root, ["底图"])
        QTreeWidgetItem(root, ["道路网络"])
        QTreeWidgetItem(root, ["建筑物"])
        QTreeWidgetItem(root, ["水系"])
        QTreeWidgetItem(root, ["植被"])

        layout.addWidget(self.layer_tree)


class AlgorithmPanel(QWidget):
    """算法工具面板"""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        title = QLabel("地图综合算法")
        title.setStyleSheet("font-weight: bold; padding: 8px; background: #e0e0e0;")
        layout.addWidget(title)

        self.algo_tree = QTreeWidget()
        self.algo_tree.setHeaderHidden(True)
        self.algo_tree.setStyleSheet("""
            QTreeWidget {
                border: none;
                background: #fafafa;
            }
            QTreeWidget::item {
                padding: 4px;
            }
            QTreeWidget::item:hover {
                background: #e8f5e9;
            }
        """)

        # 线要素综合
        line_cat = QTreeWidgetItem(self.algo_tree, ["线要素综合"])
        line_cat.setExpanded(True)
        QTreeWidgetItem(line_cat, ["Douglas-Peucker简化"])
        QTreeWidgetItem(line_cat, ["Visvalingam-Whyatt简化"])
        QTreeWidgetItem(line_cat, ["曲线平滑"])
        QTreeWidgetItem(line_cat, ["线要素合并"])
        QTreeWidgetItem(line_cat, ["道路选取"])

        # 面要素综合
        polygon_cat = QTreeWidgetItem(self.algo_tree, ["面要素综合"])
        polygon_cat.setExpanded(True)
        QTreeWidgetItem(polygon_cat, ["建筑物简化"])
        QTreeWidgetItem(polygon_cat, ["建筑物合并"])
        QTreeWidgetItem(polygon_cat, ["建筑物位移"])
        QTreeWidgetItem(polygon_cat, ["面积阈值选取"])
        QTreeWidgetItem(polygon_cat, ["典型化处理"])

        # 点要素综合
        point_cat = QTreeWidgetItem(self.algo_tree, ["点要素综合"])
        point_cat.setExpanded(True)
        QTreeWidgetItem(point_cat, ["点聚类"])
        QTreeWidgetItem(point_cat, ["点选取"])
        QTreeWidgetItem(point_cat, ["点位移"])
        QTreeWidgetItem(point_cat, ["密度分析"])

        # 冲突检测与处理
        conflict_cat = QTreeWidgetItem(self.algo_tree, ["冲突检测与处理"])
        conflict_cat.setExpanded(True)
        QTreeWidgetItem(conflict_cat, ["重叠检测"])
        QTreeWidgetItem(conflict_cat, ["自动位移"])
        QTreeWidgetItem(conflict_cat, ["缓冲区分析"])

        # 制图综合
        carto_cat = QTreeWidgetItem(self.algo_tree, ["制图综合"])
        carto_cat.setExpanded(True)
        QTreeWidgetItem(carto_cat, ["注记配置"])
        QTreeWidgetItem(carto_cat, ["符号化"])
        QTreeWidgetItem(carto_cat, ["比例尺转换"])

        layout.addWidget(self.algo_tree)


class ChatMessage(QFrame):
    """聊天消息气泡"""
    def __init__(self, text, is_user=True, parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.Shape.NoFrame)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)

        msg_label = QLabel(text)
        msg_label.setWordWrap(True)
        msg_label.setMaximumWidth(280)

        if is_user:
            layout.addStretch()
            msg_label.setStyleSheet("""
                QLabel {
                    background: #1976d2;
                    color: white;
                    padding: 10px 14px;
                    border-radius: 12px;
                    font-size: 13px;
                }
            """)
        else:
            msg_label.setStyleSheet("""
                QLabel {
                    background: #e8e8e8;
                    color: #333;
                    padding: 10px 14px;
                    border-radius: 12px;
                    font-size: 13px;
                }
            """)

        layout.addWidget(msg_label)

        if not is_user:
            layout.addStretch()


class AIAssistantPanel(QWidget):
    """AI助手面板"""
    message_sent = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # 标题栏
        header = QFrame()
        header.setStyleSheet("background: #1976d2;")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(12, 8, 12, 8)

        title = QLabel("AI 地图综合助手")
        title.setStyleSheet("color: white; font-weight: bold; font-size: 14px;")
        header_layout.addWidget(title)

        status = QLabel("在线")
        status.setStyleSheet("color: #90caf9; font-size: 12px;")
        header_layout.addStretch()
        header_layout.addWidget(status)

        layout.addWidget(header)

        # 聊天区域
        self.chat_scroll = QScrollArea()
        self.chat_scroll.setWidgetResizable(True)
        self.chat_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.chat_scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background: #fafafa;
            }
        """)

        self.chat_container = QWidget()
        self.chat_layout = QVBoxLayout(self.chat_container)
        self.chat_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.chat_layout.setSpacing(8)
        self.chat_layout.setContentsMargins(8, 8, 8, 8)

        welcome = ChatMessage(
            "你好！我是MapGenAI智能助手。\n\n"
            "我可以帮助你进行地图综合操作，包括：\n"
            "• 线要素简化与平滑\n"
            "• 建筑物综合处理\n"
            "• 冲突检测与解决\n"
            "• 自动制图综合\n\n"
            "请告诉我你需要什么帮助？",
            is_user=False
        )
        self.chat_layout.addWidget(welcome)

        self.chat_scroll.setWidget(self.chat_container)
        layout.addWidget(self.chat_scroll, 1)

        # 输入区域
        input_frame = QFrame()
        input_frame.setStyleSheet("background: white; border-top: 1px solid #ddd;")
        input_layout = QHBoxLayout(input_frame)
        input_layout.setContentsMargins(8, 8, 8, 8)

        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("输入消息，描述你的地图综合需求...")
        self.input_field.setStyleSheet("""
            QLineEdit {
                border: 1px solid #ddd;
                border-radius: 18px;
                padding: 8px 16px;
                font-size: 13px;
                background: #f5f5f5;
            }
            QLineEdit:focus {
                border-color: #1976d2;
                background: white;
            }
        """)
        self.input_field.returnPressed.connect(self.send_message)

        self.send_btn = QPushButton("发送")
        self.send_btn.setStyleSheet("""
            QPushButton {
                background: #1976d2;
                color: white;
                border: none;
                border-radius: 18px;
                padding: 8px 20px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #1565c0;
            }
            QPushButton:pressed {
                background: #0d47a1;
            }
        """)
        self.send_btn.clicked.connect(self.send_message)

        input_layout.addWidget(self.input_field)
        input_layout.addWidget(self.send_btn)

        layout.addWidget(input_frame)

    def send_message(self):
        text = self.input_field.text().strip()
        if text:
            user_msg = ChatMessage(text, is_user=True)
            self.chat_layout.addWidget(user_msg)
            self.input_field.clear()
            self.add_ai_response(text)
            self.chat_scroll.verticalScrollBar().setValue(
                self.chat_scroll.verticalScrollBar().maximum()
            )

    def add_ai_response(self, user_text):
        response = self.generate_response(user_text)
        ai_msg = ChatMessage(response, is_user=False)
        self.chat_layout.addWidget(ai_msg)

    def generate_response(self, text):
        text_lower = text.lower()

        if "简化" in text or "douglas" in text_lower:
            return ("好的，我可以帮你进行线要素简化。\n\n"
                    "请先选择要简化的图层，然后我会使用Douglas-Peucker算法进行处理。\n\n"
                    "你可以指定简化容差值（默认为10米）。")
        elif "建筑" in text:
            return ("建筑物综合包括多种操作：\n\n"
                    "1. 建筑物简化 - 简化建筑物轮廓\n"
                    "2. 建筑物合并 - 合并相邻建筑\n"
                    "3. 建筑物位移 - 解决重叠冲突\n\n"
                    "请告诉我你需要哪种操作？")
        elif "冲突" in text or "重叠" in text:
            return ("我可以帮你检测和处理地图要素冲突。\n\n"
                    "冲突处理流程：\n"
                    "1. 检测重叠区域\n"
                    "2. 计算位移向量\n"
                    "3. 执行自动位移\n\n"
                    "是否开始冲突检测？")
        elif "帮助" in text or "help" in text_lower:
            return ("我可以帮助你完成以下地图综合任务：\n\n"
                    "• 线要素：简化、平滑、合并\n"
                    "• 面要素：建筑物综合、面积选取\n"
                    "• 点要素：聚类、选取、位移\n"
                    "• 冲突处理：检测、位移、缓冲\n"
                    "• 制图：注记、符号化、比例尺转换\n\n"
                    "请描述你的具体需求。")
        else:
            return (f"收到你的请求：\"{text}\"\n\n"
                    "我正在分析你的需求。请确保已加载地图数据，"
                    "然后我会为你推荐合适的综合算法。")


class MainWindow(QMainWindow):
    """主窗口"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MapGenAI - 地图综合智能体 v0.1")
        self.setMinimumSize(1200, 800)
        self.setup_ui()
        self.setup_menu()
        self.setup_toolbar()
        self.setup_statusbar()

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)

        # 左侧面板
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)

        self.left_tabs = QTabWidget()
        self.left_tabs.setStyleSheet("""
            QTabWidget::pane {
                border: none;
            }
            QTabBar::tab {
                padding: 8px 16px;
                background: #e0e0e0;
            }
            QTabBar::tab:selected {
                background: #1976d2;
                color: white;
            }
        """)

        self.layer_panel = LayerPanel()
        self.algo_panel = AlgorithmPanel()

        self.left_tabs.addTab(self.layer_panel, "图层")
        self.left_tabs.addTab(self.algo_panel, "算法")

        left_layout.addWidget(self.left_tabs)
        left_panel.setMinimumWidth(250)
        left_panel.setMaximumWidth(350)

        # 中间地图区域
        self.map_canvas = MapCanvas()

        # 右侧AI助手
        self.ai_panel = AIAssistantPanel()
        self.ai_panel.setMinimumWidth(320)
        self.ai_panel.setMaximumWidth(400)

        self.main_splitter.addWidget(left_panel)
        self.main_splitter.addWidget(self.map_canvas)
        self.main_splitter.addWidget(self.ai_panel)
        self.main_splitter.setSizes([280, 600, 350])

        main_layout.addWidget(self.main_splitter)

    def setup_menu(self):
        menubar = self.menuBar()
        menubar.setStyleSheet("""
            QMenuBar {
                background: #f5f5f5;
                padding: 4px;
            }
            QMenuBar::item {
                padding: 6px 12px;
                background: transparent;
            }
            QMenuBar::item:selected {
                background: #e0e0e0;
            }
            QMenu {
                background: white;
                border: 1px solid #ddd;
            }
            QMenu::item {
                padding: 8px 24px;
            }
            QMenu::item:selected {
                background: #1976d2;
                color: white;
            }
        """)

        # 文件菜单
        file_menu = menubar.addMenu("文件")
        file_menu.addAction("新建项目", self.new_project)
        file_menu.addAction("打开项目", self.open_project)
        file_menu.addAction("保存项目", self.save_project)
        file_menu.addSeparator()
        file_menu.addAction("导入数据", self.import_data)
        file_menu.addAction("导出数据", self.export_data)
        file_menu.addSeparator()
        file_menu.addAction("退出", self.close)

        # 查看菜单
        view_menu = menubar.addMenu("查看")
        view_menu.addAction("放大")
        view_menu.addAction("缩小")
        view_menu.addAction("全图显示")
        view_menu.addSeparator()
        view_menu.addAction("显示图层面板")
        view_menu.addAction("显示AI助手")

        # 编辑菜单
        edit_menu = menubar.addMenu("编辑")
        edit_menu.addAction("撤销")
        edit_menu.addAction("重做")
        edit_menu.addSeparator()
        edit_menu.addAction("选择全部")
        edit_menu.addAction("清除选择")

        # 图层菜单
        layer_menu = menubar.addMenu("图层")
        layer_menu.addAction("添加图层")
        layer_menu.addAction("删除图层")
        layer_menu.addSeparator()
        layer_menu.addAction("图层属性")
        layer_menu.addAction("图层样式")

        # 综合菜单
        gen_menu = menubar.addMenu("综合")

        line_menu = gen_menu.addMenu("线要素综合")
        line_menu.addAction("Douglas-Peucker简化")
        line_menu.addAction("Visvalingam-Whyatt简化")
        line_menu.addAction("曲线平滑")

        polygon_menu = gen_menu.addMenu("面要素综合")
        polygon_menu.addAction("建筑物简化")
        polygon_menu.addAction("建筑物合并")
        polygon_menu.addAction("建筑物位移")

        point_menu = gen_menu.addMenu("点要素综合")
        point_menu.addAction("点聚类")
        point_menu.addAction("点选取")

        gen_menu.addSeparator()
        gen_menu.addAction("冲突检测")
        gen_menu.addAction("自动综合")

        # 配置菜单
        config_menu = menubar.addMenu("配置")
        config_menu.addAction("AI设置", self.ai_settings)
        config_menu.addAction("算法参数")
        config_menu.addSeparator()
        config_menu.addAction("首选项")

        # 帮助菜单
        help_menu = menubar.addMenu("帮助")
        help_menu.addAction("使用文档")
        help_menu.addAction("关于", self.show_about)

    def setup_toolbar(self):
        toolbar = QToolBar("主工具栏")
        toolbar.setMovable(False)
        toolbar.setStyleSheet("""
            QToolBar {
                background: #fafafa;
                border-bottom: 1px solid #ddd;
                padding: 4px;
                spacing: 4px;
            }
            QToolButton {
                padding: 6px 12px;
                border-radius: 4px;
            }
            QToolButton:hover {
                background: #e0e0e0;
            }
        """)
        self.addToolBar(toolbar)

        toolbar.addAction("新建")
        toolbar.addAction("打开")
        toolbar.addAction("保存")
        toolbar.addSeparator()
        toolbar.addAction("导入")
        toolbar.addAction("导出")
        toolbar.addSeparator()
        toolbar.addAction("选择")
        toolbar.addAction("平移")
        toolbar.addAction("缩放")
        toolbar.addSeparator()
        toolbar.addAction("简化")
        toolbar.addAction("平滑")
        toolbar.addAction("合并")

    def setup_statusbar(self):
        statusbar = self.statusBar()
        statusbar.setStyleSheet("""
            QStatusBar {
                background: #f5f5f5;
                border-top: 1px solid #ddd;
            }
        """)
        statusbar.showMessage("就绪 | 坐标: -- | 比例尺: 1:10000")

        self.progress = QProgressBar()
        self.progress.setMaximumWidth(150)
        self.progress.setVisible(False)
        statusbar.addPermanentWidget(self.progress)

    def new_project(self):
        QMessageBox.information(self, "新建项目", "创建新的地图综合项目")

    def open_project(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "打开项目", "", "MapGenAI项目 (*.mgai);;所有文件 (*)"
        )
        if file_path:
            QMessageBox.information(self, "打开项目", f"打开: {file_path}")

    def save_project(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存项目", "", "MapGenAI项目 (*.mgai)"
        )
        if file_path:
            QMessageBox.information(self, "保存项目", f"保存到: {file_path}")

    def import_data(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "导入数据", "",
            "Shapefile (*.shp);;GeoJSON (*.geojson);;GeoPackage (*.gpkg);;所有文件 (*)"
        )
        if file_path:
            QMessageBox.information(self, "导入数据", f"导入: {file_path}")

    def export_data(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "导出数据", "",
            "Shapefile (*.shp);;GeoJSON (*.geojson);;GeoPackage (*.gpkg)"
        )
        if file_path:
            QMessageBox.information(self, "导出数据", f"导出到: {file_path}")

    def ai_settings(self):
        QMessageBox.information(self, "AI设置", "配置AI模型和API密钥")

    def show_about(self):
        QMessageBox.about(
            self, "关于 MapGenAI",
            "<h3>MapGenAI - 地图综合智能体</h3>"
            "<p>版本: 0.1.0</p>"
            "<p>一个结合AI技术的地图综合工具，集成了QGIS、ArcGIS等软件中的地图综合算法。</p>"
        )


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
