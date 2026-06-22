# Interface-display

## 数据读取情况
- 读取数据目录：`E:\GradOpening\MapGenAI\amap_poi_dataset_expanded`
- 主数据文件：`all_poi_expanded.geojson`，自动识别城市、类别、行政区和点坐标字段。
- 结构标签文件：`E:\GradOpening\MapGenAI\paper_figures\expanded_structure_map\poi_structure_labels_8types.csv`，用于匹配论文中的 8 种空间结构。

## 随机抽取的代表样本
- 固定随机种子：`20260409`。
- 主展示样本：`北京`，结构类型为 `单核心集聚型`。
- 多簇集型：洛阳（候选池：潍坊, 洛阳, 绍兴）
- 单核心集聚型：北京（候选池：北京）
- 双核心/多核心型：南通（候选池：南通, 大同, 天水）
- 廊道带状型：乌鲁木齐（候选池：乌鲁木齐, 中山, 亳州）
- 边缘偏置型：广州（候选池：天津, 泉州, 广州）
- 稀疏离散型：南平（候选池：上饶, 南平, 南阳）
- 均匀分布型：三沙（候选池：三沙）
- 混合过渡型：嘉兴（候选池：大连, 嘉兴, 太原）

## 八种结构匹配说明
- 论文结构名称与 `structure_type_8` 字段直接对应，为 1:1 匹配。
- 本次未出现必须重命名的结构标签，映射说明同步记录在 `selected_samples.json/csv` 中。

## 页面打开方式
- 直接打开 `index.html` 即可查看静态展示页面。
- 页面使用 `demo_bundle.js` 内嵌数据，不依赖本地后端服务。

## 如何导出 SVG
- 页面主视图与预生成 SVG 均已对齐论文展示风格，可直接截图使用。
- 本轮新增 `interface_display_v2.svg`（推荐），并保留 `interface_display.svg`（原命名兼容）。

## 顾及符号表达的设计实现
- 不再使用单一圆点，而是使用菱形、方形、三角形、十字、星形、六边形表达不同 POI 类别。
- 颜色、描边和尺寸统一在柔和浅色体系内控制，兼顾类别区分与学术图面克制感。
- 综合后结果采用类别保持 + 最小间距 + 网格约束策略，以突出图面可读性。

## 主要文件
- `index.html`：静态展示主页面。
- `styles.css`：页面样式。
- `app.js`：页面交互与 SVG 导出。
- `demo_bundle.js`：页面直接读取的数据包。
- `selected_samples.json / csv`：8 类结构样本清单。
- `demo_points.json`：前端演示数据。
- `demo_data.geojson`：前后对比 GeoJSON。
- `interface_display.svg`：预生成 SVG（兼容原文件名）。
- `interface_display_v2.svg`：论文展示版 SVG（推荐）。
