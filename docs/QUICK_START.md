# MapGenAI 快速入门

欢迎使用 MapGenAI。本指南帮助你在 10 分钟内完成第一个地图综合任务。

## 目录
- [1. 安装 MapGenAI](#1-安装-mapgenai)
- [2. 配置环境（Python + QGIS）](#2-配置环境python--qgis)
- [3. 加载第一个数据](#3-加载第一个数据)
- [4. 执行第一个任务](#4-执行第一个任务)
- [5. 查看结果](#5-查看结果)
- [6. 下一步](#6-下一步)

## 1. 安装 MapGenAI

### 1.1 获取代码
```powershell
git clone <your-repo-url>
cd MapGenAI
```

### 1.2 安装依赖
```powershell
pip install -r requirements.txt
```

### 1.3 启动程序
```powershell
python main.py
```

## 2. 配置环境（Python + QGIS）

首次启动后，检查底部状态栏：
- `Python: ✅` 表示 Python 环境可用
- `QGIS: ✅/❌` 表示 QGIS 环境状态（可选）

如需配置：
1. 点击状态栏 `Configure`，或使用菜单 `配置 -> 环境配置`
2. 设置 Python 可执行文件路径（必填）
3. 设置 QGIS 安装目录（可选）
4. 点击 `Verify Now` 验证
5. 点击 `Save` 保存

建议优先使用 `Auto Detect` 自动检测路径。

## 3. 加载第一个数据

支持通过以下方式加载：
- 菜单导入矢量数据（如 `shp`、`geojson`、`gpkg`）
- 在左侧工作区打开并双击文件

示例数据建议：`roads.shp`（线要素）。

## 4. 执行第一个任务

在 AI 助手输入框中输入：

```text
对 roads.shp 进行 Douglas-Peucker 简化，容差 10 米，保持拓扑正确性
```

系统会自动：
1. 识别任务类型（线简化）
2. 提取参数（容差=10米）
3. 生成并执行代码
4. 加载输出图层

## 5. 查看结果

建议检查以下内容：
- 地图显示是否正常
- 输出图层是否已加载
- 属性表与要素数量是否符合预期
- 几何是否存在明显破碎或偏移

若结果不理想，可调整容差后重试（例如 5m、20m）。

## 6. 下一步

- 阅读 [用户指南](./USER_GUIDE.md)
- 查看 [任务示例库](./EXAMPLES.md)
- 遇到问题时优先查看 FAQ：`USER_GUIDE.md` 第 5 章
