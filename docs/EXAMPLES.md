# MapGenAI 任务示例库

本示例库提供 10 个可复现的典型任务。建议先阅读 `QUICK_START.md` 与 `USER_GUIDE.md`。

## 目录
- [示例1：道路简化（Douglas-Peucker）](#示例1道路简化douglas-peucker)
- [示例2：建筑物合并（Dissolve）](#示例2建筑物合并dissolve)
- [示例3：缓冲区分析](#示例3缓冲区分析)
- [示例4：坐标系转换（WGS84 -> UTM）](#示例4坐标系转换wgs84---utm)
- [示例5：拓扑修复（修复自相交）](#示例5拓扑修复修复自相交)
- [示例6：空间连接（点与面）](#示例6空间连接点与面)
- [示例7：裁剪分析（面裁剪面）](#示例7裁剪分析面裁剪面)
- [示例8：叠加分析（面叠加面）](#示例8叠加分析面叠加面)
- [示例9：点聚类（DBSCAN）](#示例9点聚类dbscan)
- [示例10：线平滑（Chaikin）](#示例10线平滑chaikin)

## 示例1：道路简化（Douglas-Peucker）
### 任务描述
对道路图层进行 Douglas-Peucker 简化，容差 10 米，保持拓扑。
### 输入数据
- 文件：`data/roads.shp`
- 类型：线要素
- 坐标系：EPSG:4326
### 用户输入
`对 roads.shp 进行 Douglas-Peucker 简化，容差 10 米，保持拓扑正确性`
### 预期结果
- 输出：`data/roads_simplified.shp`
- 拓扑保持，要素数不变，节点数下降

## 示例2：建筑物合并（Dissolve）
### 任务描述
按行政区字段合并建筑面。
### 输入数据
- 文件：`data/buildings.shp`
- 类型：面要素
### 用户输入
`将 buildings.shp 按 district 字段合并`
### 预期结果
- 输出：`data/buildings_dissolved.shp`
- 同一 `district` 形成聚合面

## 示例3：缓冲区分析
### 任务描述
对学校点创建 500 米缓冲区。
### 输入数据
- 文件：`data/schools.shp`
- 类型：点要素
### 用户输入
`对 schools.shp 创建 500 米缓冲区`
### 预期结果
- 输出：`data/schools_buffer_500m.shp`
- 生成覆盖范围用于服务半径分析

## 示例4：坐标系转换（WGS84 -> UTM）
### 任务描述
将道路数据从地理坐标转换为投影坐标。
### 输入数据
- 文件：`data/roads.shp`
- 坐标系：EPSG:4326
### 用户输入
`将 roads.shp 从 EPSG:4326 转换为 EPSG:32649`
### 预期结果
- 输出：`data/roads_utm49n.shp`
- 可用于距离/面积准确计算

## 示例5：拓扑修复（修复自相交）
### 任务描述
检查并修复面要素自相交问题。
### 输入数据
- 文件：`data/parcels_invalid.shp`
- 类型：面要素
### 用户输入
`修复 parcels_invalid.shp 的自相交拓扑错误`
### 预期结果
- 输出：`data/parcels_fixed.shp`
- 几何有效率提升

## 示例6：空间连接（点与面）
### 任务描述
将 POI 点按所属行政区进行属性关联。
### 输入数据
- 点文件：`data/poi.shp`
- 面文件：`data/districts.shp`
### 用户输入
`将 poi.shp 与 districts.shp 做空间连接，保留行政区名称`
### 预期结果
- 输出：`data/poi_joined.shp`
- 点要素新增行政区字段

## 示例7：裁剪分析（面裁剪面）
### 任务描述
以研究区边界裁剪土地利用图层。
### 输入数据
- 被裁剪：`data/landuse.shp`
- 裁剪边界：`data/study_area.shp`
### 用户输入
`用 study_area.shp 裁剪 landuse.shp`
### 预期结果
- 输出：`data/landuse_clip.shp`
- 仅保留研究区内面要素

## 示例8：叠加分析（面叠加面）
### 任务描述
对生态红线与土地利用做叠加分析。
### 输入数据
- 图层 A：`data/ecoline.shp`
- 图层 B：`data/landuse.shp`
### 用户输入
`对 ecoline.shp 和 landuse.shp 做叠加分析，输出交集`
### 预期结果
- 输出：`data/ecoline_landuse_intersection.shp`
- 获取冲突或重叠区域

## 示例9：点聚类（DBSCAN）
### 任务描述
对事故点进行 DBSCAN 聚类，识别热点。
### 输入数据
- 文件：`data/incidents.shp`
- 类型：点要素
### 用户输入
`对 incidents.shp 进行 DBSCAN 聚类，半径 100 米，最小样本数 5`
### 预期结果
- 输出：`data/incidents_clustered.shp`
- 新增聚类编号字段

## 示例10：线平滑（Chaikin）
### 任务描述
对河流线进行平滑处理。
### 输入数据
- 文件：`data/rivers.shp`
- 类型：线要素
### 用户输入
`对 rivers.shp 使用 Chaikin 算法平滑，迭代 3 次`
### 预期结果
- 输出：`data/rivers_smoothed.shp`
- 线形更平滑，视觉连续性更好
