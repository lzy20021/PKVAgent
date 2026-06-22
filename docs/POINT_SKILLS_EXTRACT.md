# 平台智能体 Skills 提取（点综合相关）

## 1) 核心 Skill：点要素综合
- Skill 名称：`point_generalization_skill`
- 文件路径：`skills/point_generalization_skill.md`
- 定位：面向点要素（POI、地铁站、医院、学校、监测点等）的地图综合，采用 VLM + LLM 联合决策与验收。

### 关键能力
- 结构识别：`cluster / corridor / center-periphery / uniform / multi-core`
- 语义识别：交通点、公共服务点、商业 POI、监测点
- 联合决策：按“空间结构 × 语义类型”选算法，而不是固定删点
- 质量验收：VLM 视觉验收 + LLM 规则验收

### 关键约束（摘要）
- 目标比例尺 >= `1:200000` 时，冲突目标趋近 `0`，需持续迭代到无重叠
- 廊道型点：轴向覆盖率 `>= 0.60`，中段空洞率 `<= 0.35`
- 商业热点：热点覆盖率 `>= 0.50`，热点代表点数 `>= 3`
- 稀疏区域（1-2 点）至少保留 1 点；中密/高密区域禁止普遍塌缩成单点
- 轮廓保真：边界控制点（8方向锚点）必须保留，避免整体形态塌缩

## 2) 基础母 Skill：地图综合总流程
- Skill 名称：`map-generalization`
- 文件路径：`skills/map_generalization_skill.md`
- 作用：点综合前置规范（比例尺约束、最小可读性间距、算子流程、验收闭环）

### 与点综合的关系
- 点综合每次都应先继承 `map-generalization` 再执行点专项
- 标准流程：`selection -> simplify/typification -> displacement -> polish`

## 3) 平台内置技能注册与加载机制
- 管理代码：`src/agent/skills.py`
- 机制：
  - 内置技能：`BUILTIN_SKILLS`（含 `map-generalization`）
  - 本地扩展：自动读取 `skills/*.md` 和 `skills/*.json`
  - 工具接口：`list_skills / get_skill / save_skill`

## 4) 目前你平台中“点综合”可直接调用的 Skill 清单
1. `point_generalization_skill`（点综合专项）
2. `map-generalization`（通用综合母流程）

