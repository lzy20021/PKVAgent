# Skill: map-generalization（导入版）

## 1. 适用范围
用于多尺度制图综合，尤其是从大比例尺到中小比例尺时的可读性优化、冲突消解与质量验收。

## 2. 核心流程
1. 明确 `source_scale` 与 `target_scale`。  
2. 设定可读性约束：最小符号间隔、最小可见尺寸、保留率底线。  
3. 按几何类型选算子并按顺序执行：`selection -> simplify/typification -> displacement -> polish`。  
4. 每轮执行后做质量检查：冲突、拓扑、结构、视觉。  
5. 输出数据 + 参数表 + 验收报告。

## 3. 点要素扩展（与 point_generalization_skill 联动）
- corridor：启用主轴分箱，强制中段代表点保留。  
- clustered：热点代表 + 外围补点，防止只剩中心。  
- uniform：覆盖优先，控制空洞面积。  
- center-periphery / multi-core：分区配额，维持层级。
- dense hotspot：每个热点网格至少保留代表点，禁止热点区整体消失。
- sparse region：每个有效稀疏区域至少保留 1 个代表点，优先保“区域出现过点”的事实表达。

## 4. 质量阈值（默认）
- 最小符号间隔：`0.30 mm`（图上）  
- 轴向覆盖率（点廊道）：`>= 0.60`  
- 中段最大空洞率（点廊道）：`<= 0.35`  
- 热点覆盖率（点团簇）：`>= 0.50`  
- VLM 通过分：`>= 80`

## 5. 验收
必须执行 VLM 视觉验收，并与规则校验联合判定：
- `pass`: 输出 accepted  
- `fail`: 输出 revise_required 并给出修正参数
