# 任意比例尺地图综合：算子选择与冲突检测框架

## 1. 目标

构建一个可用于任意目标比例尺 `1:St` 的综合决策框架，使 LLM 能自动选择综合算子与参数，VLM 能基于规则验证结果是否达标，并形成可迭代闭环。

## 2. 核心思路

1. 用“连续尺度函数”替代“只支持标准比例尺”的离散规则。
2. 先做可量化冲突检测，再决定算子组合，而不是先固定算子。
3. LLM 负责规划与调用工具，VLM 负责视觉与规则一致性验收。
4. 每轮输出结构化日志，失败后按冲突类型定向修正。

## 3. 比例尺与符号换算规则

设：
- 源比例尺分母 `Ss`，目标比例尺分母 `St`
- 缩尺比 `R = St / Ss`
- 地图上 `1 mm` 对应地面距离 `St / 1000` 米

常用换算：
- 最小可分辨长度（地面）`L_min_m = L_min_mm * St / 1000`
- 最小可保留面积（地面）`A_min_m2 = A_min_mm2 * (St / 1000)^2`
- 最小符号间距（地面）`D_min_m = D_min_mm * St / 1000`

建议初值（可按图式调整）：
- `L_min_mm = 0.35`
- `A_min_mm2 = 1.0`
- `D_min_mm = 0.30`

## 4. 任意比例尺算子选择规则

### 4.1 算子候选池（按几何）

- 点：`selection`, `typification`, `displacement`
- 线：`selection`, `simplify`, `smooth`, `merge`, `displacement`
- 面：`selection`, `eliminate`, `aggregate`, `simplify`, `displacement`

### 4.2 执行顺序（固定骨架）

1. 选择/删除（Selection/Elimination）
2. 简化（Simplify）
3. 合并/聚合（Merge/Aggregate）
4. 位移解冲突（Displacement）
5. 平滑与符号微调（Smooth/Symbol tuning）

### 4.3 连续尺度参数函数（关键）

不要只写 `1:25000`、`1:50000` 两档，改为分段线性插值。

设某参数在两个标准比例尺下为：
- `p25`（对应 `St=25000`）
- `p50`（对应 `St=50000`）

则当 `St` 落在 `[25000,50000]`：

`p(St) = p25 + (p50 - p25) * (St - 25000) / 25000`

### 4.4 1:35000 示例（线简化容差）

若道路简化容差设定：
- `tol25 = 8 m`
- `tol50 = 20 m`

则 `St=35000`：

`tol35 = 8 + (20 - 8) * (35000 - 25000) / 25000 = 12.8 m`

同理可对最小面积、最小间距、分类阈值全部做连续插值。

## 5. 冲突检测规则（规则引擎）

## 5.1 几何/符号冲突类型

- 点-点冲突：符号中心距离 `< D_min_m`
- 点-线冲突：点到线距离 `< D_min_m`
- 线-线冲突：并行线最小间距 `< D_min_m`
- 面-面冲突：微小缝隙/重叠导致不可读
- 标签冲突（后续可加）：标签框重叠率超阈值

### 5.2 可量化质量指标（每轮必须输出）

- `conflict_count_total`
- `conflict_density_per_km2`
- `min_symbol_separation_m`
- `topology_error_count`
- `retention_rate_by_class`
- `shape_distortion_score`（可先用简化前后面积/长度偏差近似）

### 5.3 拓扑约束（硬约束）

- 道路/河流网络连通性不下降
- 行政边界不交叉、不自相交
- 面要素不出现非法几何（invalid geometry）

## 6. LLM + VLM 协同闭环

### 6.1 角色分工

- LLM（Planner-Agent）
  - 读取目标比例尺、图层统计、当前冲突指标
  - 产出本轮算子序列与参数
  - 调用你的 AI 助手工具执行
- VLM（Verifier-Agent）
  - 读取渲染图 + 指标表
  - 判断“可读性/遮挡/平衡性”是否达标
  - 给出失败类型与修正建议

### 6.2 闭环流程

1. LLM 产出计划 `plan_v1`
2. 执行算子并导出图层 + 渲染截图
3. 规则引擎输出量化指标
4. VLM 给出 `pass/fail + evidence`
5. 若 `fail`，LLM 仅对失败指标定向调参并进入下一轮
6. 达到阈值或轮次上限即停止

### 6.3 停止条件

- `conflict_count_total <= T_conflict`
- `topology_error_count == 0`
- `retention_rate_by_class` 不低于类别下限
- VLM 打分 `>= T_vlm_score`
- 或达到 `max_rounds`

## 7. 提示词框架（可直接用）

### 7.1 LLM 系统提示词（Planner）

```text
你是地图综合规划代理。目标是在任意比例尺下，基于图层统计和冲突指标，选择最小必要算子集合。

硬约束：
1) 先做选择/删除，再简化，再聚合，再位移，最后平滑。
2) 不得破坏网络连通性、行政边界合法性与几何有效性。
3) 所有阈值必须由比例尺换算或标准比例尺插值推导，不允许拍脑袋常数。
4) 每轮必须输出：算子序列、参数、预期改善指标、回滚条件。

输出必须为 JSON，不要输出自然语言解释。
```

### 7.2 LLM 用户提示模板（每轮）

```text
任务: 地图综合
源比例尺: 1:{Ss}
目标比例尺: 1:{St}
图层统计: {layer_stats_json}
当前指标: {qa_metrics_json}
上一轮失败原因: {last_fail_reason}
可用算子: [selection, simplify, smooth, aggregate, merge, displacement, eliminate, typification]
请输出下一轮计划 JSON。
```

### 7.3 LLM 输出 JSON 协议

```json
{
  "round": 2,
  "target_scale": 35000,
  "operators": [
    {"name": "selection", "params": {"min_area_m2": 1800}},
    {"name": "simplify", "params": {"tolerance_m": 12.8, "preserve_topology": true}},
    {"name": "displacement", "params": {"min_sep_m": 10.5, "max_shift_m": 18}}
  ],
  "expected_delta": {
    "conflict_count_total": -35,
    "topology_error_count": 0
  },
  "rollback_if": [
    "retention_rate_by_class.primary_road < 0.98",
    "topology_error_count > 0"
  ]
}
```

### 7.4 VLM 系统提示词（Verifier）

```text
你是地图综合验收代理。你会看到：综合前后截图、冲突指标、拓扑检查结果。
请只判断是否通过验收，并指出证据。

验收优先级：
1) 拓扑硬约束（必须通过）
2) 遮挡与可读性（冲突是否明显下降）
3) 形状保持与类别层级（主干要素是否保留）

输出 JSON:
{
  "pass": true/false,
  "score": 0-100,
  "failed_rules": [],
  "evidence": [],
  "suggested_fixes": []
}
```

### 7.5 VLM 用户提示模板

```text
目标比例尺: 1:{St}
前后对比图: {before_after_image_refs}
量化指标: {qa_metrics_json}
拓扑报告: {topology_report_json}
请返回验收 JSON。
```

## 8. 规则参数建议（首版可用）

- 冲突阈值：
  - `T_conflict = max(5, feature_count * 0.005)`
- VLM 通过阈值：
  - `T_vlm_score = 80`
- 最大迭代轮次：
  - `max_rounds = 4`
- 类别保留率下限（示例）：
  - 主干路 `>= 0.98`
  - 次干路 `>= 0.90`
  - 支路 `>= 0.70`

## 9. 与你现有 AI 助手的对接建议

将每轮流程绑定为：

1. `load_layer` / 数据加载
2. `run_python_script` / 执行算子管线
3. `load_layer` / 加载结果图层
4. `zoom_to_layer` / 固定视图导出截图
5. `run_python_script` / 输出 QA 指标 JSON
6. 送入 VLM 验证

建议保留统一执行日志：
- 输入比例尺、算子参数、指标变化、VLM 结论、最终是否通过

## 10. 最小可行版本（MVP）边界

首版先只覆盖：
- 几何类型：道路（线）+ 建筑（面）
- 冲突类型：线线、面面、点线（如 POI）
- 验收：拓扑 + 遮挡 + 保留率

跑通后再扩展标签冲突、语义冲突和多图层联动。
