# 点综合八类 POI 符号与 VLM 校核使用说明

## 本次接入的八类符号

| POI 类别 | point_symbol | point_symbol_color | point_symbol_label |
| --- | --- | --- | --- |
| 便利店 | convenience | #d23b7d | 24 |
| 学校 | school | #2e7d32 | S |
| 停车场 | parking | #1976d2 | P |
| 酒店 | hotel | #7b4fa1 | H |
| 银行 | bank | #00897b | B |
| 医院 | hospital | #d32f2f | + |
| 餐饮 | restaurant | #ef8a24 | R |
| 交通设施 | transit | #4e79a7 | T |

常见别名会被自动归一，例如 `公交站`、`地铁站`、`火车站` 会归到 `交通设施`，`餐厅`、`饭店` 会归到 `餐饮`。

## 智能体输出字段

点综合输出 GeoJSON 会自动写入这些字段：

- `poi_category`：归一后的八类 POI 类别。
- `point_symbol`：项目 UI 和导出图使用的符号名称。
- `point_symbol_color`：该类符号推荐颜色。
- `point_symbol_label`：符号内部标签。

如果输入数据已有 `query_category`、`poi_category`、`category`、`type`、`类别`、`类型` 等字段，智能体会自动选择可用字段识别类别。

## 如何测试点综合

1. 在项目 UI 中加载包含 POI 类别字段的点图层。
2. 对智能体发送类似任务：

```text
对当前点图层做 1:220000 点综合，输出 geojson 和 png，要求使用八类 POI 符号，并检查符号遮挡。
```

3. 输出 GeoJSON 加载回项目 UI 后，样式推断会读取 `point_symbol` 字段。
4. 若综合结果仍不满意，可在属性表或 ArcGIS/QGIS 中按 `poi_category`、`point_symbol`、`point_symbol_color` 手工调符号。

## 如何确认真实 VLM 被调用

智能体现在会把以下内容传给 `vlm_verify` 回调：

- 综合前截图路径 `before_image`
- 综合后截图路径 `after_image`
- 点综合指标 `metrics`
- 八类符号规则 `symbol_context.poi_symbol_rules`
- 当前输出实际包含的类别 `symbol_context.poi_symbol_categories`

如果没有注册真实 VLM 回调，系统会明确返回：

- `vlm_mode = heuristic_fallback`
- `real_vlm_called = False`

如果要强制必须调用真实 VLM，启动前设置：

```powershell
$env:MAPGENAI_REQUIRE_REAL_VLM = "1"
```

此时若没有注册 `vlm_verify`，验收会失败，并提示 `real_vlm_required`。

## vlm_verify 回调接口

注册回调：

```python
from src.agent.tools import register_ui_callback

def my_vlm_verify(payload: dict) -> dict:
    before_image = payload["before_image"]
    after_image = payload["after_image"]
    symbol_context = payload.get("symbol_context", {})
    metrics = payload.get("metrics", {})

    # 在这里调用你的视觉语言模型，检查：
    # 1. 八类 POI 是否按规则出符号；
    # 2. 符号是否太大、是否遮挡；
    # 3. 综合前后结构是否保持。
    return {
        "pass": True,
        "score": 90,
        "failed_rules": [],
        "evidence": [
            f"checked before={before_image}",
            f"checked after={after_image}",
            f"categories={symbol_context.get('poi_symbol_categories', [])}",
        ],
    }

register_ui_callback("vlm_verify", my_vlm_verify)
```

只要回调返回包含 `pass` 的 dict，智能体会标记：

- `vlm_mode = callback`
- `real_vlm_called = True`

## 主要修改文件

- `src/ui/point_symbol_core.py`：新增八类 POI 符号规则、类别别名归一、固定符号映射。
- `src/ui/point_symbol_render.py`：原生画布新增八类 POI 图标绘制。
- `src/ui/main_window.py`：样式面板新增八类 POI 符号可选项。
- `src/agent/core.py`：点综合生成代码自动写入 `poi_category`、`point_symbol`、`point_symbol_color`、`point_symbol_label`；VLM payload 增加符号上下文。
- `src/agent/vlm_verifier.py`：增加真实 VLM 调用标记、强制真实 VLM 模式和 symbol_context 传递。
- `src/skills/point_generalization_dlm_dcm.py`：DLM/DCM 默认符号规则对齐八类 POI。

