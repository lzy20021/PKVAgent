# PointGenAI 迭代过程实验序列说明

- 样本区域: `E:\GradOpening\MapGenAI\TestDatasets\Scale_1_86000\sample_13.geojson`
- 智能体输入: `E:\GradOpening\MapGenAI\output\pointgenai_process_sequence_sample13_1_250000\agent_input_sample13_8class.geojson`
- 目标比例尺: `1:250000`
- 比例尺选择说明: 本次正式输出采用优先目标比例尺 1:250,000。运行时未注册宽松本地 VLM 回调，使用项目 `src/agent/vlm_verifier.py` 的质量评价路径，因此 PointGenAI 生成了可审计的多轮反馈序列。
- 符号来源: `src/ui/point_symbol_core.py`、`src/ui/point_symbol_render.py` 和 `output/arcgis_10_2_poi_symbols/` 中已有 8 类 POI 符号。
- 未新增 POI 类别、颜色、图标或类别样式。

## 阶段文件

| 阶段 | 地图 | 点图层 | 日志 |
|---|---|---|---|
| R0 | `R0_input.png` / `R0_input.svg` | `R0_input.geojson` / `R0_input.csv` | `R0_input_agent_log.md` |
| R1 | `R1_iter1.png` / `R1_iter1.svg` | `R1_iter1.geojson` / `R1_iter1.csv` | `R1_iter1_agent_log.md` |
| R2 | `R2_iter2.png` / `R2_iter2.svg` | `R2_iter2.geojson` / `R2_iter2.csv` | `R2_iter2_agent_log.md` |
| R3 | `R3_final.png` / `R3_final.svg` | `R3_final.geojson` / `R3_final.csv` | `R3_final_agent_log.md` |

## 关键指标

详见 `iteration_metrics.csv`。

## 智能体原始记录

- `agent_reply.txt`: PointGenAI 返回文本。
- `agent_events.json`: 事件、工具调用和每轮 stdout 指标摘要。
- `agent_internal_rounds.json`: 智能体内部 R1-R4 输出路径。
