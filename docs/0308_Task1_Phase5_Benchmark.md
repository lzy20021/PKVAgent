# 0308 Task1 Phase 5：性能报告与压测入口

## GUI 导出（JSON/CSV）
- 打开地图画布下方的“性能面板”
- 点击 `导出JSON` 或 `导出CSV`
- 文件包含渲染历史、耗时、内存与缓存指标

## 压测脚本入口

### 快速压测（推荐先跑）
```bash
python tests/benchmark_render_phase5.py --profile quick --rounds 3
```

### 全量压测（按规划目标规模）
```bash
python tests/benchmark_render_phase5.py --profile full --rounds 3
```

### 输出目录
- 默认输出到 `tests/tmp/bench/`
- 会生成：
  - `phase5_benchmark_<profile>_<timestamp>.json`
  - `phase5_benchmark_<profile>_<timestamp>.csv`

## 验收建议
- `single_100k`：平均耗时目标 `<100ms`
- `multi_20_500k`：平均耗时目标 `<300ms`
- 报告中 `pass_target=true` 视为单项通过
