# PointGenAI Iterative Process Figure

## Outputs

- `PointGenAI_iterative_process.svg`: editable vector layout.
- `PointGenAI_iterative_process.png`: 600 dpi preview.
- `PointGenAI_iterative_process.tif`: 600 dpi TIFF preview.
- `PointGenAI_iterative_process_caption.txt`: Chinese/English title and caption.

## Data Source

- Sample area: `TestDatasets/Scale_1_86000/sample_13.geojson`.
- Iteration sequence: `output/pointgenai_process_sequence_sample13_1_250000/`.
- Stage layers: `R0_input.geojson`, `R1_iter1.geojson`, `R2_iter2.geojson`, `R3_final.geojson`.
- Target scale: `1:250,000`.

## Symbol Source

- Category rules: `src/ui/point_symbol_core.py::POI_SYMBOL_RULES`.
- Symbol rendering logic: `src/ui/point_symbol_render.py` and the existing classic-symbol drawing style used by `scripts/render_classic_symbol_two_figures.py`.
- Raster symbol reference from step 2: `output/arcgis_10_2_poi_symbols/`.
- No new POI category, color, icon, or category style was introduced.

## Per-Round Strategy

### R0 任务解析与问题识别

- Skills: 符号层级配置、符号压盖校核、视觉可读性评估
- point_count: 904
- conflict_pairs: 802
- conflict_resolution_rate: 0.0000
- visual_readability: 35
- constraint_satisfaction: 0.4875
- pattern_preservation: 1.0000

### R1 初始策略生成与 Skills 调用

- Skills: 重要点优先保留、比例尺驱动删减、密集区点筛选、聚类中心保持、多核结构保持、符号层级配置、视觉可读性评估
- point_count: 232
- conflict_pairs: 0
- conflict_resolution_rate: 1.0000
- visual_readability: 75
- constraint_satisfaction: 0.9313
- pattern_preservation: 0.9730

### R2 VLM 反馈驱动的冲突消解与均衡优化

- Skills: 邻近冲突检测、邻近冲突消解、均衡优化、稀疏格局保持、符号压盖校核、视觉可读性评估、结果反馈修正
- point_count: 182
- conflict_pairs: 0
- conflict_resolution_rate: 1.0000
- visual_readability: 75
- constraint_satisfaction: 0.9281
- pattern_preservation: 0.9534

### R3 视觉可读性评估与最终校核

- Skills: 视觉可读性评估、结果反馈修正、符号压盖校核、稀疏格局保持、均衡优化
- point_count: 144
- conflict_pairs: 0
- conflict_resolution_rate: 1.0000
- visual_readability: 75
- constraint_satisfaction: 0.9313
- pattern_preservation: 0.9573

## Generation Script

- Figure script: `scripts/make_pointgenai_iterative_process_figure.py`.
- Input sequence script from step 2: `scripts/export_pointgenai_process_sequence.py`.
- Metrics table: `output/pointgenai_process_sequence_sample13_1_250000/iteration_metrics.csv`.
