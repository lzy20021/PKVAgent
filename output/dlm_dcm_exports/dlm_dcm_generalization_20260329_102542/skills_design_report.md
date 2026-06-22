# DLM / DCM Skills Design Report

## DLM 层
- `representative_poi_retain`: 以类别稀有度、中心性、局部代表性构造代理重要性。
- `structure_aware_selection`: 用聚类和 PCA 判断 clustered / linear / sparse / multi-core。
- `category_balance_preserve`: 在局部网格先保每类代表点，避免单一类别塌缩。

## DCM 层
- `symbol_size_aware_generalization`: 把符号尺寸和最小间距换算为地面冲突阈值。
- `cross_category_conflict_resolution`: 在异类冲突中保留 minority 类别并允许位移。
- `layout_aesthetic_optimization`: 对局部空白和偏斜做补点与轻量重排。

## DLM -> DCM
- DLM 输出结构标签、重要性和局部类别多样性目标。
- DCM 基于这些变量完成符号保留、冲突消解与图面优化。