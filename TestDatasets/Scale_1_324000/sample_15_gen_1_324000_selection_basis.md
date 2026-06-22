# Point Generalization Selection Basis

## Task
- input: `E:\GradOpening\MapGenAI\TestDatasets\Scale_1_324000\sample_15.geojson`
- target_scale: `1:324000`
- feature_count: `1714`
- geometry_type: `Point`
- source_crs: `EPSG:4326`
- projected_crs: `EPSG:32649`

## Skill-Driven Analysis
- skill reference: `skills/point_generalization_skill.md`
- semantic_type: `commercial_poi_points`
- structure_type: `multi-core`
- semantic_evidence: ['sample values contain catering keywords', 'detected POI/business-like fields']
- structure_evidence: ['high nearest-neighbor CV=1.34', 'grid hot cells top2 share=0.53']
- candidate_fields: ['name', 'type', 'typecode', 'pname', 'cityname', 'adname']
- poi_symbol_rules: ['便利店', '学校', '停车场', '酒店', '银行', '医院', '餐饮', '交通设施']

## Operator Decision
- `selection` params={"keep_ratio": 0.066, "structure_type": "multi-core", "semantic_type": "commercial_poi_points"}
- `simplify` params={"tolerance_m": 0.0, "preserve_topology": true}
- `typification` params={"pattern": "multi_core_balanced"}
- `displacement` params={"min_sep_m": 97.2, "max_shift_m": 174.96, "strict_no_overlap": true, "strict_pixel_gap": 32.0, "strict_density_factor": 0.33}

## Algorithm Snippet
```python
score = semantic_score + structure_score + balance_score + sparse_region_bonus
selected = grouped_anchor_keep + region_quota_keep + hotspot_representative_keep + axis_bin_continuity_keep + grid_best_keep + top_score_fill
selected = resolve_conflicts_by_min_separation(selected)
```

## Validation (VLM + LLM)
- pass: False
- score: 25
- vlm_mode: heuristic_fallback
- real_vlm_called: False
- failed_rules: ['retention_rate >= class_min', 'point_sparse_region_coverage_ratio >= min', 'point_rgb_hotspot_overdraw <= max']
- evidence: ['retention_rate=0.032, min=0.120', 'point_axis_coverage_ratio=0.864, min=0.600', 'point_middle_gap_ratio=0.045, max=0.350', 'point_hotspot_rep_count=10, min=3', 'point_hotspot_coverage_ratio=0.625, min=0.500', 'point_region_coverage_ratio=0.724, min=0.720', 'point_sparse_region_coverage_ratio=0.571, min=0.850', 'point_dense_region_singleton_ratio=0.000, max=0.250', 'image_diff_ratio=0.0009', 'point_rgb_hotspot_overdraw=0.5365, max=0.2000']

## Outputs
- `E:\GradOpening\MapGenAI\TestDatasets\Scale_1_324000\sample_15_gen_1_324000_r4.geojson`
- `E:\GradOpening\MapGenAI\TestDatasets\Scale_1_324000\sample_15_gen_1_324000_r4.png`
- `E:\GradOpening\MapGenAI\TestDatasets\Scale_1_324000\sample_15_gen_1_324000_r4.pdf`
