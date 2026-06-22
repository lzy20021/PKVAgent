# Point Generalization Selection Basis

## Task
- input: `E:\GradOpening\MapGenAI\TestDatasets\Scale_1_86000\sample_13.geojson`
- target_scale: `1:50000`
- feature_count: `904`
- geometry_type: `Point`
- source_crs: `EPSG:4326`
- projected_crs: `EPSG:32650`

## Skill-Driven Analysis
- skill reference: `skills/point_generalization_skill.md`
- semantic_type: `commercial_poi_points`
- structure_type: `center-periphery`
- semantic_evidence: ['detected POI/business-like fields']
- structure_evidence: ['high nearest-neighbor CV=0.96', 'center share=0.40']
- candidate_fields: ['name', 'type', 'typecode', 'pname', 'cityname', 'adname']
- poi_symbol_rules: ['便利店', '学校', '停车场', '酒店', '银行', '医院', '餐饮', '交通设施']

## Operator Decision
- `selection` params={"keep_ratio": 0.438, "structure_type": "center-periphery", "semantic_type": "commercial_poi_points"}
- `simplify` params={"tolerance_m": 0.0, "preserve_topology": true}
- `typification` params={"pattern": "center_periphery_balanced"}
- `displacement` params={"min_sep_m": 15.0, "max_shift_m": 27.0, "strict_no_overlap": false, "strict_pixel_gap": 0, "strict_density_factor": 1.0}

## Algorithm Snippet
```python
score = semantic_score + structure_score + balance_score + sparse_region_bonus
selected = grouped_anchor_keep + region_quota_keep + hotspot_representative_keep + axis_bin_continuity_keep + grid_best_keep + top_score_fill
selected = resolve_conflicts_by_min_separation(selected)
```

## Validation (VLM + LLM)
- pass: False
- score: 75
- vlm_mode: heuristic_fallback
- real_vlm_called: False
- failed_rules: ['point_rgb_hotspot_overdraw <= max']
- evidence: ['point_axis_coverage_ratio=1.000, min=0.600', 'point_middle_gap_ratio=0.000, max=0.350', 'point_hotspot_rep_count=141, min=3', 'point_hotspot_coverage_ratio=1.000, min=0.500', 'point_region_coverage_ratio=1.000, min=0.720', 'point_sparse_region_coverage_ratio=1.000, min=0.850', 'point_dense_region_singleton_ratio=0.000, max=0.250', 'image_diff_ratio=0.0005', 'point_rgb_hotspot_overdraw=0.7882, max=0.2000']

## Outputs
- `E:\GradOpening\MapGenAI\TestDatasets\Scale_1_86000\sample_13_gen_1_50000_r2.geojson`
- `E:\GradOpening\MapGenAI\TestDatasets\Scale_1_86000\sample_13_gen_1_50000_r2.png`
- `E:\GradOpening\MapGenAI\TestDatasets\Scale_1_86000\sample_13_gen_1_50000_r2.pdf`
