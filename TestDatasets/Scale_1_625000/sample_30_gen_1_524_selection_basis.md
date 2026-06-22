# Point Generalization Selection Basis

## Task
- input: `E:\GradOpening\MapGenAI\TestDatasets\Scale_1_625000\sample_30.geojson`
- target_scale: `1:524`
- feature_count: `1651`
- geometry_type: `Point`
- source_crs: `EPSG:4326`
- projected_crs: `EPSG:32648`

## Skill-Driven Analysis
- skill reference: `skills/point_generalization_skill.md`
- semantic_type: `commercial_poi_points`
- structure_type: `clustered`
- semantic_evidence: ['sample values contain fuel or charging keywords', 'detected POI/business-like fields']
- structure_evidence: ['high nearest-neighbor CV=1.57']
- candidate_fields: ['name', 'type', 'typecode', 'pname', 'cityname', 'adname']
- poi_symbol_rules: ['便利店', '学校', '停车场', '酒店', '银行', '医院', '餐饮', '交通设施']

## Operator Decision
- `selection` params={"keep_ratio": 0.52, "structure_type": "clustered", "semantic_type": "commercial_poi_points"}
- `simplify` params={"tolerance_m": 0.0, "preserve_topology": true}
- `typification` params={"pattern": "cluster_representative"}
- `displacement` params={"min_sep_m": 0.157, "max_shift_m": 0.283, "strict_no_overlap": false, "strict_pixel_gap": 0, "strict_density_factor": 1.0}

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
- evidence: ['point_axis_coverage_ratio=1.000, min=0.600', 'point_middle_gap_ratio=0.000, max=0.350', 'point_hotspot_rep_count=386, min=3', 'point_hotspot_coverage_ratio=1.000, min=0.500', 'point_region_coverage_ratio=1.000, min=0.720', 'point_sparse_region_coverage_ratio=1.000, min=0.850', 'point_dense_region_singleton_ratio=0.000, max=0.250', 'image_diff_ratio=0.0003', 'point_rgb_hotspot_overdraw=0.8646, max=0.2000']

## Outputs
- `E:\GradOpening\MapGenAI\TestDatasets\Scale_1_625000\sample_30_gen_1_524_r3.geojson`
- `E:\GradOpening\MapGenAI\TestDatasets\Scale_1_625000\sample_30_gen_1_524_r3.png`
- `E:\GradOpening\MapGenAI\TestDatasets\Scale_1_625000\sample_30_gen_1_524_r3.pdf`
