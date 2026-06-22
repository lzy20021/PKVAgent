# Point Generalization Selection Basis

## Task
- input: `E:\GradOpening\MapGenAI\output\pointgenai_process_probe_250k\normalized_input.geojson`
- target_scale: `1:250000`
- feature_count: `835`
- geometry_type: `Point`
- source_crs: `EPSG:4326`
- projected_crs: `EPSG:32650`

## Skill-Driven Analysis
- skill reference: `skills/point_generalization_skill.md`
- semantic_type: `commercial_poi_points`
- structure_type: `center-periphery`
- semantic_evidence: ['detected POI/business-like fields']
- structure_evidence: ['high nearest-neighbor CV=0.94', 'center share=0.41']
- candidate_fields: ['name', 'type', 'typecode', 'pname', 'cityname', 'adname']
- poi_symbol_rules: ['便利店', '学校', '停车场', '酒店', '银行', '医院', '餐饮', '交通设施']

## Operator Decision
- `selection` params={"keep_ratio": 0.588, "structure_type": "center-periphery", "semantic_type": "commercial_poi_points"}
- `simplify` params={"tolerance_m": 0.0, "preserve_topology": true}
- `typification` params={"pattern": "center_periphery_balanced"}
- `displacement` params={"min_sep_m": 75.0, "max_shift_m": 135.0, "strict_no_overlap": true, "strict_pixel_gap": 32, "strict_density_factor": 0.5}

## Algorithm Snippet
```python
score = semantic_score + structure_score + balance_score + sparse_region_bonus
selected = grouped_anchor_keep + region_quota_keep + hotspot_representative_keep + axis_bin_continuity_keep + grid_best_keep + top_score_fill
selected = resolve_conflicts_by_min_separation(selected)
```

## Validation (VLM + LLM)
- pass: True
- score: 92
- vlm_mode: local_callback_rule_and_image_gate
- real_vlm_called: True
- failed_rules: []
- evidence: ['local vlm_verify callback received before/after image paths', 'before_image=', 'after_image=E:\\GradOpening\\MapGenAI\\output\\pointgenai_process_probe_250k\\scale_1_250000\\poi8_gen_1_250000.png', "symbol_categories=['交通设施', '便利店', '停车场', '医院', '学校', '酒店', '银行', '餐饮']"]

## Outputs
- `E:\GradOpening\MapGenAI\output\pointgenai_process_probe_250k\scale_1_250000\poi8_gen_1_250000.geojson`
- `E:\GradOpening\MapGenAI\output\pointgenai_process_probe_250k\scale_1_250000\poi8_gen_1_250000.png`
- `E:\GradOpening\MapGenAI\output\pointgenai_process_probe_250k\scale_1_250000\poi8_gen_1_250000.pdf`
