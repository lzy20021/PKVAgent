# Point Generalization Selection Basis

## Task
- input: `E:\GradOpening\MapGenAI\output\classic_symbol_figures\agent_intermediate\大同市_平城区_1_532000_before.geojson`
- target_scale: `1:532000`
- feature_count: `320`
- geometry_type: `Point`
- source_crs: `EPSG:4326`
- projected_crs: `EPSG:32649`

## Skill-Driven Analysis
- skill reference: `skills/point_generalization_skill.md`
- semantic_type: `commercial_poi_points`
- structure_type: `mixed`
- semantic_evidence: ['sample values contain catering keywords', 'detected POI/business-like fields']
- structure_evidence: ['moderate nearest-neighbor CV=0.64']
- candidate_fields: ['name', 'type']

## Operator Decision
- `selection` params={"keep_ratio": 0.02, "structure_type": "mixed", "semantic_type": "commercial_poi_points"}
- `simplify` params={"tolerance_m": 0.0, "preserve_topology": true}
- `typification` params={"pattern": "spatial_even"}
- `displacement` params={"min_sep_m": 159.6, "max_shift_m": 287.28, "strict_no_overlap": true, "strict_pixel_gap": 32.0, "strict_density_factor": 0.55}

## Algorithm Snippet
```python
score = semantic_score + structure_score + balance_score + sparse_region_bonus
selected = grouped_anchor_keep + region_quota_keep + hotspot_representative_keep + axis_bin_continuity_keep + grid_best_keep + top_score_fill
selected = resolve_conflicts_by_min_separation(selected)
```

## Validation (VLM + LLM)
- pass: False
- score: 73
- failed_rules: ['conflict_count_total <= target_conflict_count', 'point_sparse_region_coverage_ratio >= min']
- evidence: ['conflict=3, target=0', 'point_axis_coverage_ratio=0.893, min=0.600', 'point_middle_gap_ratio=0.036, max=0.350', 'point_hotspot_rep_count=15, min=3', 'point_hotspot_coverage_ratio=0.778, min=0.500', 'point_region_coverage_ratio=0.840, min=0.720', 'point_sparse_region_coverage_ratio=0.643, min=0.850', 'point_dense_region_singleton_ratio=0.000, max=0.250']

## Outputs
- `E:\GradOpening\MapGenAI\output\classic_symbol_figures\agent_intermediate\大同市_平城区_1_532000_after.geojson`
