import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.generalization_planner import GeneralizationPlanner


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a generalization operator plan.")
    parser.add_argument("--source-scale", type=int, default=10000)
    parser.add_argument("--target-scale", type=int, required=True)
    parser.add_argument("--geometry", type=str, default="line", choices=["point", "line", "polygon"])
    parser.add_argument("--feature-count", type=int, default=10000)
    parser.add_argument("--conflicts", type=int, default=120)
    parser.add_argument("--conflict-density", type=float, default=18.0)
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--rules", type=str, default="knowledge_base/generalization_rules.yaml")
    args = parser.parse_args()

    planner = GeneralizationPlanner(rules_path=args.rules)
    payload = {
        "source_scale": args.source_scale,
        "target_scale": args.target_scale,
        "round": args.round,
        "layer_stats": {
            "geometry_type": args.geometry,
            "feature_count": args.feature_count,
        },
        "qa_metrics": {
            "conflict_count_total": args.conflicts,
            "conflict_density_per_km2": args.conflict_density,
            "topology_error_count": 0,
        },
    }
    plan = planner.plan_from_dict(payload)
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
