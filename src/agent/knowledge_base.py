from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class AlgorithmKnowledgeBase:
    """Load/search GIS algorithm templates from local knowledge base."""

    def __init__(self, kb_dir: str = "knowledge_base"):
        self.kb_dir = Path(kb_dir)
        self.algorithms: List[Dict[str, Any]] = []
        self.load_algorithms()

    def load_algorithms(self) -> List[Dict[str, Any]]:
        """Reload and return all algorithm definitions."""
        loaded: List[Dict[str, Any]] = []
        algorithms_dir = self.kb_dir / "algorithms"
        if not algorithms_dir.exists():
            self.algorithms = loaded
            return loaded

        for json_file in sorted(algorithms_dir.rglob("*.json")):
            try:
                payload = json.loads(json_file.read_text(encoding="utf-8"))
            except Exception:
                continue

            if not isinstance(payload, dict):
                continue

            payload["_path"] = str(json_file)
            loaded.append(payload)

        self.algorithms = loaded
        return loaded

    def search(
        self,
        task_type: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Search algorithms by task_type and free keywords (tags/name/description)."""
        norm_task = (task_type or "").strip().lower()
        norm_keywords = [k.strip().lower() for k in (keywords or []) if str(k).strip()]

        scored: List[tuple[int, Dict[str, Any]]] = []
        for algo in self.algorithms:
            score = 0
            algo_task = str(algo.get("task_type", "")).lower()
            name = str(algo.get("name", "")).lower()
            desc = str(algo.get("description", "")).lower()
            tags = [str(t).lower() for t in algo.get("tags", [])]

            if norm_task:
                if algo_task != norm_task:
                    continue
                score += 10

            for kw in norm_keywords:
                if kw in tags:
                    score += 5
                if kw in name:
                    score += 3
                if kw in desc:
                    score += 2

            # If only keywords were provided, require at least one hit.
            if not norm_task and norm_keywords and score == 0:
                continue

            scored.append((score, algo))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in scored[: max(1, limit)]]

    def get_algorithm(self, algorithm_name: str) -> Optional[Dict[str, Any]]:
        """Get algorithm by identifier (`algorithm`) or human-readable `name`."""
        target = (algorithm_name or "").strip().lower()
        if not target:
            return None

        for algo in self.algorithms:
            if str(algo.get("algorithm", "")).lower() == target:
                return algo
            if str(algo.get("name", "")).lower() == target:
                return algo
        return None

    def get_template(self, algorithm_name: str) -> str:
        """Return code template for an algorithm; empty string if missing."""
        algo = self.get_algorithm(algorithm_name)
        return str(algo.get("code_template", "")) if algo else ""

    def list_task_types(self) -> List[str]:
        """Return sorted task types available in current KB."""
        return sorted({str(a.get("task_type", "")) for a in self.algorithms if a.get("task_type")})

    def list_algorithms(self) -> List[Dict[str, Any]]:
        """Return brief metadata for all algorithms."""
        return [
            {
                "name": algo.get("name", ""),
                "algorithm": algo.get("algorithm", ""),
                "task_type": algo.get("task_type", ""),
                "description": algo.get("description", ""),
            }
            for algo in self.algorithms
        ]

    # Backward-compatible helper used by some existing call sites.
    def get_examples(self, task_type: str, limit: int = 3) -> List[Dict[str, Any]]:
        examples: List[Dict[str, Any]] = []
        for algo in self.search(task_type=task_type, limit=100):
            for item in algo.get("examples", []):
                if isinstance(item, dict):
                    examples.append(item)
                    if len(examples) >= limit:
                        return examples
        return examples
