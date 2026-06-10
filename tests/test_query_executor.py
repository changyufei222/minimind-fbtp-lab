from __future__ import annotations

import sys
from pathlib import Path


LAB_ROOT = Path(__file__).resolve().parents[1]
if str(LAB_ROOT) not in sys.path:
    sys.path.insert(0, str(LAB_ROOT))

from query_compiler.executor import execute_query_plan  # type: ignore  # noqa: E402


def test_execute_query_plan_filters_and_sorts_candidates() -> None:
    rows = [
        {
            "candidate_id": "A",
            "scaffold_type": "kunitz",
            "oral_class": "High",
            "best_affinity_value": 5.0,
            "engineered": True,
        },
        {
            "candidate_id": "B",
            "scaffold_type": "kunitz",
            "oral_class": "High",
            "best_affinity_value": 9.0,
            "engineered": True,
        },
        {
            "candidate_id": "C",
            "scaffold_type": "knottin",
            "oral_class": "High",
            "best_affinity_value": 3.0,
            "engineered": True,
        },
    ]
    plan = {
        "intent": "candidate_search",
        "entity": "protein_candidate",
        "filters": [
            {"field": "scaffold_type", "op": "eq", "value": "kunitz"},
            {"field": "oral_class", "op": "eq", "value": "High"},
        ],
        "sort": [{"field": "best_affinity_value", "direction": "asc"}],
        "limit": 1,
    }

    result = execute_query_plan(plan, rows)

    assert [row["candidate_id"] for row in result["rows"]] == ["A"]
    assert result["metadata"]["filtered_count"] == 2


def test_execute_query_plan_supports_boolean_filters() -> None:
    rows = [
        {"candidate_id": "A", "engineered": True, "has_experimental_affinity": True},
        {"candidate_id": "B", "engineered": False, "has_experimental_affinity": True},
    ]
    plan = {
        "intent": "candidate_search",
        "entity": "protein_candidate",
        "filters": [{"field": "engineered", "op": "eq", "value": True}],
        "sort": [],
        "limit": 10,
    }

    result = execute_query_plan(plan, rows)

    assert [row["candidate_id"] for row in result["rows"]] == ["A"]
