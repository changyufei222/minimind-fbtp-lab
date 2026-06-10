from __future__ import annotations

import sys
from pathlib import Path


LAB_ROOT = Path(__file__).resolve().parents[1]
if str(LAB_ROOT) not in sys.path:
    sys.path.insert(0, str(LAB_ROOT))

from query_compiler.scoring import score_prediction, summarize_scores  # type: ignore  # noqa: E402


def test_score_query_compiler_prediction_reports_core_metrics() -> None:
    gold_plan = {
        "intent": "candidate_search",
        "entity": "protein_candidate",
        "filters": [{"field": "engineered", "op": "eq", "value": True}],
        "sort": [{"field": "best_affinity_value", "direction": "asc"}],
        "limit": 1,
    }
    predicted_plan = {
        "intent": "candidate_search",
        "entity": "protein_candidate",
        "filters": [{"field": "engineered", "op": "eq", "value": True}],
        "sort": [{"field": "best_affinity_value", "direction": "asc"}],
        "limit": 1,
    }
    rows = [
        {"candidate_id": "A", "engineered": True, "best_affinity_value": 4.0},
        {"candidate_id": "B", "engineered": False, "best_affinity_value": 1.0},
    ]

    result = score_prediction(predicted_plan, gold_plan, rows)

    assert result["draft_valid"] is True
    assert result["json_parsed"] is True
    assert result["non_empty_filter"] is True
    assert result["field_value_exact_match"] == 1.0
    assert result["slot_accuracy"] == 1.0
    assert result["execution_success"] is True
    assert result["result_overlap_at_k"] == 1.0


def test_summarize_scores_aggregates_metric_rates() -> None:
    rows = [
        {
            "draft_valid": True,
            "json_parsed": True,
            "non_empty_filter": True,
            "field_value_exact_match": 1.0,
            "slot_accuracy": 1.0,
            "execution_success": True,
            "result_overlap_at_k": 1.0,
        },
        {
            "draft_valid": False,
            "json_parsed": False,
            "non_empty_filter": False,
            "field_value_exact_match": 0.0,
            "slot_accuracy": 0.5,
            "execution_success": False,
            "result_overlap_at_k": 0.0,
        },
    ]

    summary = summarize_scores(rows)

    assert summary["plan_valid_rate"] == 0.5
    assert summary["json_parse_rate"] == 0.5
    assert summary["non_empty_filter_rate"] == 0.5
    assert summary["field_value_exact_match"] == 0.5
    assert summary["slot_accuracy"] == 0.75
    assert summary["execution_success_rate"] == 0.5
    assert summary["result_overlap_at_k"] == 0.5
