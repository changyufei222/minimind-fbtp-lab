from __future__ import annotations

import json
from typing import Any

from .executor import execute_query_plan


def _json_value(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _slot_set(plan: dict[str, Any]) -> set[tuple[str, ...]]:
    slots: set[tuple[str, ...]] = {
        ("intent", str(plan.get("intent"))),
        ("entity", str(plan.get("entity"))),
        ("limit", str(plan.get("limit"))),
    }
    for flt in plan.get("filters", []):
        slots.add(("filter", str(flt.get("field")), str(flt.get("op")), _json_value(flt.get("value"))))
    for sort in plan.get("sort", []):
        slots.add(("sort", str(sort.get("field")), str(sort.get("direction"))))
    return slots


def _slot_accuracy(predicted_plan: dict[str, Any], gold_plan: dict[str, Any]) -> float:
    gold_slots = _slot_set(gold_plan)
    if not gold_slots:
        return 1.0
    predicted_slots = _slot_set(predicted_plan)
    return round(len(gold_slots & predicted_slots) / len(gold_slots), 4)


def _filter_set(plan: dict[str, Any]) -> set[tuple[str, ...]]:
    filters: set[tuple[str, ...]] = set()
    for flt in plan.get("filters", []):
        filters.add((str(flt.get("field")), str(flt.get("op")), _json_value(flt.get("value"))))
    return filters


def _field_value_exact_match(predicted_plan: dict[str, Any], gold_plan: dict[str, Any]) -> float:
    gold_filters = _filter_set(gold_plan)
    if not gold_filters:
        return 1.0 if not _filter_set(predicted_plan) else 0.0
    predicted_filters = _filter_set(predicted_plan)
    return round(len(gold_filters & predicted_filters) / len(gold_filters), 4)


def score_prediction(
    predicted_plan: dict[str, Any] | None,
    gold_plan: dict[str, Any],
    rows: list[dict[str, Any]],
    json_parsed: bool | None = None,
) -> dict[str, Any]:
    json_parsed = predicted_plan is not None if json_parsed is None else json_parsed
    draft_valid = isinstance(predicted_plan, dict) and bool(predicted_plan.get("intent")) and bool(predicted_plan.get("entity"))
    if not draft_valid:
        return {
            "draft_valid": False,
            "json_parsed": bool(json_parsed),
            "non_empty_filter": False,
            "field_value_exact_match": 0.0,
            "slot_accuracy": 0.0,
            "execution_success": False,
            "result_overlap_at_k": 0.0,
        }

    slot_accuracy = _slot_accuracy(predicted_plan, gold_plan)
    non_empty_filter = bool(predicted_plan.get("filters"))
    field_value_exact_match = _field_value_exact_match(predicted_plan, gold_plan)
    try:
        predicted_result = execute_query_plan(predicted_plan, rows)
        gold_result = execute_query_plan(gold_plan, rows)
        predicted_ids = {row["candidate_id"] for row in predicted_result["rows"]}
        gold_ids = {row["candidate_id"] for row in gold_result["rows"]}
        if not gold_ids:
            overlap = 1.0 if not predicted_ids else 0.0
        else:
            overlap = round(len(predicted_ids & gold_ids) / len(gold_ids), 4)
        execution_success = True
    except Exception:
        overlap = 0.0
        execution_success = False

    return {
        "draft_valid": True,
        "json_parsed": bool(json_parsed),
        "non_empty_filter": non_empty_filter,
        "field_value_exact_match": field_value_exact_match,
        "slot_accuracy": slot_accuracy,
        "execution_success": execution_success,
        "result_overlap_at_k": overlap,
    }


def summarize_scores(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {
            "plan_valid_rate": 0.0,
            "json_parse_rate": 0.0,
            "non_empty_filter_rate": 0.0,
            "field_value_exact_match": 0.0,
            "slot_accuracy": 0.0,
            "execution_success_rate": 0.0,
            "result_overlap_at_k": 0.0,
        }

    total = len(rows)
    return {
        "plan_valid_rate": round(sum(1 for row in rows if row["draft_valid"]) / total, 4),
        "json_parse_rate": round(sum(1 for row in rows if row.get("json_parsed")) / total, 4),
        "non_empty_filter_rate": round(sum(1 for row in rows if row.get("non_empty_filter")) / total, 4),
        "field_value_exact_match": round(sum(float(row.get("field_value_exact_match", 0.0)) for row in rows) / total, 4),
        "slot_accuracy": round(sum(float(row["slot_accuracy"]) for row in rows) / total, 4),
        "execution_success_rate": round(sum(1 for row in rows if row["execution_success"]) / total, 4),
        "result_overlap_at_k": round(sum(float(row["result_overlap_at_k"]) for row in rows) / total, 4),
    }
