from __future__ import annotations

from copy import deepcopy
from typing import Any

from .prompting import extract_first_json_object, render_plan_json
from .request_hints import infer_request_hints, semantic_mismatch_reasons
from .sketch import FILTER_SLOT_FIELDS, SKETCH_REQUIRED_KEYS, plan_to_sketch
from .validator import validate_query_draft


REPAIRABLE_ERROR_CODES = {
    "parse_failed",
    "invalid_draft_type",
    "invalid_slot_value",
    "missing_filters",
    "invalid_filters",
    "empty_filters",
    "invalid_sort",
    "invalid_limit",
    "semantic_mismatch_scaffold_type",
    "semantic_mismatch_oral_class",
    "semantic_mismatch_engineered",
    "semantic_mismatch_has_experimental_affinity",
    "semantic_mismatch_limit",
    "semantic_mismatch_sort",
}


def repair_reason_list(normalized: dict[str, Any]) -> list[str]:
    trace = normalized.get("trace", {})
    if not isinstance(trace, dict):
        return ["invalid_trace"]

    reasons: list[str] = []
    errors = trace.get("errors", [])
    if isinstance(errors, list):
        reasons.extend(str(item) for item in errors if str(item) in REPAIRABLE_ERROR_CODES)

    if trace.get("parsed") is False and "parse_failed" not in reasons:
        reasons.append("parse_failed")
    if trace.get("empty_filters") is True and "empty_filters" not in reasons:
        reasons.append("empty_filters")
    if trace.get("schema_ok") is False and not reasons:
        reasons.append("schema_failed")
    return reasons


def should_repair(normalized: dict[str, Any]) -> bool:
    return bool(repair_reason_list(normalized))


def _parse_and_validate(answer: str) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    parsed = extract_first_json_object(answer, required_keys=SKETCH_REQUIRED_KEYS)
    normalized = validate_query_draft(parsed, parsed=parsed is not None)
    return parsed, normalized


def _semantic_repair_reasons(normalized: dict[str, Any], original_request: str | None) -> list[str]:
    if not original_request:
        return []
    plan = normalized.get("plan", {})
    if not isinstance(plan, dict):
        return []
    return semantic_mismatch_reasons(plan, original_request)


def _hint_match_stats(plan: dict[str, Any], original_request: str | None) -> tuple[int, int, int]:
    if not original_request:
        return (0, 0, 0)

    hints = infer_request_hints(original_request)
    if not hints:
        return (0, 0, 0)

    filter_map = {item.get("field"): item.get("value") for item in plan.get("filters", [])}
    matched = 0
    mismatched = 0
    possible = 0

    for field in ("scaffold_type", "oral_class", "engineered", "has_experimental_affinity", "organism"):
        expected = hints.get(field)
        if expected is None:
            continue
        possible += 1
        if filter_map.get(field) == expected:
            matched += 1
        else:
            mismatched += 1

    expected_limit = hints.get("limit")
    if expected_limit is not None:
        possible += 1
        if plan.get("limit") == expected_limit:
            matched += 1
        else:
            mismatched += 1

    expected_sort_by = hints.get("sort_by")
    if expected_sort_by is not None:
        possible += 1
        sort_items = plan.get("sort", [])
        first_sort = sort_items[0] if sort_items else {}
        if first_sort.get("field") == expected_sort_by and first_sort.get("direction") == hints.get("sort_dir", "asc"):
            matched += 1
        else:
            mismatched += 1

    return matched, mismatched, possible


def _project_plan_from_request_hints(plan: dict[str, Any], original_request: str | None) -> tuple[dict[str, Any] | None, list[str]]:
    if not original_request or not isinstance(plan, dict):
        return None, []

    hints = infer_request_hints(original_request)
    if not hints:
        return None, []

    filter_map = {item.get("field"): item.get("value") for item in plan.get("filters", [])}
    projected_plan = deepcopy(plan)
    projection_notes: list[str] = []

    for field in ("scaffold_type", "oral_class", "engineered", "has_experimental_affinity", "organism"):
        expected = hints.get(field)
        if expected is None:
            continue
        if filter_map.get(field) != expected:
            filter_map[field] = expected
            projection_notes.append(f"semantic projection aligned {field} with request hints")

    expected_limit = hints.get("limit")
    if expected_limit is not None and projected_plan.get("limit") != expected_limit:
        projected_plan["limit"] = expected_limit
        projection_notes.append("semantic projection aligned limit with request hints")

    expected_sort_by = hints.get("sort_by")
    if expected_sort_by is not None:
        expected_sort_dir = hints.get("sort_dir", "asc")
        current_sort = projected_plan.get("sort", [])
        first_sort = current_sort[0] if current_sort else {}
        if first_sort.get("field") != expected_sort_by or first_sort.get("direction") != expected_sort_dir:
            projected_plan["sort"] = [{"field": expected_sort_by, "direction": expected_sort_dir}]
            projection_notes.append("semantic projection aligned sort with request hints")

    if not projection_notes:
        return None, []

    projected_plan["filters"] = [
        {"field": field, "op": "eq", "value": filter_map[field]}
        for field in FILTER_SLOT_FIELDS
        if filter_map.get(field) is not None
    ]
    projected_normalized = validate_query_draft(projected_plan, parsed=True)
    projected_trace = projected_normalized.get("trace", {})
    if isinstance(projected_trace, dict):
        projected_trace.setdefault("repairs", []).extend(projection_notes)
        projected_trace["semantic_projection"] = True
    return projected_normalized, projection_notes


def _candidate_quality(normalized: dict[str, Any], original_request: str | None) -> tuple[int, ...]:
    plan = normalized.get("plan", {})
    trace = normalized.get("trace", {})
    errors = trace.get("errors", []) if isinstance(trace, dict) else []
    if not isinstance(errors, list):
        errors = []

    filters = plan.get("filters", []) if isinstance(plan, dict) else []
    sort_items = plan.get("sort", []) if isinstance(plan, dict) else []
    matched, mismatched, possible = _hint_match_stats(plan if isinstance(plan, dict) else {}, original_request)

    return (
        1 if trace.get("parsed") else 0,
        1 if filters else 0,
        matched,
        -mismatched,
        1 if "invalid_slot_value" not in errors else 0,
        len(filters),
        1 if sort_items else 0,
        matched - mismatched if possible else 0,
        1 if not trace.get("semantic_projection") else 0,
        -len(errors),
    )


def finalize_prediction(first_answer: str, original_request: str | None = None, repaired_answer: str | None = None) -> dict[str, Any]:
    first_parsed, first_normalized = _parse_and_validate(first_answer)
    reasons = repair_reason_list(first_normalized)
    for item in _semantic_repair_reasons(first_normalized, original_request):
        if item not in reasons:
            reasons.append(item)

    repaired_parsed: dict[str, Any] | None = None
    repaired_normalized: dict[str, Any] | None = None
    projected_parsed: dict[str, Any] | None = None
    projected_normalized: dict[str, Any] | None = None
    repair_attempted = bool(reasons and repaired_answer is not None)
    used_repair = False
    projection_attempted = False
    used_projection = False
    projection_reasons: list[str] = []

    final_parsed = first_parsed
    final_normalized = first_normalized
    final_answer = first_answer

    if repair_attempted:
        repaired_parsed, repaired_normalized = _parse_and_validate(repaired_answer)
        first_quality = _candidate_quality(first_normalized, original_request)
        repaired_quality = _candidate_quality(repaired_normalized, original_request)
        if repaired_quality > first_quality:
            final_parsed = repaired_parsed
            final_normalized = repaired_normalized
            final_answer = repaired_answer
            used_repair = True

    projected_normalized, projection_reasons = _project_plan_from_request_hints(final_normalized["plan"], original_request)
    projection_attempted = projected_normalized is not None
    if projected_normalized is not None:
        final_quality = _candidate_quality(final_normalized, original_request)
        projected_quality = _candidate_quality(projected_normalized, original_request)
        if projected_quality > final_quality:
            projected_parsed = plan_to_sketch(projected_normalized["plan"])
            final_parsed = projected_parsed
            final_normalized = projected_normalized
            final_answer = render_plan_json(projected_parsed)
            used_projection = True

    return {
        "repair_attempted": repair_attempted,
        "used_repair": used_repair,
        "repair_reasons": reasons,
        "projection_attempted": projection_attempted,
        "used_projection": used_projection,
        "projection_reasons": projection_reasons,
        "first_answer": first_answer,
        "first_parsed_draft": first_parsed,
        "first_normalized": first_normalized,
        "repaired_answer": repaired_answer,
        "repaired_parsed_draft": repaired_parsed,
        "repaired_normalized": repaired_normalized,
        "projected_parsed_draft": projected_parsed,
        "projected_normalized": projected_normalized,
        "final_answer": final_answer,
        "final_parsed_draft": final_parsed,
        "final_plan": final_normalized["plan"],
        "final_trace": final_normalized["trace"],
    }
