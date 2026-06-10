from __future__ import annotations

from typing import Any

from .dsl import DEFAULT_ENTITY, DEFAULT_INTENT, DEFAULT_LIMIT, DEFAULT_RETURN_FIELDS, LEGAL_OPERATORS
from .field_registry import canonicalize_field, normalize_operator, normalize_sort_direction, normalize_value
from .sketch import build_plan_from_sketch, is_sketch_draft


def _normalize_limit(value: Any, repairs: list[str], errors: list[str]) -> int:
    if value is None:
        return DEFAULT_LIMIT
    try:
        normalized = int(value)
    except (TypeError, ValueError):
        repairs.append(f"invalid limit {value!r}; using default {DEFAULT_LIMIT}")
        errors.append("invalid_limit")
        return DEFAULT_LIMIT
    if normalized <= 0:
        repairs.append(f"non-positive limit {normalized}; using default {DEFAULT_LIMIT}")
        errors.append("invalid_limit")
        return DEFAULT_LIMIT
    return normalized


def _normalize_filters(filters: Any, repairs: list[str], errors: list[str]) -> list[dict[str, Any]]:
    normalized_filters: list[dict[str, Any]] = []
    if filters is None:
        errors.append("missing_filters")
        return normalized_filters
    if not isinstance(filters, list):
        repairs.append("filters was not a list; dropped invalid filter block")
        errors.append("invalid_filters")
        return normalized_filters

    for item in filters:
        if not isinstance(item, dict):
            repairs.append("dropped non-dict filter entry")
            continue

        original_field = item.get("field")
        field = canonicalize_field(original_field)
        if field is None:
            repairs.append(f"dropped unsupported field {original_field!r}")
            continue

        op = normalize_operator(item.get("op", "eq"))
        if op is None or op not in LEGAL_OPERATORS:
            repairs.append(f"dropped unsupported operator {item.get('op')!r} for field {field}")
            continue

        value = normalize_value(field, item.get("value"))
        if op != "exists" and value is None:
            repairs.append(f"dropped filter for {field} because value could not be normalized")
            continue

        normalized_filters.append({"field": field, "op": op, "value": value})

        if field != original_field or op != item.get("op") or value != item.get("value"):
            repairs.append(f"normalized filter {original_field!r} -> {field}")

    return normalized_filters


def _normalize_sort(sort: Any, repairs: list[str], errors: list[str]) -> list[dict[str, Any]]:
    normalized_sort: list[dict[str, Any]] = []
    if not isinstance(sort, list):
        if sort is not None:
            repairs.append("sort was not a list; dropped invalid sort block")
            errors.append("invalid_sort")
        return normalized_sort

    for item in sort:
        if not isinstance(item, dict):
            repairs.append("dropped non-dict sort entry")
            continue
        original_field = item.get("field")
        field = canonicalize_field(original_field)
        if field is None:
            repairs.append(f"dropped unsupported sort field {original_field!r}")
            continue
        direction = normalize_sort_direction(field, item.get("direction", "asc"))
        if direction is None:
            repairs.append(f"dropped unsupported sort direction {item.get('direction')!r} for field {field}")
            continue
        normalized_sort.append({"field": field, "direction": direction})
        if field != original_field or direction != item.get("direction"):
            repairs.append(f"normalized sort {original_field!r} -> {field}")
    return normalized_sort


def validate_query_draft(draft: dict[str, Any] | None, parsed: bool = True) -> dict[str, Any]:
    repairs: list[str] = []
    errors: list[str] = []
    if not parsed:
        errors.append("parse_failed")

    if draft is None:
        draft = {}
    if not isinstance(draft, dict):
        repairs.append("draft was not an object; using empty draft")
        errors.append("invalid_draft_type")
        draft = {}

    if is_sketch_draft(draft):
        normalized_plan = build_plan_from_sketch(draft, repairs, errors)
        normalized_plan["limit"] = _normalize_limit(normalized_plan.get("limit"), repairs, errors)
        normalized_filters = normalized_plan["filters"]
        normalized_sort = normalized_plan["sort"]
    else:
        normalized_filters = _normalize_filters(draft.get("filters"), repairs, errors)
        normalized_sort = _normalize_sort(draft.get("sort", []), repairs, errors)
        normalized_plan = {
            "intent": DEFAULT_INTENT,
            "entity": DEFAULT_ENTITY,
            "filters": normalized_filters,
            "sort": normalized_sort,
            "limit": _normalize_limit(draft.get("limit"), repairs, errors),
            "return_fields": list(DEFAULT_RETURN_FIELDS),
        }

    if not normalized_filters:
        errors.append("empty_filters")
    deduped_errors = list(dict.fromkeys(errors))
    trace = {
        "input": draft,
        "parsed": parsed,
        "schema_ok": not deduped_errors,
        "errors": deduped_errors,
        "empty_filters": not normalized_filters,
        "repairs": repairs,
        "draft_format": "sketch" if is_sketch_draft(draft) else "plan",
    }
    return {"plan": normalized_plan, "trace": trace}
