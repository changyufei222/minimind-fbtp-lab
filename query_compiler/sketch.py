from __future__ import annotations

from typing import Any

from .dsl import DEFAULT_ENTITY, DEFAULT_INTENT, DEFAULT_LIMIT, DEFAULT_RETURN_FIELDS
from .field_registry import canonicalize_field, normalize_sort_direction, normalize_value


FILTER_SLOT_FIELDS = (
    "candidate_id",
    "scaffold_type",
    "oral_class",
    "engineered",
    "has_experimental_affinity",
    "organism",
)
SKETCH_FIELDS = FILTER_SLOT_FIELDS + (
    "sort_by",
    "sort_dir",
    "limit",
)
SKETCH_REQUIRED_KEYS = {"limit"}
NULL_LIKE_VALUES = {"", "null", "none", "n/a", "na"}


def empty_sketch() -> dict[str, Any]:
    sketch = {field: None for field in SKETCH_FIELDS}
    sketch["limit"] = DEFAULT_LIMIT
    return sketch


def plan_to_sketch(plan: dict[str, Any]) -> dict[str, Any]:
    sketch = empty_sketch()
    sketch["limit"] = plan.get("limit", DEFAULT_LIMIT)

    for flt in plan.get("filters", []):
        field = canonicalize_field(flt.get("field"))
        if field in FILTER_SLOT_FIELDS and flt.get("op") == "eq":
            sketch[field] = flt.get("value")

    sort_items = plan.get("sort", [])
    if sort_items:
        first = sort_items[0]
        field = canonicalize_field(first.get("field"))
        if field is not None:
            sketch["sort_by"] = field
            sketch["sort_dir"] = first.get("direction")

    return sketch


def is_sketch_draft(draft: Any) -> bool:
    return isinstance(draft, dict) and "filters" not in draft and any(field in draft for field in SKETCH_FIELDS)


def _clean_slot_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in NULL_LIKE_VALUES:
        return None
    return value


def build_plan_from_sketch(draft: dict[str, Any], repairs: list[str], errors: list[str]) -> dict[str, Any]:
    normalized_filters: list[dict[str, Any]] = []
    for field in FILTER_SLOT_FIELDS:
        raw_value = _clean_slot_value(draft.get(field))
        if raw_value is None:
            continue

        normalized_value = normalize_value(field, raw_value)
        if normalized_value is None:
            repairs.append(f"dropped sketch slot {field!r} because value could not be normalized")
            errors.append("invalid_slot_value")
            continue

        normalized_filters.append({"field": field, "op": "eq", "value": normalized_value})
        if normalized_value != raw_value:
            repairs.append(f"normalized sketch slot {field!r}")

    normalized_sort: list[dict[str, Any]] = []
    raw_sort_by = _clean_slot_value(draft.get("sort_by"))
    raw_sort_dir = _clean_slot_value(draft.get("sort_dir"))
    if raw_sort_by is not None:
        sort_field = canonicalize_field(raw_sort_by)
        if sort_field is None:
            repairs.append(f"dropped unsupported sketch sort field {raw_sort_by!r}")
            errors.append("invalid_sort")
        else:
            direction = normalize_sort_direction(sort_field, raw_sort_dir or "asc")
            if direction is None:
                repairs.append(f"dropped unsupported sketch sort direction {raw_sort_dir!r} for field {sort_field}")
                errors.append("invalid_sort")
            else:
                normalized_sort.append({"field": sort_field, "direction": direction})
                if sort_field != raw_sort_by or direction != raw_sort_dir:
                    repairs.append(f"normalized sketch sort {raw_sort_by!r} -> {sort_field}")

    return {
        "intent": DEFAULT_INTENT,
        "entity": DEFAULT_ENTITY,
        "filters": normalized_filters,
        "sort": normalized_sort,
        "limit": draft.get("limit", DEFAULT_LIMIT),
        "return_fields": list(DEFAULT_RETURN_FIELDS),
    }
