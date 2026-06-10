from __future__ import annotations

from typing import Any


def _matches_filter(row: dict[str, Any], flt: dict[str, Any]) -> bool:
    field = flt["field"]
    op = flt["op"]
    expected = flt.get("value")
    actual = row.get(field)

    if op == "eq":
        return actual == expected
    if op == "in":
        return actual in (expected or [])
    if op == "gte":
        return actual is not None and expected is not None and actual >= expected
    if op == "lte":
        return actual is not None and expected is not None and actual <= expected
    if op == "exists":
        return actual is not None
    raise ValueError(f"Unsupported operator: {op}")


def _sort_key(row: dict[str, Any], field: str) -> tuple[int, Any]:
    value = row.get(field)
    return (1, None) if value is None else (0, value)


def execute_query_plan(plan: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    filtered = list(rows)
    for flt in plan.get("filters", []):
        filtered = [row for row in filtered if _matches_filter(row, flt)]

    for sort_spec in reversed(plan.get("sort", [])):
        field = sort_spec["field"]
        reverse = sort_spec.get("direction") == "desc"
        filtered.sort(key=lambda row, field_name=field: _sort_key(row, field_name), reverse=reverse)

    limit = int(plan.get("limit", len(filtered)))
    limited_rows = filtered[:limit]

    return {
        "rows": limited_rows,
        "metadata": {
            "total_rows": len(rows),
            "filtered_count": len(filtered),
            "returned_count": len(limited_rows),
            "limit": limit,
        },
    }
