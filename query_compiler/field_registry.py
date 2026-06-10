from __future__ import annotations

from typing import Any


SCAFFOLD_TYPE_VALUES = (
    "EVH1domain",
    "Ibody",
    "adnectin",
    "affimer",
    "avimer",
    "betarolldomain",
    "centyrin",
    "cyclotide",
    "knottin",
    "kunitz",
    "obody",
    "phdfingerdomain",
)
ORAL_CLASS_VALUES = ("High", "Low", "Medium")

CANONICAL_FIELDS = {
    "candidate_id",
    "canonical_name",
    "organism",
    "scaffold_type",
    "engineered",
    "oral_class",
    "expression_system",
    "stability_signal",
    "inhibitory",
    "has_experimental_affinity",
    "best_affinity_value",
}
SCHEMA_WORD_VALUES = CANONICAL_FIELDS | {
    "candidate_search",
    "protein_candidate",
    "filters",
    "sort",
    "return_fields",
}

FIELD_ALIASES = {
    "candidate": "candidate_id",
    "name": "canonical_name",
    "species": "organism",
    "scaffold": "scaffold_type",
    "scaffold_class": "scaffold_type",
    "oral": "oral_class",
    "oral_properties": "oral_class",
    "stability": "stability_signal",
    "is_engineered": "engineered",
    "affinity": "best_affinity_value",
    "binding_strength": "best_affinity_value",
    "experimental_affinity": "has_experimental_affinity",
}

FIELD_VALUE_NORMALIZERS = {
    "oral_class": {
        "high": "High",
        "medium": "Medium",
        "low": "Low",
        "unknown": "Unknown",
    }
}
SCAFFOLD_TYPE_NORMALIZERS = {value.lower(): value for value in SCAFFOLD_TYPE_VALUES}

BOOLEAN_TRUE_VALUES = {"true", "yes", "y", "1", "engineered"}
BOOLEAN_FALSE_VALUES = {"false", "no", "n", "0"}

OPERATOR_ALIASES = {
    "=": "eq",
    "==": "eq",
    "eq": "eq",
    "in": "in",
    ">=": "gte",
    "gte": "gte",
    "<=": "lte",
    "lte": "lte",
    "exists": "exists",
}

SORT_DIRECTION_ALIASES = {
    "asc": "asc",
    "ascending": "asc",
    "strongest": "asc",
    "desc": "desc",
    "descending": "desc",
    "weakest": "desc",
}


def canonicalize_field(field: Any) -> str | None:
    if field is None:
        return None
    key = str(field).strip()
    if not key:
        return None
    lowered = key.lower()
    canonical = FIELD_ALIASES.get(lowered, lowered)
    return canonical if canonical in CANONICAL_FIELDS else None


def normalize_operator(op: Any) -> str | None:
    if op is None:
        return None
    return OPERATOR_ALIASES.get(str(op).strip().lower())


def normalize_value(field: str, value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    raw_text = str(value).strip()
    lowered_raw = raw_text.lower()
    if lowered_raw in SCHEMA_WORD_VALUES and field not in {"candidate_id"}:
        return None
    if field == "scaffold_type":
        lowered = lowered_raw
        return SCAFFOLD_TYPE_NORMALIZERS.get(lowered)
    if field in {"engineered", "inhibitory", "has_experimental_affinity"}:
        lowered = lowered_raw
        if lowered in BOOLEAN_TRUE_VALUES:
            return True
        if lowered in BOOLEAN_FALSE_VALUES:
            return False
        return None
    if field in FIELD_VALUE_NORMALIZERS:
        lowered = lowered_raw
        return FIELD_VALUE_NORMALIZERS[field].get(lowered)
    return raw_text


def normalize_sort_direction(field: str, direction: Any) -> str | None:
    if direction is None:
        return None
    normalized = SORT_DIRECTION_ALIASES.get(str(direction).strip().lower())
    if normalized == "asc" and field == "best_affinity_value":
        return "asc"
    return normalized
