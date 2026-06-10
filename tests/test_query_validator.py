from __future__ import annotations

import sys
from pathlib import Path


LAB_ROOT = Path(__file__).resolve().parents[1]
if str(LAB_ROOT) not in sys.path:
    sys.path.insert(0, str(LAB_ROOT))

from query_compiler.validator import validate_query_draft  # type: ignore  # noqa: E402


def test_validate_query_draft_normalizes_aliases_and_defaults() -> None:
    draft = {
        "filters": [
            {"field": "oral", "op": "=", "value": "high"},
            {"field": "scaffold", "op": "=", "value": "kunitz"},
        ],
        "sort": [{"field": "affinity", "direction": "strongest"}],
    }

    result = validate_query_draft(draft)

    assert result["plan"]["intent"] == "candidate_search"
    assert result["plan"]["entity"] == "protein_candidate"
    assert result["plan"]["limit"] == 20
    assert result["plan"]["filters"][0] == {"field": "oral_class", "op": "eq", "value": "High"}
    assert result["plan"]["filters"][1] == {"field": "scaffold_type", "op": "eq", "value": "kunitz"}
    assert result["plan"]["sort"][0] == {"field": "best_affinity_value", "direction": "asc"}
    assert result["trace"]["parsed"] is True
    assert result["trace"]["schema_ok"] is True
    assert result["trace"]["errors"] == []
    assert result["trace"]["empty_filters"] is False
    assert result["trace"]["repairs"]


def test_validate_query_draft_drops_unsupported_fields_and_records_trace() -> None:
    draft = {
        "filters": [
            {"field": "solubility_class", "op": "=", "value": "High"},
            {"field": "engineered", "op": "=", "value": "yes"},
        ],
        "limit": "7",
    }

    result = validate_query_draft(draft)

    assert result["plan"]["filters"] == [{"field": "engineered", "op": "eq", "value": True}]
    assert result["plan"]["limit"] == 7
    assert result["trace"]["schema_ok"] is True
    assert result["trace"]["empty_filters"] is False
    assert any("solubility_class" in repair for repair in result["trace"]["repairs"])


def test_validate_query_draft_marks_empty_filters_as_error() -> None:
    draft = {
        "filters": [
            {"field": "solubility_class", "op": "=", "value": "High"},
        ],
        "sort": [],
    }

    result = validate_query_draft(draft)

    assert result["plan"]["filters"] == []
    assert result["trace"]["parsed"] is True
    assert result["trace"]["schema_ok"] is False
    assert result["trace"]["empty_filters"] is True
    assert "empty_filters" in result["trace"]["errors"]


def test_validate_query_draft_accepts_flat_sketch_format() -> None:
    draft = {
        "candidate_id": None,
        "scaffold_type": "knottin",
        "oral_class": "low",
        "engineered": "false",
        "has_experimental_affinity": True,
        "organism": None,
        "sort_by": "best_affinity_value",
        "sort_dir": "asc",
        "limit": 10,
    }

    result = validate_query_draft(draft)

    assert result["plan"]["filters"] == [
        {"field": "scaffold_type", "op": "eq", "value": "knottin"},
        {"field": "oral_class", "op": "eq", "value": "Low"},
        {"field": "engineered", "op": "eq", "value": False},
        {"field": "has_experimental_affinity", "op": "eq", "value": True},
    ]
    assert result["plan"]["sort"] == [{"field": "best_affinity_value", "direction": "asc"}]
    assert result["plan"]["limit"] == 10
    assert result["trace"]["draft_format"] == "sketch"


def test_validate_query_draft_drops_invalid_scaffold_enum_values() -> None:
    draft = {
        "candidate_id": None,
        "scaffold_type": "FBBP",
        "oral_class": "Low",
        "engineered": False,
        "has_experimental_affinity": True,
        "organism": None,
        "sort_by": "best_affinity_value",
        "sort_dir": "asc",
        "limit": 10,
    }

    result = validate_query_draft(draft)

    assert all(item["field"] != "scaffold_type" for item in result["plan"]["filters"])
    assert "invalid_slot_value" in result["trace"]["errors"]
