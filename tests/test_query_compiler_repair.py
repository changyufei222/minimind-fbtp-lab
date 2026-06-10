from __future__ import annotations

import sys
from pathlib import Path


LAB_ROOT = Path(__file__).resolve().parents[1]
if str(LAB_ROOT) not in sys.path:
    sys.path.insert(0, str(LAB_ROOT))

from query_compiler.repair import finalize_prediction, repair_reason_list, should_repair  # type: ignore  # noqa: E402


def test_should_repair_when_filters_normalize_to_empty() -> None:
    normalized = {
        "plan": {
            "intent": "candidate_search",
            "entity": "protein_candidate",
            "filters": [],
            "sort": [],
            "limit": 20,
            "return_fields": ["candidate_id"],
        },
        "trace": {
            "repairs": ["dropped unsupported field 'foo'"],
            "errors": ["empty_filters"],
            "parsed": True,
            "schema_ok": False,
            "empty_filters": True,
        },
    }

    assert should_repair(normalized) is True
    assert repair_reason_list(normalized) == ["empty_filters"]


def test_should_not_repair_when_plan_is_valid_and_non_empty() -> None:
    normalized = {
        "plan": {
            "intent": "candidate_search",
            "entity": "protein_candidate",
            "filters": [{"field": "oral_class", "op": "eq", "value": "High"}],
            "sort": [],
            "limit": 20,
            "return_fields": ["candidate_id"],
        },
        "trace": {
            "repairs": [],
            "errors": [],
            "parsed": True,
            "schema_ok": True,
            "empty_filters": False,
        },
    }

    assert should_repair(normalized) is False
    assert repair_reason_list(normalized) == []


def test_finalize_prediction_uses_repair_when_first_draft_is_empty() -> None:
    first_answer = (
        '{"candidate_id":null,"scaffold_type":null,"oral_class":null,"engineered":null,'
        '"has_experimental_affinity":null,"organism":null,"sort_by":null,"sort_dir":null,"limit":20}'
    )
    repaired_answer = (
        '{"candidate_id":null,"scaffold_type":"knottin","oral_class":"High","engineered":false,'
        '"has_experimental_affinity":true,"organism":null,"sort_by":null,"sort_dir":null,"limit":20}'
    )

    result = finalize_prediction(
        first_answer=first_answer,
        original_request="找 knottin 候选、oral high、non-engineered、带实验亲和力，前 20 个。",
        repaired_answer=repaired_answer,
    )

    assert result["used_repair"] is True
    assert result["final_plan"]["filters"] == [
        {"field": "scaffold_type", "op": "eq", "value": "knottin"},
        {"field": "oral_class", "op": "eq", "value": "High"},
        {"field": "engineered", "op": "eq", "value": False},
        {"field": "has_experimental_affinity", "op": "eq", "value": True},
    ]
    assert "empty_filters" in result["repair_reasons"]


def test_finalize_prediction_keeps_first_draft_when_no_repair_needed() -> None:
    first_answer = (
        '{"candidate_id":null,"scaffold_type":null,"oral_class":null,"engineered":true,'
        '"has_experimental_affinity":null,"organism":null,"sort_by":null,"sort_dir":null,"limit":20}'
    )

    result = finalize_prediction(first_answer=first_answer, original_request="找 engineered 候选，前 20 个。")

    assert result["used_repair"] is False
    assert "semantic_mismatch_engineered" not in result["repair_reasons"]
    assert result["final_plan"]["filters"] == [{"field": "engineered", "op": "eq", "value": True}]


def test_finalize_prediction_marks_semantic_mismatch_for_repair() -> None:
    first_answer = (
        '{"candidate_id":null,"scaffold_type":"knottin","oral_class":"Medium","engineered":false,'
        '"has_experimental_affinity":true,"organism":null,"sort_by":"best_affinity_value","sort_dir":"asc","limit":20}'
    )
    repaired_answer = (
        '{"candidate_id":null,"scaffold_type":"knottin","oral_class":"High","engineered":true,'
        '"has_experimental_affinity":true,"organism":null,"sort_by":"best_affinity_value","sort_dir":"asc","limit":20}'
    )

    result = finalize_prediction(
        first_answer=first_answer,
        original_request="找 knottin 候选、oral high、engineered、带实验亲和力，按 affinity 最强排前 20，输出查询计划 JSON。",
        repaired_answer=repaired_answer,
    )

    assert result["used_repair"] is True
    assert "semantic_mismatch_oral_class" in result["repair_reasons"]
    assert "semantic_mismatch_engineered" in result["repair_reasons"]
    assert result["final_plan"]["filters"][1] == {"field": "oral_class", "op": "eq", "value": "High"}


def test_finalize_prediction_can_use_semantic_projection_when_repair_is_missing() -> None:
    first_answer = (
        '{"candidate_id":null,"scaffold_type":"knottin","oral_class":"High","engineered":false,'
        '"has_experimental_affinity":true,"organism":null,"sort_by":"best_affinity_value","sort_dir":"asc","limit":10}'
    )

    result = finalize_prediction(
        first_answer=first_answer,
        original_request="I need a candidate shortlist restricted to scaffold family knottin, oral class High, engineered molecules only, experimental affinity evidence must be present; prioritize better affinity values and trim the output to 10.",
    )

    assert result["projection_attempted"] is True
    assert result["used_projection"] is True
    assert result["used_repair"] is False
    filter_map = {item["field"]: item["value"] for item in result["final_plan"]["filters"]}
    assert filter_map["engineered"] is True
    assert "semantic projection aligned engineered with request hints" in result["projection_reasons"]


def test_finalize_prediction_keeps_first_when_repair_is_worse() -> None:
    first_answer = (
        '{"candidate_id":null,"scaffold_type":"phdfingerdomain","oral_class":"Medium","engineered":false,'
        '"has_experimental_affinity":true,"organism":null,"sort_by":"best_affinity_value","sort_dir":"asc","limit":10}'
    )
    repaired_answer = (
        '{"candidate_id":null,"scaffold_type":"phdfingerdomain","oral_class":"Medium","engineered":false,'
        '"has_experimental_affinity":true,"organism":null,"sort_by":"best_affinity_value","sort_dir":"asc","limit":10,"}'
    )

    result = finalize_prediction(
        first_answer=first_answer,
        original_request="帮我筛一批 phdfingerdomain 候选、oral high、non-engineered、带实验亲和力 的候选，口径按真实数据库字段来，优先把亲和力更强的放前面，先给我前 10 个。",
        repaired_answer=repaired_answer,
    )

    assert result["repair_attempted"] is True
    assert result["used_repair"] is False
    assert result["used_projection"] is True
    assert result["final_plan"]["filters"] == [
        {"field": "scaffold_type", "op": "eq", "value": "phdfingerdomain"},
        {"field": "oral_class", "op": "eq", "value": "High"},
        {"field": "engineered", "op": "eq", "value": False},
        {"field": "has_experimental_affinity", "op": "eq", "value": True},
    ]


def test_finalize_prediction_can_use_salvaged_first_draft_without_repair() -> None:
    first_answer = (
        '{"candidate_id":null,"statper_class":"knottin","higheral_class":"High","engimber_":false,'
        '"has_experimental_affinity":true'
    )

    result = finalize_prediction(
        first_answer=first_answer,
        original_request="Compile this FBBP candidate request to JSON only: scaffold=knottin; oral high; non-engineered; experimental affinity required; rank=strongest affinity first; top=20.",
    )

    assert result["repair_attempted"] is False
    assert result["used_repair"] is False
    assert result["final_plan"]["filters"] == [
        {"field": "scaffold_type", "op": "eq", "value": "knottin"},
        {"field": "oral_class", "op": "eq", "value": "High"},
        {"field": "engineered", "op": "eq", "value": False},
        {"field": "has_experimental_affinity", "op": "eq", "value": True},
    ]


def test_finalize_prediction_drops_schema_words_used_as_slot_values() -> None:
    first_answer = (
        '{"candidate_id":null,"scaffold_type":"candidate_id","oral_class":"candidate_id","engineered":"candidate_id",'
        '"has_experimental_affinity":true,"organism":null,"sort_by":"best_affinity_value","sort_dir":"asc","limit":10}'
    )

    result = finalize_prediction(
        first_answer=first_answer,
        original_request="From the FBBP registry, pull candidates with scaffold family obody, oral class Low, non-engineered molecules only, experimental affinity evidence must be present; keep the strongest binders first; stop after 10 results.",
    )

    assert result["first_normalized"]["plan"]["filters"] == [
        {"field": "has_experimental_affinity", "op": "eq", "value": True},
    ]
    assert "empty_filters" not in result["first_normalized"]["trace"]["errors"]
