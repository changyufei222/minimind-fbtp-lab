from __future__ import annotations

import json
import sys
from pathlib import Path


LAB_ROOT = Path(__file__).resolve().parents[1]
if str(LAB_ROOT) not in sys.path:
    sys.path.insert(0, str(LAB_ROOT))

from query_compiler.prompting import (  # type: ignore  # noqa: E402
    build_compiler_messages,
    build_repair_messages,
    extract_first_json_object,
    render_plan_json,
)


def test_build_compiler_messages_includes_json_only_system_prompt() -> None:
    messages = build_compiler_messages("找 knottin 候选，前 5 个。")

    assert messages[0]["role"] == "system"
    assert "Output one JSON object only" in messages[0]["content"]
    assert "scaffold_type" in messages[0]["content"]
    assert "filters" in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert "Req: 找 knottin 候选，前 5 个。" in messages[1]["content"]
    assert "Hints:" in messages[1]["content"]
    assert "scaffold_type=knottin" in messages[1]["content"]
    assert "limit=5" in messages[1]["content"]
    assert "JSON only." in messages[1]["content"]


def test_build_compiler_messages_can_disable_hints_and_request_wrapper() -> None:
    messages = build_compiler_messages(
        "From the FBBP registry, pull candidates with scaffold family knottin and oral class High.",
        include_hints=False,
        wrap_request=False,
    )

    assert messages[1]["content"] == "From the FBBP registry, pull candidates with scaffold family knottin and oral class High."
    assert "Hints:" not in messages[1]["content"]
    assert "Req:" not in messages[1]["content"]


def test_render_plan_json_is_compact_and_round_trips() -> None:
    plan = {
        "intent": "candidate_search",
        "entity": "protein_candidate",
        "filters": [{"field": "scaffold_type", "op": "eq", "value": "knottin"}],
        "sort": [{"field": "best_affinity_value", "direction": "asc"}],
        "limit": 5,
    }

    rendered = render_plan_json(plan)

    assert rendered.startswith("{")
    assert "\n" not in rendered
    assert json.loads(rendered) == plan


def test_extract_first_json_object_handles_markdown_fence() -> None:
    answer = """先给你结果。\n```json\n{\"intent\":\"candidate_search\",\"entity\":\"protein_candidate\",\"filters\":[],\"sort\":[],\"limit\":20}\n```"""

    parsed = extract_first_json_object(answer)

    assert parsed is not None
    assert parsed["intent"] == "candidate_search"


def test_extract_first_json_object_handles_prefixed_text() -> None:
    answer = '结果如下: {"intent":"candidate_search","entity":"protein_candidate","filters":[],"sort":[],"limit":10} 谢谢'

    parsed = extract_first_json_object(answer)

    assert parsed is not None
    assert parsed["limit"] == 10


def test_extract_first_json_object_can_require_top_level_keys() -> None:
    answer = (
        '{"intent":"candidate_search","entity":"protein_candidate","filters":'
        '[{"field":"scaffold_type","op":"eq","value":"obody"},{"field":"outer","op":"eq","value":"Low"}'
    )

    parsed = extract_first_json_object(answer, required_keys={"intent", "entity", "filters", "limit"})

    assert parsed is None


def test_extract_first_json_object_can_close_truncated_flat_sketch() -> None:
    answer = (
        '{"candidate_id":null,"scaffold_type":"knottin","oral_class":"Low",'
        '"engineered":false,"has_experimental_affinity":true,"organism":null,'
        '"sort_by":"best_affinity_value","sort_dir":"asc","limit":20'
    )

    parsed = extract_first_json_object(answer, required_keys={"limit"})

    assert parsed is not None
    assert parsed["scaffold_type"] == "knottin"
    assert parsed["limit"] == 20


def test_extract_first_json_object_can_salvage_typo_keys_and_missing_closure() -> None:
    answer = (
        '{"candidate_id":null,"statper_class":"knottin","higheral_class":"Low",'
        '"engimber_":fal,"has_experimental_affinity":true'
    )

    parsed = extract_first_json_object(answer, required_keys={"limit"})

    assert parsed is not None
    assert parsed["scaffold_type"] == "knottin"
    assert parsed["oral_class"] == "Low"
    assert parsed["engineered"] is False
    assert parsed["limit"] == 20


def test_build_repair_messages_includes_error_summary_and_invalid_output() -> None:
    messages = build_repair_messages(
        original_request="找 knottin 候选、oral low、non-engineered，前 5 个。",
        invalid_output='{"scaffold_type":null,"oral_class":null,"engineered":null,"has_experimental_affinity":null,"organism":null,"sort_by":null,"sort_dir":null,"limit":20}',
        errors=["empty_filters", "semantic_mismatch_oral_class"],
    )

    assert messages[0]["role"] == "system"
    assert "repair" in messages[0]["content"].lower()
    assert "Issues:" in messages[1]["content"]
    assert "keep at least one non-null filter slot" in messages[1]["content"]
    assert "oral_class must match the request oral level" in messages[1]["content"]
    assert "找 knottin 候选" in messages[1]["content"]
    assert "Hints:" in messages[1]["content"]
    assert "scaffold_type=knottin" in messages[1]["content"]
    assert "keep at least one non-null filter slot" in messages[0]["content"].lower()
    assert 'Bad: {"scaffold_type":null' in messages[1]["content"]


def test_build_repair_messages_can_disable_hints_and_req_wrapper() -> None:
    messages = build_repair_messages(
        original_request="From the FBBP registry, pull candidates with scaffold family knottin and oral class High.",
        invalid_output='{"candidate_id":null,"statper_class":"knottin"',
        errors=["parse_failed"],
        include_hints=False,
        wrap_request=False,
    )

    assert "Hints:" not in messages[1]["content"]
    assert "Req:" not in messages[1]["content"]
    assert "Original request:" in messages[1]["content"]
