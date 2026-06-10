from __future__ import annotations

import sys
from pathlib import Path


LAB_ROOT = Path(__file__).resolve().parents[1]
if str(LAB_ROOT) not in sys.path:
    sys.path.insert(0, str(LAB_ROOT))

from query_compiler.request_hints import infer_request_hints, semantic_mismatch_reasons  # type: ignore  # noqa: E402


def test_infer_request_hints_extracts_core_slots() -> None:
    hints = infer_request_hints("找 knottin 候选、oral high、engineered、带实验亲和力，按 affinity 最强排前 20，输出查询计划 JSON。")

    assert hints["scaffold_type"] == "knottin"
    assert hints["oral_class"] == "High"
    assert hints["engineered"] is True
    assert hints["has_experimental_affinity"] is True
    assert hints["sort_by"] == "best_affinity_value"
    assert hints["sort_dir"] == "asc"
    assert hints["limit"] == 20


def test_semantic_mismatch_reasons_flags_wrong_slots() -> None:
    plan = {
        "intent": "candidate_search",
        "entity": "protein_candidate",
        "filters": [
            {"field": "scaffold_type", "op": "eq", "value": "knottin"},
            {"field": "oral_class", "op": "eq", "value": "Medium"},
            {"field": "engineered", "op": "eq", "value": False},
            {"field": "has_experimental_affinity", "op": "eq", "value": True},
        ],
        "sort": [{"field": "best_affinity_value", "direction": "asc"}],
        "limit": 20,
    }

    reasons = semantic_mismatch_reasons(
        plan,
        "找 knottin 候选、oral high、engineered、带实验亲和力，按 affinity 最强排前 20，输出查询计划 JSON。",
    )

    assert "semantic_mismatch_oral_class" in reasons
    assert "semantic_mismatch_engineered" in reasons


def test_infer_request_hints_handles_true_holdout_english_phrasings() -> None:
    hints = infer_request_hints(
        "From the FBBP registry, pull candidates with scaffold family obody, oral class Low, non-engineered molecules only, "
        "experimental affinity evidence must be present; keep the strongest binders first; stop after 10 results."
    )

    assert hints["scaffold_type"] == "obody"
    assert hints["oral_class"] == "Low"
    assert hints["engineered"] is False
    assert hints["has_experimental_affinity"] is True
    assert hints["sort_by"] == "best_affinity_value"
    assert hints["sort_dir"] == "asc"
    assert hints["limit"] == 10


def test_infer_request_hints_handles_true_holdout_chinese_phrasings() -> None:
    hints = infer_request_hints(
        "研究问题是这样的：候选必须满足 骨架类型是 cyclotide、oral 分档是 Medium、只看 engineered 条目、必须带实验亲和力证据；"
        "结果部分请 把结合最强的放在最前面；返回数量不要超过 20。"
    )

    assert hints["scaffold_type"] == "cyclotide"
    assert hints["oral_class"] == "Medium"
    assert hints["engineered"] is True
    assert hints["has_experimental_affinity"] is True
    assert hints["sort_by"] == "best_affinity_value"
    assert hints["sort_dir"] == "asc"
    assert hints["limit"] == 20


def test_infer_request_hints_handles_v12_farther_english_phrasings() -> None:
    hints = infer_request_hints(
        "I need a candidate shortlist restricted to scaffold family knottin, oral class High, engineered molecules only, "
        "experimental affinity evidence must be present; prioritize better affinity values and trim the output to 10."
    )

    assert hints["scaffold_type"] == "knottin"
    assert hints["oral_class"] == "High"
    assert hints["engineered"] is True
    assert hints["has_experimental_affinity"] is True
    assert hints["sort_by"] == "best_affinity_value"
    assert hints["sort_dir"] == "asc"
    assert hints["limit"] == 10


def test_infer_request_hints_handles_v12_farther_chinese_limit_phrasings() -> None:
    hints = infer_request_hints(
        "我想从候选库里锁定 骨架类型是 obody、oral 分档是 Low、只看 non-engineered 条目、必须带实验亲和力证据 的分子，"
        "排序时先看最强结合，名单控制在 5 个以内。"
    )

    assert hints["scaffold_type"] == "obody"
    assert hints["oral_class"] == "Low"
    assert hints["engineered"] is False
    assert hints["has_experimental_affinity"] is True
    assert hints["sort_by"] == "best_affinity_value"
    assert hints["sort_dir"] == "asc"
    assert hints["limit"] == 5


def test_infer_request_hints_handles_v13_farther_english_phrasings() -> None:
    hints = infer_request_hints(
        "For downstream review, retain only protein candidates tagged with scaffold family obody, oral class Low, "
        "non-engineered molecules only, experimental affinity evidence must be present. "
        "Put the highest-priority binders first and show no more than 20 records."
    )

    assert hints["scaffold_type"] == "obody"
    assert hints["oral_class"] == "Low"
    assert hints["engineered"] is False
    assert hints["has_experimental_affinity"] is True
    assert hints["sort_by"] == "best_affinity_value"
    assert hints["sort_dir"] == "asc"
    assert hints["limit"] == 20


def test_infer_request_hints_handles_v13_farther_chinese_phrasings() -> None:
    hints = infer_request_hints(
        "候选清单需求：骨架类型是 adnectin、oral 分档是 Medium、只看 non-engineered 条目、必须带实验亲和力证据；"
        "结果按最优结合强度排序；输出限制为 5 条。"
    )

    assert hints["scaffold_type"] == "adnectin"
    assert hints["oral_class"] == "Medium"
    assert hints["engineered"] is False
    assert hints["has_experimental_affinity"] is True
    assert hints["sort_by"] == "best_affinity_value"
    assert hints["sort_dir"] == "asc"
    assert hints["limit"] == 5
