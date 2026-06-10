from __future__ import annotations

import re
from typing import Any

from .field_registry import ORAL_CLASS_VALUES, SCAFFOLD_TYPE_VALUES


ORAL_CLASS_BY_LOWER = {value.lower(): value for value in ORAL_CLASS_VALUES}
KNOWN_SCAFFOLDS_BY_LOWER = {value.lower(): value for value in SCAFFOLD_TYPE_VALUES}
LIMIT_PATTERNS = (
    re.compile(r"前\s*(\d+)\s*个"),
    re.compile(r"前\s*(\d+)(?!\s*个)"),
    re.compile(r"最多(?:保留|返回)?\s*(\d+)\s*条"),
    re.compile(r"只保留\s*(\d+)\s*条"),
    re.compile(r"只留前\s*(\d+)\s*个?"),
    re.compile(r"截到\s*(\d+)\s*条"),
    re.compile(r"返回数量不要超过\s*(\d+)"),
    re.compile(r"不要超过\s*(\d+)"),
    re.compile(r"最多输出\s*(\d+)\s*条"),
    re.compile(r"结果上限\s*(\d+)\s*条"),
    re.compile(r"输出限制为\s*(\d+)\s*条"),
    re.compile(r"最终不要超过\s*(\d+)\s*项"),
    re.compile(r"控制在\s*(\d+)\s*个以内"),
    re.compile(r"limit\s*(\d+)", flags=re.IGNORECASE),
    re.compile(r"top\s*(\d+)", flags=re.IGNORECASE),
    re.compile(r"stop\s+after\s+(\d+)\s+results?", flags=re.IGNORECASE),
    re.compile(r"cap\s+the\s+list\s+at\s+(\d+)", flags=re.IGNORECASE),
    re.compile(r"return\s+the\s+first\s+(\d+)", flags=re.IGNORECASE),
    re.compile(r"keep\s+only\s+(\d+)", flags=re.IGNORECASE),
    re.compile(r"at\s+most\s+(\d+)", flags=re.IGNORECASE),
    re.compile(r"trim\s+the\s+output\s+to\s+(\d+)", flags=re.IGNORECASE),
    re.compile(r"keep\s+no\s+more\s+than\s+(\d+)\s+hits?", flags=re.IGNORECASE),
    re.compile(r"stop\s+once\s+(\d+)\s+rows?\s+remain", flags=re.IGNORECASE),
    re.compile(r"keep\s+the\s+top\s+(\d+)", flags=re.IGNORECASE),
    re.compile(r"show\s+no\s+more\s+than\s+(\d+)\s+records?", flags=re.IGNORECASE),
    re.compile(r"cap\s+the\s+response\s+at\s+(\d+)", flags=re.IGNORECASE),
    re.compile(r"stop\s+at\s+(\d+)", flags=re.IGNORECASE),
    re.compile(r"to\s+(\d+)\.?$", flags=re.IGNORECASE),
)
ORAL_PATTERNS = (
    re.compile(r"oral\s*(?:class|level)?\s*(?:is|=)?\s*(high|medium|low)", flags=re.IGNORECASE),
    re.compile(r"oral\s*(high|medium|low)", flags=re.IGNORECASE),
    re.compile(r"oral\s*分档\s*(?:是|为)?\s*(high|medium|low)", flags=re.IGNORECASE),
    re.compile(r"口服\s*分档\s*(?:是|为)?\s*(high|medium|low)", flags=re.IGNORECASE),
)
AFFINITY_CONTEXT_PATTERNS = (
    re.compile(r"affinity", flags=re.IGNORECASE),
    re.compile(r"affinity values?", flags=re.IGNORECASE),
    re.compile(r"affinity quality", flags=re.IGNORECASE),
    re.compile(r"binding strength", flags=re.IGNORECASE),
    re.compile(r"binding potency", flags=re.IGNORECASE),
    re.compile(r"binders?", flags=re.IGNORECASE),
    re.compile(r"亲和力"),
    re.compile(r"结合强度"),
)
SORT_PATTERNS = (
    re.compile(r"最强"),
    re.compile(r"放前面"),
    re.compile(r"优先"),
    re.compile(r"排序"),
    re.compile(r"排前"),
    re.compile(r"best affinity first", flags=re.IGNORECASE),
    re.compile(r"strongest affinity first", flags=re.IGNORECASE),
    re.compile(r"strongest binders? first", flags=re.IGNORECASE),
    re.compile(r"best binders? first", flags=re.IGNORECASE),
    re.compile(r"ranking by binding strength", flags=re.IGNORECASE),
    re.compile(r"rank(?:ed|ing)? by affinity", flags=re.IGNORECASE),
    re.compile(r"rank by affinity quality", flags=re.IGNORECASE),
    re.compile(r"rank the survivors by measured binding quality", flags=re.IGNORECASE),
    re.compile(r"sort from best measured affinity to weaker binding", flags=re.IGNORECASE),
    re.compile(r"order the list with the best affinity first", flags=re.IGNORECASE),
    re.compile(r"keep the strongest binders? first", flags=re.IGNORECASE),
    re.compile(r"put the best binders? first", flags=re.IGNORECASE),
    re.compile(r"put the highest-priority binders? first", flags=re.IGNORECASE),
    re.compile(r"arrange the output by strongest measured affinity", flags=re.IGNORECASE),
    re.compile(r"prioriti[sz]e better affinity values?", flags=re.IGNORECASE),
    re.compile(r"sorting by binding potency", flags=re.IGNORECASE),
    re.compile(r"按测得亲和力优先排序"),
    re.compile(r"先放最强结合的记录"),
    re.compile(r"按最佳亲和力优先"),
    re.compile(r"按最优结合强度排序"),
)


def infer_request_hints(text: str) -> dict[str, Any]:
    lowered = text.lower()
    hints: dict[str, Any] = {}

    for scaffold_lower, scaffold_value in KNOWN_SCAFFOLDS_BY_LOWER.items():
        if scaffold_lower in lowered:
            hints["scaffold_type"] = scaffold_value
            break

    for pattern in ORAL_PATTERNS:
        oral_match = pattern.search(text)
        if oral_match:
            hints["oral_class"] = ORAL_CLASS_BY_LOWER.get(oral_match.group(1).lower())
            break

    if "non-engineered" in lowered:
        hints["engineered"] = False
    elif re.search(r"(?<!non-)engineered", lowered):
        hints["engineered"] = True

    if "带实验亲和力" in text or "实验亲和力" in text or "experimental affinity" in lowered:
        hints["has_experimental_affinity"] = True

    for pattern in LIMIT_PATTERNS:
        match = pattern.search(text)
        if match:
            try:
                hints["limit"] = int(match.group(1))
            except ValueError:
                pass
            break

    has_affinity_context = any(pattern.search(text) for pattern in AFFINITY_CONTEXT_PATTERNS)
    has_sort_signal = any(pattern.search(text) for pattern in SORT_PATTERNS) or "sort" in lowered
    if has_affinity_context and has_sort_signal:
            hints["sort_by"] = "best_affinity_value"
            hints["sort_dir"] = "asc"

    return hints


def format_request_hints(hints: dict[str, Any]) -> str:
    if not hints:
        return "No deterministic slot hints extracted."
    parts = [f"{key}={value}" for key, value in hints.items()]
    return ", ".join(parts)


def semantic_mismatch_reasons(plan: dict[str, Any], request_text: str) -> list[str]:
    hints = infer_request_hints(request_text)
    if not hints:
        return []

    filter_map = {item.get("field"): item.get("value") for item in plan.get("filters", [])}
    reasons: list[str] = []

    for field in ("scaffold_type", "oral_class", "engineered", "has_experimental_affinity"):
        expected = hints.get(field)
        if expected is None:
            continue
        if filter_map.get(field) != expected:
            reasons.append(f"semantic_mismatch_{field}")

    expected_limit = hints.get("limit")
    if expected_limit is not None and plan.get("limit") != expected_limit:
        reasons.append("semantic_mismatch_limit")

    expected_sort_by = hints.get("sort_by")
    if expected_sort_by is not None:
        sort_items = plan.get("sort", [])
        first_sort = sort_items[0] if sort_items else {}
        if first_sort.get("field") != expected_sort_by or first_sort.get("direction") != hints.get("sort_dir", "asc"):
            reasons.append("semantic_mismatch_sort")

    return reasons
