from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from .candidate_snapshot import LAB_ROOT, build_candidate_snapshot
from .field_registry import ORAL_CLASS_VALUES, SCAFFOLD_TYPE_VALUES
from .prompting import build_compiler_messages, build_repair_messages, render_plan_json
from .sketch import plan_to_sketch


DEFAULT_OUTPUT_DIR = LAB_ROOT / "data" / "processed"


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _plan_from_candidate(row: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    filters: list[dict[str, Any]] = []

    if row.get("scaffold_type") and row["scaffold_type"] != "unknown":
        filters.append({"field": "scaffold_type", "op": "eq", "value": row["scaffold_type"]})
    if row.get("oral_class") and row["oral_class"] != "Unknown":
        filters.append({"field": "oral_class", "op": "eq", "value": row["oral_class"]})
    if row.get("engineered") is not None:
        filters.append({"field": "engineered", "op": "eq", "value": row["engineered"]})
    if row.get("has_experimental_affinity"):
        filters.append({"field": "has_experimental_affinity", "op": "eq", "value": True})
    if row.get("organism") and row["organism"] != "Unknown" and rng.random() < 0.35:
        filters.append({"field": "organism", "op": "eq", "value": row["organism"]})

    if not filters:
        filters.append({"field": "candidate_id", "op": "eq", "value": row["candidate_id"]})

    sort: list[dict[str, Any]] = []
    if row.get("best_affinity_value") is not None:
        sort.append({"field": "best_affinity_value", "direction": "asc"})

    return {
        "intent": "candidate_search",
        "entity": "protein_candidate",
        "filters": filters[: max(1, min(len(filters), 4))],
        "sort": sort,
        "limit": rng.choice([5, 10, 20]),
    }


def _render_filters(plan: dict[str, Any]) -> str:
    parts: list[str] = []
    for flt in plan["filters"]:
        field = flt["field"]
        value = flt["value"]
        if field == "scaffold_type":
            parts.append(f"{value} 候选")
        elif field == "oral_class":
            parts.append(f"oral {str(value).lower()}")
        elif field == "engineered":
            parts.append("engineered" if value else "non-engineered")
        elif field == "has_experimental_affinity":
            parts.append("带实验亲和力")
        elif field == "organism":
            parts.append(f"来源于 {value}")
        elif field == "candidate_id":
            parts.append(f"id 为 {value}")
    return "、".join(parts)


def _prompt_for_plan(plan: dict[str, Any], style: str) -> str:
    filter_text = _render_filters(plan)
    limit = plan["limit"]
    has_affinity_sort = any(item["field"] == "best_affinity_value" for item in plan["sort"])

    if style == "hard":
        if has_affinity_sort:
            return f"帮我筛一批 {filter_text} 的候选，口径按真实数据库字段来，优先把亲和力更强的放前面，先给我前 {limit} 个。"
        return f"帮我筛一批 {filter_text} 的候选，先给我前 {limit} 个，输出可执行查询计划 JSON。"
    if style == "dev":
        return f"找 {filter_text}，输出查询计划 JSON，limit {limit}。"
    return f"找 {filter_text}，按 affinity 最强排前 {limit}，输出查询计划 JSON。" if has_affinity_sort else f"找 {filter_text} 前 {limit}，输出查询计划 JSON。"


def _prompt_variants_for_plan(plan: dict[str, Any], style: str) -> list[str]:
    filter_text = _render_filters(plan)
    anti_hype_filter_text = _render_anti_hype_filters(plan)
    limit = plan["limit"]
    sort_clause = "优先把亲和力更强的放前面" if any(item["field"] == "best_affinity_value" for item in plan["sort"]) else "不用额外排序"
    limit_clause = f"先给我前 {limit} 个"
    sort_clause_en = "strongest affinity first" if any(item["field"] == "best_affinity_value" for item in plan["sort"]) else "no extra ranking"

    if style == "train":
        return [
            f"找 {filter_text}，{sort_clause}，{limit_clause}。",
            f"帮我筛一批 {filter_text} 的候选，{sort_clause}，{limit_clause}。",
            f"把数据库里符合 {filter_text} 的 candidate 找出来，{sort_clause}，limit {limit}。",
            f"我要查 {filter_text} 的 protein candidate，{sort_clause}，top {limit}。",
            f"按真实 FBBP 字段编译这个请求：{filter_text}，{sort_clause}，{limit_clause}。",
            f"给我一个 candidate_search 计划，条件是 {filter_text}，{sort_clause}，{limit_clause}。",
            f"严格输出 slot sketch：scaffold_type 直接复制请求里的 scaffold 名，oral 只能写 High/Low/Medium，engineered 用 true/false。请求是：{filter_text}，{sort_clause}，{limit_clause}。",
            f"Compile this FBBP candidate request to JSON only: {anti_hype_filter_text}; rank={sort_clause_en}; top={limit}.",
            f"Need an executable protein-candidate query. Constraints: {anti_hype_filter_text}. Ranking: {sort_clause_en}. Return top {limit} only.",
            f"Convert the following retrieval request into FBBP JSON: {anti_hype_filter_text}; {sort_clause_en}; limit {limit}.",
            f"数据库筛选请求，不要总结：{anti_hype_filter_text}；按 {sort_clause_en}；top {limit}；输出 JSON。",
            f"只输出查询 JSON。需求如下：{anti_hype_filter_text}。排序规则：{sort_clause_en}。结果数：{limit}。",
            f"Translate this retrieval intent into compact slot-sketch JSON only: {anti_hype_filter_text}; ranking={sort_clause_en}; limit={limit}.",
        ]
    if style == "hard":
        return [
            f"帮我筛一批 {filter_text} 的候选，口径按真实数据库字段来，{sort_clause}，{limit_clause}。",
            f"按数据库字段理解这句话：{filter_text}，{sort_clause}，{limit_clause}。",
        ]
    if style == "dev":
        return [
            f"找 {filter_text}，{sort_clause}，limit {limit}。",
            f"筛 {filter_text}，{sort_clause}，前 {limit}。",
        ]
    return [_prompt_for_plan(plan, style)]


def _conversation(prompt: str, plan: dict[str, Any], row: dict[str, Any], include_annotations: bool = True) -> dict[str, Any]:
    sketch = plan_to_sketch(plan)
    record = {
        "conversations": build_compiler_messages(prompt)
        + [{"role": "assistant", "content": render_plan_json(sketch)}]
    }
    if include_annotations:
        record["prompt"] = prompt
        record["gold_plan"] = plan
        record["metadata"] = {
            "candidate_id": row["candidate_id"],
            "source_trace": row["trace"],
        }
    return record


def _bare_no_hints_conversation(prompt: str, plan: dict[str, Any], row: dict[str, Any], include_annotations: bool = True) -> dict[str, Any]:
    sketch = plan_to_sketch(plan)
    record = {
        "conversations": build_compiler_messages(prompt, include_hints=False, wrap_request=False)
        + [{"role": "assistant", "content": render_plan_json(sketch)}]
    }
    if include_annotations:
        record["prompt"] = prompt
        record["gold_plan"] = plan
        record["metadata"] = {
            "candidate_id": row["candidate_id"],
            "source_trace": row["trace"],
            "prompt_mode": "bare_no_hints",
        }
    return record


def _row_matches_engineered_value(row: dict[str, Any], engineered_value: bool | None) -> bool:
    if engineered_value is None:
        return False
    return row.get("engineered") is engineered_value


def _corrupted_repair_examples(plan: dict[str, Any]) -> list[tuple[str, list[str]]]:
    sketch = plan_to_sketch(plan)
    sketch_json = render_plan_json(sketch)

    corrupted: list[tuple[str, list[str]]] = [
        (render_plan_json({**sketch, "scaffold_type": None, "oral_class": None, "engineered": None, "has_experimental_affinity": None, "organism": None, "candidate_id": None}), ["empty_filters"]),
        (f"请修正这个 JSON: {sketch_json}", ["wrapped_prose"]),
        (render_plan_json({key: value for key, value in sketch.items() if key != "limit"}), ["invalid_limit"]),
    ]

    if sketch.get("scaffold_type") is not None:
        corrupted.append((render_plan_json({**sketch, "scaffold_type": "candidate_search"}), ["schema_key_as_value"]))
        corrupted.append((render_plan_json({**sketch, "scaffold_type": "best_affinity_value"}), ["semantic_mismatch_scaffold_type"]))
        corrupted.append((render_plan_json({**sketch, "scaffold_type": "candidate_id"}), ["schema_key_as_value", "semantic_mismatch_scaffold_type"]))
        corrupted.append((render_plan_json({**sketch, "scaffold_type": "ABC"}), ["invalid_slot_value", "semantic_mismatch_scaffold_type"]))
        alternative_scaffold = next((value for value in SCAFFOLD_TYPE_VALUES if value != sketch["scaffold_type"]), None)
        if alternative_scaffold is not None:
            corrupted.append((render_plan_json({**sketch, "scaffold_type": alternative_scaffold}), ["semantic_mismatch_scaffold_type"]))

    if sketch.get("oral_class") is not None:
        corrupted.append((render_plan_json({**sketch, "oral_class": "oral_class"}), ["schema_key_as_value"]))
        corrupted.append((render_plan_json({**sketch, "oral_class": "candidate_id"}), ["schema_key_as_value", "semantic_mismatch_oral_class"]))
        corrupted.append((render_plan_json({**sketch, "oral_class": "Medi"}), ["semantic_mismatch_oral_class"]))
        alternative_oral = next((value for value in ORAL_CLASS_VALUES if value != sketch["oral_class"]), None)
        if alternative_oral is not None:
            corrupted.append((render_plan_json({**sketch, "oral_class": alternative_oral}), ["semantic_mismatch_oral_class"]))

    if sketch.get("engineered") is not None:
        corrupted.append((render_plan_json({**sketch, "engineered": "Age"}), ["invalid_boolean_value"]))
        corrupted.append((render_plan_json({**sketch, "engineered": "candidate_id"}), ["schema_key_as_value", "semantic_mismatch_engineered"]))
        corrupted.append((render_plan_json({**sketch, "engineered": not sketch["engineered"]}), ["semantic_mismatch_engineered"]))

    if sketch.get("has_experimental_affinity") is True:
        corrupted.append((render_plan_json({**sketch, "has_experimental_affinity": False}), ["semantic_mismatch_has_experimental_affinity"]))

    if sketch.get("sort_by") is not None:
        corrupted.append((render_plan_json({**sketch, "sort_by": "affinity"}), ["invalid_sort"]))
        corrupted.append((sketch_json.replace('"sort_dir":"asc"', '"sort_dir":"\\")'), ["parse_failed", "invalid_sort"]))

    corrupted.append((render_plan_json({**sketch, "outer": "Low"}), ["unsupported_field_alias"]))
    corrupted.append((render_plan_json({**sketch, "limit": "top five"}), ["invalid_limit"]))
    if sketch.get("limit") in {5, 10, 20}:
        alternative_limit = next(value for value in (5, 10, 20) if value != sketch["limit"])
        corrupted.append((render_plan_json({**sketch, "limit": alternative_limit}), ["semantic_mismatch_limit"]))

    corrupted.append((sketch_json[:-1], ["parse_failed"]))
    corrupted.append((sketch_json[:-1] + ',"length":20', ["parse_failed"]))

    if '"scaffold_type":"' in sketch_json:
        corrupted.append((sketch_json.replace('"scaffold_type":', '"statper_class":', 1), ["semantic_mismatch_scaffold_type"]))
    if '"oral_class":"' in sketch_json:
        corrupted.append((sketch_json.replace('"oral_class":', '"higheral_class":', 1), ["semantic_mismatch_oral_class"]))
    if '"engineered":' in sketch_json:
        corrupted.append((sketch_json.replace('"engineered":', '"engimber_":', 1), ["semantic_mismatch_engineered"]))

    return corrupted


def _schema_word_corruption_examples(plan: dict[str, Any]) -> list[tuple[str, list[str]]]:
    sketch = plan_to_sketch(plan)
    corrupted: list[tuple[str, list[str]]] = []

    if sketch.get("scaffold_type") is not None:
        corrupted.append((render_plan_json({**sketch, "scaffold_type": "candidate_id"}), ["schema_key_as_value", "semantic_mismatch_scaffold_type"]))
        corrupted.append((render_plan_json({**sketch, "scaffold_type": "best_affinity_value"}), ["schema_key_as_value", "semantic_mismatch_scaffold_type"]))

    if sketch.get("oral_class") is not None:
        corrupted.append((render_plan_json({**sketch, "oral_class": "candidate_id"}), ["schema_key_as_value", "semantic_mismatch_oral_class"]))
        corrupted.append((render_plan_json({**sketch, "oral_class": "scaffold_type"}), ["schema_key_as_value", "semantic_mismatch_oral_class"]))

    if sketch.get("engineered") is not None:
        corrupted.append((render_plan_json({**sketch, "engineered": "candidate_id"}), ["schema_key_as_value", "semantic_mismatch_engineered"]))

    if sketch.get("sort_by") is not None:
        corrupted.append((render_plan_json({**sketch, "sort_by": "candidate_id"}), ["schema_key_as_value", "invalid_sort"]))

    return corrupted


def _targeted_completion_gate_corruption_examples(plan: dict[str, Any]) -> list[tuple[str, list[str]]]:
    sketch = plan_to_sketch(plan)
    corrupted: list[tuple[str, list[str]]] = []

    if sketch.get("engineered") is True:
        corrupted.append(
            (
                render_plan_json({**sketch, "engineered": False}),
                ["semantic_mismatch_engineered"],
            )
        )

    if sketch.get("has_experimental_affinity") is True:
        corrupted.append(
            (
                render_plan_json({**sketch, "has_experimental_affinity": "tightness"}),
                ["invalid_boolean_value", "semantic_mismatch_has_experimental_affinity"],
            )
        )
        corrupted.append(
            (
                render_plan_json(sketch).replace('"has_experimental_affinity":true', '"has_experimental_affinity":tightness', 1),
                ["parse_failed", "semantic_mismatch_has_experimental_affinity"],
            )
        )

    return corrupted


def _farthest_no_hints_repair_prompts(plan: dict[str, Any]) -> list[str]:
    return [
        _true_holdout_prompt_for_plan(plan, index=0),
        _v12_holdout_prompt_for_plan(plan, index=0),
        _v13_holdout_prompt_for_plan(plan, index=0),
        _engineered_bridge_prompt_for_plan(plan, index=0),
        _final_bridge_prompt_for_plan(plan, index=0),
    ]


def _repair_conversation(
    original_request: str,
    invalid_output: str,
    errors: list[str],
    plan: dict[str, Any],
    row: dict[str, Any],
    include_annotations: bool = True,
) -> dict[str, Any]:
    sketch = plan_to_sketch(plan)
    record = {
        "conversations": build_repair_messages(
            original_request=original_request,
            invalid_output=invalid_output,
            errors=errors,
        )
        + [{"role": "assistant", "content": render_plan_json(sketch)}]
    }
    if include_annotations:
        record["prompt"] = original_request
        record["gold_plan"] = plan
        record["metadata"] = {
            "candidate_id": row["candidate_id"],
            "source_trace": row["trace"],
            "repair_errors": errors,
        }
    return record


def _render_anti_hype_filters(plan: dict[str, Any]) -> str:
    parts: list[str] = []
    for flt in plan["filters"]:
        field = flt["field"]
        value = flt["value"]
        if field == "scaffold_type":
            parts.append(f"scaffold={value}")
        elif field == "oral_class":
            parts.append(f"oral {str(value).lower()}")
        elif field == "engineered":
            parts.append("engineered=true" if value else "non-engineered")
        elif field == "has_experimental_affinity":
            parts.append("experimental affinity required")
        elif field == "organism":
            parts.append(f"organism={value}")
        elif field == "candidate_id":
            parts.append(f"candidate_id={value}")
    return "; ".join(parts)


def _anti_hype_prompt_for_plan(plan: dict[str, Any], index: int) -> str:
    filter_text = _render_anti_hype_filters(plan)
    limit = plan["limit"]
    has_affinity_sort = any(item["field"] == "best_affinity_value" for item in plan["sort"])
    sort_text = "strongest affinity first" if has_affinity_sort else "no extra ranking"

    templates = [
        "Compile this FBBP candidate request to JSON only: {filters}; rank={sort}; top={limit}.",
        "Need an executable protein-candidate query. Constraints: {filters}. Ranking: {sort}. Return top {limit} only.",
        "请把这个检索需求编译成可执行 JSON，不要解释：{filters}；排序 {sort}；只要前 {limit} 个。",
        "只输出查询 JSON。需求如下：{filters}。排序规则：{sort}。结果数：{limit}。",
        "Convert the following retrieval request into FBBP JSON: {filters}; {sort}; limit {limit}.",
        "数据库筛选请求，不要总结：{filters}；按 {sort}；top {limit}；输出 JSON。",
    ]
    template = templates[index % len(templates)]
    return template.format(filters=filter_text, sort=sort_text, limit=limit)


def _render_true_holdout_clauses(plan: dict[str, Any], language: str) -> list[str]:
    clauses: list[str] = []
    filter_map = {flt["field"]: flt["value"] for flt in plan["filters"]}

    scaffold = filter_map.get("scaffold_type")
    if scaffold is not None:
        clauses.append(f"scaffold family {scaffold}" if language == "en" else f"骨架类型是 {scaffold}")

    oral = filter_map.get("oral_class")
    if oral is not None:
        clauses.append(f"oral class {oral}" if language == "en" else f"oral 分档是 {oral}")

    engineered = filter_map.get("engineered")
    if engineered is True:
        clauses.append("engineered molecules only" if language == "en" else "只看 engineered 条目")
    elif engineered is False:
        clauses.append("non-engineered molecules only" if language == "en" else "只看 non-engineered 条目")

    if filter_map.get("has_experimental_affinity") is True:
        clauses.append("experimental affinity evidence must be present" if language == "en" else "必须带实验亲和力证据")

    organism = filter_map.get("organism")
    if organism is not None:
        clauses.append(f"organism {organism}" if language == "en" else f"来源物种是 {organism}")

    candidate_id = filter_map.get("candidate_id")
    if candidate_id is not None:
        clauses.append(f"candidate id {candidate_id}" if language == "en" else f"候选编号是 {candidate_id}")

    return clauses


def _true_holdout_prompt_templates() -> list[Any]:
    return [
        lambda c_en, c_cn, rank_en, rank_cn, limit: f"From the FBBP registry, pull candidates with {', '.join(c_en)}; {rank_en}; stop after {limit} results.",
        lambda c_en, c_cn, rank_en, rank_cn, limit: f"I am shortlisting entries from the candidate table. Keep records that have {', '.join(c_en)} and then {rank_en}; cap the list at {limit}.",
        lambda c_en, c_cn, rank_en, rank_cn, limit: f"For this screen, I only want molecules with {', '.join(c_en)}. After ranking by binding strength, return the first {limit}.",
        lambda c_en, c_cn, rank_en, rank_cn, limit: f"请按数据库语义理解这个检索：我要 {'，'.join(c_cn)} 的候选，{rank_cn}，最多保留 {limit} 条。",
        lambda c_en, c_cn, rank_en, rank_cn, limit: f"我在库里做初筛，只看 {'，'.join(c_cn)} 的记录，最后 {rank_cn}，只留前 {limit} 个。",
        lambda c_en, c_cn, rank_en, rank_cn, limit: f"研究问题是这样的：候选必须满足 {'，'.join(c_cn)}；结果部分请 {rank_cn}；返回数量不要超过 {limit}。",
        lambda c_en, c_cn, rank_en, rank_cn, limit: f"Keep a database shortlist for candidates showing {', '.join(c_en)}. Order the list with the best affinity first and keep only {limit}.",
        lambda c_en, c_cn, rank_en, rank_cn, limit: f"做一个候选清单给我，条件是 {'，'.join(c_cn)}，并且 {rank_cn}，最终看前 {limit} 条。",
    ]


def _true_holdout_prompt_for_plan(plan: dict[str, Any], index: int) -> str:
    limit = plan["limit"]
    has_affinity_sort = any(item["field"] == "best_affinity_value" for item in plan["sort"])
    rank_en = "keep the strongest binders first" if has_affinity_sort else "no extra ranking is needed"
    rank_cn = "把结合最强的放在最前面" if has_affinity_sort else "不需要额外排序"

    english_clauses = _render_true_holdout_clauses(plan, language="en")
    chinese_clauses = _render_true_holdout_clauses(plan, language="zh")
    templates = _true_holdout_prompt_templates()
    template = templates[index % len(templates)]
    return template(english_clauses, chinese_clauses, rank_en, rank_cn, limit)


def _v12_holdout_prompt_templates() -> list[Any]:
    return [
        lambda c_en, c_cn, rank_en, rank_cn, limit: f"Search the candidate registry for entries satisfying {', '.join(c_en)}. Keep the top {limit} after sorting by binding potency.",
        lambda c_en, c_cn, rank_en, rank_cn, limit: f"I need a candidate shortlist restricted to {', '.join(c_en)}; prioritize better affinity values and trim the output to {limit}.",
        lambda c_en, c_cn, rank_en, rank_cn, limit: f"Within the protein-candidate table, retain only rows with {', '.join(c_en)}. Rank by affinity quality and keep no more than {limit} hits.",
        lambda c_en, c_cn, rank_en, rank_cn, limit: f"请直接按数据库语义筛选：候选需要满足 {'、'.join(c_cn)}，随后按结合强度优先，只保留 {limit} 条。",
        lambda c_en, c_cn, rank_en, rank_cn, limit: f"做一个不带解释的候选筛选，约束是 {'、'.join(c_cn)}，并按亲和力优先，输出不要超过 {limit} 条。",
        lambda c_en, c_cn, rank_en, rank_cn, limit: f"我想从候选库里锁定 {'、'.join(c_cn)} 的分子，排序时先看最强结合，名单控制在 {limit} 个以内。",
        lambda c_en, c_cn, rank_en, rank_cn, limit: f"Build a database shortlist for candidates meeting {', '.join(c_en)}. Put the best binders first and stop once {limit} rows remain.",
        lambda c_en, c_cn, rank_en, rank_cn, limit: f"候选初筛条件如下：{'、'.join(c_cn)}；请先看结合最强的，再把结果截到 {limit} 条。",
    ]


def _v12_holdout_prompt_for_plan(plan: dict[str, Any], index: int) -> str:
    limit = plan["limit"]
    has_affinity_sort = any(item["field"] == "best_affinity_value" for item in plan["sort"])
    rank_en = "prioritize stronger affinity" if has_affinity_sort else "no extra ranking is needed"
    rank_cn = "按结合强度优先" if has_affinity_sort else "不需要额外排序"
    english_clauses = _render_true_holdout_clauses(plan, language="en")
    chinese_clauses = _render_true_holdout_clauses(plan, language="zh")
    templates = _v12_holdout_prompt_templates()
    template = templates[index % len(templates)]
    return template(english_clauses, chinese_clauses, rank_en, rank_cn, limit)


def _v13_holdout_prompt_templates() -> list[Any]:
    return [
        lambda c_en, c_cn, rank_en, rank_cn, limit: f"Filter the candidate inventory down to rows annotated with {', '.join(c_en)}. Then rank the survivors by measured binding quality and return at most {limit}.",
        lambda c_en, c_cn, rank_en, rank_cn, limit: f"For downstream review, retain only protein candidates tagged with {', '.join(c_en)}. Put the highest-priority binders first and show no more than {limit} records.",
        lambda c_en, c_cn, rank_en, rank_cn, limit: f"Database query request: among candidate entries, keep those marked as {', '.join(c_en)}; sort from best measured affinity to weaker binding and stop at {limit}.",
        lambda c_en, c_cn, rank_en, rank_cn, limit: f"请从候选表里筛出满足 {'、'.join(c_cn)} 的条目，然后按测得亲和力优先排序，结果上限 {limit} 条。",
        lambda c_en, c_cn, rank_en, rank_cn, limit: f"只保留 {'、'.join(c_cn)} 的候选，查看时先放最强结合的记录，最多输出 {limit} 条。",
        lambda c_en, c_cn, rank_en, rank_cn, limit: f"我要一个数据库候选列表，要求 {'、'.join(c_cn)}，并按最佳亲和力优先，最终不要超过 {limit} 项。",
        lambda c_en, c_cn, rank_en, rank_cn, limit: f"Keep candidate rows carrying {', '.join(c_en)}. Arrange the output by strongest measured affinity and cap the response at {limit}.",
        lambda c_en, c_cn, rank_en, rank_cn, limit: f"候选清单需求：{'、'.join(c_cn)}；结果按最优结合强度排序；输出限制为 {limit} 条。",
    ]


def _v13_holdout_prompt_for_plan(plan: dict[str, Any], index: int) -> str:
    limit = plan["limit"]
    has_affinity_sort = any(item["field"] == "best_affinity_value" for item in plan["sort"])
    rank_en = "best measured affinity first" if has_affinity_sort else "no extra ranking is needed"
    rank_cn = "按最佳亲和力优先" if has_affinity_sort else "不需要额外排序"
    english_clauses = _render_true_holdout_clauses(plan, language="en")
    chinese_clauses = _render_true_holdout_clauses(plan, language="zh")
    templates = _v13_holdout_prompt_templates()
    template = templates[index % len(templates)]
    return template(english_clauses, chinese_clauses, rank_en, rank_cn, limit)


def _v15_holdout_prompt_templates() -> list[Any]:
    return [
        lambda c_en, c_cn, rank_en, rank_cn, limit: f"Within the curation pass, keep candidate records tagged with {', '.join(c_en)}. Let measured binder priority decide the order and cut the sheet down to {limit} rows.",
        lambda c_en, c_cn, rank_en, rank_cn, limit: f"Need a review-ready candidate slate where the annotations say {', '.join(c_en)}. Arrange entries by tighter measured binding before weaker ones; output ceiling {limit}.",
        lambda c_en, c_cn, rank_en, rank_cn, limit: f"From the candidate ledger, retain only rows carrying {', '.join(c_en)}. Use experimental binder strength as the ordering signal and stop once {limit} remain.",
        lambda c_en, c_cn, rank_en, rank_cn, limit: f"请在候选账本里保留标注为 {'、'.join(c_cn)} 的条目，按实测结合更强者靠前，最终收敛到 {limit} 条。",
        lambda c_en, c_cn, rank_en, rank_cn, limit: f"我只要注释上写着 {'、'.join(c_cn)} 的候选，查看顺序按实测结合由强到弱，名单压到 {limit} 条。",
        lambda c_en, c_cn, rank_en, rank_cn, limit: f"做一个候选留样单：满足 {'、'.join(c_cn)}；排序依据是实测结合更强者靠前；结果封顶 {limit} 条。",
        lambda c_en, c_cn, rank_en, rank_cn, limit: f"Keep a curation-ready shortlist for rows labeled {', '.join(c_en)}. Resolve ordering with measured binding tightness and keep the sheet at {limit} rows or fewer.",
        lambda c_en, c_cn, rank_en, rank_cn, limit: f"库内复核时只看 {'、'.join(c_cn)} 的候选，排序遵循实测结合更强优先，最后不要留过 {limit} 条。",
    ]


def _v15_holdout_prompt_for_plan(plan: dict[str, Any], index: int) -> str:
    limit = plan["limit"]
    has_affinity_sort = any(item["field"] == "best_affinity_value" for item in plan["sort"])
    rank_en = "measured binding tighter to weaker" if has_affinity_sort else "no extra ranking is needed"
    rank_cn = "实测结合更强者靠前" if has_affinity_sort else "不需要额外排序"
    english_clauses = _render_true_holdout_clauses(plan, language="en")
    chinese_clauses = _render_true_holdout_clauses(plan, language="zh")
    templates = _v15_holdout_prompt_templates()
    template = templates[index % len(templates)]
    return template(english_clauses, chinese_clauses, rank_en, rank_cn, limit)


def _engineered_bridge_prompt_templates() -> list[Any]:
    return [
        lambda c_en, c_cn, rank_en, rank_cn, limit: f"For the curation worksheet, keep entries whose annotation block says {', '.join(c_en)}. Rank them by measured binding strength and return at most {limit}.",
        lambda c_en, c_cn, rank_en, rank_cn, limit: f"I need a review sheet of candidates marked {', '.join(c_en)}. Order the rows by observed affinity quality and cap the list at {limit}.",
        lambda c_en, c_cn, rank_en, rank_cn, limit: f"Screen candidate annotations for {', '.join(c_en)}. Use measured binding evidence as the ranking signal and stop at {limit} results.",
        lambda c_en, c_cn, rank_en, rank_cn, limit: f"做数据库复核单，只保留注释栏注明 {'、'.join(c_cn)} 的候选，按实测结合强度排序，最多 {limit} 条。",
        lambda c_en, c_cn, rank_en, rank_cn, limit: f"候选审阅表里只放 {'、'.join(c_cn)} 的记录，顺序参考实测亲和力强弱，数量上限 {limit}。",
        lambda c_en, c_cn, rank_en, rank_cn, limit: f"按注释信息筛候选：满足 {'、'.join(c_cn)}，再按实测结合由强到弱排，保留前 {limit} 条。",
    ]


def _engineered_bridge_prompt_for_plan(plan: dict[str, Any], index: int) -> str:
    limit = plan["limit"]
    has_affinity_sort = any(item["field"] == "best_affinity_value" for item in plan["sort"])
    rank_en = "measured binding tighter to weaker" if has_affinity_sort else "no extra ranking is needed"
    rank_cn = "实测结合更强者靠前" if has_affinity_sort else "不需要额外排序"
    english_clauses = _render_true_holdout_clauses(plan, language="en")
    chinese_clauses = _render_true_holdout_clauses(plan, language="zh")
    templates = _engineered_bridge_prompt_templates()
    template = templates[index % len(templates)]
    return template(english_clauses, chinese_clauses, rank_en, rank_cn, limit)


def _final_bridge_prompt_templates() -> list[Any]:
    return [
        lambda c_en, c_cn, rank_en, rank_cn, limit: f"Review the candidate ledger and keep entries annotated with {', '.join(c_en)}. Let measured binder tightness set the order and stop at {limit} rows.",
        lambda c_en, c_cn, rank_en, rank_cn, limit: f"Prepare a shortlist sheet for candidate records carrying {', '.join(c_en)}. Use experimental binding tightness as the priority signal and cap the output at {limit}.",
        lambda c_en, c_cn, rank_en, rank_cn, limit: f"For curation handoff, retain only candidate rows marked {', '.join(c_en)}. Rank them by measured binder evidence and keep no more than {limit} results.",
        lambda c_en, c_cn, rank_en, rank_cn, limit: f"做候选账页复核，只留下标注为 {'、'.join(c_cn)} 的条目，排序看实测结合紧密度，最多 {limit} 条。",
        lambda c_en, c_cn, rank_en, rank_cn, limit: f"整理一个候选留样清单，记录需要满足 {'、'.join(c_cn)}，优先级按实验结合紧密度走，数量压到 {limit} 条以内。",
        lambda c_en, c_cn, rank_en, rank_cn, limit: f"交付复核时，只保留 {'、'.join(c_cn)} 的候选，顺序由实测结合证据决定，最终不超过 {limit} 条。",
    ]


def _final_bridge_prompt_for_plan(plan: dict[str, Any], index: int) -> str:
    limit = plan["limit"]
    has_affinity_sort = any(item["field"] == "best_affinity_value" for item in plan["sort"])
    rank_en = "measured binding tighter to weaker" if has_affinity_sort else "no extra ranking is needed"
    rank_cn = "实测结合更强者靠前" if has_affinity_sort else "不需要额外排序"
    english_clauses = _render_true_holdout_clauses(plan, language="en")
    chinese_clauses = _render_true_holdout_clauses(plan, language="zh")
    templates = _final_bridge_prompt_templates()
    template = templates[index % len(templates)]
    return template(english_clauses, chinese_clauses, rank_en, rank_cn, limit)


def _completion_english_bridge_prompt_templates() -> list[Any]:
    return [
        lambda clauses, limit: f"During curation review, keep candidate rows annotated with {', '.join(clauses)}. Let measured binder ranking determine the order and trim the sheet to {limit} rows.",
        lambda clauses, limit: f"In the candidate ledger, retain rows carrying {', '.join(clauses)}. Use measured binding strength to order the records and stop after {limit}.",
        lambda clauses, limit: f"Keep a curation shortlist for entries labeled {', '.join(clauses)}. Resolve the order with measured binding tightness and leave no more than {limit} rows.",
        lambda clauses, limit: f"Build a reviewer slate for candidates whose annotations read {', '.join(clauses)}. Arrange the list by tighter measured binding and cap the output at {limit}.",
        lambda clauses, limit: f"For screening handoff, keep only records marked {', '.join(clauses)}. Rank them by measured binder priority and return up to {limit} rows.",
    ]


def _completion_english_bridge_prompt_for_plan(plan: dict[str, Any], index: int) -> str:
    clauses = _render_true_holdout_clauses(plan, language="en")
    templates = _completion_english_bridge_prompt_templates()
    template = templates[index % len(templates)]
    return template(clauses, plan["limit"])


def _final_knottin_high_bridge_prompt_templates() -> list[Any]:
    return [
        lambda clauses, limit: f"From the candidate ledger, keep rows carrying {', '.join(clauses)}. Use experimental binder strength as the ordering signal and stop after {limit} remain.",
        lambda clauses, limit: f"Keep a curation-ready shortlist of rows labeled {', '.join(clauses)}. Resolve ordering with measured binding tightness and keep the sheet at {limit} rows or fewer.",
    ]


def _final_knottin_high_bridge_prompt_for_plan(plan: dict[str, Any], index: int) -> str:
    clauses = _render_true_holdout_clauses(plan, language="en")
    templates = _final_knottin_high_bridge_prompt_templates()
    template = templates[index % len(templates)]
    return template(clauses, plan["limit"])


def _hotspot_shortlist_contrast_prompt_templates() -> list[Any]:
    return [
        lambda clauses, limit: f"Keep a curation-ready shortlist for candidate rows labeled {', '.join(clauses)}. Resolve ordering with measured binding tightness and keep the sheet at {limit} rows or fewer.",
        lambda clauses, limit: f"Prepare a curation-ready shortlist for rows labeled {', '.join(clauses)}. Use measured binder priority to order the sheet and stop once {limit} rows remain.",
        lambda clauses, limit: f"Need a curation-ready shortlist for rows labeled {', '.join(clauses)}. Let measured binding tightness determine the order and cap the list at {limit}.",
    ]


def _hotspot_shortlist_contrast_prompt_for_plan(plan: dict[str, Any], index: int) -> str:
    clauses = _render_true_holdout_clauses(plan, language="en")
    templates = _hotspot_shortlist_contrast_prompt_templates()
    template = templates[index % len(templates)]
    return template(clauses, plan["limit"])


def _hotspot_exact_prefix_prompt_templates() -> list[Any]:
    return [
        lambda clauses, limit: f"Keep a curation-ready shortlist for rows labeled {', '.join(clauses)}. Use measured binder strength as the ordering signal and stop once {limit} remain.",
        lambda clauses, limit: f"Keep a curation-ready shortlist for rows labeled {', '.join(clauses)}. Let measured binder priority decide the order and cap the list at {limit}.",
        lambda clauses, limit: f"Keep a curation-ready shortlist for rows labeled {', '.join(clauses)}. Arrange entries by tighter measured binding before weaker ones and keep no more than {limit}.",
    ]


def _hotspot_exact_prefix_prompt_for_plan(plan: dict[str, Any], index: int) -> str:
    clauses = _render_true_holdout_clauses(plan, language="en")
    templates = _hotspot_exact_prefix_prompt_templates()
    template = templates[index % len(templates)]
    return template(clauses, plan["limit"])


def _is_projection_hotspot_plan(plan: dict[str, Any]) -> bool:
    sketch = plan_to_sketch(plan)
    return (
        sketch.get("scaffold_type") == "knottin"
        and sketch.get("oral_class") == "High"
        and sketch.get("engineered") is True
        and sketch.get("has_experimental_affinity") is True
    )


def build_query_compiler_dataset(
    output_dir: Path | None = None,
    seed: int = 7,
    train_size: int = 600,
    train_variants_per_row: int = 3,
    repair_examples_per_row: int = 0,
    bare_no_hints_examples_per_row: int = 0,
    farther_no_hints_examples_per_row: int = 0,
    engineered_true_no_hints_examples_per_row: int = 0,
    final_bridge_examples_per_row: int = 0,
    completion_english_bridge_examples_per_row: int = 0,
    projection_hotspot_examples_per_row: int = 0,
    hotspot_shortlist_contrast_examples_per_row: int = 0,
    schema_word_examples_per_row: int = 0,
    dev_size: int = 40,
    test_seen_size: int = 40,
    test_hard_size: int = 40,
) -> dict[str, Any]:
    output_dir = output_dir or DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    rows = build_candidate_snapshot()
    rng.shuffle(rows)

    total_needed = min(len(rows), train_size + dev_size + test_seen_size + test_hard_size)
    selected = rows[:total_needed]

    train_rows = selected[:train_size]
    dev_rows = selected[train_size:train_size + dev_size]
    test_seen_rows = selected[train_size + dev_size:train_size + dev_size + test_seen_size]
    test_hard_rows = selected[train_size + dev_size + test_seen_size:total_needed]

    def build_examples(
        source_rows: list[dict[str, Any]],
        style: str,
        include_annotations: bool = True,
        variants_per_row: int = 1,
    ) -> list[dict[str, Any]]:
        examples: list[dict[str, Any]] = []
        for row in source_rows:
            plan = _plan_from_candidate(row, rng)
            variants = _prompt_variants_for_plan(plan, style=style)
            if style == "train":
                rng.shuffle(variants)
            for prompt in variants[:variants_per_row]:
                examples.append(_conversation(prompt, plan, row, include_annotations=include_annotations))
        return examples

    def build_repair_examples(
        source_rows: list[dict[str, Any]],
        include_annotations: bool = False,
        examples_per_row: int = 0,
        add_schema_word_negatives: int = 0,
    ) -> list[dict[str, Any]]:
        examples: list[dict[str, Any]] = []
        if examples_per_row <= 0 and add_schema_word_negatives <= 0:
            return examples

        for row in source_rows:
            plan = _plan_from_candidate(row, rng)
            targeted_corrupted = _targeted_completion_gate_corruption_examples(plan)
            corrupted_examples = _corrupted_repair_examples(plan)
            if add_schema_word_negatives > 0:
                corrupted_examples.extend(_schema_word_corruption_examples(plan))
            prompts = [_prompt_variants_for_plan(plan, style="train")[0]]
            prompts.extend(_farthest_no_hints_repair_prompts(plan))
            prompts.extend(
                _completion_english_bridge_prompt_for_plan(plan, index=offset)
                for offset in range(2)
            )
            if _is_projection_hotspot_plan(plan):
                hotspot_prompts = [
                    _final_knottin_high_bridge_prompt_for_plan(plan, index=offset)
                    for offset in range(2)
                ]
                hotspot_prompts.extend(
                    _hotspot_shortlist_contrast_prompt_for_plan(plan, index=offset)
                    for offset in range(2)
                )
                hotspot_prompts.extend(
                    _hotspot_exact_prefix_prompt_for_plan(plan, index=offset)
                    for offset in range(2)
                )
                prompts = hotspot_prompts + prompts
            rng.shuffle(prompts)
            rng.shuffle(corrupted_examples)
            max_examples = max(0, examples_per_row) + max(0, add_schema_word_negatives)
            selected_corrupted: list[tuple[str, list[str]]] = []
            seen_keys: set[str] = set()
            for invalid_output, errors in targeted_corrupted:
                key = f"{invalid_output}|{'|'.join(errors)}"
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                selected_corrupted.append((invalid_output, errors))
                if len(selected_corrupted) >= max_examples:
                    break
            if len(selected_corrupted) < max_examples:
                for invalid_output, errors in corrupted_examples:
                    key = f"{invalid_output}|{'|'.join(errors)}"
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)
                    selected_corrupted.append((invalid_output, errors))
                    if len(selected_corrupted) >= max_examples:
                        break
            for index, (invalid_output, errors) in enumerate(selected_corrupted):
                original_prompt = prompts[index % len(prompts)]
                examples.append(
                    _repair_conversation(
                        original_request=original_prompt,
                        invalid_output=invalid_output,
                        errors=errors,
                        plan=plan,
                        row=row,
                        include_annotations=include_annotations,
                    )
                )
        return examples

    def build_bare_no_hints_examples(
        source_rows: list[dict[str, Any]],
        examples_per_row: int = 0,
    ) -> list[dict[str, Any]]:
        examples: list[dict[str, Any]] = []
        if examples_per_row <= 0:
            return examples

        for row_index, row in enumerate(source_rows):
            plan = _plan_from_candidate(row, rng)
            variants = [
                _true_holdout_prompt_for_plan(plan, index=row_index * examples_per_row + offset)
                for offset in range(examples_per_row)
            ]
            for prompt in variants:
                examples.append(_bare_no_hints_conversation(prompt, plan, row, include_annotations=False))
        return examples

    def build_farther_no_hints_examples(
        source_rows: list[dict[str, Any]],
        examples_per_row: int = 0,
    ) -> list[dict[str, Any]]:
        examples: list[dict[str, Any]] = []
        if examples_per_row <= 0:
            return examples

        for row_index, row in enumerate(source_rows):
            plan = _plan_from_candidate(row, rng)
            variants = [
                _v12_holdout_prompt_for_plan(plan, index=row_index * examples_per_row + offset)
                for offset in range(examples_per_row)
            ]
            for prompt in variants:
                examples.append(_bare_no_hints_conversation(prompt, plan, row, include_annotations=False))
        return examples

    def oversample_engineered_true_no_hints(
        source_rows: list[dict[str, Any]],
        examples_per_row: int = 0,
    ) -> list[dict[str, Any]]:
        examples: list[dict[str, Any]] = []
        if examples_per_row <= 0:
            return examples

        engineered_rows = [row for row in source_rows if _row_matches_engineered_value(row, True)]
        for row_index, row in enumerate(engineered_rows):
            plan = _plan_from_candidate(row, rng)
            variants = [
                _true_holdout_prompt_for_plan(plan, index=row_index * examples_per_row + offset)
                for offset in range(examples_per_row)
            ]
            variants.extend(
                _v12_holdout_prompt_for_plan(plan, index=row_index * examples_per_row + offset)
                for offset in range(examples_per_row)
            )
            variants.extend(
                _v13_holdout_prompt_for_plan(plan, index=row_index * examples_per_row + offset)
                for offset in range(examples_per_row)
            )
            variants.extend(
                _engineered_bridge_prompt_for_plan(plan, index=row_index * examples_per_row + offset)
                for offset in range(examples_per_row)
            )
            for prompt in variants:
                examples.append(_bare_no_hints_conversation(prompt, plan, row, include_annotations=False))
        return examples

    def build_final_bridge_examples(
        source_rows: list[dict[str, Any]],
        examples_per_row: int = 0,
    ) -> list[dict[str, Any]]:
        examples: list[dict[str, Any]] = []
        if examples_per_row <= 0:
            return examples

        affinity_rows = [row for row in source_rows if row.get("has_experimental_affinity")]
        for row_index, row in enumerate(affinity_rows):
            plan = _plan_from_candidate(row, rng)
            variants = [
                _final_bridge_prompt_for_plan(plan, index=row_index * examples_per_row + offset)
                for offset in range(examples_per_row)
            ]
            for prompt in variants:
                examples.append(_bare_no_hints_conversation(prompt, plan, row, include_annotations=False))
        return examples

    def build_completion_english_bridge_examples(
        source_rows: list[dict[str, Any]],
        examples_per_row: int = 0,
    ) -> list[dict[str, Any]]:
        examples: list[dict[str, Any]] = []
        if examples_per_row <= 0:
            return examples

        engineered_rows = [row for row in source_rows if _row_matches_engineered_value(row, True)]
        for row_index, row in enumerate(engineered_rows):
            plan = _plan_from_candidate(row, rng)
            variants = [
                _completion_english_bridge_prompt_for_plan(plan, index=row_index * examples_per_row + offset)
                for offset in range(examples_per_row)
            ]
            if _is_projection_hotspot_plan(plan):
                variants.extend(
                    _final_knottin_high_bridge_prompt_for_plan(plan, index=offset)
                    for offset in range(2)
                )
            for prompt in variants:
                examples.append(_bare_no_hints_conversation(prompt, plan, row, include_annotations=False))
        return examples

    def build_projection_hotspot_examples(
        source_rows: list[dict[str, Any]],
        examples_per_row: int = 0,
    ) -> list[dict[str, Any]]:
        examples: list[dict[str, Any]] = []
        if examples_per_row <= 0:
            return examples

        hotspot_rows = []
        for row in source_rows:
            plan = _plan_from_candidate(row, rng)
            if _is_projection_hotspot_plan(plan):
                hotspot_rows.append((row, plan))

        for row_index, (row, plan) in enumerate(hotspot_rows):
            variants: list[str] = []
            variants.extend(
                _final_knottin_high_bridge_prompt_for_plan(plan, index=row_index * examples_per_row + offset)
                for offset in range(examples_per_row)
            )
            variants.extend(
                _completion_english_bridge_prompt_for_plan(plan, index=row_index * examples_per_row + offset)
                for offset in range(examples_per_row)
            )
            variants.extend(
                _hotspot_exact_prefix_prompt_for_plan(plan, index=row_index * examples_per_row + offset)
                for offset in range(examples_per_row)
            )
            for prompt in variants:
                examples.append(_bare_no_hints_conversation(prompt, plan, row, include_annotations=False))
        return examples

    def build_hotspot_shortlist_contrast_examples(
        source_rows: list[dict[str, Any]],
        examples_per_row: int = 0,
    ) -> list[dict[str, Any]]:
        examples: list[dict[str, Any]] = []
        if examples_per_row <= 0:
            return examples

        contrast_rows = []
        for row in source_rows:
            plan = _plan_from_candidate(row, rng)
            sketch = plan_to_sketch(plan)
            if (
                sketch.get("scaffold_type") == "knottin"
                and sketch.get("oral_class") == "High"
                and sketch.get("has_experimental_affinity") is True
                and sketch.get("engineered") in {True, False}
            ):
                contrast_rows.append((row, plan))

        for row_index, (row, plan) in enumerate(contrast_rows):
            variants = [
                _hotspot_shortlist_contrast_prompt_for_plan(plan, index=row_index * examples_per_row + offset)
                for offset in range(examples_per_row)
            ]
            if _is_projection_hotspot_plan(plan):
                variants.extend(
                    _hotspot_exact_prefix_prompt_for_plan(plan, index=row_index * examples_per_row + offset)
                    for offset in range(examples_per_row)
                )
            for prompt in variants:
                examples.append(_bare_no_hints_conversation(prompt, plan, row, include_annotations=False))
        return examples

    train_examples = build_examples(
        train_rows,
        style="train",
        include_annotations=False,
        variants_per_row=max(1, train_variants_per_row),
    )
    bare_no_hints_examples = build_bare_no_hints_examples(
        train_rows,
        examples_per_row=max(0, bare_no_hints_examples_per_row),
    )
    farther_no_hints_examples = build_farther_no_hints_examples(
        train_rows,
        examples_per_row=max(0, farther_no_hints_examples_per_row),
    )
    engineered_true_no_hints_examples = oversample_engineered_true_no_hints(
        train_rows,
        examples_per_row=max(0, engineered_true_no_hints_examples_per_row),
    )
    final_bridge_examples = build_final_bridge_examples(
        train_rows,
        examples_per_row=max(0, final_bridge_examples_per_row),
    )
    completion_english_bridge_examples = build_completion_english_bridge_examples(
        train_rows,
        examples_per_row=max(0, completion_english_bridge_examples_per_row),
    )
    projection_hotspot_examples = build_projection_hotspot_examples(
        train_rows,
        examples_per_row=max(0, projection_hotspot_examples_per_row),
    )
    hotspot_shortlist_contrast_examples = build_hotspot_shortlist_contrast_examples(
        train_rows,
        examples_per_row=max(0, hotspot_shortlist_contrast_examples_per_row),
    )
    repair_examples = build_repair_examples(
        train_rows,
        include_annotations=False,
        examples_per_row=max(0, repair_examples_per_row),
        add_schema_word_negatives=max(0, schema_word_examples_per_row),
    )
    train_examples.extend(bare_no_hints_examples)
    train_examples.extend(farther_no_hints_examples)
    train_examples.extend(engineered_true_no_hints_examples)
    train_examples.extend(final_bridge_examples)
    train_examples.extend(completion_english_bridge_examples)
    train_examples.extend(projection_hotspot_examples)
    train_examples.extend(hotspot_shortlist_contrast_examples)
    train_examples.extend(repair_examples)
    dev_examples = build_examples(dev_rows, style="dev")
    test_seen_examples = build_examples(test_seen_rows, style="seen")
    test_hard_examples = build_examples(test_hard_rows, style="hard")

    train_path = output_dir / "fbbp_query_compiler_train.jsonl"
    dev_path = output_dir / "fbbp_query_compiler_dev.jsonl"
    test_seen_path = output_dir / "fbbp_query_compiler_test_seen.jsonl"
    test_hard_path = output_dir / "fbbp_query_compiler_test_hard.jsonl"
    eval_path = output_dir / "fbbp_query_compiler_eval_prompts.jsonl"
    manifest_path = output_dir / "fbbp_query_compiler_manifest.json"

    _write_jsonl(train_path, train_examples)
    _write_jsonl(dev_path, dev_examples)
    _write_jsonl(test_seen_path, test_seen_examples)
    _write_jsonl(test_hard_path, test_hard_examples)

    eval_prompts: list[dict[str, Any]] = []
    for index, example in enumerate(test_seen_examples + test_hard_examples, start=1):
        eval_prompts.append(
            {
                "id": f"query_compiler_{index}",
                "category": "fbbp-query-compiler",
                "prompt": example["prompt"],
                "gold_plan": example["gold_plan"],
                "reference_note": "Should output candidate_search JSON for protein_candidate.",
            }
        )
    _write_jsonl(eval_path, eval_prompts)

    manifest = {
        "seed": seed,
        "counts": {
            "train": len(train_examples),
            "bare_no_hints_examples": len(bare_no_hints_examples),
            "farther_no_hints_examples": len(farther_no_hints_examples),
            "engineered_true_no_hints_examples": len(engineered_true_no_hints_examples),
            "final_bridge_examples": len(final_bridge_examples),
            "completion_english_bridge_examples": len(completion_english_bridge_examples),
            "projection_hotspot_examples": len(projection_hotspot_examples),
            "hotspot_shortlist_contrast_examples": len(hotspot_shortlist_contrast_examples),
            "repair_examples": len(repair_examples),
            "dev": len(dev_examples),
            "test_seen": len(test_seen_examples),
            "test_hard": len(test_hard_examples),
            "eval_prompts": len(eval_prompts),
        },
        "source_snapshot_rows": len(rows),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def build_query_compiler_anti_hype_eval(
    output_path: Path,
    audit_path: Path | None = None,
    seed: int = 97,
    eval_size: int = 80,
    train_size: int = 600,
    dev_size: int = 40,
    test_seen_size: int = 40,
    test_hard_size: int = 40,
) -> dict[str, Any]:
    rng = random.Random(seed)
    rows = build_candidate_snapshot()
    rng.shuffle(rows)

    seen_cutoff = min(len(rows), train_size + dev_size + test_seen_size + test_hard_size)
    holdout_rows = rows[seen_cutoff:]
    if not holdout_rows:
        holdout_rows = rows

    eval_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for index, row in enumerate(holdout_rows):
        plan = _plan_from_candidate(row, rng)
        prompt = _anti_hype_prompt_for_plan(plan, index=index)
        item = {
            "id": f"anti_hype_query_compiler_{len(eval_rows) + 1}",
            "category": "fbbp-query-compiler-anti-hype",
            "prompt": prompt,
            "gold_plan": plan,
            "reference_note": "Out-of-template anti-hype holdout prompt.",
        }
        eval_rows.append(item)
        if len(audit_rows) < 20:
            audit_rows.append(item)
        if len(eval_rows) >= eval_size:
            break

    _write_jsonl(output_path, eval_rows)

    if audit_path is not None:
        lines = [
            "# FBBP Query Compiler Anti-Hype Audit",
            "",
            f"- seed: `{seed}`",
            f"- eval_size: `{len(eval_rows)}`",
            f"- holdout_offset: `{seen_cutoff}`",
            "",
        ]
        for row in audit_rows:
            lines.append(f"## {row['id']}")
            lines.append("")
            lines.append(f"- Prompt: `{row['prompt']}`")
            lines.append(f"- Gold: `{json.dumps(row['gold_plan'], ensure_ascii=False)}`")
            lines.append("")
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "seed": seed,
        "eval_size": len(eval_rows),
        "holdout_offset": seen_cutoff,
        "audit_examples": len(audit_rows),
        "source_snapshot_rows": len(rows),
        "output_path": str(output_path),
    }


def build_query_compiler_true_holdout_eval(
    output_path: Path,
    audit_path: Path | None = None,
    seed: int = 131,
    eval_size: int = 80,
    train_size: int = 600,
    dev_size: int = 40,
    test_seen_size: int = 40,
    test_hard_size: int = 40,
) -> dict[str, Any]:
    rng = random.Random(seed)
    rows = build_candidate_snapshot()
    rng.shuffle(rows)

    seen_cutoff = min(len(rows), train_size + dev_size + test_seen_size + test_hard_size)
    holdout_rows = rows[seen_cutoff:]
    if not holdout_rows:
        holdout_rows = rows

    eval_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for index, row in enumerate(holdout_rows):
        plan = _plan_from_candidate(row, rng)
        prompt = _true_holdout_prompt_for_plan(plan, index=index)
        item = {
            "id": f"true_holdout_query_compiler_{len(eval_rows) + 1}",
            "category": "fbbp-query-compiler-v10-true-holdout",
            "prompt_mode": "bare_no_hints",
            "prompt": prompt,
            "gold_plan": plan,
            "reference_note": "True holdout prompt family with no injected hints.",
        }
        eval_rows.append(item)
        if len(audit_rows) < 20:
            audit_rows.append(item)
        if len(eval_rows) >= eval_size:
            break

    _write_jsonl(output_path, eval_rows)

    if audit_path is not None:
        lines = [
            "# FBBP Query Compiler V10 True Holdout Audit",
            "",
            f"- seed: `{seed}`",
            f"- eval_size: `{len(eval_rows)}`",
            f"- holdout_offset: `{seen_cutoff}`",
            "",
        ]
        for row in audit_rows:
            lines.append(f"## {row['id']}")
            lines.append("")
            lines.append(f"- PromptMode: `{row['prompt_mode']}`")
            lines.append(f"- Prompt: `{row['prompt']}`")
            lines.append(f"- Gold: `{json.dumps(row['gold_plan'], ensure_ascii=False)}`")
            lines.append("")
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "seed": seed,
        "eval_size": len(eval_rows),
        "holdout_offset": seen_cutoff,
        "audit_examples": len(audit_rows),
        "source_snapshot_rows": len(rows),
        "output_path": str(output_path),
    }


def build_query_compiler_v12_true_holdout_eval(
    output_path: Path,
    audit_path: Path | None = None,
    seed: int = 173,
    eval_size: int = 80,
    train_size: int = 600,
    dev_size: int = 40,
    test_seen_size: int = 40,
    test_hard_size: int = 40,
) -> dict[str, Any]:
    rng = random.Random(seed)
    rows = build_candidate_snapshot()
    rng.shuffle(rows)

    seen_cutoff = min(len(rows), train_size + dev_size + test_seen_size + test_hard_size)
    holdout_rows = rows[seen_cutoff:]
    if not holdout_rows:
        holdout_rows = rows

    eval_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for index, row in enumerate(holdout_rows):
        plan = _plan_from_candidate(row, rng)
        prompt = _v12_holdout_prompt_for_plan(plan, index=index)
        item = {
            "id": f"v12_true_holdout_query_compiler_{len(eval_rows) + 1}",
            "category": "fbbp-query-compiler-v12-true-holdout",
            "prompt_mode": "bare_no_hints",
            "prompt": prompt,
            "gold_plan": plan,
            "reference_note": "Farther no-hints holdout prompt family for v12 robustness validation.",
        }
        eval_rows.append(item)
        if len(audit_rows) < 20:
            audit_rows.append(item)
        if len(eval_rows) >= eval_size:
            break

    _write_jsonl(output_path, eval_rows)

    if audit_path is not None:
        lines = [
            "# FBBP Query Compiler V12 True Holdout Audit",
            "",
            f"- seed: `{seed}`",
            f"- eval_size: `{len(eval_rows)}`",
            f"- holdout_offset: `{seen_cutoff}`",
            "",
        ]
        for row in audit_rows:
            lines.append(f"## {row['id']}")
            lines.append("")
            lines.append(f"- PromptMode: `{row['prompt_mode']}`")
            lines.append(f"- Prompt: `{row['prompt']}`")
            lines.append(f"- Gold: `{json.dumps(row['gold_plan'], ensure_ascii=False)}`")
            lines.append("")
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "seed": seed,
        "eval_size": len(eval_rows),
        "holdout_offset": seen_cutoff,
        "audit_examples": len(audit_rows),
        "source_snapshot_rows": len(rows),
        "output_path": str(output_path),
    }


def build_query_compiler_v13_true_holdout_eval(
    output_path: Path,
    audit_path: Path | None = None,
    seed: int = 211,
    eval_size: int = 80,
    train_size: int = 600,
    dev_size: int = 40,
    test_seen_size: int = 40,
    test_hard_size: int = 40,
) -> dict[str, Any]:
    rng = random.Random(seed)
    rows = build_candidate_snapshot()
    rng.shuffle(rows)

    seen_cutoff = min(len(rows), train_size + dev_size + test_seen_size + test_hard_size)
    holdout_rows = rows[seen_cutoff:]
    if not holdout_rows:
        holdout_rows = rows

    eval_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for index, row in enumerate(holdout_rows):
        plan = _plan_from_candidate(row, rng)
        prompt = _v13_holdout_prompt_for_plan(plan, index=index)
        item = {
            "id": f"v13_true_holdout_query_compiler_{len(eval_rows) + 1}",
            "category": "fbbp-query-compiler-v13-true-holdout",
            "prompt_mode": "bare_no_hints",
            "prompt": prompt,
            "gold_plan": plan,
            "reference_note": "Farther no-hints holdout prompt family for v13 robustness validation.",
        }
        eval_rows.append(item)
        if len(audit_rows) < 20:
            audit_rows.append(item)
        if len(eval_rows) >= eval_size:
            break

    _write_jsonl(output_path, eval_rows)

    if audit_path is not None:
        lines = [
            "# FBBP Query Compiler V13 True Holdout Audit",
            "",
            f"- seed: `{seed}`",
            f"- eval_size: `{len(eval_rows)}`",
            f"- holdout_offset: `{seen_cutoff}`",
            "",
        ]
        for row in audit_rows:
            lines.append(f"## {row['id']}")
            lines.append("")
            lines.append(f"- PromptMode: `{row['prompt_mode']}`")
            lines.append(f"- Prompt: `{row['prompt']}`")
            lines.append(f"- Gold: `{json.dumps(row['gold_plan'], ensure_ascii=False)}`")
            lines.append("")
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "seed": seed,
        "eval_size": len(eval_rows),
        "holdout_offset": seen_cutoff,
        "audit_examples": len(audit_rows),
        "source_snapshot_rows": len(rows),
        "output_path": str(output_path),
    }


def build_query_compiler_v15_true_holdout_eval(
    output_path: Path,
    audit_path: Path | None = None,
    seed: int = 241,
    eval_size: int = 80,
    train_size: int = 600,
    dev_size: int = 40,
    test_seen_size: int = 40,
    test_hard_size: int = 40,
) -> dict[str, Any]:
    rng = random.Random(seed)
    rows = build_candidate_snapshot()
    rng.shuffle(rows)

    seen_cutoff = min(len(rows), train_size + dev_size + test_seen_size + test_hard_size)
    holdout_rows = rows[seen_cutoff:]
    if not holdout_rows:
        holdout_rows = rows

    eval_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for index, row in enumerate(holdout_rows):
        plan = _plan_from_candidate(row, rng)
        prompt = _v15_holdout_prompt_for_plan(plan, index=index)
        item = {
            "id": f"v15_true_holdout_query_compiler_{len(eval_rows) + 1}",
            "category": "fbbp-query-compiler-v15-true-holdout",
            "prompt_mode": "bare_no_hints",
            "prompt": prompt,
            "gold_plan": plan,
            "reference_note": "Farther no-hints holdout prompt family reserved for final robustness validation.",
        }
        eval_rows.append(item)
        if len(audit_rows) < 20:
            audit_rows.append(item)
        if len(eval_rows) >= eval_size:
            break

    _write_jsonl(output_path, eval_rows)

    if audit_path is not None:
        lines = [
            "# FBBP Query Compiler V15 True Holdout Audit",
            "",
            f"- seed: `{seed}`",
            f"- eval_size: `{len(eval_rows)}`",
            f"- holdout_offset: `{seen_cutoff}`",
            "",
        ]
        for row in audit_rows:
            lines.append(f"## {row['id']}")
            lines.append("")
            lines.append(f"- PromptMode: `{row['prompt_mode']}`")
            lines.append(f"- Prompt: `{row['prompt']}`")
            lines.append(f"- Gold: `{json.dumps(row['gold_plan'], ensure_ascii=False)}`")
            lines.append("")
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "seed": seed,
        "eval_size": len(eval_rows),
        "holdout_offset": seen_cutoff,
        "audit_examples": len(audit_rows),
        "source_snapshot_rows": len(rows),
        "output_path": str(output_path),
    }
