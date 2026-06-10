from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize query-compiler eval raw.jsonl by family, repair usage, and metric slices")
    parser.add_argument("--raw", required=True, help="Path to eval raw.jsonl")
    parser.add_argument("--output", default=None, help="Optional output JSON path")
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def family_label(prompt: str) -> str:
    prefixes = {
        "From the FBBP registry": "from_registry_en",
        "I am shortlisting entries from the candidate table": "shortlisting_en",
        "For this screen, I only want molecules with": "screen_en",
        "请按数据库语义理解这个检索": "zh_semantic",
        "我在库里做初筛": "zh_prescreen",
        "研究问题是这样的": "zh_research",
        "Keep a database shortlist for candidates showing": "keep_shortlist_en",
        "做一个候选清单给我": "zh_shortlist",
        "Search the candidate registry for entries satisfying": "search_registry_en",
        "I need a candidate shortlist restricted to": "restricted_shortlist_en",
        "Within the protein-candidate table, retain only rows with": "retain_rows_en",
        "请直接按数据库语义筛选": "zh_direct_filter",
        "做一个不带解释的候选筛选": "zh_no_explain",
        "我想从候选库里锁定": "zh_lock_candidates",
        "Build a database shortlist for candidates meeting": "build_shortlist_en",
        "候选初筛条件如下": "zh_prescreen_brief",
        "Filter the candidate inventory down to rows annotated with": "inventory_filter_en",
        "For downstream review, retain only protein candidates tagged with": "downstream_review_en",
        "Database query request: among candidate entries": "db_request_en",
        "请从候选表里筛出满足": "zh_table_filter",
        "只保留": "zh_keep_only",
        "我要一个数据库候选列表": "zh_db_candidate_list",
        "Keep candidate rows carrying": "keep_rows_en",
        "候选清单需求": "zh_candidate_list_need",
    }
    for prefix, label in prefixes.items():
        if prompt.startswith(prefix):
            return label
    return "other"


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    repair_reason_counts = Counter()
    family_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    failed_rows: list[dict[str, Any]] = []

    for row in rows:
        family = family_label(row.get("prompt", ""))
        family_rows[family].append(row)
        for reason in row.get("repair_reasons", []):
            repair_reason_counts[reason] += 1
        if not row.get("execution_success") or float(row.get("result_overlap_at_k", 0.0)) == 0.0:
            failed_rows.append(
                {
                    "id": row.get("id"),
                    "family": family,
                    "field_value_exact_match": row.get("field_value_exact_match"),
                    "slot_accuracy": row.get("slot_accuracy"),
                    "used_repair": row.get("used_repair"),
                    "repair_reasons": row.get("repair_reasons", []),
                    "prompt": row.get("prompt"),
                    "answer": row.get("answer"),
                }
            )

    family_summary = {}
    for family, items in sorted(family_rows.items()):
        total = len(items)
        family_summary[family] = {
            "rows": total,
            "mean_overlap": round(sum(float(item.get("result_overlap_at_k", 0.0)) for item in items) / total, 4),
            "mean_field": round(sum(float(item.get("field_value_exact_match", 0.0)) for item in items) / total, 4),
            "mean_slot": round(sum(float(item.get("slot_accuracy", 0.0)) for item in items) / total, 4),
            "repair_attempted": sum(1 for item in items if item.get("repair_attempted")),
            "used_repair": sum(1 for item in items if item.get("used_repair")),
            "non_empty_filter": sum(1 for item in items if item.get("non_empty_filter")),
        }

    return {
        "rows": len(rows),
        "repair_reason_counts": dict(repair_reason_counts.most_common()),
        "family_summary": family_summary,
        "failed_rows_sample": failed_rows[:20],
    }


def main() -> None:
    args = parse_args()
    rows = load_rows(Path(args.raw))
    summary = summarize(rows)
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
