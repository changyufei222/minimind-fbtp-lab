from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


LAB_ROOT = Path(__file__).resolve().parents[1]
if str(LAB_ROOT) not in sys.path:
    sys.path.insert(0, str(LAB_ROOT))

from query_compiler.request_hints import infer_request_hints  # type: ignore  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit deterministic hint coverage on query-compiler holdout prompts")
    parser.add_argument("--prompts", required=True, help="Path to holdout prompt JSONL")
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    parser.add_argument("--show-misses", type=int, default=12, help="Number of miss examples to keep")
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def family_label(prompt: str) -> str:
    if prompt.startswith("From the FBBP registry"):
        return "from_registry_en"
    if prompt.startswith("I am shortlisting entries from the candidate table"):
        return "shortlisting_en"
    if prompt.startswith("For this screen, I only want molecules with"):
        return "screen_en"
    if prompt.startswith("请按数据库语义理解这个检索"):
        return "zh_semantic"
    if prompt.startswith("我在库里做初筛"):
        return "zh_prescreen"
    if prompt.startswith("研究问题是这样的"):
        return "zh_research"
    if prompt.startswith("Keep a database shortlist for candidates showing"):
        return "keep_shortlist_en"
    if prompt.startswith("做一个候选清单给我"):
        return "zh_shortlist"
    if prompt.startswith("Search the candidate registry for entries satisfying"):
        return "search_registry_en"
    if prompt.startswith("I need a candidate shortlist restricted to"):
        return "restricted_shortlist_en"
    if prompt.startswith("Within the protein-candidate table, retain only rows with"):
        return "retain_rows_en"
    if prompt.startswith("请直接按数据库语义筛选"):
        return "zh_direct_filter"
    if prompt.startswith("做一个不带解释的候选筛选"):
        return "zh_no_explain"
    if prompt.startswith("我想从候选库里锁定"):
        return "zh_lock_candidates"
    if prompt.startswith("Build a database shortlist for candidates meeting"):
        return "build_shortlist_en"
    if prompt.startswith("候选初筛条件如下"):
        return "zh_prescreen_brief"
    if prompt.startswith("Filter the candidate inventory down to rows annotated with"):
        return "inventory_filter_en"
    if prompt.startswith("For downstream review, retain only protein candidates tagged with"):
        return "downstream_review_en"
    if prompt.startswith("Database query request: among candidate entries"):
        return "db_request_en"
    if prompt.startswith("请从候选表里筛出满足"):
        return "zh_table_filter"
    if prompt.startswith("只保留"):
        return "zh_keep_only"
    if prompt.startswith("我要一个数据库候选列表"):
        return "zh_db_candidate_list"
    if prompt.startswith("Keep candidate rows carrying"):
        return "keep_rows_en"
    if prompt.startswith("候选清单需求"):
        return "zh_candidate_list_need"
    return "other"


def main() -> None:
    args = parse_args()
    rows = load_rows(Path(args.prompts))

    field_totals = Counter()
    field_matches = Counter()
    family_totals = Counter()
    misses: list[dict[str, Any]] = []

    for row in rows:
        prompt = row["prompt"]
        gold = row["gold_plan"]
        hints = infer_request_hints(prompt)
        family = family_label(prompt)
        family_totals[family] += 1
        filter_map = {flt["field"]: flt["value"] for flt in gold.get("filters", [])}
        expected = {
            "scaffold_type": filter_map.get("scaffold_type"),
            "oral_class": filter_map.get("oral_class"),
            "engineered": filter_map.get("engineered"),
            "has_experimental_affinity": filter_map.get("has_experimental_affinity"),
            "limit": gold.get("limit"),
            "sort_by": "best_affinity_value" if gold.get("sort") else None,
        }

        for field, exp in expected.items():
            if exp is None:
                continue
            field_totals[field] += 1
            got = hints.get(field)
            if got == exp:
                field_matches[field] += 1
            elif len(misses) < args.show_misses:
                misses.append(
                    {
                        "id": row["id"],
                        "family": family,
                        "field": field,
                        "expected": exp,
                        "got": got,
                        "prompt": prompt,
                    }
                )

    summary = {
        "rows": len(rows),
        "field_coverage": {
            field: {
                "matched": field_matches[field],
                "present": field_totals[field],
                "rate": round(field_matches[field] / field_totals[field], 4) if field_totals[field] else None,
            }
            for field in ("scaffold_type", "oral_class", "engineered", "has_experimental_affinity", "limit", "sort_by")
        },
        "families": dict(sorted(family_totals.items())),
        "sample_misses": misses,
    }

    text = json.dumps(summary, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
