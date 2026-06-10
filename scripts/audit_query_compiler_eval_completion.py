from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit query-compiler eval raw.jsonl for completion-critical slices")
    parser.add_argument("--raw", required=True, help="Path to eval raw.jsonl")
    parser.add_argument("--output", default=None, help="Optional output JSON path")
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def attach_return_fields(plan: dict[str, Any] | None, reference: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(plan, dict):
        return None
    fixed = dict(plan)
    if isinstance(reference, dict) and "return_fields" in reference:
        fixed["return_fields"] = reference["return_fields"]
    return fixed


def is_final_perfect(row: dict[str, Any]) -> bool:
    return (
        bool(row.get("draft_valid"))
        and bool(row.get("json_parsed"))
        and bool(row.get("non_empty_filter"))
        and float(row.get("field_value_exact_match", 0.0)) == 1.0
        and float(row.get("slot_accuracy", 0.0)) == 1.0
        and bool(row.get("execution_success"))
        and float(row.get("result_overlap_at_k", 0.0)) == 1.0
    )


def is_first_pass_perfect(row: dict[str, Any]) -> bool:
    first_trace = row.get("first_trace") or {}
    if not isinstance(first_trace, dict) or not first_trace.get("schema_ok"):
        return False
    first_plan = attach_return_fields(row.get("first_normalized_plan"), row.get("normalized_plan"))
    gold_plan = attach_return_fields(row.get("gold_plan"), row.get("normalized_plan"))
    return first_plan == gold_plan


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    repair_attempted = sum(1 for row in rows if row.get("repair_attempted"))
    used_repair = sum(1 for row in rows if row.get("used_repair"))
    projection_attempted = sum(1 for row in rows if row.get("projection_attempted"))
    used_projection = sum(1 for row in rows if row.get("used_projection"))
    final_perfect = sum(1 for row in rows if is_final_perfect(row))
    first_perfect = sum(1 for row in rows if is_first_pass_perfect(row))
    first_schema_ok = sum(1 for row in rows if (row.get("first_trace") or {}).get("schema_ok") is True)

    projection_examples = [
        {
            "id": row.get("id"),
            "prompt": row.get("prompt"),
            "projection_reasons": row.get("projection_reasons", []),
            "repair_reasons": row.get("repair_reasons", []),
            "first_answer": row.get("first_answer"),
            "answer": row.get("answer"),
        }
        for row in rows
        if row.get("used_projection")
    ][:10]

    repair_examples = [
        {
            "id": row.get("id"),
            "prompt": row.get("prompt"),
            "repair_reasons": row.get("repair_reasons", []),
            "first_answer": row.get("first_answer"),
            "repaired_answer": row.get("repaired_answer"),
            "answer": row.get("answer"),
        }
        for row in rows
        if row.get("used_repair")
    ][:10]

    return {
        "rows": total,
        "final_perfect_rows": final_perfect,
        "final_perfect_rate": round(final_perfect / total, 4) if total else 0.0,
        "first_schema_ok_rows": first_schema_ok,
        "first_schema_ok_rate": round(first_schema_ok / total, 4) if total else 0.0,
        "first_pass_perfect_rows": first_perfect,
        "first_pass_perfect_rate": round(first_perfect / total, 4) if total else 0.0,
        "repair_attempted_rows": repair_attempted,
        "repair_attempted_rate": round(repair_attempted / total, 4) if total else 0.0,
        "used_repair_rows": used_repair,
        "used_repair_rate": round(used_repair / total, 4) if total else 0.0,
        "projection_attempted_rows": projection_attempted,
        "projection_attempted_rate": round(projection_attempted / total, 4) if total else 0.0,
        "used_projection_rows": used_projection,
        "used_projection_rate": round(used_projection / total, 4) if total else 0.0,
        "projection_examples": projection_examples,
        "repair_examples": repair_examples,
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
