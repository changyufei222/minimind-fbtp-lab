from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


LAB_ROOT = Path(__file__).resolve().parents[1]
if str(LAB_ROOT) not in sys.path:
    sys.path.insert(0, str(LAB_ROOT))

from query_compiler.scoring import summarize_scores  # type: ignore  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score query-compiler evaluation outputs")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_markdown(path: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    lines = [
        "# Query Compiler Eval Score",
        "",
        f"- plan_valid_rate: `{summary['plan_valid_rate']}`",
        f"- json_parse_rate: `{summary['json_parse_rate']}`",
        f"- non_empty_filter_rate: `{summary['non_empty_filter_rate']}`",
        f"- field_value_exact_match: `{summary['field_value_exact_match']}`",
        f"- slot_accuracy: `{summary['slot_accuracy']}`",
        f"- execution_success_rate: `{summary['execution_success_rate']}`",
        f"- result_overlap_at_k: `{summary['result_overlap_at_k']}`",
        "",
        "| id | draft_valid | json_parsed | non_empty_filter | field_value_exact_match | slot_accuracy | execution_success | result_overlap_at_k |",
        "|---|---|---|---|---:|---:|---|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('id', 'n/a')} | {row['draft_valid']} | {row.get('json_parsed', False)} | {row.get('non_empty_filter', False)} | {row.get('field_value_exact_match', 0.0)} | {row['slot_accuracy']} | {row['execution_success']} | {row['result_overlap_at_k']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_jsonl)
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)

    rows = _load_jsonl(input_path)
    summary = summarize_scores(rows)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(output_md, rows, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
