from __future__ import annotations

import json

from .executor import execute_query_plan
from .rule_baseline import infer_rule_draft
from .validator import validate_query_draft


def render_demo(query: str, snapshot_rows: list[dict]) -> str:
    scaffold_values = [str(row["scaffold_type"]) for row in snapshot_rows if row.get("scaffold_type")]
    raw_draft = infer_rule_draft(query, scaffold_values)
    normalized = validate_query_draft(raw_draft)
    result = execute_query_plan(normalized["plan"], snapshot_rows)
    preview_rows = result["rows"][:5]

    lines = [
        "=== Query Compiler Demo ===",
        f"Request: {query}",
        "",
        "Raw Draft:",
        json.dumps(raw_draft, ensure_ascii=False, indent=2),
        "",
        "Normalized Plan:",
        json.dumps(normalized["plan"], ensure_ascii=False, indent=2),
        "",
        "Repairs:",
        json.dumps(normalized["trace"]["repairs"], ensure_ascii=False, indent=2),
        "",
        "Top Results:",
        json.dumps(preview_rows, ensure_ascii=False, indent=2),
    ]
    return "\n".join(lines)
