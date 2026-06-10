from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any


LAB_ROOT = Path(__file__).resolve().parents[1]
if str(LAB_ROOT) not in sys.path:
    sys.path.insert(0, str(LAB_ROOT))

from query_compiler.candidate_snapshot import build_candidate_snapshot  # type: ignore  # noqa: E402
from query_compiler.rule_baseline import infer_rule_draft  # type: ignore  # noqa: E402
from query_compiler.scoring import score_prediction  # type: ignore  # noqa: E402
from query_compiler.validator import validate_query_draft  # type: ignore  # noqa: E402


EVAL_PROMPTS_PATH = LAB_ROOT / "data" / "eval" / "fbbp_query_compiler_eval_prompts.jsonl"
OUTPUT_PATH = LAB_ROOT / "reports" / "eval" / "query_compiler_rule_eval.jsonl"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    snapshot_rows = build_candidate_snapshot()
    scaffold_values = [str(row["scaffold_type"]) for row in snapshot_rows if row.get("scaffold_type")]
    prompts = _load_jsonl(EVAL_PROMPTS_PATH)
    output_rows: list[dict[str, Any]] = []

    for item in prompts:
        raw_draft = infer_rule_draft(item["prompt"], scaffold_values)
        normalized = validate_query_draft(raw_draft)
        score = score_prediction(normalized["plan"], item["gold_plan"], snapshot_rows)
        output_rows.append(
            {
                "id": item["id"],
                "category": item["category"],
                "prompt": item["prompt"],
                "draft": raw_draft,
                "normalized_plan": normalized["plan"],
                "trace": normalized["trace"],
                "gold_plan": item["gold_plan"],
                **score,
            }
        )

    _write_jsonl(OUTPUT_PATH, output_rows)
    print(json.dumps({"output_path": str(OUTPUT_PATH), "rows": len(output_rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
