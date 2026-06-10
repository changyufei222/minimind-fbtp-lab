from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


LAB_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB_ROOT))

from query_compiler.prompting import build_compiler_messages, render_plan_json  # noqa: E402
from query_compiler.sketch import plan_to_sketch  # noqa: E402


DEFAULT_SOURCE = LAB_ROOT / "data" / "eval" / "fbbp_query_compiler_v15_true_holdout_prompts.jsonl"
DEFAULT_OUTPUT = LAB_ROOT / "data" / "processed" / "fbbp_query_compiler_dpo_smoke.jsonl"
DEFAULT_HELDOUT_SOURCE = LAB_ROOT / "data" / "eval" / "fbbp_query_compiler_v13_true_holdout_prompts.jsonl"
DEFAULT_HELDOUT_OUTPUT = LAB_ROOT / "data" / "processed" / "fbbp_query_compiler_dpo_heldout_v13.jsonl"
DEFAULT_REPORT = LAB_ROOT / "reports" / "algorithm_resume" / "query_compiler_dpo_dataset_report.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build DPO chosen/rejected pairs for the FBBP query compiler")
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--source-kind", choices=["auto", "eval", "conversation"], default="auto")
    parser.add_argument("--heldout-source", default=str(DEFAULT_HELDOUT_SOURCE))
    parser.add_argument("--heldout-output", default=str(DEFAULT_HELDOUT_OUTPUT))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--limit", type=int, default=80)
    parser.add_argument("--heldout-limit", type=int, default=80)
    parser.add_argument("--variants-per-row", type=int, default=1)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def corrupt_sketch(sketch: dict[str, Any], idx: int) -> dict[str, Any]:
    rejected = dict(sketch)
    mode = idx % 10
    if mode == 0 and isinstance(rejected.get("engineered"), bool):
        rejected["engineered"] = not rejected["engineered"]
    elif mode == 1:
        rejected["oral_class"] = {"High": "Low", "Low": "High", "Medium": "Low"}.get(rejected.get("oral_class"), "Medium")
    elif mode == 2:
        rejected["sort_by"] = None
        rejected["sort_dir"] = None
    elif mode == 3:
        rejected["limit"] = 20 if rejected.get("limit") != 20 else 5
    elif mode == 4:
        rejected["has_experimental_affinity"] = False
    elif mode == 5:
        rejected["scaffold_type"] = "candidate_id"
    elif mode == 6:
        rejected["oral_class"] = "oral_class"
    elif mode == 7:
        rejected["sort_by"] = "candidate_id"
        rejected["sort_dir"] = "desc"
    elif mode == 8:
        rejected["limit"] = "top five"
    else:
        rejected = {
            "candidate_id": None,
            "scaffold_type": None,
            "oral_class": None,
            "engineered": None,
            "has_experimental_affinity": None,
            "organism": None,
            "sort_by": None,
            "sort_dir": None,
            "limit": 20,
        }
    return rejected


def build_eval_pair(row: dict[str, Any], idx: int) -> dict[str, Any]:
    prompt = row["prompt"]
    gold_plan = row["gold_plan"]
    chosen_sketch = plan_to_sketch(gold_plan)
    rejected_sketch = corrupt_sketch(chosen_sketch, idx)
    base_messages = build_compiler_messages(prompt, include_hints=False, wrap_request=False)
    chosen = base_messages + [{"role": "assistant", "content": render_plan_json(chosen_sketch)}]
    rejected = base_messages + [{"role": "assistant", "content": render_plan_json(rejected_sketch)}]
    return {
        "id": row.get("id", f"dpo_pair_{idx + 1}"),
        "chosen": chosen,
        "rejected": rejected,
        "preference_source": "gold_plan_vs_rule_corruption",
        "corruption_family": str(idx % 10),
    }


def build_conversation_pair(row: dict[str, Any], idx: int) -> dict[str, Any]:
    conversations = row["conversations"]
    if not conversations or conversations[-1].get("role") != "assistant":
        raise ValueError("conversation source rows must end with an assistant message")
    chosen_sketch = json.loads(conversations[-1]["content"])
    rejected_sketch = corrupt_sketch(chosen_sketch, idx)
    base_messages = conversations[:-1]
    chosen = base_messages + [{"role": "assistant", "content": render_plan_json(chosen_sketch)}]
    rejected = base_messages + [{"role": "assistant", "content": render_plan_json(rejected_sketch)}]
    return {
        "id": row.get("id", f"dpo_conv_pair_{idx + 1}"),
        "chosen": chosen,
        "rejected": rejected,
        "preference_source": "assistant_plan_vs_rule_corruption",
        "corruption_family": str(idx % 10),
    }


def infer_source_kind(rows: list[dict[str, Any]], requested: str) -> str:
    if requested != "auto":
        return requested
    if rows and "conversations" in rows[0]:
        return "conversation"
    return "eval"


def build_pairs(rows: list[dict[str, Any]], source_kind: str, variants_per_row: int) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    builder = build_conversation_pair if source_kind == "conversation" else build_eval_pair
    for row_index, row in enumerate(rows):
        for variant in range(max(1, variants_per_row)):
            pair_index = row_index * max(1, variants_per_row) + variant
            pair = builder(row, pair_index)
            if variants_per_row > 1:
                pair["id"] = f"{pair['id']}_var{variant + 1}"
            pairs.append(pair)
    return pairs


def main() -> None:
    args = parse_args()
    source = Path(args.source)
    output = Path(args.output)
    report = Path(args.report)
    rows = load_jsonl(source)[: args.limit]
    source_kind = infer_source_kind(rows, args.source_kind)
    heldout_source = Path(args.heldout_source)
    heldout_output = Path(args.heldout_output)
    heldout_rows = load_jsonl(heldout_source)[: args.heldout_limit]
    pairs = build_pairs(rows, source_kind, args.variants_per_row)
    heldout_pairs = build_pairs(heldout_rows, "eval", 1)
    write_jsonl(output, pairs)
    write_jsonl(heldout_output, heldout_pairs)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        "\n".join(
            [
                "# Query Compiler DPO Dataset Report",
                "",
                f"- source: `{source}`",
                f"- source_kind: `{source_kind}`",
                f"- output: `{output}`",
                f"- pairs: `{len(pairs)}`",
                f"- variants_per_row: `{max(1, args.variants_per_row)}`",
                f"- heldout_source: `{heldout_source}`",
                f"- heldout_output: `{heldout_output}`",
                f"- heldout_pairs: `{len(heldout_pairs)}`",
                "- chosen: gold executable query sketch",
                "- rejected: deterministic corruption of semantic slots or schema-validity cues",
                "- purpose: DPO smoke training for preference learning over query-plan correctness",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "source_kind": source_kind,
                "pairs": len(pairs),
                "variants_per_row": max(1, args.variants_per_row),
                "heldout_output": str(heldout_output),
                "heldout_pairs": len(heldout_pairs),
                "report": str(report),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
