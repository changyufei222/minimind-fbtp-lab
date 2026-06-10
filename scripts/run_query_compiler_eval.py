from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LAB_ROOT = Path(__file__).resolve().parents[1]
if str(LAB_ROOT) not in sys.path:
    sys.path.insert(0, str(LAB_ROOT))
UPSTREAM_ROOT = PROJECT_ROOT / "upstream-minimind-full"
sys.path.insert(0, str(UPSTREAM_ROOT))

from model.model_lora import apply_lora, load_lora  # type: ignore  # noqa: E402
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM  # type: ignore  # noqa: E402
from trainer.trainer_utils import get_model_params  # type: ignore  # noqa: E402

from query_compiler.candidate_snapshot import build_candidate_snapshot  # type: ignore  # noqa: E402
from query_compiler.eval_paths import resolve_base_weight_path, resolve_lora_weight_path  # type: ignore  # noqa: E402
from query_compiler.prompting import build_compiler_messages, build_repair_messages  # type: ignore  # noqa: E402
from query_compiler.repair import finalize_prediction  # type: ignore  # noqa: E402
from query_compiler.scoring import score_prediction, summarize_scores  # type: ignore  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run query-compiler evaluation for MiniMind-FBBP")
    parser.add_argument("--save-dir", default=str((LAB_ROOT / "reports" / "checkpoints").resolve()))
    parser.add_argument("--weight", default="full_sft")
    parser.add_argument("--lora-weight", default="None")
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--num-hidden-layers", type=int, default=16)
    parser.add_argument("--use-moe", type=int, default=0, choices=[0, 1])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--prompts", default=str((LAB_ROOT / "data" / "eval" / "fbbp_query_compiler_eval_prompts.jsonl").resolve()))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--label", default="baseline")
    parser.add_argument("--lora-subdir", default="lora")
    return parser.parse_args()


def load_prompts(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def init_model(args: argparse.Namespace):
    tokenizer = AutoTokenizer.from_pretrained(str((UPSTREAM_ROOT / "model").resolve()))
    model = MiniMindForCausalLM(
        MiniMindConfig(
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            use_moe=bool(args.use_moe),
        )
    )

    weight_path = resolve_base_weight_path(
        weight=args.weight,
        hidden_size=args.hidden_size,
        save_dir=Path(args.save_dir),
        upstream_root=UPSTREAM_ROOT,
        lab_root=LAB_ROOT,
    )
    model.load_state_dict(torch.load(weight_path, map_location=args.device), strict=True)

    if args.lora_weight != "None":
        apply_lora(model)
        lora_path = resolve_lora_weight_path(
            lora_weight=args.lora_weight,
            hidden_size=args.hidden_size,
            save_dir=Path(args.save_dir),
            lora_subdir=args.lora_subdir,
            lab_root=LAB_ROOT,
        )
        load_lora(model, str(lora_path))

    get_model_params(model, model.config)
    return model.eval().to(args.device), tokenizer


def build_prompt_inputs(tokenizer, messages: list[dict[str, str]]) -> dict[str, torch.Tensor]:
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tokenizer(rendered, return_tensors="pt", truncation=True)


def generate_answer(model, tokenizer, messages: list[dict[str, str]], device: str, max_new_tokens: int) -> str:
    inputs = build_prompt_inputs(tokenizer, messages)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        generated = model.generate(
            inputs=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(generated[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
    return response.strip()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_markdown(path: Path, rows: list[dict[str, Any]], summary: dict[str, Any], label: str) -> None:
    lines = [
        f"# Query Compiler Eval ({label})",
        "",
        f"- plan_valid_rate: `{summary['plan_valid_rate']}`",
        f"- json_parse_rate: `{summary['json_parse_rate']}`",
        f"- non_empty_filter_rate: `{summary['non_empty_filter_rate']}`",
        f"- field_value_exact_match: `{summary['field_value_exact_match']}`",
        f"- slot_accuracy: `{summary['slot_accuracy']}`",
        f"- execution_success_rate: `{summary['execution_success_rate']}`",
        f"- result_overlap_at_k: `{summary['result_overlap_at_k']}`",
        "",
        "| id | draft_valid | json_parsed | non_empty_filter | field_value_exact_match | used_repair | execution_success | result_overlap_at_k | latency_ms |",
        "|---|---|---|---|---:|---|---|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['id']} | {row['draft_valid']} | {row['json_parsed']} | {row['non_empty_filter']} | {row['field_value_exact_match']} | {row['used_repair']} | {row['execution_success']} | {row['result_overlap_at_k']} | {row['latency_ms']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def prompt_mode_options(item: dict[str, Any]) -> dict[str, bool]:
    mode = str(item.get("prompt_mode", "default"))
    if mode == "bare_no_hints":
        return {"include_hints": False, "wrap_request": False}
    return {"include_hints": True, "wrap_request": True}


def main() -> None:
    args = parse_args()
    prompts = load_prompts(Path(args.prompts))
    snapshot_rows = build_candidate_snapshot()
    model, tokenizer = init_model(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / f"{args.label}_raw.jsonl"
    score_path = output_dir / f"{args.label}_score.json"
    md_path = output_dir / f"{args.label}_score.md"

    rows: list[dict[str, Any]] = []
    for item in prompts:
        started = time.time()
        message_options = prompt_mode_options(item)
        first_answer = generate_answer(
            model,
            tokenizer,
            build_compiler_messages(item["prompt"], **message_options),
            args.device,
            args.max_new_tokens,
        )
        finalized = finalize_prediction(first_answer=first_answer, original_request=item["prompt"])
        repair_answer: str | None = None
        if finalized["repair_reasons"]:
            repair_answer = generate_answer(
                model,
                tokenizer,
                build_repair_messages(
                    original_request=item["prompt"],
                    invalid_output=first_answer,
                    errors=finalized["repair_reasons"],
                    **message_options,
                ),
                args.device,
                args.max_new_tokens,
            )
            finalized = finalize_prediction(first_answer=first_answer, original_request=item["prompt"], repaired_answer=repair_answer)
        latency_ms = round((time.time() - started) * 1000, 2)
        final_parsed = finalized["final_parsed_draft"]
        score = score_prediction(
            finalized["final_plan"] if final_parsed is not None else None,
            item["gold_plan"],
            snapshot_rows,
            json_parsed=final_parsed is not None,
        )
        rows.append(
            {
                "id": item["id"],
                "category": item["category"],
                "prompt": item["prompt"],
                "answer": finalized["final_answer"],
                "first_answer": first_answer,
                "first_parsed_draft": finalized["first_parsed_draft"],
                "first_normalized_plan": finalized["first_normalized"]["plan"],
                "first_trace": finalized["first_normalized"]["trace"],
                "repaired_answer": repair_answer,
                "repaired_parsed_draft": finalized["repaired_parsed_draft"],
                "repaired_normalized_plan": finalized["repaired_normalized"]["plan"] if finalized["repaired_normalized"] else None,
                "repaired_trace": finalized["repaired_normalized"]["trace"] if finalized["repaired_normalized"] else None,
                "parsed_draft": final_parsed,
                "normalized_plan": finalized["final_plan"],
                "trace": finalized["final_trace"],
                "repair_attempted": finalized["repair_attempted"],
                "repair_reasons": finalized["repair_reasons"],
                "used_repair": finalized["used_repair"],
                "projection_attempted": finalized["projection_attempted"],
                "projection_reasons": finalized["projection_reasons"],
                "used_projection": finalized["used_projection"],
                "gold_plan": item["gold_plan"],
                "latency_ms": latency_ms,
                **score,
            }
        )

    summary = summarize_scores(rows)
    write_jsonl(raw_path, rows)
    score_path.write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(md_path, rows, summary, args.label)
    print(json.dumps({"output_dir": str(output_dir), "summary": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
