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
UPSTREAM_ROOT = PROJECT_ROOT / "upstream-minimind-full"
sys.path.insert(0, str(UPSTREAM_ROOT))

from model.model_lora import apply_lora, load_lora  # type: ignore  # noqa: E402
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM  # type: ignore  # noqa: E402
from trainer.trainer_utils import get_model_params  # type: ignore  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline/LoRA comparison prompts for MiniMind-FBTP")
    parser.add_argument("--save-dir", default=str((LAB_ROOT / "reports" / "checkpoints").resolve()))
    parser.add_argument("--weight", default="full_sft")
    parser.add_argument("--lora-weight", default="None")
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--num-hidden-layers", type=int, default=8)
    parser.add_argument("--use-moe", type=int, default=0, choices=[0, 1])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--prompts", default=str((LAB_ROOT / "data" / "eval" / "fbtp_eval_prompts.jsonl").resolve()))
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--label", default="baseline")
    parser.add_argument("--lora-subdir", default="lora")
    return parser.parse_args()


def load_prompts(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
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

    weight_path = Path(args.save_dir) / f"{args.weight}_{args.hidden_size}.pth"
    model.load_state_dict(torch.load(weight_path, map_location=args.device), strict=True)

    if args.lora_weight != "None":
        apply_lora(model)
        load_lora(model, str(Path(args.save_dir) / args.lora_subdir / f"{args.lora_weight}_{args.hidden_size}.pth"))

    get_model_params(model, model.config)
    return model.eval().to(args.device), tokenizer


def build_prompt(tokenizer, prompt: str) -> dict[str, torch.Tensor]:
    conversation = [{"role": "user", "content": prompt}]
    rendered = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    return tokenizer(rendered, return_tensors="pt", truncation=True)


def generate_answer(model, tokenizer, prompt: str, device: str, max_new_tokens: int) -> str:
    inputs = build_prompt(tokenizer, prompt)
    inputs = {k: v.to(device) for k, v in inputs.items()}
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
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_markdown(path: Path, rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    lines = [
        f"# MiniMind-FBTP Eval ({args.label})",
        "",
        f"- Base weight: `{args.weight}_{args.hidden_size}.pth`",
        f"- LoRA weight: `{args.lora_weight}`",
        f"- Device: `{args.device}`",
        "",
        "| id | category | prompt | answer | reference_note | latency_ms |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["id"]).replace("|", "/"),
                    str(row["category"]).replace("|", "/"),
                    str(row["prompt"]).replace("\n", " ").replace("|", "/")[:120],
                    str(row["answer"]).replace("\n", " ").replace("|", "/")[:240],
                    str(row["reference_note"]).replace("\n", " ").replace("|", "/")[:120],
                    str(row["latency_ms"]),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    prompts = load_prompts(Path(args.prompts))
    model, tokenizer = init_model(args)

    rows: list[dict[str, Any]] = []
    for item in prompts:
        started = time.time()
        answer = generate_answer(model, tokenizer, item["prompt"], args.device, args.max_new_tokens)
        latency_ms = round((time.time() - started) * 1000, 2)
        rows.append(
            {
                "id": item["id"],
                "category": item["category"],
                "prompt": item["prompt"],
                "reference_note": item.get("reference_note", ""),
                "answer": answer,
                "latency_ms": latency_ms,
                "label": args.label,
            }
        )

    output_jsonl = Path(args.output_jsonl)
    output_md = Path(args.output_md)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_jsonl, rows)
    write_markdown(output_md, rows, args)
    print(json.dumps({"output_jsonl": str(output_jsonl), "output_md": str(output_md)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
