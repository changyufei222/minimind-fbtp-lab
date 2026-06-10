from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LAB_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_ROOT = PROJECT_ROOT / "upstream-minimind-full"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check whether query-compiler SFT examples keep assistant supervision after truncation")
    parser.add_argument("--config", required=True, help="Training config JSON path")
    parser.add_argument("--data-path", default=None, help="Optional explicit dataset path")
    parser.add_argument("--sample-size", type=int, default=0, help="Only inspect the first N rows when > 0")
    parser.add_argument("--max-zero-rate", type=float, default=0.02, help="Fail if zero-supervision rate exceeds this threshold")
    return parser.parse_args()


def generate_labels(input_ids: list[int], bos_id: list[int], eos_id: list[int], max_length: int) -> list[int]:
    labels = [-100] * len(input_ids)
    i = 0
    while i < len(input_ids):
        if input_ids[i:i + len(bos_id)] == bos_id:
            start = i + len(bos_id)
            end = start
            while end < len(input_ids):
                if input_ids[end:end + len(eos_id)] == eos_id:
                    break
                end += 1
            for j in range(start, min(end + len(eos_id), max_length)):
                labels[j] = input_ids[j]
            i = end + len(eos_id) if end < len(input_ids) else len(input_ids)
        else:
            i += 1
    return labels


def main() -> None:
    args = parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    max_length = int(config["max_seq_len"])
    data_path = Path(args.data_path) if args.data_path else (LAB_ROOT / config["data_path"])
    tokenizer = AutoTokenizer.from_pretrained(str((UPSTREAM_ROOT / "model").resolve()))
    bos_id = tokenizer(f"{tokenizer.bos_token}assistant\n", add_special_tokens=False).input_ids
    eos_id = tokenizer(f"{tokenizer.eos_token}\n", add_special_tokens=False).input_ids

    lines = [line for line in data_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.sample_size > 0:
        lines = lines[:args.sample_size]

    zero_supervision = 0
    max_prompt_before_assistant = 0
    max_total_tokens = 0
    checked = 0
    for line in lines:
        row = json.loads(line)
        prompt = tokenizer.apply_chat_template(row["conversations"], tokenize=False, add_generation_prompt=False)
        ids = tokenizer(prompt).input_ids[:max_length]
        labels = generate_labels(ids, bos_id=bos_id, eos_id=eos_id, max_length=max_length)
        supervised = sum(1 for item in labels if item != -100)
        before_assistant = tokenizer.apply_chat_template(row["conversations"][:-1], tokenize=False, add_generation_prompt=True)
        before_len = len(tokenizer(before_assistant).input_ids)
        total_len = len(tokenizer(prompt).input_ids)
        max_prompt_before_assistant = max(max_prompt_before_assistant, before_len)
        max_total_tokens = max(max_total_tokens, total_len)
        zero_supervision += int(supervised == 0)
        checked += 1

    zero_rate = zero_supervision / max(checked, 1)
    summary = {
        "checked_rows": checked,
        "max_seq_len": max_length,
        "zero_supervision_rows": zero_supervision,
        "zero_supervision_rate": round(zero_rate, 4),
        "max_prompt_before_assistant_tokens": max_prompt_before_assistant,
        "max_total_tokens_before_truncation": max_total_tokens,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if zero_rate > args.max_zero_rate:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
