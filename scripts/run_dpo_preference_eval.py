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

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM  # type: ignore  # noqa: E402
from trainer.trainer_utils import get_model_params  # type: ignore  # noqa: E402

from query_compiler.eval_paths import resolve_base_weight_path  # type: ignore  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DPO chosen/rejected preference margins")
    parser.add_argument("--pairs", default=str((LAB_ROOT / "data" / "processed" / "fbbp_query_compiler_dpo_heldout_v13.jsonl").resolve()))
    parser.add_argument("--save-dir", default=str((LAB_ROOT / "reports" / "checkpoints").resolve()))
    parser.add_argument("--weight", default="full_sft")
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--num-hidden-layers", type=int, default=16)
    parser.add_argument("--use-moe", type=int, default=0, choices=[0, 1])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--label", default="model")
    parser.add_argument("--output-dir", default=str((LAB_ROOT / "reports" / "eval" / "dpo_preference").resolve()))
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
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
    get_model_params(model, model.config)
    return model.eval().to(args.device), tokenizer


def assistant_loss_mask(tokenizer, input_ids: list[int], max_len: int) -> list[int]:
    bos_id = tokenizer(f"{tokenizer.bos_token}assistant\n", add_special_tokens=False).input_ids
    eos_id = tokenizer(f"{tokenizer.eos_token}\n", add_special_tokens=False).input_ids
    mask = [0] * len(input_ids)
    i = 0
    while i < len(input_ids):
        if input_ids[i : i + len(bos_id)] == bos_id:
            start = i + len(bos_id)
            end = start
            while end < len(input_ids):
                if input_ids[end : end + len(eos_id)] == eos_id:
                    break
                end += 1
            for j in range(start, min(end + len(eos_id), max_len)):
                mask[j] = 1
            i = end + len(eos_id) if end < len(input_ids) else len(input_ids)
        else:
            i += 1
    return mask


def sequence_logprob(model, tokenizer, messages: list[dict[str, str]], device: str, max_seq_len: int) -> tuple[float, int]:
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    encoded = tokenizer(rendered, truncation=True, max_length=max_seq_len, padding="max_length")
    input_ids = encoded["input_ids"]
    mask = assistant_loss_mask(tokenizer, input_ids, max_seq_len)
    x = torch.tensor(input_ids[:-1], dtype=torch.long, device=device).unsqueeze(0)
    y = torch.tensor(input_ids[1:], dtype=torch.long, device=device)
    y_mask = torch.tensor(mask[1:], dtype=torch.bool, device=device)
    with torch.no_grad():
        logits = model(x).logits[0]
        log_probs = torch.log_softmax(logits, dim=-1)
        token_log_probs = log_probs[torch.arange(y.shape[0], device=device), y]
        selected = token_log_probs[y_mask]
    if selected.numel() == 0:
        return 0.0, 0
    return float(selected.mean().item()), int(selected.numel())


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    pairs = load_jsonl(Path(args.pairs))
    out_dir = Path(args.output_dir) / args.label
    out_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer = init_model(args)

    rows: list[dict[str, Any]] = []
    start = time.time()
    for pair in pairs:
        chosen_logp, chosen_tokens = sequence_logprob(model, tokenizer, pair["chosen"], args.device, args.max_seq_len)
        rejected_logp, rejected_tokens = sequence_logprob(model, tokenizer, pair["rejected"], args.device, args.max_seq_len)
        margin = chosen_logp - rejected_logp
        dpo_loss = -torch.nn.functional.logsigmoid(torch.tensor(args.beta * margin)).item()
        rows.append(
            {
                "id": pair.get("id"),
                "chosen_logp": chosen_logp,
                "rejected_logp": rejected_logp,
                "preference_margin": margin,
                "preferred_chosen": margin > 0,
                "dpo_loss": dpo_loss,
                "chosen_tokens": chosen_tokens,
                "rejected_tokens": rejected_tokens,
                "corruption_family": pair.get("corruption_family"),
            }
        )

    total = len(rows)
    wins = sum(1 for row in rows if row["preferred_chosen"])
    avg_margin = sum(float(row["preference_margin"]) for row in rows) / total if total else 0.0
    avg_loss = sum(float(row["dpo_loss"]) for row in rows) / total if total else 0.0
    summary = {
        "label": args.label,
        "pairs": total,
        "preference_accuracy": round(wins / total, 4) if total else 0.0,
        "chosen_wins": wins,
        "avg_preference_margin": round(avg_margin, 6),
        "avg_heldout_dpo_loss": round(avg_loss, 6),
        "elapsed_sec": round(time.time() - start, 2),
        "pairs_path": str(Path(args.pairs)),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_jsonl(out_dir / "rows.jsonl", rows)

    lines = [
        f"# DPO Preference Eval ({args.label})",
        "",
        f"- pairs: `{summary['pairs']}`",
        f"- preference_accuracy: `{summary['preference_accuracy']}`",
        f"- chosen_wins: `{summary['chosen_wins']}`",
        f"- avg_preference_margin: `{summary['avg_preference_margin']}`",
        f"- avg_heldout_dpo_loss: `{summary['avg_heldout_dpo_loss']}`",
        f"- pairs_path: `{summary['pairs_path']}`",
        "",
        "| id | chosen_logp | rejected_logp | margin | chosen_win | dpo_loss |",
        "|---|---:|---:|---:|---|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['id']} | {row['chosen_logp']:.6f} | {row['rejected_logp']:.6f} | {row['preference_margin']:.6f} | {row['preferred_chosen']} | {row['dpo_loss']:.6f} |"
        )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
