from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import urllib.request
import zipfile
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


CEVAL_URL = "https://huggingface.co/datasets/ceval/ceval-exam/resolve/main/ceval-exam.zip"
SUBJECT_NAMES = {
    "basic_medicine": "基础医学",
    "clinical_medicine": "临床医学",
    "physician": "医师资格",
    "veterinary_medicine": "兽医学",
}
DEFAULT_SUBJECTS = ["basic_medicine", "clinical_medicine", "physician"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MiniMind on official C-Eval medical validation subjects")
    parser.add_argument("--ceval-root", default=str((LAB_ROOT / "data" / "external" / "ceval-exam").resolve()))
    parser.add_argument("--download-if-missing", action="store_true")
    parser.add_argument("--subjects", nargs="+", default=DEFAULT_SUBJECTS)
    parser.add_argument("--split", default="val", choices=["dev", "val"])
    parser.add_argument("--save-dir", default=str((LAB_ROOT / "reports" / "checkpoints").resolve()))
    parser.add_argument("--weight", default="full_sft")
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--num-hidden-layers", type=int, default=16)
    parser.add_argument("--use-moe", type=int, default=0, choices=[0, 1])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--n-shot", type=int, default=0)
    parser.add_argument("--scoring", choices=["generate", "choice_logprob"], default="generate")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--label", default="model")
    parser.add_argument("--output-dir", default=str((LAB_ROOT / "reports" / "eval" / "official_ceval_medical").resolve()))
    return parser.parse_args()


def ensure_ceval(root: Path, download_if_missing: bool) -> None:
    marker = root / "val" / "basic_medicine_val.csv"
    if marker.exists():
        return
    if not download_if_missing:
        raise FileNotFoundError(f"C-Eval data not found at {root}; rerun with --download-if-missing")

    root.parent.mkdir(parents=True, exist_ok=True)
    zip_path = root.parent / "ceval-exam.zip"
    if not zip_path.exists():
        print(f"Downloading official C-Eval data: {CEVAL_URL}")
        urllib.request.urlretrieve(CEVAL_URL, zip_path)
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(root.parent)
    if not marker.exists():
        raise FileNotFoundError(f"Downloaded C-Eval zip, but expected file is still missing: {marker}")


def read_subject(root: Path, split: str, subject: str, limit: int) -> list[dict[str, str]]:
    path = root / split / f"{subject}_{split}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(row)
            if limit and len(rows) >= limit:
                break
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


def format_question(row: dict[str, str], include_answer: bool) -> str:
    text = (
        f"{row['question']}\n"
        f"A. {row['A']}\n"
        f"B. {row['B']}\n"
        f"C. {row['C']}\n"
        f"D. {row['D']}\n"
    )
    if include_answer:
        return text + f"答案：{row.get('answer', '').strip().upper()}\n"
    return text + "答案："


def build_prompt(row: dict[str, str], subject: str, shots: list[dict[str, str]] | None = None) -> str:
    subject_name = SUBJECT_NAMES.get(subject, subject)
    parts = [f"以下是中国关于{subject_name}考试的单项选择题，请选出其中的正确答案。"]
    for shot in shots or []:
        parts.append(format_question(shot, include_answer=True))
    parts.append(format_question(row, include_answer=False))
    return "\n\n".join(parts)


def extract_answer(text: str) -> str:
    cleaned = text.strip().upper()
    match = re.search(r"(?:答案|选项|ANSWER)?\s*[:：]?\s*([ABCD])", cleaned)
    if match:
        return match.group(1)
    match = re.search(r"\b([ABCD])\b", cleaned)
    return match.group(1) if match else ""


def render_prompt(tokenizer, prompt: str) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt


def generate_answer(model, tokenizer, prompt: str, device: str, max_new_tokens: int) -> tuple[str, str, int]:
    rendered = render_prompt(tokenizer, prompt)
    inputs = tokenizer(rendered, return_tensors="pt", truncation=True).to(device)
    start = time.time()
    with torch.no_grad():
        generated = model.generate(
            inputs=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    latency_ms = int((time.time() - start) * 1000)
    response = tokenizer.decode(generated[0][len(inputs["input_ids"][0]) :], skip_special_tokens=True)
    return response.strip(), extract_answer(response), latency_ms


def score_choice_logprobs(model, tokenizer, prompt: str, device: str) -> tuple[dict[str, float], str, int]:
    rendered = render_prompt(tokenizer, prompt)
    prompt_ids = tokenizer(rendered, return_tensors="pt", truncation=True)["input_ids"][0].to(device)
    scores: dict[str, float] = {}
    start = time.time()
    with torch.no_grad():
        for choice in ["A", "B", "C", "D"]:
            choice_ids = tokenizer(choice, add_special_tokens=False, return_tensors="pt")["input_ids"][0].to(device)
            input_ids = torch.cat([prompt_ids, choice_ids], dim=0).unsqueeze(0)
            outputs = model(input_ids)
            logits = outputs.logits[0]
            log_probs = torch.log_softmax(logits, dim=-1)
            score = 0.0
            start_pos = len(prompt_ids) - 1
            for offset, token_id in enumerate(choice_ids):
                score += float(log_probs[start_pos + offset, int(token_id)].item())
            scores[choice] = score
    latency_ms = int((time.time() - start) * 1000)
    prediction = max(scores, key=scores.get)
    return scores, prediction, latency_ms


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    ceval_root = Path(args.ceval_root)
    output_dir = Path(args.output_dir) / args.label
    output_dir.mkdir(parents=True, exist_ok=True)
    ensure_ceval(ceval_root, args.download_if_missing)
    model, tokenizer = init_model(args)

    all_rows: list[dict[str, Any]] = []
    by_subject: dict[str, dict[str, Any]] = {}
    for subject in args.subjects:
        rows = read_subject(ceval_root, args.split, subject, args.limit)
        shots = read_subject(ceval_root, "dev", subject, args.n_shot) if args.n_shot > 0 else []
        correct = 0
        for row in rows:
            prompt = build_prompt(row, subject, shots)
            if args.scoring == "choice_logprob":
                choice_scores, prediction, latency_ms = score_choice_logprobs(model, tokenizer, prompt, args.device)
                response = json.dumps(choice_scores, ensure_ascii=False)
            else:
                response, prediction, latency_ms = generate_answer(
                    model, tokenizer, prompt, args.device, args.max_new_tokens
                )
            gold = row.get("answer", "").strip().upper()
            is_correct = prediction == gold
            correct += int(is_correct)
            all_rows.append(
                {
                    "subject": subject,
                    "split": args.split,
                    "id": row.get("id", ""),
                    "gold": gold,
                    "prediction": prediction,
                    "correct": is_correct,
                    "response": response,
                    "scoring": args.scoring,
                    "n_shot": args.n_shot,
                    "latency_ms": latency_ms,
                }
            )
        by_subject[subject] = {
            "n": len(rows),
            "correct": correct,
            "accuracy": correct / len(rows) if rows else 0.0,
        }

    total = sum(item["n"] for item in by_subject.values())
    total_correct = sum(item["correct"] for item in by_subject.values())
    summary = {
        "label": args.label,
        "official_ceval": True,
        "official_ceval_split": args.split,
        "official_ceval_test_leaderboard": False,
        "scoring": args.scoring,
        "n_shot": args.n_shot,
        "source": CEVAL_URL,
        "subjects": by_subject,
        "total": {"n": total, "correct": total_correct, "accuracy": total_correct / total if total else 0.0},
        "note": "Scored official C-Eval validation/dev split locally. Test leaderboard scores require official submission.",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_jsonl(output_dir / "predictions.jsonl", all_rows)

    lines = [
        f"# Official C-Eval Medical Validation ({args.label})",
        "",
        f"- official_ceval: `{summary['official_ceval']}`",
        f"- split: `{args.split}`",
        f"- scoring: `{args.scoring}`",
        f"- n_shot: `{args.n_shot}`",
        f"- official_test_leaderboard: `{summary['official_ceval_test_leaderboard']}`",
        f"- total_accuracy: `{summary['total']['accuracy']:.4f}` ({total_correct}/{total})",
        f"- source: `{CEVAL_URL}`",
        "",
        "| subject | n | correct | accuracy |",
        "|---|---:|---:|---:|",
    ]
    for subject, item in by_subject.items():
        lines.append(f"| {subject} | {item['n']} | {item['correct']} | {item['accuracy']:.4f} |")
    lines.append("")
    lines.append("This is a local score on the official C-Eval validation/dev split, not a submitted test leaderboard result.")
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
