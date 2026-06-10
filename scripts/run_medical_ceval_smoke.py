from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


LAB_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = LAB_ROOT / "data" / "eval" / "medical_ceval_smoke.jsonl"
DEFAULT_OUTPUT = LAB_ROOT / "reports" / "algorithm_resume" / "medical_ceval_smoke_score.json"
DEFAULT_MD = LAB_ROOT / "reports" / "algorithm_resume" / "medical_ceval_smoke_score.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score a C-Eval-style medical multiple-choice smoke set")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--predictions", help="Optional JSONL with id and prediction fields")
    parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--output-md", default=str(DEFAULT_MD))
    parser.add_argument("--oracle", action="store_true", help="Use gold answers as predictions to validate the scorer")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_predictions(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    return {str(row["id"]): str(row.get("prediction", "")).strip().upper()[:1] for row in load_jsonl(path)}


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset)
    rows = load_jsonl(dataset_path)
    predictions = load_predictions(Path(args.predictions) if args.predictions else None)
    scored = []
    correct = 0
    for row in rows:
        pred = row["answer"] if args.oracle else predictions.get(str(row["id"]), "")
        ok = pred == row["answer"]
        correct += int(ok)
        scored.append({**row, "prediction": pred, "correct": ok})
    accuracy = correct / len(rows) if rows else 0.0
    summary = {
        "dataset": str(dataset_path),
        "mode": "oracle_scorer_check" if args.oracle else "prediction_scoring",
        "n": len(rows),
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "official_ceval": False,
        "note": "C-Eval-compatible local smoke set, not the official C-Eval benchmark.",
    }
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps({"summary": summary, "rows": scored}, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Medical C-Eval-Style Smoke Score",
        "",
        f"- mode: `{summary['mode']}`",
        f"- official C-Eval: `{summary['official_ceval']}`",
        f"- n: `{summary['n']}`",
        f"- accuracy: `{summary['accuracy']}`",
        f"- note: {summary['note']}",
        "",
        "| id | category | answer | prediction | correct |",
        "|---|---|---:|---:|---:|",
    ]
    for row in scored:
        lines.append(f"| {row['id']} | {row['category']} | {row['answer']} | {row['prediction']} | {row['correct']} |")
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
