from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


LAB_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = LAB_ROOT / "data" / "external" / "ceval-exam"
DEFAULT_OUTPUT = LAB_ROOT / "data" / "processed" / "medical_ceval_dpo_dev.jsonl"
DEFAULT_REPORT = LAB_ROOT / "reports" / "algorithm_resume" / "medical_ceval_dpo_dataset_report.md"
SUBJECT_NAMES = {
    "basic_medicine": "基础医学",
    "clinical_medicine": "临床医学",
    "physician": "医师资格",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build medical QA DPO pairs from C-Eval dev subjects")
    parser.add_argument("--ceval-root", default=str(DEFAULT_ROOT))
    parser.add_argument("--subjects", nargs="+", default=list(SUBJECT_NAMES))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def prompt_for(row: dict[str, str], subject: str) -> str:
    subject_name = SUBJECT_NAMES.get(subject, subject)
    return (
        f"以下是中国关于{subject_name}考试的单项选择题。请只输出正确选项字母。\n\n"
        f"{row['question']}\n"
        f"A. {row['A']}\n"
        f"B. {row['B']}\n"
        f"C. {row['C']}\n"
        f"D. {row['D']}\n"
        "答案："
    )


def wrong_answer(answer: str) -> str:
    options = ["A", "B", "C", "D"]
    idx = options.index(answer)
    return options[(idx + 1) % len(options)]


def build_pair(row: dict[str, str], subject: str, idx: int) -> dict[str, Any]:
    answer = row["answer"].strip().upper()
    rejected = wrong_answer(answer)
    user = {"role": "user", "content": prompt_for(row, subject)}
    return {
        "id": f"{subject}_dev_{row.get('id', idx)}",
        "subject": subject,
        "chosen": [user, {"role": "assistant", "content": answer}],
        "rejected": [user, {"role": "assistant", "content": rejected}],
        "preference_source": "ceval_dev_correct_option_vs_deterministic_wrong_option",
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    args = parse_args()
    root = Path(args.ceval_root)
    rows: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for subject in args.subjects:
        csv_path = root / "dev" / f"{subject}_dev.csv"
        subject_rows = read_csv(csv_path)
        counts[subject] = len(subject_rows)
        for idx, row in enumerate(subject_rows):
            rows.append(build_pair(row, subject, idx))

    output = Path(args.output)
    report = Path(args.report)
    write_jsonl(output, rows)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        "\n".join(
            [
                "# Medical C-Eval DPO Dataset Report",
                "",
                f"- source_root: `{root}`",
                f"- output: `{output}`",
                f"- pairs: `{len(rows)}`",
                f"- subjects: `{', '.join(args.subjects)}`",
                f"- subject_counts: `{counts}`",
                "- chosen: correct C-Eval dev option",
                "- rejected: deterministic wrong option",
                "- intended use: separate medical-QA DPO branch, not the FBBP query-compiler result",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), "pairs": len(rows), "counts": counts, "report": str(report)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
