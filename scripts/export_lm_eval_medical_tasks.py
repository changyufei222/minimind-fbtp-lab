from __future__ import annotations

import json
from pathlib import Path


LAB_ROOT = Path(__file__).resolve().parents[1]
SOURCE = LAB_ROOT / "data" / "eval" / "medical_ceval_smoke.jsonl"
OUT_DIR = LAB_ROOT / "reports" / "algorithm_resume" / "lm_eval_medical_smoke"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    docs_path = OUT_DIR / "medical_ceval_smoke_docs.jsonl"
    yaml_path = OUT_DIR / "medical_ceval_smoke.yaml"
    docs = []
    with SOURCE.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            docs.append(
                {
                    "id": row["id"],
                    "query": f"{row['question']}\nA. {row['A']}\nB. {row['B']}\nC. {row['C']}\nD. {row['D']}\n答案：",
                    "choices": ["A", "B", "C", "D"],
                    "gold": ["A", "B", "C", "D"].index(row["answer"]),
                    "category": row["category"],
                }
            )
    with docs_path.open("w", encoding="utf-8") as handle:
        for doc in docs:
            handle.write(json.dumps(doc, ensure_ascii=False) + "\n")
    yaml_path.write_text(
        "\n".join(
            [
                "task: medical_ceval_smoke",
                "dataset_path: json",
                f"dataset_kwargs: {{data_files: {docs_path.as_posix()}}}",
                "test_split: train",
                "output_type: multiple_choice",
                "doc_to_text: '{{query}}'",
                "doc_to_choice: '{{choices}}'",
                "doc_to_target: '{{gold}}'",
                "metric_list:",
                "  - metric: acc",
                "    aggregation: mean",
                "    higher_is_better: true",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps({"docs": str(docs_path), "yaml": str(yaml_path), "n": len(docs)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
