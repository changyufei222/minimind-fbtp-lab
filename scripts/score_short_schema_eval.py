from __future__ import annotations

import argparse
import json
from pathlib import Path


EXPECTED_KEYS = [
    "RecordType",
    "Antibody",
    "TargetLineage",
    "BindingOutcome",
    "EpitopeDomain",
    "PDBID",
    "ResolutionA",
    "SourceIdentifier",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score strict short-schema evaluation outputs")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_lines(answer: str) -> list[str]:
    if "\n" in answer:
        return [line.strip() for line in answer.splitlines() if line.strip()]
    parts = []
    for key in EXPECTED_KEYS:
        marker = f"{key}:"
        idx = answer.find(marker)
        if idx >= 0:
            parts.append((idx, key))
    parts.sort()
    if not parts:
        return [answer.strip()] if answer.strip() else []
    lines: list[str] = []
    for i, (idx, key) in enumerate(parts):
        end = parts[i + 1][0] if i + 1 < len(parts) else len(answer)
        lines.append(answer[idx:end].strip())
    return lines


def score_row(row: dict) -> dict:
    answer = row.get("answer", "")
    lines = normalize_lines(answer)
    line_map = {}
    for line in lines:
        if ":" in line:
            key, value = line.split(":", 1)
            line_map[key.strip()] = value.strip()

    key_hits = sum(1 for key in EXPECTED_KEYS if key in line_map)
    exact_8_lines = len(lines) == 8
    recordtype_ok = line_map.get("RecordType") == "AntibodyAntigenEvidence"
    extra_text = 0
    if lines:
        extra_text = 1 if any(":" not in line for line in lines) else 0
    compliant = exact_8_lines and key_hits == 8 and recordtype_ok and extra_text == 0
    return {
        "id": row.get("id"),
        "key_hits": key_hits,
        "exact_8_lines": exact_8_lines,
        "recordtype_ok": recordtype_ok,
        "extra_text": extra_text,
        "compliant": compliant,
        "answer_preview": answer[:240],
    }


def write_markdown(path: Path, rows: list[dict], summary: dict) -> None:
    lines = [
        "# Short Schema Eval Score",
        "",
        f"- total: `{summary['total']}`",
        f"- compliant: `{summary['compliant']}`",
        f"- compliance_rate: `{summary['compliance_rate']}`",
        f"- exact_8_lines: `{summary['exact_8_lines']}`",
        f"- average_key_hits: `{summary['average_key_hits']}`",
        "",
        "| id | key_hits | exact_8_lines | recordtype_ok | compliant | answer_preview |",
        "|---|---:|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['id']} | {row['key_hits']} | {row['exact_8_lines']} | {row['recordtype_ok']} | {row['compliant']} | {row['answer_preview'].replace('|', '/')} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = load_rows(Path(args.input_jsonl))
    scored = [score_row(row) for row in rows]
    total = len(scored)
    compliant = sum(1 for row in scored if row["compliant"])
    exact_8_lines = sum(1 for row in scored if row["exact_8_lines"])
    average_key_hits = round(sum(row["key_hits"] for row in scored) / total, 2) if total else 0.0
    summary = {
        "total": total,
        "compliant": compliant,
        "compliance_rate": round(compliant / total, 4) if total else 0.0,
        "exact_8_lines": exact_8_lines,
        "average_key_hits": average_key_hits,
    }

    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps({"summary": summary, "rows": scored}, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(output_md, scored, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
