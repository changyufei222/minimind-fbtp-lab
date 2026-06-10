from __future__ import annotations

import argparse
import json
import re
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

KEY_PATTERN = re.compile(r"(RecordType|Antibody|TargetLineage|BindingOutcome|EpitopeDomain|PDBID|ResolutionA|SourceIdentifier)\s*:")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize short-schema eval outputs conservatively")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
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


def extract_fields(answer: str) -> dict[str, str]:
    matches = list(KEY_PATTERN.finditer(answer))
    if not matches:
        return {}

    extracted: dict[str, str] = {}
    for index, match in enumerate(matches):
        key = match.group(1)
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(answer)
        value = answer[start:end].strip()
        value = re.sub(r"\s+", " ", value)
        value = value.strip(" -|")
        if value and key not in extracted:
            extracted[key] = value
    return extracted


def normalize_answer(answer: str) -> tuple[str, dict[str, str]]:
    extracted = extract_fields(answer)
    if not extracted:
        return answer.strip(), {}

    normalized: dict[str, str] = {}
    if "RecordType" in extracted:
        normalized["RecordType"] = extracted["RecordType"]
    elif len([k for k in extracted if k != "RecordType"]) >= 5:
        normalized["RecordType"] = "AntibodyAntigenEvidence"

    for key in EXPECTED_KEYS[1:]:
        if key in extracted:
            normalized[key] = extracted[key]

    lines = [f"{key}: {normalized[key]}" for key in EXPECTED_KEYS if key in normalized]
    return "\n".join(lines).strip(), normalized


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_markdown(path: Path, rows: list[dict]) -> None:
    lines = [
        "# Normalized Short Schema Eval",
        "",
        "| id | normalized_key_hits | normalized_answer | raw_answer_preview |",
        "|---|---:|---|---|",
    ]
    for row in rows:
        normalized = str(row["answer"]).replace("\n", "<br>").replace("|", "/")[:240]
        raw_preview = str(row["raw_answer"]).replace("\n", " ").replace("|", "/")[:200]
        lines.append(f"| {row['id']} | {row['normalized_key_hits']} | {normalized} | {raw_preview} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = load_rows(Path(args.input_jsonl))
    normalized_rows: list[dict] = []
    for row in rows:
        raw_answer = row.get("answer", "")
        normalized_answer, extracted = normalize_answer(raw_answer)
        new_row = dict(row)
        new_row["raw_answer"] = raw_answer
        new_row["answer"] = normalized_answer
        new_row["normalized_key_hits"] = len(extracted)
        new_row["normalizer_changed"] = normalized_answer != raw_answer.strip()
        normalized_rows.append(new_row)

    output_jsonl = Path(args.output_jsonl)
    output_md = Path(args.output_md)
    write_jsonl(output_jsonl, normalized_rows)
    write_markdown(output_md, normalized_rows)
    print(json.dumps({"rows": len(normalized_rows), "output_jsonl": str(output_jsonl), "output_md": str(output_md)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
