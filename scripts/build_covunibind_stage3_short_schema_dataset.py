from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LAB_ROOT = Path(__file__).resolve().parents[1]
LOCAL_COVUNIBIND_CSV = LAB_ROOT / "data" / "raw" / "covunibind_covabdab_binding_ingest.csv"
PROJECT_COVUNIBIND_CSV = PROJECT_ROOT / "llm-rag-knowledge-base" / "data" / "hf_covunibind_sample" / "covunibind_covabdab_binding_ingest.csv"
OUT_DIR = LAB_ROOT / "data" / "processed"
EVAL_DIR = LAB_ROOT / "data" / "eval"
TRAIN_PATH = OUT_DIR / "covunibind_stage3_short_schema_train.jsonl"
DEV_PATH = OUT_DIR / "covunibind_stage3_short_schema_dev.jsonl"
MANIFEST_PATH = OUT_DIR / "covunibind_stage3_short_schema_manifest.json"
EVAL_PROMPTS_PATH = EVAL_DIR / "covunibind_stage3_short_schema_eval_prompts.jsonl"


PROMPT_TEMPLATES = [
    "Rewrite the following record into the exact 8-line schema shown below. Output only the schema lines.\n\n{record}",
    "Normalize this antibody-antigen record into the fixed 8-line schema. Do not add any explanation.\n\n{record}",
    "Convert the record below into the exact schema with these keys only: RecordType, Antibody, TargetLineage, BindingOutcome, EpitopeDomain, PDBID, ResolutionA, SourceIdentifier.\n\n{record}",
    "Read the following entry and output only the normalized 8-line evidence schema.\n\n{record}",
    "Transform this CoVUniBind example into the fixed schema. Keep each key on its own line and do not add extra text.\n\n{record}",
    "Produce the canonical short evidence card from the fields below. Output only the 8 required lines.\n\n{record}",
    "Standardize the following record into the short FBTP schema with no commentary.\n\n{record}",
    "Map this entry to the exact 8-field schema. Preserve values faithfully and output nothing else.\n\n{record}",
    "Return only the normalized short schema for the record below.\n\n{record}",
    "Format the following antibody-lineage entry using the strict 8-line schema and no additional prose.\n\n{record}",
]

SCHEMA_KEYS = [
    "RecordType",
    "Antibody",
    "TargetLineage",
    "BindingOutcome",
    "EpitopeDomain",
    "PDBID",
    "ResolutionA",
    "SourceIdentifier",
]


@dataclass(frozen=True)
class CovRecord:
    antibody: str
    lineage: str
    binding_value: str
    domain: str
    pdb_id: str
    resolution: str
    source_id: str


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _conversation(user: str, assistant: str) -> dict:
    return {
        "conversations": [
            {"role": "user", "content": user.strip()},
            {"role": "assistant", "content": assistant.strip()},
        ]
    }


def _clean(value: object) -> str:
    if pd.isna(value) or value is None:
        return "NA"
    text = str(value).strip()
    return text if text else "NA"


def to_record(row: pd.Series) -> CovRecord:
    return CovRecord(
        antibody=_clean(row.get("Proteins_canonical_name")),
        lineage=_clean(row.get("Targets_gene_name")),
        binding_value=_clean(row.get("target_value")),
        domain=_clean(row.get("epitope_domain")),
        pdb_id=_clean(row.get("pdb_id")),
        resolution=_clean(row.get("PDB_Resolution_A")),
        source_id=_clean(row.get("Sources_identifier")),
    )


def render_input(record: CovRecord) -> str:
    return (
        f"Antibody: {record.antibody}\n"
        f"TargetLineage: {record.lineage}\n"
        f"BindingOutcome: {record.binding_value}\n"
        f"EpitopeDomain: {record.domain}\n"
        f"PDBID: {record.pdb_id}\n"
        f"ResolutionA: {record.resolution}\n"
        f"SourceIdentifier: {record.source_id}"
    )


def render_output(record: CovRecord) -> str:
    return (
        "RecordType: AntibodyAntigenEvidence\n"
        f"Antibody: {record.antibody}\n"
        f"TargetLineage: {record.lineage}\n"
        f"BindingOutcome: {record.binding_value}\n"
        f"EpitopeDomain: {record.domain}\n"
        f"PDBID: {record.pdb_id}\n"
        f"ResolutionA: {record.resolution}\n"
        f"SourceIdentifier: {record.source_id}"
    )


def build_rows(df: pd.DataFrame) -> tuple[list[dict], list[dict], list[dict]]:
    sorted_df = df.sort_values(
        by=["Proteins_canonical_name", "Targets_gene_name", "pdb_id", "Sources_identifier"],
        kind="stable",
    ).reset_index(drop=True)
    dev_base = sorted_df.head(6)
    train_base = sorted_df.iloc[6:]

    def expand(frame: pd.DataFrame) -> list[dict]:
        rows: list[dict] = []
        for _, raw_row in frame.iterrows():
            record = to_record(raw_row)
            record_text = render_input(record)
            target_text = render_output(record)
            for template in PROMPT_TEMPLATES:
                rows.append(_conversation(template.format(record=record_text), target_text))
        return rows

    train_rows = expand(train_base)
    dev_rows = expand(dev_base)

    eval_rows: list[dict] = []
    for idx, (_, raw_row) in enumerate(dev_base.iterrows(), start=1):
        record = to_record(raw_row)
        eval_rows.append(
            {
                "id": f"cov_stage3_{idx}",
                "category": "covunibind-short-schema",
                "prompt": PROMPT_TEMPLATES[0].format(record=render_input(record)),
                "reference_note": (
                    "Should output exactly the 8-line schema with keys: "
                    + ", ".join(SCHEMA_KEYS)
                ),
            }
        )

    return train_rows, dev_rows, eval_rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    source_csv = LOCAL_COVUNIBIND_CSV if LOCAL_COVUNIBIND_CSV.exists() else PROJECT_COVUNIBIND_CSV
    df = pd.read_csv(source_csv)
    train_rows, dev_rows, eval_rows = build_rows(df)

    _write_jsonl(TRAIN_PATH, train_rows)
    _write_jsonl(DEV_PATH, dev_rows)
    _write_jsonl(EVAL_PROMPTS_PATH, eval_rows)

    manifest = {
        "source_csv": str(source_csv),
        "source_rows": int(len(df)),
        "split_strategy": {
            "train_source_rows": int(len(df) - 6),
            "dev_source_rows": 6,
            "prompt_variants_per_row": len(PROMPT_TEMPLATES),
        },
        "counts": {
            "train": len(train_rows),
            "dev": len(dev_rows),
            "total": len(train_rows) + len(dev_rows),
            "eval_prompts": len(eval_rows),
        },
        "target_schema_keys": SCHEMA_KEYS,
        "notes": [
            "Stage-3 dataset uses a much narrower task: fixed 8-line schema only.",
            "Designed for MiniMind2 104M (768dim) to maximize the chance of measurable gains.",
            "This dataset removes long summaries, mutation spans, and free-form explanations.",
        ],
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
