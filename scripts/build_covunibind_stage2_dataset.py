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
TRAIN_PATH = OUT_DIR / "covunibind_stage2_train.jsonl"
DEV_PATH = OUT_DIR / "covunibind_stage2_dev.jsonl"
MANIFEST_PATH = OUT_DIR / "covunibind_stage2_manifest.json"
EVAL_PROMPTS_PATH = EVAL_DIR / "covunibind_stage2_eval_prompts.jsonl"


PROMPT_TEMPLATES = [
    "Convert the following antibody-antigen record into the standard evidence card format.\n\n{record}",
    "Normalize this CoVUniBind record into a compact structured summary. Keep every field on its own line.\n\n{record}",
    "Turn this record into an evidence-focused note for a protein binding knowledge base.\n\n{record}",
    "Read the structured record below and rewrite it as a standard antibody-antigen evidence card.\n\n{record}",
    "Prepare a comparison-ready structured summary from the following CoVUniBind entry.\n\n{record}",
    "Summarize the following antibody record into the FBTP structured note schema.\n\n{record}",
    "Create a compact, machine-readable evidence card from this CoVUniBind example.\n\n{record}",
    "Rewrite this antibody-lineage binding entry into the standard fields used for structured analysis.\n\n{record}",
]


@dataclass(frozen=True)
class CovRecord:
    file_name: str
    antibody: str
    lineage: str
    host: str
    assay: str
    pdb_id: str
    domain: str
    resolution: str
    binding_value: str
    source_title: str
    source_id: str
    source_date: str
    mutations: str
    epitope_residues: str
    summary: str


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
    if pd.isna(value):
        return "NA"
    if value is None:
        return "NA"
    text = str(value).strip()
    return text if text else "NA"


def to_record(row: pd.Series) -> CovRecord:
    return CovRecord(
        file_name=_clean(row.get("File_Name")),
        antibody=_clean(row.get("Proteins_canonical_name")),
        lineage=_clean(row.get("Targets_gene_name")),
        host=_clean(row.get("Targets_species_name")),
        assay=_clean(row.get("Structural_Info_method")),
        pdb_id=_clean(row.get("pdb_id")),
        domain=_clean(row.get("epitope_domain")),
        resolution=_clean(row.get("PDB_Resolution_A")),
        binding_value=_clean(row.get("target_value")),
        source_title=_clean(row.get("Sources_title")),
        source_id=_clean(row.get("Sources_identifier")),
        source_date=_clean(row.get("Sources_publication_date")),
        mutations=_clean(row.get("mutations")),
        epitope_residues=_clean(row.get("epitope_residues")),
        summary=_clean(row.get("Proteins_description")),
    )


def render_input(record: CovRecord) -> str:
    return (
        f"File_Name: {record.file_name}\n"
        f"Antibody: {record.antibody}\n"
        f"Target lineage: {record.lineage}\n"
        f"Host species: {record.host}\n"
        f"Assay: {record.assay}\n"
        f"Binding outcome: {record.binding_value}\n"
        f"Epitope domain: {record.domain}\n"
        f"PDB ID: {record.pdb_id}\n"
        f"Resolution (A): {record.resolution}\n"
        f"Mutations: {record.mutations}\n"
        f"Epitope residues: {record.epitope_residues}\n"
        f"Source title: {record.source_title}\n"
        f"Source identifier: {record.source_id}\n"
        f"Source publication date: {record.source_date}\n"
        f"Summary: {record.summary}"
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
        f"HostSpecies: {record.host}\n"
        f"Assay: {record.assay}\n"
        f"KeyMutations: {record.mutations}\n"
        f"EpitopeResidues: {record.epitope_residues}\n"
        f"SourceTitle: {record.source_title}\n"
        f"SourceIdentifier: {record.source_id}\n"
        f"SourceDate: {record.source_date}\n"
        f"Summary: {record.summary}\n"
        f"Interpretation: {record.antibody} is reported against lineage {record.lineage} with epitope domain {record.domain}; "
        f"representative structure {record.pdb_id} has resolution {record.resolution} A and source identifier {record.source_id}."
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
                "id": f"cov_stage2_{idx}",
                "category": "covunibind",
                "prompt": PROMPT_TEMPLATES[0].format(record=render_input(record)),
                "reference_note": (
                    "Should preserve the standard evidence-card schema and mention antibody, lineage, "
                    "binding outcome, epitope domain, PDB ID, resolution, and source identifier."
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
        "notes": [
            "Stage-2 dataset is CoVUniBind-only to reduce task entropy for the 26M MiniMind model.",
            "Target output uses a single normalized evidence-card schema for every sample.",
            "Recommended for round-2 LoRA with lower learning rate and fewer epochs.",
        ],
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
