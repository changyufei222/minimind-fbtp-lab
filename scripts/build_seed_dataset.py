from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAGPPI_CSV = PROJECT_ROOT / "llm-rag-knowledge-base" / "data" / "hf_ragppi_sample" / "ragppi_ingest.csv"
COVUNIBIND_CSV = PROJECT_ROOT / "llm-rag-knowledge-base" / "data" / "hf_covunibind_sample" / "covunibind_covabdab_binding_ingest.csv"
OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
TRAIN_PATH = OUT_DIR / "fbtp_sft_seed_train.jsonl"
DEV_PATH = OUT_DIR / "fbtp_sft_seed_dev.jsonl"
MANIFEST_PATH = OUT_DIR / "fbtp_sft_seed_manifest.json"


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
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


def build_ragppi_rows(df: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    for _, row in df.iterrows():
        protein_a = str(row.get("Proteins_canonical_name", "")).strip()
        protein_b = str(row.get("Targets_gene_name", "")).strip()
        abstract = str(row.get("Proteins_description", "")).strip()
        answer = str(row.get("GT_answer", "")).strip()
        source_id = str(row.get("Sources_identifier", "")).strip()
        source_title = str(row.get("Sources_title", "")).strip()

        if protein_a and protein_b and answer:
            prompt = (
                f"Based on the following protein interaction abstract, summarize what is reported about the interaction "
                f"between {protein_a} and {protein_b}.\n\nAbstrac<local_path_removed>"
            )
            rows.append(_conversation(prompt, answer))

            prompt2 = (
                f"Read the following protein interaction record and provide a concise evidence-backed summary. "
                f"Also mention the source identifier.\n\n"
                f"Protein A: {protein_a}\nProtein B: {protein_b}\n"
                f"Source: {source_title}\nIdentifier: {source_id}\n"
                f"Abstract: {abstract}"
            )
            answer2 = f"{answer}\n\nSource identifier: {source_id}"
            rows.append(_conversation(prompt2, answer2))

    return rows


def build_cov_rows(df: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    for idx, row in df.iterrows():
        antibody = str(row.get("Proteins_canonical_name", "")).strip()
        lineage = str(row.get("Targets_gene_name", "")).strip()
        host = str(row.get("Targets_species_name", "")).strip()
        pdb_id = str(row.get("pdb_id", "")).strip()
        domain = str(row.get("epitope_domain", "")).strip()
        source_id = str(row.get("Sources_identifier", "")).strip()
        resolution = row.get("PDB_Resolution_A")
        summary = str(row.get("Proteins_description", "")).strip()
        epitope = str(row.get("epitope_residues", "")).strip()

        assistant = (
            f"Antibody: {antibody}\n"
            f"Target lineage: {lineage}\n"
            f"Epitope domain: {domain}\n"
            f"PDB ID: {pdb_id}\n"
            f"Resolution: {resolution}\n"
            f"Source identifier: {source_id}\n"
            f"Host: {host}\n"
            f"Summary: {summary}"
        )
        prompt = (
            "Summarize the following structured antibody-antigen record into a compact evidence note.\n\n"
            f"{assistant}\nEpitope residues: {epitope}"
        )
        rows.append(_conversation(prompt, assistant))

        if idx < 25:
            prompt2 = (
                "Turn the following structured biological record into a short comparison-style summary with the key fields only.\n\n"
                f"Antibody: {antibody}\nLineage: {lineage}\nDomain: {domain}\nPDB: {pdb_id}\nResolution: {resolution}\nSource: {source_id}\n"
            )
            answer2 = (
                f"{antibody} targets lineage {lineage} through the {domain} domain. "
                f"The representative structure is {pdb_id} with resolution {resolution}. "
                f"Source identifier: {source_id}."
            )
            rows.append(_conversation(prompt2, answer2))

    return rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rag_df = pd.read_csv(RAGPPI_CSV)
    cov_df = pd.read_csv(COVUNIBIND_CSV)

    all_rows = build_ragppi_rows(rag_df) + build_cov_rows(cov_df)
    dev_rows = all_rows[:20]
    train_rows = all_rows[20:]

    _write_jsonl(TRAIN_PATH, train_rows)
    _write_jsonl(DEV_PATH, dev_rows)

    manifest = {
        "sources": {
            "ragppi_csv": str(RAGPPI_CSV),
            "covunibind_csv": str(COVUNIBIND_CSV),
        },
        "counts": {
            "train": len(train_rows),
            "dev": len(dev_rows),
            "total": len(all_rows),
        },
        "notes": [
            "Generated from local public protein datasets already prepared in llm-rag-knowledge-base",
            "Intended as a first-stage seed SFT dataset for MiniMind-FBTP LoRA / SFT experiments",
        ],
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

