from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LAB_ROOT = Path(__file__).resolve().parents[1]
PROTEIN_CARDS_PATH = PROJECT_ROOT / "llm-rag-knowledge-base" / "data" / "schema_tables_rag_ready" / "protein_cards.jsonl"
INTERACTION_CARDS_PATH = PROJECT_ROOT / "llm-rag-knowledge-base" / "data" / "schema_tables_rag_ready" / "interaction_cards.jsonl"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _parse_card_content(content: str) -> tuple[dict[str, str], dict[str, list[str]]]:
    scalars: dict[str, str] = {}
    sections: dict[str, list[str]] = {}
    current_section: str | None = None

    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.endswith(":") and ": " not in line:
            current_section = line[:-1]
            sections.setdefault(current_section, [])
            continue
        if current_section is not None:
            if line.startswith("- "):
                sections[current_section].append(line[2:].strip())
                continue
            if ": " in line:
                sections[current_section].append(line)
                continue
        if ": " in line:
            key, value = line.split(": ", 1)
            scalars[key.strip()] = value.strip()
            current_section = None
    return scalars, sections


def _first_match(pattern: str, text: str) -> str | None:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    return match.group(1).strip() if match else None


def _normalize_boolean(value: str | None) -> bool | None:
    if value is None:
        return None
    lowered = value.strip().lower()
    if lowered in {"yes", "true", "1"}:
        return True
    if lowered in {"no", "false", "0"}:
        return False
    return None


def _to_float(value: str | None) -> float | None:
    if value is None:
        return None
    match = re.search(r"(-?\d+(?:\.\d+)?)", value)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _first_non_empty(values: list[str | None]) -> str | None:
    for value in values:
        if value and value.lower() not in {"n/a", "na", "unknown", "none"}:
            return value
    return None


def _merge_interaction_signals(interaction_rows: list[dict[str, Any]]) -> dict[str, Any]:
    inhibitory_values: list[bool] = []
    affinity_values: list[float] = []
    experimental = False
    interaction_ids: list[str] = []

    for row in interaction_rows:
        interaction_ids.append(str(row.get("interaction_id")))
        content = str(row.get("content", ""))
        inhibitory = _normalize_boolean(_first_match(r"Is Inhibitor<local_path_removed>", content))
        if inhibitory is not None:
            inhibitory_values.append(inhibitory)
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith("- "):
                numeric = _to_float(stripped)
                if numeric is not None:
                    affinity_values.append(numeric)
                if "(Experimental)" in stripped:
                    experimental = True

    return {
        "interaction_ids": [interaction_id for interaction_id in interaction_ids if interaction_id and interaction_id != "None"],
        "inhibitory": True if any(inhibitory_values) else (False if inhibitory_values else None),
        "best_affinity_value": min(affinity_values) if affinity_values else None,
        "has_experimental_affinity": experimental or bool(affinity_values),
    }


def build_candidate_snapshot(limit: int | None = None) -> list[dict[str, Any]]:
    protein_cards = _load_jsonl(PROTEIN_CARDS_PATH)
    interaction_cards = _load_jsonl(INTERACTION_CARDS_PATH)

    interactions_by_protein: dict[str, list[dict[str, Any]]] = {}
    for row in interaction_cards:
        protein_id = row.get("metadata", {}).get("protein_id")
        if protein_id:
            interactions_by_protein.setdefault(str(protein_id), []).append(row)

    snapshot_rows: list[dict[str, Any]] = []
    source_rows = protein_cards[:limit] if limit is not None else protein_cards

    for row in source_rows:
        metadata = row.get("metadata", {})
        scalars, sections = _parse_card_content(str(row.get("content", "")))

        protein_id = str(metadata.get("protein_id") or scalars.get("Protein ID") or row.get("chunk_id"))
        domain_entries = sections.get("Domains", [])
        cmc_entries = sections.get("CMC / Developability", [])
        interaction_entries = sections.get("Interactions", [])
        affinity_entries = sections.get("Affinity", [])

        scaffold_type = _first_non_empty([_first_match(r"scaffold=([^;]+)", entry) for entry in domain_entries])
        engineered = _normalize_boolean(_first_non_empty([_first_match(r"engineered=([^;]+)", entry) for entry in domain_entries]))

        oral_class = _first_non_empty([_first_match(r"oral=([^;]+)", entry) for entry in cmc_entries])
        expression_system = _first_non_empty([_first_match(r"expression=([^;]+)", entry) for entry in cmc_entries])
        stability_signal = _first_non_empty([_first_match(r"stability=(.+)", entry) for entry in cmc_entries])

        inhibitory = _normalize_boolean(_first_non_empty([_first_match(r"inhibitory=([^;]+)", entry) for entry in interaction_entries]))
        affinity_values = [_to_float(entry) for entry in affinity_entries]
        affinity_values = [value for value in affinity_values if value is not None]
        has_experimental_affinity = any("experimental" in entry.lower() for entry in affinity_entries) or bool(affinity_values)

        merged = _merge_interaction_signals(interactions_by_protein.get(protein_id, []))
        if inhibitory is None:
            inhibitory = merged["inhibitory"]
        if not affinity_values and merged["best_affinity_value"] is not None:
            affinity_values = [merged["best_affinity_value"]]
        if not has_experimental_affinity:
            has_experimental_affinity = bool(merged["has_experimental_affinity"])

        snapshot_rows.append(
            {
                "candidate_id": protein_id,
                "canonical_name": scalars.get("Canonical Name") or scalars.get("Protein Card") or protein_id,
                "organism": scalars.get("Organism") or "Unknown",
                "scaffold_type": scaffold_type or "unknown",
                "engineered": engineered,
                "oral_class": oral_class or "Unknown",
                "expression_system": expression_system or "Unknown",
                "stability_signal": stability_signal or "Unknown",
                "inhibitory": inhibitory,
                "has_experimental_affinity": has_experimental_affinity,
                "best_affinity_value": min(affinity_values) if affinity_values else None,
                "trace": {
                    "protein_id": protein_id,
                    "interaction_ids": metadata.get("interaction_ids") or merged["interaction_ids"],
                    "table_sources": metadata.get("table_sources", []),
                    "card_source": row.get("source"),
                },
            }
        )

    return snapshot_rows


def write_candidate_snapshot(rows_path: Path, manifest_path: Path, limit: int | None = None) -> dict[str, Any]:
    rows = build_candidate_snapshot(limit=limit)
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with rows_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "source_paths": {
            "protein_cards": str(PROTEIN_CARDS_PATH),
            "interaction_cards": str(INTERACTION_CARDS_PATH),
        },
        "sources": ["protein_cards", "interaction_cards"],
        "counts": {"rows": len(rows)},
        "limit": limit,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest
