from __future__ import annotations

import sys
from pathlib import Path


LAB_ROOT = Path(__file__).resolve().parents[1]
if str(LAB_ROOT) not in sys.path:
    sys.path.insert(0, str(LAB_ROOT))

from query_compiler.candidate_snapshot import build_candidate_snapshot, write_candidate_snapshot  # type: ignore  # noqa: E402


def test_build_candidate_snapshot_emits_one_row_per_protein_candidate() -> None:
    rows = build_candidate_snapshot(limit=3)

    assert len(rows) == 3

    first = rows[0]
    assert "candidate_id" in first
    assert "canonical_name" in first
    assert "scaffold_type" in first
    assert "oral_class" in first
    assert "best_affinity_value" in first
    assert "trace" in first
    assert "protein_id" in first["trace"]


def test_write_candidate_snapshot_outputs_rows_and_manifest(tmp_path: Path) -> None:
    rows_path = tmp_path / "snapshot.jsonl"
    manifest_path = tmp_path / "manifest.json"

    manifest = write_candidate_snapshot(rows_path, manifest_path, limit=5)

    assert rows_path.exists()
    assert manifest_path.exists()
    assert manifest["counts"]["rows"] == 5
    assert "protein_cards" in manifest["sources"]
    assert "interaction_cards" in manifest["sources"]
