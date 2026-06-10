from __future__ import annotations

import json
import sys
from pathlib import Path


LAB_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB_ROOT / "scripts"))

from plot_training_loss import parse_logs  # noqa: E402


def test_dpo_smoke_dataset_has_chosen_rejected_pairs() -> None:
    path = LAB_ROOT / "data" / "processed" / "fbbp_query_compiler_dpo_smoke.jsonl"
    first = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert "chosen" in first
    assert "rejected" in first
    assert first["chosen"][-1]["role"] == "assistant"
    assert first["rejected"][-1]["role"] == "assistant"
    assert first["chosen"][-1]["content"] != first["rejected"][-1]["content"]


def test_medical_ceval_smoke_dataset_shape() -> None:
    path = LAB_ROOT / "data" / "eval" / "medical_ceval_smoke.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) >= 5
    assert all(row["answer"] in {"A", "B", "C", "D"} for row in rows)


def test_loss_parser_handles_minimind_lora_log() -> None:
    rows = parse_logs("tests/fixtures/*.log")
    assert rows
    assert rows[0]["loss"] == "2.1234"
