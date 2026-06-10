from __future__ import annotations

import sys
from pathlib import Path


LAB_ROOT = Path(__file__).resolve().parents[1]
if str(LAB_ROOT) not in sys.path:
    sys.path.insert(0, str(LAB_ROOT))

from query_compiler.eval_paths import resolve_base_weight_path, resolve_lora_weight_path  # type: ignore  # noqa: E402


def test_resolve_base_weight_path_falls_back_to_upstream_out(tmp_path: Path) -> None:
    save_dir = tmp_path / "reports" / "checkpoints"
    upstream_root = tmp_path / "upstream-minimind-full"
    lab_root = tmp_path / "minimind-fbtp-lab"
    expected = upstream_root / "out" / "full_sft_768.pth"
    expected.parent.mkdir(parents=True, exist_ok=True)
    expected.write_bytes(b"base")

    resolved = resolve_base_weight_path(
        weight="full_sft",
        hidden_size=768,
        save_dir=save_dir,
        upstream_root=upstream_root,
        lab_root=lab_root,
    )

    assert resolved == expected


def test_resolve_lora_weight_path_falls_back_to_lab_checkpoint_subdir(tmp_path: Path) -> None:
    save_dir = tmp_path / "upstream-minimind-full" / "out"
    upstream_root = tmp_path / "upstream-minimind-full"
    lab_root = tmp_path / "minimind-fbtp-lab"
    expected = lab_root / "reports" / "checkpoints" / "lora_query_compiler_104m" / "fbbp_query_compiler_104m_768.pth"
    expected.parent.mkdir(parents=True, exist_ok=True)
    expected.write_bytes(b"lora")

    resolved = resolve_lora_weight_path(
        lora_weight="fbbp_query_compiler_104m",
        hidden_size=768,
        save_dir=save_dir,
        lora_subdir="lora_query_compiler_104m",
        lab_root=lab_root,
    )

    assert resolved == expected
