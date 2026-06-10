from __future__ import annotations

from pathlib import Path


def _first_existing(candidates: list[Path]) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("No checkpoint found in candidates: " + ", ".join(str(item) for item in candidates))


def resolve_base_weight_path(
    *,
    weight: str,
    hidden_size: int,
    save_dir: Path,
    upstream_root: Path,
    lab_root: Path,
) -> Path:
    filename = f"{weight}_{hidden_size}.pth"
    candidates = [
        save_dir / filename,
        upstream_root / "out" / filename,
        lab_root / "reports" / "checkpoints" / filename,
    ]
    return _first_existing(candidates)


def resolve_lora_weight_path(
    *,
    lora_weight: str,
    hidden_size: int,
    save_dir: Path,
    lora_subdir: str,
    lab_root: Path,
) -> Path:
    filename = f"{lora_weight}_{hidden_size}.pth"
    candidates = [
        save_dir / lora_subdir / filename,
        lab_root / "reports" / "checkpoints" / lora_subdir / filename,
        save_dir / filename,
    ]
    return _first_existing(candidates)
