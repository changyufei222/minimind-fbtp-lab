from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LAB_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_ROOT = PROJECT_ROOT / "upstream-minimind-full"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch MiniMind training from a local config")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    data_path = (LAB_ROOT / config.get("data_path", "data/processed/fbtp_sft_seed_train.jsonl")).resolve()

    trainer = config["trainer"]
    if trainer == "lora":
        trainer_script = UPSTREAM_ROOT / "trainer" / "train_lora.py"
        save_dir = (LAB_ROOT / config.get("save_dir", "reports/checkpoints/lora")).resolve()
        command = [
            sys.executable,
            str(trainer_script),
            "--save_dir", str(save_dir),
            "--lora_name", config["experiment_name"],
            "--epochs", str(config["epochs"]),
            "--batch_size", str(config["batch_size"]),
            "--learning_rate", str(config["learning_rate"]),
            "--device", config["device"],
            "--dtype", config["dtype"],
            "--num_workers", str(config["num_workers"]),
            "--accumulation_steps", str(config["accumulation_steps"]),
            "--log_interval", str(config["log_interval"]),
            "--save_interval", str(config["save_interval"]),
            "--hidden_size", str(config["hidden_size"]),
            "--num_hidden_layers", str(config["num_hidden_layers"]),
            "--max_seq_len", str(config["max_seq_len"]),
            "--use_moe", str(config["use_moe"]),
            "--data_path", str(data_path),
            "--from_weight", config["from_weight"],
        ]
    elif trainer == "full_sft":
        trainer_script = UPSTREAM_ROOT / "trainer" / "train_full_sft.py"
        save_dir = (LAB_ROOT / config.get("save_dir", "reports/checkpoints/full_sft")).resolve()
        command = [
            sys.executable,
            str(trainer_script),
            "--save_dir", str(save_dir),
            "--save_weight", config["save_weight"],
            "--epochs", str(config["epochs"]),
            "--batch_size", str(config["batch_size"]),
            "--learning_rate", str(config["learning_rate"]),
            "--device", config["device"],
            "--dtype", config["dtype"],
            "--num_workers", str(config["num_workers"]),
            "--accumulation_steps", str(config["accumulation_steps"]),
            "--log_interval", str(config["log_interval"]),
            "--save_interval", str(config["save_interval"]),
            "--hidden_size", str(config["hidden_size"]),
            "--num_hidden_layers", str(config["num_hidden_layers"]),
            "--max_seq_len", str(config["max_seq_len"]),
            "--use_moe", str(config["use_moe"]),
            "--data_path", str(data_path),
            "--from_weight", config["from_weight"],
        ]
    elif trainer == "dpo":
        trainer_script = UPSTREAM_ROOT / "trainer" / "train_dpo.py"
        save_dir = (LAB_ROOT / config.get("save_dir", "reports/checkpoints/dpo")).resolve()
        command = [
            sys.executable,
            str(trainer_script),
            "--save_dir", str(save_dir),
            "--save_weight", config["experiment_name"],
            "--epochs", str(config["epochs"]),
            "--batch_size", str(config["batch_size"]),
            "--learning_rate", str(config["learning_rate"]),
            "--device", config["device"],
            "--dtype", config["dtype"],
            "--num_workers", str(config["num_workers"]),
            "--accumulation_steps", str(config["accumulation_steps"]),
            "--log_interval", str(config["log_interval"]),
            "--save_interval", str(config["save_interval"]),
            "--hidden_size", str(config["hidden_size"]),
            "--num_hidden_layers", str(config["num_hidden_layers"]),
            "--max_seq_len", str(config["max_seq_len"]),
            "--use_moe", str(config["use_moe"]),
            "--data_path", str(data_path),
            "--from_weight", config["from_weight"],
            "--beta", str(config.get("beta", 0.1)),
        ]
    else:
        raise ValueError(f"Unsupported trainer: {trainer}")

    print("Launching:", " ".join(command))
    subprocess.run(command, cwd=str(UPSTREAM_ROOT / "trainer"), check=True)


if __name__ == "__main__":
    main()
