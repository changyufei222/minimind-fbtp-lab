from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


LAB_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = LAB_ROOT / "reports" / "algorithm_resume"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    metrics = ["Plan valid", "Slot acc", "Exec success", "Overlap@k"]
    baseline = [1.0, 0.8141, 1.0, 0.28]
    lora = [1.0, 1.0, 1.0, 1.0]

    versions = ["v18", "v19", "v20", "v21", "v22", "v23"]
    first_pass = [0.9625, 0.9625, 0.975, 0.975, 0.975, 1.0]
    projection_used = [0.0375, 0.0375, 0.025, 0.025, 0.025, 0.0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=180)
    fig.patch.set_facecolor("white")

    x = range(len(metrics))
    width = 0.36
    axes[0].bar([i - width / 2 for i in x], baseline, width, label="Baseline v23", color="#8aa1b1")
    axes[0].bar([i + width / 2 for i in x], lora, width, label="LoRA v23", color="#2f7d6d")
    axes[0].set_title("Reserved v15 True Holdout Eval")
    axes[0].set_ylim(0, 1.08)
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(metrics, rotation=20, ha="right")
    axes[0].set_ylabel("Score")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)

    for idx, value in enumerate(baseline):
        axes[0].text(idx - width / 2, value + 0.025, f"{value:.2f}" if value != 0.8141 else "0.814", ha="center", va="bottom", fontsize=8)
    for idx, value in enumerate(lora):
        axes[0].text(idx + width / 2, value + 0.025, f"{value:.1f}", ha="center", va="bottom", fontsize=8)

    axes[1].plot(versions, first_pass, marker="o", linewidth=2.2, label="First-pass perfect", color="#2f7d6d")
    axes[1].plot(versions, projection_used, marker="o", linewidth=2.2, label="Projection used", color="#b65f4a")
    axes[1].set_title("Hardening Trajectory")
    axes[1].set_ylim(-0.03, 1.08)
    axes[1].set_ylabel("Rate")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False, loc="center right")

    fig.suptitle("MiniMind FBBP Query Compiler: Checkpoint and Eval Comparison", fontsize=13, fontweight="bold")
    fig.text(
        0.01,
        0.01,
        "Note: local training-loss logs are not fully archived in this repo; this figure shows checkpoint/eval evidence and the staged hardening gate.",
        fontsize=8,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))

    png_path = OUT_DIR / "query_compiler_training_eval_comparison.png"
    svg_path = OUT_DIR / "query_compiler_training_eval_comparison.svg"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    print(png_path)
    print(svg_path)


if __name__ == "__main__":
    main()

