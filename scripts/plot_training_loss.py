from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt


LAB_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GLOB = "reports/**/*.log"
DEFAULT_CSV = LAB_ROOT / "reports" / "algorithm_resume" / "training_loss_curve.csv"
DEFAULT_PNG = LAB_ROOT / "reports" / "algorithm_resume" / "training_loss_curve.png"
DEFAULT_MD = LAB_ROOT / "reports" / "algorithm_resume" / "training_loss_curve_report.md"

LOSS_RE = re.compile(
    r"Epoc<local_path_removed><epoch>\d+)/(?P<epochs>\d+)\]\((?P<step>\d+)/(?P<steps>\d+)\),\s+"
    r"(?:(?:los<local_path_removed><loss>[0-9.]+).*?(?:lr|learning_rate):\s+(?P<lr>[0-9.eE-]+))|"
    r"(?:Actor Los<local_path_removed><actor_loss>[0-9.]+).*?Rewar<local_path_removed><reward>-?[0-9.]+)))"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse MiniMind training logs and plot loss curves")
    parser.add_argument("--glob", default=DEFAULT_GLOB)
    parser.add_argument("--output-csv", default=str(DEFAULT_CSV))
    parser.add_argument("--output-png", default=str(DEFAULT_PNG))
    parser.add_argument("--output-md", default=str(DEFAULT_MD))
    return parser.parse_args()


def parse_logs(pattern: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in LAB_ROOT.glob(pattern):
        text = path.read_text(encoding="utf-8", errors="ignore")
        for match in LOSS_RE.finditer(text):
            data = match.groupdict()
            rows.append(
                {
                    "source": str(path.relative_to(LAB_ROOT)),
                    "epoch": data["epoch"] or "",
                    "step": data["step"] or "",
                    "loss": data.get("loss") or data.get("actor_loss") or "",
                    "lr": data.get("lr") or "",
                    "reward": data.get("reward") or "",
                }
            )
    return rows


def write_outputs(rows: list[dict[str, str]], csv_path: Path, png_path: Path, md_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source", "epoch", "step", "loss", "lr", "reward"])
        writer.writeheader()
        writer.writerows(rows)
    if rows:
        y = [float(row["loss"]) for row in rows if row["loss"]]
        x = list(range(1, len(y) + 1))
        fig, ax = plt.subplots(figsize=(7, 4), dpi=180)
        ax.plot(x, y, marker="o", linewidth=1.8, color="#2f7d6d")
        ax.set_title("MiniMind Training Loss")
        ax.set_xlabel("Logged point")
        ax.set_ylabel("Loss")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(png_path, bbox_inches="tight")
        status = "parsed"
    else:
        fig, ax = plt.subplots(figsize=(7, 4), dpi=180)
        ax.text(0.5, 0.55, "No local loss log entries found", ha="center", va="center", fontsize=14)
        ax.text(0.5, 0.42, "Run training or pull cluster logs, then rerun scripts/plot_training_loss.py", ha="center", va="center", fontsize=9)
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(png_path, bbox_inches="tight")
        status = "no_local_loss_logs"
    md_path.write_text(
        "\n".join(
            [
                "# Training Loss Curve Report",
                "",
                f"- status: `{status}`",
                f"- parsed rows: `{len(rows)}`",
                f"- csv: `{csv_path}`",
                f"- figure: `{png_path}`",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    rows = parse_logs(args.glob)
    write_outputs(rows, Path(args.output_csv), Path(args.output_png), Path(args.output_md))
    print(json.dumps({"rows": len(rows), "csv": args.output_csv, "png": args.output_png, "md": args.output_md}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
