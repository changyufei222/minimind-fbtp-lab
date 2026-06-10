from __future__ import annotations

import argparse
import sys
from pathlib import Path


LAB_ROOT = Path(__file__).resolve().parents[1]
if str(LAB_ROOT) not in sys.path:
    sys.path.insert(0, str(LAB_ROOT))

from query_compiler.candidate_snapshot import build_candidate_snapshot  # type: ignore  # noqa: E402
from query_compiler.demo import render_demo  # type: ignore  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the FBBP query compiler thin demo")
    parser.add_argument("--query", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    snapshot_rows = build_candidate_snapshot()
    print(render_demo(args.query, snapshot_rows))


if __name__ == "__main__":
    main()
