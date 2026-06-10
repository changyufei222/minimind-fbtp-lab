from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


LAB_ROOT = Path(__file__).resolve().parents[1]
if str(LAB_ROOT) not in sys.path:
    sys.path.insert(0, str(LAB_ROOT))

from query_compiler.synthetic_data import build_query_compiler_true_holdout_eval  # type: ignore  # noqa: E402


EVAL_DIR = LAB_ROOT / "data" / "eval"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the v10 true holdout eval set for the FBBP query compiler")
    parser.add_argument("--seed", type=int, default=131)
    parser.add_argument("--eval-size", type=int, default=80)
    parser.add_argument("--train-size", type=int, default=600)
    parser.add_argument("--dev-size", type=int, default=40)
    parser.add_argument("--test-seen-size", type=int, default=40)
    parser.add_argument("--test-hard-size", type=int, default=40)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    output_path = EVAL_DIR / "fbbp_query_compiler_v10_true_holdout_prompts.jsonl"
    audit_path = EVAL_DIR / "fbbp_query_compiler_v10_true_holdout_audit20.md"
    manifest = build_query_compiler_true_holdout_eval(
        output_path=output_path,
        audit_path=audit_path,
        seed=args.seed,
        eval_size=args.eval_size,
        train_size=args.train_size,
        dev_size=args.dev_size,
        test_seen_size=args.test_seen_size,
        test_hard_size=args.test_hard_size,
    )
    print(
        json.dumps(
            {
                "eval_prompts": str(output_path),
                "audit_path": str(audit_path),
                "manifest": manifest,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
