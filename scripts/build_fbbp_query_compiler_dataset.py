from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


LAB_ROOT = Path(__file__).resolve().parents[1]
if str(LAB_ROOT) not in sys.path:
    sys.path.insert(0, str(LAB_ROOT))

from query_compiler.synthetic_data import build_query_compiler_dataset  # type: ignore  # noqa: E402
from query_compiler.synthetic_data import build_query_compiler_v12_true_holdout_eval  # type: ignore  # noqa: E402
from query_compiler.synthetic_data import build_query_compiler_v13_true_holdout_eval  # type: ignore  # noqa: E402
from query_compiler.synthetic_data import build_query_compiler_v15_true_holdout_eval  # type: ignore  # noqa: E402


OUTPUT_DIR = LAB_ROOT / "data" / "processed"
EVAL_DIR = LAB_ROOT / "data" / "eval"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the FBBP query compiler dataset")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--train-size", type=int, default=600)
    parser.add_argument("--train-variants-per-row", type=int, default=3)
    parser.add_argument("--repair-examples-per-row", type=int, default=1)
    parser.add_argument("--bare-no-hints-examples-per-row", type=int, default=0)
    parser.add_argument("--farther-no-hints-examples-per-row", type=int, default=0)
    parser.add_argument("--engineered-true-no-hints-examples-per-row", type=int, default=0)
    parser.add_argument("--final-bridge-examples-per-row", type=int, default=0)
    parser.add_argument("--completion-english-bridge-examples-per-row", type=int, default=0)
    parser.add_argument("--projection-hotspot-examples-per-row", type=int, default=0)
    parser.add_argument("--hotspot-shortlist-contrast-examples-per-row", type=int, default=0)
    parser.add_argument("--schema-word-examples-per-row", type=int, default=0)
    parser.add_argument("--dev-size", type=int, default=40)
    parser.add_argument("--test-seen-size", type=int, default=40)
    parser.add_argument("--test-hard-size", type=int, default=40)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_query_compiler_dataset(
        output_dir=OUTPUT_DIR,
        seed=args.seed,
        train_size=args.train_size,
        train_variants_per_row=args.train_variants_per_row,
        repair_examples_per_row=args.repair_examples_per_row,
        bare_no_hints_examples_per_row=args.bare_no_hints_examples_per_row,
        farther_no_hints_examples_per_row=args.farther_no_hints_examples_per_row,
        engineered_true_no_hints_examples_per_row=args.engineered_true_no_hints_examples_per_row,
        final_bridge_examples_per_row=args.final_bridge_examples_per_row,
        completion_english_bridge_examples_per_row=args.completion_english_bridge_examples_per_row,
        projection_hotspot_examples_per_row=args.projection_hotspot_examples_per_row,
        hotspot_shortlist_contrast_examples_per_row=args.hotspot_shortlist_contrast_examples_per_row,
        schema_word_examples_per_row=args.schema_word_examples_per_row,
        dev_size=args.dev_size,
        test_seen_size=args.test_seen_size,
        test_hard_size=args.test_hard_size,
    )
    eval_source = OUTPUT_DIR / "fbbp_query_compiler_eval_prompts.jsonl"
    eval_target = EVAL_DIR / "fbbp_query_compiler_eval_prompts.jsonl"
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    eval_target.write_text(eval_source.read_text(encoding="utf-8"), encoding="utf-8")
    v12_eval_target = EVAL_DIR / "fbbp_query_compiler_v12_true_holdout_prompts.jsonl"
    v12_audit_target = EVAL_DIR / "fbbp_query_compiler_v12_true_holdout_audit20.md"
    v12_holdout = build_query_compiler_v12_true_holdout_eval(
        output_path=v12_eval_target,
        audit_path=v12_audit_target,
        train_size=args.train_size,
        dev_size=args.dev_size,
        test_seen_size=args.test_seen_size,
        test_hard_size=args.test_hard_size,
    )
    v13_eval_target = EVAL_DIR / "fbbp_query_compiler_v13_true_holdout_prompts.jsonl"
    v13_audit_target = EVAL_DIR / "fbbp_query_compiler_v13_true_holdout_audit20.md"
    v13_holdout = build_query_compiler_v13_true_holdout_eval(
        output_path=v13_eval_target,
        audit_path=v13_audit_target,
        train_size=args.train_size,
        dev_size=args.dev_size,
        test_seen_size=args.test_seen_size,
        test_hard_size=args.test_hard_size,
    )
    v15_eval_target = EVAL_DIR / "fbbp_query_compiler_v15_true_holdout_prompts.jsonl"
    v15_audit_target = EVAL_DIR / "fbbp_query_compiler_v15_true_holdout_audit20.md"
    v15_holdout = build_query_compiler_v15_true_holdout_eval(
        output_path=v15_eval_target,
        audit_path=v15_audit_target,
        train_size=args.train_size,
        dev_size=args.dev_size,
        test_seen_size=args.test_seen_size,
        test_hard_size=args.test_hard_size,
    )
    print(
        json.dumps(
            {
                "output_dir": str(OUTPUT_DIR),
                "eval_prompts": str(eval_target),
                "v12_true_holdout_prompts": str(v12_eval_target),
                "v13_true_holdout_prompts": str(v13_eval_target),
                "v15_true_holdout_prompts": str(v15_eval_target),
                "manifest": manifest,
                "v12_true_holdout": v12_holdout,
                "v13_true_holdout": v13_holdout,
                "v15_true_holdout": v15_holdout,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
