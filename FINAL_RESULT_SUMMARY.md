# Final Result Summary

Updated on `2026-05-05`.

This file is the official summary for the public-facing project:

- **MiniMind FBBP Query Compiler Lab**

Repository directory retained for compatibility:

- `minimind-fbtp-lab`

The repo evolved from earlier evidence-card formatting work toward the FBBP query-compiler line.

## Current Positioning

The project is now positioned as:

- a small-model LoRA lab for FBBP natural-language query compilation
- plus validator-backed execution over the real FBBP candidate-card lineage

## Official Status

- Status: query-compiler line completed for this stage
- Active line: `MiniMind FBBP query compiler`
- Historical frozen line retained: Round 3 / Round 3.1 evidence-card extraction results

## Current Evidence Chain

The current stage has already landed:

- candidate snapshot generation from the checked-in FBBP card line
- a first-stage query DSL and validator
- a snapshot executor for candidate filtering and top-k sorting
- synthetic train/dev/test/test_hard query-compiler datasets
- a rule baseline over the fixed evaluation prompts
- a thin runnable demo

Current reproducible commands:

- snapshot build:
  - `python .\scripts\build_fbbp_candidate_snapshot.py`
- dataset build:
  - `python .\scripts\build_fbbp_query_compiler_dataset.py`
- rule baseline:
  - `python .\scripts\run_query_compiler_rule_baseline.py`
- scoring:
  - `python .\scripts\score_query_compiler_eval.py --input-jsonl .\reports\eval\query_compiler_rule_eval.jsonl --output-json .\reports\eval\query_compiler_rule_score.json --output-md .\reports\eval\query_compiler_rule_score.md`
- thin demo:
  - `python .\scripts\demo_query_compiler.py --query "帮我筛一批 engineered 的 kunitz 候选，口服别太差，优先亲和力更强的"`

## Query-Compiler Evolution

The bootstrap rule baseline on the fixed synthetic evaluation set is still strong:

- `plan_valid_rate = 1.0`
- `json_parse_rate = 1.0`
- `non_empty_filter_rate = 1.0`
- `field_value_exact_match = 1.0`
- `slot_accuracy = 1.0`
- `execution_success_rate = 1.0`
- `result_overlap_at_k = 1.0`

The first two remote MiniMind comparisons established the actual difficulty of the task:

- **Round 1**
  - `base`: `plan_valid_rate = 0.075`, `slot_accuracy = 0.0234`, `result_overlap_at_k = 0.0`
  - `lora`: `plan_valid_rate = 0.0875`, `slot_accuracy = 0.0281`, `result_overlap_at_k = 0.0`
- **Round 2 (`JSON-first`)**
  - `base v2`: all major metrics remained `0.0`
  - `lora v2`: `plan_valid_rate = 0.1875`, `slot_accuracy = 0.0578`, `result_overlap_at_k = 0.0`

That moved the repo past "can the pipeline run" and into the real compiler problem:

- force valid JSON
- keep filters non-empty
- predict the right field-value pairs
- finally move `result_overlap_at_k` above zero

The later hardening sequence on the reserved `v15 true holdout` was:

- **v18**
  - `final_perfect_rate = 1.0`
  - `first_pass_perfect_rate = 0.9625`
  - `used_projection_rate = 0.0375`
- **v19**
  - `final_perfect_rate = 1.0`
  - `first_pass_perfect_rate = 0.9625`
  - `used_projection_rate = 0.0375`
- **v20**
  - `final_perfect_rate = 1.0`
  - `first_pass_perfect_rate = 0.975`
  - `used_projection_rate = 0.025`
- **v21**
  - `final_perfect_rate = 1.0`
  - `first_pass_perfect_rate = 0.975`
  - `used_projection_rate = 0.025`
- **v22**
  - `final_perfect_rate = 1.0`
  - `first_pass_perfect_rate = 0.975`
  - `used_projection_rate = 0.025`
- **v23**
  - `final_perfect_rate = 1.0`
  - `first_pass_perfect_rate = 1.0`
  - `used_repair_rate = 0.0`
  - `used_projection_rate = 0.0`

## Final Query-Compiler Result

The final active result for this repo is `v23` on the reserved no-hints `v15 true holdout`.

Score-side metrics:

- `plan_valid_rate = 1.0`
- `json_parse_rate = 1.0`
- `non_empty_filter_rate = 1.0`
- `field_value_exact_match = 1.0`
- `slot_accuracy = 1.0`
- `execution_success_rate = 1.0`
- `result_overlap_at_k = 1.0`

Completion-critical metrics:

- `rows = 80`
- `final_perfect_rate = 1.0`
- `first_schema_ok_rate = 1.0`
- `first_pass_perfect_rate = 1.0`
- `repair_attempted_rate = 0.0`
- `used_repair_rate = 0.0`
- `projection_attempted_rate = 0.0`
- `used_projection_rate = 0.0`

This is the first point in the repo where we can honestly claim:

- the model is not merely producing valid JSON
- the model is not depending on repair
- the model is not depending on projection
- the model is achieving first-pass semantic grounding on a reserved no-hints true holdout family

## Interpretation Boundary

The final `1.0` metrics should be interpreted as a protocol-complete result, not as an unlimited external generalization claim.

- The database entity layer is real and derived from the checked-in FBBP candidate-card lineage.
- The natural-language supervision and much of the benchmark prompting are still programmatically constructed.
- The right external claim is:
  - the repo demonstrates a strong and reproducible result on a reserved no-hints holdout inside the current protocol
  - the repo does not by itself prove universal robustness across arbitrary future human query language

## Final Evidence

- Final score summary:
  - `reports/eval/v23_completion_audit/lora_v23_v15_true_holdout_score_2238710.md`
- Final completion audit:
  - `reports/eval/v23_completion_audit/lora_v23_v15_true_holdout_audit_2238710.json`
- Baseline reference:
  - `reports/eval/v23_completion_audit/baseline_v23_v15_true_holdout_score_2238710.md`

## Historical Archive

The previous frozen evidence-card line still matters as background:

- Round 3 positive result
- Round 3.1 failed refinement
- historical summary report: `reports/final_experiment_summary.md`

Use that line as archive context, not as the main active project story.

## Canonical Artifacts

- `data/processed/fbbp_candidate_snapshot.jsonl`
- `data/processed/fbbp_query_compiler_train.jsonl`
- `data/processed/fbbp_query_compiler_dev.jsonl`
- `data/processed/fbbp_query_compiler_test_seen.jsonl`
- `data/processed/fbbp_query_compiler_test_hard.jsonl`
- `data/eval/fbbp_query_compiler_eval_prompts.jsonl`
- `reports/eval/query_compiler_rule_score.md`
- `reports/eval/local_smoke_v3/smoke_v3_score.md`
- `docs/superpowers/specs/2026-04-20-minimind-fbbp-query-compiler-design.md`
- `docs/superpowers/specs/2026-04-26-minimind-fbbp-query-compiler-v3-design.md`

## Resume Direction

The preferred active resume framing is now:

- built a small-model query compiler for a private FBBP database, compiling natural-language candidate-search requests into executable structured query plans
- designed a validator-backed execution and evaluation loop with fixed no-hints true holdout testing, repair/projection tracing, and failure-mode-driven data hardening
- drove the final `v23` model to `100%` first-pass perfection on the reserved no-hints true holdout under the current evaluation protocol, with `0` repair usage and `0` projection usage
