# Query Compiler Baseline Matrix

This file compresses the public query-compiler evidence into one comparison table that can be read quickly by an algorithm interviewer.

## Main Comparison

| Method | Split | Plan Valid | JSON Parse | Non-empty Filter | Field/Value Exact | Slot Acc | Exec Success | Overlap@k | Repair Used |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Rule baseline | fixed eval prompts | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0 |
| Base round 1 | early remote holdout | 0.075 | n/a | n/a | n/a | 0.0234 | n/a | 0.0 | n/a |
| LoRA round 1 | early remote holdout | 0.0875 | n/a | n/a | n/a | 0.0281 | n/a | 0.0 | n/a |
| Base round 2 (JSON-first) | early remote holdout | 0.0 | n/a | n/a | n/a | 0.0 | n/a | 0.0 | n/a |
| LoRA round 2 (JSON-first) | early remote holdout | 0.1875 | n/a | n/a | n/a | 0.0578 | n/a | 0.0 | n/a |
| Baseline v23 | reserved `v15 true holdout` | 1.0 | 1.0 | 1.0 | 1.0 | 0.8141 | 1.0 | 0.28 | 1/80 rows |
| LoRA v23 | reserved `v15 true holdout` | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0/80 rows |

## Hardening Trajectory

| Version | Split | Final Perfect | First-pass Perfect | Projection Used | Repair Used | Main Interpretation |
|---|---|---:|---:|---:|---:|---|
| v18 | `v15 true holdout` | 1.0 | 0.9625 | 0.0375 | 0.0 | semantics mostly fixed, still depends on projection for a few rows |
| v19 | `v15 true holdout` | 1.0 | 0.9625 | 0.0375 | 0.0 | stable replication of v18 |
| v20 | `v15 true holdout` | 1.0 | 0.975 | 0.025 | 0.0 | projection dependency reduced |
| v21 | `v15 true holdout` | 1.0 | 0.975 | 0.025 | 0.0 | stable replication of v20 |
| v22 | `v15 true holdout` | 1.0 | 0.975 | 0.025 | 0.0 | still two projection-sensitive rows |
| v23 | `v15 true holdout` | 1.0 | 1.0 | 0.0 | 0.0 | first-pass perfect without repair or projection |

## Interview Readout

- The strongest algorithm-facing evidence is not simply “valid JSON generation”, but the jump from `baseline_v23` to `lora_v23` on the same reserved no-hints holdout.
- The main gain is semantic rather than syntactic:
  - `plan_valid_rate` and `json_parse_rate` are already saturated in `baseline_v23`
  - the real improvement is `slot_accuracy: 0.8141 -> 1.0`
  - and `result_overlap_at_k: 0.28 -> 1.0`
- The later v18-v23 sequence shows that the final result is not a one-shot lucky run. It is a staged hardening process that reduces projection dependence from `3.75%` to `0%`.

## Canonical Sources

- `FINAL_RESULT_SUMMARY.md`
- `reports/eval/query_compiler_rule_score.json`
- `reports/eval/v23_completion_audit/baseline_v23_v15_true_holdout_score_2238710.md`
- `reports/eval/v23_completion_audit/lora_v23_v15_true_holdout_score_2238710.md`
- `reports/eval/v23_completion_audit/lora_v23_v15_true_holdout_audit_2238710.json`
