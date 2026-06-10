# Query Compiler Algorithm Resume Summary

## What this project is

`MiniMind FBBP Query Compiler Lab` is an algorithm-facing small-model project, not merely a prompt demo.  

The core task is:

- input: natural-language candidate search request
- output: executable structured query plan over a real FBBP candidate snapshot

This makes it a task about:

- structured output learning
- semantic slot grounding
- executable plan correctness
- downstream retrieval alignment

## Why it is algorithm-relevant

The project already contains the key pieces expected by algorithm interviewers:

- fixed protocol splits:
  - train / dev / test_seen / test_hard
- reserved `true holdout`
- `no-hints` and `farther no-hints` evaluation families
- rule baseline
- base-model historical rounds
- LoRA-tuned model rounds
- repair / projection usage tracing
- execution-aligned metrics instead of only JSON formatting metrics

## Best headline evidence

The strongest current comparison is:

- `baseline_v23` on reserved `v15 true holdout`
  - `plan_valid_rate = 1.0`
  - `slot_accuracy = 0.8141`
  - `result_overlap_at_k = 0.28`
- `lora_v23` on the same reserved holdout
  - `plan_valid_rate = 1.0`
  - `slot_accuracy = 1.0`
  - `result_overlap_at_k = 1.0`
  - `used_repair_rate = 0.0`
  - `used_projection_rate = 0.0`

This is a strong algorithm story because the gain is not about syntax; it is about semantic execution fidelity.

## What was actually improved

The later hardening sequence shows a meaningful learning curve:

- earlier rounds could still depend on projection to repair residual semantic mistakes
- later rounds reduced projection-sensitive cases
- `v23` is the first stage that reaches:
  - `first_pass_perfect_rate = 1.0`
  - `used_repair_rate = 0.0`
  - `used_projection_rate = 0.0`

So the final result is:

- not “JSON looks valid”
- but “first-pass executable semantic correctness on a reserved no-hints holdout”

## Resume-ready wording

You can safely describe it as:

> Built a small-model query compiler for a private FBBP database, mapping natural-language candidate-search requests into executable structured query plans. Designed fixed train/dev/test/true-holdout evaluation with validator-backed execution, and improved the final reserved no-hints holdout from a strong structured-output baseline (`slot_accuracy 0.8141`, `overlap@k 0.28`) to a LoRA-tuned model with `100%` first-pass execution and `0` repair / `0` projection dependence.
