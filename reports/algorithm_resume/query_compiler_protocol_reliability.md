# Query Compiler Protocol and Reliability Note

## 1. Why this note exists

The query-compiler result is strong enough that it needs a separate reliability explanation.  

Without this note, a reviewer may reasonably ask:

- how the holdout was constructed
- whether the test prompts leaked into training
- whether the final `1.0` result is only a formatting trick
- whether the result implies unlimited external generalization

This note answers those questions in a compact, interview-friendly way.

## 2. Dataset and split structure

The checked-in query-compiler manifest currently reports:

- `train = 17366`
- `dev = 40`
- `test_seen = 40`
- `test_hard = 40`
- `eval_prompts = 80`
- source snapshot rows: `1996`

So this is already a fixed train/dev/test-style protocol rather than an ad hoc prompt demo.

## 3. What the reserved holdout actually is

The strongest current result is reported on the reserved `v15 true holdout`.

Important properties of this holdout:

- it is a dedicated evaluation family, separate from the main train/dev/test partitions
- it includes **no-hints** prompt variants
- later rows are explicitly marked as **farther no-hints** prompt families reserved for robustness validation
- the public audit file records:
  - `seed = 241`
  - `eval_size = 80`
  - `holdout_offset = 720`

This means the final score is not based on the same surface prompt templates used during ordinary training loops.

## 4. Why the result is not just formatting

The project does not treat “valid JSON” as the final success condition.

The public metric set includes:

- `plan_valid_rate`
- `json_parse_rate`
- `non_empty_filter_rate`
- `field_value_exact_match`
- `slot_accuracy`
- `execution_success_rate`
- `result_overlap_at_k`

This matters because a model can:

- emit valid JSON
- pass schema checks
- still return the wrong candidate set

That is exactly what the later public baseline shows:

- `baseline_v23`
  - `plan_valid_rate = 1.0`
  - `json_parse_rate = 1.0`
  - `slot_accuracy = 0.8141`
  - `result_overlap_at_k = 0.28`

So the protocol is not fooled by formatting correctness alone.

## 5. Why we can talk about leakage resistance

The strongest credibility signals already present in the repo are:

1. the final claim is tied to a **reserved** holdout rather than the training split
2. the final promoted version explicitly says the holdout remained unchanged as the completion gate
3. the v23 notes state that the exact-prefix shortlist hardening was added **without reusing the held-out sentence endings**
4. later holdout rows are described as `farther no-hints` prompt families, which are intentionally more distant from the ordinary templated phrasing

This does not prove absolute zero leakage in a formal paper sense, but it is a much stronger credibility story than:

- “we tuned until a random dev set looked good”

## 6. What the final `1.0` does and does not mean

The correct interpretation of the final `lora_v23` result is:

- within the current frozen protocol
- on the reserved no-hints true holdout
- under validator-backed executable-plan scoring
- the model achieves first-pass semantic correctness

More concretely:

- `plan_valid_rate = 1.0`
- `slot_accuracy = 1.0`
- `execution_success_rate = 1.0`
- `result_overlap_at_k = 1.0`
- `first_pass_perfect_rate = 1.0`
- `used_repair_rate = 0.0`
- `used_projection_rate = 0.0`

What this **does not** automatically mean:

- unlimited robustness to arbitrary future human phrasing
- guaranteed cross-domain transfer outside the current FBBP candidate-query protocol
- elimination of all possible prompt-distribution shift

That boundary should be stated explicitly in interviews.

## 7. Recommended interview wording

A safe and strong wording is:

> We froze a reserved no-hints true holdout and used it as the sole promotion gate.  
> The final v23 LoRA model reached perfect first-pass plan execution on that holdout without relying on repair or projection.  
> I would describe this as protocol-complete performance inside the current query-compiler benchmark, not as proof of unlimited open-world generalization.

## 8. Why this is algorithm-meaningful

This protocol is algorithm-relevant because it separates:

- surface structured-output correctness
- semantic slot correctness
- executable retrieval correctness
- auxiliary repair dependence

That is much closer to a real model-evaluation protocol than a one-shot prompt demo.
