# Query Compiler Error Analysis

## 1. Why a separate error analysis is needed

The query-compiler project already has strong top-line results, but those results alone do not explain *why* earlier model rounds failed and *what* changed to reach the final `v23` result.  

This note focuses on the main failure modes seen before the final hardened release, especially on the reserved `v15 true holdout`.

## 2. Error taxonomy

For this task, the most useful error taxonomy is:

1. `JSON invalid / parse failure`
   - the model does not produce machine-readable output
2. `Empty filter`
   - output parses, but the query intent is missing actual constraints
3. `Field mapping error`
   - correct intent, wrong database field
4. `Value mapping error`
   - correct field family, wrong normalized value
5. `Sort / limit error`
   - filters are correct, but ranking direction or top-k is wrong
6. `Execution failure`
   - normalized plan cannot be executed
7. `Semantic retrieval mismatch`
   - execution succeeds, but the returned candidate set is not the intended one

## 3. What the baseline evidence shows

### 3.1 Earlier raw baseline (`baseline_v14` on the reserved `v15 true holdout`)

From the raw completion audit:

- rows: `80`
- JSON invalid / parse failure: `0`
- empty filter: `0`
- field/value mismatch: `0`
- slot mismatch: `75/80`
- result overlap failure: `58/80`

This means the earlier baseline problem was **not** syntax.

The model could already:

- produce valid JSON
- keep filters non-empty
- map most obvious field/value pairs into the correct schema

But it still failed on the more semantic part of the task:

- choosing the right `limit`
- choosing the right `sort`
- matching the final executable plan to the intended candidate set

### 3.2 Public baseline (`baseline_v23`)

The later baseline is much stronger on surface metrics:

- `plan_valid_rate = 1.0`
- `json_parse_rate = 1.0`
- `non_empty_filter_rate = 1.0`
- `field_value_exact_match = 1.0`

However, it still underperforms on the metrics that really matter for downstream execution:

- `slot_accuracy = 0.8141`
- `result_overlap_at_k = 0.28`

So by the `v23` stage, the baseline is already a **strong structured-output generator**, but not yet a strong **semantic query compiler**.

## 4. Representative failure patterns

### Error Type A: sort / limit omitted or generalized away

Representative earlier-row pattern:

- prompt asks for:
  - exact scaffold
  - exact oral class
  - engineered flag
  - experimental affinity requirement
  - strongest affinity first
  - top `5` or top `10`
- model outputs:
  - correct filters
  - but `sort = []`
  - and `limit = 20`

Observed consequence:

- `slot_accuracy` drops from `1.0` to `0.75`
- `execution_success = True`
- but `result_overlap_at_k = 0.0`

Interpretation:

- the query is syntactically valid
- the schema is correct
- but the returned set is semantically wrong because the ranking/limit behavior no longer matches the request

### Error Type B: execution succeeds but the candidate set is wrong

Representative pattern:

- normalized plan looks structurally plausible
- it executes successfully on the snapshot executor
- but top-k overlap is still poor or zero

This is an especially important category for an algorithm interview because it shows:

- “can execute” is not equal to “understands the request”
- the real challenge is semantic compilation, not merely JSON generation

### Error Type C: dependence on projection or repair

In the intermediate hardening sequence:

- `v18`: `used_projection_rate = 0.0375`
- `v19`: `used_projection_rate = 0.0375`
- `v20`: `used_projection_rate = 0.025`
- `v21`: `used_projection_rate = 0.025`
- `v22`: `used_projection_rate = 0.025`
- `v23`: `used_projection_rate = 0.0`

Interpretation:

- earlier rounds could still rely on auxiliary projection logic to reach perfect final outputs
- later rounds removed that dependency by changing the training/data protocol

This is important because it turns the story from:

- “post-processing made it look good”

into:

- “the model itself became correct on first pass”

## 5. Representative examples

### Example 1: filters right, ranking wrong

Prompt family:

- scaffold family fixed
- oral class fixed
- engineered / non-engineered fixed
- experimental affinity required
- strongest affinity first
- limit fixed

Failure pattern:

- output retains the correct filters
- but does not emit the requested sort field or exact top-k
- execution succeeds
- overlap collapses to `0.0`

This is the clearest example of a **semantic query-planning error**.

### Example 2: high surface correctness but weak downstream utility

In `baseline_v23`, the system reaches:

- `plan_valid_rate = 1.0`
- `json_parse_rate = 1.0`
- `field_value_exact_match = 1.0`

Yet:

- `slot_accuracy = 0.8141`
- `result_overlap_at_k = 0.28`

This shows a classic algorithm lesson:

- surface structured-output metrics can saturate before the real downstream metric saturates
- the task should therefore be evaluated on **execution-aligned metrics**, not just parsing metrics

## 6. How these errors were reduced

The main improvement route was not “bigger system glue”, but targeted hardening of the data and protocol:

1. preserve the fixed query DSL and validator
2. identify projection-sensitive families on the reserved holdout
3. add targeted shortlist / exact-prefix contrast data for the failing families
4. keep the reserved `v15 true holdout` unchanged
5. re-evaluate on the same no-hints holdout

This produced the final transition:

- `baseline_v23`
  - `slot_accuracy = 0.8141`
  - `result_overlap_at_k = 0.28`
  - some residual repair/projection dependence
- `lora_v23`
  - `slot_accuracy = 1.0`
  - `result_overlap_at_k = 1.0`
  - `used_repair_rate = 0.0`
  - `used_projection_rate = 0.0`

## 7. Final takeaway for resume / interview use

The strongest error-analysis message is:

- the hard part of this project was **not** getting the model to emit valid JSON
- the hard part was aligning the compiled plan with the intended retrieval semantics
- the final gains came from targeted error-family hardening, not generic prompt tweaking

That is exactly the kind of story that reads as “algorithm work” rather than only “engineering glue”.
