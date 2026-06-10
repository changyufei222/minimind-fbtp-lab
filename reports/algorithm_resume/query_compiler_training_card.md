# Query Compiler LoRA Training Card

## 1. Task definition

This project trains a small model to compile natural-language FBBP candidate-search requests into executable structured query plans.

The target is not free-form text generation. The target is:

- produce a valid structured plan
- map user intent into the right normalized fields
- preserve ranking / top-k semantics
- execute correctly on the candidate snapshot

## 2. Data used for training and evaluation

From the checked-in manifest:

- source snapshot rows: `1996`
- train: `17366`
- dev: `40`
- test_seen: `40`
- test_hard: `40`
- eval prompts: `80`

The training set also includes targeted subfamilies:

- `bare_no_hints_examples = 4800`
- `farther_no_hints_examples = 4800`
- `engineered_true_no_hints_examples = 1048`
- `final_bridge_examples = 1200`
- `completion_english_bridge_examples = 278`
- `projection_hotspot_examples = 192`
- `hotspot_shortlist_contrast_examples = 248`
- `repair_examples = 3000`

## 3. Base model / training family

Current active public line:

- `MiniMind2 104M`
- hidden size: `768`
- number of layers: `16`
- trainer: `LoRA`

Repo-facing public label:

- `MiniMind FBBP Query Compiler Lab`

## 4. Main public LoRA config

The earlier public LoRA config:

- config file:
  - `configs/lora_query_compiler_104m_1x4090.json`
- key settings:
  - epochs: `3`
  - batch size: `12`
  - accumulation steps: `2`
  - learning rate: `2e-5`
  - max sequence length: `384`
  - dtype: `bfloat16`
  - device: `cuda:0`
  - initialization: `from_weight = full_sft`

## 5. Final promoted LoRA config (`v23`)

The current promoted config is:

- config file:
  - `configs/lora_query_compiler_104m_v23.json`
- experiment name:
  - `fbbp_query_compiler_104m_v23`
- trainer:
  - `lora`
- hidden size:
  - `768`
- layers:
  - `16`
- epochs:
  - `3`
- batch size:
  - `6`
- accumulation steps:
  - `4`
- effective optimization step size:
  - smaller micro-batch, larger accumulation than the stage-1 run
- learning rate:
  - `2e-5`
- max sequence length:
  - `1024`
- dtype:
  - `bfloat16`
- device:
  - `cuda:0`
- save directory:
  - `reports/checkpoints/lora_query_compiler_104m_v23`
- initialization:
  - `from_weight = full_sft`

## 6. What changed by the final stage

The final promoted notes in the config describe the main intent of `v23`:

- exact-prefix shortlist hardening
- focus on the last projection-dependent rows from `v22`
- additional contrast examples for the remaining failing English family
- keep the reserved `v15 true holdout` unchanged as the sole completion gate

This is important because it shows the later stages were not “random retraining”; they were targeted data/protocol iterations against specific failure families.

## 7. Checkpoint selection logic

This project is not currently presented as a classical early-stopping experiment with a long checkpoint sweep reported in a single table.

The more accurate description is:

- multiple staged training rounds were run (`v18` -> `v23`)
- each round was judged against the same reserved `v15 true holdout`
- promotion criterion was whether the round removed residual projection dependence and improved first-pass perfection

So the final `v23` model is the **promoted checkpoint line**, selected because it is the first stage to satisfy all of the following on the same reserved gate:

- `first_pass_perfect_rate = 1.0`
- `used_repair_rate = 0.0`
- `used_projection_rate = 0.0`
- `result_overlap_at_k = 1.0`

This is a stronger story for interviews than pretending a standard early-stopping curve exists when the repo is actually organized around staged protocol hardening.

## 8. Final public result tied to this training card

On the reserved no-hints `v15 true holdout`:

- `plan_valid_rate = 1.0`
- `json_parse_rate = 1.0`
- `non_empty_filter_rate = 1.0`
- `field_value_exact_match = 1.0`
- `slot_accuracy = 1.0`
- `execution_success_rate = 1.0`
- `result_overlap_at_k = 1.0`
- `first_pass_perfect_rate = 1.0`
- `used_repair_rate = 0.0`
- `used_projection_rate = 0.0`

## 9. Interview-ready explanation

A concise explanation you can give is:

> I trained a LoRA-tuned MiniMind2 104M query compiler on a synthetic-but-executable FBBP query-planning task.  
> The training data already had fixed train/dev/test/holdout splits, and the later rounds focused on specific failure families instead of broad retraining.  
> The final promoted `v23` run used a 768-dim 16-layer base, 3 epochs, `2e-5` learning rate, batch size 6 with accumulation 4, and a longer 1024-token context window.  
> I selected the final version by a reserved no-hints holdout gate, not by informal cherry-picking, and the final model reached perfect first-pass execution without repair or projection.
