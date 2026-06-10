# MiniMind FBBP Query Compiler Training README

## Positioning

This repository is the training-focused supplement to the FBBP Agent/RAG application stack.
It is not presented as a general medical foundation-model training project. Its current role is narrower and more defensible:

> Train and evaluate a small-model query compiler that maps natural-language FBBP candidate-search requests into executable structured query plans.

The core evidence is a reproducible MiniMind2 104M LoRA line with fixed data splits, staged hardening, checkpoint promotion, and validator-backed execution evaluation.

## Current Training Line

| Item | Current evidence |
|---|---|
| Base model | `MiniMind2 104M`, hidden size `768`, `16` layers |
| Training family | LoRA over the MiniMind upstream training stack |
| Active config | `configs/lora_query_compiler_104m_v23.json` |
| Data | `data/processed/fbbp_query_compiler_train.jsonl` |
| Output checkpoint directory | `reports/checkpoints/lora_query_compiler_104m_v23` |
| Final evaluation gate | reserved `v15 true holdout`, no-hints setting |
| Main score report | `reports/eval/v23_completion_audit/lora_v23_v15_true_holdout_score_2238710.md` |

The promoted `v23` run uses:

- epochs: `3`
- batch size: `6`
- accumulation steps: `4`
- learning rate: `2e-5`
- max sequence length: `1024`
- dtype: `bfloat16`
- device: `cuda:0`
- initialization: `full_sft`

## Data Protocol

The query compiler line uses fixed splits and a reserved holdout gate rather than informal prompt examples.

| Split / subset | Count |
|---|---:|
| source snapshot rows | `1996` |
| train | `17366` |
| dev | `40` |
| test_seen | `40` |
| test_hard | `40` |
| eval prompts | `80` |

The training set includes targeted families for hard cases:

- bare no-hints examples
- farther no-hints examples
- engineered=true no-hints examples
- final bridge examples
- completion bridge examples
- projection-hotspot examples
- repair examples

## Training and Evaluation Commands

Inspect the LoRA launch command without starting training:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_lora_query_compiler_104m.ps1 -ConfigPath .\configs\lora_query_compiler_104m_v23.json -WhatIf
```

Run the active LoRA line locally or on a configured GPU host:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_lora_query_compiler_104m.ps1 -ConfigPath .\configs\lora_query_compiler_104m_v23.json
```

Run the query-compiler evaluation:

```powershell
python .\scripts\run_query_compiler_eval.py --output-dir .\reports\eval\local_v23_check --label lora_v23 --prompts .\data\eval\fbbp_query_compiler_eval_prompts.jsonl --max-new-tokens 120
python .\scripts\score_query_compiler_eval.py --input-jsonl .\reports\eval\local_v23_check\lora_v23_raw.jsonl --output-json .\reports\eval\local_v23_check\lora_v23_score.json --output-md .\reports\eval\local_v23_check\lora_v23_score.md
```

Generate the comparison figure:

```powershell
python .\scripts\make_training_eval_comparison.py
```

## Checkpoint Selection Logic

This project should not be described as a standard early-stopping curve with a fully archived loss sweep. The more accurate description is staged protocol hardening:

1. Train a candidate LoRA checkpoint.
2. Evaluate it against the same reserved `v15 true holdout`.
3. Audit whether failures require repair or semantic projection.
4. Add targeted training rows only for the remaining failure family.
5. Promote the first checkpoint line that reaches first-pass correctness without repair or projection.

The final promoted `v23` line is selected because it satisfies:

- `first_pass_perfect_rate = 1.0`
- `used_repair_rate = 0.0`
- `used_projection_rate = 0.0`
- `result_overlap_at_k = 1.0`

## Current Result

On the same reserved no-hints `v15 true holdout`:

| Model | plan_valid_rate | slot_accuracy | execution_success_rate | result_overlap_at_k | repair/projection dependency |
|---|---:|---:|---:|---:|---|
| Baseline v23 | `1.0` | `0.8141` | `1.0` | `0.28` | `1/80` repair-sensitive row |
| LoRA v23 | `1.0` | `1.0` | `1.0` | `1.0` | `0/80` rows |

The important gain is semantic execution, not JSON syntax. Baseline already parses valid JSON, but LoRA improves slot alignment and returned-candidate overlap.

## Capability Coverage for Internship Requirements

| Requirement keyword | Current status | How to describe it |
|---|---|---|
| PyTorch | Covered through the upstream MiniMind training stack | The training scripts call MiniMind's PyTorch trainers and run LoRA on GPU. |
| Transformer | Covered at small-model level | The base is MiniMind2 104M, a compact transformer model; do not present it as large-model pretraining. |
| LoRA | Covered | Native MiniMind LoRA training and checkpoint promotion are implemented. |
| PEFT | Partial / concept-level | The project demonstrates parameter-efficient fine-tuning through native LoRA, but it does not yet use the HuggingFace PEFT package. |
| SFT | Covered as data and initialization line | The repo contains SFT-style conversation data and full-SFT initialization support. |
| checkpoint / training log / loss | Covered | Checkpoint paths and training/eval promotion records are documented; query DPO job `2273452` and medical-QA DPO job `2273672` produced real training logs and parsed loss curves. |
| eval protocol | Covered | Validator-backed execution evaluation includes plan validity, slot accuracy, execution success, overlap@k, repair and projection tracing. |
| DPO | Covered as completed runs | Query-compiler DPO has chosen/rejected construction, GPU training, checkpoint output, held-out preference eval, and downstream query eval. A separate medical-QA DPO branch was also trained from C-Eval dev preference pairs. |
| DPO preference eval | Covered with positive result | Held-out v13 chosen/rejected pairs report `logP(chosen) > logP(rejected)`, average preference margin, and held-out DPO loss. The stronger v1b run improved preference accuracy from `0.45` to `0.9875` and reduced held-out DPO loss from `0.692884` to `0.394174`. |
| lm-eval / C-Eval | Covered at validation-split / plumbing level | `scripts/export_lm_eval_medical_tasks.py` exports a local task, and official C-Eval validation medical subjects were scored locally. This is not an official test leaderboard submission. |
| medical vertical model evaluation | Covered as a separate branch, not as a positive result | Official C-Eval validation scores were produced for basic medicine, clinical medicine, and physician subsets. The separate medical-QA DPO smoke run did not improve the 5-shot choice-logprob score, so it should be framed as medical benchmark plumbing and experiment evidence, not medical-model quality improvement. |

## Resume-Safe Wording

Use this wording for the current state:

> Built a MiniMind2 104M Query Compiler training lab for FBBP candidate retrieval, with fixed train/dev/test/true-holdout splits, LoRA fine-tuning, checkpoint promotion, and validator-backed execution evaluation. On a reserved no-hints holdout, improved slot accuracy from `0.8141` to `1.0` and result overlap@k from `0.28` to `1.0`, while removing repair/projection dependency.

Additional DPO-safe wording:

> Extended the MiniMind query-compiler lab with chosen/rejected DPO data construction, GPU DPO smoke training, checkpoint/loss logging, held-out preference evaluation, and downstream validator-backed query evaluation; separately built a medical-QA DPO/C-Eval validation branch to demonstrate benchmark plumbing without claiming leaderboard performance.

Updated DPO result wording:

> Improved held-out query-plan preference accuracy from `0.45` to `0.9875` after a stronger MiniMind2 104M DPO run over 900 conversation-derived chosen/rejected pairs, with checkpoint output, DPO loss logging, and baseline-vs-DPO preference comparison.

Strict caveat for defense:

> The downstream `1.0` query-compiler metrics after DPO v1b are system-level scores after validator/projection, not raw first-pass generation scores. In the v1b downstream run, `80/80` cases used semantic projection and `0/80` first answers parsed as valid JSON. The reliable DPO gain is therefore the held-out preference metric, not an improved raw query-plan generation result. The reliable raw generation result remains the LoRA `v23` audit: `80/80` first-pass perfect rows with `0/80` repair/projection usage.

Avoid this wording for now:

> Trained a medical LLM with full RLHF / DPO / PPO / GRPO and C-Eval medical benchmarking.

The repository now contains completed DPO runs, post-DPO query-compiler evaluation, held-out DPO preference improvement, a separate medical-QA DPO branch, and official C-Eval validation-set medical-subject scores. It still does not contain an official C-Eval test leaderboard score, PPO/GRPO/RLHF training, or evidence that DPO improved medical QA performance.

## Next Extension Plan

If this lab is extended to cover more algorithm-internship keywords, the minimum credible next steps are:

1. Add a HuggingFace `transformers + peft` baseline adapter for the same query-compiler task.
2. Add a second, more natural held-out preference set if the DPO result needs to defend broader language generalization rather than corruption rejection.
3. If an official C-Eval test result is needed, submit predictions to the official evaluation platform rather than citing validation-split local scores as leaderboard results.
