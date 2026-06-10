# Resume Claim Boundary, 2026-06-09

This note separates resume-safe claims from auxiliary evidence for the MiniMind FBTP lab.

## Primary resume result

Use the LoRA `v23` query-compiler audit as the main result.

Evidence:

- Evaluation: `reports/eval/v23_completion_audit/lora_v23_v15_true_holdout_audit_2238710.json`
- Held-out set: reserved `v15 true holdout`
- Rows: `80`
- `first_pass_perfect_rate`: `1.0`
- `first_pass_perfect_rows`: `80/80`
- `used_projection_rate`: `0.0`
- `used_projection_rows`: `0/80`
- `projection_attempted_rows`: `0/80`

Resume-safe wording:

> Built a MiniMind2 104M query-compiler training lab for FBBP candidate retrieval, with LoRA fine-tuning, fixed train/dev/test/true-holdout splits, checkpoint promotion, and validator-backed execution evaluation. On a reserved no-hints holdout, the promoted LoRA v23 checkpoint reached `80/80` first-pass perfect query-plan generations with `0/80` repair/projection usage.

## DPO result

Use DPO as preference-learning and training-pipeline evidence, not as raw query-generation improvement evidence.

Evidence:

- Training config: `configs/dpo_query_compiler_104m_stronger_v1b.json`
- Preference data: `data/processed/fbbp_query_compiler_dpo_stronger_v1b.jsonl`
- Loss curve: `reports/algorithm_resume/dpo_training_loss_2273689_stronger_v1b.png`
- Preference eval: `reports/eval/dpo_preference_2273690/dpo_query_compiler_104m_stronger_v1b/summary.md`

Observed result:

- Baseline preference accuracy: `0.45`
- DPO v1b preference accuracy: `0.9875`
- Baseline average preference margin: `0.00544`
- DPO v1b average preference margin: `9.728064`
- Baseline held-out DPO loss: `0.692884`
- DPO v1b held-out DPO loss: `0.394174`

Resume-safe wording:

> Extended the query-compiler lab with chosen/rejected DPO data construction and GPU DPO training; on held-out synthetic preference pairs, improved `logP(chosen) > logP(rejected)` accuracy from `0.45` to `0.9875`, with checkpoint output and parsed DPO loss curve.

Important boundary:

- The DPO v1b downstream query-compiler run is not raw generation evidence.
- In that downstream run, `0/80` first answers parsed as valid JSON.
- `80/80` cases used semantic projection.
- Therefore, DPO v1b did not demonstrate improved raw first-pass query-plan generation.

Do not write:

> DPO improved raw query-plan generation to 100%.

## C-Eval and medical branch

C-Eval should stay as auxiliary benchmark plumbing, not as the main result.

Evidence:

- Query-DPO C-Eval validation scoring was run locally on official C-Eval validation medical subjects.
- A separate medical-QA DPO branch was trained from C-Eval dev preference pairs.
- The medical-QA DPO validation result did not improve over baseline.

Resume-safe wording:

> Built a separate medical-QA DPO/C-Eval validation branch to demonstrate medical preference-data construction, GPU DPO training, checkpoint/loss logging, and local validation-split benchmark scoring.

Do not write:

> DPO improved medical C-Eval performance.

> Official C-Eval leaderboard result.

## Interview defense

If asked why many downstream metrics are `1.0`, answer directly:

> There are two evaluation layers. The LoRA v23 result is raw first-pass generation and is the main claim. The DPO v1b downstream `1.0` result is system-level after validator/projection, so I do not use it as raw generation evidence. For DPO, I only claim held-out preference ranking improvement.
