# DPO, Preference Eval, and C-Eval Remote Run Evidence

## Remote jobs

- DPO training job: `2273452`, state `COMPLETED`, elapsed `00:02:47`, exit code `0:0`
- Query-compiler evaluation job: `2273453`, state `COMPLETED`, elapsed `00:12:57`, exit code `0:0`
- Official C-Eval medical validation job: `2273658`, state `COMPLETED`, elapsed `00:04:04`, exit code `0:0`
- Improved C-Eval scoring job: `2273668`, state `COMPLETED`, elapsed `00:01:29`, exit code `0:0`
- DPO preference evaluation job: `2273671`, state `COMPLETED`, elapsed `00:02:17`, exit code `0:0`
- Medical-QA DPO training job: `2273672`, state `COMPLETED`, elapsed `00:02:19`, exit code `0:0`
- Medical-QA DPO C-Eval validation job: `2273673`, state `COMPLETED`, elapsed `00:01:27`, exit code `0:0`
- Stronger query-compiler DPO v1 training job: `2273682`, state `COMPLETED`, elapsed `00:25:42`, exit code `0:0`
- Stronger query-compiler DPO v1 preference eval job: `2273686`, state `COMPLETED`, elapsed `00:01:44`, exit code `0:0`
- Stronger query-compiler DPO v1b training job: `2273689`, state `COMPLETED`, elapsed `00:06:29`, exit code `0:0`
- Stronger query-compiler DPO v1b preference eval job: `2273690`, state `COMPLETED`, elapsed `00:02:11`, exit code `0:0`
- Stronger query-compiler DPO v1b downstream eval job: `2273692`, state `COMPLETED`, elapsed `00:11:34`, exit code `0:0`
- Remote checkpoint: `/data/run01/scv7sd2/minimind-fbtp-lab/reports/checkpoints/dpo_query_compiler_104m_smoke/fbbp_query_compiler_dpo_104m_smoke_768.pth`
- Stronger v1b checkpoint: `/data/run01/scv7sd2/minimind-fbtp-lab/reports/checkpoints/dpo_query_compiler_104m_stronger_v1b/fbbp_query_compiler_dpo_104m_stronger_v1b_768.pth`
- Medical-QA DPO checkpoint: `/data/run01/scv7sd2/minimind-fbtp-lab/reports/checkpoints/dpo_medical_ceval_104m_smoke/medical_ceval_dpo_104m_smoke_768.pth`

## DPO training

- Data: `data/processed/fbbp_query_compiler_dpo_smoke.jsonl`
- Preference pairs: `80`
- Model: MiniMind2 104M, hidden size `768`, layers `16`
- Training: `1` epoch, `40` steps, batch size `2`, accumulation `4`
- Loss log: `reports/logs/dpo_query_compiler_104m_2273452.log`
- Parsed loss curve: `reports/algorithm_resume/dpo_training_loss_2273452.png`

Observed DPO loss checkpoints:

| step | loss |
|---:|---:|
| 5 | 0.6921 |
| 10 | 0.6824 |
| 15 | 0.6714 |
| 20 | 0.6624 |
| 25 | 0.6907 |
| 30 | 0.6602 |
| 35 | 0.6908 |
| 39 | 0.6896 |
| 40 | 0.6895 |

## Query-compiler eval after DPO

Baseline full-SFT and DPO checkpoint were both evaluated on the same query-compiler holdout.

| model | plan_valid | json_parse | slot_accuracy | execution_success | overlap@k |
|---|---:|---:|---:|---:|---:|
| baseline_full_sft | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| dpo_query_compiler_104m_smoke | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |

Evidence files:

- `reports/eval/baseline_full_sft_query_compiler_2273453/baseline_full_sft_score.md`
- `reports/eval/dpo_query_compiler_104m_smoke_query_compiler_2273453/dpo_query_compiler_104m_smoke_score.md`

## Held-out DPO preference eval

This is the primary DPO-specific evaluation for the FBBP query-compiler branch. It scores held-out chosen/rejected pairs from `data/processed/fbbp_query_compiler_dpo_heldout_v13.jsonl` and reports whether the model assigns higher mean assistant-token log probability to the chosen plan.

| model | pairs | preference_accuracy | chosen_wins | avg_preference_margin | avg_heldout_dpo_loss |
|---|---:|---:|---:|---:|---:|
| baseline_full_sft | 80 | 0.4875 | 39 | 0.008489 | 0.692725 |
| dpo_query_compiler_104m_smoke | 80 | 0.4875 | 39 | 0.008539 | 0.692722 |

Interpretation: the first DPO smoke run completed and the evaluation was implemented, but the small run did not provide evidence of a meaningful preference-learning gain. It was kept as wiring evidence and followed by stronger DPO retries.

Evidence files:

- `reports/eval/dpo_preference_2273671/baseline_full_sft/summary.md`
- `reports/eval/dpo_preference_2273671/dpo_query_compiler_104m_smoke/summary.md`
- `reports/eval/dpo_preference_2273671/baseline_full_sft/rows.jsonl`
- `reports/eval/dpo_preference_2273671/dpo_query_compiler_104m_smoke/rows.jsonl`

## Stronger query-compiler DPO retries

The no-improvement smoke result was followed by larger conversation-derived DPO training sets and stronger learning rates. These are still query-compiler DPO experiments, not medical QA DPO results.

| run | train pairs | epochs | learning_rate | held-out pairs | preference_accuracy | avg_preference_margin | avg_heldout_dpo_loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline_full_sft | 0 | 0 | n/a | 80 | 0.45 | 0.005440 | 0.692884 |
| stronger_v1 | 2400 | 2 | 2e-6 | 80 | 0.7125 | 1.556623 | 0.626922 |
| stronger_v1b | 900 | 1 | 3e-6 | 80 | 0.9875 | 9.728064 | 0.394174 |

The best current result is `stronger_v1b`: `79/80` held-out chosen/rejected pairs assigned higher probability to the chosen query plan. The DPO loss curve for v1b shows the training objective falling from roughly `0.68` early in training to `0.1595` at the final logged step.

Downstream query-compiler eval after v1b:

| model | plan_valid | json_parse | slot_accuracy | execution_success | overlap@k |
|---|---:|---:|---:|---:|---:|
| baseline_full_sft | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| dpo_query_compiler_104m_stronger_v1b | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |

Important caveat: these are system-level scores after validator/projection. In the v1b downstream run, raw first-pass parsing was `0/80`, `used_projection` was `80/80`, and the first answers were not valid query JSON. Therefore this table should only be used to say that the DPO checkpoint did not break the projected execution pipeline. It should not be used to claim improved raw query-plan generation.

Evidence files:

- `configs/dpo_query_compiler_104m_stronger_v1b.json`
- `data/processed/fbbp_query_compiler_dpo_stronger_v1b.jsonl`
- `reports/algorithm_resume/query_compiler_dpo_stronger_v1b_dataset_report.md`
- `reports/logs/dpo_query_compiler_104m_2273689.log`
- `reports/algorithm_resume/dpo_training_loss_2273689_stronger_v1b.png`
- `reports/eval/dpo_preference_2273690/baseline_full_sft/summary.md`
- `reports/eval/dpo_preference_2273690/dpo_query_compiler_104m_stronger_v1b/summary.md`
- `reports/eval/dpo_query_compiler_104m_stronger_v1b_query_compiler_2273692/dpo_query_compiler_104m_stronger_v1b_score.md`

## Official C-Eval medical validation

This is a local score on the official C-Eval validation split, not a submitted official test leaderboard score. C-Eval is kept as auxiliary medical benchmark plumbing, not as the main result for the FBBP query-compiler DPO branch.

| model | subjects | total n | correct | accuracy |
|---|---|---:|---:|---:|
| baseline_full_sft | basic_medicine, clinical_medicine, physician | 90 | 11 | 0.1222 |
| dpo_query_compiler_104m_smoke | basic_medicine, clinical_medicine, physician | 90 | 11 | 0.1222 |

Subject-level scores:

| model | basic_medicine | clinical_medicine | physician |
|---|---:|---:|---:|
| baseline_full_sft | 4/19, 0.2105 | 2/22, 0.0909 | 5/49, 0.1020 |
| dpo_query_compiler_104m_smoke | 4/19, 0.2105 | 2/22, 0.0909 | 5/49, 0.1020 |

Evidence files:

- `reports/eval/official_ceval_medical_2273658/baseline_full_sft/summary.md`
- `reports/eval/official_ceval_medical_2273658/dpo_query_compiler_104m_smoke/summary.md`
- `reports/eval/official_ceval_medical_2273658/baseline_full_sft/predictions.jsonl`
- `reports/eval/official_ceval_medical_2273658/dpo_query_compiler_104m_smoke/predictions.jsonl`

## Improved C-Eval scoring protocol

The original zero-shot generate scorer produced a very low score. A more appropriate local validation scorer uses C-Eval dev examples as 5-shot context and scores A/B/C/D by choice log probability. This raised both baseline and query-DPO validation accuracy to `21/90 = 0.2333`, but still did not show a DPO improvement.

| model | scoring | n_shot | total n | correct | accuracy |
|---|---|---:|---:|---:|---:|
| baseline_full_sft | choice_logprob | 5 | 90 | 21 | 0.2333 |
| dpo_query_compiler_104m_smoke | choice_logprob | 5 | 90 | 21 | 0.2333 |

Evidence files:

- `reports/eval/official_ceval_medical_2273668/baseline_full_sft/summary.md`
- `reports/eval/official_ceval_medical_2273668/dpo_query_compiler_104m_smoke/summary.md`

## Separate medical-QA DPO branch

A separate medical-QA DPO branch was built from C-Eval dev chosen/rejected answer pairs, then evaluated on official C-Eval validation medical subjects with the same 5-shot choice-logprob protocol.

Medical-QA DPO training:

- Data: `data/processed/medical_ceval_dpo_dev.jsonl`
- Generated pairs: `15`
- Model: MiniMind2 104M, hidden size `768`, layers `16`
- Training: `2` epochs, `8` steps per epoch, batch size `2`, accumulation `2`
- Loss log: `reports/logs/dpo_medical_ceval_104m_2273672.log`
- Parsed loss curve: `reports/algorithm_resume/medical_dpo_training_loss_2273672.png`

Observed medical-QA DPO loss checkpoints:

| epoch | step | loss |
|---:|---:|---:|
| 1 | 2 | 0.6573 |
| 1 | 4 | 0.7875 |
| 1 | 6 | 0.6931 |
| 1 | 7 | 0.7235 |
| 1 | 8 | 0.6949 |
| 2 | 2 | 0.6964 |
| 2 | 4 | 0.6477 |
| 2 | 6 | 0.6837 |
| 2 | 7 | 0.6699 |
| 2 | 8 | 0.6612 |

C-Eval validation after the separate medical-QA DPO run:

| model | scoring | n_shot | total n | correct | accuracy |
|---|---|---:|---:|---:|---:|
| baseline_full_sft | choice_logprob | 5 | 90 | 21 | 0.2333 |
| medical_ceval_dpo_104m_smoke | choice_logprob | 5 | 90 | 21 | 0.2333 |

Subject-level scores:

| model | basic_medicine | clinical_medicine | physician |
|---|---:|---:|---:|
| baseline_full_sft | 3/19, 0.1579 | 5/22, 0.2273 | 13/49, 0.2653 |
| medical_ceval_dpo_104m_smoke | 3/19, 0.1579 | 5/22, 0.2273 | 13/49, 0.2653 |

Interpretation: the separate branch demonstrates medical-DPO data construction, GPU DPO training, checkpointing, loss logging, and medical benchmark evaluation. It does not yet demonstrate medical QA performance improvement.

Evidence files:

- `reports/logs/build_medical_ceval_dpo_dataset_2273672.log`
- `reports/logs/dpo_medical_ceval_104m_2273672.log`
- `reports/algorithm_resume/medical_dpo_training_loss_2273672.png`
- `reports/eval/official_ceval_medical_2273673/baseline_full_sft/summary.md`
- `reports/eval/official_ceval_medical_2273673/medical_ceval_dpo_104m_smoke/summary.md`

## Resume-safe wording

Safe:

> Completed a MiniMind2 104M DPO smoke training run for an FBBP query-compiler task, including chosen/rejected preference data construction, GPU training logs, checkpoint output, loss-curve parsing, and post-DPO validator-backed evaluation.

Safe:

> Added held-out DPO preference evaluation for the query-compiler branch, reporting preference accuracy, average preference margin, and held-out DPO loss against baseline full-SFT.

Safe:

> Improved held-out query-plan preference accuracy from `0.45` to `0.9875` with a stronger MiniMind2 104M DPO run; downstream validator/projection-backed execution remained at `1.0`, but raw first-pass generation was not improved in this run.

Safe:

> Built a separate medical-QA DPO/C-Eval validation branch with C-Eval dev preference-pair construction, GPU DPO training, checkpoint/loss logging, and local official C-Eval validation scoring.

Avoid:

> DPO improved medical C-Eval performance.

Avoid:

> Official C-Eval test leaderboard score.

Avoid:

> Completed full RLHF / PPO / GRPO training.
