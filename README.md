> Interface guide: [English](./INTERFACE_GUIDE_EN.md) | [中文](./INTERFACE_GUIDE_CN.md)

# MiniMind FBBP Query Compiler Lab

Public-facing project name:

- **MiniMind FBBP Query Compiler Lab**

Repository directory retained for compatibility:

- `minimind-fbtp-lab`

This repo is the training-focused supplement to the adjacent application stack:

- `llm-rag-knowledge-base`
- `fbtp-mcp-rag-server`
- `deerflow-fbtp-research-agent`

This repo does not attempt to reimplement all of MiniMind. It isolates small, reproducible fine-tuning experiments that can run on `1 x RTX 4090` and produce defensible comparison results.

## Current Direction

The active rebuilt direction is:

- **FBBP query compiler**
- objective: compile natural-language candidate-search requests into executable query plans
- model role: generate constrained query-draft JSON
- system role: validate, normalize, execute, and benchmark those plans against the real FBBP lineage

This repo is no longer best described as only an evidence-card formatting lab.

The query-compiler line now includes:

- candidate snapshot generation from the checked-in FBBP card line
- a small query DSL and validator
- synthetic executable supervision
- repair-oriented supervision for malformed drafts
- rule baseline scoring
- LoRA wiring for the `MiniMind2 104M / 768dim` line
- fixed no-hints true holdout evaluation
- a thin demo for `request -> draft -> normalized plan -> results`

## Active Query-Compiler Status

The active line is the query-compiler benchmark, not the archived evidence-card line.

The early rounds established the difficulty curve:

- **Round 1**
  - `base`: `plan_valid_rate = 0.075`, `slot_accuracy = 0.0234`, `result_overlap_at_k = 0.0`
  - `lora`: `plan_valid_rate = 0.0875`, `slot_accuracy = 0.0281`, `result_overlap_at_k = 0.0`
- **Round 2 (`JSON-first`)**
  - `base v2`: all major metrics stayed `0.0`
  - `lora v2`: `plan_valid_rate = 0.1875`, `slot_accuracy = 0.0578`, `result_overlap_at_k = 0.0`

The final hardening run is the `v18 -> v23` sequence on the fixed `v15 true holdout`:

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

That `v23` completion audit is the first point where the repo has strong protocol-level evidence that the model is not merely producing valid JSON, but is performing stable first-pass semantic grounding on the reserved no-hints true holdout.

## Current Status

- **Round 1**: mixed `RAGPPI + CoVUniBind` seed dataset, LoRA pipeline validated end-to-end
- **Round 1 finding**: training completed, but baseline vs LoRA quality did not improve in a meaningful way
- **Round 2 goal**: switch to a narrower `CoVUniBind-only` task with a single normalized output schema
- **Round 3 goal**: switch to `MiniMind2 104M / 768dim` and an even narrower fixed 8-line schema task
- **Round 3.1 goal**: keep the 104M line and tighten exact line-by-line compliance

## Final Status For This Stage

The active query-compiler line now has a finished protocol result:

- **Final active result**: `v23` on the reserved `v15 true holdout`
- **Final score summary**:
  - `plan_valid_rate = 1.0`
  - `json_parse_rate = 1.0`
  - `non_empty_filter_rate = 1.0`
  - `field_value_exact_match = 1.0`
  - `slot_accuracy = 1.0`
  - `execution_success_rate = 1.0`
  - `result_overlap_at_k = 1.0`
- **Final completion audit**:
  - `first_pass_perfect_rate = 1.0`
  - `used_repair_rate = 0.0`
  - `used_projection_rate = 0.0`

This is the active result to present for the query-compiler project under the current evaluation protocol.

## Protocol Boundary

The `1.0` result should be presented carefully.

- It is a result on a **reserved no-hints true holdout** under the current query-compiler protocol.
- It is **not** a blanket claim that arbitrary future user phrasing is universally solved.
- The database entity layer is real, but the natural-language supervision is still largely programmatically constructed.
- The correct external-facing interpretation is:
  - the repo demonstrates a strong, reproducible internal benchmark result
  - not yet a universal generalization claim over open-ended real-world query language

The older evidence-card rounds remain archived only as historical context:

- `reports/final_experiment_summary.md`
- `reports/baseline_vs_lora_round3_short_schema_104m.md`
- `reports/baseline_vs_lora_round3_1_strict_104m.md`

## Repo Layout

- `configs/` - experiment configs
- `data/processed/` - generated training datasets
- `data/eval/` - fixed evaluation prompts
- `scripts/` - dataset builders, training wrappers, evaluation scripts
- `cluster/` - Slurm job scripts for the GPU cluster
- `weights/` - notes for upstream checkpoints
- `reports/` - checkpoints, logs, eval outputs, and comparison templates

## Final Query-Compiler Artifacts

- `reports/eval/v23_completion_audit/lora_v23_v15_true_holdout_score_2238710.md`
- `reports/eval/v23_completion_audit/lora_v23_v15_true_holdout_audit_2238710.json`
- `reports/eval/v23_completion_audit/baseline_v23_v15_true_holdout_score_2238710.md`

## Training Evidence

- `TRAINING_README.md` - training protocol, checkpoint promotion logic, capability coverage, and resume-safe wording
- `reports/algorithm_resume/query_compiler_training_eval_comparison.png` - checkpoint/eval comparison figure
- `reports/algorithm_resume/query_compiler_training_eval_comparison.svg` - editable vector version of the same figure
- `data/processed/fbbp_query_compiler_dpo_smoke.jsonl` - DPO chosen/rejected query-plan preference pairs
- `configs/dpo_query_compiler_104m_smoke.json` - MiniMind DPO smoke training config
- `reports/logs/dpo_query_compiler_104m_2273452.log` - completed remote DPO GPU training log
- `reports/algorithm_resume/dpo_training_loss_2273452.png` - parsed DPO loss curve from the completed remote run
- `reports/eval/dpo_query_compiler_104m_smoke_query_compiler_2273453/dpo_query_compiler_104m_smoke_score.md` - post-DPO query-compiler holdout evaluation
- `reports/eval/dpo_preference_2273671/dpo_query_compiler_104m_smoke/summary.md` - held-out chosen/rejected preference evaluation for the query-compiler DPO checkpoint
- `configs/dpo_query_compiler_104m_stronger_v1b.json` - stronger DPO retry config with 900 conversation-derived preference pairs
- `reports/eval/dpo_preference_2273690/dpo_query_compiler_104m_stronger_v1b/summary.md` - improved held-out DPO preference result for the stronger v1b checkpoint
- `reports/eval/dpo_query_compiler_104m_stronger_v1b_query_compiler_2273692/dpo_query_compiler_104m_stronger_v1b_score.md` - downstream query-compiler eval after the stronger v1b DPO checkpoint; system-level projection-backed score, not raw first-pass generation evidence
- `reports/algorithm_resume/dpo_training_loss_2273689_stronger_v1b.png` - parsed DPO loss curve for the stronger v1b run
- `reports/eval/official_ceval_medical_2273668/dpo_query_compiler_104m_smoke/summary.md` - official C-Eval validation-set medical-subject score for the query-compiler DPO checkpoint, using 5-shot choice-logprob scoring
- `reports/eval/official_ceval_medical_2273673/medical_ceval_dpo_104m_smoke/summary.md` - separate medical-QA DPO branch evaluation on official C-Eval validation subjects
- `reports/algorithm_resume/dpo_ceval_remote_run_20260609.md` - consolidated DPO, preference-eval, and C-Eval remote-run evidence
- `reports/algorithm_resume/resume_claim_boundary_20260609.md` - resume-safe claim boundary separating LoRA v23 raw generation evidence, DPO preference-eval evidence, and C-Eval benchmark plumbing
- `reports/algorithm_resume/medical_dpo_training_loss_2273672.png` - parsed loss curve for the separate medical-QA DPO smoke run
- `reports/algorithm_resume/medical_ceval_smoke_score.md` - local C-Eval-style medical smoke scorer output
- `reports/algorithm_resume/lm_eval_medical_smoke/medical_ceval_smoke.yaml` - lm-eval-compatible task export
- `reports/algorithm_resume/training_loss_curve.png` - MiniMind loss-curve parser smoke output
- `reports/algorithm_resume/training_loss_parser_check.png` - parser-check curve generated from a MiniMind-format fixture log

## Resume / Defense Framing

Preferred concise framing:

- built a small-model query compiler for a private FBBP database, compiling natural-language candidate-search requests into executable structured query plans
- designed a validator-backed execution and evaluation loop with fixed no-hints true holdout testing, repair/projection tracing, and failure-mode-driven data hardening
- drove the final `v23` model to `100%` first-pass perfection on the reserved no-hints true holdout under the current evaluation protocol, with `0` repair usage and `0` projection usage

The defense-friendly interpretation is:

- the contribution is not only "the model can emit legal JSON"
- the stronger result is that the model can stably bind domain semantics to structured query slots on a held-out no-hints prompt family without downstream correction
- the safe limitation statement is that this is still protocol-bound evidence, not a claim of universal open-ended robustness

## Round 1 Assets

- Seed train/dev:
  - `data/processed/fbtp_sft_seed_train.jsonl`
  - `data/processed/fbtp_sft_seed_dev.jsonl`
- Config:
  - `configs/lora_stage1_1x4090.json`
- Eval outputs:
  - `reports/eval/baseline_509_eval.md`
  - `reports/eval/lora_509_eval.md`

Round 1 established that the pipeline works, but the model did not learn the target tasks well enough. The main issues were:

- model size is very small (`26M`)
- dataset was too small (`155` train rows)
- task types were mixed
- learning rate was too aggressive (`1e-4`)

## Round 2 Plan

Round 2 is intentionally narrower:

- **dataset**: `CoVUniBind` only
- **task**: one structured evidence-card summarization format
- **train size**: `352`
- **dev size**: `48`
- **eval prompts**: `6`
- **learning rate**: `5e-5`
- **epochs**: `4`

### Round 2 files

- Dataset builder:
  - `scripts/build_covunibind_stage2_dataset.py`
- Local raw source vendored into this repo:
  - `data/raw/covunibind_covabdab_binding_ingest.csv`
- Generated outputs:
  - `data/processed/covunibind_stage2_train.jsonl`
  - `data/processed/covunibind_stage2_dev.jsonl`
  - `data/processed/covunibind_stage2_manifest.json`
  - `data/eval/covunibind_stage2_eval_prompts.jsonl`
- Training config:
  - `configs/lora_stage2_covunibind_1x4090.json`

## Round 3 Plan

Round 3 is the highest-priority line if the goal is to get a positive result:

- **model**: `MiniMind2 104M / 768dim`
- **base weight**: `full_sft_768.pth`
- **task**: strict 8-line short schema only
- **train size**: `440`
- **dev size**: `60`
- **eval prompts**: `6`
- **learning rate**: `2e-5`
- **epochs**: `3`

### Round 3 files

- Dataset builder:
  - `scripts/build_covunibind_stage3_short_schema_dataset.py`
- Generated outputs:
  - `data/processed/covunibind_stage3_short_schema_train.jsonl`
  - `data/processed/covunibind_stage3_short_schema_dev.jsonl`
  - `data/processed/covunibind_stage3_short_schema_manifest.json`
  - `data/eval/covunibind_stage3_short_schema_eval_prompts.jsonl`
- Training config:
  - `configs/lora_stage3_covunibind_short_104m_1x4090.json`

## Round 3.1 Plan

Round 3.1 is the formatting-tightening pass:

- **model**: `MiniMind2 104M / 768dim`
- **base weight**: `full_sft_768.pth`
- **task**: exact 8-line schema, first line fixed to `RecordType: AntibodyAntigenEvidence`
- **train size**: `220`
- **dev size**: `30`
- **eval prompts**: `6`
- **learning rate**: `1.5e-5`
- **epochs**: `2`

### Round 3.1 files

- Dataset builder:
  - `scripts/build_covunibind_stage3_1_strict_schema_dataset.py`
- Generated outputs:
  - `data/processed/covunibind_stage3_1_strict_schema_train.jsonl`
  - `data/processed/covunibind_stage3_1_strict_schema_dev.jsonl`
  - `data/processed/covunibind_stage3_1_strict_schema_manifest.json`
  - `data/eval/covunibind_stage3_1_strict_schema_eval_prompts.jsonl`
- Training config:
  - `configs/lora_stage3_1_covunibind_strict_104m_1x4090.json`

## Required Base Weight

For LoRA on top of the aligned MiniMind checkpoint, place:

- `upstream-minimind-full/out/full_sft_512.pth`

See `weights/README.md`.

## Local Usage

### Build the FBBP candidate snapshot

```powershell
cd <local_path_removed>
python .\minimind-fbtp-lab\scripts\build_fbbp_candidate_snapshot.py
```

### Build the query-compiler dataset

```powershell
cd <local_path_removed>
python .\minimind-fbtp-lab\scripts\build_fbbp_query_compiler_dataset.py
```

The default V3 build now includes repair-oriented training rows. To inspect the count:

```powershell
cd <local_path_removed>
Get-Content .\minimind-fbtp-lab\data\processed\fbbp_query_compiler_manifest.json
```

### Run the rule baseline

```powershell
cd <local_path_removed>
python .\minimind-fbtp-lab\scripts\run_query_compiler_rule_baseline.py
python .\minimind-fbtp-lab\scripts\score_query_compiler_eval.py --input-jsonl .\minimind-fbtp-lab\reports\eval\query_compiler_rule_eval.jsonl --output-json .\minimind-fbtp-lab\reports\eval\query_compiler_rule_score.json --output-md .\minimind-fbtp-lab\reports\eval\query_compiler_rule_score.md
```

### Run the thin demo

```powershell
cd <local_path_removed>
python .\minimind-fbtp-lab\scripts\demo_query_compiler.py --query "帮我筛一批 engineered 的 kunitz 候选，口服别太差，优先亲和力更强的"
```

### Inspect the LoRA launch commands without starting training

```powershell
cd <local_path_removed>
powershell -ExecutionPolicy Bypass -File .\minimind-fbtp-lab\scripts\run_lora_query_compiler_104m.ps1 -ConfigPath .\minimind-fbtp-lab\configs\lora_query_compiler_104m_v3.json -WhatIf
```

### Run a local V3 smoke eval

```powershell
cd <local_path_removed>
Get-Content .\minimind-fbtp-lab\data\eval\fbbp_query_compiler_eval_prompts.jsonl -TotalCount 1 | Set-Content .\minimind-fbtp-lab\data\eval\fbbp_query_compiler_eval_prompts_smoke_v3.jsonl -Encoding utf8
python .\minimind-fbtp-lab\scripts\run_query_compiler_eval.py --output-dir .\minimind-fbtp-lab\reports\eval\local_smoke_v3 --label smoke_v3 --prompts .\minimind-fbtp-lab\data\eval\fbbp_query_compiler_eval_prompts_smoke_v3.jsonl --max-new-tokens 120
```

### Build the round-2 dataset

```powershell
cd <local_path_removed>
python .\minimind-fbtp-lab\scripts\build_covunibind_stage2_dataset.py
```

### Run the round-2 LoRA line

```powershell
cd <local_path_removed>
powershell -ExecutionPolicy Bypass -File .\scripts\run_lora_stage2_covunibind.ps1
```

### Run the round-3 LoRA line

```powershell
cd <local_path_removed>
powershell -ExecutionPolicy Bypass -File .\scripts\run_lora_stage3_covunibind_short_104m.ps1
```

### Run the round-3.1 LoRA line

```powershell
cd <local_path_removed>
powershell -ExecutionPolicy Bypass -File .\scripts\run_lora_stage3_1_covunibind_strict_104m.ps1
```

## Cluster Usage

For the active query-compiler line, use the current cluster wrappers and the `gpu` partition rather than the old archived `gpu_4090` examples.

### Query-compiler V3 train

```bash
cd /data/run01/scv7sd2/minimind-fbtp-lab/cluster
TRAIN_CONFIG_PATH=/data/run01/scv7sd2/minimind-fbtp-lab/configs/lora_query_compiler_104m_v3.json bash ./submit_lora_query_compiler_104m_1x4090.sh
```

### Query-compiler V3 eval

```bash
cd /data/run01/scv7sd2/minimind-fbtp-lab/cluster
QUERY_COMPILER_LORA_NAME=fbbp_query_compiler_104m_v3 QUERY_COMPILER_LORA_SUBDIR=lora_query_compiler_104m_v3 BASELINE_LABEL=baseline_v3 LORA_LABEL=lora_v3 bash ./submit_eval_query_compiler_104m_1x4090.sh
```

The older sections below are historical archive instructions for the evidence-card lines.

Prepare the environment on the login node first:

```bash
cd /data/run01/scw6fnk/chang
bash ./minimind-fbtp-lab/cluster/prepare_env.sh /data/run01/scw6fnk/chang
```

Then submit the round-2 LoRA job:

```bash
cd /data/run01/scw6fnk/chang/minimind-fbtp-lab/cluster
sbatch -p gpu_4090 --gpus=1 ./run_lora_stage2_covunibind_1x4090.sh
```

After training, submit the round-2 comparison eval:

```bash
cd /data/run01/scw6fnk/chang/minimind-fbtp-lab/cluster
sbatch -p gpu_4090 --gpus=1 ./run_eval_lora_stage2_covunibind_1x4090.sh
```

For the higher-probability positive-result line, submit round 3 instead:

```bash
cd /data/run01/scw6fnk/chang/minimind-fbtp-lab/cluster
sbatch -p gpu_4090 --gpus=1 ./run_lora_stage3_covunibind_short_104m_1x4090.sh
```

Then evaluate it with:

```bash
cd /data/run01/scw6fnk/chang/minimind-fbtp-lab/cluster
sbatch -p gpu_4090 --gpus=1 ./run_eval_lora_stage3_covunibind_short_104m_1x4090.sh
```

If you want the formatting-tightening pass, run round 3.1:

```bash
cd /data/run01/scw6fnk/chang/minimind-fbtp-lab/cluster
sbatch -p gpu_4090 --gpus=1 ./run_lora_stage3_1_covunibind_strict_104m_1x4090.sh
```

Then evaluate it with:

```bash
cd /data/run01/scw6fnk/chang/minimind-fbtp-lab/cluster
sbatch -p gpu_4090 --gpus=1 ./run_eval_lora_stage3_1_covunibind_strict_104m_1x4090.sh
```

## Deliverables

The minimum useful deliverables for this repo are:

- one stable LoRA config
- one narrow, reproducible dataset
- one output checkpoint
- one baseline vs LoRA comparison table
- one short note explaining what changed between rounds

For the new query-compiler line, the minimum useful deliverables are:

- one candidate snapshot build path
- one executable query DSL
- one validator + executor path
- one synthetic dataset + fixed eval set
- one rule baseline score
- one baseline vs LoRA comparison report
- one thin demo path
