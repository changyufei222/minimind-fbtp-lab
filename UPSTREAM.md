# Upstream Notes

- Intended derivative repo: `minimind-fbtp-lab`
- Upstream baseline: `../upstream-minimind-full`
- Upstream repository: `https://github.com/jingyaogong/minimind`

## What This Repo Owns
- FBTP-oriented dataset construction
- stage-1 LoRA / SFT configs
- local training wrappers
- experiment result tables and conclusions

## What Stays Upstream
- core model architecture
- tokenizer
- training implementations (`train_lora.py`, `train_full_sft.py`, etc.)
- general-purpose MiniMind training stack

## Strategy
- keep upstream code intact
- call upstream trainers from this repo
- put all experiment-specific data, config, and reporting here

