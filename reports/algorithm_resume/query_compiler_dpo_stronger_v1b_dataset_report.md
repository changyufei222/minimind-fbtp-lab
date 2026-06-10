# Query Compiler DPO Dataset Report

- source: `data\processed\fbbp_query_compiler_train.jsonl`
- source_kind: `conversation`
- output: `data\processed\fbbp_query_compiler_dpo_stronger_v1b.jsonl`
- pairs: `900`
- variants_per_row: `3`
- heldout_source: `<local_path_removed>
- heldout_output: `data\processed\fbbp_query_compiler_dpo_heldout_v13.jsonl`
- heldout_pairs: `80`
- chosen: gold executable query sketch
- rejected: deterministic corruption of semantic slots or schema-validity cues
- purpose: DPO smoke training for preference learning over query-plan correctness
