from __future__ import annotations


DEFAULT_INTENT = "candidate_search"
DEFAULT_ENTITY = "protein_candidate"
DEFAULT_LIMIT = 20

LEGAL_OPERATORS = {"eq", "in", "gte", "lte", "exists"}

DEFAULT_RETURN_FIELDS = [
    "candidate_id",
    "canonical_name",
    "scaffold_type",
    "oral_class",
    "best_affinity_value",
]
