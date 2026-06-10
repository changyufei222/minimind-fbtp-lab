from __future__ import annotations

import re


def infer_limit(prompt: str) -> int:
    match = re.search(r"(?:前\s*|limit\s*)(\d+)", prompt, flags=re.IGNORECASE)
    return int(match.group(1)) if match else 20


def infer_rule_draft(prompt: str, scaffold_values: list[str]) -> dict:
    prompt_lower = prompt.lower()
    draft: dict = {"filters": [], "sort": [], "limit": infer_limit(prompt)}

    for scaffold in sorted({value for value in scaffold_values if value and value != "unknown"}):
        if scaffold.lower() in prompt_lower:
            draft["filters"].append({"field": "scaffold", "op": "=", "value": scaffold})
            break

    for oral in ("high", "medium", "low"):
        if f"oral {oral}" in prompt_lower:
            draft["filters"].append({"field": "oral", "op": "=", "value": oral})
            break

    if "non-engineered" in prompt_lower:
        draft["filters"].append({"field": "engineered", "op": "=", "value": False})
    elif "engineered" in prompt_lower:
        draft["filters"].append({"field": "engineered", "op": "=", "value": True})

    if "实验亲和力" in prompt or "experimental affinity" in prompt_lower or "亲和力" in prompt or "affinity" in prompt_lower:
        draft["filters"].append({"field": "experimental_affinity", "op": "=", "value": True})
        draft["sort"].append({"field": "affinity", "direction": "strongest"})

    return draft
