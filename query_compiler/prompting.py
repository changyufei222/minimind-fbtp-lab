from __future__ import annotations

from difflib import get_close_matches
import json
import re
from typing import Any

from .dsl import DEFAULT_LIMIT
from .field_registry import ORAL_CLASS_VALUES, SCAFFOLD_TYPE_VALUES
from .request_hints import format_request_hints, infer_request_hints
from .sketch import SKETCH_FIELDS


SCAFFOLD_ENUM_TEXT = ", ".join(f'"{value}"' for value in SCAFFOLD_TYPE_VALUES)
ORAL_ENUM_TEXT = ", ".join(f'"{value}"' for value in ORAL_CLASS_VALUES)

QUERY_COMPILER_SYSTEM_PROMPT = """
FBBP query compiler. Output one JSON object only.
Keys: candidate_id, scaffold_type, oral_class, engineered, has_experimental_affinity, organism, sort_by, sort_dir, limit.
Rules:
- use null for missing slots
- scaffold_type in [{scaffold_enum}]
- oral_class in [{oral_enum}]
- engineered and has_experimental_affinity are true, false, or null
- sort_by is "best_affinity_value" or null
- sort_dir is "asc", "desc", or null
- limit is an integer; default 20
- do not output intent, entity, filters, sort, or return_fields
- copy scaffold_type from the request exactly
- copy oral high / medium / low exactly to High / Medium / Low
- engineered means true; non-engineered means false
- prefer deterministic hints from the user message over default priors
- never use schema words like candidate_id or best_affinity_value as values
- use the exact key spellings above and keep that key order
- close the JSON object immediately after limit
""".strip().format(scaffold_enum=SCAFFOLD_ENUM_TEXT, oral_enum=ORAL_ENUM_TEXT)


QUERY_COMPILER_REPAIR_SYSTEM_PROMPT = """
Repair one FBBP slot-sketch JSON. Output one JSON object only.
Keys: candidate_id, scaffold_type, oral_class, engineered, has_experimental_affinity, organism, sort_by, sort_dir, limit.
Rules:
- use null for missing slots
- scaffold_type in [{scaffold_enum}]
- oral_class in [{oral_enum}]
- engineered and has_experimental_affinity are true, false, or null
- sort_by is "best_affinity_value" or null
- sort_dir is "asc", "desc", or null
- limit is an integer; default 20
- keep at least one non-null filter slot
- copy scaffold_type from the request exactly
- copy oral high / medium / low exactly to High / Medium / Low
- engineered means true; non-engineered means false
- prefer deterministic hints from the user message over default priors
- never use schema words like candidate_id or best_affinity_value as values
- preserve intact slot values from the bad JSON when they already match the request
- if a slot key is misspelled but the value is clear, repair the key and keep the value
- close the JSON object immediately after limit
""".strip().format(scaffold_enum=SCAFFOLD_ENUM_TEXT, oral_enum=ORAL_ENUM_TEXT)


SKETCH_KEY_BY_LOWER = {field.lower(): field for field in SKETCH_FIELDS}
SKETCH_KEY_ALIASES = {
    "engimber_": "engineered",
    "enginered": "engineered",
    "higheral_class": "oral_class",
    "statper_class": "scaffold_type",
}


def _build_compiler_user_prompt(user_prompt: str, include_hints: bool = True, wrap_request: bool = True) -> str:
    if not include_hints and not wrap_request:
        return user_prompt

    lines: list[str] = []
    if wrap_request:
        lines.append(f"Req: {user_prompt}")
    else:
        lines.append(user_prompt)

    if include_hints:
        hint_block = format_request_hints(infer_request_hints(user_prompt))
        lines.append(f"Hints: {hint_block}")

    lines.append("JSON only.")
    return "\n".join(lines)


def _describe_repair_errors(original_request: str, errors: list[str]) -> str:
    hints = infer_request_hints(original_request)
    descriptions: list[str] = []
    seen: set[str] = set()

    for error in errors:
        if error in seen:
            continue
        seen.add(error)

        if error == "parse_failed":
            descriptions.append("the JSON is malformed or truncated")
        elif error == "invalid_draft_type":
            descriptions.append("the answer must be one JSON object, not prose or a list")
        elif error == "invalid_slot_value":
            descriptions.append("replace unsupported slot values with valid enum or boolean values")
        elif error in {"missing_filters", "invalid_filters", "empty_filters"}:
            descriptions.append("keep at least one non-null filter slot")
        elif error == "invalid_sort":
            descriptions.append('sort_by must be "best_affinity_value" or null, and sort_dir must be "asc", "desc", or null')
        elif error == "invalid_limit":
            descriptions.append("limit must be a positive integer")
        elif error == "semantic_mismatch_scaffold_type":
            expected = hints.get("scaffold_type")
            descriptions.append(f"scaffold_type must match the request scaffold ({expected})" if expected else "scaffold_type must match the request scaffold")
        elif error == "semantic_mismatch_oral_class":
            expected = hints.get("oral_class")
            descriptions.append(f"oral_class must match the request oral level ({expected})" if expected else "oral_class must match the request oral level")
        elif error == "semantic_mismatch_engineered":
            expected = hints.get("engineered")
            if expected is True:
                descriptions.append("engineered must be true because the request says engineered")
            elif expected is False:
                descriptions.append("engineered must be false because the request says non-engineered")
            else:
                descriptions.append("engineered must match the request")
        elif error == "semantic_mismatch_has_experimental_affinity":
            descriptions.append("has_experimental_affinity must match the request")
        elif error == "semantic_mismatch_limit":
            expected = hints.get("limit")
            descriptions.append(f"limit must match the request ({expected})" if expected is not None else "limit must match the request")
        elif error == "semantic_mismatch_sort":
            descriptions.append("sort_by and sort_dir must match the requested affinity ordering")
        else:
            descriptions.append(error.replace("_", " "))

    if not descriptions:
        return "no repair hints available"
    return "; ".join(descriptions)


def build_compiler_messages(
    user_prompt: str,
    *,
    include_hints: bool = True,
    wrap_request: bool = True,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": QUERY_COMPILER_SYSTEM_PROMPT},
        {"role": "user", "content": _build_compiler_user_prompt(user_prompt, include_hints=include_hints, wrap_request=wrap_request)},
    ]


def build_repair_messages(
    original_request: str,
    invalid_output: str,
    errors: list[str],
    *,
    include_hints: bool = True,
    wrap_request: bool = True,
) -> list[dict[str, str]]:
    issue_block = _describe_repair_errors(original_request, errors)
    lines = ["Fix JSON."]
    if wrap_request:
        lines.append(f"Req: {original_request}")
    else:
        lines.append(f"Original request: {original_request}")
    if include_hints:
        hint_block = format_request_hints(infer_request_hints(original_request))
        lines.append(f"Hints: {hint_block}")
    lines.extend(
        [
            f"Issues: {issue_block}",
            f"Bad: {invalid_output}",
            "JSON only.",
        ]
    )
    user_prompt = "\n".join(lines)
    return [
        {"role": "system", "content": QUERY_COMPILER_REPAIR_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def render_plan_json(plan: dict[str, Any]) -> str:
    return json.dumps(plan, ensure_ascii=False, separators=(",", ":"))


def _close_truncated_json_candidate(text: str) -> str:
    open_braces = text.count("{")
    close_braces = text.count("}")
    open_brackets = text.count("[")
    close_brackets = text.count("]")
    if close_braces > open_braces or close_brackets > open_brackets:
        return text
    return text + ("]" * (open_brackets - close_brackets)) + ("}" * (open_braces - close_braces))


def _matches_required_keys(parsed: dict[str, Any], required_keys: set[str] | None) -> bool:
    if not required_keys:
        return True
    return required_keys.issubset(set(parsed.keys()))


def _parse_relaxed_scalar(raw_value: str) -> Any:
    token = raw_value.strip().rstrip(",")
    if not token:
        return None

    lowered = token.lower()
    if lowered.startswith("null"):
        return None
    if lowered.startswith("true") or lowered == "tru":
        return True
    if lowered.startswith("false") or lowered == "fal":
        return False

    if token.startswith('"'):
        if token.endswith('"') and len(token) >= 2:
            try:
                return json.loads(token)
            except json.JSONDecodeError:
                pass
        return token[1:].rstrip('"')

    number_match = re.match(r"-?\d+", token)
    if number_match:
        try:
            return int(number_match.group(0))
        except ValueError:
            return None
    return None


def _infer_sketch_key(raw_key: str, raw_value: str) -> str | None:
    cleaned = raw_key.strip().lower().replace("-", "_").replace(" ", "_")
    if cleaned in SKETCH_KEY_BY_LOWER:
        return SKETCH_KEY_BY_LOWER[cleaned]
    if cleaned in SKETCH_KEY_ALIASES:
        return SKETCH_KEY_ALIASES[cleaned]

    parsed_value = _parse_relaxed_scalar(raw_value)

    if "oral" in cleaned and "class" in cleaned:
        return "oral_class"
    if ("engi" in cleaned or "imber" in cleaned or "engine" in cleaned) and "organ" not in cleaned:
        return "engineered"
    if "candidate" in cleaned and "id" in cleaned:
        return "candidate_id"
    if "sort" in cleaned and "dir" in cleaned:
        return "sort_dir"
    if "sort" in cleaned and ("by" in cleaned or "field" in cleaned or "rank" in cleaned):
        return "sort_by"
    if "organ" in cleaned:
        return "organism"
    if "affinity" in cleaned or "exper" in cleaned:
        return "has_experimental_affinity"
    if "limit" in cleaned or cleaned in {"top", "k", "length"}:
        return "limit"

    if isinstance(parsed_value, str):
        if parsed_value in SCAFFOLD_TYPE_VALUES:
            return "scaffold_type"
        if parsed_value in ORAL_CLASS_VALUES:
            return "oral_class"
        if cleaned.endswith("_class") and parsed_value in SCAFFOLD_TYPE_VALUES:
            return "scaffold_type"

    fuzzy = get_close_matches(cleaned, list(SKETCH_KEY_BY_LOWER.keys()), n=1, cutoff=0.84)
    if fuzzy:
        return SKETCH_KEY_BY_LOWER[fuzzy[0]]
    return None


def _extract_relaxed_flat_sketch(text: str, required_keys: set[str] | None = None) -> dict[str, Any] | None:
    pair_pattern = re.compile(
        r'"(?P<key>[^"]+)"\s*:\s*(?P<raw>"(?:\\.|[^"\\])*"?|true|false|tru|fal|null|-?\d+)',
        flags=re.IGNORECASE,
    )
    salvaged: dict[str, Any] = {}

    for match in pair_pattern.finditer(text):
        canonical_key = _infer_sketch_key(match.group("key"), match.group("raw"))
        if canonical_key is None or canonical_key in salvaged:
            continue
        value = _parse_relaxed_scalar(match.group("raw"))
        if canonical_key == "limit":
            if isinstance(value, int) and value > 0:
                salvaged[canonical_key] = value
            continue
        salvaged[canonical_key] = value

    useful_keys = set(salvaged.keys()) - {"limit"}
    if not useful_keys:
        return None

    salvaged.setdefault("limit", DEFAULT_LIMIT)
    return salvaged if _matches_required_keys(salvaged, required_keys) else None


def extract_first_json_object(text: str, required_keys: set[str] | None = None) -> dict[str, Any] | None:
    text = text.strip()
    if not text:
        return None

    fenced_blocks = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    for block in fenced_blocks:
        parsed = extract_first_json_object(block, required_keys=required_keys)
        if parsed is not None:
            return parsed

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and _matches_required_keys(parsed, required_keys):
            return parsed
    except json.JSONDecodeError:
        pass

    repaired = _close_truncated_json_candidate(text)
    if repaired != text:
        try:
            parsed = json.loads(repaired)
            if isinstance(parsed, dict) and _matches_required_keys(parsed, required_keys):
                return parsed
        except json.JSONDecodeError:
            pass

    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            candidate = _close_truncated_json_candidate(text[index:])
            if candidate == text[index:]:
                continue
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                continue
        if isinstance(parsed, dict) and _matches_required_keys(parsed, required_keys):
            return parsed

    salvaged = _extract_relaxed_flat_sketch(text, required_keys=required_keys)
    if salvaged is not None:
        return salvaged

    return None
