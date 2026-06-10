from __future__ import annotations

import json
import sys
from pathlib import Path


LAB_ROOT = Path(__file__).resolve().parents[1]
if str(LAB_ROOT) not in sys.path:
    sys.path.insert(0, str(LAB_ROOT))

from query_compiler.prompting import QUERY_COMPILER_REPAIR_SYSTEM_PROMPT  # type: ignore  # noqa: E402
from query_compiler.synthetic_data import (  # type: ignore  # noqa: E402
    build_query_compiler_anti_hype_eval,
    build_query_compiler_dataset,
    build_query_compiler_true_holdout_eval,
    build_query_compiler_v12_true_holdout_eval,
    build_query_compiler_v13_true_holdout_eval,
    build_query_compiler_v15_true_holdout_eval,
)


def test_build_query_compiler_dataset_writes_all_expected_splits(tmp_path: Path) -> None:
    manifest = build_query_compiler_dataset(output_dir=tmp_path, seed=7, train_size=20)

    assert manifest["counts"]["train"] > 0
    assert manifest["counts"]["dev"] > 0
    assert manifest["counts"]["test_seen"] > 0
    assert manifest["counts"]["test_hard"] > 0
    assert (tmp_path / "fbbp_query_compiler_train.jsonl").exists()
    assert (tmp_path / "fbbp_query_compiler_test_hard.jsonl").exists()
    assert (tmp_path / "fbbp_query_compiler_eval_prompts.jsonl").exists()


def test_dataset_examples_include_prompt_and_gold_plan(tmp_path: Path) -> None:
    manifest = build_query_compiler_dataset(output_dir=tmp_path, seed=11, train_size=12)

    train_path = tmp_path / "fbbp_query_compiler_train.jsonl"
    first_record = json.loads(train_path.read_text(encoding="utf-8").splitlines()[0])
    assistant_content = first_record["conversations"][2]["content"]
    assistant_json = json.loads(assistant_content)

    assert "conversations" in first_record
    assert first_record["conversations"][0]["role"] == "system"
    assert first_record["conversations"][1]["role"] == "user"
    assert first_record["conversations"][2]["role"] == "assistant"
    assert "filters" not in assistant_json
    assert "limit" in assistant_json
    assert "scaffold_type" in assistant_json
    assert manifest["seed"] == 11


def test_train_split_contains_only_conversations_for_sft_loader_compatibility(tmp_path: Path) -> None:
    build_query_compiler_dataset(output_dir=tmp_path, seed=5, train_size=10)

    train_path = tmp_path / "fbbp_query_compiler_train.jsonl"
    first_record = json.loads(train_path.read_text(encoding="utf-8").splitlines()[0])

    assert set(first_record.keys()) == {"conversations"}


def test_train_split_expands_with_multiple_prompt_variants(tmp_path: Path) -> None:
    manifest = build_query_compiler_dataset(output_dir=tmp_path, seed=13, train_size=5, train_variants_per_row=3)

    train_path = tmp_path / "fbbp_query_compiler_train.jsonl"
    rows = [json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    assert manifest["counts"]["train"] == 15
    assert len(rows) == 15


def test_build_query_compiler_dataset_can_emit_repair_examples(tmp_path: Path) -> None:
    manifest = build_query_compiler_dataset(
        output_dir=tmp_path,
        seed=17,
        train_size=6,
        train_variants_per_row=1,
        repair_examples_per_row=1,
    )

    train_path = tmp_path / "fbbp_query_compiler_train.jsonl"
    rows = [json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    assert any(row["conversations"][0]["content"] == QUERY_COMPILER_REPAIR_SYSTEM_PROMPT for row in rows)
    assert manifest["counts"]["repair_examples"] == 6
    assert manifest["counts"]["train"] == 12
    repair_rows = [row for row in rows if row["conversations"][0]["content"] == QUERY_COMPILER_REPAIR_SYSTEM_PROMPT]
    assert any("Req:" in row["conversations"][1]["content"] for row in repair_rows)


def test_train_prompt_variants_cover_anti_hype_families(tmp_path: Path) -> None:
    build_query_compiler_dataset(
        output_dir=tmp_path,
        seed=29,
        train_size=2,
        train_variants_per_row=6,
        repair_examples_per_row=0,
    )

    train_path = tmp_path / "fbbp_query_compiler_train.jsonl"
    rows = [json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    user_messages = [row["conversations"][1]["content"] for row in rows]

    assert any("Compile this FBBP candidate request to JSON only" in message for message in user_messages)
    assert any("数据库筛选请求，不要总结" in message or "只输出查询 JSON" in message for message in user_messages)


def test_dataset_can_emit_bare_no_hints_training_examples(tmp_path: Path) -> None:
    manifest = build_query_compiler_dataset(
        output_dir=tmp_path,
        seed=37,
        train_size=3,
        train_variants_per_row=1,
        repair_examples_per_row=0,
        bare_no_hints_examples_per_row=1,
    )

    train_path = tmp_path / "fbbp_query_compiler_train.jsonl"
    rows = [json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    user_messages = [row["conversations"][1]["content"] for row in rows]

    assert manifest["counts"]["bare_no_hints_examples"] == 3
    assert any("From the FBBP registry" in message or "请按数据库语义理解这个检索" in message for message in user_messages)
    assert any("Hints:" not in message and "Req:" not in message for message in user_messages)


def test_bare_no_hints_training_examples_rotate_across_true_holdout_families(tmp_path: Path) -> None:
    manifest = build_query_compiler_dataset(
        output_dir=tmp_path,
        seed=43,
        train_size=8,
        train_variants_per_row=1,
        repair_examples_per_row=0,
        bare_no_hints_examples_per_row=8,
    )

    train_path = tmp_path / "fbbp_query_compiler_train.jsonl"
    rows = [json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    user_messages = [row["conversations"][1]["content"] for row in rows if "Hints:" not in row["conversations"][1]["content"] and "Req:" not in row["conversations"][1]["content"]]

    assert manifest["counts"]["bare_no_hints_examples"] == 64
    assert any("From the FBBP registry" in message for message in user_messages)
    assert any("I am shortlisting entries from the candidate table" in message for message in user_messages)
    assert any("For this screen, I only want molecules with" in message for message in user_messages)
    assert any("请按数据库语义理解这个检索" in message for message in user_messages)
    assert any("我在库里做初筛" in message for message in user_messages)
    assert any("研究问题是这样的" in message for message in user_messages)
    assert any("Keep a database shortlist for candidates showing" in message for message in user_messages)
    assert any("做一个候选清单给我" in message for message in user_messages)


def test_farther_no_hints_training_examples_rotate_across_v12_families(tmp_path: Path) -> None:
    manifest = build_query_compiler_dataset(
        output_dir=tmp_path,
        seed=53,
        train_size=8,
        train_variants_per_row=1,
        repair_examples_per_row=0,
        bare_no_hints_examples_per_row=0,
        farther_no_hints_examples_per_row=8,
    )

    train_path = tmp_path / "fbbp_query_compiler_train.jsonl"
    rows = [json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    user_messages = [row["conversations"][1]["content"] for row in rows if "Hints:" not in row["conversations"][1]["content"] and "Req:" not in row["conversations"][1]["content"]]

    assert manifest["counts"]["farther_no_hints_examples"] == 64
    assert any("Search the candidate registry for entries satisfying" in message for message in user_messages)
    assert any("I need a candidate shortlist restricted to" in message for message in user_messages)
    assert any("Within the protein-candidate table, retain only rows with" in message for message in user_messages)
    assert any("请直接按数据库语义筛选" in message for message in user_messages)
    assert any("做一个不带解释的候选筛选" in message for message in user_messages)
    assert any("我想从候选库里锁定" in message for message in user_messages)
    assert any("Build a database shortlist for candidates meeting" in message for message in user_messages)
    assert any("候选初筛条件如下" in message for message in user_messages)


def test_dataset_can_oversample_engineered_true_no_hints_examples(tmp_path: Path) -> None:
    manifest = build_query_compiler_dataset(
        output_dir=tmp_path,
        seed=61,
        train_size=20,
        train_variants_per_row=1,
        repair_examples_per_row=0,
        bare_no_hints_examples_per_row=0,
        farther_no_hints_examples_per_row=0,
        engineered_true_no_hints_examples_per_row=1,
    )

    assert manifest["counts"]["engineered_true_no_hints_examples"] > 0

    train_path = tmp_path / "fbbp_query_compiler_train.jsonl"
    rows = [json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    user_messages = [row["conversations"][1]["content"] for row in rows]

    assert any("engineered molecules only" in message or "只看 engineered 条目" in message for message in user_messages)
    assert any(
        "annotation block says" in message
        or "review sheet of candidates marked" in message
        or "注释栏注明" in message
        or "候选审阅表里只放" in message
        for message in user_messages
    )


def test_dataset_can_emit_final_bridge_no_hints_examples(tmp_path: Path) -> None:
    manifest = build_query_compiler_dataset(
        output_dir=tmp_path,
        seed=71,
        train_size=12,
        train_variants_per_row=1,
        repair_examples_per_row=0,
        bare_no_hints_examples_per_row=0,
        farther_no_hints_examples_per_row=0,
        engineered_true_no_hints_examples_per_row=0,
        final_bridge_examples_per_row=1,
    )

    assert manifest["counts"]["final_bridge_examples"] > 0

    train_path = tmp_path / "fbbp_query_compiler_train.jsonl"
    rows = [json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    user_messages = [row["conversations"][1]["content"] for row in rows]

    assert any("Review the candidate ledger" in message or "Prepare a shortlist sheet" in message for message in user_messages)
    assert any("做候选账页复核" in message or "整理一个候选留样清单" in message for message in user_messages)


def test_dataset_can_emit_completion_english_bridge_examples(tmp_path: Path) -> None:
    manifest = build_query_compiler_dataset(
        output_dir=tmp_path,
        seed=79,
        train_size=12,
        train_variants_per_row=1,
        repair_examples_per_row=0,
        bare_no_hints_examples_per_row=0,
        farther_no_hints_examples_per_row=0,
        engineered_true_no_hints_examples_per_row=0,
        final_bridge_examples_per_row=0,
        completion_english_bridge_examples_per_row=1,
    )

    assert manifest["counts"]["completion_english_bridge_examples"] > 0

    train_path = tmp_path / "fbbp_query_compiler_train.jsonl"
    rows = [json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    user_messages = [row["conversations"][1]["content"] for row in rows]

    assert any("During curation review" in message or "In the candidate ledger" in message for message in user_messages)
    assert any("engineered molecules only" in message for message in user_messages)


def test_projection_hotspot_knottin_high_true_examples_get_final_bridge_variants(tmp_path: Path) -> None:
    manifest = build_query_compiler_dataset(
        output_dir=tmp_path,
        seed=83,
        train_size=80,
        train_variants_per_row=1,
        repair_examples_per_row=0,
        bare_no_hints_examples_per_row=0,
        farther_no_hints_examples_per_row=0,
        engineered_true_no_hints_examples_per_row=0,
        final_bridge_examples_per_row=0,
        completion_english_bridge_examples_per_row=1,
    )

    assert manifest["counts"]["completion_english_bridge_examples"] > 0

    train_path = tmp_path / "fbbp_query_compiler_train.jsonl"
    rows = [json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    user_messages = [row["conversations"][1]["content"] for row in rows]

    assert any(
        message.startswith("From the candidate ledger, keep rows carrying scaffold family knottin, oral class High, engineered molecules only, experimental affinity evidence must be present.")
        for message in user_messages
    )
    assert any(
        message.startswith("Keep a curation-ready shortlist of rows labeled scaffold family knottin, oral class High, engineered molecules only, experimental affinity evidence must be present.")
        for message in user_messages
    )


def test_dataset_can_emit_projection_hotspot_examples(tmp_path: Path) -> None:
    manifest = build_query_compiler_dataset(
        output_dir=tmp_path,
        seed=89,
        train_size=80,
        train_variants_per_row=1,
        repair_examples_per_row=0,
        bare_no_hints_examples_per_row=0,
        farther_no_hints_examples_per_row=0,
        engineered_true_no_hints_examples_per_row=0,
        final_bridge_examples_per_row=0,
        completion_english_bridge_examples_per_row=0,
        projection_hotspot_examples_per_row=2,
    )

    assert manifest["counts"]["projection_hotspot_examples"] > 0

    train_path = tmp_path / "fbbp_query_compiler_train.jsonl"
    rows = [json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    messages = [row["conversations"][1]["content"] for row in rows]

    assert any(
        message.startswith("From the candidate ledger, keep rows carrying scaffold family knottin, oral class High, engineered molecules only, experimental affinity evidence must be present.")
        for message in messages
    )
    assert any(
        message.startswith("Keep a curation-ready shortlist of rows labeled scaffold family knottin, oral class High, engineered molecules only, experimental affinity evidence must be present.")
        for message in messages
    )


def test_dataset_can_emit_hotspot_shortlist_contrast_examples(tmp_path: Path) -> None:
    manifest = build_query_compiler_dataset(
        output_dir=tmp_path,
        seed=97,
        train_size=120,
        train_variants_per_row=1,
        repair_examples_per_row=0,
        bare_no_hints_examples_per_row=0,
        farther_no_hints_examples_per_row=0,
        engineered_true_no_hints_examples_per_row=0,
        final_bridge_examples_per_row=0,
        completion_english_bridge_examples_per_row=0,
        projection_hotspot_examples_per_row=0,
        hotspot_shortlist_contrast_examples_per_row=2,
    )

    assert manifest["counts"]["hotspot_shortlist_contrast_examples"] > 0

    train_path = tmp_path / "fbbp_query_compiler_train.jsonl"
    rows = [json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    messages = [row["conversations"][1]["content"] for row in rows]

    assert any(
        message.startswith("Keep a curation-ready shortlist for candidate rows labeled scaffold family knottin, oral class High, engineered molecules only, experimental affinity evidence must be present.")
        for message in messages
    )
    assert any(
        message.startswith("Prepare a curation-ready shortlist for rows labeled scaffold family knottin, oral class High, engineered molecules only, experimental affinity evidence must be present.")
        for message in messages
    )
    assert any(
        message.startswith("Need a curation-ready shortlist for rows labeled scaffold family knottin, oral class High, engineered molecules only, experimental affinity evidence must be present.")
        for message in messages
    )
    assert any(
        message.startswith("Keep a curation-ready shortlist for rows labeled scaffold family knottin, oral class High, engineered molecules only, experimental affinity evidence must be present.")
        for message in messages
    )


def test_dataset_schema_word_negative_examples_expand_repair_rows(tmp_path: Path) -> None:
    manifest = build_query_compiler_dataset(
        output_dir=tmp_path,
        seed=41,
        train_size=4,
        train_variants_per_row=1,
        repair_examples_per_row=1,
        schema_word_examples_per_row=1,
    )

    train_path = tmp_path / "fbbp_query_compiler_train.jsonl"
    rows = [json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    repair_rows = [row for row in rows if row["conversations"][0]["content"] == QUERY_COMPILER_REPAIR_SYSTEM_PROMPT]

    assert manifest["counts"]["repair_examples"] >= 8
    assert len(repair_rows) >= 8
    assert any("Bad:" in row["conversations"][1]["content"] and "candidate_id" in row["conversations"][1]["content"] for row in repair_rows)


def test_targeted_completion_gate_corruptions_expand_repair_rows(tmp_path: Path) -> None:
    manifest = build_query_compiler_dataset(
        output_dir=tmp_path,
        seed=73,
        train_size=12,
        train_variants_per_row=1,
        repair_examples_per_row=2,
        bare_no_hints_examples_per_row=0,
        farther_no_hints_examples_per_row=0,
        engineered_true_no_hints_examples_per_row=1,
        final_bridge_examples_per_row=1,
        completion_english_bridge_examples_per_row=1,
        schema_word_examples_per_row=0,
    )

    assert manifest["counts"]["repair_examples"] > 0

    train_path = tmp_path / "fbbp_query_compiler_train.jsonl"
    rows = [json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    repair_rows = [row for row in rows if row["conversations"][0]["content"] == QUERY_COMPILER_REPAIR_SYSTEM_PROMPT]
    repair_messages = [row["conversations"][1]["content"] for row in repair_rows]

    assert any("tightness" in message for message in repair_messages)
    assert any("engineered must be true because the request says engineered" in message for message in repair_messages)


def test_repair_examples_can_include_no_hints_holdout_prompt_families(tmp_path: Path) -> None:
    build_query_compiler_dataset(
        output_dir=tmp_path,
        seed=67,
        train_size=10,
        train_variants_per_row=1,
        repair_examples_per_row=3,
        schema_word_examples_per_row=0,
    )

    train_path = tmp_path / "fbbp_query_compiler_train.jsonl"
    rows = [json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    repair_rows = [row for row in rows if row["conversations"][0]["content"] == QUERY_COMPILER_REPAIR_SYSTEM_PROMPT]
    repair_messages = [row["conversations"][1]["content"] for row in repair_rows]

    assert any("Req: From the FBBP registry" in message for message in repair_messages)
    assert any(
        "Req: Search the candidate registry for entries satisfying" in message
        or "Req: Filter the candidate inventory down to rows annotated with" in message
        or "Req: Search the candidate registry for entries satisfying" in message
        for message in repair_messages
    )


def test_eval_prompts_keep_standard_gold_plan_schema(tmp_path: Path) -> None:
    build_query_compiler_dataset(output_dir=tmp_path, seed=19, train_size=8, dev_size=2, test_seen_size=2, test_hard_size=2)

    eval_path = tmp_path / "fbbp_query_compiler_eval_prompts.jsonl"
    first_eval = json.loads(eval_path.read_text(encoding="utf-8").splitlines()[0])

    assert "Req:" not in first_eval["prompt"]
    assert "Hints:" not in first_eval["prompt"]
    assert first_eval["gold_plan"]["intent"] == "candidate_search"
    assert first_eval["gold_plan"]["entity"] == "protein_candidate"
    assert isinstance(first_eval["gold_plan"]["filters"], list)


def test_build_query_compiler_anti_hype_eval_emits_raw_holdout_prompts(tmp_path: Path) -> None:
    output_path = tmp_path / "anti_hype_eval.jsonl"
    audit_path = tmp_path / "anti_hype_audit.md"

    manifest = build_query_compiler_anti_hype_eval(
        output_path=output_path,
        audit_path=audit_path,
        eval_size=6,
        train_size=8,
        dev_size=2,
        test_seen_size=2,
        test_hard_size=2,
        seed=23,
    )

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    assert manifest["eval_size"] == 6
    assert len(rows) == 6
    assert audit_path.exists()
    assert rows[0]["id"].startswith("anti_hype_query_compiler_")
    assert "Hints:" not in rows[0]["prompt"]
    assert "Req:" not in rows[0]["prompt"]
    assert rows[0]["category"] == "fbbp-query-compiler-anti-hype"


def test_build_query_compiler_true_holdout_eval_avoids_seen_family_wording_and_hints(tmp_path: Path) -> None:
    output_path = tmp_path / "v10_true_holdout.jsonl"
    audit_path = tmp_path / "v10_true_holdout_audit.md"

    manifest = build_query_compiler_true_holdout_eval(
        output_path=output_path,
        audit_path=audit_path,
        eval_size=6,
        train_size=8,
        dev_size=2,
        test_seen_size=2,
        test_hard_size=2,
        seed=31,
    )

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    assert manifest["eval_size"] == 6
    assert len(rows) == 6
    assert audit_path.exists()
    assert rows[0]["id"].startswith("true_holdout_query_compiler_")
    assert rows[0]["category"] == "fbbp-query-compiler-v10-true-holdout"
    assert rows[0]["prompt_mode"] == "bare_no_hints"
    prompts = [row["prompt"] for row in rows]
    assert all("Hints:" not in prompt for prompt in prompts)
    assert all("Req:" not in prompt for prompt in prompts)
    assert all("Compile this FBBP candidate request to JSON only" not in prompt for prompt in prompts)
    assert all("Need an executable protein-candidate query" not in prompt for prompt in prompts)
    assert all("数据库筛选请求，不要总结" not in prompt for prompt in prompts)
    assert all("只输出查询 JSON" not in prompt for prompt in prompts)


def test_build_query_compiler_v12_true_holdout_eval_uses_farther_wording(tmp_path: Path) -> None:
    output_path = tmp_path / "v12_true_holdout.jsonl"
    audit_path = tmp_path / "v12_true_holdout_audit.md"

    manifest = build_query_compiler_v12_true_holdout_eval(
        output_path=output_path,
        audit_path=audit_path,
        eval_size=6,
        train_size=8,
        dev_size=2,
        test_seen_size=2,
        test_hard_size=2,
        seed=47,
    )

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    prompts = [row["prompt"] for row in rows]

    assert manifest["eval_size"] == 6
    assert len(rows) == 6
    assert audit_path.exists()
    assert rows[0]["id"].startswith("v12_true_holdout_query_compiler_")
    assert rows[0]["category"] == "fbbp-query-compiler-v12-true-holdout"
    assert rows[0]["prompt_mode"] == "bare_no_hints"
    assert all("Hints:" not in prompt for prompt in prompts)
    assert all("Req:" not in prompt for prompt in prompts)
    assert all("From the FBBP registry" not in prompt for prompt in prompts)
    assert all("I am shortlisting entries from the candidate table" not in prompt for prompt in prompts)
    assert all("For this screen, I only want molecules with" not in prompt for prompt in prompts)
    assert all("请按数据库语义理解这个检索" not in prompt for prompt in prompts)
    assert all("我在库里做初筛" not in prompt for prompt in prompts)
    assert all("研究问题是这样的" not in prompt for prompt in prompts)


def test_build_query_compiler_v13_true_holdout_eval_uses_even_farther_wording(tmp_path: Path) -> None:
    output_path = tmp_path / "v13_true_holdout.jsonl"
    audit_path = tmp_path / "v13_true_holdout_audit.md"

    manifest = build_query_compiler_v13_true_holdout_eval(
        output_path=output_path,
        audit_path=audit_path,
        eval_size=6,
        train_size=8,
        dev_size=2,
        test_seen_size=2,
        test_hard_size=2,
        seed=59,
    )

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    prompts = [row["prompt"] for row in rows]

    assert manifest["eval_size"] == 6
    assert len(rows) == 6
    assert audit_path.exists()
    assert rows[0]["id"].startswith("v13_true_holdout_query_compiler_")
    assert rows[0]["category"] == "fbbp-query-compiler-v13-true-holdout"
    assert rows[0]["prompt_mode"] == "bare_no_hints"
    assert all("Hints:" not in prompt for prompt in prompts)
    assert all("Req:" not in prompt for prompt in prompts)
    assert all("Search the candidate registry for entries satisfying" not in prompt for prompt in prompts)
    assert all("I need a candidate shortlist restricted to" not in prompt for prompt in prompts)
    assert all("Within the protein-candidate table, retain only rows with" not in prompt for prompt in prompts)
    assert all("请直接按数据库语义筛选" not in prompt for prompt in prompts)
    assert all("做一个不带解释的候选筛选" not in prompt for prompt in prompts)
    assert all("我想从候选库里锁定" not in prompt for prompt in prompts)


def test_build_query_compiler_v15_true_holdout_eval_uses_reserved_final_wording(tmp_path: Path) -> None:
    output_path = tmp_path / "v15_true_holdout.jsonl"
    audit_path = tmp_path / "v15_true_holdout_audit.md"

    manifest = build_query_compiler_v15_true_holdout_eval(
        output_path=output_path,
        audit_path=audit_path,
        eval_size=6,
        train_size=8,
        dev_size=2,
        test_seen_size=2,
        test_hard_size=2,
        seed=71,
    )

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    prompts = [row["prompt"] for row in rows]

    assert manifest["eval_size"] == 6
    assert len(rows) == 6
    assert audit_path.exists()
    assert rows[0]["id"].startswith("v15_true_holdout_query_compiler_")
    assert rows[0]["category"] == "fbbp-query-compiler-v15-true-holdout"
    assert rows[0]["prompt_mode"] == "bare_no_hints"
    assert all("Hints:" not in prompt for prompt in prompts)
    assert all("Req:" not in prompt for prompt in prompts)
    assert all("From the FBBP registry" not in prompt for prompt in prompts)
    assert all("Search the candidate registry for entries satisfying" not in prompt for prompt in prompts)
    assert all("请按数据库语义理解这个检索" not in prompt for prompt in prompts)
    assert all("请直接按数据库语义筛选" not in prompt for prompt in prompts)
