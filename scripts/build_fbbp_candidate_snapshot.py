from __future__ import annotations

import json
import sys
from pathlib import Path


LAB_ROOT = Path(__file__).resolve().parents[1]
if str(LAB_ROOT) not in sys.path:
    sys.path.insert(0, str(LAB_ROOT))

from query_compiler.candidate_snapshot import write_candidate_snapshot  # type: ignore  # noqa: E402


ROWS_PATH = LAB_ROOT / "data" / "processed" / "fbbp_candidate_snapshot.jsonl"
MANIFEST_PATH = LAB_ROOT / "data" / "processed" / "fbbp_candidate_snapshot_manifest.json"


def main() -> None:
    manifest = write_candidate_snapshot(ROWS_PATH, MANIFEST_PATH)
    print(json.dumps({"rows_path": str(ROWS_PATH), "manifest": manifest}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
