"""Validate every JSON file under a directory (Prompt.md - Phase 13).

    .venv/Scripts/python.exe scripts/validate_json.py [outputs/json]

Exit 0 iff every file parses and has the required top-level keys.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REQUIRED = {"store_name", "date", "items", "total_amount",
            "low_confidence_fields", "flags", "meta", "plain"}


def main() -> int:
    json_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/json")
    files = sorted(json_dir.glob("*.json"))
    bad: list[str] = []
    for path in files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            missing = REQUIRED - set(data)
            if missing:
                bad.append(f"{path.name}: missing {sorted(missing)}")
        except (json.JSONDecodeError, OSError) as exc:
            bad.append(f"{path.name}: {exc}")

    print(f"{len(files)} files | {len(files) - len(bad)} valid | {len(bad)} invalid")
    for line in bad:
        print("  ", line)
    return 1 if bad or not files else 0


if __name__ == "__main__":
    raise SystemExit(main())
