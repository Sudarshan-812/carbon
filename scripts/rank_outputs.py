"""Rank receipt JSONs by reliability signals (Prompt.md - Phase 13).

    .venv/Scripts/python.exe scripts/rank_outputs.py [outputs/json] [N]

Prints the N receipts with the lowest mean field confidence and the N with the
most flags, so they can be eyeballed against the source images.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path


def _mean_field_conf(data: dict) -> float:
    confs: list[float] = []
    for key in ("store_name", "date", "total_amount"):
        node = data.get(key, {})
        if node.get("value") is not None:
            confs.append(node.get("confidence", 0.0))
    for item in data.get("items", []):
        for part in ("name", "price"):
            node = item.get(part, {})
            if node.get("value") is not None:
                confs.append(node.get("confidence", 0.0))
    return statistics.fmean(confs) if confs else 0.0


def _summary_line(data: dict) -> str:
    def val(key: str) -> object:
        return (data.get(key) or {}).get("value")
    return (f"store={val('store_name')!r} date={val('date')!r} "
            f"total={val('total_amount')!r} items={len(data.get('items', []))}")


def main() -> int:
    json_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/json")
    top_n = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    rows = []
    for path in sorted(json_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        rows.append({
            "id": path.stem,
            "mean_conf": _mean_field_conf(data),
            "n_flags": len(data.get("flags", [])),
            "flags": data.get("flags", []),
            "summary": _summary_line(data),
        })

    print(f"{len(rows)} receipts\n")

    print(f"== {top_n} lowest mean field confidence ==")
    for r in sorted(rows, key=lambda r: r["mean_conf"])[:top_n]:
        print(f"  {r['id']:20} conf={r['mean_conf']:.3f}  flags={r['n_flags']}  {r['summary']}")

    print(f"\n== {top_n} most flags ==")
    for r in sorted(rows, key=lambda r: (-r["n_flags"], r["mean_conf"]))[:top_n]:
        print(f"  {r['id']:20} flags={r['n_flags']} {r['flags']}  conf={r['mean_conf']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
