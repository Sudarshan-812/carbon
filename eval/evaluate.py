"""Evaluate pipeline output against a hand-labelled subset.

See Prompt.md - Phase 11. Run AFTER filling eval/labels.csv.

    python eval/evaluate.py --json-dir outputs/json --labels eval/labels.csv

Reports per-field accuracy (store fuzzy, date exact, total to 2dp / within 0.05,
n_items exact), overall coverage on all 371 receipts, and a confidence-bucket
calibration table + eval/calibration.png.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-dir", default="outputs/json")
    ap.add_argument("--labels", default="eval/labels.csv")
    ap.add_argument("--out", default="eval/eval_report.md")
    ap.parse_args()
    raise NotImplementedError("Phase 11")


if __name__ == "__main__":
    raise SystemExit(main())
