"""Aggregate per-receipt extractions into the financial/expense summary.

See Prompt.md - Phase 7.
"""
from __future__ import annotations

import csv
from pathlib import Path

from .config import Config
from .schema import ExpenseSummary, ReceiptExtraction
from .utils import get_logger

log = get_logger(__name__)


def build_summary(results: list[ReceiptExtraction], cfg: Config) -> ExpenseSummary:
    """Total spend, transaction counts, per-store breakdown, date range.

    TODO(Phase 7):
      * total_spend = sum of total_amount.value where confidence >= min_total_conf
      * group spend_per_store by the fuzzy-normalized vendor name ("UNKNOWN" bucket)
      * date_range from valid ISO dates
      * record every receipt excluded from total_spend with a reason
    """
    raise NotImplementedError("Phase 7")


def load_results_from_json(json_dir: str | Path) -> list[ReceiptExtraction]:
    """Rehydrate ReceiptExtraction objects from outputs/json/*.json. TODO(Phase 7)."""
    raise NotImplementedError("Phase 7")


def write_summary(summary: ExpenseSummary, out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "expense_summary.json").write_text(
        summary.model_dump_json(indent=2), encoding="utf-8"
    )
    with (out_dir / "expense_summary.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["store", "count", "spend", "mean_confidence"])
        for store, s in sorted(summary.spend_per_store.items()):
            writer.writerow([store, s.count, f"{s.spend:.2f}", f"{s.mean_confidence:.3f}"])
