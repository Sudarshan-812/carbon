"""Aggregate per-receipt extractions into the financial/expense summary.

See Prompt.md - Phase 7.
"""
from __future__ import annotations

import csv
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from .config import Config
from .extract import VendorRegistry
from .schema import ExpenseSummary, FieldConf, Item, ReceiptExtraction, StoreSpend
from .utils import get_logger, parse_money

log = get_logger(__name__)

_ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_UNKNOWN = "UNKNOWN"


def _majority_currency(results: list[ReceiptExtraction]) -> str | None:
    """Most common ``meta['currency_detected']`` value, if any receipt set one."""
    seen = Counter(
        r.meta["currency_detected"] for r in results if r.meta.get("currency_detected")
    )
    if not seen:
        return None
    return seen.most_common(1)[0][0]


def build_summary(results: list[ReceiptExtraction], cfg: Config) -> ExpenseSummary:
    """Total spend, transaction counts, per-store breakdown, date range."""
    min_conf = cfg.summary.min_total_conf
    summary = ExpenseSummary(currency=cfg.summary.currency, num_transactions=len(results))

    detected = _majority_currency(results)
    if detected and detected != summary.currency:
        summary.notes.append(
            f"currency '{detected}' seen on most receipts; config default is "
            f"'{summary.currency}'"
        )
        summary.currency = detected

    vendors = VendorRegistry(cfg.extract.store_name.vendor_merge_ratio)
    per_store: dict[str, list[tuple[float | None, float]]] = defaultdict(list)
    total_spend = 0.0
    n_with_total = 0
    valid_dates: list[str] = []

    for result in results:
        image_id = str(result.meta.get("image_id", "unknown"))

        store_value = result.store_name.value
        store_key = vendors.canonical(store_value)[0] if store_value else _UNKNOWN

        total_value = parse_money(result.total_amount.value)
        total_conf = result.total_amount.confidence
        counted = total_value is not None and total_conf >= min_conf

        if counted:
            total_spend += total_value
            n_with_total += 1
            per_store[store_key].append((total_value, total_conf))
        else:
            per_store[store_key].append((None, total_conf))
            reason = ("missing_total" if total_value is None
                      else f"total_confidence {total_conf:.2f} < {min_conf}")
            summary.excluded.append({"image_id": image_id, "store": store_key, "reason": reason})

        if result.date.value and _ISO_RE.match(result.date.value):
            valid_dates.append(result.date.value)

    summary.total_spend = round(total_spend, 2)
    summary.num_transactions_with_total = n_with_total

    for store_key, entries in per_store.items():
        spends = [value for value, _ in entries if value is not None]
        summary.spend_per_store[store_key] = StoreSpend(
            count=len(entries),
            spend=round(sum(spends), 2),
            mean_confidence=round(statistics.fmean(conf for _, conf in entries), 3),
        )

    summary.date_range = (
        {"start": min(valid_dates), "end": max(valid_dates)}
        if valid_dates else {"start": None, "end": None}
    )

    if summary.excluded:
        summary.notes.append(
            f"{len(summary.excluded)} of {len(results)} receipts excluded from "
            "total_spend (missing or low-confidence total)"
        )
    summary.notes.append(f"total_spend includes only totals with confidence >= {min_conf}")
    return summary


def result_from_json_dict(data: dict) -> ReceiptExtraction:
    """Rehydrate one :class:`ReceiptExtraction` from a parsed ``<id>.json`` dict."""
    return ReceiptExtraction(
        store_name=FieldConf(**data["store_name"]),
        date=FieldConf(**data["date"]),
        total_amount=FieldConf(**data["total_amount"]),
        items=[
            Item(name=FieldConf(**it["name"]), price=FieldConf(**it["price"]),
                 meta={"qty": it.get("qty")})
            for it in data.get("items", [])
        ],
        low_confidence_fields=data.get("low_confidence_fields", []),
        flags=data.get("flags", []),
        meta=data.get("meta", {}),
    )


def load_one_result(path: str | Path) -> ReceiptExtraction:
    """Rehydrate a single ``outputs/json/<id>.json`` file."""
    return result_from_json_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def load_results_from_json(json_dir: str | Path) -> list[ReceiptExtraction]:
    """Rehydrate :class:`ReceiptExtraction` objects from ``outputs/json/*.json``."""
    return [load_one_result(p) for p in sorted(Path(json_dir).glob("*.json"))]


def _store_rows(summary: ExpenseSummary) -> list[tuple[str, StoreSpend]]:
    return sorted(summary.spend_per_store.items(), key=lambda kv: (-kv[1].spend, kv[0]))


def write_summary(summary: ExpenseSummary, out_dir: str | Path) -> None:
    """Write ``expense_summary.json`` + ``.csv`` and append a section to run_report.md."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "expense_summary.json").write_text(
        summary.model_dump_json(indent=2), encoding="utf-8"
    )

    with (out_dir / "expense_summary.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["store", "count", "spend", "mean_confidence"])
        for store, spend in _store_rows(summary):
            writer.writerow([store, spend.count, f"{spend.spend:.2f}",
                             f"{spend.mean_confidence:.3f}"])

    _append_report_section(out_dir / "run_report.md", summary)


def _append_report_section(path: Path, summary: ExpenseSummary) -> None:
    """Append (or replace) the ``## Financial summary`` section of run_report.md."""
    cur = summary.currency
    marker = "\n## Financial summary\n"
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    cut = existing.find(marker)
    if cut != -1:  # drop a previous run's section so re-runs stay idempotent
        existing = existing[:cut].rstrip() + "\n"

    lines = [
        "", "## Financial summary", "",
        f"- Currency: **{cur}**",
        (f"- Total spend: **{cur} {summary.total_spend:,.2f}** across "
        f"{summary.num_transactions_with_total} of {summary.num_transactions} "
        "receipts with a confident total"),
        (f"- Date range: {summary.date_range.get('start') or '?'} "
        f"-> {summary.date_range.get('end') or '?'}"),
        f"- Excluded from total_spend: {len(summary.excluded)} receipt(s)",
        "",
        "| Store | Receipts | Spend | Mean total conf |",
        "| --- | ---: | ---: | ---: |",
    ]
    lines += [
        f"| {store} | {s.count} | {cur} {s.spend:,.2f} | {s.mean_confidence:.3f} |"
        for store, s in _store_rows(summary)
    ]
    lines += ["", *(f"- _{note}_" for note in summary.notes), ""]
    path.write_text(existing + "\n".join(lines) + "\n", encoding="utf-8")
