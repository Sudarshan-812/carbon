"""Financial-summary tests. See Prompt.md - Phase 7."""
from __future__ import annotations

import json

from src.schema import FieldConf, ReceiptExtraction
from src.summary import build_summary, load_results_from_json, write_summary


def _r(image_id: str, store: str | None, total: str | None, conf: float,
       date: str | None = None) -> ReceiptExtraction:
    return ReceiptExtraction(
        store_name=FieldConf(value=store, confidence=0.9 if store else 0.0),
        date=FieldConf(value=date, confidence=0.9 if date else 0.0),
        total_amount=FieldConf(value=total, confidence=conf),
        meta={"image_id": image_id},
    )


def test_total_spend_sums_only_confident_totals(cfg):
    results = [
        _r("a", "STORE X", "10.00", 0.90),
        _r("b", "STORE X", "5.00", 0.40),      # below min_total_conf
        _r("c", "STORE Y", "20.00", 0.50),     # exactly at threshold -> counted
        _r("d", "STORE Y", None, 0.00),        # missing
    ]
    s = build_summary(results, cfg)
    assert s.total_spend == 30.00
    assert s.num_transactions == 4
    assert s.num_transactions_with_total == 2
    reasons = {e["image_id"]: e["reason"] for e in s.excluded}
    assert set(reasons) == {"b", "d"}
    assert "missing_total" == reasons["d"]
    assert reasons["b"].startswith("total_confidence")


def test_spend_per_store_fuzzy_merges_vendors(cfg):
    results = [
        _r("a", "TESCO STORES SDN BHD", "10.00", 0.9),
        _r("b", "TESCO STORES", "15.00", 0.9),
        _r("c", "AEON", "20.00", 0.9),
    ]
    s = build_summary(results, cfg)
    assert len(s.spend_per_store) == 2
    tesco = s.spend_per_store["TESCO STORES SDN BHD"]
    assert tesco.count == 2
    assert tesco.spend == 25.00
    assert s.spend_per_store["AEON"].spend == 20.00


def test_missing_store_bucketed_as_unknown(cfg):
    s = build_summary([_r("a", None, "8.00", 0.9)], cfg)
    assert "UNKNOWN" in s.spend_per_store
    assert s.spend_per_store["UNKNOWN"].spend == 8.00
    assert s.spend_per_store["UNKNOWN"].count == 1


def test_mean_confidence_per_store(cfg):
    s = build_summary([
        _r("a", "STORE X", "10.00", 0.9),
        _r("b", "STORE X", "10.00", 0.5),
    ], cfg)
    assert s.spend_per_store["STORE X"].mean_confidence == 0.7


def test_date_range(cfg):
    s = build_summary([
        _r("a", "S", "1.00", 0.9, date="2018-05-01"),
        _r("b", "S", "1.00", 0.9, date="2018-01-15"),
        _r("c", "S", "1.00", 0.9, date="2018-12-31"),
        _r("d", "S", "1.00", 0.9, date=None),
    ], cfg)
    assert s.date_range == {"start": "2018-01-15", "end": "2018-12-31"}


def test_date_range_empty_when_no_valid_dates(cfg):
    s = build_summary([_r("a", "S", "1.00", 0.9)], cfg)
    assert s.date_range == {"start": None, "end": None}


def test_notes_mention_exclusions(cfg):
    s = build_summary([_r("a", "S", None, 0.0), _r("b", "S", "9.00", 0.9)], cfg)
    assert any("excluded from" in n for n in s.notes)
    assert any("confidence >=" in n for n in s.notes)


def test_write_summary_produces_json_csv_and_report(tmp_path, cfg):
    s = build_summary([
        _r("a", "STORE X", "10.00", 0.9, date="2018-03-01"),
        _r("b", "STORE Y", "20.00", 0.9, date="2018-03-02"),
        _r("c", "STORE Y", None, 0.0),
    ], cfg)
    write_summary(s, tmp_path)

    data = json.loads((tmp_path / "expense_summary.json").read_text(encoding="utf-8"))
    assert data["total_spend"] == 30.00
    assert data["num_transactions"] == 3

    csv_lines = (tmp_path / "expense_summary.csv").read_text(encoding="utf-8").splitlines()
    assert csv_lines[0] == "store,count,spend,mean_confidence"
    assert any(line.startswith("STORE Y,2,20.00") for line in csv_lines)

    report = (tmp_path / "run_report.md").read_text(encoding="utf-8")
    assert "## Financial summary" in report
    assert "RM 30.00" in report


def test_load_results_from_json_roundtrip(tmp_path, cfg):
    from src.pipeline import write_receipt_json

    src = _r("recpt1", "MY STORE", "42.50", 0.88, date="2019-06-01")
    write_receipt_json(src, tmp_path)
    (loaded,) = load_results_from_json(tmp_path)
    assert loaded.store_name.value == "MY STORE"
    assert loaded.total_amount.value == "42.50"
    assert loaded.total_amount.confidence == 0.88
    assert loaded.meta["image_id"] == "recpt1"

    s = build_summary([loaded], cfg)
    assert s.total_spend == 42.50
