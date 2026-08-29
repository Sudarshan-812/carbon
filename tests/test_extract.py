"""Extraction rule tests. See Prompt.md - Phase 4.

Each test builds a synthetic ``OcrResult`` via the ``make_ocr`` fixture and
asserts the picked raw values / provenance.
"""
from __future__ import annotations

from src.extract import (
    VendorRegistry,
    extract_date,
    extract_fields,
    extract_items,
    extract_store_name,
    extract_total,
)

# --- total amount --------------------------------------------------

def test_clean_receipt_all_fields(make_ocr, cfg):
    ocr = make_ocr([
        ("MY STORE SDN BHD", 0.95),
        ("DATE 12/03/2018", 0.9),
        ("MILK 1L 5.90", 0.9),
        ("BREAD 3.10", 0.9),
        ("TOTAL 9.00", 0.95),
    ])
    raw = extract_fields(ocr, cfg)
    assert "MY STORE" in raw.store_name.value
    assert raw.date.value == "2018-03-12"
    assert raw.total_amount.value == "9.00"
    assert [n.value for n, _ in raw.items] == ["MILK 1L", "BREAD"]
    assert [p.value for _, p in raw.items] == ["5.90", "3.10"]


def test_total_prefers_grand_total_over_subtotal(make_ocr, cfg):
    ocr = make_ocr([
        ("MILK 5.90", 0.9), ("BREAD 3.10", 0.9),
        ("SUBTOTAL 9.00", 0.95), ("GST 0.54", 0.9),
        ("GRAND TOTAL 9.54", 0.97), ("CASH 10.00", 0.9), ("CHANGE 0.46", 0.9),
    ])
    raw = extract_total(ocr, cfg)
    assert raw.value == "9.54"
    assert raw.rule == "tier_a_keyword"


def test_total_fallback_when_no_keyword(make_ocr, cfg):
    ocr = make_ocr([("ITEM A 2.00", 0.9), ("ITEM B 3.00", 0.9), ("5.00", 0.8)])
    raw = extract_total(ocr, cfg)
    assert raw.value == "5.00"
    assert raw.rule == "fallback_max"
    assert raw.signals["from_fallback"] is True


def test_total_european_decimals(make_ocr, cfg):
    ocr = make_ocr([("STUFF", 0.9), ("GRAND TOTAL 1.234,50", 0.95)])
    assert extract_total(ocr, cfg).value == "1234.50"


def test_total_keyword_line_without_money_uses_next_line(make_ocr, cfg):
    ocr = make_ocr([("GRAND TOTAL", 0.95), ("RM 55.40", 0.9)])
    raw = extract_total(ocr, cfg)
    assert raw.value == "55.40"
    assert raw.line_index == 1


def test_total_tier_a_last_occurrence_wins_and_flags_conflict(make_ocr, cfg):
    ocr = make_ocr([
        ("TOTAL AMOUNT 10.00", 0.9),
        ("XXX", 0.9),
        ("GRAND TOTAL 12.00", 0.9),
    ])
    raw = extract_total(ocr, cfg)
    assert raw.value == "12.00"
    assert raw.signals["conflicting"] is True


def test_total_tier_b_used_when_no_tier_a(make_ocr, cfg):
    raw = extract_total(make_ocr([("TOTAL 8.80", 0.9)]), cfg)
    assert raw.value == "8.80"
    assert raw.rule == "tier_b_keyword"


def test_total_inclusive_of_gst_line_is_kept(make_ocr, cfg):
    ocr = make_ocr([
        ("SUBTOTAL 60.00", 0.9),
        ("GST 6% 3.60", 0.9),
        ("TOTAL INCLUSIVE OF GST 63.60", 0.95),
    ])
    assert extract_total(ocr, cfg).value == "63.60"


# --- date --------------------------------------------------------

def test_date_iso_normalization_marks_ambiguous(make_ocr, cfg):
    raw = extract_date(make_ocr([("DATE: 12/03/2018 14:22", 0.9)]), cfg)
    assert raw.value == "2018-03-12"
    assert raw.signals["dayfirst_assumed"] is True
    assert raw.rule == "keyword_line"


def test_date_unambiguous_has_no_dayfirst_flag(make_ocr, cfg):
    raw = extract_date(make_ocr([("Tarikh 25/12/2018", 0.9)]), cfg)
    assert raw.value == "2018-12-25"
    assert raw.signals["dayfirst_assumed"] is False


def test_date_iso_input_format(make_ocr, cfg):
    assert extract_date(make_ocr([("2019-07-04", 0.9)]), cfg).value == "2019-07-04"


def test_date_textual_month(make_ocr, cfg):
    assert extract_date(make_ocr([("12 JAN 2018", 0.9)]), cfg).value == "2018-01-12"


def test_date_missing(make_ocr, cfg):
    raw = extract_date(make_ocr([("NO DATE HERE", 0.9), ("MILK 5.90", 0.9)]), cfg)
    assert raw.value is None
    assert raw.rule == "missing"


def test_date_two_candidates_prefers_keyword_line(make_ocr, cfg):
    ocr = make_ocr([
        ("BILL 01/01/2018", 0.9),
        ("XXX", 0.9), ("XXX", 0.9), ("XXX", 0.9),
        ("INVOICE DATE 15/06/2019", 0.9),
    ])
    raw = extract_date(ocr, cfg)
    assert raw.value == "2019-06-15"
    assert {alt["value"] for alt in raw.alternatives} == {"2018-01-01", "2019-06-15"}


# --- store name ------------------------------------------------

def test_store_name_second_line_under_logo(make_ocr, cfg):
    ocr = make_ocr([
        ("* * *", 0.3),
        ("SUPER MART SDN BHD", 0.95),
        ("NO 1 JALAN ABC", 0.9),
    ])
    raw = extract_store_name(ocr, cfg)
    assert "SUPER MART" in raw.value
    assert raw.rule == "company_suffix"


def test_store_name_rejects_tax_invoice_header(make_ocr, cfg):
    ocr = make_ocr([("TAX INVOICE", 0.98), ("KEDAI RUNCIT ABC", 0.9)])
    raw = extract_store_name(ocr, cfg)
    assert raw.value == "KEDAI RUNCIT ABC"
    assert raw.line_index == 1


def test_vendor_registry_merges_variants(make_ocr, cfg):
    registry = VendorRegistry(ratio=cfg.extract.store_name.vendor_merge_ratio)
    first = extract_store_name(make_ocr([("TESCO STORES (M) SDN BHD", 0.9)]), cfg, registry)
    second = extract_store_name(make_ocr([("TESCO STORES", 0.9)]), cfg, registry)
    assert second.value == first.value
    assert second.rule == "vendor_match"


# --- items -----------------------------------------------------

def test_items_pair_name_and_price_across_lines(make_ocr, cfg):
    ocr = make_ocr([
        ("STORE SDN BHD", 0.9),
        ("DATE 01/02/2018", 0.9),
        ("SPECIAL FRIED RICE", 0.9),
        ("12.90", 0.9),
        ("TOTAL 12.90", 0.95),
    ])
    items = extract_items(ocr, cfg)
    assert len(items) == 1
    name, price = items[0]
    assert name.value == "SPECIAL FRIED RICE"
    assert price.value == "12.90"
    assert price.rule == "next_line"


def test_items_drops_gst_and_subtotal_lines(make_ocr, cfg):
    ocr = make_ocr([
        ("STORE SDN BHD", 0.9),
        ("DATE 01/02/2018", 0.9),
        ("COKE 2.50", 0.9),
        ("GST 0.15", 0.9),
        ("SUBTOTAL 2.50", 0.9),
        ("TOTAL 2.65", 0.95),
    ])
    items = extract_items(ocr, cfg)
    assert [n.value for n, _ in items] == ["COKE"]


def test_items_skip_rows_without_a_real_name(make_ocr, cfg):
    ocr = make_ocr([
        ("STORE SDN BHD", 0.9),
        ("DATE 01/02/2018", 0.9),
        ("SR 101.76", 0.6),          # OCR noise, not an item
        ("NASI GORENG 8.00", 0.9),
        ("TOTAL 8.00", 0.95),
    ])
    assert [n.value for n, _ in extract_items(ocr, cfg)] == ["NASI GORENG"]


def test_items_strip_leading_quantity(make_ocr, cfg):
    ocr = make_ocr([
        ("STORE SDN BHD", 0.9),
        ("DATE 01/02/2018", 0.9),
        ("3 NASI LEMAK 6.00", 0.9),
        ("TOTAL 6.00", 0.95),
    ])
    (name, price), = extract_items(ocr, cfg)
    assert name.value == "NASI LEMAK"
    assert name.signals["qty"] == 3
    assert price.value == "6.00"
