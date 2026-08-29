"""Extraction rule tests. See Prompt.md - Phase 4.

These are written first (red) as the spec for the extractors. Un-skip as each
rule lands.
"""
import pytest

pytestmark = pytest.mark.skip(reason="Phase 4 - implement src/extract.py")


def test_total_prefers_grand_total_over_subtotal(make_ocr, cfg):
    from src.extract import extract_total

    ocr = make_ocr([
        ("MILK 1L 5.90", 0.9),
        ("BREAD 3.10", 0.9),
        ("SUBTOTAL 9.00", 0.95),
        ("GST 6% 0.54", 0.9),
        ("GRAND TOTAL 9.54", 0.97),
        ("CASH 10.00", 0.9),
        ("CHANGE 0.46", 0.9),
    ])
    raw = extract_total(ocr, cfg)
    assert raw.value == "9.54"


def test_total_fallback_when_no_keyword(make_ocr, cfg):
    from src.extract import extract_total

    ocr = make_ocr([
        ("ITEM A 2.00", 0.9),
        ("ITEM B 3.00", 0.9),
        ("5.00", 0.8),
    ])
    raw = extract_total(ocr, cfg)
    assert raw.value == "5.00"
    assert "fallback" in raw.rule


def test_store_name_second_line_under_logo(make_ocr, cfg):
    from src.extract import extract_store_name

    ocr = make_ocr([
        ("*** ***", 0.3),
        ("SUPER MART SDN BHD", 0.95),
        ("NO 1 JALAN ABC", 0.9),
    ])
    raw = extract_store_name(ocr, cfg)
    assert "SUPER MART" in raw.value


def test_date_iso_normalization(make_ocr, cfg):
    from src.extract import extract_date

    ocr = make_ocr([("DATE: 12/03/2018 14:22", 0.9)])
    raw = extract_date(ocr, cfg)
    assert raw.value == "2018-03-12"
