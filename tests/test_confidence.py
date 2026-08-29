"""Confidence-scoring tests. See Prompt.md - Phase 5."""
from __future__ import annotations

import pytest

from src.confidence import (
    _cross_check_total,
    _pattern_conf_date,
    _pattern_conf_money,
    _pattern_conf_text,
    score,
)
from src.extract import FieldRaw, RawExtraction
from src.ocr import OcrResult


def _ocr(mean_conf: float = 0.9) -> OcrResult:
    return OcrResult(tokens=[], lines=[], full_text="", mean_conf=mean_conf, engine="fake")


def _item(name: str, name_conf: float, price: str, price_conf: float):
    return FieldRaw(name, [name_conf]), FieldRaw(price, [price_conf])


# --- signal helpers ---------------------------------------------

def test_pattern_conf_date():
    assert _pattern_conf_date("2018-03-12", False) == 1.0
    assert _pattern_conf_date("2018-03-12", True) == 0.5
    assert _pattern_conf_date(None, False) == 0.0
    assert _pattern_conf_date("not-a-date", False) == 0.0


def test_pattern_conf_money():
    assert _pattern_conf_money("12.90") == 1.0
    assert _pattern_conf_money("12.9") == 0.6            # parses, wrong format
    assert _pattern_conf_money("999999.00") == 0.25     # out of range
    assert _pattern_conf_money("O.OO") == 0.0           # unparseable
    assert _pattern_conf_money(None) == 0.0


def test_pattern_conf_text():
    assert _pattern_conf_text("KEDAI RUNCIT") > 0.8
    assert _pattern_conf_text("12345") == 0.0
    assert _pattern_conf_text("AB") == pytest.approx(0.5)


def test_cross_check_total():
    assert _cross_check_total(10.0, [4.0, 6.0], _cfg()) == 1.0
    assert _cross_check_total(10.0, [4.0, 6.30], _cfg()) == 1.0   # within abs tolerance
    assert _cross_check_total(10.0, [2.0], _cfg()) < 0.1
    assert _cross_check_total(None, [1.0], _cfg()) is None
    assert _cross_check_total(10.0, [], _cfg()) is None


def _cfg():
    from src.config import load_config
    return load_config()


# --- score(): ordering ----------------------------------------

def test_confidence_ordering_clean_gt_fallback_gt_unparseable(cfg):
    clean = RawExtraction(
        total_amount=FieldRaw("9.54", [0.97], "tier_a_keyword", signals={"tier": "A"}),
        items=[_item("A", 0.9, "5.94", 0.9), _item("B", 0.9, "3.60", 0.9)],
    )
    fallback = RawExtraction(
        total_amount=FieldRaw("5.00", [0.8], "fallback_max", signals={"from_fallback": True}),
    )
    unparseable = RawExtraction(
        total_amount=FieldRaw("O.OO", [0.5], "tier_b_keyword"),
    )
    c = score(clean, _ocr(), cfg).total_amount.confidence
    f = score(fallback, _ocr(), cfg).total_amount.confidence
    u = score(unparseable, _ocr(), cfg).total_amount.confidence
    assert c > f > u
    assert c > 0.9


def test_matching_items_boost_total_confidence(cfg):
    base = FieldRaw("20.00", [0.9], "tier_b_keyword")
    with_items = RawExtraction(
        total_amount=base,
        items=[_item("A", 0.9, "10.00", 0.9), _item("B", 0.9, "10.00", 0.9)],
    )
    without = RawExtraction(total_amount=base)
    assert (score(with_items, _ocr(), cfg).total_amount.confidence
            > score(without, _ocr(), cfg).total_amount.confidence)


# --- score(): flags & reliability --------------------------

def test_missing_fields_null_value_zero_conf_and_flags(cfg):
    result = score(RawExtraction(), _ocr(0.9), cfg)
    assert result.total_amount.value is None
    assert result.total_amount.confidence == 0.0
    assert set(result.flags) >= {"missing_store", "missing_date", "missing_total", "no_items"}
    assert set(result.low_confidence_fields) >= {"store_name", "date", "total_amount"}


def test_low_confidence_field_flagged_below_threshold(cfg):
    raw = RawExtraction(store_name=FieldRaw("XY", [0.3], "weak"))
    result = score(raw, _ocr(0.9), cfg)
    assert result.store_name.confidence < 0.7
    assert "store_name" in result.low_confidence_fields


def test_items_price_sum_mismatch_flag(cfg):
    raw = RawExtraction(
        total_amount=FieldRaw("100.00", [0.9], "tier_a_keyword"),
        items=[_item("A", 0.9, "5.00", 0.9)],
    )
    assert "items_price_sum_mismatch" in score(raw, _ocr(0.9), cfg).flags


def test_conflicting_total_penalty_and_alternatives(cfg):
    raw = RawExtraction(total_amount=FieldRaw(
        "12.00", [0.95], "tier_a_keyword",
        alternatives=[{"value": "12.00", "tier": "A", "line_index": 2},
                      {"value": "10.00", "tier": "A", "line_index": 0}],
        signals={"conflicting": True, "tier": "A"},
    ))
    plain = RawExtraction(total_amount=FieldRaw("12.00", [0.95], "tier_a_keyword"))
    result = score(raw, _ocr(0.9), cfg)
    assert "conflicting_total" in result.flags
    assert result.total_amount.confidence == pytest.approx(
        score(plain, _ocr(0.9), cfg).total_amount.confidence - 0.15, abs=0.005)
    assert result.meta["alternatives"]["total_amount"] == [
        {"value": "10.00", "tier": "A", "line_index": 0}]


def test_fallback_and_ambiguous_date_flags(cfg):
    raw = RawExtraction(
        total_amount=FieldRaw("5.00", [0.8], "fallback_max"),
        date=FieldRaw("2018-03-12", [0.9], "keyword_line",
                      signals={"dayfirst_assumed": True, "n_distinct": 1}),
    )
    flags = score(raw, _ocr(0.9), cfg).flags
    assert "total_from_fallback" in flags
    assert "date_order_ambiguous" in flags


def test_low_mean_ocr_conf_flag(cfg):
    assert "low_mean_ocr_conf" in score(RawExtraction(), _ocr(0.3), cfg).flags
    assert "low_mean_ocr_conf" not in score(RawExtraction(), _ocr(0.9), cfg).flags


def test_item_price_low_confidence_granular_entry(cfg):
    raw = RawExtraction(items=[
        _item("GOOD ITEM", 0.95, "9.99", 0.95),
        _item("BAD ITEM", 0.2, "8.88", 0.15),
    ])
    low = score(raw, _ocr(0.9), cfg).low_confidence_fields
    assert "items[1].price" in low
    assert "items[0].price" not in low
