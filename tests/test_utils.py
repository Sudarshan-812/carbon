"""Tests for src/utils.py. See Prompt.md - Phase 1."""
from __future__ import annotations

import numpy as np
import pytest

from src.utils import Timer, imread_unicode, imwrite_unicode, parse_money, timer

# --- parse_money ---------------------------------------------------------

MONEY_CASES = [
    # plain US-style
    ("12.90", 12.90),
    ("1234.50", 1234.50),
    ("0.05", 0.05),
    ("1000", 1000.0),
    # currency prefixes
    ("RM 1,234.50", 1234.50),
    ("RM1,234.50", 1234.50),
    ("MYR 12.90", 12.90),
    ("USD 7.00", 7.00),
    ("$12.90", 12.90),
    # thousands separators
    ("1,234.50", 1234.50),
    ("1,000", 1000.0),
    ("1.234.567,89", 1234567.89),   # European multi-group
    # European decimal comma
    ("1.234,50", 1234.50),
    ("1,23", 1.23),
    # trailing minus (refund marker) stays positive
    ("5.90-", 5.90),
    ("5.90 -", 5.90),
    # missing cents
    ("RM 5", 5.0),
    ("RM5", 5.0),
    # embedded in a label line
    ("TOTAL  54.20", 54.20),
    ("GRAND TOTAL   RM 108.00", 108.00),
    # failures
    ("", None),
    (None, None),
    ("no digits here", None),
    ("N/A", None),
]


@pytest.mark.parametrize("text,expected", MONEY_CASES)
def test_parse_money(text, expected):
    assert parse_money(text) == expected


def test_parse_money_matches_left_most_number():
    # parse_money is a low-level "first money-ish token" parser. Stripping a
    # leading qty like "2 x" is the extraction layer's job, not this one's.
    assert parse_money("2 x 3.50") == 2.0
    assert parse_money("x 3.50") == 3.50


# --- Timer -------------------------------------------------------------

def test_timer_reports_non_negative_elapsed_ms():
    with Timer() as t:
        pass
    assert t() >= 0.0
    assert t.ms == pytest.approx(t(), abs=1.0)


def test_timer_alias_is_the_class():
    assert timer is Timer


def test_timer_measures_time_inside_block():
    with Timer() as t:
        live = t()
    assert live >= 0.0


# --- image IO --------------------------------------------------------

def test_imread_unicode_roundtrip(tmp_path):
    path = tmp_path / "café_收据_test.png"
    img = np.full((10, 12, 3), 128, dtype=np.uint8)
    assert imwrite_unicode(path, img) is True
    back = imread_unicode(path)
    assert back is not None
    assert back.shape[:2] == (10, 12)


def test_imread_unicode_missing_file_returns_none(tmp_path):
    assert imread_unicode(tmp_path / "does_not_exist.png") is None


def test_imread_unicode_empty_file_returns_none(tmp_path):
    empty = tmp_path / "empty.png"
    empty.write_bytes(b"")
    assert imread_unicode(empty) is None


def test_imread_unicode_garbage_file_returns_none(tmp_path):
    junk = tmp_path / "junk.png"
    junk.write_bytes(b"not really a png")
    assert imread_unicode(junk) is None
