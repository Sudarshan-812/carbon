"""Money-parsing tests. See Prompt.md - Phase 1."""
import pytest

from src.utils import parse_money

CASES = [
    ("RM 1,234.50", 1234.50),
    ("12.90", 12.90),
    ("RM12.90", 12.90),
    ("1.234,50", 1234.50),   # European style
    ("TOTAL  54.20", 54.20),
    ("5.90-", 5.90),         # trailing minus (receipt refund marker)
    ("1000", 1000.0),
    ("", None),
    ("no digits here", None),
]


@pytest.mark.parametrize("text,expected", CASES)
def test_parse_money(text, expected):
    assert parse_money(text) == expected
