"""OCR line-reconstruction + token-merge tests. See Prompt.md - Phase 3.

The engine wrappers themselves need EasyOCR/Tesseract and are exercised by the
`python -m src.ocr` CLI smoke run, not here.
"""
from __future__ import annotations

from src.ocr import Token, merge_split_numbers, reconstruct_lines


def _tok(text: str, x: float, y: float, w: float = 40.0, h: float = 20.0, conf: float = 0.9) -> Token:
    return Token(text, conf, (x, y, w, h))


# --- reconstruct_lines -------------------------------------------------

def test_reconstruct_groups_rows_and_orders_reading_order(cfg):
    tokens = [
        _tok("PRICE", 300, 12),      # row 1, right
        _tok("ITEM", 10, 10),        # row 1, left
        _tok("5.90", 300, 60),       # row 2, right
        _tok("MILK", 10, 62),        # row 2, left
    ]
    lines = reconstruct_lines(tokens, cfg)
    assert [line.text for line in lines] == ["ITEM PRICE", "MILK 5.90"]


def test_reconstruct_line_conf_is_mean_and_box_is_union(cfg):
    tokens = [_tok("A", 0, 0, conf=0.8), _tok("B", 50, 2, conf=0.6)]
    (line,) = reconstruct_lines(tokens, cfg)
    assert line.conf == 0.7
    assert line.box[0] == 0 and line.box[2] == 90  # 50 + 40


def test_reconstruct_empty_returns_empty(cfg):
    assert reconstruct_lines([], cfg) == []


def test_reconstruct_far_apart_rows_stay_separate(cfg):
    tokens = [_tok("TOP", 0, 0), _tok("BOTTOM", 0, 500)]
    lines = reconstruct_lines(tokens, cfg)
    assert [line.text for line in lines] == ["TOP", "BOTTOM"]


# --- merge_split_numbers --------------------------------------------

def test_merge_two_part_price():
    merged = merge_split_numbers([_tok("12", 100, 10, w=20), _tok(".90", 122, 10, w=18)])
    assert [t.text for t in merged] == ["12.90"]
    assert merged[0].box[0] == 100


def test_merge_three_part_price():
    merged = merge_split_numbers([
        _tok("12", 100, 10, w=18),
        _tok(".", 119, 10, w=4),
        _tok("90", 124, 10, w=18),
    ])
    assert [t.text for t in merged] == ["12.90"]


def test_merge_leaves_ordinary_words_alone():
    merged = merge_split_numbers([_tok("MILK", 10, 10), _tok("BREAD", 55, 10)])
    assert [t.text for t in merged] == ["MILK", "BREAD"]


def test_merge_does_not_cross_rows():
    merged = merge_split_numbers([_tok("12", 100, 10), _tok(".90", 100, 200)])
    assert [t.text for t in merged] == ["12", ".90"]


def test_merge_keeps_min_conf():
    merged = merge_split_numbers([
        _tok("7", 10, 10, w=12, conf=0.95),
        _tok(".50", 23, 10, w=18, conf=0.40),
    ])
    assert merged[0].conf == 0.40


def test_merge_ignores_non_numeric_prefix():
    # "NO." + "12" must not fuse into a bogus number
    merged = merge_split_numbers([_tok("NO.", 10, 10, w=22), _tok("12", 33, 10, w=16)])
    assert [t.text for t in merged] == ["NO.", "12"]
