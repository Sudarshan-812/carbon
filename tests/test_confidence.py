"""Confidence-scoring tests. See Prompt.md - Phase 5."""
import pytest

pytestmark = pytest.mark.skip(reason="Phase 5 - implement src/confidence.py")


def test_matching_items_boost_total_confidence(make_ocr, cfg):
    """A total that equals the sum of item prices should score higher than the
    same total reached via the bottom-third fallback rule."""
    raise NotImplementedError


def test_low_confidence_fields_flagged_below_threshold(make_ocr, cfg):
    raise NotImplementedError


def test_missing_total_sets_null_value_and_flag(make_ocr, cfg):
    raise NotImplementedError
