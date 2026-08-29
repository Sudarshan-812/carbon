"""Field-level confidence scoring + reliability handling.

See Prompt.md - Phase 5.

field_confidence = weighted sum of:
  ocr_conf       - mean OCR confidence of the value's tokens
  pattern_conf   - format validity (date parses, currency regex, ranges)
  heuristic_conf - strength of the rule that found it (tier-A keyword vs guess)
  cross_check    - total_amount only: sum(item prices) vs total
Weights per field come from ``cfg.confidence.weights`` and sum to 1.
"""
from __future__ import annotations

from .config import Config
from .extract import RawExtraction
from .ocr import OcrResult
from .schema import ReceiptExtraction
from .utils import get_logger

log = get_logger(__name__)


def score(raw: RawExtraction, ocr: OcrResult, cfg: Config) -> ReceiptExtraction:
    """Turn a :class:`RawExtraction` into a scored :class:`ReceiptExtraction`.

    TODO(Phase 5):
      * compute the four signals per field, combine with configured weights
      * clamp/round to 3dp; stash the signal breakdown in ``meta``
      * populate ``low_confidence_fields`` (< threshold, incl. per-item prices)
      * populate ``flags`` (missing_*, items_price_sum_mismatch,
        total_from_fallback, date_order_ambiguous, low_mean_ocr_conf,
        conflicting_total)
      * missing field -> value=None, confidence=0.0
      * conflicting candidates -> keep pick, -0.15 confidence, list rejects in meta
    """
    raise NotImplementedError("Phase 5")


def _pattern_conf_date(value: str | None, dayfirst_assumed: bool) -> float:
    raise NotImplementedError("Phase 5")


def _pattern_conf_money(value: str | None) -> float:
    raise NotImplementedError("Phase 5")


def _cross_check_total(total: float | None, item_prices: list[float], cfg: Config) -> float:
    raise NotImplementedError("Phase 5")
