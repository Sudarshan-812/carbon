"""Field-level confidence scoring + reliability handling.

See Prompt.md - Phase 5.

``field_confidence`` = weighted sum of up to four signals:
  ocr_conf       - mean OCR confidence of the value's tokens
  pattern_conf   - format validity (date parses, currency regex, ranges)
  heuristic_conf - strength of the rule that found it (tier-A keyword vs guess)
  cross_check    - total_amount only: sum(item prices) vs total
Weights per field come from ``cfg.confidence.weights`` and sum to 1. When a
signal is not applicable (e.g. no items, so no cross-check) it is dropped and
the remaining weights are renormalised.
"""
from __future__ import annotations

import re
import statistics
from typing import Any

from .config import Config, FieldWeights
from .extract import RawExtraction
from .ocr import OcrResult
from .schema import FieldConf, Item, ReceiptExtraction
from .utils import get_logger, parse_money

log = get_logger(__name__)

_ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_MONEY_RE = re.compile(r"^\d+\.\d{2}$")

_STORE_HEURISTIC = {"vendor_match": 1.0, "company_suffix": 1.0, "top_scored": 0.6,
                    "weak": 0.3, "missing": 0.0}
_DATE_HEURISTIC = {"keyword_line": 1.0, "regex_only": 0.65, "missing": 0.0}
_TOTAL_HEURISTIC = {"tier_a_keyword": 1.0, "tier_b_keyword": 0.7, "fallback_max": 0.3,
                    "missing": 0.0}

_CONFLICT_PENALTY = 0.15


# --- signal functions ----------------------------------------------

def _ocr_conf(confs: list[float]) -> float:
    return statistics.fmean(confs) if confs else 0.0


def _pattern_conf_date(value: str | None, dayfirst_assumed: bool) -> float:
    if not value or not _ISO_RE.match(value):
        return 0.0
    return 0.5 if dayfirst_assumed else 1.0


def _pattern_conf_money(value: str | None) -> float:
    if not value:
        return 0.0
    number = parse_money(value)
    if number is None:
        return 0.0
    in_range = 0.01 <= number <= 100_000
    if _MONEY_RE.match(value.strip()) and in_range:
        return 1.0
    return 0.6 if in_range else 0.25


def _pattern_conf_text(value: str | None) -> float:
    """Alphabetic-character ratio; penalise very short or digit-only values."""
    if not value:
        return 0.0
    alpha = sum(c.isalpha() for c in value)
    if alpha == 0:
        return 0.0
    ratio = alpha / len(value)
    return ratio * 0.5 if len(value) < 3 else ratio


def _cross_check_total(
    total: float | None, item_prices: list[float], cfg: Config
) -> float | None:
    """Compare ``sum(item_prices)`` to *total*; ``None`` when not applicable."""
    if total is None or total <= 0 or not item_prices:
        return None
    subtotal = sum(item_prices)
    if subtotal <= 0:
        return None
    lo, hi = sorted((subtotal, total))
    tolerance = max(cfg.confidence.cross_check.rel_tolerance * total,
                    cfg.confidence.cross_check.abs_tolerance)
    if abs(subtotal - total) <= tolerance:
        return 1.0
    return (lo / hi) ** 2


# --- combination -------------------------------------------------

def _combine(
    signals: dict[str, float | None], weights: FieldWeights
) -> tuple[float, dict[str, Any]]:
    """Weighted mean of the present signals, weights renormalised to sum 1."""
    weight_of = {"ocr": weights.ocr, "pattern": weights.pattern,
                 "heuristic": weights.heuristic, "cross_check": weights.cross_check}
    active = {k: v for k, v in signals.items()
              if v is not None and weight_of.get(k, 0.0) > 0.0}
    total_weight = sum(weight_of[k] for k in active)
    if total_weight == 0.0:
        return 0.0, {}
    conf = sum(weight_of[k] * active[k] for k in active) / total_weight
    breakdown = {k: round(active[k], 3) for k in active}
    breakdown["weights"] = {k: round(weight_of[k] / total_weight, 3) for k in active}
    return conf, breakdown


# --- entry point -----------------------------------------------

def score(raw: RawExtraction, ocr: OcrResult, cfg: Config) -> ReceiptExtraction:
    """Turn a :class:`RawExtraction` into a scored :class:`ReceiptExtraction`."""
    weights = cfg.confidence.weights
    threshold = cfg.confidence.low_conf_threshold
    result = ReceiptExtraction()
    breakdown: dict[str, Any] = {}
    alternatives: dict[str, Any] = {}

    item_prices = [
        price for _, price_raw in raw.items
        if (price := parse_money(price_raw.value)) is not None
    ]

    # -- store_name --
    store = raw.store_name
    if store.value is None:
        result.store_name = FieldConf(value=None, confidence=0.0)
    else:
        conf, breakdown["store_name"] = _combine(
            {"ocr": _ocr_conf(store.tokens_conf),
             "pattern": _pattern_conf_text(store.value),
             "heuristic": _STORE_HEURISTIC.get(store.rule, 0.5)},
            weights.store_name,
        )
        result.store_name = FieldConf(value=store.value, confidence=conf)

    # -- date --
    date = raw.date
    if date.value is None:
        result.date = FieldConf(value=None, confidence=0.0)
    else:
        ambiguous = bool(date.signals.get("dayfirst_assumed"))
        conf, bd = _combine(
            {"ocr": _ocr_conf(date.tokens_conf),
             "pattern": _pattern_conf_date(date.value, ambiguous),
             "heuristic": _DATE_HEURISTIC.get(date.rule, 0.5)},
            weights.date,
        )
        if date.signals.get("n_distinct", 1) >= 2:
            conf -= _CONFLICT_PENALTY
            bd["conflict_penalty"] = -_CONFLICT_PENALTY
            alternatives["date"] = [a for a in date.alternatives if a.get("value") != date.value]
        result.date = FieldConf(value=date.value, confidence=conf)
        breakdown["date"] = bd

    # -- total_amount --
    total = raw.total_amount
    cross = None
    if total.value is None:
        result.total_amount = FieldConf(value=None, confidence=0.0)
    else:
        cross = _cross_check_total(parse_money(total.value), item_prices, cfg)
        conf, bd = _combine(
            {"ocr": _ocr_conf(total.tokens_conf),
             "pattern": _pattern_conf_money(total.value),
             "heuristic": _TOTAL_HEURISTIC.get(total.rule, 0.5),
             "cross_check": cross},
            weights.total_amount,
        )
        if total.signals.get("conflicting"):
            conf -= _CONFLICT_PENALTY
            bd["conflict_penalty"] = -_CONFLICT_PENALTY
            alternatives["total_amount"] = [
                a for a in total.alternatives if a.get("value") != total.value
            ]
        result.total_amount = FieldConf(value=total.value, confidence=conf)
        breakdown["total_amount"] = bd

    # -- items --
    for idx, (name_raw, price_raw) in enumerate(raw.items):
        name_conf, name_bd = _combine(
            {"ocr": _ocr_conf(name_raw.tokens_conf),
             "pattern": _pattern_conf_text(name_raw.value)},
            weights.item_name,
        )
        price_conf, price_bd = _combine(
            {"ocr": _ocr_conf(price_raw.tokens_conf),
             "pattern": _pattern_conf_money(price_raw.value)},
            weights.item_price,
        )
        result.items.append(Item(
            name=FieldConf(value=name_raw.value, confidence=name_conf),
            price=FieldConf(value=price_raw.value, confidence=price_conf),
            meta={"qty": name_raw.signals.get("qty"),
                  "line_index": name_raw.line_index,
                  "breakdown": {"name": name_bd, "price": price_bd}},
        ))

    # -- reliability --
    result.flags = _flags(raw, ocr, cross, cfg)
    result.low_confidence_fields = _low_confidence_fields(result, threshold)
    result.meta["confidence_breakdown"] = breakdown
    if alternatives:
        result.meta["alternatives"] = alternatives
    return result


def _flags(
    raw: RawExtraction, ocr: OcrResult, cross: float | None, cfg: Config
) -> list[str]:
    flags: list[str] = []
    if raw.store_name.value is None:
        flags.append("missing_store")
    if raw.date.value is None:
        flags.append("missing_date")
    if raw.total_amount.value is None:
        flags.append("missing_total")
    if not raw.items:
        flags.append("no_items")
    if raw.total_amount.rule == "fallback_max":
        flags.append("total_from_fallback")
    if raw.total_amount.signals.get("conflicting"):
        flags.append("conflicting_total")
    if raw.date.signals.get("dayfirst_assumed"):
        flags.append("date_order_ambiguous")
    if cross is not None and cross < cfg.confidence.flags.mismatch_cross_check:
        flags.append("items_price_sum_mismatch")
    if ocr.mean_conf < cfg.confidence.flags.low_mean_ocr_conf:
        flags.append("low_mean_ocr_conf")
    return flags


def _low_confidence_fields(result: ReceiptExtraction, threshold: float) -> list[str]:
    low: list[str] = []
    for name, field in (("store_name", result.store_name), ("date", result.date),
                        ("total_amount", result.total_amount)):
        if field.confidence < threshold:
            low.append(name)
    for idx, item in enumerate(result.items):
        if item.name.confidence < threshold:
            low.append(f"items[{idx}].name")
        if item.price.confidence < threshold:
            low.append(f"items[{idx}].price")
    return low
